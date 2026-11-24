"""Hiring Signals: the request, one sentence per signal, and the text a model reads.

A signal is one event at one company, detected from its job postings: a hiring
surge, a first job in a country, a leadership search, a new technology, a
slowdown. `GET /api/signals` lists them across companies, and
`GET /api/companies/{id}/signals` lists one company's.

Pure functions over the API's JSON, with no HTTP in sight. The filter keys
follow the shared connector contract (`Jobo.Connectors/contracts/signals.json`),
and the sentences follow `describeSignal` in the connector core, which words
them the way Jobo's dashboard and alert emails do.

Signals are billed per signal returned, at ten times the price of a job, and a
signal read again is billed again.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID

from langchain_jobo.errors import (
    JoboError,
    JoboInsufficientCreditsError,
    JoboPermissionError,
    JoboRateLimitError,
    JoboValidationError,
)
from langchain_jobo.filters import to_wire_timestamp

SignalType = Literal[
    "hiring_surge",
    "new_function",
    "new_country",
    "leadership_hire",
    "new_skill",
    "hiring_slowdown",
    "first_role",
    "new_technology",
    "function_surge",
    "new_city",
    "hard_to_fill",
    "leadership_role_closed",
    "hiring_freeze",
    "hiring_resumed",
]
#: `seniority` values, lowest first.
SignalSeniority = Literal[
    "junior", "mid", "senior", "director", "vp", "c_level", "executive"
]

#: A short name for each signal type, as the dashboard shows it.
SIGNAL_TYPE_LABELS: dict[str, str] = {
    "hiring_surge": "Hiring surge",
    "new_function": "New function",
    "new_country": "New country",
    "leadership_hire": "Leadership hire",
    "new_skill": "New skill",
    "hiring_slowdown": "Hiring slowdown",
    "first_role": "First role",
    "new_technology": "New technology",
    "function_surge": "Function surge",
    "new_city": "New city",
    "hard_to_fill": "Hard to fill",
    "leadership_role_closed": "Leadership role closed",
    "hiring_freeze": "Hiring freeze",
    "hiring_resumed": "Hiring resumed",
}

SIGNALS_PATH = "/api/signals"
#: Signals per page: `limit` runs from 1 to this, and defaults to 50 at the API.
SIGNAL_MAX_LIMIT = 200
SIGNAL_DEFAULT_LIMIT = 50
#: The most values one comma-separated filter takes.
SIGNAL_MAX_LIST_VALUES = 100

#: Every query parameter `GET /api/signals` takes besides `limit` and `cursor`,
#: in the contract's order.
SIGNAL_FILTER_KEYS: tuple[str, ...] = (
    "types",
    "domains",
    "company_ids",
    "exclude_domains",
    "countries",
    "company_sizes",
    "industries",
    "categories",
    "funding_stages",
    "founded_after",
    "exclude_agencies",
    "functions",
    "roles",
    "seniority",
    "technologies",
    "skills",
    "technology_categories",
    "job_countries",
    "job_regions",
    "job_cities",
    "min_strength",
    "min_new_jobs",
    "min_days_open",
    "reposted_only",
    "detected_after",
    "detected_before",
    "created_after",
)
#: The filters sent as one comma-separated list.
SIGNAL_LIST_FILTER_KEYS: frozenset[str] = frozenset(
    {
        "types",
        "domains",
        "company_ids",
        "exclude_domains",
        "countries",
        "company_sizes",
        "industries",
        "categories",
        "funding_stages",
        "functions",
        "roles",
        "seniority",
        "technologies",
        "skills",
        "technology_categories",
        "job_countries",
        "job_regions",
        "job_cities",
    }
)
#: The filters `GET /api/companies/{id}/signals` takes; the company is in the path.
COMPANY_SIGNAL_FILTER_KEYS: tuple[str, ...] = (
    "types",
    "min_strength",
    "detected_after",
    "detected_before",
    "created_after",
)

#: An instant as the API reads one: date, time with seconds, an optional
#: fraction of up to seven digits, and an optional `Z` or offset.
_INSTANT = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,7})?(Z|[+-]\d{2}:\d{2})?"
)


# ------------------------------------------------------------------- request


def company_signals_path(company_id: str) -> str:
    """`GET /api/companies/{id}/signals` for one company.

    Args:
        company_id: A Jobo company id (a UUID), in any spelling `uuid.UUID`
            reads.

    Raises:
        ValueError: The id is not a UUID.
    """
    try:
        canonical = str(UUID(company_id.strip()))
    except ValueError as exc:
        msg = (
            f'"{company_id.strip()}" is not a Jobo company id. Company ids are '
            "UUIDs, such as the Company ID shown with each signal."
        )
        raise ValueError(msg) from exc
    return f"/api/companies/{canonical}/signals"


def to_signal_instant(value: Any) -> Any:
    """Check an instant for `created_after` and keep it as it was written.

    `created_after` is exclusive, so the API's own `created_at` must reach it
    unchanged: parsing it would drop a seventh fractional digit and read the
    same signal, and bill it, again. A `datetime` is written out as UTC.

    Raises:
        ValueError: The text is not an ISO 8601 date and time with seconds.
    """
    if isinstance(value, datetime):
        return to_wire_timestamp(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if _INSTANT.fullmatch(text) is None:
            msg = (
                f'"{text}" is not an ISO 8601 date and time. created_after takes '
                "one with seconds, e.g. 2026-11-12T09:30:00Z."
            )
            raise ValueError(msg)
        return text
    return value


def _clean(value: Any) -> list[str]:
    """A list filter's values: trimmed, without blanks or repeats."""
    if isinstance(value, str):
        raw: list[Any] = [value]
    elif isinstance(value, (list, tuple, set, frozenset)):
        raw = list(value)
    else:
        raw = [value]
    values: list[str] = []
    for item in raw:
        text = "" if item is None else str(item).strip()
        if text and text not in values:
            values.append(text)
    return values


def check_list_values(key: str, values: Sequence[str]) -> None:
    """Refuse a list the API would misread or reject.

    The API reads a list filter as one comma-separated value, so a value that
    contains a comma would silently become two filters.

    Raises:
        ValueError: A value contains a comma, or there are more than 100.
    """
    with_comma = next((value for value in values if "," in value), None)
    if with_comma is not None:
        msg = (
            f'"{with_comma}" cannot be used in {key}: the signals API separates '
            "values with commas, so a value that contains one would be read as "
            "two. Pass separate values as separate list items, and leave the "
            "comma out of a single value."
        )
        raise ValueError(msg)
    if len(values) > SIGNAL_MAX_LIST_VALUES:
        msg = (
            f"{key} takes at most {SIGNAL_MAX_LIST_VALUES} values; "
            f"{len(values)} were given."
        )
        raise ValueError(msg)


def _is_set(value: Any) -> bool:
    """Whether a filter value would be sent. `False` and 0 are the unset state."""
    if value is None or isinstance(value, bool):
        return value is True
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, (str, list, tuple, set, frozenset)):
        return bool(_clean(value))
    return True


def company_filters_refused(filters: Mapping[str, Any]) -> list[str]:
    """The set filters `GET /api/companies/{id}/signals` does not take."""
    return [
        key
        for key in SIGNAL_FILTER_KEYS
        if key not in COMPANY_SIGNAL_FILTER_KEYS and _is_set(filters.get(key))
    ]


def build_signals_request(
    filters: Mapping[str, Any] | None = None,
    *,
    company_id: str | None = None,
    limit: int | None = None,
    cursor: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Map filters onto `GET /api/signals`, or one company's signals.

    Lists are sent comma-separated, which is the only form the API reads.
    Empty values are left out, so an untouched filter never becomes one;
    `False` and numbers below 1 are the unset state and are left out too.

    Args:
        filters: Filter values keyed as the API names them (`types`,
            `domains`, `min_strength`, ...). A list may also be one string.
        company_id: Read this one company's signals from
            `GET /api/companies/{id}/signals`, which takes only `types`,
            `min_strength`, `detected_after`, `detected_before` and
            `created_after`.
        limit: Signals per page, 1-200.
        cursor: The `next_cursor` of the previous page.

    Returns:
        The path and the query parameters.

    Raises:
        ValueError: An unknown filter; a list value that contains a comma;
            more than 100 values in one list; a `limit` outside 1-200; a
            `company_id` that is not a UUID; or, with `company_id`, a filter
            the per-company endpoint does not take.
    """
    values = dict(filters or {})
    unknown = sorted(set(values) - set(SIGNAL_FILTER_KEYS))
    if unknown:
        msg = (
            f"Unknown hiring signal filter(s): {', '.join(unknown)}. Known: "
            f"{', '.join(SIGNAL_FILTER_KEYS)}."
        )
        raise ValueError(msg)

    path = SIGNALS_PATH
    company = (company_id or "").strip()
    if company:
        path = company_signals_path(company)
        refused = company_filters_refused(values)
        if refused:
            raise ValueError(refused_company_filters_message(refused))

    params: dict[str, Any] = {}
    for key in SIGNAL_FILTER_KEYS:
        value = values.get(key)
        if value is None:
            continue
        if key in SIGNAL_LIST_FILTER_KEYS:
            entries = _clean(value)
            if entries:
                check_list_values(key, entries)
                params[key] = ",".join(entries)
        elif isinstance(value, bool):
            if value:
                params[key] = True
        elif isinstance(value, (int, float)):
            if value > 0:
                params[key] = int(value)
        elif isinstance(value, datetime):
            params[key] = to_wire_timestamp(value)
        elif isinstance(value, date):
            params[key] = value.isoformat()
        else:
            text = str(value).strip()
            if text:
                params[key] = text

    if limit is not None:
        if not 1 <= limit <= SIGNAL_MAX_LIMIT:
            msg = f"limit must be between 1 and {SIGNAL_MAX_LIMIT}, got {limit}."
            raise ValueError(msg)
        params["limit"] = limit
    position = (cursor or "").strip()
    if position:
        params["cursor"] = position
    return path, params


def refused_company_filters_message(refused: Sequence[str]) -> str:
    """Why filters were refused beside `company_id`, and what to do instead."""
    return (
        "company_id reads one company's signals, and that endpoint takes only "
        f"{', '.join(COMPANY_SIGNAL_FILTER_KEYS)}, limit and cursor. Leave out "
        f"{', '.join(refused)}, or drop company_id and name the company in "
        "domains instead."
    )


# --------------------------------------------------------------------- words


def signal_type_label(signal_type: Any) -> str:
    """The label of a type, or the type in words for one added after this."""
    if isinstance(signal_type, str) and signal_type in SIGNAL_TYPE_LABELS:
        return SIGNAL_TYPE_LABELS[signal_type]
    words = ("" if signal_type is None else str(signal_type)).replace("_", " ").strip()
    return words[:1].upper() + words[1:] if words else "Signal"


def signal_strength_band(strength: float) -> str:
    """Strength in a word: `Very strong`, `Strong`, `Moderate` or `Mild`.

    Strength runs from 1 to 100 and compares signals of one type, so it reads
    better as one of four bands than as a score.
    """
    if strength >= 75:
        return "Very strong"
    if strength >= 50:
        return "Strong"
    if strength >= 25:
        return "Moderate"
    return "Mild"


def _scaled(value: float, digits: int) -> int:
    """`value` rounded to `digits` decimals, half away from zero, times 10**digits.

    As the connector core rounds, and the API: `Math.round` of the absolute
    value, nudged up by 1e-9 so 1050.25 does not round down to 1050.2.
    """
    units = math.floor(abs(value) * 10.0**digits + 1e-9 + 0.5)
    return -units if value < 0 else units


def _whole(value: float) -> str:
    return f"{_scaled(value, 0):,}"


def _decimal(value: float) -> str:
    """At most one decimal, and none when it would be ".0"."""
    tenths = _scaled(value, 1)
    sign = "-" if tenths < 0 else ""
    integer, fraction = divmod(abs(tenths), 10)
    text = f"{sign}{integer:,}"
    return f"{text}.{fraction}" if fraction else text


def _counted(count: float, noun: str) -> str:
    return f"1 {noun}" if abs(count - 1) < 0.0001 else f"{_whole(count)} {noun}s"


def _lookback(days: float | None) -> str:
    if days is None or days <= 0:
        return "six months"
    if days < 60:
        return f"{_whole(days)} days"
    months = _scaled(days / 30, 0)
    return "six months" if months == 6 else f"{months} months"


def _fact_number(facts: Mapping[str, Any], name: str) -> float | None:
    value = facts.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _fact_text(facts: Mapping[str, Any], name: str) -> str | None:
    value = facts.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _sentence(signal_type: Any, facts: Mapping[str, Any]) -> str | None:
    """The sentence for a type, or `None` when `facts` lack what it needs."""

    def number(name: str) -> float | None:
        return _fact_number(facts, name)

    def text(name: str) -> str | None:
        return _fact_text(facts, name)

    new_jobs = number("new_jobs")
    period = _lookback(number("lookback_days"))

    if signal_type == "hiring_surge":
        baseline = number("baseline_weekly_avg")
        if new_jobs is None or baseline is None:
            return None
        return (
            f"{_counted(new_jobs, 'new job')} this week vs {_decimal(baseline)} "
            "a week normally"
        )
    if signal_type == "hiring_slowdown":
        now = number("open_roles")
        before = number("open_roles_before")
        if now is None or before is None:
            return None
        change = number("change_pct")
        drop = abs(change) if change is not None else 0.0
        day = text("compared_to")
        since = f" since {day}" if day else ""
        return (
            f"Open roles down {_decimal(drop)}%{since}: {_whole(now)} now, "
            f"{_whole(before)} then"
        )
    if signal_type == "new_function":
        category = text("category")
        if not category:
            return None
        if new_jobs is not None and new_jobs > 1:
            return f"{_counted(new_jobs, 'new job')} in {category}, a first in {period}"
        return f"First {category} job in {period}"
    if signal_type == "new_country":
        country = text("country")
        if not country:
            return None
        if new_jobs is not None and new_jobs > 1:
            return f"{_counted(new_jobs, 'new job')} in {country}, a first in {period}"
        return f"First job in {country} in {period}"
    if signal_type == "new_skill":
        skill = text("skill")
        if not skill:
            return None
        if new_jobs is not None and new_jobs > 1:
            return f"{_whole(new_jobs)} new jobs ask for {skill}, a first in {period}"
        return f"First job asking for {skill} in {period}"
    if signal_type == "leadership_hire":
        title = text("title")
        return f"Hiring a leader: {title}" if title else None
    if signal_type == "first_role":
        occupation = text("occupation")
        return f"First {occupation} role" if occupation else None
    if signal_type == "new_technology":
        technology = text("technology")
        if not technology:
            return None
        role = text("title")
        if role:
            return f"First mention of {technology}, in a {role} role"
        return f"First mention of {technology}"
    if signal_type == "function_surge":
        category = text("category")
        usual = number("baseline_avg")
        if not category or new_jobs is None or usual is None:
            return None
        days = number("window_days")
        span = f"{_whole(days)} days" if days is not None else "two weeks"
        noun = f"new {category} job"
        return f"{_counted(new_jobs, noun)} in {span} vs {_decimal(usual)} normally"
    if signal_type == "new_city":
        city = text("city")
        if not city:
            return None
        region = text("region")
        place = f"{city}, {region}" if region else city
        if new_jobs is not None and new_jobs > 1:
            return f"{_counted(new_jobs, 'new job')} in {place}, a first there"
        return f"First job in {place}"
    if signal_type == "hard_to_fill":
        title = text("title")
        days_open = number("days_open")
        if not title or days_open is None:
            return None
        city = text("city")
        at = f" ({city})" if city else ""
        if text("reason") != "reposted":
            return f"{title}{at} open {_counted(days_open, 'day')}"
        closed = number("closed_days")
        if closed is not None and closed > 0:
            return f"{title}{at} re-posted after {_counted(closed, 'day')} closed"
        return f"{title}{at} re-posted"
    # Says the posting ended, never that anyone was hired.
    if signal_type == "leadership_role_closed":
        title = text("title")
        days_open = number("days_open")
        if not title or days_open is None:
            return None
        return f"{title} search closed after {_counted(days_open, 'day')}"
    if signal_type == "hiring_freeze":
        removed = number("closed_jobs")
        had = number("open_roles_before")
        if removed is None or had is None:
            return None
        last = text("closed_to")
        week = f" in the week to {last}" if last else ""
        return f"{_whole(removed)} of {_counted(had, 'open role')} removed{week}"
    if signal_type == "hiring_resumed":
        gap = number("gap_days")
        return f"First new job in {_counted(gap, 'day')}" if gap is not None else None
    return None


def describe_signal(signal: Mapping[str, Any]) -> str:
    """One plain sentence for a signal, in the words of Jobo's dashboard.

    For example "17 new jobs this week vs 3.1 a week normally", as the
    dashboard and the alert emails put it. Falls back to the type's label when
    `details` lacks the facts the sentence needs, or the type is one this
    version does not know.

    Args:
        signal: A signal as the API returns it; only `type` and `details` are
            read.
    """
    signal_type = signal.get("type")
    details = signal.get("details")
    if not isinstance(details, Mapping):
        return signal_type_label(signal_type)
    return _sentence(signal_type, details) or signal_type_label(signal_type)


# ---------------------------------------------------------------------- text


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [text for text in (_text(item) for item in value) if text]


def signal_list(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The signals of a response body, as the API sent them."""
    signals = payload.get("signals")
    if not isinstance(signals, Sequence) or isinstance(signals, (str, bytes)):
        return []
    return [dict(signal) for signal in signals if isinstance(signal, Mapping)]


def format_signal(signal: Mapping[str, Any], index: int) -> str:
    """Render one signal as a compact block: who, what, when, how strong, why."""
    company = _mapping(signal.get("company"))
    name = _text(company.get("name")) or "Unnamed company"
    domain = _text(company.get("domain"))
    label = signal_type_label(signal.get("type"))
    who = f"{name} ({domain})" if domain and domain != name else name
    lines = [f"{index}. {who} - {label}"]

    sentence = describe_signal(signal)
    if sentence != label:
        lines.append(sentence)

    facts: list[str] = []
    detected = _text(signal.get("detected_on"))
    if detected:
        facts.append(f"Detected: {detected}")
    strength = signal.get("strength")
    if isinstance(strength, (int, float)) and not isinstance(strength, bool):
        facts.append(f"Strength: {signal_strength_band(strength)} ({strength:g}/100)")
    if facts:
        lines.append(" | ".join(facts))

    company_id = _text(company.get("id"))
    if company_id:
        lines.append(f"Company ID: {company_id}")
    evidence = _strings(signal.get("evidence_job_ids"))
    if evidence:
        lines.append(f"Evidence jobs: {', '.join(evidence)}")
    return "\n".join([lines[0], *(f"   {line}" for line in lines[1:])])


def _signals(count: int) -> str:
    return f"{count} hiring signal{'' if count == 1 else 's'}"


def signals_header(count: int, *, next_cursor: str | None = None) -> str:
    """How many signals came back, and how to read the next page if there is one.

    Args:
        count: Signals on this page.
        next_cursor: The cursor of the next page, when more signals match.
    """
    if count == 0:
        return (
            "No hiring signals matched. Widen the search: fewer types, an "
            "earlier detected_after, a lower min_strength or fewer focus filters."
        )
    header = f"{_signals(count)}, newest detection day first."
    if next_cursor:
        return (
            f'{header} More match: call again with the same arguments and cursor "'
            f'{next_cursor}" to read the next page.'
        )
    return f"{header} No more match."


def signals_billing_line(count: int, usage: Mapping[str, Any]) -> str:
    """What the page cost, from the metering headers of the response.

    Args:
        count: Signals on the page: each one is billed.
        usage: The metering the API sent: `credits_deducted`,
            `credits_balance`, `quota_limit`, `quota_remaining`,
            `billing_status`. Absent keys were not sent.
    """
    if count == 0:
        return "Billed: nothing. A request that returns no signals is free."
    parts: list[str] = []
    remaining = usage.get("quota_remaining")
    included = usage.get("quota_limit")
    if isinstance(remaining, int):
        of = f" of {included:,}" if isinstance(included, int) else ""
        parts.append(f"{remaining:,}{of} included signals left this month")
    deducted = usage.get("credits_deducted")
    if isinstance(deducted, int):
        parts.append(f"{deducted:,} credits charged to the wallet")
    balance = usage.get("credits_balance")
    if isinstance(balance, int):
        parts.append(f"wallet balance {balance:,} credits")
    status = usage.get("billing_status")
    if isinstance(status, str) and status:
        parts.append(f"billing status {status}")
    if parts:
        return (
            f"Billed: {_signals(count)} ({'; '.join(parts)}). Reading the same "
            "signals again is billed again."
        )
    return (
        f"Billed: {_signals(count)}. The API reported no charge, as on the "
        "Unlimited plan, which has no per-signal charge."
    )


def format_signals(
    payload: Mapping[str, Any],
    usage: Mapping[str, Any],
    *,
    details_tool: str,
    capped_limit: int | None = None,
) -> str:
    """Render a page of signals: a header, one block per signal, what it cost.

    Args:
        payload: The response body of `GET /api/signals`.
        usage: The metering the API sent with it.
        details_tool: The name of the tool that reads a job, for the evidence.
        capped_limit: Set when the requested limit was reduced, so the text
            says so instead of silently returning fewer signals.
    """
    signals = signal_list(payload)
    parts: list[str] = []
    if capped_limit is not None:
        parts.append(
            f"Note: limit was capped at {capped_limit}, this tool's configured maximum."
        )
    cursor = _text(payload.get("next_cursor"))
    more = cursor if payload.get("has_more") is True and cursor else None
    parts.append(signals_header(len(signals), next_cursor=more))
    parts += [format_signal(signal, index) for index, signal in enumerate(signals, 1)]
    if any(_strings(signal.get("evidence_job_ids")) for signal in signals):
        parts.append(
            "Evidence jobs are the postings behind a signal: read one free with "
            f"{details_tool}."
        )
    parts.append(signals_billing_line(len(signals), usage))
    return "\n\n".join(parts)


# -------------------------------------------------------------------- errors

#: What a sandbox key gets instead of signals.
NO_SIGNALS_IN_SANDBOX = (
    "Hiring signals cannot be read here: a sandbox API key (jbe_test_...) does "
    "not serve them. Do not call this tool again. Tell the user that hiring "
    "signals need a live API key (jbe_live_...)."
)


def signals_error_advice(error: JoboError) -> str | None:
    """What a model can do about a failed signals request, or `None`."""
    if isinstance(error, JoboInsufficientCreditsError):
        return (
            "Every hiring signal is billed, and the plan's included signals and "
            "the wallet together cannot cover this page. Ask for fewer signals "
            "with a smaller limit, or stop and tell the user the wallet needs a "
            "top-up."
        )
    if isinstance(error, JoboPermissionError):
        if error.code == "account_key_required":
            return (
                "Hiring signals are charged to an account's wallet, and this API "
                "key belongs to no account. Do not call this tool again: tell "
                "the user an account API key is needed."
            )
        return (
            "This API key may not read hiring signals. Do not call this tool "
            "again: tell the user."
        )
    if isinstance(error, JoboRateLimitError):
        return (
            "Wait before calling this tool again, or tell the user the rate "
            "limit was reached."
        )
    if isinstance(error, JoboValidationError):
        return "Correct the arguments named above and call again."
    return None


def signals_error_message(error: JoboError) -> str:
    """The error's message, followed by what a model can do about it."""
    advice = signals_error_advice(error)
    return f"{error}\n{advice}" if advice else str(error)


__all__ = [
    "COMPANY_SIGNAL_FILTER_KEYS",
    "NO_SIGNALS_IN_SANDBOX",
    "SIGNALS_PATH",
    "SIGNAL_DEFAULT_LIMIT",
    "SIGNAL_FILTER_KEYS",
    "SIGNAL_LIST_FILTER_KEYS",
    "SIGNAL_MAX_LIMIT",
    "SIGNAL_MAX_LIST_VALUES",
    "SIGNAL_TYPE_LABELS",
    "SignalSeniority",
    "SignalType",
    "build_signals_request",
    "check_list_values",
    "company_filters_refused",
    "company_signals_path",
    "describe_signal",
    "format_signal",
    "format_signals",
    "refused_company_filters_message",
    "signal_list",
    "signal_strength_band",
    "signal_type_label",
    "signals_billing_line",
    "signals_error_advice",
    "signals_error_message",
    "signals_header",
    "to_signal_instant",
]
