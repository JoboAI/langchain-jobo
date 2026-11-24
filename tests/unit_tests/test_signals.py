"""The hiring signals tool: its request, its text, its cost and its opt-in."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import httpx
import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException
from pydantic import ValidationError

from langchain_jobo import (
    JoboConfigurationError,
    JoboConnectionError,
    JoboHiringSignalsTool,
    JoboInsufficientCreditsError,
    JoboNotFoundError,
    JoboPermissionError,
    JoboSearchTool,
    JoboServerError,
    JoboValidationError,
    get_jobo_tools,
)
from langchain_jobo.signals import (
    NO_SIGNALS_IN_SANDBOX,
    build_signals_request,
    describe_signal,
    signal_strength_band,
    signal_type_label,
)
from tests.unit_tests.helpers import (
    INSUFFICIENT_CREDITS,
    NOT_IN_SANDBOX,
    RATE_LIMITED,
    SANDBOX_KEY,
    TEST_KEY,
    VALIDATION_PROBLEM,
    Handler,
    MockJobo,
    reply,
)

COMPANY_ID = "6c1f0a9e-3b7d-4c55-8e21-5a4d9f0b7c33"
OTHER_COMPANY_ID = "11111111-1111-4111-8111-111111111111"
EVIDENCE = [
    "a41c7e02-58b1-4d0f-b6c9-0e2f7d5a1c88",
    "00000000-0000-4000-8000-000000000042",
]
CURSOR = "eyJkIjoiMjAyNi0xMS0wOSJ9"

SURGE: dict[str, Any] = {
    "id": "0d0f5c1e-7a52-4f0e-9d3b-2f6f1c7a9b10",
    "type": "hiring_surge",
    "detected_on": "2026-11-10",
    "created_at": "2026-11-11T02:14:09.1234567Z",
    "window_start": "2026-11-04",
    "window_end": "2026-11-10",
    "strength": 88,
    "company": {
        "id": COMPANY_ID,
        "name": "Northwind Health",
        "domain": "northwind.example",
        "country_code": "US",
        "size": "1001-5000",
        "industry": "Hospitals and Health Care",
        "industries": ["Hospitals and Health Care"],
        "categories": [],
        "is_agency": False,
    },
    "details": {
        "new_jobs": 42,
        "baseline_weekly_avg": 9,
        "ratio": 4.7,
        "open_roles": 187,
        "is_agency": False,
    },
    "focus": {
        "functions": ["healthcare"],
        "topics": [],
        "job_countries": ["US"],
        "job_regions": ["texas"],
        "job_cities": [],
    },
    "evidence_job_ids": EVIDENCE,
}

#: A type added to the API after this package was written.
NEW_TYPE: dict[str, Any] = {
    "id": "2a6f3c1e-0000-4f0e-9d3b-2f6f1c7a9b11",
    "type": "remote_shift",
    "detected_on": "2026-11-09",
    "created_at": "2026-11-10T01:00:00Z",
    "strength": 12,
    "company": {"id": OTHER_COMPANY_ID, "name": "Globex", "domain": None},
    "details": {"share": 0.4},
    "evidence_job_ids": [],
}

PAGE: dict[str, Any] = {
    "signals": [SURGE, NEW_TYPE],
    "next_cursor": CURSOR,
    "has_more": True,
}

#: A Job Search plan whose included signals are used up: the wallet pays.
METERING = {
    "X-Credits-Deducted": "60",
    "X-Credits-Balance": "4940",
    "X-Quota-Limit": "250",
    "X-Quota-Remaining": "0",
}

PAGE_TEXT = "\n\n".join(
    [
        "2 hiring signals, newest detection day first. More match: call again "
        f'with the same arguments and cursor "{CURSOR}" to read the next page.',
        "1. Northwind Health (northwind.example) - Hiring surge\n"
        "   42 new jobs this week vs 9 a week normally\n"
        "   Detected: 2026-11-10 | Strength: Very strong (88/100)\n"
        f"   Company ID: {COMPANY_ID}\n"
        f"   Evidence jobs: {EVIDENCE[0]}, {EVIDENCE[1]}",
        "2. Globex - Remote shift\n"
        "   Detected: 2026-11-09 | Strength: Mild (12/100)\n"
        f"   Company ID: {OTHER_COMPANY_ID}",
        "Evidence jobs are the postings behind a signal: read one free with "
        "jobo_job_details.",
        "Billed: 2 hiring signals (0 of 250 included signals left this month; 60 "
        "credits charged to the wallet; wallet balance 4,940 credits). Reading "
        "the same signals again is billed again.",
    ]
)

ACCOUNT_KEY_REQUIRED = {
    "type": "https://tools.ietf.org/html/rfc9110#section-15.5.4",
    "title": "Account API key required",
    "status": 403,
    "detail": (
        "Hiring Signals is charged per signal to an account's wallet. Call it "
        "with an account API key."
    ),
    "code": "account_key_required",
}


def signals_api(
    body: dict[str, Any] | None = None, headers: dict[str, str] | None = None
) -> MockJobo:
    """A mock API that answers every request with a page of signals."""
    handler = reply(
        200,
        PAGE if body is None else body,
        headers=METERING if headers is None else headers,
    )
    return MockJobo(handler)


def tool_call(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "tool_call",
        "id": "call-1",
        "name": "jobo_hiring_signals",
        "args": args,
    }


# -------------------------------------------------------------------- request


def test_the_request_joins_each_list_with_commas() -> None:
    api = signals_api()
    tool = JoboHiringSignalsTool(**api.kwargs)

    tool.invoke(
        {
            "types": "hiring_surge, New_Country",
            "domains": ["stripe.com", " shopify.com ", "stripe.com"],
            "countries": ["DE", "AT"],
            "company_sizes": ["51-200", "201-500"],
            "industries": ["Financial Services"],
            "funding_stages": ["Seed", "Series A"],
            "functions": ["Sales"],
            "roles": ["account executive"],
            "seniority": ["Director", "vp"],
            "technologies": "Salesforce, HubSpot",
            "skills": ["Negotiation"],
            "job_countries": ["DE"],
            "min_strength": 60,
            "detected_after": "2026-11-01",
            "detected_before": "2026-11-23",
            "created_after": "2026-11-20T08:00:00.1234567Z",
            "exclude_agencies": True,
            "limit": 3,
            "cursor": "abc",
        }
    )

    assert api.last.method == "GET"
    assert api.last.url.path == "/api/signals"
    assert dict(api.last.url.params) == {
        "types": "hiring_surge,new_country",
        "domains": "stripe.com,shopify.com",
        "countries": "DE,AT",
        "company_sizes": "51-200,201-500",
        "industries": "Financial Services",
        "funding_stages": "Seed,Series A",
        "exclude_agencies": "true",
        "functions": "Sales",
        "roles": "account executive",
        "seniority": "director,vp",
        "technologies": "Salesforce,HubSpot",
        "skills": "Negotiation",
        "job_countries": "DE",
        "min_strength": "60",
        "detected_after": "2026-11-01",
        "detected_before": "2026-11-23",
        # Sent as written: `created_after` is exclusive, and a parsed instant
        # would lose the seventh digit and read the same signal again.
        "created_after": "2026-11-20T08:00:00.1234567Z",
        "limit": "3",
        "cursor": "abc",
    }
    assert api.last.headers["X-Api-Key"] == TEST_KEY


def test_an_empty_call_asks_for_five_signals_and_nothing_else() -> None:
    api = signals_api()

    JoboHiringSignalsTool(**api.kwargs).invoke({})

    assert dict(api.last.url.params) == {"limit": "5"}


def test_unset_values_are_left_out() -> None:
    api = signals_api()

    JoboHiringSignalsTool(**api.kwargs).invoke(
        {"domains": [" ", ""], "exclude_agencies": False, "cursor": " "}
    )

    assert dict(api.last.url.params) == {"limit": "5"}


def test_limit_is_capped_and_the_model_is_told() -> None:
    api = signals_api()
    tool = JoboHiringSignalsTool(**api.kwargs)

    content = tool.invoke({"limit": 200})

    assert api.last.url.params["limit"] == "25"
    assert content.startswith(
        "Note: limit was capped at 25, this tool's configured maximum."
    )


@pytest.mark.parametrize(("cap", "default"), [(25, 5), (3, 3), (200, 5)])
def test_the_schema_states_the_cap_as_the_maximum(cap: int, default: int) -> None:
    api = signals_api()
    tool = JoboHiringSignalsTool(max_signals=cap, **api.kwargs)

    schema = tool.args["limit"]
    tool.invoke({"limit": 200})

    assert schema["maximum"] == cap
    assert schema["minimum"] == 1
    assert schema["default"] == default
    assert f"at most {cap}" in schema["description"]
    assert api.last.url.params["limit"] == str(cap)


@pytest.mark.parametrize("cap", [0, 201])
def test_the_cap_cannot_leave_the_api_range(cap: int) -> None:
    with pytest.raises(ValidationError):
        JoboHiringSignalsTool(api_key=TEST_KEY, max_signals=cap)


def test_company_id_reads_one_company_from_its_own_endpoint() -> None:
    api = signals_api()
    tool = JoboHiringSignalsTool(**api.kwargs)

    tool.invoke(
        {
            "company_id": f" {COMPANY_ID.upper()} ",
            "types": ["hiring_freeze", "hiring_slowdown"],
            "min_strength": 50,
            "detected_after": "2026-10-01",
            "exclude_agencies": False,
        }
    )

    assert api.last.url.path == f"/api/companies/{COMPANY_ID}/signals"
    assert dict(api.last.url.params) == {
        "types": "hiring_freeze,hiring_slowdown",
        "min_strength": "50",
        "detected_after": "2026-10-01",
        "limit": "5",
    }


def test_company_id_refuses_the_filters_its_endpoint_ignores() -> None:
    """The endpoint would ignore them, and bill signals they meant to leave out."""
    api = signals_api()
    tool = JoboHiringSignalsTool(**api.kwargs)

    with pytest.raises(ValidationError, match="Leave out countries, functions,"):
        tool.invoke(
            {
                "company_id": COMPANY_ID,
                "types": ["hiring_surge"],
                "countries": ["DE"],
                "functions": ["Sales"],
            }
        )

    assert api.requests == []


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"company_id": "acme.com"}, '"acme.com" is not a Jobo company id'),
        (
            {"industries": ["Film, TV & Stage Production"]},
            "cannot be used in industries",
        ),
        # A name is one value, so a comma in it is refused, not split.
        ({"funding_stages": "Series A, Series B"}, "cannot be used in funding_stages"),
        (
            {"domains": [f"company{index}.example" for index in range(101)]},
            "domains takes at most 100 values; 101 were given",
        ),
        ({"created_after": "2026-11-20"}, "not an ISO 8601 date and time"),
    ],
)
def test_values_the_api_would_misread_are_refused_before_any_request(
    args: dict[str, Any], message: str
) -> None:
    api = signals_api()

    with pytest.raises(ValidationError, match=message):
        JoboHiringSignalsTool(**api.kwargs).invoke(args)

    assert api.requests == []


@pytest.mark.parametrize(
    "args",
    [
        {"company_ids": [COMPANY_ID]},  # the tool takes one company_id
        {"exclude_domains": ["acme.com"]},  # not an argument of the tool
        {"types": ["remote_shift"]},
        {"seniority": ["intern"]},
        {"company_sizes": ["1-5"]},
        {"min_strength": 101},
        {"detected_after": "last week"},
        {"limit": 0},
    ],
)
def test_invalid_arguments_are_rejected_before_any_request(
    args: dict[str, Any],
) -> None:
    api = signals_api()

    with pytest.raises(ValidationError):
        JoboHiringSignalsTool(**api.kwargs).invoke(args)

    assert api.requests == []


def test_build_signals_request_takes_every_filter_of_the_api() -> None:
    """Also the ones the tool leaves to the developer."""
    path, params = build_signals_request(
        {
            "exclude_domains": "acme.com",
            "categories": ["SaaS"],
            "founded_after": 2018,
            "job_cities": ["Austin"],
            "min_days_open": 60.0,
            "reposted_only": True,
            "min_new_jobs": 0,
            "detected_after": date(2026, 11, 1),
            "created_after": datetime(2026, 11, 20, 8, 0, tzinfo=timezone.utc),
        },
        limit=200,
        cursor=" next ",
    )

    assert path == "/api/signals"
    assert params == {
        "exclude_domains": "acme.com",
        "categories": "SaaS",
        "founded_after": 2018,
        "job_cities": "Austin",
        "min_days_open": 60,
        "reposted_only": True,
        "detected_after": "2026-11-01",
        "created_after": "2026-11-20T08:00:00Z",
        "limit": 200,
        "cursor": "next",
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"filters": {"company": ["acme.com"]}}, "Unknown hiring signal filter"),
        ({"limit": 0}, "limit must be between 1 and 200"),
        ({"limit": 201}, "limit must be between 1 and 200"),
        (
            {"filters": {"job_regions": ["Texas"]}, "company_id": COMPANY_ID},
            "Leave out job_regions,",
        ),
    ],
)
def test_build_signals_request_refuses_what_the_api_would_not_read(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_signals_request(**kwargs)


# ----------------------------------------------------------------------- text


def test_the_model_reads_one_compact_block_per_signal_and_the_cost() -> None:
    api = signals_api()
    tool = JoboHiringSignalsTool(**api.kwargs)

    message = tool.invoke(tool_call({"types": ["hiring_surge"]}))

    assert isinstance(message, ToolMessage)
    assert message.name == "jobo_hiring_signals"
    assert message.status == "success"
    assert message.content == PAGE_TEXT
    assert message.artifact == {
        "signals": [SURGE, NEW_TYPE],
        "next_cursor": CURSOR,
        "has_more": True,
        "usage": {
            "credits_deducted": 60,
            "credits_balance": 4940,
            "quota_limit": 250,
            "quota_remaining": 0,
        },
    }


def test_the_last_page_says_no_more_match() -> None:
    api = signals_api({"signals": [SURGE], "has_more": False}, headers={})

    content = JoboHiringSignalsTool(**api.kwargs).invoke({})

    assert content.startswith(
        "1 hiring signal, newest detection day first. No more match.\n\n"
    )
    assert "cursor" not in content
    # Unlimited sends no metering headers: there is no per-signal charge.
    assert content.endswith(
        "Billed: 1 hiring signal. The API reported no charge, as on the "
        "Unlimited plan, which has no per-signal charge."
    )


def test_included_signals_alone_are_reported_as_such() -> None:
    headers = {"X-Quota-Limit": "1000", "X-Quota-Remaining": "998"}
    api = signals_api({"signals": [SURGE], "has_more": False}, headers=headers)

    content = JoboHiringSignalsTool(**api.kwargs).invoke({})

    assert content.endswith(
        "Billed: 1 hiring signal (998 of 1,000 included signals left this "
        "month). Reading the same signals again is billed again."
    )


def test_an_empty_page_says_it_cost_nothing() -> None:
    api = signals_api({"signals": [], "has_more": False}, headers={})

    message = JoboHiringSignalsTool(**api.kwargs).invoke(tool_call({}))

    assert message.content == (
        "No hiring signals matched. Widen the search: fewer types, an earlier "
        "detected_after, a lower min_strength or fewer focus filters.\n\n"
        "Billed: nothing. A request that returns no signals is free."
    )
    assert message.artifact == {
        "signals": [],
        "next_cursor": None,
        "has_more": False,
        "usage": {},
    }


def test_a_signal_with_missing_parts_still_renders() -> None:
    bare = {"type": "leadership_hire", "details": None, "company": None}
    api = signals_api({"signals": [bare]}, headers={})

    content = JoboHiringSignalsTool(**api.kwargs).invoke({})

    assert "1. Unnamed company - Leadership hire" in content


def test_the_description_says_what_a_signal_is_and_what_it_costs() -> None:
    description = JoboHiringSignalsTool(api_key=TEST_KEY).description

    assert description.startswith("Read hiring signals: events at companies")
    for phrase in (
        "ten times the price of a job",
        "a signal read again is billed again",
        "keep limit small",
        "returns no signals costs nothing",
        "jobo_job_details reads for free",
    ):
        assert phrase in description, phrase


# --------------------------------------------------------------------- errors


def lost(error: type[httpx.TransportError]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error("the response was lost", request=request)

    return handler


@pytest.mark.parametrize(
    ("handler", "cause"),
    [
        (reply(500, {"message": "An error occurred"}), JoboServerError),
        (lost(httpx.ReadTimeout), JoboConnectionError),
        (lost(httpx.RemoteProtocolError), JoboConnectionError),
    ],
)
def test_a_failure_after_the_request_was_sent_is_not_replayed(
    handler: Handler, cause: type[Exception]
) -> None:
    """The page may have been billed: a replay would bill it again."""
    api = MockJobo(handler)
    tool = JoboHiringSignalsTool(**api.kwargs)

    with pytest.raises(ToolException) as caught:
        tool.invoke({"domains": ["stripe.com"]})

    assert isinstance(caught.value.__cause__, cause)
    assert len(api.requests) == 1


def test_402_tells_the_model_to_ask_for_fewer() -> None:
    api = MockJobo(reply(402, INSUFFICIENT_CREDITS))
    tool = JoboHiringSignalsTool(**api.kwargs)

    with pytest.raises(ToolException, match="with a smaller limit") as caught:
        tool.invoke({"limit": 20})

    assert isinstance(caught.value.__cause__, JoboInsufficientCreditsError)
    assert "https://enterprise.jobo.world/credits?topup=10" in str(caught.value)
    assert len(api.requests) == 1


def test_403_without_an_account_key_tells_the_model_to_stop() -> None:
    api = MockJobo(reply(403, ACCOUNT_KEY_REQUIRED))
    tool = JoboHiringSignalsTool(handle_tool_error=True, **api.kwargs)

    message = tool.invoke(tool_call({}))

    assert message.status == "error"
    assert "(account_key_required)" in message.content
    assert "Do not call this tool again" in message.content
    assert "an account API key is needed" in message.content


def test_another_403_is_still_explained() -> None:
    api = MockJobo(reply(403, {"title": "Forbidden", "status": 403}))

    with pytest.raises(ToolException, match="may not read hiring signals") as caught:
        JoboHiringSignalsTool(**api.kwargs).invoke({})

    assert isinstance(caught.value.__cause__, JoboPermissionError)


def test_429_tells_the_model_how_long_to_wait() -> None:
    api = MockJobo(reply(429, RATE_LIMITED, headers={"Retry-After": "12"}))
    tool = JoboHiringSignalsTool(max_retries=0, handle_tool_error=True, **api.kwargs)

    message = tool.invoke(tool_call({}))

    assert message.status == "error"
    assert "Retry after 12 seconds." in message.content
    assert message.content.endswith(
        "Wait before calling this tool again, or tell the user the rate limit "
        "was reached."
    )


def test_400_names_the_fields_and_asks_for_a_correction() -> None:
    api = MockJobo(reply(400, VALIDATION_PROBLEM))

    with pytest.raises(ToolException, match="Correct the arguments") as caught:
        JoboHiringSignalsTool(**api.kwargs).invoke({})

    assert isinstance(caught.value.__cause__, JoboValidationError)
    assert "page_size: page_size must be between 1 and 100" in str(caught.value)


def test_a_sandbox_key_is_answered_instead_of_raising() -> None:
    """`handle_tool_error` is off: a raise here would end the agent's run."""
    api = MockJobo(reply(404, {**NOT_IN_SANDBOX, "instance": "/api/signals"}))
    tool = JoboHiringSignalsTool(
        api_key=SANDBOX_KEY,
        http_client=api.client,
        http_async_client=api.async_client,
    )

    message = tool.invoke(tool_call({"types": ["hiring_surge"]}))

    assert message.status == "success"
    assert message.content == NO_SIGNALS_IN_SANDBOX
    assert "Do not call this tool again." in message.content
    assert message.artifact["signals"] == []
    assert message.artifact["unavailable"] == "not_available_in_sandbox"
    assert len(api.requests) == 1


def test_any_other_404_is_an_error() -> None:
    api = MockJobo(reply(404, {"title": "Not Found", "status": 404}))

    with pytest.raises(ToolException, match="Jobo API error 404") as caught:
        JoboHiringSignalsTool(**api.kwargs).invoke({"company_id": COMPANY_ID})

    assert isinstance(caught.value.__cause__, JoboNotFoundError)


# ---------------------------------------------------------------------- async


async def test_async_sends_the_same_request_and_text() -> None:
    api = signals_api()
    tool = JoboHiringSignalsTool(**api.kwargs)

    message = await tool.ainvoke(
        tool_call({"company_id": COMPANY_ID, "types": "hiring_surge", "limit": 2})
    )

    assert api.last.url.path == f"/api/companies/{COMPANY_ID}/signals"
    assert dict(api.last.url.params) == {"types": "hiring_surge", "limit": "2"}
    assert message.content == PAGE_TEXT
    assert message.artifact["next_cursor"] == CURSOR


async def test_async_500_is_not_replayed() -> None:
    api = MockJobo(reply(500, {"message": "An error occurred"}))

    with pytest.raises(ToolException) as caught:
        await JoboHiringSignalsTool(**api.kwargs).ainvoke({})

    assert isinstance(caught.value.__cause__, JoboServerError)
    assert len(api.requests) == 1


async def test_async_errors_carry_the_same_advice() -> None:
    api = MockJobo(reply(402, INSUFFICIENT_CREDITS))

    with pytest.raises(ToolException, match="with a smaller limit"):
        await JoboHiringSignalsTool(**api.kwargs).ainvoke({})


async def test_async_sandbox_key_is_answered() -> None:
    api = MockJobo(reply(404, NOT_IN_SANDBOX))
    tool = JoboHiringSignalsTool(
        api_key=SANDBOX_KEY,
        http_client=api.client,
        http_async_client=api.async_client,
    )

    content = await tool.ainvoke({})

    assert content == NO_SIGNALS_IN_SANDBOX


# --------------------------------------------------------------------- opt-in


def test_get_jobo_tools_leaves_hiring_signals_out_by_default() -> None:
    names = [tool.name for tool in get_jobo_tools(TEST_KEY)]

    assert "jobo_hiring_signals" not in names
    assert len(names) == 5


def test_get_jobo_tools_adds_hiring_signals_when_asked() -> None:
    api = signals_api()

    tools = get_jobo_tools(
        include_hiring_signals=True,
        include_lookups=False,
        max_signals=7,
        handle_tool_error=True,
        **api.kwargs,
    )
    tools[-1].invoke({"limit": 50})

    assert [tool.name for tool in tools] == [
        "jobo_job_search",
        "jobo_job_details",
        "jobo_hiring_signals",
    ]
    signals = tools[-1]
    assert isinstance(signals, JoboHiringSignalsTool)
    assert signals.max_signals == 7
    assert signals.handle_tool_error is True
    assert signals.response_format == "content_and_artifact"
    assert api.last.url.params["limit"] == "7"


def test_get_jobo_tools_shares_one_connection_pool_with_the_signals_tool() -> None:
    tools = get_jobo_tools(TEST_KEY, include_hiring_signals=True)
    search, signals = tools[0], tools[-1]

    assert isinstance(search, JoboSearchTool)
    assert isinstance(signals, JoboHiringSignalsTool)
    assert signals._api() is search._api()
    assert len(tools) == 6


def test_the_tool_needs_a_key_and_rejects_mistyped_arguments() -> None:
    with pytest.raises(JoboConfigurationError, match="JOBO_API_KEY"):
        JoboHiringSignalsTool()
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        JoboHiringSignalsTool(api_key=TEST_KEY, max_signal=3)


# ------------------------------------------------------------------ sentences

#: The connector core's table (signal-summary.test.ts): the sentences Jobo's
#: dashboard and alert emails write.
SENTENCES: list[tuple[str, dict[str, Any], str]] = [
    (
        "hiring_surge",
        {"new_jobs": 17, "baseline_weekly_avg": 3.1},
        "17 new jobs this week vs 3.1 a week normally",
    ),
    (
        "hiring_surge",
        {"new_jobs": 1, "baseline_weekly_avg": 0.04},
        "1 new job this week vs 0 a week normally",
    ),
    (
        "hiring_surge",
        {"new_jobs": 1200, "baseline_weekly_avg": 1050.25},
        "1,200 new jobs this week vs 1,050.3 a week normally",
    ),
    (
        "hiring_slowdown",
        {
            "open_roles": 40,
            "open_roles_before": 90,
            "change_pct": -55.6,
            "compared_to": "2026-09-21",
        },
        "Open roles down 55.6% since 2026-09-21: 40 now, 90 then",
    ),
    (
        "new_function",
        {"category": "Sales", "new_jobs": 4, "lookback_days": 180},
        "4 new jobs in Sales, a first in six months",
    ),
    (
        "new_function",
        {"category": "Sales", "lookback_days": 90},
        "First Sales job in 3 months",
    ),
    (
        "new_country",
        {"country": "Germany", "new_jobs": 1},
        "First job in Germany in six months",
    ),
    (
        "new_country",
        {"country": "Germany", "new_jobs": 3, "lookback_days": 45},
        "3 new jobs in Germany, a first in 45 days",
    ),
    (
        "new_skill",
        {"skill": "Rust", "new_jobs": 5},
        "5 new jobs ask for Rust, a first in six months",
    ),
    (
        "new_skill",
        {"skill": "Rust", "named": True},
        "First job asking for Rust in six months",
    ),
    (
        "leadership_hire",
        {"title": "VP of Engineering"},
        "Hiring a leader: VP of Engineering",
    ),
    ("first_role", {"occupation": "Data Engineer"}, "First Data Engineer role"),
    (
        "new_technology",
        {"technology": "Snowflake", "title": "Data Engineer"},
        "First mention of Snowflake, in a Data Engineer role",
    ),
    ("new_technology", {"technology": "Snowflake"}, "First mention of Snowflake"),
    (
        "function_surge",
        {
            "category": "Technology",
            "new_jobs": 12,
            "baseline_avg": 2.5,
            "window_days": 14,
        },
        "12 new Technology jobs in 14 days vs 2.5 normally",
    ),
    (
        "function_surge",
        {"category": "Technology", "new_jobs": 1, "baseline_avg": 0.2},
        "1 new Technology job in two weeks vs 0.2 normally",
    ),
    (
        "new_city",
        {"city": "Austin", "region": "Texas", "new_jobs": 6},
        "6 new jobs in Austin, Texas, a first there",
    ),
    ("new_city", {"city": "Austin"}, "First job in Austin"),
    (
        "hard_to_fill",
        {"title": "Staff Nurse", "days_open": 92, "city": "Leeds"},
        "Staff Nurse (Leeds) open 92 days",
    ),
    (
        "hard_to_fill",
        {
            "title": "Staff Nurse",
            "days_open": 30,
            "reason": "reposted",
            "closed_days": 1,
        },
        "Staff Nurse re-posted after 1 day closed",
    ),
    (
        "hard_to_fill",
        {"title": "Staff Nurse", "days_open": 30, "reason": "reposted"},
        "Staff Nurse re-posted",
    ),
    (
        "leadership_role_closed",
        {"title": "Chief Financial Officer", "days_open": 61},
        "Chief Financial Officer search closed after 61 days",
    ),
    (
        "hiring_freeze",
        {"closed_jobs": 45, "open_roles_before": 52, "closed_to": "2026-11-15"},
        "45 of 52 open roles removed in the week to 2026-11-15",
    ),
    ("hiring_resumed", {"gap_days": 74}, "First new job in 74 days"),
]


@pytest.mark.parametrize(("signal_type", "details", "sentence"), SENTENCES)
def test_one_sentence_per_signal(
    signal_type: str, details: dict[str, Any], sentence: str
) -> None:
    assert describe_signal({"type": signal_type, "details": details}) == sentence


def test_the_sentence_table_covers_every_type() -> None:
    covered = {signal_type for signal_type, _, _ in SENTENCES}

    assert len(covered) == 14


def test_the_sentence_falls_back_to_the_label_when_facts_are_missing() -> None:
    surge: dict[str, Any] = {"type": "hiring_surge", "details": {"new_jobs": 17}}
    leader: dict[str, Any] = {"type": "leadership_hire", "details": {}}
    city: dict[str, Any] = {"type": "new_city", "details": None}
    # A number must be a number: true is not 1.
    resumed: dict[str, Any] = {"type": "hiring_resumed", "details": {"gap_days": True}}

    assert describe_signal(surge) == "Hiring surge"
    assert describe_signal(leader) == "Leadership hire"
    assert describe_signal(city) == "New city"
    assert describe_signal(resumed) == "Hiring resumed"


def test_a_type_this_version_does_not_know_is_named_not_failed_on() -> None:
    unknown: dict[str, Any] = {"type": "remote_shift", "details": {"share": 0.4}}

    assert describe_signal(unknown) == "Remote shift"
    assert signal_type_label("remote_shift") == "Remote shift"
    assert signal_type_label("") == "Signal"
    assert signal_type_label(None) == "Signal"


def test_strength_reads_as_one_of_four_bands() -> None:
    bands = [signal_strength_band(value) for value in (100, 75, 74, 50, 49, 25, 24, 1)]

    assert bands == [
        "Very strong",
        "Very strong",
        "Strong",
        "Strong",
        "Moderate",
        "Moderate",
        "Mild",
        "Mild",
    ]
