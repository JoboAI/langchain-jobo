"""Render Jobo job objects as text a model can read and cite.

Pure functions over the API's job JSON (`JobDto`), with no HTTP in sight. A
field the posting does not state is left out of the text and of the metadata:
nothing here fills a gap with a guess.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.documents import Document

_PERIOD_PHRASES = {
    "hourly": "per hour",
    "daily": "per day",
    "weekly": "per week",
    "monthly": "per month",
    "yearly": "per year",
    "per-diem": "per diem",
}

_MAX_SUGGESTED_COMPANIES = 3


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [text for text in (_text(item) for item in value) if text]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _money(value: float) -> str:
    return f"{value:,.0f}" if value == int(value) else f"{value:,.2f}"


def _day(value: Any) -> str:
    """The calendar date of an ISO 8601 timestamp, or an empty string."""
    return _text(value)[:10]


def _place(location: Mapping[str, Any]) -> str:
    label = _text(location.get("location"))
    if label:
        return label
    parts = (_text(location.get(key)) for key in ("city", "region", "country"))
    return ", ".join(part for part in parts if part)


def job_places(job: Mapping[str, Any]) -> list[str]:
    """Every distinct place the job lists, in the order the API returned them."""
    places: list[str] = []
    locations = job.get("locations")
    if isinstance(locations, Sequence) and not isinstance(locations, (str, bytes)):
        for location in locations:
            place = _place(_mapping(location))
            if place and place not in places:
                places.append(place)
    return places


def format_location(job: Mapping[str, Any]) -> str:
    """Render the job's location as one short phrase."""
    places = job_places(job)
    if not places:
        return "Remote" if job.get("workplace_type") == "Remote" else "Not stated"
    if len(places) == 1:
        return places[0]
    return f"{places[0]} (+{len(places) - 1} more)"


def format_compensation(job: Mapping[str, Any]) -> str:
    """Render compensation as published, or `""` when the posting has none."""
    compensation = _mapping(job.get("compensation"))
    low = _number(compensation.get("min"))
    high = _number(compensation.get("max"))
    if low is None and high is None:
        return ""

    if low is not None and high is not None:
        amount = _money(low) if low == high else f"{_money(low)}-{_money(high)}"
    elif low is not None:
        amount = f"from {_money(low)}"
    else:
        amount = f"up to {_money(high)}"  # type: ignore[arg-type]

    currency = _text(compensation.get("currency"))
    period = _text(compensation.get("period"))
    phrase = _PERIOD_PHRASES.get(period, period)
    return " ".join(part for part in (currency, amount, phrase) if part)


def format_occupation(job: Mapping[str, Any]) -> str:
    """Render the occupation as `Registered Nurse (Healthcare > Nursing)`."""
    occupation = _mapping(job.get("occupation"))
    name = _text(occupation.get("name"))
    if not name:
        return ""
    path = [
        part
        for part in (
            _text(occupation.get("category")),
            _text(occupation.get("subcategory")),
        )
        if part
    ]
    return f"{name} ({' > '.join(path)})" if path else name


def _company_name(job: Mapping[str, Any]) -> str:
    return _text(_mapping(job.get("company")).get("name"))


def _job_url(job: Mapping[str, Any]) -> str:
    return _text(job.get("listing_url")) or _text(job.get("apply_url"))


def format_job_summary(job: Mapping[str, Any], index: int | None = None) -> str:
    """Render one job as a compact, citeable block for a search result list."""
    title = _text(job.get("title")) or "Untitled job"
    company = _company_name(job)
    heading = f"{title} - {company}" if company else title
    lines = [f"{index}. {heading}" if index is not None else heading]

    facts = [f"Location: {format_location(job)}"]
    for label, key in (
        ("Work model", "workplace_type"),
        ("Employment", "employment_type"),
        ("Level", "experience_level"),
    ):
        value = _text(job.get(key))
        if value:
            facts.append(f"{label}: {value}")
    lines.append(" | ".join(facts))

    pay = format_compensation(job)
    if pay:
        lines.append(f"Salary: {pay}")

    posted = _day(job.get("date_posted"))
    if posted:
        lines.append(f"Posted: {posted}")
    else:
        indexed = _day(job.get("created_at"))
        if indexed:
            lines.append(f"First indexed: {indexed}")

    url = _job_url(job)
    if url:
        lines.append(f"URL: {url}")
    lines.append(f"ID: {_text(job.get('id'))}")

    indent = "   " if index is not None else ""
    return "\n".join([lines[0], *(f"{indent}{line}" for line in lines[1:])])


def _warning_notes(payload: Mapping[str, Any]) -> list[str]:
    notes: list[str] = []
    warnings = payload.get("warnings")
    if isinstance(warnings, Sequence) and not isinstance(warnings, (str, bytes)):
        for warning in warnings:
            message = _text(_mapping(warning).get("message"))
            if message:
                notes.append(f"Note: {message}")

    companies = _mapping(_mapping(payload.get("filters")).get("companies"))
    suggestions = companies.get("suggestions")
    if isinstance(suggestions, Sequence) and not isinstance(suggestions, (str, bytes)):
        for entry in suggestions:
            entry_map = _mapping(entry)
            query = _text(entry_map.get("query"))
            options = entry_map.get("companies")
            if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
                continue
            rendered = []
            for option in options[:_MAX_SUGGESTED_COMPANIES]:
                option_map = _mapping(option)
                name = _text(option_map.get("name"))
                if not name:
                    continue
                ids = ", ".join(_strings(option_map.get("ids")))
                domain = _text(option_map.get("domain"))
                extra = "; ".join(
                    part for part in (domain, f"ids: {ids}" if ids else "") if part
                )
                rendered.append(f"{name} ({extra})" if extra else name)
            if rendered:
                notes.append(f'Closest companies to "{query}": {" | ".join(rendered)}')
    return notes


def format_search_results(
    payload: Mapping[str, Any], *, capped_page_size: int | None = None
) -> str:
    """Render a search response as a numbered list of compact job blocks.

    Args:
        payload: The `POST /api/jobs/search` response body.
        capped_page_size: Set when the requested page size was reduced, so the
            text says so instead of silently returning fewer jobs.
    """
    jobs = payload.get("jobs")
    job_list = (
        list(jobs)
        if isinstance(jobs, Sequence) and not isinstance(jobs, (str, bytes))
        else []
    )
    notes = _warning_notes(payload)
    if capped_page_size is not None:
        notes.append(
            f"Note: page_size was capped at {capped_page_size}, this tool's "
            "configured maximum."
        )

    if not job_list:
        notes.append(
            "No jobs matched. Relax a filter or broaden the query. Enum filters "
            "take lowercase keys, and an unrecognised filter value matches "
            "nothing."
        )
        return "\n".join(notes)

    total = payload.get("total")
    page = payload.get("page")
    total_pages = payload.get("total_pages")
    shown = len(job_list)
    header = f"Showing {shown} job{'' if shown == 1 else 's'}"
    if isinstance(total, int):
        header = f"{total:,} matching job{'' if total == 1 else 's'}; showing {shown}"
    if isinstance(page, int) and isinstance(total_pages, int):
        header += f" (page {page} of {total_pages:,})"
    header += "."

    blocks = [
        format_job_summary(_mapping(job), index)
        for index, job in enumerate(job_list, start=1)
    ]
    parts = [*notes, header, *blocks]
    if isinstance(page, int) and isinstance(total_pages, int) and page < total_pages:
        parts.append(f"More results are available: request page {page + 1}.")
    return "\n\n".join(parts)


def _bullets(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _qualification_items(group: Any) -> list[str]:
    group_map = _mapping(group)
    skills = group_map.get("skills")
    names: list[str] = []
    if isinstance(skills, Sequence) and not isinstance(skills, (str, bytes)):
        names = [
            name
            for name in (_text(_mapping(skill).get("name")) for skill in skills)
            if name
        ]
    return [
        *names,
        *_strings(group_map.get("education")),
        *_strings(group_map.get("certifications")),
    ]


def format_job_document(
    job: Mapping[str, Any], *, include_description: bool = True
) -> str:
    """Flatten a job into self-contained prose.

    Everything that would justify citing the job is in the text, so a reader
    never has to open the URL: the structured fields first (location,
    compensation, responsibilities, qualifications, benefits), then the
    employer's own description.

    Args:
        job: A job object as returned by search, feed or job-by-id.
        include_description: Append the full description. Turn it off for a
            shorter document built from the structured fields only.
    """
    title = _text(job.get("title")) or "Untitled job"
    company = _mapping(job.get("company"))
    company_name = _text(company.get("name"))
    sections: list[str] = [f"# {title}"]

    facts: list[str] = []
    if company_name:
        facts.append(f"Company: {company_name}")
    brand = _text(_mapping(job.get("brand")).get("name"))
    if brand and brand != company_name:
        facts.append(f"Brand: {brand}")
    facts.append(f"Location: {format_location(job)}")
    occupation = format_occupation(job)
    if occupation:
        facts.append(f"Occupation: {occupation}")
    for label, key in (
        ("Work model", "workplace_type"),
        ("Employment", "employment_type"),
        ("Level", "experience_level"),
    ):
        value = _text(job.get(key))
        if value:
            facts.append(f"{label}: {value}")
    pay = format_compensation(job)
    if pay:
        facts.append(f"Compensation: {pay}")
    posted = _day(job.get("date_posted"))
    if posted:
        facts.append(f"Posted: {posted}")
    closes = _day(job.get("valid_through"))
    if closes:
        facts.append(f"Closes: {closes}")
    industries = _strings(company.get("industries"))
    if industries:
        facts.append(f"Industry: {', '.join(industries)}")
    sections.append("\n".join(facts))

    company_summary = _text(company.get("summary"))
    if company_summary and company_name:
        sections.append(f"## About {company_name}\n{company_summary}")
    summary = _text(job.get("summary"))
    if summary:
        sections.append(f"## Summary\n{summary}")

    responsibilities = _strings(job.get("responsibilities"))
    if responsibilities:
        sections.append(f"## Responsibilities\n{_bullets(responsibilities)}")

    qualifications = _mapping(job.get("qualifications"))
    for label, key in (
        ("Required qualifications", "must_have"),
        ("Preferred qualifications", "preferred"),
    ):
        items = _qualification_items(qualifications.get(key))
        if items:
            sections.append(f"## {label}\n{_bullets(items)}")

    benefits = _strings(job.get("benefits"))
    if benefits:
        sections.append(f"## Benefits\n{_bullets(benefits)}")

    eligibility: list[str] = []
    if job.get("is_h1b_sponsor") is True:
        eligibility.append("Employer has sponsored H-1B visas.")
    if job.get("is_work_auth_required") is True:
        eligibility.append("Existing work authorisation required.")
    if job.get("is_clearance_required") is True:
        eligibility.append("Security clearance required.")
    if eligibility:
        sections.append(f"## Eligibility\n{_bullets(eligibility)}")

    places = job_places(job)
    if len(places) > 1:
        sections.append(f"## All locations\n{_bullets(places)}")

    description = _text(job.get("description"))
    if include_description and description:
        sections.append(f"## Description\n{description}")

    links: list[str] = []
    listing_url = _text(job.get("listing_url"))
    apply_url = _text(job.get("apply_url"))
    if listing_url:
        links.append(f"Listing: {listing_url}")
    if apply_url and apply_url != listing_url:
        links.append(f"Apply: {apply_url}")
    source = _text(job.get("source"))
    if source:
        links.append(f"Source: {source}")
    links.append(f"Job ID: {_text(job.get('id'))}")
    sections.append("\n".join(links))

    return "\n\n".join(sections)


def job_metadata(job: Mapping[str, Any]) -> dict[str, Any]:
    """Flat `Document` metadata for a job. Keys without a value are omitted.

    Values are strings and numbers only, so the metadata is accepted by vector
    stores that reject nested or null values.
    """
    company = _mapping(job.get("company"))
    compensation = _mapping(job.get("compensation"))
    candidates: dict[str, Any] = {
        "id": _text(job.get("id")),
        "url": _job_url(job),
        "apply_url": _text(job.get("apply_url")),
        "title": _text(job.get("title")),
        "company": _text(company.get("name")),
        "company_id": _text(company.get("id")),
        "location": format_location(job) if job_places(job) else "",
        "workplace_type": _text(job.get("workplace_type")),
        "employment_type": _text(job.get("employment_type")),
        "experience_level": _text(job.get("experience_level")),
        "salary": format_compensation(job),
        "salary_min": _number(compensation.get("min")),
        "salary_max": _number(compensation.get("max")),
        "salary_currency": _text(compensation.get("currency")),
        "salary_period": _text(compensation.get("period")),
        "date_posted": _text(job.get("date_posted")),
        "created_at": _text(job.get("created_at")),
        "updated_at": _text(job.get("updated_at")),
        "valid_through": _text(job.get("valid_through")),
        "source": _text(job.get("source")),
    }
    return {key: value for key, value in candidates.items() if value not in ("", None)}


def job_to_document(
    job: Mapping[str, Any], *, include_description: bool = True
) -> Document:
    """Turn a job object into a `Document` whose id is the Jobo job id."""
    job_id = _text(job.get("id")) or None
    return Document(
        id=job_id,
        page_content=format_job_document(job, include_description=include_description),
        metadata=job_metadata(job),
    )


def format_company_suggestions(
    query: str, suggestions: Sequence[Mapping[str, Any]]
) -> str:
    """Render company lookups: one line per company with every id to filter on."""
    if not suggestions:
        return (
            f'No company matched "{query}". Lookups match the start of a company '
            "name or website domain and need at least 2 characters: try a shorter "
            "or different spelling, or the company's domain."
        )

    lines: list[str] = []
    for index, suggestion in enumerate(suggestions, start=1):
        name = _text(suggestion.get("name")) or "Unnamed company"
        facts = [
            fact
            for fact in (
                _text(suggestion.get("domain")),
                _text(suggestion.get("country")),
            )
            if fact
        ]
        line = f"{index}. {name}"
        if facts:
            line += f" ({', '.join(facts)})"
        job_count = suggestion.get("job_count")
        if isinstance(job_count, int) and not isinstance(job_count, bool):
            line += f" - {job_count:,} live job{'' if job_count == 1 else 's'}"
        ids = _strings(suggestion.get("ids")) or _strings([suggestion.get("id")])
        lines.append(f"{line}\n   company ids: {', '.join(ids)}")

    count = len(suggestions)
    header = (
        f'{count} compan{"y" if count == 1 else "ies"} matching "{query}". To '
        "search one company's jobs, pass all of its ids as the `companies` filter "
        "of jobo_job_search."
    )
    return "\n\n".join([header, "\n".join(lines)])


__all__ = [
    "format_company_suggestions",
    "format_compensation",
    "format_job_document",
    "format_job_summary",
    "format_location",
    "format_occupation",
    "format_search_results",
    "job_metadata",
    "job_places",
    "job_to_document",
]
