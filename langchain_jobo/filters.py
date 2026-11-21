"""Search and feed filters, and their mapping onto the API request bodies.

`JoboSearchFilters` carries one field per filter in the shared connector
contract (`Jobo.Connectors/contracts/filters.json`), so this package cannot
quietly miss a filter the other Jobo connectors support. `build_search_body`
maps the flat fields onto `POST /api/jobs/search`, where the include/exclude
filters are nested objects: a flat `exclude_skills` key in the body would be
ignored by the API rather than rejected.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

WorkModel = Literal["remote", "hybrid", "onsite"]
EmploymentType = Literal[
    "full-time", "part-time", "contract", "internship", "freelance", "temporary"
]
ExperienceLevel = Literal["intern", "entry", "mid", "senior", "lead", "executive"]
SearchSort = Literal["relevance", "discovered_asc", "discovered_desc"]
DescriptionFormat = Literal["markdown", "html", "enhanced", "enhanced_html"]
CompanySize = Literal[
    "1-10",
    "11-50",
    "51-200",
    "201-500",
    "501-1000",
    "1001-5000",
    "5001-10000",
    "10001+",
]
CompanyRevenue = Literal[
    "under-1m", "1m-10m", "10m-50m", "50m-100m", "100m-200m", "200m-1b", "over-1b"
]

#: The API's own page-size ceiling on search.
API_MAX_PAGE_SIZE = 100
#: The API's own batch-size ceiling on the feed.
API_MAX_BATCH_SIZE = 1000


def _clean(values: list[Any]) -> list[str] | None:
    """Trim, drop blanks and duplicates; `None` when nothing is left."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        if raw is None:
            continue
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out or None


def _to_list(value: Any) -> Any:
    """Coerce a list field's input.

    Lists pass through entry by entry (a value such as an industry name may
    contain a comma). A single string, which is what a model often hands a list
    field, is split on commas and newlines.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return _clean(value.replace("\n", ",").split(","))
    if isinstance(value, (list, tuple, set, frozenset)):
        return _clean(list(value))
    return value


def _to_enum_list(value: Any) -> Any:
    """Like `_to_list`, and lowercased: enum filters take lowercase keys."""
    values = _to_list(value)
    if isinstance(values, list):
        return _clean([entry.lower() for entry in values])
    return values


def _lower(value: Any) -> Any:
    return value.strip().lower() if isinstance(value, str) else value


StrList = Annotated[list[str] | None, BeforeValidator(_to_list)]


class JoboSearchFilters(BaseModel):
    """Filters for a Jobo job search. Every field is optional.

    Filters combine with AND; the values inside one list filter combine with
    OR. Enum filters take lowercase keys (`full-time`), not the display values
    jobs come back with (`Full-time`): an unrecognised value matches nothing
    rather than failing.
    """

    model_config = ConfigDict(extra="forbid")

    location: str | None = Field(
        default=None,
        description=(
            'Free-form location: a city, region or country, e.g. "Berlin, Germany".'
        ),
    )
    sources: StrList = Field(
        default=None,
        description=(
            "Restrict to these ATS provider ids, e.g. greenhouse, lever, workday."
        ),
    )
    companies: StrList = Field(
        default=None,
        description=(
            "Only jobs at these companies. Each value is a Jobo company id, a "
            'website domain ("stripe.com") or a whole company name ("Stripe"). '
            "Names are matched whole, never partially."
        ),
    )
    exclude_companies: StrList = Field(
        default=None,
        description="Leave out jobs at these companies (ids, domains or whole names).",
    )
    work_model: Annotated[list[WorkModel] | None, BeforeValidator(_to_enum_list)] = (
        Field(default=None, description="Work models: remote, hybrid, onsite.")
    )
    employment_type: Annotated[
        list[EmploymentType] | None, BeforeValidator(_to_enum_list)
    ] = Field(
        default=None,
        description=(
            "Employment types: full-time, part-time, contract, internship, "
            "freelance, temporary."
        ),
    )
    experience_level: Annotated[
        list[ExperienceLevel] | None, BeforeValidator(_to_enum_list)
    ] = Field(
        default=None,
        description="Experience levels: intern, entry, mid, senior, lead, executive.",
    )
    language: StrList = Field(
        default=None,
        description='Description languages as ISO 639-1 codes, e.g. "en", "de".',
    )
    skills: StrList = Field(
        default=None,
        description="Require one or more skills, e.g. Python, Kubernetes.",
    )
    exclude_skills: StrList = Field(
        default=None, description="Leave out jobs asking for any of these skills."
    )
    industries: StrList = Field(
        default=None,
        description=(
            "Restrict to companies in these industries. Matched on the whole "
            "industry name, case-insensitively."
        ),
    )
    exclude_industries: StrList = Field(
        default=None, description="Leave out companies in any of these industries."
    )
    company_categories: Annotated[list[str] | None, BeforeValidator(_to_enum_list)] = (
        Field(
            default=None,
            description=(
                "Hiring company's business model, e.g. b2b, b2c, saas, marketplace, "
                "agency, nonprofit, government."
            ),
        )
    )
    exclude_company_categories: Annotated[
        list[str] | None, BeforeValidator(_to_enum_list)
    ] = Field(
        default=None,
        description=(
            "Leave out companies with any of these business models, e.g. agency."
        ),
    )
    job_categories: StrList = Field(
        default=None,
        description='Occupation category names, e.g. "Healthcare". Case-insensitive.',
    )
    exclude_job_categories: StrList = Field(
        default=None, description="Leave out jobs in these occupation categories."
    )
    job_subcategories: StrList = Field(
        default=None,
        description='Occupation subcategory names, e.g. "Nursing". Case-insensitive.',
    )
    exclude_job_subcategories: StrList = Field(
        default=None, description="Leave out jobs in these occupation subcategories."
    )
    occupation_ids: StrList = Field(
        default=None,
        description=(
            'Occupation ids, e.g. "esco:8d3e8aaa-791b-4c75-a465-f3f827028f50". '
            "Ids, not names."
        ),
    )
    exclude_occupation_ids: StrList = Field(
        default=None, description="Leave out jobs with these occupation ids."
    )
    min_salary_usd: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Minimum annual salary in USD. Jobs without disclosed compensation "
            "are excluded when a salary bound is set. 0 means no bound."
        ),
    )
    max_salary_usd: int | None = Field(
        default=None,
        ge=0,
        description="Maximum annual salary in USD. 0 means no bound.",
    )
    posted_after: datetime | None = Field(
        default=None,
        description=(
            "Only jobs the employer posted on or after this ISO 8601 timestamp."
        ),
    )
    posted_before: datetime | None = Field(
        default=None,
        description=(
            "Only jobs the employer posted on or before this ISO 8601 timestamp."
        ),
    )
    discovered_after: datetime | None = Field(
        default=None,
        description=(
            "Only jobs Jobo first indexed on or after this ISO 8601 UTC timestamp."
        ),
    )
    discovered_before: datetime | None = Field(
        default=None,
        description=(
            "Only jobs Jobo first indexed on or before this ISO 8601 UTC timestamp."
        ),
    )
    search_description: bool | None = Field(
        default=None,
        description=(
            "true (default) matches the query against title, company, popular "
            "skills and summary; false restricts it to job titles."
        ),
    )
    sort: Annotated[SearchSort | None, BeforeValidator(_lower)] = Field(
        default=None,
        description=(
            "relevance (default), discovered_asc or discovered_desc (ordered by "
            "when Jobo first indexed the job)."
        ),
    )
    description_format: Annotated[DescriptionFormat | None, BeforeValidator(_lower)] = (
        Field(
            default=None,
            description=(
                "Format of each job's description: markdown (default), html, "
                "enhanced or enhanced_html."
            ),
        )
    )


class JoboSearchInput(JoboSearchFilters):
    """Arguments of the `jobo_job_search` tool."""

    query: str | None = Field(
        default=None,
        description=(
            "Free-text query, e.g. a job title or skill. Wrap it in double quotes "
            "for an exact phrase match on the job title."
        ),
    )
    page: int = Field(default=1, ge=1, description="Page number, starting at 1.")
    page_size: int = Field(
        default=5,
        ge=1,
        le=API_MAX_PAGE_SIZE,
        description=(
            "Jobs to return. Every returned job is billed, so ask only for what "
            "you need. The tool caps this at its configured maximum."
        ),
    )


#: Contract filter keys whose field is named differently in this package.
CONTRACT_KEY_TO_FIELD: dict[str, str] = {"q": "query"}

_ENUM_LIST_FIELDS: tuple[tuple[str, str], ...] = (
    ("work_model", "work_models"),
    ("employment_type", "employment_types"),
    ("experience_level", "experience_levels"),
    ("language", "languages"),
)

_INCLUDE_EXCLUDE_FIELDS: tuple[str, ...] = (
    "skills",
    "companies",
    "industries",
    "company_categories",
    "job_categories",
    "job_subcategories",
    "occupation_ids",
)

_DATE_FIELDS: tuple[str, ...] = (
    "posted_after",
    "posted_before",
    "discovered_after",
    "discovered_before",
)


def to_wire_timestamp(value: datetime) -> str:
    """Serialise a timestamp for the API. Naive values are sent as UTC."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def build_search_body(
    filters: JoboSearchFilters,
    *,
    query: str | None = None,
    page: int | None = None,
    page_size: int | None = None,
) -> dict[str, Any]:
    """Map filters onto a `POST /api/jobs/search` request body.

    Unset and empty values are left out, so an untouched filter never becomes
    one, and server defaults (`search_description: true`, `sort: relevance`,
    `description_format: markdown`) are not sent.
    """
    body: dict[str, Any] = {}

    text = (query or "").strip()
    if text:
        body["queries"] = [text]
    if filters.search_description is False:
        body["search_description"] = False

    location = (filters.location or "").strip()
    if location:
        body["locations"] = [location]

    if filters.sources:
        body["sources"] = list(filters.sources)

    for field, wire_key in _ENUM_LIST_FIELDS:
        values = getattr(filters, field)
        if values:
            body[wire_key] = list(values)

    for field in _INCLUDE_EXCLUDE_FIELDS:
        pair: dict[str, list[str]] = {}
        include = getattr(filters, field)
        exclude = getattr(filters, f"exclude_{field}")
        if include:
            pair["include"] = list(include)
        if exclude:
            pair["exclude"] = list(exclude)
        if pair:
            body[field] = pair

    salary: dict[str, int] = {}
    if filters.min_salary_usd:
        salary["min"] = filters.min_salary_usd
    if filters.max_salary_usd:
        salary["max"] = filters.max_salary_usd
    if salary:
        body["salary_usd"] = salary

    for field in _DATE_FIELDS:
        value = getattr(filters, field)
        if value is not None:
            body[field] = to_wire_timestamp(value)

    if filters.sort and filters.sort != "relevance":
        body["sort"] = filters.sort
    if filters.description_format and filters.description_format != "markdown":
        body["description_format"] = filters.description_format

    if page is not None:
        body["page"] = page
    if page_size is not None:
        body["page_size"] = page_size
    return body


class IncludeExclude(BaseModel):
    """An include/exclude filter. A bare list is read as `include`."""

    model_config = ConfigDict(extra="forbid")

    include: StrList = Field(default=None, description="Values to include.")
    exclude: StrList = Field(default=None, description="Values to exclude.")


def _to_include_exclude(value: Any) -> Any:
    if isinstance(value, (str, list, tuple, set, frozenset)):
        return {"include": value}
    return value


IncludeExcludeFilter = Annotated[
    IncludeExclude | None, BeforeValidator(_to_include_exclude)
]


class CompanySizeFilter(BaseModel):
    """Include/exclude filter on company headcount bands."""

    model_config = ConfigDict(extra="forbid")

    include: list[CompanySize] | None = None
    exclude: list[CompanySize] | None = None


class CompanyRevenueFilter(BaseModel):
    """Include/exclude filter on company annual revenue bands."""

    model_config = ConfigDict(extra="forbid")

    include: list[CompanyRevenue] | None = None
    exclude: list[CompanyRevenue] | None = None


class FeedLocation(BaseModel):
    """One feed location, matched at its most specific level.

    `country` is required whenever `city` or `region` is set. Use
    `{"country": "Remote"}` to match remote jobs.
    """

    model_config = ConfigDict(extra="forbid")

    country: str | None = None
    region: str | None = None
    city: str | None = None


class JoboFeedFilters(BaseModel):
    """Filters for `POST /api/jobs/feed`. Every field is optional.

    This is a different surface from search: list filters are plural and have
    `exclude_` siblings, and the occupation and company filters are
    `{"include": [...], "exclude": [...]}` objects. Unknown keys are rejected
    here because the feed itself ignores them silently, and a mistyped filter
    would otherwise stream (and bill) an unfiltered feed.
    """

    model_config = ConfigDict(extra="forbid")

    locations: list[FeedLocation] | None = None
    sources: StrList = None
    work_models: Annotated[list[WorkModel] | None, BeforeValidator(_to_enum_list)] = (
        None
    )
    employment_types: Annotated[
        list[EmploymentType] | None, BeforeValidator(_to_enum_list)
    ] = None
    experience_levels: Annotated[
        list[ExperienceLevel] | None, BeforeValidator(_to_enum_list)
    ] = None
    languages: StrList = None
    exclude_sources: StrList = None
    exclude_work_models: Annotated[
        list[WorkModel] | None, BeforeValidator(_to_enum_list)
    ] = None
    exclude_employment_types: Annotated[
        list[EmploymentType] | None, BeforeValidator(_to_enum_list)
    ] = None
    exclude_experience_levels: Annotated[
        list[ExperienceLevel] | None, BeforeValidator(_to_enum_list)
    ] = None
    exclude_languages: StrList = None
    companies: IncludeExcludeFilter = None
    job_categories: IncludeExcludeFilter = None
    job_subcategories: IncludeExcludeFilter = None
    occupation_ids: IncludeExcludeFilter = None
    industries: IncludeExcludeFilter = None
    company_primary_industries: IncludeExcludeFilter = None
    company_categories: IncludeExcludeFilter = None
    company_sizes: CompanySizeFilter | None = None
    company_revenues: CompanyRevenueFilter | None = None
    company_is_agency: bool | None = None
    posted_after: datetime | None = None
    include_reactivated: bool | None = None
    description_format: Annotated[DescriptionFormat | None, BeforeValidator(_lower)] = (
        None
    )
    updated_after: datetime | None = None


def build_feed_body(
    filters: JoboFeedFilters, *, batch_size: int | None = None
) -> dict[str, Any]:
    """Map filters onto the first `POST /api/jobs/feed` request of a scan.

    Continuation requests carry only the cursor: the cursor keeps the first
    request's filters, and filters sent beside it are ignored.
    """
    body: dict[str, Any] = {}
    for name in JoboFeedFilters.model_fields:
        value = getattr(filters, name)
        if value is None:
            continue
        if isinstance(value, datetime):
            body[name] = to_wire_timestamp(value)
        elif isinstance(value, BaseModel):
            dumped = value.model_dump(exclude_none=True)
            if dumped:
                body[name] = dumped
        elif isinstance(value, list):
            items = [
                item.model_dump(exclude_none=True)
                if isinstance(item, BaseModel)
                else item
                for item in value
            ]
            items = [item for item in items if item not in ({}, "")]
            if items:
                body[name] = items
        elif name == "description_format":
            if value != "markdown":
                body[name] = value
        else:
            body[name] = value
    if batch_size is not None:
        body["batch_size"] = batch_size
    return body


__all__ = [
    "API_MAX_BATCH_SIZE",
    "API_MAX_PAGE_SIZE",
    "CONTRACT_KEY_TO_FIELD",
    "CompanyRevenueFilter",
    "CompanySizeFilter",
    "FeedLocation",
    "IncludeExclude",
    "JoboFeedFilters",
    "JoboSearchFilters",
    "JoboSearchInput",
    "build_feed_body",
    "build_search_body",
    "to_wire_timestamp",
]
