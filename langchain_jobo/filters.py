"""Search and feed filters, and their mapping onto the API request bodies.

`JoboSearchFilters` carries one field per filter in the shared connector
contract (`Jobo.Connectors/contracts/filters.json`), so this package cannot
quietly miss a filter the other Jobo connectors support. `build_search_body`
maps the flat fields onto `POST /api/jobs/search`, where the include/exclude
filters are nested objects: a flat `exclude_skills` key in the body would be
ignored by the API rather than rejected.

Every filter set has two spellings: a pydantic model (`JoboSearchFilters`,
`JoboFeedFilters`) and a `TypedDict` for the same keys as a plain dict
(`JoboSearchFiltersDict`, `JoboFeedFiltersDict`). Both are checked when a
component is built, and both autocomplete in an editor.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypedDict

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    GetJsonSchemaHandler,
    create_model,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue

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
CompanyCategory = Literal[
    "b2b",
    "b2c",
    "b2b2c",
    "b2g",
    "saas",
    "marketplace",
    "e-commerce",
    "service-provider",
    "hardware",
    "agency",
    "nonprofit",
    "government",
    "education",
]
#: Filters whose values the API lists, with live job counts
#: (`GET /api/connectors/filter-options/{facet}`).
FilterFacet = Literal[
    "sources",
    "industries",
    "skills",
    "company_categories",
    "job_categories",
    "job_subcategories",
    "occupations",
]

#: The API's own page-size ceiling on search.
API_MAX_PAGE_SIZE = 100
#: The API's own batch-size ceiling on the feed.
API_MAX_BATCH_SIZE = 1000
#: Jobs the search tool asks for when the model names no page size.
DEFAULT_PAGE_SIZE = 5


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
    """Coerce the input of a list of ids, codes, keys or skills.

    Lists pass through entry by entry. A single string, which is what a model
    often hands a list field, is split on commas and newlines: these values
    do not contain either.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return _clean(value.replace("\n", ",").split(","))
    if isinstance(value, (list, tuple, set, frozenset)):
        return _clean(list(value))
    return value


def _to_name_list(value: Any) -> Any:
    """Coerce the input of a list of names.

    A name can contain a comma ("FedEx, Inc.", "Film, TV & Stage Production"),
    so a single string is one name, or one name per line. It is never split on
    commas.
    """
    if isinstance(value, str):
        return _clean(value.split("\n"))
    return _to_list(value)


def _to_enum_list(value: Any) -> Any:
    """Like `_to_list`, and lowercased: enum filters take lowercase keys."""
    values = _to_list(value)
    if isinstance(values, list):
        return _clean([entry.lower() for entry in values])
    return values


def _to_locations(value: Any) -> Any:
    """One location stays a string ("Berlin, Germany"); a list is cleaned."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (list, tuple, set, frozenset)):
        return _clean(list(value))
    return value


def _lower(value: Any) -> Any:
    return value.strip().lower() if isinstance(value, str) else value


def _blank_to_none(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip() or None
    return value


#: A list of ids, codes, keys or skills. A single string is split on commas.
StrList = Annotated[list[str] | None, BeforeValidator(_to_list)]
#: A list of names. A single string is one name: names can contain commas.
NameList = Annotated[list[str] | None, BeforeValidator(_to_name_list)]
CompanyCategoryList = Annotated[
    list[CompanyCategory] | None, BeforeValidator(_to_enum_list)
]


@dataclass(frozen=True)
class _SchemaMaximum:
    """Put a `maximum` in a field's JSON schema without enforcing it.

    The search tool reduces a larger `page_size` to its cap instead of
    failing the call, so the model must be told the cap but not held to it.
    """

    maximum: int

    def __get_pydantic_json_schema__(
        self, core_schema: Any, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        schema = handler(core_schema)
        schema["maximum"] = self.maximum
        return schema


class _SearchQuery(BaseModel):
    """The free-text query.

    Declared in a class of its own so that it is the first argument of the
    search tool: pydantic orders a model's fields base class first.
    """

    model_config = ConfigDict(extra="forbid")

    query: str | None = Field(
        default=None,
        description=(
            "Free-text query, e.g. a job title or skill. Wrap it in double quotes "
            "for an exact phrase match on the job title."
        ),
    )


class SearchFilterFields(BaseModel):
    """The search filters a model may set. Base of `JoboSearchFilters`.

    Filters combine with AND; the values inside one list filter combine with
    OR. Enum filters take lowercase keys (`full-time`), not the display values
    jobs come back with (`Full-time`): an unrecognised value is rejected here,
    because the API would answer it with zero results rather than an error.

    A list filter also accepts a single string. For ids, codes, enum keys and
    skills it is split on commas. For names (`companies`, `industries`,
    `job_categories`, `job_subcategories` and their `exclude_` twins) it is one
    value, because a name can contain a comma ("FedEx, Inc.").
    """

    model_config = ConfigDict(extra="forbid")

    location: Annotated[str | list[str] | None, BeforeValidator(_to_locations)] = Field(
        default=None,
        description=(
            'Free-form location: a city, region or country, e.g. "Berlin, '
            'Germany". A list matches jobs in any of the places. '
            "jobo_location_search resolves an ambiguous place name."
        ),
    )
    sources: StrList = Field(
        default=None,
        description=(
            "Restrict to these ATS provider ids, e.g. greenhouse, lever, workday. "
            "jobo_filter_values with facet `sources` lists them."
        ),
    )
    companies: NameList = Field(
        default=None,
        description=(
            "Only jobs at these companies. Each value is a Jobo company id, a "
            'website domain ("stripe.com") or a whole company name ("Stripe"). '
            "Names are matched whole, never partially: call jobo_company_search "
            "for a company's ids, which match exactly."
        ),
    )
    exclude_companies: NameList = Field(
        default=None,
        description=(
            "Leave out jobs at these companies (ids, domains or whole names). "
            "jobo_company_search returns the ids."
        ),
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
        description=(
            "Require one or more skills, e.g. Python, Kubernetes. "
            "jobo_filter_values with facet `skills` lists the names in use."
        ),
    )
    exclude_skills: StrList = Field(
        default=None, description="Leave out jobs asking for any of these skills."
    )
    industries: NameList = Field(
        default=None,
        description=(
            "Restrict to companies in these industries. Matched on the whole "
            "industry name, case-insensitively: get the names from "
            "jobo_filter_values with facet `industries`."
        ),
    )
    exclude_industries: NameList = Field(
        default=None, description="Leave out companies in any of these industries."
    )
    company_categories: CompanyCategoryList = Field(
        default=None,
        description=(
            "Hiring company's business model: b2b, b2c, b2b2c, b2g, saas, "
            "marketplace, e-commerce, service-provider, hardware, agency, "
            "nonprofit, government, education."
        ),
    )
    exclude_company_categories: CompanyCategoryList = Field(
        default=None,
        description=(
            "Leave out companies with any of these business models, e.g. agency."
        ),
    )
    job_categories: NameList = Field(
        default=None,
        description=(
            'Occupation category names, e.g. "Healthcare". Case-insensitive. '
            "jobo_filter_values with facet `job_categories` lists them."
        ),
    )
    exclude_job_categories: NameList = Field(
        default=None, description="Leave out jobs in these occupation categories."
    )
    job_subcategories: NameList = Field(
        default=None,
        description=(
            'Occupation subcategory names, e.g. "Nursing". Case-insensitive. '
            "jobo_filter_values with facet `job_subcategories` lists them."
        ),
    )
    exclude_job_subcategories: NameList = Field(
        default=None, description="Leave out jobs in these occupation subcategories."
    )
    occupation_ids: StrList = Field(
        default=None,
        description=(
            'Occupation ids, e.g. "esco:8d3e8aaa-791b-4c75-a465-f3f827028f50". '
            "Ids, not names: call jobo_filter_values with facet `occupations` "
            "and the occupation's name to get them."
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


class JoboSearchFilters(SearchFilterFields):
    """Filters for a Jobo job search. Every field is optional.

    Pass it as `filters=` to `JoboRetriever`. The same keys as a plain dict
    are typed by `JoboSearchFiltersDict`.

    Filters combine with AND; the values inside one list filter combine with
    OR. Enum filters take lowercase keys (`full-time`), not the display values
    jobs come back with (`Full-time`): an unrecognised value is rejected,
    because the API would answer it with zero results rather than an error.

    Example:
        ```python
        from langchain_jobo import JoboSearchFilters

        filters = JoboSearchFilters(
            location=["Berlin, Germany", "Munich, Germany"],
            work_model=["remote", "hybrid"],
            companies=["FedEx, Inc."],
            min_salary_usd=90000,
        )
        ```
    """

    description_format: Annotated[DescriptionFormat | None, BeforeValidator(_lower)] = (
        Field(
            default=None,
            description=(
                "Format of each job's description: markdown (default), html, "
                "enhanced or enhanced_html."
            ),
        )
    )


def _page_size_description(maximum: int) -> str:
    return (
        f"Jobs to return, at most {maximum}. Every returned job is billed, so "
        "ask only for what you need. A larger number is reduced to the maximum."
    )


class JoboSearchInput(SearchFilterFields, _SearchQuery):
    """Arguments of the `jobo_job_search` tool: `query`, the filters, the page.

    `description_format` is not among them. It only changes the raw jobs in
    the tool's artifact, never the text the model reads, so it is an option of
    `JoboSearchTool` itself.
    """

    page: int = Field(default=1, ge=1, description="Page number, starting at 1.")
    page_size: Annotated[int, _SchemaMaximum(API_MAX_PAGE_SIZE)] = Field(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        description=_page_size_description(API_MAX_PAGE_SIZE),
    )


_capped_schemas: dict[int, type[JoboSearchInput]] = {}


def search_input_schema(max_page_size: int) -> type[JoboSearchInput]:
    """`JoboSearchInput` as a tool capped at `max_page_size` shows it.

    The schema states the tool's real cap as `page_size`'s maximum and keeps
    the default page under it. A larger `page_size` still validates: the tool
    reduces it and says so, rather than failing the model's call.

    One class is made per cap and reused.
    """
    schema = _capped_schemas.get(max_page_size)
    if schema is None:
        schema = create_model(
            "JoboSearchInput",
            __base__=JoboSearchInput,
            __doc__=JoboSearchInput.__doc__,
            page_size=(
                Annotated[int, _SchemaMaximum(max_page_size)],
                Field(
                    default=min(DEFAULT_PAGE_SIZE, max_page_size),
                    ge=1,
                    description=_page_size_description(max_page_size),
                ),
            ),
        )
        _capped_schemas[max_page_size] = schema
    return schema


def is_stock_search_schema(schema: object) -> bool:
    """Whether `schema` is `JoboSearchInput` or one of its capped copies.

    A schema a caller wrote is theirs to keep; a stock one follows the tool's
    cap.
    """
    return schema is JoboSearchInput or schema in _capped_schemas.values()


#: Contract filter keys whose field is named differently in this package.
CONTRACT_KEY_TO_FIELD: dict[str, str] = {"q": "query"}
#: Contract filter keys that are an option of `JoboSearchTool`, not one of its
#: arguments: they change the artifact only, so the model is not shown them.
CONTRACT_KEYS_SET_ON_THE_TOOL: frozenset[str] = frozenset({"description_format"})

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
    filters: SearchFilterFields,
    *,
    query: str | None = None,
    page: int | None = None,
    page_size: int | None = None,
    description_format: DescriptionFormat | None = None,
) -> dict[str, Any]:
    """Map filters onto a `POST /api/jobs/search` request body.

    Unset and empty values are left out, so an untouched filter never becomes
    one, and server defaults (`search_description: true`, `sort: relevance`,
    `description_format: markdown`) are not sent.

    Args:
        filters: The filters, as `JoboSearchFilters` or a tool's arguments.
        query: Free-text query.
        page: Page number.
        page_size: Jobs per page.
        description_format: Format of each job's description. Defaults to the
            one on `filters` when that is a `JoboSearchFilters`.
    """
    body: dict[str, Any] = {}

    text = (query or "").strip()
    if text:
        body["queries"] = [text]
    if filters.search_description is False:
        body["search_description"] = False

    location = filters.location
    locations = _clean([location] if isinstance(location, str) else location or [])
    if locations:
        body["locations"] = locations

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
    if description_format is None and isinstance(filters, JoboSearchFilters):
        description_format = filters.description_format
    if description_format and description_format != "markdown":
        body["description_format"] = description_format

    if page is not None:
        body["page"] = page
    if page_size is not None:
        body["page_size"] = page_size
    return body


class IncludeExclude(BaseModel):
    """An include/exclude feed filter. A bare list is read as `include`.

    Values are taken whole: a single string is one value, never split on
    commas, because company and industry names can contain one.

    Example:
        ```python
        from langchain_jobo import IncludeExclude

        IncludeExclude(include=["fedex.com", "UPS"], exclude=["DHL"])
        ```
    """

    model_config = ConfigDict(extra="forbid")

    include: NameList = Field(default=None, description="Values to include.")
    exclude: NameList = Field(default=None, description="Values to exclude.")


def _to_include_exclude(value: Any) -> Any:
    if isinstance(value, (str, list, tuple, set, frozenset)):
        return {"include": value}
    return value


def _to_id_include_exclude(value: Any) -> Any:
    """Like `_to_include_exclude`, and a single string of ids is split."""
    value = _to_include_exclude(value)
    if isinstance(value, dict):
        return {
            key: _to_list(entry) if isinstance(entry, str) else entry
            for key, entry in value.items()
        }
    return value


IncludeExcludeFilter = Annotated[
    IncludeExclude | None, BeforeValidator(_to_include_exclude)
]
IdIncludeExcludeFilter = Annotated[
    IncludeExclude | None, BeforeValidator(_to_id_include_exclude)
]


class CompanyCategoryFilter(BaseModel):
    """Include/exclude feed filter on the hiring company's business model."""

    model_config = ConfigDict(extra="forbid")

    include: CompanyCategoryList = None
    exclude: CompanyCategoryList = None


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


_OptionalText = Annotated[str | None, BeforeValidator(_blank_to_none)]


class FeedLocation(BaseModel):
    """One feed location, matched at its most specific level.

    `country` is required: alone it matches the whole country, and a `region`
    or `city` narrows it. Use `FeedLocation(country="Remote")` to match remote
    jobs. A location without a country is rejected here, as the API rejects
    it, before any request is made.

    Example:
        ```python
        from langchain_jobo import FeedLocation

        FeedLocation(country="United States", region="Ohio", city="Columbus")
        ```
    """

    model_config = ConfigDict(extra="forbid")

    country: _OptionalText = Field(
        default=None,
        description='Country name or ISO code, e.g. "United States" or "US".',
    )
    region: _OptionalText = Field(
        default=None, description='State, province or region, e.g. "Ohio".'
    )
    city: _OptionalText = Field(default=None, description='City, e.g. "Columbus".')

    @model_validator(mode="after")
    def _require_country(self) -> FeedLocation:
        if self.country:
            return self
        if self.city or self.region:
            msg = (
                "country is required when city or region is set (a name or ISO "
                'code, e.g. "United States" or "US").'
            )
        else:
            msg = (
                "a location needs a country (a name or ISO code, e.g. "
                '"United States" or "US"; "Remote" matches remote jobs).'
            )
        raise ValueError(msg)


class JoboFeedFilters(BaseModel):
    """Filters for `POST /api/jobs/feed`. Every field is optional.

    This is a different surface from search: list filters are plural and have
    `exclude_` siblings, and the occupation and company filters are
    `{"include": [...], "exclude": [...]}` objects. Unknown keys are rejected
    here because the feed itself ignores them silently, and a mistyped filter
    would otherwise stream (and bill) an unfiltered feed.

    The same keys as a plain dict are typed by `JoboFeedFiltersDict`.

    Example:
        ```python
        from langchain_jobo import FeedLocation, IncludeExclude, JoboFeedFilters

        filters = JoboFeedFilters(
            locations=[FeedLocation(country="United States", region="Ohio")],
            work_models=["remote"],
            job_categories=IncludeExclude(include=["Healthcare"]),
        )
        ```
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
    occupation_ids: IdIncludeExcludeFilter = None
    industries: IncludeExcludeFilter = None
    company_primary_industries: IncludeExcludeFilter = None
    company_categories: Annotated[
        CompanyCategoryFilter | None, BeforeValidator(_to_include_exclude)
    ] = None
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


# --------------------------------------------------------------- dict spellings
#
# The same filters as plain dicts. Every key is optional, and `None` means
# "not set". The models above validate them at run time; these exist so an
# editor completes the keys and a type checker accepts
# `JoboRetriever(filters={"work_model": ["remote"]})`.


class IncludeExcludeDict(TypedDict, total=False):
    """`IncludeExclude` as a dict."""

    include: list[str] | None
    exclude: list[str] | None


class CompanyCategoryFilterDict(TypedDict, total=False):
    """`CompanyCategoryFilter` as a dict."""

    include: list[CompanyCategory] | None
    exclude: list[CompanyCategory] | None


class CompanySizeFilterDict(TypedDict, total=False):
    """`CompanySizeFilter` as a dict."""

    include: list[CompanySize] | None
    exclude: list[CompanySize] | None


class CompanyRevenueFilterDict(TypedDict, total=False):
    """`CompanyRevenueFilter` as a dict."""

    include: list[CompanyRevenue] | None
    exclude: list[CompanyRevenue] | None


class FeedLocationDict(TypedDict, total=False):
    """`FeedLocation` as a dict. `country` is required by the API."""

    country: str | None
    region: str | None
    city: str | None


class JoboSearchFiltersDict(TypedDict, total=False):
    """`JoboSearchFilters` as a dict: the keys `JoboRetriever(filters=...)` takes.

    Timestamps may be `datetime`s or ISO 8601 strings.
    """

    location: str | list[str] | None
    sources: list[str] | None
    companies: list[str] | None
    exclude_companies: list[str] | None
    work_model: list[WorkModel] | None
    employment_type: list[EmploymentType] | None
    experience_level: list[ExperienceLevel] | None
    language: list[str] | None
    skills: list[str] | None
    exclude_skills: list[str] | None
    industries: list[str] | None
    exclude_industries: list[str] | None
    company_categories: list[CompanyCategory] | None
    exclude_company_categories: list[CompanyCategory] | None
    job_categories: list[str] | None
    exclude_job_categories: list[str] | None
    job_subcategories: list[str] | None
    exclude_job_subcategories: list[str] | None
    occupation_ids: list[str] | None
    exclude_occupation_ids: list[str] | None
    min_salary_usd: int | None
    max_salary_usd: int | None
    posted_after: datetime | str | None
    posted_before: datetime | str | None
    discovered_after: datetime | str | None
    discovered_before: datetime | str | None
    search_description: bool | None
    sort: SearchSort | None
    description_format: DescriptionFormat | None


class JoboFeedFiltersDict(TypedDict, total=False):
    """`JoboFeedFilters` as a dict: the keys `JoboFeedLoader(filters=...)` takes.

    An include/exclude filter is `{"include": [...], "exclude": [...]}`; a bare
    list is read as `include`. Timestamps may be `datetime`s or ISO 8601
    strings.
    """

    locations: list[FeedLocationDict] | None
    sources: list[str] | None
    work_models: list[WorkModel] | None
    employment_types: list[EmploymentType] | None
    experience_levels: list[ExperienceLevel] | None
    languages: list[str] | None
    exclude_sources: list[str] | None
    exclude_work_models: list[WorkModel] | None
    exclude_employment_types: list[EmploymentType] | None
    exclude_experience_levels: list[ExperienceLevel] | None
    exclude_languages: list[str] | None
    companies: IncludeExcludeDict | list[str] | None
    job_categories: IncludeExcludeDict | list[str] | None
    job_subcategories: IncludeExcludeDict | list[str] | None
    occupation_ids: IncludeExcludeDict | list[str] | None
    industries: IncludeExcludeDict | list[str] | None
    company_primary_industries: IncludeExcludeDict | list[str] | None
    company_categories: CompanyCategoryFilterDict | list[CompanyCategory] | None
    company_sizes: CompanySizeFilterDict | None
    company_revenues: CompanyRevenueFilterDict | None
    company_is_agency: bool | None
    posted_after: datetime | str | None
    include_reactivated: bool | None
    description_format: DescriptionFormat | None
    updated_after: datetime | str | None


def merge_search_filters(
    base: JoboSearchFilters,
    override: JoboSearchFilters | JoboSearchFiltersDict | None,
) -> JoboSearchFilters:
    """Lay `override` over `base`: its keys win, the others are kept.

    A key set to `None` in `override` removes that filter. The result is
    validated like any other filter set, so an unknown key is rejected.
    """
    if override is None:
        return base
    changes = (
        override.model_dump(exclude_unset=True)
        if isinstance(override, JoboSearchFilters)
        else dict(override)
    )
    if not changes:
        return base
    return JoboSearchFilters.model_validate(
        {**base.model_dump(exclude_unset=True), **changes}
    )


__all__ = [
    "API_MAX_BATCH_SIZE",
    "API_MAX_PAGE_SIZE",
    "CONTRACT_KEYS_SET_ON_THE_TOOL",
    "CONTRACT_KEY_TO_FIELD",
    "DEFAULT_PAGE_SIZE",
    "CompanyCategory",
    "CompanyCategoryFilter",
    "CompanyCategoryFilterDict",
    "CompanyRevenue",
    "CompanyRevenueFilter",
    "CompanyRevenueFilterDict",
    "CompanySize",
    "CompanySizeFilter",
    "CompanySizeFilterDict",
    "DescriptionFormat",
    "EmploymentType",
    "ExperienceLevel",
    "FeedLocation",
    "FeedLocationDict",
    "FilterFacet",
    "IncludeExclude",
    "IncludeExcludeDict",
    "JoboFeedFilters",
    "JoboFeedFiltersDict",
    "JoboSearchFilters",
    "JoboSearchFiltersDict",
    "JoboSearchInput",
    "SearchFilterFields",
    "SearchSort",
    "WorkModel",
    "build_feed_body",
    "build_search_body",
    "is_stock_search_schema",
    "merge_search_filters",
    "search_input_schema",
    "to_wire_timestamp",
]
