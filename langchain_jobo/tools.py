"""LangChain tools for the Jobo jobs API."""

from __future__ import annotations

import re
import time
from datetime import date
from typing import Annotated, Any, Literal
from urllib.parse import quote
from uuid import UUID

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    create_model,
    model_validator,
)

from langchain_jobo._base import JoboSettings, given
from langchain_jobo._client import SANDBOX_KEY_PREFIX, JoboResponse
from langchain_jobo.errors import (
    NOT_AVAILABLE_IN_SANDBOX,
    JoboError,
    JoboNotFoundError,
)
from langchain_jobo.filters import (
    API_MAX_PAGE_SIZE,
    CompanySize,
    DescriptionFormat,
    FilterFacet,
    JoboSearchInput,
    NameList,
    StrList,
    _SchemaMaximum,
    _to_enum_list,
    build_search_body,
    is_stock_search_schema,
    search_input_schema,
)
from langchain_jobo.formatting import (
    format_company_suggestions,
    format_filter_values,
    format_job_document,
    format_location_suggestions,
    format_occupation_matches,
    format_search_results,
)
from langchain_jobo.signals import (
    NO_SIGNALS_IN_SANDBOX,
    SIGNAL_MAX_LIMIT,
    SignalSeniority,
    SignalType,
    build_signals_request,
    format_signals,
    signal_list,
    signals_error_message,
    to_signal_instant,
)

DEFAULT_MAX_PAGE_SIZE = 25
_MAX_COMPANY_SUGGESTIONS = 20
_MAX_FILTER_VALUES = 50
_MAX_LOCATION_SUGGESTIONS = 10
#: The occupation taxonomy changes with a release, not a minute.
_TAXONOMY_TTL_SECONDS = 600.0

_Answer = tuple[str, dict[str, Any]]

_SANDBOX_ONLY_SERVES = (
    f"a sandbox API key ({SANDBOX_KEY_PREFIX}...) serves job search and job "
    "details only"
)


def _tool_error(error: JoboError) -> ToolException:
    """Wrap an API failure so `handle_tool_error` can hand it to the model."""
    return ToolException(str(error))


def _refused_by_sandbox(error: JoboError) -> bool:
    """Whether a sandbox key called an endpoint the sandbox does not serve."""
    return isinstance(error, JoboNotFoundError) and error.not_available_in_sandbox


class _JoboTool(BaseTool, JoboSettings):
    """What the Jobo tools share: connection settings and the artifact format.

    Each tool spells out its own `__init__` all the same. A type checker
    builds a pydantic model's constructor from its fields unless the class
    itself defines one, and that built one would take `api_key` only as a
    `SecretStr`.
    """

    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    def __init__(
        self,
        *,
        api_key: SecretStr | str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client: httpx.Client | None = None,
        http_async_client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the tool.

        Args:
            api_key: Jobo API key. Read from the `JOBO_API_KEY` environment
                variable when not passed.
            base_url: API base URL. Defaults to `https://connect.jobo.world`.
            timeout: Response timeout in seconds. Defaults to 30.
            max_retries: Retries after the first attempt, for 429, 500 and 503
                responses and failures to connect. Defaults to 2.
            http_client: Optional `httpx.Client` to send sync requests with.
            http_async_client: Optional `httpx.AsyncClient` for async requests.
            **kwargs: Other fields of the tool and of LangChain's `BaseTool`,
                e.g. `handle_tool_error`, `tags` or `metadata`. An unknown
                name is rejected.

        Raises:
            JoboConfigurationError: No API key was passed or found.
        """
        super().__init__(
            **given(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
                http_client=http_client,
                http_async_client=http_async_client,
            ),
            **kwargs,
        )


class JoboSearchTool(_JoboTool):
    """Search live job listings from employer career sites and ATS platforms.

    Calls `POST /api/jobs/search`. Search is billed per job returned, so the
    tool asks for a small page by default (5) and never requests more than
    `max_page_size` jobs in one call, whatever the model asks for.

    Setup:
        Install `langchain-jobo` and set the `JOBO_API_KEY` environment variable.

        ```bash
        pip install -U langchain-jobo
        export JOBO_API_KEY="jbe_live_..."
        ```

    Instantiate:
        ```python
        from langchain_jobo import JoboSearchTool

        tool = JoboSearchTool(max_page_size=10)
        ```

    Invoke directly with args:
        ```python
        tool.invoke(
            {"query": "backend engineer", "work_model": ["remote"], "page_size": 3}
        )
        ```

    Invoke with a tool call to also get the raw jobs as the artifact:
        ```python
        message = tool.invoke(
            {
                "type": "tool_call",
                "id": "1",
                "name": "jobo_job_search",
                "args": {"query": "nurse", "location": "Ohio"},
            }
        )
        message.content  # compact, citeable text
        message.artifact["jobs"]  # the raw job objects
        ```
    """

    name: str = "jobo_job_search"
    description: str = (
        "Search live job listings sourced directly from employer career sites and "
        "applicant tracking systems. Filter by free-text query, location, work "
        "model, employment type, experience level, skills, industries, companies, "
        "salary and posting date. Returns each job's id, title, company, location, "
        "work model, salary when the posting states one, posted date and listing "
        "URL. Call jobo_job_details with a job id for the full posting. Every "
        "returned job is billed, so keep page_size small."
    )
    args_schema: type[BaseModel] = JoboSearchInput

    max_page_size: int = Field(
        default=DEFAULT_MAX_PAGE_SIZE,
        ge=1,
        le=API_MAX_PAGE_SIZE,
        description=(
            "Hard cap on jobs requested per call. A larger page_size from the "
            "model is reduced to this."
        ),
    )
    description_format: DescriptionFormat = Field(
        default="markdown",
        description=(
            "Format of each job's description in the artifact. The model never "
            "sees descriptions in search results, so it is not a tool argument."
        ),
    )

    def __init__(
        self,
        *,
        api_key: SecretStr | str | None = None,
        max_page_size: int | None = None,
        description_format: DescriptionFormat | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client: httpx.Client | None = None,
        http_async_client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the search tool.

        Args:
            api_key: Jobo API key. Read from the `JOBO_API_KEY` environment
                variable when not passed.
            max_page_size: Hard cap on jobs requested per call, 1-100.
                Defaults to 25. A larger `page_size` from the model is reduced
                to it, and the tool's schema states it as the maximum.
            description_format: Format of each job's description in the
                artifact: `markdown` (default), `html`, `enhanced` or
                `enhanced_html`.
            base_url: API base URL. Defaults to `https://connect.jobo.world`.
            timeout: Response timeout in seconds. Defaults to 30.
            max_retries: Retries after the first attempt, for 429 responses,
                503 responses with `Retry-After` and failures to connect. A
                500 is not retried: search may have billed before it failed.
                Defaults to 2.
            http_client: Optional `httpx.Client` to send sync requests with.
            http_async_client: Optional `httpx.AsyncClient` for async requests.
            **kwargs: Other fields of LangChain's `BaseTool`, e.g.
                `handle_tool_error`, `tags` or `metadata`. An unknown name is
                rejected.

        Raises:
            JoboConfigurationError: No API key was passed or found.
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
            http_async_client=http_async_client,
            **given(max_page_size=max_page_size, description_format=description_format),
            **kwargs,
        )

    @model_validator(mode="after")
    def _state_the_cap_in_the_schema(self) -> JoboSearchTool:
        # Also when the tool is rebuilt from another one's fields with a new
        # cap, as `configurable_fields` does: the schema follows the cap.
        if is_stock_search_schema(self.args_schema):
            self.args_schema = search_input_schema(self.max_page_size)
        return self

    def _prepare(self, kwargs: dict[str, Any]) -> tuple[dict[str, Any], int | None]:
        args = JoboSearchInput.model_validate(kwargs)
        page_size = min(args.page_size, self.max_page_size)
        capped = page_size if page_size < args.page_size else None
        body = build_search_body(
            args,
            query=args.query,
            page=args.page,
            page_size=page_size,
            description_format=self.description_format,
        )
        # Facets are left out of the answer: do not have the API compute them.
        body["include_facets"] = []
        return body, capped

    @staticmethod
    def _finish(response: JoboResponse, capped: int | None) -> _Answer:
        payload = response.data if isinstance(response.data, dict) else {}
        artifact = {
            "jobs": payload.get("jobs") or [],
            "total": payload.get("total"),
            "page": payload.get("page"),
            "page_size": payload.get("page_size"),
            "total_pages": payload.get("total_pages"),
            "warnings": payload.get("warnings") or [],
            "usage": response.usage.as_dict(),
        }
        return format_search_results(payload, capped_page_size=capped), artifact

    def _run(
        self,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> _Answer:
        body, capped = self._prepare(kwargs)
        try:
            response = self._api().request("POST", "/api/jobs/search", json=body)
        except JoboError as error:
            raise _tool_error(error) from error
        return self._finish(response, capped)

    async def _arun(
        self,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> _Answer:
        body, capped = self._prepare(kwargs)
        try:
            response = await self._api().arequest("POST", "/api/jobs/search", json=body)
        except JoboError as error:
            raise _tool_error(error) from error
        return self._finish(response, capped)


class JoboJobDetailsInput(BaseModel):
    """Arguments of the `jobo_job_details` tool."""

    job_id: str = Field(
        description="The Jobo job id (a UUID), as returned by jobo_job_search."
    )


def _is_job_not_found(error: JoboError) -> bool:
    """Whether the API itself said that no job has this id.

    `GET /api/jobs/{id}` answers an unknown id with the problem "Job not
    found" and no `code`. Every other 404 (a wrong `base_url`, a proxy, an
    endpoint the sandbox lacks) says something is misconfigured, and must
    reach the developer instead of being read as "no such job".
    """
    if not isinstance(error, JoboNotFoundError):
        return False
    if error.code == "job_not_found":
        return True
    title = error.body.get("title") if isinstance(error.body, dict) else None
    return isinstance(title, str) and title.strip().lower() == "job not found"


class JoboJobDetailsTool(_JoboTool):
    """Fetch one job listing by id and render it as readable prose.

    Calls `GET /api/jobs/{id}`, which is free: it is never billed, though it
    counts toward the search rate limit.

    An id that is not a UUID, or that the API says does not exist, comes back
    as a plain answer the model can act on. Any other failure, including a 404
    that is not the API's own "job not found", raises.

    Instantiate:
        ```python
        from langchain_jobo import JoboJobDetailsTool

        tool = JoboJobDetailsTool()
        ```

    Invoke:
        ```python
        tool.invoke({"job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"})
        ```
    """

    name: str = "jobo_job_details"
    description: str = (
        "Get the full details of one job listing by its Jobo job id: company, "
        "location, compensation, responsibilities, qualifications, benefits, the "
        "employer's description and the listing and apply URLs. Use the id "
        "returned by jobo_job_search."
    )
    args_schema: type[BaseModel] = JoboJobDetailsInput

    include_description: bool = Field(
        default=True,
        description=(
            "Append the employer's full description after the structured fields."
        ),
    )
    description_format: DescriptionFormat = Field(
        default="markdown",
        description="Format of the job's description as returned by the API.",
    )

    def __init__(
        self,
        *,
        api_key: SecretStr | str | None = None,
        include_description: bool | None = None,
        description_format: DescriptionFormat | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client: httpx.Client | None = None,
        http_async_client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the details tool.

        Args:
            api_key: Jobo API key. Read from the `JOBO_API_KEY` environment
                variable when not passed.
            include_description: Append the employer's full description after
                the structured fields. Defaults to true.
            description_format: Format of the description: `markdown`
                (default), `html`, `enhanced` or `enhanced_html`.
            base_url: API base URL. Defaults to `https://connect.jobo.world`.
            timeout: Response timeout in seconds. Defaults to 30.
            max_retries: Retries after the first attempt, for 429, 500 and 503
                responses and failures to connect. Defaults to 2.
            http_client: Optional `httpx.Client` to send sync requests with.
            http_async_client: Optional `httpx.AsyncClient` for async requests.
            **kwargs: Other fields of LangChain's `BaseTool`, e.g.
                `handle_tool_error`, `tags` or `metadata`. An unknown name is
                rejected.

        Raises:
            JoboConfigurationError: No API key was passed or found.
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
            http_async_client=http_async_client,
            **given(
                include_description=include_description,
                description_format=description_format,
            ),
            **kwargs,
        )

    def _path_and_params(self, job_id: str) -> tuple[str, dict[str, str]] | None:
        """The request for a job id, or `None` when it cannot be one."""
        try:
            canonical = str(UUID(job_id.strip()))
        except ValueError:
            return None
        params: dict[str, str] = {}
        if self.description_format != "markdown":
            params["description_format"] = self.description_format
        return f"/api/jobs/{quote(canonical, safe='')}", params

    def _finish(self, response: JoboResponse) -> _Answer:
        job = response.data if isinstance(response.data, dict) else {}
        content = format_job_document(job, include_description=self.include_description)
        return content, job

    @staticmethod
    def _not_an_id(job_id: str) -> _Answer:
        return (
            f'"{job_id.strip()}" is not a Jobo job id. Job ids are UUIDs: the `ID` '
            "values jobo_job_search returns.",
            {},
        )

    @staticmethod
    def _recover(job_id: str, error: JoboError) -> _Answer:
        if _is_job_not_found(error):
            return (
                f'No job exists with id "{job_id.strip()}". It may have closed. '
                "Job ids are the `ID` values jobo_job_search returns.",
                {},
            )
        raise _tool_error(error) from error

    def _run(
        self,
        job_id: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> _Answer:
        request = self._path_and_params(job_id)
        if request is None:
            return self._not_an_id(job_id)
        path, params = request
        try:
            response = self._api().request("GET", path, params=params)
        except JoboError as error:
            return self._recover(job_id, error)
        return self._finish(response)

    async def _arun(
        self,
        job_id: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> _Answer:
        request = self._path_and_params(job_id)
        if request is None:
            return self._not_an_id(job_id)
        path, params = request
        try:
            response = await self._api().arequest("GET", path, params=params)
        except JoboError as error:
            return self._recover(job_id, error)
        return self._finish(response)


class JoboCompanySearchInput(BaseModel):
    """Arguments of the `jobo_company_search` tool."""

    query: str = Field(
        min_length=2,
        description=(
            "The start of a company name or of its website domain, at least 2 "
            'characters, e.g. "strip" or "stripe.com".'
        ),
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=_MAX_COMPANY_SUGGESTIONS,
        description="Maximum number of companies to return.",
    )


class JoboCompanySearchTool(_JoboTool):
    """Look up companies in Jobo's catalog to get the ids the search filter takes.

    Calls `GET /api/connectors/companies/suggest`, the company typeahead the
    other Jobo connectors use. It is rate limited but never billed.

    The `companies` filter of `JoboSearchTool` matches a company *name* whole,
    and a name also matches every other company called that. This tool returns
    each company's ids, which match exactly.

    The sandbox does not serve this lookup. With a sandbox key the tool does
    not fail: it tells the model to filter on the company's name or domain.

    Instantiate:
        ```python
        from langchain_jobo import JoboCompanySearchTool

        tool = JoboCompanySearchTool()
        ```

    Invoke:
        ```python
        tool.invoke({"query": "stripe"})
        ```
    """

    name: str = "jobo_company_search"
    description: str = (
        "Look up companies by the start of their name or website domain. Returns "
        "each matching company's name, domain, country, live job count and Jobo "
        "company ids. Pass all of a company's ids as the `companies` filter of "
        "jobo_job_search to search that company's jobs exactly. Free to call."
    )
    args_schema: type[BaseModel] = JoboCompanySearchInput

    def __init__(
        self,
        *,
        api_key: SecretStr | str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client: httpx.Client | None = None,
        http_async_client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the company lookup tool.

        Args:
            api_key: Jobo API key. Read from the `JOBO_API_KEY` environment
                variable when not passed.
            base_url: API base URL. Defaults to `https://connect.jobo.world`.
            timeout: Response timeout in seconds. Defaults to 30.
            max_retries: Retries after the first attempt, for 429, 500 and 503
                responses and failures to connect. Defaults to 2.
            http_client: Optional `httpx.Client` to send sync requests with.
            http_async_client: Optional `httpx.AsyncClient` for async requests.
            **kwargs: Other fields of LangChain's `BaseTool`, e.g.
                `handle_tool_error`, `tags` or `metadata`. An unknown name is
                rejected.

        Raises:
            JoboConfigurationError: No API key was passed or found.
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
            http_async_client=http_async_client,
            **kwargs,
        )

    @staticmethod
    def _finish(query: str, response: JoboResponse) -> _Answer:
        payload = response.data if isinstance(response.data, dict) else {}
        raw = payload.get("suggestions")
        suggestions = [item for item in raw if isinstance(item, dict)] if raw else []
        content = format_company_suggestions(query, suggestions)
        return content, {"suggestions": suggestions}

    @staticmethod
    def _recover(error: JoboError) -> _Answer:
        if _refused_by_sandbox(error):
            return (
                f"Companies cannot be looked up here: {_SANDBOX_ONLY_SERVES}. "
                "Do not call this tool again. Pass the company's name or website "
                "domain as the `companies` filter of jobo_job_search instead.",
                {"suggestions": [], "unavailable": NOT_AVAILABLE_IN_SANDBOX},
            )
        raise _tool_error(error) from error

    def _run(
        self,
        query: str,
        limit: int = 5,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> _Answer:
        params = {"q": query.strip(), "limit": limit}
        try:
            response = self._api().request(
                "GET", "/api/connectors/companies/suggest", params=params
            )
        except JoboError as error:
            return self._recover(error)
        return self._finish(query.strip(), response)

    async def _arun(
        self,
        query: str,
        limit: int = 5,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> _Answer:
        params = {"q": query.strip(), "limit": limit}
        try:
            response = await self._api().arequest(
                "GET", "/api/connectors/companies/suggest", params=params
            )
        except JoboError as error:
            return self._recover(error)
        return self._finish(query.strip(), response)


class JoboFilterValuesInput(BaseModel):
    """Arguments of the `jobo_filter_values` tool."""

    facet: FilterFacet = Field(
        description=(
            "Which filter's values to list: sources (ATS providers), industries, "
            "skills, company_categories, job_categories, job_subcategories or "
            "occupations (for the occupation_ids filter)."
        )
    )
    query: str | None = Field(
        default=None,
        description=(
            "Optional text to narrow the list. For occupations it is searched "
            'in the occupation names, e.g. "nurse". For the other facets a '
            'value must start with it, e.g. "soft". Leave it out for the most '
            "common values."
        ),
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=_MAX_FILTER_VALUES,
        description="Maximum number of values to return.",
    )


def _flatten_taxonomy(data: Any) -> list[dict[str, str]]:
    """`GET /api/occupations` as one row per occupation."""
    rows: list[dict[str, str]] = []
    categories = data.get("categories") if isinstance(data, dict) else None
    for category in categories if isinstance(categories, list) else []:
        if not isinstance(category, dict):
            continue
        subcategories = category.get("subcategories")
        for subcategory in subcategories if isinstance(subcategories, list) else []:
            if not isinstance(subcategory, dict):
                continue
            occupations = subcategory.get("occupations")
            for occupation in occupations if isinstance(occupations, list) else []:
                if not isinstance(occupation, dict):
                    continue
                occupation_id, name = occupation.get("id"), occupation.get("name")
                if isinstance(occupation_id, str) and isinstance(name, str):
                    rows.append(
                        {
                            "id": occupation_id,
                            "name": name,
                            "category": str(category.get("name") or ""),
                            "subcategory": str(subcategory.get("name") or ""),
                        }
                    )
    return rows


#: The words of an occupation name, for whole-word matching.
_WORD = re.compile(r"[a-z0-9]+")


def _match_occupations(
    rows: list[dict[str, str]], query: str, limit: int
) -> list[dict[str, str]]:
    """Occupations whose name holds every word of `query`, best match first.

    A whole-name match leads, then names that hold every query word as a
    whole word ("nurse" finds Registered Nurse before Nursery School
    Teacher), then names that start with the query, then names that contain
    its words, then occupations whose category or subcategory does.
    """
    needle = query.lower()
    words = needle.split()
    ranked: list[tuple[int, int, dict[str, str]]] = []
    for index, row in enumerate(rows):
        name = row["name"].lower()
        name_words = set(_WORD.findall(name))
        if name == needle:
            rank = 0
        elif all(word in name_words for word in words):
            rank = 1
        elif name.startswith(needle):
            rank = 2
        elif all(word in name for word in words):
            rank = 3
        elif all(
            word in f"{name} {row['subcategory']} {row['category']}".lower()
            for word in words
        ):
            rank = 4
        else:
            continue
        ranked.append((rank, index, row))
    ranked.sort(key=lambda entry: entry[:2])
    return [row for _, _, row in ranked[:limit]]


class JoboFilterValuesTool(_JoboTool):
    """List the values a search filter accepts, with live job counts.

    A filter value the index does not hold matches no jobs, and the API does
    not say so. This tool lets a model look the values up first: ATS
    `sources`, `industries`, `skills`, `company_categories`, `job_categories`,
    `job_subcategories` and `occupations` (the ids `occupation_ids` takes).

    Calls `GET /api/connectors/filter-options/{facet}`, the lookup the other
    Jobo connectors use. Occupations searched by name are read from
    `GET /api/occupations` instead, because the facet lookup matches
    occupation ids, not names; the taxonomy is cached on the tool for ten
    minutes. Both are rate limited but never billed.

    The sandbox serves neither. With a sandbox key the tool does not fail: it
    tells the model to search without the lookup.

    Instantiate:
        ```python
        from langchain_jobo import JoboFilterValuesTool

        tool = JoboFilterValuesTool()
        ```

    Invoke:
        ```python
        tool.invoke({"facet": "industries", "query": "soft"})
        tool.invoke({"facet": "occupations", "query": "nurse"})
        ```
    """

    name: str = "jobo_filter_values"
    description: str = (
        "List the values a jobo_job_search filter accepts, each with its number "
        "of live jobs: ATS sources, industries, skills, company categories, job "
        "categories, job subcategories and occupations. Call it before "
        "filtering on one of these, because a value the index does not hold "
        "matches no jobs. For occupations, pass the occupation's name as the "
        "query to get the id the occupation_ids filter takes. For companies use "
        "jobo_company_search. Free to call."
    )
    args_schema: type[BaseModel] = JoboFilterValuesInput

    _taxonomy: tuple[float, list[dict[str, str]]] | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        api_key: SecretStr | str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client: httpx.Client | None = None,
        http_async_client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the filter value lookup tool.

        Args:
            api_key: Jobo API key. Read from the `JOBO_API_KEY` environment
                variable when not passed.
            base_url: API base URL. Defaults to `https://connect.jobo.world`.
            timeout: Response timeout in seconds. Defaults to 30.
            max_retries: Retries after the first attempt, for 429, 500 and 503
                responses and failures to connect. Defaults to 2.
            http_client: Optional `httpx.Client` to send sync requests with.
            http_async_client: Optional `httpx.AsyncClient` for async requests.
            **kwargs: Other fields of LangChain's `BaseTool`, e.g.
                `handle_tool_error`, `tags` or `metadata`. An unknown name is
                rejected.

        Raises:
            JoboConfigurationError: No API key was passed or found.
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
            http_async_client=http_async_client,
            **kwargs,
        )

    def _cached_taxonomy(self) -> list[dict[str, str]] | None:
        if self._taxonomy is None:
            return None
        loaded_at, rows = self._taxonomy
        if time.monotonic() - loaded_at > _TAXONOMY_TTL_SECONDS:
            return None
        return rows

    def _keep_taxonomy(self, response: JoboResponse) -> list[dict[str, str]]:
        rows = _flatten_taxonomy(response.data)
        self._taxonomy = (time.monotonic(), rows)
        return rows

    @staticmethod
    def _options_request(
        facet: str, query: str, limit: int
    ) -> tuple[str, dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if query:
            params["q"] = query
        return f"/api/connectors/filter-options/{quote(facet, safe='')}", params

    @staticmethod
    def _finish(facet: str, query: str, response: JoboResponse) -> _Answer:
        payload = response.data if isinstance(response.data, dict) else {}
        raw = payload.get("options")
        options = [item for item in raw if isinstance(item, dict)] if raw else []
        content = format_filter_values(facet, query, options)
        return content, {"facet": facet, "query": query or None, "options": options}

    @staticmethod
    def _finish_occupations(
        query: str, limit: int, rows: list[dict[str, str]]
    ) -> _Answer:
        matches = _match_occupations(rows, query, limit)
        return format_occupation_matches(query, matches), {
            "facet": "occupations",
            "query": query,
            "options": [{"key": match["id"], **match} for match in matches],
        }

    @staticmethod
    def _recover(facet: str, query: str, error: JoboError) -> _Answer:
        if _refused_by_sandbox(error):
            return (
                f"Filter values cannot be looked up here: {_SANDBOX_ONLY_SERVES}. "
                "Do not call this tool again. Call jobo_job_search without the "
                "lookup; the fixed-value filters (work_model, employment_type, "
                "experience_level, company_categories) list their values in "
                "that tool's arguments.",
                {
                    "facet": facet,
                    "query": query or None,
                    "options": [],
                    "unavailable": NOT_AVAILABLE_IN_SANDBOX,
                },
            )
        raise _tool_error(error) from error

    def _run(
        self,
        facet: str,
        query: str | None = None,
        limit: int = 20,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> _Answer:
        text = (query or "").strip()
        try:
            if facet == "occupations" and text:
                rows = self._cached_taxonomy()
                if rows is None:
                    rows = self._keep_taxonomy(
                        self._api().request("GET", "/api/occupations")
                    )
                return self._finish_occupations(text, limit, rows)
            path, params = self._options_request(facet, text, limit)
            response = self._api().request("GET", path, params=params)
        except JoboError as error:
            return self._recover(facet, text, error)
        return self._finish(facet, text, response)

    async def _arun(
        self,
        facet: str,
        query: str | None = None,
        limit: int = 20,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> _Answer:
        text = (query or "").strip()
        try:
            if facet == "occupations" and text:
                rows = self._cached_taxonomy()
                if rows is None:
                    rows = self._keep_taxonomy(
                        await self._api().arequest("GET", "/api/occupations")
                    )
                return self._finish_occupations(text, limit, rows)
            path, params = self._options_request(facet, text, limit)
            response = await self._api().arequest("GET", path, params=params)
        except JoboError as error:
            return self._recover(facet, text, error)
        return self._finish(facet, text, response)


class JoboLocationSearchInput(BaseModel):
    """Arguments of the `jobo_location_search` tool."""

    query: str = Field(
        min_length=2,
        description=(
            "A place name or the start of one, at least 2 characters, e.g. "
            '"Springfield" or "Berl".'
        ),
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=_MAX_LOCATION_SUGGESTIONS,
        description="Maximum number of places to return.",
    )


class JoboLocationSearchTool(_JoboTool):
    """Look up places, written the way the search `location` filter takes them.

    Calls `GET /api/connectors/locations/suggest`, the location typeahead the
    other Jobo connectors use. It is rate limited but never billed.

    The search resolves a free-form `location` itself, so this lookup is for
    the cases where that is not enough: an ambiguous name ("Springfield"), or
    a search that came back with a `locations_unresolved` warning.

    The sandbox does not serve this lookup. With a sandbox key the tool does
    not fail: it tells the model to pass the place name to the search as it is.

    Instantiate:
        ```python
        from langchain_jobo import JoboLocationSearchTool

        tool = JoboLocationSearchTool()
        ```

    Invoke:
        ```python
        tool.invoke({"query": "Springfield"})
        ```
    """

    name: str = "jobo_location_search"
    description: str = (
        "Look up places by name. Returns the matching cities, regions and "
        "countries, written the way the `location` filter of jobo_job_search "
        "takes them. Use it when a place name is ambiguous, or when a search "
        "reports a location it could not resolve. Free to call."
    )
    args_schema: type[BaseModel] = JoboLocationSearchInput

    def __init__(
        self,
        *,
        api_key: SecretStr | str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client: httpx.Client | None = None,
        http_async_client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the location lookup tool.

        Args:
            api_key: Jobo API key. Read from the `JOBO_API_KEY` environment
                variable when not passed.
            base_url: API base URL. Defaults to `https://connect.jobo.world`.
            timeout: Response timeout in seconds. Defaults to 30.
            max_retries: Retries after the first attempt, for 429, 500 and 503
                responses and failures to connect. Defaults to 2.
            http_client: Optional `httpx.Client` to send sync requests with.
            http_async_client: Optional `httpx.AsyncClient` for async requests.
            **kwargs: Other fields of LangChain's `BaseTool`, e.g.
                `handle_tool_error`, `tags` or `metadata`. An unknown name is
                rejected.

        Raises:
            JoboConfigurationError: No API key was passed or found.
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
            http_async_client=http_async_client,
            **kwargs,
        )

    @staticmethod
    def _finish(query: str, response: JoboResponse) -> _Answer:
        payload = response.data if isinstance(response.data, dict) else {}
        raw = payload.get("suggestions")
        suggestions = [item for item in raw if isinstance(item, dict)] if raw else []
        content = format_location_suggestions(query, suggestions)
        return content, {"suggestions": suggestions}

    @staticmethod
    def _recover(error: JoboError) -> _Answer:
        if _refused_by_sandbox(error):
            return (
                f"Places cannot be looked up here: {_SANDBOX_ONLY_SERVES}. Do not "
                "call this tool again. Pass the place name as the `location` "
                "filter of jobo_job_search as it is: the search resolves "
                "free-form locations itself.",
                {"suggestions": [], "unavailable": NOT_AVAILABLE_IN_SANDBOX},
            )
        raise _tool_error(error) from error

    def _run(
        self,
        query: str,
        limit: int = 5,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> _Answer:
        params = {"q": query.strip(), "limit": limit}
        try:
            response = self._api().request(
                "GET", "/api/connectors/locations/suggest", params=params
            )
        except JoboError as error:
            return self._recover(error)
        return self._finish(query.strip(), response)

    async def _arun(
        self,
        query: str,
        limit: int = 5,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> _Answer:
        params = {"q": query.strip(), "limit": limit}
        try:
            response = await self._api().arequest(
                "GET", "/api/connectors/locations/suggest", params=params
            )
        except JoboError as error:
            return self._recover(error)
        return self._finish(query.strip(), response)


# -------------------------------------------------------------------- signals

DEFAULT_MAX_SIGNALS = 25
#: Signals the tool asks for when the model names no limit.
DEFAULT_SIGNAL_LIMIT = 5

_SignalTypeList = Annotated[list[SignalType] | None, BeforeValidator(_to_enum_list)]
_SeniorityList = Annotated[list[SignalSeniority] | None, BeforeValidator(_to_enum_list)]
_CompanySizeList = Annotated[list[CompanySize] | None, BeforeValidator(_to_enum_list)]
#: Arguments of the signals tool that are not filters of the API.
_NOT_SIGNAL_FILTERS = frozenset({"company_id", "limit", "cursor"})


def _signal_limit_description(maximum: int) -> str:
    return (
        f"Signals to return, at most {maximum}. Every returned signal is billed, "
        "at ten times the price of a job, so ask only for what you need. A "
        "larger number is reduced to the maximum."
    )


class JoboHiringSignalsInput(BaseModel):
    """Arguments of the `jobo_hiring_signals` tool.

    Every filter is optional, and filters combine with AND; the values inside
    one list combine with OR. The focus filters (`functions`, `roles`,
    `seniority`, `technologies`, `skills`, `job_countries`) say what a signal
    must be about. A value cannot contain a comma: the API separates values
    with commas, so one is refused here rather than read as two.
    """

    model_config = ConfigDict(extra="forbid")

    types: _SignalTypeList = Field(
        default=None,
        description=(
            "Only these kinds of signal, e.g. hiring_surge, new_country, "
            "leadership_hire. Leave empty for every kind."
        ),
    )
    domains: StrList = Field(
        default=None,
        description=(
            'Only signals at these companies, by website domain, e.g. "stripe.com". '
            "At most 100."
        ),
    )
    company_id: str | None = Field(
        default=None,
        description=(
            "One company's signals, by Jobo company id (a UUID): the Company ID "
            "shown with each signal. It takes only types, min_strength, "
            "detected_after, detected_before, created_after, limit and cursor. "
            "To name companies by website, use domains instead."
        ),
    )
    countries: StrList = Field(
        default=None,
        description=(
            "Countries the companies are headquartered in, as two-letter codes, "
            'e.g. "US", "DE".'
        ),
    )
    company_sizes: _CompanySizeList = Field(
        default=None, description="Headcount bands of the companies."
    )
    industries: NameList = Field(
        default=None,
        description=(
            "Industries of the companies, matched against the primary industry "
            "and the industry list. A value cannot contain a comma."
        ),
    )
    funding_stages: NameList = Field(
        default=None,
        description='Latest funding round of the companies, e.g. "Seed", "Series A".',
    )
    functions: NameList = Field(
        default=None,
        description=(
            'Job categories the signal must be about, e.g. "Technology", '
            '"Sales". A focus filter.'
        ),
    )
    roles: NameList = Field(
        default=None,
        description=(
            "Words or phrases to look for in the job title or occupation behind "
            'the signal, e.g. "data engineer". At least 2 characters each. A '
            "focus filter."
        ),
    )
    seniority: _SeniorityList = Field(
        default=None,
        description="Seniority of the role behind the signal. A focus filter.",
    )
    technologies: StrList = Field(
        default=None,
        description=(
            'Named tools the signal must be about, e.g. "Snowflake". A focus '
            "filter, combined with skills."
        ),
    )
    skills: StrList = Field(
        default=None,
        description=(
            "Named skills the signal must be about. A focus filter, combined "
            "with technologies."
        ),
    )
    job_countries: StrList = Field(
        default=None,
        description=(
            "Countries the jobs are in (not the company's headquarters), as "
            "two-letter codes. A focus filter."
        ),
    )
    min_strength: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description=(
            "Only signals at least this strong. Strength runs from 1 to 100 and "
            "compares signals of the same type."
        ),
    )
    detected_after: date | None = Field(
        default=None,
        description="Earliest detection day, inclusive, as yyyy-MM-dd.",
    )
    detected_before: date | None = Field(
        default=None,
        description="Latest detection day, inclusive, as yyyy-MM-dd.",
    )
    created_after: Annotated[str | None, BeforeValidator(to_signal_instant)] = Field(
        default=None,
        description=(
            "Only signals first stored after this instant, exclusive, e.g. "
            "2026-11-12T09:30:00Z. To ask for recent signals, detected_after is "
            "simpler."
        ),
    )
    exclude_agencies: bool | None = Field(
        default=None,
        description=(
            "Leave out staffing agencies, whose postings are their clients' hiring."
        ),
    )
    limit: Annotated[int, _SchemaMaximum(SIGNAL_MAX_LIMIT)] = Field(
        default=DEFAULT_SIGNAL_LIMIT,
        ge=1,
        description=_signal_limit_description(SIGNAL_MAX_LIMIT),
    )
    cursor: str | None = Field(
        default=None,
        description=(
            "The cursor a previous answer gave, to read its next page. Pass the "
            "same other arguments with it. Leave it out for the first page."
        ),
    )

    def filters(self) -> dict[str, Any]:
        """The filters, keyed as the API names them. `None` means not set."""
        return {
            name: getattr(self, name)
            for name in type(self).model_fields
            if name not in _NOT_SIGNAL_FILTERS
        }

    @model_validator(mode="after")
    def _refuse_what_the_api_would_misread(self) -> JoboHiringSignalsInput:
        # A comma would split a value in two, and the per-company endpoint
        # ignores the filters it does not take: both are refused here, before
        # a request is made and billed.
        build_signals_request(self.filters(), company_id=self.company_id)
        return self


_capped_signal_schemas: dict[int, type[JoboHiringSignalsInput]] = {}


def signals_input_schema(max_signals: int) -> type[JoboHiringSignalsInput]:
    """`JoboHiringSignalsInput` as a tool capped at `max_signals` shows it.

    The schema states the tool's cap as `limit`'s maximum and keeps the
    default under it. A larger `limit` still validates: the tool reduces it
    and says so, rather than failing the model's call.

    One class is made per cap and reused.
    """
    schema = _capped_signal_schemas.get(max_signals)
    if schema is None:
        schema = create_model(
            "JoboHiringSignalsInput",
            __base__=JoboHiringSignalsInput,
            __doc__=JoboHiringSignalsInput.__doc__,
            limit=(
                Annotated[int, _SchemaMaximum(max_signals)],
                Field(
                    default=min(DEFAULT_SIGNAL_LIMIT, max_signals),
                    ge=1,
                    description=_signal_limit_description(max_signals),
                ),
            ),
        )
        _capped_signal_schemas[max_signals] = schema
    return schema


def _is_stock_signals_schema(schema: object) -> bool:
    """Whether `schema` is `JoboHiringSignalsInput` or one of its capped copies."""
    capped = _capped_signal_schemas.values()
    return schema is JoboHiringSignalsInput or schema in capped


class JoboHiringSignalsTool(_JoboTool):
    """Read hiring signals: events at companies, detected from their job postings.

    A signal is one event at one company: a hiring surge, a first job in a
    country, a leadership search, a new technology, a slowdown. The tool calls
    `GET /api/signals`, or `GET /api/companies/{id}/signals` when the model
    names one company, and gives each signal's company, type, a one-line
    summary, strength and the ids of up to five jobs behind it, which
    `JoboJobDetailsTool` reads for free.

    Every signal returned is billed, at ten times the price of a job, and a
    signal read again is billed again. The tool asks for 5 signals by default
    and never for more than `max_signals` in one call. It never pages on its
    own: it hands the model the cursor of the next page instead.

    `get_jobo_tools` leaves it out unless `include_hiring_signals=True`.

    Instantiate:
        ```python
        from langchain_jobo import JoboHiringSignalsTool

        tool = JoboHiringSignalsTool(max_signals=10)
        ```

    Invoke:
        ```python
        tool.invoke(
            {
                "types": ["function_surge", "new_function"],
                "functions": ["Sales"],
                "countries": ["DE"],
                "industries": ["Financial Services"],
                "detected_after": "2026-11-01",
            }
        )
        ```
    """

    name: str = "jobo_hiring_signals"
    description: str = (
        "Read hiring signals: events at companies detected from their job "
        "postings, such as a hiring surge, a first job in a new country or "
        "function, a leadership search, a new technology, a slowdown or a "
        "hiring freeze. Use it to find companies that are growing, entering a "
        "market or building a team, or to check whether one company is hiring "
        "or freezing. Filter by signal type; by company (domains, or company_id "
        "for one company); by the companies' country, size, industry and "
        "funding; and by what the signal is about (functions, roles, seniority, "
        "technologies, skills, job_countries). Returns each signal's company, "
        "type, a one-line summary, the day it was detected, its strength and "
        "the ids of the jobs behind it, which jobo_job_details reads for free. "
        "Every signal returned is billed, at ten times the price of a job, and "
        "a signal read again is billed again: keep limit small and do not "
        "repeat a call you already have the answer to. A call that returns no "
        "signals costs nothing."
    )
    args_schema: type[BaseModel] = JoboHiringSignalsInput

    max_signals: int = Field(
        default=DEFAULT_MAX_SIGNALS,
        ge=1,
        le=SIGNAL_MAX_LIMIT,
        description=(
            "Hard cap on signals requested per call. A larger limit from the "
            "model is reduced to this."
        ),
    )

    def __init__(
        self,
        *,
        api_key: SecretStr | str | None = None,
        max_signals: int | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client: httpx.Client | None = None,
        http_async_client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the hiring signals tool.

        Args:
            api_key: Jobo API key. Read from the `JOBO_API_KEY` environment
                variable when not passed. Hiring signals need a live account
                key: a sandbox key gets a note telling the model to stop.
            max_signals: Hard cap on signals requested per call, 1-200.
                Defaults to 25. A larger `limit` from the model is reduced to
                it, and the tool's schema states it as the maximum.
            base_url: API base URL. Defaults to `https://connect.jobo.world`.
            timeout: Response timeout in seconds. Defaults to 30.
            max_retries: Retries after the first attempt, for 429 responses,
                503 responses with `Retry-After` and failures to connect. A
                500, a read timeout or a dropped connection is not retried:
                the signals may have been billed before it. Defaults to 2.
            http_client: Optional `httpx.Client` to send sync requests with.
            http_async_client: Optional `httpx.AsyncClient` for async requests.
            **kwargs: Other fields of LangChain's `BaseTool`, e.g.
                `handle_tool_error`, `tags` or `metadata`. An unknown name is
                rejected.

        Raises:
            JoboConfigurationError: No API key was passed or found.
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
            http_async_client=http_async_client,
            **given(max_signals=max_signals),
            **kwargs,
        )

    @model_validator(mode="after")
    def _state_the_cap_in_the_schema(self) -> JoboHiringSignalsTool:
        # Also when the tool is rebuilt from another one's fields with a new
        # cap, as `configurable_fields` does: the schema follows the cap.
        if _is_stock_signals_schema(self.args_schema):
            self.args_schema = signals_input_schema(self.max_signals)
        return self

    def _prepare(
        self, kwargs: dict[str, Any]
    ) -> tuple[str, dict[str, Any], int | None]:
        args = JoboHiringSignalsInput.model_validate(kwargs)
        limit = min(args.limit, self.max_signals)
        capped = limit if limit < args.limit else None
        path, params = build_signals_request(
            args.filters(), company_id=args.company_id, limit=limit, cursor=args.cursor
        )
        return path, params, capped

    @staticmethod
    def _finish(response: JoboResponse, capped: int | None) -> _Answer:
        payload = response.data if isinstance(response.data, dict) else {}
        usage = response.usage.as_dict()
        artifact = {
            "signals": signal_list(payload),
            "next_cursor": payload.get("next_cursor"),
            "has_more": payload.get("has_more") is True,
            "usage": usage,
        }
        content = format_signals(
            payload, usage, details_tool="jobo_job_details", capped_limit=capped
        )
        return content, artifact

    @staticmethod
    def _recover(error: JoboError) -> _Answer:
        if _refused_by_sandbox(error):
            return (
                NO_SIGNALS_IN_SANDBOX,
                {
                    "signals": [],
                    "next_cursor": None,
                    "has_more": False,
                    "usage": {},
                    "unavailable": NOT_AVAILABLE_IN_SANDBOX,
                },
            )
        raise ToolException(signals_error_message(error)) from error

    def _run(
        self,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> _Answer:
        path, params, capped = self._prepare(kwargs)
        try:
            response = self._api().request("GET", path, params=params)
        except JoboError as error:
            return self._recover(error)
        return self._finish(response, capped)

    async def _arun(
        self,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> _Answer:
        path, params, capped = self._prepare(kwargs)
        try:
            response = await self._api().arequest("GET", path, params=params)
        except JoboError as error:
            return self._recover(error)
        return self._finish(response, capped)


__all__ = [
    "DEFAULT_MAX_PAGE_SIZE",
    "DEFAULT_MAX_SIGNALS",
    "DEFAULT_SIGNAL_LIMIT",
    "JoboCompanySearchInput",
    "JoboCompanySearchTool",
    "JoboFilterValuesInput",
    "JoboFilterValuesTool",
    "JoboHiringSignalsInput",
    "JoboHiringSignalsTool",
    "JoboJobDetailsInput",
    "JoboJobDetailsTool",
    "JoboLocationSearchInput",
    "JoboLocationSearchTool",
    "JoboSearchTool",
    "signals_input_schema",
]
