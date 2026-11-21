"""LangChain tools for the Jobo jobs API."""

from __future__ import annotations

from typing import Any, Literal
from urllib.parse import quote

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from langchain_jobo._base import JoboSettings
from langchain_jobo._client import JoboResponse
from langchain_jobo.errors import JoboError, JoboNotFoundError
from langchain_jobo.filters import (
    API_MAX_PAGE_SIZE,
    DescriptionFormat,
    JoboSearchInput,
    build_search_body,
)
from langchain_jobo.formatting import (
    format_company_suggestions,
    format_job_document,
    format_search_results,
)

DEFAULT_MAX_PAGE_SIZE = 25
_MAX_COMPANY_SUGGESTIONS = 20


def _tool_error(error: JoboError) -> ToolException:
    """Wrap an API failure so `handle_tool_error` can hand it to the model."""
    return ToolException(str(error))


class JoboSearchTool(BaseTool, JoboSettings):
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
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    max_page_size: int = Field(
        default=DEFAULT_MAX_PAGE_SIZE,
        ge=1,
        le=API_MAX_PAGE_SIZE,
        description=(
            "Hard cap on jobs requested per call. A larger page_size from the "
            "model is reduced to this."
        ),
    )

    def _prepare(self, kwargs: dict[str, Any]) -> tuple[dict[str, Any], int | None]:
        args = JoboSearchInput.model_validate(kwargs)
        page_size = min(args.page_size, self.max_page_size)
        capped = page_size if page_size < args.page_size else None
        body = build_search_body(
            args, query=args.query, page=args.page, page_size=page_size
        )
        return body, capped

    @staticmethod
    def _finish(
        response: JoboResponse, capped: int | None
    ) -> tuple[str, dict[str, Any]]:
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
    ) -> tuple[str, dict[str, Any]]:
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
    ) -> tuple[str, dict[str, Any]]:
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


class JoboJobDetailsTool(BaseTool, JoboSettings):
    """Fetch one job listing by id and render it as readable prose.

    Calls `GET /api/jobs/{id}`, which is free: it is never billed, though it
    counts toward the search rate limit.

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
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

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

    def _path_and_params(self, job_id: str) -> tuple[str, dict[str, str]]:
        cleaned = job_id.strip()
        if not cleaned:
            msg = "job_id is required."
            raise ToolException(msg)
        params: dict[str, str] = {}
        if self.description_format != "markdown":
            params["description_format"] = self.description_format
        return f"/api/jobs/{quote(cleaned, safe='')}", params

    def _finish(self, response: JoboResponse) -> tuple[str, dict[str, Any]]:
        job = response.data if isinstance(response.data, dict) else {}
        content = format_job_document(job, include_description=self.include_description)
        return content, job

    @staticmethod
    def _not_found(job_id: str) -> tuple[str, dict[str, Any]]:
        return (
            f'No job exists with id "{job_id.strip()}". Job ids are the `id` values '
            "jobo_job_search returns.",
            {},
        )

    def _run(
        self,
        job_id: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        path, params = self._path_and_params(job_id)
        try:
            response = self._api().request("GET", path, params=params)
        except JoboNotFoundError:
            return self._not_found(job_id)
        except JoboError as error:
            raise _tool_error(error) from error
        return self._finish(response)

    async def _arun(
        self,
        job_id: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        path, params = self._path_and_params(job_id)
        try:
            response = await self._api().arequest("GET", path, params=params)
        except JoboNotFoundError:
            return self._not_found(job_id)
        except JoboError as error:
            raise _tool_error(error) from error
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


class JoboCompanySearchTool(BaseTool, JoboSettings):
    """Look up companies in Jobo's catalog to get the ids the search filter takes.

    Calls `GET /api/connectors/companies/suggest`, the company typeahead the
    other Jobo connectors use. It is rate limited but never billed.

    The `companies` filter of `JoboSearchTool` matches a company *name* whole,
    and a name also matches every other company called that. This tool returns
    each company's ids, which match exactly.

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
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    @staticmethod
    def _finish(query: str, response: JoboResponse) -> tuple[str, dict[str, Any]]:
        payload = response.data if isinstance(response.data, dict) else {}
        raw = payload.get("suggestions")
        suggestions = [item for item in raw if isinstance(item, dict)] if raw else []
        content = format_company_suggestions(query, suggestions)
        return content, {"suggestions": suggestions}

    def _run(
        self,
        query: str,
        limit: int = 5,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        params = {"q": query.strip(), "limit": limit}
        try:
            response = self._api().request(
                "GET", "/api/connectors/companies/suggest", params=params
            )
        except JoboError as error:
            raise _tool_error(error) from error
        return self._finish(query.strip(), response)

    async def _arun(
        self,
        query: str,
        limit: int = 5,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        params = {"q": query.strip(), "limit": limit}
        try:
            response = await self._api().arequest(
                "GET", "/api/connectors/companies/suggest", params=params
            )
        except JoboError as error:
            raise _tool_error(error) from error
        return self._finish(query.strip(), response)


__all__ = [
    "JoboCompanySearchInput",
    "JoboCompanySearchTool",
    "JoboJobDetailsInput",
    "JoboJobDetailsTool",
    "JoboSearchTool",
]
