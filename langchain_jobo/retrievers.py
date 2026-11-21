"""Retriever over the Jobo job search."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr, SecretStr

from langchain_jobo._base import JoboSettings, given
from langchain_jobo._client import JoboResponse
from langchain_jobo.filters import (
    API_MAX_PAGE_SIZE,
    JoboSearchFilters,
    JoboSearchFiltersDict,
    build_search_body,
    merge_search_filters,
)
from langchain_jobo.formatting import (
    job_to_document,
    response_warnings,
    warning_messages,
)

logger = logging.getLogger(__name__)

#: Per-call arguments of `invoke`. `verbose` is LangChain's own.
_CALL_ARGUMENTS = frozenset({"k", "filters", "page", "verbose"})


class JoboRetriever(BaseRetriever, JoboSettings):
    """Retrieve live job listings as `Document`s.

    Each query runs one `POST /api/jobs/search` and returns up to `k` jobs.
    Search is billed per job returned, so `k` is also what a query can cost.

    `page_content` is self-contained prose about the job; `metadata` carries
    `id`, `source` and `url` (the listing URL), `title`, `company`,
    `location`, the dates, the salary and `ats` (the applicant tracking system
    the listing came from). The `Document.id` is the Jobo job id, which is
    stable and safe to dedupe on.

    Setup:
        Install `langchain-jobo` and set the `JOBO_API_KEY` environment variable.

        ```bash
        pip install -U langchain-jobo
        export JOBO_API_KEY="jbe_live_..."
        ```

    Instantiate:
        ```python
        from langchain_jobo import JoboRetriever, JoboSearchFilters

        retriever = JoboRetriever(
            k=5,
            filters=JoboSearchFilters(
                work_model=["remote"], experience_level=["senior"]
            ),
        )
        ```

    Usage:
        ```python
        docs = retriever.invoke("staff data engineer")
        for doc in docs:
            print(doc.metadata["title"], doc.metadata["source"])
        ```

    Override `k`, the filters or the page for one call:
        ```python
        retriever.invoke("staff data engineer", k=2)
        retriever.invoke("staff data engineer", filters={"location": "Berlin"})
        retriever.invoke("staff data engineer", page=2)
        ```

    Warnings:
        A filter value the API cannot use does not fail the search: it comes
        back as a warning, often with no jobs. Each warning is logged on the
        `langchain_jobo` logger, and the warnings of the most recent query are
        kept on `last_warnings`.

        ```python
        if not docs and retriever.last_warnings:
            print(retriever.last_warnings[0]["message"])
        ```
    """

    k: int = Field(
        default=5,
        ge=1,
        le=API_MAX_PAGE_SIZE,
        description="Jobs to return per query. Every returned job is billed.",
    )
    filters: JoboSearchFilters = Field(
        default_factory=JoboSearchFilters,
        description="Filters applied to every query.",
    )
    include_description: bool = Field(
        default=True,
        description=(
            "Append the employer's full description to page_content after the "
            "structured fields."
        ),
    )

    _last_warnings: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        *,
        api_key: SecretStr | str | None = None,
        k: int | None = None,
        filters: JoboSearchFilters | JoboSearchFiltersDict | None = None,
        include_description: bool | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client: httpx.Client | None = None,
        http_async_client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the retriever.

        Args:
            api_key: Jobo API key. Read from the `JOBO_API_KEY` environment
                variable when not passed.
            k: Jobs to return per query, 1-100. Defaults to 5. Every returned
                job is billed.
            filters: Filters applied to every query: a `JoboSearchFilters`, or
                a dict with its keys. Unknown keys are rejected.
            include_description: Append the employer's full description to
                `page_content`. Defaults to true.
            base_url: API base URL. Defaults to `https://connect.jobo.world`.
            timeout: Response timeout in seconds. Defaults to 30.
            max_retries: Retries after the first attempt, for 429, 500 and 503
                responses and failures to connect. Defaults to 2.
            http_client: Optional `httpx.Client` to send sync requests with.
            http_async_client: Optional `httpx.AsyncClient` for async requests.
            **kwargs: Other fields of LangChain's `BaseRetriever`, e.g. `tags`
                or `metadata`. An unknown name is rejected.

        Raises:
            JoboConfigurationError: No API key was passed or found.
        """
        super().__init__(
            **given(
                api_key=api_key,
                k=k,
                filters=filters,
                include_description=include_description,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
                http_client=http_client,
                http_async_client=http_async_client,
            ),
            **kwargs,
        )

    @property
    def last_warnings(self) -> list[dict[str, Any]]:
        """Warnings the API sent with the most recent query on this retriever.

        Each is a dict with `code`, `field` and `message`, e.g.
        `companies_unmatched` when a `companies` value matched no company.
        Empty when there were none. A retriever shared between concurrent
        callers holds whichever query finished last: read the log instead.
        """
        return list(self._last_warnings)

    def _body(self, query: str, call: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(call) - _CALL_ARGUMENTS)
        if unknown:
            msg = (
                f"JoboRetriever.invoke() got unexpected arguments: "
                f"{', '.join(unknown)}. It takes k, filters and page."
            )
            raise TypeError(msg)

        size = self.k if call.get("k") is None else int(call["k"])
        if not 1 <= size <= API_MAX_PAGE_SIZE:
            msg = f"k must be between 1 and {API_MAX_PAGE_SIZE}, got {size}."
            raise ValueError(msg)
        page = 1 if call.get("page") is None else int(call["page"])
        if page < 1:
            msg = f"page must be 1 or greater, got {page}."
            raise ValueError(msg)

        filters = merge_search_filters(self.filters, call.get("filters"))
        return build_search_body(filters, query=query, page=page, page_size=size)

    def _documents(self, response: JoboResponse) -> list[Document]:
        payload = response.data if isinstance(response.data, dict) else {}
        self._last_warnings = response_warnings(payload)
        for message in warning_messages(payload):
            logger.warning("Jobo search: %s", message)
        jobs = payload.get("jobs") or []
        return [
            job_to_document(job, include_description=self.include_description)
            for job in jobs
            if isinstance(job, dict)
        ]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        body = self._body(query, kwargs)
        response = self._api().request("POST", "/api/jobs/search", json=body)
        return self._documents(response)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        body = self._body(query, kwargs)
        response = await self._api().arequest("POST", "/api/jobs/search", json=body)
        return self._documents(response)


__all__ = ["JoboRetriever"]
