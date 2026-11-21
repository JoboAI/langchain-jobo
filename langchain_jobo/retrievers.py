"""Retriever over the Jobo job search."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from langchain_jobo._base import JoboSettings
from langchain_jobo._client import JoboResponse
from langchain_jobo.filters import (
    API_MAX_PAGE_SIZE,
    JoboSearchFilters,
    build_search_body,
)
from langchain_jobo.formatting import job_to_document


class JoboRetriever(BaseRetriever, JoboSettings):
    """Retrieve live job listings as `Document`s.

    Each query runs one `POST /api/jobs/search` and returns up to `k` jobs.
    Search is billed per job returned, so `k` is also what a query can cost.

    `page_content` is self-contained prose about the job; `metadata` carries
    `id`, `url`, `title`, `company`, `location`, the dates, the salary and the
    `source` (the ATS the listing came from). The `Document.id` is the Jobo job
    id, which is stable and safe to dedupe on.

    Setup:
        Install `langchain-jobo` and set the `JOBO_API_KEY` environment variable.

        ```bash
        pip install -U langchain-jobo
        export JOBO_API_KEY="jbe_live_..."
        ```

    Instantiate:
        ```python
        from langchain_jobo import JoboRetriever

        retriever = JoboRetriever(
            k=5,
            filters={"work_model": ["remote"], "experience_level": ["senior"]},
        )
        ```

    Usage:
        ```python
        docs = retriever.invoke("staff data engineer")
        for doc in docs:
            print(doc.metadata["title"], doc.metadata["url"])
        ```

    Override `k` for one call:
        ```python
        retriever.invoke("staff data engineer", k=2)
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

    def _body(self, query: str, k: Any) -> dict[str, Any]:
        size = self.k if k is None else int(k)
        if not 1 <= size <= API_MAX_PAGE_SIZE:
            msg = f"k must be between 1 and {API_MAX_PAGE_SIZE}, got {size}."
            raise ValueError(msg)
        return build_search_body(self.filters, query=query, page=1, page_size=size)

    def _documents(self, response: JoboResponse) -> list[Document]:
        payload = response.data if isinstance(response.data, dict) else {}
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
        body = self._body(query, kwargs.get("k"))
        response = self._api().request("POST", "/api/jobs/search", json=body)
        return self._documents(response)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        body = self._body(query, kwargs.get("k"))
        response = await self._api().arequest("POST", "/api/jobs/search", json=body)
        return self._documents(response)


__all__ = ["JoboRetriever"]
