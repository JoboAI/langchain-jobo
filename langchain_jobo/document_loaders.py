"""Document loader over the Jobo job feed."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

import httpx
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.utils import secret_from_env
from pydantic import SecretStr

from langchain_jobo._base import _MISSING_KEY_MESSAGE
from langchain_jobo._client import (
    DEFAULT_BASE_URL,
    DEFAULT_FEED_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    JoboClient,
    JoboResponse,
)
from langchain_jobo.filters import (
    API_MAX_BATCH_SIZE,
    JoboFeedFilters,
    build_feed_body,
)
from langchain_jobo.formatting import job_to_document

_FEED_PATH = "/api/jobs/feed"


def _plan_batch_size(max_jobs: int | None, batch_size: int) -> int:
    """Pick a batch size whose whole batches add up to at most `max_jobs`.

    The feed keeps the smallest batch size a scan has ever used: lowering it
    for one last partial batch would shrink every later page read from the
    same cursor. So the loader never asks for a partial batch. It splits
    `max_jobs` into the fewest equal batches that fit under `batch_size`.
    """
    if max_jobs is None:
        return batch_size
    batches = -(-max_jobs // batch_size)
    return max_jobs // batches


class _Scan:
    """Cursor bookkeeping for one pass over the feed, shared by sync and async."""

    def __init__(self, loader: JoboFeedLoader) -> None:
        self._loader = loader
        self._cursor = loader.cursor
        self._first = True
        self._done = False
        self._pending_cursor: str | None = None
        self._pending_has_more = False
        self._pending_jobs = 0
        loader.last_cursor = loader.cursor
        loader.has_more = None
        loader.jobs_loaded = 0
        loader.jobs_billed = 0
        loader.estimated_total = None
        loader.warnings = []

    def next_request(self) -> dict[str, Any] | None:
        """The body of the next request, or `None` when the pass is over."""
        loader = self._loader
        if self._done:
            return None
        if (
            loader.max_jobs is not None
            and loader.max_jobs - loader.jobs_loaded < loader.batch_size
        ):
            # The next whole batch would not fit under max_jobs.
            return None

        first, self._first = self._first, False
        if self._cursor is None:
            return build_feed_body(loader.filters, batch_size=loader.batch_size)
        body: dict[str, Any] = {"cursor": self._cursor}
        if first and loader.max_jobs is not None:
            # Resuming: the cursor carries its scan's batch size, which may be
            # larger than this run's budget allows per request.
            body["batch_size"] = loader.batch_size
        return body

    def read(self, response: JoboResponse) -> list[dict[str, Any]]:
        """Take a feed page: note its cursor, return its jobs."""
        loader = self._loader
        payload = response.data if isinstance(response.data, dict) else {}
        raw_jobs = payload.get("jobs") or []
        jobs = [job for job in raw_jobs if isinstance(job, dict)]

        next_cursor = payload.get("next_cursor")
        self._pending_cursor = next_cursor if isinstance(next_cursor, str) else None
        self._pending_has_more = bool(payload.get("has_more"))
        self._pending_jobs = len(jobs)

        if response.usage.jobs_billed is not None:
            loader.jobs_billed += response.usage.jobs_billed
        estimated_total = payload.get("estimated_total")
        if isinstance(estimated_total, int) and loader.estimated_total is None:
            loader.estimated_total = estimated_total
        warnings = payload.get("warnings")
        if isinstance(warnings, list):
            loader.warnings.extend(w for w in warnings if isinstance(w, dict))
        return jobs

    def batch_done(self) -> None:
        """Every job of the page was handed out: commit its cursor."""
        loader = self._loader
        loader.has_more = self._pending_has_more
        stalled = self._pending_cursor == self._cursor and self._pending_jobs == 0
        if self._pending_cursor:
            loader.last_cursor = self._pending_cursor
        if not self._pending_has_more or not self._pending_cursor or stalled:
            self._done = True
        self._cursor = self._pending_cursor


class JoboFeedLoader(BaseLoader):
    """Load jobs from the Jobo feed (`POST /api/jobs/feed`) as `Document`s.

    The feed streams every matching active job, oldest first, following a
    cursor. It is billed per *new* job: a job you have not received through
    Feed, Export or Outbound in the last 30 days. An unbounded scan can
    therefore be a large bill, so `max_jobs` is required: pass a number, or
    `max_jobs=None` to opt in to reading the whole feed.

    The loader reads whole batches only and never more than `max_jobs` jobs.
    When `max_jobs` is not a multiple of the batch size it settles for slightly
    fewer rather than ask the feed for a partial batch (see `batch_size`).

    The feed is not available to sandbox (`jbe_test_`) keys.

    Setup:
        Install `langchain-jobo` and set the `JOBO_API_KEY` environment variable.

        ```bash
        pip install -U langchain-jobo
        export JOBO_API_KEY="jbe_live_..."
        ```

    Instantiate:
        ```python
        from langchain_jobo import JoboFeedLoader

        loader = JoboFeedLoader(
            max_jobs=500,
            filters={
                "sources": ["greenhouse"],
                "work_models": ["remote"],
                "job_categories": {"include": ["Healthcare"]},
            },
        )
        ```

    Lazy load, then keep the cursor to resume later:
        ```python
        for doc in loader.lazy_load():
            print(doc.metadata["title"])

        if loader.has_more:
            save(loader.last_cursor)
        ```

    Resume where the previous run stopped:
        ```python
        loader = JoboFeedLoader(max_jobs=500, cursor=saved_cursor)
        ```

    Attributes:
        last_cursor: Cursor to resume from. It advances once every job of a
            batch has been yielded, so a consumer that stops part-way through a
            batch resumes from the start of that batch (re-delivery of a job
            you already received is free on the feed; dedupe on `Document.id`).
        has_more: Whether the feed had more jobs after the last completed
            batch. `None` until a batch completes.
        jobs_loaded: Documents yielded by the current or last pass.
        jobs_billed: Jobs the API reported as billed (`X-Jobs-Billed`) during
            the current or last pass. Stays 0 when the API does not send it.
        estimated_total: The API's estimate of how many jobs the scan will
            deliver, sent on the first page of a new scan.
        warnings: Non-fatal warnings from the API, e.g. `companies_unmatched`.
        batch_size: Jobs requested per call, after planning for `max_jobs`.
    """

    def __init__(
        self,
        *,
        max_jobs: int | None,
        api_key: SecretStr | str | None = None,
        filters: JoboFeedFilters | Mapping[str, Any] | None = None,
        cursor: str | None = None,
        batch_size: int = API_MAX_BATCH_SIZE,
        include_description: bool = True,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_FEED_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: httpx.Client | None = None,
        http_async_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create the loader.

        Args:
            max_jobs: Required. The most jobs one pass may load, or `None` to
                read the whole feed. There is no default on purpose.
            api_key: Jobo API key. Read from `JOBO_API_KEY` when not passed.
            filters: Feed filters (`JoboFeedFilters` or a dict with its keys).
                Unknown keys are rejected. Ignored when `cursor` is given: a
                cursor keeps the filters of the scan it came from.
            cursor: `last_cursor` of an earlier run, to resume that scan.
            batch_size: Upper bound on jobs per request, 1-1000. With
                `max_jobs` set, the loader uses the largest size at or below
                this that splits `max_jobs` into equal whole batches. The API
                checks the wallet for a full batch before each request.
            include_description: Append the employer's full description to
                `page_content` after the structured fields.
            base_url: API base URL.
            timeout: Response timeout in seconds. The API asks for at least
                120 on feed calls.
            max_retries: Retries after the first attempt, for 429, 500 and 503
                responses and failures to connect. 402 and 409 are never
                retried.
            http_client: Optional `httpx.Client` to send sync requests with.
            http_async_client: Optional `httpx.AsyncClient` for async requests.
        """
        if max_jobs is not None and (
            isinstance(max_jobs, bool) or not isinstance(max_jobs, int) or max_jobs < 1
        ):
            msg = "max_jobs must be a positive integer, or None to read the whole feed."
            raise ValueError(msg)
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not 1 <= batch_size <= API_MAX_BATCH_SIZE
        ):
            msg = f"batch_size must be between 1 and {API_MAX_BATCH_SIZE}."
            raise ValueError(msg)

        if api_key is None:
            api_key = secret_from_env(
                "JOBO_API_KEY", error_message=_MISSING_KEY_MESSAGE
            )()

        self.max_jobs = max_jobs
        self.filters = JoboFeedFilters.model_validate(filters or {})
        self.cursor = cursor.strip() if cursor and cursor.strip() else None
        self.batch_size = _plan_batch_size(max_jobs, batch_size)
        self.include_description = include_description
        self.timeout = timeout
        self._client = JoboClient(
            api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
            http_async_client=http_async_client,
        )

        self.last_cursor: str | None = self.cursor
        self.has_more: bool | None = None
        self.jobs_loaded = 0
        self.jobs_billed = 0
        self.estimated_total: int | None = None
        self.warnings: list[dict[str, Any]] = []

    def _document(self, job: Mapping[str, Any]) -> Document:
        self.jobs_loaded += 1
        return job_to_document(job, include_description=self.include_description)

    def lazy_load(self) -> Iterator[Document]:
        """Yield one `Document` per job, a batch at a time.

        Each call starts a fresh pass from `cursor` (or from the beginning of
        the feed) and loads at most `max_jobs` jobs.
        """
        scan = _Scan(self)
        while (body := scan.next_request()) is not None:
            response = self._client.request(
                "POST", _FEED_PATH, json=body, timeout=self.timeout
            )
            for job in scan.read(response):
                yield self._document(job)
            scan.batch_done()

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Async twin of `lazy_load`."""
        scan = _Scan(self)
        while (body := scan.next_request()) is not None:
            response = await self._client.arequest(
                "POST", _FEED_PATH, json=body, timeout=self.timeout
            )
            for job in scan.read(response):
                yield self._document(job)
            scan.batch_done()


__all__ = ["JoboFeedLoader"]
