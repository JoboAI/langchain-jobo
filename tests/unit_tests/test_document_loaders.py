"""`JoboFeedLoader`: cursor paging, the `max_jobs` budget and resuming."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from langchain_jobo import (
    JoboFeedCursorRestartRequiredError,
    JoboFeedFilters,
    JoboFeedLoader,
    JoboInsufficientCreditsError,
    JoboNotFoundError,
)
from langchain_jobo.document_loaders import _plan_batch_size
from tests.unit_tests.helpers import (
    CURSOR_RESTART,
    INSUFFICIENT_CREDITS,
    TEST_KEY,
    MockJobo,
    make_job,
)


class FakeFeed:
    """A feed of `total` jobs that behaves like `POST /api/jobs/feed`.

    The cursor is `"<offset>:<batch size>"`. As on the real feed, the scan's
    batch size is fixed by its first request, and a continuation may lower it
    (for good) but never raise it.
    """

    def __init__(self, total: int) -> None:
        self.total = total

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "cursor" in body:
            offset, size = (int(part) for part in body["cursor"].split(":"))
            size = min(size, int(body.get("batch_size", size)))
        else:
            offset, size = 0, int(body.get("batch_size", 1000))

        jobs = [make_job(i) for i in range(offset, min(offset + size, self.total))]
        next_offset = offset + len(jobs)
        has_more = next_offset < self.total
        payload: dict[str, Any] = {
            "jobs": jobs,
            "next_cursor": f"{next_offset}:{size}" if has_more else None,
            "has_more": has_more,
        }
        if "cursor" not in body:
            payload["estimated_total"] = self.total
        return httpx.Response(
            200, json=payload, headers={"X-Jobs-Billed": str(len(jobs))}
        )


def feed(total: int) -> MockJobo:
    return MockJobo(FakeFeed(total))


def titles(documents: list[Document]) -> list[str]:
    return [doc.metadata["title"] for doc in documents]


# ----------------------------------------------------------------- the budget


def test_max_jobs_has_no_default() -> None:
    with pytest.raises(TypeError, match="max_jobs"):
        JoboFeedLoader(api_key=TEST_KEY)  # type: ignore[call-arg]


@pytest.mark.parametrize("max_jobs", [0, -5, 1.5, "10", True])
def test_max_jobs_must_be_a_positive_integer_or_none(max_jobs: Any) -> None:
    with pytest.raises(ValueError, match="max_jobs"):
        JoboFeedLoader(api_key=TEST_KEY, max_jobs=max_jobs)


@pytest.mark.parametrize("batch_size", [0, 1001])
def test_batch_size_is_bounded_by_the_api(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        JoboFeedLoader(api_key=TEST_KEY, max_jobs=10, batch_size=batch_size)


@pytest.mark.parametrize(
    ("max_jobs", "batch_size", "planned"),
    [
        (None, 1000, 1000),
        (5, 1000, 5),
        (1000, 1000, 1000),
        (3000, 1000, 1000),
        (1500, 1000, 750),
        (2500, 1000, 833),
        (1001, 1000, 500),
        (10, 4, 3),
    ],
)
def test_batches_are_planned_to_fit_under_max_jobs(
    max_jobs: int | None, batch_size: int, planned: int
) -> None:
    assert _plan_batch_size(max_jobs, batch_size) == planned
    loader = JoboFeedLoader(api_key=TEST_KEY, max_jobs=max_jobs, batch_size=batch_size)
    assert loader.batch_size == planned


def test_max_jobs_bounds_what_is_requested_not_just_what_is_yielded() -> None:
    api = feed(total=100)
    loader = JoboFeedLoader(max_jobs=6, batch_size=3, **api.kwargs)

    documents = loader.load()

    assert len(documents) == 6
    assert api.bodies == [{"batch_size": 3}, {"cursor": "3:3"}]
    # Nothing was delivered (and billed) beyond the budget.
    assert loader.jobs_billed == 6
    assert loader.jobs_loaded == 6
    assert loader.has_more is True
    assert loader.last_cursor == "6:3"


def test_a_small_budget_is_one_exact_batch() -> None:
    api = feed(total=100)
    loader = JoboFeedLoader(max_jobs=7, **api.kwargs)

    assert len(loader.load()) == 7
    assert api.bodies == [{"batch_size": 7}]


def test_budget_that_is_not_a_whole_number_of_batches_stops_short() -> None:
    """A partial last batch would shrink the scan's batch size for good."""
    api = feed(total=100)
    loader = JoboFeedLoader(max_jobs=10, batch_size=4, **api.kwargs)

    documents = loader.load()

    assert loader.batch_size == 3
    assert len(documents) == 9
    assert all("batch_size" not in body for body in api.bodies[1:])
    assert loader.last_cursor == "9:3"


def test_unbounded_load_is_an_explicit_opt_in_and_reads_to_the_end() -> None:
    api = feed(total=2500)
    loader = JoboFeedLoader(max_jobs=None, **api.kwargs)

    documents = loader.load()

    assert len(documents) == 2500
    assert api.bodies == [
        {"batch_size": 1000},
        {"cursor": "1000:1000"},
        {"cursor": "2000:1000"},
    ]
    assert loader.has_more is False
    assert loader.estimated_total == 2500


def test_feed_shorter_than_the_budget_ends_cleanly() -> None:
    api = feed(total=4)
    loader = JoboFeedLoader(max_jobs=50, **api.kwargs)

    assert len(loader.load()) == 4
    assert loader.has_more is False
    assert len(api.requests) == 1


def test_empty_feed() -> None:
    api = feed(total=0)
    loader = JoboFeedLoader(max_jobs=50, **api.kwargs)

    assert loader.load() == []
    assert loader.has_more is False
    assert loader.last_cursor is None


# -------------------------------------------------------------------- requests


def test_first_request_carries_filters_and_continuations_only_the_cursor() -> None:
    api = feed(total=100)
    loader = JoboFeedLoader(
        max_jobs=4,
        batch_size=2,
        filters={
            "sources": ["greenhouse"],
            "work_models": ["remote"],
            "job_categories": {"include": ["Healthcare"]},
            "exclude_employment_types": ["internship"],
        },
        **api.kwargs,
    )

    loader.load()

    assert api.last.method == "POST"
    assert api.last.url.path == "/api/jobs/feed"
    assert api.bodies == [
        {
            "sources": ["greenhouse"],
            "work_models": ["remote"],
            "exclude_employment_types": ["internship"],
            "job_categories": {"include": ["Healthcare"]},
            "batch_size": 2,
        },
        {"cursor": "2:2"},
    ]


def test_filters_accept_a_model_and_reject_unknown_keys() -> None:
    api = feed(total=1)
    JoboFeedLoader(
        max_jobs=1, filters=JoboFeedFilters(company_is_agency=False), **api.kwargs
    ).load()

    assert api.bodies == [{"company_is_agency": False, "batch_size": 1}]
    with pytest.raises(ValidationError):
        JoboFeedLoader(api_key=TEST_KEY, max_jobs=1, filters={"work_model": ["remote"]})


def test_feed_requests_use_the_long_timeout_and_identify_the_client() -> None:
    api = feed(total=1)
    JoboFeedLoader(max_jobs=1, **api.kwargs).load()

    assert api.last.extensions["timeout"]["read"] == 120.0
    assert api.last.headers["X-Jobo-Client"] == "langchain"
    assert api.last.headers["X-Api-Key"] == TEST_KEY


def test_api_key_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOBO_API_KEY", "jbe_live_from-env")
    api = feed(total=1)

    JoboFeedLoader(max_jobs=1, http_client=api.client).load()

    assert api.last.headers["X-Api-Key"] == "jbe_live_from-env"


def test_missing_api_key_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="JOBO_API_KEY"):
        JoboFeedLoader(max_jobs=1)


# ------------------------------------------------------------------ documents


def test_documents_are_rendered_jobs() -> None:
    api = feed(total=2)

    documents = JoboFeedLoader(max_jobs=2, **api.kwargs).load()

    assert titles(documents) == [
        "Senior Backend Engineer 0",
        "Senior Backend Engineer 1",
    ]
    assert documents[0].id == "00000000-0000-4000-8000-000000000000"
    assert documents[0].page_content.startswith("# Senior Backend Engineer 0")
    assert "## Description" in documents[0].page_content
    assert documents[0].metadata["source"] == "greenhouse"
    assert documents[0].metadata["url"] == "https://jobs.acme.example/0"


def test_description_can_be_left_out() -> None:
    api = feed(total=1)

    documents = JoboFeedLoader(
        max_jobs=1, include_description=False, **api.kwargs
    ).load()

    assert "## Description" not in documents[0].page_content


# --------------------------------------------------------------------- cursor


def test_cursor_advances_only_once_a_batch_is_fully_consumed() -> None:
    api = feed(total=100)
    loader = JoboFeedLoader(max_jobs=4, batch_size=2, **api.kwargs)
    iterator = loader.lazy_load()

    def state() -> tuple[str | None, bool | None, int]:
        return loader.last_cursor, loader.has_more, loader.jobs_loaded

    next(iterator)
    assert state() == (None, None, 1)

    next(iterator)
    assert state() == (None, None, 2)  # the consumer has not come back yet

    next(iterator)  # asking for the third job closes the first batch
    assert state() == ("2:2", True, 3)

    assert len(list(iterator)) == 1
    assert state() == ("4:2", True, 4)
    assert loader.jobs_loaded == 4


def test_resuming_from_last_cursor_continues_without_gaps_or_repeats() -> None:
    api = feed(total=10)
    first = JoboFeedLoader(max_jobs=4, batch_size=2, **api.kwargs)
    first_documents = first.load()

    second = JoboFeedLoader(
        max_jobs=4, batch_size=2, cursor=first.last_cursor, **api.kwargs
    )
    second_documents = second.load()

    assert second.cursor == "4:2"
    assert titles(first_documents + second_documents) == [
        f"Senior Backend Engineer {i}" for i in range(8)
    ]
    assert second.last_cursor == "8:2"
    assert second.has_more is True


def test_resume_sends_only_the_cursor_and_this_runs_batch_size() -> None:
    """Filters sent beside a cursor are ignored by the API, so none are sent."""
    api = feed(total=100)
    loader = JoboFeedLoader(
        max_jobs=4,
        batch_size=2,
        cursor="10:2",
        filters={"sources": ["greenhouse"]},
        **api.kwargs,
    )

    loader.load()

    assert api.bodies == [{"cursor": "10:2", "batch_size": 2}, {"cursor": "12:2"}]


def test_resume_never_reads_past_the_budget_of_a_larger_cursor() -> None:
    api = feed(total=5000)
    loader = JoboFeedLoader(max_jobs=3, cursor="1000:1000", **api.kwargs)

    documents = loader.load()

    assert len(documents) == 3
    assert api.bodies == [{"cursor": "1000:1000", "batch_size": 3}]
    assert loader.jobs_billed == 3


def test_unbounded_resume_sends_the_cursor_alone() -> None:
    api = feed(total=30)
    loader = JoboFeedLoader(max_jobs=None, cursor="10:10", **api.kwargs)

    assert len(loader.load()) == 20
    assert api.bodies == [{"cursor": "10:10"}, {"cursor": "20:10"}]


def test_each_pass_starts_again_from_the_constructor_cursor() -> None:
    api = feed(total=10)
    loader = JoboFeedLoader(max_jobs=2, **api.kwargs)

    assert titles(loader.load()) == titles(loader.load())
    assert loader.jobs_loaded == 2
    assert loader.jobs_billed == 2


def test_blank_cursor_means_start_from_the_beginning() -> None:
    api = feed(total=3)
    loader = JoboFeedLoader(max_jobs=1, cursor="  ", **api.kwargs)

    loader.load()

    assert loader.cursor is None
    assert api.bodies == [{"batch_size": 1}]


def test_warnings_from_the_first_page_are_kept() -> None:
    warning = {
        "code": "companies_unmatched",
        "field": "companies.include",
        "message": "No company matched: Acme.",
    }
    page = {
        "jobs": [],
        "next_cursor": None,
        "has_more": False,
        "estimated_total": 0,
        "warnings": [warning],
    }
    api = MockJobo(lambda _: httpx.Response(200, json=page))
    loader = JoboFeedLoader(max_jobs=10, filters={"companies": ["Acme"]}, **api.kwargs)

    assert loader.load() == []
    assert api.bodies == [{"companies": {"include": ["Acme"]}, "batch_size": 10}]
    assert loader.warnings == [warning]
    assert loader.estimated_total == 0
    assert loader.jobs_billed == 0


def test_a_feed_that_stops_advancing_does_not_loop_forever() -> None:
    stuck = {"jobs": [], "next_cursor": "same", "has_more": True}
    api = MockJobo(lambda _: httpx.Response(200, json=stuck))
    loader = JoboFeedLoader(max_jobs=None, cursor="same", **api.kwargs)

    assert loader.load() == []
    assert len(api.requests) == 1


# --------------------------------------------------------------------- errors


def test_409_cursor_restart_is_raised_and_never_retried() -> None:
    api = MockJobo(lambda _: httpx.Response(409, json=CURSOR_RESTART))
    loader = JoboFeedLoader(max_jobs=10, cursor="stale", **api.kwargs)

    with pytest.raises(JoboFeedCursorRestartRequiredError) as caught:
        loader.load()

    assert caught.value.replacement_request == {"stable_scan": True, "batch_size": 1000}
    assert len(api.requests) == 1


def test_402_mid_scan_keeps_the_cursor_of_the_last_completed_batch() -> None:
    fake = FakeFeed(total=100)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 2:
            return httpx.Response(402, json=INSUFFICIENT_CREDITS)
        return fake(request)

    api = MockJobo(handler)
    loader = JoboFeedLoader(max_jobs=6, batch_size=2, **api.kwargs)
    loaded: list[Document] = []

    with pytest.raises(JoboInsufficientCreditsError):
        loaded.extend(loader.lazy_load())

    assert len(api.requests) == 2  # the 402 was not retried
    assert loader.jobs_loaded == 2
    assert loader.last_cursor == "2:2"


def test_503_is_retried_with_the_same_cursor() -> None:
    fake = FakeFeed(total=4)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 2:
            return httpx.Response(
                503, json={"title": "Feed unavailable"}, headers={"Retry-After": "5"}
            )
        return fake(request)

    api = MockJobo(handler)
    loader = JoboFeedLoader(max_jobs=4, batch_size=2, **api.kwargs)
    sleeps: list[float] = []
    loader._client._sleep = sleeps.append

    documents = loader.load()

    assert len(documents) == 4
    assert sleeps == [5.0]
    assert api.bodies == [{"batch_size": 2}, {"cursor": "2:2"}, {"cursor": "2:2"}]


def test_sandbox_keys_get_a_typed_not_found() -> None:
    problem = {
        "title": "Not available in the sandbox",
        "status": 404,
        "code": "not_available_in_sandbox",
    }
    api = MockJobo(lambda _: httpx.Response(404, json=problem))

    with pytest.raises(JoboNotFoundError) as caught:
        JoboFeedLoader(max_jobs=1, **api.kwargs).load()

    assert caught.value.code == "not_available_in_sandbox"


# ---------------------------------------------------------------------- async


async def test_alazy_load_pages_and_respects_the_budget() -> None:
    api = feed(total=100)
    loader = JoboFeedLoader(
        max_jobs=6, batch_size=3, filters={"sources": ["lever"]}, **api.kwargs
    )

    documents = [doc async for doc in loader.alazy_load()]

    assert titles(documents) == [f"Senior Backend Engineer {i}" for i in range(6)]
    assert api.bodies == [{"sources": ["lever"], "batch_size": 3}, {"cursor": "3:3"}]
    assert loader.last_cursor == "6:3"
    assert loader.has_more is True
    assert loader.jobs_billed == 6


async def test_aload_reads_an_unbounded_feed_to_the_end() -> None:
    api = feed(total=5)
    loader = JoboFeedLoader(max_jobs=None, batch_size=2, **api.kwargs)

    documents = await loader.aload()

    assert len(documents) == 5
    assert loader.has_more is False


async def test_async_resume_and_errors() -> None:
    api = feed(total=10)
    loader = JoboFeedLoader(max_jobs=2, cursor="4:2", **api.kwargs)

    documents = await loader.aload()

    assert titles(documents) == [
        "Senior Backend Engineer 4",
        "Senior Backend Engineer 5",
    ]
    assert api.bodies == [{"cursor": "4:2", "batch_size": 2}]

    failing = MockJobo(lambda _: httpx.Response(409, json=CURSOR_RESTART))
    with pytest.raises(JoboFeedCursorRestartRequiredError):
        await JoboFeedLoader(max_jobs=2, cursor="stale", **failing.kwargs).aload()
    assert len(failing.requests) == 1
