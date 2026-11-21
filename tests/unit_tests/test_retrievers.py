"""`JoboRetriever`: query in, job `Document`s out."""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from langchain_jobo import (
    JoboConfigurationError,
    JoboInsufficientCreditsError,
    JoboRetriever,
    JoboSearchFilters,
)
from tests.unit_tests.helpers import (
    INSUFFICIENT_CREDITS,
    SANDBOX_KEY,
    TEST_KEY,
    MockJobo,
    search_page,
)
from tests.unit_tests.helpers import search_handler as search_api_handler


def test_query_and_constructor_filters_build_the_search_body(
    search_api: MockJobo,
) -> None:
    retriever = JoboRetriever(
        k=3,
        filters={
            "location": "London",
            "work_model": ["remote"],
            "exclude_companies": ["Acme"],
            "min_salary_usd": 90000,
        },
        **search_api.kwargs,
    )

    retriever.invoke("staff data engineer")

    assert search_api.last.url.path == "/api/jobs/search"
    assert search_api.bodies == [
        {
            "queries": ["staff data engineer"],
            "locations": ["London"],
            "work_models": ["remote"],
            "companies": {"exclude": ["Acme"]},
            "salary_usd": {"min": 90000},
            "page": 1,
            "page_size": 3,
        }
    ]


def test_filters_accept_a_model(search_api: MockJobo) -> None:
    retriever = JoboRetriever(
        filters=JoboSearchFilters(skills=["Rust"]), **search_api.kwargs
    )

    retriever.invoke("engineer")

    assert search_api.bodies[0]["skills"] == {"include": ["Rust"]}
    assert search_api.bodies[0]["page_size"] == 5


def test_unknown_filters_are_rejected() -> None:
    # A type checker rejects the key as well: `filters` is a typed dict.
    with pytest.raises(ValidationError):
        JoboRetriever(
            api_key=TEST_KEY,
            filters={"is_remote": True},  # type: ignore[arg-type]
        )


def test_documents_carry_prose_and_metadata(search_api: MockJobo) -> None:
    documents = JoboRetriever(k=2, **search_api.kwargs).invoke("backend engineer")

    assert len(documents) == 2
    first = documents[0]
    assert isinstance(first, Document)
    assert first.id == "00000000-0000-4000-8000-000000000000"
    assert first.page_content.startswith(
        "# Senior Backend Engineer 0\n\nCompany: Acme Corp"
    )
    assert "## Description\nWe are hiring" in first.page_content
    assert first.metadata["id"] == first.id
    assert first.metadata["url"] == "https://jobs.acme.example/0"
    assert first.metadata["title"] == "Senior Backend Engineer 0"
    assert first.metadata["company"] == "Acme Corp"
    assert first.metadata["location"] == "Berlin, Germany"
    assert first.metadata["date_posted"] == "2026-11-16T00:00:00Z"
    assert first.metadata["salary"] == "EUR 80,000-100,000 per year"
    assert first.metadata["source"] == "https://jobs.acme.example/0"
    assert first.metadata["ats"] == "greenhouse"


def test_description_can_be_left_out_of_page_content(search_api: MockJobo) -> None:
    retriever = JoboRetriever(include_description=False, **search_api.kwargs)

    documents = retriever.invoke("backend engineer")

    assert all("## Description" not in doc.page_content for doc in documents)


def test_k_can_be_overridden_per_call(search_api: MockJobo) -> None:
    retriever = JoboRetriever(k=5, **search_api.kwargs)

    assert len(retriever.invoke("nurse", k=1)) == 1
    assert len(retriever.invoke("nurse")) == 5
    assert [body["page_size"] for body in search_api.bodies] == [1, 5]


@pytest.mark.parametrize("k", [0, 101])
def test_k_is_bounded_by_the_api_page_size(k: int, search_api: MockJobo) -> None:
    with pytest.raises(ValidationError):
        JoboRetriever(k=k, **search_api.kwargs)

    retriever = JoboRetriever(**search_api.kwargs)
    with pytest.raises(ValueError, match="k must be between 1 and 100"):
        retriever.invoke("nurse", k=k)
    assert search_api.requests == []


def test_no_matches_is_an_empty_list() -> None:
    empty = {"jobs": [], "total": 0, "page": 1, "page_size": 5, "total_pages": 0}
    api = MockJobo(lambda _: httpx.Response(200, json=empty))

    assert JoboRetriever(**api.kwargs).invoke("unicorn wrangler") == []


def test_api_errors_propagate_typed_and_unretried() -> None:
    api = MockJobo(lambda _: httpx.Response(402, json=INSUFFICIENT_CREDITS))

    with pytest.raises(JoboInsufficientCreditsError) as caught:
        JoboRetriever(**api.kwargs).invoke("nurse")

    assert caught.value.credits_required == 75
    assert len(api.requests) == 1


async def test_async_retrieval(search_api: MockJobo) -> None:
    retriever = JoboRetriever(
        k=4,
        filters={"experience_level": ["senior"]},
        **search_api.kwargs,
    )

    documents = await retriever.ainvoke("nurse")
    one = await retriever.ainvoke("nurse", k=1)

    assert len(documents) == 4
    assert len(one) == 1
    assert search_api.bodies[0] == {
        "queries": ["nurse"],
        "experience_levels": ["senior"],
        "page": 1,
        "page_size": 4,
    }
    assert search_api.last.headers["X-Jobo-Client"] == "langchain"


async def test_async_errors_propagate_typed() -> None:
    api = MockJobo(lambda _: httpx.Response(402, json=INSUFFICIENT_CREDITS))

    with pytest.raises(JoboInsufficientCreditsError):
        await JoboRetriever(**api.kwargs).ainvoke("nurse")

    assert len(api.requests) == 1


# ------------------------------------------------------------ per-call options


def test_filters_can_be_overridden_per_call(search_api: MockJobo) -> None:
    retriever = JoboRetriever(
        filters=JoboSearchFilters(work_model=["remote"], location="London"),
        **search_api.kwargs,
    )

    retriever.invoke("nurse", filters={"location": ["Berlin", "Paris"]})
    retriever.invoke("nurse", filters=JoboSearchFilters(skills=["Go"]))
    retriever.invoke("nurse", filters={"work_model": None})
    retriever.invoke("nurse")

    assert [
        {k: v for k, v in body.items() if k not in ("queries", "page", "page_size")}
        for body in search_api.bodies
    ] == [
        {"locations": ["Berlin", "Paris"], "work_models": ["remote"]},
        {
            "locations": ["London"],
            "work_models": ["remote"],
            "skills": {"include": ["Go"]},
        },
        {"locations": ["London"]},
        # The override was for one call: the retriever's own filters are intact.
        {"locations": ["London"], "work_models": ["remote"]},
    ]


def test_page_can_be_chosen_per_call(search_api: MockJobo) -> None:
    retriever = JoboRetriever(k=2, **search_api.kwargs)

    retriever.invoke("nurse", page=3)
    retriever.invoke("nurse")

    assert [body["page"] for body in search_api.bodies] == [3, 1]


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        ({"page": 0}, ValueError, "page must be 1 or greater"),
        ({"filters": {"is_remote": True}}, ValidationError, "is_remote"),
        ({"filters": {"work_model": ["anywhere"]}}, ValidationError, "work_model"),
        ({"K": 2}, TypeError, "unexpected arguments: K. It takes k, filters and page"),
        ({"top_k": 2, "filter": {}}, TypeError, "filter, top_k"),
    ],
)
def test_bad_per_call_options_fail_before_any_request(
    call: dict[str, Any],
    error: type[Exception],
    message: str,
    search_api: MockJobo,
) -> None:
    """A mistyped `k` would otherwise bill the default page without a word."""
    retriever = JoboRetriever(**search_api.kwargs)

    with pytest.raises(error, match=message):
        retriever.invoke("nurse", **call)

    assert search_api.requests == []


def test_langchains_own_invoke_arguments_are_not_mistaken_for_ours(
    search_api: MockJobo,
) -> None:
    retriever = JoboRetriever(k=1, **search_api.kwargs)

    assert len(retriever.invoke("nurse", verbose=True)) == 1
    assert len(retriever.invoke("nurse", config={"tags": ["a"]})) == 1


async def test_per_call_options_async(search_api: MockJobo) -> None:
    retriever = JoboRetriever(filters={"work_model": ["remote"]}, **search_api.kwargs)

    await retriever.ainvoke("nurse", k=2, page=4, filters={"sources": ["lever"]})

    assert search_api.bodies == [
        {
            "queries": ["nurse"],
            "sources": ["lever"],
            "work_models": ["remote"],
            "page": 4,
            "page_size": 2,
        }
    ]
    with pytest.raises(TypeError, match="unexpected arguments: K"):
        await retriever.ainvoke("nurse", K=2)


def test_description_format_is_a_filter_of_the_retriever(search_api: MockJobo) -> None:
    """Here it changes `page_content`, so unlike on the tool it is a filter."""
    retriever = JoboRetriever(
        filters={"description_format": "html"}, **search_api.kwargs
    )

    retriever.invoke("nurse")

    assert search_api.bodies[0]["description_format"] == "html"


# -------------------------------------------------------------------- warnings

UNMATCHED = {
    "code": "companies_unmatched",
    "field": "companies.include",
    "message": "No company matched: Acme.",
}


def test_api_warnings_are_logged_and_kept(caplog: pytest.LogCaptureFixture) -> None:
    """An unmatched company comes back as no jobs and a warning, not an error."""
    pages = iter([search_page(0, warnings=[UNMATCHED]), search_page(1)])
    api = MockJobo(lambda _: httpx.Response(200, json=next(pages)))
    retriever = JoboRetriever(filters={"companies": ["Acme"]}, **api.kwargs)
    assert retriever.last_warnings == []

    with caplog.at_level(logging.WARNING, logger="langchain_jobo"):
        documents = retriever.invoke("nurse")

    assert documents == []
    assert retriever.last_warnings == [UNMATCHED]
    assert caplog.messages == ["Jobo search: No company matched: Acme."]
    assert caplog.records[0].name == "langchain_jobo.retrievers"

    # The next query replaces them: they describe one query, not the retriever.
    assert len(retriever.invoke("nurse")) == 1
    assert retriever.last_warnings == []


async def test_api_warnings_async(caplog: pytest.LogCaptureFixture) -> None:
    page = search_page(
        0,
        warnings=[UNMATCHED],
        filters={
            "companies": {
                "suggestions": [
                    {
                        "query": "Acme",
                        "companies": [{"name": "Acme Corp", "ids": ["x"]}],
                    }
                ]
            }
        },
    )
    api = MockJobo(lambda _: httpx.Response(200, json=page))
    retriever = JoboRetriever(**api.kwargs)

    with caplog.at_level(logging.WARNING, logger="langchain_jobo"):
        assert await retriever.ainvoke("nurse") == []

    assert retriever.last_warnings == [UNMATCHED]
    assert caplog.messages == [
        "Jobo search: No company matched: Acme.",
        'Jobo search: Closest companies to "Acme": Acme Corp (ids: x)',
    ]


def test_last_warnings_cannot_be_edited_from_outside() -> None:
    api = MockJobo(
        lambda _: httpx.Response(200, json=search_page(0, warnings=[UNMATCHED]))
    )
    retriever = JoboRetriever(**api.kwargs)
    retriever.invoke("nurse")

    retriever.last_warnings.clear()

    assert retriever.last_warnings == [UNMATCHED]


# ---------------------------------------------------------------- construction


@pytest.mark.parametrize(
    "kwargs", [{"K": 2}, {"max_page_size": 3}, {"filter": {}}, {"apikey": "x"}]
)
def test_mistyped_constructor_arguments_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        JoboRetriever(api_key=TEST_KEY, **kwargs)


def test_langchain_retriever_fields_are_still_accepted() -> None:
    retriever = JoboRetriever(api_key=TEST_KEY, tags=["jobs"], metadata={"team": "a"})

    assert retriever.tags == ["jobs"]
    assert retriever.metadata == {"team": "a"}


def test_missing_api_key_is_a_clear_error_at_construction() -> None:
    with pytest.raises(JoboConfigurationError, match="JOBO_API_KEY") as caught:
        JoboRetriever()

    assert "https://enterprise.jobo.world/api-keys" in str(caught.value)


@pytest.mark.parametrize("key", ["", "  "])
def test_blank_api_key_is_refused_at_construction(key: str) -> None:
    with pytest.raises(JoboConfigurationError, match="JOBO_API_KEY"):
        JoboRetriever(api_key=key)


def test_api_key_is_read_from_the_environment_and_kept_secret(
    monkeypatch: pytest.MonkeyPatch, search_api: MockJobo
) -> None:
    monkeypatch.setenv("JOBO_API_KEY", " jbe_live_from-env ")
    retriever = JoboRetriever(http_client=search_api.client)

    retriever.invoke("nurse")

    assert search_api.last.headers["X-Api-Key"] == "jbe_live_from-env"
    assert "from-env" not in repr(retriever)


def test_sandbox_key_works_with_the_retriever(search_api: MockJobo) -> None:
    """The sandbox serves search, so nothing is refused here."""
    retriever = JoboRetriever(api_key=SANDBOX_KEY, http_client=search_api.client, k=1)

    assert len(retriever.invoke("nurse")) == 1
    assert search_api.last.headers["X-Api-Key"] == SANDBOX_KEY


def test_a_subclass_can_change_defaults_and_add_fields(search_api: MockJobo) -> None:
    class TwoRemoteJobs(JoboRetriever):
        k: int = 2
        filters: JoboSearchFilters = JoboSearchFilters(work_model=["remote"])
        team: str = "data"

    retriever = TwoRemoteJobs(team="platform", **search_api.kwargs)

    assert len(retriever.invoke("nurse")) == 2
    assert retriever.team == "platform"
    assert search_api.bodies[0]["work_models"] == ["remote"]
    assert len(TwoRemoteJobs(k=4, **search_api.kwargs).invoke("nurse")) == 4
    with pytest.raises(ValidationError):
        TwoRemoteJobs(squad="x", **search_api.kwargs)


# ----------------------------------------------------------------- connections


def test_retriever_is_a_context_manager_and_leaves_your_client_open(
    search_api: MockJobo,
) -> None:
    with JoboRetriever(k=1, **search_api.kwargs) as retriever:
        assert len(retriever.invoke("nurse")) == 1
    retriever.close()

    assert isinstance(retriever, JoboRetriever)
    assert not search_api.client.is_closed


def test_closing_the_retriever_closes_the_connections_it_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list[httpx.Client] = []
    real = httpx.Client

    def make() -> httpx.Client:
        made.append(real(transport=httpx.MockTransport(search_api_handler)))
        return made[-1]

    monkeypatch.setattr(httpx, "Client", make)
    retriever = JoboRetriever(api_key=TEST_KEY, k=1)

    retriever.invoke("nurse")
    retriever.invoke("nurse")
    assert len(made) == 1  # one pool for every query
    retriever.close()

    assert made[0].is_closed
    assert len(retriever.invoke("nurse")) == 1  # reusable after closing
    assert len(made) == 2


async def test_retriever_is_an_async_context_manager(search_api: MockJobo) -> None:
    async with JoboRetriever(k=1, **search_api.kwargs) as retriever:
        assert len(await retriever.ainvoke("nurse")) == 1
    await retriever.aclose()

    assert not search_api.async_client.is_closed
