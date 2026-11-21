"""`JoboRetriever`: query in, job `Document`s out."""

from __future__ import annotations

import httpx
import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from langchain_jobo import (
    JoboInsufficientCreditsError,
    JoboRetriever,
    JoboSearchFilters,
)
from tests.unit_tests.helpers import INSUFFICIENT_CREDITS, TEST_KEY, MockJobo


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
    with pytest.raises(ValidationError):
        JoboRetriever(api_key=TEST_KEY, filters={"is_remote": True})


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
    assert first.metadata["source"] == "greenhouse"


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
