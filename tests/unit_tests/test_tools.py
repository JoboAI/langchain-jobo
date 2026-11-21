"""The three tools and the `get_jobo_tools` helper."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException
from pydantic import ValidationError

from langchain_jobo import (
    JoboCompanySearchTool,
    JoboInsufficientCreditsError,
    JoboJobDetailsTool,
    JoboRateLimitError,
    JoboSearchTool,
    get_jobo_tools,
)
from tests.unit_tests.helpers import (
    INSUFFICIENT_CREDITS,
    RATE_LIMITED,
    TEST_KEY,
    MockJobo,
    make_job,
)

JOB_ID = "00000000-0000-4000-8000-000000000042"


def tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_call", "id": "call-1", "name": name, "args": args}


# --------------------------------------------------------------------- search


def test_search_maps_arguments_onto_the_post_body(search_api: MockJobo) -> None:
    tool = JoboSearchTool(**search_api.kwargs)

    tool.invoke(
        {
            "query": "backend engineer",
            "location": "Berlin, Germany",
            "work_model": ["remote", "hybrid"],
            "employment_type": ["full-time"],
            "experience_level": ["senior"],
            "skills": ["Python", "Go"],
            "exclude_skills": ["PHP"],
            "industries": ["Software Development"],
            "companies": ["stripe.com"],
            "exclude_companies": ["Acme"],
            "min_salary_usd": 150000,
            "posted_after": "2026-11-01T00:00:00Z",
            "sort": "discovered_desc",
            "page": 2,
            "page_size": 10,
        }
    )

    request = search_api.last
    assert request.method == "POST"
    assert request.url.path == "/api/jobs/search"
    assert search_api.bodies == [
        {
            "queries": ["backend engineer"],
            "locations": ["Berlin, Germany"],
            "work_models": ["remote", "hybrid"],
            "employment_types": ["full-time"],
            "experience_levels": ["senior"],
            "skills": {"include": ["Python", "Go"], "exclude": ["PHP"]},
            "companies": {"include": ["stripe.com"], "exclude": ["Acme"]},
            "industries": {"include": ["Software Development"]},
            "salary_usd": {"min": 150000},
            "posted_after": "2026-11-01T00:00:00Z",
            "sort": "discovered_desc",
            "page": 2,
            "page_size": 10,
        }
    ]


def test_search_defaults_to_a_small_page(search_api: MockJobo) -> None:
    JoboSearchTool(**search_api.kwargs).invoke({"query": "nurse"})

    assert search_api.bodies == [{"queries": ["nurse"], "page": 1, "page_size": 5}]


def test_search_page_size_is_capped(search_api: MockJobo) -> None:
    tool = JoboSearchTool(**search_api.kwargs)

    content = tool.invoke({"query": "nurse", "page_size": 100})

    assert search_api.bodies[0]["page_size"] == 25
    assert "page_size was capped at 25" in content
    assert "showing 25" in content


def test_search_cap_is_configurable(search_api: MockJobo) -> None:
    tool = JoboSearchTool(max_page_size=3, **search_api.kwargs)

    tool.invoke({"query": "nurse", "page_size": 50})
    tool.invoke({"query": "nurse", "page_size": 2})

    assert [body["page_size"] for body in search_api.bodies] == [3, 2]


@pytest.mark.parametrize("cap", [0, 101])
def test_search_cap_cannot_exceed_the_api_maximum(cap: int) -> None:
    with pytest.raises(ValidationError):
        JoboSearchTool(api_key=TEST_KEY, max_page_size=cap)


def test_search_rejects_a_page_size_above_the_api_maximum(
    search_api: MockJobo,
) -> None:
    tool = JoboSearchTool(**search_api.kwargs)

    with pytest.raises(ValidationError):
        tool.invoke({"query": "nurse", "page_size": 500})

    assert search_api.requests == []


def test_search_rejects_unknown_arguments_before_calling_the_api(
    search_api: MockJobo,
) -> None:
    tool = JoboSearchTool(**search_api.kwargs)

    with pytest.raises(ValidationError):
        tool.invoke({"query": "nurse", "is_remote": True})
    with pytest.raises(ValidationError):
        tool.invoke({"query": "nurse", "work_model": ["anywhere"]})

    assert search_api.requests == []


def test_search_returns_compact_content_and_the_raw_jobs_as_artifact(
    search_api: MockJobo,
) -> None:
    tool = JoboSearchTool(**search_api.kwargs)

    message = tool.invoke(tool_call("jobo_job_search", {"query": "x", "page_size": 2}))

    assert isinstance(message, ToolMessage)
    assert message.name == "jobo_job_search"
    assert isinstance(message.content, str)
    assert message.content.startswith("1,234 matching jobs; showing 2 (page 1 of 617).")
    assert "1. Senior Backend Engineer 0 - Acme Corp" in message.content
    assert "URL: https://jobs.acme.example/0" in message.content
    assert "ID: 00000000-0000-4000-8000-000000000000" in message.content
    # The description stays out of the model's context; it is in the artifact.
    assert "We are hiring" not in message.content
    assert message.artifact["jobs"] == [make_job(0), make_job(1)]
    assert message.artifact["total"] == 1234
    assert message.artifact["page"] == 1
    assert message.artifact["total_pages"] == 617
    assert message.artifact["usage"] == {
        "credits_deducted": 6,
        "credits_balance": 4000,
    }


def test_search_with_no_matches_still_returns_an_artifact() -> None:
    empty = {"jobs": [], "total": 0, "page": 1, "page_size": 5, "total_pages": 0}
    api = MockJobo(lambda _: httpx.Response(200, json=empty))
    tool = JoboSearchTool(**api.kwargs)

    message = tool.invoke(tool_call("jobo_job_search", {"query": "unicorn wrangler"}))

    assert message.content.startswith("No jobs matched.")
    assert message.artifact["jobs"] == []
    assert message.artifact["usage"] == {}


def test_search_api_errors_become_tool_exceptions() -> None:
    api = MockJobo(lambda _: httpx.Response(402, json=INSUFFICIENT_CREDITS))
    tool = JoboSearchTool(**api.kwargs)

    with pytest.raises(ToolException, match="This request needs 75 credits") as caught:
        tool.invoke({"query": "nurse"})

    assert isinstance(caught.value.__cause__, JoboInsufficientCreditsError)
    assert len(api.requests) == 1
    assert TEST_KEY not in str(caught.value)


def test_handle_tool_error_hands_the_error_to_the_model() -> None:
    api = MockJobo(lambda _: httpx.Response(402, json=INSUFFICIENT_CREDITS))
    tool = JoboSearchTool(handle_tool_error=True, **api.kwargs)

    message = tool.invoke(tool_call("jobo_job_search", {"query": "nurse"}))

    assert message.status == "error"
    assert "This request needs 75 credits" in message.content
    assert "https://enterprise.jobo.world/credits?topup=10" in message.content


async def test_search_async(search_api: MockJobo) -> None:
    tool = JoboSearchTool(**search_api.kwargs)

    message = await tool.ainvoke(
        tool_call(
            "jobo_job_search", {"query": "nurse", "exclude_industries": ["Staffing"]}
        )
    )

    assert search_api.bodies == [
        {
            "queries": ["nurse"],
            "industries": {"exclude": ["Staffing"]},
            "page": 1,
            "page_size": 5,
        }
    ]
    assert len(message.artifact["jobs"]) == 5
    assert "showing 5" in message.content


async def test_search_async_errors_become_tool_exceptions() -> None:
    api = MockJobo(lambda _: httpx.Response(429, json=RATE_LIMITED))
    tool = JoboSearchTool(max_retries=0, **api.kwargs)

    with pytest.raises(ToolException, match="request limit of 60 exceeded") as caught:
        await tool.ainvoke({"query": "nurse"})

    assert isinstance(caught.value.__cause__, JoboRateLimitError)


# -------------------------------------------------------------------- details


def test_details_fetches_one_job_and_renders_prose() -> None:
    api = MockJobo(lambda _: httpx.Response(200, json=make_job(42)))
    tool = JoboJobDetailsTool(**api.kwargs)

    message = tool.invoke(tool_call("jobo_job_details", {"job_id": f" {JOB_ID} "}))

    assert api.last.method == "GET"
    assert str(api.last.url) == f"https://connect.jobo.world/api/jobs/{JOB_ID}"
    assert message.content.startswith("# Senior Backend Engineer 42")
    assert "## Responsibilities\n- Design APIs" in message.content
    assert "## Description\nWe are hiring" in message.content
    assert message.artifact == make_job(42)


def test_details_options_shape_the_request_and_the_text() -> None:
    api = MockJobo(lambda _: httpx.Response(200, json=make_job(42)))
    tool = JoboJobDetailsTool(
        include_description=False, description_format="enhanced", **api.kwargs
    )

    content = tool.invoke({"job_id": JOB_ID})

    assert api.last.url.params["description_format"] == "enhanced"
    assert "## Description" not in content


def test_details_escapes_the_id_in_the_path() -> None:
    api = MockJobo(lambda _: httpx.Response(404, json={"title": "Not Found"}))
    tool = JoboJobDetailsTool(**api.kwargs)

    tool.invoke({"job_id": "../companies/x?y=1"})

    assert api.last.url.raw_path == b"/api/jobs/..%2Fcompanies%2Fx%3Fy%3D1"


def test_details_unknown_id_is_an_answer_not_an_error() -> None:
    api = MockJobo(
        lambda _: httpx.Response(404, json={"title": "Not Found", "status": 404})
    )
    tool = JoboJobDetailsTool(**api.kwargs)

    message = tool.invoke(tool_call("jobo_job_details", {"job_id": JOB_ID}))

    assert message.content.startswith(f'No job exists with id "{JOB_ID}".')
    assert message.status == "success"


def test_details_blank_id_is_rejected_without_a_request() -> None:
    api = MockJobo(lambda _: httpx.Response(200, json=make_job()))
    tool = JoboJobDetailsTool(**api.kwargs)

    with pytest.raises(ToolException, match="job_id is required"):
        tool.invoke({"job_id": "   "})

    assert api.requests == []


def test_details_other_errors_become_tool_exceptions() -> None:
    api = MockJobo(
        lambda _: httpx.Response(401, json={"error": "Invalid or expired API key"})
    )
    tool = JoboJobDetailsTool(**api.kwargs)

    with pytest.raises(ToolException, match="Invalid or expired API key"):
        tool.invoke({"job_id": JOB_ID})


async def test_details_async() -> None:
    api = MockJobo(lambda _: httpx.Response(200, json=make_job(42)))
    tool = JoboJobDetailsTool(**api.kwargs)

    message = await tool.ainvoke(tool_call("jobo_job_details", {"job_id": JOB_ID}))

    assert api.last.url.path == f"/api/jobs/{JOB_ID}"
    assert message.artifact["id"] == JOB_ID
    assert "Compensation: EUR 80,000-100,000 per year" in message.content


async def test_details_async_unknown_id() -> None:
    api = MockJobo(lambda _: httpx.Response(404, json={"title": "Not Found"}))
    tool = JoboJobDetailsTool(**api.kwargs)

    content = await tool.ainvoke({"job_id": JOB_ID})

    assert content.startswith("No job exists")


# -------------------------------------------------------------------- company

SUGGESTIONS = {
    "suggestions": [
        {
            "id": "id-1",
            "ids": ["id-1", "id-2"],
            "name": "Stripe",
            "domain": "stripe.com",
            "logo_url": None,
            "country": "US",
            "job_count": 1204,
        }
    ]
}


def test_company_search_calls_the_suggest_endpoint() -> None:
    api = MockJobo(lambda _: httpx.Response(200, json=SUGGESTIONS))
    tool = JoboCompanySearchTool(**api.kwargs)

    message = tool.invoke(
        tool_call("jobo_company_search", {"query": " strip ", "limit": 3})
    )

    assert api.last.method == "GET"
    assert api.last.url.path == "/api/connectors/companies/suggest"
    assert dict(api.last.url.params) == {"q": "strip", "limit": "3"}
    assert "1. Stripe (stripe.com, US) - 1,204 live jobs" in message.content
    assert "company ids: id-1, id-2" in message.content
    assert message.artifact == SUGGESTIONS


def test_company_search_with_no_match() -> None:
    api = MockJobo(lambda _: httpx.Response(200, json={"suggestions": []}))
    tool = JoboCompanySearchTool(**api.kwargs)

    message = tool.invoke(tool_call("jobo_company_search", {"query": "zzzz"}))

    assert message.content.startswith('No company matched "zzzz".')
    assert message.artifact == {"suggestions": []}
    assert dict(api.last.url.params) == {"q": "zzzz", "limit": "5"}


@pytest.mark.parametrize("args", [{"query": "a"}, {"query": "acme", "limit": 21}])
def test_company_search_validates_arguments(args: dict[str, Any]) -> None:
    api = MockJobo(lambda _: httpx.Response(200, json=SUGGESTIONS))
    tool = JoboCompanySearchTool(**api.kwargs)

    with pytest.raises(ValidationError):
        tool.invoke(args)

    assert api.requests == []


async def test_company_search_async() -> None:
    api = MockJobo(lambda _: httpx.Response(200, json=SUGGESTIONS))
    tool = JoboCompanySearchTool(**api.kwargs)

    content = await tool.ainvoke({"query": "stripe.com"})

    assert dict(api.last.url.params) == {"q": "stripe.com", "limit": "5"}
    assert "Stripe" in content


# ------------------------------------------------------------------- settings


def test_api_key_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, search_api: MockJobo
) -> None:
    monkeypatch.setenv("JOBO_API_KEY", "jbe_test_from-env")
    tool = JoboSearchTool(http_client=search_api.client)

    tool.invoke({"query": "nurse"})

    assert tool.api_key.get_secret_value() == "jbe_test_from-env"
    assert search_api.last.headers["X-Api-Key"] == "jbe_test_from-env"


def test_missing_api_key_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="JOBO_API_KEY"):
        JoboSearchTool()


def test_api_key_is_masked_in_repr_and_dumps() -> None:
    tool = JoboSearchTool(api_key="jbe_live_do-not-print-me")

    assert "do-not-print-me" not in repr(tool)
    assert "do-not-print-me" not in str(tool)
    assert "do-not-print-me" not in str(tool.model_dump())


def test_tool_names_and_schemas() -> None:
    search, details, company = get_jobo_tools(api_key=TEST_KEY)

    assert search.name == "jobo_job_search"
    assert details.name == "jobo_job_details"
    assert company.name == "jobo_company_search"
    for tool in (search, details, company):
        assert tool.description
        assert tool.response_format == "content_and_artifact"
    assert set(details.args) == {"job_id"}
    assert set(company.args) == {"query", "limit"}
    assert {"query", "location", "work_model", "skills", "page_size"} <= set(
        search.args
    )
    # Connection settings never leak into what the model is shown.
    assert "api_key" not in search.args


def test_get_jobo_tools_passes_settings_to_every_tool(search_api: MockJobo) -> None:
    tools = get_jobo_tools(
        max_page_size=7,
        base_url="https://example.test",
        handle_tool_error=True,
        **search_api.kwargs,
    )

    assert [type(tool) for tool in tools] == [
        JoboSearchTool,
        JoboJobDetailsTool,
        JoboCompanySearchTool,
    ]
    assert isinstance(tools[0], JoboSearchTool)
    assert tools[0].max_page_size == 7
    for tool in tools:
        assert tool.handle_tool_error is True
        assert getattr(tool, "base_url", None) == "https://example.test"

    tools[0].invoke({"query": "nurse"})
    assert search_api.last.url.host == "example.test"


def test_get_jobo_tools_reads_the_key_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBO_API_KEY", "jbe_test_from-env")

    tools = get_jobo_tools(include_company_search=False)

    assert [tool.name for tool in tools] == ["jobo_job_search", "jobo_job_details"]
