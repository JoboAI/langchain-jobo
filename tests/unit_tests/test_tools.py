"""The five tools and the `get_jobo_tools` helper."""

from __future__ import annotations

import copy
from typing import Any

import httpx
import pytest
from langchain_core.messages import ToolMessage
from langchain_core.runnables import ConfigurableField
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ValidationError

from langchain_jobo import (
    JoboCompanySearchTool,
    JoboConfigurationError,
    JoboFilterValuesTool,
    JoboInsufficientCreditsError,
    JoboJobDetailsTool,
    JoboLocationSearchTool,
    JoboNotFoundError,
    JoboRateLimitError,
    JoboSearchTool,
    get_jobo_tools,
)
from tests.unit_tests.helpers import (
    INSUFFICIENT_CREDITS,
    JOB_NOT_FOUND,
    NOT_IN_SANDBOX,
    RATE_LIMITED,
    SANDBOX_KEY,
    TEST_KEY,
    MockJobo,
    make_job,
    reply,
    search_handler,
)

JOB_ID = "00000000-0000-4000-8000-000000000042"

ALL_TOOLS: list[type[BaseTool]] = [
    JoboSearchTool,
    JoboJobDetailsTool,
    JoboCompanySearchTool,
    JoboFilterValuesTool,
    JoboLocationSearchTool,
]


def tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_call", "id": "call-1", "name": name, "args": args}


def model_facing(tool: BaseTool) -> dict[str, Any]:
    """The argument schema a chat model is sent for the tool."""
    schema = tool.tool_call_schema
    assert isinstance(schema, type)
    assert issubclass(schema, BaseModel)
    properties: dict[str, Any] = schema.model_json_schema()["properties"]
    return properties


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
            "include_facets": [],
        }
    ]


def test_search_defaults_to_a_small_page(search_api: MockJobo) -> None:
    JoboSearchTool(**search_api.kwargs).invoke({"query": "nurse"})

    assert search_api.bodies == [
        {"queries": ["nurse"], "page": 1, "page_size": 5, "include_facets": []}
    ]


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


def test_search_caps_any_page_size_instead_of_failing_the_call(
    search_api: MockJobo,
) -> None:
    """A model that asks for 500 jobs gets the cap and is told, not an error."""
    tool = JoboSearchTool(**search_api.kwargs)

    content = tool.invoke({"query": "nurse", "page_size": 500})

    assert search_api.bodies[0]["page_size"] == 25
    assert "page_size was capped at 25" in content


def test_search_still_rejects_a_page_size_below_one(search_api: MockJobo) -> None:
    tool = JoboSearchTool(**search_api.kwargs)

    with pytest.raises(ValidationError):
        tool.invoke({"query": "nurse", "page_size": 0})

    assert search_api.requests == []


@pytest.mark.parametrize(("cap", "default"), [(25, 5), (10, 5), (3, 3), (100, 5)])
def test_search_schema_states_the_real_cap_as_the_maximum(
    cap: int, default: int, search_api: MockJobo
) -> None:
    tool = JoboSearchTool(max_page_size=cap, **search_api.kwargs)

    for schema in (tool.args["page_size"], model_facing(tool)["page_size"]):
        assert schema["maximum"] == cap
        assert schema["minimum"] == 1
        assert schema["default"] == default
        assert f"at most {cap}" in schema["description"]

    tool.invoke({"query": "nurse"})
    assert search_api.bodies[0]["page_size"] == default


def test_search_schema_follows_the_cap_when_the_tool_is_reconfigured(
    search_api: MockJobo,
) -> None:
    """`configurable_fields` rebuilds the tool from its fields with a new cap."""
    tool = JoboSearchTool(**search_api.kwargs)
    fields = {name: getattr(tool, name) for name in JoboSearchTool.model_fields}

    rebuilt = JoboSearchTool(**{**fields, "max_page_size": 3})
    configurable = tool.configurable_fields(
        max_page_size=ConfigurableField(id="jobo_max_page_size")
    )
    configurable.with_config(configurable={"jobo_max_page_size": 4}).invoke(
        {"query": "nurse", "page_size": 50}
    )

    assert rebuilt.args["page_size"]["maximum"] == 3
    assert tool.args["page_size"]["maximum"] == 25
    assert search_api.bodies[0]["page_size"] == 4


def test_a_custom_search_schema_is_left_alone(search_api: MockJobo) -> None:
    class QueryOnly(BaseModel):
        query: str

    tool = JoboSearchTool(args_schema=QueryOnly, max_page_size=3, **search_api.kwargs)

    tool.invoke({"query": "nurse"})

    assert tool.args_schema is QueryOnly
    assert list(tool.args) == ["query"]
    assert search_api.bodies == [
        {"queries": ["nurse"], "page": 1, "page_size": 3, "include_facets": []}
    ]


def test_search_schema_leads_with_the_query() -> None:
    tool = JoboSearchTool(api_key=TEST_KEY)

    assert list(tool.args)[:2] == ["query", "location"]
    assert list(model_facing(tool))[:2] == ["query", "location"]
    assert list(tool.args)[-2:] == ["page", "page_size"]


def test_search_takes_several_locations(search_api: MockJobo) -> None:
    tool = JoboSearchTool(**search_api.kwargs)

    tool.invoke({"query": "nurse", "location": ["Berlin, Germany", "Paris"]})
    tool.invoke({"query": "nurse", "location": "Berlin, Germany"})

    assert [body["locations"] for body in search_api.bodies] == [
        ["Berlin, Germany", "Paris"],
        ["Berlin, Germany"],
    ]
    location = tool.args["location"]
    assert {"type": "string"} in location["anyOf"]
    assert {"type": "array", "items": {"type": "string"}} in location["anyOf"]


def test_search_does_not_split_a_company_name_on_its_comma(
    search_api: MockJobo,
) -> None:
    tool = JoboSearchTool(**search_api.kwargs)

    tool.invoke({"companies": "FedEx, Inc.", "skills": "Python, Go"})

    assert search_api.bodies[0]["companies"] == {"include": ["FedEx, Inc."]}
    assert search_api.bodies[0]["skills"] == {"include": ["Python", "Go"]}


def test_description_format_is_an_option_of_the_tool_not_an_argument(
    search_api: MockJobo,
) -> None:
    """It changes the artifact only, so the model is not offered it."""
    tool = JoboSearchTool(description_format="html", **search_api.kwargs)

    assert "description_format" not in tool.args
    with pytest.raises(ValidationError):
        tool.invoke({"query": "nurse", "description_format": "enhanced"})

    tool.invoke({"query": "nurse"})
    JoboSearchTool(**search_api.kwargs).invoke({"query": "nurse"})

    assert search_api.bodies[0]["description_format"] == "html"
    assert "description_format" not in search_api.bodies[1]


def test_search_field_descriptions_point_at_the_lookup_tools() -> None:
    args = JoboSearchTool(api_key=TEST_KEY).args

    assert "jobo_company_search" in args["companies"]["description"]
    assert "jobo_company_search" in args["exclude_companies"]["description"]
    assert "jobo_location_search" in args["location"]["description"]
    for field, facet in {
        "sources": "sources",
        "skills": "skills",
        "industries": "industries",
        "job_categories": "job_categories",
        "job_subcategories": "job_subcategories",
        "occupation_ids": "occupations",
    }.items():
        description = args[field]["description"]
        assert f"jobo_filter_values with facet `{facet}`" in description, field


def test_search_company_categories_are_a_closed_set(search_api: MockJobo) -> None:
    tool = JoboSearchTool(**search_api.kwargs)
    schema = tool.args["company_categories"]["anyOf"][0]

    assert "saas" in schema["items"]["enum"]
    assert len(schema["items"]["enum"]) == 13
    with pytest.raises(ValidationError):
        tool.invoke({"company_categories": ["startup"]})
    tool.invoke(
        {"company_categories": ["SaaS"], "exclude_company_categories": "agency"}
    )

    assert search_api.bodies == [
        {
            "company_categories": {"include": ["saas"], "exclude": ["agency"]},
            "page": 1,
            "page_size": 5,
            "include_facets": [],
        }
    ]


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
            "include_facets": [],
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


def test_rate_limit_answer_tells_the_model_how_long_to_wait() -> None:
    api = MockJobo(reply(429, RATE_LIMITED, headers={"Retry-After": "12"}))
    tool = JoboJobDetailsTool(max_retries=0, handle_tool_error=True, **api.kwargs)

    message = tool.invoke(tool_call("jobo_job_details", {"job_id": JOB_ID}))

    assert message.status == "error"
    assert message.content.endswith("Retry after 12 seconds.")


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


@pytest.mark.parametrize(
    "job_id", ["../companies/x?y=1", "42", "   ", "not-a-uuid", f"{JOB_ID}/extra"]
)
def test_details_id_that_is_not_a_uuid_is_answered_without_a_request(
    job_id: str,
) -> None:
    """The API routes only UUIDs: anything else would be a bare, unexplained 404."""
    api = MockJobo(lambda _: httpx.Response(200, json=make_job()))
    tool = JoboJobDetailsTool(**api.kwargs)

    message = tool.invoke(tool_call("jobo_job_details", {"job_id": job_id}))

    assert message.status == "success"
    assert message.content == (
        f'"{job_id.strip()}" is not a Jobo job id. Job ids are UUIDs: the `ID` '
        "values jobo_job_search returns."
    )
    assert message.artifact == {}
    assert api.requests == []


@pytest.mark.parametrize(
    "job_id",
    [JOB_ID.upper(), JOB_ID.replace("-", ""), f"{{{JOB_ID}}}", f"urn:uuid:{JOB_ID}"],
)
def test_details_sends_any_spelling_of_a_uuid_in_its_canonical_form(
    job_id: str,
) -> None:
    api = MockJobo(lambda _: httpx.Response(200, json=make_job(42)))

    JoboJobDetailsTool(**api.kwargs).invoke({"job_id": job_id})

    assert api.last.url.raw_path == f"/api/jobs/{JOB_ID}".encode()


def test_details_unknown_id_is_an_answer_not_an_error() -> None:
    api = MockJobo(reply(404, JOB_NOT_FOUND))
    tool = JoboJobDetailsTool(**api.kwargs)

    message = tool.invoke(tool_call("jobo_job_details", {"job_id": JOB_ID}))

    assert message.content.startswith(f'No job exists with id "{JOB_ID}".')
    assert message.status == "success"
    assert message.artifact == {}


@pytest.mark.parametrize(
    "body",
    [
        {"title": "Not Found", "status": 404},  # a route that does not exist
        {"message": "no Route matched with those values"},  # a gateway
        NOT_IN_SANDBOX,
        "<html>404 page not found</html>",
    ],
)
def test_details_404_that_is_not_job_not_found_is_an_error(body: Any) -> None:
    """A wrong `base_url` must not read as "that job does not exist"."""
    handler = reply(404, text=body) if isinstance(body, str) else reply(404, body)
    api = MockJobo(handler)
    tool = JoboJobDetailsTool(base_url="https://example.test/v9", **api.kwargs)

    with pytest.raises(ToolException, match="Jobo API error 404") as caught:
        tool.invoke({"job_id": JOB_ID})

    assert isinstance(caught.value.__cause__, JoboNotFoundError)
    assert str(api.last.url) == f"https://example.test/v9/api/jobs/{JOB_ID}"


def test_details_recognises_a_job_not_found_code_should_the_api_send_one() -> None:
    api = MockJobo(reply(404, {"title": "Gone", "code": "job_not_found"}))

    content = JoboJobDetailsTool(**api.kwargs).invoke({"job_id": JOB_ID})

    assert content.startswith("No job exists")


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
    api = MockJobo(reply(404, JOB_NOT_FOUND))
    tool = JoboJobDetailsTool(**api.kwargs)

    content = await tool.ainvoke({"job_id": JOB_ID})
    not_an_id = await tool.ainvoke({"job_id": "42"})

    assert content.startswith("No job exists")
    assert not_an_id.startswith('"42" is not a Jobo job id.')
    assert len(api.requests) == 1


async def test_details_async_other_404_is_an_error() -> None:
    api = MockJobo(reply(404, {"title": "Not Found"}))

    with pytest.raises(ToolException, match="Jobo API error 404"):
        await JoboJobDetailsTool(**api.kwargs).ainvoke({"job_id": JOB_ID})


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


@pytest.mark.parametrize("tool_class", ALL_TOOLS)
def test_missing_api_key_is_a_clear_error_at_construction(
    tool_class: type[BaseTool],
) -> None:
    with pytest.raises(JoboConfigurationError, match="JOBO_API_KEY") as caught:
        tool_class()

    assert "https://enterprise.jobo.world/api-keys" in str(caught.value)


@pytest.mark.parametrize("tool_class", ALL_TOOLS)
@pytest.mark.parametrize("key", ["", "   "])
def test_blank_api_key_is_refused_at_construction_not_on_first_call(
    tool_class: type[BaseTool], key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(JoboConfigurationError, match="JOBO_API_KEY"):
        tool_class(api_key=key)

    monkeypatch.setenv("JOBO_API_KEY", key)
    with pytest.raises(JoboConfigurationError, match="JOBO_API_KEY"):
        tool_class()


def test_api_key_is_trimmed_and_stays_a_secret(search_api: MockJobo) -> None:
    tool = JoboSearchTool(api_key=f" {TEST_KEY}\n", http_client=search_api.client)

    tool.invoke({"query": "nurse"})

    assert tool.api_key.get_secret_value() == TEST_KEY
    assert search_api.last.headers["X-Api-Key"] == TEST_KEY


@pytest.mark.parametrize("tool_class", ALL_TOOLS)
@pytest.mark.parametrize(
    "kwargs", [{"max_pagesize": 3}, {"apikey": "x"}, {"max_retry": 0}, {"k": 2}]
)
def test_mistyped_constructor_arguments_are_rejected(
    tool_class: type[BaseTool], kwargs: dict[str, Any]
) -> None:
    """`JoboSearchTool(max_pagesize=3)` used to build a tool capped at 25."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        tool_class(api_key=TEST_KEY, **kwargs)


@pytest.mark.parametrize("tool_class", ALL_TOOLS)
def test_langchain_tool_fields_are_still_accepted(tool_class: type[BaseTool]) -> None:
    tool = tool_class(
        api_key=TEST_KEY,
        handle_tool_error=True,
        tags=["jobs"],
        metadata={"team": "a"},
        return_direct=True,
    )

    assert tool.handle_tool_error is True
    assert tool.tags == ["jobs"]
    assert tool.return_direct is True


def test_a_subclass_can_change_defaults_and_add_fields(search_api: MockJobo) -> None:
    class SmallSearch(JoboSearchTool):
        name: str = "small_job_search"
        max_page_size: int = 2
        team: str = "data"

    tool = SmallSearch(team="platform", **search_api.kwargs)

    tool.invoke({"query": "nurse", "page_size": 50})

    assert tool.name == "small_job_search"
    assert tool.team == "platform"
    assert search_api.bodies[0]["page_size"] == 2
    assert tool.args["page_size"]["maximum"] == 2
    assert SmallSearch(max_page_size=7, **search_api.kwargs).max_page_size == 7
    with pytest.raises(ValidationError):
        SmallSearch(squad="x", **search_api.kwargs)


def test_tools_copy_and_dump_like_any_pydantic_model(search_api: MockJobo) -> None:
    tool = JoboSearchTool(max_page_size=7, **search_api.kwargs)

    copy = tool.model_copy(update={"description": "Find jobs."})
    dumped = tool.model_dump()

    assert copy.description == "Find jobs."
    assert copy.max_page_size == 7
    assert copy.invoke({"query": "nurse"}).startswith("1,234 matching jobs")
    assert dumped["max_page_size"] == 7
    assert "http_client" not in dumped


def test_api_key_is_masked_in_repr_and_dumps() -> None:
    tool = JoboSearchTool(api_key="jbe_live_do-not-print-me")

    assert "do-not-print-me" not in repr(tool)
    assert "do-not-print-me" not in str(tool)
    assert "do-not-print-me" not in str(tool.model_dump())


def test_tool_names_and_schemas() -> None:
    search, details, company, values, location = get_jobo_tools(api_key=TEST_KEY)

    assert search.name == "jobo_job_search"
    assert details.name == "jobo_job_details"
    assert company.name == "jobo_company_search"
    assert values.name == "jobo_filter_values"
    assert location.name == "jobo_location_search"
    for tool in (search, details, company, values, location):
        assert tool.description
        assert tool.response_format == "content_and_artifact"
        # Connection settings never leak into what the model is shown.
        assert "api_key" not in tool.args
    assert set(details.args) == {"job_id"}
    assert set(company.args) == {"query", "limit"}
    assert list(values.args) == ["facet", "query", "limit"]
    assert set(location.args) == {"query", "limit"}
    assert {"query", "location", "work_model", "skills", "page_size"} <= set(
        search.args
    )
    for free in (company, values, location):
        assert free.description.endswith("Free to call.")


def test_tools_can_be_deep_copied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Some agent frameworks copy the tools they are handed."""
    real = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda: real(transport=httpx.MockTransport(search_handler))
    )
    tool = JoboSearchTool(api_key=TEST_KEY, max_page_size=3)
    tool.invoke({"query": "nurse"})

    for twin in (copy.deepcopy(tool), tool.model_copy(deep=True)):
        assert twin.max_page_size == 3
        assert twin.api_key.get_secret_value() == TEST_KEY
        assert twin.invoke({"query": "nurse"}).startswith("1,234 matching jobs")


def test_get_jobo_tools_passes_settings_to_every_tool(search_api: MockJobo) -> None:
    tools = get_jobo_tools(
        max_page_size=7,
        base_url="https://example.test",
        timeout=12.0,
        max_retries=0,
        handle_tool_error=True,
        handle_validation_error="Fix the arguments.",
        **search_api.kwargs,
    )

    assert [type(tool) for tool in tools] == ALL_TOOLS
    assert isinstance(tools[0], JoboSearchTool)
    assert tools[0].max_page_size == 7
    for tool in tools:
        assert tool.handle_tool_error is True
        assert tool.handle_validation_error == "Fix the arguments."
        assert getattr(tool, "base_url", None) == "https://example.test"
        assert getattr(tool, "timeout", None) == 12.0
        assert getattr(tool, "max_retries", None) == 0

    tools[0].invoke({"query": "nurse"})
    assert search_api.last.url.host == "example.test"
    assert search_api.last.extensions["timeout"]["read"] == 12.0
    assert tools[0].invoke({"work_model": ["anywhere"]}) == "Fix the arguments."


def test_get_jobo_tools_defaults_leave_errors_raising(search_api: MockJobo) -> None:
    tools = get_jobo_tools(**search_api.kwargs)

    for tool in tools:
        assert tool.handle_tool_error is False
        assert tool.handle_validation_error is False
        assert getattr(tool, "base_url", None) == "https://connect.jobo.world"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_retry": 0},
        {"max_pagesize": 3},
        {"include_company_search": False},
        {"handle_tool_errors": True},
    ],
)
def test_get_jobo_tools_rejects_a_mistyped_argument(kwargs: dict[str, Any]) -> None:
    """It used to swallow any keyword and hand back default tools."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        get_jobo_tools(TEST_KEY, **kwargs)


def test_get_jobo_tools_reads_the_key_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBO_API_KEY", "jbe_test_from-env")

    tools = get_jobo_tools(include_lookups=False)

    assert [tool.name for tool in tools] == ["jobo_job_search", "jobo_job_details"]


def test_get_jobo_tools_without_a_key_is_a_clear_error() -> None:
    with pytest.raises(JoboConfigurationError, match="JOBO_API_KEY"):
        get_jobo_tools()
    with pytest.raises(JoboConfigurationError, match="JOBO_API_KEY"):
        get_jobo_tools(api_key=" ")


def test_get_jobo_tools_share_one_connection_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list[httpx.Client] = []
    real = httpx.Client

    def make() -> httpx.Client:
        made.append(real(transport=httpx.MockTransport(everything)))
        return made[-1]

    monkeypatch.setattr(httpx, "Client", make)
    search, details, company, values, location = get_jobo_tools(TEST_KEY)

    search.invoke({"query": "nurse"})
    details.invoke({"job_id": JOB_ID})
    company.invoke({"query": "stripe"})
    values.invoke({"facet": "skills"})
    location.invoke({"query": "berlin"})

    assert len(made) == 1
    assert isinstance(search, JoboSearchTool)
    search.close()
    assert made[0].is_closed
    # Closing one tool closes the shared pool; the others reopen it on demand.
    assert "Stripe" in company.invoke({"query": "stripe"})
    assert len(made) == 2


def test_a_tool_is_a_context_manager_and_leaves_your_client_open(
    search_api: MockJobo,
) -> None:
    with JoboSearchTool(**search_api.kwargs) as tool:
        tool.invoke({"query": "nurse"})

    assert isinstance(tool, JoboSearchTool)
    assert not search_api.client.is_closed


async def test_a_tool_is_an_async_context_manager(search_api: MockJobo) -> None:
    async with JoboSearchTool(**search_api.kwargs) as tool:
        await tool.ainvoke({"query": "nurse"})
    await tool.aclose()

    assert not search_api.async_client.is_closed


# --------------------------------------------------------------- filter values

INDUSTRIES = {
    "facet": "industries",
    "options": [
        {"key": "Software Development", "count": 120431},
        {"key": "Software, Data & AI", "count": 77},
    ],
}

TAXONOMY = {
    "version": "2026.11",
    "categories": [
        {
            "name": "Healthcare",
            "subcategories": [
                {
                    "name": "Nursing",
                    "occupations": [
                        {"id": "esco:nurse-aide", "name": "Nurse Aide"},
                        {"id": "esco:rn", "name": "Registered Nurse"},
                        {"id": "esco:midwife", "name": "Midwife"},
                        {"id": "esco:nurse", "name": "Nurse"},
                    ],
                },
                {
                    "name": "Medicine",
                    "occupations": [{"id": "esco:gp", "name": "General Practitioner"}],
                },
            ],
        },
        {
            "name": "Technology",
            "subcategories": [
                {
                    "name": "Software Engineering",
                    "occupations": [
                        {"id": "jobo:backend-developer", "name": "Backend Developer"}
                    ],
                }
            ],
        },
    ],
}


def test_filter_values_lists_a_facet_with_job_counts() -> None:
    api = MockJobo(reply(200, INDUSTRIES))
    tool = JoboFilterValuesTool(**api.kwargs)

    message = tool.invoke(
        tool_call("jobo_filter_values", {"facet": "industries", "query": " soft "})
    )

    assert api.last.method == "GET"
    assert api.last.url.path == "/api/connectors/filter-options/industries"
    assert dict(api.last.url.params) == {"limit": "20", "q": "soft"}
    assert message.content.startswith('2 `industries` values matching "soft"')
    assert "1. Software Development - 120,431 jobs" in message.content
    assert "`industries` (or `exclude_industries`) filter" in message.content
    assert message.artifact == {
        "facet": "industries",
        "query": "soft",
        "options": INDUSTRIES["options"],
    }


@pytest.mark.parametrize(
    "facet",
    [
        "sources",
        "industries",
        "skills",
        "company_categories",
        "job_categories",
        "job_subcategories",
        "occupations",
    ],
)
def test_filter_values_without_a_query_asks_for_the_most_common(facet: str) -> None:
    api = MockJobo(reply(200, {"facet": facet, "options": [{"key": "x", "count": 1}]}))
    tool = JoboFilterValuesTool(**api.kwargs)

    message = tool.invoke(tool_call("jobo_filter_values", {"facet": facet, "limit": 3}))

    assert api.last.url.path == f"/api/connectors/filter-options/{facet}"
    assert dict(api.last.url.params) == {"limit": "3"}
    assert message.artifact["query"] is None
    assert message.content.startswith(f"1 `{facet}` value, most jobs first.")


@pytest.mark.parametrize(
    "args",
    [
        {"facet": "companies"},  # jobo_company_search does companies
        {"facet": "locations"},
        {"facet": "skills", "limit": 51},
        {"facet": "skills", "limit": 0},
        {},
    ],
)
def test_filter_values_validates_arguments(args: dict[str, Any]) -> None:
    api = MockJobo(reply(200, INDUSTRIES))
    tool = JoboFilterValuesTool(**api.kwargs)

    with pytest.raises(ValidationError):
        tool.invoke(args)

    assert api.requests == []


def test_filter_values_finds_occupations_by_name_in_the_taxonomy() -> None:
    """The facet lookup matches occupation ids, so names need the taxonomy."""
    api = MockJobo(reply(200, TAXONOMY))
    tool = JoboFilterValuesTool(**api.kwargs)

    message = tool.invoke(
        tool_call("jobo_filter_values", {"facet": "occupations", "query": "Nurse"})
    )

    assert api.last.url.path == "/api/occupations"
    assert dict(api.last.url.params) == {}
    # The whole name first, then names that start with it, then the rest.
    assert message.content == (
        '3 occupations matching "Nurse". Pass the ids in the `occupation_ids` (or '
        "`exclude_occupation_ids`) filter of jobo_job_search.\n\n"
        "1. Nurse - id: esco:nurse (Healthcare > Nursing)\n"
        "2. Nurse Aide - id: esco:nurse-aide (Healthcare > Nursing)\n"
        "3. Registered Nurse - id: esco:rn (Healthcare > Nursing)"
    )
    assert message.artifact["options"][0] == {
        "key": "esco:nurse",
        "id": "esco:nurse",
        "name": "Nurse",
        "category": "Healthcare",
        "subcategory": "Nursing",
    }


def test_filter_values_ranks_a_whole_word_above_a_word_that_starts_with_it() -> None:
    """Seen on the live taxonomy: "nurse" led with a nursery school teacher."""
    taxonomy = {
        "version": "v1",
        "categories": [
            {
                "name": "Education",
                "subcategories": [
                    {
                        "name": "Early Education",
                        "occupations": [
                            {"id": "esco:nursery", "name": "Nursery School Teacher"}
                        ],
                    }
                ],
            },
            {
                "name": "Healthcare",
                "subcategories": [
                    {
                        "name": "Nursing",
                        "occupations": [
                            {"id": "esco:rn", "name": "Registered Nurse"},
                            {"id": "esco:na", "name": "Nurse Assistant"},
                        ],
                    }
                ],
            },
        ],
    }
    api = MockJobo(reply(200, taxonomy))
    tool = JoboFilterValuesTool(**api.kwargs)

    message = tool.invoke(
        tool_call("jobo_filter_values", {"facet": "occupations", "query": "nurse"})
    )

    assert [option["name"] for option in message.artifact["options"]] == [
        "Registered Nurse",
        "Nurse Assistant",
        "Nursery School Teacher",
    ]


def test_filter_values_occupation_search_matches_every_word_and_the_limit() -> None:
    api = MockJobo(reply(200, TAXONOMY))
    tool = JoboFilterValuesTool(**api.kwargs)

    words = tool.invoke({"facet": "occupations", "query": "developer backend"})
    by_category = tool.invoke({"facet": "occupations", "query": "healthcare"})
    by_subcategory = tool.invoke({"facet": "occupations", "query": "nursing"})
    limited = tool.invoke({"facet": "occupations", "query": "nurse", "limit": 2})
    nothing = tool.invoke({"facet": "occupations", "query": "astronaut"})

    assert "1. Backend Developer - id: jobo:backend-developer" in words
    # A category or subcategory name finds the occupations filed under it.
    assert by_category.startswith('5 occupations matching "healthcare".')
    assert by_subcategory.startswith('4 occupations matching "nursing".')
    assert "Midwife - id: esco:midwife (Healthcare > Nursing)" in by_subcategory
    assert limited.startswith('2 occupations matching "nurse".')
    assert nothing.startswith('No occupation matches "astronaut".')


def test_filter_values_keeps_the_taxonomy_between_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = MockJobo(reply(200, TAXONOMY))
    tool = JoboFilterValuesTool(**api.kwargs)
    clock = [1000.0]
    monkeypatch.setattr("langchain_jobo.tools.time.monotonic", lambda: clock[0])

    tool.invoke({"facet": "occupations", "query": "nurse"})
    tool.invoke({"facet": "occupations", "query": "midwife"})
    assert len(api.requests) == 1

    clock[0] += 601  # ten minutes on, it is read again
    tool.invoke({"facet": "occupations", "query": "nurse"})
    assert len(api.requests) == 2


def test_filter_values_api_errors_become_tool_exceptions() -> None:
    api = MockJobo(reply(401, {"error": "Invalid or expired API key"}))
    tool = JoboFilterValuesTool(**api.kwargs)

    with pytest.raises(ToolException, match="Invalid or expired API key"):
        tool.invoke({"facet": "skills"})
    with pytest.raises(ToolException, match="Invalid or expired API key"):
        tool.invoke({"facet": "occupations", "query": "nurse"})


async def test_filter_values_async() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/occupations":
            return httpx.Response(200, json=TAXONOMY)
        return httpx.Response(200, json=INDUSTRIES)

    api = MockJobo(handler)
    tool = JoboFilterValuesTool(**api.kwargs)

    values = await tool.ainvoke({"facet": "industries", "query": "soft", "limit": 5})
    occupations = await tool.ainvoke({"facet": "occupations", "query": "midwife"})
    again = await tool.ainvoke({"facet": "occupations", "query": "nurse aide"})

    assert "Software Development - 120,431 jobs" in values
    assert dict(api.requests[0].url.params) == {"limit": "5", "q": "soft"}
    assert "1. Midwife - id: esco:midwife" in occupations
    assert "1. Nurse Aide - id: esco:nurse-aide" in again
    assert [request.url.path for request in api.requests] == [
        "/api/connectors/filter-options/industries",
        "/api/occupations",
    ]


# -------------------------------------------------------------------- location

PLACES = {
    "suggestions": [
        {
            "city": "Springfield",
            "region": "Illinois",
            "country": "United States",
            "latitude": 39.8,
            "longitude": -89.6,
            "display_name": "Springfield, Illinois, United States",
            "country_code": "US",
        },
        {
            "city": "Springfield",
            "region": "Missouri",
            "country": "United States",
            "display_name": None,
            "country_code": "US",
        },
    ]
}


def test_location_search_calls_the_suggest_endpoint() -> None:
    api = MockJobo(reply(200, PLACES))
    tool = JoboLocationSearchTool(**api.kwargs)

    message = tool.invoke(
        tool_call("jobo_location_search", {"query": " Springfield ", "limit": 2})
    )

    assert api.last.method == "GET"
    assert api.last.url.path == "/api/connectors/locations/suggest"
    assert dict(api.last.url.params) == {"q": "Springfield", "limit": "2"}
    assert message.content == (
        '2 places matching "Springfield". Pass one, exactly as written, as the '
        "`location` filter of jobo_job_search.\n\n"
        "1. Springfield, Illinois, United States\n"
        "2. Springfield, Missouri, United States"
    )
    assert message.artifact == PLACES


def test_location_search_with_no_match() -> None:
    api = MockJobo(reply(200, {"suggestions": []}))

    message = JoboLocationSearchTool(**api.kwargs).invoke(
        tool_call("jobo_location_search", {"query": "zz"})
    )

    assert message.content.startswith('No place matched "zz".')
    assert message.artifact == {"suggestions": []}
    assert dict(api.last.url.params) == {"q": "zz", "limit": "5"}


@pytest.mark.parametrize("args", [{"query": "a"}, {"query": "berlin", "limit": 11}])
def test_location_search_validates_arguments(args: dict[str, Any]) -> None:
    api = MockJobo(reply(200, PLACES))

    with pytest.raises(ValidationError):
        JoboLocationSearchTool(**api.kwargs).invoke(args)

    assert api.requests == []


async def test_location_search_async() -> None:
    api = MockJobo(reply(200, PLACES))

    content = await JoboLocationSearchTool(**api.kwargs).ainvoke({"query": "spring"})

    assert dict(api.last.url.params) == {"q": "spring", "limit": "5"}
    assert "1. Springfield, Illinois, United States" in content


# --------------------------------------------------------------------- sandbox


def everything(request: httpx.Request) -> httpx.Response:
    """A live API: every endpoint the five tools call answers."""
    path = request.url.path
    if path == "/api/jobs/search":
        return search_handler(request)
    if path == "/api/connectors/companies/suggest":
        return httpx.Response(200, json=SUGGESTIONS)
    if path == "/api/connectors/locations/suggest":
        return httpx.Response(200, json=PLACES)
    if path.startswith("/api/connectors/filter-options/"):
        return httpx.Response(200, json=INDUSTRIES)
    if path == "/api/occupations":
        return httpx.Response(200, json=TAXONOMY)
    return httpx.Response(200, json=make_job(42))


def sandbox(request: httpx.Request) -> httpx.Response:
    """The sandbox: search and job lookups only, a coded 404 for the rest."""
    path = request.url.path
    if path == "/api/jobs/search":
        return search_handler(request)
    if path.startswith("/api/jobs/"):
        return httpx.Response(200, json=make_job(42))
    return httpx.Response(404, json={**NOT_IN_SANDBOX, "instance": path})


def sandbox_tools(**kwargs: Any) -> tuple[MockJobo, dict[str, BaseTool]]:
    api = MockJobo(sandbox)
    tools = get_jobo_tools(
        SANDBOX_KEY,
        http_client=api.client,
        http_async_client=api.async_client,
        **kwargs,
    )
    return api, {tool.name: tool for tool in tools}


LOOKUP_CALLS: list[tuple[str, dict[str, Any], str, str]] = [
    (
        "jobo_company_search",
        {"query": "stripe"},
        "Companies cannot be looked up here",
        "as the `companies` filter of jobo_job_search",
    ),
    (
        "jobo_filter_values",
        {"facet": "industries"},
        "Filter values cannot be looked up here",
        "Call jobo_job_search without the lookup",
    ),
    (
        "jobo_filter_values",
        {"facet": "occupations", "query": "nurse"},
        "Filter values cannot be looked up here",
        "Call jobo_job_search without the lookup",
    ),
    (
        "jobo_location_search",
        {"query": "berlin"},
        "Places cannot be looked up here",
        "as the `location` filter of jobo_job_search",
    ),
]


@pytest.mark.parametrize(("name", "args", "opening", "advice"), LOOKUP_CALLS)
def test_a_lookup_the_sandbox_lacks_answers_the_model_instead_of_raising(
    name: str, args: dict[str, Any], opening: str, advice: str
) -> None:
    """`handle_tool_error` is off: a raise here would end the agent's run."""
    api, tools = sandbox_tools()

    message = tools[name].invoke(tool_call(name, args))

    assert message.status == "success"
    assert message.content.startswith(opening)
    assert "a sandbox API key (jbe_test_...) serves job search" in message.content
    assert "Do not call this tool again." in message.content
    assert advice in message.content
    assert message.artifact["unavailable"] == "not_available_in_sandbox"
    found = message.artifact.get("suggestions", message.artifact.get("options"))
    assert found == []
    assert len(api.requests) == 1
    assert SANDBOX_KEY not in message.content


@pytest.mark.parametrize(("name", "args", "opening", "advice"), LOOKUP_CALLS)
async def test_a_lookup_the_sandbox_lacks_answers_the_model_async(
    name: str, args: dict[str, Any], opening: str, advice: str
) -> None:
    _, tools = sandbox_tools()

    content = await tools[name].ainvoke(args)

    assert content.startswith(opening)
    assert advice in content


def test_search_and_details_work_with_a_sandbox_key() -> None:
    api, tools = sandbox_tools()

    found = tools["jobo_job_search"].invoke({"query": "nurse", "page_size": 1})
    job = tools["jobo_job_details"].invoke({"job_id": JOB_ID})

    assert found.startswith("1,234 matching jobs; showing 1")
    assert job.startswith("# Senior Backend Engineer 42")
    assert api.last.headers["X-Api-Key"] == SANDBOX_KEY


@pytest.mark.parametrize(
    ("tool_class", "args"),
    [
        (JoboCompanySearchTool, {"query": "stripe"}),
        (JoboFilterValuesTool, {"facet": "skills"}),
        (JoboFilterValuesTool, {"facet": "occupations", "query": "nurse"}),
        (JoboLocationSearchTool, {"query": "berlin"}),
    ],
)
def test_any_other_404_from_a_lookup_is_still_an_error(
    tool_class: type[BaseTool], args: dict[str, Any]
) -> None:
    """Only the sandbox's own refusal is an answer: a wrong `base_url` is not."""
    api = MockJobo(reply(404, {"title": "Not Found", "status": 404}))
    tool = tool_class(**api.kwargs)

    with pytest.raises(ToolException, match="Jobo API error 404") as caught:
        tool.invoke(args)

    assert isinstance(caught.value.__cause__, JoboNotFoundError)
