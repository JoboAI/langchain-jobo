"""LangChain's standard test suites, run offline against a mocked API.

`langchain-tests` ships unit tests for tools only. Its tool and retriever
*integration* suites need nothing but a working component, so they run here
too, with the HTTP layer replaced by a mock transport.
"""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
    ToolsIntegrationTests,
)
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_jobo import (
    JoboCompanySearchTool,
    JoboFilterValuesTool,
    JoboJobDetailsTool,
    JoboLocationSearchTool,
    JoboRetriever,
    JoboSearchTool,
)
from tests.unit_tests.helpers import TEST_KEY, MockJobo, make_job, search_handler

EnvParams = tuple[dict[str, str], dict[str, Any], dict[str, Any]]
ENV_PARAMS: EnvParams = (
    {"JOBO_API_KEY": "jbe_test_from-env"},
    {},
    {"api_key": "jbe_test_from-env"},
)

SUGGESTIONS = {
    "suggestions": [
        {
            "id": "id-1",
            "ids": ["id-1"],
            "name": "Stripe",
            "domain": "stripe.com",
            "logo_url": None,
            "country": "US",
            "job_count": 12,
        }
    ]
}


FILTER_OPTIONS = {
    "facet": "industries",
    "options": [{"key": "Software Development", "count": 120431}],
}

PLACES = {
    "suggestions": [
        {
            "city": "Berlin",
            "region": "Berlin",
            "country": "Germany",
            "display_name": "Berlin, Germany",
            "country_code": "DE",
        }
    ]
}


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/jobs/search":
        return search_handler(request)
    if path == "/api/connectors/companies/suggest":
        return httpx.Response(200, json=SUGGESTIONS)
    if path == "/api/connectors/locations/suggest":
        return httpx.Response(200, json=PLACES)
    if path.startswith("/api/connectors/filter-options/"):
        return httpx.Response(200, json=FILTER_OPTIONS)
    if path.startswith("/api/jobs/"):
        return httpx.Response(200, json=make_job(1))
    return httpx.Response(404, json={"title": "Not Found"})


def _mocked() -> dict[str, Any]:
    return MockJobo(_handler).kwargs


class TestJoboSearchToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return JoboSearchTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": TEST_KEY}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "query": "backend engineer",
            "location": "Berlin, Germany",
            "work_model": ["remote"],
            "page_size": 3,
        }

    @property
    def init_from_env_params(self) -> EnvParams:
        return ENV_PARAMS


class TestJoboSearchToolMocked(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return JoboSearchTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return _mocked()

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "backend engineer", "work_model": ["remote"], "page_size": 3}


class TestJoboJobDetailsToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return JoboJobDetailsTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": TEST_KEY}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"job_id": "00000000-0000-4000-8000-000000000001"}

    @property
    def init_from_env_params(self) -> EnvParams:
        return ENV_PARAMS


class TestJoboJobDetailsToolMocked(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return JoboJobDetailsTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return _mocked()

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"job_id": "00000000-0000-4000-8000-000000000001"}


class TestJoboCompanySearchToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return JoboCompanySearchTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": TEST_KEY}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "stripe", "limit": 3}

    @property
    def init_from_env_params(self) -> EnvParams:
        return ENV_PARAMS


class TestJoboCompanySearchToolMocked(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return JoboCompanySearchTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return _mocked()

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "stripe", "limit": 3}


class TestJoboFilterValuesToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return JoboFilterValuesTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": TEST_KEY}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"facet": "industries", "query": "soft", "limit": 3}

    @property
    def init_from_env_params(self) -> EnvParams:
        return ENV_PARAMS


class TestJoboFilterValuesToolMocked(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return JoboFilterValuesTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return _mocked()

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"facet": "industries", "query": "soft", "limit": 3}


class TestJoboLocationSearchToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return JoboLocationSearchTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": TEST_KEY}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "berlin", "limit": 3}

    @property
    def init_from_env_params(self) -> EnvParams:
        return ENV_PARAMS


class TestJoboLocationSearchToolMocked(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return JoboLocationSearchTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return _mocked()

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "berlin", "limit": 3}


class TestJoboRetrieverMocked(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> type[BaseRetriever]:
        return JoboRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        return _mocked()

    @property
    def retriever_query_example(self) -> str:
        return "backend engineer"
