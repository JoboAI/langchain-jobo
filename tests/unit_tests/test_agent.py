"""The tools inside a LangChain agent, as the README shows them.

A scripted chat model stands in for a real one, so this runs offline. It
proves `create_agent` accepts the tools, calls them with model-written
arguments, and feeds the compact text (not the raw jobs) back to the model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import ToolException

from langchain_jobo import get_jobo_tools
from tests.unit_tests.helpers import (
    INSUFFICIENT_CREDITS,
    NOT_IN_SANDBOX,
    SANDBOX_KEY,
    MockJobo,
    reply,
    search_handler,
)

agents = pytest.importorskip("langchain.agents")


class ScriptedChatModel(GenericFakeChatModel):
    """A fake chat model that tolerates `bind_tools`."""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> BaseChatModel:
        return self


def test_agent_searches_and_cites_a_job(search_api: MockJobo) -> None:
    script = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "type": "tool_call",
                        "id": "call-1",
                        "name": "jobo_job_search",
                        "args": {
                            "query": "backend engineer",
                            "work_model": ["remote"],
                            "page_size": 2,
                        },
                    }
                ],
            ),
            AIMessage(content="Acme Corp is hiring: https://jobs.acme.example/0"),
        ]
    )
    agent = agents.create_agent(
        model=ScriptedChatModel(messages=script),
        tools=get_jobo_tools(**search_api.kwargs),
        system_prompt="You help people find jobs. Cite the listing URL.",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Find remote backend jobs."}]}
    )

    assert search_api.bodies == [
        {
            "queries": ["backend engineer"],
            "work_models": ["remote"],
            "page": 1,
            "page_size": 2,
        }
    ]
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "URL: https://jobs.acme.example/0" in tool_messages[0].content
    assert len(tool_messages[0].artifact["jobs"]) == 2
    assert result["messages"][-1].content.endswith("https://jobs.acme.example/0")


def _one_search_call() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "type": "tool_call",
                "id": "call-1",
                "name": "jobo_job_search",
                "args": {"query": "nurse"},
            }
        ],
    )


def test_api_error_stops_the_run_by_default() -> None:
    api = MockJobo(reply(402, INSUFFICIENT_CREDITS))
    agent = agents.create_agent(
        model=ScriptedChatModel(messages=iter([_one_search_call()])),
        tools=get_jobo_tools(**api.kwargs),
    )

    with pytest.raises(ToolException, match="This request needs 75 credits"):
        agent.invoke({"messages": [{"role": "user", "content": "Find nurse jobs."}]})

    assert len(api.requests) == 1


def test_handle_tool_error_lets_the_model_see_the_api_error() -> None:
    api = MockJobo(reply(402, INSUFFICIENT_CREDITS))
    script = iter([_one_search_call(), AIMessage(content="Your wallet is empty.")])
    agent = agents.create_agent(
        model=ScriptedChatModel(messages=script),
        tools=get_jobo_tools(handle_tool_error=True, **api.kwargs),
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Find nurse jobs."}]}
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages[0].status == "error"
    assert "This request needs 75 credits" in tool_messages[0].content
    assert result["messages"][-1].content == "Your wallet is empty."
    assert len(api.requests) == 1


def _sandbox(request: httpx.Request) -> httpx.Response:
    """The sandbox serves search and job lookups, and refuses the rest."""
    if request.url.path == "/api/jobs/search":
        return search_handler(request)
    return httpx.Response(404, json=NOT_IN_SANDBOX)


def test_default_toolkit_with_a_sandbox_key_survives_a_lookup() -> None:
    """The lookups are not served by the sandbox. That must not end the run."""
    api = MockJobo(_sandbox)
    script = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "type": "tool_call",
                        "id": "call-1",
                        "name": "jobo_company_search",
                        "args": {"query": "acme"},
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "type": "tool_call",
                        "id": "call-2",
                        "name": "jobo_job_search",
                        "args": {"companies": ["Acme Corp"], "page_size": 1},
                    }
                ],
            ),
            AIMessage(content="Acme Corp is hiring: https://jobs.acme.example/0"),
        ]
    )
    agent = agents.create_agent(
        model=ScriptedChatModel(messages=script),
        tools=get_jobo_tools(
            SANDBOX_KEY, http_client=api.client, http_async_client=api.async_client
        ),
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Find jobs at Acme."}]}
    )

    lookup, search = (m for m in result["messages"] if isinstance(m, ToolMessage))
    assert lookup.status == "success"
    assert isinstance(lookup.content, str)
    assert lookup.content.startswith("Companies cannot be looked up here")
    assert "`companies` filter of jobo_job_search" in lookup.content
    assert search.status == "success"
    assert len(search.artifact["jobs"]) == 1
    assert [request.url.path for request in api.requests] == [
        "/api/connectors/companies/suggest",
        "/api/jobs/search",
    ]
    assert api.bodies[-1]["companies"] == {"include": ["Acme Corp"]}
    assert result["messages"][-1].content.endswith("https://jobs.acme.example/0")


def test_agent_is_given_all_five_tools_by_default(search_api: MockJobo) -> None:
    tools = get_jobo_tools(**search_api.kwargs)
    agent = agents.create_agent(
        model=ScriptedChatModel(messages=iter([AIMessage(content="Hello.")])),
        tools=tools,
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "Hi."}]})

    assert [tool.name for tool in tools] == [
        "jobo_job_search",
        "jobo_job_details",
        "jobo_company_search",
        "jobo_filter_values",
        "jobo_location_search",
    ]
    assert result["messages"][-1].content == "Hello."
    assert search_api.requests == []
