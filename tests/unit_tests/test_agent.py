"""The tools inside a LangChain agent, as the README shows them.

A scripted chat model stands in for a real one, so this runs offline. It
proves `create_agent` accepts the tools, calls them with model-written
arguments, and feeds the compact text (not the raw jobs) back to the model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import ToolException

from langchain_jobo import get_jobo_tools
from tests.unit_tests.helpers import INSUFFICIENT_CREDITS, MockJobo, reply

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
