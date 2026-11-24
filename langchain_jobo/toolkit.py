"""One call that returns the Jobo tools, ready to hand to an agent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
from langchain_core.tools import BaseTool, ToolException
from pydantic import SecretStr, ValidationError

from langchain_jobo._base import given
from langchain_jobo._client import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
)
from langchain_jobo.tools import (
    DEFAULT_MAX_PAGE_SIZE,
    JoboCompanySearchTool,
    JoboFilterValuesTool,
    JoboJobDetailsTool,
    JoboLocationSearchTool,
    JoboSearchTool,
    _JoboTool,
)


def get_jobo_tools(
    api_key: SecretStr | str | None = None,
    *,
    max_page_size: int = DEFAULT_MAX_PAGE_SIZE,
    include_lookups: bool = True,
    handle_tool_error: bool | str | Callable[[ToolException], str] = False,
    handle_validation_error: bool | str | Callable[[ValidationError], str] = False,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    http_client: httpx.Client | None = None,
    http_async_client: httpx.AsyncClient | None = None,
) -> list[BaseTool]:
    """Return the Jobo tools: job search, job details and the free lookups.

    Every parameter is named, so a mistyped one is an error and not silently
    ignored. The tools share one connection pool.

    Args:
        api_key: Jobo API key. Read from the `JOBO_API_KEY` environment
            variable when not passed.
        max_page_size: Hard cap on jobs the search tool requests per call.
            Search is billed per job returned.
        include_lookups: Include the three free lookups a model uses to find
            valid filter values: `jobo_company_search`, `jobo_filter_values`
            and `jobo_location_search`. The sandbox does not serve them: with
            a sandbox key they answer the model with a plain note instead of
            failing, and `include_lookups=False` leaves them out.
        handle_tool_error: Hand an API error (a rate limit, an empty wallet)
            back to the model as the tool's answer instead of raising
            `ToolException`. `True`, a fixed message, or a function of the
            exception.
        handle_validation_error: The same for arguments the model got wrong,
            e.g. an unknown `work_model`. By default they raise.
        base_url: API base URL.
        timeout: Response timeout in seconds.
        max_retries: Retries after the first attempt, for 429, 500 and 503
            responses and failures to connect. A 503 without `Retry-After` is
            not retried, nor is a 500 on the billed search.
        http_client: Optional `httpx.Client` to send sync requests with.
        http_async_client: Optional `httpx.AsyncClient` for async requests.

    Returns:
        `[JoboSearchTool, JoboJobDetailsTool, JoboCompanySearchTool,
        JoboFilterValuesTool, JoboLocationSearchTool]`, or the first two when
        `include_lookups` is false.

    Raises:
        JoboConfigurationError: No API key was passed or found.

    Example:
        ```python
        from langchain_jobo import get_jobo_tools

        tools = get_jobo_tools(max_page_size=10, handle_tool_error=True)
        ```
    """
    shared: dict[str, Any] = {
        "base_url": base_url,
        "timeout": timeout,
        "max_retries": max_retries,
        "handle_tool_error": handle_tool_error,
        "handle_validation_error": handle_validation_error,
        **given(
            api_key=api_key,
            http_client=http_client,
            http_async_client=http_async_client,
        ),
    }
    tools: list[_JoboTool] = [
        JoboSearchTool(max_page_size=max_page_size, **shared),
        JoboJobDetailsTool(**shared),
    ]
    if include_lookups:
        tools += [
            JoboCompanySearchTool(**shared),
            JoboFilterValuesTool(**shared),
            JoboLocationSearchTool(**shared),
        ]
    for tool in tools[1:]:
        tool._share_connections_with(tools[0])
    return list(tools)


__all__ = ["get_jobo_tools"]
