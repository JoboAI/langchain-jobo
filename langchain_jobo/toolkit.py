"""One call that returns the Jobo tools, ready to hand to an agent."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import SecretStr

from langchain_jobo.tools import (
    DEFAULT_MAX_PAGE_SIZE,
    JoboCompanySearchTool,
    JoboJobDetailsTool,
    JoboSearchTool,
)


def get_jobo_tools(
    api_key: SecretStr | str | None = None,
    *,
    max_page_size: int = DEFAULT_MAX_PAGE_SIZE,
    include_company_search: bool = True,
    **kwargs: Any,
) -> list[BaseTool]:
    """Return the Jobo tools: job search, job details and company lookup.

    Args:
        api_key: Jobo API key. Read from the `JOBO_API_KEY` environment
            variable when not passed.
        max_page_size: Hard cap on jobs the search tool requests per call.
            Search is billed per job returned.
        include_company_search: Include `jobo_company_search`, the free lookup
            that resolves a company name to the ids the search filter takes.
        **kwargs: Connection settings passed to every tool: `base_url`,
            `timeout`, `max_retries`, `http_client`, `http_async_client`, and
            `handle_tool_error` to hand API errors back to the model instead
            of raising.

    Returns:
        `[JoboSearchTool, JoboJobDetailsTool, JoboCompanySearchTool]`, without
        the last one when `include_company_search` is false.

    Example:
        ```python
        from langchain_jobo import get_jobo_tools

        tools = get_jobo_tools(max_page_size=10, handle_tool_error=True)
        ```
    """
    if api_key is not None:
        kwargs["api_key"] = api_key
    tools: list[BaseTool] = [
        JoboSearchTool(max_page_size=max_page_size, **kwargs),
        JoboJobDetailsTool(**kwargs),
    ]
    if include_company_search:
        tools.append(JoboCompanySearchTool(**kwargs))
    return tools


__all__ = ["get_jobo_tools"]
