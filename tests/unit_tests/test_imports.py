"""The public surface of the package."""

from __future__ import annotations

import langchain_jobo

EXPECTED_ALL = [
    "JoboAPIError",
    "JoboAuthenticationError",
    "JoboCompanySearchTool",
    "JoboConflictError",
    "JoboConnectionError",
    "JoboError",
    "JoboFeedCursorRestartRequiredError",
    "JoboFeedFilters",
    "JoboFeedLoader",
    "JoboInsufficientCreditsError",
    "JoboJobDetailsTool",
    "JoboNotFoundError",
    "JoboPermissionError",
    "JoboRateLimitError",
    "JoboRetriever",
    "JoboSearchFilters",
    "JoboSearchInput",
    "JoboSearchTool",
    "JoboServerError",
    "JoboValidationError",
    "__version__",
    "get_jobo_tools",
]


def test_all_imports() -> None:
    assert sorted(langchain_jobo.__all__) == sorted(EXPECTED_ALL)
    for name in EXPECTED_ALL:
        assert hasattr(langchain_jobo, name), name


def test_version_is_a_release_string() -> None:
    assert langchain_jobo.__version__ == "0.1.0"
