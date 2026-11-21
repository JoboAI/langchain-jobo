"""The public surface of the package."""

from __future__ import annotations

import langchain_jobo

COMPONENTS = [
    "JoboCompanySearchTool",
    "JoboFeedLoader",
    "JoboFilterValuesTool",
    "JoboJobDetailsTool",
    "JoboLocationSearchTool",
    "JoboRetriever",
    "JoboSearchInput",
    "JoboSearchTool",
    "get_jobo_tools",
]

ERRORS = [
    "JoboAPIError",
    "JoboAuthenticationError",
    "JoboConfigurationError",
    "JoboConflictError",
    "JoboConnectionError",
    "JoboError",
    "JoboFeedCursorRestartRequiredError",
    "JoboInsufficientCreditsError",
    "JoboNotFoundError",
    "JoboPermissionError",
    "JoboRateLimitError",
    "JoboServerError",
    "JoboValidationError",
]

#: The filter models and the helper models a feed filter is built from.
FILTER_MODELS = [
    "CompanyCategoryFilter",
    "CompanyRevenueFilter",
    "CompanySizeFilter",
    "FeedLocation",
    "IncludeExclude",
    "JoboFeedFilters",
    "JoboSearchFilters",
]

#: The same filters spelled as plain dicts.
FILTER_DICTS = [
    "CompanyCategoryFilterDict",
    "CompanyRevenueFilterDict",
    "CompanySizeFilterDict",
    "FeedLocationDict",
    "IncludeExcludeDict",
    "JoboFeedFiltersDict",
    "JoboSearchFiltersDict",
]

#: The `Literal` aliases of the fixed value sets.
LITERALS = [
    "CompanyCategory",
    "CompanyRevenue",
    "CompanySize",
    "DescriptionFormat",
    "EmploymentType",
    "ExperienceLevel",
    "FilterFacet",
    "SearchSort",
    "WorkModel",
]

EXPECTED_ALL = [
    *COMPONENTS,
    *ERRORS,
    *FILTER_MODELS,
    *FILTER_DICTS,
    *LITERALS,
    "__version__",
]


def test_all_imports() -> None:
    assert sorted(langchain_jobo.__all__) == sorted(EXPECTED_ALL)
    assert len(set(langchain_jobo.__all__)) == len(langchain_jobo.__all__)
    for name in EXPECTED_ALL:
        assert hasattr(langchain_jobo, name), name


def test_every_error_derives_from_the_package_base() -> None:
    for name in ERRORS:
        assert issubclass(getattr(langchain_jobo, name), langchain_jobo.JoboError)


def test_every_filter_model_has_a_dict_spelling() -> None:
    assert [f"{name}Dict" for name in FILTER_MODELS] == FILTER_DICTS


def test_version_is_a_release_string() -> None:
    assert langchain_jobo.__version__ == "0.1.0"
