"""Filter validation and the mapping onto search and feed request bodies."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from langchain_jobo import JoboFeedFilters, JoboSearchFilters, JoboSearchInput
from langchain_jobo.filters import build_feed_body, build_search_body


def search_body(**fields: Any) -> dict[str, Any]:
    return build_search_body(JoboSearchFilters(**fields))


def test_no_filters_builds_an_empty_body() -> None:
    assert search_body() == {}


def test_query_and_location_become_single_entry_lists() -> None:
    body = build_search_body(
        JoboSearchFilters(location="  Berlin, Germany "), query="  data engineer "
    )

    assert body == {"queries": ["data engineer"], "locations": ["Berlin, Germany"]}


def test_blank_query_and_location_are_left_out() -> None:
    assert build_search_body(JoboSearchFilters(location="  "), query=" ") == {}


@pytest.mark.parametrize(
    "field",
    [
        "skills",
        "companies",
        "industries",
        "company_categories",
        "job_categories",
        "job_subcategories",
        "occupation_ids",
    ],
)
def test_include_exclude_pairs_nest_under_one_key(field: str) -> None:
    both = search_body(**{field: ["a", "b"], f"exclude_{field}": ["c"]})
    include_only = search_body(**{field: ["a"]})
    exclude_only = search_body(**{f"exclude_{field}": ["c"]})

    assert both == {field: {"include": ["a", "b"], "exclude": ["c"]}}
    assert include_only == {field: {"include": ["a"]}}
    assert exclude_only == {field: {"exclude": ["c"]}}
    # A flat `exclude_*` key is ignored by the API, so it must never be sent.
    assert f"exclude_{field}" not in both


def test_enum_list_filters_use_the_plural_body_keys() -> None:
    body = search_body(
        work_model=["remote", "hybrid"],
        employment_type=["full-time"],
        experience_level=["senior", "lead"],
        language=["en", "de"],
        sources=["greenhouse", "lever"],
    )

    assert body == {
        "sources": ["greenhouse", "lever"],
        "work_models": ["remote", "hybrid"],
        "employment_types": ["full-time"],
        "experience_levels": ["senior", "lead"],
        "languages": ["en", "de"],
    }


def test_a_single_string_is_split_on_commas_and_newlines() -> None:
    filters = JoboSearchFilters(
        skills="Python, Kubernetes\nGo",
        work_model="Remote, HYBRID",
    )

    assert filters.skills == ["Python", "Kubernetes", "Go"]
    assert filters.work_model == ["remote", "hybrid"]


def test_list_entries_are_not_comma_split() -> None:
    """Some values contain commas, e.g. a subcategory name."""
    filters = JoboSearchFilters(job_subcategories=["Film, TV & Stage Production"])

    assert filters.job_subcategories == ["Film, TV & Stage Production"]


def test_lists_are_trimmed_and_deduplicated_and_blanks_dropped() -> None:
    filters = JoboSearchFilters(skills=[" Python ", "", "Python", "Go", "  "])

    assert filters.skills == ["Python", "Go"]
    assert search_body(skills=["", "  "]) == {}


def test_enum_values_are_lowercased() -> None:
    filters = JoboSearchFilters(
        work_model=["Remote"],
        employment_type=["Full-Time"],
        sort="DISCOVERED_DESC",
    )

    assert filters.work_model == ["remote"]
    assert filters.employment_type == ["full-time"]
    assert filters.sort == "discovered_desc"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("work_model", ["on-site"]),
        ("employment_type", ["full_time"]),
        ("experience_level", ["senior level"]),
        ("sort", "newest"),
        ("description_format", "pdf"),
    ],
)
def test_unknown_enum_values_are_rejected(field: str, value: Any) -> None:
    """The API answers an unknown enum value with zero results, not an error."""
    with pytest.raises(ValidationError):
        JoboSearchFilters(**{field: value})


def test_unknown_filter_names_are_rejected() -> None:
    with pytest.raises(ValidationError):
        JoboSearchFilters(is_remote=True)  # type: ignore[call-arg]


def test_salary_bounds_nest_under_salary_usd() -> None:
    assert search_body(min_salary_usd=120000) == {"salary_usd": {"min": 120000}}
    assert search_body(max_salary_usd=200000) == {"salary_usd": {"max": 200000}}
    assert search_body(min_salary_usd=1, max_salary_usd=2) == {
        "salary_usd": {"min": 1, "max": 2}
    }


def test_zero_salary_means_no_bound() -> None:
    assert search_body(min_salary_usd=0, max_salary_usd=0) == {}


def test_negative_salary_is_rejected() -> None:
    with pytest.raises(ValidationError):
        JoboSearchFilters(min_salary_usd=-1)


def test_timestamps_are_serialised_as_iso_8601() -> None:
    plus_two = timezone(timedelta(hours=2))
    body = search_body(
        posted_after="2026-11-01",
        posted_before="2026-11-20T12:30:00Z",
        discovered_after=datetime(2026, 11, 1, 8, 0, 0),
        discovered_before=datetime(2026, 11, 2, 8, 0, 0, tzinfo=plus_two),
    )

    assert body == {
        "posted_after": "2026-11-01T00:00:00Z",
        "posted_before": "2026-11-20T12:30:00Z",
        "discovered_after": "2026-11-01T08:00:00Z",
        "discovered_before": "2026-11-02T08:00:00+02:00",
    }


def test_server_defaults_are_not_sent() -> None:
    body = search_body(
        search_description=True, sort="relevance", description_format="markdown"
    )

    assert body == {}


def test_non_default_options_are_sent() -> None:
    body = search_body(
        search_description=False, sort="discovered_asc", description_format="html"
    )

    assert body == {
        "search_description": False,
        "sort": "discovered_asc",
        "description_format": "html",
    }


def test_paging_is_added_when_given() -> None:
    assert build_search_body(JoboSearchFilters(), page=2, page_size=10) == {
        "page": 2,
        "page_size": 10,
    }


def test_search_input_defaults_to_a_small_page() -> None:
    args = JoboSearchInput()

    assert args.page == 1
    assert args.page_size == 5


@pytest.mark.parametrize("page_size", [0, 101])
def test_search_input_rejects_page_sizes_the_api_would(page_size: int) -> None:
    with pytest.raises(ValidationError):
        JoboSearchInput(page_size=page_size)


def test_feed_body_keeps_the_feed_wire_shape() -> None:
    filters = JoboFeedFilters(
        locations=[{"country": "United States", "region": "Ohio"}],
        sources=["greenhouse", "lever"],
        exclude_sources=["lever"],
        work_models=["Remote"],
        exclude_employment_types=["internship"],
        companies={"include": ["fedex.com", "UPS"], "exclude": ["DHL"]},
        job_subcategories=["Nursing"],
        company_sizes={"include": ["201-500", "10001+"]},
        company_is_agency=False,
        updated_after="2026-11-01T00:00:00Z",
    )

    assert build_feed_body(filters, batch_size=500) == {
        "locations": [{"country": "United States", "region": "Ohio"}],
        "sources": ["greenhouse", "lever"],
        "work_models": ["remote"],
        "exclude_sources": ["lever"],
        "exclude_employment_types": ["internship"],
        "companies": {"include": ["fedex.com", "UPS"], "exclude": ["DHL"]},
        "job_subcategories": {"include": ["Nursing"]},
        "company_sizes": {"include": ["201-500", "10001+"]},
        "company_is_agency": False,
        "updated_after": "2026-11-01T00:00:00Z",
        "batch_size": 500,
    }


def test_empty_feed_filters_send_only_the_batch_size() -> None:
    assert build_feed_body(JoboFeedFilters(), batch_size=1000) == {"batch_size": 1000}
    assert build_feed_body(JoboFeedFilters()) == {}


def test_feed_default_description_format_is_not_sent() -> None:
    assert build_feed_body(JoboFeedFilters(description_format="markdown")) == {}
    assert build_feed_body(JoboFeedFilters(description_format="enhanced")) == {
        "description_format": "enhanced"
    }


@pytest.mark.parametrize(
    "filters",
    [
        {"work_model": ["remote"]},  # the search spelling, not the feed's
        {"exclude_skills": ["PHP"]},
        {"company_sizes": {"include": ["huge"]}},
        {"companies": {"includes": ["Acme"]}},
        {"locations": [{"town": "Columbus"}]},
    ],
)
def test_feed_filters_reject_unknown_keys_and_values(
    filters: dict[str, Any],
) -> None:
    """The feed ignores an unknown key without a warning, so it is caught here."""
    with pytest.raises(ValidationError):
        JoboFeedFilters.model_validate(filters)
