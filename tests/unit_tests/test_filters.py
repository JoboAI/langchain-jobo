"""Filter validation and the mapping onto search and feed request bodies."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from langchain_jobo import (
    CompanyCategoryFilter,
    FeedLocation,
    IncludeExclude,
    JoboFeedFilters,
    JoboFeedFiltersDict,
    JoboSearchFilters,
    JoboSearchFiltersDict,
    JoboSearchInput,
)
from langchain_jobo.filters import (
    CompanyCategoryFilterDict,
    CompanyRevenueFilter,
    CompanyRevenueFilterDict,
    CompanySizeFilter,
    CompanySizeFilterDict,
    FeedLocationDict,
    IncludeExcludeDict,
    build_feed_body,
    build_search_body,
    merge_search_filters,
    search_input_schema,
)


def search_body(**fields: Any) -> dict[str, Any]:
    return build_search_body(JoboSearchFilters(**fields))


def test_no_filters_builds_an_empty_body() -> None:
    assert search_body() == {}


def test_query_and_location_become_single_entry_lists() -> None:
    body = build_search_body(
        JoboSearchFilters(location="  Berlin, Germany "), query="  data engineer "
    )

    assert body == {"queries": ["data engineer"], "locations": ["Berlin, Germany"]}


def test_location_takes_a_list_as_the_api_does() -> None:
    filters = JoboSearchFilters(
        location=[" Berlin, Germany ", "", "Paris", "Berlin, Germany"]
    )

    assert filters.location == ["Berlin, Germany", "Paris"]
    assert build_search_body(filters) == {"locations": ["Berlin, Germany", "Paris"]}
    assert build_search_body(JoboSearchFilters(location=[" ", ""])) == {}


def test_blank_query_and_location_are_left_out() -> None:
    assert build_search_body(JoboSearchFilters(location="  "), query=" ") == {}


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("skills", ("a", "b", "c")),
        ("companies", ("a", "b", "c")),
        ("industries", ("a", "b", "c")),
        ("company_categories", ("b2b", "saas", "agency")),
        ("job_categories", ("a", "b", "c")),
        ("job_subcategories", ("a", "b", "c")),
        ("occupation_ids", ("a", "b", "c")),
    ],
)
def test_include_exclude_pairs_nest_under_one_key(
    field: str, values: tuple[str, str, str]
) -> None:
    a, b, c = values
    both = search_body(**{field: [a, b], f"exclude_{field}": [c]})
    include_only = search_body(**{field: [a]})
    exclude_only = search_body(**{f"exclude_{field}": [c]})

    assert both == {field: {"include": [a, b], "exclude": [c]}}
    assert include_only == {field: {"include": [a]}}
    assert exclude_only == {field: {"exclude": [c]}}
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


def test_a_single_string_of_ids_keys_or_skills_is_split_on_commas() -> None:
    """A model often hands a list field one comma-separated string."""
    filters = JoboSearchFilters.model_validate(
        {
            "skills": "Python, Kubernetes\nGo",
            "work_model": "Remote, HYBRID",
            "sources": "greenhouse,lever",
            "language": "en, de",
            "occupation_ids": "esco:a, jobo:b",
            "company_categories": "B2B, saas",
        }
    )

    assert filters.skills == ["Python", "Kubernetes", "Go"]
    assert filters.work_model == ["remote", "hybrid"]
    assert filters.sources == ["greenhouse", "lever"]
    assert filters.language == ["en", "de"]
    assert filters.occupation_ids == ["esco:a", "jobo:b"]
    assert filters.company_categories == ["b2b", "saas"]


@pytest.mark.parametrize(
    "field",
    [
        "companies",
        "exclude_companies",
        "industries",
        "exclude_industries",
        "job_categories",
        "exclude_job_categories",
        "job_subcategories",
        "exclude_job_subcategories",
    ],
)
def test_a_single_string_naming_something_is_never_split_on_commas(
    field: str,
) -> None:
    """`companies="FedEx, Inc."` is one company, not two."""
    one = JoboSearchFilters.model_validate({field: " FedEx, Inc. "})
    lines = JoboSearchFilters.model_validate({field: "FedEx, Inc.\nUPS\n"})

    assert getattr(one, field) == ["FedEx, Inc."]
    assert getattr(lines, field) == ["FedEx, Inc.", "UPS"]


def test_a_company_name_with_a_comma_reaches_the_api_whole() -> None:
    body = build_search_body(JoboSearchFilters.model_validate({"companies": "A, B"}))

    assert body == {"companies": {"include": ["A, B"]}}


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
        ("company_categories", ["startup"]),
        ("exclude_company_categories", ["b2b", "recruiter"]),
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


def test_search_input_leads_with_the_query_and_ends_with_the_page() -> None:
    fields = list(JoboSearchInput.model_fields)

    assert fields[0] == "query"
    assert fields[1] == "location"
    assert fields[-2:] == ["page", "page_size"]


def test_search_input_has_every_filter_but_the_description_format() -> None:
    """It only changes the artifact, so the model is not shown it."""
    assert set(JoboSearchFilters.model_fields) - set(JoboSearchInput.model_fields) == {
        "description_format"
    }
    with pytest.raises(ValidationError):
        JoboSearchInput.model_validate({"description_format": "html"})


def test_search_input_states_a_maximum_it_does_not_enforce() -> None:
    """The tool reduces a larger page to its cap instead of failing the call."""
    schema = JoboSearchInput.model_json_schema()["properties"]["page_size"]

    assert schema["maximum"] == 100
    assert schema["minimum"] == 1
    assert JoboSearchInput(page_size=500).page_size == 500
    with pytest.raises(ValidationError):
        JoboSearchInput(page_size=0)


def test_search_input_schema_for_a_cap_states_that_cap() -> None:
    capped = search_input_schema(3)
    schema = capped.model_json_schema()["properties"]["page_size"]

    assert issubclass(capped, JoboSearchInput)
    assert capped is search_input_schema(3)
    assert schema["maximum"] == 3
    assert schema["default"] == 3  # the default page never exceeds the cap
    assert "at most 3" in schema["description"]
    assert capped().page_size == 3
    assert capped(page_size=40).page_size == 40
    assert search_input_schema(25)().page_size == 5
    assert list(capped.model_fields) == list(JoboSearchInput.model_fields)


def test_description_format_can_be_given_beside_the_filters() -> None:
    args = JoboSearchInput(query="nurse")

    assert build_search_body(args, query=args.query, description_format="html") == {
        "queries": ["nurse"],
        "description_format": "html",
    }
    assert build_search_body(args, description_format="markdown") == {}
    # Given explicitly, it wins over the one on the filters.
    filters = JoboSearchFilters(description_format="enhanced")
    assert build_search_body(filters, description_format="html") == {
        "description_format": "html"
    }


def test_filters_can_be_laid_over_other_filters() -> None:
    base = JoboSearchFilters(work_model=["remote"], location="London", skills=["Go"])

    merged = merge_search_filters(base, {"location": "Berlin", "skills": None})
    from_model = merge_search_filters(base, JoboSearchFilters(sources=["lever"]))

    assert merged.model_dump(exclude_none=True) == {
        "work_model": ["remote"],
        "location": "Berlin",
    }
    assert from_model.model_dump(exclude_none=True) == {
        "work_model": ["remote"],
        "location": "London",
        "skills": ["Go"],
        "sources": ["lever"],
    }
    assert merge_search_filters(base, None) is base
    assert merge_search_filters(base, {}) is base
    with pytest.raises(ValidationError):
        merge_search_filters(base, {"is_remote": True})  # type: ignore[arg-type]


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
        {"company_categories": {"include": ["startup"]}},
        {"company_categories": ["startup"]},
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


@pytest.mark.parametrize(
    "location",
    [
        {"city": "Columbus"},
        {"region": "Ohio"},
        {"city": "Columbus", "region": "Ohio"},
        {"country": "  ", "city": "Columbus"},
    ],
)
def test_feed_location_needs_a_country_beside_a_city_or_region(
    location: dict[str, str],
) -> None:
    """The API answers this with a 400, after the request is made."""
    with pytest.raises(ValidationError, match="country is required when city"):
        FeedLocation.model_validate(location)
    with pytest.raises(ValidationError, match="country is required when city"):
        JoboFeedFilters.model_validate({"locations": [location]})


@pytest.mark.parametrize("location", [{}, {"country": ""}, {"country": None}])
def test_feed_location_cannot_be_empty(location: dict[str, Any]) -> None:
    """An empty location would be dropped, and the feed would come unfiltered."""
    with pytest.raises(ValidationError, match="a location needs a country"):
        JoboFeedFilters.model_validate({"locations": [location]})


def test_feed_location_is_trimmed_and_sent_without_blanks() -> None:
    filters = JoboFeedFilters(
        locations=[
            FeedLocation(country=" United States ", region="Ohio", city=" "),
            FeedLocation(country="Remote"),
        ]
    )

    assert build_feed_body(filters) == {
        "locations": [
            {"country": "United States", "region": "Ohio"},
            {"country": "Remote"},
        ]
    }


def test_feed_names_are_never_split_on_commas() -> None:
    filters = JoboFeedFilters.model_validate(
        {
            "companies": "FedEx, Inc.",
            "industries": {"include": "Technology, Information and Internet"},
            "job_subcategories": ["Film, TV & Stage Production"],
        }
    )

    assert build_feed_body(filters) == {
        "companies": {"include": ["FedEx, Inc."]},
        "job_subcategories": {"include": ["Film, TV & Stage Production"]},
        "industries": {"include": ["Technology, Information and Internet"]},
    }


def test_feed_ids_and_keys_given_as_one_string_are_split() -> None:
    filters = JoboFeedFilters.model_validate(
        {
            "occupation_ids": "esco:a, jobo:b",
            "sources": "greenhouse, lever",
            "company_categories": {"exclude": "Agency, B2G"},
        }
    )

    assert build_feed_body(filters) == {
        "sources": ["greenhouse", "lever"],
        "occupation_ids": {"include": ["esco:a", "jobo:b"]},
        "company_categories": {"exclude": ["agency", "b2g"]},
    }


def test_feed_filters_take_the_helper_models() -> None:
    filters = JoboFeedFilters(
        companies=IncludeExclude(include=["fedex.com"], exclude=["DHL"]),
        company_categories=CompanyCategoryFilter(include=["saas"]),
        company_sizes=CompanySizeFilter(include=["11-50"]),
        company_revenues=CompanyRevenueFilter(exclude=["over-1b"]),
    )

    assert build_feed_body(filters) == {
        "companies": {"include": ["fedex.com"], "exclude": ["DHL"]},
        "company_categories": {"include": ["saas"]},
        "company_sizes": {"include": ["11-50"]},
        "company_revenues": {"exclude": ["over-1b"]},
    }


# -------------------------------------------------------------- dict spellings


@pytest.mark.parametrize(
    ("typed_dict", "model"),
    [
        (JoboSearchFiltersDict, JoboSearchFilters),
        (JoboFeedFiltersDict, JoboFeedFilters),
        (IncludeExcludeDict, IncludeExclude),
        (FeedLocationDict, FeedLocation),
        (CompanyCategoryFilterDict, CompanyCategoryFilter),
        (CompanySizeFilterDict, CompanySizeFilter),
        (CompanyRevenueFilterDict, CompanyRevenueFilter),
    ],
)
def test_typed_dicts_have_exactly_the_keys_of_their_models(
    typed_dict: Any, model: Any
) -> None:
    """A filter added to a model must be added to its dict spelling too."""
    assert list(typed_dict.__annotations__) == list(model.model_fields)
    assert typed_dict.__required_keys__ == frozenset()


def test_typed_dicts_validate_as_their_models() -> None:
    search: JoboSearchFiltersDict = {
        "location": ["Berlin, Germany"],
        "work_model": ["remote"],
        "company_categories": ["saas"],
        "posted_after": "2026-11-01T00:00:00Z",
        "sort": "discovered_desc",
    }
    feed: JoboFeedFiltersDict = {
        "locations": [{"country": "US", "region": "Ohio"}],
        "work_models": ["remote"],
        "companies": {"include": ["fedex.com"]},
        "job_categories": ["Healthcare"],
        "company_sizes": {"include": ["11-50"]},
        "updated_after": "2026-11-01T00:00:00Z",
    }

    assert build_search_body(JoboSearchFilters.model_validate(search)) == {
        "locations": ["Berlin, Germany"],
        "work_models": ["remote"],
        "company_categories": {"include": ["saas"]},
        "posted_after": "2026-11-01T00:00:00Z",
        "sort": "discovered_desc",
    }
    assert build_feed_body(JoboFeedFilters.model_validate(feed)) == {
        "locations": [{"country": "US", "region": "Ohio"}],
        "work_models": ["remote"],
        "companies": {"include": ["fedex.com"]},
        "job_categories": {"include": ["Healthcare"]},
        "company_sizes": {"include": ["11-50"]},
        "updated_after": "2026-11-01T00:00:00Z",
    }
