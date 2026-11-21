"""Rendering jobs into text and metadata."""

from __future__ import annotations

from typing import Any

import pytest

from langchain_jobo.formatting import (
    format_company_suggestions,
    format_compensation,
    format_filter_values,
    format_job_document,
    format_job_summary,
    format_location,
    format_location_suggestions,
    format_occupation,
    format_occupation_matches,
    format_search_results,
    job_metadata,
    job_to_document,
    response_warnings,
    warning_messages,
)
from tests.unit_tests.helpers import make_job, search_page


@pytest.mark.parametrize(
    ("compensation", "expected"),
    [
        (
            {"min": 80000, "max": 100000, "currency": "EUR", "period": "yearly"},
            "EUR 80,000-100,000 per year",
        ),
        (
            {"min": 25.5, "max": 30, "currency": "USD", "period": "hourly"},
            "USD 25.50-30 per hour",
        ),
        (
            {"min": 4000, "max": None, "currency": "GBP", "period": "monthly"},
            "GBP from 4,000 per month",
        ),
        (
            {"min": None, "max": 300, "currency": "USD", "period": "daily"},
            "USD up to 300 per day",
        ),
        (
            {"min": 500, "max": 500, "currency": "USD", "period": "per-diem"},
            "USD 500 per diem",
        ),
        (
            {"min": 900, "max": 1100, "currency": "USD", "period": "weekly"},
            "USD 900-1,100 per week",
        ),
        (
            {"min": 50000, "max": 60000, "currency": None, "period": None},
            "50,000-60,000",
        ),
        ({"min": None, "max": None, "currency": "USD", "period": "yearly"}, ""),
        (None, ""),
    ],
)
def test_format_compensation(compensation: Any, expected: str) -> None:
    assert format_compensation({"compensation": compensation}) == expected


def test_compensation_never_invents_a_currency_or_period() -> None:
    text = format_compensation({"compensation": {"min": 10, "max": 20}})

    assert text == "10-20"


@pytest.mark.parametrize(
    ("job", "expected"),
    [
        ({"locations": [{"location": "Berlin, Germany"}]}, "Berlin, Germany"),
        (
            {
                "locations": [
                    {"city": "Columbus", "region": "Ohio", "country": "United States"}
                ]
            },
            "Columbus, Ohio, United States",
        ),
        (
            {
                "locations": [
                    {"location": "Paris"},
                    {"location": "Lyon"},
                    {"location": "Paris"},
                ]
            },
            "Paris (+1 more)",
        ),
        ({"locations": [], "workplace_type": "Remote"}, "Remote"),
        ({"locations": [], "workplace_type": "On-site"}, "Not stated"),
        (
            {"locations": [{"location": None, "city": None, "country": None}]},
            "Not stated",
        ),
        ({}, "Not stated"),
    ],
)
def test_format_location(job: dict[str, Any], expected: str) -> None:
    assert format_location(job) == expected


def test_format_occupation() -> None:
    assert (
        format_occupation(make_job())
        == "Backend Developer (Technology > Software Engineering)"
    )
    assert format_occupation({"occupation": {"name": "Nurse"}}) == "Nurse"
    assert format_occupation({"occupation": None}) == ""


def test_job_summary_holds_everything_an_agent_needs_to_cite() -> None:
    text = format_job_summary(make_job(7), 2)

    assert text.splitlines() == [
        "2. Senior Backend Engineer 7 - Acme Corp",
        "   Location: Berlin, Germany | Work model: Hybrid | Employment: Full-time "
        "| Level: Senior",
        "   Salary: EUR 80,000-100,000 per year",
        "   Posted: 2026-11-16",
        "   URL: https://jobs.acme.example/7",
        "   ID: 00000000-0000-4000-8000-000000000007",
    ]


def test_job_summary_leaves_out_what_the_posting_does_not_state() -> None:
    job = make_job(
        compensation=None,
        date_posted=None,
        workplace_type=None,
        employment_type=None,
        experience_level=None,
    )

    text = format_job_summary(job)

    assert "Salary" not in text
    assert "Posted" not in text
    assert "Work model" not in text
    assert "None" not in text
    # The first-indexed date is labelled as such, not passed off as a post date.
    assert "First indexed: 2026-11-17" in text


def test_job_summary_falls_back_to_the_apply_url() -> None:
    job = make_job(listing_url="", apply_url="https://apply.example/1")

    assert "URL: https://apply.example/1" in format_job_summary(job)


def test_search_results_have_a_header_blocks_and_a_next_page_hint() -> None:
    text = format_search_results(search_page(2, total=1234))

    assert text.startswith("1,234 matching jobs; showing 2 (page 1 of 617).")
    assert "1. Senior Backend Engineer 0 - Acme Corp" in text
    assert "2. Senior Backend Engineer 1 - Acme Corp" in text
    assert text.endswith("More results are available: request page 2.")


def test_search_results_on_the_last_page_have_no_next_page_hint() -> None:
    text = format_search_results(search_page(1))

    assert text.startswith("1 matching job; showing 1 (page 1 of 1).")
    assert "More results" not in text


def test_search_results_say_when_the_page_size_was_capped() -> None:
    text = format_search_results(search_page(1), capped_page_size=10)

    assert "page_size was capped at 10" in text


def test_empty_search_results_explain_themselves() -> None:
    text = format_search_results(search_page(0))

    assert text.startswith("No jobs matched.")


def test_search_warnings_and_company_suggestions_are_surfaced() -> None:
    payload = search_page(
        0,
        warnings=[
            {
                "code": "companies_unmatched",
                "field": "companies.include",
                "message": "No company matched: Glenmark.",
            }
        ],
        filters={
            "companies": {
                "unmatched": ["Glenmark"],
                "suggestions": [
                    {
                        "query": "Glenmark",
                        "companies": [
                            {
                                "ids": ["id-1", "id-2"],
                                "name": "Glenmark Pharmaceuticals",
                                "domain": "glenmarkpharma.com",
                                "country": "IN",
                            }
                        ],
                    }
                ],
            }
        },
    )

    text = format_search_results(payload)

    assert "Note: No company matched: Glenmark." in text
    assert (
        'Closest companies to "Glenmark": Glenmark Pharmaceuticals '
        "(glenmarkpharma.com; ids: id-1, id-2)" in text
    )


def test_job_document_is_self_contained_prose() -> None:
    text = format_job_document(make_job(3))

    assert text.startswith("# Senior Backend Engineer 3\n\nCompany: Acme Corp\n")
    for expected in (
        "Location: Berlin, Germany",
        "Occupation: Backend Developer (Technology > Software Engineering)",
        "Work model: Hybrid",
        "Employment: Full-time",
        "Level: Senior",
        "Compensation: EUR 80,000-100,000 per year",
        "Posted: 2026-11-16",
        "Industry: Aerospace, Software Development",
        "## About Acme Corp\nAcme builds rockets for coyotes.",
        "## Summary\nBuild and run the public API.",
        "## Responsibilities\n- Design APIs\n- Run the on-call rota",
        "## Required qualifications\n- Python\n- BSc in Computer Science",
        "## Preferred qualifications\n- Kubernetes\n- AWS Solutions Architect",
        "## Benefits\n- 30 days of leave",
        "## Eligibility\n- Existing work authorisation required.",
        "## Description\nWe are hiring a **backend engineer** to build the API.",
        "Listing: https://jobs.acme.example/3",
        "Apply: https://jobs.acme.example/3/apply",
        "ATS: greenhouse",
        "Job ID: 00000000-0000-4000-8000-000000000003",
    ):
        assert expected in text
    assert "None" not in text
    assert "Closes:" not in text
    assert "All locations" not in text


def test_job_document_can_leave_the_description_out() -> None:
    text = format_job_document(make_job(), include_description=False)

    assert "## Description" not in text
    assert "## Responsibilities" in text


def test_job_document_of_a_sparse_job_has_no_empty_sections() -> None:
    sparse: dict[str, Any] = {
        "id": "abc",
        "title": "Driver",
        "company": {"name": "Haulage Ltd", "industries": [], "summary": None},
        "description": "",
        "summary": None,
        "listing_url": "https://example.test/driver",
        "apply_url": "https://example.test/driver",
        "locations": [],
        "compensation": None,
        "employment_type": None,
        "workplace_type": None,
        "experience_level": None,
        "source": "workday",
        "qualifications": {
            "must_have": {"education": [], "certifications": [], "skills": []},
            "preferred": {"education": [], "certifications": [], "skills": []},
        },
        "responsibilities": [],
        "benefits": [],
        "is_work_auth_required": None,
        "is_h1b_sponsor": None,
        "is_clearance_required": None,
    }

    assert format_job_document(sparse) == (
        "# Driver\n\n"
        "Company: Haulage Ltd\n"
        "Location: Not stated\n\n"
        "Listing: https://example.test/driver\n"
        "ATS: workday\n"
        "Job ID: abc"
    )


def test_job_document_lists_every_location_and_a_distinct_brand() -> None:
    job = make_job(
        locations=[{"location": "Paris, France"}, {"location": "Lyon, France"}],
        brand={"id": "b", "name": "Rocket Brand", "domain": None, "logo_url": None},
        valid_through="2026-12-31T00:00:00Z",
        is_h1b_sponsor=True,
        is_clearance_required=True,
    )

    text = format_job_document(job)

    assert "Brand: Rocket Brand" in text
    assert "Location: Paris, France (+1 more)" in text
    assert "## All locations\n- Paris, France\n- Lyon, France" in text
    assert "Closes: 2026-12-31" in text
    assert "- Employer has sponsored H-1B visas." in text
    assert "- Security clearance required." in text


def test_job_metadata_is_flat_and_complete() -> None:
    assert job_metadata(make_job(5)) == {
        "id": "00000000-0000-4000-8000-000000000005",
        "source": "https://jobs.acme.example/5",
        "url": "https://jobs.acme.example/5",
        "apply_url": "https://jobs.acme.example/5/apply",
        "title": "Senior Backend Engineer 5",
        "company": "Acme Corp",
        "company_id": "11111111-1111-4111-8111-111111111111",
        "location": "Berlin, Germany",
        "workplace_type": "Hybrid",
        "employment_type": "Full-time",
        "experience_level": "Senior",
        "salary": "EUR 80,000-100,000 per year",
        "salary_min": 80000.0,
        "salary_max": 100000.0,
        "salary_currency": "EUR",
        "salary_period": "yearly",
        "date_posted": "2026-11-16T00:00:00Z",
        "created_at": "2026-11-17T08:00:00Z",
        "updated_at": "2026-11-18T09:30:00Z",
        "ats": "greenhouse",
    }


def test_metadata_source_says_where_the_document_came_from() -> None:
    """LangChain convention: `source` locates the document. The ATS is `ats`."""
    listed = job_metadata(make_job(1))
    apply_only = job_metadata(make_job(1, listing_url=None))
    no_url = job_metadata(make_job(1, listing_url=None, apply_url=""))

    assert listed["source"] == listed["url"] == "https://jobs.acme.example/1"
    assert listed["ats"] == "greenhouse"
    assert apply_only["source"] == "https://jobs.acme.example/1/apply"
    assert "source" not in no_url
    assert "url" not in no_url
    assert no_url["ats"] == "greenhouse"


def test_job_metadata_omits_missing_values() -> None:
    metadata = job_metadata(
        make_job(compensation=None, date_posted=None, locations=[], company=None)
    )

    for absent in ("salary", "salary_min", "date_posted", "location", "company"):
        assert absent not in metadata
    assert all(value is not None for value in metadata.values())
    assert all(isinstance(value, (str, int, float)) for value in metadata.values())


def test_job_to_document_uses_the_job_id_as_document_id() -> None:
    document = job_to_document(make_job(9))

    assert document.id == "00000000-0000-4000-8000-000000000009"
    assert document.metadata["id"] == document.id
    assert document.page_content.startswith("# Senior Backend Engineer 9")


def test_company_suggestions_list_every_id() -> None:
    text = format_company_suggestions(
        "strip",
        [
            {
                "id": "id-1",
                "ids": ["id-1", "id-2"],
                "name": "Stripe",
                "domain": "stripe.com",
                "country": "US",
                "job_count": 1204,
            },
            {
                "id": "id-9",
                "ids": [],
                "name": "Stripes Ltd",
                "domain": None,
                "country": None,
                "job_count": None,
            },
        ],
    )

    assert text.startswith('2 companies matching "strip".')
    assert (
        "1. Stripe (stripe.com, US) - 1,204 live jobs\n   company ids: id-1, id-2"
        in text
    )
    assert "2. Stripes Ltd\n   company ids: id-9" in text


def test_no_company_suggestions_explains_the_prefix_match() -> None:
    text = format_company_suggestions("zzzz", [])

    assert text.startswith('No company matched "zzzz".')


def test_warnings_are_read_as_messages_with_close_company_matches() -> None:
    warning = {
        "code": "companies_unmatched",
        "field": "companies.include",
        "message": "No company matched: Glenmark.",
    }
    payload = search_page(
        0,
        warnings=[warning, "not a warning", {"code": "silent"}],
        filters={
            "companies": {
                "suggestions": [
                    {
                        "query": "Glenmark",
                        "companies": [{"name": "Glenmark Pharma", "ids": ["id-1"]}],
                    }
                ]
            }
        },
    )

    assert response_warnings(payload) == [warning, {"code": "silent"}]
    assert warning_messages(payload) == [
        "No company matched: Glenmark.",
        'Closest companies to "Glenmark": Glenmark Pharma (ids: id-1)',
    ]
    assert response_warnings(search_page(1)) == []
    assert warning_messages({"warnings": "broken"}) == []


def test_filter_values_list_each_value_with_its_job_count() -> None:
    text = format_filter_values(
        "industries",
        "soft",
        [
            {"key": "Software Development", "count": 120431},
            {"key": "Software, Data & AI", "count": 1},
        ],
    )

    assert text == (
        '2 `industries` values matching "soft", most jobs first. Pass the values '
        "exactly as written in the `industries` (or `exclude_industries`) filter "
        "of jobo_job_search.\n\n"
        "1. Software Development - 120,431 jobs\n"
        "2. Software, Data & AI - 1 job"
    )


def test_filter_values_of_sources_name_no_exclude_twin() -> None:
    text = format_filter_values("sources", "", [{"key": "greenhouse", "count": 9}])

    assert text.startswith("1 `sources` value, most jobs first.")
    assert "in the `sources` filter of jobo_job_search." in text
    assert "exclude_sources" not in text


def test_filter_values_of_named_facets_show_the_name_and_its_parent() -> None:
    categories = format_filter_values(
        "job_categories", "", [{"key": "healthcare", "name": "Healthcare", "count": 5}]
    )
    subcategories = format_filter_values(
        "job_subcategories",
        "nurs",
        [{"key": "nursing", "name": "Nursing", "parent": "healthcare", "count": 3}],
    )

    assert "1. Healthcare - 5 jobs" in categories
    assert "1. Nursing (category: healthcare) - 3 jobs" in subcategories
    assert "`job_subcategories` (or `exclude_job_subcategories`)" in subcategories


def test_filter_values_of_occupations_give_the_id_to_filter_on() -> None:
    text = format_filter_values(
        "occupations",
        "",
        [
            {
                "key": "esco:8d3e8aaa",
                "name": "Registered Nurse",
                "parent": "nursing",
                "count": 4210,
            },
            {"key": "jobo:unlabelled", "count": 2},
        ],
    )

    assert "Pass the ids in the `occupation_ids` (or `exclude_occupation_ids`)" in text
    assert (
        "1. Registered Nurse - id: esco:8d3e8aaa (subcategory: nursing) - 4,210 jobs"
        in text
    )
    assert "2. jobo:unlabelled - id: jobo:unlabelled - 2 jobs" in text


def test_no_filter_values_explains_the_prefix_match() -> None:
    assert format_filter_values("skills", "zzz", []).startswith(
        'No `skills` value matches "zzz". The lookup matches the start of a value'
    )
    assert format_filter_values("skills", "", []) == (
        "The API lists no `skills` values at the moment."
    )


def test_occupation_matches_show_the_id_and_where_it_sits() -> None:
    text = format_occupation_matches(
        "nurse",
        [
            {
                "id": "esco:8d3e8aaa",
                "name": "Registered Nurse",
                "category": "Healthcare",
                "subcategory": "Nursing",
            },
            {"id": "jobo:x", "name": "Nurse Aide", "category": "", "subcategory": ""},
        ],
    )

    assert text == (
        '2 occupations matching "nurse". Pass the ids in the `occupation_ids` (or '
        "`exclude_occupation_ids`) filter of jobo_job_search.\n\n"
        "1. Registered Nurse - id: esco:8d3e8aaa (Healthcare > Nursing)\n"
        "2. Nurse Aide - id: jobo:x"
    )
    assert format_occupation_matches("zzz", []).startswith(
        'No occupation matches "zzz".'
    )


def test_location_suggestions_are_written_as_the_location_filter_takes_them() -> None:
    text = format_location_suggestions(
        "spring",
        [
            {"display_name": "Springfield, Illinois, United States", "city": "x"},
            {"city": "Springfield", "region": "Missouri", "country": "United States"},
            {"city": "Berlin", "region": "Berlin", "country": "Germany"},
            {"display_name": "Springfield, Illinois, United States"},  # a repeat
            {"latitude": 1.0},  # nothing to show
        ],
    )

    assert text == (
        '3 places matching "spring". Pass one, exactly as written, as the '
        "`location` filter of jobo_job_search.\n\n"
        "1. Springfield, Illinois, United States\n"
        "2. Springfield, Missouri, United States\n"
        "3. Berlin, Germany"
    )


def test_no_location_suggestions_point_back_at_the_search() -> None:
    text = format_location_suggestions("zz", [])

    assert text.startswith('No place matched "zz".')
    assert "jobo_job_search also takes a free-form `location`" in text
