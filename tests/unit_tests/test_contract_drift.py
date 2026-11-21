"""Drift guard against the shared connector filter contract.

`Jobo.Connectors/contracts/filters.json` is generated from the connector core
and lists every filter the Jobo connectors support. These tests fail when the
contract gains a filter this package does not expose or does not send.

The contract lives one directory above this package in the Jobo monorepo. CI
mirrors this directory to a public repository on its own, and there the file
is absent, so the module is skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, get_args

import pytest

from langchain_jobo import JoboFeedFilters, JoboSearchInput, JoboSearchTool
from langchain_jobo.filters import (
    CONTRACT_KEY_TO_FIELD,
    CompanyRevenue,
    CompanySize,
    DescriptionFormat,
    EmploymentType,
    ExperienceLevel,
    SearchSort,
    WorkModel,
    build_search_body,
)

CONTRACT_PATH = Path(__file__).resolve().parents[3] / "contracts" / "filters.json"

pytestmark = pytest.mark.skipif(
    not CONTRACT_PATH.is_file(),
    reason="filters.json is only present inside the Jobo.Connectors monorepo",
)

#: A sample value per contract `kind`, valid for any filter of that kind.
SAMPLE_BY_KIND: dict[str, Any] = {
    "string": "sample",
    "number": 123,
    "boolean": False,
    "dateTime": "2026-11-01T00:00:00Z",
    "stringList": ["sample"],
}


def contract() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return loaded


def search_filters() -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = contract()["search_filters"]
    return filters


def field_for(key: str) -> str:
    return CONTRACT_KEY_TO_FIELD.get(key, key)


def sample_for(descriptor: dict[str, Any]) -> Any:
    kind = descriptor["kind"]
    options: list[str] = descriptor.get("options") or []
    if kind == "enum":
        # The last option is never the server default, which is not sent.
        return options[-1]
    if kind == "enumList":
        return [options[0]] if options else ["sample"]
    return SAMPLE_BY_KIND[kind]


def test_contract_was_found_and_has_filters() -> None:
    assert len(search_filters()) >= 30
    assert len(contract()["feed_filter_keys"]) >= 25


def test_tool_schema_has_every_search_filter_in_the_contract() -> None:
    schema_fields = set(JoboSearchInput.model_fields)
    tool_args = set(JoboSearchTool(api_key="jbe_test_drift").args)

    missing = sorted(
        descriptor["key"]
        for descriptor in search_filters()
        if field_for(descriptor["key"]) not in schema_fields
        or field_for(descriptor["key"]) not in tool_args
    )

    assert missing == [], (
        f"filters.json has search filters the jobo_job_search schema lacks: {missing}"
    )


def test_every_search_filter_in_the_contract_reaches_the_request_body() -> None:
    """A field that validates but is never sent would be a silent no-op."""
    unsent: list[str] = []
    for descriptor in search_filters():
        field = field_for(descriptor["key"])
        if field not in JoboSearchInput.model_fields:
            unsent.append(descriptor["key"])
            continue
        args = JoboSearchInput.model_validate({field: sample_for(descriptor)})
        body = build_search_body(args, query=args.query)
        if not body:
            unsent.append(descriptor["key"])

    assert unsent == []


def test_alias_map_only_names_real_contract_keys() -> None:
    keys = {descriptor["key"] for descriptor in search_filters()}

    assert set(CONTRACT_KEY_TO_FIELD) <= keys


@pytest.mark.parametrize(
    ("contract_enum", "literal"),
    [
        ("work_models", WorkModel),
        ("employment_types", EmploymentType),
        ("experience_levels", ExperienceLevel),
        ("search_sorts", SearchSort),
        ("description_formats", DescriptionFormat),
        ("company_sizes", CompanySize),
        ("company_revenues", CompanyRevenue),
    ],
)
def test_enum_values_match_the_contract(contract_enum: str, literal: Any) -> None:
    assert list(get_args(literal)) == contract()["enums"][contract_enum]


def test_search_filter_options_match_the_typed_fields() -> None:
    """Where a search filter is a closed enum here, its options match the contract."""
    typed = {
        "work_model": WorkModel,
        "employment_type": EmploymentType,
        "experience_level": ExperienceLevel,
        "sort": SearchSort,
        "description_format": DescriptionFormat,
    }
    by_key = {descriptor["key"]: descriptor for descriptor in search_filters()}

    for key, literal in typed.items():
        assert list(get_args(literal)) == by_key[key]["options"], key


def test_feed_filters_cover_every_feed_key_in_the_contract() -> None:
    missing = sorted(
        set(contract()["feed_filter_keys"]) - set(JoboFeedFilters.model_fields)
    )

    assert missing == [], f"filters.json has feed filters the loader lacks: {missing}"


def test_feed_filters_have_no_keys_the_contract_does_not_know() -> None:
    extra = sorted(
        set(JoboFeedFilters.model_fields) - set(contract()["feed_filter_keys"])
    )

    assert extra == []
