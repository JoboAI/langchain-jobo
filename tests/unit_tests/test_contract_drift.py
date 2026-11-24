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

from langchain_jobo import (
    JoboFeedFilters,
    JoboFilterValuesTool,
    JoboSearchFilters,
    JoboSearchInput,
    JoboSearchTool,
)
from langchain_jobo.filters import (
    CONTRACT_KEY_TO_FIELD,
    CONTRACT_KEYS_SET_ON_THE_TOOL,
    CompanyCategory,
    CompanyRevenue,
    CompanySize,
    DescriptionFormat,
    EmploymentType,
    ExperienceLevel,
    FilterFacet,
    SearchSort,
    WorkModel,
    build_search_body,
)
from langchain_jobo.formatting import FACET_FILTERS

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
    """Every contract filter is a tool argument, or an option set on the tool."""
    schema_fields = set(JoboSearchInput.model_fields)
    tool_args = set(JoboSearchTool(api_key="jbe_test_drift").args)

    missing = sorted(
        descriptor["key"]
        for descriptor in search_filters()
        if descriptor["key"] not in CONTRACT_KEYS_SET_ON_THE_TOOL
        and (
            field_for(descriptor["key"]) not in schema_fields
            or field_for(descriptor["key"]) not in tool_args
        )
    )

    assert missing == [], (
        f"filters.json has search filters the jobo_job_search schema lacks: {missing}"
    )


def test_filters_kept_off_the_tool_schema_are_options_of_the_tool() -> None:
    """A filter the model is not shown must still be settable, and sent."""
    keys = {descriptor["key"] for descriptor in search_filters()}
    by_key = {descriptor["key"]: descriptor for descriptor in search_filters()}

    assert keys >= CONTRACT_KEYS_SET_ON_THE_TOOL
    for key in CONTRACT_KEYS_SET_ON_THE_TOOL:
        assert key in JoboSearchTool.model_fields, key
        assert key not in JoboSearchInput.model_fields, key
        body = build_search_body(JoboSearchInput(), **{key: sample_for(by_key[key])})
        assert body, key


def test_search_filters_model_has_every_search_filter_in_the_contract() -> None:
    """`JoboSearchFilters` is what the retriever takes: it carries them all."""
    missing = sorted(
        descriptor["key"]
        for descriptor in search_filters()
        if descriptor["key"] not in CONTRACT_KEY_TO_FIELD
        and descriptor["key"] not in JoboSearchFilters.model_fields
    )

    assert missing == []


def test_every_search_filter_in_the_contract_reaches_the_request_body() -> None:
    """A field that validates but is never sent would be a silent no-op."""
    unsent: list[str] = []
    for descriptor in search_filters():
        key = descriptor["key"]
        sample = sample_for(descriptor)
        if key in CONTRACT_KEY_TO_FIELD:
            args = JoboSearchInput.model_validate({field_for(key): sample})
            body = build_search_body(args, query=args.query)
        elif key in JoboSearchFilters.model_fields:
            body = build_search_body(JoboSearchFilters.model_validate({key: sample}))
        else:
            body = {}
        if not body:
            unsent.append(key)

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
        ("company_categories", CompanyCategory),
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
        "company_categories": CompanyCategory,
        "exclude_company_categories": CompanyCategory,
        "sort": SearchSort,
        "description_format": DescriptionFormat,
    }
    by_key = {descriptor["key"]: descriptor for descriptor in search_filters()}

    for key, literal in typed.items():
        assert list(get_args(literal)) == by_key[key]["options"], key


def test_language_accepts_every_option_of_the_contract() -> None:
    """`language` stays open (the API takes names such as "English" too).

    It is not a closed enum here, so check that every code the contract lists
    is accepted and sent as it is.
    """
    options: list[str] = next(
        descriptor["options"]
        for descriptor in search_filters()
        if descriptor["key"] == "language"
    )

    assert options == contract()["enums"]["languages"]
    filters = JoboSearchFilters.model_validate({"language": options})
    assert filters.language == options
    assert build_search_body(filters)["languages"] == options
    args = JoboSearchInput.model_validate({"language": options})
    assert args.language == options


def test_company_category_fields_reject_values_outside_the_contract() -> None:
    values = contract()["enums"]["company_categories"]

    for field in ("company_categories", "exclude_company_categories"):
        assert getattr(JoboSearchFilters.model_validate({field: values}), field) == (
            values
        )
        schema = JoboSearchInput.model_json_schema()["properties"][field]
        assert schema["anyOf"][0]["items"]["enum"] == values, field


def test_lookup_facets_are_the_dynamic_option_sources_of_the_contract() -> None:
    """`jobo_filter_values` lists every value set the contract loads at run time.

    `companies` is the one source it does not list: `jobo_company_search` does.
    """
    sources = {
        descriptor["dynamicOptions"]
        for descriptor in search_filters()
        if "dynamicOptions" in descriptor
    }
    schema = JoboFilterValuesTool(api_key="jbe_test_drift").args["facet"]

    assert set(get_args(FilterFacet)) == sources - {"companies"}
    assert set(schema["enum"]) == sources - {"companies"}
    assert set(FACET_FILTERS) == set(get_args(FilterFacet))


def test_each_lookup_facet_names_a_real_search_filter() -> None:
    by_key = {descriptor["key"]: descriptor for descriptor in search_filters()}

    for facet, filter_name in FACET_FILTERS.items():
        assert by_key[filter_name]["dynamicOptions"] == facet


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
