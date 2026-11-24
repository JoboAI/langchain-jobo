"""Drift guard against the shared hiring signals contract.

`Jobo.Connectors/contracts/signals.json` is generated from the connector core
and describes `GET /api/signals`: its filters, which of them the per-company
endpoint takes, the signal types, the seniority levels, the labels and the
page limits. These tests fail when this package no longer matches it, and
when the contract gains a filter the hiring signals tool neither takes nor
names below as left out on purpose.

The contract lives one directory above this package in the Jobo monorepo. The
public mirror of this directory has no contracts folder, so there the module
is skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from langchain_jobo import (
    JoboHiringSignalsInput,
    JoboHiringSignalsTool,
    SignalSeniority,
    SignalType,
)
from langchain_jobo.filters import CompanySize
from langchain_jobo.signals import (
    COMPANY_SIGNAL_FILTER_KEYS,
    SIGNAL_DEFAULT_LIMIT,
    SIGNAL_FILTER_KEYS,
    SIGNAL_LIST_FILTER_KEYS,
    SIGNAL_MAX_LIMIT,
    SIGNAL_MAX_LIST_VALUES,
    SIGNAL_TYPE_LABELS,
    build_signals_request,
)

CONTRACT_PATH = Path(__file__).resolve().parents[3] / "contracts" / "signals.json"

pytestmark = pytest.mark.skipif(
    not CONTRACT_PATH.is_file(),
    reason="signals.json is only present inside the Jobo.Connectors monorepo",
)

#: Contract filters the `jobo_hiring_signals` tool leaves out on purpose. They
#: are exclusions, thresholds that narrow one or two types, and the less used
#: company and location filters: a developer sets them better than a model.
#: `build_signals_request` takes every one of them.
LEFT_OFF_THE_TOOL = frozenset(
    {
        "exclude_domains",
        "categories",
        "founded_after",
        "technology_categories",
        "job_regions",
        "job_cities",
        "min_new_jobs",
        "min_days_open",
        "reposted_only",
    }
)
#: Contract filters the tool takes in another form: `company_id` reads one
#: company from `GET /api/companies/{id}/signals` instead of a list of ids.
ARGUMENT_FOR = {"company_ids": "company_id"}
#: Tool arguments that are not filters: the company in the path, and paging.
NOT_FILTERS = frozenset({"company_id", "limit", "cursor"})

COMPANY_ID = "6c1f0a9e-3b7d-4c55-8e21-5a4d9f0b7c33"

#: A sample value per contract `kind`, valid for any filter of that kind.
SAMPLE_BY_KIND: dict[str, Any] = {
    "stringList": ["sample"],
    "boolean": True,
    "date": "2026-11-01",
    "dateTime": "2026-11-01T00:00:00Z",
}


def contract() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return loaded


def contract_filters() -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = contract()["filters"]
    return filters


def by_key() -> dict[str, dict[str, Any]]:
    return {descriptor["key"]: descriptor for descriptor in contract_filters()}


def tool_filters() -> set[str]:
    """The tool's arguments that are API filters, under the API's names."""
    return set(JoboHiringSignalsInput.model_fields) - NOT_FILTERS


def sample_for(descriptor: dict[str, Any]) -> Any:
    kind = descriptor["kind"]
    if kind == "enumList":
        return [descriptor["options"][0]]
    if kind == "number":
        return descriptor["min"]
    return SAMPLE_BY_KIND[kind]


def test_contract_was_found_and_has_filters() -> None:
    assert len(contract_filters()) >= 20


@pytest.mark.parametrize(
    ("contract_enum", "literal"),
    [
        ("signal_types", SignalType),
        ("seniority", SignalSeniority),
        ("company_sizes", CompanySize),
    ],
)
def test_enum_values_match_the_contract(contract_enum: str, literal: Any) -> None:
    assert list(get_args(literal)) == contract()["enums"][contract_enum]


def test_every_type_has_the_dashboard_label() -> None:
    labels = contract()["type_labels"]

    assert labels == SIGNAL_TYPE_LABELS
    assert list(SIGNAL_TYPE_LABELS) == contract()["enums"]["signal_types"]


def test_page_limits_match_the_contract() -> None:
    limits = contract()["limits"]

    assert limits == {
        "default_limit": SIGNAL_DEFAULT_LIMIT,
        "max_limit": SIGNAL_MAX_LIMIT,
        "max_list_values": SIGNAL_MAX_LIST_VALUES,
    }


def test_the_request_builder_knows_every_filter_and_its_kind() -> None:
    lists = {
        descriptor["key"]
        for descriptor in contract_filters()
        if descriptor["kind"] in ("stringList", "enumList")
    }

    assert list(SIGNAL_FILTER_KEYS) == [d["key"] for d in contract_filters()]
    assert lists == SIGNAL_LIST_FILTER_KEYS


def test_the_per_company_filters_match_the_contract() -> None:
    marked = [d["key"] for d in contract_filters() if d.get("company")]

    assert list(COMPANY_SIGNAL_FILTER_KEYS) == contract()["company_filter_keys"]
    assert list(COMPANY_SIGNAL_FILTER_KEYS) == marked


def test_the_tool_takes_every_filter_or_it_is_left_out_on_purpose() -> None:
    arguments = set(JoboHiringSignalsInput.model_fields)
    missing = sorted(
        key
        for key in by_key()
        if key not in tool_filters()
        and key not in LEFT_OFF_THE_TOOL
        and ARGUMENT_FOR.get(key) not in arguments
    )

    assert missing == [], (
        f"signals.json has filters jobo_hiring_signals lacks: {missing}"
    )


def test_the_tool_has_no_filter_the_contract_does_not_know() -> None:
    assert sorted(tool_filters() - set(by_key())) == []


def test_the_keys_named_here_are_real_contract_keys_and_off_the_tool() -> None:
    keys = set(by_key())

    assert keys >= LEFT_OFF_THE_TOOL
    assert keys >= set(ARGUMENT_FOR)
    assert LEFT_OFF_THE_TOOL.isdisjoint(tool_filters())
    assert set(ARGUMENT_FOR.values()) <= set(JoboHiringSignalsInput.model_fields)


@pytest.mark.parametrize("key", ["types", "seniority", "company_sizes"])
def test_the_tool_offers_exactly_the_options_of_the_contract(key: str) -> None:
    schema = JoboHiringSignalsInput.model_json_schema()["properties"][key]

    assert schema["anyOf"][0]["items"]["enum"] == by_key()[key]["options"]


def test_every_filter_of_the_tool_reaches_the_query() -> None:
    """An argument that validates but is never sent would be a silent no-op."""
    unsent: list[str] = []
    for descriptor in contract_filters():
        key = descriptor["key"]
        if key not in tool_filters():
            continue
        args = JoboHiringSignalsInput.model_validate({key: sample_for(descriptor)})
        _, params = build_signals_request(args.filters())
        if key not in params:
            unsent.append(key)

    assert unsent == []


def test_every_left_out_filter_still_reaches_the_query() -> None:
    """A developer can send what the tool leaves out."""
    descriptors = by_key()

    for key in sorted(LEFT_OFF_THE_TOOL):
        _, params = build_signals_request({key: sample_for(descriptors[key])})
        assert key in params, key


def test_company_id_refuses_exactly_what_its_endpoint_does_not_take() -> None:
    company_keys = set(contract()["company_filter_keys"])

    for descriptor in contract_filters():
        key = descriptor["key"]
        if key not in tool_filters():
            continue
        args = {key: sample_for(descriptor), "company_id": COMPANY_ID}
        if key in company_keys:
            JoboHiringSignalsInput.model_validate(args)
            continue
        with pytest.raises(ValidationError, match="company_id reads one company"):
            JoboHiringSignalsInput.model_validate(args)


def test_the_tool_cap_stays_within_the_api_limit() -> None:
    maximum = contract()["limits"]["max_limit"]

    tool = JoboHiringSignalsTool(api_key="jbe_test_drift", max_signals=maximum)

    assert tool.max_signals == maximum
    with pytest.raises(ValidationError):
        JoboHiringSignalsTool(api_key="jbe_test_drift", max_signals=maximum + 1)
