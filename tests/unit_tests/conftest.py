"""Shared fixtures. No test touches the network: sockets are disabled."""

from __future__ import annotations

import pytest

from tests.unit_tests.helpers import MockJobo, search_handler


@pytest.fixture(autouse=True)
def _no_api_key_in_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never read a real key from the developer's environment."""
    monkeypatch.delenv("JOBO_API_KEY", raising=False)


@pytest.fixture
def search_api() -> MockJobo:
    """A mock API whose search returns as many jobs as were asked for."""
    return MockJobo(search_handler)
