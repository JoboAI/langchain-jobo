"""The HTTP layer: headers, retry policy and error mapping."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from langchain_jobo import (
    JoboAPIError,
    JoboAuthenticationError,
    JoboConflictError,
    JoboConnectionError,
    JoboError,
    JoboFeedCursorRestartRequiredError,
    JoboInsufficientCreditsError,
    JoboNotFoundError,
    JoboPermissionError,
    JoboRateLimitError,
    JoboServerError,
    JoboValidationError,
    __version__,
)
from langchain_jobo._client import JoboClient
from tests.unit_tests.helpers import (
    CURSOR_RESTART,
    INSUFFICIENT_CREDITS,
    RATE_LIMITED,
    TEST_KEY,
    Handler,
    MockJobo,
    reply,
    sequence,
)

ok = reply(200, {"ok": True})


def build(handler: Handler, **kwargs: Any) -> tuple[JoboClient, MockJobo, list[float]]:
    """A client wired to a mock API, with sleeping replaced by a recorder."""
    api = MockJobo(handler)
    client = JoboClient(
        TEST_KEY,
        http_client=api.client,
        http_async_client=api.async_client,
        **kwargs,
    )
    sleeps: list[float] = []

    async def asleep(seconds: float) -> None:
        sleeps.append(seconds)

    client._sleep = sleeps.append
    client._asleep = asleep
    client._random = lambda _low, _high: 1.0
    return client, api, sleeps


def test_sends_identifying_headers_and_json_body() -> None:
    client, api, _ = build(ok)

    response = client.request("POST", "/api/jobs/search", json={"page_size": 1})

    request = api.last
    assert str(request.url) == "https://connect.jobo.world/api/jobs/search"
    assert request.headers["X-Api-Key"] == TEST_KEY
    assert request.headers["X-Jobo-Client"] == "langchain"
    assert request.headers["User-Agent"] == f"langchain-jobo/{__version__}"
    assert request.headers["Accept"] == "application/json"
    assert request.headers["Content-Type"] == "application/json"
    assert api.bodies == [{"page_size": 1}]
    assert response.data == {"ok": True}


def test_base_url_is_configurable_and_trailing_slash_is_dropped() -> None:
    client, api, _ = build(ok, base_url="https://example.test/")

    client.request("GET", "/api/jobs/abc", params={"description_format": "html"})

    assert (
        str(api.last.url) == "https://example.test/api/jobs/abc?description_format=html"
    )


def test_timeout_is_passed_per_request() -> None:
    client, api, _ = build(ok, timeout=12.0)

    client.request("GET", "/x")
    client.request("GET", "/x", timeout=120.0)

    assert api.requests[0].extensions["timeout"]["read"] == 12.0
    assert api.requests[1].extensions["timeout"]["read"] == 120.0


@pytest.mark.parametrize("key", ["", "   "])
def test_blank_api_key_is_rejected(key: str) -> None:
    with pytest.raises(ValueError, match="JOBO_API_KEY"):
        JoboClient(key)


def test_api_key_is_never_exposed() -> None:
    secret = "jbe_live_super-secret-value"  # noqa: S105
    api = MockJobo(reply(401, {"error": "Invalid or expired API key"}))
    client = JoboClient(SecretStr(secret), http_client=api.client)

    assert secret not in repr(client)
    assert secret not in str(vars(client))
    with pytest.raises(JoboAuthenticationError) as caught:
        client.request("GET", "/api/jobs/x")
    assert secret not in str(caught.value)
    assert secret not in repr(caught.value.body)


def test_usage_headers_are_parsed() -> None:
    headers = {
        "X-Credits-Deducted": "75",
        "X-Credits-Balance": "4925",
        "X-Quota-Limit": "100000",
        "X-Quota-Remaining": "99975",
        "X-Jobs-Billed": "25",
        "X-Billing-Status": "Unconfirmed",
    }
    client, _, _ = build(reply(200, {}, headers=headers))

    usage = client.request("POST", "/api/jobs/feed", json={}).usage

    assert usage.as_dict() == {
        "credits_deducted": 75,
        "credits_balance": 4925,
        "quota_limit": 100000,
        "quota_remaining": 99975,
        "jobs_billed": 25,
        "billing_status": "Unconfirmed",
    }


def test_absent_usage_headers_are_left_out() -> None:
    client, _, _ = build(ok)

    assert client.request("GET", "/x").usage.as_dict() == {}


@pytest.mark.parametrize(
    ("status", "body", "expected", "code"),
    [
        (
            400,
            {"title": "Invalid cursor", "code": "invalid_cursor"},
            JoboValidationError,
            "invalid_cursor",
        ),
        (401, {"error": "Invalid or expired API key"}, JoboAuthenticationError, None),
        (402, INSUFFICIENT_CREDITS, JoboInsufficientCreditsError, None),
        (
            403,
            {"title": "Forbidden", "code": "company_enrichment_not_enabled"},
            JoboPermissionError,
            "company_enrichment_not_enabled",
        ),
        (
            404,
            {"title": "Not found", "code": "not_available_in_sandbox"},
            JoboNotFoundError,
            "not_available_in_sandbox",
        ),
        (
            409,
            {"title": "Conflict", "code": "request_conflict"},
            JoboConflictError,
            "request_conflict",
        ),
        (
            409,
            CURSOR_RESTART,
            JoboFeedCursorRestartRequiredError,
            "feed_cursor_restart_required",
        ),
        (
            422,
            {"title": "Unprocessable", "code": "unprocessable_request"},
            JoboAPIError,
            "unprocessable_request",
        ),
    ],
)
def test_error_statuses_map_to_typed_errors_and_are_not_retried(
    status: int, body: dict[str, Any], expected: type[JoboAPIError], code: str | None
) -> None:
    client, api, sleeps = build(lambda _: httpx.Response(status, json=body))

    with pytest.raises(expected) as caught:
        client.request("POST", "/api/jobs/search", json={})

    assert type(caught.value) is expected
    assert caught.value.status_code == status
    assert caught.value.code == code
    assert caught.value.body == body
    assert len(api.requests) == 1
    assert sleeps == []


def test_402_carries_the_wallet_figures_and_top_up_link() -> None:
    client, api, sleeps = build(reply(402, INSUFFICIENT_CREDITS))

    with pytest.raises(JoboInsufficientCreditsError) as caught:
        client.request("POST", "/api/jobs/search", json={"page_size": 25})

    error = caught.value
    assert error.credits_required == 75
    assert error.credits_balance == 12
    assert error.top_up_url == "https://enterprise.jobo.world/credits?topup=10"
    assert error.upgrade_url == INSUFFICIENT_CREDITS["upgrade_url"]
    assert "This request needs 75 credits" in str(error)
    assert "https://enterprise.jobo.world/credits?topup=10" in str(error)
    assert len(api.requests) == 1
    assert sleeps == []


def test_409_cursor_restart_exposes_the_replacement_request() -> None:
    client, api, sleeps = build(reply(409, CURSOR_RESTART))

    with pytest.raises(JoboFeedCursorRestartRequiredError) as caught:
        client.request("POST", "/api/jobs/feed", json={"cursor": "stale"})

    assert caught.value.replacement_request == {"stable_scan": True, "batch_size": 1000}
    assert isinstance(caught.value, JoboConflictError)
    assert len(api.requests) == 1
    assert sleeps == []


def test_429_honours_retry_after_header_then_succeeds() -> None:
    limited = reply(429, RATE_LIMITED, headers={"Retry-After": "7"})
    client, api, sleeps = build(sequence(limited, ok))

    response = client.request("POST", "/api/jobs/search", json={})

    assert response.data == {"ok": True}
    assert sleeps == [7.0]
    assert len(api.requests) == 2


def test_429_falls_back_to_retry_after_seconds_in_the_body() -> None:
    client, _, sleeps = build(sequence(reply(429, RATE_LIMITED), ok))

    client.request("POST", "/api/jobs/search", json={})

    assert sleeps == [7.0]


def test_503_honours_retry_after_and_replays_the_same_request() -> None:
    busy = reply(503, {"title": "Feed unavailable"}, headers={"Retry-After": "5"})
    client, api, sleeps = build(sequence(busy, busy, ok))

    client.request("POST", "/api/jobs/feed", json={"cursor": "abc"})

    assert sleeps == [5.0, 5.0]
    assert api.bodies == [{"cursor": "abc"}] * 3


def test_500_backs_off_exponentially_and_gives_up_after_max_retries() -> None:
    client, api, sleeps = build(reply(500, {"message": "An error occurred"}))

    with pytest.raises(JoboServerError) as caught:
        client.request("GET", "/api/jobs/x")

    assert caught.value.status_code == 500
    assert "An error occurred" in str(caught.value)
    assert sleeps == [1.0, 2.0]
    assert len(api.requests) == 3


def test_max_retries_zero_never_retries() -> None:
    limited = reply(429, RATE_LIMITED, headers={"Retry-After": "1"})
    client, api, sleeps = build(limited, max_retries=0)

    with pytest.raises(JoboRateLimitError) as caught:
        client.request("GET", "/x")

    assert caught.value.retry_after == 1.0
    assert caught.value.code == "rate_limit_exceeded"
    assert len(api.requests) == 1
    assert sleeps == []


def test_retry_after_longer_than_the_wait_cap_is_raised_not_slept() -> None:
    limited = reply(429, RATE_LIMITED, headers={"Retry-After": "3600"})
    client, api, sleeps = build(limited, max_retry_wait=60.0)

    with pytest.raises(JoboRateLimitError) as caught:
        client.request("GET", "/x")

    assert caught.value.retry_after == 3600.0
    assert len(api.requests) == 1
    assert sleeps == []


def test_non_json_error_body_is_kept_as_text() -> None:
    client, _, _ = build(reply(502, text="Bad Gateway"))

    with pytest.raises(JoboServerError) as caught:
        client.request("GET", "/x")

    assert caught.value.body == "Bad Gateway"
    assert caught.value.status_code == 502


def test_502_is_not_retried() -> None:
    client, api, sleeps = build(reply(502, text="Bad Gateway"))

    with pytest.raises(JoboServerError):
        client.request("GET", "/x")

    assert len(api.requests) == 1
    assert sleeps == []


def test_success_with_invalid_json_is_an_error() -> None:
    client, _, _ = build(reply(200, text="<html>"))

    with pytest.raises(JoboError, match="not valid JSON"):
        client.request("GET", "/x")


def test_failure_to_connect_is_retried() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return ok(request)

    client, _, sleeps = build(handler)

    assert client.request("GET", "/x").data == {"ok": True}
    assert sleeps == [1.0]


def test_read_timeout_is_not_replayed() -> None:
    """A page may have been delivered and billed before the timeout."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client, api, sleeps = build(handler)

    with pytest.raises(JoboConnectionError, match="ReadTimeout"):
        client.request("POST", "/api/jobs/search", json={})

    assert len(api.requests) == 1
    assert sleeps == []


async def test_async_request_sends_headers_and_parses_usage() -> None:
    client, api, _ = build(
        lambda _: httpx.Response(
            200, json={"ok": True}, headers={"X-Credits-Deducted": "3"}
        )
    )

    response = await client.arequest("POST", "/api/jobs/search", json={"page_size": 1})

    assert response.data == {"ok": True}
    assert response.usage.credits_deducted == 3
    assert api.last.headers["X-Jobo-Client"] == "langchain"
    assert api.last.headers["X-Api-Key"] == TEST_KEY


async def test_async_429_honours_retry_after_then_succeeds() -> None:
    limited = reply(429, RATE_LIMITED, headers={"Retry-After": "7"})
    client, api, sleeps = build(sequence(limited, ok))

    response = await client.arequest("POST", "/api/jobs/search", json={})

    assert response.data == {"ok": True}
    assert sleeps == [7.0]
    assert len(api.requests) == 2


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, {"error": "Missing X-Api-Key header"}, JoboAuthenticationError),
        (402, INSUFFICIENT_CREDITS, JoboInsufficientCreditsError),
        (409, CURSOR_RESTART, JoboFeedCursorRestartRequiredError),
    ],
)
async def test_async_terminal_errors_are_typed_and_not_retried(
    status: int, body: dict[str, Any], expected: type[JoboAPIError]
) -> None:
    client, api, sleeps = build(lambda _: httpx.Response(status, json=body))

    with pytest.raises(expected):
        await client.arequest("POST", "/api/jobs/feed", json={})

    assert len(api.requests) == 1
    assert sleeps == []


async def test_async_500_gives_up_after_max_retries() -> None:
    client, api, sleeps = build(reply(500, {"message": "boom"}))

    with pytest.raises(JoboServerError):
        await client.arequest("GET", "/x")

    assert sleeps == [1.0, 2.0]
    assert len(api.requests) == 3
