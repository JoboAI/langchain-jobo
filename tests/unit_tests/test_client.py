"""The HTTP layer: the key, headers, connections, retries and error mapping."""

from __future__ import annotations

import asyncio
import copy
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from langchain_jobo import (
    JoboAPIError,
    JoboAuthenticationError,
    JoboConfigurationError,
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
from langchain_jobo._client import (
    JoboClient,
    is_sandbox_key,
    require_live_key,
    resolve_api_key,
)
from tests.unit_tests.helpers import (
    CURSOR_RESTART,
    INSUFFICIENT_CREDITS,
    RATE_LIMITED,
    SANDBOX_KEY,
    TEST_KEY,
    VALIDATION_PROBLEM,
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


@pytest.mark.parametrize("key", ["", "   ", SecretStr(" ")])
def test_blank_api_key_is_rejected_with_where_to_get_one(key: str | SecretStr) -> None:
    with pytest.raises(JoboConfigurationError, match="JOBO_API_KEY") as caught:
        JoboClient(key)

    assert "https://enterprise.jobo.world/api-keys" in str(caught.value)
    assert isinstance(caught.value, JoboError)


def test_missing_key_is_read_from_the_environment_or_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(JoboConfigurationError, match="JOBO_API_KEY"):
        resolve_api_key(None)

    monkeypatch.setenv("JOBO_API_KEY", "  jbe_live_from-env\n")

    assert resolve_api_key(None).get_secret_value() == "jbe_live_from-env"


def test_key_is_trimmed_before_it_is_sent() -> None:
    api = MockJobo(ok)
    client = JoboClient(f"  {TEST_KEY}\n", http_client=api.client)

    client.request("GET", "/x")

    assert api.last.headers["X-Api-Key"] == TEST_KEY


def test_sandbox_keys_are_told_apart_by_their_prefix() -> None:
    assert is_sandbox_key(SANDBOX_KEY)
    assert is_sandbox_key(SecretStr(f" {SANDBOX_KEY}"))
    assert not is_sandbox_key(TEST_KEY)
    assert not is_sandbox_key("jbe_legacy-key")

    require_live_key(TEST_KEY, "JoboFeedLoader", "the job feed")
    with pytest.raises(JoboConfigurationError, match="JoboFeedLoader needs a live"):
        require_live_key(SANDBOX_KEY, "JoboFeedLoader", "the job feed")


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


def test_400_folds_the_field_errors_into_the_message() -> None:
    """Without them the message is only "One or more validation errors occurred."."""
    client, api, _ = build(reply(400, VALIDATION_PROBLEM))

    with pytest.raises(JoboValidationError) as caught:
        client.request("POST", "/api/jobs/feed", json={})

    assert str(caught.value) == (
        "Jobo API error 400: One or more validation errors occurred. "
        "page_size: page_size must be between 1 and 100; "
        "locations[0].country: country is required when city or region is set "
        '(a name or ISO code, e.g. "United States" or "US").'
    )
    assert caught.value.errors == VALIDATION_PROBLEM["errors"]
    assert caught.value.detail == "One or more validation errors occurred."
    assert len(api.requests) == 1


@pytest.mark.parametrize(
    "body",
    [
        {"title": "Invalid cursor", "code": "invalid_cursor"},
        {"title": "Bad", "errors": None},
        {"title": "Bad", "errors": {"page": []}},
        {"title": "Bad", "errors": ["not", "a", "map"]},
        "plain text",
    ],
)
def test_400_without_field_errors_has_an_empty_errors_map(body: Any) -> None:
    handler = reply(400, text=body) if isinstance(body, str) else reply(400, body)
    client, _, _ = build(handler)

    with pytest.raises(JoboValidationError) as caught:
        client.request("POST", "/api/jobs/feed", json={})

    assert caught.value.errors == {}
    assert str(caught.value).startswith("Jobo API error 400")


def test_400_field_errors_given_as_a_single_string_are_kept() -> None:
    client, _, _ = build(reply(400, {"title": "Bad", "errors": {"cursor": "stale"}}))

    with pytest.raises(JoboValidationError) as caught:
        client.request("POST", "/api/jobs/feed", json={})

    assert caught.value.errors == {"cursor": ["stale"]}
    assert str(caught.value) == "Jobo API error 400: Bad. cursor: stale."


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"Retry-After": "12"}, "Retry after 12 seconds."),
        ({}, "Retry after 7 seconds."),  # `retry_after_seconds` in the body
        ({"Retry-After": "0.5"}, "Retry after 0.5 seconds."),
    ],
)
def test_429_message_says_how_long_to_wait(
    headers: dict[str, str], expected: str
) -> None:
    client, _, _ = build(reply(429, RATE_LIMITED, headers=headers), max_retries=0)

    with pytest.raises(JoboRateLimitError) as caught:
        client.request("POST", "/api/jobs/search", json={})

    assert str(caught.value) == (
        "Jobo API error 429 (rate_limit_exceeded): Per-minute JobSearch request "
        f"limit of 60 exceeded. Try again shortly. {expected}"
    )


def test_429_without_a_wait_does_not_invent_one() -> None:
    client, _, _ = build(reply(429, {"error": "Too many requests"}), max_retries=0)

    with pytest.raises(JoboRateLimitError) as caught:
        client.request("GET", "/x")

    assert caught.value.retry_after is None
    assert str(caught.value) == "Jobo API error 429: Too many requests"


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


def test_500_on_the_billed_search_is_not_replayed() -> None:
    """Search takes its charge before it builds the answer: a replay bills twice."""
    client, api, sleeps = build(reply(500, {"message": "An error occurred"}))

    with pytest.raises(JoboServerError) as caught:
        client.request("POST", "/api/jobs/search", json={"page_size": 5})

    assert caught.value.status_code == 500
    assert len(api.requests) == 1
    assert sleeps == []


def test_503_with_retry_after_on_the_billed_search_is_retried() -> None:
    busy = reply(503, {"title": "Search unavailable"}, headers={"Retry-After": "5"})
    client, api, sleeps = build(sequence(busy, ok))

    response = client.request("POST", "/api/jobs/search", json={"page_size": 5})

    assert response.data == {"ok": True}
    assert sleeps == [5.0]
    assert api.bodies == [{"page_size": 5}] * 2


SIGNALS_PATHS = [
    "/api/signals",
    "/api/companies/6c1f0a9e-3b7d-4c55-8e21-5a4d9f0b7c33/signals",
]


@pytest.mark.parametrize("path", SIGNALS_PATHS)
def test_500_on_hiring_signals_is_not_replayed(path: str) -> None:
    """Signals are billed before the answer is built: a replay bills twice."""
    client, api, sleeps = build(reply(500, {"message": "An error occurred"}))

    with pytest.raises(JoboServerError):
        client.request("GET", path, params={"limit": 5})

    assert len(api.requests) == 1
    assert sleeps == []


@pytest.mark.parametrize("path", SIGNALS_PATHS)
def test_hiring_signals_retry_only_a_refusal_that_came_before_the_charge(
    path: str,
) -> None:
    limited = reply(429, RATE_LIMITED, headers={"Retry-After": "2"})
    busy = reply(503, {"title": "Unavailable"}, headers={"Retry-After": "5"})
    client, api, sleeps = build(sequence(limited, busy, ok))

    response = client.request("GET", path, params={"limit": 5})

    assert response.data == {"ok": True}
    assert sleeps == [2.0, 5.0]
    assert len(api.requests) == 3


@pytest.mark.parametrize("path", SIGNALS_PATHS)
@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.RemoteProtocolError])
def test_a_lost_hiring_signals_response_is_not_replayed(
    path: str, error: type[httpx.TransportError]
) -> None:
    """The page may have been delivered and billed before the connection broke."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise error("the response was lost", request=request)

    client, api, sleeps = build(handler)

    with pytest.raises(JoboConnectionError, match=error.__name__):
        client.request("GET", path)

    assert len(api.requests) == 1
    assert sleeps == []


def test_a_500_on_another_company_path_is_still_retried() -> None:
    """Only the signals reads are billed; a company's jobs list is not."""
    client, api, sleeps = build(reply(500, {"message": "boom"}))

    with pytest.raises(JoboServerError):
        client.request("GET", "/api/companies/6c1f0a9e/jobs")

    assert len(api.requests) == 3
    assert sleeps == [1.0, 2.0]


async def test_async_hiring_signals_are_not_replayed_after_a_500() -> None:
    client, api, sleeps = build(reply(500, {"message": "boom"}))

    with pytest.raises(JoboServerError):
        await client.arequest("GET", "/api/signals", params={"types": "hiring_surge"})

    assert len(api.requests) == 1
    assert sleeps == []


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/jobs/search"),
        ("POST", "/api/jobs/feed"),
        ("GET", "/api/jobs/x"),
        ("GET", "/api/signals"),
    ],
)
def test_503_without_retry_after_is_not_retried(method: str, path: str) -> None:
    """A "Search took too long" 503 fails the same way again: no Retry-After."""
    too_long = reply(503, {"title": "Search took too long"})
    client, api, sleeps = build(too_long)

    with pytest.raises(JoboServerError) as caught:
        client.request(method, path, json={} if method == "POST" else None)

    assert caught.value.status_code == 503
    assert caught.value.retry_after is None
    assert len(api.requests) == 1
    assert sleeps == []


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


async def test_async_billed_search_retries_only_before_the_charge() -> None:
    failed = reply(500, {"message": "boom"})
    too_long = reply(503, {"title": "Search took too long"})
    busy = reply(503, {"title": "Search unavailable"}, headers={"Retry-After": "5"})
    client, api, sleeps = build(sequence(failed, too_long, busy, ok))

    with pytest.raises(JoboServerError, match="500"):
        await client.arequest("POST", "/api/jobs/search", json={})
    with pytest.raises(JoboServerError, match="503"):
        await client.arequest("POST", "/api/jobs/search", json={})
    response = await client.arequest("POST", "/api/jobs/search", json={})

    assert response.data == {"ok": True}
    assert sleeps == [5.0]
    assert len(api.requests) == 4


# ---------------------------------------------------------------- connections


class CountingClients:
    """Stands in for `httpx.Client` / `httpx.AsyncClient`, counting what is made."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.sync: list[httpx.Client] = []
        self.asynchronous: list[httpx.AsyncClient] = []
        real_sync, real_async = httpx.Client, httpx.AsyncClient

        def make_sync() -> httpx.Client:
            self.sync.append(real_sync(transport=httpx.MockTransport(ok)))
            return self.sync[-1]

        def make_async() -> httpx.AsyncClient:
            self.asynchronous.append(real_async(transport=httpx.MockTransport(ok)))
            return self.asynchronous[-1]

        monkeypatch.setattr(httpx, "Client", make_sync)
        monkeypatch.setattr(httpx, "AsyncClient", make_async)


@pytest.fixture
def made(monkeypatch: pytest.MonkeyPatch) -> CountingClients:
    return CountingClients(monkeypatch)


def test_sync_requests_reuse_one_connection_pool(made: CountingClients) -> None:
    client = JoboClient(TEST_KEY)

    for _ in range(3):
        assert client.request("GET", "/x").data == {"ok": True}

    assert len(made.sync) == 1
    assert not made.sync[0].is_closed


def test_close_closes_the_pool_and_the_client_stays_usable(
    made: CountingClients,
) -> None:
    client = JoboClient(TEST_KEY)
    client.request("GET", "/x")

    client.close()
    client.close()  # closing twice is harmless

    assert made.sync[0].is_closed
    assert client.request("GET", "/x").data == {"ok": True}
    assert len(made.sync) == 2


def test_client_is_a_context_manager(made: CountingClients) -> None:
    with JoboClient(TEST_KEY) as client:
        client.request("GET", "/x")

    assert made.sync[0].is_closed


def test_a_client_that_was_passed_in_is_never_closed() -> None:
    api = MockJobo(ok)

    with JoboClient(TEST_KEY, http_client=api.client) as client:
        client.request("GET", "/x")

    assert not api.client.is_closed


async def test_async_requests_reuse_one_connection_pool(
    made: CountingClients,
) -> None:
    client = JoboClient(TEST_KEY)

    for _ in range(3):
        assert (await client.arequest("GET", "/x")).data == {"ok": True}

    assert len(made.asynchronous) == 1
    assert made.sync == []


async def test_aclose_closes_the_async_and_the_sync_pool(
    made: CountingClients,
) -> None:
    client = JoboClient(TEST_KEY)
    await client.arequest("GET", "/x")
    client.request("GET", "/x")

    await client.aclose()
    await client.aclose()

    assert made.asynchronous[0].is_closed
    assert made.sync[0].is_closed
    assert (await client.arequest("GET", "/x")).data == {"ok": True}
    assert len(made.asynchronous) == 2


async def test_client_is_an_async_context_manager(made: CountingClients) -> None:
    async with JoboClient(TEST_KEY) as client:
        await client.arequest("GET", "/x")

    assert made.asynchronous[0].is_closed


async def test_an_async_client_that_was_passed_in_is_never_closed() -> None:
    api = MockJobo(ok)

    async with JoboClient(TEST_KEY, http_async_client=api.async_client) as client:
        await client.arequest("GET", "/x")

    assert not api.async_client.is_closed


def test_each_event_loop_gets_its_own_async_client(made: CountingClients) -> None:
    """An `httpx.AsyncClient` cannot be shared between event loops."""
    client = JoboClient(TEST_KEY)

    async def twice() -> None:
        await client.arequest("GET", "/x")
        await client.arequest("GET", "/x")

    asyncio.run(twice())
    asyncio.run(twice())

    assert len(made.asynchronous) == 2


def test_clients_of_closed_event_loops_are_let_go(made: CountingClients) -> None:
    """A long-lived component called through `asyncio.run()` must not grow."""
    client = JoboClient(TEST_KEY)

    for _ in range(5):
        asyncio.run(client.arequest("GET", "/x"))

    assert len(made.asynchronous) == 5
    assert len(client._owned_async_clients) == 1


def test_a_deep_copy_keeps_the_settings_and_opens_its_own_connections(
    made: CountingClients,
) -> None:
    client = JoboClient(TEST_KEY, base_url="https://example.test", max_retries=0)
    client.request("GET", "/x")

    twin = copy.deepcopy(client)
    twin.request("GET", "/x")

    assert twin.base_url == "https://example.test"
    assert twin.max_retries == 0
    assert len(made.sync) == 2
    client.close()
    assert not made.sync[1].is_closed


def test_a_deep_copy_shares_a_client_that_was_passed_in() -> None:
    api = MockJobo(ok)
    client = JoboClient(TEST_KEY, http_client=api.client)

    copy.deepcopy(client).request("GET", "/x")

    assert api.last.headers["X-Api-Key"] == TEST_KEY
