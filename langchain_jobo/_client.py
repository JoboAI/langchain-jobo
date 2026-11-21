"""Small sync + async HTTP client for the Jobo API.

Internal. It owns the three things every component needs to get right:

- headers: `X-Api-Key`, a `User-Agent`, and `X-Jobo-Client: langchain` so
  traffic can be attributed to this integration;
- retries: only `429`, `500` and `503` (and failures to connect) are retried,
  `Retry-After` is honoured, and `402` / `409` are never retried;
- errors: every non-2xx response is raised as a typed `JoboAPIError`.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import httpx
from pydantic import SecretStr

from langchain_jobo._version import __version__
from langchain_jobo.errors import (
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
)

DEFAULT_BASE_URL = "https://connect.jobo.world"
DEFAULT_TIMEOUT = 30.0
#: The feed can work for up to 90 seconds before answering; the API docs ask
#: for a response timeout of at least 120 seconds on feed calls.
DEFAULT_FEED_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 2
#: A `Retry-After` longer than this is not slept through: the error is raised
#: with `retry_after` set so the caller can decide what to do.
DEFAULT_MAX_RETRY_WAIT = 60.0

CLIENT_NAME = "langchain"
USER_AGENT = f"langchain-jobo/{__version__}"

_RETRYABLE_STATUSES = frozenset({429, 500, 503})
_BACKOFF_CAP_SECONDS = 30.0


@dataclass(frozen=True)
class JoboUsage:
    """Metering reported on a response's headers. `None` means "not sent".

    Attributes:
        credits_deducted: Credits charged to the wallet (1,000 credits = $1.00).
        credits_balance: Wallet balance in credits after the charge.
        quota_limit: The plan's total included jobs.
        quota_remaining: Included jobs remaining after this response.
        jobs_billed: Feed only: jobs on the page that were billed.
        billing_status: Rare `Failed` / `Unconfirmed` settlement outcome.
    """

    credits_deducted: int | None = None
    credits_balance: int | None = None
    quota_limit: int | None = None
    quota_remaining: int | None = None
    jobs_billed: int | None = None
    billing_status: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return only the values the API actually sent."""
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class JoboResponse:
    """A successful API response: the parsed JSON body and its metering."""

    data: Any
    usage: JoboUsage


class _AttemptError(Exception):
    """An attempt that failed, and whether another attempt may help."""

    def __init__(self, error: JoboError, *, retryable: bool) -> None:
        super().__init__(str(error))
        self.error = error
        self.retryable = retryable


class JoboClient:
    """Minimal Jobo API client used by the tools, retriever and loader."""

    def __init__(
        self,
        api_key: SecretStr | str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_retry_wait: float = DEFAULT_MAX_RETRY_WAIT,
        http_client: httpx.Client | None = None,
        http_async_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a client.

        Args:
            api_key: Jobo API key. Held as a `SecretStr` and only ever written
                into the `X-Api-Key` request header.
            base_url: API base URL.
            timeout: Default response timeout in seconds.
            max_retries: Retries after the first attempt for retryable failures.
            max_retry_wait: Longest single wait, in seconds, the client will
                sleep through before giving up and raising.
            http_client: Optional `httpx.Client` to send sync requests with.
            http_async_client: Optional `httpx.AsyncClient` for async requests.
        """
        secret = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        if not secret.get_secret_value().strip():
            msg = (
                "A Jobo API key is required. Pass api_key=... or set the "
                "JOBO_API_KEY environment variable. Create a key at "
                "https://enterprise.jobo.world/api-keys"
            )
            raise ValueError(msg)
        if max_retries < 0:
            msg = "max_retries must be 0 or greater."
            raise ValueError(msg)

        self._api_key = secret
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_retry_wait = max_retry_wait
        self._http_client = http_client
        self._http_async_client = http_async_client
        self._owned_client: httpx.Client | None = None
        self._sleep: Callable[[float], None] = time.sleep
        self._asleep: Callable[[float], Awaitable[None]] = asyncio.sleep
        self._random: Callable[[float, float], float] = random.uniform

    def __repr__(self) -> str:
        """Represent the client without its key."""
        return f"JoboClient(base_url={self.base_url!r})"

    # ------------------------------------------------------------------ sync

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        timeout: float | None = None,
    ) -> JoboResponse:
        """Send a request, retrying only what the API documents as retryable."""
        kwargs = self._request_kwargs(method, path, params, json, timeout)
        attempt = 0
        while True:
            try:
                return self._attempt(kwargs)
            except _AttemptError as failure:
                delay = self._retry_delay(failure, attempt)
                if delay is None:
                    raise failure.error from failure.__cause__
                self._sleep(delay)
                attempt += 1

    def _attempt(self, kwargs: dict[str, Any]) -> JoboResponse:
        try:
            response = self._sync_client().request(**kwargs)
        except httpx.TransportError as exc:
            raise _transport_failure(exc) from exc
        return _handle_response(response)

    def _sync_client(self) -> httpx.Client:
        if self._http_client is not None:
            return self._http_client
        if self._owned_client is None:
            self._owned_client = httpx.Client()
        return self._owned_client

    def close(self) -> None:
        """Close the HTTP client this instance created, if it created one."""
        if self._owned_client is not None:
            self._owned_client.close()
            self._owned_client = None

    # ----------------------------------------------------------------- async

    async def arequest(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        timeout: float | None = None,
    ) -> JoboResponse:
        """Async twin of `request`."""
        kwargs = self._request_kwargs(method, path, params, json, timeout)
        if self._http_async_client is not None:
            return await self._arequest_with(self._http_async_client, kwargs)
        # A fresh client per call: an `httpx.AsyncClient` must not be shared
        # across event loops, and callers may well use more than one.
        async with httpx.AsyncClient() as client:
            return await self._arequest_with(client, kwargs)

    async def _arequest_with(
        self, client: httpx.AsyncClient, kwargs: dict[str, Any]
    ) -> JoboResponse:
        attempt = 0
        while True:
            try:
                try:
                    response = await client.request(**kwargs)
                except httpx.TransportError as exc:
                    raise _transport_failure(exc) from exc
                return _handle_response(response)
            except _AttemptError as failure:
                delay = self._retry_delay(failure, attempt)
                if delay is None:
                    raise failure.error from failure.__cause__
                await self._asleep(delay)
                attempt += 1

    # ---------------------------------------------------------------- shared

    def _request_kwargs(
        self,
        method: str,
        path: str,
        params: Mapping[str, Any] | None,
        json: Any,
        timeout: float | None,
    ) -> dict[str, Any]:
        headers = {
            "X-Api-Key": self._api_key.get_secret_value().strip(),
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            "X-Jobo-Client": CLIENT_NAME,
        }
        kwargs: dict[str, Any] = {
            "method": method,
            "url": f"{self.base_url}{path}",
            "headers": headers,
            "timeout": self.timeout if timeout is None else timeout,
        }
        if params:
            kwargs["params"] = dict(params)
        if json is not None:
            kwargs["json"] = json
        return kwargs

    def _retry_delay(self, failure: _AttemptError, attempt: int) -> float | None:
        """Seconds to wait before the next attempt, or `None` to give up."""
        if not failure.retryable or attempt >= self.max_retries:
            return None
        error = failure.error
        retry_after = error.retry_after if isinstance(error, JoboAPIError) else None
        if retry_after is not None:
            # Honoured literally: the API says "wait at least this long".
            delay = max(retry_after, 0.0)
        else:
            base = min(float(2**attempt), _BACKOFF_CAP_SECONDS)
            delay = base * self._random(0.75, 1.25)
        return delay if delay <= self.max_retry_wait else None


def _transport_failure(exc: httpx.TransportError) -> _AttemptError:
    # Only a failure to connect is certain not to have reached the API. A
    # timeout or dropped connection after the request was sent may hide a
    # delivered (and billed) page, so it is surfaced instead of replayed.
    retryable = isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))
    error = JoboConnectionError(
        f"Could not get a response from the Jobo API: {type(exc).__name__}: {exc}"
    )
    return _AttemptError(error, retryable=retryable)


def _handle_response(response: httpx.Response) -> JoboResponse:
    if response.is_success:
        try:
            data = response.json()
        except ValueError as exc:
            error = JoboError(
                f"The Jobo API answered HTTP {response.status_code} with a body "
                "that is not valid JSON."
            )
            raise _AttemptError(error, retryable=False) from exc
        return JoboResponse(data=data, usage=_parse_usage(response.headers))

    api_error = _error_from_response(response)
    raise _AttemptError(
        api_error, retryable=response.status_code in _RETRYABLE_STATUSES
    )


def _parse_usage(headers: Mapping[str, str]) -> JoboUsage:
    return JoboUsage(
        credits_deducted=_to_int(headers.get("x-credits-deducted")),
        credits_balance=_to_int(headers.get("x-credits-balance")),
        quota_limit=_to_int(headers.get("x-quota-limit")),
        quota_remaining=_to_int(headers.get("x-quota-remaining")),
        jobs_billed=_to_int(headers.get("x-jobs-billed")),
        billing_status=headers.get("x-billing-status") or None,
    )


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


def _to_seconds(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


_ERROR_BY_STATUS: dict[int, type[JoboAPIError]] = {
    400: JoboValidationError,
    401: JoboAuthenticationError,
    402: JoboInsufficientCreditsError,
    403: JoboPermissionError,
    404: JoboNotFoundError,
    409: JoboConflictError,
    429: JoboRateLimitError,
}


def _error_from_response(response: httpx.Response) -> JoboAPIError:
    status = response.status_code
    body: Any
    try:
        body = response.json()
    except ValueError:
        body = response.text

    code: str | None = None
    detail: str | None = None
    retry_after = _to_seconds(response.headers.get("retry-after"))
    if isinstance(body, dict):
        raw_code = body.get("code")
        code = raw_code if isinstance(raw_code, str) and raw_code else None
        # Gateway errors carry `error` + `detail`, endpoint errors are RFC 7807
        # problems (`title` + `detail`), unhandled ones only have `message`.
        for key in ("detail", "error", "title", "message"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                detail = value.strip()
                break
        if retry_after is None:
            retry_after = _to_seconds(body.get("retry_after_seconds"))

    message = f"Jobo API error {status}"
    if code:
        message += f" ({code})"
    if detail:
        message += f": {detail}"

    error_cls = _ERROR_BY_STATUS.get(status)
    if status == 409 and code == "feed_cursor_restart_required":
        error_cls = JoboFeedCursorRestartRequiredError
    elif error_cls is None:
        error_cls = JoboServerError if status >= 500 else JoboAPIError

    error = error_cls(
        message,
        status_code=status,
        code=code,
        detail=detail,
        body=body,
        retry_after=retry_after,
    )
    if isinstance(error, JoboInsufficientCreditsError) and error.top_up_url:
        error.args = (f"{message} Top up: {error.top_up_url}",)
    return error
