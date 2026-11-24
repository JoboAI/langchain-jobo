"""Small sync + async HTTP client for the Jobo API.

Internal. It owns the things every component needs to get right:

- the key: resolved from the argument or `JOBO_API_KEY` when the component is
  built, so a missing key fails there and not on the first request;
- headers: `X-Api-Key`, a `User-Agent`, and `X-Jobo-Client: langchain` so
  traffic can be attributed to this integration;
- connections: one `httpx.Client` and one `httpx.AsyncClient` per event loop
  are kept and reused, and closed by `close()` / `aclose()`;
- retries: only `429`, `500` and `503` (and failures to connect) are retried,
  `Retry-After` is honoured, and `402` / `409` are never retried. A `503`
  without `Retry-After` is not retried, and neither is a `500` on a billed
  read (search and hiring signals);
- errors: every non-2xx response is raised as a typed `JoboAPIError`.
"""

from __future__ import annotations

import asyncio
import os
import random
import re
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from types import TracebackType
from typing import Any

import httpx
from pydantic import SecretStr

from langchain_jobo._version import __version__
from langchain_jobo.errors import (
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
    field_errors,
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

API_KEY_ENV = "JOBO_API_KEY"
API_KEYS_URL = "https://enterprise.jobo.world/api-keys"
#: Keys with this prefix are answered from the sandbox's fictional data.
SANDBOX_KEY_PREFIX = "jbe_test_"
MISSING_KEY_MESSAGE = (
    f"A Jobo API key is required. Pass api_key=... or set the {API_KEY_ENV} "
    f"environment variable. Create a key at {API_KEYS_URL}"
)

_RETRYABLE_STATUSES = frozenset({429, 500, 503})
#: `POST /api/jobs/search` bills every job it returns, and the hiring signals
#: reads every signal, every time. Both take the charge before they build the
#: response: a `500` can follow a charge, and a replay would charge again. Only
#: a refusal that comes before any work is retried there.
_BILLED_RETRYABLE_STATUSES = frozenset({429, 503})
_BILLED_SEARCH_PATH = "/api/jobs/search"
#: `GET /api/signals` and `GET /api/companies/{id}/signals`.
_BILLED_SIGNALS_PATH = re.compile(r"/api/signals|/api/companies/[^/]+/signals")
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


def resolve_api_key(api_key: SecretStr | str | None = None) -> SecretStr:
    """Return the key to use, trimmed, or raise when there is none.

    Args:
        api_key: The key a component was given. `None` reads `JOBO_API_KEY`.

    Raises:
        JoboConfigurationError: No key was given and the environment variable
            is unset, or the key is blank.
    """
    if api_key is None:
        api_key = os.environ.get(API_KEY_ENV, "")
    value = (
        api_key.get_secret_value() if isinstance(api_key, SecretStr) else str(api_key)
    ).strip()
    if not value:
        raise JoboConfigurationError(MISSING_KEY_MESSAGE)
    return SecretStr(value)


def is_sandbox_key(api_key: SecretStr | str) -> bool:
    """Whether the key is a sandbox key (`jbe_test_...`)."""
    value = (
        api_key.get_secret_value() if isinstance(api_key, SecretStr) else str(api_key)
    )
    return value.strip().startswith(SANDBOX_KEY_PREFIX)


def require_live_key(api_key: SecretStr | str, component: str, serves: str) -> None:
    """Refuse a sandbox key for a component the sandbox does not serve.

    Args:
        api_key: The resolved key.
        component: Class name for the message, e.g. `JoboFeedLoader`.
        serves: What the component reads, e.g. `the job feed`.

    Raises:
        JoboConfigurationError: The key is a sandbox key.
    """
    if is_sandbox_key(api_key):
        msg = (
            f"{component} needs a live API key (jbe_live_...). Sandbox keys "
            f"({SANDBOX_KEY_PREFIX}...) serve job search and job lookups only, "
            f"not {serves}. Create a live key at {API_KEYS_URL}"
        )
        raise JoboConfigurationError(msg)


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
                It is never closed by this client.
            http_async_client: Optional `httpx.AsyncClient` for async requests.
                It is never closed by this client.

        Raises:
            JoboConfigurationError: The key is blank.
            ValueError: `max_retries` is negative.
        """
        if max_retries < 0:
            msg = "max_retries must be 0 or greater."
            raise ValueError(msg)

        self._api_key = resolve_api_key(api_key)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_retry_wait = max_retry_wait
        self._http_client = http_client
        self._http_async_client = http_async_client
        self._owned_client: httpx.Client | None = None
        self._owned_client_lock = threading.Lock()
        # An `httpx.AsyncClient` belongs to the event loop it first ran on, so
        # one is kept per loop, until that loop has closed.
        self._owned_async_clients: dict[
            asyncio.AbstractEventLoop, httpx.AsyncClient
        ] = {}
        self._sleep: Callable[[float], None] = time.sleep
        self._asleep: Callable[[float], Awaitable[None]] = asyncio.sleep
        self._random: Callable[[float, float], float] = random.uniform

    def __repr__(self) -> str:
        """Represent the client without its key."""
        return f"JoboClient(base_url={self.base_url!r})"

    def __deepcopy__(self, memo: dict[int, Any]) -> JoboClient:
        """Copy the settings; the copy opens connections of its own.

        Connection pools and locks cannot be copied. HTTP clients that were
        passed in are shared with the copy, not duplicated.
        """
        return JoboClient(
            self._api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            max_retry_wait=self.max_retry_wait,
            http_client=self._http_client,
            http_async_client=self._http_async_client,
        )

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
        billed = _is_billed(method, path)
        attempt = 0
        while True:
            try:
                return self._attempt(kwargs, billed=billed)
            except _AttemptError as failure:
                delay = self._retry_delay(failure, attempt)
                if delay is None:
                    raise failure.error from failure.__cause__
                self._sleep(delay)
                attempt += 1

    def _attempt(self, kwargs: dict[str, Any], *, billed: bool) -> JoboResponse:
        try:
            response = self._sync_client().request(**kwargs)
        except httpx.TransportError as exc:
            raise _transport_failure(exc) from exc
        return _handle_response(response, billed=billed)

    def _sync_client(self) -> httpx.Client:
        if self._http_client is not None:
            return self._http_client
        with self._owned_client_lock:
            if self._owned_client is None:
                self._owned_client = httpx.Client()
            return self._owned_client

    def close(self) -> None:
        """Close the sync HTTP client this instance created, if it created one.

        A client passed in as `http_client` is left open. The instance stays
        usable: the next request opens a new client.
        """
        with self._owned_client_lock:
            client, self._owned_client = self._owned_client, None
        if client is not None:
            client.close()

    def __enter__(self) -> JoboClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

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
        billed = _is_billed(method, path)
        client = self._async_client()
        attempt = 0
        while True:
            try:
                try:
                    response = await client.request(**kwargs)
                except httpx.TransportError as exc:
                    raise _transport_failure(exc) from exc
                return _handle_response(response, billed=billed)
            except _AttemptError as failure:
                delay = self._retry_delay(failure, attempt)
                if delay is None:
                    raise failure.error from failure.__cause__
                await self._asleep(delay)
                attempt += 1

    def _async_client(self) -> httpx.AsyncClient:
        """The client for the running event loop, opened on first use."""
        if self._http_async_client is not None:
            return self._http_async_client
        loop = asyncio.get_running_loop()
        client = self._owned_async_clients.get(loop)
        if client is None:
            # A client whose loop has closed can neither be used nor awaited
            # shut: let go of it, so repeated `asyncio.run()` calls do not
            # pile up one dead client each.
            for closed in [o for o in self._owned_async_clients if o.is_closed()]:
                del self._owned_async_clients[closed]
            client = httpx.AsyncClient()
            self._owned_async_clients[loop] = client
        return client

    async def aclose(self) -> None:
        """Close the HTTP clients this instance created.

        Closes the async client of the running event loop and the sync client.
        Clients passed in as `http_client` / `http_async_client` are left
        open. The instance stays usable: the next request opens a new client.
        """
        client = self._owned_async_clients.pop(asyncio.get_running_loop(), None)
        if client is not None:
            await client.aclose()
        self.close()

    async def __aenter__(self) -> JoboClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

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
            "X-Api-Key": self._api_key.get_secret_value(),
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


def _is_billed(method: str, path: str) -> bool:
    """Whether the request bills what it returns: a search, or hiring signals."""
    verb = method.upper()
    if verb == "POST":
        return path == _BILLED_SEARCH_PATH
    return verb == "GET" and _BILLED_SIGNALS_PATH.fullmatch(path) is not None


def _is_retryable(error: JoboAPIError, *, billed: bool) -> bool:
    status = error.status_code
    if status not in (_BILLED_RETRYABLE_STATUSES if billed else _RETRYABLE_STATUSES):
        return False
    # The API sends `Retry-After` with a 503 it expects to pass ("unavailable").
    # A 503 without it, such as "Search took too long", fails the same way
    # again, and each replay holds a search thread on the server.
    return status != 503 or error.retry_after is not None


def _handle_response(response: httpx.Response, *, billed: bool) -> JoboResponse:
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
    raise _AttemptError(api_error, retryable=_is_retryable(api_error, billed=billed))


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


def _sentence(text: str) -> str:
    """End the text with a full stop unless it already ends a sentence."""
    return text if text.endswith((".", "!", "?")) else f"{text}."


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
    # A validation problem names the offending fields in `errors`; without
    # them the message is only "One or more validation errors occurred."
    fields = "; ".join(
        f"{field}: {'; '.join(text.rstrip('.') for text in texts)}"
        for field, texts in field_errors(body).items()
    )
    if fields:
        message = f"{_sentence(message)} {fields}."
    if status == 429 and retry_after is not None:
        message = f"{_sentence(message)} Retry after {retry_after:g} seconds."

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
