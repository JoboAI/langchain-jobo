"""Exceptions raised by the Jobo integration.

Every non-2xx response becomes a `JoboAPIError` subclass chosen by status code.
Branch on the class or on `status_code` first, then on the stable `code` the
API sends on most endpoint errors (`invalid_cursor`, `rate_limit_exceeded`,
`feed_cursor_restart_required`, ...). `detail` is prose and may be reworded.

A component that cannot work as configured (no API key, or a sandbox key given
to the feed loader) raises `JoboConfigurationError` when it is constructed,
before any request is made.
"""

from __future__ import annotations

from typing import Any

#: `code` of the 404 a sandbox key gets from an endpoint the sandbox lacks.
NOT_AVAILABLE_IN_SANDBOX = "not_available_in_sandbox"


class JoboError(Exception):
    """Base class for everything this package raises."""


class JoboConfigurationError(JoboError):
    """A component was set up in a way that cannot work. Raised at construction.

    Two cases: no API key was passed and `JOBO_API_KEY` is unset or blank, or
    a sandbox key (`jbe_test_`) was given to a component the sandbox does not
    serve, such as `JoboFeedLoader`.
    """


class JoboConnectionError(JoboError):
    """The request never produced an HTTP response (DNS, TLS, socket, timeout)."""


class JoboAPIError(JoboError):
    """A non-2xx response from the Jobo API.

    Attributes:
        status_code: HTTP status of the response.
        code: Stable machine-readable error code, when the API sent one.
        detail: Human-readable explanation from the response body, when present.
        body: The parsed response body (a `dict` for JSON, otherwise text).
        retry_after: Seconds the API asked the client to wait, when it said so.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        code: str | None = None,
        detail: str | None = None,
        body: Any = None,
        retry_after: float | None = None,
    ) -> None:
        """Create the error from the parts of an API error response."""
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.body = body
        self.retry_after = retry_after


class JoboValidationError(JoboAPIError):
    """HTTP 400: the request is invalid. Includes `invalid_cursor` on the feed.

    Attributes:
        errors: The API's per-field messages, e.g.
            `{"page_size": ["page_size must be between 1 and 100."]}`. Empty
            when the response named no fields. The same text is part of the
            exception message.
    """

    @property
    def errors(self) -> dict[str, list[str]]:
        """The API's per-field validation messages, keyed by field."""
        return field_errors(self.body)


class JoboAuthenticationError(JoboAPIError):
    """HTTP 401: the API key is missing, invalid, revoked or expired."""


class JoboInsufficientCreditsError(JoboAPIError):
    """HTTP 402: included jobs plus wallet balance cannot cover the request.

    Never retried. The API prices the worst case (a full page or batch) before
    the request runs, so the same request fails the same way until the wallet is
    topped up or a smaller `page_size` / `batch_size` is sent.

    Attributes:
        credits_required: Credits the request needs beyond remaining included jobs.
        credits_balance: Current wallet balance in credits (1,000 credits = $1.00).
        top_up_url: Link that opens a wallet top-up covering the shortfall.
        upgrade_url: Link that opens checkout for the recommended plan, if any.
    """

    @property
    def credits_required(self) -> int | None:
        """Credits the request needs beyond remaining included jobs."""
        return _int_field(self.body, "credits_required")

    @property
    def credits_balance(self) -> int | None:
        """Current wallet balance in credits."""
        return _int_field(self.body, "credits_balance")

    @property
    def top_up_url(self) -> str | None:
        """Link that opens a wallet top-up covering the shortfall."""
        return _str_field(self.body, "top_up_url")

    @property
    def upgrade_url(self) -> str | None:
        """Link that opens checkout for the recommended plan, when one applies."""
        return _str_field(self.body, "upgrade_url")


class JoboPermissionError(JoboAPIError):
    """HTTP 403: this key or account may not use the endpoint."""


class JoboNotFoundError(JoboAPIError):
    """HTTP 404: unknown id, or an endpoint a sandbox key cannot call.

    Attributes:
        not_available_in_sandbox: Whether the API refused the endpoint because
            the key is a sandbox key (`code == "not_available_in_sandbox"`).
    """

    @property
    def not_available_in_sandbox(self) -> bool:
        """Whether a sandbox key called an endpoint the sandbox does not serve."""
        return self.code == NOT_AVAILABLE_IN_SANDBOX


class JoboConflictError(JoboAPIError):
    """HTTP 409: the request conflicts with current state. Never retried."""


class JoboFeedCursorRestartRequiredError(JoboConflictError):
    """HTTP 409 `feed_cursor_restart_required`: the feed cursor is void.

    Never retried: replaying the cursor is guaranteed to fail. Discard it and
    start a new scan.

    Attributes:
        replacement_request: The request body the API suggests starting over with.
    """

    @property
    def replacement_request(self) -> Any:
        """The request body the API suggests starting over with, when sent."""
        if isinstance(self.body, dict):
            return self.body.get("replacement_request")
        return None


class JoboRateLimitError(JoboAPIError):
    """HTTP 429: a rate-limit window is exhausted.

    `retry_after` holds the seconds to wait, and the message states them.
    """


class JoboServerError(JoboAPIError):
    """HTTP 5xx: the API failed or is temporarily unavailable."""


def field_errors(body: Any) -> dict[str, list[str]]:
    """The `errors` map of a validation problem: field name to its messages."""
    raw = body.get("errors") if isinstance(body, dict) else None
    if not isinstance(raw, dict):
        return {}
    errors: dict[str, list[str]] = {}
    for field, messages in raw.items():
        if isinstance(messages, str):
            messages = [messages]
        if not isinstance(messages, list):
            continue
        texts = [m.strip() for m in messages if isinstance(m, str) and m.strip()]
        if texts:
            errors[str(field)] = texts
    return errors


def _int_field(body: Any, key: str) -> int | None:
    if isinstance(body, dict):
        value = body.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _str_field(body: Any, key: str) -> str | None:
    if isinstance(body, dict):
        value = body.get(key)
        if isinstance(value, str) and value:
            return value
    return None


__all__ = [
    "NOT_AVAILABLE_IN_SANDBOX",
    "JoboAPIError",
    "JoboAuthenticationError",
    "JoboConfigurationError",
    "JoboConflictError",
    "JoboConnectionError",
    "JoboError",
    "JoboFeedCursorRestartRequiredError",
    "JoboInsufficientCreditsError",
    "JoboNotFoundError",
    "JoboPermissionError",
    "JoboRateLimitError",
    "JoboServerError",
    "JoboValidationError",
    "field_errors",
]
