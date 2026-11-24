"""Connection settings shared by the tools and the retriever."""

from __future__ import annotations

from types import TracebackType
from typing import Any, TypeVar

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    field_validator,
    model_validator,
)

from langchain_jobo._client import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    JoboClient,
    resolve_api_key,
)

_Self = TypeVar("_Self", bound="JoboSettings")


def _api_key_from_env() -> SecretStr:
    return resolve_api_key(None)


def given(**values: Any) -> dict[str, Any]:
    """Keep the constructor arguments that were actually passed.

    The components take their settings as keyword arguments defaulting to
    `None`, so a type checker and an editor see every name and type. Only the
    ones a caller set are handed on to pydantic: a field left out keeps its
    default, including a default a subclass has overridden.
    """
    return {name: value for name, value in values.items() if value is not None}


class JoboSettings(BaseModel):
    """Fields every Jobo component takes to reach the API.

    Unknown fields are rejected, so a mistyped argument such as
    `max_pagesize=3` fails when the component is built instead of being
    ignored.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    api_key: SecretStr = Field(
        default_factory=_api_key_from_env,
        description=(
            "Jobo API key. Read from the JOBO_API_KEY environment variable when "
            "not passed."
        ),
    )
    base_url: str = Field(default=DEFAULT_BASE_URL, description="API base URL.")
    timeout: float = Field(
        default=DEFAULT_TIMEOUT, gt=0, description="Response timeout in seconds."
    )
    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        ge=0,
        description=(
            "Retries after the first attempt, for 429, 500 and 503 responses and "
            "failures to connect. 402 and 409 are never retried, nor is a 503 "
            "without Retry-After or a 500 on a billed read (search, hiring "
            "signals)."
        ),
    )
    http_client: httpx.Client | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description=(
            "Optional httpx.Client to send sync requests with. It is not closed "
            "by close()."
        ),
    )
    http_async_client: httpx.AsyncClient | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description=(
            "Optional httpx.AsyncClient to send async requests with. It is not "
            "closed by aclose()."
        ),
    )

    _jobo_client: JoboClient | None = PrivateAttr(default=None)

    @field_validator("api_key", mode="after")
    @classmethod
    def _require_api_key(cls, value: SecretStr) -> SecretStr:
        # Raises JoboConfigurationError, which pydantic does not wrap: a blank
        # key is the same typed error from every component.
        return resolve_api_key(value)

    @model_validator(mode="after")
    def _connect(self: _Self) -> _Self:
        # Built with the component, not on first use: tools run on worker
        # threads, and two of them must not race to create the client.
        self._jobo_client = self._new_client()
        return self

    def _new_client(self) -> JoboClient:
        return JoboClient(
            self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client=self.http_client,
            http_async_client=self.http_async_client,
        )

    def _api(self) -> JoboClient:
        """The API client of this component."""
        if self._jobo_client is None:
            # Only reachable through `model_construct`, which skips validation.
            self._jobo_client = self._new_client()
        return self._jobo_client

    def _share_connections_with(self, other: JoboSettings) -> None:
        """Use `other`'s API client, and with it its connection pool."""
        self._jobo_client = other._api()

    def close(self) -> None:
        """Close the HTTP connections this component opened.

        An `http_client` you passed in is left open. The component stays
        usable: its next call opens a new connection.
        """
        self._api().close()

    async def aclose(self) -> None:
        """Close the HTTP connections this component opened, sync and async.

        Clients you passed in are left open. Call it on the event loop the
        component was used on.
        """
        await self._api().aclose()

    def __enter__(self: _Self) -> _Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self: _Self) -> _Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()
