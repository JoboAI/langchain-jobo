"""Connection settings shared by the tools and the retriever."""

from __future__ import annotations

import httpx
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr

from langchain_jobo._client import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    JoboClient,
)

_MISSING_KEY_MESSAGE = (
    "A Jobo API key is required. Pass api_key=... or set the JOBO_API_KEY "
    "environment variable. Create a key at https://enterprise.jobo.world/api-keys"
)


class JoboSettings(BaseModel):
    """Fields every Jobo component takes to reach the API."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: SecretStr = Field(
        default_factory=secret_from_env(
            "JOBO_API_KEY", error_message=_MISSING_KEY_MESSAGE
        ),
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
            "failures to connect. 402 and 409 are never retried."
        ),
    )
    http_client: httpx.Client | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Optional httpx.Client to send sync requests with.",
    )
    http_async_client: httpx.AsyncClient | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Optional httpx.AsyncClient to send async requests with.",
    )

    _jobo_client: JoboClient | None = PrivateAttr(default=None)

    def _api(self) -> JoboClient:
        """The API client for this component, built on first use."""
        if self._jobo_client is None:
            self._jobo_client = JoboClient(
                self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
                http_client=self.http_client,
                http_async_client=self.http_async_client,
            )
        return self._jobo_client
