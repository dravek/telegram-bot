"""OpenAI provider wrapper.

Uses the official ``openai`` async client.  Retries transient network and
rate-limit errors with exponential back-off; surfaces 403 errors as
``PermissionError`` so the bot can return a consistent user-facing message.
"""

import asyncio
import logging

from openai import AsyncOpenAI, APIConnectionError, APIError, RateLimitError

from memory import Message
from providers.base import BaseProvider

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds
_RETRYABLE = (APIConnectionError, RateLimitError)


class OpenAIProvider(BaseProvider):
    """Async wrapper around the OpenAI Chat Completions API."""

    def __init__(self, api_key: str, model: str) -> None:
        """Initialise with credentials and model selection.

        Args:
            api_key: OpenAI API key.
            model:   Chat model identifier (e.g. ``"gpt-4o-mini"``).
        """
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        """Return ``"openai"``."""
        return "openai"

    @property
    def model(self) -> str:
        """Return the configured model identifier."""
        return self._model

    async def complete(self, messages: list[Message], system: str) -> str:
        """Call the OpenAI Chat Completions API.

        Retries up to :data:`_MAX_RETRIES` times on transient failures.

        Args:
            messages: Conversation history (oldest first).
            system:   System prompt text.

        Returns:
            The assistant's reply text.

        Raises:
            PermissionError: On HTTP 403.
            openai.APIError:  On unrecoverable API errors.
        """
        payload: list[dict[str, str]] = [{"role": "system", "content": system}]
        payload += [{"role": m["role"], "content": m["content"]} for m in messages]

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=payload,  # type: ignore[arg-type]
                )
                return response.choices[0].message.content or ""

            except _RETRYABLE as exc:
                if attempt == _MAX_RETRIES:
                    raise
                delay = _BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "OpenAI transient error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

            except APIError as exc:
                if getattr(exc, "status_code", None) == 403:
                    logger.error("OpenAI 403 permission denied: %s", exc)
                    raise PermissionError("403 from OpenAI") from exc
                raise

        return ""  # unreachable, satisfies type checker
