"""Anthropic provider wrapper.

Uses the official ``anthropic`` async client.  Retries transient network and
rate-limit errors with exponential back-off; surfaces 403 errors as
``PermissionError`` so the bot can return a consistent user-facing message.
"""

import asyncio
import logging

from anthropic import AsyncAnthropic, APIConnectionError, APIStatusError, RateLimitError

from memory import Message
from providers.base import BaseProvider

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds
_RETRYABLE = (APIConnectionError, RateLimitError)


class AnthropicProvider(BaseProvider):
    """Async wrapper around the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str) -> None:
        """Initialise with credentials and model selection.

        Args:
            api_key: Anthropic API key.
            model:   Model identifier (e.g. ``"claude-3-5-haiku-latest"``).
        """
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        """Return ``"anthropic"``."""
        return "anthropic"

    @property
    def model(self) -> str:
        """Return the configured model identifier."""
        return self._model

    async def complete(self, messages: list[Message], system: str) -> str:
        """Call the Anthropic Messages API.

        Retries up to :data:`_MAX_RETRIES` times on transient failures.

        Args:
            messages: Conversation history (oldest first).
            system:   System prompt text.

        Returns:
            The assistant's reply text.

        Raises:
            PermissionError:           On HTTP 403.
            anthropic.APIStatusError:  On unrecoverable API errors.
        """
        api_messages = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=4096,
                    system=system,
                    messages=api_messages,  # type: ignore[arg-type]
                )
                return response.content[0].text  # type: ignore[union-attr]

            except _RETRYABLE as exc:
                if attempt == _MAX_RETRIES:
                    raise
                delay = _BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Anthropic transient error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

            except APIStatusError as exc:
                if exc.status_code == 403:
                    logger.error("Anthropic 403 permission denied: %s", exc)
                    raise PermissionError("403 from Anthropic") from exc
                raise

        return ""  # unreachable, satisfies type checker
