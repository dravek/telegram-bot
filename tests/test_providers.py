"""Tests for provider wrappers — behaviour with mocked network calls."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from memory import Message
from providers.openai_provider import OpenAIProvider
from providers.anthropic_provider import AnthropicProvider


MESSAGES: list[Message] = [
    {"role": "user", "content": "Hello"},
]
SYSTEM = "Be concise."


# ── OpenAI provider ───────────────────────────────────────────────────────────

class TestOpenAIProvider:
    def _make_response(self, text: str):
        choice = MagicMock()
        choice.message.content = text
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    @pytest.mark.asyncio
    async def test_returns_text_on_success(self):
        provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini")
        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=self._make_response("Hi there!"),
        ):
            result = await provider.complete(MESSAGES, SYSTEM)
        assert result == "Hi there!"

    @pytest.mark.asyncio
    async def test_raises_permission_error_on_403(self):
        from openai import APIStatusError

        provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini")
        exc = APIStatusError("forbidden", response=MagicMock(status_code=403), body=None)

        with patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=exc,
        ):
            with pytest.raises(PermissionError):
                await provider.complete(MESSAGES, SYSTEM)

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self):
        from openai import APIConnectionError as OAIConnError

        provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini")

        call_count = 0

        async def flaky(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OAIConnError(request=MagicMock())
            return self._make_response("finally!")

        with patch.object(
            provider._client.chat.completions,
            "create",
            side_effect=flaky,
        ):
            with patch("providers.openai_provider.asyncio.sleep", new_callable=AsyncMock):
                result = await provider.complete(MESSAGES, SYSTEM)

        assert result == "finally!"
        assert call_count == 3

    def test_name_and_model(self):
        p = OpenAIProvider(api_key="k", model="gpt-4o")
        assert p.name == "openai"
        assert p.model == "gpt-4o"


# ── Anthropic provider ────────────────────────────────────────────────────────

class TestAnthropicProvider:
    def _make_response(self, text: str):
        block = MagicMock()
        block.text = text
        resp = MagicMock()
        resp.content = [block]
        return resp

    @pytest.mark.asyncio
    async def test_returns_text_on_success(self):
        provider = AnthropicProvider(api_key="sk-ant", model="claude-3-5-haiku-latest")
        with patch.object(
            provider._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=self._make_response("Hello human!"),
        ):
            result = await provider.complete(MESSAGES, SYSTEM)
        assert result == "Hello human!"

    @pytest.mark.asyncio
    async def test_raises_permission_error_on_403(self):
        from anthropic import APIStatusError

        provider = AnthropicProvider(api_key="sk-ant", model="claude-3-5-haiku-latest")
        exc = APIStatusError(
            "forbidden",
            response=MagicMock(status_code=403),
            body={"error": {"type": "permission_error"}},
        )

        with patch.object(
            provider._client.messages,
            "create",
            new_callable=AsyncMock,
            side_effect=exc,
        ):
            with pytest.raises(PermissionError):
                await provider.complete(MESSAGES, SYSTEM)

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self):
        from anthropic import APIConnectionError as AntConnError

        provider = AnthropicProvider(api_key="sk-ant", model="claude-3-5-haiku-latest")

        call_count = 0

        async def flaky(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise AntConnError(request=MagicMock())
            return self._make_response("success")

        with patch.object(
            provider._client.messages,
            "create",
            side_effect=flaky,
        ):
            with patch("providers.anthropic_provider.asyncio.sleep", new_callable=AsyncMock):
                result = await provider.complete(MESSAGES, SYSTEM)

        assert result == "success"
        assert call_count == 3

    def test_name_and_model(self):
        p = AnthropicProvider(api_key="k", model="claude-3-opus-20240229")
        assert p.name == "anthropic"
        assert p.model == "claude-3-opus-20240229"
