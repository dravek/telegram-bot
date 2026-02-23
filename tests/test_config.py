"""Tests for config.py — provider selection and required-key validation."""

import pytest

from config import load_config


def _env(monkeypatch, **kwargs: str) -> None:
    """Helper: clear relevant env vars then set only those provided."""
    for key in (
        "TELEGRAM_BOT_TOKEN",
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_MODEL",
        "ANTHROPIC_MODEL",
        "MEMORY_SIZE",
        "BASENOTES_BASE_URL",
        "BASENOTES_TIMEOUT",
        "MEMORY_DB_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in kwargs.items():
        monkeypatch.setenv(key, value)


class TestLoadConfig:
    def test_valid_openai_config(self, monkeypatch):
        _env(
            monkeypatch,
            TELEGRAM_BOT_TOKEN="tgtoken",
            LLM_PROVIDER="openai",
            OPENAI_API_KEY="sk-test",
        )
        cfg = load_config()
        assert cfg.llm_provider == "openai"
        assert cfg.openai_api_key == "sk-test"
        assert cfg.openai_model == "gpt-4o-mini"  # default
        assert cfg.memory_size == 30               # default

    def test_valid_anthropic_config(self, monkeypatch):
        _env(
            monkeypatch,
            TELEGRAM_BOT_TOKEN="tgtoken",
            LLM_PROVIDER="anthropic",
            ANTHROPIC_API_KEY="sk-ant-test",
            ANTHROPIC_MODEL="claude-3-5-haiku-latest",
            MEMORY_SIZE="5",
        )
        cfg = load_config()
        assert cfg.llm_provider == "anthropic"
        assert cfg.anthropic_api_key == "sk-ant-test"
        assert cfg.anthropic_model == "claude-3-5-haiku-latest"
        assert cfg.memory_size == 5

    def test_missing_telegram_token_raises(self, monkeypatch):
        _env(monkeypatch, LLM_PROVIDER="openai", OPENAI_API_KEY="sk-test")
        with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
            load_config()

    def test_invalid_provider_raises(self, monkeypatch):
        _env(
            monkeypatch,
            TELEGRAM_BOT_TOKEN="tgtoken",
            LLM_PROVIDER="gemini",
        )
        with pytest.raises(ValueError, match="LLM_PROVIDER"):
            load_config()

    def test_empty_provider_raises(self, monkeypatch):
        _env(monkeypatch, TELEGRAM_BOT_TOKEN="tgtoken")
        with pytest.raises(ValueError, match="LLM_PROVIDER"):
            load_config()

    def test_openai_missing_key_raises(self, monkeypatch):
        _env(monkeypatch, TELEGRAM_BOT_TOKEN="tgtoken", LLM_PROVIDER="openai")
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            load_config()

    def test_anthropic_missing_key_raises(self, monkeypatch):
        _env(
            monkeypatch,
            TELEGRAM_BOT_TOKEN="tgtoken",
            LLM_PROVIDER="anthropic",
        )
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            load_config()

    def test_invalid_memory_size_raises(self, monkeypatch):
        _env(
            monkeypatch,
            TELEGRAM_BOT_TOKEN="tgtoken",
            LLM_PROVIDER="openai",
            OPENAI_API_KEY="sk-test",
            MEMORY_SIZE="bad",
        )
        with pytest.raises(ValueError, match="MEMORY_SIZE"):
            load_config()

    def test_zero_memory_size_raises(self, monkeypatch):
        _env(
            monkeypatch,
            TELEGRAM_BOT_TOKEN="tgtoken",
            LLM_PROVIDER="openai",
            OPENAI_API_KEY="sk-test",
            MEMORY_SIZE="0",
        )
        with pytest.raises(ValueError, match="MEMORY_SIZE"):
            load_config()

    def test_custom_openai_model(self, monkeypatch):
        _env(
            monkeypatch,
            TELEGRAM_BOT_TOKEN="tgtoken",
            LLM_PROVIDER="openai",
            OPENAI_API_KEY="sk-test",
            OPENAI_MODEL="gpt-4o",
        )
        cfg = load_config()
        assert cfg.openai_model == "gpt-4o"
