"""Configuration loading and validation for the Telegram AI bot.

All settings are read from environment variables.  Call :func:`load_config`
once at startup; it raises ``ValueError`` with a descriptive message on any
misconfiguration so the process exits immediately rather than failing later.
"""

import os
from dataclasses import dataclass
from typing import Literal

LLMProvider = Literal["openai", "anthropic"]


@dataclass(frozen=True)
class Config:
    """Immutable application configuration."""

    telegram_bot_token: str
    llm_provider: LLMProvider
    openai_api_key: str | None
    openai_model: str
    anthropic_api_key: str | None
    anthropic_model: str
    memory_size: int
    # Research agent settings
    research_model: str | None      # None → use the main chat model
    research_results: int           # default source count
    research_snippet_chars: int     # chars extracted per page
    search_cache_ttl: int           # seconds to cache search results
    memory_db_path: str             # path to SQLite file for long-term memory


def load_config() -> Config:
    """Load and validate configuration from environment variables.

    Raises:
        ValueError: If any required variable is missing or invalid.
    """
    token = _require("TELEGRAM_BOT_TOKEN")

    raw_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if raw_provider not in ("openai", "anthropic"):
        raise ValueError(
            f"LLM_PROVIDER must be 'openai' or 'anthropic', got: {raw_provider!r}. "
            "Set LLM_PROVIDER=openai or LLM_PROVIDER=anthropic."
        )
    provider: LLMProvider = raw_provider  # type: ignore[assignment]

    openai_key = os.getenv("OPENAI_API_KEY") or None
    anthropic_key = os.getenv("ANTHROPIC_API_KEY") or None

    if provider == "openai" and not openai_key:
        raise ValueError(
            "OPENAI_API_KEY must be set when LLM_PROVIDER=openai."
        )
    if provider == "anthropic" and not anthropic_key:
        raise ValueError(
            "ANTHROPIC_API_KEY must be set when LLM_PROVIDER=anthropic."
        )

    return Config(
        telegram_bot_token=token,
        llm_provider=provider,
        openai_api_key=openai_key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        anthropic_api_key=anthropic_key,
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
        memory_size=_require_int("MEMORY_SIZE", default="30", minimum=1),
        research_model=os.getenv("RESEARCH_MODEL", "").strip() or None,
        research_results=_require_int("RESEARCH_RESULTS", default="5", minimum=1),
        research_snippet_chars=_require_int("RESEARCH_SNIPPET_CHARS", default="1200", minimum=100),
        search_cache_ttl=_require_int("SEARCH_CACHE_TTL", default="180", minimum=0),
        memory_db_path=os.getenv("MEMORY_DB_PATH", "memory.db"),
    )


def _require_int(name: str, *, default: str, minimum: int = 1) -> int:
    """Return env var *name* as an ``int``, or raise ``ValueError``.

    Args:
        name:    Environment variable name.
        default: Default string value if the variable is unset.
        minimum: Minimum acceptable value (inclusive).
    """
    raw = os.getenv(name, default)
    try:
        val = int(raw)
        if val < minimum:
            raise ValueError
    except ValueError:
        raise ValueError(
            f"{name} must be >= {minimum}, got: {raw!r}."
        )
    return val


def _require(name: str) -> str:
    """Return the value of *name* or raise ``ValueError`` if unset/empty."""
    val = os.getenv(name, "").strip()
    if not val:
        raise ValueError(f"Required environment variable {name!r} is not set.")
    return val
