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

    raw_memory = os.getenv("MEMORY_SIZE", "30")
    try:
        memory_size = int(raw_memory)
        if memory_size < 1:
            raise ValueError
    except ValueError:
        raise ValueError(
            f"MEMORY_SIZE must be a positive integer, got: {raw_memory!r}."
        )

    research_model = os.getenv("RESEARCH_MODEL", "").strip() or None

    raw_rr = os.getenv("RESEARCH_RESULTS", "5")
    try:
        research_results = int(raw_rr)
        if research_results < 1:
            raise ValueError
    except ValueError:
        raise ValueError(
            f"RESEARCH_RESULTS must be a positive integer, got: {raw_rr!r}."
        )

    raw_rsc = os.getenv("RESEARCH_SNIPPET_CHARS", "1200")
    try:
        research_snippet_chars = int(raw_rsc)
        if research_snippet_chars < 100:
            raise ValueError
    except ValueError:
        raise ValueError(
            f"RESEARCH_SNIPPET_CHARS must be >= 100, got: {raw_rsc!r}."
        )

    raw_ttl = os.getenv("SEARCH_CACHE_TTL", "180")
    try:
        search_cache_ttl = int(raw_ttl)
        if search_cache_ttl < 0:
            raise ValueError
    except ValueError:
        raise ValueError(
            f"SEARCH_CACHE_TTL must be a non-negative integer, got: {raw_ttl!r}."
        )

    return Config(
        telegram_bot_token=token,
        llm_provider=provider,
        openai_api_key=openai_key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        anthropic_api_key=anthropic_key,
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
        memory_size=memory_size,
        research_model=research_model,
        research_results=research_results,
        research_snippet_chars=research_snippet_chars,
        search_cache_ttl=search_cache_ttl,
        memory_db_path=os.getenv("MEMORY_DB_PATH", "memory.db"),
    )


def _require(name: str) -> str:
    """Return the value of *name* or raise ``ValueError`` if unset/empty."""
    val = os.getenv(name, "").strip()
    if not val:
        raise ValueError(f"Required environment variable {name!r} is not set.")
    return val
