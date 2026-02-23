"""Entry point for the Telegram AI bot.

Usage::

    python app.py

Required environment variables (see .env.example):
    TELEGRAM_BOT_TOKEN, LLM_PROVIDER, and the matching provider API key.
"""

import logging
import sys
from typing import Any

from config import Config, load_config
from basenotes import BasenotesClient
from memory import BasenotesTokenStore, ConversationMemory, LongTermMemory
from providers.base import BaseProvider
from bot import build_application


def setup_logging() -> None:
    """Configure structured logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )
    # Reduce noise from low-level HTTP libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


# ── Provider factory ──────────────────────────────────────────────────────────

# Maps LLM_PROVIDER value → (module path, class name, api_key config attr)
_PROVIDER_MAP: dict[str, tuple[str, str, str]] = {
    "openai":    ("providers.openai_provider",    "OpenAIProvider",    "openai_api_key"),
    "anthropic": ("providers.anthropic_provider",  "AnthropicProvider", "anthropic_api_key"),
}


def _create_provider(config: Config, model: str | None = None) -> BaseProvider:
    """Instantiate the LLM provider from *config*, optionally overriding the model."""
    import importlib

    mod_path, cls_name, key_attr = _PROVIDER_MAP[config.llm_provider]
    module = importlib.import_module(mod_path)
    cls = getattr(module, cls_name)
    api_key: str = getattr(config, key_attr)
    default_model = getattr(config, f"{config.llm_provider}_model")
    return cls(api_key=api_key, model=model or default_model)


def main() -> None:
    """Load configuration, build the bot, and start long polling."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        config = load_config()
    except ValueError as exc:
        # Log to stderr so the error is visible even if stdout is redirected
        logging.critical("Configuration error: %s", exc)
        sys.exit(1)

    provider = _create_provider(config)
    research_provider = (
        _create_provider(config, model=config.research_model)
        if config.research_model
        else provider
    )

    memory = ConversationMemory(max_size=config.memory_size, db_path=config.memory_db_path)
    long_term_memory = LongTermMemory(db_path=config.memory_db_path)
    basenotes_client = BasenotesClient(
        base_url=config.basenotes_base_url,
        timeout=config.basenotes_timeout,
    )
    basenotes_tokens = BasenotesTokenStore(db_path=config.memory_db_path)

    application = build_application(
        config,
        provider,
        memory,
        research_provider,
        long_term_memory,
        basenotes_client,
        basenotes_tokens,
    )

    logger.info(
        "Bot starting — provider=%s model=%s research_model=%s memory_size=%d long_term_memory_db=%s",
        provider.name,
        provider.model,
        research_provider.model,
        config.memory_size,
        config.memory_db_path,
    )

    application.run_polling(
        allowed_updates=["message"],
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
