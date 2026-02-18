"""Entry point for the Telegram AI bot.

Usage::

    python app.py

Required environment variables (see .env.example):
    TELEGRAM_BOT_TOKEN, LLM_PROVIDER, and the matching provider API key.
"""

import logging
import sys

from config import load_config
from memory import ConversationMemory, LongTermMemory
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

    if config.llm_provider == "openai":
        from providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(
            api_key=config.openai_api_key,  # type: ignore[arg-type]
            model=config.openai_model,
        )
        research_provider = (
            OpenAIProvider(
                api_key=config.openai_api_key,  # type: ignore[arg-type]
                model=config.research_model,
            )
            if config.research_model
            else provider
        )
    else:
        from providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(
            api_key=config.anthropic_api_key,  # type: ignore[arg-type]
            model=config.anthropic_model,
        )
        research_provider = (
            AnthropicProvider(
                api_key=config.anthropic_api_key,  # type: ignore[arg-type]
                model=config.research_model,
            )
            if config.research_model
            else provider
        )

    memory = ConversationMemory(max_size=config.memory_size, db_path=config.memory_db_path)
    long_term_memory = LongTermMemory(db_path=config.memory_db_path)
    application = build_application(config, provider, memory, research_provider, long_term_memory)

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
