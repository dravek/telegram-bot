"""Telegram bot command and message handlers.

All handler functions follow the signature required by ``python-telegram-bot``
v21+.  Shared state (provider, memory) is stored in ``context.bot_data`` so
handlers remain stateless functions rather than methods on a class.
"""

import logging
import time
from datetime import timedelta

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import Config
from memory import ConversationMemory
from providers.base import BaseProvider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful, concise Telegram assistant. "
    "Keep your answers short and to the point unless the user asks you to elaborate. "
    "If you don't know something, say so honestly."
)

_START_TIME = time.monotonic()


def _uptime() -> str:
    """Return a human-readable uptime string, e.g. ``"1:23:45"``."""
    elapsed = int(time.monotonic() - _START_TIME)
    return str(timedelta(seconds=elapsed))


# ── Command handlers ──────────────────────────────────────────────────────────

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command — send a brief greeting."""
    await update.message.reply_text(  # type: ignore[union-attr]
        "👋 Hi! I'm an AI-powered Telegram bot.\n\n"
        "Send me any message to start chatting, or use /help for a list of commands.\n"
        "The active AI provider is shown with /provider."
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /help command — list all available commands."""
    await update.message.reply_text(  # type: ignore[union-attr]
        "🤖 *Available commands*\n\n"
        "/start    — greeting\n"
        "/help     — this message\n"
        "/ping     — check the bot is alive + uptime\n"
        "/provider — show the active AI provider and model\n"
        "/reset    — clear your conversation history\n"
        "/research — research a topic and get a cited summary\n\n"
        "Just type any message to chat with the AI.\n\n"
        "💡 Switch providers by setting `LLM_PROVIDER=openai` or `LLM_PROVIDER=anthropic`.",
        parse_mode="Markdown",
    )


async def ping_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /ping command — respond with pong and uptime."""
    await update.message.reply_text(f"pong 🏓  (uptime: {_uptime()})")  # type: ignore[union-attr]


async def provider_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /provider command — display current provider and model."""
    provider: BaseProvider = context.bot_data["provider"]
    await update.message.reply_text(  # type: ignore[union-attr]
        f"Provider: `{provider.name}`\nModel: `{provider.model}`",
        parse_mode="Markdown",
    )


async def reset_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /reset command — clear this chat's conversation memory."""
    memory: ConversationMemory = context.bot_data["memory"]
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    memory.reset(chat_id)
    await update.message.reply_text("✅ Conversation history cleared.")  # type: ignore[union-attr]


# ── Research handler ──────────────────────────────────────────────────────────

async def research_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /research [--quick|--deep] <query> — retrieve and summarise sources.

    Searches the web, fetches the top pages, and returns a cited plain-text
    summary.  The result is intentionally **not** added to conversation memory
    so it does not inflate future chat context.
    """
    from agents.researcher import research  # local import keeps startup fast

    research_provider: BaseProvider = context.bot_data["research_provider"]
    config: Config = context.bot_data["config"]
    chat_id = update.effective_chat.id  # type: ignore[union-attr]

    args = list(context.args or [])
    mode = "default"
    query_parts: list[str] = []

    for arg in args:
        if arg == "--quick":
            mode = "quick"
        elif arg == "--deep":
            mode = "deep"
        else:
            query_parts.append(arg)

    query = " ".join(query_parts).strip()

    if not query:
        await update.message.reply_text(  # type: ignore[union-attr]
            "Usage: /research [--quick|--deep] <your question>\n\n"
            "Examples:\n"
            "  /research Python asyncio explained\n"
            "  /research --quick latest AI news\n"
            "  /research --deep climate change causes"
        )
        return

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        answer = await research(
            query,
            research_provider,
            mode_name=mode,
            cache_ttl=config.search_cache_ttl,
            default_sources=config.research_results,
            default_snippet_chars=config.research_snippet_chars,
        )
        # Plain text — research content may contain characters that break Markdown
        await update.message.reply_text(answer)  # type: ignore[union-attr]

    except Exception as exc:
        logger.error("Research error for chat %d: %s", chat_id, exc, exc_info=True)
        await update.message.reply_text(  # type: ignore[union-attr]
            "⚠️ Research failed. Please try again in a moment."
        )


# ── Natural language handler ──────────────────────────────────────────────────

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle plain-text messages — call the AI provider and reply."""
    provider: BaseProvider = context.bot_data["provider"]
    memory: ConversationMemory = context.bot_data["memory"]
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    user_text = (update.message.text or "").strip()  # type: ignore[union-attr]

    if not user_text:
        return

    memory.add(chat_id, "user", user_text)
    messages = memory.get(chat_id)

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        reply = await provider.complete(messages, system=SYSTEM_PROMPT)
        memory.add(chat_id, "assistant", reply)
        await update.message.reply_text(reply)  # type: ignore[union-attr]

    except PermissionError:
        logger.warning("403 permission error for chat %d", chat_id)
        # Remove the user message we already stored so memory stays consistent
        memory.reset(chat_id)
        await update.message.reply_text(  # type: ignore[union-attr]
            "I don't have access to that resource (403). "
            "Please check permissions / sharing settings."
        )

    except Exception as exc:
        logger.error(
            "Provider error for chat %d: %s", chat_id, exc, exc_info=True
        )
        await update.message.reply_text(  # type: ignore[union-attr]
            "⚠️ Sorry, I couldn't get a response from the AI provider right now. "
            "Please try again in a moment."
        )


# ── Application factory ───────────────────────────────────────────────────────

def build_application(
    config: Config,
    provider: BaseProvider,
    memory: ConversationMemory,
    research_provider: BaseProvider,
) -> Application:
    """Build and configure the Telegram :class:`Application`.

    Registers all command and message handlers and stores shared state in
    ``bot_data`` so every handler can access it without globals.

    Args:
        config:             Validated application configuration.
        provider:           Active LLM provider instance for chat.
        memory:             Shared per-chat conversation memory.
        research_provider:  LLM provider instance for research summarisation
                            (may be the same object as *provider*).

    Returns:
        A fully configured :class:`Application` ready to call
        :meth:`~Application.run_polling`.
    """
    app: Application = (
        Application.builder()
        .token(config.telegram_bot_token)
        .build()
    )

    app.bot_data["config"] = config
    app.bot_data["provider"] = provider
    app.bot_data["memory"] = memory
    app.bot_data["research_provider"] = research_provider

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("ping", ping_handler))
    app.add_handler(CommandHandler("provider", provider_handler))
    app.add_handler(CommandHandler("reset", reset_handler))
    app.add_handler(CommandHandler("research", research_handler))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler)
    )

    return app
