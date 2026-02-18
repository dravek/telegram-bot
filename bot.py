"""Telegram bot command and message handlers.

All handler functions follow the signature required by ``python-telegram-bot``
v21+.  Shared state (provider, memory) is stored in ``context.bot_data`` so
handlers remain stateless functions rather than methods on a class.
"""

import asyncio
import json
import logging
import re
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
from memory import ConversationMemory, LongTermMemory
from providers.base import BaseProvider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful, concise Telegram assistant. "
    "Keep your answers short and to the point unless the user asks you to elaborate. "
    "If you don't know something, say so honestly."
)

# ── Long-term memory helpers ──────────────────────────────────────────────────

_FACT_EXTRACTOR_SYSTEM = (
    "Extract personal facts worth remembering long-term from the user's message.\n"
    "Examples: user's name or nickname, bot name, location, language preference, "
    "relationships, likes/dislikes.\n"
    "Return ONLY a JSON array of short fact strings, e.g.:\n"
    '["User\'s name is David", "User wants the bot to be called Jarvis"]\n'
    "Return an empty array [] if there is nothing worth remembering.\n"
    "Be concise. One fact per item. No explanation outside the JSON array."
)


def _build_system_prompt(facts: list[str]) -> str:
    """Return the system prompt, optionally enriched with long-term facts."""
    if not facts:
        return SYSTEM_PROMPT
    facts_block = "\n".join(f"- {f}" for f in facts)
    return (
        SYSTEM_PROMPT
        + "\n\n## What you know about this user (long-term memory)\n"
        + facts_block
    )


async def _extract_facts(text: str, provider: BaseProvider) -> list[str]:
    """Ask the LLM to extract any memorable facts from *text*.

    Returns a (possibly empty) list of fact strings.  Never raises — any
    error is logged and an empty list is returned so the main flow continues.
    """
    try:
        result = await provider.complete(
            [{"role": "user", "content": text}],
            system=_FACT_EXTRACTOR_SYSTEM,
        )
        match = re.search(r"\[.*?\]", result, re.DOTALL)
        if not match:
            return []
        facts = json.loads(match.group())
        return [f for f in facts if isinstance(f, str) and f.strip()]
    except Exception as exc:
        logger.warning("Fact extraction failed: %s", exc)
        return []


async def _store_extracted_facts(
    chat_id: int,
    text: str,
    provider: BaseProvider,
    long_term: LongTermMemory,
) -> None:
    """Background task: extract facts from *text* and persist them."""
    facts = await _extract_facts(text, provider)
    for fact in facts:
        long_term.add(chat_id, fact)

_START_TIME = time.monotonic()

# ── LLM-based search intent router ───────────────────────────────────────────

# The router prompt is intentionally terse — we want exactly one word back.
# Using the research_provider (cheap model) keeps the cost negligible:
# ~50 input tokens + 1 output token ≈ $0.000008 per message with gpt-4o-mini.
_ROUTER_SYSTEM = (
    "You are a routing assistant. Decide if the user's message requires "
    "searching the web for current or live information.\n"
    "Reply with ONLY one word — no explanation, no punctuation:\n"
    "  SEARCH  — if the user wants news, current events, live prices, recent "
    "information, or explicitly asks to search/find/look something up.\n"
    "  CHAT    — if it is a general question, creative task, coding help, "
    "explanation, or anything that does not need live web data.\n"
    "Reply with ONLY the single word SEARCH or CHAT."
)


async def _needs_search(text: str, provider: BaseProvider) -> bool:
    """Ask the LLM whether *text* requires a live web search.

    Uses the research provider (cheapest configured model) for a single
    low-token classification call.  Defaults to ``False`` (normal chat) on
    any error so the bot never blocks on a routing failure.

    Works in any language — the LLM handles the classification regardless of
    what language the user writes in.

    Args:
        text:     The user's message.
        provider: LLM provider to use for classification.

    Returns:
        ``True`` if a web search should be performed.
    """
    try:
        result = await provider.complete(
            [{"role": "user", "content": text}],
            system=_ROUTER_SYSTEM,
        )
        return result.strip().upper().startswith("SEARCH")
    except Exception as exc:
        logger.warning("Router LLM call failed, defaulting to CHAT: %s", exc)
        return False


def _uptime() -> str:
    """Return a human-readable uptime string, e.g. ``"1:23:45"``."""
    elapsed = int(time.monotonic() - _START_TIME)
    return str(timedelta(seconds=elapsed))


# ── Command handlers ──────────────────────────────────────────────────────────

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command — send a brief greeting."""
    await update.message.reply_text(  # type: ignore[union-attr]
        "👋 Hi! I'm an AI-powered Telegram bot.\n\n"
        "Just chat with me naturally — I'll search the web automatically "
        "when you ask about news, prices, or anything current.\n\n"
        "Use /help to see all commands."
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
        "/research — explicit web research with optional depth flag\n\n"
        "*Long-term memory* 🧠\n"
        "/memories         — list everything I remember about you\n"
        "/remember <fact>  — manually store a fact\n"
        "/forget           — wipe all my long-term memories\n\n"
        "*Auto-search* 🔍\n"
        "You don't need commands for web searches. Just ask naturally:\n"
        "  _\"what's the latest AI news?\"\n"
        "  \"find me Python tutorials\"\n"
        "  \"what happened with OpenAI today\"_\n\n"
        "For deeper research: `/research --deep <topic>`\n\n"
        "💡 Switch providers: set `LLM_PROVIDER=openai` or `=anthropic`.",
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


async def memories_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /memories — list all stored long-term facts for this chat."""
    long_term: LongTermMemory = context.bot_data["long_term_memory"]
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    facts = long_term.get_all(chat_id)
    if not facts:
        await update.message.reply_text(  # type: ignore[union-attr]
            "🧠 I don't have any long-term memories yet.\n"
            "Just tell me things like your name or what to call me!"
        )
        return
    lines = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))
    await update.message.reply_text(  # type: ignore[union-attr]
        f"🧠 *What I remember about you:*\n\n{lines}",
        parse_mode="Markdown",
    )


async def remember_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /remember <fact> — manually store a long-term fact."""
    long_term: LongTermMemory = context.bot_data["long_term_memory"]
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    fact = " ".join(context.args or []).strip()
    if not fact:
        await update.message.reply_text(  # type: ignore[union-attr]
            "Usage: /remember <fact>\nExample: /remember My dog is called Rex"
        )
        return
    stored = long_term.add(chat_id, fact)
    if stored:
        await update.message.reply_text("✅ Got it, I'll remember that!")  # type: ignore[union-attr]
    else:
        await update.message.reply_text(  # type: ignore[union-attr]
            f"⚠️ Memory is full ({LongTermMemory.MAX_FACTS} facts). "
            "Use /forget to clear it first."
        )


async def forget_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /forget — wipe all long-term memories for this chat."""
    long_term: LongTermMemory = context.bot_data["long_term_memory"]
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    long_term.clear(chat_id)
    await update.message.reply_text("🗑️ Long-term memories cleared.")  # type: ignore[union-attr]


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
    """Handle plain-text messages — auto-route to web research or normal chat.

    If the message contains signals that fresh web information is needed
    (news, prices, current events, explicit search requests) the research
    pipeline is invoked automatically.  Otherwise the message goes to the
    normal chat path with conversation memory.
    """
    provider: BaseProvider = context.bot_data["provider"]
    research_provider: BaseProvider = context.bot_data["research_provider"]
    memory: ConversationMemory = context.bot_data["memory"]
    long_term: LongTermMemory = context.bot_data["long_term_memory"]
    config: Config = context.bot_data["config"]
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    user_text = (update.message.text or "").strip()  # type: ignore[union-attr]

    if not user_text:
        return

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    # ── Auto-route: web research vs normal chat ───────────────────────────────
    if await _needs_search(user_text, research_provider):
        from agents.researcher import research  # local import keeps startup fast

        try:
            answer = await research(
                user_text,
                research_provider,
                mode_name="default",
                cache_ttl=config.search_cache_ttl,
                default_sources=config.research_results,
                default_snippet_chars=config.research_snippet_chars,
            )
            # Research results are intentionally not stored in chat memory —
            # they are long and would inflate context on every future turn.
            await update.message.reply_text(answer)  # type: ignore[union-attr]
        except Exception as exc:
            logger.error(
                "Auto-research error for chat %d: %s", chat_id, exc, exc_info=True
            )
            await update.message.reply_text(  # type: ignore[union-attr]
                "⚠️ I tried searching the web but something went wrong. "
                "Try /research <query> or rephrase your question."
            )
        return

    # ── Normal chat with memory ───────────────────────────────────────────────
    memory.add(chat_id, "user", user_text)
    messages = memory.get(chat_id)
    system_prompt = _build_system_prompt(long_term.get_all(chat_id))

    try:
        reply = await provider.complete(messages, system=system_prompt)
        memory.add(chat_id, "assistant", reply)
        await update.message.reply_text(reply)  # type: ignore[union-attr]
        # Extract and store any memorable facts in the background (no latency cost)
        asyncio.create_task(
            _store_extracted_facts(chat_id, user_text, research_provider, long_term)
        )

    except PermissionError:
        logger.warning("403 permission error for chat %d", chat_id)
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
    long_term_memory: LongTermMemory,
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
        long_term_memory:   Persistent SQLite-backed fact store.

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
    app.bot_data["long_term_memory"] = long_term_memory

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("ping", ping_handler))
    app.add_handler(CommandHandler("provider", provider_handler))
    app.add_handler(CommandHandler("reset", reset_handler))
    app.add_handler(CommandHandler("memories", memories_handler))
    app.add_handler(CommandHandler("remember", remember_handler))
    app.add_handler(CommandHandler("forget", forget_handler))
    app.add_handler(CommandHandler("research", research_handler))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler)
    )

    return app
