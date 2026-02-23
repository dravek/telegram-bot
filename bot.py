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
from datetime import datetime, timezone

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from basenotes import BasenotesAuthError, BasenotesClient, BasenotesError
from config import Config
from memory import BasenotesTokenStore, ConversationMemory, LongTermMemory
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
    "If an existing fact list is provided after '---EXISTING FACTS---', check whether\n"
    "any new fact *updates* or *replaces* an existing one (e.g. a new name supersedes\n"
    "an old name). In that case set the 'replaces' field to the exact existing fact\n"
    "string it supersedes; otherwise leave 'replaces' as null.\n"
    "Return ONLY a JSON array of objects, e.g.:\n"
    '[{"fact": "User\'s name is David", "replaces": null},\n'
    ' {"fact": "User wants the bot to be called Jarvis", "replaces": "Bot name is Max"}]\n'
    "Return an empty array [] if there is nothing worth remembering.\n"
    "Be concise. One fact per item. No explanation outside the JSON array."
)

_SUMMARIZER_SYSTEM = (
    "You are a conversation summariser. Given an optional existing summary and "
    "a block of chat messages, produce a single concise paragraph (max 120 words) "
    "that captures the key topics, decisions, and context.\n"
    "If an existing summary is provided, merge it with the new messages.\n"
    "Write in the third person: 'The user asked about...', 'The assistant explained...'.\n"
    "Return ONLY the summary paragraph — no headings, no extra text."
)


def _build_system_prompt(facts: list[str], summary: str | None = None) -> str:
    """Return the system prompt, enriched with rolling summary and long-term facts."""
    parts = [SYSTEM_PROMPT]
    if summary:
        parts.append(
            "\n\n## Earlier conversation summary\n" + summary
        )
    if facts:
        facts_block = "\n".join(f"- {f}" for f in facts)
        parts.append(
            "\n\n## What you know about this user (long-term memory)\n" + facts_block
        )
    return "".join(parts)


async def _extract_facts(
    text: str, provider: BaseProvider, existing_facts: list[str] | None = None
) -> list[dict]:
    """Ask the LLM to extract memorable facts from *text*.

    Returns a (possibly empty) list of ``{"fact": str, "replaces": str|None}``
    dicts.  Never raises — errors are logged and an empty list is returned.
    """
    try:
        content = text
        if existing_facts:
            content += (
                "\n\n---EXISTING FACTS---\n"
                + "\n".join(f"- {f}" for f in existing_facts)
            )
        result = await provider.complete(
            [{"role": "user", "content": content}],
            system=_FACT_EXTRACTOR_SYSTEM,
        )
        match = re.search(r"\[.*?\]", result, re.DOTALL)
        if not match:
            return []
        raw = json.loads(match.group())
        out = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                # Tolerate plain-string fallback from older prompts
                out.append({"fact": item.strip(), "replaces": None})
            elif isinstance(item, dict) and isinstance(item.get("fact"), str):
                out.append({
                    "fact": item["fact"].strip(),
                    "replaces": item.get("replaces") or None,
                })
        return out
    except Exception as exc:
        logger.warning("Fact extraction failed: %s", exc)
        return []


async def _store_extracted_facts(
    chat_id: int,
    text: str,
    provider: BaseProvider,
    long_term: LongTermMemory,
) -> None:
    """Background task: extract facts from *text* and persist them.

    Skips very short messages (fewer than 5 words) to avoid wasting LLM tokens.
    Uses existing facts as context so the extractor can detect updates/replacements.
    """
    if len(text.split()) < 5:
        return
    existing = long_term.get_all(chat_id)
    items = await _extract_facts(text, provider, existing_facts=existing or None)
    for item in items:
        fact = item["fact"]
        replaces = item["replaces"]
        if replaces:
            long_term.replace_fact(chat_id, replaces, fact)
        else:
            long_term.add(chat_id, fact)


async def _update_rolling_summary(
    chat_id: int,
    old_messages: list,
    existing_summary: str | None,
    provider: BaseProvider,
    memory: ConversationMemory,
) -> None:
    """Background task: condense *old_messages* into a rolling summary.

    Merges with *existing_summary* if present, then persists the result so
    the next conversation turn can inject it into the system prompt.
    """
    try:
        transcript = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in old_messages
        )
        if existing_summary:
            user_content = (
                f"Existing summary:\n{existing_summary}\n\nNew messages:\n{transcript}"
            )
        else:
            user_content = transcript
        result = await provider.complete(
            [{"role": "user", "content": user_content}],
            system=_SUMMARIZER_SYSTEM,
        )
        memory.set_summary(chat_id, result.strip())
    except Exception as exc:
        logger.warning("Rolling summarisation failed for chat %d: %s", chat_id, exc)

_START_TIME = time.monotonic()

# Hold references to background tasks so they are not garbage-collected mid-flight.
# See: https://docs.python.org/3/library/asyncio-task.html#creating-tasks
_background_tasks: set[asyncio.Task] = set()

# ── LLM-based intent router ───────────────────────────────────────────────────

# Returns JSON so the LLM can also rewrite the search query in context, which
# lets it resolve references like "find more about what we discussed" or handle
# complex sentences such as "while chatting, also look up the latest rates".
_ROUTER_SYSTEM = (
    "You are a routing assistant. Analyse the user's latest message and the\n"
    "optional conversation context provided.\n\n"
    "Decide whether the request needs a live web search for current or recent\n"
    "information (news, prices, events, explicit search/research requests).\n\n"
    "Return ONLY a single-line JSON object — no explanation, no markdown:\n"
    '  {"action": "SEARCH", "query": "<concise keyword search query>"}\n'
    '  {"action": "CHAT"}\n\n'
    "Rules for the query field (only when action is SEARCH):\n"
    "- Extract ONLY the search-worthy part; strip conversational fluff.\n"
    "- Use conversation context to resolve vague references "
    '(e.g. "that topic" → actual topic name).\n'
    "- Keep it under 80 characters, keyword-style.\n\n"
    "Choose SEARCH when the user wants: news, current events, live prices,\n"
    "recent information, or explicitly says search/find/look up/research.\n"
    "Choose CHAT for everything else: explanations, creative tasks, coding,\n"
    "opinions, follow-up questions on already-retrieved information."
)


async def _route_message(
    text: str,
    context_snippet: str,
    provider: BaseProvider,
) -> tuple[bool, str]:
    """Decide if *text* needs a web search; return ``(needs_search, refined_query)``.

    Passes recent conversation context so the router can resolve vague references
    and extract a clean search query from complex, multi-part sentences.
    Defaults to ``(False, text)`` on any error so the bot never blocks.

    Args:
        text:             The user's latest message.
        context_snippet:  Last few turns + rolling summary for reference resolution.
        provider:         LLM provider used for classification.

    Returns:
        Tuple of (needs_search, refined_query).  ``refined_query`` is a clean
        keyword string suitable for DDG; falls back to the original *text*.
    """
    try:
        content = text
        if context_snippet:
            content = f"[Conversation context]\n{context_snippet}\n\n[Latest message]\n{text}"
        result = await provider.complete(
            [{"role": "user", "content": content}],
            system=_ROUTER_SYSTEM,
        )
        # Extract the first JSON object from the response
        match = re.search(r"\{.*?\}", result.strip(), re.DOTALL)
        if not match:
            return False, text
        data = json.loads(match.group())
        if data.get("action", "").upper() == "SEARCH":
            query = (data.get("query") or text).strip() or text
            return True, query
        return False, text
    except Exception as exc:
        logger.warning("Router LLM call failed, defaulting to CHAT: %s", exc)
        return False, text


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
        "*Basenotes* 📝\n"
        "/notes_token <token>            — save your Basenotes API token\n"
        "/notes [cursor]                 — list notes (optional cursor)\n"
        "/note_create <title> | <body>   — create a note\n"
        "/note_create <title>\\n<body>    — create a note (newline body)\n"
        "/note_edit <id> <title> | <body> — edit a note\n"
        "/note_edit <id> <body>          — edit body only\n"
        "/note_edit <id> <title>\\n<body> — edit with newline body\n\n"
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


# ── Basenotes helpers and handlers ────────────────────────────────────────────

def _command_payload(update: Update, command: str) -> str:
    text = update.message.text or ""  # type: ignore[union-attr]
    prefix = f"/{command}"
    if not text.startswith(prefix):
        return ""
    return text[len(prefix):].strip()


def _split_title_body(payload: str) -> tuple[str, str] | None:
    """Split a payload into (title, body) using '|' or newline."""
    if "|" in payload:
        left, right = payload.split("|", 1)
        title = left.strip()
        body = right.strip()
        if title or body:
            return title, body
        return None
    if "\n" in payload:
        first, rest = payload.split("\n", 1)
        title = first.strip()
        body = rest.strip()
        if title or body:
            return title, body
    return None


def _format_ts(value: object) -> str:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, str):
        return value
    return "unknown"


async def notes_token_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /notes_token — store Basenotes API token for this chat."""
    tokens: BasenotesTokenStore = context.bot_data["basenotes_tokens"]
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    payload = _command_payload(update, "notes_token")
    if not payload:
        await update.message.reply_text(  # type: ignore[union-attr]
            "Usage: /notes_token <token>\n"
            "Generate it in Basenotes → Settings → API Tokens."
        )
        return
    tokens.set(chat_id, payload)
    masked = f"...{payload[-6:]}" if len(payload) > 6 else "(saved)"
    await update.message.reply_text(  # type: ignore[union-attr]
        f"✅ Basenotes token saved {masked}."
    )


async def notes_list_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /notes — list notes for the stored token."""
    client: BasenotesClient = context.bot_data["basenotes_client"]
    tokens: BasenotesTokenStore = context.bot_data["basenotes_tokens"]
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    token = tokens.get(chat_id)
    if not token:
        await update.message.reply_text(  # type: ignore[union-attr]
            "Set your Basenotes token first: /notes_token <token>"
        )
        return
    cursor = _command_payload(update, "notes") or None
    try:
        payload = await client.list_notes(token, cursor=cursor)
    except BasenotesAuthError:
        await update.message.reply_text(  # type: ignore[union-attr]
            "🔒 Basenotes token is invalid or missing permissions."
        )
        return
    except BasenotesError as exc:
        await update.message.reply_text(  # type: ignore[union-attr]
            f"⚠️ Basenotes error: {exc}"
        )
        return

    data = payload.get("data") or []
    if not data:
        await update.message.reply_text("No notes found.")  # type: ignore[union-attr]
        return
    lines = []
    for i, item in enumerate(data, start=1):
        note_id = item.get("id", "?")
        title = item.get("title") or "(untitled)"
        updated = _format_ts(item.get("updated_at"))
        lines.append(f"{i}. {note_id} — {title} (updated {updated})")
    message = "📝 Notes\n" + "\n".join(lines)
    next_cursor = payload.get("next_cursor")
    if next_cursor:
        message += f"\n\nNext cursor: {next_cursor}\nUsage: /notes {next_cursor}"
    await update.message.reply_text(message)  # type: ignore[union-attr]


async def note_create_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /note_create — create a new Basenotes note."""
    client: BasenotesClient = context.bot_data["basenotes_client"]
    tokens: BasenotesTokenStore = context.bot_data["basenotes_tokens"]
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    token = tokens.get(chat_id)
    if not token:
        await update.message.reply_text(  # type: ignore[union-attr]
            "Set your Basenotes token first: /notes_token <token>"
        )
        return
    payload = _command_payload(update, "note_create")
    parts = _split_title_body(payload)
    if not parts or not parts[0] or not parts[1]:
        await update.message.reply_text(  # type: ignore[union-attr]
            "Usage:\n"
            "/note_create <title> | <body>\n"
            "or\n"
            "/note_create <title>\\n<body>"
        )
        return
    title, body = parts
    try:
        created = await client.create_note(token, title=title, content=body)
    except BasenotesAuthError:
        await update.message.reply_text(  # type: ignore[union-attr]
            "🔒 Basenotes token is invalid or missing permissions."
        )
        return
    except BasenotesError as exc:
        await update.message.reply_text(  # type: ignore[union-attr]
            f"⚠️ Basenotes error: {exc}"
        )
        return
    note_id = created.get("id") or "(unknown id)"
    await update.message.reply_text(  # type: ignore[union-attr]
        f"✅ Note created: {note_id}"
    )


async def note_edit_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /note_edit — update an existing Basenotes note."""
    client: BasenotesClient = context.bot_data["basenotes_client"]
    tokens: BasenotesTokenStore = context.bot_data["basenotes_tokens"]
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    token = tokens.get(chat_id)
    if not token:
        await update.message.reply_text(  # type: ignore[union-attr]
            "Set your Basenotes token first: /notes_token <token>"
        )
        return
    payload = _command_payload(update, "note_edit")
    if not payload:
        await update.message.reply_text(  # type: ignore[union-attr]
            "Usage: /note_edit <id> <title> | <body>"
        )
        return
    parts = payload.split(None, 1)
    if len(parts) < 2:
        await update.message.reply_text(  # type: ignore[union-attr]
            "Usage:\n"
            "/note_edit <id> <title> | <body>\n"
            "or\n"
            "/note_edit <id> <body>\n"
            "or\n"
            "/note_edit <id> <title>\\n<body>"
        )
        return
    note_id, rest = parts[0], parts[1]
    title_body = _split_title_body(rest)
    if title_body:
        title, body = title_body
    else:
        # Treat rest as body-only update
        title, body = None, rest.strip()
        if not body:
            await update.message.reply_text(  # type: ignore[union-attr]
                "Usage:\n"
                "/note_edit <id> <title> | <body>\n"
                "or\n"
                "/note_edit <id> <body>\n"
                "or\n"
                "/note_edit <id> <title>\\n<body>"
            )
            return
    try:
        await client.update_note(token, note_id, title=title, content=body)
    except BasenotesAuthError:
        await update.message.reply_text(  # type: ignore[union-attr]
            "🔒 Basenotes token is invalid or missing permissions."
        )
        return
    except BasenotesError as exc:
        await update.message.reply_text(  # type: ignore[union-attr]
            f"⚠️ Basenotes error: {exc}"
        )
        return
    await update.message.reply_text("✅ Note updated.")  # type: ignore[union-attr]


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

    # Load facts BEFORE any branching so both paths have access to them
    facts = long_term.get_all(chat_id)
    summary = memory.get_summary(chat_id)

    # Build a brief context snippet for the router so it can resolve references
    # like "that topic" or understand multi-part sentences.
    recent_msgs = memory.get(chat_id)[-6:]  # last 3 turns (6 messages max)
    context_parts: list[str] = []
    if summary:
        context_parts.append(f"Summary: {summary}")
    if recent_msgs:
        context_parts.append(
            "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent_msgs)
        )
    context_snippet = "\n".join(context_parts)

    # ── Auto-route: web research vs normal chat ───────────────────────────────
    needs_search, refined_query = await _route_message(
        user_text, context_snippet, research_provider
    )
    if needs_search:
        from agents.researcher import research  # local import keeps startup fast

        # Pass user context + rolling summary to the summariser
        research_context_parts: list[str] = []
        if summary:
            research_context_parts.append(f"Conversation summary: {summary}")
        if facts:
            research_context_parts.append("User context: " + "; ".join(facts))
        context_note = (
            "\n\n[" + " | ".join(research_context_parts) + "]"
            if research_context_parts else ""
        )

        try:
            answer = await research(
                refined_query,
                research_provider,
                mode_name="default",
                cache_ttl=config.search_cache_ttl,
                default_sources=config.research_results,
                default_snippet_chars=config.research_snippet_chars,
                context_note=context_note,
            )
            await update.message.reply_text(answer)  # type: ignore[union-attr]

            # Add research exchange to conversation memory so follow-ups work
            memory.add(chat_id, "user", user_text)
            memory.add(chat_id, "assistant", answer)

            # Extract and persist any memorable facts from this message too
            task = asyncio.create_task(
                _store_extracted_facts(chat_id, user_text, research_provider, long_term)
            )
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
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
    # Rolling summarisation: when the buffer is nearly full, pop the oldest
    # half and condense it in the background so old context is never silently lost.
    if memory.count(chat_id) >= memory.max_size - 1:
        old_msgs = memory.pop_oldest(chat_id, memory.max_size // 2)
        if old_msgs:
            task = asyncio.create_task(
                _update_rolling_summary(
                    chat_id, old_msgs, summary,
                    research_provider, memory,
                )
            )
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

    memory.add(chat_id, "user", user_text)
    messages = memory.get(chat_id)
    system_prompt = _build_system_prompt(facts, summary)  # reuse already-loaded facts and summary

    try:
        reply = await provider.complete(messages, system=system_prompt)
        memory.add(chat_id, "assistant", reply)
        await update.message.reply_text(reply)  # type: ignore[union-attr]
        # Extract and store any memorable facts in the background (no latency cost)
        task = asyncio.create_task(
            _store_extracted_facts(chat_id, user_text, research_provider, long_term)
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

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
    basenotes_client: BasenotesClient,
    basenotes_tokens: BasenotesTokenStore,
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
        basenotes_client:   Basenotes API client for notes operations.
        basenotes_tokens:   Persistent per-chat token store.

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
    app.bot_data["basenotes_client"] = basenotes_client
    app.bot_data["basenotes_tokens"] = basenotes_tokens

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("ping", ping_handler))
    app.add_handler(CommandHandler("provider", provider_handler))
    app.add_handler(CommandHandler("reset", reset_handler))
    app.add_handler(CommandHandler("memories", memories_handler))
    app.add_handler(CommandHandler("remember", remember_handler))
    app.add_handler(CommandHandler("forget", forget_handler))
    app.add_handler(CommandHandler("notes_token", notes_token_handler))
    app.add_handler(CommandHandler("notes", notes_list_handler))
    app.add_handler(CommandHandler("note_create", note_create_handler))
    app.add_handler(CommandHandler("notes_create", note_create_handler))
    app.add_handler(CommandHandler("note_edit", note_edit_handler))
    app.add_handler(CommandHandler("notes_edit", note_edit_handler))
    app.add_handler(CommandHandler("research", research_handler))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler)
    )

    return app
