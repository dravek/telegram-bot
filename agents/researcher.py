"""Research agent: search → fetch → summarise pipeline.

A bounded-cost research pipeline that:
1. Decomposes the query into sub-questions (rule-based, no LLM call)
2. Searches each sub-question via DuckDuckGo (free, cached)
3. Fetches the top N pages and clips them to a configurable length
4. Sends all clipped sources to the LLM in a single summarisation call
5. Returns a plain-text cited answer with numbered references

Maximum LLM calls per invocation: **1** (the summarisation step).
"""

import asyncio
import logging
import re
import urllib.parse
from dataclasses import dataclass
from datetime import date

from memory import Message
from providers.base import BaseProvider
from tools.fetch_page import PageText, fetch_page
from tools.web_search import SearchResult, search

logger = logging.getLogger(__name__)

# ── Research modes ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ResearchMode:
    """Parameters controlling how much data the pipeline gathers."""

    sources: int        # number of unique URLs to fetch
    snippet_chars: int  # max chars extracted per page


MODES: dict[str, ResearchMode] = {
    "quick":   ResearchMode(sources=3, snippet_chars=800),
    "default": ResearchMode(sources=5, snippet_chars=1200),
    "deep":    ResearchMode(sources=8, snippet_chars=1500),
}

# ── Summarisation prompt ──────────────────────────────────────────────────────

_SUMMARISE_SYSTEM = (
    "You are a research assistant. Synthesise a clear, cited answer from the "
    "web sources provided by the user.\n\n"
    "Rules:\n"
    "- Use ONLY the information in the provided sources. "
    "Do not add knowledge from your training data.\n"
    "- If sources are insufficient or contradictory, say so explicitly.\n"
    "- Cite claims with numbered references like [1], [2].\n"
    "- End with a 'References' section listing each source as:\n"
    "  [N] Title — domain — URL\n"
    "- Keep the answer concise (3–6 short paragraphs) unless the mode is 'deep'.\n"
    "- Use plain text. Do not use Markdown symbols like **, __, or #."
)

_MAX_TELEGRAM_CHARS = 4000  # Telegram message limit with a small safety buffer


# ── Query cleaning ────────────────────────────────────────────────────────────

# Conversational prefixes that add no value to a keyword search
_CONV_PREFIX = re.compile(
    r"^(?:give\s+me|tell\s+me|show\s+me|find\s+me|can\s+you|please\s+"
    r"|i\s+want\s+to\s+know|what(?:'s|\s+is|\s+are)|\w+\s+me\s+)\s*",
    re.IGNORECASE,
)
# Filler noun-phrases ("a summary of", "an overview of", …)
_FILLER_PHRASE = re.compile(
    r"\b(?:a\s+summary\s+of|an?\s+overview\s+of"
    r"|some\s+info(?:rmation)?\s+(?:about|on)"
    r"|info(?:rmation)?\s+(?:about|on))\b",
    re.IGNORECASE,
)
# Vague temporal references → replace with current year so DDG finds recent pages
_TEMPORAL = {
    re.compile(r"\bthis\s+week\b", re.IGNORECASE): str(date.today().year),
    re.compile(r"\bthis\s+month\b", re.IGNORECASE): str(date.today().year),
    re.compile(r"\bthis\s+year\b", re.IGNORECASE): str(date.today().year),
    re.compile(r"\btoday\b", re.IGNORECASE): str(date.today().year),
    re.compile(r"\bright\s+now\b", re.IGNORECASE): str(date.today().year),
}
_MAX_SEARCH_QUERY_LEN = 80


def _to_search_query(text: str) -> str:
    """Convert a natural-language question to a concise DDG search query.

    Strips conversational lead-ins ("give me a summary of …"), filler phrases,
    and replaces vague temporal references with the current year so DDG doesn't
    receive an overly long or conversational string (which can trigger its bot
    detection / CAPTCHA).

    Args:
        text: Raw user query.

    Returns:
        Cleaned, keyword-style search string capped at
        :data:`_MAX_SEARCH_QUERY_LEN` characters.

    Examples:
        "give me a summary of the top AI news for this week"
        → "top AI news 2026"

        "tell me about Python asyncio"
        → "Python asyncio"
    """
    q = text.strip()

    # Strip conversational prefix (up to 3 passes for nested forms)
    for _ in range(3):
        cleaned = _CONV_PREFIX.sub("", q).strip()
        if cleaned == q:
            break
        q = cleaned

    # Strip "about" / "the" if they now lead the query
    q = re.sub(r"^(?:about|the)\s+", "", q, flags=re.IGNORECASE).strip()

    # Remove filler noun-phrases
    q = _FILLER_PHRASE.sub("", q).strip()

    # Collapse multiple spaces left by removals
    q = re.sub(r"\s+", " ", q).strip()

    # Replace vague temporal references
    for pattern, replacement in _TEMPORAL.items():
        q = pattern.sub(replacement, q)

    return q[:_MAX_SEARCH_QUERY_LEN]


# ── Query decomposition (rule-based, free) ────────────────────────────────────

def _decompose(query: str) -> list[str]:
    """Split a query into 1–3 sub-queries using simple rules.

    Avoids an LLM call by applying pattern matching:
    - "X vs Y" / "X versus Y"  →  search each side separately
    - Long query with " and "  →  search whole + each half
    - Everything else          →  single search

    Args:
        query: Raw user query string.

    Returns:
        List of search strings (de-duplicated, non-empty).
    """
    q = query.strip()
    q_lower = q.lower()

    for sep in (" vs ", " versus "):
        idx = q_lower.find(sep)
        if idx != -1:
            left = q[:idx].strip()
            right = q[idx + len(sep):].strip()
            if left and right:
                return [left, right]

    if " and " in q_lower and len(q) > 40:
        parts = q.split(" and ", 1)
        if len(parts[0].strip()) > 10 and len(parts[1].strip()) > 10:
            return [q, parts[0].strip(), parts[1].strip()]

    return [q]


# ── Source assembly ───────────────────────────────────────────────────────────

def _domain(url: str) -> str:
    """Return the bare domain from a URL, e.g. ``"en.wikipedia.org"``."""
    try:
        return urllib.parse.urlparse(url).netloc or url
    except Exception:
        return url


def _build_prompt(query: str, sources: list[dict]) -> str:
    """Assemble the user message that contains all sources for the LLM.

    Args:
        query:   Original user query.
        sources: List of dicts with keys: index, title, url, text.

    Returns:
        Formatted prompt string.
    """
    lines = [f"Query: {query}\n"]
    for s in sources:
        lines.append(
            f"[{s['index']}] {s['title']}\n"
            f"URL: {s['url']}\n"
            f"{s['text']}\n"
        )
    lines.append(
        "\nWrite a concise, cited answer using only the sources above. "
        "Follow the rules in the system prompt."
    )
    return "\n".join(lines)


# ── Pipeline ──────────────────────────────────────────────────────────────────

async def research(
    query: str,
    provider: BaseProvider,
    *,
    mode_name: str = "default",
    cache_ttl: int = 180,
    default_sources: int = 5,
    default_snippet_chars: int = 1200,
) -> str:
    """Run the full research pipeline and return a cited plain-text answer.

    Steps:
        1. Decompose query → sub-queries (rule-based)
        2. Search each sub-query (cached, with retry)
        3. Deduplicate + cap URLs at ``mode.sources``
        4. Fetch each page in parallel (with retry, timeout)
        5. Clip text; fall back to search snippet if fetch fails/empty
        6. One LLM summarisation call

    Args:
        query:                 User's research question.
        provider:              LLM provider to call for summarisation.
        mode_name:             ``"quick"``, ``"default"``, or ``"deep"``.
        cache_ttl:             Search cache TTL in seconds.
        default_sources:       Source count when mode is ``"default"``
                               (overridden by RESEARCH_RESULTS env var via config).
        default_snippet_chars: Clip length when mode is ``"default"``.

    Returns:
        A plain-text cited answer, or a user-friendly error string.
    """
    mode = MODES.get(mode_name, MODES["default"])
    # Allow config-level overrides on the default mode
    if mode_name == "default":
        mode = ResearchMode(
            sources=default_sources,
            snippet_chars=default_snippet_chars,
        )

    # ── 1. Clean query for search, keep original for LLM prompt ──────────────
    search_query = _to_search_query(query)
    sub_queries = _decompose(search_query)
    logger.info(
        "Research query=%r search_query=%r mode=%s sub_queries=%s",
        query, search_query, mode_name, sub_queries,
    )

    # ── 2. Search ─────────────────────────────────────────────────────────────
    search_tasks = [
        search(sq, ttl=cache_ttl, max_results=mode.sources)
        for sq in sub_queries
    ]
    search_results_per_query: list[list[SearchResult]] = await asyncio.gather(
        *search_tasks
    )

    # ── 3. Deduplicate and cap ────────────────────────────────────────────────
    seen_urls: set[str] = set()
    all_results: list[SearchResult] = []
    for results in search_results_per_query:
        for r in results:
            if r.url and r.url not in seen_urls:
                seen_urls.add(r.url)
                all_results.append(r)
            if len(all_results) >= mode.sources:
                break
        if len(all_results) >= mode.sources:
            break

    if not all_results:
        logger.warning("No search results returned for query %r", query)
        return (
            "I couldn't find any search results for that query. "
            "Please try rephrasing or try again later."
        )

    # ── 4. Fetch pages in parallel ────────────────────────────────────────────
    fetch_tasks = [
        fetch_page(r.url, max_chars=mode.snippet_chars) for r in all_results
    ]
    fetched: list[PageText | BaseException] = await asyncio.gather(
        *fetch_tasks, return_exceptions=True
    )

    # ── 5. Assemble sources list ──────────────────────────────────────────────
    sources: list[dict] = []
    for i, (result, page) in enumerate(zip(all_results, fetched), start=1):
        if isinstance(page, BaseException):
            logger.warning("Fetch exception for %s: %s", result.url, page)
            text = result.snippet or "(text unavailable)"
            title = result.title
        else:
            # Prefer fetched page text; fall back to search snippet
            text = page.text.strip() if page.text.strip() else (result.snippet or "")
            title = page.title.strip() if page.title.strip() else result.title

        if not text:
            text = "(no text retrieved)"

        sources.append(
            {
                "index": i,
                "title": title or _domain(result.url),
                "url": result.url,
                "text": text,
            }
        )

    logger.info("Assembled %d sources for summarisation", len(sources))

    # ── 6. Summarise ──────────────────────────────────────────────────────────
    prompt = _build_prompt(query, sources)
    messages: list[Message] = [{"role": "user", "content": prompt}]

    try:
        answer = await provider.complete(messages, system=_SUMMARISE_SYSTEM)
    except PermissionError:
        return (
            "I don't have access to that resource (403). "
            "Please check permissions / sharing settings."
        )
    except Exception as exc:
        logger.error("Summarisation LLM call failed: %s", exc, exc_info=True)
        return (
            "⚠️ The AI provider failed to summarise the results. "
            "Please try again in a moment."
        )

    # Guard against oversized responses
    if len(answer) > _MAX_TELEGRAM_CHARS:
        answer = answer[:_MAX_TELEGRAM_CHARS] + "\n\n(response truncated)"

    return answer
