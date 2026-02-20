"""DuckDuckGo HTML search with in-memory TTL caching.

Uses the public HTML endpoint — no API key required.  Results are cached per
normalised query to avoid hammering the endpoint on repeated lookups.

Caveat: DDG may change its HTML structure.  If ``search()`` returns an empty
list unexpectedly, check the WARNING log for a parse hint.
"""

import asyncio
import gzip
import hashlib
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser

from tools.http_utils import DEFAULT_HEADERS

logger = logging.getLogger(__name__)


_SEARCH_URL = "https://html.duckduckgo.com/html/?q={}&kl=us-en"
_TIMEOUT = 10  # seconds per request
_MAX_RETRIES = 3
_BASE_DELAY = 1.0


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single search result returned by DuckDuckGo."""

    url: str
    title: str
    snippet: str


# ── TTL cache ─────────────────────────────────────────────────────────────────

_MAX_CACHE_SIZE = 256  # max entries before forced cleanup


class _CacheEntry:
    def __init__(self, results: list[SearchResult], ttl: int) -> None:
        self.results = results
        self._expires = time.monotonic() + ttl

    def is_fresh(self) -> bool:
        """Return True if the entry has not yet expired."""
        return time.monotonic() < self._expires


_cache: dict[str, _CacheEntry] = {}


def _cache_key(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


def _evict_expired() -> None:
    """Remove expired entries from the cache to prevent unbounded growth."""
    expired = [k for k, v in _cache.items() if not v.is_fresh()]
    for k in expired:
        del _cache[k]


# ── HTML parser ───────────────────────────────────────────────────────────────

def _extract_url(href: str) -> str:
    """Resolve a DuckDuckGo redirect href to the real destination URL."""
    if not href:
        return ""
    # DDG wraps result links: /l/?uddg=<encoded-url>&rut=...
    if "/l/?" in href:
        if href.startswith("//"):
            href = "https:" + href
        elif not href.startswith("http"):
            href = "https://duckduckgo.com" + href
        parsed = urllib.parse.urlparse(href)
        qs = urllib.parse.parse_qs(parsed.query)
        urls = qs.get("uddg", [])
        return urls[0] if urls else ""
    return href


# Tags that are void (self-closing) and should not increment depth tracking
_VOID_ELEMENTS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input",
     "link", "meta", "param", "source", "track", "wbr"}
)


class _DDGParser(HTMLParser):
    """Extract title, URL, and snippet from DuckDuckGo's HTML results page.

    State machine:
      mode=0  waiting for next result element
      mode=1  inside a ``result__a`` anchor (title + URL)
      mode=2  inside a ``result__snippet`` element
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[SearchResult] = []
        self._mode = 0
        self._depth = 0
        self._url = ""
        self._title = ""
        self._snippet = ""

    # -- internal helpers -----------------------------------------------------

    def _flush_pending(self) -> None:
        """Emit the current in-progress result (may have no snippet yet)."""
        if self._url and self._title:
            self.results.append(
                SearchResult(
                    url=self._url,
                    title=self._title.strip(),
                    snippet=self._snippet.strip(),
                )
            )
        self._url = ""
        self._title = ""
        self._snippet = ""

    # -- HTMLParser interface --------------------------------------------------

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attr = dict(attrs)
        css = (attr.get("class") or "").split()
        href = attr.get("href") or ""

        if self._mode == 0:
            if "result__a" in css:
                # Flush any pending result that had a title but no snippet
                self._flush_pending()
                self._url = _extract_url(href)
                self._title = ""
                self._snippet = ""
                self._mode = 1
                self._depth = 1
            elif "result__snippet" in css:
                self._snippet = ""
                self._mode = 2
                self._depth = 1
        else:
            # Track nesting depth so we know when the active element closes
            # Skip void elements — they have no closing tag
            if tag not in _VOID_ELEMENTS:
                self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._mode != 0 and self._depth > 0:
            self._depth -= 1
            if self._depth == 0:
                if self._mode == 2:
                    # Snippet closed — emit complete result
                    self._flush_pending()
                self._mode = 0

    def handle_data(self, data: str) -> None:
        if self._mode == 1:
            self._title += data
        elif self._mode == 2:
            self._snippet += data


# ── Synchronous search (runs in thread pool) ──────────────────────────────────

def _search_sync(query: str, max_results: int) -> list[SearchResult]:
    """Perform a blocking DuckDuckGo HTML search.

    Retries up to :data:`_MAX_RETRIES` times on network errors.
    HTTP 4xx errors are not retried (permanent client errors).
    """
    url = _SEARCH_URL.format(urllib.parse.quote_plus(query))
    req = urllib.request.Request(url, headers=DEFAULT_HEADERS)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                raw: bytes = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                html = raw.decode("utf-8", errors="replace")

            parser = _DDGParser()
            parser.feed(html)

            results = parser.results[:max_results]
            # Filter out results with no URL (parse artefacts)
            results = [r for r in results if r.url.startswith("http")]

            if not results:
                logger.warning(
                    "DDG search returned 0 parsed results for %r "
                    "(HTML structure may have changed)",
                    query,
                )
            return results

        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                # 4xx — not a transient error, don't retry
                logger.error("DDG search HTTP %d for query %r", exc.code, query)
                return []
            logger.warning(
                "DDG search HTTP %d (attempt %d/%d)", exc.code, attempt, _MAX_RETRIES
            )

        except urllib.error.URLError as exc:
            logger.warning(
                "DDG search network error (attempt %d/%d): %s", attempt, _MAX_RETRIES, exc
            )

        if attempt < _MAX_RETRIES:
            time.sleep(_BASE_DELAY * (2 ** (attempt - 1)))

    return []


# ── Public async API ──────────────────────────────────────────────────────────

async def search(
    query: str,
    *,
    ttl: int = 180,
    max_results: int = 10,
) -> list[SearchResult]:
    """Search DuckDuckGo and return up to *max_results* results.

    Results are cached for *ttl* seconds to avoid redundant requests.

    Args:
        query:       Search query string.
        ttl:         Cache lifetime in seconds.
        max_results: Maximum number of results to return.

    Returns:
        List of :class:`SearchResult` objects, possibly empty on failure.
    """
    key = _cache_key(query)
    entry = _cache.get(key)
    if entry and entry.is_fresh():
        logger.debug("Cache hit for query %r", query)
        return entry.results

    # Prevent unbounded cache growth
    if len(_cache) >= _MAX_CACHE_SIZE:
        _evict_expired()

    results = await asyncio.to_thread(_search_sync, query, max_results)
    _cache[key] = _CacheEntry(results, ttl)
    return results
