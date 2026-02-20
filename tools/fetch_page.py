"""Fetch a web page and extract clean readable text.

Downloads a URL, strips all HTML tags (skipping noise elements like scripts,
nav bars, and footers), and returns the first *max_chars* characters of the
visible text body.  Falls back gracefully on network or parse errors.
"""

import asyncio
import gzip
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser

from tools.http_utils import DEFAULT_HEADERS

logger = logging.getLogger(__name__)


_TIMEOUT = 10       # seconds
_MAX_RETRIES = 2
_BASE_DELAY = 0.5
_MAX_DOWNLOAD = 512 * 1024  # 512 KB — enough for any article

# Tags whose content we discard entirely (ads, nav, scripts, etc.)
_SKIP_TAGS = frozenset(
    {"script", "style", "nav", "footer", "header", "aside", "noscript", "form"}
)


# ── Data type ─────────────────────────────────────────────────────────────────

@dataclass
class PageText:
    """Extracted text from a fetched web page."""

    url: str
    title: str
    text: str  # clipped to max_chars


# ── HTML text extractor ───────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    """Strip HTML and collect visible text, ignoring noise elements.

    Uses a depth counter for skipped tags so nested elements are handled
    correctly even in malformed HTML.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self.title = ""
        self.chunks: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title" and self._skip_depth == 0:
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title += text
        elif self._skip_depth == 0:
            self.chunks.append(text)

    def get_text(self, max_chars: int) -> str:
        """Join collected chunks and clip to *max_chars*."""
        return " ".join(self.chunks)[:max_chars]


# ── Synchronous fetch (runs in thread pool) ───────────────────────────────────

def _fetch_sync(url: str, max_chars: int) -> PageText:
    """Fetch *url* and return extracted text clipped to *max_chars*.

    Retries up to :data:`_MAX_RETRIES` times on transient network errors.
    Returns an empty :class:`PageText` on unrecoverable failure.
    """
    req = urllib.request.Request(url, headers=DEFAULT_HEADERS)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                # Skip non-HTML content types
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    logger.debug("Skipping non-HTML page: %s (%s)", url, content_type)
                    return PageText(url=url, title="", text="")

                raw: bytes = resp.read(_MAX_DOWNLOAD)
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)

            extractor = _TextExtractor()
            extractor.feed(raw.decode("utf-8", errors="replace"))
            return PageText(
                url=url,
                title=extractor.title.strip(),
                text=extractor.get_text(max_chars),
            )

        except urllib.error.HTTPError as exc:
            if exc.code == 403:
                logger.warning("403 fetching %s — skipping", url)
                return PageText(url=url, title="", text="")
            if exc.code < 500:
                logger.debug("HTTP %d fetching %s — skipping", exc.code, url)
                return PageText(url=url, title="", text="")
            logger.warning(
                "HTTP %d fetching %s (attempt %d/%d)", exc.code, url, attempt, _MAX_RETRIES
            )

        except urllib.error.URLError as exc:
            logger.warning(
                "Network error fetching %s (attempt %d/%d): %s",
                url, attempt, _MAX_RETRIES, exc,
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected error fetching %s: %s", url, exc)
            return PageText(url=url, title="", text="")

        if attempt < _MAX_RETRIES:
            time.sleep(_BASE_DELAY * (2 ** (attempt - 1)))

    return PageText(url=url, title="", text="")


# ── Public async API ──────────────────────────────────────────────────────────

async def fetch_page(url: str, *, max_chars: int = 1200) -> PageText:
    """Fetch *url* and return extracted text clipped to *max_chars*.

    Never raises — returns an empty :class:`PageText` on any failure so the
    research pipeline can degrade gracefully to snippet-only.

    Args:
        url:       URL to fetch.
        max_chars: Maximum number of characters to extract from the page body.

    Returns:
        :class:`PageText` with title and clipped body text.
    """
    return await asyncio.to_thread(_fetch_sync, url, max_chars)
