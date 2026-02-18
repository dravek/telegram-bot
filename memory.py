"""Per-chat conversation memory buffer and persistent long-term fact store.

Short-term memory is backed by SQLite so context survives bot restarts.
When the buffer approaches *max_size* the caller can pop the oldest messages
via :meth:`ConversationMemory.pop_oldest` and produce a rolling summary
(see bot.py) rather than silently discarding old context.

:class:`LongTermMemory` provides a SQLite-backed store for facts that persist
across bot restarts (user name, preferences, bot name, etc.).  Duplicate
facts are ignored and the oldest fact is evicted (LRU) when the cap is hit.
"""

import sqlite3
from pathlib import Path
from threading import Lock
from typing import TypedDict


class Message(TypedDict):
    """A single chat message as expected by OpenAI / Anthropic APIs."""

    role: str    # "user" or "assistant"
    content: str


class ConversationMemory:
    """Thread-safe per-chat message buffer backed by SQLite.

    Messages persist across bot restarts.  :meth:`get` always returns the
    most recent *max_size* messages.  When the total stored count reaches
    ``max_size - 1``, the caller should call :meth:`pop_oldest` to reclaim
    old messages for rolling summarisation before adding new ones.
    """

    def __init__(self, max_size: int = 30, db_path: str = "memory.db") -> None:
        self._max_size = max_size
        self._db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._lock = Lock()
        self._init_db()

    @property
    def max_size(self) -> int:
        return self._max_size

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id    INTEGER NOT NULL,
                    role       TEXT    NOT NULL,
                    content    TEXT    NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_ch_chat
                    ON conversation_history(chat_id);
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    chat_id    INTEGER PRIMARY KEY,
                    summary    TEXT    NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            self._conn.commit()

    # ── message buffer ─────────────────────────────────────────────────────

    def add(self, chat_id: int, role: str, content: str) -> None:
        """Append a message to *chat_id*'s persistent history."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO conversation_history (chat_id, role, content)"
                " VALUES (?, ?, ?)",
                (chat_id, role, content),
            )
            self._conn.commit()

    def get(self, chat_id: int) -> list[Message]:
        """Return the most recent *max_size* messages for *chat_id*, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT role, content
                FROM conversation_history
                WHERE chat_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (chat_id, self._max_size),
            ).fetchall()
        return [Message(role=r, content=c) for r, c in reversed(rows)]

    def count(self, chat_id: int) -> int:
        """Return the total number of stored messages for *chat_id*."""
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM conversation_history WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()[0]

    def pop_oldest(self, chat_id: int, n: int) -> list[Message]:
        """Return and permanently remove the oldest *n* messages for *chat_id*.

        Used by the rolling summarisation logic in bot.py to condense old
        context before the buffer grows beyond *max_size*.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, role, content
                FROM conversation_history
                WHERE chat_id = ?
                ORDER BY id
                LIMIT ?
                """,
                (chat_id, n),
            ).fetchall()
            if rows:
                ids = [row[0] for row in rows]
                self._conn.execute(
                    f"DELETE FROM conversation_history WHERE id IN"
                    f" ({','.join('?' * len(ids))})",
                    ids,
                )
                self._conn.commit()
        return [Message(role=r, content=c) for _, r, c in rows]

    # ── rolling summary ────────────────────────────────────────────────────

    def get_summary(self, chat_id: int) -> str | None:
        """Return the stored rolling summary for *chat_id*, or ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT summary FROM conversation_summaries WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
        return row[0] if row else None

    def set_summary(self, chat_id: int, summary: str) -> None:
        """Persist a rolling summary for *chat_id*, replacing any previous one."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO conversation_summaries (chat_id, summary, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(chat_id) DO UPDATE SET
                    summary    = excluded.summary,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (chat_id, summary),
            )
            self._conn.commit()

    # ── lifecycle ──────────────────────────────────────────────────────────

    def reset(self, chat_id: int) -> None:
        """Clear all messages *and* the rolling summary for *chat_id*."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM conversation_history WHERE chat_id = ?", (chat_id,)
            )
            self._conn.execute(
                "DELETE FROM conversation_summaries WHERE chat_id = ?", (chat_id,)
            )
            self._conn.commit()


class LongTermMemory:
    """Persistent per-chat fact store backed by SQLite.

    Stores free-text facts (e.g. "User's name is David") that survive bot
    restarts and are injected into the system prompt.

    Duplicate facts (exact match) are silently ignored.  When the per-chat
    cap is reached the *oldest* fact is evicted so the newest information is
    always retained (LRU eviction).
    """

    MAX_FACTS = 50

    def __init__(self, db_path: str = "memory.db") -> None:
        """Open (or create) the SQLite database at *db_path*."""
        self._db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._lock = Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id    INTEGER NOT NULL,
                    fact       TEXT    NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_ltm_chat
                    ON long_term_memory(chat_id);
                """
            )
            self._conn.commit()

    def add(self, chat_id: int, fact: str) -> bool:
        """Store *fact* for *chat_id*.

        Duplicate facts (exact match) are silently ignored.  When the cap is
        reached the oldest fact is evicted to make room (LRU eviction).

        Returns:
            ``True`` if the fact was stored or already existed.
        """
        fact = fact.strip()
        if not fact:
            return False
        with self._lock:
            # Deduplicate: skip if the exact fact is already stored
            existing = self._conn.execute(
                "SELECT 1 FROM long_term_memory WHERE chat_id = ? AND fact = ?",
                (chat_id, fact),
            ).fetchone()
            if existing:
                return True

            # LRU eviction: delete the oldest fact when at cap
            count = self._conn.execute(
                "SELECT COUNT(*) FROM long_term_memory WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()[0]
            if count >= self.MAX_FACTS:
                self._conn.execute(
                    """
                    DELETE FROM long_term_memory
                    WHERE id = (
                        SELECT id FROM long_term_memory
                        WHERE chat_id = ? ORDER BY id LIMIT 1
                    )
                    """,
                    (chat_id,),
                )

            self._conn.execute(
                "INSERT INTO long_term_memory (chat_id, fact) VALUES (?, ?)",
                (chat_id, fact),
            )
            self._conn.commit()
        return True

    def get_all(self, chat_id: int) -> list[str]:
        """Return all facts for *chat_id*, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT fact FROM long_term_memory WHERE chat_id = ? ORDER BY id",
                (chat_id,),
            ).fetchall()
        return [row[0] for row in rows]

    def count(self, chat_id: int) -> int:
        """Return the number of stored facts for *chat_id*."""
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM long_term_memory WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()[0]

    def clear(self, chat_id: int) -> None:
        """Delete all facts for *chat_id*."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM long_term_memory WHERE chat_id = ?", (chat_id,)
            )
            self._conn.commit()
