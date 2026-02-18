"""Per-chat conversation memory buffer and persistent long-term fact store.

Each chat gets its own sliding window of the last *N* messages so the LLM
has context without sending unbounded history on every request.

:class:`LongTermMemory` provides a SQLite-backed store for facts that persist
across bot restarts (user name, preferences, bot name, etc.).
"""

import sqlite3
from collections import deque
from pathlib import Path
from threading import Lock
from typing import TypedDict


class Message(TypedDict):
    """A single chat message as expected by OpenAI / Anthropic APIs."""

    role: str    # "user" or "assistant"
    content: str


class ConversationMemory:
    """Thread-safe per-chat message buffer with a configurable maximum size.

    When the buffer is full the oldest message is automatically evicted
    (FIFO) so the window always contains the most recent *max_size* turns.
    """

    def __init__(self, max_size: int = 10) -> None:
        """Initialise memory with a given window size.

        Args:
            max_size: Maximum number of messages stored per chat.
        """
        self._max_size = max_size
        self._chats: dict[int, deque[Message]] = {}
        self._lock = Lock()

    def add(self, chat_id: int, role: str, content: str) -> None:
        """Append a message to *chat_id*'s history.

        Automatically evicts the oldest message when the buffer is full.
        """
        with self._lock:
            if chat_id not in self._chats:
                self._chats[chat_id] = deque(maxlen=self._max_size)
            self._chats[chat_id].append(Message(role=role, content=content))

    def get(self, chat_id: int) -> list[Message]:
        """Return a snapshot of *chat_id*'s messages, oldest first."""
        with self._lock:
            return list(self._chats.get(chat_id, []))

    def reset(self, chat_id: int) -> None:
        """Clear all stored messages for *chat_id*."""
        with self._lock:
            self._chats.pop(chat_id, None)


class LongTermMemory:
    """Persistent per-chat fact store backed by SQLite.

    Stores free-text facts (e.g. "User's name is David", "Bot should be called
    Jarvis") that survive bot restarts and are injected into the system prompt
    so the LLM always has long-term context.
    """

    MAX_FACTS = 50  # hard cap per chat to keep the system prompt manageable

    def __init__(self, db_path: str = "memory.db") -> None:
        """Open (or create) the SQLite database at *db_path*."""
        self._db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._lock = Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id   INTEGER NOT NULL,
                    fact      TEXT    NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ltm_chat ON long_term_memory(chat_id)"
            )
            self._conn.commit()

    def add(self, chat_id: int, fact: str) -> bool:
        """Store *fact* for *chat_id*.

        Returns:
            ``True`` if stored, ``False`` if the per-chat limit was already reached.
        """
        fact = fact.strip()
        if not fact:
            return False
        with self._lock:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM long_term_memory WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()[0]
            if count >= self.MAX_FACTS:
                return False
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
