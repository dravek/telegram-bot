"""Per-chat conversation memory buffer.

Each chat gets its own sliding window of the last *N* messages so the LLM
has context without sending unbounded history on every request.
"""

from collections import deque
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
