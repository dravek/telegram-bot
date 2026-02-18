"""Tests for memory.py — ConversationMemory buffer behaviour."""

import pytest

from memory import ConversationMemory


class TestConversationMemory:
    def test_add_and_get_single_message(self):
        mem = ConversationMemory(max_size=5)
        mem.add(1, "user", "hello")
        msgs = mem.get(1)
        assert len(msgs) == 1
        assert msgs[0] == {"role": "user", "content": "hello"}

    def test_messages_ordered_oldest_first(self):
        mem = ConversationMemory(max_size=5)
        mem.add(1, "user", "first")
        mem.add(1, "assistant", "second")
        mem.add(1, "user", "third")
        msgs = mem.get(1)
        assert [m["content"] for m in msgs] == ["first", "second", "third"]

    def test_evicts_oldest_when_full(self):
        mem = ConversationMemory(max_size=3)
        for i in range(4):
            mem.add(1, "user", str(i))
        msgs = mem.get(1)
        assert len(msgs) == 3
        assert [m["content"] for m in msgs] == ["1", "2", "3"]

    def test_reset_clears_chat(self):
        mem = ConversationMemory(max_size=5)
        mem.add(1, "user", "hello")
        mem.reset(1)
        assert mem.get(1) == []

    def test_reset_nonexistent_chat_is_noop(self):
        mem = ConversationMemory(max_size=5)
        mem.reset(999)  # should not raise
        assert mem.get(999) == []

    def test_separate_chats_are_isolated(self):
        mem = ConversationMemory(max_size=5)
        mem.add(1, "user", "chat one")
        mem.add(2, "user", "chat two")
        assert mem.get(1)[0]["content"] == "chat one"
        assert mem.get(2)[0]["content"] == "chat two"

    def test_get_returns_copy(self):
        """Mutating the returned list must not affect internal state."""
        mem = ConversationMemory(max_size=5)
        mem.add(1, "user", "hello")
        msgs = mem.get(1)
        msgs.clear()
        assert len(mem.get(1)) == 1

    def test_get_unknown_chat_returns_empty(self):
        mem = ConversationMemory(max_size=5)
        assert mem.get(42) == []
