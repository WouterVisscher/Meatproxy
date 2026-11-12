"""Conversation store (SQLite) and context building."""

import pytest

from meatproxy.conversations import ConversationStore, StoredMessage, build_context


@pytest.fixture
async def store(tmp_path):
    s = ConversationStore(str(tmp_path / "conv.db"))
    await s.open()
    yield s
    await s.aclose()


async def test_append_and_history(store):
    await store.ensure_conversation("c1")
    await store.append_message("c1", "user", "hi")
    await store.append_message("c1", "assistant", "hello")
    history = await store.history("c1")
    assert [(m.role, m.content) for m in history] == [("user", "hi"), ("assistant", "hello")]


async def test_exists_and_delete(store):
    await store.ensure_conversation("c1")
    await store.append_message("c1", "user", "hi")
    assert await store.exists("c1")
    assert await store.delete("c1") is True
    assert await store.exists("c1") is False
    assert await store.history("c1") == []
    assert await store.delete("c1") is False


async def test_list_conversations(store):
    await store.ensure_conversation("a")
    await store.ensure_conversation("b")
    await store.append_message("a", "user", "x")
    listing = await store.list_conversations()
    assert {c["id"] for c in listing} == {"a", "b"}
    by_id = {c["id"]: c for c in listing}
    assert by_id["a"]["message_count"] == 1
    assert by_id["b"]["message_count"] == 0


async def test_build_context_keeps_system_and_trims_oldest():
    history = [StoredMessage("system", "be nice")] + [
        StoredMessage("user", f"u{i}")
        if i % 2 == 0
        else StoredMessage("assistant", f"a{i}")
        for i in range(10)
    ]
    context = build_context(history, max_messages=5)
    # 1 system + 4 non-system (u6 a7 u8 a9)
    assert len(context) == 5
    assert context[0] == {"role": "system", "content": "be nice"}
    assert [m["content"] for m in context[1:]] == ["u6", "a7", "u8", "a9"]


async def test_build_context_drops_orphan_tool_messages():
    history = [
        StoredMessage("user", "u0"),
        StoredMessage("assistant", "a0"),
        StoredMessage("tool", "orphan result"),
    ]
    context = build_context(history, max_messages=10)
    assert [m["role"] for m in context] == ["user", "assistant"]


async def test_build_context_single_message():
    context = build_context([StoredMessage("user", "hi")], max_messages=1)
    assert context == [{"role": "user", "content": "hi"}]
