"""Conversation persistence in SQLite (WAL).

meatproxy is the source of truth for conversation history: the IDE sends
only the newest user message plus an ``X-Conversation-Id`` header, and the
router reconstructs context from this store for stateless backends.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    last_active_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE TABLE IF NOT EXISTS messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    seq              INTEGER NOT NULL,
    role             TEXT NOT NULL,
    content          TEXT NOT NULL DEFAULT '',
    tool_calls       TEXT,
    tool_call_id     TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, seq);
"""


@dataclass
class StoredMessage:
    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ConversationStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._db: aiosqlite.Connection | None = None

    async def open(self) -> None:
        self._db = await aiosqlite.connect(self.path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.executescript(_SCHEMA)
        cursor = await self._db.execute("PRAGMA table_info(messages)")
        columns = {row[1] for row in await cursor.fetchall()}
        await cursor.close()
        if "tool_call_id" not in columns:
            await self._db.execute("ALTER TABLE messages ADD COLUMN tool_call_id TEXT")
        await self._db.commit()

    async def aclose(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("ConversationStore is not open")
        return self._db

    async def ensure_conversation(self, conv_id: str) -> None:
        await self.db.execute(
            "INSERT OR IGNORE INTO conversations(id) VALUES (?)", (conv_id,)
        )

    async def append_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        await self.db.execute(
            "INSERT INTO messages(conversation_id, seq, role, content, tool_calls, tool_call_id) "
            "VALUES (?, (SELECT COUNT(*) FROM messages WHERE conversation_id = ?) + 1, ?, ?, ?, ?)",
            (conv_id, conv_id, role, content, json.dumps(tool_calls) if tool_calls else None, tool_call_id),
        )
        await self.db.execute(
            "UPDATE conversations SET last_active_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
            "WHERE id = ?",
            (conv_id,),
        )
        await self.db.commit()

    async def history(self, conv_id: str) -> list[StoredMessage]:
        cursor = await self.db.execute(
            "SELECT role, content, tool_calls, tool_call_id FROM messages "
            "WHERE conversation_id = ? ORDER BY seq",
            (conv_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            StoredMessage(
                role=role,
                content=content,
                tool_calls=json.loads(tc) if tc else None,
                tool_call_id=tc_id,
            )
            for role, content, tc, tc_id in rows
        ]

    async def exists(self, conv_id: str) -> bool:
        cursor = await self.db.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conv_id,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row is not None

    async def delete(self, conv_id: str) -> bool:
        cursor = await self.db.execute(
            "DELETE FROM conversations WHERE id = ?", (conv_id,)
        )
        await self.db.commit()
        return cursor.rowcount > 0

    async def list_conversations(self) -> list[dict[str, Any]]:
        cursor = await self.db.execute(
            "SELECT c.id, c.created_at, c.last_active_at, "
            "(SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS messages "
            "FROM conversations c ORDER BY c.last_active_at DESC"
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            {
                "id": row[0],
                "created_at": row[1],
                "last_active_at": row[2],
                "message_count": row[3],
            }
            for row in rows
        ]


def build_context(history: list[StoredMessage], max_messages: int) -> list[dict[str, Any]]:
    """Assemble the message list for a stateless backend.

    Keeps at most ``max_messages`` messages, preserving system/developer
    messages and dropping tool results whose assistant call was trimmed.
    """
    system = [
        {"role": m.role, "content": m.content, **({"tool_calls": m.tool_calls} if m.tool_calls else {})}
        for m in history
        if m.role in ("system", "developer")
    ]
    rest = [
        {
            "role": m.role,
            "content": m.content,
            **({"tool_calls": m.tool_calls} if m.tool_calls else {}),
            **({"tool_call_id": m.tool_call_id} if m.tool_call_id else {}),
        }
        for m in history
        if m.role not in ("system", "developer")
    ]
    budget = max(1, max_messages - len(system))
    rest = rest[-budget:]
    # Keep results only when their assistant call survived trimming.
    known_calls: set[str] = set()
    valid = []
    for msg in rest:
        if msg["role"] == "tool" and msg.get("tool_call_id") not in known_calls:
            continue
        known_calls.update(tc.get("id") for tc in msg.get("tool_calls", []))
        valid.append(msg)
    return system + valid
