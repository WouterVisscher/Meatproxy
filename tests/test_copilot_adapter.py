"""Copilot adapter: session pooling, gauge, ephemeral session cleanup.

Uses a fake SDK (monkeypatched into copilot_adapter._copilot) so no real
Copilot runtime or subscription is needed.
"""

from __future__ import annotations

import asyncio

import pytest

import meatproxy.copilot_adapter as ca
from meatproxy.config import CopilotSpec
from meatproxy.metrics import conversations_active


# -- fake SDK ------------------------------------------------------------------


class FakeDeltaData:
    def __init__(self, text: str) -> None:
        self.delta_content = text


class FakeFinalData:
    def __init__(self, text: str) -> None:
        self.content = text


class FakeIdleData:
    mode = "idle"  # anything != AUTOPILOT settles the turn


class FakeAutopilotStoppedData:
    def __init__(self) -> None:
        self.request_id = None


class FakeErrorData:
    pass


class FakeSessionMode:
    AUTOPILOT = "autopilot"


class FakeEvent:
    def __init__(self, data) -> None:
        self.data = data
        self.agent_id = None


class FakeSession:
    def __init__(self) -> None:
        self.disconnected = False
        self._handlers: list = []

    def on(self, handler):
        self._handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return unsubscribe

    async def send(self, prompt: str) -> str:
        for handler in list(self._handlers):
            handler(FakeEvent(FakeDeltaData("hello ")))
            handler(FakeEvent(FakeDeltaData("world")))
            handler(FakeEvent(FakeIdleData()))
        return "req-1"

    async def disconnect(self) -> None:
        self.disconnected = True


class FakeClient:
    def __init__(self) -> None:
        self.created: list[FakeSession] = []
        self.stopped = False

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        self.stopped = True

    async def create_session(self, session_id=None, **kwargs) -> FakeSession:
        session = FakeSession()
        self.created.append(session)
        return session

    async def resume_session(self, session_id, **kwargs):
        raise RuntimeError("no persisted session")


def _install_fake_sdk(
    monkeypatch: pytest.MonkeyPatch, client: FakeClient | None = None
) -> FakeClient:
    client = client or FakeClient()
    monkeypatch.setattr(
        ca,
        "_copilot",
        {
            "CopilotClient": lambda **kwargs: client,
            "AssistantMessageData": FakeFinalData,
            "AssistantMessageDeltaData": FakeDeltaData,
            "SessionAutopilotStoppedData": FakeAutopilotStoppedData,
            "SessionErrorData": FakeErrorData,
            "SessionIdleData": FakeIdleData,
            "SessionMode": FakeSessionMode,
            "deny_tools": lambda request, invocation: None,
        },
    )
    return client


# -- tests ----------------------------------------------------------------------


async def test_ephemeral_session_disconnected(monkeypatch):
    """Stateless requests create an ephemeral session that must be released
    after the turn (no conversation id -> not pooled)."""
    client = _install_fake_sdk(monkeypatch)
    conversations_active.set(0)
    adapter = ca.CopilotAdapter(CopilotSpec())

    events = [ev async for ev in adapter.stream("hi")]

    assert "".join(ev.text for ev in events) == "hello world"
    assert len(client.created) == 1
    assert client.created[0].disconnected is True
    assert adapter.active_sessions() == 0
    assert conversations_active._value.get() == 0


async def test_pooled_session_gauge_and_stop(monkeypatch):
    client = _install_fake_sdk(monkeypatch)
    conversations_active.set(0)
    adapter = ca.CopilotAdapter(CopilotSpec())

    events = [ev async for ev in adapter.stream("hi", conversation_id="conv-1")]

    assert "".join(ev.text for ev in events) == "hello world"
    assert adapter.active_sessions() == 1
    assert conversations_active._value.get() == 1
    assert client.created[0].disconnected is False

    await adapter.stop()
    assert conversations_active._value.get() == 0
    assert client.created[0].disconnected is True
    assert client.stopped is True


async def test_pooled_session_reused_across_turns(monkeypatch):
    client = _install_fake_sdk(monkeypatch)
    conversations_active.set(0)
    adapter = ca.CopilotAdapter(CopilotSpec())

    [ev async for ev in adapter.stream("hi", conversation_id="conv-1")]
    [ev async for ev in adapter.stream("again", conversation_id="conv-1")]

    assert len(client.created) == 1  # second turn reused the pooled session
    assert conversations_active._value.get() == 1
    await adapter.stop()


class LateIdleSession(FakeSession):
    """The real runtime's interleaving: the delta is delivered before the
    idle event, so by the time the turn ends the consumer has already
    consumed the delta and is blocked on the next queue.get()."""

    async def send(self, prompt: str) -> str:
        for handler in list(self._handlers):
            handler(FakeEvent(FakeDeltaData("hi")))
        await asyncio.sleep(0)  # let the consumer consume the delta and block
        for handler in list(self._handlers):
            handler(FakeEvent(FakeIdleData()))
        return "req-1"


class LateIdleClient(FakeClient):
    async def create_session(self, session_id=None, **kwargs) -> LateIdleSession:
        session = LateIdleSession()
        self.created.append(session)
        return session


async def test_turn_end_wakes_blocked_consumer(monkeypatch):
    """End-of-turn events enqueue nothing; without the _TURN_DONE sentinel
    the consumer stayed blocked on queue.get() and the stream stalled until
    the 600s timeout (observed against the live runtime)."""
    client = _install_fake_sdk(monkeypatch, LateIdleClient())
    adapter = ca.CopilotAdapter(CopilotSpec())

    events = [ev async for ev in adapter.stream("hi")]

    assert "".join(ev.text for ev in events) == "hi"
    assert client.created[0].disconnected is True
