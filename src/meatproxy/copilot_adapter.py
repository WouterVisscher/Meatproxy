"""Adapter for the GitHub Copilot backend, built on github-copilot-sdk.

Copilot is an *agent runtime*, not a chat API, so this adapter:

* runs one shared :class:`CopilotClient` (one CLI runtime process) for the
  whole app;
* maps conversation ids to persistent Copilot sessions (``session_id`` ==
  conversation id). Live sessions are pooled and resumed across evictions
  via ``resume_session`` — session state persists in ``base_directory``,
  which should be a durable volume.
* maps stateless requests to ephemeral sessions (disconnected after the turn).
* disables built-in tools (``available_tools=[]``): meatproxy is a proxy,
  not an agent sandbox, so Copilot answers with text only and the
  request's ``tools`` are deliberately dropped on this path.

Model selection uses Copilot's own "auto" mode by default
(``model="auto"``), optionally nudged with an auto tier preference.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator

from .config import CopilotSpec
from .metrics import conversations_active
from .openai_client import FirstChunkTimeout, StreamEvent, TextDelta, UpstreamError

log = logging.getLogger(__name__)

# Lazy imports: the SDK is only needed when a copilot model is configured,
# and importing it is expensive.
_copilot = None


def _sdk():
    global _copilot
    if _copilot is None:
        import copilot  # noqa: F401  (marks the SDK as loaded)
        from copilot import (
            CopilotClient,
            PermissionRequest,
            PermissionRequestResult,
        )
        from copilot.rpc import PermissionDecisionReject
        from copilot.session_events import (
            AssistantMessageData,
            AssistantMessageDeltaData,
            SessionAutopilotStoppedData,
            SessionErrorData,
            SessionIdleData,
            SessionMode,
        )

        def deny_tools(request: PermissionRequest, invocation: Any) -> PermissionRequestResult:
            # Defensive: with available_tools=[] no tools exist, but if
            # anything still asks for permission, the answer is "no".
            return PermissionDecisionReject(
                feedback="Tool use is disabled in meatproxy; answer with text only."
            )

        _copilot = {
            "CopilotClient": CopilotClient,
            "AssistantMessageData": AssistantMessageData,
            "AssistantMessageDeltaData": AssistantMessageDeltaData,
            "SessionAutopilotStoppedData": SessionAutopilotStoppedData,
            "SessionErrorData": SessionErrorData,
            "SessionIdleData": SessionIdleData,
            "SessionMode": SessionMode,
            "deny_tools": deny_tools,
        }
    return _copilot


@dataclass
class _PooledSession:
    session: Any
    last_active: float


class CopilotAdapter:
    """Drives Copilot sessions behind a simple text-stream interface."""

    def __init__(self, spec: CopilotSpec, default_timeout_s: float = 60.0) -> None:
        self.spec = spec
        self.default_timeout_s = default_timeout_s
        self._idle_timeout = 900.0
        self._client: Any | None = None
        self._sessions: dict[str, _PooledSession] = {}
        self._client_lock = asyncio.Lock()
        self._evictor_task: asyncio.Task | None = None

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        await self._ensure_client()
        if self._evictor_task is None:
            self._evictor_task = asyncio.create_task(self._evict_idle())

    async def stop(self) -> None:
        if self._evictor_task is not None:
            self._evictor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._evictor_task
            self._evictor_task = None
        if self._client is not None:
            for entry in list(self._sessions.values()):
                with contextlib.suppress(Exception):
                    await entry.session.disconnect()
            self._sessions.clear()
            self._sync_gauge()
            with contextlib.suppress(Exception):
                await self._client.stop()
            self._client = None

    async def _ensure_client(self) -> Any:
        async with self._client_lock:
            if self._client is not None:
                return self._client
            sdk = _sdk()
            kwargs: dict[str, Any] = {
                "log_level": "info",
                "env": {"COPILOT_TELEMETRY_DISABLED": "true"},
            }
            if self.spec.base_directory:
                kwargs["base_directory"] = self.spec.base_directory
            token = None
            if self.spec.token_env:
                token = os.environ.get(self.spec.token_env)
            if token:
                kwargs["github_token"] = token
            else:
                # use the credentials from `copilot login`
                kwargs["use_logged_in_user"] = True
            client = sdk["CopilotClient"](**kwargs)
            try:
                await client.start()
            except Exception:
                await client.stop()
                raise
            self._client = client
            return client

    # -- session pool ------------------------------------------------------

    def active_sessions(self) -> int:
        return len(self._sessions)

    def _sync_gauge(self) -> None:
        conversations_active.set(len(self._sessions))

    async def _evict_idle(self) -> None:
        while True:
            await asyncio.sleep(30)
            now = asyncio.get_event_loop().time()
            cutoff = self._idle_timeout
            evicted = False
            for conv_id, entry in list(self._sessions.items()):
                if now - entry.last_active > cutoff:
                    self._sessions.pop(conv_id, None)
                    with contextlib.suppress(Exception):
                        await entry.session.disconnect()
                    log.info("evicted idle copilot session %s", conv_id)
                    evicted = True
            if evicted:
                self._sync_gauge()

    def set_idle_timeout(self, seconds: float) -> None:
        self._idle_timeout = seconds

    async def _session_for(self, conversation_id: str | None, model: str) -> tuple[Any, bool]:
        """Return (session, created_fresh). Pooled sessions are reused."""
        client = await self._ensure_client()
        sdk = _sdk()
        now = asyncio.get_event_loop().time()
        common = dict(
            on_permission_request=sdk["deny_tools"],
            available_tools=[],
            streaming=True,
            interactive_user=True,
        )
        if conversation_id and conversation_id in self._sessions:
            self._sessions[conversation_id].last_active = now
            return self._sessions[conversation_id].session, False

        if conversation_id:
            # Prefer resuming a persistent session (state in base_directory).
            try:
                session = await client.resume_session(
                    conversation_id, model=model, capi=self._capi(), **common
                )
                self._sessions[conversation_id] = _PooledSession(session, now)
                self._sync_gauge()
                return session, False
            except Exception:
                pass  # fall through to a fresh session with the same id

        session = await client.create_session(
            session_id=conversation_id or uuid.uuid4().hex,
            model=model,
            capi=self._capi(),
            **common,
        )
        if conversation_id:
            self._sessions[conversation_id] = _PooledSession(session, now)
            self._sync_gauge()
        return session, True

    def _capi(self) -> dict | None:
        if self.spec.auto_tier:
            return {"auto_tier": self.spec.auto_tier}
        return None

    # -- streaming ----------------------------------------------------------

    async def stream(
        self,
        prompt: str,
        conversation_id: str | None = None,
        model: str = "auto",
        timeout_s: float | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield TextDelta events until the Copilot turn is finished.

        Raises UpstreamError when the session errors out, or FirstChunkTimeout
        when no text arrives within the budget.
        """
        timeout = timeout_s or self.default_timeout_s
        session, _ = await self._session_for(conversation_id, model)
        sdk = _sdk()

        queue: asyncio.Queue[StreamEvent | Exception] = asyncio.Queue()
        done = asyncio.Event()
        sent: dict[str, Any] = {}  # filled with the send's request_id
        state = {"delta": False, "final": ""}

        def on_event(event: Any) -> None:
            if getattr(event, "agent_id", None):
                return  # sub-agent noise
            data = event.data
            if isinstance(data, sdk["AssistantMessageDeltaData"]):
                text = data.delta_content or ""
                if text:
                    state["delta"] = True
                    queue.put_nowait(TextDelta(text=text))
            elif isinstance(data, sdk["AssistantMessageData"]):
                state["final"] = data.content or ""
            elif isinstance(data, sdk["SessionErrorData"]):
                queue.put_nowait(UpstreamError(f"copilot session error: {data!r}"))
                done.set()
            elif isinstance(data, sdk["SessionIdleData"]) and data.mode != sdk["SessionMode"].AUTOPILOT:
                done.set()
            elif isinstance(data, sdk["SessionAutopilotStoppedData"]):
                # Only settle on our own request's stop event.
                if sent.get("id") is None or getattr(data, "request_id", None) == sent["id"]:
                    done.set()

        unsubscribe = session.on(on_event)
        try:
            async def do_send() -> None:
                try:
                    sent["id"] = await session.send(prompt)
                except Exception as e:  # session dead, runtime error, ...
                    if not queue.empty() or done.is_set():
                        return
                    queue.put_nowait(UpstreamError(f"copilot send failed: {e}"))
                    done.set()

            send_task = asyncio.create_task(do_send())
            try:
                try:
                    first = await asyncio.wait_for(queue.get(), timeout=timeout)
                except asyncio.TimeoutError as e:
                    raise FirstChunkTimeout(f"no copilot response within {timeout}s") from e
                if isinstance(first, Exception):
                    raise first
                yield first
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=600)
                    except asyncio.TimeoutError:
                        raise UpstreamError("copilot stream stalled (600s)") from None
                    if isinstance(item, Exception):
                        raise item
                    yield item
                    if done.is_set() and queue.empty():
                        break
                if not state["delta"] and state["final"]:
                    # No deltas streamed (should be rare); emit the whole
                    # final content so the client still gets an answer.
                    yield TextDelta(text=state["final"])
            finally:
                send_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await send_task
        finally:
            with contextlib.suppress(Exception):
                unsubscribe()
            if conversation_id:
                entry = self._sessions.get(conversation_id)
                if entry is not None:
                    entry.last_active = asyncio.get_event_loop().time()
            else:
                # stateless request: the session was ephemeral, release it
                with contextlib.suppress(Exception):
                    await session.disconnect()
