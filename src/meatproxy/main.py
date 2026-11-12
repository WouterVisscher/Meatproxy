"""meatproxy application: FastAPI app, OpenAI-compatible endpoints, wiring.

Endpoints
---------
POST /v1/chat/completions   chat (SSE when stream: true; JSON by default, like OpenAI)
GET  /v1/models             virtual model list (auto + configured models)
GET  /v1/conversations      list stored conversations
DELETE /v1/conversations/{id}
GET  /health                liveness + backend reachability
GET  /metrics               Prometheus metrics

Conversation handling: send an ``X-Conversation-Id`` header to keep
conversation state in meatproxy (SQLite). With a conversation id the
client only needs to send the newest user message; the router rebuilds
context from the store.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import __version__
from .auth import Auth
from .backends import BackendPool, TransportFactory, build_registry
from .config import Config, ConfigError, ConfigManager
from .conversations import ConversationStore, build_context
from .copilot_adapter import CopilotAdapter
from .logsetup import emit, setup_logging
from .metrics import (
    conversations_active,
    render_metrics,
    requests_total,
)
from .openai_client import (
    PASSTHROUGH_PARAMS,
    FirstChunkTimeout,
    RawChunk,
    UpstreamError,
    chunk_content,
    chunk_usage,
    now_ts,
)
from .router import AUTO_MODEL, Router, RoutePlan
from .triage import Triage

log = logging.getLogger(__name__)

CONVERSATION_ID_HEADER = "x-conversation-id"


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


class AppState:
    """All long-lived state, swapped under lock on config hot-reload."""

    def __init__(
        self,
        manager: ConfigManager,
        store: ConversationStore,
        pool: BackendPool,
        copilot: CopilotAdapter,
    ) -> None:
        self.manager = manager
        self.store = store
        self.pool = pool
        self.copilot = copilot
        self.auth = Auth()
        self.triage: Triage | None = None
        cfg = self.manager.get()
        self.pool.set_registry(build_registry(cfg, self.pool.transport_factory))
        self.router = Router(
            get_config=manager.get,
            get_registry=lambda: self.pool.current,
            copilot=copilot,
            get_triage=lambda: self.triage,
        )
        self._locks: dict[str, asyncio.Lock] = {}
        self._reload_lock = asyncio.Lock()
        self._rejected_mtime: float | None = None
        self._apply_config(cfg)

    def _apply_config(self, cfg: Config) -> None:
        self.triage = build_triage(cfg, self.pool)
        self.copilot.set_idle_timeout(cfg.conversations.copilot_idle_timeout_s)
        conversations_active.set(self.copilot.active_sessions())

    def conversation_lock(self, conv_id: str) -> asyncio.Lock:
        return self._locks.setdefault(conv_id, asyncio.Lock())

    async def refresh(self) -> None:
        """Hot-reload the config file if it changed.

        An invalid new file is rejected: the previous config keeps serving
        and the rejection is logged once per file version.
        """
        async with self._reload_lock:
            try:
                changed = self.manager.refresh()
            except ConfigError as e:
                try:
                    bad_mtime = self.manager.path.stat().st_mtime
                except OSError:
                    bad_mtime = None
                if bad_mtime != self._rejected_mtime:
                    self._rejected_mtime = bad_mtime
                    emit(
                        log,
                        logging.WARNING,
                        "config reload rejected, keeping previous config",
                        error=str(e),
                    )
                return
            if not changed:
                return
            cfg = self.manager.get()
            old_registry = self.pool.current
            await self.pool.rebuild(cfg)
            self._apply_config(cfg)
            await old_registry.aclose_all()
            self._rejected_mtime = None
            emit(log, logging.INFO, "config reloaded", models=sorted(cfg.models))


def build_triage(cfg: Config, pool: BackendPool) -> Triage | None:
    if not cfg.triage:
        return None
    spec = cfg.triage
    registry = pool.current
    return Triage(
        spec=spec,
        backend=registry.backend_for(spec.model),
        upstream_model=registry.upstream_for(spec.model),
    )


# ---------------------------------------------------------------------------
# request helpers
# ---------------------------------------------------------------------------


def message_text(content: Any) -> str:
    """Flatten message content (string or multimodal parts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )
    return "" if content is None else str(content)


def _last_user_message(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg
    raise HTTPException(status_code=400, detail="no user message in request")


def _extra_params(body: dict[str, Any]) -> dict[str, Any]:
    return {k: body[k] for k in PASSTHROUGH_PARAMS if k in body}


def _validate_messages(messages: Any) -> None:
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="'messages' must be a non-empty list")
    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg:
            raise HTTPException(
                status_code=400, detail="each message needs 'role' and 'content'"
            )
        if "content" not in msg and not (msg["role"] == "assistant" and msg.get("tool_calls")):
            raise HTTPException(status_code=400, detail="each message needs 'role' and 'content'")
        if msg["role"] == "tool" and (
            not isinstance(msg.get("tool_call_id"), str) or not msg["tool_call_id"]
        ):
            raise HTTPException(status_code=400, detail="tool messages need 'tool_call_id'")


def _validate_tool_params(body: dict[str, Any]) -> None:
    tools = body.get("tools") or []
    if not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools):
        raise HTTPException(status_code=400, detail="'tools' must be a list of tool definitions")
    choice = body.get("tool_choice", "auto")
    if choice == "required" and not tools:
        raise HTTPException(status_code=400, detail="tool_choice requires an available function tool")
    if isinstance(choice, dict):
        function = choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if not name or not any(
            t.get("type") == "function" and isinstance(t.get("function"), dict)
            and t["function"].get("name") == name for t in tools
        ):
            raise HTTPException(status_code=400, detail="named tool_choice must match an available function tool")
    if "parallel_tool_calls" in body and not isinstance(body["parallel_tool_calls"], bool):
        raise HTTPException(status_code=400, detail="'parallel_tool_calls' must be a boolean")


def merge_tool_calls(chunks: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Reassemble OpenAI tool-call deltas spread over stream chunks."""
    calls: dict[int, dict[str, Any]] = {}
    for chunk in chunks:
        try:
            choices = chunk["choices"]
        except (KeyError, TypeError):
            continue
        if not isinstance(choices, list):
            continue
        for choice in choices:
            delta = (choice or {}).get("delta") or {}
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0) or 0
                slot = calls.setdefault(
                    idx, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
    if not calls:
        return None
    return [calls[k] for k in sorted(calls)]


def sse(obj: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n".encode()


def sse_error(message: str, etype: str = "upstream_error") -> bytes:
    return sse({"error": {"message": message, "type": etype, "code": None}})


# ---------------------------------------------------------------------------
# response generators
# ---------------------------------------------------------------------------


async def _run_and_stream(
    state: AppState,
    plan: RoutePlan,
    messages: list[dict[str, Any]],
    extra: dict[str, Any],
    conversation_id: str | None,
) -> AsyncIterator[bytes]:
    """Yield SSE bytes for one routed request; store results; emit metrics."""
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = now_ts()
    content_parts: list[str] = []
    tool_chunks: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None
    served: str | None = None
    finished = False  # upstream already sent a finish_reason chunk
    start = time.monotonic()

    try:
        async for event in state.router.run(
            plan, messages, extra, conversation_id
        ):
            if isinstance(event, RawChunk):
                data = event.data
                choices = data.get("choices")
                if isinstance(choices, list):
                    tool_chunks.append(data)
                    for choice in choices:
                        if (choice or {}).get("finish_reason"):
                            finished = True
                content_parts.extend(chunk_content(data))
                usage = chunk_usage(data) or usage
                served = data.get("model") or served
                yield sse(
                    {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": served or AUTO_MODEL,
                        "choices": choices,
                        **({"usage": usage} if usage else {}),
                    }
                )
            else:
                content_parts.append(event.text)
                served = served or "copilot"
                yield sse(
                    {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": served,
                        "choices": [
                            {"index": 0, "delta": {"content": event.text}, "finish_reason": None}
                        ],
                    }
                )

        # terminal chunk (only when the upstream stream did not close it
        # itself; Copilot text deltas never do)
        if not finished:
            terminal: dict[str, Any] = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": served or AUTO_MODEL,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            if usage:
                terminal["usage"] = usage
            yield sse(terminal)
        yield b"data: [DONE]\n\n"

        if conversation_id:
            tool_calls = merge_tool_calls(tool_chunks)
            await state.store.append_message(
                conversation_id, "assistant", "".join(content_parts), tool_calls
            )
        emit(
            log,
            logging.INFO,
            "completed request",
            model=served,
            tier=plan.tier,
            source=plan.source,
            conversation=conversation_id,
            latency_s=round(time.monotonic() - start, 3),
        )
    except (UpstreamError, FirstChunkTimeout) as e:
        yield sse_error(str(e))
        emit(log, logging.WARNING, "request failed", error=str(e), tier=plan.tier)
    except Exception as e:
        yield sse_error(f"internal error: {e}", etype="internal_error")
        log.exception("internal error while streaming")
    finally:
        model_label = served or plan.models[0].name
        requests_total.labels(
            model=model_label, tier=plan.tier, source=plan.source
        ).inc()


async def _collect(
    state: AppState,
    plan: RoutePlan,
    messages: list[dict[str, Any]],
    extra: dict[str, Any],
    conversation_id: str | None,
) -> dict[str, Any]:
    """Run the route non-streamingly and build the OpenAI JSON response."""
    content_parts: list[str] = []
    tool_chunks: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None
    served: str | None = None
    start = time.monotonic()
    finish_reason = "stop"
    async for event in state.router.run(plan, messages, extra, conversation_id):
        if isinstance(event, RawChunk):
            data = event.data
            choices = data.get("choices")
            if isinstance(choices, list):
                tool_chunks.append(data)
                for choice in choices:
                    finish_reason = choice.get("finish_reason") or finish_reason
            content_parts.extend(chunk_content(data))
            usage = chunk_usage(data) or usage
            served = data.get("model") or served
        else:
            content_parts.append(event.text)
            served = served or "copilot"

    if conversation_id:
        tool_calls = merge_tool_calls(tool_chunks)
        await state.store.append_message(
            conversation_id, "assistant", "".join(content_parts), tool_calls
        )
    emit(
        log,
        logging.INFO,
        "completed request (non-stream)",
        model=served,
        tier=plan.tier,
        source=plan.source,
        conversation=conversation_id,
        latency_s=round(time.monotonic() - start, 3),
    )
    requests_total.labels(
        model=served or plan.models[0].name, tier=plan.tier, source=plan.source
    ).inc()

    message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
    tool_calls = merge_tool_calls(tool_chunks)
    if tool_calls:
        message["tool_calls"] = tool_calls
        if finish_reason == "stop":
            finish_reason = "tool_calls"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now_ts(),
        "model": served or AUTO_MODEL,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage
        or {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
    }


# ---------------------------------------------------------------------------
# app factory
# ---------------------------------------------------------------------------


def create_app(
    config_path: str | os.PathLike,
    transport_factory: TransportFactory | None = None,
) -> FastAPI:
    try:
        manager = ConfigManager(config_path)
    except ConfigError as e:
        log.error("config error: %s", e)
        raise SystemExit(2) from e
    cfg = manager.get()

    store = ConversationStore(cfg.conversations.db_path)
    pool = BackendPool(transport_factory)
    copilot = CopilotAdapter(cfg.copilot, default_timeout_s=cfg.copilot.timeout_s)

    state = AppState(manager, store, pool, copilot)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await store.open()
        if any(spec.kind == "copilot" for spec in cfg.models.values()):
            try:
                await copilot.start()
                status = await copilot.auth_status()
                if status:
                    if status["authenticated"]:
                        emit(
                            log,
                            logging.INFO,
                            "copilot: logged in as",
                            login=status["login"],
                            host=status["host"],
                        )
                    else:
                        emit(
                            log,
                            logging.INFO,
                            "copilot: not logged in (run 'meatproxy copilot login' to enable copilot models)",
                        )
            except Exception as e:
                emit(
                    log,
                    logging.WARNING,
                    "copilot runtime did not start (will retry lazily)",
                    error=str(e),
                )
        emit(
            log,
            logging.INFO,
            "meatproxy started",
            version=__version__,
            models=sorted(cfg.models),
            triage=bool(cfg.triage),
        )
        yield
        await copilot.stop()
        await store.aclose()
        await pool.aclose_all()

    app = FastAPI(title="meatproxy", version=__version__, lifespan=lifespan)
    app.state.state = state

    # -- endpoints ------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        state.auth.check(request)
        await state.refresh()

        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid JSON body") from None
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="invalid JSON body")

        model = str(body.get("model") or AUTO_MODEL)
        messages = body.get("messages")
        _validate_messages(messages)
        _validate_tool_params(body)
        stream = bool(body.get("stream", False))
        extra = _extra_params(body)
        cfg = state.manager.get()

        if model != AUTO_MODEL and model not in cfg.models:
            raise HTTPException(status_code=404, detail=f"unknown model {model!r}")

        conv_id = (request.headers.get(CONVERSATION_ID_HEADER) or "").strip() or None
        if conv_id is not None and len(conv_id) > 128:
            raise HTTPException(status_code=400, detail="conversation id too long")
        if conv_id and messages[-1]["role"] == "tool":
            history = await state.store.history(conv_id)
            if not history:
                raise HTTPException(status_code=400, detail="tool results need an existing conversation")
            last_assistant = next((m for m in reversed(history) if m.role == "assistant"), None)
            expected = {tc["id"] for tc in (last_assistant.tool_calls or [])} if last_assistant else set()
            results = []
            for msg in reversed(messages):
                if msg["role"] != "tool":
                    break
                results.append(msg["tool_call_id"])
            if not expected or len(results) != len(set(results)) or set(results) != expected:
                raise HTTPException(status_code=400, detail="tool results must match the pending assistant tool calls")
            prompt = next((m.content for m in reversed(history) if m.role == "user"), "")
        else:
            last_user = _last_user_message(messages)
            prompt = message_text(last_user.get("content"))

        lock = state.conversation_lock(conv_id) if conv_id else None
        if lock is not None and lock.locked():
            raise HTTPException(
                status_code=409,
                detail="conversation already has an in-flight request",
            )

        if conv_id is None:
            plan = await state.router.plan(model, prompt, cfg)
            if stream:
                return StreamingResponse(
                    _run_and_stream(state, plan, messages, extra, None),
                    media_type="text/event-stream",
                )
            try:
                result = await _collect(state, plan, messages, extra, None)
            except UpstreamError as e:
                return JSONResponse(
                    {"error": {"message": str(e), "type": "upstream_error", "code": None}},
                    status_code=502,
                )
            return JSONResponse(result)

        # --- conversation request: router owns the history ---------------------

        async def _prep() -> tuple[RoutePlan, list[dict[str, Any]]]:
            await state.store.ensure_conversation(conv_id)
            if messages[-1]["role"] == "tool":
                # The proxy already stored the assistant's tool calls. Append
                # only this batch of results, not the client's repeated history.
                batch = []
                for msg in reversed(messages):
                    if msg["role"] != "tool":
                        break
                    batch.append(msg)
                for msg in reversed(batch):
                    await state.store.append_message(
                        conv_id, "tool", message_text(msg.get("content")),
                        tool_call_id=msg["tool_call_id"],
                    )
            else:
                history = await state.store.history(conv_id)
                if not history:
                    for msg in messages:
                        if msg["role"] in ("system", "developer"):
                            await state.store.append_message(
                                conv_id, msg["role"], message_text(msg.get("content"))
                            )
                await state.store.append_message(conv_id, "user", prompt)
            history = await state.store.history(conv_id)
            context = build_context(history, cfg.conversations.max_context_messages)
            plan = await state.router.plan(model, prompt, cfg)
            return plan, context

        if stream:

            async def conv_stream() -> AsyncIterator[bytes]:
                # the lock was free when the response started; re-check in case
                # a racing request grabbed it before the stream began
                if lock.locked():
                    yield sse_error(
                        "conversation already has an in-flight request",
                        etype="conflict",
                    )
                    return
                async with lock:
                    plan, context = await _prep()
                    async for chunk in _run_and_stream(
                        state, plan, context, extra, conv_id
                    ):
                        yield chunk

            return StreamingResponse(conv_stream(), media_type="text/event-stream")

        async with lock:
            plan, context = await _prep()
            try:
                result = await _collect(state, plan, context, extra, conv_id)
            except UpstreamError as e:
                return JSONResponse(
                    {"error": {"message": str(e), "type": "upstream_error", "code": None}},
                    status_code=502,
                )
            return JSONResponse(result)

    @app.get("/v1/models")
    async def list_models(request: Request) -> Any:
        state.auth.check(request)
        await state.refresh()
        cfg = state.manager.get()
        data = [
            {
                "id": AUTO_MODEL,
                "object": "model",
                "created": now_ts(),
                "owned_by": "meatproxy-router",
            }
        ]
        data.extend(
            {
                "id": name,
                "object": "model",
                "created": now_ts(),
                "owned_by": spec.tier,
            }
            for name, spec in sorted(cfg.models.items())
        )
        return {"object": "list", "data": data}

    @app.get("/v1/conversations")
    async def list_conversations(request: Request) -> Any:
        state.auth.check(request)
        return {"object": "list", "data": await state.store.list_conversations()}

    @app.delete("/v1/conversations/{conv_id}")
    async def delete_conversation(conv_id: str, request: Request) -> Any:
        state.auth.check(request)
        lock = state.conversation_lock(conv_id)
        if lock.locked():
            raise HTTPException(
                status_code=409, detail="conversation has an in-flight request"
            )
        async with lock:
            deleted = await state.store.delete(conv_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="conversation not found")
        state._locks.pop(conv_id, None)
        return {"deleted": conv_id}

    @app.get("/health")
    async def health() -> Any:
        cfg = state.manager.get()
        status: dict[str, Any] = {"status": "ok", "models": {}}
        registry = state.pool.current
        for name, spec in sorted(cfg.models.items()):
            if spec.kind == "copilot":
                status["models"][name] = "configured"
            else:
                status["models"][name] = (
                    "up" if await registry.backend_for(name).reachable() else "down"
                )
        status["copilot_sessions"] = state.copilot.active_sessions()
        status["version"] = __version__
        return status

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(render_metrics(), media_type="text/plain; charset=utf-8")

    @app.get("/")
    async def root() -> Any:
        return {"name": "meatproxy", "version": __version__}

    return app


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def serve(
    config_path: str | os.PathLike,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """Run the proxy from an explicit config path (and optional overrides)."""
    config_path = os.path.expanduser(str(config_path))
    app = create_app(config_path)
    cfg: Config = app.state.state.manager.get()
    host = host or cfg.host
    port = port if port is not None else int(os.environ.get("MEATPROXY_PORT", cfg.port))
    emit(log, logging.INFO, "starting", host=host, port=port, config=config_path)
    uvicorn.run(app, host=host, port=port, log_config=None)


def run() -> None:
    """Legacy entrypoint: env-var driven (MEATPROXY_CONFIG / MEATPROXY_PORT)."""
    setup_logging(os.environ.get("MEATPROXY_LOG_LEVEL", "info"))
    config_path = os.environ.get("MEATPROXY_CONFIG", "meatproxy.yaml")
    serve(config_path)
