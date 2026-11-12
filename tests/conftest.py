"""Shared fixtures: fake OpenAI upstreams (mock transports) + test app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from meatproxy.main import create_app

API_KEY = "test-key"


class FakeUpstream:
    """Stateful fake OpenAI upstream.

    * triage calls (max_tokens == 96) answer "heavy" when the last message
      contains a configured trigger word, else "light";
    * stream calls yield two content chunks + usage + [DONE];
    * when ``fail`` is True, chat calls answer 503;
    * records every request body.
    """

    def __init__(self, name: str, triage_model: str | None = None, trigger: str = "refactor") -> None:
        self.name = name
        self.triage_model = triage_model
        self.trigger = trigger
        self.fail = False
        self.fail_models: set[str] = set()
        self.triage_delay = 0.0
        self.stream_error: str | None = None  # answer streams with an SSE error payload
        self.requests: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] | None = None

    async def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/models"):
            return httpx.Response(
                200, json={"data": [{"id": self.name}]}
            )
        body = json.loads(request.content)
        self.requests.append(body)

        if body.get("max_tokens") == 96 and body["model"] == (self.triage_model or ""):
            if self.triage_delay:
                import asyncio

                await asyncio.sleep(self.triage_delay)
            if self.fail:
                return httpx.Response(500, text="triage down")
            text = body["messages"][-1]["content"]
            tier = "heavy" if self.trigger in text else "light"
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": tier}}]},
            )

        model = body["model"]
        if self.fail or model in self.fail_models:
            return httpx.Response(503, text="upstream boom")

        if body.get("stream") and self.stream_error:
            # Some servers (e.g. Lemonade NPU) deliver errors as an SSE
            # payload inside an HTTP 200 instead of using an error status.
            payload = json.dumps(
                {"error": {"message": self.stream_error, "type": "inference_error"}}
            )
            return httpx.Response(
                200,
                text=f"data: {payload}\n\ndata: [DONE]\n\n",
                headers={"Content-Type": "text/event-stream"},
            )

        if body.get("stream"):
            if self.tool_calls and body.get("tools") and body["messages"][-1]["role"] != "tool":
                chunks = [
                    {"model": model, "choices": [{"index": 0, "delta": {
                        "tool_calls": [dict(call, index=i) for i, call in enumerate(self.tool_calls)]
                    }, "finish_reason": None}]},
                    {"model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
                ]
                text = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                return httpx.Response(200, text=text, headers={"Content-Type": "text/event-stream"})
            chunks = [
                {
                    "id": "c-1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant", "content": f"hello from {model} "}, "finish_reason": None}
                    ],
                },
                {
                    "id": "c-1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": "world"}, "finish_reason": None}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                },
            ]
            sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(
                200, text=sse, headers={"Content-Type": "text/event-stream"}
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": f"hello from {model} world"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )


TEST_CONFIG = """
models:
  npu-small: {tier: light, endpoint: http://npu/v1, model: fastflow-small}
  onprem: {tier: mid, endpoint: http://gpu/v1, model: qwen}
  copilot: {tier: heavy, kind: copilot}

routing:
  triage:
    model: npu-small
    timeout_s: 2
    on_failure: onprem
    cache_ttl_s: 300
  chains:
    light: [npu-small, onprem]
    mid: [onprem]
    heavy: [onprem]

conversations:
  db_path: __DB__
  max_context_messages: 40
  copilot_idle_timeout_s: 900

copilot:
  base_directory: __COPILOT__
  timeout_s: 60
"""


@pytest.fixture
def fakes() -> dict[str, FakeUpstream]:
    return {
        "npu": FakeUpstream("fastflow-small", triage_model="fastflow-small"),
        "gpu": FakeUpstream("qwen"),
    }


@pytest.fixture
def config_path(tmp_path: Path, fakes: dict[str, FakeUpstream]) -> Path:
    path = tmp_path / "meatproxy.yaml"
    path.write_text(
        TEST_CONFIG.replace("__DB__", str(tmp_path / "conv.db")).replace(
            "__COPILOT__", str(tmp_path / "copilot")
        )
    )
    return path


@pytest.fixture
def app(config_path: Path, fakes: dict[str, FakeUpstream], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEATPROXY_API_KEY", API_KEY)
    transport_factory = lambda endpoint: httpx.MockTransport(  # noqa: E731
        fakes[endpoint.split("/")[2]].handler
    )
    return create_app(config_path, transport_factory=transport_factory)


@pytest.fixture
async def client(app) -> httpx.AsyncClient:
    state = app.state.state
    await state.store.open()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await state.store.aclose()


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}


def parse_sse(text: str) -> list[dict[str, Any]]:
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            continue
        out.append(json.loads(payload))
    return out
