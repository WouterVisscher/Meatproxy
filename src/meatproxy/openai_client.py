"""Async OpenAI-compatible client used for local (NPU), on-prem and triage calls.

Streamed upstream chunks are passed through almost verbatim: each SSE chunk
is parsed and re-emitted so the router can rewrite the ``model`` field (the
client sees the virtual model name, not the upstream one). Tool-call chunks
pass through unchanged, so function calling works transparently.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

log = logging.getLogger(__name__)

# Parameters we pass straight through from the request to the upstream call.
PASSTHROUGH_PARAMS = (
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "seed",
    "n",
    "frequency_penalty",
    "presence_penalty",
    "tools",
    "tool_choice",
    "response_format",
    "reasoning_effort",
)


class UpstreamError(Exception):
    """An upstream model call failed (network, HTTP error, timeout...)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FirstChunkTimeout(UpstreamError):
    """No first token within the model's timeout budget."""


def _error_message(error: Any) -> str:
    """Extract a human-readable message from an upstream ``error`` field."""
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error)


@dataclass
class RawChunk:
    """One upstream OpenAI chat.completion.chunk, model field already rewritten."""

    data: dict[str, Any]


@dataclass
class TextDelta:
    """A plain text delta (used by the Copilot adapter)."""

    text: str


# What backend streams yield: either a passthrough chunk or a text delta.
StreamEvent = RawChunk | TextDelta


class OpenAIBackend:
    """Client for one OpenAI-compatible base endpoint (e.g. Lemonade, vLLM)."""

    def __init__(
        self,
        endpoint: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        verify: bool | str = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = endpoint.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            verify=verify,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _body(
        self,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        for key in PASSTHROUGH_PARAMS:
            if key in extra:
                body[key] = extra[key]
        return body

    async def stream_chat(
        self, model: str, messages: list[dict[str, Any]], extra: dict[str, Any]
    ) -> AsyncIterator[RawChunk]:
        """Yields upstream chunks (SSE ``data:`` payloads) for a streaming call."""
        body = self._body(model, messages, stream=True, extra=extra)
        async with self._client.stream("POST", "/chat/completions", json=body) as response:
            if response.status_code >= 400:
                detail = (await response.aread()).decode("utf-8", "replace")[:500]
                raise UpstreamError(
                    f"upstream {self.base_url} returned {response.status_code}: {detail}",
                    status_code=response.status_code,
                )
            async for line in response.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    return
                chunk = json.loads(payload)
                # Some servers (e.g. Lemonade) answer errors as an SSE payload
                # inside an HTTP 200: data: {"error": {...}}. Treat that as a
                # failure so the router can fail over instead of passing a
                # choiceless chunk to the client.
                if isinstance(chunk, dict) and "error" in chunk and "choices" not in chunk:
                    raise UpstreamError(
                        f"upstream {self.base_url} stream error: "
                        f"{_error_message(chunk['error'])}"
                    )
                yield RawChunk(data=chunk)

    async def complete(
        self, model: str, messages: list[dict[str, Any]], extra: dict[str, Any] | None = None
    ) -> str:
        """Non-streaming call; returns the assistant text."""
        extra = extra or {}
        body = self._body(model, messages, stream=False, extra=extra)
        try:
            response = await self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as e:
            raise FirstChunkTimeout(f"upstream {self.base_url} timed out: {e}") from e
        except httpx.HTTPError as e:
            raise UpstreamError(f"upstream {self.base_url} error: {e}") from e
        if response.status_code >= 400:
            raise UpstreamError(
                f"upstream {self.base_url} returned {response.status_code}: "
                f"{response.text[:500]}",
                status_code=response.status_code,
            )
        data = response.json()
        if isinstance(data, dict) and "error" in data and "choices" not in data:
            raise UpstreamError(
                f"upstream {self.base_url} error: {_error_message(data['error'])}"
            )
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise UpstreamError(f"unexpected upstream response: {data!r}") from e

    async def reachable(self, timeout_s: float = 3.0) -> bool:
        """Cheap liveness probe (GET /models)."""
        try:
            response = await self._client.get(
                "/models", timeout=httpx.Timeout(timeout_s, connect=2.0)
            )
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    async def list_models(self, timeout_s: float = 3.0) -> list[str]:
        try:
            response = await self._client.get(
                "/models", timeout=httpx.Timeout(timeout_s, connect=2.0)
            )
        except httpx.HTTPError:
            return []
        if response.status_code >= 400:
            return []
        try:
            return sorted(m["id"] for m in response.json()["data"])
        except (KeyError, TypeError):
            return []


class OpenAIRegistry:
    """Maps virtual model names to (backend, upstream model name)."""

    def __init__(
        self,
        backends: dict[str, OpenAIBackend],
        upstream: dict[str, str],
        model_backends: dict[str, OpenAIBackend],
    ) -> None:
        self._backends = backends  # endpoint -> backend
        self._upstream = upstream  # model name -> upstream model name
        self._model_backends = model_backends  # model name -> backend

    def stream(
        self,
        model_name: str,
        messages: list[dict[str, Any]],
        extra: dict[str, Any],
        first_chunk_timeout_s: float,
    ) -> AsyncIterator[StreamEvent]:
        backend = self._model_backends[model_name]
        upstream_model = self._upstream[model_name]
        chunks = backend.stream_chat(upstream_model, messages, extra)
        return _with_first_chunk_timeout(chunks, first_chunk_timeout_s, model_name)

    def backend_for(self, model_name: str) -> OpenAIBackend:
        return self._model_backends[model_name]

    def upstream_for(self, model_name: str) -> str:
        return self._upstream[model_name]

    async def aclose_all(self) -> None:
        for backend in self._backends.values():
            await backend.aclose()


async def _with_first_chunk_timeout(
    chunks: AsyncIterator[RawChunk],
    timeout_s: float,
    model_name: str,
) -> AsyncIterator[RawChunk]:
    iterator = chunks.__aiter__()
    try:
        try:
            first = await asyncio.wait_for(anext(iterator), timeout=timeout_s)
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError as e:
            raise FirstChunkTimeout(
                f"no first token from {model_name!r} within {timeout_s}s"
            ) from e
        yield RawChunk(data={**first.data, "model": model_name})
        async for chunk in iterator:
            yield RawChunk(data={**chunk.data, "model": model_name})
    finally:
        # release any open upstream stream on early exit
        close = getattr(iterator, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception:
                pass


def chunk_content(chunk: dict[str, Any]) -> str:
    """Extract delta content from an OpenAI chunk, if any."""
    try:
        delta = chunk["choices"][0].get("delta") or {}
        return delta.get("content") or ""
    except (KeyError, IndexError, TypeError):
        return ""


def chunk_usage(chunk: dict[str, Any]) -> dict[str, Any] | None:
    return chunk.get("usage")


def now_ts() -> int:
    return int(time.time())
