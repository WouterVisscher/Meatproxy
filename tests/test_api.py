"""End-to-end API tests against the app with fake upstreams."""

import logging
import time

from tests.conftest import auth, parse_sse


async def test_missing_auth_rejected(client):
    r = await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401


async def test_wrong_key_rejected(client):
    r = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer nope"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 401


async def test_unknown_model_404(client):
    r = await client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"model": "ghost", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 404


async def test_invalid_messages_400(client):
    r = await client.post("/v1/chat/completions", headers=auth(), json={"messages": []})
    assert r.status_code == 400


async def test_streaming_light_request(client, fakes):
    r = await client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"model": "auto", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    chunks = parse_sse(r.text)
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c.get("choices"))
    assert "hello from fastflow-small world" in content
    # model name rewritten to the virtual name
    assert all(c["model"] == "npu-small" for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1].get("usage")
    # triage + actual call both went to the NPU upstream
    assert len(fakes["npu"].requests) == 2
    assert len(fakes["gpu"].requests) == 0


async def test_streaming_heavy_request_routed_to_chain(client, fakes):
    r = await client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={
            "model": "auto",
            "stream": True,
            "messages": [{"role": "user", "content": "refactor this big architecture"}],
        },
    )
    chunks = parse_sse(r.text)
    # heavy chain is ["onprem"] in the test config
    assert all(c["model"] == "onprem" for c in chunks if c.get("choices"))
    assert any("hello from qwen" in c["choices"][0]["delta"].get("content", "") for c in chunks)


async def test_failover_to_next_model(client, fakes):
    fakes["npu"].fail_models.add("fastflow-small")
    r = await client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"model": "npu-small", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    chunks = parse_sse(r.text)
    assert all(c["model"] == "onprem" for c in chunks if c.get("choices"))


async def test_sse_error_payload_triggers_failover(client, fakes):
    """Upstreams may deliver errors as SSE payloads inside an HTTP 200
    (Lemonade does this for context overflow). That must be treated as a
    failure and fail over, not passed through as a choiceless chunk —
    which would yield an empty response to the client.
    """
    fakes["npu"].stream_error = (
        "max_length (34048) cannot be greater than model context_length (32768)"
    )
    r = await client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"model": "auto", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    chunks = parse_sse(r.text)
    # triage says "light" -> npu-small stream errors -> fail over to onprem
    assert all(c["model"] == "onprem" for c in chunks if c.get("choices"))
    assert any("hello from qwen" in c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert all("error" not in c for c in chunks)


async def test_all_models_down_returns_error_event(client, fakes):
    fakes["npu"].fail = True
    fakes["gpu"].fail = True
    r = await client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"model": "npu-small", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    events = parse_sse(r.text)
    assert "error" in events[-1]


async def test_non_stream_request(client, fakes):
    r = await client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"model": "auto", "stream": False, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "npu-small"
    assert body["choices"][0]["message"]["content"] == "hello from fastflow-small world"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == 7


async def test_omitted_stream_defaults_to_non_stream(client, fakes):
    """OpenAI default: no `stream` field means a JSON (non-streaming) response.

    The AI SDK omits `stream` for non-streaming calls; returning SSE there
    made clients fail with 'Invalid JSON response'.
    """
    r = await client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert not r.headers["content-type"].startswith("text/event-stream")
    assert r.json()["object"] == "chat.completion"


async def test_models_endpoint(client):
    r = await client.get("/v1/models", headers=auth())
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["data"]}
    assert ids == {"auto", "npu-small", "onprem", "copilot"}


async def test_conversation_flow(client, fakes):
    headers = {**auth(), "X-Conversation-Id": "conv-1"}
    for i in range(2):
        r = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "auto", "stream": True, "messages": [{"role": "user", "content": f"hi {i}"}]},
        )
        assert r.status_code == 200
        parse_sse(r.text)

    # the router rebuilt history: 2nd call sent 3 messages (u0, a0, u1)
    chat_calls = [b for b in fakes["npu"].requests if b.get("max_tokens") != 8]
    assert [len(b["messages"]) for b in chat_calls] == [1, 3]

    listing = await client.get("/v1/conversations", headers=auth())
    convs = {c["id"]: c for c in listing.json()["data"]}
    assert convs["conv-1"]["message_count"] == 4

    r = await client.delete("/v1/conversations/conv-1", headers=auth())
    assert r.status_code == 200
    listing = await client.get("/v1/conversations", headers=auth())
    assert listing.json()["data"] == []

    r = await client.delete("/v1/conversations/conv-1", headers=auth())
    assert r.status_code == 404


async def test_health(client, fakes):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models"]["npu-small"] == "up"
    assert body["models"]["copilot"] == "configured"


async def test_metrics_endpoint(client):
    await client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"model": "auto", "stream": False, "messages": [{"role": "user", "content": "hi"}]},
    )
    r = await client.get("/metrics")
    assert r.status_code == 200
    assert "meatproxy_requests_total" in r.text


async def test_invalid_config_reload_keeps_serving(client, config_path, caplog):
    """An invalid config file must not take the API down: the old config
    keeps serving, and the rejection is logged once per file version."""
    r = await client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200

    config_path.write_text("models: [not a mapping")
    time.sleep(0.05)  # ensure mtime differs

    with caplog.at_level(logging.WARNING, logger="meatproxy.main"):
        r = await client.post(
            "/v1/chat/completions",
            headers=auth(),
            json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
        assert r.json()["object"] == "chat.completion"
        # second request: still serving, no duplicate warning
        r = await client.post(
            "/v1/chat/completions",
            headers=auth(),
            json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
    warnings = [
        rec for rec in caplog.records if "config reload rejected" in rec.getMessage()
    ]
    assert len(warnings) == 1


async def test_no_user_message_400(client):
    r = await client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"messages": [{"role": "assistant", "content": "nope"}]},
    )
    assert r.status_code == 400
