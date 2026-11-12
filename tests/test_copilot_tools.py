"""Client-executed tool calls through the SDK terminal-tool bridge."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from meatproxy import copilot_tools as ct
from meatproxy.config import CopilotSpec
from meatproxy.copilot_adapter import CopilotAdapter
from meatproxy.openai_client import UpstreamError
from tests.test_copilot_adapter import (
    FakeClient, FakeSession, FakeEvent, FakeDeltaData, FakeIdleData,
    _install_fake_sdk,
)
from tests.conftest import auth, parse_sse


TOOLS = [{"type": "function", "function": {
    "name": "bash", "description": "Run a client command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
}}]
MESSAGES = [{"role": "user", "content": "refactor using bash"}]


class ToolSession(FakeSession):
    def __init__(self, kwargs, owner):
        super().__init__()
        self.kwargs = kwargs
        self.owner = owner
        self.aborted = False

    async def abort(self):
        self.aborted = True

    async def send(self, prompt):
        self.owner.prompts.append(prompt)
        if self.owner.fail:
            raise RuntimeError("runtime failed")
        if self.owner.stall:
            await asyncio.Event().wait()
        transcript = json.loads(prompt[prompt.index("["):])
        tools = self.kwargs["tools"]
        if tools and transcript[-1]["role"] != "tool" and not self.owner.ignore_tools:
            for i in range(self.owner.call_count):
                result = await tools[0].handler(SimpleNamespace(
                    tool_name=tools[0].name, arguments={"command": f"echo {i}"}, agent_id=None,
                ))
                self.owner.results.append(result)
        else:
            for handler in list(self._handlers):
                handler(FakeEvent(FakeDeltaData("result received")))
        await asyncio.sleep(0)
        for handler in list(self._handlers):
            handler(FakeEvent(FakeIdleData()))
        return "request-1"


class ToolClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.prompts = []
        self.results = []
        self.fail = False
        self.stall = False
        self.ignore_tools = False
        self.call_count = 1

    async def create_session(self, **kwargs):
        session = ToolSession(kwargs, self)
        self.created.append(session)
        return session


@pytest.fixture
def tool_client(monkeypatch):
    client = ToolClient()
    _install_fake_sdk(monkeypatch, client)
    monkeypatch.setattr(ct, "_tool_sdk", lambda: {"Tool": SimpleNamespace, "ToolResult": SimpleNamespace})
    return client


async def collect(adapter, messages=None, extra=None, timeout=1):
    return [event.data async for event in ct.stream_tools(
        adapter, messages or MESSAGES, extra if extra is not None else {"tools": TOOLS},
        "auto", "copilot", timeout,
    )]


async def test_terminal_tools_are_forwarded_not_executed(tool_client):
    adapter = CopilotAdapter(CopilotSpec())
    chunks = await collect(adapter)
    call = chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert call["function"] == {"name": "bash", "arguments": '{"command":"echo 0"}'}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    session = tool_client.created[0]
    assert session.kwargs["tools"][0].is_terminal is True
    assert session.kwargs["available_tools"] == ["meatproxy_client_0"]
    assert session.kwargs["tools"][0].skip_permission is True
    assert session.disconnected is True


async def test_roundtrip_rebuilds_full_transcript(tool_client):
    adapter = CopilotAdapter(CopilotSpec())
    chunks = await collect(adapter)
    call = chunks[0]["choices"][0]["delta"]["tool_calls"][0].copy()
    call.pop("index")
    transcript = MESSAGES + [
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": "real client result"},
    ]
    result = await collect(adapter, transcript)
    assert result[0]["choices"][0]["delta"]["content"] == "result received"
    assert "real client result" in tool_client.prompts[-1]
    assert len(tool_client.created) == 2
    assert all(s.disconnected for s in tool_client.created)


@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "bash"}}])
async def test_required_and_named_choices(tool_client, choice):
    chunks = await collect(CopilotAdapter(CopilotSpec()), extra={"tools": TOOLS, "tool_choice": choice})
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


async def test_tool_choice_none_hides_tools(tool_client):
    chunks = await collect(CopilotAdapter(CopilotSpec()), extra={"tools": TOOLS, "tool_choice": "none"})
    assert tool_client.created[0].kwargs["tools"] == []
    assert tool_client.created[0].kwargs["available_tools"] == []
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


async def test_required_choice_not_honored_fails_before_output(tool_client):
    tool_client.ignore_tools = True
    with pytest.raises(UpstreamError, match="did not honor"):
        await collect(CopilotAdapter(CopilotSpec()), extra={"tools": TOOLS, "tool_choice": "required"})
    assert tool_client.created[0].aborted
    assert tool_client.created[0].disconnected


@pytest.mark.parametrize("parallel, count", [(True, 2), (False, 1)])
async def test_parallel_call_indices(tool_client, parallel, count):
    tool_client.call_count = 2
    chunks = await collect(CopilotAdapter(CopilotSpec()), extra={"tools": TOOLS, "parallel_tool_calls": parallel})
    calls = [c["choices"][0]["delta"]["tool_calls"][0] for c in chunks[:-1]]
    assert [c["index"] for c in calls] == list(range(count))
    assert len({c["id"] for c in calls}) == count


@pytest.mark.parametrize("extra", [
    {"tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "missing"}}},
    {"tools": [{"type": "custom"}]},
    {"tools": TOOLS * 2},
    {"tools": [], "tool_choice": "required"},
    {"tool_choice": "required"},
])
async def test_unsupported_tools_fail_explicitly(tool_client, extra):
    with pytest.raises(UpstreamError):
        await collect(CopilotAdapter(CopilotSpec()), extra=extra)
    assert not tool_client.created


@pytest.mark.parametrize("failure", ["fail", "stall"])
async def test_runtime_failure_and_timeout_cleanup(tool_client, failure):
    setattr(tool_client, failure, True)
    with pytest.raises(UpstreamError):
        await collect(CopilotAdapter(CopilotSpec()), timeout=0.01)
    assert tool_client.created[0].disconnected
    assert tool_client.created[0].aborted


async def test_cancelled_response_cleans_up_session(tool_client):
    adapter = CopilotAdapter(CopilotSpec())
    stream = ct.stream_tools(adapter, MESSAGES, {"tools": TOOLS}, "auto", "copilot", 1)
    await anext(stream)
    await stream.aclose()
    assert tool_client.created[0].disconnected
    assert tool_client.created[0].aborted


@pytest.mark.parametrize("stream", [False, True])
async def test_api_auto_triage_to_copilot_tool_roundtrip(client, config_path, tool_client, stream):
    config_path.write_text(config_path.read_text().replace("heavy: [onprem]", "heavy: [copilot, onprem]"))
    response = await client.post("/v1/chat/completions", headers=auth(), json={
        "model": "auto", "stream": stream, "messages": MESSAGES, "tools": TOOLS,
    })
    assert response.status_code == 200
    if stream:
        chunks = parse_sse(response.text)
        call = chunks[0]["choices"][0]["delta"]["tool_calls"][0].copy()
        call.pop("index")
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
        assert response.text.endswith("data: [DONE]\n\n")
        assistant = {"role": "assistant", "tool_calls": [call]}
    else:
        choice = response.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assistant = choice["message"]
        call = assistant["tool_calls"][0]
    result = await client.post("/v1/chat/completions", headers=auth(), json={
        "model": "auto", "messages": MESSAGES + [assistant, {
            "role": "tool", "tool_call_id": call["id"], "content": "client execution output",
        }], "tools": TOOLS,
    })
    assert result.status_code == 200
    assert result.json()["choices"][0]["message"]["content"] == "result received"
    assert "client execution output" in tool_client.prompts[-1]


async def test_api_stored_conversation_accepts_tool_only_results(client, app, tool_client):
    headers = {**auth(), "X-Conversation-Id": "tools-conversation"}
    first = await client.post("/v1/chat/completions", headers=headers, json={
        "model": "copilot", "messages": MESSAGES, "tools": TOOLS,
    })
    call = first.json()["choices"][0]["message"]["tool_calls"][0]
    result = await client.post("/v1/chat/completions", headers=headers, json={
        "model": "copilot", "messages": [{
            "role": "tool", "tool_call_id": call["id"], "content": "actual result",
        }], "tools": TOOLS,
    })
    assert result.status_code == 200
    assert result.json()["choices"][0]["message"]["content"] == "result received"
    history = await app.state.state.store.history("tools-conversation")
    assert [m.role for m in history] == ["user", "assistant", "tool", "assistant"]
    assert history[2].tool_call_id == call["id"]
    assert "actual result" in tool_client.prompts[-1]


async def test_conversation_rejects_mismatched_results(client, tool_client):
    headers = {**auth(), "X-Conversation-Id": "tools-conversation"}
    await client.post("/v1/chat/completions", headers=headers, json={
        "model": "copilot", "messages": MESSAGES, "tools": TOOLS,
    })
    result = await client.post("/v1/chat/completions", headers=headers, json={
        "model": "copilot", "messages": [{"role": "tool", "tool_call_id": "wrong", "content": "output"}],
        "tools": TOOLS,
    })
    assert result.status_code == 400
    assert len(tool_client.created) == 1


async def test_system_instructions_not_demoted_to_user_transcript(tool_client):
    await collect(CopilotAdapter(CopilotSpec()), [{"role": "system", "content": "client instructions"}] + MESSAGES)
    assert "client instructions" in tool_client.created[0].kwargs["system_message"]["content"]
    assert "client instructions" not in tool_client.prompts[0]


async def test_local_failover_reaches_copilot_with_tools(client, config_path, fakes, tool_client):
    config_path.write_text(config_path.read_text().replace(
        "light: [npu-small, onprem]", "light: [npu-small, onprem, copilot]"
    ))
    fakes["npu"].fail_models.add("fastflow-small")
    fakes["gpu"].fail = True
    result = await client.post("/v1/chat/completions", headers=auth(), json={
        "model": "auto", "messages": [{"role": "user", "content": "use bash"}], "tools": TOOLS,
    })
    assert result.status_code == 200
    assert result.json()["model"] == "copilot"
    assert result.json()["choices"][0]["finish_reason"] == "tool_calls"


async def test_copilot_failover_keeps_tools_on_local_backend(client, config_path, fakes, tool_client):
    config_path.write_text(config_path.read_text().replace("heavy: [onprem]", "heavy: [copilot, onprem]"))
    tool_client.fail = True
    fakes["gpu"].tool_calls = [{"id": "local-call", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]
    result = await client.post("/v1/chat/completions", headers=auth(), json={
        "model": "auto", "messages": MESSAGES, "tools": TOOLS,
    })
    assert result.status_code == 200
    assert result.json()["model"] == "onprem"
    assert result.json()["choices"][0]["finish_reason"] == "tool_calls"
    assert fakes["gpu"].requests[-1]["tools"] == TOOLS


async def test_required_choice_without_tools_is_not_silently_ignored(client, tool_client):
    response = await client.post("/v1/chat/completions", headers=auth(), json={
        "model": "copilot", "messages": MESSAGES, "tool_choice": "required",
    })
    assert response.status_code == 400
    assert "available function tool" in response.json()["detail"]
    assert not tool_client.created
