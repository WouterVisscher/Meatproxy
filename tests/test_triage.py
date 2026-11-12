"""Triage classification, caching and failure modes."""

import json

import httpx
import pytest

from meatproxy.config import TriageSpec
from meatproxy.openai_client import OpenAIBackend
from meatproxy.triage import Triage, TriageUnavailable, _parse_tier


def _backend(responses: list[str], calls: list) -> OpenAIBackend:
    n = {"i": 0}

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        assert body["max_tokens"] == 8
        calls.append(body)
        answer = responses[min(n["i"], len(responses) - 1)]
        n["i"] += 1
        return httpx.Response(
            200, json={"choices": [{"message": {"content": answer}}]}
        )

    return OpenAIBackend(
        "http://npu/v1", timeout_s=5.0, transport=httpx.MockTransport(handler)
    )


def _triage(backend) -> Triage:
    return Triage(
        spec=TriageSpec(model="npu-small", timeout_s=2.0, on_failure="copilot", cache_ttl_s=300),
        backend=backend,
        upstream_model="fastflow-small",
    )


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("light", "light"),
        ("LIGHT", "light"),
        ("the answer is: heavy", "heavy"),
        ("medium", "mid"),
        ("mid", "mid"),
    ],
)
def test_parse_tier(answer, expected):
    assert _parse_tier(answer) == expected


def test_parse_tier_garbage():
    assert _parse_tier("I don't know") is None


async def test_classify_and_cache():
    calls: list = []
    triage = _triage(_backend(["light"], calls))
    first = await triage.classify("say hi")
    second = await triage.classify("say hi")
    assert first.tier == "light" and first.cached is False
    assert second.cached is True
    assert len(calls) == 1  # cache hit, no second upstream call


async def test_classify_different_prompts_not_cached():
    calls: list = []
    triage = _triage(_backend(["light", "heavy"], calls))
    await triage.classify("a")
    await triage.classify("b")
    assert len(calls) == 2


async def test_unparsable_answer_raises():
    triage = _triage(_backend(["huh?"], []))
    with pytest.raises(TriageUnavailable):
        await triage.classify("x")


async def test_upstream_error_raises():
    def handler(request):
        return httpx.Response(500, text="boom")

    backend = OpenAIBackend(
        "http://npu/v1", timeout_s=5.0, transport=httpx.MockTransport(handler)
    )
    triage = _triage(backend)
    with pytest.raises(TriageUnavailable):
        await triage.classify("x")
