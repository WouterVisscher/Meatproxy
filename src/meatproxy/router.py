"""Routing: turn a request into a plan, then run it with up-chain failover.

A plan is a list of models to try in order. If a model fails *before the
first token is emitted*, the next model in the chain takes over. Once
tokens have been emitted we are committed to that model — errors propagate
to the client (a half-sent stream cannot be "un-sent").
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator

from .config import AUTO_TIER_FALLBACKS, Config, ConfigError, ModelSpec
from .metrics import failovers_total, triage_decisions_total, triage_latency_seconds
from .openai_client import StreamEvent, UpstreamError

if TYPE_CHECKING:
    from .copilot_adapter import CopilotAdapter

log = logging.getLogger(__name__)

AUTO_MODEL = "auto"


@dataclass(frozen=True)
class RoutePlan:
    tier: str
    models: list[ModelSpec]
    source: str  # "auto" | "direct" | "triage-failure" | "auto-default"


@dataclass(frozen=True)
class RequestedRoute:
    """What the client asked for, before routing."""

    model: str
    prompt: str  # last user text, used for triage


class Router:
    def __init__(
        self,
        get_config: Any,  # callable -> Config
        get_registry: Any,  # callable -> OpenAIRegistry
        copilot: "CopilotAdapter",
        get_triage: Any,  # callable -> Triage | None
    ) -> None:
        self.get_config = get_config
        self.get_registry = get_registry
        self.copilot = copilot
        self.get_triage = get_triage

    # -- planning ------------------------------------------------------------

    async def plan(self, requested_model: str, prompt: str, cfg: Config) -> RoutePlan:
        if requested_model != AUTO_MODEL:
            return RoutePlan(
                tier=cfg.model(requested_model).tier,
                models=cfg.chain_for_model(requested_model),
                source="direct",
            )

        triage = self.get_triage()
        if triage is not None:
            started = time.monotonic()
            try:
                decision = await asyncio.wait_for(
                    triage.classify(prompt), timeout=triage.spec.timeout_s
                )
            except Exception as e:
                triage_latency_seconds.observe(time.monotonic() - started)
                triage_decisions_total.labels(tier="failed", cached="false").inc()
                target = triage.spec.on_failure or _first_chain_model(cfg)
                if target is None:
                    raise ConfigError("no models configured")
                log.warning(
                    "triage failed (%s: %s); routing to %s", type(e).__name__, e, target
                )
                return RoutePlan(
                    tier=cfg.model(target).tier,
                    models=cfg.chain_for_model(target),
                    source="triage-failure",
                )
            triage_latency_seconds.observe(time.monotonic() - started)
            triage_decisions_total.labels(tier=decision.tier, cached=str(decision.cached)).inc()
            return RoutePlan(
                tier=decision.tier, models=cfg.chain_for(decision.tier), source="auto"
            )

        # triage disabled: go to the highest tier that has a chain
        for tier in AUTO_TIER_FALLBACKS:
            if cfg.chains.get(tier):
                return RoutePlan(tier=tier, models=cfg.chain_for(tier), source="auto-default")
        raise ConfigError("no routing chains configured")

    # -- execution ------------------------------------------------------------

    async def run(
        self,
        plan: RoutePlan,
        messages: list[dict[str, Any]],
        extra: dict[str, Any],
        conversation_id: str | None,
    ) -> AsyncIterator[StreamEvent]:
        last_error: Exception | None = None
        for index, spec in enumerate(plan.models):
            started = False
            try:
                if spec.kind == "copilot":
                    events: AsyncIterator[StreamEvent] = self.copilot.stream(
                        prompt=_prompt_from_messages(messages),
                        conversation_id=conversation_id,
                        model=spec.model or "auto",
                        timeout_s=spec.timeout,
                    )
                else:
                    events = self.get_registry().stream(
                        spec.name, messages, extra, spec.timeout
                    )
                async for event in events:
                    started = True
                    yield event
                log.info(
                    "served by model=%s tier=%s source=%s",
                    spec.name,
                    spec.tier,
                    plan.source,
                )
                return
            except UpstreamError as e:
                if started:
                    raise
                last_error = e
                if index + 1 < len(plan.models):
                    failovers_total.labels(
                        from_model=spec.name, to_model=plan.models[index + 1].name
                    ).inc()
                log.warning(
                    "model %s failed before first token (%s); failing over",
                    spec.name,
                    e,
                )
            except Exception:
                if started:
                    log.exception("unexpected error from model %s", spec.name)
                    raise
                log.exception("unexpected error from model %s", spec.name)
                last_error = last_error or UpstreamError(f"unexpected error from {spec.name}")
                if index + 1 < len(plan.models):
                    failovers_total.labels(
                        from_model=spec.name, to_model=plan.models[index + 1].name
                    ).inc()
        assert last_error is not None
        raise last_error


def _first_chain_model(cfg: Config) -> str | None:
    for tier in AUTO_TIER_FALLBACKS:
        names = cfg.chains.get(tier)
        if names:
            return names[0]
    return None


def _prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    """Build the text prompt for Copilot from the (stored) message list."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                # multimodal: flatten text parts
                return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            return content or ""
    return messages[-1].get("content", "") if messages else ""
