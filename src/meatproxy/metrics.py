"""Prometheus metrics for meatproxy."""

from __future__ import annotations

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

requests_total = Counter(
    "meatproxy_requests_total",
    "Routed chat-completion requests.",
    ["model", "tier", "source"],
)

failovers_total = Counter(
    "meatproxy_failovers_total",
    "Failover events: a model failed before its first token and the next "
    "model in the chain took over.",
    ["from_model", "to_model"],
)

triage_latency_seconds = Histogram(
    "meatproxy_triage_latency_seconds",
    "Time spent in the triage (classification) call, including cache hits.",
)

triage_decisions_total = Counter(
    "meatproxy_triage_decisions_total",
    "Triage classification results.",
    ["tier", "cached"],
)

conversations_active = Gauge(
    "meatproxy_conversations_active",
    "Conversations with a live Copilot session in the pool."
)


def render_metrics() -> bytes:
    return generate_latest()
