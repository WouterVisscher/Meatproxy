"""Triage: classify prompts into tiers, caching a brief decision summary."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field, replace

from .config import TIER_ORDER, TriageSpec
from .openai_client import OpenAIBackend

DEFAULT_SYSTEM = (
    "You are a routing classifier for an LLM gateway. Decide which tier of "
    "model should handle the user request below:\n"
    "- light: simple, short, low-stakes work (greeting, quick lookup, trivial "
    "edits, formatting, single small function).\n"
    "- mid: moderate work (writing or refactoring medium-sized code, explaining "
    "a concept, multi-step but bounded tasks).\n"
    "- heavy: complex or high-stakes work (architecture decisions, large "
    "multi-file changes, deep debugging, long reasoning, sensitive decisions).\n"
    "Reply on one line: <tier> | <brief decision summary>. "
    "Start with light, mid or heavy, then a pipe and a short explanation of "
    "the task complexity (at most 20 words). Do not provide step-by-step "
    "reasoning or quote user content, secrets, or code."
)

TRIAGE_MAX_TOKENS = 96
LEGACY_REASON = "Classifier returned a tier without a decision summary."


class TriageUnavailable(Exception):
    """Triage could not produce a decision (timeout, error, bad answer)."""


@dataclass(frozen=True)
class TriageDecision:
    tier: str
    cached: bool = False
    # Model-generated diagnostic summary, not internal reasoning.
    reason: str = LEGACY_REASON


@dataclass
class Triage:
    spec: TriageSpec
    backend: OpenAIBackend
    upstream_model: str
    _cache: dict[str, tuple[TriageDecision, float]] = field(default_factory=dict)
    _cache_max: int = 1024

    def _key(self, text: str) -> str:
        payload = f"{self.upstream_model}\x00{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def classify(self, user_text: str) -> TriageDecision:
        """Classify and return the tier. Raises TriageUnavailable on failure."""
        key = self._key(user_text)
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached and now - cached[1] < self.spec.cache_ttl_s:
            return replace(cached[0], cached=True)

        decision = await self._classify_once(user_text)
        self._cache[key] = (decision, now)
        if len(self._cache) > self._cache_max:
            oldest = sorted(self._cache.items(), key=lambda kv: kv[1][1])[:100]
            for k, _ in oldest:
                self._cache.pop(k, None)
        return decision

    async def _classify_once(self, text: str) -> TriageDecision:
        system = self.spec.prompt or DEFAULT_SYSTEM
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text[:4000]},
        ]
        try:
            answer = await self.backend.complete(
                self.upstream_model,
                messages,
                extra={"max_tokens": TRIAGE_MAX_TOKENS, "temperature": 0.0},
            )
        except Exception as e:
            raise TriageUnavailable(str(e)) from e
        tier, reason = _parse_decision(answer)
        if tier is None:
            raise TriageUnavailable("unparsable triage answer")
        return TriageDecision(tier=tier, reason=reason)


def _parse_decision(answer: str) -> tuple[str | None, str]:
    """Accept tier-first summaries and legacy tier-only classifier prompts."""
    label, separator, summary = answer.strip().partition("|")
    tier = _parse_tier(label)
    reason = " ".join(summary.split())[:240] if separator else ""
    return tier, reason or LEGACY_REASON


def _parse_tier(answer: str) -> str | None:
    """Find the first tier word in the (hopefully tiny) model answer."""
    match = re.search(r"\b(light|mid|medium|heavy|hard)\b", answer, re.IGNORECASE)
    if not match:
        return None
    word = match.group(1).lower()
    if word == "medium":
        return "mid"
    if word == "hard":
        return "heavy"
    return word if word in TIER_ORDER else None
