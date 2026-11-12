"""Triage: a small local model classifies the incoming prompt.

The decision is one of the configured tiers (light / mid / heavy) and is
cached by prompt hash. When triage is unavailable or too slow, requests go
straight to the configured ``on_failure`` model.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field

from .config import TIER_ORDER, TriageSpec
from .openai_client import OpenAIBackend

log = logging.getLogger(__name__)

DEFAULT_SYSTEM = (
    "You are a routing classifier for an LLM gateway. Decide which tier of "
    "model should handle the user request below:\n"
    "- light: simple, short, low-stakes work (greeting, quick lookup, trivial "
    "edits, formatting, single small function).\n"
    "- mid: moderate work (writing or refactoring medium-sized code, explaining "
    "a concept, multi-step but bounded tasks).\n"
    "- heavy: complex or high-stakes work (architecture decisions, large "
    "multi-file changes, deep debugging, long reasoning, sensitive decisions).\n"
    "Reply with exactly one word: light, mid or heavy. No other text."
)


class TriageUnavailable(Exception):
    """Triage could not produce a decision (timeout, error, bad answer)."""


@dataclass(frozen=True)
class TriageDecision:
    tier: str
    cached: bool = False


@dataclass
class Triage:
    spec: TriageSpec
    backend: OpenAIBackend  # backend serving the triage model
    upstream_model: str
    _cache: dict[str, tuple[str, float]] = field(default_factory=dict)
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
            return TriageDecision(tier=cached[0], cached=True)

        tier = await self._classify_once(user_text)
        self._cache[key] = (tier, now)
        if len(self._cache) > self._cache_max:
            oldest = sorted(self._cache.items(), key=lambda kv: kv[1])[:100]
            for k, _ in oldest:
                self._cache.pop(k, None)
        return TriageDecision(tier=tier, cached=False)

    async def _classify_once(self, text: str) -> str:
        system = self.spec.prompt or DEFAULT_SYSTEM
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text[:4000]},
        ]
        try:
            answer = await self.backend.complete(
                self.upstream_model,
                messages,
                extra={"max_tokens": 8, "temperature": 0.0},
            )
        except Exception as e:
            raise TriageUnavailable(str(e)) from e
        tier = _parse_tier(answer)
        if tier is None:
            raise TriageUnavailable(f"unparsable triage answer: {answer!r}")
        return tier


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
