"""Configuration loading, validation and hot-reload.

The config is a single YAML file. Secrets are never stored in it: auth
material is referenced by environment-variable name.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TIER_ORDER = ("light", "mid", "heavy")

# When triage is disabled, "auto" requests start here and fall back down.
AUTO_TIER_FALLBACKS = ("heavy", "mid", "light")


class ConfigError(Exception):
    """Raised when the configuration file is invalid."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    tier: str
    kind: str  # "openai" | "copilot"
    endpoint: str | None = None
    model: str | None = None  # upstream model name (OpenAI backends)
    api_key_env: str | None = None
    timeout_s: float = 60.0
    ca_bundle: str | None = None  # path to a CA bundle for internal CAs
    ssl_insecure: bool = False  # skip TLS verification entirely (not recommended)

    def __post_init__(self) -> None:
        if self.kind not in ("openai", "copilot"):
            raise ConfigError(f"model {self.name!r}: kind must be 'openai' or 'copilot'")
        if self.kind == "openai" and not self.endpoint:
            raise ConfigError(f"model {self.name!r}: openai models need an endpoint")
        if self.kind == "openai" and not self.model:
            raise ConfigError(f"model {self.name!r}: openai models need a model name")
        if self.tier not in TIER_ORDER:
            raise ConfigError(f"model {self.name!r}: tier must be one of {TIER_ORDER}")

    @property
    def timeout(self) -> float:
        return self.timeout_s


@dataclass(frozen=True)
class TriageSpec:
    model: str
    on_failure: str
    timeout_s: float = 2.0
    cache_ttl_s: float = 300.0
    prompt: str | None = None  # optional custom classification prompt


@dataclass(frozen=True)
class ConversationsSpec:
    db_path: str = "meatproxy.db"
    max_context_messages: int = 40
    copilot_idle_timeout_s: float = 900.0


@dataclass(frozen=True)
class CopilotSpec:
    base_directory: str | None = None
    auto_tier: str | None = None
    token_env: str | None = None  # e.g. "COPILOT_GITHUB_TOKEN"; None = logged-in user
    timeout_s: float = 60.0


@dataclass
class Config:
    models: dict[str, ModelSpec]
    triage: TriageSpec | None
    chains: dict[str, list[str]]
    conversations: ConversationsSpec
    copilot: CopilotSpec
    host: str = "0.0.0.0"
    port: int = 8700
    _config_path: Path | None = field(default=None, compare=False)

    def model(self, name: str) -> ModelSpec:
        try:
            return self.models[name]
        except KeyError:
            raise ConfigError(f"unknown model {name!r}") from None

    def chain_for(self, tier: str) -> list[ModelSpec]:
        names = self.chains.get(tier, [])
        return [self.model(n) for n in names]

    def chain_for_model(self, name: str) -> list[ModelSpec]:
        """Direct route: the named model first, then the rest of its tier chain."""
        spec = self.model(name)
        names = self.chains.get(spec.tier, [])
        chain = [n for n in names if n != name]
        return [spec] + [self.model(n) for n in chain]


def _section(raw: dict[str, Any], name: str, required: bool = True) -> dict[str, Any]:
    value = raw.get(name, {}) if raw else {}
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ConfigError(f"{name!r} must be a mapping")
    if not value and required:
        raise ConfigError(f"missing required section: {name!r}")
    return value


def _model_spec(name: str, raw: Any) -> ModelSpec:
    if not isinstance(raw, dict):
        raise ConfigError(f"model {name!r}: must be a mapping")
    tier = raw.get("tier")
    if tier is None:
        raise ConfigError(f"model {name!r}: missing 'tier'")
    return ModelSpec(
        name=name,
        tier=tier,
        kind=raw.get("kind", "openai"),
        endpoint=raw.get("endpoint"),
        model=raw.get("model"),
        api_key_env=raw.get("api_key_env"),
        timeout_s=float(raw.get("timeout_s", 60.0)),
        ca_bundle=raw.get("ca_bundle"),
        ssl_insecure=bool(raw.get("ssl_insecure", False)),
    )


def load_config(path: str | Path) -> Config:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping in {path}")

    api = _section(raw, "api", required=False)
    models_raw = _section(raw, "models")
    models = {name: _model_spec(name, m) for name, m in models_raw.items()}

    routing = _section(raw, "routing")
    triage_raw = routing.get("triage")
    triage = None
    if triage_raw:
        if not isinstance(triage_raw, dict):
            raise ConfigError("'routing.triage' must be a mapping")
        triage_model = triage_raw.get("model")
        if not triage_model:
            raise ConfigError("'routing.triage.model' is required")
        triage = TriageSpec(
            model=triage_model,
            timeout_s=float(triage_raw.get("timeout_s", 2.0)),
            on_failure=triage_raw.get("on_failure", ""),
            cache_ttl_s=float(triage_raw.get("cache_ttl_s", 300.0)),
            prompt=triage_raw.get("prompt"),
        )

    chains_raw = _section(routing, "chains")
    chains: dict[str, list[str]] = {}
    for tier, names in chains_raw.items():
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ConfigError(f"routing.chains.{tier} must be a list of model names")
        chains[str(tier)] = list(names)

    conv_raw = _section(raw, "conversations", required=False)
    conversations = ConversationsSpec(
        db_path=conv_raw.get("db_path", "meatproxy.db"),
        max_context_messages=int(conv_raw.get("max_context_messages", 40)),
        copilot_idle_timeout_s=float(conv_raw.get("copilot_idle_timeout_s", 900.0)),
    )

    copilot_raw = _section(raw, "copilot", required=False)
    copilot = CopilotSpec(
        base_directory=copilot_raw.get("base_directory"),
        auto_tier=copilot_raw.get("auto_tier") or None,
        token_env=copilot_raw.get("token_env"),
        timeout_s=float(copilot_raw.get("timeout_s", 60.0)),
    )

    cfg = Config(
        models=models,
        triage=triage,
        chains=chains,
        conversations=conversations,
        copilot=copilot,
        host=api.get("host", "0.0.0.0"),
        port=int(api.get("port", 8700)),
        _config_path=path,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    # chain members must exist
    for tier, names in cfg.chains.items():
        for name in names:
            if name not in cfg.models:
                raise ConfigError(f"routing.chains.{tier}: unknown model {name!r}")
    if cfg.triage:
        if cfg.triage.model not in cfg.models:
            raise ConfigError(f"routing.triage.model: unknown model {cfg.triage.model!r}")
        if cfg.triage.on_failure and cfg.triage.on_failure not in cfg.models:
            raise ConfigError(
                f"routing.triage.on_failure: unknown model {cfg.triage.on_failure!r}"
            )


def resolve_api_key(spec: ModelSpec) -> str | None:
    """Resolve an upstream API key via the env var named in the model spec."""
    if not spec.api_key_env:
        return None
    return os.environ.get(spec.api_key_env)


class ConfigManager:
    """Holds the active config and hot-reloads it when the file changes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._config = load_config(self.path)
        self._mtime = self.path.stat().st_mtime

    def get(self) -> Config:
        with self._lock:
            return self._config

    def refresh(self) -> bool:
        """Reload if the file changed. Returns True if the config was replaced.

        Invalid new files are rejected (ConfigError raised) and the old
        config stays active.
        """
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return False
        if mtime == self._mtime:
            return False
        new = load_config(self.path)
        with self._lock:
            self._config = new
            self._mtime = mtime
        return True
