"""Builds the OpenAI backend registry from config; rebuilt on hot-reload."""

from __future__ import annotations

import logging
from typing import Any, Callable

from .config import Config, resolve_api_key
from .openai_client import OpenAIBackend, OpenAIRegistry

log = logging.getLogger(__name__)

# test hook: endpoint -> transport
TransportFactory = Callable[[str], Any]


class BackendPool:
    """Holds the active OpenAIRegistry and rebuilds it when config changes."""

    def __init__(self, transport_factory: TransportFactory | None = None) -> None:
        self._registry: OpenAIRegistry | None = None
        self.transport_factory = transport_factory

    @property
    def current(self) -> OpenAIRegistry:
        if self._registry is None:
            raise RuntimeError("backend pool has no config")
        return self._registry

    def set_registry(self, registry: OpenAIRegistry) -> None:
        self._registry = registry

    async def rebuild(self, cfg: Config) -> None:
        old, self._registry = self._registry, build_registry(
            cfg, self.transport_factory
        )
        if old is not None:
            await old.aclose_all()

    async def aclose_all(self) -> None:
        if self._registry is not None:
            await self._registry.aclose_all()


def _endpoint_verify(cfg: Config, endpoint: str) -> bool | str:
    """Resolve the TLS verification setting for an endpoint.

    First model on the endpoint wins (consistent with _first_api_key).
    """
    for spec in cfg.models.values():
        if spec.kind != "openai" or spec.endpoint != endpoint:
            continue
        if spec.ssl_insecure:
            log.warning(
                "model %s: TLS verification DISABLED for %s — do not use on a shared network",
                spec.name,
                endpoint,
            )
            return False
        if spec.ca_bundle:
            return spec.ca_bundle
    return True


def build_registry(
    cfg: Config,
    transport_factory: TransportFactory | None = None,
) -> OpenAIRegistry:
    backends: dict[str, OpenAIBackend] = {}
    timeouts: dict[str, float] = {}
    for spec in cfg.models.values():
        if spec.kind != "openai" or not spec.endpoint:
            continue
        endpoint = spec.endpoint
        timeouts[endpoint] = min(timeouts.get(endpoint, spec.timeout), spec.timeout)
    upstream: dict[str, str] = {}
    model_backends: dict[str, OpenAIBackend] = {}
    for name, spec in cfg.models.items():
        if spec.kind != "openai" or not spec.endpoint:
            continue
        endpoint = spec.endpoint
        if endpoint not in backends:
            backends[endpoint] = OpenAIBackend(
                endpoint,
                api_key=_first_api_key(cfg, endpoint),
                timeout_s=timeouts[endpoint],
                verify=_endpoint_verify(cfg, endpoint),
                transport=transport_factory(endpoint) if transport_factory else None,
            )
        upstream[name] = spec.model
        model_backends[name] = backends[endpoint]
    return OpenAIRegistry(backends, upstream, model_backends)


def _first_api_key(cfg: Config, endpoint: str) -> str | None:
    for spec in cfg.models.values():
        if spec.kind == "openai" and spec.endpoint == endpoint:
            key = resolve_api_key(spec)
            if key:
                return key
    return None
