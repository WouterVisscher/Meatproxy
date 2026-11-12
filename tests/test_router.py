"""Router: plan selection and up-chain failover."""

import httpx
import pytest

from meatproxy.backends import build_registry
from meatproxy.config import ConfigManager
from meatproxy.conversations import ConversationStore
from meatproxy.copilot_adapter import CopilotAdapter
from meatproxy.main import AppState
from meatproxy.router import Router
from tests.conftest import TEST_CONFIG

TRIAGE_BLOCK = """  triage:
    model: npu-small
    timeout_s: 2
    on_failure: onprem
    cache_ttl_s: 300
"""


def make_router(tmp_path, fakes, triage: bool = True, content: str | None = None):
    base = content if content is not None else TEST_CONFIG
    content = base.replace("__DB__", str(tmp_path / "conv.db")).replace(
        "__COPILOT__", str(tmp_path / "copilot")
    )
    if not triage:
        content = content.replace(TRIAGE_BLOCK, "")
    path = tmp_path / "cfg.yaml"
    path.write_text(content)
    manager = ConfigManager(path)
    cfg = manager.get()

    factory = lambda endpoint: httpx.MockTransport(  # noqa: E731
        fakes[endpoint.split("/")[2]].handler
    )
    registry = build_registry(cfg, factory)

    state = AppState.__new__(AppState)  # bypass __init__ (no real store needed)
    state.manager = manager
    state.store = ConversationStore(str(tmp_path / "conv.db"))
    state.pool = type(
        "Pool", (), {"current": registry, "set_registry": lambda s, r: None}
    )()
    state.copilot = CopilotAdapter(cfg.copilot)
    state.triage = None
    state.router = Router(
        get_config=manager.get,
        get_registry=lambda: registry,
        copilot=state.copilot,
        get_triage=lambda: state.triage,
    )
    state._locks = {}
    state._reload_lock = None
    import asyncio

    state._reload_lock = asyncio.Lock()
    state._apply_config(cfg)
    return state.router, cfg, state


async def test_direct_plan(tmp_path, fakes):
    router, cfg, _ = make_router(tmp_path, fakes)
    plan = await router.plan("onprem", "anything", cfg)
    assert plan.source == "direct"
    assert [m.name for m in plan.models] == ["onprem"]


async def test_auto_plan_light(tmp_path, fakes):
    router, cfg, _ = make_router(tmp_path, fakes)
    plan = await router.plan("auto", "hello", cfg)
    assert plan.source == "auto"
    assert plan.tier == "light"
    assert [m.name for m in plan.models] == ["npu-small", "onprem"]


async def test_auto_plan_heavy(tmp_path, fakes):
    router, cfg, _ = make_router(tmp_path, fakes)
    plan = await router.plan("auto", "please refactor the whole architecture", cfg)
    assert plan.source == "auto"
    assert plan.tier == "heavy"


async def test_triage_failure_routes_to_on_failure(tmp_path, fakes):
    fakes["npu"].fail = True  # triage upstream is down
    router, cfg, _ = make_router(tmp_path, fakes)
    plan = await router.plan("auto", "hello", cfg)
    assert plan.source == "triage-failure"
    assert plan.models[0].name == "onprem"


async def test_auto_without_triage_goes_to_top_tier(tmp_path, fakes):
    router, cfg, _ = make_router(tmp_path, fakes, triage=False)
    plan = await router.plan("auto", "hello", cfg)
    assert plan.source == "auto-default"
    assert [m.name for m in plan.models] == ["onprem"]


async def test_run_failover_before_first_token(tmp_path, fakes):
    fakes["npu"].fail_models.add("fastflow-small")  # light model down
    router, cfg, _ = make_router(tmp_path, fakes)
    plan = await router.plan("auto", "hello", cfg)
    events = [ev async for ev in router.run(plan, [{"role": "user", "content": "hi"}], {}, None)]
    assert len(events) == 2  # both chunks came from the on-prem model
    assert all(ev.data["model"] == "onprem" for ev in events)
    assert len(fakes["gpu"].requests) == 1


async def test_run_exhausts_chain(tmp_path, fakes):
    fakes["npu"].fail = True
    fakes["gpu"].fail = True
    router, cfg, _ = make_router(tmp_path, fakes)
    plan = await router.plan("onprem", "hi", cfg)
    with pytest.raises(Exception):
        [ev async for ev in router.run(plan, [{"role": "user", "content": "hi"}], {}, None)]


async def test_triage_picks_tier_without_chain(tmp_path, fakes):
    """Triage may pick a tier that has no chain (key absent from
    routing.chains); the router must fall back to the triage on_failure
    model instead of producing an empty plan."""
    content = TEST_CONFIG.replace("    light: [npu-small, onprem]\n", "")
    router, cfg, _ = make_router(tmp_path, fakes, content=content)
    plan = await router.plan("auto", "hello", cfg)  # no trigger word -> "light"
    assert plan.source == "triage-failure"
    assert plan.models[0].name == "onprem"


async def test_triage_timeout_routes_to_on_failure(tmp_path, fakes):
    import dataclasses

    fakes["npu"].triage_delay = 5.0  # triage takes forever
    router, cfg, state = make_router(tmp_path, fakes)
    state.triage.spec = dataclasses.replace(state.triage.spec, timeout_s=0.01)
    plan = await router.plan("auto", "hello", cfg)
    assert plan.source == "triage-failure"
    assert plan.models[0].name == "onprem"
