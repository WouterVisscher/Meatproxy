"""Config loading, validation and hot-reload."""

import textwrap

import pytest

from meatproxy.config import ConfigError, ConfigManager, load_config

VALID = textwrap.dedent(
    """
    models:
      small: {tier: light, endpoint: http://a/v1, model: s}
      big: {tier: heavy, endpoint: http://b/v1, model: b}

    routing:
      triage:
        model: small
        on_failure: big
      chains:
        light: [small, big]
        mid: [big]
        heavy: [big]
    """
)


def _write(tmp_path, content):
    p = tmp_path / "cfg.yaml"
    p.write_text(content)
    return p


def test_load_valid(tmp_path):
    cfg = load_config(_write(tmp_path, VALID))
    assert set(cfg.models) == {"small", "big"}
    assert cfg.triage is not None and cfg.triage.model == "small"
    assert cfg.chains["light"] == ["small", "big"]
    assert cfg.model("small").tier == "light"


def test_missing_models(tmp_path):
    with pytest.raises(ConfigError, match="models"):
        load_config(_write(tmp_path, "routing:\n  chains:\n    light: []\n"))


def test_unknown_model_in_chain(tmp_path):
    bad = VALID.replace("light: [small, big]", "light: [small, ghost]")
    with pytest.raises(ConfigError, match="ghost"):
        load_config(_write(tmp_path, bad))


def test_unknown_triage_model(tmp_path):
    bad = VALID.replace("model: small", "model: ghost")
    with pytest.raises(ConfigError, match="triage"):
        load_config(_write(tmp_path, bad))


def test_bad_tier(tmp_path):
    bad = VALID.replace("tier: light", "tier: tiny")
    with pytest.raises(ConfigError, match="tier"):
        load_config(_write(tmp_path, bad))


def test_openai_model_requires_endpoint(tmp_path):
    bad = (
        "models:\n  small: {tier: light, model: s}\n"
        "routing:\n  chains:\n    light: [small]\n"
    )
    with pytest.raises(ConfigError, match="endpoint"):
        load_config(_write(tmp_path, bad))


def test_chain_for_model_orders_direct_first(tmp_path):
    cfg = load_config(_write(tmp_path, VALID))
    chain = cfg.chain_for_model("small")
    assert [m.name for m in chain] == ["small", "big"]


def test_hot_reload(tmp_path):
    path = _write(tmp_path, VALID)
    manager = ConfigManager(path)
    assert manager.get().port == 8700
    path.write_text(VALID + "\napi:\n  port: 9999\n")
    import time

    time.sleep(0.05)  # ensure mtime differs
    assert manager.refresh() is True
    assert manager.get().port == 9999
    # no change -> no reload
    assert manager.refresh() is False


def test_hot_reload_rejects_bad_file(tmp_path):
    path = _write(tmp_path, VALID)
    manager = ConfigManager(path)
    path.write_text("models: [not a mapping")
    import time

    time.sleep(0.05)
    with pytest.raises(ConfigError):
        manager.refresh()
    # old config still active
    assert manager.get().port == 8700
