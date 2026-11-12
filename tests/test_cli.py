"""CLI: subcommand dispatch, serve delegation, config check, copilot commands.

Copilot commands use the fake SDK (monkeypatched into
copilot_auth._auth_sdk); serve is tested by patching uvicorn.run.
"""

from __future__ import annotations

import io

import pytest

import meatproxy.cli as cli
import meatproxy.main as m
from meatproxy import __version__
from tests.fake_auth_sdk import (
    ResultStatus,
    StepKind,
    WriteKind,
    FakeAccountStatus,
    FakeResult,
    FakeStatus,
    FakeStep,
    install,
    make_client,
)


def _device_code_step() -> FakeStep:
    return FakeStep(
        StepKind.DEVICE_CODE,
        user_code="ABCD-1234",
        verification_uri="https://github.com/login/device",
    )


def _completed_step(login: str = "octocat") -> FakeStep:
    return FakeStep(
        StepKind.COMPLETED,
        result=FakeResult(ResultStatus.COMPLETED, host="github.com", login=login),
    )


# -- version / config -------------------------------------------------------------


def test_version(capsys):
    assert cli.main(["version"]) == 0
    assert f"meatproxy {__version__}" in capsys.readouterr().out


def test_config_check(config_path, capsys):
    assert cli.main(["config", "check", "--config", str(config_path)]) == 0
    out = capsys.readouterr().out
    assert "config ok" in out
    assert "npu-small" in out
    assert "chain light" in out
    assert "copilot base_directory" in out


def test_config_check_missing(capsys):
    assert cli.main(["config", "check", "--config", "/nonexistent.yaml"]) == 2
    assert "invalid" in capsys.readouterr().err


def test_config_check_invalid(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("models: []\n")
    assert cli.main(["config", "check", "--config", str(bad)]) == 2
    assert "invalid" in capsys.readouterr().err


# -- serve ------------------------------------------------------------------------


def test_no_subcommand_serves(config_path, monkeypatch, capsys):
    calls: list[dict] = []
    monkeypatch.setattr(m.uvicorn, "run", lambda app, **kw: calls.append(kw))
    monkeypatch.setenv("MEATPROXY_CONFIG", str(config_path))

    assert cli.main([]) == 0

    assert calls, "uvicorn.run was not called"
    assert calls[0]["port"] == 8700
    assert calls[0]["host"] == "0.0.0.0"


def test_serve_explicit_flags(config_path, monkeypatch, capsys):
    calls: list[dict] = []
    monkeypatch.setattr(m.uvicorn, "run", lambda app, **kw: calls.append(kw))

    assert cli.main(["serve", "--config", str(config_path), "--port", "9999"]) == 0

    assert calls[0]["port"] == 9999


def test_serve_missing_config(monkeypatch, capsys):
    monkeypatch.delenv("MEATPROXY_CONFIG", raising=False)
    monkeypatch.chdir("/tmp")

    with pytest.raises(SystemExit):
        cli.main(["serve", "--config", "/nonexistent.yaml"])


# -- copilot status -----------------------------------------------------------------


def test_copilot_status_not_logged_in(monkeypatch, capsys):
    monkeypatch.delenv("MEATPROXY_CONFIG", raising=False)
    client = make_client(FakeStatus(False))
    install(monkeypatch, client)

    assert cli.main(["copilot", "status"]) == 1

    out = capsys.readouterr().out
    assert "Not logged in" in out
    assert client.stopped is True


def test_copilot_status_logged_in(monkeypatch, capsys):
    monkeypatch.delenv("MEATPROXY_CONFIG", raising=False)
    client = make_client(FakeStatus(True, login="octocat", host="github.com"))
    install(monkeypatch, client)

    assert cli.main(["copilot", "status"]) == 0

    out = capsys.readouterr().out
    assert "Logged in as octocat (github.com)" in out


def test_copilot_status_uses_config_base_directory(
    config_path, monkeypatch, capsys
):
    """base_directory from the config file is passed to the client."""
    kwargs_seen: list[dict] = []
    client = make_client(FakeStatus(False))
    install(monkeypatch, client, captured_kwargs=kwargs_seen)

    assert cli.main(["copilot", "status", "--config", str(config_path)]) == 1

    assert kwargs_seen[0]["base_directory"] == str(config_path.parent / "copilot")


def test_copilot_status_explicit_base_directory(monkeypatch, capsys):
    monkeypatch.delenv("MEATPROXY_CONFIG", raising=False)
    kwargs_seen: list[dict] = []
    client = make_client(FakeStatus(False))
    install(monkeypatch, client, captured_kwargs=kwargs_seen)

    assert cli.main(["copilot", "status", "--base-directory", "/x/copilot"]) == 1

    assert kwargs_seen[0]["base_directory"] == "/x/copilot"


def test_copilot_status_missing_config_warns(monkeypatch, capsys):
    monkeypatch.setenv("MEATPROXY_CONFIG", "/nonexistent.yaml")
    client = make_client(FakeStatus(False))
    install(monkeypatch, client)

    assert cli.main(["copilot", "status"]) == 1

    err = capsys.readouterr().err
    assert "not found" in err
    assert "~/.copilot" in err


# -- copilot login -------------------------------------------------------------------


def test_copilot_login_token(monkeypatch, capsys):
    monkeypatch.delenv("MEATPROXY_CONFIG", raising=False)
    client = make_client(FakeStatus(True, login="octocat", host="github.com"))
    install(monkeypatch, client)

    assert cli.main(["copilot", "login", "--token", "ghp_abc"]) == 0

    out = capsys.readouterr().out
    assert "logged in as octocat" in out
    cmd = client.accounts_api.set_calls[0].command
    assert cmd.kind is WriteKind.SET_CREDENTIALS
    assert cmd.token == "ghp_abc"


def test_copilot_login_token_stdin(monkeypatch, capsys):
    monkeypatch.delenv("MEATPROXY_CONFIG", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO("tok123\n"))
    client = make_client(FakeStatus(True, login="octocat", host="github.com"))
    install(monkeypatch, client)

    assert cli.main(["copilot", "login", "--token-stdin"]) == 0

    cmd = client.accounts_api.set_calls[0].command
    assert cmd.token == "tok123"


def test_copilot_login_token_empty_stdin(monkeypatch, capsys):
    monkeypatch.delenv("MEATPROXY_CONFIG", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO("\n"))
    client = make_client(FakeStatus(False))
    install(monkeypatch, client)

    assert cli.main(["copilot", "login", "--token-stdin"]) == 2
    assert "no token" in capsys.readouterr().err


def test_copilot_login_device_code(monkeypatch, capsys):
    monkeypatch.delenv("MEATPROXY_CONFIG", raising=False)
    client = make_client(
        FakeStatus(False), steps=[_device_code_step(), _completed_step()]
    )
    install(monkeypatch, client)

    assert cli.main(["copilot", "login", "--yes"]) == 0

    out = capsys.readouterr().out
    assert "ABCD-1234" in out
    assert "Logged in as octocat" in out
    assert client.stopped is True


def test_copilot_login_failure(monkeypatch, capsys):
    monkeypatch.delenv("MEATPROXY_CONFIG", raising=False)
    client = make_client(
        FakeStatus(False),
        steps=[
            _device_code_step(),
            FakeStep(StepKind.ERROR, message="device code expired"),
        ],
    )
    install(monkeypatch, client)

    assert cli.main(["copilot", "login", "--yes"]) == 1
    assert "device code expired" in capsys.readouterr().err


# -- copilot logout --------------------------------------------------------------------


def test_copilot_logout(monkeypatch, capsys):
    monkeypatch.delenv("MEATPROXY_CONFIG", raising=False)
    account = FakeAccountStatus(True, "github.com", "octocat", account_id="acc-1")
    client = make_client(FakeStatus(False), accounts=[account])
    install(monkeypatch, client)

    assert cli.main(["copilot", "logout"]) == 0

    out = capsys.readouterr().out
    assert "Logged out." in out
    cmd = client.accounts_api.set_calls[0].command
    assert cmd.kind is WriteKind.LOGOUT


def test_copilot_logout_when_not_logged_in(monkeypatch, capsys):
    monkeypatch.delenv("MEATPROXY_CONFIG", raising=False)
    client = make_client(FakeStatus(False), accounts=[])
    install(monkeypatch, client)

    assert cli.main(["copilot", "logout"]) == 0
    assert "nothing to do" in capsys.readouterr().out
