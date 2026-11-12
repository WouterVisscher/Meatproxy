"""copilot_auth: login flow, consent, account selection, token, logout.

Uses the fake SDK (monkeypatched into copilot_auth._auth_sdk) so no real
Copilot runtime or subscription is needed.
"""

from __future__ import annotations

import pytest

import meatproxy.copilot_auth as cauth
from tests.fake_auth_sdk import (
    Method,
    ProviderKind,
    ResultStatus,
    StepKind,
    WriteKind,
    FakeAccountStatus,
    FakeLoginAccount,
    FakeResult,
    FakeStatus,
    FakeStep,
    consent_inputs,
    install,
    make_client,
)


def _device_code_step() -> FakeStep:
    return FakeStep(
        StepKind.DEVICE_CODE,
        user_code="ABCD-1234",
        verification_uri="https://github.com/login/device",
        expires_in=900,
    )


def _completed_step(login: str = "octocat") -> FakeStep:
    return FakeStep(
        StepKind.COMPLETED,
        result=FakeResult(ResultStatus.COMPLETED, host="github.com", login=login),
    )


# -- login -----------------------------------------------------------------------


async def test_login_device_code_success(monkeypatch):
    client = make_client(
        FakeStatus(False),
        steps=[_device_code_step(), _completed_step()],
    )
    install(monkeypatch, client)
    out: list[str] = []

    name = await cauth.login(client, open_browser=False, out=out.append)

    assert name == "octocat"
    assert any("ABCD-1234" in line for line in out)
    assert any("https://github.com/login/device" in line for line in out)
    login_api = client.accounts_api.login
    assert login_api.begin_params.kind is ProviderKind.GITHUB_DOT_COM
    assert login_api.begin_params.method is Method.DEVICE_CODE
    assert len(login_api.advanced) == 1
    assert login_api.cancelled == []
    assert client.sessions[0].disconnected is True


async def test_login_web_method(monkeypatch):
    client = make_client(
        FakeStatus(False),
        steps=[
            FakeStep(StepKind.OPEN_URL, url="https://github.com/login/oauth"),
            _completed_step(),
        ],
    )
    install(monkeypatch, client)

    name = await cauth.login(client, web=True, open_browser=False, out=lambda s: None)

    assert name == "octocat"
    assert (
        client.accounts_api.login.begin_params.method is Method.WEB
    )


async def test_login_consent_assume_yes(monkeypatch):
    client = make_client(
        FakeStatus(False),
        steps=[
            _device_code_step(),
            FakeStep(
                StepKind.AWAITING,
                result=FakeResult(ResultStatus.NEEDS_PLAINTEXT_CONSENT),
            ),
            _completed_step(),
        ],
    )
    install(monkeypatch, client)

    name = await cauth.login(
        client, assume_yes=True, open_browser=False, out=lambda s: None
    )

    assert name == "octocat"
    assert consent_inputs(client.accounts_api.login) == ["yes"]


async def test_login_consent_retries_then_fails(monkeypatch):
    """If the runtime keeps answering needs-plaintext-consent, the
    alternative token is tried once, then a clear error is raised."""
    stuck = FakeStep(
        StepKind.AWAITING, result=FakeResult(ResultStatus.NEEDS_PLAINTEXT_CONSENT)
    )
    client = make_client(
        FakeStatus(False),
        steps=[_device_code_step(), stuck, stuck, stuck],
    )
    install(monkeypatch, client)

    with pytest.raises(cauth.CopilotAuthError, match="plaintext-storage consent"):
        await cauth.login(
            client, assume_yes=True, open_browser=False, out=lambda s: None
        )

    login_api = client.accounts_api.login
    assert consent_inputs(login_api) == ["yes", "y"]
    assert login_api.cancelled == ["flow-1"]


async def test_login_consent_declined_interactive(monkeypatch):
    client = make_client(
        FakeStatus(False),
        steps=[
            _device_code_step(),
            FakeStep(
                StepKind.AWAITING,
                result=FakeResult(ResultStatus.NEEDS_PLAINTEXT_CONSENT),
            ),
            FakeStep(
                StepKind.COMPLETED, result=FakeResult(ResultStatus.DECLINED)
            ),
        ],
    )
    install(monkeypatch, client)

    with pytest.raises(cauth.LoginDeclined):
        await cauth.login(
            client,
            ask=lambda prompt: "n",
            open_browser=False,
            out=lambda s: None,
        )

    assert consent_inputs(client.accounts_api.login) == ["n"]


async def test_login_interactive_consent_not_understood(monkeypatch):
    """An interactive answer the runtime ignores fails with a clear error
    instead of looping forever."""
    stuck = FakeStep(
        StepKind.AWAITING, result=FakeResult(ResultStatus.NEEDS_PLAINTEXT_CONSENT)
    )
    client = make_client(FakeStatus(False), steps=[_device_code_step(), stuck, stuck])
    install(monkeypatch, client)

    with pytest.raises(cauth.CopilotAuthError, match="consent answer"):
        await cauth.login(
            client, ask=lambda prompt: "sure", open_browser=False, out=lambda s: None
        )


async def test_login_account_selection(monkeypatch):
    choices = [
        FakeLoginAccount("github.com", "github", "alice", "sel-1"),
        FakeLoginAccount("github.com", "github", "bob", "sel-2"),
    ]
    client = make_client(
        FakeStatus(False),
        steps=[
            _device_code_step(),
            FakeStep(
                StepKind.AWAITING,
                result=FakeResult(
                    ResultStatus.NEEDS_ACCOUNT_SELECTION, accounts=choices
                ),
            ),
            _completed_step("bob"),
        ],
    )
    install(monkeypatch, client)
    out: list[str] = []

    name = await cauth.login(
        client, ask=lambda prompt: "2", open_browser=False, out=out.append
    )

    assert name == "bob"
    assert consent_inputs(client.accounts_api.login) == ["sel-2"]
    assert any("alice" in line for line in out)
    assert any("bob" in line for line in out)


async def test_login_account_selection_assume_yes_single(monkeypatch):
    choices = [FakeLoginAccount("github.com", "github", "alice", "sel-1")]
    client = make_client(
        FakeStatus(False),
        steps=[
            _device_code_step(),
            FakeStep(
                StepKind.AWAITING,
                result=FakeResult(
                    ResultStatus.NEEDS_ACCOUNT_SELECTION, accounts=choices
                ),
            ),
            _completed_step("alice"),
        ],
    )
    install(monkeypatch, client)

    name = await cauth.login(
        client, assume_yes=True, open_browser=False, out=lambda s: None
    )

    assert name == "alice"
    assert consent_inputs(client.accounts_api.login) == ["sel-1"]


async def test_login_account_selection_assume_yes_multiple(monkeypatch):
    choices = [
        FakeLoginAccount("github.com", "github", "alice", "sel-1"),
        FakeLoginAccount("github.com", "github", "bob", "sel-2"),
    ]
    client = make_client(
        FakeStatus(False),
        steps=[
            _device_code_step(),
            FakeStep(
                StepKind.AWAITING,
                result=FakeResult(
                    ResultStatus.NEEDS_ACCOUNT_SELECTION, accounts=choices
                ),
            ),
        ],
    )
    install(monkeypatch, client)

    with pytest.raises(cauth.CopilotAuthError, match="without --yes"):
        await cauth.login(
            client, assume_yes=True, open_browser=False, out=lambda s: None
        )


async def test_login_error_step(monkeypatch):
    client = make_client(
        FakeStatus(False),
        steps=[
            _device_code_step(),
            FakeStep(StepKind.ERROR, message="device code expired"),
        ],
    )
    install(monkeypatch, client)

    with pytest.raises(cauth.CopilotAuthError, match="device code expired"):
        await cauth.login(client, open_browser=False, out=lambda s: None)

    assert client.accounts_api.login.cancelled == ["flow-1"]


async def test_login_input_required_step(monkeypatch):
    client = make_client(
        FakeStatus(False),
        steps=[
            FakeStep(StepKind.INPUT_REQUIRED, prompt="GitHub Enterprise host: "),
            _completed_step("ghe-user"),
        ],
    )
    install(monkeypatch, client)

    name = await cauth.login(
        client, ask=lambda prompt: "corp.ghe.com", open_browser=False, out=lambda s: None
    )

    assert name == "ghe-user"
    assert consent_inputs(client.accounts_api.login) == ["corp.ghe.com"]


# -- token / status / logout ------------------------------------------------------


async def test_install_token(monkeypatch):
    client = make_client(
        FakeStatus(True, login="octocat", host="github.com"),
        steps=[_completed_step()],
    )
    install(monkeypatch, client)

    name = await cauth.install_token(client, "ghp_123", host="github.com")

    assert name == "octocat"
    cmd = client.accounts_api.set_calls[0].command
    assert cmd.kind is WriteKind.SET_CREDENTIALS
    assert cmd.token == "ghp_123"
    assert cmd.host == "github.com"


async def test_install_token_rejected(monkeypatch):
    client = make_client(FakeStatus(False), steps=[_completed_step()], set_ok=False)
    install(monkeypatch, client)

    with pytest.raises(cauth.CopilotAuthError, match="rejected the token"):
        await cauth.install_token(client, "ghp_bogus")


async def test_get_status(monkeypatch):
    client = make_client(FakeStatus(True, login="octocat", host="github.com"))
    install(monkeypatch, client)

    status = await cauth.get_status(client)

    assert status == {
        "authenticated": True,
        "login": "octocat",
        "host": "github.com",
        "status_message": None,
    }


async def test_logout_active_account(monkeypatch):
    account = FakeAccountStatus(True, "github.com", "octocat", account_id="acc-1")
    client = make_client(FakeStatus(False), accounts=[account])
    install(monkeypatch, client)

    assert await cauth.logout(client) is True

    cmd = client.accounts_api.set_calls[0].command
    assert cmd.kind is WriteKind.LOGOUT
    assert cmd.account_id == "acc-1"
    assert cmd.login == "octocat"


async def test_logout_no_accounts(monkeypatch):
    client = make_client(FakeStatus(False), accounts=[])
    install(monkeypatch, client)

    assert await cauth.logout(client) is False
    assert client.accounts_api.set_calls == []


async def test_logout_unknown_account(monkeypatch):
    account = FakeAccountStatus(True, "github.com", "octocat", account_id="acc-1")
    client = make_client(FakeStatus(False), accounts=[account])
    install(monkeypatch, client)

    with pytest.raises(cauth.CopilotAuthError, match="no signed-in account"):
        await cauth.logout(client, account="nobody")


async def test_logout_multiple_requires_account(monkeypatch):
    # no active account to target: both signed in, neither active
    accounts = [
        FakeAccountStatus(False, "github.com", "alice", account_id="acc-1"),
        FakeAccountStatus(False, "github.com", "bob", account_id="acc-2"),
    ]
    client = make_client(FakeStatus(False), accounts=accounts)
    install(monkeypatch, client)

    with pytest.raises(cauth.CopilotAuthError, match="--account"):
        await cauth.logout(client)


# -- client kwargs -----------------------------------------------------------------


def test_client_kwargs():
    assert cauth.client_kwargs(None)["use_logged_in_user"] is True
    assert "base_directory" not in cauth.client_kwargs(None)
    assert cauth.client_kwargs("/data/copilot")["base_directory"] == "/data/copilot"
