"""Fake github-copilot-sdk for copilot_auth / cli tests.

Monkeypatched into ``copilot_auth._auth_sdk`` so no real Copilot runtime
or subscription is needed. The fake login flow is a scripted list of
steps: the first is returned by ``begin``, the rest by successive
``advance`` calls.
"""

from __future__ import annotations

import enum
from typing import Any

import meatproxy.copilot_auth as cauth


# -- enum stand-ins -------------------------------------------------------------


class StepKind(enum.Enum):
    DEVICE_CODE = "device-code"
    OPEN_URL = "open-url"
    NEEDS_INTERACTION = "needs-interaction"
    INPUT_REQUIRED = "input-required"
    AWAITING = "awaiting"
    COMPLETED = "completed"
    ERROR = "error"


class ResultStatus(enum.Enum):
    COMPLETED = "completed"
    DECLINED = "declined"
    NEEDS_PLAINTEXT_CONSENT = "needs-plaintext-consent"
    NEEDS_ACCOUNT_SELECTION = "needs-account-selection"


class WriteKind(enum.Enum):
    LOGOUT = "logout"
    SET_CREDENTIALS = "setCredentials"
    SWITCH_ACTIVE = "switchActive"


class QueryKind(enum.Enum):
    ACCOUNTS = "accounts"
    LOGIN_ACCOUNTS = "loginAccounts"
    PROVIDERS = "providers"


class ProviderKind(enum.Enum):
    GITHUB_DOT_COM = "githubDotCom"
    ENTRA = "entra"
    PROXIMA = "proxima"


class Method(enum.Enum):
    DEVICE_CODE = "device-code"
    WEB = "web"


# -- request / response stand-ins ----------------------------------------------


class Req:
    """Generic request: records whatever kwargs it was built with."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class FakeStep:
    def __init__(
        self,
        kind: StepKind,
        url: str | None = None,
        prompt: str | None = None,
        result: "FakeResult | None" = None,
        message: str | None = None,
        user_code: str | None = None,
        verification_uri: str | None = None,
        expires_in: int | None = None,
    ) -> None:
        self.kind = kind
        self.url = url
        self.prompt = prompt
        self.result = result
        self.message = message
        self.user_code = user_code
        self.verification_uri = verification_uri
        self.expires_in = expires_in


class FakeResult:
    def __init__(
        self,
        status: ResultStatus,
        host: str | None = None,
        login: str | None = None,
        accounts: list["FakeLoginAccount"] | None = None,
    ) -> None:
        self.status = status
        self.host = host
        self.login = login
        self.accounts = accounts


class FakeLoginAccount:
    """Account choice during needs-account-selection."""

    def __init__(self, host: str, kind: str, login: str, selection_id: str) -> None:
        self.host = host
        self.kind = kind
        self.login = login
        self.selection_id = selection_id


class FakeAccountStatus:
    """Signed-in account entry from accounts.enumerate."""

    def __init__(
        self,
        active: bool,
        host: str,
        login: str,
        account_id: str | None = None,
    ) -> None:
        self.active = active
        self.host = host
        self.kind = "github"
        self.login = login
        self.account_id = account_id


class FakeBegun:
    def __init__(self, flow_id: str, step: FakeStep) -> None:
        self.flow_id = flow_id
        self.step = step


class FakeWriteResult:
    def __init__(self, ok: bool, more_users: bool | None = None) -> None:
        self.ok = ok
        self.more_users = more_users


class FakeValueAccounts:
    def __init__(self, items: list[FakeAccountStatus]) -> None:
        self.items = items


class FakeStatus:
    """Client-level auth.getStatus response."""

    def __init__(
        self,
        authenticated: bool,
        login: str | None = None,
        host: str | None = None,
        status_message: str | None = None,
    ) -> None:
        self.isAuthenticated = authenticated
        self.login = login
        self.host = host
        self.statusMessage = status_message


# -- fake RPC surface ------------------------------------------------------------


class FakeLoginApi:
    def __init__(self, steps: list[FakeStep]) -> None:
        # the first step is served by begin(); the rest by advance()
        self.first = steps[0] if steps else None
        self.steps = list(steps[1:])
        self.begin_params: Req | None = None
        self.advanced: list[Req] = []
        self.cancelled: list[str] = []

    async def begin(self, params: Req) -> FakeBegun:
        self.begin_params = params
        return FakeBegun("flow-1", self.first)

    async def advance(self, params: Req) -> FakeStep:
        self.advanced.append(params)
        if not self.steps:
            raise AssertionError("advance called with no scripted step left")
        return self.steps.pop(0)

    async def cancel(self, params: Req) -> None:
        self.cancelled.append(params.flow_id)


class FakeAccountsApi:
    def __init__(
        self,
        login_api: FakeLoginApi,
        accounts: list[FakeAccountStatus],
        set_ok: bool = True,
    ) -> None:
        self.login = login_api
        self.accounts = accounts
        self.set_ok = set_ok
        self.enumerate_calls: list[Req] = []
        self.set_calls: list[Req] = []

    async def enumerate(self, params: Req) -> FakeValueAccounts:
        self.enumerate_calls.append(params)
        return FakeValueAccounts(self.accounts)

    async def set(self, params: Req) -> FakeWriteResult:
        self.set_calls.append(params)
        return FakeWriteResult(self.set_ok)


class FakeRpc:
    def __init__(self, accounts_api: FakeAccountsApi) -> None:
        self.accounts = accounts_api


class FakeSession:
    def __init__(self, accounts_api: FakeAccountsApi) -> None:
        self._rpc = FakeRpc(accounts_api)
        self.disconnected = False

    @property
    def rpc(self) -> FakeRpc:
        return self._rpc

    async def disconnect(self) -> None:
        self.disconnected = True


class FakeClient:
    def __init__(
        self,
        accounts_api: FakeAccountsApi,
        status: FakeStatus,
    ) -> None:
        self.accounts_api = accounts_api
        self._status = status
        self.kwargs: dict[str, Any] | None = None
        self.stopped = False
        self.sessions: list[FakeSession] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        self.stopped = True

    async def create_session(self, **kwargs: Any) -> FakeSession:
        session = FakeSession(self.accounts_api)
        self.sessions.append(session)
        return session

    async def get_auth_status(self) -> FakeStatus:
        return self._status


# -- install --------------------------------------------------------------------


def install(
    monkeypatch,
    client: FakeClient,
    captured_kwargs: list[dict[str, Any]] | None = None,
) -> None:
    """Monkeypatch copilot_auth._auth_sdk with the fakes; the fake
    CopilotClient is the given client (kwargs captured for assertions)."""

    def fake_client_class(**kwargs: Any) -> FakeClient:
        if captured_kwargs is not None:
            captured_kwargs.append(kwargs)
        client.kwargs = kwargs
        return client

    monkeypatch.setattr(
        cauth,
        "_auth_sdk",
        {
            "CopilotClient": fake_client_class,
            "AccountsEnumerateRequest": Req,
            "AccountsSetRequest": Req,
            "AuthEnumerateQuery": Req,
            "AuthEnumerateQueryKind": QueryKind,
            "AuthLoginAdvanceRequest": Req,
            "AuthLoginBeginRequest": Req,
            "AuthLoginCancelRequest": Req,
            "AuthLoginMethod": Method,
            "AuthLoginResultStatus": ResultStatus,
            "AuthLoginStepKind": StepKind,
            "AuthWrite": Req,
            "AuthWriteKind": WriteKind,
            "LoginProviderKind": ProviderKind,
        },
    )


def make_client(
    status: FakeStatus,
    steps: list[FakeStep] | None = None,
    accounts: list[FakeAccountStatus] | None = None,
    set_ok: bool = True,
) -> FakeClient:
    """Convenience: a FakeClient with a scripted login flow and account list."""
    login_api = FakeLoginApi(steps or [])
    accounts_api = FakeAccountsApi(login_api, accounts or [], set_ok=set_ok)
    return FakeClient(accounts_api, status)


def consent_inputs(login_api: FakeLoginApi) -> list[str]:
    """The advance inputs that carried a consent/account decision."""
    return [p.input for p in login_api.advanced if getattr(p, "input", None)]
