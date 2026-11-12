"""GitHub Copilot login / status / logout, driven through the SDK's accounts RPC.

``meatproxy copilot login`` drives the Copilot CLI runtime's interactive
login API (``session.accounts.login.begin/advance/cancel``), so there is
no hand-rolled OIDC. Credentials are persisted under the same
``base_directory`` the server uses (the Copilot home), so a running
meatproxy picks them up with zero config changes.

The accounts RPC surface is *experimental* in the SDK; the symbols are
imported lazily and any future SDK that drops or renames them fails with
a clear error instead of an import traceback.
"""

from __future__ import annotations

import contextlib
import webbrowser
from contextlib import asynccontextmanager
from typing import Any, Callable

# Lazy import: the SDK is only needed for copilot auth commands, and
# importing it is expensive.
_auth_sdk: dict[str, Any] | None = None


def _sdk() -> dict[str, Any]:
    global _auth_sdk
    if _auth_sdk is None:
        try:
            from copilot import CopilotClient
            from copilot.rpc import (
                AccountsEnumerateRequest,
                AccountsSetRequest,
                AuthEnumerateQuery,
                AuthEnumerateQueryKind,
                AuthLoginAdvanceRequest,
                AuthLoginBeginRequest,
                AuthLoginCancelRequest,
                AuthLoginMethod,
                AuthLoginResultStatus,
                AuthLoginStepKind,
                AuthWrite,
                AuthWriteKind,
                LoginProviderKind,
            )
        except ImportError as e:
            raise CopilotAuthError(
                "the installed github-copilot-sdk does not expose the accounts "
                f"login API ({e}); upgrade the SDK or use 'copilot login --token'"
            ) from e
        _auth_sdk = {
            "CopilotClient": CopilotClient,
            "AccountsEnumerateRequest": AccountsEnumerateRequest,
            "AccountsSetRequest": AccountsSetRequest,
            "AuthEnumerateQuery": AuthEnumerateQuery,
            "AuthEnumerateQueryKind": AuthEnumerateQueryKind,
            "AuthLoginAdvanceRequest": AuthLoginAdvanceRequest,
            "AuthLoginBeginRequest": AuthLoginBeginRequest,
            "AuthLoginCancelRequest": AuthLoginCancelRequest,
            "AuthLoginMethod": AuthLoginMethod,
            "AuthLoginResultStatus": AuthLoginResultStatus,
            "AuthLoginStepKind": AuthLoginStepKind,
            "AuthWrite": AuthWrite,
            "AuthWriteKind": AuthWriteKind,
            "LoginProviderKind": LoginProviderKind,
        }
    return _auth_sdk


class CopilotAuthError(Exception):
    """User-facing Copilot auth failure."""


class LoginDeclined(CopilotAuthError):
    """The user declined the sign-in (e.g. refused plaintext storage)."""


# ---------------------------------------------------------------------------
# client lifecycle
# ---------------------------------------------------------------------------


def client_kwargs(base_directory: str | None) -> dict[str, Any]:
    """CopilotClient kwargs shared with the server adapter: same home, so
    credentials land where the running proxy expects them."""
    kwargs: dict[str, Any] = {
        "log_level": "info",
        "env": {"COPILOT_TELEMETRY_DISABLED": "true"},
        # logged-in-user credentials; the client starts fine while logged out
        "use_logged_in_user": True,
    }
    if base_directory:
        kwargs["base_directory"] = base_directory
    return kwargs


async def open_client(base_directory: str | None) -> Any:
    sdk = _sdk()
    client = sdk["CopilotClient"](**client_kwargs(base_directory))
    try:
        await client.start()
    except Exception:
        with contextlib.suppress(Exception):
            await client.stop()
        raise
    return client


@asynccontextmanager
async def auth_client(base_directory: str | None):
    client = await open_client(base_directory)
    try:
        yield client
    finally:
        with contextlib.suppress(Exception):
            await client.stop()


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


async def get_status(client: Any) -> dict[str, Any]:
    """Client-level auth status (``auth.getStatus``)."""
    st = await client.get_auth_status()
    return {
        "authenticated": bool(st.isAuthenticated),
        "login": st.login,
        "host": st.host,
        "status_message": st.statusMessage,
    }


# ---------------------------------------------------------------------------
# interactive login
# ---------------------------------------------------------------------------


def _maybe_open_browser(url: str | None) -> None:
    if not url:
        return
    with contextlib.suppress(Exception):
        webbrowser.open(url)


def _out(line: str) -> None:
    # flush: the login output is often redirected (captured code, containers)
    print(line, flush=True)


def _consent_decision(assume_yes: bool, ask: Callable[[str], str], tries: int) -> str:
    """The plaintext-consent answer to send via advance(input=...).

    The exact token the runtime expects is not documented, so the answer is
    verified by the resulting status: if the runtime still reports
    needs-plaintext-consent, one alternative token is tried, then the
    failure is surfaced clearly.
    """
    if assume_yes:
        if tries == 0:
            return "yes"
        if tries == 1:
            return "y"
        raise CopilotAuthError(
            "the runtime did not accept the plaintext-storage consent; "
            "re-run without --yes to answer interactively"
        )
    if tries > 0:
        raise CopilotAuthError(
            "the runtime did not accept the plaintext-storage consent answer; "
            "re-run 'meatproxy copilot login' and try again"
        )
    return ask("Store the auth token in plaintext (no keychain in this environment)? [y/N] ")


def _pick_account(
    accounts: list[Any],
    assume_yes: bool,
    ask: Callable[[str], str],
    out: Callable[[str], None],
) -> Any:
    if assume_yes:
        if len(accounts) == 1:
            return accounts[0]
        raise CopilotAuthError(
            f"{len(accounts)} accounts are available; re-run without --yes to choose"
        )
    for i, acc in enumerate(accounts, start=1):
        out(f"  {i}. {acc.login} ({acc.host})")
    while True:
        raw = ask(f"Select an account [1-{len(accounts)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(accounts):
            return accounts[int(raw) - 1]
        out(f"Invalid choice {raw!r}; enter a number between 1 and {len(accounts)}.")


async def login(
    client: Any,
    *,
    web: bool = False,
    assume_yes: bool = False,
    open_browser: bool = True,
    out: Callable[[str], None] = _out,
    ask: Callable[[str], str] = input,
) -> str:
    """Drive the interactive GitHub sign-in flow; return the login name.

    Default is the device-code flow (works from any terminal, including
    containers); ``web=True`` uses the browser loopback variant.
    """
    sdk = _sdk()
    method = sdk["AuthLoginMethod"].WEB if web else sdk["AuthLoginMethod"].DEVICE_CODE
    session = await client.create_session(available_tools=[])
    login_api = session.rpc.accounts.login
    begun = await login_api.begin(
        sdk["AuthLoginBeginRequest"](
            kind=sdk["LoginProviderKind"].GITHUB_DOT_COM, method=method
        )
    )
    flow_id = begun.flow_id
    step = begun.step
    consent_tries = 0
    try:
        while True:
            # pending decisions (consent / account selection) come back in
            # the step's result and take priority over the step kind
            result = step.result
            if result is not None:
                status = result.status
                if status == sdk["AuthLoginResultStatus"].COMPLETED:
                    return result.login or "unknown"
                if status == sdk["AuthLoginResultStatus"].DECLINED:
                    raise LoginDeclined(
                        "sign-in was declined; no credentials were stored. "
                        "Run 'meatproxy copilot login' again to try again."
                    )
                if status == sdk["AuthLoginResultStatus"].NEEDS_PLAINTEXT_CONSENT:
                    decision = _consent_decision(assume_yes, ask, consent_tries)
                    consent_tries += 1
                    step = await login_api.advance(
                        sdk["AuthLoginAdvanceRequest"](
                            flow_id=flow_id, input=decision
                        )
                    )
                    continue
                if status == sdk["AuthLoginResultStatus"].NEEDS_ACCOUNT_SELECTION:
                    pick = _pick_account(
                        result.accounts or [], assume_yes, ask, out
                    )
                    step = await login_api.advance(
                        sdk["AuthLoginAdvanceRequest"](
                            flow_id=flow_id, input=pick.selection_id
                        )
                    )
                    continue
                raise CopilotAuthError(
                    f"unexpected login result status: {status!r}"
                )

            kind = step.kind
            if kind == sdk["AuthLoginStepKind"].DEVICE_CODE:
                out(
                    f"Open {step.verification_uri} and enter the code "
                    f"{step.user_code}"
                )
                if step.expires_in:
                    out(f"(the code expires in {step.expires_in}s)")
                if open_browser:
                    _maybe_open_browser(step.verification_uri)
                step = await login_api.advance(
                    sdk["AuthLoginAdvanceRequest"](flow_id=flow_id)
                )
            elif kind == sdk["AuthLoginStepKind"].OPEN_URL:
                out(f"Open {step.url} to sign in")
                if open_browser:
                    _maybe_open_browser(step.url)
                step = await login_api.advance(
                    sdk["AuthLoginAdvanceRequest"](flow_id=flow_id)
                )
            elif kind == sdk["AuthLoginStepKind"].NEEDS_INTERACTION:
                out("Complete the sign-in in the window that opened; waiting here.")
                step = await login_api.advance(
                    sdk["AuthLoginAdvanceRequest"](flow_id=flow_id)
                )
            elif kind == sdk["AuthLoginStepKind"].INPUT_REQUIRED:
                answer = ask(step.prompt or "Input required: ")
                step = await login_api.advance(
                    sdk["AuthLoginAdvanceRequest"](flow_id=flow_id, input=answer)
                )
            elif kind == sdk["AuthLoginStepKind"].AWAITING:
                step = await login_api.advance(
                    sdk["AuthLoginAdvanceRequest"](flow_id=flow_id)
                )
            elif kind == sdk["AuthLoginStepKind"].ERROR:
                raise CopilotAuthError(step.message or "login failed")
            elif kind == sdk["AuthLoginStepKind"].COMPLETED:
                # a completed step always carries its result (handled above)
                raise CopilotAuthError("login completed without a result")
            else:  # pragma: no cover - defensive
                raise CopilotAuthError(f"unknown login step: {kind!r}")
            # loop back: the new step may carry a result
    except BaseException:
        with contextlib.suppress(Exception):
            await login_api.cancel(
                sdk["AuthLoginCancelRequest"](flow_id=flow_id)
            )
        raise
    finally:
        with contextlib.suppress(Exception):
            await session.disconnect()


# ---------------------------------------------------------------------------
# token install / logout
# ---------------------------------------------------------------------------


async def install_token(client: Any, token: str, host: str = "github.com") -> str:
    """Install a GitHub token via ``session.accounts.set``; return the login."""
    sdk = _sdk()
    session = await client.create_session(available_tools=[])
    try:
        result = await session.rpc.accounts.set(
            sdk["AccountsSetRequest"](
                command=sdk["AuthWrite"](
                    kind=sdk["AuthWriteKind"].SET_CREDENTIALS, host=host, token=token
                )
            )
        )
        if not result.ok:
            raise CopilotAuthError(
                f"the runtime rejected the token for host {host!r} "
                "(check that it is a valid GitHub token with Copilot access)"
            )
    finally:
        with contextlib.suppress(Exception):
            await session.disconnect()
    status = await get_status(client)
    return status["login"] or "unknown"


async def logout(client: Any, account: str | None = None) -> bool:
    """Log out the active account (or the one named ``account``).

    Returns True when an account was logged out, False when none was signed in.
    """
    sdk = _sdk()
    session = await client.create_session(available_tools=[])
    try:
        value = await session.rpc.accounts.enumerate(
            sdk["AccountsEnumerateRequest"](
                query=sdk["AuthEnumerateQuery"](
                    kind=sdk["AuthEnumerateQueryKind"].ACCOUNTS
                )
            )
        )
        items = value.items
        if not items:
            return False

        target = None
        if account is not None:
            for item in items:
                if item.login == account:
                    target = item
                    break
            if target is None:
                raise CopilotAuthError(f"no signed-in account named {account!r}")
        else:
            active = [i for i in items if i.active]
            if len(active) == 1:
                target = active[0]
            elif len(items) == 1:
                target = items[0]
            else:
                raise CopilotAuthError(
                    f"{len(items)} accounts are signed in; pick one with --account"
                )

        result = await session.rpc.accounts.set(
            sdk["AccountsSetRequest"](
                command=sdk["AuthWrite"](
                    kind=sdk["AuthWriteKind"].LOGOUT,
                    account_id=target.account_id,
                    host=target.host,
                    login=target.login,
                )
            )
        )
        if not result.ok:
            raise CopilotAuthError(f"logout failed for {target.login!r}")
        return True
    finally:
        with contextlib.suppress(Exception):
            await session.disconnect()
