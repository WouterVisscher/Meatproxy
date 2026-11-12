"""meatproxy command line interface (argparse; no extra dependencies).

    meatproxy [serve]              run the proxy (default when no subcommand)
    meatproxy copilot login        sign in to GitHub Copilot
    meatproxy copilot status       show the Copilot sign-in state
    meatproxy copilot logout       sign out of GitHub Copilot
    meatproxy config check         validate the config file
    meatproxy version              print the version

Flags override the MEATPROXY_* environment variables. For the copilot
commands the config file is optional: without it the SDK default Copilot
home (~/.copilot) is used.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from . import __version__
from . import copilot_auth
from .config import ConfigError, load_config
from .logsetup import setup_logging

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meatproxy",
        description="A small, unbloated LLM router.",
    )
    sub = parser.add_subparsers(dest="command")

    serve_p = sub.add_parser("serve", help="run the proxy (default)")
    serve_p.add_argument(
        "--config",
        default=None,
        help="config file (default: $MEATPROXY_CONFIG or meatproxy.yaml)",
    )
    serve_p.add_argument(
        "--host", default=None, help="bind host (default: config api.host)"
    )
    serve_p.add_argument(
        "--port", type=int, default=None, help="bind port (default: $MEATPROXY_PORT or config api.port)"
    )
    serve_p.add_argument(
        "--log-level",
        default=None,
        help="log level (default: $MEATPROXY_LOG_LEVEL or info)",
    )

    cop_p = sub.add_parser("copilot", help="GitHub Copilot sign-in")
    cop_sub = cop_p.add_subparsers(dest="copilot_command", required=True)

    login_p = cop_sub.add_parser("login", help="sign in to GitHub Copilot")
    login_p.add_argument(
        "--web",
        action="store_true",
        help="browser sign-in instead of the device-code flow",
    )
    login_p.add_argument(
        "--token",
        default=None,
        help="install a GitHub token instead of interactive login",
    )
    login_p.add_argument(
        "--token-stdin",
        action="store_true",
        help="read the token from stdin (one line)",
    )
    login_p.add_argument(
        "--host",
        default="github.com",
        help="GitHub host for --token / --token-stdin (default: github.com)",
    )
    login_p.add_argument(
        "--yes",
        action="store_true",
        help="assume yes to the plaintext-storage consent prompt "
        "(no interactive prompts; for containers)",
    )
    login_p.add_argument(
        "--base-directory",
        default=None,
        help="Copilot state directory (default: config copilot.base_directory)",
    )
    login_p.add_argument(
        "--config",
        default=None,
        help="config file (default: $MEATPROXY_CONFIG; optional here)",
    )

    status_p = cop_sub.add_parser("status", help="show the Copilot sign-in state")
    status_p.add_argument(
        "--base-directory",
        default=None,
        help="Copilot state directory (default: config copilot.base_directory)",
    )
    status_p.add_argument(
        "--config",
        default=None,
        help="config file (default: $MEATPROXY_CONFIG; optional here)",
    )

    logout_p = cop_sub.add_parser("logout", help="sign out of GitHub Copilot")
    logout_p.add_argument(
        "--account",
        default=None,
        help="account login to sign out (default: the active one)",
    )
    logout_p.add_argument(
        "--base-directory",
        default=None,
        help="Copilot state directory (default: config copilot.base_directory)",
    )
    logout_p.add_argument(
        "--config",
        default=None,
        help="config file (default: $MEATPROXY_CONFIG; optional here)",
    )

    cfg_p = sub.add_parser("config", help="config utilities")
    cfg_sub = cfg_p.add_subparsers(dest="config_command", required=True)
    check_p = cfg_sub.add_parser("check", help="validate the config file")
    check_p.add_argument(
        "--config",
        default=None,
        help="config file (default: $MEATPROXY_CONFIG or meatproxy.yaml)",
    )

    sub.add_parser("version", help="print the version")
    return parser


def _resolve_base_directory(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Resolve the Copilot state dir for copilot commands.

    Returns (base_directory, warning). The config is optional here: without
    it (or when it is missing/invalid) the SDK default (~/.copilot) applies.
    """
    if getattr(args, "base_directory", None):
        return args.base_directory, None
    config_path = args.config or os.environ.get("MEATPROXY_CONFIG")
    if config_path:
        if Path(config_path).is_file():
            try:
                cfg = load_config(config_path)
                return cfg.copilot.base_directory, None
            except ConfigError as e:
                return None, f"warning: ignoring invalid config {config_path}: {e}"
        return None, (
            f"warning: config {config_path} not found; "
            "using the default Copilot home (~/.copilot)"
        )
    return None, None


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> int:
    setup_logging(args.log_level or os.environ.get("MEATPROXY_LOG_LEVEL", "info"))
    from .main import serve

    config_path = args.config or os.environ.get("MEATPROXY_CONFIG", "meatproxy.yaml")
    serve(config_path, host=args.host, port=args.port)
    return EXIT_OK


def _cmd_copilot_login(args: argparse.Namespace, base_directory: str | None) -> int:
    if args.token_stdin:
        token = sys.stdin.read().strip()
    elif args.token is not None:
        token = args.token.strip()
    else:
        token = None
    if token is not None:
        if not token:
            print("error: no token provided", file=sys.stderr)
            return EXIT_USAGE

        async def run() -> str:
            async with copilot_auth.auth_client(base_directory) as client:
                return await copilot_auth.install_token(client, token, host=args.host)

        login_name = asyncio.run(run())
        print(f"Token installed; logged in as {login_name}.")
        return EXIT_OK

    async def run() -> str:
        async with copilot_auth.auth_client(base_directory) as client:
            return await copilot_auth.login(
                client, web=args.web, assume_yes=args.yes
            )

    login_name = asyncio.run(run())
    print(f"Logged in as {login_name}.")
    print("Copilot models are now available to a running meatproxy.")
    return EXIT_OK


def _cmd_copilot_status(args: argparse.Namespace, base_directory: str | None) -> int:
    async def run() -> dict:
        async with copilot_auth.auth_client(base_directory) as client:
            return await copilot_auth.get_status(client)

    status = asyncio.run(run())
    if status["authenticated"]:
        print(f"Logged in as {status['login']} ({status['host']})")
        return EXIT_OK
    print("Not logged in. Run 'meatproxy copilot login' to sign in.")
    return EXIT_ERROR


def _cmd_copilot_logout(args: argparse.Namespace, base_directory: str | None) -> int:
    async def run() -> bool:
        async with copilot_auth.auth_client(base_directory) as client:
            return await copilot_auth.logout(client, account=args.account)

    if asyncio.run(run()):
        print("Logged out.")
    else:
        print("Not logged in; nothing to do.")
    return EXIT_OK


def _cmd_copilot(args: argparse.Namespace) -> int:
    base_directory, warning = _resolve_base_directory(args)
    if warning:
        print(warning, file=sys.stderr)
    if args.copilot_command == "login":
        return _cmd_copilot_login(args, base_directory)
    if args.copilot_command == "status":
        return _cmd_copilot_status(args, base_directory)
    if args.copilot_command == "logout":
        return _cmd_copilot_logout(args, base_directory)
    raise AssertionError(f"unhandled copilot command {args.copilot_command!r}")


def _cmd_config(args: argparse.Namespace) -> int:
    if args.config_command != "check":
        raise AssertionError(f"unhandled config command {args.config_command!r}")
    config_path = args.config or os.environ.get("MEATPROXY_CONFIG", "meatproxy.yaml")
    try:
        cfg = load_config(config_path)
    except ConfigError as e:
        print(f"invalid: {e}", file=sys.stderr)
        return EXIT_USAGE
    print(f"config ok: {config_path}")
    for name, spec in sorted(cfg.models.items()):
        print(f"  model {name}: tier={spec.tier} kind={spec.kind}")
    for tier, names in sorted(cfg.chains.items()):
        print(f"  chain {tier}: {' -> '.join(names)}")
    print(
        f"  copilot base_directory: "
        f"{cfg.copilot.base_directory or '(SDK default ~/.copilot)'}"
    )
    return EXIT_OK


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        # no subcommand: serve (backward compatible with bare `meatproxy`)
        args = parser.parse_args(["serve", *(argv or [])])
    command = args.command or "serve"
    try:
        if command == "serve":
            return _cmd_serve(args)
        if command == "copilot":
            return _cmd_copilot(args)
        if command == "config":
            return _cmd_config(args)
        if command == "version":
            print(f"meatproxy {__version__}")
            return EXIT_OK
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except copilot_auth.CopilotAuthError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    raise AssertionError(f"unhandled command {command!r}")


if __name__ == "__main__":
    sys.exit(main())
