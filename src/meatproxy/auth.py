"""API key authentication for the router itself.

The key is read from the environment (``MEATPROXY_API_KEY``) and never
stored in the YAML config. If unset, auth is disabled (dev mode) and a
loud warning is emitted at startup.
"""

from __future__ import annotations

import hmac
import logging
import os
from fastapi import HTTPException, Request

log = logging.getLogger(__name__)

API_KEY_ENV = "MEATPROXY_API_KEY"


class Auth:
    def __init__(self) -> None:
        self.key = os.environ.get(API_KEY_ENV, "").strip()
        if not self.key:
            log.warning(
                "%s is not set — meatproxy is running WITHOUT authentication. "
                "Never expose it on an untrusted network like this.",
                API_KEY_ENV,
            )

    def check(self, request: Request) -> None:
        if not self.key:
            return
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(status_code=401, detail="missing Bearer token")
        if not hmac.compare_digest(token.encode(), self.key.encode()):
            raise HTTPException(status_code=401, detail="invalid API key")
