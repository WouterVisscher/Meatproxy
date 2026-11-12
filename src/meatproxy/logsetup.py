"""Structured (JSON lines) logging."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        kv = getattr(record, "kv", None)
        if kv:
            payload.update(kv)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "info") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    # quiet down chatty libraries
    for name in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(name).setLevel(logging.WARNING)


def emit(logger: logging.Logger, level: int, msg: str, **kv: object) -> None:
    """Log with structured fields, e.g. emit(log, logging.INFO, "routed", model=x)."""
    record = logger.makeRecord(
        logger.name, level, __file__, 0, msg, (), None
    )
    record.kv = kv  # type: ignore[attr-defined]
    logger.handle(record)
