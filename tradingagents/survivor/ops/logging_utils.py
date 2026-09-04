"""Structured JSON-line rotating logs with defensive secret redaction."""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import re

LOGS_DIR = os.path.join(os.path.expanduser("~"), ".tradingagents", "survivor", "logs")

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(api[_-]?key|token|secret|password|authorization)\s*[=:]\s*\S+", re.IGNORECASE),
)


def redact_secrets(text: str) -> str:
    """Defensively redact likely secrets before anything is logged."""
    result = text or ""
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_secrets(str(record.msg))
        if record.args:
            record.args = tuple(redact_secrets(str(a)) for a in record.args)
        return True


class JsonRotateLogger:
    """JSON-lines logger with size-based rotation (10MB x 5 backups default)."""

    def __init__(self, name: str, logs_dir: str | None = None,
                 max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5):
        self.logs_dir = logs_dir or LOGS_DIR
        os.makedirs(self.logs_dir, exist_ok=True)
        self.logger = logging.getLogger(f"survivor.{name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.handlers.RotatingFileHandler(
                os.path.join(self.logs_dir, f"{name}.log"),
                maxBytes=max_bytes, backupCount=backup_count,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            handler.addFilter(_RedactFilter())
            self.logger.addHandler(handler)

    def log(self, level: str, event: str, **fields) -> None:
        payload = {"level": level, "event": event, **fields}
        self.logger.info(json.dumps(payload, sort_keys=True, default=str))

    def info(self, event: str, **fields) -> None:
        self.log("INFO", event, **fields)

    def error(self, event: str, **fields) -> None:
        self.log("ERROR", event, **fields)
