"""Filesystem roots for mutable Survivor runtime state."""

from __future__ import annotations

import os


def runtime_dir() -> str:
    """Return the configured survivor state root, or the normal user root."""
    configured = os.environ.get("SURVIVOR_RUNTIME_DIR", "").strip()
    return configured or os.path.join(os.path.expanduser("~"), ".tradingagents", "survivor")
