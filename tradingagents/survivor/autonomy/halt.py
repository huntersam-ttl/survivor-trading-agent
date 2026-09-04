"""External kill switch: ~/.tradingagents/survivor/HALT.

When the HALT file exists, autonomous cycles do nothing: no scan, no research,
no AI calls, no proposals, no paper executions. Status commands keep working.
`survivor-resume` only removes the PAPER autonomy halt — it can never enable
live trading (no code path exists).
"""

from __future__ import annotations

import os

HALT_DIR = os.path.join(os.path.expanduser("~"), ".tradingagents", "survivor")
HALT_PATH = os.path.join(HALT_DIR, "HALT")


def is_halted(halt_path: str | None = None) -> bool:
    return os.path.exists(halt_path or HALT_PATH)


def set_halt(halt_path: str | None = None) -> str:
    path = halt_path or HALT_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("SURVIVOR PAPER AUTONOMY HALT\n")
    return path


def clear_halt(halt_path: str | None = None) -> bool:
    """Remove the paper-autonomy halt. NEVER enables live trading."""
    path = halt_path or HALT_PATH
    if os.path.exists(path):
        os.remove(path)
        return True
    return False
