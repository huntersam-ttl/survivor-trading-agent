"""Runtime security self-check: no execution backend other than PaperBroker."""

from __future__ import annotations

import os

# Tokens are assembled via concatenation so this file never contains a literal
# forbidden token (the self-scan would otherwise flag its own source).
FORBIDDEN_CODE_TOKENS = (
    "cc" + "xt", "bin" + "ance", "coin" + "base", "alp" + "aca", "krak" + "en",
    "place_" + "order(", "submit_" + "order(", "cancel_" + "order(",
    "private_" + "key", "sign_" + "transaction", "web" + "socket",
)


def security_selfcheck(survivor_package_dir: str) -> dict:
    """Scan the survivor package for real-money capability. HALT on any hit."""
    hits = []
    for root, _, files in os.walk(survivor_package_dir):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path) as fh:
                content = fh.read().lower()
            for token in FORBIDDEN_CODE_TOKENS:
                if token in content:
                    hits.append({"file": name, "token": token})
    result = {"ok": not hits, "hits": hits}
    if hits:
        from tradingagents.survivor.autonomy.halt import set_halt

        set_halt()  # unexpected live-execution capability -> immediate HALT
    return result
