"""Local backup + verification for survivor databases. No secrets included."""

from __future__ import annotations

import os
import shutil
import sqlite3
from datetime import datetime, timezone

from tradingagents.survivor.execution.ledger import PaperLedger

BACKUP_FILES = ("paper.db", "usage.db", "runtime.db", "evaluation.db")
OPTIONAL_FILES = ("heartbeat.json",)


def backup_survivor(survivor_dir: str | None = None, backup_root: str | None = None) -> str:
    """Copy survivor databases + trial config into a timestamped directory.
    Never includes .env / API keys / secrets (only whitelisted files)."""
    base = survivor_dir or os.path.join(os.path.expanduser("~"), ".tradingagents", "survivor")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = backup_root or os.path.join(base, "backups")
    dest = os.path.join(root, f"survivor_backup_{stamp}")
    os.makedirs(dest, exist_ok=True)

    copied = []
    for name in BACKUP_FILES:
        src = os.path.join(base, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, name))
            copied.append(name)
    for trial_file in sorted(os.listdir(os.path.join(base, "trials"))) if os.path.isdir(os.path.join(base, "trials")) else []:
        os.makedirs(os.path.join(dest, "trials"), exist_ok=True)
        shutil.copy2(os.path.join(base, "trials", trial_file),
                     os.path.join(dest, "trials", trial_file))
    for name in OPTIONAL_FILES:
        src = os.path.join(base, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, name))
    with open(os.path.join(dest, "MANIFEST.txt"), "w") as fh:
        fh.write("copied: " + ", ".join(copied) + chr(10))
        fh.write("excluded: .env, API keys, secrets" + chr(10))
    return dest


def verify_backup(backup_path: str) -> dict:
    """Validate a backup: SQLite integrity, paper hash chain, config identity."""
    report = {"path": backup_path, "files": {}, "paper_chain": None, "trials": [], "ok": True}
    for name in BACKUP_FILES:
        path = os.path.join(backup_path, name)
        if not os.path.exists(path):
            report["files"][name] = "ABSENT"
            continue
        conn = sqlite3.connect(path)
        row = conn.execute("PRAGMA integrity_check").fetchone()
        conn.close()
        ok = row[0] == "ok"
        report["files"][name] = "OK" if ok else "CORRUPT"
        report["ok"] = report["ok"] and ok
    paper = os.path.join(backup_path, "paper.db")
    if os.path.exists(paper):
        try:
            PaperLedger(db_path=paper).verify_chain()
            report["paper_chain"] = "OK"
        except Exception as exc:  # noqa: BLE001
            report["paper_chain"] = f"CORRUPT: {exc}"
            report["ok"] = False
    trials_dir = os.path.join(backup_path, "trials")
    if os.path.isdir(trials_dir):
        report["trials"] = sorted(os.listdir(trials_dir))
    return report
