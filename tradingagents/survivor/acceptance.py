"""Durable one-shot reservation primitives for operator paper acceptance.

This module deliberately does not dispatch cycles or alter HALT/opt-in. A
DISPATCHED record is never replayable; after a crash it must be reconciled.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tradingagents.survivor.paths import runtime_dir

DEFAULT_ACCEPTANCE_DB = os.path.join(runtime_dir(), "paper-acceptance.db")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
_TERMINAL = {"COMPLETED", "REJECTED", "UNCERTAIN", "RECONCILIATION_REQUIRED"}


class AcceptanceConflict(ValueError):
    pass


@dataclass(frozen=True)
class AcceptanceRecord:
    acceptance_id: str
    fingerprint: str
    status: str
    result: dict[str, Any]


class PaperAcceptanceStore:
    """SQLite reservation serialized by BEGIN IMMEDIATE across processes."""

    def __init__(self, path: str | os.PathLike[str] = DEFAULT_ACCEPTANCE_DB):
        self.path = str(path)
        Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        # Simultaneous first-open can contend while SQLite changes journal
        # mode. The default rollback journal is still transaction-safe; a
        # later connection will enable WAL once the competing initializer exits.
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                raise
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _init(self):
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS paper_acceptance (
                acceptance_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
                status TEXT NOT NULL, result_json TEXT NOT NULL DEFAULT '{}',
                created_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                updated_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            )""")

    @staticmethod
    def fingerprint(request: dict[str, Any]) -> str:
        if not isinstance(request, dict):
            raise AcceptanceConflict("acceptance request must be an object")
        try:
            encoded = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            raise AcceptanceConflict("acceptance request is not canonical JSON") from None
        return hashlib.sha256(encoded.encode()).hexdigest()

    def reserve(self, acceptance_id: str, request: dict[str, Any]) -> tuple[AcceptanceRecord, bool]:
        if not isinstance(acceptance_id, str) or not _ID.fullmatch(acceptance_id):
            raise AcceptanceConflict("invalid operator acceptance ID")
        fingerprint = self.fingerprint(request)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM paper_acceptance WHERE acceptance_id=?", (acceptance_id,)).fetchone()
            if row:
                if row["fingerprint"] != fingerprint:
                    db.rollback()
                    raise AcceptanceConflict("acceptance ID is already bound to a different request")
                db.commit()
                return self._record(row), False
            db.execute("INSERT INTO paper_acceptance(acceptance_id,fingerprint,status) VALUES(?,?,?)",
                       (acceptance_id, fingerprint, "RESERVED"))
            db.commit()
        return AcceptanceRecord(acceptance_id, fingerprint, "RESERVED", {}), True

    def transition(self, acceptance_id: str, expected: set[str], status: str,
                   result: dict[str, Any] | None = None) -> AcceptanceRecord:
        if status not in {"AUTHORIZED", "DISPATCHED", *_TERMINAL}:
            raise AcceptanceConflict("invalid acceptance state")
        payload = json.dumps(result or {}, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM paper_acceptance WHERE acceptance_id=?", (acceptance_id,)).fetchone()
            if row is None:
                db.rollback(); raise AcceptanceConflict("acceptance reservation not found")
            if row["status"] == status and row["result_json"] == payload:
                db.commit(); return self._record(row)
            if row["status"] in _TERMINAL:
                db.rollback(); raise AcceptanceConflict("terminal acceptance state cannot transition")
            if row["status"] not in expected:
                db.rollback(); raise AcceptanceConflict("acceptance transition is not permitted")
            db.execute("UPDATE paper_acceptance SET status=?,result_json=?,updated_utc=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE acceptance_id=?",
                       (status, payload, acceptance_id))
            updated = db.execute("SELECT * FROM paper_acceptance WHERE acceptance_id=?", (acceptance_id,)).fetchone()
            db.commit()
            return self._record(updated)

    def recover_interrupted(self) -> list[AcceptanceRecord]:
        """Convert all persisted DISPATCHED work to uncertain; never retry it."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT * FROM paper_acceptance WHERE status='DISPATCHED'").fetchall()
            for row in rows:
                result = {"reason": "process_restart_after_dispatch", "operator_action_required": True}
                db.execute("UPDATE paper_acceptance SET status='RECONCILIATION_REQUIRED',result_json=?,updated_utc=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE acceptance_id=?",
                           (json.dumps(result, sort_keys=True), row["acceptance_id"]))
            db.commit()
        return [self.get(row["acceptance_id"]) for row in rows]

    def get(self, acceptance_id: str) -> AcceptanceRecord | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM paper_acceptance WHERE acceptance_id=?", (acceptance_id,)).fetchone()
        return self._record(row) if row else None

    @staticmethod
    def _record(row) -> AcceptanceRecord:
        return AcceptanceRecord(row["acceptance_id"], row["fingerprint"], row["status"], json.loads(row["result_json"]))
