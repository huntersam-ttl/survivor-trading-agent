"""Append-only, hash-chained SQLite ledger for paper trading.

Every event stores previous_hash and event_hash = sha256(canonical JSON of
all fields including previous_hash). Recovery verifies the chain and the
accounting invariants; corruption fails closed (no silent repair).

A partial UNIQUE index on proposal_id for TRADE_EXECUTED events enforces
duplicate/replay protection at the storage layer. For MARK_UPDATED events the
column ``execution_price_pence`` carries the conservative mark (bid).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from tradingagents.survivor.execution.portfolio import Portfolio

DEFAULT_PAPER_DB_PATH = os.path.join(
    os.path.expanduser("~"), ".tradingagents", "survivor", "paper.db"
)

_COLUMNS = (
    "event_id", "timestamp_utc", "run_id", "symbol", "event_type", "side",
    "quantity", "execution_price_pence", "notional_pence", "fee_pence",
    "slippage_pence", "cash_before", "cash_after", "exposure_before",
    "exposure_after", "equity_before", "equity_after", "risk_decision",
    "risk_reason", "proposal_id", "previous_hash",
)


class LedgerCorruptionError(Exception):
    """Raised when the ledger chain or replayed accounting is inconsistent."""


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class PaperLedger:
    def __init__(self, db_path: str | None = None):
        self.db_path = str(db_path or DEFAULT_PAPER_DB_PATH)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp_utc TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    side TEXT,
                    quantity INTEGER NOT NULL,
                    execution_price_pence INTEGER NOT NULL,
                    notional_pence INTEGER NOT NULL,
                    fee_pence INTEGER NOT NULL,
                    slippage_pence INTEGER NOT NULL,
                    cash_before INTEGER NOT NULL,
                    cash_after INTEGER NOT NULL,
                    exposure_before INTEGER NOT NULL,
                    exposure_after INTEGER NOT NULL,
                    equity_before INTEGER NOT NULL,
                    equity_after INTEGER NOT NULL,
                    risk_decision TEXT,
                    risk_reason TEXT,
                    proposal_id TEXT,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_exec_proposal "
                "ON paper_events(proposal_id) WHERE event_type = 'TRADE_EXECUTED';"
            )

    # --- append -----------------------------------------------------------------
    def append_event(self, **kwargs: Any) -> dict:
        """Append one event. Fills ids/timestamps and hashes the chain."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT event_hash FROM paper_events ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            previous_hash = row["event_hash"] if row else "GENESIS"
        event = {col: kwargs.get(col) for col in _COLUMNS}
        event["event_id"] = event["event_id"] or str(uuid.uuid4())
        event["timestamp_utc"] = event["timestamp_utc"] or datetime.now(timezone.utc).isoformat()
        for key in (
            "quantity", "execution_price_pence", "notional_pence", "fee_pence",
            "slippage_pence", "cash_before", "cash_after", "exposure_before",
            "exposure_after", "equity_before", "equity_after",
        ):
            event[key] = int(event[key] or 0)
        event["previous_hash"] = previous_hash
        event["event_hash"] = hashlib.sha256(_canonical(event).encode("utf-8")).hexdigest()

        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO paper_events ({', '.join(_COLUMNS)}, event_hash) "
                f"VALUES ({', '.join('?' for _ in range(len(_COLUMNS) + 1))})",
                [event[col] for col in _COLUMNS] + [event["event_hash"]],
            )
        return event

    # --- queries ----------------------------------------------------------------
    def events(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {', '.join(_COLUMNS)}, event_hash FROM paper_events ORDER BY rowid"
            ).fetchall()
        return [dict(r) for r in rows]

    def verify_chain(self) -> None:
        """Verify hash links and hashes; raise LedgerCorruptionError on any mismatch."""
        previous = "GENESIS"
        for event in self.events():
            if event["previous_hash"] != previous:
                raise LedgerCorruptionError(
                    f"chain broken at {event['event_id']}: expected previous {previous}, "
                    f"found {event['previous_hash']}"
                )
            recomputed = hashlib.sha256(
                _canonical({k: event[k] for k in _COLUMNS}).encode("utf-8")
            ).hexdigest()
            if recomputed != event["event_hash"]:
                raise LedgerCorruptionError(f"hash mismatch at event {event['event_id']}")
            previous = event["event_hash"]

    def has_executed(self, proposal_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM paper_events WHERE proposal_id = ? AND event_type = 'TRADE_EXECUTED'",
                (proposal_id,),
            ).fetchone()
        return row is not None

    def pnl_today(self, now: datetime | None = None) -> int:
        """Realized + unrealized P/L so far today (UTC): sum of equity deltas."""
        day = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT SUM(equity_after - equity_before) AS total FROM paper_events "
                "WHERE timestamp_utc LIKE ? AND event_type IN ('TRADE_EXECUTED', 'MARK_UPDATED')",
                (f"{day}%",),
            ).fetchone()
        return int(row["total"] or 0)

    # --- recovery ---------------------------------------------------------------
    def recover_portfolio(self) -> Portfolio:
        """Rebuild portfolio state from the ledger, verifying chain + invariants.

        Fails closed with LedgerCorruptionError; never silently repairs.
        """
        self.verify_chain()
        portfolio = Portfolio()
        for event in self.events():
            etype = event["event_type"]
            if etype == "TREASURY_INITIALIZED":
                portfolio.cash_pence = event["cash_after"]
                portfolio.starting_equity_pence = event["cash_after"]
                portfolio.high_water_mark_pence = event["equity_after"]
            elif etype == "TRADE_EXECUTED":
                portfolio.buy(
                    event["symbol"], event["quantity"], event["execution_price_pence"],
                    event["fee_pence"], now=event["timestamp_utc"],
                )
            elif etype == "MARK_UPDATED":
                mark = event["execution_price_pence"]
                position = portfolio.positions.get(event["symbol"])
                if position is not None and mark > 0:
                    position.current_mark_pence = mark
                    equity = portfolio.equity_pence
                    if equity > portfolio.high_water_mark_pence:
                        portfolio.high_water_mark_pence = equity
            # invariants checked at every step
            if portfolio.cash_pence < 0:
                raise LedgerCorruptionError(f"negative cash at {event['event_id']}")
            if any(p.quantity < 0 for p in portfolio.positions.values()):
                raise LedgerCorruptionError(f"negative holdings at {event['event_id']}")
        portfolio.realized_pnl_today_pence = self.pnl_today()
        return portfolio


