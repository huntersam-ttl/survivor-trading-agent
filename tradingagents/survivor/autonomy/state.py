"""Autonomous-runtime state database (~/.tradingagents/survivor/runtime.db).

Tracks cycles, candidates, research runs, and point-in-time market
observations. Does NOT duplicate financial accounting (paper.db) or inference
spend (usage.db) — those remain the source of truth for their domains.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from tradingagents.survivor.markets.types import MarketSnapshot

DEFAULT_RUNTIME_DB_PATH = os.path.join(
    os.path.expanduser("~"), ".tradingagents", "survivor", "runtime.db"
)


def snapshot_data_hash(snapshot: MarketSnapshot) -> str:
    """Deterministic hash of the exact decision-time market snapshot."""
    payload = {
        "market_id": snapshot.market_id, "provider": snapshot.provider,
        "bid": snapshot.bid, "ask": snapshot.ask,
        "market_probability_bps": snapshot.market_probability_bps,
        "liquidity": snapshot.liquidity.minor_units if snapshot.liquidity else None,
        "volume_24h": snapshot.volume_24h.minor_units if snapshot.volume_24h else None,
        "close_time_utc": snapshot.close_time_utc,
        "quote_currency": snapshot.quote_currency,
        "source_timestamp_utc": snapshot.source_timestamp_utc,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class RuntimeState:
    def __init__(self, db_path: str | None = None):
        self.db_path = str(db_path or DEFAULT_RUNTIME_DB_PATH)
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
                CREATE TABLE IF NOT EXISTS cycles (
                    cycle_id TEXT PRIMARY KEY,
                    started_utc TEXT NOT NULL,
                    finished_utc TEXT,
                    status TEXT NOT NULL,
                    reason TEXT,
                    discovered INTEGER NOT NULL DEFAULT 0,
                    passed INTEGER NOT NULL DEFAULT 0,
                    research_candidates INTEGER NOT NULL DEFAULT 0,
                    researched INTEGER NOT NULL DEFAULT 0,
                    skipped_budget INTEGER NOT NULL DEFAULT 0,
                    proposals INTEGER NOT NULL DEFAULT 0,
                    approved INTEGER NOT NULL DEFAULT 0,
                    rejected INTEGER NOT NULL DEFAULT 0,
                    executed INTEGER NOT NULL DEFAULT 0,
                    ai_cost_pence INTEGER NOT NULL DEFAULT 0,
                    duration_sec REAL NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS candidates ("
                "cycle_id TEXT NOT NULL, market_id TEXT NOT NULL, rank INTEGER,"
                " score REAL, decision TEXT, status TEXT NOT NULL, reason TEXT)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    decision_time_utc TEXT NOT NULL,
                    bid INTEGER, ask INTEGER,
                    market_probability_bps INTEGER,
                    liquidity_minor INTEGER, liquidity_currency TEXT,
                    volume_24h_minor INTEGER, volume_24h_currency TEXT,
                    close_time_utc TEXT,
                    quote_currency TEXT NOT NULL,
                    data_hash TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS research_runs ("
                "id TEXT PRIMARY KEY, cycle_id TEXT NOT NULL, market_id TEXT NOT NULL,"
                " run_id TEXT, status TEXT NOT NULL, reason TEXT)"
            )

    # --- writes -----------------------------------------------------------------
    def start_cycle(self, cycle_id: str, now: datetime | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cycles (cycle_id, started_utc, status) VALUES (?, ?, 'RUNNING')",
                (cycle_id, (now or datetime.now(timezone.utc)).isoformat()),
            )

    def finish_cycle(self, cycle_id: str, status: str, reason: str | None = None,
                     metrics: dict[str, Any] | None = None, now: datetime | None = None) -> None:
        metrics = metrics or {}
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE cycles SET finished_utc = ?, status = ?, reason = ?,
                    discovered = ?, passed = ?, research_candidates = ?, researched = ?,
                    skipped_budget = ?, proposals = ?, approved = ?, rejected = ?,
                    executed = ?, ai_cost_pence = ?, duration_sec = ?
                WHERE cycle_id = ?
                """,
                (
                    (now or datetime.now(timezone.utc)).isoformat(), status, reason,
                    metrics.get("discovered", 0), metrics.get("passed", 0),
                    metrics.get("research_candidates", 0), metrics.get("researched", 0),
                    metrics.get("skipped_budget", 0), metrics.get("proposals", 0),
                    metrics.get("approved", 0), metrics.get("rejected", 0),
                    metrics.get("executed", 0), metrics.get("ai_cost_pence", 0),
                    metrics.get("duration_sec", 0.0), cycle_id,
                ),
            )

    def record_candidate(self, cycle_id: str, market_id: str, rank: int | None,
                         score: float | None, decision: str, status: str, reason: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO candidates (cycle_id, market_id, rank, score, decision, status, reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (cycle_id, market_id, rank, score, decision, status, reason),
            )

    def record_research(self, cycle_id: str, market_id: str, run_id: str | None,
                        status: str, reason: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO research_runs (id, cycle_id, market_id, run_id, status, reason) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), cycle_id, market_id, run_id, status, reason),
            )

    def record_observation(self, cycle_id: str, snapshot: MarketSnapshot,
                           decision_time_utc: str) -> None:
        """Preserve the exact decision-time snapshot (immutable evidence)."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO observations (
                    id, cycle_id, market_id, provider, timestamp_utc, decision_time_utc,
                    bid, ask, market_probability_bps, liquidity_minor, liquidity_currency,
                    volume_24h_minor, volume_24h_currency, close_time_utc, quote_currency, data_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()), cycle_id, snapshot.market_id, snapshot.provider,
                    snapshot.timestamp_utc, decision_time_utc,
                    snapshot.bid, snapshot.ask, snapshot.market_probability_bps,
                    snapshot.liquidity.minor_units if snapshot.liquidity else None,
                    snapshot.liquidity.currency_upper if snapshot.liquidity else None,
                    snapshot.volume_24h.minor_units if snapshot.volume_24h else None,
                    snapshot.volume_24h.currency_upper if snapshot.volume_24h else None,
                    snapshot.close_time_utc, snapshot.quote_currency,
                    snapshot_data_hash(snapshot),
                ),
            )

    # --- reads --------------------------------------------------------------------
    def cycle_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM cycles").fetchone()
        return int(row["n"])

    def last_cycle(self) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM cycles WHERE finished_utc IS NOT NULL "
                "ORDER BY started_utc DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

