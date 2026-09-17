"""Append-only, read-only market history for offline evaluation.

This store is deliberately separate from the live paper ledger, runtime DB,
heartbeat, and strategy configuration. Capture is best-effort: a history
storage failure must never change or stop a PAPER cycle.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Iterable

from tradingagents.survivor.markets.types import MarketSnapshot, ScanRejection
from tradingagents.survivor.paths import runtime_dir

SCHEMA_VERSION = "1"


def default_history_path() -> str:
    return os.environ.get("SURVIVOR_HISTORY_DB_PATH", "").strip() or os.path.join(
        runtime_dir(), "history.db"
    )


class HistoryStore:
    def __init__(self, db_path: str | None = None, *, read_only: bool = False,
                 initialize: bool = True):
        self.db_path = str(db_path or default_history_path())
        self._read_only = read_only
        if initialize and not read_only:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self._init_db()

    def _connect(self):
        if self._read_only:
            return sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=30.0)
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_snapshots (
                    cycle_id TEXT NOT NULL, market_id TEXT NOT NULL,
                    observed_at TEXT NOT NULL, question TEXT NOT NULL,
                    condition_id TEXT, event_id TEXT, market_status TEXT,
                    resolution_timestamp TEXT, bid_bps INTEGER, ask_bps INTEGER,
                    mid_bps INTEGER, last_trade_price_bps INTEGER,
                    spread_bps REAL, liquidity_minor INTEGER,
                    liquidity_currency TEXT, volume_minor INTEGER,
                    volume_currency TEXT, yes_bid_bps INTEGER, yes_ask_bps INTEGER,
                    no_bid_bps INTEGER, no_ask_bps INTEGER,
                    probability_bps INTEGER, provider TEXT NOT NULL,
                    schema_version TEXT NOT NULL, normalized_json TEXT NOT NULL,
                    filter_result TEXT NOT NULL, rejection_reason TEXT,
                    thresholds_json TEXT NOT NULL, strategy_version TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    PRIMARY KEY (cycle_id, market_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_outcomes (
                    market_id TEXT PRIMARY KEY, resolution_timestamp TEXT NOT NULL,
                    outcome REAL NOT NULL, final_value REAL, source TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_counters (
                    name TEXT PRIMARY KEY, count INTEGER NOT NULL
                )
            """)

    @staticmethod
    def _price_bps(raw: dict):
        value = raw.get("lastTradePrice")
        if value is None:
            value = raw.get("last_trade_price")
        try:
            return int(round(float(value) * 10000)) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _json_snapshot(snapshot: MarketSnapshot) -> str:
        liquidity = snapshot.liquidity
        volume = snapshot.volume_24h
        data = {
            "market_id": snapshot.market_id, "provider": snapshot.provider,
            "market_type": snapshot.market_type, "symbol_or_slug": snapshot.symbol_or_slug,
            "question": snapshot.question, "timestamp_utc": snapshot.timestamp_utc,
            "close_time_utc": snapshot.close_time_utc,
            "resolution_time_utc": snapshot.resolution_time_utc,
            "bid": snapshot.bid, "ask": snapshot.ask, "mid": snapshot.mid,
            "yes_bid": snapshot.yes_bid, "yes_ask": snapshot.yes_ask,
            "no_bid": snapshot.no_bid, "no_ask": snapshot.no_ask,
            "market_probability_bps": snapshot.market_probability_bps,
            "liquidity": {"minor_units": liquidity.minor_units, "currency": liquidity.currency} if liquidity else None,
            "volume_24h": {"minor_units": volume.minor_units, "currency": volume.currency} if volume else None,
            "quote_currency": snapshot.quote_currency,
            "market_status": snapshot.market_status.value,
            "resolution_status": snapshot.resolution_status.value,
            "source_timestamp_utc": snapshot.source_timestamp_utc,
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    def record_scan(
        self, cycle_id: str, observations: Iterable[tuple[dict, MarketSnapshot | None, ScanRejection | None]],
        thresholds: dict, strategy_version: str, config_hash: str,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        quality = {"missing_bid": 0, "missing_ask": 0, "invalid_timestamp": 0, "schema_errors": 0}
        for raw, snapshot, reason in observations:
            market_id = snapshot.market_id if snapshot else str(raw.get("id") or raw.get("market_id") or "<unnormalizable>")
            if snapshot is None:
                quality["schema_errors"] += 1
                normalized = json.dumps(raw, sort_keys=True, default=str)
                fields = (None,) * 14
                provider = str(raw.get("provider") or "polymarket")
                question = str(raw.get("question") or "")
            else:
                if snapshot.bid is None: quality["missing_bid"] += 1
                if snapshot.ask is None: quality["missing_ask"] += 1
                if not snapshot.close_time_utc: quality["invalid_timestamp"] += 1
                liq = snapshot.liquidity; vol = snapshot.volume_24h
                spread = ((snapshot.ask - snapshot.bid) * 10000 / snapshot.ask
                          if snapshot.bid is not None and snapshot.ask else None)
                fields = (
                    snapshot.close_time_utc, snapshot.bid, snapshot.ask, snapshot.mid,
                    HistoryStore._price_bps(raw), spread, liq.minor_units if liq else None, liq.currency if liq else None,
                    vol.minor_units if vol else None, vol.currency if vol else None,
                    snapshot.yes_bid, snapshot.yes_ask, snapshot.no_bid, snapshot.no_ask,
                )
                normalized = HistoryStore._json_snapshot(snapshot)
                provider = snapshot.provider; question = snapshot.question
            rows.append((cycle_id, market_id, now, question,
                         raw.get("conditionId") or raw.get("condition_id"),
                         raw.get("eventId") or raw.get("event_id"),
                         snapshot.market_status.value if snapshot else None,
                         *fields, snapshot.market_probability_bps if snapshot else None,
                         provider, SCHEMA_VERSION, normalized,
                         "ACCEPTED" if reason is None else "REJECTED",
                         reason.value if reason else None, json.dumps(thresholds, sort_keys=True),
                         strategy_version, config_hash))
        with self._connect() as conn:
            placeholders = ",".join("?" for _ in range(30))
            conn.executemany(
                f"INSERT OR IGNORE INTO market_snapshots VALUES ({placeholders})", rows
            )
            for name, count in quality.items():
                conn.execute("INSERT INTO quality_counters(name,count) VALUES(?,?) ON CONFLICT(name) DO UPDATE SET count=count+excluded.count", (name, count))
        return len(rows)

    def stats(self) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*), COUNT(DISTINCT market_id), MIN(observed_at), MAX(observed_at) FROM market_snapshots").fetchone()
            resolved = conn.execute("SELECT COUNT(*) FROM market_outcomes").fetchone()[0]
            total = row[0] or 0
            unresolved = max(0, (row[1] or 0) - resolved)
            quality = dict(conn.execute("SELECT name,count FROM quality_counters"))
        return {"snapshot_count": total, "unique_markets": row[1] or 0,
                "resolved_markets": resolved, "unresolved_markets": unresolved,
                "oldest_snapshot": row[2], "newest_snapshot": row[3],
                "quality": quality}

    def unresolved_market_ids(self, limit: int = 100) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT s.market_id FROM market_snapshots s "
                "LEFT JOIN market_outcomes o ON o.market_id = s.market_id "
                "WHERE o.market_id IS NULL ORDER BY s.market_id LIMIT ?", (limit,)
            ).fetchall()
        return [row[0] for row in rows]

    def reconcile(self, adapter, limit: int = 100) -> int:
        """Bounded read-only outcome reconciliation using provider resolution state."""
        resolver = getattr(adapter, "get_resolved_outcome", None)
        if resolver is None:
            return 0
        resolved = 0
        for market_id in self.unresolved_market_ids(limit):
            outcome = resolver(market_id)
            if not outcome:
                continue
            value, resolved_at, final_value = outcome
            if self.record_outcome(market_id, resolved_at, value, final_value, adapter.provider):
                resolved += 1
        return resolved

    def record_outcome(self, market_id: str, resolution_timestamp: str, outcome: float,
                       final_value: float | None, source: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("INSERT OR IGNORE INTO market_outcomes VALUES(?,?,?,?,?,?)",
                (market_id, resolution_timestamp, outcome, final_value, source,
                 datetime.now(timezone.utc).isoformat()))
            return cur.rowcount == 1
