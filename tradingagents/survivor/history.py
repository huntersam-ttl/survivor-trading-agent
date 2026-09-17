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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cycle_summaries (
                    cycle_id TEXT PRIMARY KEY, observed_at TEXT NOT NULL,
                    strategy_version TEXT NOT NULL, config_hash TEXT NOT NULL,
                    discovered INTEGER NOT NULL, accepted INTEGER NOT NULL,
                    rejected INTEGER NOT NULL, rejection_json TEXT NOT NULL,
                    missing_bid INTEGER NOT NULL, missing_ask INTEGER NOT NULL,
                    invalid_timestamp INTEGER NOT NULL, schema_errors INTEGER NOT NULL,
                    duplicate_snapshots INTEGER NOT NULL
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
        observations = list(observations)
        rows = []
        quality = {"missing_bid": 0, "missing_ask": 0, "invalid_timestamp": 0, "schema_errors": 0}
        reasons: dict[str, int] = {}
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
            if reason is not None:
                name = reason.value
                reasons[name] = reasons.get(name, 0) + 1
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
            before = conn.execute("SELECT COUNT(*) FROM market_snapshots WHERE cycle_id = ?", (cycle_id,)).fetchone()[0]
            placeholders = ",".join("?" for _ in range(30))
            conn.executemany(
                f"INSERT OR IGNORE INTO market_snapshots VALUES ({placeholders})", rows
            )
            after = conn.execute("SELECT COUNT(*) FROM market_snapshots WHERE cycle_id = ?", (cycle_id,)).fetchone()[0]
            inserted = max(0, after - before)
            duplicate_count = max(0, len(rows) - inserted)
            conn.execute(
                "INSERT OR REPLACE INTO cycle_summaries VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (cycle_id, now, strategy_version, config_hash, len(rows),
                 sum(1 for _, _, reason in observations if reason is None),
                 sum(reasons.values()), json.dumps(reasons, sort_keys=True),
                 quality["missing_bid"], quality["missing_ask"], quality["invalid_timestamp"],
                 quality["schema_errors"], duplicate_count),
            )
            quality["duplicate_snapshots"] = duplicate_count
            for name, count in quality.items():
                conn.execute("INSERT INTO quality_counters(name,count) VALUES(?,?) ON CONFLICT(name) DO UPDATE SET count=count+excluded.count", (name, count))
        return inserted

    def stats(self) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*), COUNT(DISTINCT market_id), MIN(observed_at), MAX(observed_at) FROM market_snapshots").fetchone()
            resolved = conn.execute("SELECT COUNT(*) FROM market_outcomes").fetchone()[0]
            counts = [r[0] for r in conn.execute("SELECT COUNT(*) FROM market_snapshots GROUP BY market_id")]
            cycles = conn.execute("SELECT cycle_id,observed_at,discovered,accepted,rejected,rejection_json,missing_bid,missing_ask,invalid_timestamp,schema_errors,duplicate_snapshots FROM cycle_summaries ORDER BY observed_at").fetchall()
            total = row[0] or 0
            unresolved = max(0, (row[1] or 0) - resolved)
            quality = dict(conn.execute("SELECT name,count FROM quality_counters"))
        span = 0.0
        if row[2] and row[3]:
            span = max(0.0, (datetime.fromisoformat(row[3]) - datetime.fromisoformat(row[2])).total_seconds())
        coverage = {f"at_least_{n}": sum(1 for count in counts if count >= n) for n in (2, 4, 8, 16, 32)}
        forecast = "NOT_READY"
        if coverage["at_least_8"] >= 3: forecast = "READY_FOR_FORECAST_SANITY"
        if coverage["at_least_32"] >= 10 and span >= 7 * 86400: forecast = "READY_FOR_FORECAST_BENCHMARK"
        return {"snapshot_count": total, "unique_markets": row[1] or 0,
                "resolved_markets": resolved, "unresolved_markets": unresolved,
                "oldest_snapshot": row[2], "newest_snapshot": row[3],
                "history_span_seconds": span, "snapshots_per_market": {
                    "min": min(counts) if counts else 0, "max": max(counts) if counts else 0,
                    "distribution": {str(n): counts.count(n) for n in sorted(set(counts))},
                }, "coverage": coverage, "resolution_coverage_pct":
                (round(resolved * 100 / (row[1] or 1), 2) if row[1] else 0.0),
                "forecast_readiness": forecast,
                "strategy_backtest_readiness": "READY" if resolved >= 30 else "NOT_READY",
                "filter_trends": [{"cycle_id": r[0], "observed_at": r[1],
                    "discovered": r[2], "accepted": r[3], "rejected": r[4],
                    "rejections_by_reason": json.loads(r[5]),
                    "quality": {"missing_bid": r[6], "missing_ask": r[7],
                                "invalid_timestamp": r[8], "schema_errors": r[9],
                                "duplicate_snapshots": r[10]}} for r in cycles[-20:]],
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
