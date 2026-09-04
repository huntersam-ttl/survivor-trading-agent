"""Immutable evaluation storage: ~/.tradingagents/survivor/evaluation.db.

Separate from paper.db (accounting), usage.db (inference spend) and
runtime.db (cycle telemetry). Records are append-only.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

from tradingagents.survivor.evaluation.types import PredictionRecord, TradeOutcome

DEFAULT_EVALUATION_DB_PATH = os.path.join(
    os.path.expanduser("~"), ".tradingagents", "survivor", "evaluation.db"
)


class EvaluationStore:
    def __init__(self, db_path: str | None = None):
        self.db_path = str(db_path or DEFAULT_EVALUATION_DB_PATH)
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
                CREATE TABLE IF NOT EXISTS prediction_records (
                    id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL, run_id TEXT NOT NULL,
                    proposal_id TEXT NOT NULL, market_id TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    strategy_version TEXT NOT NULL, config_hash TEXT NOT NULL,
                    category TEXT NOT NULL,
                    predicted_probability REAL NOT NULL,
                    market_probability REAL NOT NULL,
                    gross_edge_bps INTEGER NOT NULL, net_edge_bps INTEGER NOT NULL,
                    ai_cost_pence INTEGER NOT NULL,
                    outcome REAL, resolution_timestamp_utc TEXT,
                    snapshot_data_hash TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_outcomes (
                    id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL, run_id TEXT NOT NULL,
                    proposal_id TEXT NOT NULL, market_id TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    strategy_version TEXT NOT NULL, config_hash TEXT NOT NULL,
                    category TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price_pence INTEGER NOT NULL,
                    fees_pence INTEGER NOT NULL, slippage_pence INTEGER NOT NULL,
                    gross_pnl_pence INTEGER NOT NULL, ai_cost_pence INTEGER NOT NULL,
                    realized_return_bps INTEGER NOT NULL,
                    predicted_net_edge_bps INTEGER NOT NULL,
                    snapshot_data_hash TEXT NOT NULL,
                    decision_latency_ms INTEGER NOT NULL,
                    outcome REAL
                )
                """
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS strategy_runs ("
                "id TEXT PRIMARY KEY, cycle_id TEXT NOT NULL, strategy_version TEXT NOT NULL,"
                " config_hash TEXT NOT NULL, started_utc TEXT NOT NULL, notes TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS baseline_results ("
                "id TEXT PRIMARY KEY, cycle_id TEXT NOT NULL, baseline_name TEXT NOT NULL,"
                " brier REAL, net_pnl_pence INTEGER, details TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS evaluation_snapshots ("
                "id TEXT PRIMARY KEY, created_utc TEXT NOT NULL, strategy_version TEXT NOT NULL,"
                " config_hash TEXT NOT NULL, report_json TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cost_attribution ("
                "id TEXT PRIMARY KEY, run_id TEXT NOT NULL, market_id TEXT NOT NULL,"
                " cycle_id TEXT NOT NULL, ai_cost_pence INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS calibration_bins ("
                "id TEXT PRIMARY KEY, report_id TEXT NOT NULL, lower_bps INTEGER NOT NULL,"
                " upper_bps INTEGER NOT NULL, count INTEGER NOT NULL,"
                " mean_predicted REAL NOT NULL, outcome_frequency REAL NOT NULL,"
                " calibration_error REAL NOT NULL)"
            )

    # --- writes ----------------------------------------------------------------
    def record_prediction(self, record: PredictionRecord) -> str:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO prediction_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()), record.cycle_id, record.run_id, record.proposal_id,
                    record.market_id, record.timestamp_utc, record.strategy_version,
                    record.config_hash, record.category, record.predicted_probability,
                    record.market_probability, record.gross_edge_bps, record.net_edge_bps,
                    record.ai_cost_pence, record.outcome, record.resolution_timestamp_utc,
                    record.snapshot_data_hash,
                ),
            )
        return record.proposal_id

    def record_trade(self, record: TradeOutcome) -> str:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO trade_outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()), record.cycle_id, record.run_id, record.proposal_id,
                    record.market_id, record.timestamp_utc, record.strategy_version,
                    record.config_hash, record.category, record.quantity,
                    record.entry_price_pence, record.fees_pence, record.slippage_pence,
                    record.gross_pnl_pence, record.ai_cost_pence, record.realized_return_bps,
                    record.predicted_net_edge_bps, record.snapshot_data_hash,
                    record.decision_latency_ms, record.outcome,
                ),
            )
        return record.proposal_id

    def record_baseline(self, cycle_id: str, name: str, brier: float | None,
                        net_pnl_pence: int, details: dict | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO baseline_results VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), cycle_id, name, brier, net_pnl_pence,
                 json.dumps(details or {})),
            )

    def record_cost_attribution(self, run_id: str, market_id: str, cycle_id: str,
                                ai_cost_pence: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO cost_attribution VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), run_id, market_id, cycle_id, ai_cost_pence),
            )

    def attach_outcome(self, proposal_id: str, outcome: float, resolution_timestamp_utc: str) -> bool:
        """Attach a resolution outcome to an existing prediction (once only)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT outcome FROM prediction_records WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
            if row is None or row["outcome"] is not None:
                return False
            conn.execute(
                "UPDATE prediction_records SET outcome = ?, resolution_timestamp_utc = ? "
                "WHERE proposal_id = ?",
                (outcome, resolution_timestamp_utc, proposal_id),
            )
            return True

    # --- reads ------------------------------------------------------------------
    def predictions(self, strategy_version: str | None = None,
                    config_hash: str | None = None) -> list[PredictionRecord]:
        with self._connect() as conn:
            rows = self._query(conn, "prediction_records", strategy_version, config_hash)
        return [
            PredictionRecord(
                cycle_id=r["cycle_id"], run_id=r["run_id"], proposal_id=r["proposal_id"],
                market_id=r["market_id"], timestamp_utc=r["timestamp_utc"],
                strategy_version=r["strategy_version"], config_hash=r["config_hash"],
                category=r["category"], predicted_probability=r["predicted_probability"],
                market_probability=r["market_probability"],
                gross_edge_bps=r["gross_edge_bps"], net_edge_bps=r["net_edge_bps"],
                ai_cost_pence=r["ai_cost_pence"], outcome=r["outcome"],
                resolution_timestamp_utc=r["resolution_timestamp_utc"],
                snapshot_data_hash=r["snapshot_data_hash"],
            )
            for r in rows
        ]

    def trades(self, strategy_version: str | None = None,
               config_hash: str | None = None) -> list[TradeOutcome]:
        with self._connect() as conn:
            rows = self._query(conn, "trade_outcomes", strategy_version, config_hash)
        return [
            TradeOutcome(
                cycle_id=r["cycle_id"], run_id=r["run_id"], proposal_id=r["proposal_id"],
                market_id=r["market_id"], timestamp_utc=r["timestamp_utc"],
                strategy_version=r["strategy_version"], config_hash=r["config_hash"],
                category=r["category"], quantity=r["quantity"],
                entry_price_pence=r["entry_price_pence"], fees_pence=r["fees_pence"],
                slippage_pence=r["slippage_pence"], gross_pnl_pence=r["gross_pnl_pence"],
                ai_cost_pence=r["ai_cost_pence"], realized_return_bps=r["realized_return_bps"],
                predicted_net_edge_bps=r["predicted_net_edge_bps"],
                snapshot_data_hash=r["snapshot_data_hash"],
                decision_latency_ms=r["decision_latency_ms"], outcome=r["outcome"],
            )
            for r in rows
        ]

    @staticmethod
    def _query(conn: sqlite3.Connection, table: str,
               strategy_version: str | None, config_hash: str | None) -> list[sqlite3.Row]:
        query = f"SELECT * FROM {table}"
        clauses, params = [], []
        if strategy_version:
            clauses.append("strategy_version = ?")
            params.append(strategy_version)
        if config_hash:
            clauses.append("config_hash = ?")
            params.append(config_hash)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY timestamp_utc"
        return conn.execute(query, params).fetchall()

    def distinct_identities(self) -> list[tuple[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT strategy_version, config_hash FROM prediction_records "
                "UNION SELECT DISTINCT strategy_version, config_hash FROM trade_outcomes"
            ).fetchall()
        return [(r["strategy_version"], r["config_hash"]) for r in rows]

    def cost_attribution_total(self) -> dict[str, int]:
        """AI cost per run_id (exact, from recorded attributions)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id, SUM(ai_cost_pence) AS total FROM cost_attribution "
                "GROUP BY run_id"
            ).fetchall()
        return {r["run_id"]: int(r["total"] or 0) for r in rows}

    def save_report_snapshot(self, strategy_version: str, config_hash: str, report_json: str) -> str:
        snapshot_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO evaluation_snapshots VALUES (?, ?, ?, ?, ?)",
                (snapshot_id, datetime.now(timezone.utc).isoformat(),
                 strategy_version, config_hash, report_json),
            )
        return snapshot_id


