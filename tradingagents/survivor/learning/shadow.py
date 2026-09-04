"""Shadow mode: fictional parallel decisions that NEVER touch PaperBroker.

For every future candidate, the CURRENT strategy makes the normal paper
decision while each SHADOW_TESTING experiment records a separate fictional
shadow decision. Shadow decisions are stored, resolved, compared — and can
never execute anything: this module has no broker, no ledger, no execution
path whatsoever.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ShadowDecision:
    cycle_id: str
    experiment_id: str
    market_id: str
    timestamp_utc: str
    predicted_probability: float
    market_probability: float
    conservative_edge_bps: int
    action: str                       # OPEN (fictional) | NO_TRADE
    outcome: float | None = None
    resolution_timestamp_utc: str | None = None


class ShadowStore:
    """SQLite store of fictional shadow decisions. Metadata only — there is
    deliberately no execution capability here."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            base = Path.home() / ".tradingagents" / "survivor"
            base.mkdir(parents=True, exist_ok=True)
            db_path = str(base / "shadow.db")
        self.db_path = str(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shadow_decisions (
                    cycle_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    predicted_probability REAL NOT NULL,
                    market_probability REAL NOT NULL,
                    conservative_edge_bps INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    outcome REAL,
                    resolution_timestamp_utc TEXT,
                    PRIMARY KEY (cycle_id, experiment_id, market_id)
                )
            """)

    def record(self, decision: ShadowDecision) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO shadow_decisions VALUES "
                "(?,?,?,?,?,?,?,?,?,?)",
                (decision.cycle_id, decision.experiment_id, decision.market_id,
                 decision.timestamp_utc, decision.predicted_probability,
                 decision.market_probability, decision.conservative_edge_bps,
                 decision.action, decision.outcome,
                 decision.resolution_timestamp_utc),
            )

    def decisions(self, experiment_id: str) -> list[ShadowDecision]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM shadow_decisions WHERE experiment_id = ? "
                "ORDER BY timestamp_utc", (experiment_id,)).fetchall()
        return [ShadowDecision(
            cycle_id=r["cycle_id"], experiment_id=r["experiment_id"],
            market_id=r["market_id"], timestamp_utc=r["timestamp_utc"],
            predicted_probability=r["predicted_probability"],
            market_probability=r["market_probability"],
            conservative_edge_bps=r["conservative_edge_bps"],
            action=r["action"], outcome=r["outcome"],
            resolution_timestamp_utc=r["resolution_timestamp_utc"],
        ) for r in rows]

    def attach_outcome(self, experiment_id: str, market_id: str,
                       cycle_id: str, outcome: float,
                       resolution_timestamp_utc: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE shadow_decisions SET outcome = ?, resolution_timestamp_utc = ? "
                "WHERE experiment_id = ? AND market_id = ? AND cycle_id = ?",
                (outcome, resolution_timestamp_utc, experiment_id, market_id, cycle_id))
        return cursor.rowcount > 0


def shadow_decision_for(experiment, *, cycle_id: str, market_id: str,
                        timestamp_utc: str, survivor_probability: float,
                        market_probability: float,
                        conservative_edge_bps: int) -> ShadowDecision:
    """Fictional decision of ONE experiment for ONE candidate — deterministic
    replay of the allowlisted knobs on the current research result."""
    from tradingagents.survivor.learning.dataset import LearningRecord
    from tradingagents.survivor.learning.sandbox import _experiment_decision

    fake_record = LearningRecord(
        trial_id="shadow", strategy_version=experiment.parent_strategy_version,
        cycle_id=cycle_id, run_id=f"{cycle_id}_{experiment.experiment_id}",
        market_id=market_id, category="prediction_binary",
        timestamp_utc=timestamp_utc,
        market_probability=market_probability,
        survivor_probability=survivor_probability,
        predicted_edge_bps=conservative_edge_bps,
        conservative_edge_bps=conservative_edge_bps,
    )
    would_open = _experiment_decision(fake_record, experiment.proposed_config_diff)
    return ShadowDecision(
        cycle_id=cycle_id, experiment_id=experiment.experiment_id,
        market_id=market_id, timestamp_utc=timestamp_utc,
        predicted_probability=survivor_probability,
        market_probability=market_probability,
        conservative_edge_bps=conservative_edge_bps,
        action="OPEN" if would_open else "NO_TRADE",
    )


def compare_with_current(shadow_store: ShadowStore, experiment) -> dict:
    """Compare resolved shadow decisions vs the market baseline. Read-only."""
    from tradingagents.survivor.learning.sandbox import MIN_SHADOW_SAMPLE

    decisions = [d for d in shadow_store.decisions(experiment.experiment_id)
                 if d.outcome is not None and d.action == "OPEN"]
    if len(decisions) < MIN_SHADOW_SAMPLE:
        return {
            "experiment_id": experiment.experiment_id,
            "verdict": "INSUFFICIENT_SAMPLE", "resolved": len(decisions),
        }
    shadow_brier = sum((d.predicted_probability - d.outcome) ** 2
                       for d in decisions) / len(decisions)
    market_brier = sum((d.market_probability - d.outcome) ** 2
                       for d in decisions) / len(decisions)
    return {
        "experiment_id": experiment.experiment_id,
        "verdict": ("CANDIDATE_FOR_PROMOTION"
                    if shadow_brier < market_brier else "REJECTED"),
        "resolved": len(decisions),
        "shadow_brier": round(shadow_brier, 4),
        "market_brier": round(market_brier, 4),
    }

