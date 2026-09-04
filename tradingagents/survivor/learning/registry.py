"""Experiment + strategy-version registry with MANUAL-ONLY promotion.

There is deliberately NO code path from an experiment (even
CANDIDATE_FOR_PROMOTION) to an active production strategy without an explicit
operator call to approve_experiment(). Rollback creates a NEW version —
history is never rewritten.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from tradingagents.survivor.learning.experiments import (
    ExperimentStatus,
    StrategyExperiment,
)

MIN_CANDIDATE_SAMPLE = 20


def config_hash(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


class ExperimentStore:
    """SQLite persistence for experiments and strategy versions."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            base = Path.home() / ".tradingagents" / "survivor"
            base.mkdir(parents=True, exist_ok=True)
            db_path = str(base / "learning.db")
        self.db_path = str(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_versions (
                    version TEXT PRIMARY KEY,
                    parent_version TEXT,
                    config_json TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    experiment_id TEXT,
                    note TEXT
                )
            """)

    def save_experiment(self, experiment: StrategyExperiment) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO experiments VALUES (?,?)",
                (experiment.experiment_id, json.dumps(experiment.to_dict())))

    def get_experiment(self, experiment_id: str) -> StrategyExperiment | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM experiments WHERE experiment_id = ?",
                (experiment_id,)).fetchone()
        return StrategyExperiment.from_dict(json.loads(row["payload"])) if row else None

    def experiments(self, status: ExperimentStatus | None = None) -> list[StrategyExperiment]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM experiments").fetchall()
        found = [StrategyExperiment.from_dict(json.loads(r["payload"])) for r in rows]
        if status is not None:
            found = [e for e in found if e.status == status]
        return sorted(found, key=lambda e: e.created_at)

    def set_status(self, experiment_id: str, status: ExperimentStatus) -> StrategyExperiment:
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise KeyError(f"unknown experiment: {experiment_id}")
        updated = StrategyExperiment(
            **{**experiment.to_dict(), "status": status,
               "allowed_changes": experiment.allowed_changes}
        )
        self.save_experiment(updated)
        return updated

    def current_version(self) -> tuple[str, dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM strategy_versions ORDER BY created_at DESC, "
                "version DESC LIMIT 1").fetchone()
        if row is None:
            return "survivor-v1.0", {}
        return row["version"], json.loads(row["config_json"])

    def create_version(self, config: dict, created_at: str,
                       parent_version: str | None = None,
                       experiment_id: str | None = None,
                       note: str = "") -> str:
        with self._connect() as conn:
            n = conn.execute(
                "SELECT COUNT(*) AS n FROM strategy_versions").fetchone()["n"]
            parent_row = conn.execute(
                "SELECT version FROM strategy_versions ORDER BY created_at DESC, "
                "version DESC LIMIT 1").fetchone()
            parent = parent_version or (parent_row["version"] if parent_row else None)
            version = f"survivor-v1.{n + 1}"
            conn.execute(
                "INSERT INTO strategy_versions VALUES (?,?,?,?,?,?,?)",
                (version, parent, json.dumps(config, sort_keys=True),
                 config_hash(config), created_at, experiment_id, note))
        return version

    def history(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM strategy_versions ORDER BY created_at, version"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_version(self, version: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM strategy_versions WHERE version = ?",
                (version,)).fetchone()
        return dict(row) if row else None

    # -- promotion gate + MANUAL approval ----------------------------------

    @staticmethod
    def promotion_gate(sandbox: dict,
                       min_sample: int = MIN_CANDIDATE_SAMPLE) -> tuple[bool, list[str]]:
        """Deterministic gate for CANDIDATE_FOR_PROMOTION. Even when it passes,
        MANUAL operator approval is still required."""
        reasons = []
        if sandbox.get("resolved", 0) < min_sample:
            reasons.append("minimum sample not reached")
        exp = sandbox.get("experiment", sandbox)
        cur = sandbox.get("current", {})
        if exp.get("brier") is not None and cur.get("brier") is not None \
                and exp["brier"] >= cur["brier"]:
            reasons.append("Brier does not improve")
        if exp.get("economic_pnl_pence", 0) <= 0:
            reasons.append("economic P/L does not improve")
        return (not reasons), reasons

    def approve_experiment(self, experiment_id: str, operator: str,
                           created_at: str) -> str:
        """EXPLICIT operator action. Creates a NEW strategy version from the
        experiment's config diff, snapshots config + hash, records the
        experiment as PROMOTED_MANUALLY. Never modifies an active trial."""
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise KeyError(f"unknown experiment: {experiment_id}")
        if experiment.status != ExperimentStatus.CANDIDATE_FOR_PROMOTION:
            raise ValueError(
                f"experiment {experiment_id} is {experiment.status.value}; "
                "only CANDIDATE_FOR_PROMOTION can be approved"
            )
        if not operator or not str(operator).strip():
            raise ValueError("operator identity is required for manual approval")
        _, current_config = self.current_version()
        new_config = {**current_config, **experiment.proposed_config_diff}
        version = self.create_version(
            new_config, created_at,
            parent_version=experiment.parent_strategy_version,
            experiment_id=experiment_id,
            note=f"manual approval by {operator}",
        )
        self.set_status(experiment_id, ExperimentStatus.PROMOTED_MANUALLY)
        return version

    def rollback(self, to_version: str, operator: str, created_at: str) -> str:
        """MANUAL rollback: creates a NEW version copying the target's config.
        History is append-only — nothing is rewritten."""
        target = self.get_version(to_version)
        if target is None:
            raise KeyError(f"unknown strategy version: {to_version}")
        if not operator or not str(operator).strip():
            raise ValueError("operator identity is required for manual rollback")
        config = json.loads(target["config_json"])
        return self.create_version(
            config, created_at, parent_version=to_version,
            note=f"manual rollback to {to_version} by {operator}",
        )

