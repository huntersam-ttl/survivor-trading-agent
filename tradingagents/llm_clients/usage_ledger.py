"""Inference usage ledger for persistent SQLite accounting."""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tradingagents.survivor.types import InferenceRecord

_DEFAULT_DB_DIR = os.path.join(os.path.expanduser("~"), ".tradingagents", "survivor")
_DEFAULT_DB_PATH = os.path.join(_DEFAULT_DB_DIR, "usage.db")


class InferenceUsageLedger:
    """SQLite-backed persistent ledger for recording all inference requests and costs."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or _DEFAULT_DB_PATH)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_records (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    agent_role TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    ticker_or_market TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    reasoning_tokens INTEGER NOT NULL,
                    native_cost_minor INTEGER NOT NULL,
                    native_currency TEXT NOT NULL,
                    gbp_cost_pence INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    failure_reason TEXT,
                    latency_ms INTEGER NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_usage_date ON usage_records(timestamp_utc);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_usage_provider ON usage_records(provider);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_usage_run_id ON usage_records(run_id);"
            )

    def record_inference(
        self,
        run_id: str,
        agent_role: str,
        provider: str,
        model: str,
        ticker_or_market: str,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        native_cost_minor: int,
        native_currency: str,
        gbp_cost_pence: int,
        status: str = "SUCCESS",
        failure_reason: str | None = None,
        latency_ms: int = 0,
        timestamp_utc: str | None = None,
    ) -> InferenceRecord:
        record_id = str(uuid.uuid4())
        ts = timestamp_utc or datetime.now(timezone.utc).isoformat()

        record = InferenceRecord(
            id=record_id,
            run_id=run_id,
            timestamp_utc=ts,
            agent_role=agent_role,
            provider=provider.lower(),
            model=model,
            ticker_or_market=ticker_or_market,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            native_cost_minor=native_cost_minor,
            native_currency=native_currency,
            gbp_cost_pence=gbp_cost_pence,
            status=status,
            failure_reason=failure_reason,
            latency_ms=latency_ms,
        )

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO usage_records (
                    id, run_id, timestamp_utc, agent_role, provider, model,
                    ticker_or_market, input_tokens, output_tokens, reasoning_tokens,
                    native_cost_minor, native_currency, gbp_cost_pence, status,
                    failure_reason, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record.id,
                    record.run_id,
                    record.timestamp_utc,
                    record.agent_role,
                    record.provider,
                    record.model,
                    record.ticker_or_market,
                    record.input_tokens,
                    record.output_tokens,
                    record.reasoning_tokens,
                    record.native_cost_minor,
                    record.native_currency,
                    record.gbp_cost_pence,
                    record.status,
                    record.failure_reason,
                    record.latency_ms,
                ),
            )

        return record

    def get_daily_spend_pence(self, provider: str | None = None, date_str: str | None = None) -> int:
        """Sum gbp_cost_pence for a specific UTC day (format YYYY-MM-DD)."""
        target_date = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        query = "SELECT SUM(gbp_cost_pence) as total FROM usage_records WHERE timestamp_utc LIKE ? AND status = 'SUCCESS'"
        params: list[Any] = [f"{target_date}%"]

        if provider:
            query += " AND provider = ?"
            params.append(provider.lower())

        with self._get_connection() as conn:
            row = conn.execute(query, params).fetchone()
            return row["total"] or 0 if row else 0

    def get_native_spend_today(self, provider: str | None = None, date_str: str | None = None) -> float:
        """Sum today's SETTLED spend for a provider in its NATIVE major units
        (USD for OpenRouter et al.). Used by the cost governor's USD budget.
        FAILED rows settle 0 cost and are excluded."""
        cur_date = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        p_filter = provider.lower() if provider else None
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT provider, native_cost_minor, native_currency
                FROM usage_records
                WHERE substr(timestamp_utc, 1, 10) = ?
                  AND status = 'SUCCESS'
                """,
                (cur_date,),
            ).fetchall()
        total = 0.0
        for row in rows:
            if p_filter and row["provider"].lower() != p_filter:
                continue
            if row["native_currency"].upper() == "USD":
                total += row["native_cost_minor"] / 100.0
        return total

    def get_monthly_spend_pence(self, provider: str | None = None, year_month: str | None = None) -> int:
        """Sum gbp_cost_pence for a specific UTC month (format YYYY-MM)."""
        target_month = year_month or datetime.now(timezone.utc).strftime("%Y-%m")
        query = "SELECT SUM(gbp_cost_pence) as total FROM usage_records WHERE timestamp_utc LIKE ? AND status = 'SUCCESS'"
        params: list[Any] = [f"{target_month}%"]

        if provider:
            query += " AND provider = ?"
            params.append(provider.lower())

        with self._get_connection() as conn:
            row = conn.execute(query, params).fetchone()
            return row["total"] or 0 if row else 0

    def get_model_stats(
        self,
        provider: str,
        model: str,
        *,
        window_hours: float = 24.0,
    ) -> dict[str, Any]:
        """Aggregate recent historical performance for a single (provider, model).

        Used by ModelRouter's dynamic scoring: success rate and average latency
        over the trailing ``window_hours`` of settled calls. Returns neutral
        defaults when there is no history so routing is not biased by absence
        of data.
        """
        now_utc = datetime.now(timezone.utc)
        from_ts = (now_utc - timedelta(hours=window_hours)).isoformat()
        to_ts = now_utc.isoformat()
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT status, latency_ms
                FROM usage_records
                WHERE provider = ? AND model = ?
                  AND timestamp_utc >= ?
                  AND timestamp_utc <= ?
                """,
                (provider.lower(), model, from_ts, to_ts),
            ).fetchall()
        calls = len(rows)
        if calls == 0:
            return {"calls": 0, "successes": 0, "success_rate": 0.5, "avg_latency_ms": None}
        successes = sum(1 for r in rows if r["status"] == "SUCCESS")
        latencies = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
        avg_lat = sum(latencies) / len(latencies) if latencies else None
        return {
            "calls": calls,
            "successes": successes,
            "success_rate": successes / calls,
            "avg_latency_ms": avg_lat,
        }

    def get_run_usage_summary(self, run_id: str) -> dict[str, Any]:
        """Aggregate usage metrics by provider and role for a single run_id."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT provider, model, agent_role, COUNT(*) as calls,
                       SUM(input_tokens) as in_tok, SUM(output_tokens) as out_tok,
                       SUM(gbp_cost_pence) as cost_pence
                FROM usage_records
                WHERE run_id = ? AND status = 'SUCCESS'
                GROUP BY provider, model, agent_role;
                """,
                (run_id,),
            ).fetchall()

            total_pence = 0
            breakdown = []
            for r in rows:
                p_cost = r["cost_pence"] or 0
                total_pence += p_cost
                breakdown.append(
                    {
                        "provider": r["provider"],
                        "model": r["model"],
                        "role": r["agent_role"],
                        "calls": r["calls"],
                        "input_tokens": r["in_tok"] or 0,
                        "output_tokens": r["out_tok"] or 0,
                        "cost_gbp": f"£{p_cost / 100:.2f}",
                        "cost_pence": p_cost,
                    }
                )

            return {
                "run_id": run_id,
                "total_calls": sum(r["calls"] for r in rows),
                "total_cost_pence": total_pence,
                "total_cost_gbp": f"£{total_pence / 100:.2f}",
                "breakdown": breakdown,
            }
