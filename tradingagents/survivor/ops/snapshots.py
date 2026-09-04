"""Deterministic daily/weekly evaluation snapshots (immutable, no self-tuning)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

from tradingagents.survivor.evaluation.evaluate import evaluate_performance
from tradingagents.survivor.evaluation.store import EvaluationStore


def _iso_day(day: datetime) -> str:
    return day.strftime("%Y-%m-%d")


def daily_snapshot(
    *,
    trial_id: str,
    evaluation_store: EvaluationStore,
    cycles_scanned: int = 0,
    markets_scanned: int = 0,
    candidates_researched: int = 0,
    paper_trades: int = 0,
    paper_equity_pence: int = 2000,
    daily_pnl_pence: int = 0,
    drawdown_bps: int = 0,
    ai_spend_pence: int = 0,
    errors: int = 0,
    halt_state: str = "CLEAR",
    paper_ledger_path: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Immutable daily snapshot. NEVER modifies strategy based on it."""
    current = now or datetime.now(timezone.utc)
    report = evaluate_performance(evaluation_store, paper_ledger_path=paper_ledger_path)
    snapshot = {
        "trial_id": trial_id,
        "day": _iso_day(current),
        "cycles": cycles_scanned,
        "markets_scanned": markets_scanned,
        "candidates_researched": candidates_researched,
        "paper_trades": paper_trades,
        "resolved_trades": report.resolved_predictions,
        "paper_equity_pence": paper_equity_pence,
        "daily_pnl_pence": daily_pnl_pence,
        "drawdown_bps": drawdown_bps,
        "ai_spend_pence": ai_spend_pence,
        "brier": report.brier,
        "market_baseline_brier": report.market_brier,
        "evaluation_state": f"{report.sample_confidence.value}/{report.gate.value}",
        "errors": errors,
        "halt_state": halt_state,
    }
    return snapshot


def weekly_snapshot(
    *,
    trial_id: str,
    evaluation_store: EvaluationStore,
    week_start: datetime | None = None,
    paper_ledger_path: str | None = None,
    now: datetime | None = None,
    **perf_kwargs,
) -> dict:
    """Deterministic weekly evaluation snapshot (7-day window ending now)."""
    current = now or datetime.now(timezone.utc)
    start = week_start or (current - timedelta(days=7))
    report = evaluate_performance(evaluation_store, paper_ledger_path=paper_ledger_path, **perf_kwargs)
    return {
        "trial_id": trial_id,
        "week_start": start.isoformat(),
        "week_end": current.isoformat(),
        "resolved_sample_size": report.resolved_predictions,
        "brier": report.brier,
        "brier_improvement": report.brier_improvement,
        "net_pnl_pence": report.net_pnl_pence,
        "ai_cost_pence": report.total_ai_cost_pence,
        "economic_pnl_pence": report.net_pnl_after_ai_cost_pence,
        "max_drawdown_bps": report.max_drawdown_bps,
        "profit_concentration": report.profit_concentration,
        "out_of_sample_net_pnl_pence": report.out_of_sample_net_pnl_pence,
        "category_breakdown": report.category_breakdown,
        "warnings": report.warnings,
        "gate": report.gate.value,
    }


def persist_snapshot(snapshot: dict, snapshots_dir: str) -> str:
    """Persist a snapshot immutably as JSON (filename = day)."""
    os.makedirs(snapshots_dir, exist_ok=True)
    path = os.path.join(snapshots_dir, f"snapshot_{snapshot.get('day', datetime.now(timezone.utc).strftime('%Y-%m-%d'))}.json")
    if os.path.exists(path):
        return path  # immutable: first write wins
    with open(path, "w") as fh:
        json.dump(snapshot, fh, sort_keys=True, indent=2)
    return path
