"""Sandbox evaluation: replay experiments against HISTORICAL STORED snapshots.

Only deterministic, allowlisted knobs are replayed (uncertainty penalty,
minimum research confidence, category NO_TRADE thresholds, scanner rank
weights). The current strategy and the experiment see the SAME stored
decision-time data — nothing is re-fetched and no future data exists in the
features. Evaluation is chronological / walk-forward: first 70% in-sample,
final 30% out-of-sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tradingagents.survivor.learning.dataset import LearningRecord
from tradingagents.survivor.learning.experiments import StrategyExperiment

MIN_SHADOW_SAMPLE = 20
MAX_DRAWDOWN_REGRESSION_PENCE = 200
MAX_AI_COST_REGRESSION_PCT = 10


@dataclass(frozen=True)
class SandboxResult:
    experiment_id: str
    verdict: str                      # CANDIDATE | OVERFIT_REJECTED | INSUFFICIENT_SAMPLE
    reasons: list[str] = field(default_factory=list)
    in_sample: dict = field(default_factory=dict)
    out_of_sample: dict = field(default_factory=dict)
    current_overall: dict = field(default_factory=dict)


def _experiment_decision(record: LearningRecord, diff: dict[str, Any]) -> bool:
    """Deterministic replay of one decision under the experiment config.

    Returns True when the experiment would have opened the trade. Only
    allowlisted knobs influence the replay:
    - uncertainty_penalty_bps: reduces the conservative edge
    - min_research_confidence: requires |SURVIVOR probability - 0.5| >= min
    - category_no_trade_thresholds: requires conservative edge >= threshold
    """
    edge = record.conservative_edge_bps
    penalty = diff.get("uncertainty_penalty_bps", 0)
    if isinstance(penalty, (int, float)) and penalty > 0:
        edge -= int(penalty)
    confidence = abs(record.survivor_probability - 0.5)
    min_confidence = diff.get("min_research_confidence", 0.0)
    if isinstance(min_confidence, (int, float)) and confidence < float(min_confidence):
        return False
    thresholds = diff.get("category_no_trade_thresholds", {})
    threshold_bps = None
    if isinstance(thresholds, dict):
        raw = thresholds.get(record.category)
        if isinstance(raw, (int, float)):
            threshold_bps = int(float(raw) * 100) if raw <= 1 else int(raw)
    if threshold_bps is not None and edge < threshold_bps:
        return False
    return not (threshold_bps is None and edge <= 0)


def _experiment_brier(resolved: list[LearningRecord], diff: dict[str, Any]) -> float | None:
    """Brier of SURVIVOR probabilities with the uncertainty penalty applied as
    a deterministic shrink toward 0.5 (the only allowlisted calibration knob)."""
    if not resolved:
        return None
    shrink = float(diff.get("uncertainty_penalty_bps", 0) or 0) / 10000.0
    total = 0.0
    for record in resolved:
        p = record.survivor_probability
        if shrink:
            p = p + (0.5 - p) * min(1.0, shrink * 2)
        total += (p - record.outcome) ** 2
    return total / len(resolved)


def _metrics(records: list[LearningRecord], diff: dict[str, Any],
             start_equity: int = 2000) -> dict:
    resolved = [r for r in records if r.resolved]
    pnl = 0
    equity = [start_equity]
    peak = start_equity
    max_dd = 0
    for record in records:
        if record.resolved and _experiment_decision(record, diff):
            pnl += record.pnl_pence
            equity.append(start_equity + pnl)
            peak = max(peak, equity[-1])
            max_dd = max(max_dd, peak - equity[-1])
    ai_cost = sum(r.ai_cost_pence for r in records)
    wins = [r.pnl_pence for r in records
            if r.resolved and _experiment_decision(record, diff) and r.pnl_pence > 0]
    concentration = (sum(wins) / pnl) if pnl > 0 and wins else 0.0
    return {
        "n": len(records),
        "brier": _experiment_brier(resolved, diff),
        "economic_pnl_pence": pnl - ai_cost,
        "gross_pnl_pence": pnl,
        "max_drawdown_pence": max_dd,
        "profit_concentration": round(concentration, 3),
        "ai_cost_pence": ai_cost,
    }


def run_sandbox(experiment: StrategyExperiment,
                records: list[LearningRecord]) -> SandboxResult:
    """Chronological walk-forward comparison: CURRENT vs EXPERIMENT."""
    diff = experiment.proposed_config_diff
    chronological = sorted(records, key=lambda r: r.timestamp_utc)
    if len(chronological) < MIN_SHADOW_SAMPLE:
        return SandboxResult(
            experiment_id=experiment.experiment_id, verdict="INSUFFICIENT_SAMPLE",
            reasons=[f"only {len(chronological)} resolved records; need >={MIN_SHADOW_SAMPLE}"],
        )
    split = int(len(chronological) * 0.7)
    in_sample, out_sample = chronological[:split], chronological[split:]

    current_is = _metrics(in_sample, {})
    current_oos = _metrics(out_sample, {})
    experiment_is = _metrics(in_sample, diff)
    experiment_oos = _metrics(out_sample, diff)

    reasons: list[str] = []
    if experiment_oos["brier"] is not None and current_oos["brier"] is not None \
            and experiment_oos["brier"] > current_oos["brier"]:
        reasons.append("out-of-sample Brier degrades")
    if experiment_is["economic_pnl_pence"] > current_is["economic_pnl_pence"] and \
            experiment_oos["economic_pnl_pence"] <= current_oos["economic_pnl_pence"]:
        reasons.append("improves in-sample but not out-of-sample")
    if experiment_oos["economic_pnl_pence"] <= 0:
        reasons.append("out-of-sample economic P/L not positive")
    if experiment_oos["max_drawdown_pence"] > \
            current_oos["max_drawdown_pence"] + MAX_DRAWDOWN_REGRESSION_PENCE:
        reasons.append("materially higher drawdown")
    current_cost = max(1, current_oos["ai_cost_pence"])
    if experiment_oos["ai_cost_pence"] > current_cost * (1 + MAX_AI_COST_REGRESSION_PCT / 100):
        reasons.append("AI cost increase disproportionate")

    verdict = "CANDIDATE" if not reasons else "OVERFIT_REJECTED"
    return SandboxResult(
        experiment_id=experiment.experiment_id,
        verdict=verdict,
        reasons=reasons,
        in_sample={"current": current_is, "experiment": experiment_is},
        out_of_sample={"current": current_oos, "experiment": experiment_oos},
        current_overall=_metrics(chronological, {}),
    )
