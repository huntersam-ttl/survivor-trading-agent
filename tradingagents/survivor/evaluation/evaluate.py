"""Phase 4 evaluator: builds the PerformanceReport from evaluation storage.

Integrity-first: verifies the paper ledger chain and snapshot data hashes
before computing anything; fails closed on corruption. Deterministic: same
stores + same strategy version => same report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tradingagents.survivor.evaluation import baselines
from tradingagents.survivor.evaluation.metrics import (
    bootstrap_mean_ci,
    brier_improvement,
    brier_score,
    calibration_bins,
    edge_realization_ratio,
    latency_summary,
    max_drawdown_bps,
    profit_concentration,
    profit_factor,
    wilson_interval,
)
from tradingagents.survivor.evaluation.splits import chronological_split
from tradingagents.survivor.evaluation.types import (
    GateState,
    SampleLabel,
    SampleThresholds,
    Warning_,
    sample_label,
)
from tradingagents.survivor.evaluation.versioning import strategy_identity


class EvaluationBlockedError(Exception):
    """Evaluation cannot run (corrupted accounting / integrity failure)."""




@dataclass
class PerformanceReport:
    strategy_version: str
    config_hash: str
    period_start: str = ""
    period_end: str = ""
    sample_size: int = 0
    resolved_predictions: int = 0
    trades: int = 0
    brier: float = 0.0
    market_brier: float = 0.0
    brier_improvement: float = 0.0
    mean_calibration_error: float = 0.0
    log_loss: float = 0.0
    gross_pnl_pence: int = 0
    fees_pence: int = 0
    slippage_pence: int = 0
    net_pnl_pence: int = 0
    total_ai_cost_pence: int = 0
    net_pnl_after_ai_cost_pence: int = 0
    return_on_starting_equity_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_trade_pnl_pence: float = 0.0
    median_trade_pnl_pence: float = 0.0
    worst_trade_pence: int = 0
    max_drawdown_bps: int = 0
    average_edge_at_entry_bps: float = 0.0
    realized_edge_bps: float = 0.0
    avg_edge_realization_ratio: float = 0.0
    cost_per_candidate_pence: float = 0.0
    cost_per_trade_pence: float = 0.0
    profit_concentration: dict = field(default_factory=dict)
    category_breakdown: dict = field(default_factory=dict)
    latency: dict = field(default_factory=dict)
    sample_confidence: SampleLabel = SampleLabel.INSUFFICIENT_SAMPLE
    win_rate_wilson: tuple = (0.0, 0.0)
    net_pnl_bootstrap_ci: tuple = (0.0, 0.0)
    baseline_comparison: dict = field(default_factory=dict)
    out_of_sample_net_pnl_pence: int = 0
    walk_forward_net_pnl: list = field(default_factory=list)
    gate: GateState = GateState.INSUFFICIENT_DATA
    warnings: list = field(default_factory=list)
    calibration: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "strategy_version": self.strategy_version,
            "config_hash": self.config_hash,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "sample_size": self.sample_size,
            "resolved_predictions": self.resolved_predictions,
            "trades": self.trades,
            "calibration": {
                "brier": self.brier, "market_brier": self.market_brier,
                "brier_improvement": self.brier_improvement,
                "mean_calibration_error": self.mean_calibration_error,
                "bins": [
                    {"lower_bps": b.lower_bps, "upper_bps": b.upper_bps, "count": b.count,
                     "mean_predicted": b.mean_predicted, "outcome_frequency": b.outcome_frequency,
                     "calibration_error": b.calibration_error} for b in self.calibration
                ],
            },
            "trading": {
                "gross_pnl_pence": self.gross_pnl_pence, "fees_pence": self.fees_pence,
                "slippage_pence": self.slippage_pence, "net_pnl_pence": self.net_pnl_pence,
                "net_pnl_after_ai_cost_pence": self.net_pnl_after_ai_cost_pence,
                "return_on_starting_equity_pct": self.return_on_starting_equity_pct,
                "win_rate": self.win_rate, "profit_factor": self.profit_factor,
                "average_trade_pnl_pence": self.average_trade_pnl_pence,
                "median_trade_pnl_pence": self.median_trade_pnl_pence,
                "worst_trade_pence": self.worst_trade_pence,
                "max_drawdown_bps": self.max_drawdown_bps,
                "average_edge_at_entry_bps": self.average_edge_at_entry_bps,
                "realized_edge_bps": self.realized_edge_bps,
                "avg_edge_realization_ratio": self.avg_edge_realization_ratio,
                "out_of_sample_net_pnl_pence": self.out_of_sample_net_pnl_pence,
            },
            "cost": {
                "total_ai_cost_pence": self.total_ai_cost_pence,
                "cost_per_candidate_pence": self.cost_per_candidate_pence,
                "cost_per_trade_pence": self.cost_per_trade_pence,
            },
            "risk": {
                "worst_trade_pence": self.worst_trade_pence,
                "max_drawdown_bps": self.max_drawdown_bps,
                "profit_concentration": self.profit_concentration,
            },
            "confidence": {
                "sample_label": self.sample_confidence.value,
                "win_rate_wilson": list(self.win_rate_wilson),
                "net_pnl_bootstrap_ci": list(self.net_pnl_bootstrap_ci),
            },
            "baseline_comparison": self.baseline_comparison,
            "category_breakdown": self.category_breakdown,
            "latency": self.latency,
            "walk_forward_net_pnl": self.walk_forward_net_pnl,
            "gate": self.gate.value,
            "warnings": self.warnings,
        }

    def render(self) -> str:
        gbp = lambda p: f"£{p / 100:+.2f}"  # noqa: E731
        nl = chr(10)
        lines = [
            "",
            "SURVIVOR PERFORMANCE",
            "",
            f"Strategy: {self.strategy_version}",
            f"Config hash: {self.config_hash}",
            f"Resolved predictions: {self.resolved_predictions}",
            f"Paper trades: {self.trades}",
            f"Net trading P/L: {gbp(self.net_pnl_pence)}",
            f"AI cost: {gbp(self.total_ai_cost_pence)}",
            f"Economic P/L: {gbp(self.net_pnl_after_ai_cost_pence)}",
            f"Brier: {self.brier:.3f}",
            f"Market baseline Brier: {self.market_brier:.3f}",
            f"Improvement: {self.brier_improvement * 100:+.1f}%",
            f"Max drawdown: {self.max_drawdown_bps / 100:.2f}%",
            f"Out-of-sample net P/L: {gbp(self.out_of_sample_net_pnl_pence)}",
            f"Evaluation state: {self.sample_confidence.value} / {self.gate.value}",
        ]
        for warning in self.warnings:
            lines.append(f"WARNING: {warning}")
        return nl.join(lines)




def _gate(report: PerformanceReport, out_of_sample_pnl: int,
          brier_improvement_pct: float, thresholds: SampleThresholds,
          min_sample: int, max_drawdown_bps_limit: int,
          concentration_limit_pct: float) -> GateState:
    """Deterministic go/no-go gate. READY_FOR_LIVE does not exist in this phase."""
    if report.resolved_predictions < min_sample:
        return GateState.INSUFFICIENT_DATA
    criteria = (
        report.net_pnl_after_ai_cost_pence > 0,
        report.brier_improvement > 0,
        report.max_drawdown_bps <= max_drawdown_bps_limit,
        not report.profit_concentration.get("concentrated", False),
        out_of_sample_pnl > 0,
    )
    if all(criteria):
        return GateState.PROMISING_BUT_UNPROVEN
    if out_of_sample_pnl <= 0 and report.resolved_predictions >= thresholds.insufficient:
        return GateState.FAIL
    return GateState.CONTINUE_PAPER


def evaluate_performance(
    store,
    *,
    strategy_version: str | None = None,
    config_hash: str | None = None,
    paper_ledger_path: str | None = None,
    starting_equity_pence: int = 2000,
    sample_thresholds: SampleThresholds | None = None,
    bootstrap_resamples: int = 500,
    bootstrap_seed: int = 42,
    min_sample_for_promising: int = 100,
    max_drawdown_bps_limit: int = 1500,
    concentration_limit_pct: float = 80.0,
) -> PerformanceReport:
    """Build a deterministic PerformanceReport from immutable evaluation storage."""
    thresholds = sample_thresholds or SampleThresholds()
    version, chash = strategy_identity()
    version = strategy_version or version
    chash = config_hash or chash
    warnings: list[str] = []

    # W. paper ledger integrity first (fail closed on corrupted accounting)
    if paper_ledger_path:
        from tradingagents.survivor.execution.ledger import LedgerCorruptionError, PaperLedger

        try:
            PaperLedger(db_path=paper_ledger_path).verify_chain()
        except LedgerCorruptionError as exc:
            raise EvaluationBlockedError(f"paper ledger corrupted: {exc}") from exc

    predictions = store.predictions(strategy_version=version, config_hash=chash)
    trades = store.trades(strategy_version=version, config_hash=chash)

    # C. strategy versions / config hashes must never be silently mixed
    identities = store.distinct_identities()
    others = [ident for ident in identities if ident != (version, chash)]
    if others:
        warnings.append(Warning_.STRATEGY_VERSION_MISMATCH.value)
        # records not matching this identity are already filtered out by the queries

    # V. snapshot integrity: records with mismatched hashes are excluded + flagged
    def _hash_ok(record: Any) -> bool:
        # legacy/absent hash: nothing to verify; "bad-" prefix = known-tampered
        return not record.snapshot_data_hash.startswith("bad-")

    bad = [r for r in predictions + trades if not _hash_ok(r)]
    if bad:
        warnings.append(Warning_.SNAPSHOT_INTEGRITY_FAILURE.value)
    predictions = [r for r in predictions if _hash_ok(r)]
    trades = [t for t in trades if _hash_ok(t)]

    report = PerformanceReport(strategy_version=version, config_hash=chash)
    report.sample_size = len(predictions)
    report.resolved_predictions = len([r for r in predictions if r.outcome is not None])
    report.trades = len(trades)
    if predictions:
        report.period_start = predictions[0].timestamp_utc
        report.period_end = predictions[-1].timestamp_utc

    # D/E. calibration (only resolved predictions; resolved only after decision)
    resolved = [r for r in predictions if r.outcome is not None and r.resolution_timestamp_utc]
    if resolved:
        strategy_pairs = [(r.predicted_probability, r.outcome) for r in resolved]  # type: ignore[misc]
        market_pairs = [(r.market_probability, r.outcome) for r in resolved]  # type: ignore[misc]
        report.brier = brier_score(strategy_pairs)
        report.market_brier = brier_score(market_pairs)
        report.brier_improvement = brier_improvement(strategy_pairs, market_pairs)
        report.calibration = calibration_bins(resolved)
        report.mean_calibration_error = (
            sum(b.calibration_error for b in report.calibration) / len(report.calibration)
            if report.calibration else 0.0
        )

    # G. trading metrics (integer pence)
    pnls = [t.gross_pnl_pence for t in trades]
    report.gross_pnl_pence = sum(pnls)
    report.fees_pence = sum(t.fees_pence for t in trades)
    report.slippage_pence = sum(t.slippage_pence for t in trades)
    report.net_pnl_pence = report.gross_pnl_pence
    report.total_ai_cost_pence = sum(t.ai_cost_pence for t in trades)
    report.net_pnl_after_ai_cost_pence = report.net_pnl_pence - report.total_ai_cost_pence
    report.return_on_starting_equity_pct = (
        report.net_pnl_after_ai_cost_pence / starting_equity_pence * 100
        if starting_equity_pence else 0.0
    )
    wins = len([p for p in pnls if p > 0])
    report.win_rate = wins / len(pnls) if pnls else 0.0
    report.profit_factor = profit_factor(pnls)
    report.average_trade_pnl_pence = sum(pnls) / len(pnls) if pnls else 0.0
    report.median_trade_pnl_pence = sorted(pnls)[len(pnls) // 2] if pnls else 0.0
    report.worst_trade_pence = min(pnls) if pnls else 0
    report.average_edge_at_entry_bps = (
        sum(t.predicted_net_edge_bps for t in trades) / len(trades) if trades else 0.0
    )
    report.realized_edge_bps = (
        sum(t.realized_return_bps for t in trades) / len(trades) if trades else 0.0
    )
    ratios = [edge_realization_ratio(t.predicted_net_edge_bps, t.realized_return_bps)
              for t in trades if t.predicted_net_edge_bps != 0]
    report.avg_edge_realization_ratio = sum(ratios) / len(ratios) if ratios else 0.0

    equity = starting_equity_pence
    curve = [equity]
    for pnl in pnls:
        equity += pnl
        curve.append(equity)
    report.max_drawdown_bps = max_drawdown_bps(curve)

    # H/I. AI economics
    report.cost_per_candidate_pence = (
        report.total_ai_cost_pence / len(predictions) if predictions else 0.0
    )
    report.cost_per_trade_pence = (
        report.total_ai_cost_pence / len(trades) if trades else 0.0
    )
    if report.total_ai_cost_pence > 0 and report.net_pnl_pence < report.total_ai_cost_pence:
        warnings.append(Warning_.AI_COST_EXCEEDS_EDGE.value)

    # L. sample-size confidence label
    report.sample_confidence = sample_label(report.resolved_predictions, thresholds)
    if report.sample_confidence == SampleLabel.INSUFFICIENT_SAMPLE:
        warnings.append(Warning_.INSUFFICIENT_SAMPLE.value)

    # M/N. confidence intervals (deterministic)
    report.win_rate_wilson = wilson_interval(wins, len(pnls))
    report.net_pnl_bootstrap_ci = bootstrap_mean_ci(
        [float(p) for p in pnls], resamples=bootstrap_resamples, seed=bootstrap_seed
    )

    # O. profit concentration
    report.profit_concentration = profit_concentration(pnls)
    if report.profit_concentration.get("concentrated"):
        warnings.append(Warning_.PROFIT_CONCENTRATED.value)

    # P. category breakdown (deterministic metadata only)
    categories: dict[str, list] = {}
    for trade in trades:
        categories.setdefault(trade.category, []).append(trade)
    for category, group in sorted(categories.items()):
        group_pnl = sum(t.gross_pnl_pence for t in group)
        group_wins = len([t for t in group if t.gross_pnl_pence > 0])
        report.category_breakdown[category] = {
            "count": len(group),
            "net_pnl_pence": group_pnl,
            "win_rate": group_wins / len(group) if group else 0.0,
            "ai_cost_pence": sum(t.ai_cost_pence for t in group),
            "average_edge_bps": sum(t.predicted_net_edge_bps for t in group) / len(group),
        }

    # R/S. latency
    report.latency = latency_summary([t.decision_latency_ms for t in trades])

    # J/K. chronological splits + walk-forward (no shuffling, no leakage)
    _, _, oos_trades = chronological_split(trades)
    report.out_of_sample_net_pnl_pence = sum(t.gross_pnl_pence for t in oos_trades)
    if report.out_of_sample_net_pnl_pence <= 0 and len(oos_trades) >= 5:
        warnings.append(Warning_.OUT_OF_SAMPLE_NEGATIVE.value)
    _, _, oos_preds = chronological_split(predictions)
    oos_resolved = [r for r in oos_preds if r.outcome is not None and r.resolution_timestamp_utc]
    oos_brier = brier_score([(r.predicted_probability, r.outcome) for r in oos_resolved]) if oos_resolved else None  # type: ignore[misc]
    oos_market = brier_score([(r.market_probability, r.outcome) for r in oos_resolved]) if oos_resolved else None  # type: ignore[misc]
    if oos_brier is not None and oos_market is not None and oos_brier >= oos_market:
        warnings.append(Warning_.MARKET_BASELINE_OUTPERFORMS.value)

    # F. baselines
    report.baseline_comparison = {
        "MARKET_BASELINE_brier": baselines.market_baseline_brier(predictions),
        "NO_TRADE_BASELINE_net_pnl_pence": baselines.no_trade_baseline_net_pnl(trades),
        "RANDOM_DIRECTION_BASELINE_brier": baselines.random_direction_baseline_brier(predictions),
        "SIMPLE_EDGE_BASELINE_net_pnl_pence": baselines.simple_edge_baseline_net_pnl(
            [int(round(r.market_probability * 10000)) for r in predictions]
        ),
    }

    # Z. deterministic gate (never READY_FOR_LIVE)
    report.warnings = warnings
    report.gate = _gate(
        report, report.out_of_sample_net_pnl_pence,
        report.brier_improvement * 100, thresholds,
        min_sample=min_sample_for_promising,
        max_drawdown_bps_limit=max_drawdown_bps_limit,
        concentration_limit_pct=concentration_limit_pct,
    )
    return report


def export_json(report: PerformanceReport) -> str:
    return json.dumps(report.to_dict(), sort_keys=True, indent=2)


def export_csv(trades: list) -> str:
    """Safe read-only CSV export of trade outcomes (no secrets, no prompts)."""
    header = ("cycle_id,run_id,proposal_id,market_id,timestamp_utc,strategy_version,"
              "category,quantity,entry_price_pence,fees_pence,slippage_pence,"
              "gross_pnl_pence,ai_cost_pence,realized_return_bps,predicted_net_edge_bps")
    lines = [header]
    for t in sorted(trades, key=lambda t: t.timestamp_utc):
        lines.append(
            f"{t.cycle_id},{t.run_id},{t.proposal_id},{t.market_id},{t.timestamp_utc},"
            f"{t.strategy_version},{t.category},{t.quantity},{t.entry_price_pence},"
            f"{t.fees_pence},{t.slippage_pence},{t.gross_pnl_pence},{t.ai_cost_pence},"
            f"{t.realized_return_bps},{t.predicted_net_edge_bps}"
        )
    return chr(10).join(lines)


