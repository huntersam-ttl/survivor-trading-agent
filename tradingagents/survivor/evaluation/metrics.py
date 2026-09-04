"""Deterministic evaluation metrics. No LLM, no randomness except seeded bootstrap."""

from __future__ import annotations

import math
import random

from tradingagents.survivor.evaluation.types import CalibrationBin, PredictionRecord


def brier_score(forecasts: list[tuple[float, float]]) -> float:
    """Brier = mean((forecast - outcome)^2); outcome 1.0 = YES, 0.0 = NO."""
    if not forecasts:
        return 0.0
    return sum((p - o) ** 2 for p, o in forecasts) / len(forecasts)


def log_loss(forecasts: list[tuple[float, float]], eps: float = 1e-12) -> float:
    if not forecasts:
        return 0.0
    total = 0.0
    for p, o in forecasts:
        p = min(max(p, eps), 1.0 - eps)
        total += -(o * math.log(p) + (1 - o) * math.log(1 - p))
    return total / len(forecasts)


def brier_improvement(strategy: list[tuple[float, float]], baseline: list[tuple[float, float]]) -> float:
    """Fractional improvement of strategy Brier over baseline Brier (negative = worse)."""
    base = brier_score(baseline)
    if base == 0.0:
        return 0.0
    return (base - brier_score(strategy)) / base


def calibration_bins(
    records: list[PredictionRecord], bin_width_bps: int = 500
) -> list[CalibrationBin]:
    """Bin resolved predictions by PREDICTED probability; compare to outcome frequency."""
    resolved = [r for r in records if r.outcome is not None]
    bins: dict[int, list[PredictionRecord]] = {}
    for record in resolved:
        key = int(record.predicted_probability * 10000) // bin_width_bps * bin_width_bps
        bins.setdefault(key, []).append(record)
    result = []
    for lower in sorted(bins):
        group = bins[lower]
        mean_pred = sum(r.predicted_probability for r in group) / len(group)
        freq = sum(r.outcome for r in group) / len(group)  # type: ignore[operator]
        result.append(CalibrationBin(
            lower_bps=lower, upper_bps=lower + bin_width_bps, count=len(group),
            mean_predicted=mean_pred, outcome_frequency=freq,
            calibration_error=abs(mean_pred - freq),
        ))
    return result


def wilson_interval(successes: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Deterministic Wilson score interval for a binomial proportion (95% by default)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean_ci(
    values: list[float], resamples: int = 500, seed: int = 42, confidence: float = 0.95
) -> tuple[float, float]:
    """Deterministic bootstrap CI for the mean (fixed seed, percentile method)."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1 - confidence) / 2
    lo = means[int(alpha * resamples)]
    hi = means[min(int((1 - alpha) * resamples), resamples - 1)]
    return (lo, hi)


def max_drawdown_bps(equity_curve: list[int]) -> int:
    """Max drawdown in bps from an equity curve (integer pence). HWM monotonic."""
    if not equity_curve:
        return 0
    peak = equity_curve[0]
    worst = 0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) * 10000 // peak
            worst = max(worst, dd)
    return worst


def profit_factor(pnls: list[int]) -> float:
    """gross wins / |gross losses|; inf when no losses; 0.0 when no wins."""
    wins = sum(p for p in pnls if p > 0)
    losses = sum(-p for p in pnls if p < 0)
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def profit_concentration(pnls: list[int]) -> dict:
    """Share of positive P/L contributed by top trades. Flags dependence on few trades."""
    positive = sorted((p for p in pnls if p > 0), reverse=True)
    total_positive = sum(positive)
    if total_positive <= 0:
        return {"top1_pct": 0.0, "top5_pct": 0.0, "top10pct_pct": 0.0, "concentrated": False}
    top1 = positive[0] / total_positive * 100 if positive else 0.0
    top5 = sum(positive[:5]) / total_positive * 100
    k = max(1, int(math.ceil(len(pnls) * 0.1))) if pnls else 1
    top10 = sum(positive[:k]) / total_positive * 100
    # concentrated = a single trade (or a tiny minority) provides most of the profit
    concentrated = (len(positive) >= 3 and top1 > 80) or (len(positive) >= 10 and top5 > 90)
    return {"top1_pct": round(top1, 2), "top5_pct": round(top5, 2),
            "top10pct_pct": round(top10, 2), "concentrated": bool(concentrated)}


def percentile(values: list[float], pct: float) -> float:
    """Deterministic nearest-rank percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(pct / 100 * len(ordered))) - 1))
    return ordered[idx]


def latency_summary(latencies_ms: list[int]) -> dict:
    return {
        "p50_ms": percentile([float(v) for v in latencies_ms], 50),
        "p90_ms": percentile([float(v) for v in latencies_ms], 90),
        "p99_ms": percentile([float(v) for v in latencies_ms], 99),
    }


def edge_realization_ratio(predicted_net_edge_bps: int, realized_return_bps: int) -> float:
    """realized / predicted edge; 0.0 when prediction is zero (no division by zero)."""
    if predicted_net_edge_bps == 0:
        return 0.0
    return realized_return_bps / predicted_net_edge_bps
