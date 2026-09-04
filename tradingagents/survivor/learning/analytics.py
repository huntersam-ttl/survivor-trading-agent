"""Deterministic learning analytics. No LLM required for any statistic here."""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from tradingagents.survivor.learning.dataset import LearningRecord

MIN_SAMPLE_FOR_CONCLUSION = 20


def _parse(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat((ts or "").replace("Z", "+00:00"))
    except ValueError:
        return None


def _brier(records: list[LearningRecord]) -> float | None:
    resolved = [r for r in records if r.resolved]
    if not resolved:
        return None
    return sum((r.brier_contribution or 0.0) for r in resolved) / len(resolved)


def _market_brier(records: list[LearningRecord]) -> float | None:
    resolved = [r for r in records if r.resolved]
    if not resolved:
        return None
    return sum((r.market_error or 0.0) ** 2 for r in resolved) / len(resolved)


def _group_metrics(records: list[LearningRecord]) -> dict:
    resolved = [r for r in records if r.resolved]
    executed = [r for r in records if r.executed]
    wins = sum(1 for r in executed if r.pnl_pence > 0)
    return {
        "n": len(records),
        "resolved": len(resolved),
        "trades": len(executed),
        "brier": _brier(records),
        "market_brier": _market_brier(records),
        "brier_improvement": (
            (_market_brier(records) - _brier(records))
            if _brier(records) is not None and _market_brier(records) is not None else None
        ),
        "gross_pnl_pence": sum(r.pnl_pence for r in executed),
        "ai_cost_pence": sum(r.ai_cost_pence for r in records),
        "economic_pnl_pence": sum(r.pnl_pence for r in executed)
        - sum(r.ai_cost_pence for r in records),
        "win_rate": (wins / len(executed)) if executed else None,
    }


def category_performance(records: list[LearningRecord]) -> dict[str, dict]:
    grouped: dict[str, list[LearningRecord]] = {}
    for record in records:
        grouped.setdefault(record.category, []).append(record)
    return {cat: _group_metrics(group) for cat, group in sorted(grouped.items())}


def _bucket_of(value: float | None, edges: list[float]) -> str | None:
    if value is None:
        return None
    for i, edge in enumerate(edges):
        if value < edge:
            lo = 0.0 if i == 0 else edges[i - 1]
            return f"[{lo:.2f},{edge:.2f})"
    return f">={edges[-1]:.2f}"


def probability_bucket(p: float) -> str:
    return _bucket_of(p, [0.2, 0.4, 0.6, 0.8]) or "?"


def edge_bucket(edge_bps: int) -> str:
    return _bucket_of(edge_bps / 100.0, [2.0, 5.0, 10.0]) or "?"


def time_to_resolution_bucket(record: LearningRecord) -> str | None:
    decision = _parse(record.timestamp_utc)
    resolution = _parse(record.resolution_timestamp_utc or "")
    if decision is None or resolution is None:
        return None
    days = (resolution - decision).total_seconds() / 86400
    if days < 1:
        return "<1d"
    if days < 7:
        return "1-7d"
    if days < 30:
        return "7-30d"
    return ">=30d"


def liquidity_bucket(record: LearningRecord) -> str | None:
    value = record.preresolution.get("liquidity_usd_cents")
    return _bucket_of(value / 100.0, [10.0, 100.0, 1000.0]) if value is not None else None


def spread_bucket(record: LearningRecord) -> str | None:
    value = record.preresolution.get("spread_bps")
    return _bucket_of(float(value), [100.0, 300.0, 800.0]) if value is not None else None


def research_duration_bucket(record: LearningRecord) -> str | None:
    duration = record.preresolution.get("research_duration_ms")
    if duration is None:
        return None
    seconds = duration / 1000.0
    if seconds < 30:
        return "<30s"
    if seconds < 120:
        return "30-120s"
    return ">=120s"


def bucket_performance(records: list[LearningRecord], bucket_fn) -> dict[str, dict]:
    grouped: dict[str, list[LearningRecord]] = {}
    for record in records:
        key = bucket_fn(record)
        if key is None:
            continue
        grouped.setdefault(key, []).append(record)
    return {key: _group_metrics(group) for key, group in sorted(grouped.items())}


def full_analytics(records: list[LearningRecord]) -> dict:
    """Every core statistic the learning report and generators consume."""
    return {
        "overall": _group_metrics(records),
        "by_category": category_performance(records),
        "by_probability_bucket": bucket_performance(
            records, lambda r: probability_bucket(r.survivor_probability)),
        "by_edge_bucket": bucket_performance(
            records, lambda r: edge_bucket(r.conservative_edge_bps)),
        "by_time_to_resolution": bucket_performance(records, time_to_resolution_bucket),
        "by_liquidity": bucket_performance(records, liquidity_bucket),
        "by_spread": bucket_performance(records, spread_bucket),
        "by_research_duration": bucket_performance(records, research_duration_bucket),
        "error_labels": dict(sorted(Counter(
            label for r in records for label in _record_labels(r)
        ).items())),
    }


def _record_labels(record: LearningRecord) -> list[str]:
    from tradingagents.survivor.learning.errors import classify_errors

    return classify_errors(record)


def _role_cost(records: list[LearningRecord]) -> dict[str, dict]:
    """Per-role AI cost / calls aggregated from pre-resolution usage."""
    stats: dict[str, dict] = {}
    for record in records:
        for role, usage in (record.preresolution.get("role_usage") or {}).items():
            entry = stats.setdefault(role, {"calls": 0, "cost_pence": 0, "samples": 0})
            entry["calls"] += usage.get("calls", 0)
            entry["cost_pence"] += usage.get("cost_pence", 0)
            entry["samples"] += 1
    return stats


def role_value_analysis(records: list[LearningRecord]) -> list[dict]:
    """Estimate each research role's information value. EVIDENCE ONLY — the
    learner never removes roles itself. Conclusions require sufficient sample."""
    role_cost = _role_cost(records)
    results = []
    for role in sorted(role_cost):
        stats = role_cost[role]
        results.append({
            "role": role,
            "samples": stats["samples"],
            "calls": stats["calls"],
            "ai_cost_pence": stats["cost_pence"],
            "cost_per_resolved_forecast_pence": (
                round(stats["cost_pence"] / stats["samples"], 2) if stats["samples"] else None
            ),
            "conclusion": (
                "INSUFFICIENT_SAMPLE"
                if stats["samples"] < MIN_SAMPLE_FOR_CONCLUSION else "MEASURED"
            ),
        })

    # bear/trader divergence counterfactual (persisted pre-resolution flag)
    divergent = [r for r in records if r.preresolution.get("bear_disagreed_with_buy")]
    lost_divergent = [r for r in divergent if r.executed and r.resolved and r.outcome == 0.0]
    if divergent:
        avoided_gain = sum(
            max(0.0, (r.market_error or 0.0) ** 2 - 0.25) for r in lost_divergent
        ) / len(divergent)
        results.append({
            "role": "bear",
            "counterfactual": (
                f"When Bear disagreed with the executed BUY, avoiding the trade "
                f"would have improved Brier by {avoided_gain:.4f} across "
                f"{len(divergent)} resolved samples "
                f"({len(lost_divergent)} of them lost trades)."
            ),
            "sample_size": len(divergent),
            "sufficient_sample": len(divergent) >= MIN_SAMPLE_FOR_CONCLUSION,
        })
    return results


def model_value_analysis(records: list[LearningRecord]) -> list[dict]:
    """Provider/model/role cost & quality accounting. Produces candidate
    routing EXPERIMENTS only — never changes routing automatically."""
    groups: dict[str, dict] = {}
    for record in records:
        for role, usage in (record.preresolution.get("role_usage") or {}).items():
            key = f"{usage.get('provider')}/{usage.get('model')}/{role}"
            entry = groups.setdefault(key, {
                "provider": usage.get("provider"), "model": usage.get("model"),
                "role": role, "calls": 0, "cost_pence": 0, "samples": 0,
                "failures": 0,
            })
            entry["calls"] += usage.get("calls", 0)
            entry["cost_pence"] += usage.get("cost_pence", 0)
            entry["samples"] += 1
            if record.resolved and (record.survivor_error or 0) > (record.market_error or 0):
                entry["failures"] += 1
    results = []
    for key in sorted(groups):
        entry = dict(groups[key])
        entry["key"] = key
        entry["cost_per_successful_forecast_pence"] = round(
            entry["cost_pence"] / max(1, entry["samples"] - entry["failures"]), 2)
        entry["failure_rate"] = (
            round(entry["failures"] / entry["samples"], 3) if entry["samples"] else None
        )
        entry["sufficient_sample"] = entry["samples"] >= MIN_SAMPLE_FOR_CONCLUSION
        results.append(entry)
    return results


def calibration_weaknesses(records: list[LearningRecord]) -> list[str]:
    by_bucket = bucket_performance(
        records, lambda r: probability_bucket(r.survivor_probability))
    return [
        f"calibration worse than market in bucket {bucket}"
        for bucket, metrics in sorted(by_bucket.items())
        if metrics["brier"] is not None and metrics["market_brier"] is not None
        and metrics["brier"] > metrics["market_brier"]
        and metrics["resolved"] >= MIN_SAMPLE_FOR_CONCLUSION
    ]


def edge_realization_weaknesses(records: list[LearningRecord]) -> list[str]:
    executed = [r for r in records if r.executed and r.resolved]
    if len(executed) < MIN_SAMPLE_FOR_CONCLUSION:
        return ["edge realization: INSUFFICIENT_SAMPLE"]
    overstated = sum(1 for r in executed if r.conservative_edge_bps > 0 and r.pnl_pence <= 0)
    return [f"edge overstated in {overstated}/{len(executed)} executed trades"]

