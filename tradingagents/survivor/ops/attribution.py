"""Model/failure/cost analytics. Analysis only - no auto-routing, no auto-tuning."""

from __future__ import annotations


def model_attribution(usage_ledger) -> dict:
    """AI cost by provider/model/role + downstream trade counts, from usage.db."""
    with usage_ledger._get_connection() as conn:
        rows = conn.execute(
            "SELECT provider, model, agent_role, COUNT(*) AS calls, "
            "SUM(gbp_cost_pence) AS cost FROM usage_records "
            "WHERE status = 'SUCCESS' GROUP BY provider, model, agent_role"
        ).fetchall()
    attribution = {}
    for r in rows:
        key = f"{r['provider']}/{r['model']}"
        entry = attribution.setdefault(key, {
            "total_cost_pence": 0, "calls": 0, "roles": {}})
        entry["total_cost_pence"] += int(r["cost"] or 0)
        entry["calls"] += int(r["calls"])
        entry["roles"][r["agent_role"]] = int(r["cost"] or 0)
    return attribution


FAILURE_KEYS = (
    "market_fetch", "scanner_reject", "ai_budget_unavailable", "provider_unhealthy",
    "inference_failed", "proposal_invalid", "risk_rejected", "execution_rejected",
    "ledger_failure", "resolution_failure",
)


def failure_analytics(counts: dict) -> dict:
    """Normalize failure counts into totals + percentages."""
    clean = {k: int(counts.get(k, 0)) for k in FAILURE_KEYS}
    total = sum(clean.values())
    return {
        "counts": clean,
        "total": total,
        "percentages": {
            k: (round(v / total * 100, 2) if total else 0.0) for k, v in clean.items()
        },
    }


def cost_efficiency_metrics(
    *,
    total_ai_cost_pence: int,
    cycles: int,
    candidates_researched: int,
    valid_proposals: int,
    approved_trades: int,
    resolved_profitable_trades: int,
    economic_pnl_pence: int,
) -> dict:
    """Cost efficiency ratios. No automatic optimization."""
    def per(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    return {
        "ai_cost_per_cycle_pence": per(total_ai_cost_pence, cycles),
        "ai_cost_per_candidate_pence": per(total_ai_cost_pence, candidates_researched),
        "ai_cost_per_proposal_pence": per(total_ai_cost_pence, valid_proposals),
        "ai_cost_per_approved_trade_pence": per(total_ai_cost_pence, approved_trades),
        "ai_cost_per_resolved_profitable_trade_pence": per(
            total_ai_cost_pence, resolved_profitable_trades),
        "economic_pnl_per_ai_cost": per(economic_pnl_pence, total_ai_cost_pence),
    }


def equity_curve(points: list[dict]) -> list[dict]:
    """Validate + normalize capital-curve points (timestamp, cash, exposure,
    equity, drawdown, daily P/L). Read-only export."""
    curve = []
    for point in sorted(points, key=lambda p: p["timestamp_utc"]):
        curve.append({
            "timestamp_utc": point["timestamp_utc"],
            "cash_pence": int(point["cash_pence"]),
            "exposure_pence": int(point["exposure_pence"]),
            "equity_pence": int(point["equity_pence"]),
            "drawdown_bps": int(point.get("drawdown_bps", 0)),
            "daily_pnl_pence": int(point.get("daily_pnl_pence", 0)),
        })
    return curve
