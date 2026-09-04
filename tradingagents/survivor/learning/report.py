"""SURVIVOR LEARNING REPORT — deterministic rendering. The current strategy
 ALWAYS remains unchanged by report generation."""

from __future__ import annotations

from tradingagents.survivor.learning.analytics import (
    calibration_weaknesses,
    category_performance,
    edge_realization_weaknesses,
    full_analytics,
    model_value_analysis,
    role_value_analysis,
)
from tradingagents.survivor.learning.errors import (
    category_weakness_labels,
    model_role_weakness_labels,
)
from tradingagents.survivor.learning.experiments import generate_experiments


def generate_learning_report(records: list, current_version: str = "survivor-v1.0",
                             created_at: str = "") -> dict:
    analytics = full_analytics(records)
    return {
        "resolved_sample": analytics["overall"]["resolved"],
        "error_labels": analytics["error_labels"],
        "by_category": category_performance(records),
        "role_value": role_value_analysis(records),
        "model_value": model_value_analysis(records),
        "calibration_weaknesses": calibration_weaknesses(records),
        "edge_realization_weaknesses": edge_realization_weaknesses(records),
        "category_weakness_labels": category_weakness_labels(records),
        "model_role_weakness_labels": model_role_weakness_labels(records),
        "proposed_experiments": [
            e.to_dict() for e in generate_experiments(
                analytics, current_version, created_at)
        ],
        "current_strategy_remains": current_version,
    }


def render_learning_report(report: dict) -> str:
    overall = report["by_category"]
    lines = [
        "",
        "SURVIVOR LEARNING REPORT",
        "",
        f"Resolved sample: {report['resolved_sample']}",
        f"What worked: {sum(1 for m in overall.values() if (m['brier_improvement'] or 0) > 0)} "
        f"category(ies) beat the market baseline",
        f"What failed: top error labels "
        f"{dict(list(report['error_labels'].items())[:5])}",
        "Strongest categories: " + _extremes(report["by_category"], best=True),
        "Weakest categories: " + _extremes(report["by_category"], best=False),
        "Best model/role combinations: " + _best_models(report["model_value"]),
        "Most expensive low-value role: " + _expensive_low_value(report["role_value"]),
        "Calibration weaknesses: " + ("; ".join(report["calibration_weaknesses"]) or "none measured"),
        "Edge-realization weaknesses: " + ("; ".join(report["edge_realization_weaknesses"]) or "none measured"),
        f"Proposed experiments: {len(report['proposed_experiments'])}",
    ]
    for experiment in report["proposed_experiments"]:
        lines.append(f"  - {experiment['experiment_id']}: {experiment['hypothesis']}")
    lines.append(f"Current strategy remains: {report['current_strategy_remains']}")
    lines.append("No automatic changes applied.")
    return "\n".join(lines) + "\n"


def _extremes(by_category: dict, best: bool) -> str:
    scored = [(c, m["brier_improvement"]) for c, m in by_category.items()
              if m["brier_improvement"] is not None]
    if not scored:
        return "none measured"
    scored.sort(key=lambda x: x[1], reverse=best)
    return f"{scored[0][0]} (brier improvement {scored[0][1]:+.4f})"


def _best_models(model_value: list[dict]) -> str:
    sufficient = [m for m in model_value if m.get("sufficient_sample")]
    if not sufficient:
        return "INSUFFICIENT_SAMPLE"
    best = min(sufficient, key=lambda m: m["failure_rate"] or 1.0)
    return f"{best['key']} (failure rate {best['failure_rate']})"


def _expensive_low_value(role_value: list[dict]) -> str:
    measured = [r for r in role_value if r.get("conclusion") == "MEASURED"]
    if not measured:
        return "INSUFFICIENT_SAMPLE"
    worst = max(measured, key=lambda r: r["ai_cost_pence"])
    return f"{worst['role']} (cost {worst['ai_cost_pence']}p across {worst['samples']} samples)"
