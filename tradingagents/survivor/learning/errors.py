"""Deterministic multi-label error classification for resolved decisions.

Pure functions over LearningRecord — no LLM, no wall-clock, no network. One
decision may carry several labels; output order is deterministic.
"""

from __future__ import annotations

from collections import Counter

from tradingagents.survivor.learning.dataset import LearningRecord

LATENCY_FAILURE_MS = 120_000     # 2 minutes of research = latency failure
HIGH_AI_COST_PENCE = 25          # >£0.25 AI cost on a single binary decision
MARKET_MOVE_PENCE = 5            # entry deviates >=5p from decision-time market


def classify_errors(record: LearningRecord) -> list[str]:
    """Return every deterministic error label that applies to this record."""
    if not record.resolved:
        return []
    labels: list[str] = []
    predicted = record.survivor_probability
    market = record.market_probability
    outcome = record.outcome
    survivor_err = record.survivor_error
    market_err = record.market_error

    # calibration vs the 50/50 direction and the market baseline
    if (predicted >= 0.5) != (market >= 0.5) and survivor_err > market_err:
        labels.append("WRONG_DIRECTION")
    if predicted >= 0.8 and outcome == 0.0 or predicted <= 0.2 and outcome == 1.0:
        labels.append("OVERCONFIDENT")
    if (outcome == 1.0 and predicted < market - 0.05) or \
            (outcome == 0.0 and predicted > market + 0.05):
        labels.append("UNDERCONFIDENT")

    # edge economics (only meaningful for executed trades)
    if record.executed:
        if record.conservative_edge_bps > 0 and record.pnl_pence <= 0:
            labels.append("EDGE_OVERSTATED")
        if record.predicted_edge_bps > 0 and record.conservative_edge_bps <= 0:
            labels.append("EXECUTION_COST_ERASED_EDGE")
        if record.predicted_edge_bps > 0 and \
                record.realized_return_bps < record.predicted_edge_bps // 2:
            labels.append("EDGE_DECAY")
        if outcome == 0.0:
            labels.append("FALSE_POSITIVE")
        if abs(record.execution_price_pence - round(market * 100)) >= MARKET_MOVE_PENCE:
            labels.append("MARKET_MOVED_DURING_RESEARCH")
    elif outcome == 1.0 and record.conservative_edge_bps > 0:
        labels.append("FALSE_NEGATIVE")

    # research-stage weaknesses (require persisted role conclusions)
    bull = (record.preresolution.get("bull_conclusion") or "").lower()
    bear = (record.preresolution.get("bear_conclusion") or "").lower()
    if record.executed and outcome == 0.0 and "risk" in bull and "risk" not in bear:
        labels.append("BULL_MISSED_RISK")
    if not record.executed and outcome == 1.0 and "too uncertain" in bear:
        labels.append("BEAR_TOO_CONSERVATIVE")
    if record.executed and outcome == 0.0 and "overridden" in \
            (record.preresolution.get("research_manager_decision") or "").lower():
        labels.append("RESEARCH_MANAGER_ERROR")

    # operational
    if record.decision_latency_ms > LATENCY_FAILURE_MS:
        labels.append("LATENCY_FAILURE")
    if record.ai_cost_pence > HIGH_AI_COST_PENCE:
        labels.append("HIGH_AI_COST")
    if not record.executed and survivor_err < market_err:
        labels.append("GOOD_NO_TRADE")

    return labels


def error_frequency(records: list[LearningRecord]) -> dict[str, int]:
    """Deterministic label frequency across resolved records."""
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(classify_errors(record))
    return dict(sorted(counter.items()))


def category_weakness_labels(records: list[LearningRecord], min_sample: int = 5) -> list[str]:
    """Categories whose resolved record set is a persistent weakness."""
    by_category: dict[str, list[LearningRecord]] = {}
    for record in records:
        by_category.setdefault(record.category, []).append(record)
    weak = []
    for category, group in sorted(by_category.items()):
        if len(group) < min_sample:
            continue
        losses = sum(1 for r in group if r.resolved and r.survivor_error and r.survivor_error > r.market_error)
        if losses / len(group) > 0.5:
            weak.append(f"CATEGORY_WEAKNESS:{category}")
    return weak


def model_role_weakness_labels(records: list[LearningRecord], min_sample: int = 5) -> list[str]:
    """Provider/model/role combinations that systematically underperform."""
    weak = []
    for key, group in _group_by_role_model(records).items():
        if len(group) < min_sample:
            continue
        failures = sum(1 for r in group if r.resolved and r.survivor_error and r.survivor_error > r.market_error)
        if failures / len(group) > 0.5:
            weak.append(f"MODEL_ROLE_WEAKNESS:{key}")
    return weak


def _group_by_role_model(records: list[LearningRecord]) -> dict[str, list[LearningRecord]]:
    groups: dict[str, list[LearningRecord]] = {}
    for record in records:
        for role, usage in (record.preresolution.get("role_usage") or {}).items():
            key = f"{usage.get('provider')}/{usage.get('model')}/{role}"
            groups.setdefault(key, []).append(record)
    return groups
