"""Immutable StrategyExperiment schema with a hard allowlist of changes.

The learner can ONLY propose changes in ALLOWED_CHANGES. Anything touching
safety policy, risk limits, budgets, HALT, execution, or live-trading controls
is rejected structurally (ForbiddenChangeError) — inside and outside the
learning namespace.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ExperimentStatus(str, Enum):
    PROPOSED = "PROPOSED"
    SANDBOX_TESTING = "SANDBOX_TESTING"
    SHADOW_TESTING = "SHADOW_TESTING"
    REJECTED = "REJECTED"
    CANDIDATE_FOR_PROMOTION = "CANDIDATE_FOR_PROMOTION"
    PROMOTED_MANUALLY = "PROMOTED_MANUALLY"


class ForbiddenChangeError(ValueError):
    """A proposed change is outside the learning namespace. Fail closed."""


# Keys the learner may propose. ANYTHING ELSE is forbidden.
ALLOWED_CHANGES = frozenset({
    "scanner_rank_weights",
    "scanner_filter_bounds",
    "research_depth",
    "debate_rounds",
    "role_output_token_limits",
    "role_model_routing",
    "prompt_templates",
    "uncertainty_penalty_bps",
    "min_research_confidence",
    "category_no_trade_thresholds",
})

# The immutable boundary — substring-matched against the whole proposed diff.
# Both underscore and natural-language forms are blocked (a prompt template
# must not be able to smuggle "enable live trading" instructions either).
FORBIDDEN_PATTERNS = (
    "real_trading", "real trading", "live_trading", "live trading", "wallet",
    "broker_enabled", "withdrawal", "borrowing", "leverage", "max_position",
    "max_exposure", "daily_loss", "drawdown_halt", "halt", "budget",
    "api_hard_budget", "ledger_integrity", "execution_boundary", "paper_only",
    "owner", "admin", "risk_limit", "min_edge_bps", "paper_ledger",
)


def validate_config_diff(diff: dict[str, Any]) -> None:
    """Reject any proposed change outside the allowlist. Fail closed."""
    if not isinstance(diff, dict):
        raise ForbiddenChangeError("proposed_config_diff must be a dict")
    blob = json.dumps(diff, sort_keys=True).lower()
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in blob:
            raise ForbiddenChangeError(
                f"forbidden configuration change: contains '{pattern}' "
                "(outside the learning namespace)"
            )
    for key in diff:
        if key not in ALLOWED_CHANGES:
            raise ForbiddenChangeError(
                f"config key '{key}' is not in the experiment allowlist"
            )


def experiment_id_for(parent_version: str, created_at: str, hypothesis: str) -> str:
    digest = hashlib.sha256(
        f"{parent_version}|{created_at}|{hypothesis}".encode()
    ).hexdigest()[:12]
    return f"exp-{digest}"


@dataclass(frozen=True)
class StrategyExperiment:
    """Immutable candidate strategy improvement. Never auto-promoted."""

    experiment_id: str
    parent_strategy_version: str
    created_at: str
    hypothesis: str
    evidence: dict[str, Any]
    sample_size: int
    allowed_changes: frozenset[str]
    proposed_config_diff: dict[str, Any]
    expected_effect: str
    evaluation_plan: dict[str, Any]
    status: ExperimentStatus = ExperimentStatus.PROPOSED

    def __post_init__(self) -> None:
        if not self.hypothesis or not isinstance(self.hypothesis, str):
            raise ValueError("hypothesis is required")
        if not isinstance(self.evidence, dict) or not self.evidence:
            raise ValueError("evidence (dict) is required")
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if set(self.allowed_changes) - ALLOWED_CHANGES:
            raise ForbiddenChangeError(
                "allowed_changes contains non-allowlisted change types"
            )
        validate_config_diff(self.proposed_config_diff)
        if not isinstance(self.evaluation_plan, dict) or \
                not {"metric", "criteria"}.issubset(self.evaluation_plan):
            raise ValueError("evaluation_plan must contain 'metric' and 'criteria'")

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "parent_strategy_version": self.parent_strategy_version,
            "created_at": self.created_at,
            "hypothesis": self.hypothesis,
            "evidence": self.evidence,
            "sample_size": self.sample_size,
            "allowed_changes": sorted(self.allowed_changes),
            "proposed_config_diff": self.proposed_config_diff,
            "expected_effect": self.expected_effect,
            "evaluation_plan": self.evaluation_plan,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> StrategyExperiment:
        return cls(
            experiment_id=raw["experiment_id"],
            parent_strategy_version=raw["parent_strategy_version"],
            created_at=raw["created_at"],
            hypothesis=raw["hypothesis"],
            evidence=raw["evidence"],
            sample_size=raw["sample_size"],
            allowed_changes=frozenset(raw["allowed_changes"]),
            proposed_config_diff=raw["proposed_config_diff"],
            expected_effect=raw["expected_effect"],
            evaluation_plan=raw["evaluation_plan"],
            status=ExperimentStatus(raw["status"]),
        )


def validate_proposal_schema(raw: dict) -> StrategyExperiment:
    """Schema-validate a structured experiment proposal (from an LLM, Jarvis,
    or the deterministic generator). The proposer has NO safety configuration
    write access — anything outside the allowlist is rejected here."""
    required = ("hypothesis", "evidence", "proposed_config_diff",
                "expected_effect", "evaluation_plan", "sample_size",
                "parent_strategy_version", "created_at")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"proposal missing required fields: {missing}")
    return StrategyExperiment(
        experiment_id=experiment_id_for(
            raw["parent_strategy_version"], raw["created_at"], raw["hypothesis"]),
        parent_strategy_version=raw["parent_strategy_version"],
        created_at=raw["created_at"],
        hypothesis=raw["hypothesis"],
        evidence=raw["evidence"],
        sample_size=raw["sample_size"],
        allowed_changes=frozenset(ALLOWED_CHANGES),
        proposed_config_diff=raw["proposed_config_diff"],
        expected_effect=raw["expected_effect"],
        evaluation_plan=raw["evaluation_plan"],
        status=ExperimentStatus.PROPOSED,
    )


def generate_experiments(analytics: dict, parent_version: str,
                         created_at: str) -> list[StrategyExperiment]:
    """Deterministic (no-LLM) experiment generator from aggregated analytics.
    The optional LLM proposer must emit the same schema via
    validate_proposal_schema — it never writes safety configuration."""
    from tradingagents.survivor.learning.analytics import MIN_SAMPLE_FOR_CONCLUSION

    experiments: list[StrategyExperiment] = []

    for category, metrics in analytics.get("by_category", {}).items():
        if metrics["resolved"] >= MIN_SAMPLE_FOR_CONCLUSION and \
                metrics["brier_improvement"] is not None and \
                metrics["brier_improvement"] < 0:
            experiments.append(StrategyExperiment(
                experiment_id=experiment_id_for(
                    parent_version, created_at, f"category-gate:{category}"),
                parent_strategy_version=parent_version,
                created_at=created_at,
                hypothesis=(
                    f"Tightening the NO_TRADE threshold for category '{category}' "
                    f"will improve economic P/L without hurting Brier."
                ),
                evidence={"category": category, **metrics},
                sample_size=metrics["resolved"],
                allowed_changes=frozenset({"category_no_trade_thresholds"}),
                proposed_config_diff={"category_no_trade_thresholds": {category: 0.60}},
                expected_effect="fewer low-edge trades in weak category",
                evaluation_plan={
                    "metric": "economic_pnl_pence",
                    "criteria": "OOS economic P/L improves; Brier not worse; "
                                "drawdown not materially higher",
                },
            ))

    overconfident = analytics.get("error_labels", {}).get("OVERCONFIDENT", 0)
    if overconfident >= MIN_SAMPLE_FOR_CONCLUSION:
        experiments.append(StrategyExperiment(
            experiment_id=experiment_id_for(
                parent_version, created_at, "uncertainty-penalty"),
            parent_strategy_version=parent_version,
            created_at=created_at,
            hypothesis=(
                f"{overconfident} overconfident resolved decisions suggest a "
                f"higher uncertainty penalty will improve calibration."
            ),
            evidence={"OVERCONFIDENT": overconfident},
            sample_size=analytics["overall"]["resolved"],
            allowed_changes=frozenset({"uncertainty_penalty_bps"}),
            proposed_config_diff={"uncertainty_penalty_bps": 150},
            expected_effect="fewer marginal trades; better calibration",
            evaluation_plan={
                "metric": "brier",
                "criteria": "IS and OOS Brier improve; AI cost not >10% higher",
            },
        ))
    return experiments

