"""Phase 6 controlled self-improvement engine.

Deterministic learning from historical paper decisions. The learner can only
PROPOSE immutable, allowlisted strategy experiments and evaluate them in
sandbox/shadow. It structurally CANNOT modify safety policy, risk limits,
budgets, HALT, execution code, or any live-trading control.
"""

from tradingagents.survivor.learning.dataset import LearningRecord, build_learning_records

__all__ = ["LearningRecord", "build_learning_records"]
