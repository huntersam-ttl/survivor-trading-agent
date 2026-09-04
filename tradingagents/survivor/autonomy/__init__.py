"""Autonomous paper loop: kill switch, lock, state, budget preflight, cycle."""

from tradingagents.survivor.autonomy.budget import BudgetPreflight, budget_preflight
from tradingagents.survivor.autonomy.cycle import (
    CycleReport,
    ResearchResult,
    autonomy_enabled,
    dry_run_enabled,
    run_survivor_cycle,
)
from tradingagents.survivor.autonomy.halt import clear_halt, is_halted, set_halt
from tradingagents.survivor.autonomy.injection import Evidence, build_evidence, detect_suspicious
from tradingagents.survivor.autonomy.lock import CycleLock
from tradingagents.survivor.autonomy.state import RuntimeState, snapshot_data_hash

__all__ = [
    "BudgetPreflight", "Candidate", "CycleLock", "CycleReport", "Evidence", "ResearchResult",
    "RuntimeState", "autonomy_enabled", "budget_preflight", "build_evidence", "clear_halt",
    "detect_suspicious", "dry_run_enabled", "is_halted", "run_survivor_cycle", "set_halt",
    "snapshot_data_hash",
]
