"""AI budget operations: deterministic spend alerts for long-run trials.

Thin re-export of the autonomy budget control plane so operations code has a
single import point. Logic lives in autonomy/budget.py and is tested there;
no thresholds are duplicated or auto-adjusted here.
"""

from __future__ import annotations

from tradingagents.survivor.autonomy.budget import (
    BudgetPreflight,
    budget_preflight,
    spend_alerts,
)

__all__ = ["BudgetPreflight", "budget_preflight", "spend_alerts"]
