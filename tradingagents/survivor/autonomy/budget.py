"""AI budget preflight: SKIP candidates before research when budget is unavailable.

A candidate that cannot be fully researched must not be partially researched —
an under-informed trade is worse than no trade (NO_TRADE default).
"""

from __future__ import annotations

from dataclasses import dataclass

from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger
from tradingagents.survivor.policy import SurvivorPolicy


@dataclass(frozen=True)
class BudgetPreflight:
    ok: bool
    reason: str  # "OK" | "AI_BUDGET_UNAVAILABLE" | detail


def budget_preflight(
    policy: SurvivorPolicy,
    usage_ledger: InferenceUsageLedger,
    estimated_cost_pence: int = 0,
) -> BudgetPreflight:
    """Fail closed: research only when daily AND monthly global budget headroom exists."""
    daily_spend = usage_ledger.get_daily_spend_pence(None)
    monthly_spend = usage_ledger.get_monthly_spend_pence(None)

    if daily_spend + estimated_cost_pence >= policy.global_daily_pence:
        return BudgetPreflight(
            ok=False,
            reason=f"AI_BUDGET_UNAVAILABLE: daily spend {daily_spend}p >= limit {policy.global_daily_pence}p",
        )
    if monthly_spend + estimated_cost_pence >= policy.global_monthly_pence:
        return BudgetPreflight(
            ok=False,
            reason=f"AI_BUDGET_UNAVAILABLE: monthly spend {monthly_spend}p >= limit {policy.global_monthly_pence}p",
        )
    return BudgetPreflight(ok=True, reason="OK")
