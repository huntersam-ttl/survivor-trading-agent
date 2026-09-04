"""Survivor policy configuration and hard limits."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from .types import AgentRole, SurvivorMode


def pounds_to_pence(value: str | int | float | Decimal) -> int:
    """Convert GBP pounds string/number to integer pence deterministically without float rounding errors.

    Examples:
        "30" -> 3000
        "30.00" -> 3000
        "0.35" -> 35
    """
    if isinstance(value, int):
        return value * 100
    try:
        d = Decimal(str(value)).quantize(Decimal("0.01"))
        return int(d * 100)
    except (ValueError, TypeError, InvalidOperation) as exc:
        raise ValueError(f"Invalid monetary pound string {value!r}") from exc


@dataclass(frozen=True)
class SurvivorPolicy:
    """Immutable-ish policy defining budget caps, role limits, and zero-money trading boundaries."""

    mode: SurvivorMode = SurvivorMode.PAPER_ONLY

    # Mandatory Safety Boundaries (All MUST stay False)
    real_trading_enabled: bool = False
    wallet_enabled: bool = False
    broker_enabled: bool = False
    withdrawals_enabled: bool = False
    borrowing_enabled: bool = False
    leverage_enabled: bool = False

    # Monthly Provider Budgets (Pence)
    openai_monthly_pence: int = 3000   # £30.00
    deepseek_monthly_pence: int = 1000  # £10.00
    minimax_monthly_pence: int = 1000   # £10.00
    global_monthly_pence: int = 5000    # £50.00

    # Daily Provider Budgets (Pence)
    openai_daily_pence: int = 100      # £1.00
    deepseek_daily_pence: int = 35     # £0.35
    minimax_daily_pence: int = 35      # £0.35
    global_daily_pence: int = 170      # £1.70

    # FX Rate USD -> GBP (e.g. 1.30 means $1.30 = £1.00)
    usd_gbp_rate: Decimal = Decimal("1.30")

    # Hard retry ceiling
    max_retries_ceiling: int = 2

    # Role Output Token Caps
    role_max_output_tokens: dict[str, int] = field(
        default_factory=lambda: {
            AgentRole.QUICK_ANALYST.value: 800,
            AgentRole.MARKET_ANALYST.value: 1200,
            AgentRole.SOCIAL_ANALYST.value: 1200,
            AgentRole.NEWS_ANALYST.value: 1200,
            AgentRole.FUNDAMENTALS_ANALYST.value: 1200,
            AgentRole.BULL_RESEARCHER.value: 1000,
            AgentRole.BEAR_RESEARCHER.value: 1000,
            AgentRole.RESEARCH_MANAGER.value: 1200,
            AgentRole.TRADER.value: 800,
            AgentRole.AGGRESSIVE_RISK.value: 1000,
            AgentRole.CONSERVATIVE_RISK.value: 1000,
            AgentRole.NEUTRAL_RISK.value: 1000,
            AgentRole.PORTFOLIO_MANAGER.value: 600,
        }
    )

    def get_provider_monthly_budget(self, provider: str) -> int:
        p = provider.lower()
        if p == "openai":
            return self.openai_monthly_pence
        if p in ("deepseek", "deepseek-chat", "deepseek-reasoner"):
            return self.deepseek_monthly_pence
        if p in ("minimax", "minimax-cn"):
            return self.minimax_monthly_pence
        # Default fallback to provider allowance or global
        return self.global_monthly_pence

    def get_provider_daily_budget(self, provider: str) -> int:
        p = provider.lower()
        if p == "openai":
            return self.openai_daily_pence
        if p in ("deepseek", "deepseek-chat", "deepseek-reasoner"):
            return self.deepseek_daily_pence
        if p in ("minimax", "minimax-cn"):
            return self.minimax_daily_pence
        return self.global_daily_pence

    def get_role_max_tokens(self, role: str) -> int:
        return self.role_max_output_tokens.get(role, 1000)

    @classmethod
    def from_env(cls) -> SurvivorPolicy:
        """Create policy loading limits from environment variables with safe Decimal/int parsing."""
        def parse_gbp_env(env_var: str, default_pence: int) -> int:
            val = os.environ.get(env_var)
            if not val:
                return default_pence
            return pounds_to_pence(val)

        usd_gbp_raw = os.environ.get("SURVIVOR_USD_GBP_RATE", "1.30")
        try:
            usd_gbp_rate = Decimal(str(usd_gbp_raw))
        except (ValueError, TypeError, InvalidOperation) as exc:
            raise ValueError(f"Invalid SURVIVOR_USD_GBP_RATE: {usd_gbp_raw!r}") from exc

        return cls(
            openai_monthly_pence=parse_gbp_env("SURVIVOR_OPENAI_MONTHLY_GBP", 3000),
            deepseek_monthly_pence=parse_gbp_env("SURVIVOR_DEEPSEEK_MONTHLY_GBP", 1000),
            minimax_monthly_pence=parse_gbp_env("SURVIVOR_MINIMAX_MONTHLY_GBP", 1000),
            global_monthly_pence=parse_gbp_env("SURVIVOR_GLOBAL_MONTHLY_GBP", 5000),
            openai_daily_pence=parse_gbp_env("SURVIVOR_OPENAI_DAILY_GBP", 100),
            deepseek_daily_pence=parse_gbp_env("SURVIVOR_DEEPSEEK_DAILY_GBP", 35),
            minimax_daily_pence=parse_gbp_env("SURVIVOR_MINIMAX_DAILY_GBP", 35),
            global_daily_pence=parse_gbp_env("SURVIVOR_GLOBAL_DAILY_GBP", 170),
            usd_gbp_rate=usd_gbp_rate,
        )
