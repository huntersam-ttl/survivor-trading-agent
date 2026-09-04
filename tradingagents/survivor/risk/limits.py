"""Risk limits for the Survivor paper-trading boundary."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    """Deterministic risk policy. Money in integer pence, ratios in bps."""

    initial_equity_pence: int = 2000          # £20.00 fictional treasury
    max_single_position_pence: int = 100      # £1.00
    max_total_exposure_pence: int = 500       # £5.00
    daily_loss_limit_pence: int = 100         # £1.00 (realized + unrealized)
    max_drawdown_bps: int = 1500              # 15%
    min_conservative_net_edge_bps: int = 500  # 5%
    max_proposal_age_sec: int = 300           # stale-proposal window
    max_market_data_age_sec: int = 120        # stale-market window
    allow_shorting: bool = False
    allow_leverage: bool = False
    allow_borrowing: bool = False


def risk_limits_from_env() -> RiskLimits:
    """Optional env overrides; always integer pence/bps, fail-closed parsing."""
    def _int(env: str, default: int) -> int:
        raw = os.environ.get(env)
        if not raw:
            return default
        value = int(raw)  # ValueError on garbage -> loud failure, not silent drift
        if value < 0:
            raise ValueError(f"{env} must be non-negative, got {value}")
        return value

    return RiskLimits(
        max_single_position_pence=_int("SURVIVOR_PAPER_MAX_POSITION_PENCE", 100),
        max_total_exposure_pence=_int("SURVIVOR_PAPER_MAX_EXPOSURE_PENCE", 500),
        daily_loss_limit_pence=_int("SURVIVOR_PAPER_DAILY_LOSS_PENCE", 100),
        max_drawdown_bps=_int("SURVIVOR_PAPER_MAX_DRAWDOWN_BPS", 1500),
        min_conservative_net_edge_bps=_int("SURVIVOR_PAPER_MIN_EDGE_BPS", 500),
    )
