"""Risk decision value types (machine-readable)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RiskStatus(str, Enum):
    APPROVED = "APPROVE"
    REJECTED = "REJECT"
    HALTED = "HALT"


class ReasonCode(str, Enum):
    APPROVED = "APPROVED"
    NOT_PAPER_MODE = "NOT_PAPER_MODE"
    INVALID_PROPOSAL = "INVALID_PROPOSAL"
    UNSUPPORTED_SIDE = "UNSUPPORTED_SIDE"
    UNSUPPORTED_ACTION = "UNSUPPORTED_ACTION"
    MARKET_DATA_MISSING = "MARKET_DATA_MISSING"
    MARKET_DATA_STALE = "MARKET_DATA_STALE"
    PRICE_INVALID = "PRICE_INVALID"
    NO_SHORTING = "NO_SHORTING"
    NO_LEVERAGE = "NO_LEVERAGE"
    NO_BORROWING = "NO_BORROWING"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
    MAX_SINGLE_POSITION = "MAX_SINGLE_POSITION"
    MAX_TOTAL_EXPOSURE = "MAX_TOTAL_EXPOSURE"
    DAILY_LOSS_HALT = "DAILY_LOSS_HALT"
    DRAWDOWN_HALT = "DRAWDOWN_HALT"
    EDGE_TOO_LOW = "EDGE_TOO_LOW"
    DUPLICATE_PROPOSAL = "DUPLICATE_PROPOSAL"
    ALREADY_HALTED = "ALREADY_HALTED"


@dataclass(frozen=True)
class RiskDecision:
    status: RiskStatus
    reason_code: ReasonCode
    reason: str
    proposal_id: str = ""

    @property
    def approved(self) -> bool:
        return self.status == RiskStatus.APPROVED

    @property
    def halted(self) -> bool:
        return self.status == RiskStatus.HALTED

    @property
    def rejected(self) -> bool:
        return self.status == RiskStatus.REJECTED
