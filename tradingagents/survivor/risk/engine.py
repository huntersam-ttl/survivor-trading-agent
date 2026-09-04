"""Deterministic RiskEngine.

Uses ONLY: the validated proposal, portfolio state, treasury, deterministic
limits, and market/execution inputs. Contains no LLM calls, no network, and
no provider SDK imports. Fails closed at every step.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable
from typing import Any

from tradingagents.survivor.execution.portfolio import Portfolio
from tradingagents.survivor.execution.pricing import MarketSnapshot
from tradingagents.survivor.risk.limits import RiskLimits
from tradingagents.survivor.risk.result import ReasonCode, RiskDecision, RiskStatus
from tradingagents.survivor.trading.types import Action, Side
from tradingagents.survivor.trading.validator import ProposalValidationError, validate_proposal

PAPER_ONLY_MODE = "PAPER_ONLY"


class RiskEngine:
    """Order: mode -> schema -> side/action -> market data -> freshness -> price
    -> leverage/borrow/short -> cash -> single position -> total exposure
    -> daily loss -> drawdown -> conservative edge -> duplicate -> approve."""

    def __init__(
        self,
        limits: RiskLimits,
        duplicate_checker: Callable[[str], bool] | None = None,
        paper_mode: bool = True,
    ):
        self.limits = limits
        self.duplicate_checker = duplicate_checker
        self.paper_mode = paper_mode

    def _decision(self, status: RiskStatus, code: ReasonCode, reason: str, proposal: Any) -> RiskDecision:
        return RiskDecision(
            status=status,
            reason_code=code,
            reason=reason,
            proposal_id=getattr(proposal, "proposal_id", "") or "",
        )

    def evaluate(
        self,
        proposal: Any,
        portfolio: Portfolio,
        market_snapshot: MarketSnapshot | None,
        timestamp: _dt.datetime | None = None,
    ) -> RiskDecision:
        now = timestamp or _dt.datetime.now(_dt.timezone.utc)

        # 1. PAPER_ONLY mode check (structural, never configurable to live)
        if not self.paper_mode or PAPER_ONLY_MODE != "PAPER_ONLY":
            return self._decision(RiskStatus.HALTED, ReasonCode.NOT_PAPER_MODE, "mode is not PAPER_ONLY", proposal)

        # 2. Proposal schema valid (strict, LLM-independent)
        try:
            validate_proposal(proposal, self.limits, now=now)
        except ProposalValidationError as exc:
            return self._decision(RiskStatus.REJECTED, ReasonCode.INVALID_PROPOSAL, str(exc), proposal)

        # 3. Duplicate / replay protection (same proposal must never execute twice)
        if self.duplicate_checker is not None and self.duplicate_checker(proposal.proposal_id):
            return self._decision(
                RiskStatus.REJECTED, ReasonCode.DUPLICATE_PROPOSAL,
                f"proposal {proposal.proposal_id} already executed", proposal,
            )

        # 4. Supported side / actionable action (no shorting by construction)
        if proposal.side != Side.BUY.value:
            return self._decision(RiskStatus.REJECTED, ReasonCode.NO_SHORTING, f"side {proposal.side} unsupported", proposal)
        if proposal.action != Action.OPEN.value:
            return self._decision(
                RiskStatus.REJECTED, ReasonCode.UNSUPPORTED_ACTION,
                f"action {proposal.action} is not executable", proposal,
            )

        # 4./5. Market data available and fresh
        if market_snapshot is None or market_snapshot.symbol != proposal.symbol:
            return self._decision(RiskStatus.REJECTED, ReasonCode.MARKET_DATA_MISSING, "no matching market snapshot", proposal)
        if not market_snapshot.is_fresh(now=now, max_age_sec=self.limits.max_market_data_age_sec):
            return self._decision(RiskStatus.REJECTED, ReasonCode.MARKET_DATA_STALE, "market data is stale", proposal)

        # 6. Price valid
        if not market_snapshot.has_valid_prices():
            return self._decision(RiskStatus.REJECTED, ReasonCode.PRICE_INVALID, "invalid/zero/negative price", proposal)

        ask = market_snapshot.ask_pence
        assert ask is not None  # guarded by has_valid_prices

        # 7.-9. No leverage / borrowing / shorting: long-only, fully cash-funded
        total_cost = proposal.notional_pence + proposal.estimated_fees_pence + proposal.estimated_slippage_pence

        # 10. Sufficient fictional cash (no borrowing, no negative cash)
        if total_cost > portfolio.cash_pence:
            return self._decision(
                RiskStatus.REJECTED, ReasonCode.INSUFFICIENT_CASH,
                f"need {total_cost}p, cash {portfolio.cash_pence}p", proposal,
            )

        # 11. Max single position (exact boundary allowed)
        existing = portfolio.positions.get(proposal.symbol)
        existing_value = existing.market_value_pence if existing else 0
        projected_position = existing_value + proposal.notional_pence
        if projected_position > self.limits.max_single_position_pence:
            return self._decision(
                RiskStatus.REJECTED, ReasonCode.MAX_SINGLE_POSITION,
                f"projected position {projected_position}p > {self.limits.max_single_position_pence}p", proposal,
            )

        # 12. Max total exposure (exact boundary allowed)
        projected_exposure = portfolio.exposure_pence + proposal.notional_pence
        if projected_exposure > self.limits.max_total_exposure_pence:
            return self._decision(
                RiskStatus.REJECTED, ReasonCode.MAX_TOTAL_EXPOSURE,
                f"projected exposure {projected_exposure}p > {self.limits.max_total_exposure_pence}p", proposal,
            )

        # 13. Daily loss limit (realized + unrealized, unrealized NOT clamped)
        if portfolio.daily_pnl_pence <= -self.limits.daily_loss_limit_pence:
            return self._decision(
                RiskStatus.HALTED, ReasonCode.DAILY_LOSS_HALT,
                f"daily P/L {portfolio.daily_pnl_pence}p reached -{self.limits.daily_loss_limit_pence}p", proposal,
            )

        # 14. Drawdown halt (HWM never reset downward)
        if portfolio.drawdown_bps >= self.limits.max_drawdown_bps:
            return self._decision(
                RiskStatus.HALTED, ReasonCode.DRAWDOWN_HALT,
                f"drawdown {portfolio.drawdown_bps}bps >= {self.limits.max_drawdown_bps}bps", proposal,
            )

        # 15. Conservative net edge (exact boundary allowed)
        if proposal.conservative_net_edge_bps < self.limits.min_conservative_net_edge_bps:
            return self._decision(
                RiskStatus.REJECTED, ReasonCode.EDGE_TOO_LOW,
                f"net edge {proposal.conservative_net_edge_bps}bps < {self.limits.min_conservative_net_edge_bps}bps",
                proposal,
            )

        # 16. Duplicate / replay protection (already checked early, step 3)

        # 17. Final approval
        return self._decision(
            RiskStatus.APPROVED, ReasonCode.APPROVED,
            f"approved: cost {total_cost}p, exposure {projected_exposure}p, edge {proposal.conservative_net_edge_bps}bps",
            proposal,
        )

