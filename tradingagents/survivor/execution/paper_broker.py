"""PaperBroker: fictional-funds-only execution.

- Begins with £20.00 (2000 pence) of fictional equity.
- Never allows negative cash, negative holdings, or exposure over limits.
- Defensive enforcement of every risk limit (defense in depth with RiskEngine).
- Deterministic BUY fill: ask + explicit slippage; fees explicit.
- No HTTP, no exchange auth, no API keys, no wallets, no signing, no routing.
- Every state change is recorded as an append-only, hash-chained ledger event.
"""

from __future__ import annotations

from typing import Any

from tradingagents.survivor.execution.ledger import PaperLedger
from tradingagents.survivor.execution.portfolio import Portfolio
from tradingagents.survivor.execution.pricing import (
    MarketSnapshot,
    buy_execution_price_pence,
    conservative_long_mark_pence,
)
from tradingagents.survivor.risk.limits import RiskLimits
from tradingagents.survivor.risk.result import RiskDecision, RiskStatus
from tradingagents.survivor.trading.types import Action, Side, ceil_bps_cost
from tradingagents.survivor.trading.validator import ProposalValidationError, validate_proposal


class DuplicateProposalError(Exception):
    """The same proposal was already executed (replay attempt)."""


class ExecutionBlockedError(Exception):
    """Defensive rejection inside the broker (invariant would be violated)."""


class PaperBroker:
    def __init__(
        self,
        limits: RiskLimits | None = None,
        ledger: PaperLedger | None = None,
    ):
        self.limits = limits or RiskLimits()
        self.ledger = ledger or PaperLedger()
        self.portfolio: Portfolio | None = None
        self.recover()

    # --- lifecycle --------------------------------------------------------------
    def recover(self) -> Portfolio:
        """Verify ledger chain and rebuild state; fail closed on corruption."""
        events = self.ledger.events()
        if not events:
            self.portfolio = Portfolio()
            return self.portfolio
        self.portfolio = self.ledger.recover_portfolio()
        return self.portfolio

    def ensure_initialized(self, run_id: str = "paper_init") -> None:
        """Seed the £20 fictional treasury exactly once (ledger-evidenced)."""
        if self.ledger.events():
            return
        start = self.limits.initial_equity_pence
        self.portfolio = Portfolio(
            cash_pence=start,
            starting_equity_pence=start,
            high_water_mark_pence=start,
        )
        self.ledger.append_event(
            event_type="TREASURY_INITIALIZED",
            run_id=run_id,
            symbol="GBP",
            side=None,
            cash_before=0,
            cash_after=start,
            equity_before=0,
            equity_after=start,
            risk_decision="APPROVE",
            risk_reason="initial fictional paper treasury",
        )

    # --- execution --------------------------------------------------------------
    def execute(self, proposal: Any, snapshot: MarketSnapshot, risk_decision: RiskDecision) -> dict:
        """Execute an APPROVED proposal. Defensive checks; ledger-evidenced."""
        if not isinstance(risk_decision, RiskDecision) or risk_decision.status != RiskStatus.APPROVED:
            raise ExecutionBlockedError(
                "PaperBroker.execute requires a prior APPROVE RiskDecision "
                f"(got {getattr(risk_decision, 'status', None)})"
            )
        if self.ledger.has_executed(proposal.proposal_id):
            raise DuplicateProposalError(
                f"proposal {proposal.proposal_id} was already executed (DUPLICATE_PROPOSAL)"
            )
        # Defense in depth: re-validate and re-check limits before any mutation.
        try:
            validate_proposal(proposal, self.limits)
        except ProposalValidationError as exc:
            raise ExecutionBlockedError(f"invalid proposal: {exc}") from exc
        if proposal.side != Side.BUY.value or proposal.action != Action.OPEN.value:
            raise ExecutionBlockedError(f"unsupported execution: {proposal.side}/{proposal.action}")
        if snapshot.symbol != proposal.symbol or not snapshot.has_valid_prices():
            raise ExecutionBlockedError("market snapshot missing/invalid for execution")

        portfolio = self.portfolio if self.portfolio is not None else self.recover()
        ask = snapshot.ask_pence
        assert ask is not None
        execution_price = buy_execution_price_pence(ask, proposal.slippage_bps)
        quantity = proposal.quantity
        notional = quantity * execution_price
        fee_pence = ceil_bps_cost(notional, proposal.fee_bps)
        total_cost = notional + fee_pence
        if total_cost > portfolio.cash_pence:
            raise ExecutionBlockedError(
                f"insufficient cash: need {total_cost}p, have {portfolio.cash_pence}p"
            )
        projected_position_value = (
            portfolio.positions.get(proposal.symbol).market_value_pence
            if proposal.symbol in portfolio.positions
            else 0
        ) + proposal.notional_pence  # exposure measured at committed reference notional
        if projected_position_value > self.limits.max_single_position_pence:
            raise ExecutionBlockedError(
                f"max single position {self.limits.max_single_position_pence}p exceeded "
                f"({projected_position_value}p)"
            )
        if portfolio.exposure_pence + proposal.notional_pence > self.limits.max_total_exposure_pence:
            raise ExecutionBlockedError(
                f"max total exposure {self.limits.max_total_exposure_pence}p exceeded"
            )

        cash_before = portfolio.cash_pence
        exposure_before = portfolio.exposure_pence
        equity_before = portfolio.equity_pence
        portfolio.buy(proposal.symbol, quantity, execution_price, fee_pence)
        event = self.ledger.append_event(
            event_type="TRADE_EXECUTED",
            run_id=proposal.run_id,
            symbol=proposal.symbol,
            side=proposal.side,
            quantity=quantity,
            execution_price_pence=execution_price,
            notional_pence=notional,
            fee_pence=fee_pence,
            slippage_pence=notional - proposal.notional_pence,
            cash_before=cash_before,
            cash_after=portfolio.cash_pence,
            exposure_before=exposure_before,
            exposure_after=portfolio.exposure_pence,
            equity_before=equity_before,
            equity_after=portfolio.equity_pence,
            risk_decision=risk_decision.status.value,
            risk_reason=risk_decision.reason_code.value,
            proposal_id=proposal.proposal_id,
        )
        return event

    # --- marking & status -------------------------------------------------------
    def mark_to_market(self, snapshots: list[MarketSnapshot]) -> list[dict]:
        """Mark long positions conservatively at bid; append MARK_UPDATED events."""
        portfolio = self.portfolio if self.portfolio is not None else self.recover()
        events = []
        for snapshot in snapshots:
            mark = conservative_long_mark_pence(snapshot)
            position = portfolio.positions.get(snapshot.symbol)
            if position is None or position.quantity <= 0:
                continue
            equity_before = portfolio.equity_pence
            portfolio.mark({snapshot.symbol: mark})
            events.append(
                self.ledger.append_event(
                    event_type="MARK_UPDATED",
                    run_id="paper_mark",
                    symbol=snapshot.symbol,
                    execution_price_pence=mark,
                    cash_before=portfolio.cash_pence,
                    cash_after=portfolio.cash_pence,
                    exposure_before=equity_before - portfolio.cash_pence,
                    exposure_after=portfolio.exposure_pence,
                    equity_before=equity_before,
                    equity_after=portfolio.equity_pence,
                    risk_decision="MARK",
                    risk_reason="conservative bid mark",
                )
            )
        if self.portfolio is not None:
            self.portfolio.realized_pnl_today_pence = self.ledger.pnl_today()
        return events

    def refresh_daily_pnl(self) -> None:
        if self.portfolio is not None:
            self.portfolio.realized_pnl_today_pence = self.ledger.pnl_today()

    def trading_state(self) -> str:
        """ACTIVE / DAILY_HALT / DRAWDOWN_HALT (halt conditions only ever tighten)."""
        portfolio = self.portfolio if self.portfolio is not None else self.recover()
        self.refresh_daily_pnl()
        if portfolio.daily_pnl_pence <= -self.limits.daily_loss_limit_pence:
            return "DAILY_HALT"
        if portfolio.drawdown_bps >= self.limits.max_drawdown_bps:
            return "DRAWDOWN_HALT"
        return "ACTIVE"

    def status(self) -> dict:
        """Read-only status snapshot for CLI/reporting.

        A fresh (never-initialized) broker reports the £20 starting treasury
        without writing to the ledger.
        """
        if not self.ledger.events():
            start = self.limits.initial_equity_pence
            return {
                "mode": "PAPER ONLY",
                "starting_equity_pence": start,
                "cash_pence": start,
                "exposure_pence": 0,
                "equity_pence": start,
                "realized_pnl_pence": 0,
                "unrealized_pnl_pence": 0,
                "high_water_mark_pence": start,
                "drawdown_bps": 0,
                "daily_pnl_pence": 0,
                "open_positions": 0,
                "trading_state": "ACTIVE",
            }
        portfolio = self.portfolio if self.portfolio is not None else self.recover()
        self.refresh_daily_pnl()
        return {
            "mode": "PAPER ONLY",
            "starting_equity_pence": self.limits.initial_equity_pence,
            "cash_pence": portfolio.cash_pence,
            "exposure_pence": portfolio.exposure_pence,
            "equity_pence": portfolio.equity_pence,
            "realized_pnl_pence": portfolio.realized_pnl_pence,
            "unrealized_pnl_pence": portfolio.unrealized_pnl_pence,
            "high_water_mark_pence": portfolio.high_water_mark_pence,
            "drawdown_bps": portfolio.drawdown_bps,
            "daily_pnl_pence": portfolio.daily_pnl_pence,
            "open_positions": len([p for p in portfolio.positions.values() if p.quantity > 0]),
            "trading_state": self.trading_state(),
        }


