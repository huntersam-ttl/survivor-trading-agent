"""Deterministic prediction-market resolution settlement (paper only).

A long YES position (Phase 2 supports long-only):
  resolved YES -> pays 100p per share (face value of a binary share)
  resolved NO  -> pays 0

No real settlement occurs; events are recorded in the hash-chained paper ledger.
"""

from __future__ import annotations

from tradingagents.survivor.execution.paper_broker import PaperBroker

YES_SHARE_PAYOFF_PENCE = 100  # a binary share pays its face value (100p) on YES

RESOLVED_EVENT = "POSITION_RESOLVED"
SETTLED_EVENT = "SETTLEMENT_APPLIED"


def settle_prediction_position(
    broker: PaperBroker,
    symbol: str,
    resolved_yes: bool,
    run_id: str = "paper_resolution",
) -> dict | None:
    """Settle one long-YES paper position deterministically.

    Returns the SETTLEMENT_APPLIED event (with realized_pnl_pence and
    cost_basis_pence added), or None when there is nothing to settle.
    """
    portfolio = broker.portfolio
    position = portfolio.positions.get(symbol)
    if position is None or position.quantity <= 0:
        return None

    quantity = position.quantity
    payoff_pence = quantity * (YES_SHARE_PAYOFF_PENCE if resolved_yes else 0)
    proceeds = min(payoff_pence, quantity * YES_SHARE_PAYOFF_PENCE)  # identical; explicit cap

    equity_before = portfolio.equity_pence
    cash_before = portfolio.cash_pence

    # Remove the position; credit payoff to cash; realized P/L = payoff - cost basis.
    del portfolio.positions[symbol]
    portfolio.cash_pence += proceeds
    realized = proceeds - position.cost_basis_pence
    portfolio.realized_pnl_pence += realized
    if portfolio.equity_pence > portfolio.high_water_mark_pence:
        portfolio.high_water_mark_pence = portfolio.equity_pence

    broker.ledger.append_event(
        event_type=RESOLVED_EVENT,
        run_id=run_id,
        symbol=symbol,
        side="BUY",
        quantity=quantity,
        execution_price_pence=position.average_entry_price_pence,
        notional_pence=position.cost_basis_pence,
        cash_before=cash_before,
        cash_after=portfolio.cash_pence,
        exposure_before=position.market_value_pence,
        exposure_after=portfolio.exposure_pence,
        equity_before=equity_before,
        equity_after=portfolio.equity_pence,
        risk_decision="SETTLE",
        risk_reason=f"resolved {'YES' if resolved_yes else 'NO'}",
    )
    event = broker.ledger.append_event(
        event_type=SETTLED_EVENT,
        run_id=run_id,
        symbol=symbol,
        side="BUY",
        quantity=quantity,
        execution_price_pence=YES_SHARE_PAYOFF_PENCE if resolved_yes else 0,
        notional_pence=proceeds,
        cash_before=cash_before,
        cash_after=portfolio.cash_pence,
        exposure_before=position.market_value_pence,
        exposure_after=portfolio.exposure_pence,
        equity_before=equity_before,
        equity_after=portfolio.equity_pence,
        risk_decision="SETTLE",
        risk_reason=f"payoff {proceeds}p, realized {realized}p",
    )
    event["realized_pnl_pence"] = realized
    event["cost_basis_pence"] = position.cost_basis_pence
    event["fees_pence_total"] = position.cost_basis_pence - position.quantity * position.average_entry_price_pence
    return event
