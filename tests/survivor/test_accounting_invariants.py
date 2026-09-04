"""Portfolio-level accounting invariants and exact-boundary risk limits."""

import pytest

from tests.survivor.paper_helpers import fresh_snapshot, make_proposal
from tradingagents.survivor.execution.portfolio import Portfolio, Position
from tradingagents.survivor.risk.engine import RiskEngine
from tradingagents.survivor.risk.limits import RiskLimits
from tradingagents.survivor.risk.result import RiskStatus

LIMITS = RiskLimits()


def _engine():
    return RiskEngine(LIMITS)


def _funded():
    return Portfolio(cash_pence=2000, starting_equity_pence=2000, high_water_mark_pence=2000)


def _position(symbol, quantity, entry, mark):
    return Position(symbol=symbol, quantity=quantity, average_entry_price_pence=entry,
                    cost_basis_pence=quantity * entry, current_mark_pence=mark)


@pytest.mark.unit
def test_exposure_exactly_five_pounds_allowed():
    p = _funded()
    # five positions of £1.00 each = £5.00 exposure (exact boundary)
    for i in range(5):
        symbol = f"S{i}"
        p.positions[symbol] = _position(symbol, 1, 100, 100)
    assert p.exposure_pence == 500
    # engine has no cash left for more but check exposure boundary directly:
    proposal = make_proposal(
        symbol="S5", quantity=1, reference_price_pence=100, notional_pence=100,
        estimated_fees_pence=0, estimated_slippage_pence=0,
    )
    decision = _engine().evaluate(proposal, p, fresh_snapshot(symbol="S5", bid=99, ask=100))
    assert decision.rejected and decision.reason_code.value == "MAX_TOTAL_EXPOSURE"


@pytest.mark.unit
def test_exposure_four_pounds_ninety_nine_allows_one_more_penny_only_to_boundary():
    p = _funded()
    for i in range(4):
        symbol = f"S{i}"
        p.positions[symbol] = _position(symbol, 1, 100, 100)
    p.positions["S4"] = _position("S4", 1, 99, 99)  # 499p total
    # adding exactly 1p reaches the exact £5.00 boundary -> allowed
    proposal = make_proposal(
        symbol="S5", quantity=1, reference_price_pence=1, notional_pence=1,
        gross_edge_bps=10000, conservative_net_edge_bps=9750,
        estimated_fees_pence=0, estimated_slippage_pence=0,
        expected_probability_bps=10000, market_probability_bps=100,
    )
    d = _engine().evaluate(proposal, p, fresh_snapshot(symbol="S5", bid=1, ask=1))
    assert d.approved
    # adding 2p would cross to £5.01 -> rejected
    over = make_proposal(
        symbol="S6", quantity=2, reference_price_pence=1, notional_pence=2,
        gross_edge_bps=10000, conservative_net_edge_bps=9750,
        estimated_fees_pence=0, estimated_slippage_pence=0,
        expected_probability_bps=10000, market_probability_bps=100,
    )
    d2 = _engine().evaluate(over, p, fresh_snapshot(symbol="S6", bid=1, ask=1))
    assert d2.rejected and d2.reason_code.value == "MAX_TOTAL_EXPOSURE"


@pytest.mark.unit
def test_daily_loss_exactly_one_pound_halts_new_trades():
    p = _funded()
    p.realized_pnl_today_pence = 0
    # realized -40p + unrealized -60p = exactly -£1.00 (unrealized not clamped)
    p.realized_pnl_today_pence = -40
    p.positions["HALT"] = _position("HALT", 1, 100, 40)
    # entry 100, mark 40 -> unrealized -60; proposal targets a different symbol
    d = _engine().evaluate(make_proposal(), p, fresh_snapshot())
    assert d.halted and d.reason_code.value == "DAILY_LOSS_HALT"


@pytest.mark.unit
def test_drawdown_exactly_fifteen_percent_halts_new_trades():
    p = Portfolio(cash_pence=850, high_water_mark_pence=2000)
    p.positions["AAPL"] = _position("AAPL", 1, 100, 100)
    # equity 950: drawdown = (2000-950)/2000 = 5250bps? no: 850+100=950 -> 52.5%.
    p2 = Portfolio(cash_pence=1600, high_water_mark_pence=2000)
    p2.positions["DDOWN"] = _position("DDOWN", 1, 100, 100)
    # equity 1700 -> drawdown 1500bps exactly (15%)
    assert p2.drawdown_bps == 1500
    d = _engine().evaluate(make_proposal(), p2, fresh_snapshot())
    assert d.halted and d.reason_code.value == "DRAWDOWN_HALT"
    # just under: equity 1701 -> 1495bps -> no halt
    p3 = Portfolio(cash_pence=1601, high_water_mark_pence=2000)
    p3.positions["DDOWN"] = _position("DDOWN", 1, 100, 100)
    assert _engine().evaluate(make_proposal(), p3, fresh_snapshot()).status == RiskStatus.APPROVED


@pytest.mark.unit
def test_cash_never_negative_and_exposure_never_negative():
    p = _funded()
    p.buy("AAPL", 20, 100, 0)  # exactly drains cash: 2000 - 2000 = 0
    assert p.cash_pence == 0
    assert p.exposure_pence == 2000
    with pytest.raises(ValueError):
        p.buy("MSFT", 1, 100, 0)  # would force negative cash


@pytest.mark.unit
def test_equity_identity_cash_plus_positions():
    p = Portfolio(cash_pence=1234, high_water_mark_pence=2000)
    p.buy("AAPL", 3, 77, 5)
    assert p.cash_pence + p.exposure_pence == p.equity_pence
