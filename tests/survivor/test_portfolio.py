"""Portfolio accounting invariants and conservative mark-to-market."""

import pytest

from tradingagents.survivor.execution.portfolio import Portfolio, Position


@pytest.mark.unit
def test_equity_is_cash_plus_marked_positions():
    p = Portfolio(cash_pence=1900, starting_equity_pence=2000, high_water_mark_pence=2000)
    p.positions["AAPL"] = Position(symbol="AAPL", quantity=1, average_entry_price_pence=100,
                                   cost_basis_pence=100, current_mark_pence=99)
    assert p.exposure_pence == 99
    assert p.equity_pence == 1900 + 99


@pytest.mark.unit
def test_conservative_long_mark_uses_bid():
    p = Portfolio(cash_pence=1900, high_water_mark_pence=2000)
    p.buy("AAPL", 1, 100, 0)
    p.mark({"AAPL": 95})  # explicit bid mark
    assert p.positions["AAPL"].current_mark_pence == 95
    assert p.unrealized_pnl_pence == -5  # negative unrealized NOT clamped


@pytest.mark.unit
def test_unmarked_position_valued_at_cost_basis_and_flagged_stale():
    p = Portfolio(cash_pence=1900, high_water_mark_pence=2000)
    p.buy("AAPL", 1, 100, 0)
    assert p.positions["AAPL"].is_stale_mark
    assert p.exposure_pence == 100  # cost basis, no invented mark
    assert p.has_stale_marks


@pytest.mark.unit
def test_high_water_mark_never_resets_downward():
    p = Portfolio(cash_pence=2000, starting_equity_pence=2000, high_water_mark_pence=2000)
    p.buy("AAPL", 2, 100, 0)
    p.mark({"AAPL": 150})  # equity 2000 -> 2100? cash 1800 + 300 = 2100
    assert p.high_water_mark_pence == 2100
    p.mark({"AAPL": 50})   # crash
    assert p.high_water_mark_pence == 2100  # HWM stays
    assert p.drawdown_bps == (2100 - 1900) * 10000 // 2100


@pytest.mark.unit
def test_buy_rejects_bad_inputs():
    p = Portfolio(cash_pence=2000, high_water_mark_pence=2000)
    with pytest.raises(ValueError):
        p.buy("AAPL", 0, 100, 0)   # zero quantity
    with pytest.raises(ValueError):
        p.buy("AAPL", 1, -5, 0)    # negative price
    with pytest.raises(ValueError):
        p.buy("AAPL", 1, 100, -1)  # negative fee


@pytest.mark.unit
def test_average_entry_price_and_cost_basis():
    p = Portfolio(cash_pence=2000, high_water_mark_pence=2000)
    p.buy("AAPL", 1, 100, 0)
    p.buy("AAPL", 1, 60, 0)
    pos = p.positions["AAPL"]
    assert pos.quantity == 2
    assert pos.cost_basis_pence == 160
    assert pos.average_entry_price_pence == 80
