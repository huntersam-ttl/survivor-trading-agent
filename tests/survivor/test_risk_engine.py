"""Deterministic RiskEngine tests. Proves the engine contains no LLM/network code."""

import inspect
import tempfile

import pytest

from tests.survivor.paper_helpers import (
    default_inputs,
    fresh_snapshot,
    make_builder,
    make_proposal,
    stale_snapshot,
)
from tradingagents.survivor.execution.portfolio import Portfolio
from tradingagents.survivor.risk.engine import RiskEngine
from tradingagents.survivor.risk.limits import RiskLimits
from tradingagents.survivor.risk.result import RiskStatus


@pytest.fixture
def engine():
    return RiskEngine(RiskLimits())


@pytest.fixture
def funded_portfolio():
    return Portfolio(cash_pence=2000, starting_equity_pence=2000, high_water_mark_pence=2000)


@pytest.mark.unit
def test_valid_one_pound_proposal_approved(engine, funded_portfolio):
    decision = engine.evaluate(make_proposal(), funded_portfolio, fresh_snapshot())
    assert decision.status == RiskStatus.APPROVED
    assert decision.reason_code.value == "APPROVED"


@pytest.mark.unit
def test_one_pound_one_penny_position_rejected(engine, funded_portfolio):
    p = make_proposal(quantity=1, reference_price_pence=101, notional_pence=101)
    decision = engine.evaluate(p, funded_portfolio, fresh_snapshot())
    # rejected fail-closed: by validator schema check and/or risk limit check
    assert decision.rejected
    assert decision.reason_code.value in ("MAX_SINGLE_POSITION", "INVALID_PROPOSAL")


@pytest.mark.unit
def test_edge_exactly_five_percent_allowed_and_below_rejected(engine, funded_portfolio):
    p500 = make_proposal(gross_edge_bps=750, conservative_net_edge_bps=500)
    assert engine.evaluate(p500, funded_portfolio, fresh_snapshot()).approved
    p499 = make_proposal(gross_edge_bps=749, conservative_net_edge_bps=499)
    d = engine.evaluate(p499, funded_portfolio, fresh_snapshot())
    assert d.rejected and d.reason_code.value == "EDGE_TOO_LOW"


@pytest.mark.unit
def test_stale_and_missing_market_data_rejected(engine, funded_portfolio):
    d = engine.evaluate(make_proposal(), funded_portfolio, stale_snapshot())
    assert d.rejected and d.reason_code.value == "MARKET_DATA_STALE"
    d = engine.evaluate(make_proposal(), funded_portfolio, None)
    assert d.rejected and d.reason_code.value == "MARKET_DATA_MISSING"


@pytest.mark.unit
def test_insufficient_cash_rejected(engine):
    poor = Portfolio(cash_pence=50, high_water_mark_pence=2000)
    d = engine.evaluate(make_proposal(), poor, fresh_snapshot())
    assert d.rejected and d.reason_code.value == "INSUFFICIENT_CASH"


@pytest.mark.unit
def test_shorting_never_supported(engine, funded_portfolio):
    d = engine.evaluate(make_proposal(side="SELL"), funded_portfolio, fresh_snapshot())
    assert d.rejected
    assert d.reason_code.value in ("NO_SHORTING", "INVALID_PROPOSAL")


@pytest.mark.unit
def test_duplicate_proposal_rejected(engine, funded_portfolio):
    seen = {"abc"}
    eng = RiskEngine(RiskLimits(), duplicate_checker=lambda pid: pid in seen)
    d = eng.evaluate(make_proposal(proposal_id="abc"), funded_portfolio, fresh_snapshot())
    assert d.rejected and d.reason_code.value == "DUPLICATE_PROPOSAL"


@pytest.mark.unit
def test_daily_loss_and_drawdown_halt(engine, funded_portfolio):
    funded_portfolio.realized_pnl_today_pence = -100
    d = engine.evaluate(make_proposal(), funded_portfolio, fresh_snapshot())
    assert d.halted and d.reason_code.value == "DAILY_LOSS_HALT"

    p2 = Portfolio(cash_pence=1000, high_water_mark_pence=2000)
    from tradingagents.survivor.execution.portfolio import Position

    # daily loss check precedes drawdown; cancel the unrealized loss with realized
    # profit so only the drawdown condition fires
    p2.realized_pnl_today_pence = 165

    p2.positions["X"] = Position(symbol="X", quantity=5, average_entry_price_pence=100,
                                 cost_basis_pence=500, current_mark_pence=67)
    # equity 1335 vs HWM 2000 -> drawdown 3325bps >= 1500
    d2 = engine.evaluate(make_proposal(), p2, fresh_snapshot())
    assert d2.halted and d2.reason_code.value == "DRAWDOWN_HALT"


@pytest.mark.unit
def test_builder_to_engine_path_uses_only_deterministic_inputs():
    with tempfile.TemporaryDirectory():
        limits = RiskLimits()
        p = make_builder().build(
            run_id="r", timestamp_utc=make_proposal().timestamp_utc, market="stock",
            symbol="AAPL", decision_text="BUY", reference_ask_pence=100,
            inputs=default_inputs(),
        )
        assert RiskEngine(limits).evaluate(
            p, Portfolio(cash_pence=2000, high_water_mark_pence=2000), fresh_snapshot()
        ).approved


@pytest.mark.unit
def test_risk_engine_contains_no_llm_or_network_code():
    """Structural proof: the risk engine imports and calls no LLM/network code."""
    import ast

    import tradingagents.survivor.risk.engine as engine_module

    source = inspect.getsource(engine_module)
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    forbidden = ("llm", "openai", "anthropic", "requests", "urllib", "socket", "http", "boto3")
    for module in imported:
        for token in forbidden:
            assert token not in module.lower(), f"forbidden import in RiskEngine: {module}"
    assert not any(
        isinstance(n, ast.Attribute) and n.attr == "invoke" for n in ast.walk(tree)
    ), "RiskEngine must not call invoke() (LLM surface)"
