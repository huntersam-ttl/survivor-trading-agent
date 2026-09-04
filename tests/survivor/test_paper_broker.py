"""PaperBroker: fictional treasury, defensive limits, duplicate protection, recovery."""

import pytest

from tests.survivor.paper_helpers import fresh_snapshot, make_proposal, tmp_ledger
from tradingagents.survivor.execution.paper_broker import (
    DuplicateProposalError,
    ExecutionBlockedError,
    PaperBroker,
)
from tradingagents.survivor.risk.limits import RiskLimits
from tradingagents.survivor.risk.result import ReasonCode, RiskDecision, RiskStatus


def _approved(proposal):
    return RiskDecision(
        status=RiskStatus.APPROVED, reason_code=ReasonCode.APPROVED, reason="test",
        proposal_id=proposal.proposal_id,
    )


@pytest.fixture
def broker(tmp_path):
    b = PaperBroker(limits=RiskLimits(), ledger=tmp_ledger(tmp_path))
    b.ensure_initialized()
    return b


@pytest.mark.unit
def test_initial_treasury_is_twenty_pounds(broker):
    assert broker.portfolio.cash_pence == 2000
    assert broker.portfolio.high_water_mark_pence == 2000
    # initialization is ledger-evidenced and happens exactly once
    types = [e["event_type"] for e in broker.ledger.events()]
    assert types.count("TREASURY_INITIALIZED") == 1
    broker.ensure_initialized()
    assert [e["event_type"] for e in broker.ledger.events()].count("TREASURY_INITIALIZED") == 1


@pytest.mark.unit
def test_valid_one_pound_proposal_executes_with_explicit_costs(broker):
    proposal = make_proposal()
    event = broker.execute(proposal, fresh_snapshot(), _approved(proposal))
    # deterministic fill: ask 100p + 50bps slippage = 101p
    assert event["execution_price_pence"] == 101
    assert event["fee_pence"] == 1  # 50bps of 101p rounded up
    assert event["notional_pence"] == 101
    # cash decreased by notional + fee, never negative
    assert broker.portfolio.cash_pence == 2000 - 101 - 1
    assert broker.portfolio.cash_pence >= 0
    assert broker.portfolio.positions["AAPL"].quantity == 1


@pytest.mark.unit
def test_execute_requires_prior_risk_approval(broker):
    with pytest.raises(ExecutionBlockedError):
        broker.execute(make_proposal(), fresh_snapshot(), None)
    rejected = RiskDecision(status=RiskStatus.REJECTED, reason_code=ReasonCode.EDGE_TOO_LOW, reason="x")
    with pytest.raises(ExecutionBlockedError):
        broker.execute(make_proposal(), fresh_snapshot(), rejected)


@pytest.mark.unit
def test_same_proposal_cannot_execute_twice(broker):
    proposal = make_proposal()
    broker.execute(proposal, fresh_snapshot(), _approved(proposal))
    # identical resubmission: same stable fields -> same deterministic proposal_id
    replay = make_proposal(timestamp_utc=proposal.timestamp_utc)
    assert replay.proposal_id == proposal.proposal_id
    with pytest.raises(DuplicateProposalError):
        broker.execute(replay, fresh_snapshot(), _approved(replay))
    execs = [e for e in broker.ledger.events() if e["event_type"] == "TRADE_EXECUTED"]
    assert len(execs) == 1


@pytest.mark.unit
def test_negative_cash_is_impossible(broker):
    with pytest.raises(ValueError):
        broker.portfolio.buy("AAPL", 100, 100, 0)  # needs 10000p, has 2000p
    assert broker.portfolio.cash_pence == 2000


@pytest.mark.unit
def test_broker_defensively_enforces_position_limit(broker):
    proposal = make_proposal(quantity=5, reference_price_pence=100, notional_pence=500)
    with pytest.raises(ExecutionBlockedError):
        broker.execute(proposal, fresh_snapshot(), _approved(proposal))


@pytest.mark.unit
def test_conservative_mark_uses_bid_not_midpoint(broker):
    proposal = make_proposal()
    broker.execute(proposal, fresh_snapshot(), _approved(proposal))
    # bid 99, ask 101 -> mark must be bid (99), not midpoint (100)
    broker.mark_to_market([fresh_snapshot(bid=99, ask=101)])
    assert broker.portfolio.positions["AAPL"].current_mark_pence == 99
    # unrealized loss counted, not clamped
    assert broker.portfolio.unrealized_pnl_pence == 99 - 101


@pytest.mark.unit
def test_fresh_broker_status_shows_starting_treasury_without_writing(tmp_path):
    ledger = tmp_ledger(tmp_path)
    broker = PaperBroker(limits=RiskLimits(), ledger=ledger)
    status = broker.status()
    assert status["cash_pence"] == 2000
    assert status["equity_pence"] == 2000
    assert status["trading_state"] == "ACTIVE"
    assert ledger.events() == []  # read-only: nothing was written


@pytest.mark.unit
def test_state_survives_process_restart(tmp_path):
    limits = RiskLimits()
    ledger = tmp_ledger(tmp_path)
    broker = PaperBroker(limits=limits, ledger=ledger)
    broker.ensure_initialized()
    proposal = make_proposal()
    broker.execute(proposal, fresh_snapshot(), _approved(proposal))

    # "restart": fresh broker over the same db recovers state via chain replay
    reborn = PaperBroker(limits=limits, ledger=PaperBroker(limits=limits, ledger=tmp_ledger(tmp_path)).ledger)
    assert reborn.portfolio.cash_pence == 2000 - 101 - 1
    assert reborn.portfolio.positions["AAPL"].quantity == 1
    assert reborn.ledger.has_executed(proposal.proposal_id)
