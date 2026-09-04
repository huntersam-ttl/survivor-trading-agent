"""TradeProposal schema, ProposalBuilder determinism, and strict validation."""

import pytest

from tests.survivor.paper_helpers import default_inputs, make_builder, make_proposal, now_utc
from tradingagents.survivor.risk.limits import RiskLimits
from tradingagents.survivor.trading.proposal import parse_decision_direction
from tradingagents.survivor.trading.types import derive_proposal_id
from tradingagents.survivor.trading.validator import (
    ProposalValidationError,
    proposal_from_dict,
    validate_proposal,
)


@pytest.mark.unit
def test_builder_produces_deterministic_open_proposal():
    p = make_builder().build(
        run_id="r1", timestamp_utc="2026-09-04T12:00:00+00:00", market="stock",
        symbol="AAPL", decision_text="BUY - strong thesis",
        reference_ask_pence=100, inputs=default_inputs(),
    )
    assert p.action == "OPEN" and p.side == "BUY"
    assert p.quantity == 1 and p.notional_pence == 100
    assert p.gross_edge_bps == 800 and p.conservative_net_edge_bps == 550
    # deterministic identity
    p2 = make_builder().build(
        run_id="r1", timestamp_utc="2026-09-04T12:00:00+00:00", market="stock",
        symbol="AAPL", decision_text="BUY - strong thesis",
        reference_ask_pence=100, inputs=default_inputs(),
    )
    assert p.proposal_id == p2.proposal_id == derive_proposal_id(p)


@pytest.mark.unit
def test_builder_never_invents_missing_inputs():
    """Missing direction/price/edge components -> NO_TRADE with zeroed execution fields."""
    b = make_builder()
    for kwargs in (
        {'decision_text': "SELL everything", 'reference_ask_pence': 100, 'inputs': default_inputs()},
        {'decision_text': "BUY", 'reference_ask_pence': None, 'inputs': default_inputs()},
        {'decision_text': "BUY", 'reference_ask_pence': 100, 'inputs': None},
    ):
        p = b.build(run_id="r", timestamp_utc="2026-09-04T12:00:00+00:00",
                    market="stock", symbol="AAPL", **kwargs)
        assert p.action == "NO_TRADE"
        assert p.quantity == 0 and p.notional_pence == 0
        assert p.reference_price_pence == 0 and p.conservative_net_edge_bps == 0


@pytest.mark.unit
def test_parse_direction_and_unknown():
    assert parse_decision_direction("Final: BUY with conviction") == "BUY"
    assert parse_decision_direction("we should sell") == "SELL"
    assert parse_decision_direction("nothing actionable here") == "UNKNOWN"


@pytest.mark.unit
def test_validator_rejects_malformed_proposals():
    base = make_proposal()
    validate_proposal(base, RiskLimits())

    def bad(**over):
        p = make_proposal(**over)
        with pytest.raises(ProposalValidationError):
                validate_proposal(p, RiskLimits())

    bad(run_id="")
    bad(symbol="BAD SYMBOL!")
    bad(side="SELL")
    bad(action="CLOSE")
    bad(reference_price_pence=0)
    bad(notional_pence=-100)
    bad(reference_price_pence=-5)
    bad(expected_probability_bps=20000)  # impossible probability
    bad(conservative_net_edge_bps=99999)  # out of bounds (also mismatches components)
    # NaN / infinity cannot hide in int fields
    bad(quantity=float("nan"))
    bad(notional_pence=float("inf"))


@pytest.mark.unit
def test_validator_rejects_stale_and_future_proposals():
    from datetime import timedelta

    limits = RiskLimits()
    old = (now_utc() - timedelta(seconds=3600)).isoformat()
    with pytest.raises(ProposalValidationError):
        validate_proposal(make_proposal(timestamp_utc=old), limits)
    future = (now_utc() + timedelta(hours=2)).isoformat()
    with pytest.raises(ProposalValidationError):
        validate_proposal(make_proposal(timestamp_utc=future), limits)


@pytest.mark.unit
def test_validator_rejects_extra_fields_and_tampered_edge():
    data = make_proposal().__dict__.copy()
    data["sneaky_llm_field"] = 1
    with pytest.raises(ProposalValidationError):
        proposal_from_dict(data)
    # a tampered conservative edge that doesn't match its components is rejected
    tampered = make_proposal(conservative_net_edge_bps=5000)
    with pytest.raises(ProposalValidationError):
        validate_proposal(tampered, RiskLimits())
