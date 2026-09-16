"""Offline regression tests for the typed prediction-market decision path."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from tests.survivor.market_helpers import make_snapshot
from tradingagents.agents.schemas import (
    PortfolioRating,
    PredictionMarketDecision,
    render_pm_decision,
)
from tradingagents.agents.utils.structured import invoke_structured_or_freetext


def _decision() -> PredictionMarketDecision:
    return PredictionMarketDecision(
        rating=PortfolioRating.HOLD,
        executive_summary="Wait for confirming evidence.",
        investment_thesis="The snapshot is balanced and uncertainty remains material.",
        probability_yes=0.62,
        confidence=0.7,
        key_evidence=["The market-implied probability is near the model estimate."],
        key_risks=["Resolution depends on an uncertain future event."],
    )


def test_partial_structured_prediction_is_repaired_once():
    structured = MagicMock()
    structured.invoke.side_effect = [{"rating": "Hold"}, _decision()]

    rendered = invoke_structured_or_freetext(
        structured, MagicMock(), "prompt", render_pm_decision, "PM",
        schema=PredictionMarketDecision,
        repair_prompt="repair",
        fail_on_structured_error=True,
    )

    assert structured.invoke.call_count == 2
    assert "key_evidence" in structured.invoke.call_args.args[0]
    assert "key_risks" in structured.invoke.call_args.args[0]
    assert "**Probability YES**: 0.62" in rendered


def test_invalid_prediction_is_fail_closed_after_bounded_repair():
    structured = MagicMock()
    structured.invoke.side_effect = [{"rating": "Hold"}, {"rating": "Hold"}]

    with pytest.raises(ValueError, match="structured output and one repair"):
        invoke_structured_or_freetext(
            structured, MagicMock(), "prompt", render_pm_decision, "PM",
            schema=PredictionMarketDecision,
            repair_prompt="repair",
            fail_on_structured_error=True,
        )


def test_probability_bounds_are_enforced_and_structured_render_is_canonical():
    with pytest.raises(ValidationError):
        PredictionMarketDecision(
            **{**_decision().model_dump(), "probability_yes": 1.01}
        )
    assert "P(YES)" not in render_pm_decision(_decision())


def test_singleton_text_evidence_from_provider_is_normalized_without_invention():
    decision = PredictionMarketDecision(
        rating="Hold",
        executive_summary="Evidence is limited.",
        investment_thesis="The available evidence does not justify a trade.",
        probability_yes=0.5,
        confidence=0.5,
        key_evidence="one observed fact",
        key_risks="one unresolved risk",
    )
    assert decision.key_evidence == ["one observed fact"]
    assert decision.key_risks == ["one unresolved risk"]


def test_plain_text_remains_legacy_fallback():
    plain = MagicMock()
    plain.invoke.return_value = MagicMock(content="**Rating**: Hold\nP(YES): 55%")
    result = invoke_structured_or_freetext(
        None, plain, "prompt", render_pm_decision, "PM",
    )
    assert "P(YES): 55%" in result


def test_default_research_accepts_typed_probability(monkeypatch):
    from tradingagents.graph import trading_graph as tg_mod
    from tradingagents.survivor.autonomy.cycle import _default_research
    from tradingagents.survivor.execution.pricing import MarketSnapshot as QuoteSnapshot

    class FakeGraph:
        def __init__(self, **kwargs):
            pass

        def propagate(self, *args, **kwargs):
            return {"final_trade_decision": "typed only", "portfolio_decision": _decision()}, "Hold"

    monkeypatch.setattr(tg_mod, "TradingAgentsGraph", FakeGraph)
    snap = make_snapshot("typed-probability")
    quote = QuoteSnapshot(
        symbol=snap.market_id, bid_pence=snap.bid, ask_pence=snap.ask,
        timestamp_utc=snap.timestamp_utc,
    )
    candidate = type("Candidate", (), {"snapshot": snap, "market_id": snap.market_id})()

    result = _default_research(candidate, quote, {}, "typed-run")
    assert result.status == "OK"
    assert result.expected_probability_bps == 6200
