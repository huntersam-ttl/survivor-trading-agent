"""Offline regression tests for the typed prediction-market decision path."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
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


def test_no_trade_allows_probability_to_be_absent_but_keeps_semantics_required():
    decision = PredictionMarketDecision(
        rating="NO_TRADE",
        executive_summary="Do not enter this market.",
        investment_thesis="The available evidence is too conflicting to justify an entry.",
        confidence="0.4",
        key_evidence=["The evidence is mixed."],
        key_risks=["The outcome remains uncertain."],
    )
    assert decision.rating is PortfolioRating.HOLD
    assert decision.probability_yes is None
    assert "Probability YES" not in render_pm_decision(decision)


def test_actionable_prediction_still_requires_probability():
    with pytest.raises(ValidationError, match="probability_yes"):
        PredictionMarketDecision(
            rating="Buy",
            executive_summary="Enter the market.",
            investment_thesis="The evidence supports an entry.",
            confidence=0.7,
            key_evidence=["Positive evidence."],
            key_risks=["The event could still resolve otherwise."],
        )


def test_plain_text_remains_legacy_fallback():
    plain = MagicMock()
    plain.invoke.return_value = MagicMock(content="**Rating**: Hold\nP(YES): 55%")
    result = invoke_structured_or_freetext(
        None, plain, "prompt", render_pm_decision, "PM",
    )
    assert "P(YES): 55%" in result


def test_structured_json_inside_prose_and_fences_is_normalized():
    structured = MagicMock()
    structured.invoke.return_value = (
        "Here is the decision:\n```json\n"
        + _decision().model_dump_json()
        + "\n```"
    )

    rendered = invoke_structured_or_freetext(
        structured, MagicMock(), "prompt", render_pm_decision, "PM",
        schema=PredictionMarketDecision,
    )

    assert "**Probability YES**: 0.62" in rendered


def test_empty_structured_result_uses_one_validated_text_repair():
    structured = MagicMock()
    structured.invoke.return_value = None
    plain = MagicMock()
    plain.invoke.return_value = MagicMock(content=_decision().model_dump_json())

    rendered = invoke_structured_or_freetext(
        structured, plain, "prompt", render_pm_decision, "PM",
        schema=PredictionMarketDecision,
        repair_prompt="repair",
        fail_on_structured_error=True,
    )

    assert "**Probability YES**: 0.62" in rendered
    assert "JSON only" in structured.invoke.call_args.args[0]


def test_text_repair_still_fails_closed_on_missing_semantics():
    structured = MagicMock()
    structured.invoke.return_value = None
    plain = MagicMock()
    plain.invoke.return_value = MagicMock(content='{"rating":"Hold"}')

    with pytest.raises(ValueError, match="structured output and one repair"):
        invoke_structured_or_freetext(
            structured, plain, "prompt", render_pm_decision, "PM",
            schema=PredictionMarketDecision,
            repair_prompt="repair",
            fail_on_structured_error=True,
        )


@pytest.mark.parametrize("rating", ["BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"])
def test_declared_rating_casing_is_normalized_without_changing_semantics(rating):
    decision = {
        "rating": rating,
        "executive_summary": "The evidence supports a bounded decision.",
        "investment_thesis": "The decision is grounded in the supplied evidence.",
        "confidence": 0.6,
        "key_evidence": ["One observed fact."],
        "key_risks": ["One unresolved risk."],
        "probability_yes": None if rating == "HOLD" else 0.6,
    }
    parsed = PredictionMarketDecision.model_validate(decision)
    assert parsed.rating.value == rating.title()


def test_common_tool_envelope_and_prose_are_extracted_without_inventing_fields():
    structured = MagicMock()
    structured.invoke.return_value = {
        "type": "tool_call",
        "args": {"output": _decision().model_dump()},
    }
    rendered = invoke_structured_or_freetext(
        structured, MagicMock(), "prompt", render_pm_decision, "PM",
        schema=PredictionMarketDecision,
    )
    assert "**Rating**: Hold" in rendered
    assert "**Key Evidence**:" in rendered


def test_include_raw_envelope_recovers_json_from_unparsed_provider_message():
    structured = MagicMock()
    structured.invoke.return_value = {
        "raw": AIMessage(content=_decision().model_dump_json()),
        "parsed": None,
        "parsing_error": ValueError("no tool call"),
    }
    rendered = invoke_structured_or_freetext(
        structured, MagicMock(), "prompt", render_pm_decision, "PM",
        schema=PredictionMarketDecision,
    )
    assert "**Confidence**: 0.7" in rendered


def test_failure_diagnostics_are_bounded_and_field_specific():
    structured = MagicMock()
    structured.invoke.side_effect = [
        {"rating": "Hold", "confidence": "high"},
        {"rating": "Hold", "confidence": "high"},
    ]
    telemetry = {"structured_failure_diagnostics": []}
    from tradingagents.agents.utils.structured import (
        reset_structured_telemetry,
        structured_telemetry_scope,
    )
    token = structured_telemetry_scope(telemetry)
    try:
        with pytest.raises(ValueError, match="structured output and one repair"):
            invoke_structured_or_freetext(
                structured, MagicMock(), "prompt", render_pm_decision, "PM",
                schema=PredictionMarketDecision, repair_prompt="repair",
                fail_on_structured_error=True,
            )
    finally:
        reset_structured_telemetry(token)
    assert telemetry["structured_failure_diagnostics"]
    assert any(
        "confidence" in item["fields"]
        for item in telemetry["structured_failure_diagnostics"]
    )
    assert all(len(item["message"]) <= 500 for item in telemetry["structured_failure_diagnostics"])


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
