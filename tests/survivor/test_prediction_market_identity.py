"""Regression tests: prediction markets are separated from ticker identifiers.

Root cause covered: the Polymarket slug/question was passed as the graph's
ticker and tripped ``safe_ticker_component``'s 32-char limit
("ticker exceeds 32 chars"). Prediction markets now use a deterministic
opaque internal id (pm_<12-hex>); the full question and decision-time
snapshot reach the agents via ``survivor_instrument_context`` only, and no
stock-data lookup occurs for prediction_market assets.
"""

from unittest.mock import MagicMock, patch

import pytest

from tests.survivor.market_helpers import FakeMarketAdapter, make_snapshot
from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger
from tradingagents.survivor.autonomy import halt as halt_mod
from tradingagents.survivor.autonomy.cycle import (
    _default_research,
    _prediction_market_internal_id,
    run_survivor_cycle,
)
from tradingagents.survivor.autonomy.state import RuntimeState
from tradingagents.survivor.execution.pricing import MarketSnapshot as QuoteSnapshot
from tradingagents.survivor.policy import SurvivorPolicy

# The REAL failing value from the production run (83 chars).
REAL_SLUG = ("will-there-be-no-change-in-fed-interest-rates"
             "-after-the-september-2026-meeting-615")
REAL_QUESTION = ("Will there be no change in Fed interest rates after the "
                 "September 2026 meeting?")
OTHER_SLUG = ("will-the-fed-increase-interest-rates-by-25-bps"
              "-after-the-september-2026-meeting-649")


def _snapshot(market_id="615", slug=REAL_SLUG, question=REAL_QUESTION):
    snap = make_snapshot(market_id, question=question)
    return type(snap)(**{**snap.__dict__, "symbol_or_slug": slug})


def _candidate(snap):
    return type("C", (), {"snapshot": snap, "market_id": snap.market_id})()


def _quote_for(snap):

    return QuoteSnapshot(symbol=snap.market_id, bid_pence=snap.bid // 100,
                         ask_pence=snap.ask // 100, timestamp_utc=snap.timestamp_utc)


# --- 1/2. long slug never hits ticker validation; internal id <= 32 -----------------

@pytest.mark.unit
def test_long_polymarket_slug_never_hits_ticker_validation():
    with pytest.raises(ValueError, match="exceeds 32"):
        safe_ticker_component(REAL_SLUG)               # the old behaviour

    internal = _prediction_market_internal_id("615")
    assert len(internal) <= 32
    assert safe_ticker_component(internal) == internal  # passes validation


@pytest.mark.unit
def test_internal_id_length_bound_for_any_market_id():
    for market_id in ("615", "x" * 300, "", "mkt-with-unicode-é"):
        internal = _prediction_market_internal_id(market_id)
        assert internal.startswith("pm_")
        assert len(internal) <= 32
        assert safe_ticker_component(internal) == internal


# --- 3/4. deterministic and collision-distinct ---------------------------------------

@pytest.mark.unit
def test_same_market_id_generates_same_internal_id():
    assert _prediction_market_internal_id("615") == _prediction_market_internal_id("615")
    assert _prediction_market_internal_id(REAL_SLUG) == \
        _prediction_market_internal_id(REAL_SLUG)


@pytest.mark.unit
def test_different_market_ids_generate_different_ids():
    a = _prediction_market_internal_id("615")
    b = _prediction_market_internal_id("649")
    c = _prediction_market_internal_id(OTHER_SLUG)
    assert len({a, b, c}) == 3
    assert all(i.startswith("pm_") for i in (a, b, c))


# --- 5. full question remains available in the research context ----------------------

class _CapturingGraph:
    instances: list = []

    def __init__(self, config=None, **kwargs):
        self.config = config or {}
        _CapturingGraph.instances.append(self)

    def propagate(self, company_name, trade_date, asset_type="stock"):
        _CapturingGraph.calls.append((company_name, asset_type))
        ledger_path = self.config.get("survivor_usage_ledger_path")
        if ledger_path:
            ledger = InferenceUsageLedger(db_path=ledger_path)
            ledger.record_inference(
                run_id=self.config.get("survivor_run_id", "default_run"),
                agent_role="quick_analyst", provider="openrouter", model="fake/model",
                ticker_or_market=company_name,
                input_tokens=100, output_tokens=10, reasoning_tokens=0,
                native_cost_minor=1, native_currency="USD", gbp_cost_pence=2,
                timestamp_utc="2026-09-04T10:00:00Z",
            )
        return ({"final_trade_decision": "**Rating**: Buy\n\nP(YES): 62%"}, "Buy")


@pytest.mark.unit
def test_full_question_reaches_context_and_identifier_is_opaque(monkeypatch):
    from tradingagents.graph import trading_graph as tg_mod

    _CapturingGraph.instances = []
    _CapturingGraph.calls = []
    monkeypatch.setattr(tg_mod, "TradingAgentsGraph", _CapturingGraph)

    snap = make_snapshot("615", question=REAL_QUESTION)
    snap = type(snap)(**{**snap.__dict__, "symbol_or_slug": REAL_SLUG})
    result = _default_research(_candidate(snap), _quote_for(snap), {}, run_id="run_pm")

    assert result.status == "OK"
    graph = _CapturingGraph.instances[0]
    # ticker-shaped field carries the opaque id, NEVER the slug/question
    assert _CapturingGraph.calls[0][0] == _prediction_market_internal_id("615")
    assert _CapturingGraph.calls[0][0] != REAL_SLUG
    assert _CapturingGraph.calls[0][1] == "prediction_market"
    # the FULL question and market_id reach the agents via the evidence context
    context = graph.config["survivor_instrument_context"]
    assert REAL_QUESTION in context
    assert "market_id: 615" in context
    assert "5200" in context                       # YES probability bps
    assert "internal_instrument_id:" in context


# --- 6/7. NO stock lookup / no yfinance for prediction_market -------------------------

@pytest.mark.unit
def test_no_stock_lookup_for_prediction_market(monkeypatch):
    """If ANY stock-data path is touched during prediction-market research,
    the test explodes (yfinance, instrument identity, tool layer)."""

    import tradingagents.dataflows.utils as df_utils

    def _no_yf(*a, **k):
        raise AssertionError("stock lookup attempted for a prediction market")

    monkeypatch.setattr("yfinance.Ticker", _no_yf)
    monkeypatch.setattr(df_utils, "yf", MagicMock(side_effect=_no_yf), raising=False)
    import tradingagents.agents.utils.agent_utils as agent_utils
    from tradingagents.graph import trading_graph as tg_mod

    monkeypatch.setattr(tg_mod, "TradingAgentsGraph", _CapturingGraph)
    monkeypatch.setattr(agent_utils, "resolve_instrument_identity",
                        lambda ticker: (_ for _ in ()).throw(AssertionError(
                            "resolve_instrument_identity called for a prediction market")))
    monkeypatch.setattr(
        tg_mod, "resolve_instrument_identity",
        lambda ticker: (_ for _ in ()).throw(AssertionError(
            "resolve_instrument_identity called for a prediction market")),
        raising=False)

    _CapturingGraph.instances = []
    _CapturingGraph.calls = []
    snap = make_snapshot("615", question=REAL_QUESTION)
    snap = type(snap)(**{**snap.__dict__, "symbol_or_slug": REAL_SLUG})
    result = _default_research(_candidate(snap), _quote_for(snap), {}, run_id="run_pm2")
    assert result.status == "OK"


@pytest.mark.unit
def test_propagate_skips_memory_resolution_for_prediction_markets(monkeypatch):
    """propagate() must not resolve pending memory entries (stock returns via
    yfinance) for asset_type='prediction_market', but must for stocks."""
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    mock_llm_client = MagicMock()
    with patch("tradingagents.graph.trading_graph.create_llm_client", mock_llm_client):
        graph = TradingAgentsGraph(
            config={**DEFAULT_CONFIG, "survivor_enabled": False,
                    "checkpoint_enabled": False})
    resolved = []
    monkeypatch.setattr(graph, "_resolve_pending_entries",
                        lambda ticker: resolved.append(ticker))
    monkeypatch.setattr(graph, "_run_graph", lambda *a, **k: {})

    graph.propagate(_prediction_market_internal_id("615"), "2026-09-04",
                    asset_type="prediction_market")
    assert resolved == []                        # zero stock lookups for pm assets

    graph.propagate("AAPL", "2026-09-04", asset_type="stock")
    assert resolved == ["AAPL"]                  # stock path unchanged


# --- 8/9/10. autonomous cycle with the REAL failing slug ------------------------------

def _pm_cycle(tmp_path, monkeypatch, **cfg):
    from tradingagents.graph import trading_graph as tg_mod

    _CapturingGraph.instances = []
    _CapturingGraph.calls = []
    monkeypatch.setattr(tg_mod, "TradingAgentsGraph", _CapturingGraph)
    monkeypatch.setattr(halt_mod, "HALT_PATH", str(tmp_path / "HALT"))
    monkeypatch.delenv("SURVIVOR_DRY_RUN", raising=False)
    adapter = FakeMarketAdapter([
        _snapshot("615"), _snapshot("649", slug=OTHER_SLUG,
                                    question="Will the Fed increase interest rates by 25 bps "
                                             "after the September 2026 meeting?"),
    ])
    return run_survivor_cycle(
        {"survivor_autonomy_enabled": True,
         "_learning_db_path": str(tmp_path / "learning.db"),
         "_shadow_db_path": str(tmp_path / "shadow.db"),
         "_evaluation_db_path": str(tmp_path / "eval.db"), **cfg},
        adapter=adapter,
        research_fn=None,                        # the REAL default research path
        paper_ledger_path=str(tmp_path / "paper.db"),
        runtime_state=RuntimeState(db_path=str(tmp_path / "runtime.db")),
        usage_ledger=InferenceUsageLedger(db_path=str(tmp_path / "usage.db")),
        policy=SurvivorPolicy(),
    )


@pytest.mark.unit
def test_fake_research_completes_for_long_slugs(tmp_path, monkeypatch):
    """The two REAL failing candidates now research successfully."""
    report = _pm_cycle(tmp_path, monkeypatch)
    assert report.research_candidates == 2
    assert report.ai_researched == 2
    assert report.candidate_failures == []       # no 'ticker exceeds 32 chars'
    assert report.ai_cost_pence > 0


@pytest.mark.unit
def test_pm_dry_run_zero_paper_execution(tmp_path, monkeypatch):
    import sqlite3

    report = _pm_cycle(tmp_path, monkeypatch, survivor_dry_run=True)
    assert report.trade_proposals >= 1
    assert report.paper_trades == 0
    conn = sqlite3.connect(str(tmp_path / "paper.db"))
    rows = conn.execute(
        "SELECT COUNT(*) FROM paper_events WHERE event_type LIKE '%EXEC%'").fetchone()[0]
    conn.close()
    assert rows == 0
