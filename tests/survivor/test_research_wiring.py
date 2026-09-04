"""Regression tests: autonomous candidates are WIRED into the research pipeline.

Covers the real defect where scanner candidates reached the research queue but
no inference ever happened (and failures were silent). No real paid APIs — the
TradingAgentsGraph is faked at the class boundary; everything else is the real
deterministic cycle code.
"""

import sqlite3

import pytest

from tests.survivor.market_helpers import FakeMarketAdapter, make_snapshot
from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger
from tradingagents.survivor.autonomy import cycle as cycle_mod, halt as halt_mod
from tradingagents.survivor.autonomy.cycle import (
    ResearchResult,
    _default_research,
    run_survivor_cycle,
)
from tradingagents.survivor.autonomy.state import RuntimeState
from tradingagents.survivor.execution.pricing import MarketSnapshot as QuoteSnapshot
from tradingagents.survivor.policy import SurvivorPolicy

GOOD_EDGE = {
    "expected_probability_bps": 7000,  # gross 1800 - 250 costs = 1550bps net edge
    "quantity": 1, "spread_cost_bps": 50, "slippage_bps": 50,
    "fee_bps": 50, "uncertainty_penalty_bps": 100,
}


def _config(**over):
    base = {"survivor_autonomy_enabled": True}
    base.update(over)
    return base


def _cycle(tmp_path, adapter, monkeypatch, research_fn=None, usage_ledger=None, **cfg):
    monkeypatch.setattr(halt_mod, "HALT_PATH", str(tmp_path / "HALT"))
    return run_survivor_cycle(
        _config(**cfg),
        adapter=adapter,
        research_fn=research_fn,
        paper_ledger_path=str(tmp_path / "paper.db"),
        runtime_state=RuntimeState(db_path=str(tmp_path / "runtime.db")),
        usage_ledger=usage_ledger or InferenceUsageLedger(db_path=str(tmp_path / "usage.db")),
        policy=SurvivorPolicy(),
    )


def _ok_research(run_id: str) -> ResearchResult:
    return ResearchResult(status="OK", decision_text="BUY", run_id=run_id, **GOOD_EDGE)


def _quote_for(snap) -> QuoteSnapshot:
    return QuoteSnapshot(symbol=snap.market_id, bid_pence=snap.bid // 100,
                         ask_pence=snap.ask // 100, timestamp_utc=snap.timestamp_utc)


# --- 1. valid candidates + configured research backend => AI researched ----------

@pytest.mark.unit
def test_two_valid_candidates_are_researched(tmp_path, monkeypatch):
    adapter = FakeMarketAdapter([make_snapshot("mkt-0"), make_snapshot("mkt-1")])
    calls = []

    def research(candidate, quote, config, run_id):
        calls.append((candidate.market_id, run_id))
        return _ok_research(run_id)

    report = _cycle(tmp_path, adapter, monkeypatch, research_fn=research)
    assert report.research_candidates == 2
    assert report.ai_researched == 2
    assert report.candidate_failures == []
    assert len(calls) == 2
    # every research call carries a unique, cycle-scoped run_id
    assert len({r for _, r in calls}) == 2


# --- 2. missing research callback => explicit, never silent -----------------------

@pytest.mark.unit
def test_missing_research_callback_is_explicit_not_silent(tmp_path, monkeypatch):
    adapter = FakeMarketAdapter([make_snapshot("mkt-0"), make_snapshot("mkt-1")])
    monkeypatch.setattr(cycle_mod, "_default_research", None)

    report = _cycle(tmp_path, adapter, monkeypatch)
    assert report.status == "ACTIVE"          # cycle survives, no crash
    assert report.ai_researched == 0
    assert len(report.candidate_failures) == 2
    assert all(f["reason"] == "RESEARCH_CALLBACK_MISSING" for f in report.candidate_failures)
    # surfaced in the rendered report too — not hidden
    assert "RESEARCH_CALLBACK_MISSING" in report.render()


# --- 3+4. default research invokes the real graph boundary + usage ledger ---------

class _FakeGraph:
    """Captures the research config; simulates one guarded multi-agent run."""

    instances: list = []
    calls: list = []

    def __init__(self, config=None, **kwargs):
        self.config = config or {}
        _FakeGraph.instances.append(self)

    def propagate(self, company_name, trade_date, asset_type="stock"):
        _FakeGraph.calls.append((company_name, trade_date, asset_type))
        ledger = InferenceUsageLedger(db_path=self.config["survivor_usage_ledger_path"])
        ledger.record_inference(
            run_id=self.config["survivor_run_id"],
            agent_role="quick_analyst", provider="deepseek", model="deepseek-chat",
            ticker_or_market=company_name,
            input_tokens=100, output_tokens=10, reasoning_tokens=0,
            native_cost_minor=1, native_currency="USD", gbp_cost_pence=2,
            timestamp_utc="2026-09-04T10:00:00Z",
        )
        return (
            {"final_trade_decision": "**Rating**: Buy\n\n**Executive Summary**: go\n"
                                     "P(YES): 65%"},
            "Buy",
        )


@pytest.mark.unit
def test_default_research_invokes_tradingagents_graph_and_usage_ledger(tmp_path, monkeypatch):
    from tradingagents.graph import trading_graph as tg_mod

    _FakeGraph.instances = []
    _FakeGraph.calls = []
    monkeypatch.setattr(tg_mod, "TradingAgentsGraph", _FakeGraph)

    usage = InferenceUsageLedger(db_path=str(tmp_path / "usage.db"))
    snap = make_snapshot("mkt-poly")
    candidate = type("C", (), {"snapshot": snap, "market_id": "mkt-poly"})()

    result = _default_research(candidate, _quote_for(snap),
                               {"_usage_ledger": usage, "survivor_dry_run": True},
                               run_id="run_wiring_1")

    assert result.status == "OK"
    assert result.expected_probability_bps == 6500       # parsed deterministically
    assert result.quantity == 1
    assert result.ai_cost_pence == 2                     # from the usage ledger
    assert "Buy" in result.decision_text

    # graph got the guarded control plane + run identity + prediction-market context
    assert len(_FakeGraph.instances) == 1
    cfg = _FakeGraph.instances[0].config
    assert cfg["survivor_enabled"] is True
    assert cfg["survivor_paper_enabled"] is False        # graph never executes
    assert cfg["survivor_run_id"] == "run_wiring_1"
    assert cfg["survivor_usage_ledger_path"] == str(usage.db_path)
    context = cfg["survivor_instrument_context"]
    assert "mkt-poly" in context
    assert "5200" in context                             # market YES probability bps
    assert "P(YES): <NN>%" in context                    # decision format requirement
    assert "UNTRUSTED DATA" in context                   # injection-safe wrapper
    assert "MUST NOT replace" in context                 # point-in-time rule
    # identifier passed to propagate is the market id, never a fake stock ticker
    assert _FakeGraph.calls[0][0] == "mkt-poly"
    assert _FakeGraph.calls[0][2] == "prediction_market"

    # 4. the inference usage ledger actually received a record for this run
    summary = usage.get_run_usage_summary("run_wiring_1")
    assert summary["total_calls"] == 1
    assert summary["total_cost_pence"] == 2


# --- 5. dry run: risk may run, PaperBroker.execute never --------------------------

@pytest.mark.unit
def test_dry_run_runs_risk_but_never_executes(tmp_path, monkeypatch):
    adapter = FakeMarketAdapter([make_snapshot("mkt-0")])
    report = _cycle(tmp_path, adapter, monkeypatch,
                    research_fn=lambda c, q, cfg, r: _ok_research(r),
                    survivor_dry_run=True)
    assert report.trade_proposals == 1
    assert report.approved == 1
    assert report.paper_trades == 0                      # dry run stops execution

    conn = sqlite3.connect(str(tmp_path / "paper.db"))
    rows = conn.execute(
        "SELECT COUNT(*) FROM paper_events WHERE event_type LIKE '%EXEC%'"
    ).fetchone()
    conn.close()
    assert rows[0] == 0                                  # zero broker executions


# --- 6. provider failure => explicit candidate failure, cycle stays safe ----------

@pytest.mark.unit
def test_provider_failure_isolated_and_explicit(tmp_path, monkeypatch):
    adapter = FakeMarketAdapter([make_snapshot("mkt-0"), make_snapshot("mkt-1")])
    calls = {"n": 0}

    def research(candidate, quote, config, run_id):
        calls["n"] += 1
        if candidate.market_id == "mkt-0":
            return ResearchResult(status="FAILED", reason="INFERENCE_FAILED: provider down",
                                  run_id=run_id)
        return _ok_research(run_id)

    report = _cycle(tmp_path, adapter, monkeypatch, research_fn=research)
    assert report.status == "ACTIVE"
    assert report.ai_researched == 1                     # healthy candidate researched
    assert len(report.candidate_failures) == 1
    assert report.candidate_failures[0]["reason"].startswith("INFERENCE_FAILED")
    assert report.candidate_failures[0]["market_id"] == "mkt-0"
    assert calls["n"] == 2


@pytest.mark.unit
def test_default_research_classifies_provider_errors(tmp_path, monkeypatch):
    from tradingagents.graph import trading_graph as tg_mod

    snap = make_snapshot("mkt-err")
    candidate = type("C", (), {"snapshot": snap, "market_id": "mkt-err"})()

    class _BrokenGraph:
        def __init__(self, config=None, **kwargs):
            pass

        def propagate(self, *a, **k):
            raise RuntimeError("Error code: 401 - invalid api key")

    monkeypatch.setattr(tg_mod, "TradingAgentsGraph", _BrokenGraph)
    result = _default_research(candidate, _quote_for(snap), {}, run_id="run_err")
    assert result.status == "FAILED"
    assert result.reason.startswith("NO_LLM_PROVIDER")

    class _RouteGraph(_BrokenGraph):
        def propagate(self, *a, **k):
            raise RuntimeError("No model route configured for agent role 'trader'.")

    monkeypatch.setattr(tg_mod, "TradingAgentsGraph", _RouteGraph)
    result = _default_research(candidate, _quote_for(snap), {}, run_id="run_err2")
    assert result.reason.startswith("NO_MODEL_ROUTE")


# --- 7. budget unavailable behavior preserved -------------------------------------

@pytest.mark.unit
def test_budget_unavailable_still_skips_before_research(tmp_path, monkeypatch):
    adapter = FakeMarketAdapter([make_snapshot("mkt-0"), make_snapshot("mkt-1")])
    usage = InferenceUsageLedger(db_path=str(tmp_path / "usage.db"))
    usage.record_inference(
        run_id="pre", agent_role="trader", provider="deepseek", model="deepseek-chat",
        ticker_or_market="m", input_tokens=1, output_tokens=1, reasoning_tokens=0,
        native_cost_minor=1, native_currency="USD", gbp_cost_pence=170,
        timestamp_utc="2026-09-04T10:00:00Z",
    )
    calls = {"n": 0}

    def research(candidate, quote, config, run_id):
        calls["n"] += 1
        return _ok_research(run_id)

    report = _cycle(tmp_path, adapter, monkeypatch, research_fn=research, usage_ledger=usage)
    assert report.skipped_budget == 2
    assert report.ai_researched == 0
    assert calls["n"] == 0                   # zero AI calls when budget is gone


# --- 8. the scanner remains zero-LLM ----------------------------------------------

@pytest.mark.unit
def test_scanner_makes_zero_llm_calls(tmp_path, monkeypatch):
    """If any LLM call is attempted during scan, the test explodes."""
    from datetime import datetime, timezone

    from tradingagents.llm_clients.router import ModelRouter
    from tradingagents.survivor.markets.filters import scan_limits_from_env
    from tradingagents.survivor.markets.scanner import MarketScanner

    def _no_llm(*a, **k):
        raise AssertionError("scanner attempted an LLM call")

    monkeypatch.setattr(ModelRouter, "invoke_role", _no_llm)
    adapter = FakeMarketAdapter([make_snapshot("mkt-0"), make_snapshot("mkt-1")])
    scanner = MarketScanner(adapter, limits=scan_limits_from_env(),
                            max_candidates_per_cycle=40, max_research_candidates_per_cycle=3)
    scan = scanner.scan(now=datetime.now(timezone.utc),
                        open_position_symbols=frozenset(), trading_halted=False)
    assert len(scan.top) == 2
    assert scan.discovered == 2


# --- 9. point-in-time check vs fetch-time stamps (real-run regression) ------------

@pytest.mark.unit
def test_freshly_fetched_snapshot_is_not_skipped_as_stale(tmp_path, monkeypatch):
    """Regression: adapter snapshots are stamped at FETCH time, which is always
    after the cycle-start timestamp. They must NOT trip the point-in-time
    check — only genuinely future data may be skipped."""
    from datetime import datetime, timezone

    snap = make_snapshot("mkt-fresh",
                         source_timestamp_utc=datetime.now(timezone.utc).isoformat())
    adapter = FakeMarketAdapter([snap])
    report = _cycle(tmp_path, adapter, monkeypatch,
                    research_fn=lambda c, q, cfg, r: _ok_research(r))
    assert report.ai_researched == 1
    assert report.candidate_failures == []


@pytest.mark.unit
def test_genuinely_future_data_is_skipped_explicitly(tmp_path, monkeypatch):
    """Data timestamped after the decision moment is still rejected — and the
    skip is now surfaced in the report instead of silently vanishing."""
    from datetime import datetime, timedelta, timezone

    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    snap = make_snapshot("mkt-future", source_timestamp_utc=future)
    adapter = FakeMarketAdapter([snap])
    report = _cycle(tmp_path, adapter, monkeypatch,
                    research_fn=lambda c, q, cfg, r: _ok_research(r))
    assert report.ai_researched == 0
    assert len(report.candidate_failures) == 1
    assert report.candidate_failures[0]["reason"] == "SKIPPED_STALE_DATA"
    assert "SKIPPED_STALE_DATA" in report.render()
