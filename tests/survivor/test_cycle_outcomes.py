"""Deterministic pipeline-stage and zero-proposal outcome coverage."""

from tradingagents.survivor.autonomy.cycle import ResearchResult, run_survivor_cycle
from tradingagents.survivor.autonomy.state import RuntimeState
from tests.survivor.market_helpers import FakeMarketAdapter, make_snapshot


def test_research_no_trade_and_failure_become_zero_proposal(tmp_path, monkeypatch):
    monkeypatch.setattr("tradingagents.survivor.autonomy.halt.HALT_PATH", str(tmp_path / "HALT"))
    adapter = FakeMarketAdapter([make_snapshot("mkt-0"), make_snapshot("mkt-1")])

    def research(candidate, quote, config, run_id):
        if candidate.market_id == "mkt-1":
            return ResearchResult(status="FAILED", reason="INFERENCE_FAILED: invalid structured output")
        return ResearchResult(
            status="OK", decision_text="NO_TRADE", expected_probability_bps=5000,
            quantity=1,
        )

    report = run_survivor_cycle(
        {"survivor_autonomy_enabled": True}, adapter=adapter, research_fn=research,
        runtime_state=RuntimeState(db_path=str(tmp_path / "runtime.db")),
        paper_ledger_path=str(tmp_path / "paper.db"),
    )
    assert report.reason == "NO_PROPOSAL"
    assert report.pipeline_stage == "RESEARCH"
    assert report.research_attempts == 2
    assert report.research_successes == 1
    assert report.research_failures == 1
    assert report.research_no_trade_count == 1
    assert report.research_inference_failed_count == 1
    assert report.risk_decisions == 0
    assert report.trade_proposals == 0

    state = RuntimeState(db_path=str(tmp_path / "runtime.db"))
    row = state.last_cycle()
    assert row["reason"] == "NO_PROPOSAL"
    assert row["pipeline_stage"] == "RESEARCH"
    assert row["risk_decisions"] == 0


def test_proposal_then_risk_rejection_keeps_risk_reason(tmp_path, monkeypatch):
    monkeypatch.setattr("tradingagents.survivor.autonomy.halt.HALT_PATH", str(tmp_path / "HALT"))
    adapter = FakeMarketAdapter([make_snapshot("mkt-0")])

    def research(candidate, quote, config, run_id):
        return ResearchResult(
            status="OK", decision_text="BUY", expected_probability_bps=7000,
            quantity=1, spread_cost_bps=50,
        )

    report = run_survivor_cycle(
        {"survivor_autonomy_enabled": True}, adapter=adapter, research_fn=research,
        runtime_state=RuntimeState(db_path=str(tmp_path / "runtime.db")),
        paper_ledger_path=str(tmp_path / "paper.db"),
    )
    assert report.trade_proposals == 1
    assert report.risk_decisions == 1
    assert report.reason in {None, "RISK_REJECTED"}


def test_completed_cycle_persists_distinct_finish_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr("tradingagents.survivor.autonomy.halt.HALT_PATH", str(tmp_path / "HALT"))
    state = RuntimeState(db_path=str(tmp_path / "runtime.db"))
    report = run_survivor_cycle(
        {"survivor_autonomy_enabled": True},
        adapter=FakeMarketAdapter([make_snapshot("mkt-0")]),
        research_fn=lambda *_: ResearchResult(status="OK", decision_text="NO_TRADE"),
        runtime_state=state,
        paper_ledger_path=str(tmp_path / "paper.db"),
    )
    row = state.last_cycle()
    assert report.reason == "NO_PROPOSAL"
    assert row["finished_utc"] > row["started_utc"]
    assert row["status"] == "COMPLETED"


def test_provider_timeout_finishes_cycle_without_downstream_work(tmp_path, monkeypatch):
    from tradingagents.survivor.markets.polymarket_adapter import GammaPageError

    monkeypatch.setattr("tradingagents.survivor.autonomy.halt.HALT_PATH", str(tmp_path / "HALT"))

    class FailingAdapter(FakeMarketAdapter):
        def list_markets(self, limit=100):
            raise GammaPageError("GAMMA_TOTAL_TIMEOUT", "page deadline", {
                "pages_attempted": 1, "pages_completed": 0, "rows_fetched": 0,
                "failure_page": 1, "failure_category": "GAMMA_TOTAL_TIMEOUT",
                "pages": [], "scan_duration_ms": 30000,
            })

    state = RuntimeState(db_path=str(tmp_path / "runtime.db"))
    report = run_survivor_cycle(
        {"survivor_autonomy_enabled": True}, adapter=FailingAdapter([]),
        runtime_state=state, paper_ledger_path=str(tmp_path / "paper.db"),
    )
    row = state.last_cycle()
    assert report.reason == "PROVIDER_TIMEOUT"
    assert report.pipeline_stage == "SCAN"
    assert report.research_attempts == 0
    assert report.trade_proposals == 0
    assert row["finished_utc"] > row["started_utc"]
