"""Monitoring, resolution settlement, and the end-to-end integration scenarios."""

import pytest

from tests.survivor.market_helpers import FakeMarketAdapter, make_snapshot
from tests.survivor.paper_helpers import fresh_snapshot, make_proposal
from tradingagents.survivor.execution.paper_broker import PaperBroker
from tradingagents.survivor.execution.pricing import MarketSnapshot as Quote
from tradingagents.survivor.execution.resolution import settle_prediction_position
from tradingagents.survivor.risk.limits import RiskLimits
from tradingagents.survivor.risk.result import ReasonCode, RiskDecision, RiskStatus


def _broker_with_yes_position(tmp_path, quantity=2, entry=50):
    broker = PaperBroker(limits=RiskLimits(), ledger=__import__(
        "tests.survivor.paper_helpers", fromlist=["tmp_ledger"]).tmp_ledger(tmp_path))
    broker.ensure_initialized()
    proposal = make_proposal(symbol="PMKT", quantity=quantity,
                             reference_price_pence=entry, notional_pence=quantity * entry)
    approved = RiskDecision(RiskStatus.APPROVED, ReasonCode.APPROVED, "t",
                            proposal.proposal_id)
    broker.execute(proposal, Quote(symbol="PMKT", bid_pence=entry, ask_pence=entry,
                                   timestamp_utc=fresh_snapshot().timestamp_utc), approved)
    return broker


@pytest.mark.unit
def test_long_position_marked_at_bid_updates_unrealized(tmp_path):
    broker = _broker_with_yes_position(tmp_path)
    # bid drops 52p -> 40p: unrealized loss must flow into daily P/L (not clamped)
    broker.mark_to_market([Quote(symbol="PMKT", bid_pence=40, ask_pence=44,
                                 timestamp_utc=fresh_snapshot().timestamp_utc)])
    pos = broker.portfolio.positions["PMKT"]
    assert pos.current_mark_pence == 40            # bid, NOT midpoint (42)
    assert broker.portfolio.unrealized_pnl_pence == 2 * (40 - 51)  # -22, not clamped
    assert broker.portfolio.daily_pnl_pence < 0


@pytest.mark.unit
def test_resolved_yes_settles_at_face_value(tmp_path):
    broker = _broker_with_yes_position(tmp_path, quantity=2, entry=50)
    event = settle_prediction_position(broker, "PMKT", resolved_yes=True)
    assert event["event_type"] == "SETTLEMENT_APPLIED"
    types = [e["event_type"] for e in broker.ledger.events()]
    assert "POSITION_RESOLVED" in types and "SETTLEMENT_APPLIED" in types
    # 2 YES shares pay 100p each -> 200p; cost basis 102p + 1p fee -> realized +97p
    assert "PMKT" not in broker.portfolio.positions
    assert broker.portfolio.cash_pence == 2000 - 103 + 200
    assert broker.portfolio.realized_pnl_pence == 97


@pytest.mark.unit
def test_resolved_no_pays_zero_for_long_yes(tmp_path):
    broker = _broker_with_yes_position(tmp_path, quantity=2, entry=50)
    settle_prediction_position(broker, "PMKT", resolved_yes=False)
    # NO resolution -> payoff 0; realized loss = full cost basis
    assert "PMKT" not in broker.portfolio.positions
    assert broker.portfolio.realized_pnl_pence == -103  # cost basis loss + fee
    assert broker.portfolio.cash_pence == 2000 - 103


@pytest.mark.unit
def test_settlement_requires_open_position(tmp_path):
    broker = PaperBroker(limits=RiskLimits(), ledger=__import__(
        "tests.survivor.paper_helpers", fromlist=["tmp_ledger"]).tmp_ledger(tmp_path))
    broker.ensure_initialized()
    assert settle_prediction_position(broker, "MISSING", resolved_yes=True) is None


# ---------------------------------------------------------------------------
# AH/AJ integration scenarios: full fake end-to-end, no network, no real AI
# ---------------------------------------------------------------------------

from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger  # noqa: E402
from tradingagents.survivor.autonomy import halt as halt_mod  # noqa: E402
from tradingagents.survivor.autonomy.cycle import (  # noqa: E402
    ResearchResult,
    run_survivor_cycle,
)
from tradingagents.survivor.autonomy.state import RuntimeState  # noqa: E402
from tradingagents.survivor.policy import SurvivorPolicy  # noqa: E402


def _e2e_config(**over):
    base = {"survivor_autonomy_enabled": True}
    base.update(over)
    return base


@pytest.mark.unit
def test_full_cycle_discovery_to_execution(tmp_path, monkeypatch):
    """20 fake markets -> filters -> top 2 -> fake AI -> A approved/executed, B rejected."""
    monkeypatch.setattr(halt_mod, "HALT_PATH", str(tmp_path / "HALT"))
    snapshots = []
    for i in range(18):
        snapshots.append(make_snapshot(
            f"low-liq-{i}", liquidity_usd_cents=1000 + i))          # LOW_LIQUIDITY
    snapshots.append(make_snapshot("mkt-A", liquidity_usd_cents=500000,  # passes, good rank
                                   question="Will A happen?"))
    snapshots.append(make_snapshot("mkt-B", liquidity_usd_cents=400000,  # passes
                                   question="Will B happen?"))
    adapter = FakeMarketAdapter(snapshots)

    def research(candidate, quote, config, run_id):
        # candidate A has a genuine >5% edge; candidate B's edge is insufficient
        if candidate.market_id == "mkt-A":
            return ResearchResult(
                status="OK", decision_text="BUY",
                expected_probability_bps=7000, quantity=1,
                spread_cost_bps=50, slippage_bps=50, fee_bps=50,
                uncertainty_penalty_bps=100, ai_cost_pence=2, run_id=run_id,
            )
        return ResearchResult(
            status="OK", decision_text="BUY",
            expected_probability_bps=5240,  # net edge ~ -10bps < 500 -> REJECT
            quantity=1, spread_cost_bps=50, slippage_bps=50, fee_bps=50,
            uncertainty_penalty_bps=100, ai_cost_pence=2, run_id=run_id,
        )

    report = run_survivor_cycle(
        _e2e_config(), adapter=adapter, research_fn=research,
        paper_ledger_path=str(tmp_path / "paper.db"),
        runtime_state=RuntimeState(db_path=str(tmp_path / "runtime.db")),
        usage_ledger=InferenceUsageLedger(db_path=str(tmp_path / "usage.db")),
        policy=SurvivorPolicy(),
    )

    assert report.status == "ACTIVE"
    assert report.markets_discovered == 20
    assert report.passed_filters == 2
    assert report.research_candidates == 2
    assert report.ai_researched == 2
    assert report.trade_proposals == 2
    assert report.approved == 1 and report.rejected == 1
    assert report.paper_trades == 1
    assert report.ai_cost_pence == 4

    # ledger + portfolio updated for A only
    from tradingagents.survivor.execution.ledger import PaperLedger

    ledger = PaperLedger(db_path=str(tmp_path / "paper.db"))
    executions = [e for e in ledger.events() if e["event_type"] == "TRADE_EXECUTED"]
    assert len(executions) == 1 and executions[0]["symbol"] == "mkt-A"
    broker = PaperBroker(limits=RiskLimits(), ledger=ledger)
    assert broker.portfolio.positions["mkt-A"].quantity == 1
    assert "mkt-B" not in broker.portfolio.positions

    # traceability: cycle state recorded with metrics
    state = RuntimeState(db_path=str(tmp_path / "runtime.db"))
    assert state.cycle_count() == 1
    last = state.last_cycle()
    assert last["executed"] == 1 and last["ai_cost_pence"] == 4


@pytest.mark.unit
def test_halt_file_blocks_full_cycle(tmp_path, monkeypatch):
    """HALT exists -> run_survivor_cycle: 0 market calls, 0 AI calls, 0 executions."""
    monkeypatch.setattr(halt_mod, "HALT_PATH", str(tmp_path / "HALT"))
    adapter = FakeMarketAdapter([make_snapshot(f"mkt-{i}") for i in range(5)])
    calls = {"research": 0}

    def research(candidate, quote, config, run_id):
        calls["research"] += 1
        return ResearchResult(status="OK", decision_text="BUY")

    halt_mod.set_halt(str(tmp_path / "HALT"))
    report = run_survivor_cycle(
        _e2e_config(), adapter=adapter, research_fn=research,
        paper_ledger_path=str(tmp_path / "paper.db"),
        runtime_state=RuntimeState(db_path=str(tmp_path / "runtime.db")),
        usage_ledger=InferenceUsageLedger(db_path=str(tmp_path / "usage.db")),
    )
    assert report.status == "EXTERNAL_HALT"
    assert adapter.list_calls == 0
    assert calls["research"] == 0
    assert report.trade_proposals == 0 and report.paper_trades == 0



# __PART2__
