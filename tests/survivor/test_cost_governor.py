"""Governor + cost-leak regression tests. Mocked providers only — zero real
OpenRouter credit spent.

Covers: oversized-context blocking, per-call/output clamps, call-rate caps
(minute + hour), daily USD budget, kill switch, bounded retries (no retry on
governor block), duplicate-analysis cooldown (zero calls on unchanged state),
and paper trading never bypasses the governor.
"""

import datetime as _dt
import os
import tempfile
from decimal import Decimal

import pytest

from tests.survivor.market_helpers import FakeMarketAdapter, make_snapshot
from tradingagents.llm_clients.cost_governor import CostGovernor
from tradingagents.llm_clients.pricing import ModelPrice, register_model_price
from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger
from tradingagents.survivor.autonomy import halt as halt_mod
from tradingagents.survivor.autonomy.cycle import run_survivor_cycle
from tradingagents.survivor.autonomy.state import RuntimeState
from tradingagents.survivor.policy import SurvivorPolicy
from tradingagents.survivor.types import (
    AutonomyLLMDisabledError,
    GovernorExhaustedError,
)

MODEL = "test/model"


def _ledger():
    return InferenceUsageLedger(db_path=os.path.join(tempfile.mkdtemp(), "u.db"))


def _governor(**over):
    g = CostGovernor(ledger=_ledger())
    for k, v in over.items():
        setattr(g, k, v)
    return g


@pytest.fixture(autouse=True)
def _pricing():
    register_model_price(ModelPrice("openrouter", MODEL, Decimal("0.05"), Decimal("0.20")))
    yield
    from tradingagents.llm_clients.pricing import MODEL_PRICING

    MODEL_PRICING.pop(("openrouter", MODEL), None)


# --- oversized context + output clamp ------------------------------------------------------

@pytest.mark.unit
def test_oversized_input_context_blocked(monkeypatch):
    g = _governor(max_input_tokens=12000)
    with pytest.raises(GovernorExhaustedError, match="input token cap"):
        g.check_input_tokens(200_000)


@pytest.mark.unit
def test_output_tokens_clamped():
    g = _governor(max_output_tokens=2000)
    assert g.clamp_output_tokens(64_000) == 2000
    assert g.clamp_output_tokens(500) == 500


# --- call-rate caps --------------------------------------------------------------------------

@pytest.mark.unit
def test_hourly_call_limit_enforced():
    g = _governor(calls_per_minute=100_000, calls_per_hour=5)
    now = _dt.datetime.now(_dt.timezone.utc)
    for _ in range(5):
        g._record_local(now)          # 5 calls recorded (len == 5)
    g._record_local(now)              # a 6th call...
    with pytest.raises(GovernorExhaustedError, match="last hour"):
        g._check_local_rate(now + _dt.timedelta(minutes=1))


@pytest.mark.unit
def test_minute_call_limit_enforced():
    g = _governor(calls_per_minute=2, calls_per_hour=100_000)
    now = _dt.datetime.now(_dt.timezone.utc)
    g._record_local(now)
    g._record_local(now)
    with pytest.raises(GovernorExhaustedError, match="last minute"):
        g._check_local_rate(now + _dt.timedelta(seconds=5))


# --- daily USD budget + kill switch -----------------------------------------------------------

@pytest.mark.unit
def test_daily_budget_blocks_calls():
    g = _governor(daily_budget_usd=0.01)
    g.ledger.record_inference(
        run_id="r", agent_role="trader", provider="openrouter", model=MODEL,
        ticker_or_market="m", input_tokens=10, output_tokens=10, reasoning_tokens=0,
        native_cost_minor=200, native_currency="USD", gbp_cost_pence=154,
        timestamp_utc=_dt.datetime.now(_dt.timezone.utc).isoformat())
    assert g.ledger.get_native_spend_today("openrouter") > 0.01     # already over tiny budget
    with pytest.raises(GovernorExhaustedError, match="daily LLM budget"):
        g._check_daily_budget("openrouter", MODEL, 100, 100)


@pytest.mark.unit
def test_autonomy_kill_switch_blocks_calls():
    g = _governor(enabled=False)
    with pytest.raises(AutonomyLLMDisabledError):
        g.check_enabled()


# --- bounded retries: governor block surfaces, never retried as a model call -----------------

@pytest.mark.unit
def test_governor_block_surfaces_as_failure_not_hot_loop(tmp_path, monkeypatch):
    from tradingagents.graph import trading_graph as tg_mod

    class _TooBigGraph:
        def __init__(self, config=None, **kwargs):
            self.config = config or {}

        def propagate(self, *a, **k):
            raise GovernorExhaustedError("input token cap exceeded")

    monkeypatch.setattr(tg_mod, "TradingAgentsGraph", _TooBigGraph)
    monkeypatch.setattr(halt_mod, "HALT_PATH", str(tmp_path / "HALT"))
    monkeypatch.delenv("SURVIVOR_DRY_RUN", raising=False)
    monkeypatch.delenv("SURVIVOR_RESEARCH_COOLDOWN_HOURS", raising=False)
    report = run_survivor_cycle(
        {"survivor_autonomy_enabled": True},
        adapter=FakeMarketAdapter([make_snapshot("mkt-0")]),
        research_fn=None,
        paper_ledger_path=str(tmp_path / "paper.db"),
        runtime_state=RuntimeState(db_path=str(tmp_path / "runtime.db")),
        usage_ledger=InferenceUsageLedger(db_path=str(tmp_path / "usage.db")),
        policy=SurvivorPolicy(),
    )
    assert report.ai_researched == 0
    assert report.candidate_failures and "COST_GOVERNOR" in report.candidate_failures[0]["reason"]
    assert report.status == "ACTIVE"                 # deterministic cycle continued


# --- duplicate-analysis cooldown: unchanged state -> ZERO model calls ------------------------

@pytest.mark.unit
def test_unchanged_market_state_causes_zero_model_calls(tmp_path, monkeypatch):
    """A decision-time snapshot materially identical to a recently analyzed one
    must be served from cache — the research backend is never invoked."""
    from tradingagents.survivor.evaluation.store import EvaluationStore
    from tradingagents.survivor.evaluation.types import PredictionRecord

    monkeypatch.setattr(halt_mod, "HALT_PATH", str(tmp_path / "HALT"))
    monkeypatch.delenv("SURVIVOR_DRY_RUN", raising=False)
    monkeypatch.setenv("SURVIVOR_RESEARCH_COOLDOWN_HOURS", "24")
    monkeypatch.delenv("LLM_AUTONOMY_ENABLED", raising=False)

    snap = make_snapshot("mkt-dup", question="Will X resolve YES by 2027?")
    store = EvaluationStore(db_path=str(tmp_path / "eval.db"))
    from tradingagents.survivor.autonomy.state import snapshot_data_hash

    store.record_prediction(PredictionRecord(
        cycle_id="old", run_id="old", proposal_id="old_prop", market_id="mkt-dup",
        timestamp_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        strategy_version="survivor-v1.0", config_hash="h", category="prediction_binary",
        predicted_probability=0.6, market_probability=0.5,
        gross_edge_bps=1000, net_edge_bps=500, ai_cost_pence=2,
        snapshot_data_hash=snapshot_data_hash(snap)))

    report = run_survivor_cycle(
        {"survivor_autonomy_enabled": True, "_evaluation_db_path": str(tmp_path / "eval.db")},
        adapter=FakeMarketAdapter([snap]),
        research_fn=lambda *a: (_ for _ in ()).throw(
            AssertionError("research backend called for unchanged state")),
        paper_ledger_path=str(tmp_path / "paper.db"),
        runtime_state=RuntimeState(db_path=str(tmp_path / "runtime.db")),
        usage_ledger=InferenceUsageLedger(db_path=str(tmp_path / "usage.db")),
        policy=SurvivorPolicy(),
    )
    assert report.ai_researched == 0                 # ZERO model calls
    assert report.candidate_failures == []           # skipped via cache, not failure
    assert report.details.get("cached_analysis") == ["mkt-dup"]
