"""Phase 6 learning-engine tests. No paid API calls anywhere.

Structural + behavioral proof that the learner can only PROPOSE allowlisted
experiments and can never touch safety policy, risk limits, budgets, HALT,
PaperBroker, or any live-trading control.
"""

import sqlite3

import pytest

from tests.survivor.market_helpers import FakeMarketAdapter, make_snapshot
from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger
from tradingagents.survivor.autonomy import halt as halt_mod
from tradingagents.survivor.autonomy.cycle import ResearchResult, run_survivor_cycle
from tradingagents.survivor.autonomy.state import RuntimeState
from tradingagents.survivor.evaluation.store import EvaluationStore
from tradingagents.survivor.evaluation.types import PredictionRecord
from tradingagents.survivor.learning.analytics import (
    category_performance,
    full_analytics,
    model_value_analysis,
    role_value_analysis,
)
from tradingagents.survivor.learning.dataset import LearningRecord, build_learning_records
from tradingagents.survivor.learning.errors import classify_errors
from tradingagents.survivor.learning.experiments import (
    ExperimentStatus,
    ForbiddenChangeError,
    StrategyExperiment,
    generate_experiments,
    validate_config_diff,
    validate_proposal_schema,
)
from tradingagents.survivor.learning.interfaces import (
    EventIntelligenceProvider,
    JarvisInterface,
)
from tradingagents.survivor.learning.registry import ExperimentStore, config_hash
from tradingagents.survivor.learning.report import (
    generate_learning_report,
    render_learning_report,
)
from tradingagents.survivor.learning.sandbox import run_sandbox
from tradingagents.survivor.learning.shadow import ShadowStore, compare_with_current
from tradingagents.survivor.policy import SurvivorPolicy

NOW = "2026-09-04T12:00:00+00:00"


def _experiment(**over) -> StrategyExperiment:
    base = {
        "experiment_id": "exp-test",
        "parent_strategy_version": "survivor-v1.0",
        "created_at": NOW,
        "hypothesis": "Tighten weak-category gates to cut losers.",
        "evidence": {"OVERCONFIDENT": 25},
        "sample_size": 40,
        "allowed_changes": frozenset({"uncertainty_penalty_bps"}),
        "proposed_config_diff": {"uncertainty_penalty_bps": 150},
        "expected_effect": "better calibration",
        "evaluation_plan": {"metric": "brier", "criteria": "IS and OOS Brier improve"},
    }
    base.update(over)
    return StrategyExperiment(**base)


def _record(**over) -> LearningRecord:
    base = {
        "trial_id": "t", "strategy_version": "survivor-v1.0", "cycle_id": "c1",
        "run_id": "r1", "market_id": "m1", "category": "prediction_binary",
        "timestamp_utc": NOW, "market_probability": 0.5, "survivor_probability": 0.7,
        "predicted_edge_bps": 2000, "conservative_edge_bps": 1500,
        "outcome": 1.0, "resolution_timestamp_utc": "2026-09-06T12:00:00+00:00",
    }
    base.update(over)
    return LearningRecord(**base)


# --- 1. resolved outcomes create learning records ----------------------------------

@pytest.mark.unit
def test_resolved_outcomes_create_learning_records(tmp_path):
    store = EvaluationStore(db_path=str(tmp_path / "eval.db"))
    store.record_prediction(PredictionRecord(
        cycle_id="c1", run_id="r1", proposal_id="r1_prop", market_id="m1",
        timestamp_utc=NOW, strategy_version="survivor-v1.0", config_hash="h",
        category="prediction_binary", predicted_probability=0.7,
        market_probability=0.5, gross_edge_bps=2000, net_edge_bps=1500,
        ai_cost_pence=10, outcome=1.0,
        resolution_timestamp_utc="2026-09-06T12:00:00+00:00"))
    records = build_learning_records(store)
    assert len(records) == 1
    record = records[0]
    assert record.resolved and record.survivor_probability == 0.7
    assert record.predicted_edge_bps == 2000 and record.conservative_edge_bps == 1500
    assert record.brier_contribution == pytest.approx(0.09)
    assert record.market_baseline_error == pytest.approx(0.5)
    assert not record.executed and record.pnl_pence == 0


@pytest.mark.unit
def test_no_future_leakage_in_preresolution_features():
    """Resolution data must live OUTSIDE preresolution features."""
    record = _record()
    assert "outcome" not in record.preresolution
    assert "pnl" not in record.preresolution
    assert record.timestamp_utc < record.resolution_timestamp_utc


# --- 3. error classification --------------------------------------------------------

@pytest.mark.unit
def test_error_classifications_are_correct_and_multi_label():
    overconfident = _record(survivor_probability=0.9, market_probability=0.4,
                            outcome=0.0, executed=True, pnl_pence=-100,
                            predicted_edge_bps=3000, conservative_edge_bps=100,
                            realized_return_bps=-500, ai_cost_pence=30)
    labels = classify_errors(overconfident)
    assert "OVERCONFIDENT" in labels and "FALSE_POSITIVE" in labels
    assert "EDGE_OVERSTATED" in labels and "HIGH_AI_COST" in labels

    good_no_trade = _record(survivor_probability=0.3, market_probability=0.8,
                            outcome=0.0, executed=False)
    labels = classify_errors(good_no_trade)
    assert "GOOD_NO_TRADE" in labels
    # market said YES 80%, outcome NO: market was wrong, survivor right
    assert "UNDERCONFIDENT" not in labels

    underconfident = _record(survivor_probability=0.55, market_probability=0.75,
                             outcome=1.0, executed=False)
    assert "UNDERCONFIDENT" in classify_errors(underconfident)

    erased = _record(predicted_edge_bps=1000, conservative_edge_bps=0,
                     executed=True, outcome=0.0, pnl_pence=-10)
    assert "EXECUTION_COST_ERASED_EDGE" in classify_errors(erased)

    assert classify_errors(_record(outcome=None)) == []


# --- 4. category analysis ------------------------------------------------------------

@pytest.mark.unit
def test_category_analysis_correct():
    records = [_record(market_id=f"m{i}", category="sports") for i in range(3)]
    records += [_record(market_id=f"n{i}", category="politics") for i in range(2)]
    perf = category_performance(records)
    assert perf["sports"]["n"] == 3 and perf["politics"]["n"] == 2
    assert perf["sports"]["resolved"] == 3
    # small samples never generate experiments
    assert generate_experiments(full_analytics(records), "survivor-v1.0", NOW) == []


# --- 5. role-value calculation --------------------------------------------------------

@pytest.mark.unit
def test_role_value_calculation_correct():
    records = [_record(
        market_id=f"m{i}",
        preresolution={
            "role_usage": {"news_analyst": {
                "provider": "openrouter", "model": "m", "calls": 2, "cost_pence": 3}},
            "bear_disagreed_with_buy": i % 2 == 0,
        },
        executed=i % 2 == 0, outcome=0.0 if i % 2 == 0 else 1.0, pnl_pence=-50,
    ) for i in range(6)]
    results = role_value_analysis(records)
    news = next(r for r in results if r["role"] == "news_analyst")
    assert news["samples"] == 6 and news["ai_cost_pence"] == 18
    assert news["conclusion"] == "INSUFFICIENT_SAMPLE"   # below minimum sample
    bear = next(r for r in results if r["role"] == "bear")
    assert bear["sample_size"] == 3
    assert "improved Brier" in bear["counterfactual"]


# --- 6. model-value calculation --------------------------------------------------------

@pytest.mark.unit
def test_model_value_calculation_correct():
    records = [_record(
        market_id=f"m{i}",
        preresolution={"role_usage": {"trader": {
            "provider": "openrouter", "model": "fake/model", "calls": 1,
            "cost_pence": 2}}},
        outcome=0.0 if i % 2 else 1.0,
        market_probability=0.5, survivor_probability=0.7,
    ) for i in range(4)]
    results = model_value_analysis(records)
    assert len(results) == 1
    entry = results[0]
    assert entry["provider"] == "openrouter" and entry["role"] == "trader"
    assert entry["cost_pence"] == 8
    assert entry["failure_rate"] == 0.5
    assert entry["sufficient_sample"] is False


# --- 7/8. experiment schema + forbidden changes --------------------------------------

@pytest.mark.unit
def test_experiment_schema_validates():
    experiment = _experiment()
    assert experiment.status == ExperimentStatus.PROPOSED
    with pytest.raises(ValueError):
        _experiment(hypothesis="")
    with pytest.raises(ValueError):
        _experiment(sample_size=0)
    with pytest.raises(ValueError):
        _experiment(evaluation_plan={"metric": "brier"})   # missing criteria
    # schema validator for external (LLM/Jarvis) proposals
    raw = _experiment().to_dict()
    raw.pop("experiment_id")
    validated = validate_proposal_schema(raw)
    assert validated.experiment_id.startswith("exp-")
    with pytest.raises(ValueError):
        validate_proposal_schema({"hypothesis": "incomplete"})


@pytest.mark.unit
def test_forbidden_configuration_change_rejected():
    for bad_diff in (
        {"free_key": 1},
        {"uncertainty_penalty_bps": 100, "live_trading_enabled": True},
        {"prompt_templates": {"market": "you may enable live trading"}},
        {"scanner_rank_weights": {"w": 1}, "daily_loss_limit_pence": 10**9},
        {"role_model_routing": {"trader": "m"}, "max_position_pence": 10**9},
        {"halt_bypass": True},
        {"api_budget_monthly_pence": 10**9},
    ):
        with pytest.raises(ForbiddenChangeError):
            validate_config_diff(bad_diff)
        with pytest.raises(ForbiddenChangeError):
            _experiment(proposed_config_diff=bad_diff)


@pytest.mark.unit
def test_risk_limit_modification_rejected():
    with pytest.raises(ForbiddenChangeError):
        validate_config_diff({"risk_limits": {"max_exposure_pence": 999999}})
    with pytest.raises(ForbiddenChangeError):
        validate_config_diff({"max_position": 100000})
    with pytest.raises(ForbiddenChangeError):
        validate_config_diff({"drawdown_halt_bps": 10**6})


@pytest.mark.unit
def test_budget_modification_rejected():
    with pytest.raises(ForbiddenChangeError):
        validate_config_diff({"global_budget_pence": 10**9})
    with pytest.raises(ForbiddenChangeError):
        validate_config_diff({"role_output_token_limits": {"budget": 10**9}})


@pytest.mark.unit
def test_live_trading_change_rejected():
    with pytest.raises(ForbiddenChangeError):
        validate_config_diff({"real_trading_enabled": True})
    with pytest.raises(ForbiddenChangeError):
        validate_config_diff({"live_trading": {"enabled": True}})


@pytest.mark.unit
def test_policy_survives_experiment_application():
    """Applying an experiment diff to the safety policy changes NOTHING."""
    policy = SurvivorPolicy()
    experiment = _experiment()
    merged = {**policy.__dict__, **experiment.proposed_config_diff}
    assert merged.get("uncertainty_penalty_bps") is not None  # learning key present
    assert policy.real_trading_enabled is False
    assert policy.wallet_enabled is False and policy.broker_enabled is False
    assert policy.withdrawals_enabled is False
    assert policy.borrowing_enabled is False and policy.leverage_enabled is False
    assert policy.mode.value == "PAPER_ONLY"
    # frozen dataclass: policy mutation is structurally impossible anyway
    with pytest.raises(AttributeError):
        policy.real_trading_enabled = True


# --- 10/11. sandbox + anti-overfitting ------------------------------------------------

def _sandbox_records(n=24, winners_first=True) -> list[LearningRecord]:
    """24 resolved records, chronological; early period profitable (in-sample
    for the experiment), late period losing (out-of-sample) when winners_first."""
    records = []
    for i in range(n):
        day = 1 + (i // 2)
        outcome = 1.0 if (i < n // 2) == winners_first else 0.0
        records.append(_record(
            market_id=f"m{i}", cycle_id=f"c{i}", run_id=f"r{i}",
            timestamp_utc=f"2026-08-{day:02d}T12:00:00+00:00",
            outcome=outcome, executed=True, pnl_pence=60 if outcome == 1.0 else -60,
            survivor_probability=0.7, market_probability=0.5,
            predicted_edge_bps=2000, conservative_edge_bps=1500,
            ai_cost_pence=10,
        ))
    return records


@pytest.mark.unit
def test_sandbox_uses_stored_snapshots_only_and_chronological_order():
    records = _sandbox_records()
    experiment = _experiment(proposed_config_diff={"uncertainty_penalty_bps": 150})
    result = run_sandbox(experiment, records)
    # sandbox replays STORED records: nothing fetched, deterministic output
    again = run_sandbox(experiment, records)
    assert result.in_sample == again.in_sample
    assert result.out_of_sample == again.out_of_sample
    # chronological split: 70/30 walk-forward
    assert result.in_sample["current"]["n"] == 16
    assert result.out_of_sample["current"]["n"] == 8
    assert result.current_overall["n"] == 24


@pytest.mark.unit
def test_overfit_experiment_rejected():
    """Improves in-sample (winners early) but degrades out-of-sample."""
    result = run_sandbox(
        _experiment(proposed_config_diff={"uncertainty_penalty_bps": 150}),
        _sandbox_records(winners_first=True))
    assert result.verdict == "OVERFIT_REJECTED"
    assert any("out-of-sample" in r for r in result.reasons)


@pytest.mark.unit
def test_sandbox_insufficient_sample_rejected():
    assert run_sandbox(_experiment(), _sandbox_records(n=10)).verdict == "INSUFFICIENT_SAMPLE"


# --- 12. shadow mode ------------------------------------------------------------------

def test_shadow_records_never_execute_paperbroker(tmp_path):
    import tradingagents.survivor.learning.shadow as shadow_mod
    from tradingagents.survivor.learning.shadow import shadow_decision_for

    store = ShadowStore(db_path=str(tmp_path / "shadow.db"))
    decision = shadow_decision_for(
        _experiment(), cycle_id="c1", market_id="m1", timestamp_utc=NOW,
        survivor_probability=0.7, market_probability=0.5,
        conservative_edge_bps=1500)
    assert decision.action == "OPEN"                      # fictional decision
    store.record(decision)
    assert len(store.decisions("exp-test")) == 1
    # the shadow surface has no execution capability whatsoever
    assert not hasattr(store, "execute") and not hasattr(store, "broker")
    for forbidden in ("PaperBroker", "execute", "place_order"):
        assert not hasattr(shadow_mod, forbidden)


@pytest.mark.unit
def test_shadow_comparison_and_promotion_gate(tmp_path):
    store = ShadowStore(db_path=str(tmp_path / "shadow.db"))
    comparison = compare_with_current(store, _experiment())
    assert comparison["verdict"] == "INSUFFICIENT_SAMPLE"

    ok, reasons = ExperimentStore.promotion_gate({
        "resolved": 25, "experiment": {"brier": 0.15, "economic_pnl_pence": 100},
        "current": {"brier": 0.20},
    })
    assert ok is True and reasons == []
    ok, reasons = ExperimentStore.promotion_gate({
        "resolved": 5, "experiment": {"brier": 0.15, "economic_pnl_pence": 100},
        "current": {"brier": 0.20},
    })
    assert ok is False and "minimum sample not reached" in reasons


# --- 13/14/15. MANUAL promotion + append-only rollback --------------------------------

def test_promotion_requires_explicit_approval(tmp_path):
    store = ExperimentStore(db_path=str(tmp_path / "learning.db"))
    store.save_experiment(_experiment())
    assert store.current_version()[0] == "survivor-v1.0"
    with pytest.raises(ValueError):
        store.approve_experiment("exp-test", operator="", created_at=NOW)
    with pytest.raises(ValueError):
        store.approve_experiment("exp-test", operator="op", created_at=NOW)  # wrong status
    store.set_status("exp-test", ExperimentStatus.CANDIDATE_FOR_PROMOTION)
    new_version = store.approve_experiment("exp-test", operator="owner", created_at=NOW)
    assert new_version.startswith("survivor-v1.")
    version, config = store.current_version()
    assert version == new_version
    assert config.get("uncertainty_penalty_bps") == 150
    assert store.get_experiment("exp-test").status == ExperimentStatus.PROMOTED_MANUALLY
    row = store.get_version(new_version)
    assert row["parent_version"] == "survivor-v1.0"
    assert row["config_hash"] == config_hash(config)


def test_rollback_preserves_history(tmp_path):
    store = ExperimentStore(db_path=str(tmp_path / "learning.db"))
    store.save_experiment(_experiment())
    store.set_status("exp-test", ExperimentStatus.CANDIDATE_FOR_PROMOTION)
    v2 = store.approve_experiment("exp-test", operator="owner", created_at=NOW)
    with pytest.raises(ValueError):
        store.rollback(v2, operator="", created_at=NOW)
    v3 = store.rollback(v2, operator="owner", created_at=NOW)
    assert [h["version"] for h in store.history()] == [v2, v3]
    assert store.get_version(v2)["parent_version"] == "survivor-v1.0"
    assert store.get_version(v3)["parent_version"] == v2
    # rollback restores the parent's config as a NEW version (history intact)
    assert store.get_version(v2)["config_json"] == store.get_version(v3)["config_json"]
    assert store.current_version()[0] == v3


# --- 16/17. God's Eye + Jarvis are read-only ------------------------------------------

def test_gods_eye_cannot_execute_trades():
    provider = EventIntelligenceProvider()
    evidence = provider.get_event_evidence("fed decision", as_of_utc=NOW)
    assert "UNTRUSTED" in evidence and "fed decision" in evidence
    # no mutating capability exists on the interface
    for forbidden in ("execute", "place_order", "approve", "set_halt",
                      "broker", "change_risk", "change_budget"):
        assert not hasattr(provider, forbidden)


def test_jarvis_cannot_approve_experiments(tmp_path):
    store = ExperimentStore(db_path=str(tmp_path / "learning.db"))
    jarvis = JarvisInterface(store)
    # read-only queries work
    assert jarvis.health_summary() == {"halt_state": "CLEAR", "mode": "PAPER_ONLY"}
    assert jarvis.experiment_proposals() == []
    assert jarvis.strategy_history() == []
    # no dangerous capability exists at all
    for forbidden in ("approve_experiment", "set_halt", "clear_halt",
                      "enable_live_trading", "increase_budget", "execute",
                      "move_money", "change_risk"):
        assert not hasattr(jarvis, forbidden)
    # REQUEST only: creates a PROPOSED, inert experiment
    proposal = _experiment().to_dict()
    proposal.pop("experiment_id")
    proposal.pop("status")
    experiment_id = jarvis.request_experiment(proposal)
    assert experiment_id.startswith("exp-")
    assert store.get_experiment(experiment_id).status == ExperimentStatus.PROPOSED
    assert store.current_version()[0] == "survivor-v1.0"   # nothing activated
    # and the request path still rejects forbidden changes
    bad = dict(proposal)
    bad["hypothesis"] = "different"
    bad["proposed_config_diff"] = {"real_trading_enabled": True}
    with pytest.raises(ForbiddenChangeError):
        jarvis.request_experiment(bad)


# --- 18. learning report ---------------------------------------------------------------

def test_learning_report_renders_and_changes_nothing():
    records = _sandbox_records()
    report = generate_learning_report(records, current_version="survivor-v1.0",
                                      created_at=NOW)
    assert report["resolved_sample"] == 24
    assert report["current_strategy_remains"] == "survivor-v1.0"
    rendered = render_learning_report(report)
    assert "SURVIVOR LEARNING REPORT" in rendered
    assert "Current strategy remains: survivor-v1.0" in rendered
    assert "No automatic changes applied." in rendered


# --- 1(continued). cycle wiring: shadow decisions recorded, paper path intact ---------

def _shadow_cycle(tmp_path, monkeypatch, research_fn=None, **cfg):
    monkeypatch.setattr(halt_mod, "HALT_PATH", str(tmp_path / "HALT"))
    return run_survivor_cycle(
        {"survivor_autonomy_enabled": True, "_learning_db_path": str(tmp_path / "learning.db"),
         "_shadow_db_path": str(tmp_path / "shadow.db"), **cfg},
        adapter=FakeMarketAdapter([make_snapshot("mkt-0"), make_snapshot("mkt-1")]),
        research_fn=research_fn or (lambda c, q, cfg_, r: ResearchResult(
            status="OK", decision_text="BUY", expected_probability_bps=7000,
            quantity=1, spread_cost_bps=50, slippage_bps=50, fee_bps=50,
            uncertainty_penalty_bps=100, ai_cost_pence=2, run_id=r)),
        paper_ledger_path=str(tmp_path / "paper.db"),
        runtime_state=RuntimeState(db_path=str(tmp_path / "runtime.db")),
        usage_ledger=InferenceUsageLedger(db_path=str(tmp_path / "usage.db")),
        policy=SurvivorPolicy(),
    )


def test_cycle_records_shadow_decisions_without_affecting_paper(tmp_path, monkeypatch):
    registry = ExperimentStore(db_path=str(tmp_path / "learning.db"))
    shadow_exp = _experiment(
        allowed_changes=frozenset({"min_research_confidence"}),
        proposed_config_diff={"min_research_confidence": 0.1},
    )
    registry.save_experiment(shadow_exp)
    registry.set_status("exp-test", ExperimentStatus.SHADOW_TESTING)

    report = _shadow_cycle(tmp_path, monkeypatch, survivor_dry_run=True)
    assert report.ai_researched == 2
    assert report.paper_trades == 0                    # dry run: zero executions

    shadow = ShadowStore(db_path=str(tmp_path / "shadow.db"))
    decisions = shadow.decisions("exp-test")
    assert len(decisions) == 2                         # one fictional decision per candidate
    assert all(d.action == "OPEN" for d in decisions)  # experiment still opens at 0.1 conf

    # CURRENT paper path decided normally and is untouched by the experiment
    conn = sqlite3.connect(str(tmp_path / "paper.db"))
    execs = conn.execute(
        "SELECT COUNT(*) FROM paper_events WHERE event_type LIKE '%EXEC%'").fetchone()[0]
    conn.close()
    assert execs == 0


def test_shadow_failure_never_breaks_paper_cycle(tmp_path, monkeypatch):
    # corrupt the learning db so the shadow path throws internally
    registry = ExperimentStore(db_path=str(tmp_path / "learning.db"))
    registry.save_experiment(_experiment())
    registry.set_status("exp-test", ExperimentStatus.SHADOW_TESTING)
    (tmp_path / "shadow.db").write_text("not a database")

    report = _shadow_cycle(tmp_path, monkeypatch)
    assert report.status in ("ACTIVE", "DRY_RUN")      # cycle survived
    assert report.ai_researched == 2


# --- structural: no mutation capability anywhere in the learner -----------------------

LEARNING_DIR = "tradingagents/survivor/learning"


def test_learner_has_no_mutation_capability_over_safety_surfaces():
    """Structural scan: the learning package contains no code that can modify
    safety config, clear HALT, execute on the broker, or reach wallet/live
    trading capability. Capability = imports/instantiation, not prose."""
    import pathlib

    package = pathlib.Path(__file__).resolve().parents[2] / LEARNING_DIR
    forbidden_patterns = (
        "set_halt", "clear_halt", "paperbroker(", "place_order(",
        "from tradingagents.survivor.execution",
        "from tradingagents.survivor.execution.paper_broker",
        "real_trading_enabled =", "wallet_enabled =", "broker_enabled =",
        "withdrawals_enabled =", "borrowing_enabled =", "leverage_enabled =",
        "tradingagents_survivor_autonomy", "os.environ[", "environ.update",
        "os.putenv", "open(halt", "unlink()",
    )
    hits = []
    for path in sorted(package.glob("*.py")):
        content = path.read_text().lower()
        for pattern in forbidden_patterns:
            if pattern in content:
                hits.append(f"{path.name}: {pattern}")
    assert hits == [], f"learner contains forbidden capability: {hits}"


def test_learner_cannot_clear_halt(tmp_path):
    """Even with a HALT file present, every learning surface leaves it alone."""
    halt_path = tmp_path / "HALT"
    halt_mod.set_halt(str(halt_path))
    store = ExperimentStore(db_path=str(tmp_path / "learning.db"))
    store.save_experiment(_experiment())
    report = generate_learning_report(_sandbox_records(), created_at=NOW)
    render_learning_report(report)
    run_sandbox(_experiment(), _sandbox_records())
    jarvis = JarvisInterface(store)
    jarvis.health_summary()
    assert halt_path.exists()                          # HALT untouched
    assert halt_mod.is_halted(str(halt_path)) is True


def test_learner_cannot_access_wallet_or_broker_capability():
    """No learning module imports execution/wallet surfaces. The only place
    'wallet' may appear is the defensive FORBIDDEN_PATTERNS deny-list."""
    import pathlib

    package = pathlib.Path(__file__).resolve().parents[2] / LEARNING_DIR
    for path in sorted(package.glob("*.py")):
        content = path.read_text()
        assert "import PaperBroker" not in content
        assert "from tradingagents.survivor.execution" not in content
        if path.name != "experiments.py":
            # outside the defensive deny-list, wallet is never referenced
            assert "wallet" not in content.lower(), path.name
        else:
            assert '"wallet"' in content   # deny-list entry only
