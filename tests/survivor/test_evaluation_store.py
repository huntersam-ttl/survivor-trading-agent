"""Evaluation store, integrity checks, gate, reproducibility, integration scenarios."""

import json
import sqlite3

import pytest

from tradingagents.survivor.evaluation.evaluate import (
    EvaluationBlockedError,
    evaluate_performance,
    export_csv,
    export_json,
)
from tradingagents.survivor.evaluation.store import EvaluationStore
from tradingagents.survivor.evaluation.types import (
    GateState,
    PredictionRecord,
    SampleLabel,
    TradeOutcome,
    Warning_,
)
from tradingagents.survivor.evaluation.versioning import strategy_config_hash, strategy_identity

V = "survivor-v1.0"
H = strategy_config_hash()  # match the evaluator default identity


def _pred(i, pred, outcome, chash=H, shash="", category="macro", market=None):
    from datetime import datetime, timedelta, timezone

    ts = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)).isoformat()
    return PredictionRecord(
        cycle_id=f"c{i}", run_id=f"r{i}", proposal_id=f"p{i}", market_id=f"m{i}",
        timestamp_utc=ts, strategy_version=V, config_hash=chash, category=category,
        predicted_probability=pred, market_probability=market if market is not None else pred,
        gross_edge_bps=500, net_edge_bps=500, ai_cost_pence=2,
        outcome=outcome, resolution_timestamp_utc=ts, snapshot_data_hash=shash,
    )


def _trade(i, pnl, ai_cost=2, chash=H, shash="", category="macro", latency=1000):
    from datetime import datetime, timedelta, timezone

    ts = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)).isoformat()
    return TradeOutcome(
        cycle_id=f"c{i}", run_id=f"r{i}", proposal_id=f"p{i}", market_id=f"m{i}",
        timestamp_utc=ts, strategy_version=V, config_hash=chash, category=category,
        quantity=1, entry_price_pence=50, fees_pence=1, slippage_pence=1,
        gross_pnl_pence=pnl, ai_cost_pence=ai_cost, realized_return_bps=pnl * 100,
        predicted_net_edge_bps=500, snapshot_data_hash=shash, decision_latency_ms=latency,
    )


def _seed(store, predictions, trades):
    for p in predictions:
        store.record_prediction(p)
    for t in trades:
        store.record_trade(t)
        store.record_cost_attribution(t.run_id, t.market_id, t.cycle_id, t.ai_cost_pence)




@pytest.mark.unit
def test_store_records_and_reads(tmp_path):
    store = EvaluationStore(db_path=str(tmp_path / "e.db"))
    _seed(store, [_pred(0, 0.7, 1.0)], [_trade(0, 48)])
    assert len(store.predictions()) == 1 and len(store.trades()) == 1
    assert store.predictions()[0].outcome == 1.0
    assert store.cost_attribution_total() == {"r0": 2}


@pytest.mark.unit
def test_attach_outcome_once_only(tmp_path):
    store = EvaluationStore(db_path=str(tmp_path / "e.db"))
    store.record_prediction(_pred(0, 0.7, None))
    assert store.attach_outcome("p0", 1.0, "2026-01-02T00:00:00+00:00") is True
    assert store.attach_outcome("p0", 0.0, "2026-01-03T00:00:00+00:00") is False
    assert store.predictions()[0].outcome == 1.0


@pytest.mark.unit
def test_ai_cost_makes_profitable_trade_economically_negative(tmp_path):
    store = EvaluationStore(db_path=str(tmp_path / "e.db"))
    _seed(store, [_pred(i, 0.7, 1.0) for i in range(10)],
          [_trade(i, 8, ai_cost=20) for i in range(10)])
    report = evaluate_performance(store)
    assert report.net_pnl_pence == 80
    assert report.total_ai_cost_pence == 200
    assert report.net_pnl_after_ai_cost_pence == -120
    assert Warning_.AI_COST_EXCEEDS_EDGE.value in report.warnings


@pytest.mark.unit
def test_zero_positive_negative_trade_reports(tmp_path):
    report = evaluate_performance(EvaluationStore(db_path=str(tmp_path / "a.db")))
    assert report.trades == 0 and report.gate == GateState.INSUFFICIENT_DATA
    assert report.win_rate == 0.0 and report.profit_factor == 0.0

    store2 = EvaluationStore(db_path=str(tmp_path / "b.db"))
    _seed(store2, [], [_trade(i, 50, ai_cost=0) for i in range(10)])
    positive = evaluate_performance(store2)
    assert positive.net_pnl_pence == 500 and positive.win_rate == 1.0

    store3 = EvaluationStore(db_path=str(tmp_path / "c.db"))
    _seed(store3, [], [_trade(i, -40, ai_cost=0) for i in range(10)])
    negative = evaluate_performance(store3)
    assert negative.net_pnl_pence == -400 and negative.profit_factor == 0.0


@pytest.mark.unit
def test_corrupted_paper_ledger_blocks_evaluation(tmp_path):
    from tradingagents.survivor.execution.ledger import PaperLedger

    ledger_path = str(tmp_path / "paper.db")
    ledger = PaperLedger(db_path=ledger_path)
    ledger.append_event(event_type="TREASURY_INITIALIZED", run_id="r", symbol="GBP",
                        cash_before=0, cash_after=2000, equity_before=0, equity_after=2000)
    conn = sqlite3.connect(ledger_path)
    conn.execute("UPDATE paper_events SET equity_after = 1 WHERE rowid = 1")
    conn.commit()
    conn.close()
    store = EvaluationStore(db_path=str(tmp_path / "e.db"))
    _seed(store, [], [_trade(0, 50)])
    with pytest.raises(EvaluationBlockedError):
        evaluate_performance(store, paper_ledger_path=ledger_path)
    good = PaperLedger(db_path=str(tmp_path / "good.db"))
    good.append_event(event_type="TREASURY_INITIALIZED", run_id="r", symbol="GBP",
                      cash_before=0, cash_after=2000, equity_before=0, equity_after=2000)
    report = evaluate_performance(store, paper_ledger_path=str(tmp_path / "good.db"))
    assert report.trades == 1


@pytest.mark.unit
def test_corrupted_snapshot_excluded_and_flagged(tmp_path):
    store = EvaluationStore(db_path=str(tmp_path / "e.db"))
    _seed(store, [_pred(0, 0.7, 1.0, shash="bad-tampered"), _pred(1, 0.7, 1.0)],
          [_trade(0, 50, shash="bad-tampered"), _trade(1, 50)])
    report = evaluate_performance(store)
    assert Warning_.SNAPSHOT_INTEGRITY_FAILURE.value in report.warnings
    assert report.trades == 1 and report.resolved_predictions == 1


@pytest.mark.unit
def test_strategy_versions_cannot_be_silently_mixed(tmp_path):
    store = EvaluationStore(db_path=str(tmp_path / "e.db"))
    _seed(store, [_pred(0, 0.7, 1.0)], [_trade(0, 50)])
    store.record_prediction(_pred(1, 0.7, 1.0, chash="different-hash"))
    report = evaluate_performance(store, strategy_version=V, config_hash=H)
    assert report.trades == 1 and report.sample_size == 1
    assert Warning_.STRATEGY_VERSION_MISMATCH.value in report.warnings
    assert (V, "different-hash") in store.distinct_identities()


@pytest.mark.unit
def test_config_hash_deterministic_and_sensitive():
    a = strategy_config_hash({"min_edge_bps": 500, "rank_liquidity": 0.3})
    b = strategy_config_hash({"min_edge_bps": 500, "rank_liquidity": 0.3})
    c = strategy_config_hash({"min_edge_bps": 600, "rank_liquidity": 0.3})
    assert a == b and a != c
    assert strategy_identity()[0] == "survivor-v1.0"


@pytest.mark.unit
def test_out_of_sample_negative_flagged_despite_positive_overall(tmp_path):
    store = EvaluationStore(db_path=str(tmp_path / "e.db"))
    predictions, trades = [], []
    for i in range(30):
        predictions.append(_pred(i, 0.7, 1.0))
        trades.append(_trade(i, 50))
    for i in range(30, 40):
        predictions.append(_pred(i, 0.7, 0.0))
        trades.append(_trade(i, -60))
    _seed(store, predictions, trades)
    report = evaluate_performance(store, min_sample_for_promising=30)
    assert report.net_pnl_pence == 30 * 50 - 10 * 60
    assert report.out_of_sample_net_pnl_pence < 0
    assert Warning_.OUT_OF_SAMPLE_NEGATIVE.value in report.warnings
    assert report.gate in (GateState.FAIL, GateState.CONTINUE_PAPER)
    assert report.gate != GateState.PROMISING_BUT_UNPROVEN


@pytest.mark.unit
def test_market_baseline_outperforming_flagged(tmp_path):
    store = EvaluationStore(db_path=str(tmp_path / "e.db"))
    records = [_pred(i, 0.5, 1.0, market=0.9) for i in range(40)]
    _seed(store, records, [_trade(i, -5) for i in range(40)])
    report = evaluate_performance(store)
    assert report.brier > report.market_brier
    assert Warning_.MARKET_BASELINE_OUTPERFORMS.value in report.warnings


@pytest.mark.unit
def test_category_breakdown_correct(tmp_path):
    store = EvaluationStore(db_path=str(tmp_path / "e.db"))
    _seed(store,
          [_pred(i, 0.7, 1.0, category="politics") for i in range(3)]
          + [_pred(10 + i, 0.7, 0.0, category="crypto") for i in range(2)],
          [_trade(i, 40, category="politics") for i in range(3)]
          + [_trade(10 + i, -20, category="crypto") for i in range(2)])
    report = evaluate_performance(store)
    assert report.category_breakdown["politics"]["net_pnl_pence"] == 120
    assert report.category_breakdown["crypto"]["net_pnl_pence"] == -40
    assert report.category_breakdown["politics"]["win_rate"] == 1.0


@pytest.mark.unit
def test_evaluation_gate_never_ready_for_live(tmp_path):
    store = EvaluationStore(db_path=str(tmp_path / "e.db"))
    predictions, trades = [], []
    for i in range(600):
        outcome = 1.0 if i % 2 == 0 else 0.0
        predictions.append(_pred(i, 0.8 if i % 2 == 0 else 0.2, outcome, market=0.5))
        trades.append(_trade(i, 50, ai_cost=1))
    _seed(store, predictions, trades)
    report = evaluate_performance(store)
    assert report.gate == GateState.PROMISING_BUT_UNPROVEN
    assert "READY_FOR_LIVE" not in {s.value for s in GateState}


@pytest.mark.unit
def test_insufficient_data_gate(tmp_path):
    store = EvaluationStore(db_path=str(tmp_path / "e.db"))
    _seed(store, [_pred(i, 0.7, 1.0) for i in range(5)], [_trade(i, 50) for i in range(5)])
    report = evaluate_performance(store)
    assert report.gate == GateState.INSUFFICIENT_DATA
    assert report.sample_confidence == SampleLabel.INSUFFICIENT_SAMPLE


@pytest.mark.unit
def test_export_json_csv_no_secrets(tmp_path):
    store = EvaluationStore(db_path=str(tmp_path / "e.db"))
    _seed(store, [_pred(0, 0.7, 1.0)], [_trade(0, 50)])
    report = evaluate_performance(store)
    as_json = json.loads(export_json(report))
    assert as_json["strategy_version"] == V
    csv_text = export_csv(store.trades())
    for secret in ("api_key", "OPENAI_API_KEY", "sk-", "private_key", "password"):
        assert secret not in json.dumps(as_json)
        assert secret not in csv_text


@pytest.mark.unit
def test_report_reproducible(tmp_path):
    store = EvaluationStore(db_path=str(tmp_path / "e.db"))
    _seed(store, [_pred(i, 0.7, 1.0) for i in range(50)],
          [_trade(i, 50 if i % 3 else -20) for i in range(50)])
    r1 = evaluate_performance(store, bootstrap_resamples=100)
    r2 = evaluate_performance(store, bootstrap_resamples=100)
    assert r1.to_dict() == r2.to_dict()
    assert export_json(r1) == export_json(r2)

# ---------------------------------------------------------------------------
# AE/AF synthetic integration scenarios (deterministic, no network, no AI)
# ---------------------------------------------------------------------------

from datetime import datetime, timedelta, timezone  # noqa: E402


def _history(count, edge_fn, pnl_fn, start_index=0):
    """Build deterministic resolved history: predictions + trades."""
    predictions, trades = [], []
    for i in range(count):
        idx = start_index + i
        ts = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=idx)).isoformat()
        predicted, outcome = edge_fn(idx)
        predictions.append(PredictionRecord(
            cycle_id=f"c{idx}", run_id=f"r{idx}", proposal_id=f"p{idx}",
            market_id=f"m{idx}", timestamp_utc=ts, strategy_version=V,
            config_hash=strategy_config_hash(), category="macro",
            predicted_probability=predicted, market_probability=0.5,
            gross_edge_bps=500, net_edge_bps=500, ai_cost_pence=3,
            outcome=outcome, resolution_timestamp_utc=ts,
        ))
        trades.append(TradeOutcome(
            cycle_id=f"c{idx}", run_id=f"r{idx}", proposal_id=f"p{idx}",
            market_id=f"m{idx}", timestamp_utc=ts, strategy_version=V,
            config_hash=strategy_config_hash(), category="macro",
            quantity=1, entry_price_pence=50, fees_pence=1, slippage_pence=1,
            gross_pnl_pence=pnl_fn(idx), ai_cost_pence=3,
            realized_return_bps=pnl_fn(idx) * 100, predicted_net_edge_bps=500,
        ))
    return predictions, trades


@pytest.mark.unit
def test_deceptive_history_out_of_sample_fails_gate(tmp_path):
    """AE: first 300 look profitable, next 100 weak, last 100 negative.
    Aggregate P/L may look positive but the OOS tail fails -> NOT promising."""
    store = EvaluationStore(db_path=str(tmp_path / "e.db"))

    def edge_fn(idx):
        if idx < 300:
            return (0.8, 1.0) if idx % 2 == 0 else (0.2, 0.0)   # perfectly right
        if idx < 400:
            return (0.7, 1.0) if idx % 2 == 0 else (0.3, 0.0)   # right, weaker
        return (0.8, 0.0) if idx % 2 == 0 else (0.2, 1.0)        # always wrong

    def pnl_fn(idx):
        if idx < 300:
            return 50
        if idx < 400:
            return 5
        return -60

    predictions, trades = _history(500, edge_fn, pnl_fn)
    _seed(store, predictions, trades)
    report = evaluate_performance(store, bootstrap_resamples=100)
    assert report.net_pnl_pence > 0  # aggregate looks positive
    assert report.out_of_sample_net_pnl_pence < 0
    assert Warning_.OUT_OF_SAMPLE_NEGATIVE.value in report.warnings
    assert report.gate in (GateState.FAIL, GateState.CONTINUE_PAPER)
    assert report.gate != GateState.PROMISING_BUT_UNPROVEN


@pytest.mark.unit
def test_well_calibrated_positive_history_is_promising_but_unproven(tmp_path):
    """AF: 1000 resolved, well calibrated, positive economic P/L, beats market
    Brier, distributed profits, positive OOS => PROMISING_BUT_UNPROVEN, and
    NEVER anything resembling READY_FOR_LIVE."""
    store = EvaluationStore(db_path=str(tmp_path / "e.db"))

    def edge_fn(idx):
        # well-calibrated: predicted band matches realized frequency
        band = idx % 5
        predicted = {0: 0.55, 1: 0.60, 2: 0.65, 3: 0.70, 4: 0.75}[band]
        outcome = 1.0 if (idx * 7 + band * 3) % 10 < predicted * 10 else 0.0
        return (predicted, outcome)

    def pnl_fn(idx):
        return 40 + (idx % 20)  # positive, distributed

    predictions, trades = _history(1000, edge_fn, pnl_fn)
    _seed(store, predictions, trades)
    report = evaluate_performance(store, bootstrap_resamples=200)
    assert report.resolved_predictions == 1000
    assert report.net_pnl_after_ai_cost_pence > 0
    assert report.out_of_sample_net_pnl_pence > 0
    assert report.gate == GateState.PROMISING_BUT_UNPROVEN
    assert "READY_FOR_LIVE" not in {s.value for s in GateState}
    # sample label at the top tier, but the gate never escalates beyond unproven
    assert report.sample_confidence == SampleLabel.STRONGER_SAMPLE
