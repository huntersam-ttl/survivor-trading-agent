"""Phase 4 evaluation tests: metrics, calibration, splits, integrity, gate."""

import pytest

from tradingagents.survivor.evaluation import metrics as M
from tradingagents.survivor.evaluation.splits import chronological_split, walk_forward
from tradingagents.survivor.evaluation.types import (
    PredictionRecord,
    SampleLabel,
    TradeOutcome,
    sample_label,
)

V = "survivor-v1.0"
H = "hash-test"


def _pred(i, pred, outcome, market=None, category="macro", chash=H, shash=""):
    from datetime import datetime, timedelta, timezone

    ts = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)).isoformat()
    return PredictionRecord(
        cycle_id=f"c{i}", run_id=f"r{i}", proposal_id=f"p{i}", market_id=f"m{i}",
        timestamp_utc=ts, strategy_version=V, config_hash=chash, category=category,
        predicted_probability=pred, market_probability=market if market is not None else pred,
        gross_edge_bps=500, net_edge_bps=500, ai_cost_pence=2,
        outcome=outcome, resolution_timestamp_utc=ts, snapshot_data_hash=shash,
    )


def _trade(i, pnl, ai_cost=2, category="macro", chash=H, shash="", latency=1000,
           entry=50, edge=500):
    from datetime import datetime, timedelta, timezone

    ts = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)).isoformat()
    return TradeOutcome(
        cycle_id=f"c{i}", run_id=f"r{i}", proposal_id=f"p{i}", market_id=f"m{i}",
        timestamp_utc=ts, strategy_version=V, config_hash=chash, category=category,
        quantity=1, entry_price_pence=entry, fees_pence=1, slippage_pence=1,
        gross_pnl_pence=pnl, ai_cost_pence=ai_cost, realized_return_bps=pnl * 100,
        predicted_net_edge_bps=edge, snapshot_data_hash=shash, decision_latency_ms=latency,
    )


@pytest.mark.unit
def test_brier_exact():
    assert M.brier_score([(0.8, 1.0), (0.4, 0.0)]) == pytest.approx(0.10)
    assert M.brier_score([]) == 0.0


@pytest.mark.unit
def test_perfect_and_bad_calibration():
    perfect = M.calibration_bins([_pred(0, 0.70, 1.0), _pred(1, 0.70, 1.0), _pred(2, 0.70, 1.0)])
    assert perfect[0].outcome_frequency == 1.0
    assert perfect[0].calibration_error == pytest.approx(0.30)
    bad = M.calibration_bins([_pred(0, 0.90, 0.0)])
    assert bad[0].calibration_error == pytest.approx(0.90)


@pytest.mark.unit
def test_market_baseline_brier_comparison():
    records = [_pred(0, 0.70, 1.0, market=0.5), _pred(1, 0.30, 0.0, market=0.5)]
    strat = [(r.predicted_probability, r.outcome) for r in records]
    market = [(r.market_probability, r.outcome) for r in records]
    assert M.brier_score(strat) < M.brier_score(market)
    assert M.brier_improvement(strat, market) > 0


@pytest.mark.unit
def test_wilson_interval_deterministic():
    lo, hi = M.wilson_interval(7, 10)
    assert 0.35 <= lo <= 0.40 and 0.85 <= hi <= 0.94
    assert M.wilson_interval(7, 10) == (lo, hi)
    assert M.wilson_interval(0, 0) == (0.0, 0.0)


@pytest.mark.unit
def test_bootstrap_deterministic_fixed_seed():
    values = [10.0, -5.0, 3.0, 7.0, -2.0, 4.0]
    assert M.bootstrap_mean_ci(values, resamples=200, seed=42) \
        == M.bootstrap_mean_ci(values, resamples=200, seed=42)


@pytest.mark.unit
def test_max_drawdown_and_profit_factor():
    curve = [2000, 2100, 1900, 2050]
    assert M.max_drawdown_bps(curve) == (2100 - 1900) * 10000 // 2100
    assert M.profit_factor([50, -25, 30, -10]) == pytest.approx(80 / 35)
    assert M.profit_factor([50, 30]) == float("inf")
    assert M.profit_factor([-50, -30]) == 0.0


@pytest.mark.unit
def test_profit_concentration_detection():
    assert M.profit_concentration([100, 2, 2, -50, -10])["concentrated"] is True
    assert M.profit_concentration([100, 2, 2, -50, -10])["top1_pct"] > 90
    assert M.profit_concentration([20, 20, 20, 20, 20, -10])["concentrated"] is False


@pytest.mark.unit
def test_sample_size_labels():
    assert sample_label(5) == SampleLabel.INSUFFICIENT_SAMPLE
    assert sample_label(50) == SampleLabel.VERY_LOW_CONFIDENCE
    assert sample_label(150) == SampleLabel.LOW_CONFIDENCE
    assert sample_label(500) == SampleLabel.MODERATE_EVIDENCE
    assert sample_label(2000) == SampleLabel.STRONGER_SAMPLE


@pytest.mark.unit
def test_edge_realization_and_latency():
    assert M.edge_realization_ratio(500, 250) == 0.5
    assert M.edge_realization_ratio(0, 100) == 0.0
    lat = M.latency_summary([100, 200, 300, 400, 500])
    assert lat["p50_ms"] == 300.0 and lat["p90_ms"] == 500.0


@pytest.mark.unit
def test_chronological_split_preserves_order():
    records = [_pred(i, 0.5, 1.0) for i in range(100)]
    train, valid, test = chronological_split(records)
    assert len(train) == 60 and len(valid) == 20 and len(test) == 20
    assert train[-1].timestamp_utc < valid[0].timestamp_utc
    assert valid[-1].timestamp_utc < test[0].timestamp_utc


@pytest.mark.unit
def test_walk_forward_no_leakage():
    records = [_pred(i, 0.5, 1.0) for i in range(100)]
    windows = walk_forward(records, n_blocks=4)
    assert len(windows) == 4
    for train, test in windows:
        assert train[-1].timestamp_utc < test[0].timestamp_utc
        assert all(t.timestamp_utc < test[0].timestamp_utc for t in train)
    assert windows[0][1][0].timestamp_utc < windows[1][1][0].timestamp_utc


@pytest.mark.unit
def test_future_observation_cannot_enter_earlier_evaluation():
    records = [_pred(i, 0.5, 1.0) for i in range(50)]
    train, _, _ = chronological_split(records)
    max_train_ts = max(r.timestamp_utc for r in train)
    assert len([r for r in records if r.timestamp_utc <= max_train_ts]) == len(train)
