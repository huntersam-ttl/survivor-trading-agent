"""Phase 5 operations tests: health, budget alerts, snapshots, analytics,
security self-check, corruption halt, daemon halt semantics, and a fast
deterministic 10,000-cycle synthetic soak test.

No network, no LLMs, no live trading. Everything runs against tmp_path.
"""

import datetime as _dt
import os
import sqlite3

import pytest

from tests.survivor.market_helpers import FakeMarketAdapter, make_snapshot
from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger
from tradingagents.survivor import ops
from tradingagents.survivor.autonomy import halt as halt_mod
from tradingagents.survivor.autonomy.cycle import run_survivor_cycle
from tradingagents.survivor.autonomy.state import RuntimeState
from tradingagents.survivor.evaluation.store import EvaluationStore
from tradingagents.survivor.execution.ledger import PaperLedger
from tradingagents.survivor.ops.attribution import (
    FAILURE_KEYS,
    cost_efficiency_metrics,
    equity_curve,
    failure_analytics,
    model_attribution,
)
from tradingagents.survivor.ops.budget import spend_alerts
from tradingagents.survivor.ops.health import preflight_check, watchdog_record, watchdog_status
from tradingagents.survivor.ops.selfcheck import security_selfcheck
from tradingagents.survivor.ops.snapshots import daily_snapshot, weekly_snapshot
from tradingagents.survivor.policy import SurvivorPolicy

DAY = "2026-09-04"
FIXED_NOW = _dt.datetime(2026, 9, 4, 12, 0, tzinfo=_dt.timezone.utc)


def _ledger(db_path) -> InferenceUsageLedger:
    return InferenceUsageLedger(db_path=str(db_path))


def _record(ledger, pence: int, provider="openai", model="gpt-test", role="trader"):
    ledger.record_inference(
        run_id="run_ops",
        agent_role=role,
        provider=provider,
        model=model,
        ticker_or_market="mkt-ops",
        input_tokens=100,
        output_tokens=10,
        reasoning_tokens=0,
        native_cost_minor=1,
        native_currency="USD",
        gbp_cost_pence=pence,
        timestamp_utc=f"{DAY}T10:00:00Z",
    )


# --- AI spend alerts -------------------------------------------------------------

def _alerts(pence: int, tmp_path) -> list[str]:
    ledger = _ledger(tmp_path / f"usage_{pence}.db")
    _record(ledger, pence)
    return spend_alerts(ledger, SurvivorPolicy())


@pytest.mark.unit
def test_spend_alert_threshold_50_pct(tmp_path):
    # 85p of the 170p daily budget == exactly 50%
    assert _alerts(85, tmp_path) == ["AI_SPEND_50_PCT: 85p / 170p daily budget"]


@pytest.mark.unit
def test_spend_alert_threshold_75_pct(tmp_path):
    # 128p -> 75.29% -> 50 and 75 fire; 90/100 do not
    alerts = _alerts(128, tmp_path)
    assert [a.split(":")[0] for a in alerts] == [
        "AI_SPEND_50_PCT", "AI_SPEND_75_PCT",
    ]


@pytest.mark.unit
def test_spend_alert_threshold_90_pct(tmp_path):
    # 153p -> exactly 90%
    assert [a.split(":")[0] for a in _alerts(153, tmp_path)] == [
        "AI_SPEND_50_PCT", "AI_SPEND_75_PCT", "AI_SPEND_90_PCT",
    ]


@pytest.mark.unit
def test_spend_alert_threshold_100_pct(tmp_path):
    # 170p -> exactly 100%: all four alerts fire
    assert [a.split(":")[0] for a in _alerts(170, tmp_path)] == [
        "AI_SPEND_50_PCT", "AI_SPEND_75_PCT", "AI_SPEND_90_PCT", "AI_SPEND_100_PCT",
    ]


@pytest.mark.unit
def test_100_pct_budget_blocks_ai(tmp_path):
    """At 100% of the daily budget the preflight blocks further AI calls."""
    from tradingagents.survivor.autonomy.budget import budget_preflight

    ledger = _ledger(tmp_path / "usage_full.db")
    _record(ledger, 170)
    pre = budget_preflight(SurvivorPolicy(), ledger)
    assert pre.ok is False
    assert "AI_BUDGET_UNAVAILABLE" in pre.reason
    # One penny of headroom still allows research
    ledger2 = _ledger(tmp_path / "usage_headroom.db")
    _record(ledger2, 169)
    assert budget_preflight(SurvivorPolicy(), ledger2).ok is True


# --- deterministic snapshots -----------------------------------------------------

@pytest.mark.unit
def test_daily_snapshot_deterministic(tmp_path):
    store = EvaluationStore(db_path=str(tmp_path / "evaluation.db"))
    kwargs = {
        "trial_id": "trial-ops", "evaluation_store": store,
        "cycles_scanned": 10, "markets_scanned": 20, "candidates_researched": 5,
        "paper_trades": 2, "paper_equity_pence": 2000, "daily_pnl_pence": -25,
        "drawdown_bps": 125, "ai_spend_pence": 85, "errors": 1, "halt_state": "CLEAR",
        "now": FIXED_NOW,
    }
    a = daily_snapshot(**kwargs)
    b = daily_snapshot(**kwargs)
    assert a == b
    assert a["day"] == DAY
    assert a["ai_spend_pence"] == 85 and a["halt_state"] == "CLEAR"


@pytest.mark.unit
def test_weekly_snapshot_deterministic(tmp_path):
    store = EvaluationStore(db_path=str(tmp_path / "evaluation.db"))
    kwargs = {
        "trial_id": "trial-ops", "evaluation_store": store, "paper_ledger_path": None,
        "now": FIXED_NOW,
    }
    a = weekly_snapshot(**kwargs)
    b = weekly_snapshot(**kwargs)
    assert a == b
    assert a["week_start"].startswith("2026-08-28T12:00")
    assert a["week_end"].startswith("2026-09-04T12:00")
    assert a["gate"]  # deterministic gate state present


@pytest.mark.unit
def test_snapshot_persist_is_immutable(tmp_path):
    store = EvaluationStore(db_path=str(tmp_path / "evaluation.db"))
    snap = daily_snapshot(trial_id="t", evaluation_store=store, now=FIXED_NOW)
    path = ops.persist_snapshot(snap, str(tmp_path / "snaps"))
    assert os.path.exists(path)
    # second persist must not overwrite (first write wins)
    mtime = os.path.getmtime(path)
    ops.persist_snapshot({"day": DAY, "tampered": True}, str(tmp_path / "snaps"))
    assert os.path.getmtime(path) == mtime
    with open(path) as fh:
        assert "tampered" not in fh.read()


# --- attribution / analytics ------------------------------------------------------

@pytest.mark.unit
def test_equity_curve_sorted_and_normalized():
    points = [
        {"timestamp_utc": "2026-09-04T10:00:00Z", "cash_pence": 1900.0,
         "exposure_pence": 100, "equity_pence": 2000},
        {"timestamp_utc": "2026-09-04T09:00:00Z", "cash_pence": 2000,
         "exposure_pence": 0, "equity_pence": 2000},
        {"timestamp_utc": "2026-09-04T11:00:00Z", "cash_pence": 1850,
         "exposure_pence": 100, "equity_pence": 1950,
         "drawdown_bps": 2500.9, "daily_pnl_pence": -50.5},
    ]
    curve = equity_curve(points)
    assert [p["timestamp_utc"] for p in curve] == [
        "2026-09-04T09:00:00Z", "2026-09-04T10:00:00Z", "2026-09-04T11:00:00Z",
    ]
    assert curve[0]["equity_pence"] == 2000 and isinstance(curve[0]["cash_pence"], int)
    assert curve[1]["exposure_pence"] == 100
    assert curve[2]["drawdown_bps"] == 2500 and curve[2]["daily_pnl_pence"] == -50
    # accounting identity holds on every normalized point
    assert all(p["cash_pence"] + p["exposure_pence"] == p["equity_pence"] for p in curve)


@pytest.mark.unit
def test_model_attribution(tmp_path):
    ledger = _ledger(tmp_path / "usage_attr.db")
    _record(ledger, 40, provider="deepseek", model="deepseek-chat", role="market_analyst")
    _record(ledger, 230, provider="openai", model="gpt-test", role="trader")
    _record(ledger, 30, provider="openai", model="gpt-test", role="trader")
    attribution = model_attribution(ledger)
    assert set(attribution) == {"deepseek/deepseek-chat", "openai/gpt-test"}
    assert attribution["deepseek/deepseek-chat"]["total_cost_pence"] == 40
    assert attribution["deepseek/deepseek-chat"]["calls"] == 1
    assert attribution["openai/gpt-test"]["total_cost_pence"] == 260
    assert attribution["openai/gpt-test"]["calls"] == 2
    assert attribution["openai/gpt-test"]["roles"] == {"trader": 260}


@pytest.mark.unit
def test_failure_analytics_percentages():
    counts = {"inference_failed": 3, "risk_rejected": 1, "unknown_key": 99}
    result = failure_analytics(counts)
    assert result["total"] == 4
    assert result["counts"]["inference_failed"] == 3
    assert "unknown_key" not in result["counts"]
    assert set(result["counts"]) == set(FAILURE_KEYS)
    assert result["percentages"]["inference_failed"] == 75.0
    assert result["percentages"]["risk_rejected"] == 25.0
    empty = failure_analytics({})
    assert empty["total"] == 0 and empty["percentages"]["inference_failed"] == 0.0


@pytest.mark.unit
def test_cost_efficiency_metrics():
    m = cost_efficiency_metrics(
        total_ai_cost_pence=200, cycles=10, candidates_researched=8,
        valid_proposals=4, approved_trades=2, resolved_profitable_trades=1,
        economic_pnl_pence=100,
    )
    assert m["ai_cost_per_cycle_pence"] == 20.0
    assert m["ai_cost_per_candidate_pence"] == 25.0
    assert m["ai_cost_per_proposal_pence"] == 50.0
    assert m["ai_cost_per_approved_trade_pence"] == 100.0
    assert m["ai_cost_per_resolved_profitable_trade_pence"] == 200.0
    assert m["economic_pnl_per_ai_cost"] == 0.5
    zero = cost_efficiency_metrics(
        total_ai_cost_pence=0, cycles=0, candidates_researched=0,
        valid_proposals=0, approved_trades=0, resolved_profitable_trades=0,
        economic_pnl_pence=0,
    )
    assert all(v == 0.0 for v in zero.values())


# --- security self-check (fail-closed) -------------------------------------------

@pytest.mark.unit
def test_security_selfcheck_fails_closed(tmp_path, monkeypatch):
    halt_path = str(tmp_path / "HALT")
    monkeypatch.setattr(halt_mod, "HALT_PATH", halt_path)
    monkeypatch.setattr(halt_mod, "HALT_DIR", str(tmp_path))

    clean = tmp_path / "clean_pkg"
    clean.mkdir()
    (clean / "innocent.py").write_text("x = 1\n")
    assert security_selfcheck(str(clean)) == {"ok": True, "hits": []}
    assert not halt_mod.is_halted(halt_path)

    tainted = tmp_path / "tainted_pkg"
    tainted.mkdir()
    (tainted / "rogue.py").write_text("import ccxt  # real exchange backend\n")
    (tainted / "wallet.py").write_text("private_key = 'x'\n")
    result = security_selfcheck(str(tainted))
    assert result["ok"] is False
    tokens = {h["token"] for h in result["hits"]}
    assert "ccxt" in tokens and "private_key" in tokens
    assert halt_mod.is_halted(halt_path)  # fail closed: runtime HALTed


# --- critical corruption HALT ----------------------------------------------------

@pytest.mark.unit
def test_critical_corruption_halts(tmp_path, monkeypatch):
    halt_path = str(tmp_path / "HALT")
    monkeypatch.setattr(halt_mod, "HALT_PATH", halt_path)

    # Corrupt the paper ledger hash chain (critical accounting corruption)
    db_path = str(tmp_path / "paper.db")
    ledger = PaperLedger(db_path=db_path)
    ledger.append_event(event_type="TREASURY_INITIALIZED", run_id="r", symbol="GBP",
                        cash_before=0, cash_after=2000, equity_before=0, equity_after=2000)
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE paper_events SET cash_after = 999999 WHERE rowid = 1")
    conn.commit()
    conn.close()

    # Preflight fails closed on the corrupted chain...
    report = preflight_check(survivor_dir=str(tmp_path), paper_ledger_path=db_path)
    assert report.ok is False
    assert any("paper_ledger_chain" in f for f in report.failures)
    # ...and the full cycle HALTs the runtime (kill switch engaged by the system)
    cycle = run_survivor_cycle(
        {"survivor_autonomy_enabled": True},
        adapter=FakeMarketAdapter([make_snapshot()]),
        paper_ledger_path=db_path,
        runtime_state=RuntimeState(db_path=str(tmp_path / "runtime.db")),
        usage_ledger=InferenceUsageLedger(db_path=str(tmp_path / "usage.db")),
        lock_path=str(tmp_path / "cycle.lock"),
    )
    assert cycle.status == "HALTED"
    assert "corruption" in (cycle.reason or "").lower()
    assert halt_mod.is_halted(halt_path)


# --- daemon never auto-resumes a HALT --------------------------------------------

@pytest.mark.unit
def test_daemon_does_not_auto_resume_halt(tmp_path, monkeypatch):
    """Consecutive daemon invocations must never clear the HALT file."""
    halt_path = str(tmp_path / "HALT")
    monkeypatch.setattr(halt_mod, "HALT_PATH", halt_path)
    halt_mod.set_halt(halt_path)

    def daemon_tick():
        # The daemon loop only invokes run_survivor_cycle(); it never clears halt.
        return run_survivor_cycle(
            {"survivor_autonomy_enabled": True},
            adapter=FakeMarketAdapter([make_snapshot()]),
            paper_ledger_path=str(tmp_path / "paper.db"),
        )

    for _ in range(3):
        result = daemon_tick()
        assert result.status == "EXTERNAL_HALT"
        assert halt_mod.is_halted(halt_path)

    # Only explicit manual resume clears the halt.
    halt_mod.clear_halt(halt_path)
    assert not halt_mod.is_halted(halt_path)


# --- watchdog + heartbeat --------------------------------------------------------

@pytest.mark.unit
def test_watchdog_halts_after_threshold(tmp_path, monkeypatch):
    halt_path = str(tmp_path / "HALT")
    monkeypatch.setattr(halt_mod, "HALT_PATH", halt_path)
    state = RuntimeState(db_path=str(tmp_path / "runtime.db"))
    for i in range(4):
        value = watchdog_record(state, "consecutive_cycle_failures", success=False)
        assert value == i + 1
    assert not halt_mod.is_halted(halt_path)
    assert watchdog_record(state, "consecutive_cycle_failures", success=False) == 5
    assert halt_mod.is_halted(halt_path)  # threshold reached -> kill switch
    watchdog_record(state, "consecutive_cycle_failures", success=True)
    assert watchdog_status(state)["consecutive_cycle_failures"] == 0


@pytest.mark.unit
def test_heartbeat_written_atomically(tmp_path):
    import json

    path = str(tmp_path / "hb" / "heartbeat.json")
    written = ops.heartbeat_write(
        trial_id="t1", pid=12345, mode="PAPER",
        last_cycle_id="abc", last_cycle_status="ACTIVE",
        halt_state="CLEAR", heartbeat_path=path,
    )
    assert written == path
    with open(path) as fh:
        data = json.load(fh)
    assert data["trial_id"] == "t1" and data["pid"] == 12345
    assert data["mode"] == "PAPER" and data["halt_state"] == "CLEAR"
    leftovers = [f for f in os.listdir(os.path.dirname(path)) if f.startswith(".heartbeat")]
    assert leftovers == []  # atomic rename, no tmp litter


# --- 10,000-cycle deterministic synthetic soak -----------------------------------

@pytest.mark.unit
def test_soak_ten_thousand_synthetic_cycles(tmp_path, monkeypatch):
    """Fast deterministic soak: 10,000 synthetic cycles through the pure ops
    pipeline (alerts, analytics, efficiency, equity, watchdog). No sleeps,
    no randomness, no network, no AI."""
    halt_path = str(tmp_path / "HALT")
    monkeypatch.setattr(halt_mod, "HALT_PATH", halt_path)
    state = RuntimeState(db_path=str(tmp_path / "runtime.db"))
    daily_limit = SurvivorPolicy().global_daily_pence  # 170p

    alerts_seen = set()
    halt_engaged_at = None
    for i in range(10_000):
        # deterministic synthetic spend rising over the trial
        spend = (i * 7) % (daily_limit + 35)

        class _FakeLedger:
            def get_daily_spend_pence(self, provider=None, day=None, _spend=spend):
                return _spend

        for a in spend_alerts(_FakeLedger(), SurvivorPolicy()):
            alerts_seen.add(a.split(":")[0])

        failures = failure_analytics({
            "market_fetch": i % 3, "inference_failed": (i * 2) % 5,
            "risk_rejected": i % 7, "scanner_reject": (i // 11) % 2,
        })
        assert failures["total"] == (i % 3) + ((i * 2) % 5) + (i % 7) + ((i // 11) % 2)
        assert all(0.0 <= p <= 100.0 for p in failures["percentages"].values())

        cost = cost_efficiency_metrics(
            total_ai_cost_pence=i % 500, cycles=i + 1, candidates_researched=i + 2,
            valid_proposals=i + 3, approved_trades=i + 4,
            resolved_profitable_trades=i + 5, economic_pnl_pence=i % 90,
        )
        assert cost["ai_cost_per_cycle_pence"] == round((i % 500) / (i + 1), 4)

        cash, exposure = 2000 - (i % 100), i % 150
        curve = equity_curve([{
            "timestamp_utc": f"2026-09-04T{(i % 24):02d}:00:00Z",
            "cash_pence": cash, "exposure_pence": exposure,
            "equity_pence": cash + exposure,
        }])
        assert curve[0]["equity_pence"] == cash + exposure  # identity invariant

        # watchdog sampled every 10th cycle, all failures: consecutive counter
        # reaches the threshold of 5 at i=40 (failures at 0,10,20,30,40)
        if i % 10 == 0:
            value = watchdog_record(
                state, "consecutive_failures", success=False, halt_threshold=5,
            )
            if value >= 5 and halt_engaged_at is None:
                halt_engaged_at = i

    # alerts only ever at the four deterministic thresholds
    assert alerts_seen <= {
        "AI_SPEND_50_PCT", "AI_SPEND_75_PCT", "AI_SPEND_90_PCT", "AI_SPEND_100_PCT",
    }
    # HALT engaged exactly once (i=40: failures at 0,10,20,30,40) and stayed.
    assert halt_engaged_at == 40
    assert halt_mod.is_halted(halt_path)
    # Counter continues incrementing past halt threshold (1000 total calls,
    # every 10th of 10,000 iterations, all with success=False).
    assert state.get_watchdog_all()["consecutive_failures"] == 1000
