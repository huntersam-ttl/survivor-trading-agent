from tradingagents.survivor.autonomy.cycle import CycleReport
from tradingagents.survivor.autonomy.state import RuntimeState


def test_budget_blocked_count_is_persisted(tmp_path):
    state = RuntimeState(db_path=str(tmp_path / "runtime.db"))
    state.start_cycle("cycle-1")
    state.finish_cycle(
        "cycle-1", "ACTIVE", "AI_BUDGET_UNAVAILABLE",
        {"discovered": 978, "passed": 39, "research_candidates": 3,
         "researched": 0, "budget_blocked_count": 3, "proposals": 0,
         "approved": 0, "rejected": 0, "executed": 0},
    )
    row = state.last_cycle()
    assert row["reason"] == "AI_BUDGET_UNAVAILABLE"
    assert row["budget_blocked_count"] == 3


def test_cycle_report_renders_budget_blocker_without_risk_claim():
    report = CycleReport(cycle_id="cycle-1", markets_discovered=978,
                         passed_filters=39, research_candidates=3,
                         budget_blocked_count=3, reason="AI_BUDGET_UNAVAILABLE")
    rendered = report.render()
    assert "Budget blocked: 3" in rendered
    assert "RISK_REJECTED" not in rendered
