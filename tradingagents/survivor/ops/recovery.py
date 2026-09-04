"""Crash recovery: mark interrupted cycles, never duplicate executions."""

from __future__ import annotations


def recover_interrupted_cycles(runtime_state, paper_ledger=None) -> dict:
    """Find cycles stuck in RUNNING, inspect whether execution already occurred,
    and mark them CYCLE_INTERRUPTED. Fail closed (raise) on ambiguous state.

    Replay protection (proposal_id uniqueness in paper.db) stays active, so a
    re-run cannot duplicate trades; ambiguous ledgers raise instead.
    """
    recovered = {"interrupted": [], "ambiguous": []}
    with runtime_state._connect() as conn:
        rows = conn.execute(
            "SELECT cycle_id, started_utc FROM cycles WHERE status = 'RUNNING'"
        ).fetchall()
    for row in rows:
        cycle_id = row["cycle_id"]
        # check whether any paper ledger execution references this cycle
        executed = 0
        if paper_ledger is not None:
            events = paper_ledger.events()
            executed = len([e for e in events
                            if e["event_type"] == "TRADE_EXECUTED"
                            and e["run_id"].startswith(cycle_id)])
        if executed > 1:
            # ambiguous: multiple executions from one cycle must never happen
            recovered["ambiguous"].append(cycle_id)
            continue
        runtime_state.finish_cycle(
            cycle_id, "CYCLE_INTERRUPTED",
            f"process died mid-cycle; executions={executed} (replay protection active)",
        )
        recovered["interrupted"].append(cycle_id)
    if recovered["ambiguous"]:
        from tradingagents.survivor.autonomy.halt import set_halt

        set_halt()  # fail closed: ambiguous accounting halts the runtime
        raise RuntimeError(f"ambiguous recovery for cycles: {recovered['ambiguous']}")
    return recovered
