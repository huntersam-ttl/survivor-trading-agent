import multiprocessing

import pytest

from tradingagents.survivor.acceptance import AcceptanceConflict, PaperAcceptanceStore


def _reserve(path, barrier, output):
    store = PaperAcceptanceStore(path)
    barrier.wait(timeout=5)
    _record, created = store.reserve("operator-acceptance-001", {"symbol": "TEST", "mode": "PAPER"})
    output.put(created)


def _claim_dispatch(path, barrier, output):
    store = PaperAcceptanceStore(path)
    barrier.wait(timeout=5)
    try:
        store.mark_dispatched("operator-acceptance-race-001")
        output.put("claimed")
    except AcceptanceConflict:
        output.put("denied")


def test_independent_processes_atomically_reserve_same_acceptance(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    barrier, output = ctx.Barrier(2), ctx.Queue()
    path = str(tmp_path / "acceptance.db")
    processes = [ctx.Process(target=_reserve, args=(path, barrier, output)) for _ in range(2)]
    for proc in processes:
        proc.start()
    results = [output.get(timeout=10) for _ in processes]
    for proc in processes:
        proc.join(timeout=10)
        assert proc.exitcode == 0
    assert sorted(results) == [False, True]


def test_conflicting_fingerprint_and_terminal_replay_never_dispatch(tmp_path):
    store = PaperAcceptanceStore(tmp_path / "acceptance.db")
    first, created = store.reserve("operator-acceptance-002", {"mode": "PAPER", "symbol": "TEST"})
    assert created and first.status == "RESERVED"
    replay, created = store.reserve("operator-acceptance-002", {"symbol": "TEST", "mode": "PAPER"})
    assert not created and replay.status == "RESERVED"
    with pytest.raises(AcceptanceConflict):
        store.reserve("operator-acceptance-002", {"mode": "PAPER", "symbol": "OTHER"})
    store.transition(first.acceptance_id, {"RESERVED"}, "AUTHORIZED", {"operator": "confirmed"})
    store.transition(first.acceptance_id, {"AUTHORIZED"}, "DISPATCHED")
    recovered = store.recover_interrupted()
    assert recovered[0].status == "RECONCILIATION_REQUIRED"
    assert recovered[0].result["operator_action_required"] is True
    replay, created = store.reserve(first.acceptance_id, {"mode": "PAPER", "symbol": "TEST"})
    assert not created and replay.status == "RECONCILIATION_REQUIRED"
    with pytest.raises(AcceptanceConflict):
        store.transition(first.acceptance_id, {"RECONCILIATION_REQUIRED"}, "DISPATCHED")


def test_crash_before_dispatch_remains_reserved_and_requires_explicit_authorization(tmp_path):
    store = PaperAcceptanceStore(tmp_path / "acceptance.db")
    record, _ = store.reserve("operator-acceptance-003", {"mode": "PAPER"})
    assert store.recover_interrupted() == []
    assert store.get(record.acceptance_id).status == "RESERVED"


def test_two_processes_cannot_claim_the_dispatch_boundary_twice(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    path = str(tmp_path / "dispatch.db")
    store = PaperAcceptanceStore(path)
    store.reserve("operator-acceptance-race-001", {"mode": "PAPER"})
    store.authorize("operator-acceptance-race-001", ttl_seconds=60, operator_identity="operator.test")
    barrier, output = ctx.Barrier(2), ctx.Queue()
    processes = [ctx.Process(target=_claim_dispatch, args=(path, barrier, output)) for _ in range(2)]
    for proc in processes:
        proc.start()
    winners = [output.get(timeout=10) for _ in processes]
    for proc in processes:
        proc.join(timeout=10)
        assert proc.exitcode == 0
    assert sorted(winners) == ["claimed", "denied"]
    assert store.get("operator-acceptance-race-001").status == "DISPATCHED"
