from types import SimpleNamespace

import pytest

from tradingagents.survivor.acceptance import PaperAcceptanceStore
from tradingagents.survivor.jarvis_service import ApiError, JarvisApp, ServiceInjection, SurvivorHandlers, make_server


def _status(halted=True, opt_in=False):
    return {"mode": "PAPER ONLY", "broker_type": "paper", "live_broker_available": False,
            "live_order_transport_available": False, "paper_run_enabled": opt_in, "halted": halted,
            "service_heartbeat": {"heartbeat_type": "service_liveness", "present": True,
                "consistent": True, "age_seconds": 1, "halted": halted,
                "halt_state": "HALTED" if halted else "CLEAR"}}


def _handlers(tmp_path, cycle_fn, *, status=None, halt_path=None):
    return SurvivorHandlers(ServiceInjection(
        acceptance_store=PaperAcceptanceStore(tmp_path / "acceptance.db"),
        paper_ledger_path=str(tmp_path / "paper.db"),
        config_overrides={"_halt_path": str(halt_path or tmp_path / "HALT")},
        status_fn=lambda: status or _status(), cycle_fn=cycle_fn,
    ))


def _report():
    return SimpleNamespace(cycle_id="cycle-test", status="NO_TRADE", risk_decisions=0,
                           approved=0, rejected=0, paper_trades=0, ai_cost_pence=0)


def test_acceptance_routes_require_bearer_and_replay_status_is_read_only(tmp_path):
    handlers = _handlers(tmp_path, lambda **_kwargs: _report())
    server = make_server(port=0, token="local-test-token", inject=handlers.inject)
    try:
        import threading
        import httpx
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_address[1]}"
        denied = httpx.post(url + "/v1/acceptance/reserve", json={"acceptance_id": "operator-acceptance-201", "request": {"mode": "PAPER_RUNTIME"}})
        assert denied.status_code == 401
        headers = {"Authorization": "Bearer local-test-token"}
        created = httpx.post(url + "/v1/acceptance/reserve", headers=headers,
                             json={"acceptance_id": "operator-acceptance-201", "request": {"mode": "PAPER_RUNTIME"}})
        assert created.status_code == 200 and created.json()["created"] is True
        status = httpx.get(url + "/v1/acceptance/operator-acceptance-201", headers=headers)
        assert status.status_code == 200 and status.json()["status"] == "RESERVED"
    finally:
        server.shutdown(); server.server_close()


def test_scoped_permit_runs_once_restores_halt_and_never_changes_global_optin(tmp_path, monkeypatch):
    from tradingagents.survivor.pipeline import is_paper_enabled
    monkeypatch.delenv("TRADINGAGENTS_SURVIVOR_PAPER_ENABLED", raising=False)
    halt_path = tmp_path / "HALT"
    halt_path.write_text("HALT\n")
    calls = []
    def cycle(**kwargs):
        calls.append(kwargs)
        assert not halt_path.exists()
        assert kwargs["config"]["survivor_paper_enabled"] is True
        assert is_paper_enabled() is False
        return _report()
    handlers = _handlers(tmp_path, cycle, halt_path=halt_path)
    payload = {"acceptance_id": "operator-acceptance-202", "request": {"mode": "PAPER_RUNTIME"}}
    handlers.acceptance_reserve(payload)
    handlers.acceptance_authorize({"acceptance_id": payload["acceptance_id"], "operator_authorized": True,
                                   "operator_identity": "operator.test"})
    assert handlers.acceptance_status(payload["acceptance_id"])["result"]["operator_identity"] == "operator.test"
    result = handlers.acceptance_dispatch({"acceptance_id": payload["acceptance_id"]})
    assert result["status"] == "COMPLETED" and result["halt_restored"] is True
    assert result["result"]["operator_identity"] == "operator.test"
    assert halt_path.exists() and is_paper_enabled() is False and len(calls) == 1
    replay = handlers.acceptance_dispatch({"acceptance_id": payload["acceptance_id"]})
    assert replay["status"] == "COMPLETED" and len(calls) == 1


def test_ambiguous_cycle_and_restore_failure_never_replay(tmp_path, monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_SURVIVOR_PAPER_ENABLED", raising=False)
    halt_path = tmp_path / "HALT"
    halt_path.write_text("HALT\n")
    calls = []
    def cycle(**_kwargs):
        calls.append(1)
        raise TimeoutError("simulated response timeout")
    handlers = _handlers(tmp_path, cycle, halt_path=halt_path)
    payload = {"acceptance_id": "operator-acceptance-203", "request": {"mode": "PAPER_RUNTIME"}}
    handlers.acceptance_reserve(payload)
    handlers.acceptance_authorize({"acceptance_id": payload["acceptance_id"], "operator_authorized": True,
                                   "operator_identity": "operator.test"})
    with pytest.raises(ApiError) as exc:
        handlers.acceptance_dispatch({"acceptance_id": payload["acceptance_id"]})
    assert exc.value.code == "RECONCILIATION_REQUIRED"
    assert halt_path.exists()
    replay = handlers.acceptance_dispatch({"acceptance_id": payload["acceptance_id"]})
    assert replay["status"] == "RECONCILIATION_REQUIRED" and calls == [1]


def test_halt_restoration_failure_is_persisted_for_boot_recovery(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("block")
    impossible_halt = blocker / "HALT"
    calls = []
    handlers = _handlers(tmp_path, lambda **_kwargs: calls.append(1) or _report(),
                         status=_status(halted=True), halt_path=impossible_halt)
    acceptance_id = "operator-acceptance-204"
    handlers.acceptance_reserve({"acceptance_id": acceptance_id, "request": {"mode": "PAPER_RUNTIME"}})
    handlers.acceptance_authorize({"acceptance_id": acceptance_id, "operator_authorized": True,
                                   "operator_identity": "operator.test"})
    with pytest.raises(ApiError) as exc:
        handlers.acceptance_dispatch({"acceptance_id": acceptance_id})
    assert exc.value.code == "RECONCILIATION_REQUIRED"
    assert calls == []
    record = handlers._acceptance_store().get(acceptance_id)
    assert record.status == "RECONCILIATION_REQUIRED"
    assert record.result["halt_restored"] is False


def test_invalid_risk_result_fails_closed_and_cannot_be_replayed(tmp_path, monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_SURVIVOR_PAPER_ENABLED", raising=False)
    halt_path = tmp_path / "HALT"
    halt_path.write_text("HALT\n")
    calls = []
    invalid = SimpleNamespace(cycle_id="bad-risk", status="NO_TRADE", risk_decisions=-1,
        approved=0, rejected=0, paper_trades=0, ai_cost_pence=0)
    handlers = _handlers(tmp_path, lambda **_kwargs: calls.append(1) or invalid, halt_path=halt_path)
    acceptance_id = "operator-acceptance-205"
    handlers.acceptance_reserve({"acceptance_id": acceptance_id, "request": {"mode": "PAPER_RUNTIME"}})
    handlers.acceptance_authorize({"acceptance_id": acceptance_id, "operator_authorized": True,
                                   "operator_identity": "operator.test"})
    with pytest.raises(ApiError) as exc:
        handlers.acceptance_dispatch({"acceptance_id": acceptance_id})
    assert exc.value.code == "RECONCILIATION_REQUIRED"
    assert halt_path.exists() and calls == [1]
    assert handlers.acceptance_dispatch({"acceptance_id": acceptance_id})["status"] == "RECONCILIATION_REQUIRED"
    assert calls == [1]
