"""Jarvis localhost service tests: read-only state, dry/paper run, halt, safety.

Hermetic: a fake market adapter + injected research fn mean NO real OpenRouter
calls and NO writes to the real ~/.tradingagents databases.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from tests.survivor.market_helpers import FakeMarketAdapter, make_snapshot
from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger
from tradingagents.survivor.autonomy import halt as halt_mod
from tradingagents.survivor.autonomy.cycle import ResearchResult
from tradingagents.survivor.autonomy.state import RuntimeState
from tradingagents.survivor.execution.ledger import PaperLedger
from tradingagents.survivor.execution.paper_broker import PaperBroker
from tradingagents.survivor.jarvis_service import (
    BIND_HOST,
    ServiceInjection,
    make_server,
)
from tradingagents.survivor.policy import SurvivorPolicy
from tradingagents.survivor.risk.limits import RiskLimits

GOOD_EDGE = {
    'expected_probability_bps': 7000,
    'quantity': 1, 'spread_cost_bps': 50, 'slippage_bps': 50, 'fee_bps': 50,
    'uncertainty_penalty_bps': 100,
}

TOKEN = "test-jarvis-token"


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Build the hermetic uses and a helper to launch servers on ephemeral ports."""
    monkeypatch.setattr(halt_mod, "HALT_PATH", str(tmp_path / "HALT"))
    monkeypatch.setenv("JARVIS_INTELLIGENCE_ENABLED", "false")
    monkeypatch.delenv("SURVIVOR_DRY_RUN", raising=False)
    monkeypatch.delenv("TRADINGAGENTS_SURVIVOR_ENABLED", raising=False)
    monkeypatch.delenv("SURVIVOR_JARVIS_TOKEN", raising=False)

    research_calls = {"n": 0}

    def research(candidate, quote, config, run_id):
        research_calls["n"] += 1
        return ResearchResult(status="OK", decision_text="BUY", **GOOD_EDGE)

    adapter = FakeMarketAdapter([make_snapshot()])
    limits = RiskLimits()
    paper_path = str(tmp_path / "paper.db")
    broker = PaperBroker(limits=limits, ledger=PaperLedger(db_path=paper_path))
    eval_path = str(tmp_path / "evaluation.db")
    inject = ServiceInjection(
        adapter=adapter,
        research_fn=research,
        paper_ledger_path=paper_path,
        runtime_state=RuntimeState(db_path=str(tmp_path / "runtime.db")),
        usage_ledger=InferenceUsageLedger(db_path=str(tmp_path / "usage.db")),
        policy=SurvivorPolicy(),
        settings=limits,
        broker=broker,
        evaluation_db_path=eval_path,
        lock_path=str(tmp_path / "cycle.lock"),
        config_overrides={
            "jarvis_intelligence_enabled": False,
            "_heartbeat_path": str(tmp_path / "heartbeat.json"),
            "_evaluation_db_path": eval_path,
        },
    )

    def request(server, method, path, token=TOKEN):
        url = f"http://127.0.0.1:{server.server_address[1]}{path}"
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        req = urllib.request.Request(url, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8")
            return exc.code, json.loads(body) if body else {}

    server = None

    def start(token=TOKEN):
        nonlocal server
        server = make_server(port=0, token=token, inject=inject)
        if server.server_address[0] != BIND_HOST:
            raise AssertionError(f"service did not bind to {BIND_HOST}")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    yield start, request, inject, adapter, research_calls
    if server is not None:
        server.shutdown()
        server.server_close()


def test_health_status_budget_performance_positions(harness):
    start, request, *_ = harness
    server = start()

    # health
    status, health = request(server, "GET", "/v1/health")
    assert status == 200
    assert health["status"] == "ok"
    assert health["service"] == "survivor-jarvis"

    # status
    status, body = request(server, "GET", "/v1/status")
    assert status == 200
    assert body["mode"] == "PAPER ONLY"
    assert body["halted"] in (True, False)
    assert "paper" in body and "ai_spend" in body and "cycle_count" in body

    # budget
    status, body = request(server, "GET", "/v1/budget")
    assert status == 200
    assert "daily" in body and "monthly" in body

    # performance
    status, body = request(server, "GET", "/v1/performance")
    assert status == 200
    assert isinstance(body, dict)

    # positions (fresh broker => none open)
    status, body = request(server, "GET", "/v1/positions")
    assert status == 200
def test_dry_run_blocks_execution(harness):
    start, request, _, adapter, research_calls = harness
    server = start()

    status, report = request(server, "POST", "/v1/run/dry")
    assert status == 200
    assert report["status"] == "DRY_RUN"
    assert report["paper_trades"] == 0
    assert report["ai_researched"] >= 1
    assert research_calls["n"] >= 1
    # the fake adapter (not any network) was used
    assert adapter.list_calls > 0


def test_paper_run_executes_paper_trade(harness):
    start, request, _, adapter, research_calls = harness
    server = start()

    status, report = request(server, "POST", "/v1/run/paper")
    assert status == 200
    assert report["status"] == "ACTIVE"
    assert report["approved"] >= 1
    assert report["paper_trades"] >= 1
    assert research_calls["n"] >= 1
    assert adapter.snapshot_calls >= 0


def test_halt_sets_halt_and_blocks_runs(harness, tmp_path):
    start, request, _, _, _ = harness
    server = start()

    status, body = request(server, "POST", "/v1/halt")
    assert status == 200
    assert body["halted"] is True
    # HALT file physically present
    assert (tmp_path / "HALT").exists()
    # HALT cannot auto-clear: run/paper is refused with EXTERNAL_HALT
    status, report = request(server, "POST", "/v1/run/paper")
    assert status == 200
    assert report["status"] == "EXTERNAL_HALT"
    assert report["paper_trades"] == 0


def test_auth_failure(harness):
    start, request, *_ = harness
    server = start()

    # missing token
    status, body = request(server, "GET", "/v1/health", token=None)
    assert status == 401
    assert "error" in body
    # wrong token
    status, body = request(server, "GET", "/v1/health", token="wrong-token")
    assert status == 401
    # no endpoint is reachable without auth
    status, _ = request(server, "POST", "/v1/run/paper", token="nope")
    assert status == 401


def test_missing_token_config_fails_closed(harness):
    start, request, *_ = harness
    server = start(token="")  # no configured token
    # even health is refused until a token is configured
    status, body = request(server, "GET", "/v1/health", token=None)
    assert status == 401


def test_localhost_only_binding(harness):
    start, request, inject, *_ = harness
    server = start()
    assert BIND_HOST == "127.0.0.1"
    # bound address is loopback
    assert server.server_address[0] == "127.0.0.1"


def test_live_trade_endpoint_absent(harness):
    start, request, *_ = harness
    server = start()
    for path in ("/v1/run/live", "/v1/trade", "/v1/execute", "/v1/order"):
        status, body = request(server, "POST", path)
        assert status == 404, path
        assert "error" in body
        status, _ = request(server, "GET", path)
        assert status == 404, path


def test_risk_budget_mutation_absent(harness):
    start, request, inject, *_ = harness
    server = start()

    before = request(server, "GET", "/v1/budget")[1]
    for path in ("/v1/budget", "/v1/risk", "/v1/risk/limit", "/v1/positions", "/v1/wallet"):
        status, body = request(server, "POST", path)
        assert status == 404, path
        assert "error" in body

    # budget is read-only: GET still works and content is unchanged
    after = request(server, "GET", "/v1/budget")[1]
    assert after == before
    # no live-trading or budget-mutation handler was wired at all
    assert "run_live" not in inject.__dict__ and "set_budget" not in inject.__dict__