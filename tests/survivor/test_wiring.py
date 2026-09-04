"""Wiring tests proving agents route through the Survivor control plane.

- SURVIVOR disabled  -> raw upstream LLM objects, no router, upstream path intact.
- SURVIVOR enabled   -> every agent call goes: ModelRouter -> health check ->
  allowlist/price check -> BudgetManager reservation -> provider client ->
  settlement, with run_id/ticker context recorded in the ledger.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.setup import GraphSetup
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.budget import BudgetManager
from tradingagents.llm_clients.provider_health import ProviderHealthManager
from tradingagents.llm_clients.router import ModelRouter as RealModelRouter
from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger
from tradingagents.survivor.guard import (
    SurvivorLLM,
    get_survivor_run,
    reset_survivor_run,
    set_survivor_run,
)
from tradingagents.survivor.policy import SurvivorPolicy
from tradingagents.survivor.types import AgentRole


def _make_router(tmpdir):
    ledger = InferenceUsageLedger(db_path=os.path.join(tmpdir, "wiring.db"))
    policy = SurvivorPolicy()
    return RealModelRouter(
        policy=policy,
        budget_manager=BudgetManager(policy=policy, ledger=ledger),
        health_manager=ProviderHealthManager(),
    )


def _fake_client_factory(provider, model, **kwargs):
    """Fake provider client: records constructor args, returns canned response."""
    llm = MagicMock()
    llm._survivor_provider = provider
    llm._survivor_model = model
    response = MagicMock()
    response.content = "routed response"
    response.usage_metadata = {"input_tokens": 100, "output_tokens": 40}
    llm.invoke.return_value = response
    return llm


@pytest.mark.unit
def test_guard_disabled_returns_base_llm():
    """Without a survivor router, GraphSetup must hand agents the raw upstream LLM."""
    base_llm = MagicMock()
    setup = GraphSetup(base_llm, MagicMock(), {}, MagicMock(), survivor_router=None)
    assert setup._role_llm(base_llm, "market_analyst") is base_llm


@pytest.mark.unit
def test_guard_enabled_wraps_every_role():
    """With a survivor router, each role gets a SurvivorLLM proxy tagged with that role."""
    with tempfile.TemporaryDirectory() as tmpdir:
        router = _make_router(tmpdir)
        setup = GraphSetup(MagicMock(), MagicMock(), {}, MagicMock(), survivor_router=router)
        for role in AgentRole:
            proxy = setup._role_llm(MagicMock(), role.value)
            assert isinstance(proxy, SurvivorLLM)
            assert proxy.role == role.value


@pytest.mark.unit
def test_survivor_llm_invoke_routes_and_settles():
    """invoke() must reserve before the call and settle actuals into the ledger after."""
    with tempfile.TemporaryDirectory() as tmpdir:
        router = _make_router(tmpdir)
        llm = SurvivorLLM(router, role="market_analyst")
        tokens = set_survivor_run("run_w1", "NVDA")
        try:
            response = llm.invoke("Analyze NVDA", mock_client_factory=_fake_client_factory)
        finally:
            reset_survivor_run(tokens)

        assert response.content == "routed response"
        summary = router.budget_manager.ledger.get_run_usage_summary("run_w1")
        assert summary["total_calls"] == 1
        entry = summary["breakdown"][0]
        assert entry["role"] == "market_analyst"
        assert entry["provider"] in ("deepseek", "minimax")
        # Reservation must be released after settlement
        assert router.budget_manager.get_pending_reserved_pence() == 0


@pytest.mark.unit
def test_survivor_llm_bind_tools_transform_applied_to_routed_client():
    """bind_tools must apply to whichever provider client the router selects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        router = _make_router(tmpdir)
        llm = SurvivorLLM(router, role="news_analyst").bind_tools(["tool_a", "tool_b"])

        base = MagicMock()
        response = MagicMock()
        response.content = "ok"
        response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
        base.invoke.return_value = response
        base.bind_tools.return_value = base

        captured = {}

        def factory(provider, model, **kwargs):
            captured["provider"] = provider
            return base

        llm.invoke("News scan", mock_client_factory=factory)
        base.bind_tools.assert_called_once_with(["tool_a", "tool_b"])
        assert captured["provider"] in ("deepseek", "minimax")


@pytest.mark.unit
def test_survivor_llm_with_structured_output_transform_applied():
    with tempfile.TemporaryDirectory() as tmpdir:
        router = _make_router(tmpdir)
        schema = MagicMock(name="DecisionSchema")
        llm = SurvivorLLM(router, role="portfolio_manager").with_structured_output(schema)

        base = MagicMock()
        response = MagicMock()
        response.content = "decision"
        response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
        base.invoke.return_value = response

        base.with_structured_output.return_value = base

        def factory(provider, model, **kwargs):
            return base

        llm.invoke("Final decision", mock_client_factory=factory)
        base.with_structured_output.assert_called_once_with(schema)


@pytest.mark.unit
def test_run_context_defaults():
    assert get_survivor_run() == ("default_run", "GLOBAL")


@pytest.mark.unit
def test_survivor_disabled_preserves_upstream_graph_path(mock_llm_client):
    """survivor_enabled=False must keep the exact upstream wiring: no router, raw LLMs."""
    config = {**DEFAULT_CONFIG, "survivor_enabled": False}
    with patch("tradingagents.graph.trading_graph.create_llm_client", mock_llm_client):
        graph = TradingAgentsGraph(config=config)
    assert graph.survivor_router is None
    # Upstream LLM objects come straight from the factory, not through guard proxies
    assert not isinstance(graph.quick_thinking_llm, SurvivorLLM)
    assert not isinstance(graph.deep_thinking_llm, SurvivorLLM)
    assert graph.quick_thinking_llm is mock_llm_client.return_value.get_llm.return_value


@pytest.mark.unit
def test_survivor_enabled_routes_through_control_plane(mock_llm_client):
    """survivor_enabled=True must wire the router and wrap all agent LLM lanes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = InferenceUsageLedger(db_path=os.path.join(tmpdir, "graph.db"))

        def router_factory(p, **kwargs):
            return RealModelRouter(
                policy=p,
                budget_manager=BudgetManager(policy=p, ledger=ledger),
                health_manager=ProviderHealthManager(),
            )

        config = {**DEFAULT_CONFIG, "survivor_enabled": True}
        with patch("tradingagents.llm_clients.router.ModelRouter", router_factory):
            graph = TradingAgentsGraph(config=config)

        assert graph.survivor_router is not None
        assert isinstance(graph.quick_thinking_llm, SurvivorLLM)
        assert isinstance(graph.deep_thinking_llm, SurvivorLLM)
        # Graph nodes are role-tagged proxies for all 13 agent roles
        for role in AgentRole:
            proxy = graph.graph_setup._role_llm(MagicMock(), role.value)
            assert isinstance(proxy, SurvivorLLM)
            assert proxy.role == role.value


@pytest.mark.unit
def test_survivor_enabled_via_env_var(mock_llm_client, monkeypatch):
    """TRADINGAGENTS_SURVIVOR_ENABLED=true must enable the control plane."""
    monkeypatch.setenv("TRADINGAGENTS_SURVIVOR_ENABLED", "true")
    config = {**DEFAULT_CONFIG, "survivor_enabled": False}
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = InferenceUsageLedger(db_path=os.path.join(tmpdir, "env.db"))

        def router_factory(p, **kwargs):
            return RealModelRouter(
                policy=p, budget_manager=BudgetManager(policy=p, ledger=ledger)
            )

        with patch("tradingagents.llm_clients.router.ModelRouter", router_factory):
            graph = TradingAgentsGraph(config=config)
        assert graph.survivor_router is not None

