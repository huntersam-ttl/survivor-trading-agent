"""Unit tests for ModelRouter role selection and fail-closed routing."""

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from tradingagents.llm_clients.budget import BudgetManager
from tradingagents.llm_clients.provider_health import ProviderHealthManager
from tradingagents.llm_clients.router import ModelRouter
from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger
from tradingagents.survivor.policy import SurvivorPolicy
from tradingagents.survivor.types import (
    AgentRole,
    ModelNotAllowedError,
    ProviderErrorCategory,
    UnknownPriceError,
)


@pytest.fixture
def test_router():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_router.db")
        ledger = InferenceUsageLedger(db_path=db_path)
        policy = SurvivorPolicy()
        bm = BudgetManager(policy=policy, ledger=ledger)
        hm = ProviderHealthManager()
        router = ModelRouter(policy=policy, budget_manager=bm, health_manager=hm)
        yield router


@pytest.mark.unit
def test_router_role_route_resolution(test_router):
    routes = test_router.resolve_routes(AgentRole.BULL_RESEARCHER.value)
    assert ("deepseek", "deepseek-chat") in routes

    routes_mgr = test_router.resolve_routes(AgentRole.RESEARCH_MANAGER.value)
    assert ("openai", "gpt-5.6") in routes_mgr

    with pytest.raises(ModelNotAllowedError):
        test_router.resolve_routes("unauthorized_hacker_role")


@pytest.mark.unit
def test_router_healthy_fallback(test_router):
    router = test_router

    # Mark DeepSeek as unhealthy (rate limit cooldown)
    router.health_manager.record_failure("deepseek", ProviderErrorCategory.RATE_LIMIT)

    # Mock fake client factory
    def mock_factory(provider, model, **kwargs):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Analysis report content"
        mock_response.usage_metadata = {"input_tokens": 100, "output_tokens": 50}
        mock_llm.invoke.return_value = mock_response
        return mock_llm

    # For MARKET_ANALYST, preferred is DeepSeek (unhealthy) -> should fall back to MiniMax!
    resp = router.invoke_role(
        role=AgentRole.MARKET_ANALYST.value,
        messages="Analyze NVDA",
        run_id="test_run_1",
        ticker_or_market="NVDA",
        mock_client_factory=mock_factory,
    )
    assert resp.content == "Analysis report content"

    # Verify ledger recorded MiniMax (the fallback)
    summary = router.budget_manager.ledger.get_run_usage_summary("test_run_1")
    assert summary["total_calls"] == 1
    assert summary["breakdown"][0]["provider"] == "minimax"


@pytest.mark.unit
def test_router_unpriced_model_rejection(test_router):
    router = test_router
    # Override route to an unpriced fake model
    router.role_routes["custom_role"] = MagicMock(
        get_all_routes=MagicMock(return_value=[("openai", "unpriced-fake-gpt-99")])
    )

    with pytest.raises(UnknownPriceError):
        router.invoke_role("custom_role", "Hello")
