"""Unit tests for ModelRouter role selection and fail-closed routing."""

import os
import tempfile
import time
from unittest.mock import MagicMock

import pytest

from tradingagents.llm_clients.budget import BudgetManager
from tradingagents.llm_clients.provider_health import ProviderHealthManager
from tradingagents.llm_clients.router import (
    ModelRouter,
    _invoke_with_deadline,
    _provider_response_from_wire,
    _provider_response_wire,
    _serializable_provider_specs,
)
from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger
from tradingagents.survivor.policy import SurvivorPolicy
from tradingagents.survivor.types import (
    AgentRole,
    ModelNotAllowedError,
    ProviderErrorCategory,
    UnknownPriceError,
    ProviderCallTimeout,
)


class _BlockingClient:
    def invoke(self, messages):
        time.sleep(10)


class _FastClient:
    def invoke(self, messages):
        return {"ok": True, "messages": messages}


class _MessageClient:
    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        return AIMessage(
            content="provider response",
            usage_metadata={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        )


@pytest.mark.unit
def test_provider_call_returns_normal_response_and_cleans_worker():
    result = _invoke_with_deadline(_FastClient(), "hello", 2)
    assert result == {"ok": True, "messages": "hello"}


@pytest.mark.unit
def test_provider_message_response_round_trips_through_worker():
    result = _invoke_with_deadline(_MessageClient(), "hello", 2)
    assert result.content == "provider response"
    assert result.usage_metadata["output_tokens"] == 4


@pytest.mark.unit
def test_provider_call_has_real_total_deadline():
    started = time.monotonic()
    with pytest.raises(ProviderCallTimeout):
        _invoke_with_deadline(_BlockingClient(), "hello", 0.2)
    assert time.monotonic() - started < 3


@pytest.mark.unit
def test_provider_timeout_settles_budget_reservation(test_router, monkeypatch):
    router = test_router

    def timeout(*args, **kwargs):
        raise ProviderCallTimeout("provider call exceeded 0.2s total deadline")

    monkeypatch.setattr("tradingagents.llm_clients.router._invoke_with_deadline", timeout)
    fake_client = MagicMock()
    fake_client.get_llm.return_value = _FastClient()
    monkeypatch.setattr("tradingagents.llm_clients.router.create_llm_client", lambda **kwargs: fake_client)
    router.role_routes["timeout_role"] = MagicMock(
        get_all_routes=MagicMock(return_value=[("minimax", "minimax-text-01")])
    )
    with pytest.raises(Exception, match="provider call exceeded"):
        router.invoke_role("timeout_role", "hello")
    assert router.budget_manager.get_pending_reserved_pence() == 0


@pytest.mark.unit
def test_tool_specs_are_spawn_safe_and_keep_callable_out_of_args():
    from tradingagents.agents.utils.news_data_tools import get_global_news

    specs = _serializable_provider_specs((("bind_tools", [get_global_news], {}),))
    assert specs[0][0] == "bind_tools"
    assert specs[0][1][0]["function"]["name"] == "get_global_news"
    assert "callable" not in repr(specs[0][1][0]).lower()
    # A JSON round-trip proves the child receives data, not a tool object.
    import json

    json.loads(json.dumps(specs[0][1]))


@pytest.mark.unit
def test_provider_tool_call_round_trips_to_parent():
    from langchain_core.messages import AIMessage

    response = AIMessage(
        content="",
        tool_calls=[
            {"name": "get_global_news", "args": {"curr_date": "2026-09-19"}, "id": "call-1"}
        ],
    )
    restored = _provider_response_from_wire(_provider_response_wire(response))
    assert restored.tool_calls[0]["name"] == "get_global_news"
    assert restored.tool_calls[0]["args"]["curr_date"] == "2026-09-19"


@pytest.mark.unit
def test_parent_executes_get_global_news_callable():
    from unittest.mock import patch

    from tradingagents.agents.utils.news_data_tools import get_global_news

    with patch(
        "tradingagents.agents.utils.news_data_tools.route_to_vendor",
        return_value="parent-result",
    ):
        assert get_global_news.invoke({"curr_date": "2026-09-19", "limit": 1}) == "parent-result"


@pytest.fixture
def test_router(monkeypatch):
    # hermetic: default role routes regardless of operator OpenRouter env
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("SURVIVOR_OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("SURVIVOR_OPENROUTER_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("SURVIVOR_OPENROUTER_MODELS", raising=False)
    monkeypatch.delenv("SURVIVOR_OPENROUTER_PRICING", raising=False)
    monkeypatch.delenv("SURVIVOR_OPENROUTER_PRICING_MAP", raising=False)
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
def test_router_healthy_fallback(test_router, monkeypatch):
    router = test_router
    # hermetic: default role routes, no OpenRouter override
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("SURVIVOR_OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("SURVIVOR_OPENROUTER_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("SURVIVOR_OPENROUTER_MODELS", raising=False)
    monkeypatch.delenv("SURVIVOR_OPENROUTER_PRICING_MAP", raising=False)

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
