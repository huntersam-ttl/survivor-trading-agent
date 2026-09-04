"""Fail-closed behaviour tests: FX, rollovers, provider health, allowlist containment."""

import os
import tempfile
from decimal import Decimal

import pytest

from tradingagents.llm_clients.budget import BudgetManager
from tradingagents.llm_clients.pricing import MODEL_PRICING, calculate_cost
from tradingagents.llm_clients.provider_health import ProviderHealthManager
from tradingagents.llm_clients.router import DEFAULT_ROLE_ROUTES, ModelRouter
from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger
from tradingagents.survivor.policy import SurvivorPolicy
from tradingagents.survivor.types import (
    BudgetExhaustedError,
    ProviderErrorCategory,
    SurvivorError,
    UnknownPriceError,
)

ALLOWED_PROVIDERS = {"openai", "deepseek", "minimax"}


@pytest.fixture
def bm():
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = InferenceUsageLedger(db_path=os.path.join(tmpdir, "fc.db"))
        policy = SurvivorPolicy()
        yield BudgetManager(policy=policy, ledger=ledger)


@pytest.mark.unit
def test_missing_fx_rate_fails_closed():
    """A missing (None) or invalid (<= 0) USD->GBP FX rate must block paid calls."""
    for bad_rate in (None, Decimal("0"), Decimal("-1.30")):
        with pytest.raises(UnknownPriceError):
            calculate_cost(
                provider="openai",
                model="gpt-5.6",
                input_tokens=1000,
                output_tokens=100,
                usd_gbp_rate=bad_rate,
            )


@pytest.mark.unit
def test_unknown_model_price_fails_closed():
    with pytest.raises(UnknownPriceError):
        calculate_cost(
            provider="openai", model="totally-unknown-model", input_tokens=10, output_tokens=10
        )


@pytest.mark.unit
def test_monthly_utc_rollover(bm):
    """Exhausting a month's provider budget must not block the next UTC month."""
    ts_sep = "2026-09-28T23:59:00Z"
    ts_oct = "2026-10-01T00:01:00Z"
    bm.ledger.record_inference(
        run_id="r1",
        agent_role="trader",
        provider="openai",
        model="gpt-5.6",
        ticker_or_market="AAPL",
        input_tokens=100,
        output_tokens=100,
        reasoning_tokens=0,
        native_cost_minor=100,
        native_currency="USD",
        gbp_cost_pence=3000,  # == full openai monthly limit
        timestamp_utc=ts_sep,
    )
    with pytest.raises(BudgetExhaustedError):
        bm.authorize_and_reserve(
            provider="openai",
            model="gpt-4o-mini",
            agent_role="quick_analyst",
            est_input_tokens=100,
            est_output_tokens=100,
            timestamp_utc=ts_sep,
        )
    # New UTC month -> fresh allowance
    handle = bm.authorize_and_reserve(
        provider="openai",
        model="gpt-4o-mini",
        agent_role="quick_analyst",
        est_input_tokens=100,
        est_output_tokens=100,
        timestamp_utc=ts_oct,
    )
    assert handle is not None
    bm.release_reservation(handle)


@pytest.mark.unit
def test_unhealthy_provider_rejected_by_router(bm):
    """A provider in cooldown must never receive an inference call."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger = InferenceUsageLedger(db_path=os.path.join(tmpdir, "uh.db"))
        policy = SurvivorPolicy()
        router = ModelRouter(
            policy=policy,
            budget_manager=BudgetManager(policy=policy, ledger=ledger),
            health_manager=ProviderHealthManager(),
        )
        # Both deepseek and minimax (all fallbacks for market_analyst) down
        router.health_manager.record_failure("deepseek", ProviderErrorCategory.RATE_LIMIT)
        router.health_manager.record_failure("minimax", ProviderErrorCategory.RATE_LIMIT)

        with pytest.raises(SurvivorError):
            router.invoke_role("market_analyst", "Analyze NVDA", mock_client_factory=lambda **kw: object())

        # Nothing was recorded and nothing is reserved
        assert ledger.get_run_usage_summary("default_run")["total_calls"] == 0
        assert router.budget_manager.get_pending_reserved_pence() == 0


@pytest.mark.unit
def test_role_fallback_stays_within_allowlist():
    """Every configured route (preferred AND fallback) must be an allowlisted,
    explicitly-priced model so budgets can never be bypassed via fallback."""
    for role_value, route in DEFAULT_ROLE_ROUTES.items():
        for provider, model in route.get_all_routes():
            assert provider in ALLOWED_PROVIDERS, f"{role_value}: {provider}/{model}"
            assert (provider, model) in MODEL_PRICING, f"{role_value}: {provider}/{model} unpriced"


@pytest.mark.unit
def test_role_routes_cover_all_agent_roles():
    """All 13 agent roles must have a configured route (no role escapes routing)."""
    from tradingagents.survivor.types import AgentRole

    for role in AgentRole:
        assert role.value in DEFAULT_ROLE_ROUTES
