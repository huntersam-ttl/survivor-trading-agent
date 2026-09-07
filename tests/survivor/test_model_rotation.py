"""Tests for dynamic OpenRouter model rotation and structured-output repair.

SURVIVOR keeps model choices behind
    SurvivorLLM -> ModelRouter -> CostGovernor -> BudgetManager -> OpenRouter
and never selects an unpriced model (UNKNOWN_MODEL_PRICE). These tests prove:

- ModelRouter rotates among CONFIGURED, PRICED OpenRouter models (cheapest-first
  by default, considering cost/health/reliability/latency/history/budget).
- DeepSeek V4 Flash 0731 can be configured as the preferred default without
  making it mandatory; any configured model rotates and pricing is explicit.
- Unpriced models are never called (fails closed with UnknownPriceError).
- OpenAI/DeepSeek/MiniMax fallbacks are dropped when their API keys are absent.
- Existing daily/monthly budgets are never increased for OpenRouter.
- Structured-output failures retry the same approved model once, then rotate to
  another configured/priced OpenRouter model — never a non-OpenRouter fallback.

No real paid API calls; inference is faked via mock_client_factory.
"""

import tempfile
from decimal import Decimal

import pytest

from tradingagents.llm_clients.budget import BudgetManager
from tradingagents.llm_clients.pricing import (
    MODEL_PRICING,
    ModelPrice,
    get_model_price,
    register_model_price,
)
from tradingagents.llm_clients.provider_health import ProviderHealthManager
from tradingagents.llm_clients.router import ModelRouter, _role_routes_from_env
from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger
from tradingagents.survivor.policy import SurvivorPolicy
from tradingagents.survivor.types import RouteStatus, SurvivorError

CHEAP = "or/rotation-cheap"
PRICY = "or/rotation-pricy"
UNPRICED = "or/unpriced-model"


@pytest.fixture(autouse=True)
def _clean_catalog():
    yield
    for model in (CHEAP, PRICY, UNPRICED):
        MODEL_PRICING.pop(("openrouter", model), None)


def _router(tmp_path, monkeypatch, *, models=None, pricing_map=None):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    if models is not None:
        monkeypatch.setenv("SURVIVOR_OPENROUTER_MODELS", models)
    else:
        monkeypatch.delenv("SURVIVOR_OPENROUTER_MODELS", raising=False)
    if pricing_map is not None:
        monkeypatch.setenv("SURVIVOR_OPENROUTER_PRICING_MAP", pricing_map)
    else:
        monkeypatch.delenv("SURVIVOR_OPENROUTER_PRICING_MAP", raising=False)

    ledger = InferenceUsageLedger(db_path=tempfile.mkdtemp() + "/us.db")
    return ModelRouter(
        policy=SurvivorPolicy(),
        budget_manager=BudgetManager(SurvivorPolicy(), ledger=ledger),
        health_manager=ProviderHealthManager(),
    ), ledger


class _Resp:
    usage_metadata = {"input_tokens": 100, "output_tokens": 10}


def _tracking_factory(calls, fail_models=()):
    """Factory recording invoked (provider, model) and raising parse errors for some models."""

    def factory(provider, model, **kwargs):
        calls.append((provider, model))

        def invoke(messages=None, **kw):
            if model in fail_models:
                raise ValueError("OutputParserException: failed to parse JSON output to schema")
            return _Resp()

        class _C:
            def invoke(self, _m, **_k):
                return invoke(_m, **_k)

        return _C()

    return factory


# --- 1. rotation picks cheapest configured OpenRouter model -------------------


@pytest.mark.unit
def test_rotation_selects_cheapest_priced_openrouter_model(tmp_path, monkeypatch):
    register_model_price(ModelPrice("openrouter", CHEAP, Decimal("0.01"), Decimal("0.02")))
    register_model_price(ModelPrice("openrouter", PRICY, Decimal("0.50"), Decimal("1.00")))
    router, _ = _router(
        tmp_path,
        monkeypatch,
        models=f"{CHEAP},{PRICY}",
        pricing_map=f"{CHEAP}:0.01,0.02;{PRICY}:0.50,1.00",
    )
    calls = []
    router.invoke_role("quick_analyst", "hi", mock_client_factory=_tracking_factory(calls))
    openrouter_calls = [m for p, m in calls if p == "openrouter"]
    assert openrouter_calls[:1] == [CHEAP]  # cheapest healthy priced model first
    assert set(openrouter_calls) <= {CHEAP, PRICY}


# --- 2. DeepSeek is configurable default, never the only model ----------------


@pytest.mark.unit
def test_deepseek_default_is_configurable_and_rotation_accepts_other_models(tmp_path, monkeypatch):
    default = "deepseek/deepseek-v4-flash-0731"
    monkeypatch.setenv("SURVIVOR_OPENROUTER_DEFAULT_MODEL", default)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("SURVIVOR_OPENROUTER_MODELS", raising=False)
    monkeypatch.delenv("SURVIVOR_OPENROUTER_MODEL", raising=False)
    monkeypatch.setenv("SURVIVOR_OPENROUTER_PRICING_MAP", f"{default}:0.10,0.20")
    routes = _role_routes_from_env()
    assert routes["quick_analyst"].preferred == [("openrouter", default)]
    assert get_model_price("openrouter", default) is not None
    register_model_price(ModelPrice("openrouter", CHEAP, Decimal("0.01"), Decimal("0.02")))
    router, _ = _router(tmp_path, monkeypatch, models=CHEAP, pricing_map=f"{CHEAP}:0.01,0.02")
    route = [
        pair
        for pair in router.role_routes["quick_analyst"].get_all_routes()
        if pair[0] == "openrouter"
    ]
    assert route and route[0] == ("openrouter", CHEAP)


@pytest.mark.unit
def test_another_configured_model_replaces_deepseek_default(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("SURVIVOR_OPENROUTER_DEFAULT_MODEL", "deepseek/deepseek-v4-flash-0731")
    monkeypatch.setenv("SURVIVOR_OPENROUTER_MODELS", CHEAP)
    monkeypatch.setenv("SURVIVOR_OPENROUTER_PRICING_MAP", f"{CHEAP}:0.01,0.02")
    routes = _role_routes_from_env()
    assert routes["quick_analyst"].preferred == [("openrouter", CHEAP)]


@pytest.mark.unit
def test_verified_deepseek_prices_support_nitro_rotation(tmp_path, monkeypatch):
    standard = "deepseek/deepseek-v4-flash-0731"
    nitro = "deepseek/deepseek-v4-flash-0731:nitro"
    router, _ = _router(
        tmp_path,
        monkeypatch,
        models=f"{standard},{nitro}",
        pricing_map=f"{standard}:0.045,0.09;{nitro}:0.05,0.16",
    )
    assert str(get_model_price("openrouter", standard).input_cost_per_1m) == "0.045"
    assert str(get_model_price("openrouter", nitro).output_cost_per_1m) == "0.16"

    # Standard is preferred for the cost-sensitive role while both remain
    # eligible for structured-output roles.
    ordered, _ = router._ordered_pool("quick_analyst")
    assert ordered[0] == ("openrouter", standard)
    structured, _ = router._ordered_pool("portfolio_manager", "structured_output")
    assert set(structured) == {("openrouter", standard), ("openrouter", nitro)}

    # Reliability/history can justify the higher-cost nitro route.
    for _ in range(5):
        router._record_structured("openrouter", standard, False)
    structured_after_history, _ = router._ordered_pool("portfolio_manager", "structured_output")
    assert structured_after_history[0] == ("openrouter", nitro)


# --- 3. unpriced models are NEVER selected ------------------------------------


@pytest.mark.unit
def test_unpriced_rotation_model_never_selected(tmp_path, monkeypatch):
    register_model_price(ModelPrice("openrouter", CHEAP, Decimal("0.01"), Decimal("0.02")))
    router, _ = _router(
        tmp_path,
        monkeypatch,
        models=f"{CHEAP},{UNPRICED}",
        pricing_map=f"{CHEAP}:0.01,0.02",
    )
    assert get_model_price("openrouter", UNPRICED) is None
    calls = []
    router.invoke_role("quick_analyst", "hi", mock_client_factory=_tracking_factory(calls))
    invoked = {m for p, m in calls}
    assert UNPRICED not in invoked  # never called, even as a candidate
    assert CHEAP in invoked


@pytest.mark.unit
def test_all_unpriced_fails_closed_with_unknown_price(tmp_path, monkeypatch):
    for k in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "MINIMAX_API_KEY", "MINIMAX_CN_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    router, _ = _router(tmp_path, monkeypatch, models=UNPRICED)
    calls = []
    with pytest.raises(SurvivorError) as exc_info:
        router.invoke_role("quick_analyst", "hi", mock_client_factory=_tracking_factory(calls))
    assert exc_info.value.status == RouteStatus.UNKNOWN_MODEL_PRICE
    assert "UNKNOWN_MODEL_PRICE" in str(exc_info.value)
    assert calls == []  # do NOT call it


# --- 4. fallback providers dropped when their keys are absent -----------------


@pytest.mark.unit
def test_no_fallback_to_keyless_openai_deepseek_minimax(tmp_path, monkeypatch):
    register_model_price(ModelPrice("openrouter", CHEAP, Decimal("0.01"), Decimal("0.02")))
    for k in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "MINIMAX_API_KEY", "MINIMAX_CN_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    router, _ = _router(tmp_path, monkeypatch, models=CHEAP, pricing_map=f"{CHEAP}:0.01,0.02")
    calls = []
    router.invoke_role(
        "market_analyst", "analyze NVDA", mock_client_factory=_tracking_factory(calls)
    )
    assert {p for p, _ in calls} == {"openrouter"}


# --- 5. budgets are not increased ---------------------------------------------


@pytest.mark.unit
def test_openrouter_maps_to_existing_global_budgets():
    policy = SurvivorPolicy()
    assert policy.get_provider_monthly_budget("openrouter") == policy.global_monthly_pence
    assert policy.get_provider_daily_budget("openrouter") == policy.global_daily_pence


# --- 6. reliability (per role weights) can override pure cost -----------------


@pytest.mark.unit
def test_reliability_influences_rotation(tmp_path, monkeypatch):
    """Once the cheap model accrues structured-output failures, a pricier-but-
    reliable configured model ranks ahead for a strong role (role weighting)."""
    register_model_price(ModelPrice("openrouter", CHEAP, Decimal("0.04"), Decimal("0.08")))
    register_model_price(ModelPrice("openrouter", PRICY, Decimal("0.06"), Decimal("0.12")))
    router, _ = _router(
        tmp_path,
        monkeypatch,
        models=f"{CHEAP},{PRICY}",
        pricing_map=f"{CHEAP}:0.04,0.08;{PRICY}:0.06,0.12",
    )
    for _ in range(5):
        router._record_structured("openrouter", CHEAP, False)  # CHEAP is flaky

    ordered, _ = router._ordered_pool("portfolio_manager")
    assert ordered.index(("openrouter", PRICY)) < ordered.index(("openrouter", CHEAP))


# --- 7. structured-output repair: same model once, then rotate ----------------


@pytest.mark.unit
def test_structured_output_repair_retries_same_then_rotates(tmp_path, monkeypatch):
    register_model_price(ModelPrice("openrouter", CHEAP, Decimal("0.01"), Decimal("0.02")))
    register_model_price(ModelPrice("openrouter", PRICY, Decimal("0.50"), Decimal("1.00")))
    router, _ = _router(
        tmp_path,
        monkeypatch,
        models=f"{CHEAP},{PRICY}",
        pricing_map=f"{CHEAP}:0.01,0.02;{PRICY}:0.50,1.00",
    )
    calls = []
    factory = _tracking_factory(calls, fail_models=(CHEAP,))
    response = router.invoke_role(
        "portfolio_manager",
        "decide",
        purpose="structured_output",
        mock_client_factory=factory,
    )
    assert response.usage_metadata["output_tokens"] == 10
    assert [m for p, m in calls if m == CHEAP] == [CHEAP, CHEAP]  # retried once
    assert [m for p, m in calls if m == PRICY] == [PRICY]  # rotated within OpenRouter


@pytest.mark.unit
def test_structured_output_repair_stays_within_openrouter(tmp_path, monkeypatch):
    register_model_price(ModelPrice("openrouter", CHEAP, Decimal("0.01"), Decimal("0.02")))
    router, _ = _router(tmp_path, monkeypatch, models=CHEAP, pricing_map=f"{CHEAP}:0.01,0.02")
    calls = []
    factory = _tracking_factory(calls, fail_models=(CHEAP,))
    with pytest.raises(SurvivorError):
        router.invoke_role(
            "trader",
            "decide",
            purpose="structured_output",
            mock_client_factory=factory,
        )
    # structured repair never fell back to a non-OpenRouter provider
    assert {p for p, _ in calls} == {"openrouter"}
