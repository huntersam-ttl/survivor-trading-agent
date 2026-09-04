"""Fake OpenRouter routing tests for the SURVIVOR ModelRouter.

No real paid calls: inference is faked via ``mock_client_factory`` or a fake
graph. Proves OpenRouter reuses the existing Phase 1 control plane
(SurvivorLLM -> ModelRouter -> BudgetManager -> pricing/FX -> settlement ->
usage ledger) with the repository's EXISTING OpenRouter client support.
"""

import tempfile

import pytest

from tests.survivor.market_helpers import FakeMarketAdapter, make_snapshot
from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV
from tradingagents.llm_clients.pricing import MODEL_PRICING, get_model_price
from tradingagents.llm_clients.router import (
    DEFAULT_ROLE_ROUTES,
    ModelRouter,
    _role_routes_from_env,
)
from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger
from tradingagents.survivor.autonomy import halt as halt_mod
from tradingagents.survivor.autonomy.cycle import run_survivor_cycle
from tradingagents.survivor.autonomy.state import RuntimeState
from tradingagents.survivor.policy import SurvivorPolicy

CHEAP = "fake/fake-model"      # opaque OpenRouter-style ID (never assumed priced)
STRONG = "fake/strong-model"


@pytest.fixture(autouse=True)
def _clean_pricing_catalog():
    yield
    MODEL_PRICING.pop(("openrouter", CHEAP), None)
    MODEL_PRICING.pop(("openrouter", STRONG), None)


# --- 1. OPENROUTER_API_KEY detected -----------------------------------------------

@pytest.mark.unit
def test_openrouter_api_key_env_detected():
    assert PROVIDER_API_KEY_ENV["openrouter"] == "OPENROUTER_API_KEY"


# --- 2. OpenRouter route selected when configured ---------------------------------

@pytest.mark.unit
def test_openrouter_route_selected_for_all_roles(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("SURVIVOR_OPENROUTER_MODEL", CHEAP)
    monkeypatch.delenv("SURVIVOR_OPENROUTER_STRONG_MODEL", raising=False)

    routes = _role_routes_from_env()
    assert routes["market_analyst"].preferred == [("openrouter", CHEAP)]
    assert routes["quick_analyst"].preferred == [("openrouter", CHEAP)]
    assert routes["bull_researcher"].preferred == [("openrouter", CHEAP)]
    # one configured model serves the strong roles too when no strong model set
    assert routes["portfolio_manager"].preferred == [("openrouter", CHEAP)]

    router = ModelRouter(SurvivorPolicy())
    assert router.resolve_routes("trader")[0] == ("openrouter", CHEAP)


@pytest.mark.unit
def test_strong_roles_can_use_configured_strong_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("SURVIVOR_OPENROUTER_MODEL", CHEAP)
    monkeypatch.setenv("SURVIVOR_OPENROUTER_STRONG_MODEL", STRONG)

    routes = _role_routes_from_env()
    assert routes["research_manager"].preferred == [("openrouter", STRONG)]
    assert routes["trader"].preferred == [("openrouter", STRONG)]
    assert routes["portfolio_manager"].preferred == [("openrouter", STRONG)]
    assert routes["news_analyst"].preferred == [("openrouter", CHEAP)]


# --- 3. missing key fails closed (routes unchanged) --------------------------------

@pytest.mark.unit
def test_missing_key_leaves_default_routes(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("SURVIVOR_OPENROUTER_MODEL", CHEAP)

    routes = _role_routes_from_env()
    assert routes == dict(DEFAULT_ROLE_ROUTES)
    all_pairs = {p for r in routes.values() for p, _ in r.get_all_routes()}
    assert "openrouter" not in all_pairs


@pytest.mark.unit
def test_explicit_role_routes_win_over_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("SURVIVOR_OPENROUTER_MODEL", CHEAP)

    explicit = dict(DEFAULT_ROLE_ROUTES)
    router = ModelRouter(SurvivorPolicy(), role_routes=explicit)
    assert router.role_routes is explicit          # operator config not overridden


# --- 4. unknown OpenRouter model price fails closed ---------------------------------

@pytest.mark.unit
def test_unknown_openrouter_price_fails_closed(monkeypatch):
    from tradingagents.survivor.types import SurvivorError

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("SURVIVOR_OPENROUTER_MODEL", CHEAP)
    monkeypatch.delenv("SURVIVOR_OPENROUTER_PRICING", raising=False)

    router = ModelRouter(SurvivorPolicy())
    assert get_model_price("openrouter", CHEAP) is None   # never priced by name

    called_providers = []

    def fake_factory(provider, model, **kwargs):
        # no real network: any fallback attempt is faked and fails immediately
        called_providers.append(provider)
        raise RuntimeError("simulated provider outage")

    with pytest.raises(SurvivorError):
        router.invoke_role(role="quick_analyst", messages="hi",
                           run_id="r_price", ticker_or_market="m",
                           mock_client_factory=fake_factory)
    # the unpriced OpenRouter model was NEVER invoked — fail closed, no guess
    assert "openrouter" not in called_providers


@pytest.mark.unit
def test_explicit_pricing_config_is_registered(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("SURVIVOR_OPENROUTER_MODEL", CHEAP)
    monkeypatch.setenv("SURVIVOR_OPENROUTER_PRICING", "0.14,0.28")

    ModelRouter(SurvivorPolicy())     # constructing the router applies env pricing
    price = get_model_price("openrouter", CHEAP)
    assert price is not None and str(price.input_cost_per_1m) == "0.14"

    monkeypatch.setenv("SURVIVOR_OPENROUTER_PRICING", "not-a-number")
    with pytest.raises(ValueError):
        ModelRouter(SurvivorPolicy())


# --- 5+6. reservation BEFORE invocation; ledger records provider=openrouter --------

@pytest.mark.unit
def test_reservation_precedes_invocation_and_ledger_records_openrouter(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("SURVIVOR_OPENROUTER_MODEL", CHEAP)
    monkeypatch.setenv("SURVIVOR_OPENROUTER_PRICING", "0.14,0.28")

    ledger = InferenceUsageLedger(db_path=tempfile.mkdtemp() + "/usage.db")

    router = ModelRouter(SurvivorPolicy())
    from tradingagents.llm_clients.budget import BudgetManager

    router.budget_manager = BudgetManager(SurvivorPolicy(), ledger=ledger)

    order = []
    original_authorize = router.budget_manager.authorize_and_reserve

    def _spy_authorize(**kwargs):
        order.append("reserve")
        return original_authorize(**kwargs)

    router.budget_manager.authorize_and_reserve = _spy_authorize

    def fake_factory(provider, model, **kwargs):
        order.append("invoke")
        assert provider == "openrouter" and model == CHEAP
        return SimpleNamespace(invoke=lambda messages: SimpleNamespace(
            usage_metadata={"input_tokens": 1000, "output_tokens": 100},
        ))

    response = router.invoke_role(role="quick_analyst", messages="hello",
                                  run_id="run_or_1", ticker_or_market="mkt-x",
                                  mock_client_factory=fake_factory)
    assert order == ["reserve", "invoke"]            # reservation BEFORE the call
    assert response.usage_metadata["output_tokens"] == 100

    summary = ledger.get_run_usage_summary("run_or_1")
    assert summary["total_calls"] == 1
    assert summary["breakdown"][0]["provider"] == "openrouter"
    assert summary["total_cost_pence"] > 0           # pricing/FX applied (fail-closed OK)


# --- 7. fallback never bypasses the allowlist --------------------------------------

@pytest.mark.unit
def test_fallback_stays_within_configured_allowlist(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("SURVIVOR_OPENROUTER_MODEL", CHEAP)
    monkeypatch.setenv("SURVIVOR_OPENROUTER_STRONG_MODEL", STRONG)

    allowed_pairs = {
        pair
        for route in DEFAULT_ROLE_ROUTES.values()
        for pair in route.get_all_routes()
    } | {("openrouter", CHEAP), ("openrouter", STRONG)}

    for route in _role_routes_from_env().values():
        for pair in route.get_all_routes():
            assert pair in allowed_pairs, f"route {pair} bypasses the allowlist"


# --- 8+9. autonomous research through fake OpenRouter ------------------------------

class _FakeOpenRouterGraph:
    """Stands in for TradingAgentsGraph; records config, fakes one guarded run."""

    instances: list = []

    def __init__(self, config=None, **kwargs):
        self.config = config or {}
        _FakeOpenRouterGraph.instances.append(self)

    def propagate(self, company_name, trade_date, asset_type="stock"):
        ledger = InferenceUsageLedger(db_path=self.config["survivor_usage_ledger_path"])
        ledger.record_inference(
            run_id=self.config["survivor_run_id"],
            agent_role="quick_analyst", provider="openrouter", model=CHEAP,
            ticker_or_market=company_name,
            input_tokens=100, output_tokens=10, reasoning_tokens=0,
            native_cost_minor=1, native_currency="USD", gbp_cost_pence=2,
            timestamp_utc="2026-09-04T10:00:00Z",
        )
        return (
            {"final_trade_decision": "**Rating**: Buy\n\nP(YES): 60%"},
            "Buy",
        )


def _or_cycle(tmp_path, monkeypatch, **cfg):
    from tradingagents.graph import trading_graph as tg_mod

    _FakeOpenRouterGraph.instances = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("SURVIVOR_OPENROUTER_MODEL", CHEAP)
    monkeypatch.setenv("SURVIVOR_OPENROUTER_PRICING", "0.14,0.28")
    monkeypatch.setattr(tg_mod, "TradingAgentsGraph", _FakeOpenRouterGraph)
    monkeypatch.setattr(halt_mod, "HALT_PATH", str(tmp_path / "HALT"))
    return run_survivor_cycle(
        {"survivor_autonomy_enabled": True, **cfg},
        adapter=FakeMarketAdapter([make_snapshot("mkt-0"), make_snapshot("mkt-1")]),
        research_fn=None,                       # default research path (OpenRouter)
        paper_ledger_path=str(tmp_path / "paper.db"),
        runtime_state=RuntimeState(db_path=str(tmp_path / "runtime.db")),
        usage_ledger=InferenceUsageLedger(db_path=str(tmp_path / "usage.db")),
        policy=SurvivorPolicy(),
    )


@pytest.mark.unit
def test_autonomous_research_two_candidates_researched_via_openrouter(
    tmp_path, monkeypatch,
):
    report = _or_cycle(tmp_path, monkeypatch)
    assert report.research_candidates == 2
    assert report.ai_researched == 2
    assert report.candidate_failures == []
    assert report.ai_cost_pence > 0
    # graph construction clients ride the existing OpenRouter provider path
    for graph in _FakeOpenRouterGraph.instances:
        assert graph.config["llm_provider"] == "openrouter"
        assert graph.config["quick_think_llm"] == CHEAP
        assert graph.config["survivor_enabled"] is True


@pytest.mark.unit
def test_openrouter_dry_run_executes_zero_paper_trades(tmp_path, monkeypatch):
    report = _or_cycle(tmp_path, monkeypatch, survivor_dry_run=True)
    assert report.ai_researched == 2
    assert report.approved >= 1
    assert report.paper_trades == 0
