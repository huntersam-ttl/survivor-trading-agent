"""Regression tests: SurvivorLLM is LangChain-Runnable compatible.

Root cause being covered: ``chain = prompt | llm.bind_tools(tools)`` in the
analyst agents coerces the bound LLM via ``Runnable.coerce``, which rejected
the previous plain-class SurvivorLLM ("Expected a Runnable, callable or dict").

The invariant proven here: TradingAgents agent → SurvivorLLM → ModelRouter →
BudgetManager → provider → usage settlement, with NO raw provider client ever
escaping the guard. No real paid API calls.
"""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableSequence

from tradingagents.llm_clients.budget import BudgetManager
from tradingagents.llm_clients.provider_health import ProviderHealthManager
from tradingagents.llm_clients.router import (
    DEFAULT_ROLE_ROUTES,
    ModelRouter as RealModelRouter,
)
from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger
from tradingagents.survivor.autonomy.cycle import run_survivor_cycle
from tradingagents.survivor.autonomy.state import RuntimeState
from tradingagents.survivor.guard import (
    SurvivorLLM,
    reset_survivor_run,
    set_survivor_run,
)
from tradingagents.survivor.policy import SurvivorPolicy


def _make_guarded_llm(tmpdir, role="market_analyst"):
    ledger = InferenceUsageLedger(db_path=os.path.join(tmpdir, "usage.db"))
    policy = SurvivorPolicy()
    router = RealModelRouter(
        policy=policy,
        budget_manager=BudgetManager(policy=policy, ledger=ledger),
        health_manager=ProviderHealthManager(),
    )
    return SurvivorLLM(router, role=role), router, ledger


def _fake_client_factory(provider, model, **kwargs):
    """Mimics a real provider client: bind_tools/with_structured_output return
    a runnable whose invoke yields the configured response (like real
    provider bindings). The RESPONSE is a plain object (model output), never
    a client."""
    llm = MagicMock()
    llm._survivor_provider = provider
    llm._survivor_model = model
    response = SimpleNamespace(
        content="routed response",
        usage_metadata={"input_tokens": 100, "output_tokens": 40},
    )
    llm.invoke.return_value = response
    llm.bind_tools.return_value = llm
    llm.with_structured_output.return_value = llm
    return llm


def _patch_provider_factory(monkeypatch):
    """Patch the router-level client factory so NO real provider is touched."""
    captured = []

    def fake_create_llm_client(provider, model, **kwargs):
        client = _fake_client_factory(provider, model, **kwargs)
        captured.append((provider, model))
        return SimpleNamespace(get_llm=lambda: client)

    monkeypatch.setattr(
        "tradingagents.llm_clients.router.create_llm_client", fake_create_llm_client)
    return captured


class _RunCtx:
    def __init__(self, run_id="run_guard"):
        self._run_id = run_id

    def __enter__(self):
        self.tokens = set_survivor_run(self._run_id, "mkt-x")
        return self

    def __exit__(self, *exc):
        reset_survivor_run(self.tokens)
        return False


# --- 1. SurvivorLLM is accepted in prompt | SurvivorLLM ------------------------------

@pytest.mark.unit
def test_survivor_llm_composes_in_prompt_sequence(monkeypatch):
    _patch_provider_factory(monkeypatch)
    with tempfile.TemporaryDirectory() as tmpdir:
        llm, router, ledger = _make_guarded_llm(tmpdir)
        assert isinstance(llm, Runnable)
        prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
        chain = prompt | llm                      # the exact shape that crashed
        assert isinstance(chain, RunnableSequence)
        with _RunCtx("run_seq"):
            response = chain.invoke({"question": "Analyze this market"})
        assert response.content == "routed response"
        summary = ledger.get_run_usage_summary("run_seq")
        assert summary["total_calls"] == 1
        # whatever route was selected, it went through the guarded router
        assert summary["breakdown"][0]["role"] == "market_analyst"
        assert summary["total_cost_pence"] >= 0


@pytest.mark.unit
def test_prompt_sequence_with_bound_tools_composes_and_routes(monkeypatch):
    """The analyst shape: chain = prompt | llm.bind_tools(tools)."""
    _patch_provider_factory(monkeypatch)
    with tempfile.TemporaryDirectory() as tmpdir:
        llm, router, ledger = _make_guarded_llm(tmpdir)
        prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
        chain = prompt | llm.bind_tools(["tool_a"])
        assert isinstance(chain, RunnableSequence)
        with _RunCtx("run_seq2"):
            response = chain.invoke({"question": "Analyze"})
        assert response.content == "routed response"
        assert ledger.get_run_usage_summary("run_seq2")["total_calls"] == 1


# --- 2. invoke routes through ModelRouter ---------------------------------------------

@pytest.mark.unit
def test_invoke_routes_through_model_router_with_role_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        llm, router, ledger = _make_guarded_llm(tmpdir, role="trader")
        seen = {}
        original = router.invoke_role

        def spy(**kwargs):
            seen.update(kwargs)
            return original(**kwargs)

        router.invoke_role = spy
        with _RunCtx("run_role"):
            llm.invoke("decide", mock_client_factory=_fake_client_factory)
        assert seen["role"] == "trader"
        assert seen["run_id"] == "run_role"
        assert seen["ticker_or_market"] == "mkt-x"


@pytest.mark.unit
def test_real_provider_route_does_not_send_local_transform_into_spawn_worker():
    """Serialized specs, not local closures, cross the real provider boundary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        llm, router, _ = _make_guarded_llm(tmpdir)
        seen = {}

        def spy(**kwargs):
            seen.update(kwargs)
            return MagicMock(content="ok", usage_metadata={"input_tokens": 1, "output_tokens": 1})

        router.invoke_role = spy
        llm.bind_tools(["tool_a"]).invoke("use tools")

        assert seen["llm_transform"] is None
        assert seen["llm_specs"][0][0] == "bind_tools"


# --- 3/4. bind_tools / with_structured_output remain guarded ---------------------------

@pytest.mark.unit
def test_bind_tools_and_structured_output_return_guarded_llm():
    with tempfile.TemporaryDirectory() as tmpdir:
        llm, router, _ = _make_guarded_llm(tmpdir)
        bound = llm.bind_tools(["tool_a"])
        structured = llm.with_structured_output(schema=MagicMock(name="Schema"))
        # still the guard, never the raw provider client
        assert isinstance(bound, SurvivorLLM)
        assert isinstance(structured, SurvivorLLM)
        assert isinstance(bound, Runnable)
        assert bound.role == llm.role == "market_analyst"
        assert llm._transforms == ()             # original unmutated (immutable)


# --- 5/6. bound / structured invocations write the usage ledger ------------------------

@pytest.mark.unit
def test_bound_tool_invocation_writes_usage_ledger(monkeypatch):
    _patch_provider_factory(monkeypatch)
    with tempfile.TemporaryDirectory() as tmpdir:
        llm, router, ledger = _make_guarded_llm(tmpdir)
        bound = llm.bind_tools(["tool_a"])
        with _RunCtx("run_bound"):
            bound.invoke("use tools")
        summary = ledger.get_run_usage_summary("run_bound")
        assert summary["total_calls"] == 1
        assert router.budget_manager.get_pending_reserved_pence() == 0


@pytest.mark.unit
def test_structured_output_invocation_writes_usage_ledger(monkeypatch):
    _patch_provider_factory(monkeypatch)
    with tempfile.TemporaryDirectory() as tmpdir:
        llm, router, ledger = _make_guarded_llm(tmpdir, role="portfolio_manager")
        structured = llm.with_structured_output(schema=MagicMock(name="DecisionSchema"))
        with _RunCtx("run_struct"):
            structured.invoke("decide")
        summary = ledger.get_run_usage_summary("run_struct")
        assert summary["total_calls"] == 1
        assert summary["breakdown"][0]["role"] == "portfolio_manager"


# --- 7. BudgetManager reserves BEFORE provider invocation -------------------------------

@pytest.mark.unit
def test_budget_reservation_precedes_provider_invocation():
    with tempfile.TemporaryDirectory() as tmpdir:
        llm, router, _ = _make_guarded_llm(tmpdir)
        order = []
        original_authorize = router.budget_manager.authorize_and_reserve

        def spy_authorize(**kwargs):
            order.append("reserve")
            return original_authorize(**kwargs)

        router.budget_manager.authorize_and_reserve = spy_authorize

        def factory(provider, model, **kwargs):
            order.append("invoke")
            return _fake_client_factory(provider, model, **kwargs)

        with _RunCtx("run_order"):
            llm.invoke("hello", mock_client_factory=factory)
        assert order == ["reserve", "invoke"]


# --- 8. provider fallback remains allowlisted --------------------------------------------

@pytest.mark.unit
def test_fallback_routes_stay_within_allowlist(monkeypatch):
    # hermetic: default routes with no OpenRouter override
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("SURVIVOR_OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("SURVIVOR_OPENROUTER_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("SURVIVOR_OPENROUTER_MODELS", raising=False)
    monkeypatch.delenv("SURVIVOR_OPENROUTER_PRICING_MAP", raising=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        llm, router, _ = _make_guarded_llm(tmpdir)
        allowed = {
            pair for route in DEFAULT_ROLE_ROUTES.values()
            for pair in route.get_all_routes()
        }
        for pair in router.resolve_routes(llm.role):
            assert pair in allowed
    # with OpenRouter configured, its (provider, model) pair is also allowlisted
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("SURVIVOR_OPENROUTER_MODEL", "fake/hermetic")
    monkeypatch.setenv("SURVIVOR_OPENROUTER_PRICING", "0.1,0.2")
    with tempfile.TemporaryDirectory() as tmpdir:
        llm, router, _ = _make_guarded_llm(tmpdir)
        for pair in router.resolve_routes(llm.role):
            assert pair[0] in ("openrouter", "deepseek", "minimax")


# --- 9. no raw provider client escapes the guard -----------------------------------------

@pytest.mark.unit
def test_no_raw_provider_client_escapes_the_guard():
    with tempfile.TemporaryDirectory() as tmpdir:
        llm, router, _ = _make_guarded_llm(tmpdir)
        escaped = {}

        def inspecting_factory(provider, model, **kwargs):
            client = _fake_client_factory(provider, model, **kwargs)
            escaped["client"] = client
            return client

        with _RunCtx("run_escape"):
            result = llm.bind_tools(["tool_a"]).invoke(
                "hello", mock_client_factory=inspecting_factory)
        # the model OUTPUT is returned, never the client object
        assert result.content == "routed response"
        assert result is not escaped["client"]
        assert not hasattr(result, "bind_tools")
        # the guard wrapper itself is not a provider client
        assert not hasattr(llm, "_survivor_provider")
        assert llm.bind_tools(["t"])._transforms != ()   # binding stays lazily guarded


# --- 10/11. autonomous research invariants still hold -------------------------------------

def _guard_cycle(tmp_path, monkeypatch, **cfg):
    from tests.survivor.market_helpers import FakeMarketAdapter, make_snapshot
    from tradingagents.graph import trading_graph as tg_mod
    from tradingagents.survivor.autonomy import halt as halt_mod

    instances = []

    class _FakeGraph:
        def __init__(self, config=None, **kwargs):
            self.config = config or {}
            instances.append(self)

        def propagate(self, company_name, trade_date, asset_type="stock"):
            ledger = InferenceUsageLedger(
                db_path=self.config["survivor_usage_ledger_path"])
            ledger.record_inference(
                run_id=self.config["survivor_run_id"],
                agent_role="quick_analyst", provider="openrouter", model="fake/model",
                ticker_or_market=company_name,
                input_tokens=100, output_tokens=10, reasoning_tokens=0,
                native_cost_minor=1, native_currency="USD", gbp_cost_pence=2,
                timestamp_utc="2026-09-04T10:00:00Z",
            )
            return ({"final_trade_decision": "**Rating**: Buy\n\nP(YES): 60%"}, "Buy")

    monkeypatch.setattr(tg_mod, "TradingAgentsGraph", _FakeGraph)
    monkeypatch.setattr(halt_mod, "HALT_PATH", str(tmp_path / "HALT"))
    return run_survivor_cycle(
        {"survivor_autonomy_enabled": True, **cfg},
        adapter=FakeMarketAdapter([make_snapshot("mkt-0"), make_snapshot("mkt-1")]),
        research_fn=None,
        paper_ledger_path=str(tmp_path / "paper.db"),
        runtime_state=RuntimeState(db_path=str(tmp_path / "runtime.db")),
        usage_ledger=InferenceUsageLedger(db_path=str(tmp_path / "usage.db")),
        policy=SurvivorPolicy(),
    )


@pytest.mark.unit
def test_autonomous_fake_research_two_candidates_researched(tmp_path, monkeypatch):
    report = _guard_cycle(tmp_path, monkeypatch)
    assert report.research_candidates == 2
    assert report.ai_researched == 2
    assert report.candidate_failures == []
    assert report.ai_cost_pence > 0


@pytest.mark.unit
def test_guard_cycle_dry_run_executes_zero_paper_trades(tmp_path, monkeypatch):
    report = _guard_cycle(tmp_path, monkeypatch, survivor_dry_run=True)
    assert report.ai_researched == 2
    assert report.paper_trades == 0


# --- 12. auth key reaches the provider client (openrouter 401 regression) ------------

@pytest.mark.unit
def test_openrouter_auth_key_reaches_provider_client(monkeypatch):
    """langchain_openai 1.x dropped the `api_key` alias, silently discarding
    the key (OpenRouter 401 'Missing Authentication header' even with a valid
    key). The client must be constructed with `openai_api_key` AND an explicit
    Authorization default header."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-value")
    monkeypatch.setenv("SURVIVOR_OPENROUTER_MODEL", "fake/hermetic")
    from tradingagents.llm_clients.factory import create_llm_client

    llm = create_llm_client(provider="openrouter", model="fake/hermetic").get_llm()
    assert llm.openai_api_key is not None
    assert llm.openai_api_key.get_secret_value() == "sk-or-test-value"
    rc = llm.root_client
    assert rc.api_key == "sk-or-test-value"
    auth = {k.lower(): v for k, v in dict(rc._client.headers).items()}.get("authorization")
    assert auth is None or auth.endswith("sk-or-test-value")  # transport may defer to per-request
    assert rc.auth_headers.get("Authorization") == "Bearer sk-or-test-value"


@pytest.mark.unit
def test_router_failure_surfaces_all_attempted_providers(monkeypatch):
    """The raised error must include EVERY attempted provider (the primary
    preferred-provider failure must not be masked by the last fallback's)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        llm, router, _ = _make_guarded_llm(tmpdir)
        expected_routes = [pair[0] for pair in router.resolve_routes(llm.role)]

        def failing_factory(provider, model, **kwargs):
            raise RuntimeError(f"boom on {provider}")

        with _RunCtx("run_attempts"), pytest.raises(Exception) as excinfo:
            llm.invoke("hello", mock_client_factory=failing_factory)
        message = str(excinfo.value)
        assert "attempted:" in message
        for provider in expected_routes:            # every attempt visible
            assert f"boom on {provider}" in message
