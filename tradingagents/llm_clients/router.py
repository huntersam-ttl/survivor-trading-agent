"""Provider-aware ModelRouter with role-based routing, health checks, and budget enforcement.

SURVIVOR uses OpenRouter as the gateway and keeps *dynamic* model routing in
front of it. Every guarded call flows:

    SurvivorLLM -> ModelRouter -> CostGovernor -> BudgetManager -> OpenRouter

ModelRouter selects/rotates among the CONFIGURED, PRICED OpenRouter models for
a role using a deterministic weighted score: cost, provider/model health,
structured-output reliability, latency, historical performance, and remaining
AI budget. It NEVER selects an unpriced model (fails closed with
``UnknownPriceError`` / ``UNKNOWN_MODEL_PRICE``). All model choices are driven
by configuration (``SURVIVOR_OPENROUTER_MODEL[S]`` / ``_PRICING[_MAP]``) so
models can be changed without code changes. The default is only a fallback;
configured model pools remain eligible for dynamic rotation.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import queue
import time
from typing import Any

from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.budget import BudgetManager
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.pricing import DEFAULT_OPENROUTER_MODEL, get_model_price
from tradingagents.llm_clients.provider_health import ProviderHealthManager
from tradingagents.survivor.policy import SurvivorPolicy
from tradingagents.survivor.types import (
    AgentRole,
    BudgetExhaustedError,
    InferenceFailedError,
    ModelNotAllowedError,
    ProviderErrorCategory,
    ProviderCallTimeout,
    RoleRoute,
    RouteStatus,
    SurvivorError,
    UnhealthyProviderError,
    UnknownPriceError,
)

logger = logging.getLogger(__name__)


def _wire_value(value):
    """Convert provider metadata/results to bounded multiprocessing-safe data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _wire_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return _wire_value(value.model_dump(mode="json"))
    return str(value)


def _provider_response_wire(response):
    """Return a small plain-data envelope for the parent process."""
    if hasattr(response, "content"):
        return {
            "kind": "message",
            "content": _wire_value(response.content),
            "additional_kwargs": _wire_value(getattr(response, "additional_kwargs", {})),
            "response_metadata": _wire_value(getattr(response, "response_metadata", {})),
            "usage_metadata": _wire_value(getattr(response, "usage_metadata", {})),
        }
    if hasattr(response, "model_dump"):
        return {"kind": "value", "value": _wire_value(response.model_dump(mode="json"))}
    return {"kind": "value", "value": _wire_value(response)}


def _provider_response_from_wire(envelope):
    if envelope.get("kind") != "message":
        return envelope.get("value")
    from langchain_core.messages import AIMessage

    return AIMessage(
        content=envelope.get("content"),
        additional_kwargs=envelope.get("additional_kwargs") or {},
        response_metadata=envelope.get("response_metadata") or {},
        usage_metadata=envelope.get("usage_metadata") or None,
    )


def _provider_call_worker(client, messages, result_queue) -> None:
    """Invoke one already-configured client and return only a bounded result."""
    try:
        result_queue.put({"ok": True, "response": _provider_response_wire(client.invoke(messages))})
    except BaseException as exc:  # noqa: BLE001 - parent classifies safely
        result_queue.put({
            "ok": False,
            "error_type": type(exc).__name__,
            "message": str(exc)[:500],
        })


def _provider_call_deadline_seconds() -> float:
    try:
        return max(0.1, float(os.environ.get("SURVIVOR_LLM_CALL_DEADLINE_SECONDS", "75")))
    except ValueError:
        return 75.0


def _invoke_with_deadline(client, messages, deadline: float):
    """Run a provider call in a killable process with a hard total deadline."""
    context = multiprocessing.get_context("fork")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_provider_call_worker, args=(client, messages, result_queue), daemon=True,
    )
    deadline_at = time.monotonic() + deadline
    result = None
    process.start()
    while result is None and time.monotonic() < deadline_at:
        try:
            result = result_queue.get(timeout=min(0.1, max(0.001, deadline_at - time.monotonic())))
        except queue.Empty:
            if not process.is_alive():
                break
    if result is not None:
        process.join(max(0.0, deadline_at - time.monotonic()))
    if process.is_alive():
        process.terminate()
        process.join(2.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(2.0)
        if result is None:
            raise ProviderCallTimeout(f"provider call exceeded {deadline:.1f}s total deadline")
    if result is None:
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            result = {"ok": False, "error_type": "ProviderWorkerError", "message": "worker returned no result"}
    result_queue.close()
    result_queue.join_thread()
    process.close()
    if result.get("ok"):
        return _provider_response_from_wire(result["response"])
    raise RuntimeError(f"{result.get('error_type', 'ProviderError')}: {result.get('message', '')}")

# Roles that may use a stronger (more expensive) model when the operator
# configures one; every other role always uses the cheap model.
_STRONG_ROLES = frozenset(
    {
        AgentRole.RESEARCH_MANAGER.value,
        AgentRole.TRADER.value,
        AgentRole.PORTFOLIO_MANAGER.value,
    }
)

# Model-selection scoring weights. Role influences the weights: cheap roles
# lean harder on cost, strong roles allow more capable (pricier) models by
# weighting cost lower and historical performance higher. All weights sum to 1.
_SCORE_WEIGHTS_CHEAP = {
    "cost": 0.40,
    "reliability": 0.20,
    "latency": 0.10,
    "historical": 0.15,
    "budget": 0.15,
}
_SCORE_WEIGHTS_STRONG = {
    "cost": 0.20,
    "reliability": 0.25,
    "latency": 0.10,
    "historical": 0.20,
    "budget": 0.25,
}

# Latency reference (ms): a model at/above this speed maps to a latency score of
# 0, a model at 0 ms maps to 1. Used only to normalise the sub-score.
_LATENCY_REF_MS = 15000.0


def _weights_for_role(role: str) -> dict[str, float]:
    return dict(_SCORE_WEIGHTS_STRONG if role in _STRONG_ROLES else _SCORE_WEIGHTS_CHEAP)


def _provider_available(provider: str) -> bool:
    """Whether a provider can actually be called (its API key is configured).

    Providers that do not require a key (e.g. local runtimes) are always
    available. A provider whose key env var is unset/empty is NOT available —
    this keeps the fallback chain from ever attempting OpenAI/DeepSeek/MiniMax
    (or any keyed provider) when its key is not configured.
    """
    env_var = get_api_key_env(provider)
    if not env_var:
        return True
    return bool(os.environ.get(env_var, "").strip())


def _split_models(raw: str) -> list[str]:
    """Parse a comma-separated model list, dropping empties and whitespace."""
    return [m.strip() for m in raw.split(",") if m.strip()]


def _configured_default_model() -> str:
    """Resolve the preferred model while retaining the legacy variable."""
    return (
        os.environ.get("SURVIVOR_OPENROUTER_DEFAULT_MODEL", "").strip()
        or os.environ.get("SURVIVOR_OPENROUTER_MODEL", "").strip()
        or DEFAULT_OPENROUTER_MODEL
    )


def _role_routes_from_env() -> dict[str, RoleRoute]:
    """Build default role routes, optionally overridden by OpenRouter env config.

    OpenRouter is the gateway and the configured model set is the rotation
    pool for every role:

    - ``SURVIVOR_OPENROUTER_DEFAULT_MODEL``: preferred exact OpenRouter model
      ID. ``SURVIVOR_OPENROUTER_MODEL`` remains supported as a legacy alias.
    - ``SURVIVOR_OPENROUTER_STRONG_MODEL``: optional stronger model for the
      research manager / trader / portfolio manager roles (single-model path).
    - ``SURVIVOR_OPENROUTER_MODELS``: comma-separated approved OpenRouter
      models to *rotate* among. When set, every role may select from this set
      (ModelRouter ranks them by role/cost/health/reliability/latency/history/
      budget). Each model must be explicitly priced
      (``SURVIVOR_OPENROUTER_PRICING_MAP`` or the default catalog), else it is
      never selected (UNKNOWN_MODEL_PRICE).

    Non-OpenRouter defaults (OpenAI/DeepSeek/MiniMax) are retained ONLY as
    fallbacks and ONLY while their own API keys are configured — never used as
    fallbacks when a key is absent. Routing activates only when
    ``OPENROUTER_API_KEY`` is set.
    """
    from tradingagents.llm_clients.pricing import (
        register_openrouter_env_pricing,
        register_openrouter_env_pricing_map,
    )

    register_openrouter_env_pricing()
    register_openrouter_env_pricing_map()

    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        return dict(DEFAULT_ROLE_ROUTES)

    rotation = _split_models(os.environ.get("SURVIVOR_OPENROUTER_MODELS", ""))
    if not rotation:
        model = _configured_default_model()
        rotation = [model]

    strong = os.environ.get("SURVIVOR_OPENROUTER_STRONG_MODEL", "").strip() or rotation[0]

    routes: dict[str, RoleRoute] = {}
    for role, route in DEFAULT_ROLE_ROUTES.items():
        # Only keep non-OpenRouter fallbacks whose provider key is configured.
        available_fallback = [
            pair
            for pair in route.get_all_routes()
            if pair[0] != "openrouter" and _provider_available(pair[0])
        ]
        if len(rotation) > 1:
            # Dynamic rotation: the full configured OpenRouter set is eligible.
            preferred = [("openrouter", m) for m in rotation]
        else:
            chosen = strong if role in _STRONG_ROLES else rotation[0]
            preferred = [("openrouter", chosen)]
        routes[role] = RoleRoute(
            role=role,
            preferred=preferred,
            fallback=available_fallback,
        )
    return routes


DEFAULT_ROLE_ROUTES: dict[str, RoleRoute] = {
    AgentRole.QUICK_ANALYST.value: RoleRoute(
        role=AgentRole.QUICK_ANALYST,
        preferred=[("minimax", "minimax-text-01")],
        fallback=[("deepseek", "deepseek-chat")],
    ),
    AgentRole.MARKET_ANALYST.value: RoleRoute(
        role=AgentRole.MARKET_ANALYST,
        preferred=[("deepseek", "deepseek-chat")],
        fallback=[("minimax", "minimax-text-01")],
    ),
    AgentRole.SOCIAL_ANALYST.value: RoleRoute(
        role=AgentRole.SOCIAL_ANALYST,
        preferred=[("deepseek", "deepseek-chat")],
        fallback=[("minimax", "minimax-text-01")],
    ),
    AgentRole.NEWS_ANALYST.value: RoleRoute(
        role=AgentRole.NEWS_ANALYST,
        preferred=[("deepseek", "deepseek-chat")],
        fallback=[("minimax", "minimax-text-01")],
    ),
    AgentRole.FUNDAMENTALS_ANALYST.value: RoleRoute(
        role=AgentRole.FUNDAMENTALS_ANALYST,
        preferred=[("deepseek", "deepseek-chat")],
        fallback=[("minimax", "minimax-text-01")],
    ),
    AgentRole.BULL_RESEARCHER.value: RoleRoute(
        role=AgentRole.BULL_RESEARCHER,
        preferred=[("deepseek", "deepseek-chat")],
        fallback=[("minimax", "minimax-text-01")],
    ),
    AgentRole.BEAR_RESEARCHER.value: RoleRoute(
        role=AgentRole.BEAR_RESEARCHER,
        preferred=[("deepseek", "deepseek-chat")],
        fallback=[("minimax", "minimax-text-01")],
    ),
    AgentRole.RESEARCH_MANAGER.value: RoleRoute(
        role=AgentRole.RESEARCH_MANAGER,
        preferred=[("openai", "gpt-5.6")],
        fallback=[("deepseek", "deepseek-chat")],
    ),
    AgentRole.TRADER.value: RoleRoute(
        role=AgentRole.TRADER,
        preferred=[("openai", "gpt-5.6")],
        fallback=[("deepseek", "deepseek-chat")],
    ),
    AgentRole.AGGRESSIVE_RISK.value: RoleRoute(
        role=AgentRole.AGGRESSIVE_RISK,
        preferred=[("openai", "gpt-5.6")],
        fallback=[("deepseek", "deepseek-chat")],
    ),
    AgentRole.CONSERVATIVE_RISK.value: RoleRoute(
        role=AgentRole.CONSERVATIVE_RISK,
        preferred=[("openai", "gpt-5.6")],
        fallback=[("deepseek", "deepseek-chat")],
    ),
    AgentRole.NEUTRAL_RISK.value: RoleRoute(
        role=AgentRole.NEUTRAL_RISK,
        preferred=[("openai", "gpt-5.6")],
        fallback=[("deepseek", "deepseek-chat")],
    ),
    AgentRole.PORTFOLIO_MANAGER.value: RoleRoute(
        role=AgentRole.PORTFOLIO_MANAGER,
        preferred=[("openai", "gpt-5.6")],
        fallback=[("deepseek", "deepseek-chat")],
    ),
}


class ModelRouter:
    """Provider-aware router that selects healthy, budget-authorized models for agent roles.

    Dynamic OpenRouter model selection/rotation: for each role it ranks the
    CONFIGURED, PRICED models by cost, provider/model health, structured-output
    reliability, latency, historical performance, and remaining AI budget, then
    invokes the best one. Unpriced models are NEVER selected (fails closed with
    ``UnknownPriceError``). All model choices stay behind the control plane:

        SurvivorLLM -> ModelRouter -> CostGovernor -> BudgetManager -> OpenRouter
    """

    def __init__(
        self,
        policy: SurvivorPolicy,
        budget_manager: BudgetManager | None = None,
        health_manager: ProviderHealthManager | None = None,
        role_routes: dict[str, RoleRoute] | None = None,
    ):
        self.policy = policy
        self.budget_manager = budget_manager or BudgetManager(policy)
        self.health_manager = health_manager or ProviderHealthManager()
        self.role_routes = role_routes if role_routes is not None else _role_routes_from_env()
        # In-process structured-output reliability: (provider, model) -> [successes, failures].
        self._structured_stats: dict[tuple[str, str], list[int]] = {}

    def resolve_routes(self, role: str) -> list[tuple[str, str]]:
        route = self.role_routes.get(role)
        if not route:
            raise ModelNotAllowedError(f"No model route configured for agent role '{role}'.")
        return route.get_all_routes()

    # -- dynamic OpenRouter selection / rotation ----------------------------------

    def _record_structured(self, provider: str, model: str, success: bool) -> None:
        key = (provider.lower(), model)
        entry = self._structured_stats.setdefault(key, [0, 0])
        entry[0 if success else 1] += 1

    def _structured_reliability(self, provider: str, model: str) -> float:
        entry = self._structured_stats.get((provider.lower(), model), [0, 0])
        successes, failures = entry
        total = successes + failures
        if total == 0:
            return 1.0  # no evidence yet: optimistic, failures penalise later
        return successes / total

    def _budget_score(self, provider: str) -> float:
        """Remaining AI budget for a provider (0..1): more headroom scores higher."""
        try:
            limit = self.policy.get_provider_monthly_budget(provider)
        except Exception:
            return 0.5
        if limit <= 0:
            return 0.0
        spent = self.budget_manager.ledger.get_monthly_spend_pence(provider)
        reserved = self.budget_manager.get_pending_reserved_pence(provider=provider)
        remaining = max(0, limit - spent - reserved)
        return min(1.0, remaining / limit)

    def _ordered_pool(self, role: str, purpose: str = "standard") -> tuple[list, dict[str, list]]:
        """Rank the eligible (provider, model) candidates for a role.

        Returns ``(ordered_pool, skipped)`` where ``ordered_pool`` is the ranked
        list of healthy, priced candidates and ``skipped`` records why candidates
        were excluded (``unknown_price`` / ``unhealthy``). For ``purpose`` of
        ``"structured_output"`` the pool is restricted to configured OpenRouter
        models (structured-output repair never jumps to non-OpenRouter fallbacks).
        """
        route = self.role_routes.get(role)
        if not route:
            raise ModelNotAllowedError(f"No model route configured for agent role '{role}'.")

        all_routes = route.get_all_routes()
        openrouter = [pair for pair in all_routes if pair[0] == "openrouter"]
        others = [
            pair for pair in all_routes if pair[0] != "openrouter" and _provider_available(pair[0])
        ]

        or_ranked, or_skipped = self._rank_pool(role, openrouter)
        if purpose == "structured_output":
            return or_ranked, or_skipped
        other_ranked, other_skipped = self._rank_pool(role, others)
        ordered = or_ranked + other_ranked
        skipped: dict[str, list] = {
            "unknown_price": or_skipped.get("unknown_price", [])
            + other_skipped.get("unknown_price", []),
            "unhealthy": or_skipped.get("unhealthy", []) + other_skipped.get("unhealthy", []),
        }
        return ordered, skipped

    def _rank_pool(self, role: str, pool: list[tuple[str, str]]) -> tuple[list, dict[str, list]]:
        """Score and sort a candidate pool. Unhealthy/unpriced candidates are excluded."""
        weights = _weights_for_role(role)
        skipped: dict[str, list] = {"unknown_price": [], "unhealthy": []}
        priced: list[tuple[tuple[str, str], float]] = []
        for provider, model in pool:
            if not self.health_manager.is_healthy(provider):
                skipped["unhealthy"].append((provider, model))
                continue
            price = get_model_price(provider, model)
            if price is None:
                logger.warning(
                    "Excluding unpriced model '%s/%s' for role '%s' (UNKNOWN_MODEL_PRICE).",
                    provider,
                    model,
                    role,
                )
                skipped["unknown_price"].append((provider, model))
                continue
            cost = float(price.input_cost_per_1m + price.output_cost_per_1m)
            priced.append(((provider, model), cost))

        ordered: list[tuple[str, str]] = []
        if not priced:
            return ordered, skipped
        costs = [c for _, c in priced]
        min_cost, max_cost = min(costs), max(costs)
        span = max_cost - min_cost
        scored: list[tuple[float, tuple[str, str], float]] = []
        for (provider, model), cost in priced:
            cost_score = 1.0 if span <= 0 else 1.0 - (cost - min_cost) / span
            stats = self.budget_manager.ledger.get_model_stats(provider, model)
            avg_lat = stats["avg_latency_ms"]
            if avg_lat is not None:
                lat_score = 1.0 - (min(avg_lat, _LATENCY_REF_MS) / _LATENCY_REF_MS)
            else:
                lat_score = 0.5
            hist = stats["success_rate"]
            rel = self._structured_reliability(provider, model)
            budget = self._budget_score(provider)
            score = (
                weights["cost"] * cost_score
                + weights["reliability"] * rel
                + weights["latency"] * lat_score
                + weights["historical"] * hist
                + weights["budget"] * budget
            )
            scored.append((score, (provider, model), cost))
        scored.sort(key=lambda e: (-e[0], 0 if e[1][0] == "openrouter" else 1, e[2]))
        ordered = [pair for _, pair, _ in scored]
        return ordered, skipped

    def _pool_empty_error(self, skipped: dict[str, list]) -> SurvivorError:
        if skipped.get("unknown_price"):
            models = ", ".join(f"{p}/{m}" for p, m in skipped["unknown_price"])
            return UnknownPriceError(
                f"Pricing for OpenRouter model(s) [{models}] is UNKNOWN (UNKNOWN_MODEL_PRICE). "
                "Refusing to select an unpriced model; configure SURVIVOR_OPENROUTER_PRICING[_MAP]."
            )
        if skipped.get("unhealthy"):
            provs = ", ".join({p for p, _ in skipped["unhealthy"]})
            return UnhealthyProviderError(f"Provider(s) [{provs}] are unhealthy or in cooldown.")
        return SurvivorError(
            "No acceptable healthy, priced provider route found.", status=RouteStatus.NO_PROVIDER
        )

    def _invoke_one(
        self,
        provider: str,
        model: str,
        role: str,
        messages: Any,
        run_id: str,
        ticker_or_market: str,
        est_input_tokens: int,
        max_output_tokens: int,
        mock_client_factory: Any | None,
        llm_transform: Any | None,
        llm_kwargs: dict[str, Any],
    ) -> Any:
        """Execute one full invocation lifecycle (reserve -> invoke -> settle).

        Raises the appropriate ``SurvivorError`` on failure, carrying the error
        category so callers can implement structured-output repair.
        """
        if not self.health_manager.is_healthy(provider):
            raise UnhealthyProviderError(
                f"Provider '{provider}' is currently unhealthy or in cooldown."
            )
        try:
            reservation = self.budget_manager.authorize_and_reserve(
                provider=provider,
                model=model,
                agent_role=role,
                est_input_tokens=est_input_tokens,
                est_output_tokens=max_output_tokens,
                run_id=run_id,
                ticker_or_market=ticker_or_market,
            )
        except (BudgetExhaustedError, UnknownPriceError) as exc:
            raise exc

        start_time = time.time()
        try:
            if mock_client_factory:
                client = mock_client_factory(provider=provider, model=model, **llm_kwargs)
            else:
                client = create_llm_client(provider=provider, model=model, **llm_kwargs).get_llm()
            if llm_transform is not None:
                client = llm_transform(client)
            # Test doubles are deliberately invoked in-process so existing
            # unit seams remain deterministic. Real provider clients always
            # use the killable process boundary.
            response = (
                client.invoke(messages)
                if mock_client_factory
                else _invoke_with_deadline(client, messages, _provider_call_deadline_seconds())
            )
            latency_ms = int((time.time() - start_time) * 1000)

            usage = getattr(response, "usage_metadata", None) or {}
            actual_in = usage.get("input_tokens", est_input_tokens)
            actual_out = usage.get("output_tokens", 200)
            output_details = usage.get("output_token_details") or {}
            reasoning = usage.get("reasoning_tokens", 0) or output_details.get("reasoning", 0)

            self.budget_manager.settle_reservation(
                handle=reservation,
                actual_input_tokens=actual_in,
                actual_output_tokens=actual_out,
                actual_reasoning_tokens=reasoning,
                latency_ms=latency_ms,
                status="SUCCESS",
            )
            self.health_manager.record_success(provider)
            return response
        except Exception as exc:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "Inference call failed on '%s/%s' for role '%s': %s", provider, model, role, exc
            )

            err_msg = str(exc).lower()
            if (
                "401" in err_msg
                or "unauthorized" in err_msg
                or "invalid api key" in err_msg
                or "auth" in err_msg
            ):
                cat = ProviderErrorCategory.AUTH_ERROR
            elif (
                "validationerror" in err_msg
                or "outputparserexception" in err_msg
                or ("json" in err_msg and "parse" in err_msg)
                or ("validation" in err_msg and "schema" in err_msg)
            ):
                cat = ProviderErrorCategory.INVALID_RESPONSE
            elif "429" in err_msg or "rate limit" in err_msg or "quota" in err_msg:
                cat = ProviderErrorCategory.RATE_LIMIT
            elif (
                type(exc).__name__ == "ProviderCallTimeout"
                or "provider call exceeded" in err_msg
                or "timeout" in err_msg
            ):
                cat = ProviderErrorCategory.TIMEOUT
            else:
                cat = ProviderErrorCategory.SERVER_ERROR

            if cat != ProviderErrorCategory.INVALID_RESPONSE:
                # A structured-output parse error is a model-quality issue, not a
                # provider outage — do NOT cool down the whole provider, so a
                # structured-output repair can retry the same model then rotate to
                # another OpenRouter model on this healthy provider. Reliability is
                # tracked separately per (provider, model).
                self.health_manager.record_failure(provider, cat)
            self.budget_manager.settle_reservation(
                handle=reservation,
                actual_input_tokens=0,
                actual_output_tokens=0,
                latency_ms=latency_ms,
                status="FAILED",
                failure_reason=str(exc),
            )
            error = InferenceFailedError(f"Inference failed on '{provider}/{model}': {exc}")
            error.category = cat  # type: ignore[attr-defined]
            raise error from exc

    def invoke_role(
        self,
        role: str,
        messages: Any,
        run_id: str = "default_run",
        ticker_or_market: str = "GLOBAL",
        est_input_tokens: int = 1500,
        mock_client_factory: Any | None = None,
        llm_transform: Any | None = None,
        purpose: str = "standard",
        **llm_kwargs,
    ) -> Any:
        """Select the best configured model for ``role`` and execute inference.

        Dynamic routing: ranks the configured/priced OpenRouter models (plus any
        keyed-up fallback providers) and invokes the best one. Fails closed with
        a structured ``SurvivorError`` (incl. ``UnknownPriceError`` /
        ``UNKNOWN_MODEL_PRICE``) if no eligible priced model exists. When
        ``purpose == "structured_output"`` the healthy/priced pool is limited to
        configured OpenRouter models and a failed structured output is retried on
        the SAME approved model once before ModelRouter falls back to the next
        configured, priced OpenRouter model allowed for that role.
        """
        pool, skipped = self._ordered_pool(role, purpose)
        if not pool:
            raise self._pool_empty_error(skipped)

        max_output_tokens = llm_kwargs.get("max_tokens") or self.policy.get_role_max_tokens(role)

        # 0. Global cost governor — hard caps BEFORE any provider request:
        #    kill switch, per-call input token cap, output clamp, call-rate
        #    caps (per minute / per hour), and the daily USD budget. Governor
        #    violations raise immediately (fail-safe, NO retry of the block).
        #    The ledger backing the governor is the live budget ledger.
        from tradingagents.llm_clients.cost_governor import CostGovernor

        governor = CostGovernor(ledger=self.budget_manager.ledger)
        primary_provider, primary_model = pool[0]
        clamped_output = governor.pre_call_check(
            provider=primary_provider,
            model=primary_model,
            est_input_tokens=est_input_tokens,
            est_output_tokens=max_output_tokens,
        )
        llm_kwargs["max_tokens"] = clamped_output

        last_error: Exception | None = None
        attempted_errors: list[str] = []
        idx = 0
        # Structured-output repair: retry the SAME approved model once before
        # rotating to the next configured/priced OpenRouter model for the role.
        retried_once = dict.fromkeys(pool, False)

        while idx < len(pool):
            provider, model = pool[idx]
            pair = (provider, model)
            try:
                response = self._invoke_one(
                    provider,
                    model,
                    role,
                    messages,
                    run_id,
                    ticker_or_market,
                    est_input_tokens,
                    clamped_output,
                    mock_client_factory,
                    llm_transform,
                    llm_kwargs,
                )
                if purpose == "structured_output":
                    self._record_structured(provider, model, True)
                return response
            except SurvivorError as exc:
                last_error = exc
                attempted_errors.append(f"{provider}/{model}: {str(exc)[:160]}")
                cat = getattr(exc, "category", None)

                if purpose == "structured_output" and cat == ProviderErrorCategory.INVALID_RESPONSE:
                    self._record_structured(provider, model, False)
                    if not retried_once[pair]:
                        retried_once[pair] = True
                        logger.info(
                            "Structured-output repair: retrying '%s/%s' once.", provider, model
                        )
                        continue  # retry SAME approved model once
                    logger.info(
                        "Structured-output repair: '%s/%s' failed twice; rotating within OpenRouter.",
                        provider,
                        model,
                    )
                    idx += 1
                else:
                    idx += 1  # normal health/budget/non-structured fallback

        if last_error:
            summary = "; ".join(attempted_errors)
            last_error = type(last_error)(f"{last_error} | attempted: {summary}")
            raise last_error
        raise SurvivorError(
            "No acceptable healthy provider route found.", status=RouteStatus.NO_PROVIDER
        )
