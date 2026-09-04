"""Provider-aware ModelRouter with role-based routing, health checks, and budget enforcement."""

from __future__ import annotations

import logging
import time
from typing import Any

from tradingagents.llm_clients.budget import BudgetManager
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.pricing import get_model_price
from tradingagents.llm_clients.provider_health import ProviderHealthManager
from tradingagents.survivor.policy import SurvivorPolicy
from tradingagents.survivor.types import (
    AgentRole,
    BudgetExhaustedError,
    InferenceFailedError,
    ModelNotAllowedError,
    ProviderErrorCategory,
    RoleRoute,
    RouteStatus,
    SurvivorError,
    UnhealthyProviderError,
    UnknownPriceError,
)

logger = logging.getLogger(__name__)


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
    """Provider-aware router that selects healthy, budget-authorized models for agent roles."""

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
        self.role_routes = role_routes if role_routes is not None else dict(DEFAULT_ROLE_ROUTES)

    def resolve_routes(self, role: str) -> list[tuple[str, str]]:
        route = self.role_routes.get(role)
        if not route:
            raise ModelNotAllowedError(f"No model route configured for agent role '{role}'.")
        return route.get_all_routes()

    def invoke_role(
        self,
        role: str,
        messages: Any,
        run_id: str = "default_run",
        ticker_or_market: str = "GLOBAL",
        est_input_tokens: int = 1500,
        mock_client_factory: Any | None = None,
        llm_transform: Any | None = None,
        **llm_kwargs,
    ) -> Any:
        """Find the cheapest healthy, budget-authorized route for role and execute inference.

        Fails closed with a structured SurvivorError if no route is available.
        ``llm_transform`` (optional) is applied to the routed client before the
        call, e.g. to bind tools or structured-output schemas.
        """
        candidate_routes = self.resolve_routes(role)
        max_output_tokens = llm_kwargs.get("max_tokens") or self.policy.get_role_max_tokens(role)
        llm_kwargs["max_tokens"] = max_output_tokens

        last_error: Exception | None = None
        attempted_count = 0

        for provider, model in candidate_routes:
            attempted_count += 1
            # 1. Health check
            if not self.health_manager.is_healthy(provider):
                logger.warning("Skipping unhealthy provider '%s' for role '%s'.", provider, role)
                last_error = UnhealthyProviderError(f"Provider '{provider}' is currently unhealthy or in cooldown.")
                continue

            # 2. Price check (fail closed if unpriced)
            if get_model_price(provider, model) is None:
                logger.warning("Skipping unpriced model '%s/%s' for role '%s'.", provider, model, role)
                last_error = UnknownPriceError(f"Pricing for '{provider}/{model}' is unknown.")
                continue

            # 3. Budget Pre-Call Check & Reservation
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
            except BudgetExhaustedError as exc:
                logger.warning("Budget exhausted for '%s/%s' for role '%s': %s", provider, model, role, exc)
                last_error = exc
                continue
            except UnknownPriceError as exc:
                last_error = exc
                continue

            # 4. Invoke LLM with retries capped at policy.max_retries_ceiling
            start_time = time.time()
            try:
                if mock_client_factory:
                    client = mock_client_factory(provider=provider, model=model, **llm_kwargs)
                else:
                    client = create_llm_client(provider=provider, model=model, **llm_kwargs).get_llm()
                if llm_transform is not None:
                    client = llm_transform(client)

                response = client.invoke(messages)
                latency_ms = int((time.time() - start_time) * 1000)

                # Extract token usage if reported (LangChain usage_metadata shape,
                # with reasoning tokens nested under output_token_details)
                usage = getattr(response, "usage_metadata", None) or {}
                actual_in = usage.get("input_tokens", est_input_tokens)
                actual_out = usage.get("output_tokens", 200)
                output_details = usage.get("output_token_details") or {}
                reasoning = usage.get("reasoning_tokens", 0) or output_details.get("reasoning", 0)

                # Post-call settlement & health update
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
                logger.error("Inference call failed on '%s/%s' for role '%s': %s", provider, model, role, exc)

                # Categorize error
                err_msg = str(exc).lower()
                if "401" in err_msg or "unauthorized" in err_msg or "invalid api key" in err_msg or "auth" in err_msg:
                    cat = ProviderErrorCategory.AUTH_ERROR
                elif "429" in err_msg or "rate limit" in err_msg or "quota" in err_msg:
                    cat = ProviderErrorCategory.RATE_LIMIT
                elif "timeout" in err_msg:
                    cat = ProviderErrorCategory.TIMEOUT
                else:
                    cat = ProviderErrorCategory.SERVER_ERROR

                self.health_manager.record_failure(provider, cat)
                self.budget_manager.settle_reservation(
                    handle=reservation,
                    actual_input_tokens=0,
                    actual_output_tokens=0,
                    latency_ms=latency_ms,
                    status="FAILED",
                    failure_reason=str(exc),
                )
                last_error = InferenceFailedError(f"Inference failed on '{provider}/{model}': {exc}")
                # Try fallback route if available

        if last_error:
            raise last_error
        raise SurvivorError("No acceptable healthy provider route found.", status=RouteStatus.NO_PROVIDER)
