"""Types, enums, exceptions, and data structures for the Survivor control plane."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SurvivorMode(str, Enum):
    PAPER_ONLY = "PAPER_ONLY"


class AgentRole(str, Enum):
    QUICK_ANALYST = "quick_analyst"
    MARKET_ANALYST = "market_analyst"
    SOCIAL_ANALYST = "social_analyst"
    NEWS_ANALYST = "news_analyst"
    FUNDAMENTALS_ANALYST = "fundamentals_analyst"
    BULL_RESEARCHER = "bull_researcher"
    BEAR_RESEARCHER = "bear_researcher"
    RESEARCH_MANAGER = "research_manager"
    TRADER = "trader"
    AGGRESSIVE_RISK = "aggressive_risk"
    CONSERVATIVE_RISK = "conservative_risk"
    NEUTRAL_RISK = "neutral_risk"
    PORTFOLIO_MANAGER = "portfolio_manager"


class ProviderErrorCategory(str, Enum):
    RATE_LIMIT = "RATE_LIMIT"
    AUTH_ERROR = "AUTH_ERROR"
    TIMEOUT = "TIMEOUT"
    SERVER_ERROR = "SERVER_ERROR"
    INVALID_RESPONSE = "INVALID_RESPONSE"


class RouteStatus(str, Enum):
    SUCCESS = "SUCCESS"
    NO_PROVIDER = "NO_PROVIDER"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    DAILY_BUDGET_EXHAUSTED = "DAILY_BUDGET_EXHAUSTED"
    MODEL_NOT_ALLOWED = "MODEL_NOT_ALLOWED"
    PROVIDER_UNHEALTHY = "PROVIDER_UNHEALTHY"
    UNKNOWN_MODEL_PRICE = "UNKNOWN_MODEL_PRICE"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    FX_RATE_MISSING = "FX_RATE_MISSING"


class SurvivorError(Exception):
    """Base exception for Survivor control plane."""

    def __init__(self, message: str, status: RouteStatus = RouteStatus.INFERENCE_FAILED):
        super().__init__(message)
        self.status = status


class BudgetExhaustedError(SurvivorError):
    """Raised when monthly or daily budget ceiling is reached."""

    def __init__(self, message: str, is_daily: bool = False):
        status = RouteStatus.DAILY_BUDGET_EXHAUSTED if is_daily else RouteStatus.BUDGET_EXHAUSTED
        super().__init__(message, status=status)
        self.is_daily = is_daily


class ModelNotAllowedError(SurvivorError):
    """Raised when model or provider is not in the allowlist for a role."""

    def __init__(self, message: str):
        super().__init__(message, status=RouteStatus.MODEL_NOT_ALLOWED)


class UnhealthyProviderError(SurvivorError):
    """Raised when a provider is in cooldown or disabled due to errors."""

    def __init__(self, message: str):
        super().__init__(message, status=RouteStatus.PROVIDER_UNHEALTHY)


class UnknownPriceError(SurvivorError):
    """Raised when pricing is not configured for a model."""

    def __init__(self, message: str):
        super().__init__(message, status=RouteStatus.UNKNOWN_MODEL_PRICE)


class InferenceFailedError(SurvivorError):
    """Raised when inference execution fails."""

    def __init__(self, message: str):
        super().__init__(message, status=RouteStatus.INFERENCE_FAILED)


@dataclass(frozen=True)
class RoleRoute:
    """Configured route choices for an agent role."""

    role: AgentRole
    preferred: list[tuple[str, str]]  # list of (provider, model)
    fallback: list[tuple[str, str]] = None  # list of (provider, model)

    def get_all_routes(self) -> list[tuple[str, str]]:
        routes = list(self.preferred)
        if self.fallback:
            routes.extend(self.fallback)
        return routes


@dataclass(frozen=True)
class ReservationHandle:
    """Handle for a pre-call budget authorization reservation."""

    reservation_id: str
    provider: str
    model: str
    agent_role: str
    est_input_tokens: int
    est_output_tokens: int
    est_gbp_pence: int
    timestamp_utc: str
    run_id: str
    ticker_or_market: str


@dataclass(frozen=True)
class InferenceRecord:
    """Recorded inference call for ledger and accounting."""

    id: str
    run_id: str
    timestamp_utc: str
    agent_role: str
    provider: str
    model: str
    ticker_or_market: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    native_cost_minor: int
    native_currency: str
    gbp_cost_pence: int
    status: str
    failure_reason: str | None
    latency_ms: int
