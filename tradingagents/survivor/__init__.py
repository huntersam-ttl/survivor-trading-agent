"""Survivor Control Plane package for TradingAgents.

Provides policy enforcement, role-based model routing, deterministic budget manager,
concurrency-safe inference usage ledger, and provider health tracking.
"""

from .policy import SurvivorPolicy, pounds_to_pence
from .runtime import get_policy, is_survivor_enabled, reset_policy
from .types import (
    AgentRole,
    BudgetExhaustedError,
    InferenceFailedError,
    InferenceRecord,
    ModelNotAllowedError,
    ProviderErrorCategory,
    ReservationHandle,
    RoleRoute,
    RouteStatus,
    SurvivorError,
    SurvivorMode,
    UnhealthyProviderError,
    UnknownPriceError,
)

__all__ = [
    "AgentRole",
    "BudgetExhaustedError",
    "InferenceFailedError",
    "InferenceRecord",
    "ModelNotAllowedError",
    "ProviderErrorCategory",
    "ReservationHandle",
    "RoleRoute",
    "RouteStatus",
    "SurvivorError",
    "SurvivorMode",
    "SurvivorPolicy",
    "UnhealthyProviderError",
    "UnknownPriceError",
    "get_policy",
    "is_survivor_enabled",
    "pounds_to_pence",
    "reset_policy",
]
