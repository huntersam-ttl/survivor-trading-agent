"""Deterministic provider health tracker and circuit breaker."""

from __future__ import annotations

import time
from dataclasses import dataclass

from tradingagents.survivor.types import ProviderErrorCategory


@dataclass
class ProviderHealthState:
    provider: str
    consecutive_failures: int = 0
    last_failure_ts: float | None = None
    last_success_ts: float | None = None
    cooldown_until_ts: float | None = None
    last_error_category: ProviderErrorCategory | None = None
    disabled_permanently: bool = False

    def is_healthy(self, now: float | None = None) -> bool:
        if self.disabled_permanently:
            return False
        current_time = now if now is not None else time.time()
        return not bool(self.cooldown_until_ts and current_time < self.cooldown_until_ts)


class ProviderHealthManager:
    """Manages circuit-breaking health state for all registered LLM providers."""

    def __init__(self, rate_limit_cooldown_sec: float = 60.0, server_error_cooldown_sec: float = 120.0):
        self.rate_limit_cooldown_sec = rate_limit_cooldown_sec
        self.server_error_cooldown_sec = server_error_cooldown_sec
        self._states: dict[str, ProviderHealthState] = {}

    def get_state(self, provider: str) -> ProviderHealthState:
        p = provider.lower()
        if p not in self._states:
            self._states[p] = ProviderHealthState(provider=p)
        return self._states[p]

    def is_healthy(self, provider: str, now: float | None = None) -> bool:
        return self.get_state(provider).is_healthy(now=now)

    def record_success(self, provider: str, now: float | None = None) -> None:
        current_time = now if now is not None else time.time()
        state = self.get_state(provider)
        state.consecutive_failures = 0
        state.last_success_ts = current_time
        state.cooldown_until_ts = None

    def record_failure(
        self,
        provider: str,
        error_category: ProviderErrorCategory,
        now: float | None = None,
    ) -> None:
        current_time = now if now is not None else time.time()
        state = self.get_state(provider)
        state.consecutive_failures += 1
        state.last_failure_ts = current_time
        state.last_error_category = error_category

        if error_category == ProviderErrorCategory.AUTH_ERROR:
            # Auth errors permanently disable provider until process restart
            state.disabled_permanently = True
        elif error_category == ProviderErrorCategory.RATE_LIMIT:
            # Immediate rate limit cooldown
            state.cooldown_until_ts = current_time + self.rate_limit_cooldown_sec
        elif error_category in (
            ProviderErrorCategory.TIMEOUT,
            ProviderErrorCategory.SERVER_ERROR,
            ProviderErrorCategory.INVALID_RESPONSE,
        ) and state.consecutive_failures >= 2:
            # Cooldown triggered after 2 consecutive server/timeout errors
            state.cooldown_until_ts = current_time + self.server_error_cooldown_sec

    def reset_provider(self, provider: str) -> None:
        p = provider.lower()
        self._states[p] = ProviderHealthState(provider=p)

    def reset_all(self) -> None:
        self._states.clear()
