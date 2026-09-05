"""Global LLM cost governor — ONE hard cost-control layer for every
Survivor-guarded model call.

All autonomous/background model calls flow through ModelRouter.invoke_role,
which consults this governor BEFORE any provider request:

1. kill switch      LLM_AUTONOMY_ENABLED=false  -> zero autonomous calls
2. input cap        LLM_MAX_INPUT_TOKENS_PER_CALL  (hard block oversized calls)
3. output cap       LLM_MAX_OUTPUT_TOKENS_PER_CALL (clamps max_tokens)
4. call-rate caps   LLM_MAX_CALLS_PER_MINUTE / LLM_MAX_CALLS_PER_HOUR
5. daily USD budget AUTONOMOUS_LLM_DAILY_BUDGET_USD (estimated + settled)

When any limit is hit the governor raises GovernorExhaustedError /
AutonomyLLMDisabledError (both SurvivorError). The caller fails safe: the
cycle records an explicit candidate failure and continues deterministically.
There is NO retry of a governor-blocked call — the block itself is the
retry-storm kill switch.

All counters are persisted in the usage ledger (durable across processes);
nothing here ever stores or prints prompt contents or credentials.
"""

from __future__ import annotations

import os
from collections import deque
from datetime import datetime, timedelta, timezone

from tradingagents.llm_clients.pricing import get_model_price
from tradingagents.survivor.types import (
    AutonomyLLMDisabledError,
    GovernorExhaustedError,
)

_DEFAULT_MAX_INPUT_TOKENS = 12000
_DEFAULT_MAX_OUTPUT_TOKENS = 2000
_DEFAULT_CALLS_PER_MINUTE = 10
_DEFAULT_CALLS_PER_HOUR = 60
_DEFAULT_DAILY_BUDGET_USD = 0.50


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _usd_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


class CostGovernor:
    """Hard pre-call cost controls for EVERY guarded model call.

    Config (env, all overridable):
        LLM_AUTONOMY_ENABLED            default "true";  false = kill switch
        LLM_MAX_INPUT_TOKENS_PER_CALL   default 12000
        LLM_MAX_OUTPUT_TOKENS_PER_CALL  default 2000
        LLM_MAX_CALLS_PER_MINUTE        default 10
        LLM_MAX_CALLS_PER_HOUR          default 60
        AUTONOMOUS_LLM_DAILY_BUDGET_USD default 0.50
    """

    def __init__(self, ledger=None):
        self.ledger = ledger
        self.enabled = os.environ.get("LLM_AUTONOMY_ENABLED", "true").strip().lower() \
            not in ("false", "0", "no", "off")
        self.max_input_tokens = _int_env("LLM_MAX_INPUT_TOKENS_PER_CALL", _DEFAULT_MAX_INPUT_TOKENS)
        self.max_output_tokens = _int_env("LLM_MAX_OUTPUT_TOKENS_PER_CALL", _DEFAULT_MAX_OUTPUT_TOKENS)
        self.calls_per_minute = _int_env("LLM_MAX_CALLS_PER_MINUTE", _DEFAULT_CALLS_PER_MINUTE)
        self.calls_per_hour = _int_env("LLM_MAX_CALLS_PER_HOUR", _DEFAULT_CALLS_PER_HOUR)
        self.daily_budget_usd = _usd_env("AUTONOMOUS_LLM_DAILY_BUDGET_USD", _DEFAULT_DAILY_BUDGET_USD)
        # In-process sliding window (durable cross-process counts come from the ledger)
        self._recent: deque[datetime] = deque()

    # -- individual hard checks ------------------------------------------------

    def check_enabled(self) -> None:
        """Kill switch: autonomous/background LLM calls can be disabled globally."""
        if not self.enabled:
            raise AutonomyLLMDisabledError(
                "LLM_AUTONOMY_ENABLED=false: autonomous model calls are globally disabled "
                "(kill switch). Deterministic collection/analysis continues unaffected."
            )

    def check_input_tokens(self, est_input_tokens: int) -> None:
        """Hard-block obviously oversized contexts BEFORE any provider request."""
        if est_input_tokens > self.max_input_tokens:
            raise GovernorExhaustedError(
                f"input token cap exceeded: estimated {est_input_tokens} > "
                f"LLM_MAX_INPUT_TOKENS_PER_CALL={self.max_input_tokens}. "
                "Refusing to send an oversized context (fail-safe, no retry)."
            )

    def clamp_output_tokens(self, max_output_tokens: int) -> int:
        return min(max_output_tokens, self.max_output_tokens)

    def _record_local(self, now: datetime) -> None:
        self._recent.append(now)
        horizon = now - timedelta(hours=1)
        while self._recent and self._recent[0] < horizon:
            self._recent.popleft()

    def _check_local_rate(self, now: datetime) -> None:
        minute_floor = now - timedelta(minutes=1)
        last_minute = sum(1 for t in self._recent if t >= minute_floor)
        if last_minute >= self.calls_per_minute:
            raise GovernorExhaustedError(
                f"call rate cap exceeded: {last_minute} calls in the last minute >= "
                f"LLM_MAX_CALLS_PER_MINUTE={self.calls_per_minute}"
            )
        if len(self._recent) >= self.calls_per_hour:
            raise GovernorExhaustedError(
                f"call rate cap exceeded: {len(self._recent)} calls in the last hour >= "
                f"LLM_MAX_CALLS_PER_HOUR={self.calls_per_hour}"
            )

    def _check_daily_budget(self, provider: str, model: str,
                            est_input_tokens: int, est_output_tokens: int) -> None:
        """Daily USD budget: today's SETTLED provider spend (from the ledger, in
        the provider's own currency) + this call's estimated cost."""
        if self.ledger is None:
            return
        price = get_model_price(provider, model)
        if price is None:
            return  # unpriced models fail closed in the router anyway
        est_usd = float(
            est_input_tokens * price.input_cost_per_1m
            + est_output_tokens * price.output_cost_per_1m
        ) / 1_000_000.0
        spent_today_usd = self.ledger.get_native_spend_today(provider)
        if spent_today_usd + est_usd > self.daily_budget_usd:
            raise GovernorExhaustedError(
                f"daily LLM budget exhausted: {provider} settled spend today "
                f"${spent_today_usd:.4f} + estimated ${est_usd:.4f} >= "
                f"AUTONOMOUS_LLM_DAILY_BUDGET_USD=${self.daily_budget_usd:.2f}",
                is_daily=True,
            )

    # -- one-shot pre-call gate --------------------------------------------------

    def pre_call_check(self, provider: str, model: str,
                       est_input_tokens: int, est_output_tokens: int) -> int:
        """Run every hard limit for one call. Returns the (possibly clamped)
        max output tokens. Raises GovernorExhaustedError / AutonomyLLMDisabledError."""
        self.check_enabled()
        self.check_input_tokens(est_input_tokens)
        clamped = self.clamp_output_tokens(est_output_tokens)
        now = datetime.now(timezone.utc)
        self._record_local(now)
        self._check_local_rate(now)
        self._check_daily_budget(provider, model, est_input_tokens, clamped)
        return clamped
