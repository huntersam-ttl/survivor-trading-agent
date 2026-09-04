"""Deterministic, concurrency-safe BudgetManager with reservation semantics."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from tradingagents.llm_clients.pricing import calculate_cost
from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger
from tradingagents.survivor.policy import SurvivorPolicy
from tradingagents.survivor.types import BudgetExhaustedError, ReservationHandle


class BudgetManager:
    """Manages pre-call authorization, concurrency-safe reservation, and post-call settlement."""

    def __init__(self, policy: SurvivorPolicy, ledger: InferenceUsageLedger | None = None):
        self.policy = policy
        self.ledger = ledger or InferenceUsageLedger()
        self._lock = threading.Lock()
        # Active pending reservations: reservation_id -> ReservationHandle
        self._pending_reservations: dict[str, ReservationHandle] = {}

    def get_pending_reserved_pence(
        self, provider: str | None = None, date_str: str | None = None, month_str: str | None = None
    ) -> int:
        """Sum estimated pence of pending reservations matching date/month/provider filters."""
        now_utc = datetime.now(timezone.utc)
        cur_date = date_str or now_utc.strftime("%Y-%m-%d")
        cur_month = month_str or now_utc.strftime("%Y-%m")

        total = 0
        p_target = provider.lower() if provider else None

        for handle in self._pending_reservations.values():
            if p_target and handle.provider.lower() != p_target:
                continue
            # Filter by date or month
            handle_date = handle.timestamp_utc[:10]
            handle_month = handle.timestamp_utc[:7]
            if date_str and handle_date != cur_date:
                continue
            if month_str and handle_month != cur_month:
                continue
            total += handle.est_gbp_pence
        return total

    def authorize_and_reserve(
        self,
        provider: str,
        model: str,
        agent_role: str,
        est_input_tokens: int,
        est_output_tokens: int,
        run_id: str = "default_run",
        ticker_or_market: str = "GLOBAL",
        timestamp_utc: str | None = None,
    ) -> ReservationHandle:
        """PRE-CALL check: Verify limits and reserve budget atomically under lock."""
        ts = timestamp_utc or datetime.now(timezone.utc).isoformat()
        cur_date = ts[:10]
        cur_month = ts[:7]

        # Calculate projected cost (raises UnknownPriceError if model unpriced or FX rate missing)
        cost_est = calculate_cost(
            provider=provider,
            model=model,
            input_tokens=est_input_tokens,
            output_tokens=est_output_tokens,
            usd_gbp_rate=self.policy.usd_gbp_rate,
        )
        projected_pence = cost_est.gbp_cost_pence

        with self._lock:
            # Check 1: Provider Monthly
            p_monthly_spent = self.ledger.get_monthly_spend_pence(provider, cur_month)
            p_monthly_reserved = self.get_pending_reserved_pence(provider=provider, month_str=cur_month)
            p_monthly_limit = self.policy.get_provider_monthly_budget(provider)

            if p_monthly_spent + p_monthly_reserved + projected_pence > p_monthly_limit:
                raise BudgetExhaustedError(
                    f"Provider '{provider}' monthly budget exhausted. "
                    f"Spent: {p_monthly_spent}p, Reserved: {p_monthly_reserved}p, "
                    f"Projected: {projected_pence}p > Limit: {p_monthly_limit}p.",
                    is_daily=False,
                )

            # Check 2: Provider Daily
            p_daily_spent = self.ledger.get_daily_spend_pence(provider, cur_date)
            p_daily_reserved = self.get_pending_reserved_pence(provider=provider, date_str=cur_date)
            p_daily_limit = self.policy.get_provider_daily_budget(provider)

            if p_daily_spent + p_daily_reserved + projected_pence > p_daily_limit:
                raise BudgetExhaustedError(
                    f"Provider '{provider}' daily budget exhausted. "
                    f"Spent: {p_daily_spent}p, Reserved: {p_daily_reserved}p, "
                    f"Projected: {projected_pence}p > Limit: {p_daily_limit}p.",
                    is_daily=True,
                )

            # Check 3: Global Monthly
            g_monthly_spent = self.ledger.get_monthly_spend_pence(None, cur_month)
            g_monthly_reserved = self.get_pending_reserved_pence(provider=None, month_str=cur_month)
            g_monthly_limit = self.policy.global_monthly_pence

            if g_monthly_spent + g_monthly_reserved + projected_pence > g_monthly_limit:
                raise BudgetExhaustedError(
                    f"Global monthly budget exhausted. "
                    f"Spent: {g_monthly_spent}p, Reserved: {g_monthly_reserved}p, "
                    f"Projected: {projected_pence}p > Limit: {g_monthly_limit}p.",
                    is_daily=False,
                )

            # Check 4: Global Daily
            g_daily_spent = self.ledger.get_daily_spend_pence(None, cur_date)
            g_daily_reserved = self.get_pending_reserved_pence(provider=None, date_str=cur_date)
            g_daily_limit = self.policy.global_daily_pence

            if g_daily_spent + g_daily_reserved + projected_pence > g_daily_limit:
                raise BudgetExhaustedError(
                    f"Global daily budget exhausted. "
                    f"Spent: {g_daily_spent}p, Reserved: {g_daily_reserved}p, "
                    f"Projected: {projected_pence}p > Limit: {g_daily_limit}p.",
                    is_daily=True,
                )

            # Reservation succeeded -> create handle
            res_id = str(uuid.uuid4())
            handle = ReservationHandle(
                reservation_id=res_id,
                provider=provider.lower(),
                model=model,
                agent_role=agent_role,
                est_input_tokens=est_input_tokens,
                est_output_tokens=est_output_tokens,
                est_gbp_pence=projected_pence,
                timestamp_utc=ts,
                run_id=run_id,
                ticker_or_market=ticker_or_market,
            )
            self._pending_reservations[res_id] = handle
            return handle

    def settle_reservation(
        self,
        handle: ReservationHandle,
        actual_input_tokens: int,
        actual_output_tokens: int,
        actual_reasoning_tokens: int = 0,
        latency_ms: int = 0,
        status: str = "SUCCESS",
        failure_reason: str | None = None,
    ) -> Any:
        """POST-CALL accounting: Release reservation and record actual cost into usage ledger."""
        with self._lock:
            self._pending_reservations.pop(handle.reservation_id, None)

        if status == "SUCCESS":
            actual_cost = calculate_cost(
                provider=handle.provider,
                model=handle.model,
                input_tokens=actual_input_tokens,
                output_tokens=actual_output_tokens,
                usd_gbp_rate=self.policy.usd_gbp_rate,
            )
            native_minor = actual_cost.native_cost_minor
            native_currency = actual_cost.native_currency
            pence = actual_cost.gbp_cost_pence
        else:
            native_minor = 0
            native_currency = "USD"
            pence = 0

        return self.ledger.record_inference(
            run_id=handle.run_id,
            agent_role=handle.agent_role,
            provider=handle.provider,
            model=handle.model,
            ticker_or_market=handle.ticker_or_market,
            input_tokens=actual_input_tokens,
            output_tokens=actual_output_tokens,
            reasoning_tokens=actual_reasoning_tokens,
            native_cost_minor=native_minor,
            native_currency=native_currency,
            gbp_cost_pence=pence,
            status=status,
            failure_reason=failure_reason,
            latency_ms=latency_ms,
            timestamp_utc=handle.timestamp_utc,
        )

    def release_reservation(self, handle: ReservationHandle) -> None:
        """Release reservation without recording a charge (e.g. pre-call abort)."""
        with self._lock:
            self._pending_reservations.pop(handle.reservation_id, None)
