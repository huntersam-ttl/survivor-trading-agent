"""Unit tests for BudgetManager, boundary limits, and concurrency reservations."""

import os
import tempfile
import threading

import pytest

from tradingagents.llm_clients.budget import BudgetManager
from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger
from tradingagents.survivor.policy import SurvivorPolicy
from tradingagents.survivor.types import BudgetExhaustedError


@pytest.fixture
def temp_budget_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_budget.db")
        ledger = InferenceUsageLedger(db_path=db_path)
        policy = SurvivorPolicy(
            openai_monthly_pence=3000,   # £30.00
            deepseek_monthly_pence=1000,  # £10.00
            minimax_monthly_pence=1000,   # £10.00
            global_monthly_pence=5000,    # £50.00
            openai_daily_pence=3000,     # £30.00 (set high to isolate monthly test)
            deepseek_daily_pence=35,     # £0.35
            minimax_daily_pence=35,      # £0.35
            global_daily_pence=5000,     # £50.00
        )
        bm = BudgetManager(policy=policy, ledger=ledger)
        yield bm


@pytest.mark.unit
def test_budget_exact_boundary_monthly(temp_budget_manager):
    bm = temp_budget_manager
    ts = "2026-09-04T12:00:00Z"

    # Pre-fill ledger with 2999 pence spend for OpenAI
    bm.ledger.record_inference(
        run_id="r1",
        agent_role="trader",
        provider="openai",
        model="gpt-5.6",
        ticker_or_market="AAPL",
        input_tokens=100,
        output_tokens=100,
        reasoning_tokens=0,
        native_cost_minor=3000,
        native_currency="USD",
        gbp_cost_pence=2999,
        timestamp_utc=ts,
    )

    # Next projected cost = 1 pence -> Total = 3000 pence <= 3000 limit -> ALLOWED
    # (using mock estimation: 100 in, 100 out on gpt-4o-mini = ~1p)
    handle = bm.authorize_and_reserve(
        provider="openai",
        model="gpt-4o-mini",
        agent_role="quick_analyst",
        est_input_tokens=100,
        est_output_tokens=100,
        timestamp_utc=ts,
    )
    assert handle is not None
    assert handle.est_gbp_pence == 1

    # Settle the reservation
    bm.settle_reservation(handle, actual_input_tokens=100, actual_output_tokens=100)
    assert bm.ledger.get_monthly_spend_pence("openai", "2026-09") == 3000

    # Now spent = 3000 pence = Limit. Next projected cost = 1 pence -> EXCEEDS limit -> REJECTED
    with pytest.raises(BudgetExhaustedError) as exc_info:
        bm.authorize_and_reserve(
            provider="openai",
            model="gpt-4o-mini",
            agent_role="quick_analyst",
            est_input_tokens=100,
            est_output_tokens=100,
            timestamp_utc=ts,
        )
    assert "monthly budget exhausted" in str(exc_info.value)


@pytest.mark.unit
def test_budget_exact_boundary_daily(temp_budget_manager):
    bm = temp_budget_manager
    ts = "2026-09-04T12:00:00Z"

    # DeepSeek daily limit is 35 pence. Pre-fill 34 pence.
    bm.ledger.record_inference(
        run_id="r2",
        agent_role="news_analyst",
        provider="deepseek",
        model="deepseek-chat",
        ticker_or_market="BTC-USD",
        input_tokens=100,
        output_tokens=100,
        reasoning_tokens=0,
        native_cost_minor=50,
        native_currency="USD",
        gbp_cost_pence=34,
        timestamp_utc=ts,
    )

    # Spending 1 pence -> total 35p == 35p limit -> ALLOWED
    h = bm.authorize_and_reserve(
        provider="deepseek",
        model="deepseek-chat",
        agent_role="news_analyst",
        est_input_tokens=100,
        est_output_tokens=100,
        timestamp_utc=ts,
    )
    assert h is not None

    bm.settle_reservation(h, actual_input_tokens=100, actual_output_tokens=100)
    assert bm.ledger.get_daily_spend_pence("deepseek", "2026-09-04") == 35

    # Further spend -> REJECTED
    with pytest.raises(BudgetExhaustedError) as exc_info:
        bm.authorize_and_reserve(
            provider="deepseek",
            model="deepseek-chat",
            agent_role="news_analyst",
            est_input_tokens=100,
            est_output_tokens=100,
            timestamp_utc=ts,
        )
    assert exc_info.value.is_daily is True


@pytest.mark.unit
def test_budget_utc_date_rollover(temp_budget_manager):
    bm = temp_budget_manager
    day1 = "2026-09-04T23:59:00Z"
    day2 = "2026-09-05T00:01:00Z"

    # Fill day1 limit for DeepSeek (35p)
    bm.ledger.record_inference(
        run_id="r1",
        agent_role="news_analyst",
        provider="deepseek",
        model="deepseek-chat",
        ticker_or_market="ETH-USD",
        input_tokens=100,
        output_tokens=100,
        reasoning_tokens=0,
        native_cost_minor=50,
        native_currency="USD",
        gbp_cost_pence=35,
        timestamp_utc=day1,
    )

    # Day 1 is exhausted
    with pytest.raises(BudgetExhaustedError):
        bm.authorize_and_reserve(
            provider="deepseek",
            model="deepseek-chat",
            agent_role="news_analyst",
            est_input_tokens=100,
            est_output_tokens=100,
            timestamp_utc=day1,
        )

    # Day 2 is a new UTC date -> ALLOWED
    h = bm.authorize_and_reserve(
        provider="deepseek",
        model="deepseek-chat",
        agent_role="news_analyst",
        est_input_tokens=100,
        est_output_tokens=100,
        timestamp_utc=day2,
    )
    assert h is not None


@pytest.mark.unit
def test_budget_concurrency_reservation(temp_budget_manager):
    bm = temp_budget_manager
    ts = "2026-09-04T12:00:00Z"

    # Daily limit for MiniMax is 35 pence.
    # We spawn 10 concurrent threads each trying to reserve 10 pence.
    # Only 3 reservations (30p <= 35p) should succeed, 7 should raise BudgetExhaustedError.
    successful_handles = []
    failed_errors = []

    def worker():
        try:
            h = bm.authorize_and_reserve(
                provider="minimax",
                model="minimax-text-01",
                agent_role="quick_analyst",
                est_input_tokens=100000,  # ~10p
                est_output_tokens=50000,
                timestamp_utc=ts,
            )
            successful_handles.append(h)
        except BudgetExhaustedError as err:
            failed_errors.append(err)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Sum of successful reserved pence must be <= 35 pence
    total_reserved = sum(h.est_gbp_pence for h in successful_handles)
    assert total_reserved <= 35
    assert len(successful_handles) + len(failed_errors) == 10
