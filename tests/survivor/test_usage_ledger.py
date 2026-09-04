"""Unit tests for SQLite InferenceUsageLedger."""

import os
import tempfile

import pytest

from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger


@pytest.mark.unit
def test_usage_ledger_record_and_aggregate():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_usage.db")
        ledger = InferenceUsageLedger(db_path=db_path)

        # Record inference calls on 2026-09-04
        r1 = ledger.record_inference(
            run_id="run_101",
            agent_role="market_analyst",
            provider="deepseek",
            model="deepseek-chat",
            ticker_or_market="NVDA",
            input_tokens=1000,
            output_tokens=500,
            reasoning_tokens=0,
            native_cost_minor=50,
            native_currency="USD",
            gbp_cost_pence=40,
            timestamp_utc="2026-09-04T10:00:00Z",
        )
        assert r1.id is not None
        assert r1.gbp_cost_pence == 40

        r2 = ledger.record_inference(
            run_id="run_101",
            agent_role="trader",
            provider="openai",
            model="gpt-5.6",
            ticker_or_market="NVDA",
            input_tokens=2000,
            output_tokens=800,
            reasoning_tokens=100,
            native_cost_minor=300,
            native_currency="USD",
            gbp_cost_pence=230,
            timestamp_utc="2026-09-04T10:05:00Z",
        )
        assert r2.gbp_cost_pence == 230

        # Query daily spend
        assert ledger.get_daily_spend_pence("deepseek", "2026-09-04") == 40
        assert ledger.get_daily_spend_pence("openai", "2026-09-04") == 230
        assert ledger.get_daily_spend_pence(None, "2026-09-04") == 270

        # Query monthly spend
        assert ledger.get_monthly_spend_pence("deepseek", "2026-09") == 40
        assert ledger.get_monthly_spend_pence("openai", "2026-09") == 230
        assert ledger.get_monthly_spend_pence(None, "2026-09") == 270

        # Query summary for run
        summary = ledger.get_run_usage_summary("run_101")
        assert summary["run_id"] == "run_101"
        assert summary["total_calls"] == 2
        assert summary["total_cost_pence"] == 270
        assert summary["total_cost_gbp"] == "£2.70"
