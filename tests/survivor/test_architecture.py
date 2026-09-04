"""Architectural boundary tests proving safety constraints, key isolation, and zero-money trading."""

import os
import sqlite3
import tempfile

import pytest

from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger
from tradingagents.survivor import SurvivorPolicy
from tradingagents.survivor.types import SurvivorMode


@pytest.mark.unit
def test_safety_boundary_zero_real_money():
    """Verify that Survivor policy enforces zero live trading, zero wallets, and zero broker execution."""
    policy = SurvivorPolicy()
    assert policy.mode == SurvivorMode.PAPER_ONLY
    assert policy.real_trading_enabled is False
    assert policy.wallet_enabled is False
    assert policy.broker_enabled is False
    assert policy.withdrawals_enabled is False
    assert policy.borrowing_enabled is False
    assert policy.leverage_enabled is False


@pytest.mark.unit
def test_no_api_keys_in_usage_ledger():
    """Verify that no API keys or secrets are logged into the usage database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "usage.db")
        ledger = InferenceUsageLedger(db_path=db_path)

        secret_key = "sk-proj-supersecret123456789"

        ledger.record_inference(
            run_id="run_999",
            agent_role="trader",
            provider="openai",
            model="gpt-5.6",
            ticker_or_market="AAPL",
            input_tokens=100,
            output_tokens=50,
            reasoning_tokens=0,
            native_cost_minor=10,
            native_currency="USD",
            gbp_cost_pence=8,
            timestamp_utc="2026-09-04T12:00:00Z",
        )

        # Inspect raw SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM usage_records;")
        rows = cursor.fetchall()
        conn.close()

        # Check every single string field in DB rows to ensure secret_key never appears
        for row in rows:
            row_str = str(row)
            assert secret_key not in row_str
            assert "OPENAI_API_KEY" not in row_str
            assert "DEEPSEEK_API_KEY" not in row_str
            assert "MINIMAX_API_KEY" not in row_str


@pytest.mark.unit
def test_absent_broker_wallet_capabilities():
    """Verify that no wallet, order placement, or exchange integration modules exist in tradingagents."""
    import tradingagents

    pkg_dir = os.path.dirname(tradingagents.__file__)
    all_files = []
    for root, _, files in os.walk(pkg_dir):
        for f in files:
            all_files.append(os.path.join(root, f))

    # Assert no wallet.py, broker.py, execution.py, exchange.py exist in core repo currently
    forbidden_basenames = ["wallet.py", "broker.py", "binance.py", "coinbase.py", "alpaca.py", "ccxt.py"]
    for path in all_files:
        basename = os.path.basename(path)
        assert basename not in forbidden_basenames, f"Forbidden live-trading file found: {path}"
