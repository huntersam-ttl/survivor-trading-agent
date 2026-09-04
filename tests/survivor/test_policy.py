"""Unit tests for SurvivorPolicy and monetary conversion logic."""

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from tradingagents.survivor.policy import SurvivorPolicy, pounds_to_pence
from tradingagents.survivor.types import SurvivorMode


@pytest.mark.unit
def test_policy_defaults_and_immutability():
    policy = SurvivorPolicy()

    assert policy.mode == SurvivorMode.PAPER_ONLY
    assert policy.real_trading_enabled is False
    assert policy.wallet_enabled is False
    assert policy.broker_enabled is False
    assert policy.withdrawals_enabled is False
    assert policy.borrowing_enabled is False
    assert policy.leverage_enabled is False

    # Monthly caps in pence
    assert policy.openai_monthly_pence == 3000  # £30.00
    assert policy.deepseek_monthly_pence == 1000  # £10.00
    assert policy.minimax_monthly_pence == 1000  # £10.00
    assert policy.global_monthly_pence == 5000  # £50.00

    # Daily caps in pence
    assert policy.openai_daily_pence == 100  # £1.00
    assert policy.deepseek_daily_pence == 35  # £0.35
    assert policy.minimax_daily_pence == 35  # £0.35
    assert policy.global_daily_pence == 170  # £1.70

    # Immutability
    with pytest.raises(FrozenInstanceError):
        policy.real_trading_enabled = True  # type: ignore


@pytest.mark.unit
def test_pounds_to_pence_conversion():
    assert pounds_to_pence(30) == 3000
    assert pounds_to_pence("30") == 3000
    assert pounds_to_pence("30.00") == 3000
    assert pounds_to_pence("0.35") == 35
    assert pounds_to_pence("1.70") == 170
    assert pounds_to_pence(Decimal("10.00")) == 1000

    with pytest.raises(ValueError):
        pounds_to_pence("invalid_amount")


@pytest.mark.unit
def test_policy_from_env(monkeypatch):
    monkeypatch.setenv("SURVIVOR_OPENAI_MONTHLY_GBP", "40")
    monkeypatch.setenv("SURVIVOR_DEEPSEEK_DAILY_GBP", "0.50")
    monkeypatch.setenv("SURVIVOR_USD_GBP_RATE", "1.25")

    policy = SurvivorPolicy.from_env()

    assert policy.openai_monthly_pence == 4000
    assert policy.deepseek_daily_pence == 50
    assert policy.usd_gbp_rate == Decimal("1.25")
