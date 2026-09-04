"""Unit tests for ProviderHealthManager circuit breaking."""

import pytest

from tradingagents.llm_clients.provider_health import ProviderHealthManager
from tradingagents.survivor.types import ProviderErrorCategory


@pytest.mark.unit
def test_provider_health_auth_error_lockout():
    health = ProviderHealthManager()
    assert health.is_healthy("openai") is True

    health.record_failure("openai", ProviderErrorCategory.AUTH_ERROR)
    assert health.is_healthy("openai") is False

    # Success should not clear permanent auth failure until explicit reset
    health.record_success("openai")
    assert health.is_healthy("openai") is False

    health.reset_provider("openai")
    assert health.is_healthy("openai") is True


@pytest.mark.unit
def test_provider_health_rate_limit_cooldown():
    health = ProviderHealthManager(rate_limit_cooldown_sec=10.0)
    now = 1000.0

    assert health.is_healthy("deepseek", now=now) is True

    health.record_failure("deepseek", ProviderErrorCategory.RATE_LIMIT, now=now)
    assert health.is_healthy("deepseek", now=now + 5.0) is False
    assert health.is_healthy("deepseek", now=now + 11.0) is True


@pytest.mark.unit
def test_provider_health_consecutive_server_errors():
    health = ProviderHealthManager(server_error_cooldown_sec=20.0)
    now = 1000.0

    # 1st server error -> still healthy (no cooldown yet)
    health.record_failure("minimax", ProviderErrorCategory.SERVER_ERROR, now=now)
    assert health.is_healthy("minimax", now=now) is True

    # 2nd server error -> triggers cooldown
    health.record_failure("minimax", ProviderErrorCategory.SERVER_ERROR, now=now + 1.0)
    assert health.is_healthy("minimax", now=now + 2.0) is False
    assert health.is_healthy("minimax", now=now + 22.0) is True
