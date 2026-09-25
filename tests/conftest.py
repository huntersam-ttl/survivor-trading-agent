"""Shared pytest fixtures that prevent CI hangs when API keys are absent."""

import os
from unittest.mock import MagicMock, patch

import pytest


def pytest_configure(config):
    for marker in ("unit", "integration", "smoke"):
        config.addinivalue_line("markers", f"{marker}: {marker}-level tests")


_API_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_CN_API_KEY",
    "ZHIPU_API_KEY",
    "ZHIPU_CN_API_KEY",
    "MINIMAX_API_KEY",
    "MINIMAX_CN_API_KEY",
    "OPENROUTER_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
)


@pytest.fixture(autouse=True)
def _dummy_api_keys(monkeypatch):
    for env_var in _API_KEY_ENV_VARS:
        # `or` not a .get default: an env var present but empty (e.g. a key left
        # blank in a .env copied from .env.example) must still get the placeholder.
        monkeypatch.setenv(env_var, os.environ.get(env_var) or "placeholder")


@pytest.fixture(autouse=True)
def _disable_local_live_fallback_in_unit_tests(monkeypatch):
    """Keep the operator's local failover opt-in out of hermetic tests."""
    monkeypatch.setenv("SURVIVOR_OLLAMA_FALLBACK_ENABLED", "false")


@pytest.fixture(autouse=True)
def _isolate_survivor_runtime(tmp_path, monkeypatch):
    """Keep every survivor default mutable path inside this test's temp root."""
    root = tmp_path / "survivor-runtime"
    monkeypatch.setenv("SURVIVOR_RUNTIME_DIR", str(root))

    from tradingagents.llm_clients import usage_ledger
    from tradingagents.survivor.autonomy import halt, lock, state
    from tradingagents.survivor.evaluation import store
    from tradingagents.survivor.execution import ledger
    from tradingagents.survivor.ops import health, logging_utils, trial

    monkeypatch.setattr(halt, "HALT_DIR", str(root))
    monkeypatch.setattr(halt, "HALT_PATH", str(root / "HALT"))
    monkeypatch.setattr(lock, "LOCK_DIR", str(root))
    monkeypatch.setattr(lock, "LOCK_PATH", str(root / "cycle.lock"))
    monkeypatch.setattr(state, "DEFAULT_RUNTIME_DB_PATH", str(root / "runtime.db"))
    monkeypatch.setattr(ledger, "DEFAULT_PAPER_DB_PATH", str(root / "paper.db"))
    monkeypatch.setattr(store, "DEFAULT_EVALUATION_DB_PATH", str(root / "evaluation.db"))
    monkeypatch.setattr(usage_ledger, "_DEFAULT_DB_DIR", str(root))
    monkeypatch.setattr(usage_ledger, "_DEFAULT_DB_PATH", str(root / "usage.db"))
    monkeypatch.setattr(health, "SURVIVOR_DIR", str(root))
    monkeypatch.setattr(health, "HEARTBEAT_PATH", str(root / "heartbeat.json"))
    monkeypatch.setattr(trial, "TRIALS_DIR", str(root / "trials"))
    monkeypatch.setattr(logging_utils, "LOGS_DIR", str(root / "logs"))


@pytest.fixture(autouse=True)
def _isolate_config():
    """Reset the global dataflows config before and after each test.

    ``set_config`` merges (it never clears keys absent from the override), so a
    test that sets e.g. ``tool_vendors`` would otherwise leak into later tests
    and make routing behavior order-dependent. Replace the global outright so
    every test starts from a clean DEFAULT_CONFIG.
    """
    import copy

    import tradingagents.dataflows.config as config_module
    import tradingagents.default_config as default_config

    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    yield
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


@pytest.fixture()
def mock_llm_client():
    client = MagicMock()
    client.get_llm.return_value = MagicMock()
    with patch(
        "tradingagents.llm_clients.factory.create_llm_client",
        return_value=client,
    ):
        yield client
