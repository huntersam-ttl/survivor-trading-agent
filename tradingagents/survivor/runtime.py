"""Runtime helpers for checking and accessing Survivor control plane state."""

from __future__ import annotations

import os

from .policy import SurvivorPolicy

_BOOL_TRUE = ("true", "1", "yes", "on")


def is_survivor_enabled(config: dict | None = None) -> bool:
    """Return whether the Survivor control plane is enabled.

    Precedence: an explicitly set TRADINGAGENTS_SURVIVOR_ENABLED env var wins
    (so the env var can enable Survivor even when a DEFAULT_CONFIG-derived
    dict carries ``survivor_enabled: False``); otherwise the config dict value
    is used; otherwise disabled.
    """
    env_val = os.environ.get("TRADINGAGENTS_SURVIVOR_ENABLED", "").strip().lower()
    if env_val:
        return env_val in _BOOL_TRUE
    if config and config.get("survivor_enabled") is not None:
        return bool(config["survivor_enabled"])
    return False


_GLOBAL_POLICY: SurvivorPolicy | None = None


def get_policy(config: dict | None = None) -> SurvivorPolicy:
    """Return current SurvivorPolicy instance."""
    global _GLOBAL_POLICY
    if config and "survivor_policy" in config and isinstance(config["survivor_policy"], SurvivorPolicy):
        return config["survivor_policy"]
    if _GLOBAL_POLICY is None:
        _GLOBAL_POLICY = SurvivorPolicy.from_env()
    return _GLOBAL_POLICY


def reset_policy() -> None:
    """Reset global policy cache (used in tests)."""
    global _GLOBAL_POLICY
    _GLOBAL_POLICY = None
