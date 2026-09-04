"""Strategy versioning and deterministic configuration hashing.

Evaluation results from materially different configurations must never be
silently mixed: every record carries strategy_version + config_hash, and the
evaluator separates/reports mismatches.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from tradingagents.survivor.evaluation.types import STRATEGY_VERSION

# The strategy-relevant configuration keys. A change to ANY of these (values
# or set) changes the config hash and therefore the evaluation identity.
STRATEGY_CONFIG_KEYS = (
    "min_edge_bps",
    "max_single_position_pence",
    "max_total_exposure_pence",
    "daily_loss_limit_pence",
    "max_drawdown_bps",
    "max_spread_bps",
    "min_liquidity_minor",
    "min_probability_bps",
    "max_probability_bps",
    "max_research_candidates_per_cycle",
    "rank_liquidity",
    "rank_spread",
    "rank_volume",
    "rank_time",
    "fill_fraction_of_liquidity",
    "model_routing",
    "slippage_model",
)


def strategy_config_hash(config: dict[str, Any] | None = None) -> str:
    """Deterministic hash over the strategy-relevant config subset."""
    config = config or {}
    payload = {key: config.get(key) for key in STRATEGY_CONFIG_KEYS}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def strategy_identity(config: dict[str, Any] | None = None) -> tuple[str, str]:
    """(strategy_version, config_hash) for the current strategy configuration."""
    return STRATEGY_VERSION, strategy_config_hash(config)
