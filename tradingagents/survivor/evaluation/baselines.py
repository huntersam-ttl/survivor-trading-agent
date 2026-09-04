"""Deterministic baseline strategies. No cherry-picking, no LLM."""

from __future__ import annotations

import random

from tradingagents.survivor.evaluation.metrics import brier_score
from tradingagents.survivor.evaluation.types import PredictionRecord

MARKET_BASELINE = "MARKET_BASELINE"
NO_TRADE_BASELINE = "NO_TRADE_BASELINE"
RANDOM_DIRECTION_BASELINE = "RANDOM_DIRECTION_BASELINE"
SIMPLE_EDGE_BASELINE = "SIMPLE_EDGE_BASELINE"


def market_baseline_brier(records: list[PredictionRecord]) -> float:
    """Brier of the market probability at decision time (no AI adjustment)."""
    resolved = [r for r in records if r.outcome is not None]
    return brier_score([(r.market_probability, r.outcome) for r in resolved])  # type: ignore[list-item]


def market_baseline_net_pnl() -> int:
    """MARKET_BASELINE has zero informational edge over itself -> zero expected P/L."""
    return 0


def no_trade_baseline_net_pnl(trades: list) -> int:
    """NO_TRADE_BASELINE: never trade -> zero P/L (and zero AI cost)."""
    return 0


def random_direction_baseline_brier(
    records: list[PredictionRecord], seed: int = 42
) -> float:
    """Deterministic seeded random forecast benchmark (research diagnostic only)."""
    rng = random.Random(seed)
    resolved = [r for r in records if r.outcome is not None]
    forecasts = [(rng.random(), r.outcome) for r in resolved]  # type: ignore[list-item]
    return brier_score(forecasts)


def simple_edge_baseline_net_pnl(
    market_probabilities_bps: list[int], threshold_bps: int = 500, payoff_pence: int = 100
) -> int:
    """SIMPLE_EDGE_BASELINE: mechanically buy every market whose MARKET
    probability leaves an apparent >threshold edge with no AI research,
    resolving at face value. Deterministic; P/L per trade = payoff - entry cost."""
    pnl = 0
    for bps in market_probabilities_bps:
        if bps <= 10000 - threshold_bps:
            pnl += payoff_pence - bps // 100
    return pnl
