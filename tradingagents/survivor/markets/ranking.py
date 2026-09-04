"""Deterministic candidate ranking (0-100). No AI. Explicit configurable weights.

Tie-breaking is fully deterministic: (score desc, liquidity desc, market_id asc).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

from tradingagents.survivor.markets.types import Candidate, MarketSnapshot


@dataclass(frozen=True)
class RankingWeights:
    """Weights must sum to 1.0; each component contributes 0..weight*100."""

    liquidity: float = 0.30
    spread_quality: float = 0.20
    volume: float = 0.20
    time_remaining: float = 0.15
    probability_distance: float = 0.10
    metadata_completeness: float = 0.05
    # Reference points for normalization
    liquidity_ref_minor: int = 1000000   # $10,000 saturates the liquidity score
    volume_ref_minor: int = 500000       # $5,000 saturates the volume score
    time_ref_seconds: int = 86400        # 1 day to resolution saturates time score


def ranking_weights_from_env() -> RankingWeights:
    def _float(env: str, default: float) -> float:
        raw = os.environ.get(env)
        return float(raw) if raw else default

    return RankingWeights(
        liquidity=_float("SURVIVOR_RANK_LIQUIDITY", 0.30),
        spread_quality=_float("SURVIVOR_RANK_SPREAD", 0.20),
        volume=_float("SURVIVOR_RANK_VOLUME", 0.20),
        time_remaining=_float("SURVIVOR_RANK_TIME", 0.15),
        probability_distance=_float("SURVIVOR_RANK_PROBABILITY", 0.10),
        metadata_completeness=_float("SURVIVOR_RANK_METADATA", 0.05),
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_snapshot(snapshot: MarketSnapshot, weights: RankingWeights, now: datetime | None = None) -> float:
    """Deterministic 0-100 score."""
    current = now or datetime.now(timezone.utc)

    liquidity_score = _clamp01(
        (snapshot.liquidity.minor_units / weights.liquidity_ref_minor)
        if snapshot.liquidity else 0.0
    )
    if snapshot.bid is not None and snapshot.ask is not None and snapshot.ask > 0:
        spread_score = _clamp01(1.0 - (snapshot.ask - snapshot.bid) / snapshot.ask)
    else:
        spread_score = 0.0
    volume_score = _clamp01(
        (snapshot.volume_24h.minor_units / weights.volume_ref_minor)
        if snapshot.volume_24h else 0.0
    )
    if snapshot.close_time_utc:
        try:
            close = datetime.fromisoformat(snapshot.close_time_utc.replace("Z", "+00:00"))
            time_score = _clamp01((close - current).total_seconds() / weights.time_ref_seconds)
        except ValueError:
            time_score = 0.0
    else:
        time_score = 0.0
    prob = snapshot.market_probability_bps
    if prob is not None:
        distance = abs(prob - 5000) / 5000.0          # 0 at 50%, 1 at extremes
        probability_score = _clamp01(1.0 - distance)  # prefer non-extreme
    else:
        probability_score = 0.0
    completeness_fields = (
        snapshot.close_time_utc, snapshot.bid, snapshot.ask,
        snapshot.liquidity, snapshot.volume_24h, snapshot.question,
    )
    completeness = sum(1 for f in completeness_fields if f is not None and f != "") / len(completeness_fields)

    return round(100.0 * (
        weights.liquidity * liquidity_score
        + weights.spread_quality * spread_score
        + weights.volume * volume_score
        + weights.time_remaining * time_score
        + weights.probability_distance * probability_score
        + weights.metadata_completeness * completeness
    ), 4)


def rank_candidates(
    candidates: list[Candidate],
    weights: RankingWeights | None = None,
    now: datetime | None = None,
) -> list[Candidate]:
    """Return candidates ranked deterministically (score desc, liquidity desc, id asc)."""
    weights = weights or ranking_weights_from_env()
    # accept either snapshots or pre-wrapped candidates
    items = [c if isinstance(c, Candidate) else Candidate(snapshot=c) for c in candidates]
    scored = []
    for candidate in items:
        score = score_snapshot(candidate.snapshot, weights, now=now)
        scored.append((score, candidate.snapshot.liquidity.minor_units if candidate.snapshot.liquidity else 0, candidate.snapshot.market_id, score, candidate))
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return [
        Candidate(snapshot=candidate.snapshot, score=s, rank=i + 1)
        for i, (_, _, _, s, candidate) in enumerate(scored)
    ]
