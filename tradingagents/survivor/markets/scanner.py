"""Deterministic MarketScanner: fetch -> normalize -> filter -> rank -> top-N.

ZERO LLM CALLS by design. The scanner imports no LLM module and never will;
``tests`` enforce this structurally via AST inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from tradingagents.survivor.markets.adapter import MarketAdapter
from tradingagents.survivor.markets.filters import ScanLimits, filter_market, scan_limits_from_env
from tradingagents.survivor.markets.ranking import (
    RankingWeights,
    rank_candidates,
    ranking_weights_from_env,
)
from tradingagents.survivor.markets.types import Candidate, ScanRejection


@dataclass
class ScanResult:
    discovered: int = 0
    normalized: int = 0
    rejected: list[tuple[str, ScanRejection]] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    ranked: list[Candidate] = field(default_factory=list)
    top: list[Candidate] = field(default_factory=list)


class MarketScanner:
    """Zero-LLM deterministic scanner."""

    def __init__(
        self,
        adapter: MarketAdapter,
        limits: ScanLimits | None = None,
        weights: RankingWeights | None = None,
        max_candidates_per_cycle: int = 40,
        max_research_candidates_per_cycle: int = 3,
    ):
        self.adapter = adapter
        self.limits = limits or scan_limits_from_env()
        self.weights = weights or ranking_weights_from_env()
        self.max_candidates_per_cycle = max_candidates_per_cycle
        self.max_research_candidates_per_cycle = max_research_candidates_per_cycle

    def scan(
        self,
        now: datetime | None = None,
        open_position_symbols: frozenset[str] | None = None,
        trading_halted: bool = False,
    ) -> ScanResult:
        current = now or datetime.now(timezone.utc)
        result = ScanResult()
        raw_markets = self.adapter.list_markets()
        result.discovered = len(raw_markets)

        candidates: list[Candidate] = []
        for raw in raw_markets:
            snapshot = self.adapter.normalize(raw) if hasattr(self.adapter, "normalize") else None
            if snapshot is None:
                result.rejected.append(("<unnormalizable>", ScanRejection.UNSUPPORTED_MARKET_TYPE))
                continue
            result.normalized += 1
            reason = filter_market(
                snapshot, self.limits, now=current,
                open_position_symbols=open_position_symbols,
                trading_halted=trading_halted,
            )
            if reason is not None:
                result.rejected.append((snapshot.market_id, reason))
            else:
                candidates.append(Candidate(snapshot=snapshot))

        result.candidates = candidates[: self.max_candidates_per_cycle]
        result.ranked = rank_candidates(result.candidates, self.weights, now=current)
        result.top = result.ranked[: self.max_research_candidates_per_cycle]
        return result
