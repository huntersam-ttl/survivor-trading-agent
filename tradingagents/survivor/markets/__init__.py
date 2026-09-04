"""Market discovery package: read-only adapters, deterministic scanner (zero LLM)."""

from tradingagents.survivor.markets.adapter import FORBIDDEN_METHODS, MarketAdapter
from tradingagents.survivor.markets.filters import ScanLimits, filter_market, scan_limits_from_env
from tradingagents.survivor.markets.ranking import RankingWeights, rank_candidates, score_snapshot
from tradingagents.survivor.markets.scanner import MarketScanner, ScanResult
from tradingagents.survivor.markets.types import (
    Candidate,
    MarketSnapshot,
    MarketStatus,
    MoneyAmount,
    QuoteCurrency,
    ResolutionStatus,
    ScanRejection,
)

__all__ = [
    "Candidate", "FORBIDDEN_METHODS", "MarketAdapter", "MarketScanner", "MarketSnapshot",
    "MarketStatus", "MoneyAmount", "QuoteCurrency", "RankingWeights", "ResolutionStatus",
    "ScanLimits", "ScanRejection", "ScanResult", "filter_market", "rank_candidates",
    "scan_limits_from_env", "score_snapshot",
]
