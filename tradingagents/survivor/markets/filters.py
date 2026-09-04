"""Deterministic, fail-closed candidate filters. Every rejection is machine-readable."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

from tradingagents.survivor.markets.types import (
    MarketSnapshot,
    MarketStatus,
    QuoteCurrency,
    ResolutionStatus,
    ScanRejection,
)


@dataclass(frozen=True)
class ScanLimits:
    """Configurable conservative scan limits. Probability in bps, times in seconds,
    liquidity in minor units of the provider quote currency."""

    min_liquidity_minor: int = 100000          # $1,000.00 USD (Polymarket)
    max_spread_bps: int = 500                  # 5% of price
    min_seconds_to_resolution: int = 3600      # >= 1h
    max_seconds_to_resolution: int = 7776000   # <= 90d
    min_probability_bps: int = 500             # avoid ~0%
    max_probability_bps: int = 9500            # avoid ~100%
    max_staleness_seconds: int = 300


def scan_limits_from_env() -> ScanLimits:
    def _int(env: str, default: int) -> int:
        raw = os.environ.get(env)
        return int(raw) if raw else default  # ValueError on garbage = loud failure

    return ScanLimits(
        min_liquidity_minor=_int("SURVIVOR_MIN_LIQUIDITY_MINOR", 100000),
        max_spread_bps=_int("SURVIVOR_MAX_SPREAD_BPS", 500),
        min_seconds_to_resolution=_int("SURVIVOR_MIN_RESOLUTION_SECONDS", 3600),
        max_seconds_to_resolution=_int("SURVIVOR_MAX_RESOLUTION_SECONDS", 7776000),
        min_probability_bps=_int("SURVIVOR_MIN_PROBABILITY_BPS", 500),
        max_probability_bps=_int("SURVIVOR_MAX_PROBABILITY_BPS", 9500),
        max_staleness_seconds=_int("SURVIVOR_MARKET_MAX_STALENESS_SECONDS", 300),
    )


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def filter_market(
    snapshot: MarketSnapshot,
    limits: ScanLimits,
    now: datetime | None = None,
    open_position_symbols: frozenset[str] | None = None,
    trading_halted: bool = False,
) -> ScanRejection | None:
    """Return a ScanRejection reason, or None when the market passes all filters."""
    current = now or datetime.now(timezone.utc)

    if snapshot.market_status != MarketStatus.OPEN:
        return ScanRejection.MARKET_CLOSED
    if snapshot.resolution_status != ResolutionStatus.UNRESOLVED:
        return ScanRejection.MARKET_RESOLVED
    if snapshot.market_type != "prediction_binary":
        return ScanRejection.UNSUPPORTED_MARKET_TYPE
    if not snapshot.question or not snapshot.close_time_utc:
        return ScanRejection.MISSING_CLOSE_TIME
    if snapshot.bid is None or snapshot.ask is None or snapshot.market_probability_bps is None:
        return ScanRejection.MISSING_PRICE
    if not (0 < snapshot.bid <= 10000 and 0 < snapshot.ask <= 10000 and snapshot.bid <= snapshot.ask):
        return ScanRejection.INVALID_PRICE
    if snapshot.liquidity is None:
        return ScanRejection.MISSING_LIQUIDITY
    if snapshot.liquidity.minor_units < limits.min_liquidity_minor:
        return ScanRejection.LOW_LIQUIDITY
    spread = snapshot.ask - snapshot.bid
    if snapshot.ask > 0 and spread * 10000 > limits.max_spread_bps * snapshot.ask:
        return ScanRejection.SPREAD_TOO_WIDE
    source = _parse(snapshot.source_timestamp_utc or snapshot.timestamp_utc)
    if source is None or (current - source).total_seconds() > limits.max_staleness_seconds:
        return ScanRejection.STALE_DATA
    close = _parse(snapshot.close_time_utc)
    if close is None:
        return ScanRejection.MISSING_RESOLUTION_CRITERIA
    seconds_to_resolution = (close - current).total_seconds()
    if seconds_to_resolution < limits.min_seconds_to_resolution:
        return ScanRejection.RESOLUTION_TOO_CLOSE
    if seconds_to_resolution > limits.max_seconds_to_resolution:
        return ScanRejection.RESOLUTION_TOO_FAR
    if not (limits.min_probability_bps <= snapshot.market_probability_bps <= limits.max_probability_bps):
        return ScanRejection.INVALID_PRICE  # extreme probability treated as unpriceable edge
    if snapshot.liquidity.currency_upper != QuoteCurrency.USD.value and snapshot.liquidity.currency_upper != QuoteCurrency.GBP.value:
        return ScanRejection.UNSUPPORTED_MARKET_TYPE
    if open_position_symbols and snapshot.market_id in open_position_symbols:
        return ScanRejection.EXISTING_POSITION
    if trading_halted:
        return ScanRejection.TRADING_HALTED
    return None
