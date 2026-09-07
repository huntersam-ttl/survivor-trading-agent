"""Deterministic read-only market-signal summary for Jarvis.

This reuses the existing deterministic ``MarketScanner`` results verbatim and
derives lightweight, human/consumer-readable flags directly from the scanner's
``MarketSnapshot`` fields. It intentionally does NOT:

* make any LLM call,
* talk to any new market-data provider,
* emit a BUY/SELL recommendation,
* execute or propose an order, or
* alter strategy/risk.

There is no historical price in a single snapshot, so ``price_change`` is always
``None``. Output is bounded to ``MAX_CANDIDATES`` entries.
"""

from __future__ import annotations

from datetime import datetime, timezone

from tradingagents.survivor.markets.scanner import ScanResult
from tradingagents.survivor.markets.types import MarketSnapshot

MAX_CANDIDATES = 20

# Named deterministic thresholds for lightweight signal flags.
HIGH_VOLUME_USD_CENTS = 500_000  # >= $5,000 of 24h USD volume -> HIGH
WIDE_SPREAD_RATIO = 0.05         # quoted spread >= 5% -> HIGH volatility


def _volume_signal(minor_units: int | None) -> str | None:
    if minor_units is None or minor_units <= 0:
        return None
    return "HIGH" if minor_units >= HIGH_VOLUME_USD_CENTS else "LOW"


def _volatility_signal(snapshot: MarketSnapshot) -> str | None:
    if snapshot.bid is None or snapshot.ask is None or snapshot.ask <= 0:
        return None
    spread = (snapshot.ask - snapshot.bid) / snapshot.ask
    return "HIGH" if spread >= WIDE_SPREAD_RATIO else "LOW"


def _reason_codes(snapshot: MarketSnapshot) -> list[str]:
    """Deterministic data-presence codes describing the signal basis."""
    codes: list[str] = []
    if snapshot.volume_24h is not None:
        codes.append("HAS_VOLUME")
    if snapshot.bid is not None and snapshot.ask is not None:
        codes.append("HAS_QUOTE")
    if snapshot.liquidity is not None:
        codes.append("LIQUID")
    return codes


def _candidate_row(candidate) -> dict:
    snap = candidate.snapshot
    volume_minor = snap.volume_24h.minor_units if snap.volume_24h else None
    return {
        "symbol": snap.symbol_or_slug,
        "score": candidate.score,
        "price_change": None,  # single snapshot has no historical price
        "volume_signal": _volume_signal(volume_minor),
        "volatility_signal": _volatility_signal(snap),
        "reason_codes": _reason_codes(snap),
        "observed_at": snap.timestamp_utc or snap.source_timestamp_utc,
    }


def build_signal_summary(
    result: ScanResult,
    *,
    generated_at: datetime | None = None,
    max_candidates: int = MAX_CANDIDATES,
) -> dict:
    """Format a scanner result into a bounded read-only signal summary.

    ``result`` is expected to be a ``ScanResult`` (as produced by
    ``MarketScanner.scan``). ``market_open`` reflects whether any markets were
    discovered by the existing scanner. Never more than ``max_candidates`` rows.
    """
    if max_candidates < 0:
        raise ValueError("max_candidates must be non-negative")
    ranked = getattr(result, "ranked", None) or []
    rows = [_candidate_row(c) for c in ranked[:max_candidates]]
    return {
        "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(),
        "market_open": int(getattr(result, "discovered", 0) or 0) > 0,
        "candidates": rows,
    }
