"""Polymarket READ-ONLY market adapter over the public Gamma API.

Reuses the keyless public data client already shipped in
``tradingagents.dataflows.polymarket``. No trading API is contacted.

Currency: Gamma volumes/liquidity are USD. They are carried as
``MoneyAmount(minor_units, currency="USD")`` — never relabelled GBP.
Prices are binary-market probabilities carried in basis points (1.00 = 10000).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from tradingagents.dataflows import polymarket as gamma
from tradingagents.survivor.markets.adapter import MarketAdapter
from tradingagents.survivor.markets.types import (
    MarketSnapshot,
    MarketStatus,
    QuoteCurrency,
    ResolutionStatus,
)

PROVIDER = "polymarket"
MARKET_TYPE = "prediction_binary"


def _iso_to_iso(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return None


class PolymarketAdapter(MarketAdapter):
    """Read-only Polymarket market-data adapter (public Gamma API, keyless)."""

    provider = PROVIDER
    market_type = MARKET_TYPE

    def list_markets(self, limit: int = 100) -> list[dict]:
        """Fetch open binary markets from the public Gamma ``/markets`` collection
        (keyless, read-only), ordered by traded volume."""
        data = gamma._request(
            "markets",
            {
                "limit": min(limit, 100),
                "closed": "false",
                "order": "volumeNum",
                "ascending": "false",
            },
        )
        now = datetime.now(timezone.utc)
        markets = [m for m in data if gamma._is_forward_looking(m, now)]
        markets.sort(key=lambda m: m.get("volumeNum") or 0, reverse=True)
        return markets[:limit]

    @staticmethod
    def normalize(raw: dict, now: datetime | None = None) -> MarketSnapshot | None:
        """Normalize one raw Gamma market dict into a MarketSnapshot."""
        outcomes = gamma._parse_json_list(raw.get("outcomes"))
        prices = gamma._parse_json_list(raw.get("outcomePrices"))
        if len(outcomes) != 2 or len(prices) != 2:
            return None  # not a clean binary market -> unapplicable
        try:
            yes_prob = int(round(float(prices[0]) * 10000))
        except (ValueError, TypeError):
            return None
        if not 0 <= yes_prob <= 10000:
            return None

        def _bps(value) -> int | None:
            try:
                bps = int(round(float(value) * 10000))
            except (TypeError, ValueError):
                return None
            return bps if 0 <= bps <= 10000 else None

        best_bid = _bps(raw.get("bestBid"))
        best_ask = _bps(raw.get("bestAsk"))
        closed = bool(raw.get("closed"))
        # Gamma provides no boolean "resolved" flag on search results; a closed
        # market is treated as resolved for discovery purposes.
        resolution = ResolutionStatus.UNRESOLVED
        status = MarketStatus.CLOSED if closed else MarketStatus.OPEN

        now = now or datetime.now(timezone.utc)
        volume_num = raw.get("volumeNum")
        liquidity_num = raw.get("liquidityNum")
        canonical = json.dumps(
            {k: raw.get(k) for k in ("id", "question", "outcomes", "outcomePrices", "bestBid", "bestAsk")},
            sort_keys=True, default=str,
        )
        return MarketSnapshot(
            market_id=str(raw.get("id") or hashlib.sha256(canonical.encode()).hexdigest()[:16]),
            provider=PROVIDER,
            market_type=MARKET_TYPE,
            symbol_or_slug=str(raw.get("slug") or raw.get("id") or ""),
            question=str(raw.get("question") or ""),
            timestamp_utc=now.isoformat(),
            close_time_utc=_iso_to_iso(raw.get("endDate")),
            resolution_time_utc=_iso_to_iso(raw.get("endDate")),
            bid=best_bid,
            ask=best_ask,
            mid=(int((best_bid + best_ask) / 2) if best_bid is not None and best_ask is not None else None),
            yes_bid=best_bid,
            yes_ask=best_ask,
            no_bid=(10000 - best_ask) if best_ask is not None else None,
            no_ask=(10000 - best_bid) if best_bid is not None else None,
            market_probability_bps=yes_prob,
            liquidity=(
                MoneyAmountUSD(int(round(float(liquidity_num) * 100)))
                if isinstance(liquidity_num, (int, float)) else None
            ),
            volume_24h=(
                MoneyAmountUSD(int(round(float(volume_num) * 100)))
                if isinstance(volume_num, (int, float)) else None
            ),
            quote_currency=QuoteCurrency.USD.value,
            market_status=status,
            resolution_status=resolution,
            source_timestamp_utc=now.isoformat(),
        )

    def get_snapshot(self, market_id: str) -> MarketSnapshot | None:
        for raw in self.list_markets(limit=100):
            snap = self.normalize(raw)
            if snap and snap.market_id == market_id:
                return snap
        return None

    def get_resolution_status(self, market_id: str) -> ResolutionStatus:
        for raw in self.list_markets(limit=100):
            if str(raw.get("id")) == market_id:
                return ResolutionStatus.NO if raw.get("closed") else ResolutionStatus.UNRESOLVED
        return ResolutionStatus.UNKNOWN


def MoneyAmountUSD(minor_units: int):
    """Explicit USD money constructor (documents provider currency)."""
    from tradingagents.survivor.markets.types import MoneyAmount

    return MoneyAmount(minor_units, currency=QuoteCurrency.USD.value)
