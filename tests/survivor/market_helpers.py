"""Phase 3 helpers: fake READ-ONLY market adapter (no network, no AI)."""

from datetime import datetime, timedelta, timezone

from tradingagents.survivor.markets.adapter import MarketAdapter
from tradingagents.survivor.markets.types import (
    MarketSnapshot,
    MarketStatus,
    MoneyAmount,
    QuoteCurrency,
    ResolutionStatus,
)


def iso_in(hours: float) -> str:
    """ISO close time `hours` from now (future)."""
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def fresh_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_snapshot(
    market_id: str = "mkt-1",
    question: str = "Will X happen by 2027?",
    yes: int = 5200,
    bid: int = 5100,
    ask: int = 5200,
    liquidity_usd_cents: int = 200000,  # $2,000
    volume_usd_cents: int = 100000,     # $1,000
    close_hours: float = 24 * 7,
    **over,
) -> MarketSnapshot:
    fields = {
        'market_id': market_id,
        'provider': "fake",
        'market_type': "prediction_binary",
        'symbol_or_slug': market_id,
        'question': question,
        'timestamp_utc': fresh_ts(),
        'close_time_utc': iso_in(close_hours),
        'resolution_time_utc': iso_in(close_hours),
        'bid': bid,
        'ask': ask,
        'mid': ((bid + ask) // 2) if (bid is not None and ask is not None) else None,
        'yes_bid': bid,
        'yes_ask': ask,
        'no_bid': (10000 - ask) if ask is not None else None,
        'no_ask': (10000 - bid) if bid is not None else None,
        'market_probability_bps': yes,
        'liquidity': MoneyAmount(liquidity_usd_cents, currency=QuoteCurrency.USD.value),
        'volume_24h': MoneyAmount(volume_usd_cents, currency=QuoteCurrency.USD.value),
        'quote_currency': QuoteCurrency.USD.value,
        'market_status': MarketStatus.OPEN,
        'resolution_status': ResolutionStatus.UNRESOLVED,
        'source_timestamp_utc': fresh_ts(),
    }
    fields.update(over)
    return MarketSnapshot(**fields)


class FakeMarketAdapter(MarketAdapter):
    """Fake READ-ONLY adapter over in-memory snapshots. No network. No AI."""

    provider = "fake"
    market_type = "prediction_binary"

    def __init__(self, snapshots: list[MarketSnapshot]):
        self._snapshots = {s.market_id: s for s in snapshots}
        self.list_calls = 0
        self.snapshot_calls = 0
        self.resolution_calls = 0

    def list_markets(self, limit: int = 100) -> list[dict]:
        self.list_calls += 1
        return [{"market_id": m.market_id} for m in self._snapshots.values()][:limit]

    def normalize(self, raw: dict) -> MarketSnapshot | None:
        return self._snapshots.get(raw.get("market_id"))

    def get_snapshot(self, market_id: str) -> MarketSnapshot | None:
        self.snapshot_calls += 1
        return self._snapshots.get(market_id)

    def get_resolution_status(self, market_id: str) -> ResolutionStatus:
        self.resolution_calls += 1
        return self._snapshots[market_id].resolution_status
