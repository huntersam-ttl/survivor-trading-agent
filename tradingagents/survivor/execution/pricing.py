"""PaperBroker execution pricing model (deterministic, conservative).

- BUY execution price = ask (or single reference price) + explicit slippage.
  Never a raw mid-price; never a future price.
- Long-position mark = bid (conservative); never the midpoint.
- No price may be invented: an explicit simulated spread/slippage assumption
  must be supplied, otherwise callers fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from tradingagents.survivor.trading.types import ceil_bps_cost


@dataclass(frozen=True)
class MarketSnapshot:
    """Deterministic market inputs for one symbol. Prices in integer pence."""

    symbol: str
    bid_pence: int | None = None
    ask_pence: int | None = None
    timestamp_utc: str = ""

    def is_fresh(self, now: datetime | None = None, max_age_sec: int = 120) -> bool:
        if not self.timestamp_utc:
            return False
        try:
            ts = datetime.fromisoformat(self.timestamp_utc.replace("Z", "+00:00"))
        except ValueError:
            return False
        if ts.tzinfo is None:
            return False
        current = now or datetime.now(timezone.utc)
        return 0 <= (current - ts).total_seconds() <= max_age_sec

    def has_valid_prices(self) -> bool:
        return (
            self.ask_pence is not None
            and self.ask_pence > 0
            and (self.bid_pence is None or self.bid_pence > 0)
        )


def buy_execution_price_pence(ask_pence: int, slippage_bps: int) -> int:
    """Deterministic BUY fill: ask + explicit slippage (rounded up to the penny)."""
    if ask_pence <= 0:
        raise ValueError("ask price must be positive")
    return ask_pence + ceil_bps_cost(ask_pence, slippage_bps)


def conservative_long_mark_pence(snapshot: MarketSnapshot) -> int:
    """Long mark = bid. If bid unavailable, fail closed (no midpoint guessing)."""
    if snapshot.bid_pence is None or snapshot.bid_pence <= 0:
        raise ValueError("conservative long mark requires a valid bid price")
    return snapshot.bid_pence
