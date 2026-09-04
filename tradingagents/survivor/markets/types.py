"""Market discovery types for the autonomous paper loop.

Currency discipline: provider values are ALWAYS carried with an explicit
currency via :class:`MoneyAmount`. Polymarket liquidity/volume are USD and
must never be labelled GBP. Paper accounting remains integer GBP pence and is
only ever reached through the Phase 2 paper pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QuoteCurrency(str, Enum):
    USD = "USD"
    GBP = "GBP"


@dataclass(frozen=True)
class MoneyAmount:
    """An amount of money in minor units with an explicit currency."""

    minor_units: int
    currency: str = QuoteCurrency.USD.value

    def __post_init__(self) -> None:
        if isinstance(self.minor_units, bool) or not isinstance(self.minor_units, int):
            raise TypeError("minor_units must be an integer")
        if self.minor_units < 0:
            raise ValueError("minor_units must be non-negative")
        if not self.currency or self.currency.upper() not in (QuoteCurrency.USD.value, QuoteCurrency.GBP.value):
            raise ValueError(f"unsupported currency: {self.currency!r}")

    @property
    def currency_upper(self) -> str:
        return self.currency.upper()


class MarketStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class ResolutionStatus(str, Enum):
    UNRESOLVED = "UNRESOLVED"
    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"


class ScanRejection(str, Enum):
    MARKET_CLOSED = "MARKET_CLOSED"
    MARKET_RESOLVED = "MARKET_RESOLVED"
    MISSING_CLOSE_TIME = "MISSING_CLOSE_TIME"
    MISSING_PRICE = "MISSING_PRICE"
    INVALID_PRICE = "INVALID_PRICE"
    MISSING_LIQUIDITY = "MISSING_LIQUIDITY"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    STALE_DATA = "STALE_DATA"
    RESOLUTION_TOO_CLOSE = "RESOLUTION_TOO_CLOSE"
    RESOLUTION_TOO_FAR = "RESOLUTION_TOO_FAR"
    MISSING_RESOLUTION_CRITERIA = "MISSING_RESOLUTION_CRITERIA"
    AMBIGUOUS_MARKET = "AMBIGUOUS_MARKET"
    UNSUPPORTED_MARKET_TYPE = "UNSUPPORTED_MARKET_TYPE"
    EXISTING_POSITION = "EXISTING_POSITION"
    TRADING_HALTED = "TRADING_HALTED"
    NOT_A_BINARY_MARKET = "NOT_A_BINARY_MARKET"


@dataclass(frozen=True)
class MarketSnapshot:
    """Strict immutable provider-neutral market snapshot.

    Prices are minor units of ``quote_currency`` (probability-like binary
    prices are bps of 1.00). Monetary liquidity/volume carry explicit currency.
    """

    market_id: str
    provider: str
    market_type: str  # e.g. "prediction_binary"
    symbol_or_slug: str
    question: str
    timestamp_utc: str
    close_time_utc: str | None = None
    resolution_time_utc: str | None = None
    bid: int | None = None            # probability bps 0..10000
    ask: int | None = None            # probability bps 0..10000
    mid: int | None = None
    yes_bid: int | None = None
    yes_ask: int | None = None
    no_bid: int | None = None
    no_ask: int | None = None
    market_probability_bps: int | None = None
    liquidity: MoneyAmount | None = None
    volume_24h: MoneyAmount | None = None
    quote_currency: str = QuoteCurrency.USD.value
    market_status: MarketStatus = MarketStatus.OPEN
    resolution_status: ResolutionStatus = ResolutionStatus.UNRESOLVED
    source_timestamp_utc: str = ""

    def __post_init__(self) -> None:
        if not self.market_id or not self.provider:
            raise ValueError("market_id and provider are required")
        for name in ("bid", "ask", "mid", "yes_bid", "yes_ask", "no_bid", "no_ask", "market_probability_bps"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10000):
                raise ValueError(f"{name} must be an integer probability in bps [0, 10000], got {value}")
        if self.liquidity is not None and not isinstance(self.liquidity, MoneyAmount):
            raise TypeError("liquidity must be a MoneyAmount")
        if self.volume_24h is not None and not isinstance(self.volume_24h, MoneyAmount):
            raise TypeError("volume_24h must be a MoneyAmount")


@dataclass(frozen=True)
class Candidate:
    """A filtered market candidate ready for deterministic ranking."""

    snapshot: MarketSnapshot
    score: float | None = None
    rank: int | None = None

    @property
    def market_id(self) -> str:
        return self.snapshot.market_id

    @property
    def question(self) -> str:
        return self.snapshot.question
