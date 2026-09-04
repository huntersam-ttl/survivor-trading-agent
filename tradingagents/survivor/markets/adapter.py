"""Provider-neutral, READ-ONLY market adapter interface.

Implementations may ONLY expose market-data methods. Execution capability is
structurally prohibited: any subclass defining a forbidden name
(place_order/cancel_order/withdraw/transfer/submit_order/...) fails at class
creation time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tradingagents.survivor.markets.types import MarketSnapshot, ResolutionStatus

FORBIDDEN_METHODS = frozenset(
    {
        "place_order", "cancel_order", "submit_order", "create_order",
        "withdraw", "deposit", "transfer", "sign", "authenticate",
        "set_leverage", "borrow", "margin_trade", "execute_order",
    }
)


class MarketAdapter(ABC):
    """Read-only market data interface. No execution methods may exist."""

    provider: str = "abstract"
    market_type: str = "generic"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name in cls.__dict__:
            if name in FORBIDDEN_METHODS:
                raise TypeError(
                    f"MarketAdapter subclass {cls.__name__} defines forbidden "
                    f"execution method {name!r}; adapters are READ-ONLY"
                )

    @abstractmethod
    def list_markets(self, limit: int = 100) -> list[dict]:
        """Return raw provider market dicts (open markets)."""

    @abstractmethod
    def get_snapshot(self, market_id: str) -> MarketSnapshot | None:
        """Return a normalized snapshot for one market, or None if unknown."""

    @abstractmethod
    def get_resolution_status(self, market_id: str) -> ResolutionStatus:
        """Return the resolution status for one market."""

    def get_market(self, market_id: str) -> dict | None:
        """Optional raw market metadata lookup."""
        return None

    def get_orderbook_or_quote(self, market_id: str) -> dict | None:
        """Optional raw quote/orderbook data."""
        return None
