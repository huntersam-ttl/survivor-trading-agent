"""Paper-trading portfolio accounting (integer pence only, conservative marks)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Position:
    symbol: str
    quantity: int = 0
    average_entry_price_pence: int = 0
    cost_basis_pence: int = 0
    current_mark_pence: int | None = None  # None => not yet marked (stale)
    opened_at: str = ""
    updated_at: str = ""

    @property
    def market_value_pence(self) -> int:
        """Conservative value: marked at bid when available, else cost basis."""
        if self.quantity == 0:
            return 0
        if self.current_mark_pence is None:
            return self.cost_basis_pence
        return self.quantity * self.current_mark_pence

    @property
    def unrealized_pnl_pence(self) -> int:
        """Unrealized P/L vs cost basis; NOT clamped to zero when negative."""
        if self.quantity == 0 or self.current_mark_pence is None:
            return 0
        return self.quantity * (self.current_mark_pence - self.average_entry_price_pence)

    @property
    def is_stale_mark(self) -> bool:
        return self.quantity > 0 and self.current_mark_pence is None


@dataclass
class Portfolio:
    cash_pence: int = 0
    starting_equity_pence: int = 0
    realized_pnl_pence: int = 0
    high_water_mark_pence: int = 0
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl_today_pence: int = 0  # refreshed from the ledger by the broker

    # --- accounting aggregates -------------------------------------------------
    @property
    def exposure_pence(self) -> int:
        return sum(p.market_value_pence for p in self.positions.values())

    @property
    def unrealized_pnl_pence(self) -> int:
        return sum(p.unrealized_pnl_pence for p in self.positions.values())

    @property
    def equity_pence(self) -> int:
        """cash + marked position value (unmarked positions at cost basis)."""
        return self.cash_pence + self.exposure_pence

    @property
    def daily_pnl_pence(self) -> int:
        """realized + unrealized (unrealized losses NOT clamped to zero)."""
        return self.realized_pnl_today_pence + self.unrealized_pnl_pence

    @property
    def drawdown_bps(self) -> int:
        if self.high_water_mark_pence <= 0:
            return 0
        dd = (self.high_water_mark_pence - self.equity_pence) * 10000 // self.high_water_mark_pence
        return max(0, dd)

    @property
    def has_stale_marks(self) -> bool:
        return any(p.is_stale_mark for p in self.positions.values())

    # --- mutations -------------------------------------------------------------
    def buy(self, symbol: str, quantity: int, execution_price_pence: int, fee_pence: int, now: str | None = None) -> None:
        """Apply a long BUY fill. Raises on any invariant violation (defense in depth)."""
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if execution_price_pence <= 0:
            raise ValueError("execution price must be positive")
        if fee_pence < 0:
            raise ValueError("fee cannot be negative")
        total_cost = quantity * execution_price_pence + fee_pence
        if total_cost > self.cash_pence:
            raise ValueError(
                f"insufficient cash: need {total_cost}p, have {self.cash_pence}p (negative cash impossible)"
            )
        ts = now or utc_now_iso()
        position = self.positions.get(symbol)
        if position is None:
            position = Position(symbol=symbol, opened_at=ts)
            self.positions[symbol] = position
        new_quantity = position.quantity + quantity
        new_cost_basis = position.cost_basis_pence + quantity * execution_price_pence
        position.average_entry_price_pence = new_cost_basis // new_quantity
        position.quantity = new_quantity
        position.cost_basis_pence = new_cost_basis
        position.updated_at = ts
        self.cash_pence -= total_cost
        self.realized_pnl_pence -= fee_pence  # fees are realized costs

    def mark(self, marks: dict[str, int], now: str | None = None) -> None:
        """Mark long positions conservatively at bid. HWM never moves downward."""
        ts = now or utc_now_iso()
        for symbol, mark_pence in marks.items():
            position = self.positions.get(symbol)
            if position and position.quantity > 0:
                if mark_pence <= 0:
                    raise ValueError(f"invalid mark for {symbol}: {mark_pence}")
                position.current_mark_pence = mark_pence
                position.updated_at = ts
        equity = self.equity_pence
        if equity > self.high_water_mark_pence:
            self.high_water_mark_pence = equity
