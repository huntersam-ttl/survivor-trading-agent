"""Types for the deterministic paper-trading proposal layer (Phase 2).

All monetary values are integer minor units (pence). All percentage-like
values are integer basis points (bps). No binary floating point is used for
money anywhere in the Survivor paper-execution boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from decimal import ROUND_CEILING, Decimal
from enum import Enum


class Side(str, Enum):
    """Order side. Phase 2 supports long entries only."""

    BUY = "BUY"


class Action(str, Enum):
    """Proposal action. Only OPEN may ever reach the PaperBroker."""

    OPEN = "OPEN"
    HOLD = "HOLD"
    NO_TRADE = "NO_TRADE"


def ceil_bps_cost(amount: int, bps: int) -> int:
    """Deterministic cost in pence for ``amount`` pence at ``bps`` basis points (rounded up)."""
    if amount <= 0 or bps <= 0:
        return 0
    return int((Decimal(amount) * Decimal(bps) / Decimal(10000)).quantize(Decimal("1"), rounding=ROUND_CEILING))


@dataclass(frozen=True)
class TradeProposal:
    """Strict, deterministic trade proposal produced by ProposalBuilder.

    The LLM never fills execution parameters: it only contributes the decision
    text (mapped to a side/action) and the rationale reference. Every numeric
    field is either a deterministic input from config/data or computed here.
    """

    run_id: str
    timestamp_utc: str
    market: str
    symbol: str
    side: str
    action: str
    # Execution shape (integers only)
    quantity: int = 0
    reference_price_pence: int = 0  # reference ask for BUY
    notional_pence: int = 0
    # Edge components, all in basis points of notional
    expected_probability_bps: int | None = None
    market_probability_bps: int | None = None
    gross_edge_bps: int = 0
    spread_cost_bps: int = 0
    slippage_bps: int = 0
    fee_bps: int = 0
    uncertainty_penalty_bps: int = 0
    conservative_net_edge_bps: int = 0
    # Explicit pence cost estimates (BUY entry costs)
    estimated_fees_pence: int = 0
    estimated_slippage_pence: int = 0
    estimated_spread_cost_pence: int = 0
    # Provenance
    rationale_reference: str = ""
    source_decision: str = ""
    proposal_id: str | None = None

    def __post_init__(self) -> None:
        if self.proposal_id is None:
            object.__setattr__(self, "proposal_id", derive_proposal_id(self))

    def stable_fields(self) -> dict:
        """Canonical, order-independent dict of the identity-bearing fields."""
        return {
            "run_id": self.run_id,
            "timestamp_utc": self.timestamp_utc,
            "market": self.market,
            "symbol": self.symbol,
            "side": self.side,
            "action": self.action,
            "quantity": self.quantity,
            "reference_price_pence": self.reference_price_pence,
            "notional_pence": self.notional_pence,
            "gross_edge_bps": self.gross_edge_bps,
            "conservative_net_edge_bps": self.conservative_net_edge_bps,
            "source_decision": self.source_decision,
        }

    def field_names(self) -> set[str]:
        return {f.name for f in fields(self)}


def derive_proposal_id(proposal: TradeProposal) -> str:
    """Deterministic proposal identity from stable fields (replay detection)."""
    canonical = json.dumps(proposal.stable_fields(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_conservative_net_edge_bps(
    gross_edge_bps: int,
    spread_cost_bps: int,
    slippage_bps: int,
    fee_bps: int,
    uncertainty_penalty_bps: int,
) -> int:
    """conservative_net_edge = gross_edge - spread - slippage - fees - uncertainty (bps)."""
    return gross_edge_bps - spread_cost_bps - slippage_bps - fee_bps - uncertainty_penalty_bps
