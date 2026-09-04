"""Strict, LLM-independent validation of TradeProposal objects."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from tradingagents.survivor.trading.types import (
    Action,
    Side,
    TradeProposal,
    compute_conservative_net_edge_bps,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from tradingagents.survivor.risk.limits import RiskLimits

_SYMBOL_RE = re.compile(r"^[A-Za-z0-9.\-_]{1,20}$")


class ProposalValidationError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _require_int(value: object, name: str, errors: list[str], minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{name} must be an integer, got {type(value).__name__}")
        return
    if not math.isfinite(value):
        errors.append(f"{name} must be finite")
    if minimum is not None and value < minimum:
        errors.append(f"{name} must be >= {minimum}, got {value}")


def _parse_timestamp(timestamp_utc: str, errors: list[str]) -> datetime | None:
    try:
        return datetime.fromisoformat(str(timestamp_utc).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        errors.append(f"timestamp_utc is not a valid ISO timestamp: {timestamp_utc!r}")
        return None


def proposal_from_dict(data: dict, **overrides) -> TradeProposal:
    """Rebuild a proposal from a dict, rejecting unknown/extra fields."""
    allowed = set(TradeProposal.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = set(data) - allowed - {"proposal_id"}
    if unknown:
        raise ProposalValidationError([f"unexpected extra fields: {sorted(unknown)}"])
    return TradeProposal(**{**data, **overrides})


def validate_proposal(
    proposal: TradeProposal,
    limits: RiskLimits,
    now: datetime | None = None,
) -> None:
    """Validate every proposal invariant. Raises ProposalValidationError on any failure."""
    errors: list[str] = []

    if not proposal.run_id or not isinstance(proposal.run_id, str):
        errors.append("run_id is required")
    ts = _parse_timestamp(proposal.timestamp_utc, errors)

    if not proposal.proposal_id:
        errors.append("proposal_id is required")
    if not proposal.market or not isinstance(proposal.market, str):
        errors.append("market is required")
    if not proposal.symbol or not _SYMBOL_RE.match(proposal.symbol or ""):
        errors.append(f"malformed symbol: {proposal.symbol!r}")
    if proposal.side not in (Side.BUY.value,):
        errors.append(f"unknown/unsupported side: {proposal.side!r}")
    if proposal.action not in (Action.OPEN.value, Action.HOLD.value, Action.NO_TRADE.value):
        errors.append(f"unsupported action: {proposal.action!r}")

    for name in (
        "quantity", "reference_price_pence", "notional_pence", "gross_edge_bps",
        "spread_cost_bps", "slippage_bps", "fee_bps", "uncertainty_penalty_bps",
        "conservative_net_edge_bps", "estimated_fees_pence", "estimated_slippage_pence",
        "estimated_spread_cost_pence",
    ):
        _require_int(getattr(proposal, name), name, errors)
    for name in ("expected_probability_bps", "market_probability_bps"):
        value = getattr(proposal, name)
        if value is not None:
            _require_int(value, name, errors, minimum=0)
            if isinstance(value, int) and not isinstance(value, bool) and value > 10000:
                errors.append(f"{name} must be <= 10000 (impossible probability), got {value}")

    if errors:
        raise ProposalValidationError(errors)

    # Recompute the conservative edge: the validator never trusts the submitted value.
    recomputed = compute_conservative_net_edge_bps(
        proposal.gross_edge_bps, proposal.spread_cost_bps,
        proposal.slippage_bps, proposal.fee_bps, proposal.uncertainty_penalty_bps,
    )
    if recomputed != proposal.conservative_net_edge_bps:
        errors.append("conservative_net_edge_bps does not match its components")
    if not (-10000 <= proposal.conservative_net_edge_bps <= 10000):
        errors.append("conservative_net_edge_bps outside valid bounds [-10000, 10000]")

    # Staleness and future timestamps
    if ts is not None:
        current = now or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            errors.append("timestamp_utc must be timezone-aware")
        else:
            age = (current - ts).total_seconds()
            if age > limits.max_proposal_age_sec:
                errors.append(f"stale proposal: age {age:.0f}s > {limits.max_proposal_age_sec}s")
            if age < -60:
                errors.append("timestamp_utc is in the future")

    if proposal.action == Action.OPEN.value:
        if proposal.quantity <= 0:
            errors.append("OPEN proposal requires positive quantity")
        if proposal.reference_price_pence <= 0:
            errors.append("OPEN proposal requires positive reference price")
        if proposal.notional_pence <= 0:
            errors.append("OPEN proposal requires positive notional")
        if proposal.notional_pence != proposal.quantity * proposal.reference_price_pence:
            errors.append("notional_pence != quantity * reference_price_pence")
        if proposal.notional_pence > limits.max_single_position_pence:
            errors.append(
                f"position {proposal.notional_pence}p exceeds max_single_position "
                f"{limits.max_single_position_pence}p"
            )

    if errors:
        raise ProposalValidationError(errors)
