"""Deterministic ProposalBuilder: TradingAgents decision -> TradeProposal.

The LLM contributes ONLY the decision text (direction) and rationale
reference. All execution numbers (price, quantity, edge components) come from
deterministic config/data inputs. If any required input is missing the builder
returns a NO_TRADE proposal — it never invents values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from tradingagents.survivor.trading.types import (
    Action,
    Side,
    TradeProposal,
    ceil_bps_cost,
    compute_conservative_net_edge_bps,
)

_DIRECTION_RE = re.compile(r"\b(BUY|SELL|HOLD)\b", re.IGNORECASE)


def parse_decision_direction(decision_text: str) -> str:
    """Extract BUY/SELL/HOLD from the final decision text (first match)."""
    match = _DIRECTION_RE.search(decision_text or "")
    return match.group(1).upper() if match else "UNKNOWN"


@dataclass(frozen=True)
class RiskInputs:
    """Deterministic edge/execution inputs. Every field must be supplied by
    config or market data — never generated from LLM free text."""

    quantity: int
    expected_probability_bps: int
    market_probability_bps: int
    spread_cost_bps: int
    slippage_bps: int
    fee_bps: int
    uncertainty_penalty_bps: int


def _no_trade(run_id: str, timestamp_utc: str, market: str, symbol: str, source: str) -> TradeProposal:
    return TradeProposal(
        run_id=run_id,
        timestamp_utc=timestamp_utc,
        market=market,
        symbol=symbol,
        side=Side.BUY.value,
        action=Action.NO_TRADE.value,
        source_decision=source,
    )


class ProposalBuilder:
    """Converts a final TradingAgents decision into a validated-shape proposal."""

    def build(
        self,
        *,
        run_id: str,
        timestamp_utc: str,
        market: str,
        symbol: str,
        decision_text: str,
        reference_ask_pence: int | None = None,
        inputs: RiskInputs | None = None,
        source_decision: str = "",
    ) -> TradeProposal:
        direction = parse_decision_direction(decision_text)

        # Fail closed to NO_TRADE whenever ANY required execution input is
        # missing: direction, price, or the deterministic edge components.
        if direction != "BUY" or inputs is None or not reference_ask_pence:
            return _no_trade(run_id, timestamp_utc, market, symbol, decision_text)

        notional = int(inputs.quantity) * int(reference_ask_pence)
        gross_edge_bps = int(inputs.expected_probability_bps) - int(inputs.market_probability_bps)
        spread_cost_bps = int(inputs.spread_cost_bps)
        slippage_bps = int(inputs.slippage_bps)
        fee_bps = int(inputs.fee_bps)
        uncertainty_bps = int(inputs.uncertainty_penalty_bps)
        net_edge_bps = compute_conservative_net_edge_bps(
            gross_edge_bps, spread_cost_bps, slippage_bps, fee_bps, uncertainty_bps
        )

        return TradeProposal(
            run_id=run_id,
            timestamp_utc=timestamp_utc,
            market=market,
            symbol=symbol,
            side=Side.BUY.value,
            action=Action.OPEN.value,
            quantity=int(inputs.quantity),
            reference_price_pence=int(reference_ask_pence),
            notional_pence=notional,
            expected_probability_bps=int(inputs.expected_probability_bps),
            market_probability_bps=int(inputs.market_probability_bps),
            gross_edge_bps=gross_edge_bps,
            spread_cost_bps=spread_cost_bps,
            slippage_bps=slippage_bps,
            fee_bps=fee_bps,
            uncertainty_penalty_bps=uncertainty_bps,
            conservative_net_edge_bps=net_edge_bps,
            estimated_fees_pence=ceil_bps_cost(notional, fee_bps),
            estimated_slippage_pence=ceil_bps_cost(notional, slippage_bps),
            estimated_spread_cost_pence=ceil_bps_cost(notional, spread_cost_bps),
            rationale_reference=f"final_trade_decision:{symbol}",
            source_decision=source_decision if source_decision else decision_text,
        )


def extract_paper_inputs(config: dict[str, Any]) -> RiskInputs | dict[str, int] | None:
    """Read deterministic paper inputs from config, or None when absent."""
    raw = (config or {}).get("survivor_paper_inputs")
    if not isinstance(raw, dict):
        return None
    return raw
