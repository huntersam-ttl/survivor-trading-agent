"""Shared deterministic helpers for Phase 2 paper-trading tests. No network, no LLMs."""

from datetime import datetime, timedelta, timezone

from tradingagents.survivor.execution.ledger import PaperLedger
from tradingagents.survivor.execution.pricing import MarketSnapshot
from tradingagents.survivor.risk.limits import RiskLimits
from tradingagents.survivor.trading.proposal import ProposalBuilder, RiskInputs
from tradingagents.survivor.trading.types import Action, Side, TradeProposal


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return now_utc().isoformat()


def default_limits(**over) -> RiskLimits:
    return RiskLimits(**over)


def default_inputs(**over) -> RiskInputs:
    """Edge components: gross 800bps, costs 250bps -> net edge 550bps (>= 500)."""
    base = {
        'quantity': 1,
        'expected_probability_bps': 6000,
        'market_probability_bps': 5200,
        'spread_cost_bps': 50,
        'slippage_bps': 50,
        'fee_bps': 50,
        'uncertainty_penalty_bps': 100,
    }
    base.update(over)
    return RiskInputs(**base)


def make_proposal(**over) -> TradeProposal:
    """Valid OPEN proposal: BUY 1 unit @100p = £1.00 notional, net edge 550bps."""
    fields = {
        'run_id': over.pop("run_id", "run_t1"),
        'timestamp_utc': over.pop("timestamp_utc", iso_now()),
        'market': over.pop("market", "stock"),
        'symbol': over.pop("symbol", "AAPL"),
        'side': over.pop("side", Side.BUY.value),
        'action': over.pop("action", Action.OPEN.value),
        'quantity': over.pop("quantity", 1),
        'reference_price_pence': over.pop("reference_price_pence", 100),
        'notional_pence': over.pop("notional_pence", 100),
        'expected_probability_bps': over.pop("expected_probability_bps", 6000),
        'market_probability_bps': over.pop("market_probability_bps", 5200),
        'gross_edge_bps': over.pop("gross_edge_bps", 800),
        'spread_cost_bps': over.pop("spread_cost_bps", 50),
        'slippage_bps': over.pop("slippage_bps", 50),
        'fee_bps': over.pop("fee_bps", 50),
        'uncertainty_penalty_bps': over.pop("uncertainty_penalty_bps", 100),
        'conservative_net_edge_bps': over.pop("conservative_net_edge_bps", 550),
        'estimated_fees_pence': over.pop("estimated_fees_pence", 1),
        'estimated_slippage_pence': over.pop("estimated_slippage_pence", 1),
        'estimated_spread_cost_pence': over.pop("estimated_spread_cost_pence", 1),
        'rationale_reference': over.pop("rationale_reference", "final_trade_decision:AAPL"),
        'source_decision': over.pop("source_decision", "BUY"),
        'proposal_id': over.pop("proposal_id", None),
    }
    assert not over, f"unknown overrides: {over}"
    return TradeProposal(**fields)


def fresh_snapshot(symbol: str = "AAPL", bid: int = 99, ask: int = 100) -> MarketSnapshot:
    return MarketSnapshot(symbol=symbol, bid_pence=bid, ask_pence=ask, timestamp_utc=iso_now())


def stale_snapshot(symbol: str = "AAPL", age_sec: int = 600) -> MarketSnapshot:
    ts = (now_utc() - timedelta(seconds=age_sec)).isoformat()
    return MarketSnapshot(symbol=symbol, bid_pence=99, ask_pence=100, timestamp_utc=ts)


def make_builder() -> ProposalBuilder:
    return ProposalBuilder()


def tmp_ledger(tmpdir) -> PaperLedger:
    import os

    return PaperLedger(db_path=os.path.join(str(tmpdir), "paper.db"))
