"""Phase 2 execution boundary (opt-in).

TradingAgents final decision -> ProposalBuilder -> Validator -> RiskEngine ->
PaperBroker -> Ledger. Only runs when TRADINGAGENTS_SURVIVOR_PAPER_ENABLED is
truthy; otherwise upstream behavior is completely unchanged. The LLM never
executes anything: this module is fully deterministic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from tradingagents.survivor.execution.ledger import LedgerCorruptionError, PaperLedger
from tradingagents.survivor.execution.paper_broker import (
    DuplicateProposalError,
    ExecutionBlockedError,
    PaperBroker,
)
from tradingagents.survivor.execution.pricing import MarketSnapshot
from tradingagents.survivor.risk.engine import RiskEngine
from tradingagents.survivor.risk.limits import risk_limits_from_env
from tradingagents.survivor.risk.result import ReasonCode, RiskDecision
from tradingagents.survivor.trading.proposal import ProposalBuilder, RiskInputs

logger = logging.getLogger(__name__)

_BOOL_TRUE = ("true", "1", "yes", "on")


def is_paper_enabled(config: dict | None = None) -> bool:
    """Explicit opt-in flag for paper execution (default: disabled)."""
    env_val = str((config or {}).get("survivor_paper_enabled", "")).strip().lower()
    if env_val:
        return env_val in _BOOL_TRUE
    import os

    env_val = os.environ.get("TRADINGAGENTS_SURVIVOR_PAPER_ENABLED", "").strip().lower()
    return env_val in _BOOL_TRUE


@dataclass
class PaperPipelineResult:
    status: str  # DISABLED | NO_TRADE | EXECUTED | REJECTED | HALTED | ERROR
    proposal: Any = None
    risk_decision: RiskDecision | None = None
    event: dict | None = None


def execute_final_decision(
    final_state: dict,
    config: dict,
    ledger: PaperLedger | None = None,
) -> PaperPipelineResult:
    """Deterministic boundary: convert a final decision into a paper trade."""
    if not is_paper_enabled(config):
        return PaperPipelineResult(status="DISABLED")

    limits = risk_limits_from_env()
    ledger = ledger or PaperLedger()
    broker = PaperBroker(limits=limits, ledger=ledger)
    broker.ensure_initialized()
    broker.refresh_daily_pnl()

    paper_inputs = config.get("survivor_paper_inputs") or {}
    builder = ProposalBuilder()
    decision_text = str((final_state or {}).get("final_trade_decision") or "")

    risk_inputs = None
    ask = paper_inputs.get("ask_pence")
    bid = paper_inputs.get("bid_pence")
    if ask and bid:
        try:
            risk_inputs = RiskInputs(
                quantity=int(paper_inputs["quantity"]),
                expected_probability_bps=int(paper_inputs["expected_probability_bps"]),
                market_probability_bps=int(paper_inputs["market_probability_bps"]),
                spread_cost_bps=int(paper_inputs["spread_cost_bps"]),
                slippage_bps=int(paper_inputs["slippage_bps"]),
                fee_bps=int(paper_inputs["fee_bps"]),
                uncertainty_penalty_bps=int(paper_inputs["uncertainty_penalty_bps"]),
            )
        except (KeyError, TypeError, ValueError):
            risk_inputs = None  # missing component -> NO_TRADE (never invent)

    proposal = builder.build(
        run_id=str(paper_inputs.get("run_id") or config.get("run_id") or "paper_run"),
        timestamp_utc=paper_inputs.get("snapshot_timestamp_utc")
        or datetime.now(timezone.utc).isoformat(),
        market=str(paper_inputs.get("market") or "stock"),
        symbol=str(paper_inputs.get("symbol") or ""),
        decision_text=decision_text,
        reference_ask_pence=ask,
        inputs=risk_inputs,
    )
    ledger.append_event(
        event_type="PROPOSAL_RECEIVED",
        run_id=proposal.run_id,
        symbol=proposal.symbol,
        side=proposal.side,
        quantity=proposal.quantity,
        notional_pence=proposal.notional_pence,
        proposal_id=proposal.proposal_id,
        risk_decision="PENDING",
        risk_reason=proposal.action,
    )

    if proposal.action != "OPEN":
        return PaperPipelineResult(status="NO_TRADE", proposal=proposal)

    snapshot = MarketSnapshot(
        symbol=proposal.symbol,
        bid_pence=bid,
        ask_pence=ask,
        timestamp_utc=paper_inputs.get("snapshot_timestamp_utc") or "",
    )
    engine = RiskEngine(limits, duplicate_checker=ledger.has_executed)
    decision = engine.evaluate(proposal, broker.portfolio, snapshot)

    event_type = (
        "RISK_APPROVED" if decision.approved
        else "DAILY_HALT" if decision.reason_code == ReasonCode.DAILY_LOSS_HALT
        else "DRAWDOWN_HALT" if decision.reason_code == ReasonCode.DRAWDOWN_HALT
        else "RISK_REJECTED"
    )
    ledger.append_event(
        event_type=event_type,
        run_id=proposal.run_id,
        symbol=proposal.symbol,
        side=proposal.side,
        quantity=proposal.quantity,
        notional_pence=proposal.notional_pence,
        proposal_id=proposal.proposal_id,
        risk_decision=decision.status.value,
        risk_reason=decision.reason_code.value,
    )

    if not decision.approved:
        status = "HALTED" if decision.halted else "REJECTED"
        return PaperPipelineResult(status=status, proposal=proposal, risk_decision=decision)

    try:
        event = broker.execute(proposal, snapshot, decision)
    except (DuplicateProposalError, ExecutionBlockedError, LedgerCorruptionError) as exc:
        logger.warning("Paper execution blocked: %s", exc)
        return PaperPipelineResult(
            status="REJECTED", proposal=proposal, risk_decision=decision,
        )
    return PaperPipelineResult(
        status="EXECUTED", proposal=proposal, risk_decision=decision, event=event,
    )
