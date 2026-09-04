"""Single deterministic autonomous PAPER cycle: run_survivor_cycle().

Flow: halt check -> paper-only check -> overlap lock -> mark/resolution ->
scan (zero LLM) -> top-N candidates -> budget preflight -> research (one at a
time) -> proposal -> validate -> RiskEngine -> PaperBroker (unless dry run)
-> persist -> structured report. NO_TRADE is the default at every stage.
System-level corruption (ledger/accounting) halts the entire runtime.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger
from tradingagents.survivor.autonomy import halt as halt_mod
from tradingagents.survivor.autonomy.budget import budget_preflight
from tradingagents.survivor.autonomy.injection import build_evidence
from tradingagents.survivor.autonomy.lock import CycleLock
from tradingagents.survivor.autonomy.state import RuntimeState
from tradingagents.survivor.execution.ledger import LedgerCorruptionError
from tradingagents.survivor.execution.paper_broker import PaperBroker
from tradingagents.survivor.execution.pricing import MarketSnapshot as QuoteSnapshot
from tradingagents.survivor.execution.resolution import settle_prediction_position
from tradingagents.survivor.markets.filters import scan_limits_from_env
from tradingagents.survivor.markets.scanner import MarketScanner
from tradingagents.survivor.markets.types import Candidate
from tradingagents.survivor.policy import SurvivorPolicy
from tradingagents.survivor.risk.engine import RiskEngine
from tradingagents.survivor.risk.limits import risk_limits_from_env
from tradingagents.survivor.trading.proposal import ProposalBuilder, RiskInputs

BOOL_TRUE = ("true", "1", "yes", "on")


def _bps_to_pence(bps: int | None) -> int | None:
    """Binary-market probability bps -> paper price in pence (52% = 52p)."""
    if bps is None:
        return None
    return max(1, bps // 100)


@dataclass
class ResearchResult:
    """What one research pass produced (fakeable in tests)."""

    decision_text: str = "NO_TRADE"
    expected_probability_bps: int | None = None
    quantity: int | None = None
    spread_cost_bps: int = 0
    slippage_bps: int = 0
    fee_bps: int = 0
    uncertainty_penalty_bps: int = 0
    ai_cost_pence: int = 0
    run_id: str | None = None
    status: str = "OK"          # OK | FAILED
    reason: str | None = None


@dataclass
class CycleReport:
    cycle_id: str
    status: str = "ACTIVE"                     # ACTIVE | EXTERNAL_HALT | CYCLE_ALREADY_RUNNING | DISABLED | DRY_RUN | HALTED | ERROR
    reason: str | None = None
    markets_discovered: int = 0
    passed_filters: int = 0
    research_candidates: int = 0
    ai_researched: int = 0
    skipped_budget: int = 0
    trade_proposals: int = 0
    approved: int = 0
    rejected: int = 0
    paper_trades: int = 0
    ai_cost_pence: int = 0
    duration_sec: float = 0.0
    candidate_failures: list[dict] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def render(self) -> str:
        gbp = lambda p: f"£{p / 100:.2f}"  # noqa: E731
        return (
            "\nSURVIVOR CYCLE\n\n"
            f"Cycle: {self.cycle_id}\n"
            f"Markets discovered: {self.markets_discovered}\n"
            f"Passed deterministic filters: {self.passed_filters}\n"
            f"Research candidates: {self.research_candidates}\n"
            f"AI researched: {self.ai_researched}\n"
            f"Skipped for budget: {self.skipped_budget}\n"
            f"Trade proposals: {self.trade_proposals}\n"
            f"Approved: {self.approved}\n"
            f"Rejected: {self.rejected}\n"
            f"Paper trades: {self.paper_trades}\n"
            f"AI cost: {gbp(self.ai_cost_pence)}\n"
            f"Cycle duration: {self.duration_sec:.0f} sec\n"
            f"State: {self.status}\n"
        )


def autonomy_enabled(config: dict | None = None) -> bool:
    env_val = str((config or {}).get("survivor_autonomy_enabled", "")).strip().lower()
    if env_val:
        return env_val in BOOL_TRUE
    return os.environ.get("TRADINGAGENTS_SURVIVOR_AUTONOMY_ENABLED", "").strip().lower() in BOOL_TRUE


def dry_run_enabled(config: dict | None = None) -> bool:
    env_val = str((config or {}).get("survivor_dry_run", "")).strip().lower()
    if env_val:
        return env_val in BOOL_TRUE
    return os.environ.get("SURVIVOR_DRY_RUN", "").strip().lower() in BOOL_TRUE


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default

def _default_research(candidate: Candidate, quote: QuoteSnapshot, config: dict, run_id: str) -> ResearchResult:
    """Default research: run the existing TradingAgents multi-agent pipeline
    (analysts -> bull/bear -> research manager -> trader -> risk -> PM) under
    the mandatory SURVIVOR ModelRouter/BudgetManager control plane."""
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    question = build_evidence("market_question", candidate.snapshot.question).render()
    research_config = {
        **DEFAULT_CONFIG,
        **(config or {}),
        "survivor_paper_inputs": {
            **(config or {}).get("survivor_paper_inputs", {}),
            "symbol": candidate.snapshot.market_id,
            "market": candidate.snapshot.market_type,
            "bid_pence": quote.bid_pence,
            "ask_pence": quote.ask_pence,
            "snapshot_timestamp_utc": quote.timestamp_utc,
            "run_id": run_id,
        },
    }
    graph = TradingAgentsGraph(config=research_config)
    final_state, _signal = graph.propagate(question, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    inputs = research_config.get("survivor_paper_inputs", {})
    return ResearchResult(
        decision_text=str(final_state.get("final_trade_decision") or "NO_TRADE"),
        expected_probability_bps=inputs.get("expected_probability_bps"),
        quantity=inputs.get("quantity"),
        spread_cost_bps=inputs.get("spread_cost_bps", 0),
        slippage_bps=inputs.get("slippage_bps", 0),
        fee_bps=inputs.get("fee_bps", 0),
        uncertainty_penalty_bps=inputs.get("uncertainty_penalty_bps", 0),
        run_id=run_id,
    )


def run_survivor_cycle(
    config: dict | None = None,
    *,
    adapter: Any | None = None,
    research_fn: Callable[[Candidate, QuoteSnapshot, dict, str], ResearchResult] | None = None,
    paper_ledger_path: str | None = None,
    runtime_state: RuntimeState | None = None,
    usage_ledger: InferenceUsageLedger | None = None,
    policy: SurvivorPolicy | None = None,
    now: datetime | None = None,
    lock_path: str | None = None,
) -> CycleReport:
    started = time.time()
    current = now or datetime.now(timezone.utc)
    config = config or {}
    cycle_id = uuid.uuid4().hex[:12]
    report = CycleReport(cycle_id=cycle_id)
    state = runtime_state or RuntimeState()

    # 1. External kill switch: no scan, no AI, no proposals, no executions.
    if halt_mod.is_halted():
        report.status = "EXTERNAL_HALT"
        report.reason = "HALT file present; autonomous cycle refused"
        state.start_cycle(cycle_id, now=current)
        state.finish_cycle(cycle_id, "EXTERNAL_HALT", report.reason, now=current)
        return report

    # 2. Paper-only + explicit opt-in
    if not autonomy_enabled(config):
        report.status = "DISABLED"
        report.reason = "TRADINGAGENTS_SURVIVOR_AUTONOMY_ENABLED is not enabled"
        return report

    limits = risk_limits_from_env()
    paper_ledger = None
    if paper_ledger_path:
        from tradingagents.survivor.execution.ledger import PaperLedger

        paper_ledger = PaperLedger(db_path=paper_ledger_path)

    lock = CycleLock(lock_path=lock_path)
    if not lock.acquire():
        report.status = "CYCLE_ALREADY_RUNNING"
        report.reason = "another cycle holds the lock"
        return report

    try:
        try:
            state.start_cycle(cycle_id, now=current)
            _run_cycle_body(
                report, cycle_id, config, current, limits, adapter, research_fn,
                paper_ledger, state, usage_ledger, policy,
            )
        except LedgerCorruptionError as exc:
            halt_mod.set_halt()
            report.status = "HALTED"
            report.reason = f"paper accounting corruption: {exc}"
    finally:
        lock.release()

    report.duration_sec = time.time() - started
    metrics = {
        "discovered": report.markets_discovered, "passed": report.passed_filters,
        "research_candidates": report.research_candidates, "researched": report.ai_researched,
        "skipped_budget": report.skipped_budget, "proposals": report.trade_proposals,
        "approved": report.approved, "rejected": report.rejected,
        "executed": report.paper_trades, "ai_cost_pence": report.ai_cost_pence,
        "duration_sec": report.duration_sec,
    }
    state.finish_cycle(cycle_id, report.status, report.reason, metrics, now=current)
    return report


def _run_cycle_body(
    report: CycleReport,
    cycle_id: str,
    config: dict,
    current: datetime,
    limits: Any,
    adapter: Any | None,
    research_fn: Callable[[Candidate, QuoteSnapshot, dict, str], ResearchResult] | None,
    paper_ledger_path: str | None,
    state: RuntimeState,
    usage_ledger: InferenceUsageLedger | None,
    policy: SurvivorPolicy | None,
) -> None:

    broker = PaperBroker(limits=limits, ledger=paper_ledger_path)
    broker.ensure_initialized()
    broker.refresh_daily_pnl()
    trading_halted = broker.trading_state() != "ACTIVE"

    # 3.-4. Deterministic scan (zero LLM)
    if adapter is None:
        from tradingagents.survivor.markets.polymarket_adapter import PolymarketAdapter

        adapter = PolymarketAdapter()
    scanner = MarketScanner(
        adapter,
        limits=scan_limits_from_env(),
        max_candidates_per_cycle=_int_env("SURVIVOR_MAX_CANDIDATES_PER_CYCLE", 40),
        max_research_candidates_per_cycle=_int_env("SURVIVOR_MAX_RESEARCH_PER_CYCLE", 3),
    )
    open_symbols = frozenset(s for s, p in broker.portfolio.positions.items() if p.quantity > 0)
    scan = scanner.scan(now=current, open_position_symbols=open_symbols, trading_halted=trading_halted)
    report.markets_discovered = scan.discovered
    report.passed_filters = len(scan.candidates)
    report.research_candidates = len(scan.top)

    usage = usage_ledger or InferenceUsageLedger()
    survivor_policy = policy or SurvivorPolicy.from_env()
    research_fn = research_fn or _default_research
    builder = ProposalBuilder()
    executed_runs: dict[str, str] = {}
    from tradingagents.survivor.autonomy.state import snapshot_data_hash
    from tradingagents.survivor.evaluation.store import EvaluationStore
    from tradingagents.survivor.evaluation.types import PredictionRecord, TradeOutcome
    from tradingagents.survivor.evaluation.versioning import strategy_identity

    eval_store = EvaluationStore(db_path=config.get("_evaluation_db_path"))
    version, chash = strategy_identity({"min_edge_bps": limits.min_conservative_net_edge_bps})


    # 5.-9. Research top candidates, one at a time, with budget preflight
    for candidate in scan.top:
        snapshot = candidate.snapshot
        try:
            state.record_observation(cycle_id, snapshot, decision_time_utc=current.isoformat())
            # point-in-time safety: no future data may enter a decision
            snap_ts = datetime.fromisoformat(
                (snapshot.source_timestamp_utc or snapshot.timestamp_utc).replace("Z", "+00:00")
            )
            if snap_ts > current:
                state.record_candidate(cycle_id, snapshot.market_id, candidate.rank,
                                       candidate.score, "NO_TRADE", "SKIPPED", "STALE_DATA")
                continue

            # budget preflight BEFORE any AI call (no partial research)
            preflight = budget_preflight(survivor_policy, usage)
            if not preflight.ok:
                report.skipped_budget += 1
                state.record_candidate(cycle_id, snapshot.market_id, candidate.rank,
                                       candidate.score, "NO_TRADE", "SKIPPED", "AI_BUDGET_UNAVAILABLE")
                continue

            run_id = f"{cycle_id}_{snapshot.market_id}"
            quote = QuoteSnapshot(
                symbol=snapshot.market_id,
                bid_pence=_bps_to_pence(snapshot.bid),
                ask_pence=_bps_to_pence(snapshot.ask),
                timestamp_utc=snapshot.source_timestamp_utc or snapshot.timestamp_utc,
            )
            result = research_fn(candidate, quote, config, run_id)
            if result.status != "OK" or result.expected_probability_bps is None or result.quantity is None:
                report.candidate_failures.append(
                    {"market_id": snapshot.market_id, "reason": result.reason or "RESEARCH_FAILED"}
                )
                state.record_research(cycle_id, snapshot.market_id, run_id, "FAILED", result.reason)
                continue

            report.ai_researched += 1
            report.ai_cost_pence += result.ai_cost_pence
            state.record_research(cycle_id, snapshot.market_id, run_id, "OK")
            eval_store.record_prediction(PredictionRecord(
                cycle_id=cycle_id, run_id=run_id, proposal_id=f"{run_id}_prop",
                market_id=snapshot.market_id, timestamp_utc=current.isoformat(),
                strategy_version=version, config_hash=chash,
                category=snapshot.market_type,
                predicted_probability=result.expected_probability_bps / 10000,
                market_probability=(snapshot.market_probability_bps or 0) / 10000,
                gross_edge_bps=result.expected_probability_bps - (snapshot.market_probability_bps or 0),
                net_edge_bps=result.expected_probability_bps - (snapshot.market_probability_bps or 0)
                - result.spread_cost_bps - result.slippage_bps - result.fee_bps
                - result.uncertainty_penalty_bps,
                ai_cost_pence=result.ai_cost_pence,
                snapshot_data_hash=snapshot_data_hash(snapshot),
            ))
            eval_store.record_cost_attribution(run_id, snapshot.market_id, cycle_id,
                                               result.ai_cost_pence)

            proposal = builder.build(
                run_id=run_id,
                timestamp_utc=current.isoformat(),
                market=snapshot.market_type,
                symbol=snapshot.market_id,
                decision_text=result.decision_text,
                reference_ask_pence=_bps_to_pence(snapshot.ask),
                inputs=RiskInputs(
                    quantity=result.quantity,
                    expected_probability_bps=result.expected_probability_bps,
                    market_probability_bps=snapshot.market_probability_bps,
                    spread_cost_bps=result.spread_cost_bps,
                    slippage_bps=result.slippage_bps,
                    fee_bps=result.fee_bps,
                    uncertainty_penalty_bps=result.uncertainty_penalty_bps,
                ),
            )
            state.record_candidate(cycle_id, snapshot.market_id, candidate.rank,
                                   candidate.score, proposal.action, "PROPOSED")
            if proposal.action != "OPEN":
                continue
            report.trade_proposals += 1

            decision = RiskEngine(limits, duplicate_checker=broker.ledger.has_executed).evaluate(
                proposal, broker.portfolio, quote, timestamp=current,
            )
            if decision.approved:
                report.approved += 1
            else:
                report.rejected += 1
            if decision.approved and not dry_run_enabled(config):
                # T. conservative fill model: cap notional vs available liquidity
                # (conservative cross-currency proxy: notional in USD cents vs
                # liquidity in USD minor units; £1 position cap unchanged)
                if snapshot.liquidity is not None:
                    fill_fraction = float(os.environ.get("SURVIVOR_FILL_FRACTION", "0.05"))
                    notional_usd_cents = proposal.notional_pence * 100
                    if notional_usd_cents > fill_fraction * snapshot.liquidity.minor_units:
                        state.record_candidate(cycle_id, snapshot.market_id, candidate.rank,
                                               candidate.score, "NO_TRADE", "SKIPPED", "FILL_RISK")
                        continue
                broker.execute(proposal, quote, decision)
                report.paper_trades += 1
                executed_runs[snapshot.market_id] = run_id
        except (LedgerCorruptionError, RuntimeError) as exc:
            # system-level failure -> halt the entire runtime
            halt_mod.set_halt()
            report.status = "HALTED"
            report.reason = f"system failure: {exc}"
            return
        except Exception as exc:  # noqa: BLE001 - per-candidate failure isolation
            report.candidate_failures.append(
                {"market_id": candidate.snapshot.market_id, "reason": str(exc)[:200]}
            )
            state.record_candidate(cycle_id, candidate.snapshot.market_id, candidate.rank,
                                   candidate.score, "NO_TRADE", "FAILED", str(exc)[:200])
            continue

    # 10. mark open long positions at BID (no AI)
    marks = []
    for symbol, position in broker.portfolio.positions.items():
        if position.quantity <= 0:
            continue
        snap = adapter.get_snapshot(symbol) if hasattr(adapter, "get_snapshot") else None
        if snap is not None and snap.bid is not None and snap.bid > 0:
            marks.append(QuoteSnapshot(symbol=symbol, bid_pence=_bps_to_pence(snap.bid),
                                       ask_pence=_bps_to_pence(snap.ask), timestamp_utc=snap.timestamp_utc))
    if marks and not dry_run_enabled(config):
        broker.mark_to_market(marks)

    # 11. deterministic resolution settlement for resolved prediction markets
    if not dry_run_enabled(config):
        for symbol in list(broker.portfolio.positions.keys()):
            if hasattr(adapter, "get_resolution_status"):
                resolution = adapter.get_resolution_status(symbol)
                resolved_yes = resolution.value == "YES"
                if resolution.value in ("YES", "NO"):
                    settle_event = settle_prediction_position(broker, symbol, resolved_yes=resolved_yes)
                    if settle_event is not None:
                        run_id = executed_runs.get(symbol, f"unknown_{symbol}")
                        outcome = 1.0 if resolved_yes else 0.0
                        eval_store.attach_outcome(f"{run_id}_prop", outcome, current.isoformat())
                        eval_store.record_trade(TradeOutcome(
                            cycle_id=cycle_id, run_id=run_id,
                            proposal_id=f"{run_id}_prop", market_id=symbol,
                            timestamp_utc=current.isoformat(),
                            strategy_version=version, config_hash=chash,
                            category="prediction_binary",
                            quantity=settle_event["quantity"],
                            entry_price_pence=settle_event["execution_price_pence"],
                            fees_pence=settle_event.get("fees_pence_total", 0),
                            slippage_pence=0,
                            gross_pnl_pence=settle_event["realized_pnl_pence"],
                            ai_cost_pence=0,
                            realized_return_bps=int(
                                (settle_event["realized_pnl_pence"] * 10000)
                                // max(1, settle_event["cost_basis_pence"])
                            ),
                            predicted_net_edge_bps=0,
                            snapshot_data_hash="",
                            outcome=outcome,
                        ))


    report.status = "DRY_RUN" if dry_run_enabled(config) else "ACTIVE"





