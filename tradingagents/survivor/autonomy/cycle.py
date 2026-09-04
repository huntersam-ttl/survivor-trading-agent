"""Single deterministic autonomous PAPER cycle: run_survivor_cycle().

Flow: halt check -> paper-only check -> overlap lock -> mark/resolution ->
scan (zero LLM) -> top-N candidates -> budget preflight -> research (one at a
time) -> proposal -> validate -> RiskEngine -> PaperBroker (unless dry run)
-> persist -> structured report. NO_TRADE is the default at every stage.
System-level corruption (ledger/accounting) halts the entire runtime.
"""

from __future__ import annotations

import os
import re
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
    warnings: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def render(self) -> str:
        gbp = lambda p: f"£{p / 100:.2f}"  # noqa: E731
        lines = [
            "",
            "SURVIVOR CYCLE",
            "",
            f"Cycle: {self.cycle_id}",
            f"Markets discovered: {self.markets_discovered}",
            f"Passed deterministic filters: {self.passed_filters}",
            f"Research candidates: {self.research_candidates}",
            f"AI researched: {self.ai_researched}",
            f"Skipped for budget: {self.skipped_budget}",
            f"Trade proposals: {self.trade_proposals}",
            f"Approved: {self.approved}",
            f"Rejected: {self.rejected}",
            f"Paper trades: {self.paper_trades}",
            f"AI cost: {gbp(self.ai_cost_pence)}",
            f"Cycle duration: {self.duration_sec:.0f} sec",
            f"State: {self.status}",
        ]
        if self.candidate_failures:
            # Fail closed: research problems are surfaced, never silent.
            lines.append("Candidate failures:")
            for failure in self.candidate_failures:
                lines.append(f"  - {failure.get('market_id')}: {failure.get('reason')}")
        return "\n".join(lines) + "\n"


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


def _record_shadow_decisions(config: dict, cycle_id: str, snapshot, result, current) -> None:
    """Phase 6 shadow mode: for experiments in SHADOW_TESTING, record a
    FICTIONAL parallel decision. Best-effort; never touches PaperBroker and
    never breaks the paper cycle."""
    try:
        from tradingagents.survivor.learning.experiments import ExperimentStatus
        from tradingagents.survivor.learning.registry import ExperimentStore
        from tradingagents.survivor.learning.shadow import ShadowStore, shadow_decision_for

        experiment_store = ExperimentStore(config.get("_learning_db_path"))
        shadow_experiments = experiment_store.experiments(
            status=ExperimentStatus.SHADOW_TESTING)
        if not shadow_experiments:
            return
        shadow = ShadowStore(config.get("_shadow_db_path"))
        for experiment in shadow_experiments:
            edge = (result.expected_probability_bps or 0) \
                - (snapshot.market_probability_bps or 0) \
                - result.spread_cost_bps - result.slippage_bps \
                - result.fee_bps - result.uncertainty_penalty_bps
            shadow.record(shadow_decision_for(
                experiment, cycle_id=cycle_id, market_id=snapshot.market_id,
                timestamp_utc=current.isoformat(),
                survivor_probability=(result.expected_probability_bps or 0) / 10000,
                market_probability=(snapshot.market_probability_bps or 0) / 10000,
                conservative_edge_bps=edge,
            ))
    except Exception:  # noqa: BLE001 - shadow must never break the paper cycle
        pass


_P_YES_RE = re.compile(r"P\s*\(\s*YES\s*\)\s*[:=]\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%", re.IGNORECASE)

# API-key environment variables for the providers referenced by the ModelRouter
# role routes (deterministic preflight — never an LLM call).
_PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "xai": "XAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def _market_evidence_context(candidate: Candidate, quote: QuoteSnapshot) -> str:
    """Deterministic research context built ONLY from the stored decision-time
    snapshot. Untrusted market text is wrapped by build_evidence; the agents
    are instructed never to replace the snapshot with fresh market data."""
    snap = candidate.snapshot
    question_block = build_evidence("market_question", snap.question).render()
    header = "\n".join([
        "INSTRUMENT: binary prediction market (paper-only trial; identifier only, not a stock ticker).",
        f"market_id: {snap.market_id}",
        f"provider: {snap.provider}",
        f"symbol_or_slug: {snap.symbol_or_slug}",
        f"market_type: {snap.market_type}",
        f"resolution_status: {snap.resolution_status.value}",
        f"market-implied YES probability (bps of 1.00): {snap.market_probability_bps}",
        f"YES bid/ask (bps): {snap.bid}/{snap.ask}",
        f"liquidity (USD minor units): {snap.liquidity.minor_units if snap.liquidity else 'unknown'}",
        f"24h volume (USD minor units): {snap.volume_24h.minor_units if snap.volume_24h else 'unknown'}",
        f"close time (UTC): {snap.close_time_utc or 'unknown'}",
        f"decision-time snapshot timestamp (UTC): "
        f"{snap.source_timestamp_utc or snap.timestamp_utc or quote.timestamp_utc}",
        "resolution criteria: as defined by the official market question and the provider's rules "
        "(see the untrusted question block below).",
        "POINT-IN-TIME RULE: reason ONLY from the decision-time snapshot above. News, search and "
        "market-data tools may supply general background, but any current market quote that "
        "differs from the snapshot MUST NOT replace it.",
        "FINAL DECISION FORMAT: your final decision MUST include a line exactly "
        "'P(YES): <NN>%' where <NN> is your calibrated probability estimate (0-100) that this "
        "binary event resolves YES.",
    ])
    return f"{header}\n\n{question_block}"


def _research_failure(reason: str, run_id: str | None) -> ResearchResult:
    return ResearchResult(status="FAILED", reason=reason[:200], run_id=run_id)


def _classify_inference_error(exc: Exception) -> str:
    """Fail-closed reason classification for research backend errors."""
    name = type(exc).__name__
    msg = str(exc)
    low = msg.lower()
    if name == "BudgetExhaustedError" or "budget" in low and "exhaust" in low:
        return f"AI_BUDGET_UNAVAILABLE: {msg}"
    if name in ("ModelNotAllowedError", "UnknownPriceError", "UnhealthyProviderError", "NoRouteError") \
            or "no model route" in low or "not allowed" in low or "unpriced" in low:
        return f"NO_MODEL_ROUTE: {msg}"
    if any(k in low for k in ("api key", "unauthorized", "401", "auth", "credential", "api_key")):
        return f"NO_LLM_PROVIDER: {msg}"
    return f"INFERENCE_FAILED: {msg}"


def _default_research(candidate: Candidate, quote: QuoteSnapshot, config: dict, run_id: str) -> ResearchResult:
    """Default research: run the existing TradingAgents multi-agent pipeline
    (analysts -> bull/bear -> research manager -> trader -> risk -> PM) under
    the mandatory SURVIVOR ModelRouter/BudgetManager control plane.

    The candidate is a binary prediction market, not a stock ticker, so the
    stored decision-time snapshot is injected as the deterministic instrument
    context (untrusted text wrapped by build_evidence) and the market_id/slug
    is used purely as an identifier — never as a fake ticker.
    """
    config = config or {}
    snapshot = candidate.snapshot
    symbol = snapshot.symbol_or_slug or snapshot.market_id

    try:
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph
    except ImportError as exc:
        return _research_failure(f"RESEARCH_BACKEND_UNAVAILABLE: {exc}", run_id)

    research_config = {
        **DEFAULT_CONFIG,
        **config,
        # Phase 1 guarded inference path is mandatory for autonomous research:
        # every agent call is role-routed, budget-authorized and settled.
        "survivor_enabled": True,
        # The graph must NOT execute anything itself: the autonomous cycle owns
        # RiskEngine + PaperBroker (and dry-run stops before execute).
        "survivor_paper_enabled": False,
        "survivor_run_id": run_id,
        "survivor_instrument_context": _market_evidence_context(candidate, quote),
        "survivor_paper_inputs": {
            **(config or {}).get("survivor_paper_inputs", {}),
            "symbol": snapshot.market_id,
            "market": snapshot.market_type,
            "bid_pence": quote.bid_pence,
            "ask_pence": quote.ask_pence,
            "snapshot_timestamp_utc": quote.timestamp_utc,
            "run_id": run_id,
        },
    }
    # Route router settlements into the cycle's usage ledger DB so budget
    # preflight and cost attribution see the same spend.
    usage = config.get("_usage_ledger")
    if usage is not None:
        research_config["survivor_usage_ledger_path"] = str(usage.db_path)

    # Central OpenRouter operation (optional): when the operator configures an
    # OpenRouter key + model, the graph's construction clients use the existing
    # OpenRouter provider path too, so ONE key runs the whole research run.
    or_model = os.environ.get("SURVIVOR_OPENROUTER_MODEL", "").strip()
    if or_model and os.environ.get("OPENROUTER_API_KEY", "").strip():
        or_strong = os.environ.get("SURVIVOR_OPENROUTER_STRONG_MODEL", "").strip() or or_model
        research_config["llm_provider"] = "openrouter"
        research_config["deep_think_llm"] = or_strong
        research_config["quick_think_llm"] = or_model

    try:
        graph = TradingAgentsGraph(config=research_config)
        trade_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        final_state, _signal = graph.propagate(symbol, trade_date, asset_type="prediction_market")
    except ImportError as exc:
        return _research_failure(f"RESEARCH_BACKEND_UNAVAILABLE: {exc}", run_id)
    except Exception as exc:  # noqa: BLE001 - classified below, never silent
        return _research_failure(_classify_inference_error(exc), run_id)

    usage_summary = usage.get_run_usage_summary(run_id) if usage is not None else {}
    ai_cost_pence = int(usage_summary.get("total_cost_pence") or 0)

    decision_text = str((final_state or {}).get("final_trade_decision") or "")
    match = _P_YES_RE.search(decision_text)
    if not match:
        return _research_failure(
            "RESEARCH_NO_PROBABILITY_ESTIMATE: final decision missing 'P(YES): <NN>%' line",
            run_id,
        )
    probability_bps = min(10000, max(0, round(float(match.group(1)) * 100)))

    return ResearchResult(
        decision_text=decision_text,
        expected_probability_bps=probability_bps,
        # Deterministic paper sizing: one contract; the RiskEngine enforces the
        # position/notional caps. The LLM contributes direction + probability only.
        quantity=1,
        spread_cost_bps=max(0, (snapshot.ask or 0) - (snapshot.bid or 0)),
        slippage_bps=0,
        fee_bps=0,
        uncertainty_penalty_bps=0,
        ai_cost_pence=ai_cost_pence,
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
    # Decision time is NOW (after the scan): adapter snapshots are stamped at
    # fetch time, which is necessarily later than the cycle-start timestamp, so
    # the point-in-time check below must compare against this moment — not the
    # cycle start — or every freshly fetched snapshot looks like future data.
    current = datetime.now(timezone.utc)

    usage = usage_ledger or InferenceUsageLedger()
    # Expose the cycle's usage ledger to the research backend so router
    # settlements land in the same DB and run costs can be attributed.
    config["_usage_ledger"] = usage
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
    trial_id = config.get("_trial_id") or "no-trial"



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
                # fail closed: a selected candidate must never silently vanish
                report.candidate_failures.append(
                    {"market_id": snapshot.market_id, "reason": "SKIPPED_STALE_DATA"}
                )
                continue

            # budget preflight BEFORE any AI call (no partial research)
            preflight = budget_preflight(survivor_policy, usage)
            if not preflight.ok:
                report.skipped_budget += 1
                state.record_candidate(cycle_id, snapshot.market_id, candidate.rank,
                                       candidate.score, "NO_TRADE", "SKIPPED", "AI_BUDGET_UNAVAILABLE")
                continue

            run_id = f"{trial_id}_{cycle_id}_{snapshot.market_id}"
            quote = QuoteSnapshot(
                symbol=snapshot.market_id,
                bid_pence=_bps_to_pence(snapshot.bid),
                ask_pence=_bps_to_pence(snapshot.ask),
                timestamp_utc=snapshot.source_timestamp_utc or snapshot.timestamp_utc,
            )
            if research_fn is None:
                # Fail closed: a selected candidate must never silently vanish.
                reason = "RESEARCH_CALLBACK_MISSING"
                report.candidate_failures.append({"market_id": snapshot.market_id, "reason": reason})
                state.record_research(cycle_id, snapshot.market_id, run_id, "FAILED", reason)
                continue
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
            _record_shadow_decisions(config, cycle_id, snapshot, result, current)
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

    # Phase 5: watchdog, heartbeat, AI spend alerts (operations, no self-tuning)
    from tradingagents.survivor.ops.health import heartbeat_write, watchdog_record

    watchdog_record(state, "consecutive_cycle_failures", success=report.status in ("ACTIVE", "DRY_RUN"))
    for _failure in report.candidate_failures:
        watchdog_record(state, "consecutive_market_failures", success=False, halt_threshold=10)
    heartbeat_write(
        trial_id=trial_id, pid=os.getpid(),
        mode="DRY_RUN" if dry_run_enabled(config) else "PAPER",
        last_cycle_id=cycle_id, last_cycle_status=report.status,
        halt_state="HALTED" if halt_mod.is_halted() else "CLEAR",
        heartbeat_path=config.get("_heartbeat_path"),
    )
    # AI spend alerts at 50/75/90/100% of daily budget (100% blocks via preflight)
    from tradingagents.survivor.ops.budget import spend_alerts

    for alert in spend_alerts(usage, survivor_policy):
        report.warnings.append(alert)






