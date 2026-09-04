"""Normalized learning dataset built ONLY from persisted decision-time data.

Every LearningRecord joins an evaluated prediction (optionally resolved) with
its paper-trade outcome and per-role AI cost attribution. Pre-resolution
features are exactly the fields recorded at decision time — no future data can
enter them by construction (resolution fields live outside `preresolution`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tradingagents.survivor.evaluation.store import EvaluationStore
from tradingagents.survivor.evaluation.types import STRATEGY_VERSION, PredictionRecord


@dataclass(frozen=True)
class LearningRecord:
    """One resolved decision, normalized for learning."""

    # identity
    trial_id: str
    strategy_version: str
    cycle_id: str
    run_id: str
    market_id: str
    category: str
    timestamp_utc: str

    # decision-time (pre-resolution) features
    market_probability: float          # decision-time market-implied probability
    survivor_probability: float        # SURVIVOR estimate at decision time
    predicted_edge_bps: int            # gross edge recorded at decision time
    conservative_edge_bps: int         # net edge recorded at decision time
    preresolution: dict = field(default_factory=dict)
    # preresolution keys: analyst_outputs (refs), bull_conclusion, bear_conclusion,
    # research_manager_decision, trader_result, portfolio_manager_result,
    # role_usage: {role: {provider, model, calls, cost_pence, latency_ms}},
    # spread_bps, liquidity_usd_cents, research_duration_ms

    # resolution (never part of pre-resolution features)
    outcome: float | None = None
    resolution_timestamp_utc: str | None = None
    executed: bool = False
    execution_price_pence: int = 0
    fees_pence: int = 0
    slippage_pence: int = 0
    pnl_pence: int = 0                 # gross realized P/L
    ai_cost_pence: int = 0
    decision_latency_ms: int = 0
    realized_return_bps: int = 0

    @property
    def resolved(self) -> bool:
        return self.outcome is not None

    @property
    def survivor_error(self) -> float | None:
        return abs(self.survivor_probability - self.outcome) if self.resolved else None

    @property
    def market_error(self) -> float | None:
        return abs(self.market_probability - self.outcome) if self.resolved else None

    @property
    def brier_contribution(self) -> float | None:
        return self.survivor_error ** 2 if self.resolved else None

    @property
    def market_baseline_error(self) -> float | None:
        return self.market_error


def build_learning_records(
    store: EvaluationStore,
    usage_ledger=None,
    strategy_version: str | None = None,
    trial_id: str = "unknown-trial",
) -> list[LearningRecord]:
    """Join resolved predictions with trade outcomes and per-role usage.

    Deterministic; uses only data persisted at/after decision time. The
    usage-ledger role breakdown is keyed by run_id and was recorded during the
    decision, so it is safe pre-resolution context.
    """
    version = strategy_version or STRATEGY_VERSION
    predictions = store.predictions(strategy_version=version)
    trades_by_proposal = {t.proposal_id: t for t in store.trades(strategy_version=version)}
    usage_by_run = {}
    if usage_ledger is not None and hasattr(usage_ledger, "get_run_usage_summary"):
        seen_runs = {p.run_id for p in predictions}
        for run_id in seen_runs:
            summary = usage_ledger.get_run_usage_summary(run_id)
            role_usage = {}
            for row in summary.get("breakdown", []):
                role_usage[row["role"]] = {
                    "provider": row["provider"],
                    "model": row["model"],
                    "calls": row["calls"],
                    "cost_pence": row["cost_pence"],
                }
            usage_by_run[run_id] = role_usage

    records: list[LearningRecord] = []
    for pred in predictions:
        if pred.outcome is None:
            continue  # learning uses RESOLVED decisions only
        trade = trades_by_proposal.get(f"{pred.run_id}_prop") \
            or trades_by_proposal.get(pred.proposal_id)
        role_usage = usage_by_run.get(pred.run_id, {})
        records.append(_record_from(pred, trade, role_usage, trial_id))
    return records


def _record_from(
    pred: PredictionRecord,
    trade,
    role_usage: dict,
    trial_id: str,
) -> LearningRecord:
    preresolution = {
        "analyst_outputs": [],            # analyst report refs are not persisted yet
        "bull_conclusion": "",
        "bear_conclusion": "",
        "research_manager_decision": "",
        "trader_result": "",
        "portfolio_manager_result": "",
        "role_usage": role_usage,
        "research_duration_ms": 0,
        "spread_bps": None,
        "liquidity_usd_cents": None,
    }
    return LearningRecord(
        trial_id=trial_id,
        strategy_version=pred.strategy_version,
        cycle_id=pred.cycle_id,
        run_id=pred.run_id,
        market_id=pred.market_id,
        category=pred.category,
        timestamp_utc=pred.timestamp_utc,
        market_probability=pred.market_probability,
        survivor_probability=pred.predicted_probability,
        predicted_edge_bps=pred.gross_edge_bps,
        conservative_edge_bps=pred.net_edge_bps,
        preresolution=preresolution,
        outcome=pred.outcome,
        resolution_timestamp_utc=pred.resolution_timestamp_utc,
        executed=trade is not None,
        execution_price_pence=trade.entry_price_pence if trade else 0,
        fees_pence=trade.fees_pence if trade else 0,
        slippage_pence=trade.slippage_pence if trade else 0,
        pnl_pence=trade.gross_pnl_pence if trade else 0,
        ai_cost_pence=pred.ai_cost_pence,
        decision_latency_ms=trade.decision_latency_ms if trade else 0,
        realized_return_bps=trade.realized_return_bps if trade else 0,
    )
