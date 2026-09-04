"""Read-only external interfaces.

EventIntelligenceProvider ("God's Eye"): supplies external event evidence as
UNTRUSTED wrapped text. It structurally cannot create trades, change risk or
budget, touch PaperBroker, change strategy, or approve experiments.

JarvisInterface: operations/reporting queries plus experiment REQUESTS. Jarvis
cannot approve experiments, increase risk/budget, enable live trading, remove
HALT, or move money — those methods do not exist on this class.
"""

from __future__ import annotations

from tradingagents.survivor.autonomy.injection import build_evidence

__all__ = ["EventIntelligenceProvider", "JarvisInterface"]


class EventIntelligenceProvider:
    """Read-only event intelligence. Output is ALWAYS untrusted wrapped
    evidence; the class exposes no mutating capability by construction."""

    def get_event_evidence(self, topic: str, as_of_utc: str | None = None) -> str:
        """Return the provider's event text wrapped as untrusted data. Concrete
        sources implement `fetch_raw_events` (read-only) — never executed
        against any trading surface."""
        raw = self.fetch_raw_events(topic, as_of_utc)
        return build_evidence(f"events:{topic}", raw).render()

    def fetch_raw_events(self, topic: str, as_of_utc: str | None = None) -> str:
        """Override in a concrete provider. Must be read-only. Default returns
        an explicit no-data marker."""
        return "(no event data available)"

    def event_velocity(self, topic: str) -> int:
        """Number of related events observed (metadata only)."""
        return 0


class JarvisInterface:
    """Operations/reporting surface for an external operator assistant.
    Every method is read-only except request_experiment, which can only ever
    create a PROPOSED experiment (schema + allowlist validated)."""

    def __init__(self, experiment_store, usage_ledger=None, paper_broker_reader=None):
        self._experiments = experiment_store
        self._usage = usage_ledger
        self._broker_reader = paper_broker_reader   # optional read-only facade

    # -- read-only queries ---------------------------------------------------

    def health_summary(self) -> dict:
        from tradingagents.survivor.autonomy import halt as halt_mod
        return {
            "halt_state": "HALTED" if halt_mod.is_halted() else "CLEAR",
            "mode": "PAPER_ONLY",
        }

    def ai_spend(self) -> dict:
        if self._usage is None:
            return {"daily_pence": None, "monthly_pence": None}
        return {
            "daily_pence": self._usage.get_daily_spend_pence(None),
            "monthly_pence": self._usage.get_monthly_spend_pence(None),
        }

    def open_paper_positions(self) -> list[dict]:
        if self._broker_reader is None:
            return []
        return list(self._broker_reader.portfolio.positions.keys())

    def experiment_proposals(self) -> list[dict]:
        return [e.to_dict() for e in self._experiments.experiments()]

    def strategy_history(self) -> list[dict]:
        return self._experiments.history()

    # -- the ONLY write-like capability ----------------------------------------

    def request_experiment(self, proposal: dict) -> str:
        """Request (NOT approve) an experiment. Validated against the same
        immutable schema/allowlist; result is PROPOSED and inert until a human
        approves via ExperimentStore.approve_experiment."""
        from tradingagents.survivor.learning.experiments import (
            validate_proposal_schema,
        )

        experiment = validate_proposal_schema(proposal)
        self._experiments.save_experiment(experiment)
        return experiment.experiment_id
