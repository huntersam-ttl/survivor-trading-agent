"""Trial configuration: an immutable snapshot of a long-run PAPER experiment.

Once started, a material strategy/risk change produces a DIFFERENT config hash
and therefore requires a NEW trial - an active experiment is never mutated.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

TRIALS_DIR = os.path.join(os.path.expanduser("~"), ".tradingagents", "survivor", "trials")


@dataclass(frozen=True)
class TrialConfig:
    trial_id: str
    strategy_version: str
    config_hash: str
    start_time_utc: str
    mode: str                       # SCAN_ONLY | DRY_RUN | PAPER
    market_provider: str
    cycle_interval_seconds: int
    max_research_per_cycle: int
    ai_daily_budget_pence: int
    ai_monthly_budget_pence: int
    max_single_position_pence: int
    max_total_exposure_pence: int
    daily_loss_limit_pence: int
    max_drawdown_bps: int
    starting_equity_pence: int
    planned_duration_days: int = 30
    resolved_sample_target: int = 300
    stop_drawdown_bps: int = 2000   # early-stop paper drawdown (configurable, never live)
    status: str = "RUNNING"         # RUNNING | STOPPED | COMPLETED | HALTED
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    @staticmethod
    def from_json(raw: str) -> TrialConfig:
        data = json.loads(raw)
        return TrialConfig(**data)


def compute_trial_config_hash(config: dict[str, Any]) -> str:
    """Material-change detector: any strategy/risk/budget change changes identity."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class TrialManager:
    """Persists trial configs under <trials_dir>/<id>.json."""

    def __init__(self, trials_dir: str | None = None):
        self.trials_dir = trials_dir or TRIALS_DIR

    def _path(self, trial_id: str) -> str:
        return os.path.join(self.trials_dir, f"{trial_id}.json")

    def start_trial(
        self,
        *,
        mode: str = "DRY_RUN",
        market_provider: str = "polymarket",
        cycle_interval_seconds: int = 900,
        max_research_per_cycle: int = 3,
        ai_daily_budget_pence: int = 170,
        ai_monthly_budget_pence: int = 5000,
        max_single_position_pence: int = 100,
        max_total_exposure_pence: int = 500,
        daily_loss_limit_pence: int = 100,
        max_drawdown_bps: int = 1500,
        starting_equity_pence: int = 2000,
        strategy_version: str = "survivor-v1.0",
        planned_duration_days: int = 30,
        resolved_sample_target: int = 300,
        config: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> TrialConfig:
        if mode not in ("SCAN_ONLY", "DRY_RUN", "PAPER"):
            raise ValueError(f"invalid trial mode: {mode}")
        if mode == "PAPER" and os.environ.get(
            "TRADINGAGENTS_SURVIVOR_PAPER_ENABLED", ""
        ).lower() not in ("true", "1", "yes", "on"):
            raise ValueError(
                "PAPER trial mode requires TRADINGAGENTS_SURVIVOR_PAPER_ENABLED=true"
            )
        current = now or datetime.now(timezone.utc)
        trial = TrialConfig(
            trial_id=uuid.uuid4().hex[:12],
            strategy_version=strategy_version,
            config_hash=compute_trial_config_hash(config or {}),
            start_time_utc=current.isoformat(),
            mode=mode,
            market_provider=market_provider,
            cycle_interval_seconds=cycle_interval_seconds,
            max_research_per_cycle=max_research_per_cycle,
            ai_daily_budget_pence=ai_daily_budget_pence,
            ai_monthly_budget_pence=ai_monthly_budget_pence,
            max_single_position_pence=max_single_position_pence,
            max_total_exposure_pence=max_total_exposure_pence,
            daily_loss_limit_pence=daily_loss_limit_pence,
            max_drawdown_bps=max_drawdown_bps,
            starting_equity_pence=starting_equity_pence,
            planned_duration_days=planned_duration_days,
            resolved_sample_target=resolved_sample_target,
        )
        os.makedirs(self.trials_dir, exist_ok=True)
        with open(self._path(trial.trial_id), "w") as fh:
            fh.write(trial.to_json())
        return trial

    def load_trial(self, trial_id: str) -> TrialConfig | None:
        path = self._path(trial_id)
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            return TrialConfig.from_json(fh.read())

    def active_trial(self) -> TrialConfig | None:
        """Most recent RUNNING trial, if any."""
        if not os.path.isdir(self.trials_dir):
            return None
        running = []
        for name in os.listdir(self.trials_dir):
            if not name.endswith(".json"):
                continue
            try:
                trial = self.load_trial(name[:-5])
            except (ValueError, TypeError):
                continue
            if trial and trial.status == "RUNNING":
                running.append(trial)
        if not running:
            return None
        running.sort(key=lambda t: t.start_time_utc, reverse=True)
        return running[0]

    def all_trials(self) -> list[TrialConfig]:
        if not os.path.isdir(self.trials_dir):
            return []
        trials = []
        for name in sorted(os.listdir(self.trials_dir)):
            if name.endswith(".json"):
                try:
                    trial = self.load_trial(name[:-5])
                except (ValueError, TypeError):
                    continue
                if trial:
                    trials.append(trial)
        return trials

    def stop_trial(self, trial_id: str, status: str = "STOPPED") -> TrialConfig | None:
        trial = self.load_trial(trial_id)
        if trial is None:
            return None
        stopped = TrialConfig(**{**asdict(trial), "status": status})
        with open(self._path(trial_id), "w") as fh:
            fh.write(stopped.to_json())
        return stopped

    def config_matches(self, trial: TrialConfig, config: dict[str, Any] | None) -> bool:
        """Material configuration change requires a NEW trial."""
        return trial.config_hash == compute_trial_config_hash(config or {})


def trial_completion(
    trial: TrialConfig,
    *,
    resolved_predictions: int,
    net_pnl_after_ai_cost_pence: int,
    brier_improvement: float,
    max_drawdown_bps: int,
    profit_concentrated: bool,
    out_of_sample_net_pnl_pence: int,
    now: datetime | None = None,
    min_sample_for_promising: int = 100,
) -> dict:
    """Deterministic trial completion rules: by duration OR resolved sample target.

    Completion does NOT mean success. Result is one of the Phase 4 gate states;
    READY_FOR_LIVE does not exist.
    """
    current = now or datetime.now(timezone.utc)
    started = datetime.fromisoformat(trial.start_time_utc)
    elapsed_days = (current - started).total_seconds() / 86400
    by_duration = elapsed_days >= trial.planned_duration_days
    by_sample = resolved_predictions >= trial.resolved_sample_target
    completed = by_duration or by_sample

    if resolved_predictions < min_sample_for_promising:
        result = "INSUFFICIENT_DATA"
    elif (net_pnl_after_ai_cost_pence > 0 and brier_improvement > 0
          and max_drawdown_bps <= trial.max_drawdown_bps
          and not profit_concentrated and out_of_sample_net_pnl_pence > 0):
        result = "PROMISING_BUT_UNPROVEN"
    elif out_of_sample_net_pnl_pence <= 0:
        result = "FAIL"
    else:
        result = "CONTINUE_PAPER"
    return {
        "completed": completed,
        "by_duration": by_duration,
        "by_sample": by_sample,
        "elapsed_days": round(elapsed_days, 3),
        "resolved_predictions": resolved_predictions,
        "result": result,
    }

