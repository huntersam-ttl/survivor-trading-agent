"""Phase 5 operations: long-run PAPER trial management and monitoring."""

from tradingagents.survivor.ops.attribution import (
    cost_efficiency_metrics,
    equity_curve,
    failure_analytics,
    model_attribution,
)
from tradingagents.survivor.ops.backup import backup_survivor, verify_backup
from tradingagents.survivor.ops.health import (
    HealthReport,
    heartbeat_write,
    preflight_check,
    watchdog_record,
    watchdog_status,
)
from tradingagents.survivor.ops.logging_utils import JsonRotateLogger, redact_secrets
from tradingagents.survivor.ops.recovery import recover_interrupted_cycles
from tradingagents.survivor.ops.selfcheck import security_selfcheck
from tradingagents.survivor.ops.snapshots import daily_snapshot, persist_snapshot, weekly_snapshot
from tradingagents.survivor.ops.trial import (
    TrialConfig,
    TrialManager,
    trial_completion,
)

__all__ = [
    "HealthReport", "JsonRotateLogger", "TrialConfig", "TrialManager",
    "backup_survivor", "cost_efficiency_metrics", "daily_snapshot",
    "equity_curve", "failure_analytics", "heartbeat_write", "model_attribution",
    "persist_snapshot", "preflight_check", "recover_interrupted_cycles",
    "redact_secrets", "security_selfcheck", "trial_completion", "verify_backup",
    "watchdog_record", "watchdog_status", "weekly_snapshot",
]
