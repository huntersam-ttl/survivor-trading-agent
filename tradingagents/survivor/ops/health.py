"""Health preflight, watchdog counters, and heartbeat for long-run PAPER trials."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from tradingagents.survivor.autonomy.halt import is_halted, set_halt
from tradingagents.survivor.markets.adapter import FORBIDDEN_METHODS

SURVIVOR_DIR = os.path.join(os.path.expanduser("~"), ".tradingagents", "survivor")
HEARTBEAT_PATH = os.path.join(SURVIVOR_DIR, "heartbeat.json")
MIN_DISK_FREE_BYTES = 200 * 1024 * 1024  # 200 MB


@dataclass
class HealthReport:
    ok: bool
    checks: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)

    def render(self) -> str:
        lines = ["", "SURVIVOR HEALTH", ""]
        for name, value in self.checks.items():
            lines.append(f"{name}: {value}")
        if self.failures:
            lines.append("")
            for f in self.failures:
                lines.append(f"FAIL: {f}")
        lines.append("")
        lines.append(f"State: {'HEALTHY' if self.ok else 'UNHEALTHY'}")
        return chr(10).join(lines)


def _db_ok(path: str) -> bool:
    import sqlite3

    if not os.path.exists(path):
        return True  # absent DBs are created on demand
    conn = sqlite3.connect(path)
    row = conn.execute("PRAGMA integrity_check").fetchone()
    conn.close()
    return row[0] == "ok"


def preflight_check(
    *,
    survivor_dir: str | None = None,
    paper_ledger_path: str | None = None,
    adapter=None,
    cycle_interval_seconds: int = 900,
    min_disk_free_bytes: int = MIN_DISK_FREE_BYTES,
    require_halt_clear: bool = True,
) -> HealthReport:
    """15-point pre-flight verification before starting a long-run trial."""
    base = survivor_dir or SURVIVOR_DIR
    checks: dict = {}
    failures: list = []

    def record(name: str, ok: bool, detail: str = "OK") -> None:
        checks[name] = detail if ok else f"FAIL: {detail}"
        if not ok:
            failures.append(f"{name}: {detail}")

    # 1-3. paper-only policy / live trading / broker absence (structural)
    from tradingagents.survivor.policy import SurvivorPolicy

    policy = SurvivorPolicy()
    record("policy_paper_only", policy.real_trading_enabled is False and policy.mode.value == "PAPER_ONLY")
    record("live_trading_absent", True, "no code path exists")
    record("broker_absent", True, "no broker backend configured")
    # 4. wallet absence
    record("wallet_absent", True, "no wallet code")
    # 5. paper ledger integrity
    ledger_path = paper_ledger_path or os.path.join(base, "paper.db")
    try:
        from tradingagents.survivor.execution.ledger import PaperLedger

        PaperLedger(db_path=ledger_path).verify_chain()
        record("paper_ledger_chain", True)
    except Exception as exc:  # noqa: BLE001
        record("paper_ledger_chain", False, str(exc)[:120])
    # 6-8. databases reachable/integrity
    for name, path in (
        ("evaluation_db", os.path.join(base, "evaluation.db")),
        ("usage_db", os.path.join(base, "usage.db")),
        ("runtime_db", os.path.join(base, "runtime.db")),
    ):
        record(name, _db_ok(path))
    # 9. market adapter read-only
    if adapter is not None:
        bad = [m for m in FORBIDDEN_METHODS if hasattr(adapter, m)]
        record("market_adapter_readonly", not bad, f"forbidden methods: {bad}" if bad else "OK")
    else:
        record("market_adapter_readonly", True, "no adapter instance (validated at cycle start)")
    # 10. AI budgets valid
    record("ai_budgets_valid", policy.global_daily_pence > 0 and policy.global_monthly_pence > 0)
    # 11. FX present
    record("fx_present", policy.usd_gbp_rate is not None and policy.usd_gbp_rate > 0)
    # 12. disk space
    free = shutil.disk_usage(base if os.path.isdir(base) else os.path.expanduser("~")).free
    record("disk_space", free >= min_disk_free_bytes, f"{free} bytes free")
    # 13. interval >= hard minimum
    record("cycle_interval", cycle_interval_seconds >= 300, f"{cycle_interval_seconds}s")
    # 14. no stale runtime lock
    lock_path = os.path.join(base, "cycle.lock")
    stale = os.path.exists(lock_path) and (time.time() - os.path.getmtime(lock_path) > 3600)
    record("runtime_lock", not stale, "stale lock" if stale else "OK")
    # 15. HALT clear
    halted = is_halted(os.path.join(base, "HALT"))
    record("halt_state", (not halted) or (not require_halt_clear), "HALTED" if halted else "CLEAR")

    return HealthReport(ok=not failures, checks=checks, failures=failures)


# --- watchdog ------------------------------------------------------------------

def watchdog_record(runtime_state, key: str, *, success: bool, halt_threshold: int = 5) -> int:
    """Update a consecutive-failure counter. Halts the runtime when a counter
    reaches halt_threshold. Returns the new counter value."""
    counters = watchdog_status(runtime_state)
    value = counters.get(key, 0)
    value = 0 if success else value + 1
    runtime_state.set_watchdog(key, value)
    if value >= halt_threshold:
        set_halt()
    return value


def watchdog_status(runtime_state) -> dict:
    return runtime_state.get_watchdog_all()


# --- bounded market recovery --------------------------------------------------

def market_data_probe(adapter, *, max_age_seconds: float = 300.0) -> dict:
    """Run one bounded, read-only market-data probe.

    The probe validates provider response shape through the existing adapter's
    normalizer. It does not rank, research, or execute anything. When a
    provider does not supply a source timestamp, the observation time is used
    explicitly as the transport timestamp rather than inventing provider age.
    """
    started = time.monotonic()
    observed_at = datetime.now(timezone.utc)
    provider = str(getattr(adapter, "provider", "unknown"))
    try:
        raw = adapter.list_markets(limit=100)
        rows = raw if isinstance(raw, list) else []
        normalized = []
        for item in rows:
            snapshot = adapter.normalize(item) if hasattr(adapter, "normalize") else None
            if snapshot is not None:
                normalized.append(snapshot)
        timestamps = [
            getattr(item, "source_timestamp_utc", None)
            or getattr(item, "timestamp_utc", None)
            for item in normalized
        ]
        timestamps = [value for value in timestamps if value]
        data_timestamp = max(timestamps) if timestamps else observed_at.isoformat()
        try:
            data_age = max(0.0, (observed_at - datetime.fromisoformat(
                data_timestamp.replace("Z", "+00:00"))).total_seconds())
        except (AttributeError, TypeError, ValueError):
            data_age = 0.0 if timestamps == [] else None
        schema_valid = bool(rows) and len(normalized) == len(rows)
        fresh = schema_valid and data_age is not None and data_age <= max_age_seconds
        return {
            "timestamp": observed_at.isoformat(),
            "provider": provider,
            "response_status": "OK",
            "data_timestamp": data_timestamp,
            "candidate_count": len(normalized),
            "schema_valid": schema_valid,
            "fresh": fresh,
            "latency_ms": round((time.monotonic() - started) * 1000, 2),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - probe reports provider failure
        return {
            "timestamp": observed_at.isoformat(),
            "provider": provider,
            "response_status": "ERROR",
            "data_timestamp": None,
            "candidate_count": 0,
            "schema_valid": False,
            "fresh": False,
            "latency_ms": round((time.monotonic() - started) * 1000, 2),
            "error": str(exc)[:200],
        }


def record_market_probe(runtime_state, probe: dict, *, recovery_successes: int = 3) -> dict:
    """Record current probe state without erasing historical failure evidence."""
    counters = watchdog_status(runtime_state)
    active = int(counters.get("consecutive_market_failures", 0))
    historical = max(
        int(counters.get("historical_consecutive_market_failures", 0)), active,
    )
    streak = int(counters.get("market_probe_success_streak", 0))
    if probe.get("fresh") and probe.get("schema_valid"):
        streak += 1
        if streak >= recovery_successes:
            active = 0
    else:
        streak = 0
        active += 1
        historical = max(historical, active)
    runtime_state.set_watchdog("consecutive_market_failures", active)
    runtime_state.set_watchdog("historical_consecutive_market_failures", historical)
    runtime_state.set_watchdog("market_probe_success_streak", streak)
    return {
        "active_market_failure_count": active,
        "historical_consecutive_failures": historical,
        "successful_probe_streak": streak,
    }


def recovery_preflight(
    *,
    runtime_state,
    adapter,
    paper_status: dict,
    heartbeat_path: str | None = None,
    probe_count: int = 3,
    max_age_seconds: float = 300.0,
) -> dict:
    """Read-only recovery gate over fresh market probes and paper invariants."""
    probes = [market_data_probe(adapter, max_age_seconds=max_age_seconds)
              for _ in range(max(1, min(probe_count, 3)))]
    counters = watchdog_status(runtime_state)
    halted = is_halted()
    heartbeat = heartbeat_read(heartbeat_path)
    market_healthy = all(p["fresh"] and p["schema_valid"] for p in probes)
    paper_mode = str(paper_status.get("mode", "")) in {"PAPER", "PAPER ONLY", "PAPER_ONLY"}
    positions = int(paper_status.get("open_positions", 0) or 0)
    risk_breach = str(paper_status.get("trading_state", "ACTIVE")) not in {"ACTIVE", "PAPER"}
    reasons = []
    if not paper_mode: reasons.append("paper_mode_required")
    if not market_healthy: reasons.append("current_market_health_not_healthy")
    if positions: reasons.append("open_positions_present")
    if risk_breach: reasons.append("risk_breach_or_paper_halt")
    if not heartbeat.get("present") or not heartbeat.get("consistent"):
        reasons.append("heartbeat_inconsistent")
    return {
        "safe_to_recover": not reasons,
        "reasons": reasons,
        "paper_mode": paper_mode,
        "halt_file_present": halted,
        "current_market_health": "HEALTHY" if market_healthy else "DEGRADED",
        "active_market_failure_count": int(counters.get("consecutive_market_failures", 0)),
        "historical_consecutive_failures": int(counters.get("historical_consecutive_market_failures", counters.get("consecutive_market_failures", 0))),
        "provider_health": {str(getattr(adapter, "provider", "unknown")): "READY" if market_healthy else "DEGRADED"},
        "schema_health": "HEALTHY" if all(p["schema_valid"] for p in probes) else "DEGRADED",
        "positions": positions,
        "risk_breach": risk_breach,
        "live_broker_available": False,
        "heartbeat_consistent": bool(heartbeat.get("present") and heartbeat.get("consistent")),
        "probes": probes,
    }


def reconcile_recovery_state(
    *,
    runtime_state,
    adapter,
    heartbeat_path: str | None = None,
    halt_path: str | None = None,
    probe_count: int = 3,
    max_age_seconds: float = 300.0,
) -> dict:
    """Reconcile bounded recovery evidence without resuming autonomy.

    This is intentionally separate from the read-only preflight gate. It may
    reset only the active market-failure streak after bounded fresh probes,
    preserves the historical peak, and marks a dead supervisor heartbeat stale
    while the filesystem HALT remains authoritative.
    """
    probes = [market_data_probe(adapter, max_age_seconds=max_age_seconds)
              for _ in range(max(1, min(probe_count, 3)))]
    counter = {"active_market_failure_count": 0,
               "historical_consecutive_failures": 0,
               "successful_probe_streak": 0}
    for probe in probes:
        counter = record_market_probe(runtime_state, probe)

    heartbeat = heartbeat_read(heartbeat_path)
    if is_halted(halt_path):
        heartbeat_write(
            trial_id=str(heartbeat.get("trial_id") or "no-trial"),
            pid=0,
            mode=str(heartbeat.get("mode") or "PAPER"),
            last_cycle_id=str(heartbeat.get("last_cycle_id") or ""),
            last_cycle_status="HALTED",
            halt_state="HALTED",
            halt_path=halt_path,
            heartbeat_path=heartbeat_path,
            supervisor_state="STALE",
        )
    return {
        "halt_file_present": is_halted(halt_path),
        "heartbeat": heartbeat_read(heartbeat_path),
        "watchdog": watchdog_status(runtime_state),
        "counter_reconciliation": counter,
        "probes": probes,
    }


def reconcile_resumed_heartbeat(
    *,
    heartbeat_path: str | None = None,
    halt_path: str | None = None,
) -> dict:
    """Mark a stopped supervisor ready after the authoritative HALT is removed.

    Resuming the kill switch does not start autonomy.  It only records the
    durable, non-halted state that a stopped supervisor is expected to have.
    The caller must remove HALT first; ``heartbeat_write`` still enforces the
    filesystem kill switch if that transition is attempted too early.
    """
    if is_halted(halt_path):
        raise RuntimeError("cannot reconcile resumed heartbeat while HALT is present")

    previous = heartbeat_read(heartbeat_path)
    heartbeat_write(
        trial_id=str(previous.get("trial_id") or "no-trial"),
        pid=0,
        mode=str(previous.get("mode") or "PAPER"),
        last_cycle_id=str(previous.get("last_cycle_id") or ""),
        last_cycle_status="RECOVERY_READY",
        halt_state="CLEAR",
        halt_path=halt_path,
        heartbeat_path=heartbeat_path,
        supervisor_state="STALE",
    )
    return heartbeat_read(heartbeat_path)


# --- heartbeat ------------------------------------------------------------------

def heartbeat_write(
    *,
    trial_id: str,
    pid: int,
    mode: str,
    last_cycle_id: str = "",
    last_cycle_status: str = "",
    halt_state: str = "CLEAR",
    halt_path: str | None = None,
    heartbeat_path: str | None = None,
    supervisor_state: str | None = None,
) -> str:
    """Write heartbeat.json atomically (tmp file + rename). No secrets."""
    path = heartbeat_path or HEARTBEAT_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "trial_id": trial_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": pid,
        "mode": mode,
        "last_cycle_id": last_cycle_id,
        "last_cycle_status": last_cycle_status,
        # The filesystem kill switch is authoritative; callers cannot write
        # CLEAR while HALT exists.
        "halt_state": "HALTED" if is_halted(halt_path) else halt_state,
        "supervisor_state": supervisor_state or ("STALE" if pid <= 0 else "RUNNING"),
    }
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".heartbeat")
    with os.fdopen(fd, "w") as fh:
        json.dump(payload, fh, sort_keys=True)
    os.replace(tmp_path, path)  # atomic on POSIX
    return path


def heartbeat_read(heartbeat_path: str | None = None) -> dict:
    """Read heartbeat with derived HALT state; never trusts stale CLEAR."""
    path = heartbeat_path or HEARTBEAT_PATH
    if not os.path.exists(path):
        return {"present": False, "consistent": not is_halted(), "halt_state": "HALTED" if is_halted() else "CLEAR"}
    try:
        with open(path) as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return {"present": True, "consistent": False, "halt_state": "HALTED" if is_halted() else "UNKNOWN"}
    derived = "HALTED" if is_halted() else str(payload.get("halt_state", "CLEAR"))
    return {
        **payload,
        "halt_state": derived,
        "consistent": str(payload.get("halt_state", "")) == derived,
        "present": True,
    }


def heartbeat_age_seconds(heartbeat_path: str | None = None) -> float | None:
    path = heartbeat_path or HEARTBEAT_PATH
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        data = json.load(fh)
    ts = datetime.fromisoformat(data["timestamp_utc"])
    return (datetime.now(timezone.utc) - ts).total_seconds()


def heartbeat_is_stale(heartbeat_path: str | None = None, max_age_sec: float = 3 * 3600) -> bool:
    age = heartbeat_age_seconds(heartbeat_path)
    return age is None or age > max_age_sec
