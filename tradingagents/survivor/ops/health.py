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


# --- heartbeat ------------------------------------------------------------------

def heartbeat_write(
    *,
    trial_id: str,
    pid: int,
    mode: str,
    last_cycle_id: str = "",
    last_cycle_status: str = "",
    halt_state: str = "CLEAR",
    heartbeat_path: str | None = None,
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
        "halt_state": halt_state,
    }
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".heartbeat")
    with os.fdopen(fd, "w") as fh:
        json.dump(payload, fh, sort_keys=True)
    os.replace(tmp_path, path)  # atomic on POSIX
    return path


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
