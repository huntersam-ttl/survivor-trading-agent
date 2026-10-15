"""Process-bound liveness heartbeat for the authenticated SURVIVOR HTTP service.

This heartbeat is independent of trading cycles. It is started only by the
service's blocking ``serve`` lifecycle, refreshed by a thread in that process,
and removed on orderly shutdown. The cycle heartbeat remains separate.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from tradingagents.survivor.autonomy.halt import is_halted
from tradingagents.survivor.paths import runtime_dir

SERVICE_HEARTBEAT_FILENAME = "service-heartbeat.json"
SERVICE_HEARTBEAT_INTERVAL_SECONDS = 5.0
SERVICE_HEARTBEAT_MAX_AGE_SECONDS = 30.0


def _parse_aware_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


class ServiceLivenessHeartbeat:
    """Atomically refresh liveness owned by one live service process."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        halt_path: str | os.PathLike[str] | None = None,
        interval_seconds: float = SERVICE_HEARTBEAT_INTERVAL_SECONDS,
        max_age_seconds: float = SERVICE_HEARTBEAT_MAX_AGE_SECONDS,
        pid: int | None = None,
        now: Callable[[], datetime] | None = None,
        instance_id: str | None = None,
    ):
        if interval_seconds <= 0 or max_age_seconds <= 0:
            raise ValueError("heartbeat intervals must be positive")
        self.path = Path(path) if path is not None else Path(runtime_dir()) / SERVICE_HEARTBEAT_FILENAME
        self.halt_path = Path(halt_path) if halt_path is not None else None
        self.interval_seconds = float(interval_seconds)
        self.max_age_seconds = float(max_age_seconds)
        self.pid = int(pid if pid is not None else os.getpid())
        self.instance_id = instance_id or uuid.uuid4().hex
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._started_at = self._now()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._write_lock = threading.Lock()

    def _halt_state(self) -> tuple[bool, str]:
        halted = is_halted(str(self.halt_path) if self.halt_path else None)
        return halted, "HALTED" if halted else "CLEAR"

    def _payload(self) -> dict:
        halted, halt_state = self._halt_state()
        return {
            "schema": 1,
            "heartbeat_type": "service_liveness",
            "service_pid": self.pid,
            "service_instance_id": self.instance_id,
            "started_at_utc": self._started_at.astimezone(timezone.utc).isoformat(),
            "timestamp_utc": self._now().astimezone(timezone.utc).isoformat(),
            "halted": halted,
            "halt_state": halt_state,
        }

    def _write(self) -> None:
        payload = self._payload()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock:
            fd, temp_path = tempfile.mkstemp(dir=self.path.parent, prefix=".service-heartbeat-")
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(payload, stream, sort_keys=True)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp_path, self.path)
            except BaseException:
                try:
                    os.close(fd)
                except OSError:
                    pass
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise

    def start(self) -> None:
        """Write an initial beat, then refresh until ``stop`` or process exit."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("service heartbeat is already running")
        self._stop.clear()
        self._write()

        def refresh() -> None:
            while not self._stop.wait(self.interval_seconds):
                try:
                    self._write()
                except OSError:
                    # An unavailable heartbeat must degrade status to false;
                    # keep the service available for diagnosis and retry later.
                    continue

        self._thread = threading.Thread(
            target=refresh, name="survivor-service-heartbeat", daemon=True,
        )
        self._thread.start()

    def read(self) -> dict:
        """Return redacted status validated against this active process instance."""
        halted, derived_halt_state = self._halt_state()
        absent = {
            "present": False,
            "consistent": False,
            "heartbeat_type": "service_liveness",
            "timestamp_utc": None,
            "halted": halted,
            "halt_state": derived_halt_state,
            "service_pid": self.pid,
        }
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return absent
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {**absent, "present": True}

        if not isinstance(payload, dict):
            return {**absent, "present": True}
        timestamp = _parse_aware_timestamp(payload.get("timestamp_utc"))
        started = _parse_aware_timestamp(payload.get("started_at_utc"))
        now = self._now().astimezone(timezone.utc)
        age = (now - timestamp).total_seconds() if timestamp else None
        started_not_future = started is not None and started <= now
        fresh = age is not None and 0 <= age <= self.max_age_seconds
        halt_consistent = (
            isinstance(payload.get("halted"), bool)
            and payload.get("halted") is halted
            and payload.get("halt_state") == derived_halt_state
        )
        identity_consistent = (
            self._thread is not None
            and self._thread.is_alive()
            and payload.get("schema") == 1
            and payload.get("heartbeat_type") == "service_liveness"
            and type(payload.get("service_pid")) is int
            and payload.get("service_pid") == self.pid == os.getpid()
            and payload.get("service_instance_id") == self.instance_id
        )
        return {
            "present": True,
            "consistent": bool(identity_consistent and started_not_future and fresh and halt_consistent),
            "heartbeat_type": "service_liveness",
            "timestamp_utc": payload.get("timestamp_utc") if timestamp else None,
            "started_at_utc": payload.get("started_at_utc") if started else None,
            "halted": halted,
            "halt_state": derived_halt_state,
            "service_pid": payload.get("service_pid") if type(payload.get("service_pid")) is int else None,
            "age_seconds": age,
        }

    def stop(self) -> None:
        """Stop refreshing and remove only this instance's own heartbeat."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds + 0.5))
            self._thread = None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("service_instance_id") == self.instance_id:
                self.path.unlink(missing_ok=True)
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass


def missing_service_heartbeat(*, halt_path: str | os.PathLike[str] | None = None) -> dict:
    """Fail-closed value for status handlers without an active service owner."""
    halted = is_halted(str(halt_path) if halt_path else None)
    return {
        "present": False,
        "consistent": False,
        "heartbeat_type": "service_liveness",
        "timestamp_utc": None,
        "halted": halted,
        "halt_state": "HALTED" if halted else "CLEAR",
        "service_pid": None,
    }
