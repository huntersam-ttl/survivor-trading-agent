"""Independent service liveness must never be confused with a trading cycle."""

import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.survivor.autonomy import halt
from tradingagents.survivor.ops.service_heartbeat import ServiceLivenessHeartbeat


@pytest.fixture
def halt_path(tmp_path, monkeypatch):
    path = tmp_path / "HALT"
    monkeypatch.setattr(halt, "HALT_PATH", str(path))
    return path


@pytest.mark.unit
def test_startup_publishes_process_bound_liveness_without_cycle_fields(tmp_path, halt_path):
    heartbeat = ServiceLivenessHeartbeat(
        tmp_path / "service-heartbeat.json", halt_path=halt_path,
    )
    try:
        heartbeat.start()
        status = heartbeat.read()
        assert status["present"] is True
        assert status["consistent"] is True
        assert status["heartbeat_type"] == "service_liveness"
        assert status["service_pid"] == os.getpid()
        assert status["halt_state"] == "CLEAR"
        assert "last_cycle_at" not in status
    finally:
        heartbeat.stop()


@pytest.mark.unit
def test_service_liveness_is_healthy_and_halt_is_reported_independently(tmp_path, halt_path):
    halt.set_halt(str(halt_path))
    heartbeat = ServiceLivenessHeartbeat(
        tmp_path / "service-heartbeat.json", halt_path=halt_path,
    )
    try:
        heartbeat.start()
        status = heartbeat.read()
        assert status["present"] is True
        assert status["consistent"] is True
        assert status["halted"] is True
        assert status["halt_state"] == "HALTED"
    finally:
        heartbeat.stop()


@pytest.mark.unit
def test_missing_and_malformed_heartbeat_fail_closed(tmp_path, halt_path):
    path = tmp_path / "service-heartbeat.json"
    heartbeat = ServiceLivenessHeartbeat(path, halt_path=halt_path)
    assert heartbeat.read()["present"] is False
    assert heartbeat.read()["consistent"] is False

    path.write_text("not-json")
    status = heartbeat.read()
    assert status["present"] is True
    assert status["consistent"] is False
    assert status["timestamp_utc"] is None


@pytest.mark.unit
@pytest.mark.parametrize("mutation", [
    lambda data, now: data.update(timestamp_utc=(now - timedelta(seconds=60)).isoformat()),
    lambda data, now: data.update(timestamp_utc="not-a-timestamp"),
    lambda data, now: data.update(timestamp_utc=(now + timedelta(seconds=1)).isoformat()),
    lambda data, now: data.update(started_at_utc=(now + timedelta(seconds=1)).isoformat()),
    lambda data, now: data.update(halted=True, halt_state="HALTED"),
    lambda data, now: data.update(service_pid=123),
    lambda data, now: data.update(service_instance_id="another-process"),
])
def test_stale_skewed_contradictory_or_foreign_heartbeat_fails_closed(
    tmp_path, halt_path, mutation,
):
    path = tmp_path / "service-heartbeat.json"
    heartbeat = ServiceLivenessHeartbeat(path, halt_path=halt_path)
    try:
        heartbeat.start()
        data = json.loads(path.read_text())
        now = datetime.now(timezone.utc)
        mutation(data, now)
        path.write_text(json.dumps(data))
        assert heartbeat.read()["consistent"] is False
    finally:
        heartbeat.stop()


@pytest.mark.unit
def test_refresh_is_independent_of_cycles_and_shutdown_removes_owned_file(tmp_path, halt_path):
    path = tmp_path / "service-heartbeat.json"
    heartbeat = ServiceLivenessHeartbeat(
        path, halt_path=halt_path, interval_seconds=0.02,
    )
    heartbeat.start()
    first = json.loads(path.read_text())["timestamp_utc"]
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if json.loads(path.read_text())["timestamp_utc"] != first:
            break
        time.sleep(0.01)
    else:
        pytest.fail("service heartbeat did not refresh independently")
    assert heartbeat.read()["consistent"] is True
    heartbeat.stop()
    assert not path.exists()
    assert heartbeat.read()["consistent"] is False


@pytest.mark.unit
def test_restart_uses_new_process_instance_identity(tmp_path, halt_path):
    path = tmp_path / "service-heartbeat.json"
    first = ServiceLivenessHeartbeat(path, halt_path=halt_path)
    first.start()
    first_id = json.loads(path.read_text())["service_instance_id"]
    first.stop()

    second = ServiceLivenessHeartbeat(path, halt_path=halt_path)
    try:
        second.start()
        second_id = json.loads(path.read_text())["service_instance_id"]
        assert second_id != first_id
        assert second.read()["consistent"] is True
        assert first.read()["consistent"] is False
    finally:
        second.stop()


@pytest.mark.unit
def test_service_writer_pid_must_match_current_process(tmp_path, halt_path):
    heartbeat = ServiceLivenessHeartbeat(
        tmp_path / "service-heartbeat.json", halt_path=halt_path, pid=os.getpid() + 1,
    )
    try:
        heartbeat.start()
        assert heartbeat.read()["consistent"] is False
    finally:
        heartbeat.stop()
