"""The operator HALT path must not leak into hermetic cycle tests."""

from tests.survivor.market_helpers import FakeMarketAdapter
from tradingagents.survivor.autonomy import halt
from tradingagents.survivor.autonomy.cycle import run_survivor_cycle
from tradingagents.survivor.autonomy.state import RuntimeState


def test_injected_halt_path_ignores_real_home_state(tmp_path, monkeypatch):
    monkeypatch.setenv("SURVIVOR_DRY_RUN", "false")
    real_halt = tmp_path / "operator-home" / "HALT"
    real_halt.parent.mkdir()
    real_halt.write_text("operator halt")
    monkeypatch.setattr(halt, "HALT_PATH", str(real_halt))

    result = run_survivor_cycle(
        {"survivor_autonomy_enabled": True, "_evaluation_db_path": str(tmp_path / "evaluation.db"),
         "_heartbeat_path": str(tmp_path / "heartbeat.json")},
        adapter=FakeMarketAdapter([]),
        runtime_state=RuntimeState(db_path=str(tmp_path / "runtime.db")),
        lock_path=str(tmp_path / "cycle.lock"),
        halt_path=str(tmp_path / "isolated" / "HALT"),
    )

    assert result.status == "ACTIVE"


def test_injected_halt_file_still_blocks_cycle(tmp_path):
    halt_path = tmp_path / "HALT"
    halt_path.write_text("test halt")
    result = run_survivor_cycle(
        {"survivor_autonomy_enabled": True, "_evaluation_db_path": str(tmp_path / "evaluation.db")},
        runtime_state=RuntimeState(db_path=str(tmp_path / "runtime.db")),
        halt_path=str(halt_path),
    )
    assert result.status == "EXTERNAL_HALT"


def test_default_halt_function_still_uses_default_path(monkeypatch, tmp_path):
    default_path = tmp_path / "default" / "HALT"
    monkeypatch.setattr(halt, "HALT_PATH", str(default_path))
    assert halt.is_halted() is False
    default_path.parent.mkdir()
    default_path.write_text("halt")
    assert halt.is_halted() is True
