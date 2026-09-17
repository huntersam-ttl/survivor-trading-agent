from __future__ import annotations

import sqlite3

from tests.survivor.market_helpers import make_snapshot
from tradingagents.survivor.history import HistoryStore
from tradingagents.survivor.markets.filters import ScanLimits
from tradingagents.survivor.markets.scanner import MarketScanner
from tradingagents.survivor.markets.types import ScanRejection


def test_history_captures_normalized_filter_decision_and_is_idempotent(tmp_path):
    snapshot = make_snapshot("real-market-1", bid=1000, ask=2000)
    result = MarketScanner(_Adapter(snapshot), limits=ScanLimits(max_spread_bps=500)).scan()
    assert result.rejected == [("real-market-1", ScanRejection.SPREAD_TOO_WIDE)]
    store = HistoryStore(tmp_path / "history.db")
    kwargs = {"thresholds": {"max_spread_bps": 500}, "strategy_version": "survivor-v1.0", "config_hash": "cfg"}
    assert store.record_scan("cycle-1", result.observed, **kwargs) == 1
    assert store.record_scan("cycle-1", result.observed, **kwargs) == 1
    assert store.stats()["snapshot_count"] == 1
    row = sqlite3.connect(tmp_path / "history.db").execute(
        "select filter_result,rejection_reason,bid_bps,ask_bps,thresholds_json from market_snapshots"
    ).fetchone()
    assert row[:4] == ("REJECTED", "SPREAD_TOO_WIDE", 1000, 2000)
    assert "max_spread_bps" in row[4]


def test_history_is_separate_from_live_paths(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    assert store.db_path.endswith("history.db")
    assert store.stats()["snapshot_count"] == 0


def test_outcome_reconciliation_is_append_only(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    assert store.record_outcome("mkt-1", "2026-09-17T00:00:00+00:00", 1.0, 1.0, "polymarket")
    assert not store.record_outcome("mkt-1", "2026-09-18T00:00:00+00:00", 0.0, 0.0, "polymarket")
    assert store.stats()["resolved_markets"] == 1


class _Adapter:
    provider = "test"
    market_type = "prediction_binary"

    def __init__(self, snapshot):
        self.snapshot = snapshot

    def list_markets(self, limit=100):
        return [{"market_id": self.snapshot.market_id}]

    def normalize(self, raw):
        return self.snapshot
