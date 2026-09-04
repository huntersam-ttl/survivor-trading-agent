"""Append-only hash-chained ledger: tamper evidence, uniqueness, recovery."""

import sqlite3

import pytest

from tests.survivor.paper_helpers import fresh_snapshot, make_proposal, tmp_ledger
from tradingagents.survivor.execution.ledger import LedgerCorruptionError, PaperLedger
from tradingagents.survivor.execution.paper_broker import PaperBroker
from tradingagents.survivor.risk.limits import RiskLimits
from tradingagents.survivor.risk.result import ReasonCode, RiskDecision, RiskStatus


def _seed_broker(tmp_path):
    broker = PaperBroker(limits=RiskLimits(), ledger=tmp_ledger(tmp_path))
    broker.ensure_initialized()
    proposal = make_proposal()
    approved = RiskDecision(status=RiskStatus.APPROVED, reason_code=ReasonCode.APPROVED,
                            reason="ok", proposal_id=proposal.proposal_id)
    broker.execute(proposal, fresh_snapshot(), approved)
    broker.mark_to_market([fresh_snapshot(bid=99, ask=101)])
    return broker, proposal


@pytest.mark.unit
def test_chain_verifies_when_untampered(tmp_path):
    broker, _ = _seed_broker(tmp_path)
    broker.ledger.verify_chain()  # must not raise


@pytest.mark.unit
def test_corrupted_ledger_fails_recovery(tmp_path):
    broker, _ = _seed_broker(tmp_path)
    db = broker.ledger.db_path
    # tamper: rewrite a field of an earlier event in raw sqlite
    conn = sqlite3.connect(db)
    conn.execute("UPDATE paper_events SET notional_pence = 999 WHERE rowid = 2")
    conn.commit()
    conn.close()

    with pytest.raises(LedgerCorruptionError):
        PaperLedger(db).recover_portfolio()


@pytest.mark.unit
def test_broken_link_fails_recovery(tmp_path):
    broker, _ = _seed_broker(tmp_path)
    db = broker.ledger.db_path
    conn = sqlite3.connect(db)
    conn.execute("UPDATE paper_events SET previous_hash = 'DEADBEEF' WHERE rowid = 3")
    conn.commit()
    conn.close()
    with pytest.raises(LedgerCorruptionError):
        PaperLedger(db).verify_chain()


@pytest.mark.unit
def test_executed_proposal_unique_at_storage_layer(tmp_path):
    broker, proposal = _seed_broker(tmp_path)
    assert broker.ledger.has_executed(proposal.proposal_id)
    # direct storage-level replay is blocked by the unique partial index
    with pytest.raises(sqlite3.IntegrityError):
        broker.ledger.append_event(
            event_type="TRADE_EXECUTED", run_id=proposal.run_id, symbol=proposal.symbol,
            quantity=1, proposal_id=proposal.proposal_id,
        )


@pytest.mark.unit
def test_replay_rebuilds_portfolio_and_daily_pnl(tmp_path):
    broker, _ = _seed_broker(tmp_path)
    recovered = PaperLedger(broker.ledger.db_path).recover_portfolio()
    assert recovered.cash_pence == broker.portfolio.cash_pence
    assert recovered.positions["AAPL"].quantity == 1
    assert recovered.positions["AAPL"].current_mark_pence == 99
    # daily P/L includes the fee loss and the unrealized mark loss
    assert recovered.realized_pnl_today_pence < 0


@pytest.mark.unit
def test_ledger_records_no_secrets(tmp_path):
    broker, _ = _seed_broker(tmp_path)
    conn = sqlite3.connect(broker.ledger.db_path)
    rows = str(conn.execute("SELECT * FROM paper_events").fetchall())
    conn.close()
    for secret in ("sk-", "api_key", "API_KEY", "private_key", "OPENAI_API_KEY"):
        assert secret not in rows
