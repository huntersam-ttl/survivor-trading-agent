"""Phase 2 execution-boundary tests: graph decision -> proposal -> risk -> broker.

Proves the mandatory non-LLM boundary: approved proposals execute exactly once;
rejected proposals NEVER reach PaperBroker.execute; disabled flag preserves
upstream behavior; no network/broker/wallet code exists in the survivor tree.
"""

import os

import pytest

from tests.survivor.paper_helpers import iso_now
from tradingagents.survivor.execution.ledger import PaperLedger
from tradingagents.survivor.execution.paper_broker import PaperBroker
from tradingagents.survivor.pipeline import execute_final_decision, is_paper_enabled
from tradingagents.survivor.risk.limits import RiskLimits


def _paper_config(tmp_path, **over_inputs):
    inputs = {
        'symbol': "AAPL", 'market': "stock", 'bid_pence': 99, 'ask_pence': 100,
        'snapshot_timestamp_utc': iso_now(), 'run_id': "run_b1",
        'quantity': 1,
        'expected_probability_bps': 6000, 'market_probability_bps': 5200,
        'spread_cost_bps': 50, 'slippage_bps': 50, 'fee_bps': 50,
        'uncertainty_penalty_bps': 100,
    }
    inputs.update(over_inputs)
    return {
        "survivor_paper_enabled": True,
        "survivor_paper_inputs": inputs,
        "_paper_db": str(tmp_path) + "/boundary.db",
    }


def _run(config, final_decision="BUY - recommended by AI"):
    return execute_final_decision(
        {"final_trade_decision": final_decision}, config,
        ledger=PaperLedger(db_path=config["_paper_db"]),
    )


@pytest.mark.unit
def test_approved_decision_executes_once_and_changes_portfolio(tmp_path):
    config = _paper_config(tmp_path)
    result = _run(config)
    assert result.status == "EXECUTED"
    assert result.risk_decision.approved
    assert result.event["event_type"] == "TRADE_EXECUTED"

    broker = PaperBroker(limits=RiskLimits(), ledger=PaperLedger(db_path=config["_paper_db"]))
    # broker was seeded with £20 treasury, bought £1.00 notional (+costs)
    assert broker.portfolio.cash_pence == 2000 - 101 - 1
    assert broker.portfolio.positions["AAPL"].quantity == 1
    types = [e["event_type"] for e in broker.ledger.events()]
    assert types.count("PROPOSAL_RECEIVED") == 1
    assert types.count("RISK_APPROVED") == 1
    assert types.count("TRADE_EXECUTED") == 1


@pytest.mark.unit
def test_rejected_decision_never_reaches_paper_broker(tmp_path):
    # gross = 5200 - 5701 = -501bps -> net edge far below 500 -> deterministic REJECT
    config = _paper_config(tmp_path, expected_probability_bps=5200, market_probability_bps=5701)
    result = _run(config)
    assert result.status == "REJECTED"
    assert result.risk_decision.reason_code.value == "EDGE_TOO_LOW"

    broker = PaperBroker(limits=RiskLimits(), ledger=PaperLedger(db_path=config["_paper_db"]))
    executions = [e for e in broker.ledger.events() if e["event_type"] == "TRADE_EXECUTED"]
    assert executions == []  # PaperBroker.execute was NEVER called
    assert broker.portfolio.cash_pence == 2000  # portfolio untouched
    assert [e["event_type"] for e in broker.ledger.events()].count("RISK_REJECTED") == 1

@pytest.mark.unit
def test_duplicate_proposal_cannot_execute_twice(tmp_path):
    config = _paper_config(tmp_path)
    first = _run(config)
    assert first.status == "EXECUTED"
    second = _run(config)  # identical deterministic proposal
    assert second.status == "REJECTED"
    assert second.risk_decision.reason_code.value == "DUPLICATE_PROPOSAL"
    ledger = PaperLedger(db_path=config["_paper_db"])
    assert len([e for e in ledger.events() if e["event_type"] == "TRADE_EXECUTED"]) == 1


@pytest.mark.unit
def test_paper_disabled_preserves_upstream_behavior(tmp_path):
    assert is_paper_enabled({}) is False
    config = _paper_config(tmp_path)
    config["survivor_paper_enabled"] = False
    result = execute_final_decision(
        {"final_trade_decision": "BUY"}, config, ledger=PaperLedger(db_path=config["_paper_db"])
    )
    assert result.status == "DISABLED"
    assert result.proposal is None
    assert PaperLedger(db_path=config["_paper_db"]).events() == []  # nothing written anywhere


@pytest.mark.unit
def test_survivor_enabled_without_paper_flag_does_not_execute(tmp_path):
    """Inference control plane on, paper execution off -> no paper trades."""
    config = _paper_config(tmp_path)
    config["survivor_enabled"] = True
    config["survivor_paper_enabled"] = False
    result = execute_final_decision(
        {"final_trade_decision": "BUY"}, config, ledger=PaperLedger(db_path=config["_paper_db"])
    )
    assert result.status == "DISABLED"


@pytest.mark.unit
def test_missing_market_inputs_fail_closed_to_no_trade(tmp_path):
    config = _paper_config(tmp_path)
    del config["survivor_paper_inputs"]["ask_pence"]
    result = _run(config)
    assert result.status == "NO_TRADE"
    ledger = PaperLedger(db_path=config["_paper_db"])
    assert not any(e["event_type"] == "TRADE_EXECUTED" for e in ledger.events())


@pytest.mark.unit
def test_no_broker_exchange_wallet_network_code_in_survivor():
    """Static security scan over the whole survivor package."""
    forbidden = (
        "ccxt", "binance", "coinbase", "alpaca", "interactive brokers",
        "private_key", "place_order", "real_money",
        "requests.get", "urllib.request", "socket.socket", "boto3", "websocket", "api_key=",
    )
    pkg_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "tradingagents", "survivor",
    ))
    for root, _, files in os.walk(pkg_dir):
        for name in files:
            if not name.endswith(".py"):
                continue
            with open(os.path.join(root, name)) as fh:
                content = fh.read().lower()
            for token in forbidden:
                assert token not in content, f"{token!r} found in {name}"

