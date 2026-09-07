"""Focused tests for the Jarvis World Intelligence integration (SURVIVOR).

Jarvis is mocked only (a fake HTTP session / a monkeypatched client class). No
network, no live model/API calls, no real money. We verify the client's
fail-open + retry + validation behavior, the digest-aware research cache reuse,
and that the existing SURVIVOR cycle is unchanged whenever Jarvis fails.
"""

import datetime as _dt

import pytest
import requests as _requests

from tests.survivor.market_helpers import FakeMarketAdapter, make_snapshot
from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger
from tradingagents.survivor.autonomy import cycle as cycle_mod
from tradingagents.survivor.autonomy.cycle import (
    ResearchResult,
    _acquire_jarvis_intel,
    _default_research,
    run_survivor_cycle,
)
from tradingagents.survivor.autonomy.state import RuntimeState
from tradingagents.survivor.intelligence import (
    MAX_ANALYSES,
    JarvisDigest,
    JarvisIntelligenceClient,
    fingerprint_analyses,
    remember_digest_fingerprint,
    research_state_hash,
)
from tradingagents.survivor.policy import SurvivorPolicy


def _analysis(i: int) -> dict:
    return {
        "headline": f"headline-{i}",
        "summary": f"summary-{i}",
        "significance": 0.9,
        "confidence": 0.8,
        "affectedCountries": ["US"],
        "affectedSectors": ["Energy"],
        "affectedCommodities": ["OIL"],
        "affectedSymbols": ["USO"],
        "impactDirection": "bullish",
        "impactHorizon": "short",
        # non-whitelisted field — must never reach the research context
        "fullGodsEyePayload": {"lengthy": "x" * 500},
    }


def _payload(n: int, generated_at=None, headline_prefix: str = "headline"):
    analyses = []
    for i in range(n):
        a = _analysis(i)
        a["headline"] = f"{headline_prefix}-{i}"
        analyses.append(a)
    return {"generated_at": generated_at, "analyses": analyses}


class _FakeResponse:
    def __init__(self, status_code: int, payload=None, malformed: bool = False):
        self.status_code = status_code
        self._payload = payload
        self._malformed = malformed

    def json(self):
        if self._malformed:
            raise ValueError("not json")
        return self._payload


class _ScriptedSession:
    """A ``requests.Session``-shaped fake: each ``get`` consumes one scripted
    step, which is either a response or an exception to raise."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.meta = []

    def get(self, url, params=None, timeout=None):
        self.meta.append({"url": url, "params": params, "timeout": timeout})
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _client(steps, **kw):
    session = _ScriptedSession(steps)
    client = JarvisIntelligenceClient(
        base_url=kw.pop("base_url", "http://jarvis.test"),
        enabled=kw.pop("enabled", True),
        limit=kw.pop("limit", 10),
        session=session,
        **kw,
    )
    return client, session


def _iso_ago(seconds: int) -> str:
    return (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=seconds)).isoformat()


# --- client: valid digest / compact context / bounded ------------------------

@pytest.mark.unit
def test_valid_digest_parses_compact_fields_and_drops_non_whitelisted():
    now = _iso_ago(3)
    client, session = _client([_FakeResponse(200, _payload(2, generated_at=now))])
    digest = client.get_digest(["USO"])
    assert digest is not None
    assert len(digest.analyses) == 2
    assert digest.analyses[0]["headline"] == "headline-0"
    # whitelisted fields survive parse
    for field in ("headline", "summary", "affectedSectors", "impactDirection"):
        assert field in digest.analyses[0]
    # non-whitelisted fields are dropped (context control)
    assert "fullGodsEyePayload" not in digest.analyses[0]
    block = digest.compact_context()
    assert "JARVIS WORLD INTELLIGENCE DIGEST" in block
    assert "headline:" in block
    assert "fullGodsEyePayload" not in block
    # bounded age
    assert 0 <= digest.age_seconds() <= 5
    # symbols limit forwarded, timeout honored
    assert session.meta[0]["params"]["symbols"] == "USO"
    assert session.meta[0]["timeout"] == 5


@pytest.mark.unit
def test_bounded_to_max_10_analyses():
    client, _session = _client([_FakeResponse(200, _payload(25))])
    digest = client.get_digest(["USO"])
    assert digest is not None
    assert len(digest.analyses) == MAX_ANALYSES  # clamped to 10


@pytest.mark.unit
def test_limit_argument_is_clamped_to_10():
    client, session = _client([_FakeResponse(200, _payload(9))])
    digest = client.get_digest(["USO"], limit=999)
    assert digest is not None
    assert session.meta[0]["params"]["limit"] == MAX_ANALYSES


# --- client: disabled / unconfigured / fail-open failures --------------------

@pytest.mark.unit
def test_disabled_returns_none_without_network():
    client, session = _client([_FakeResponse(200, _payload(1))], enabled=False)
    assert client.get_digest(["USO"]) is None
    assert session.meta == []  # no network hit


@pytest.mark.unit
def test_unconfigured_returns_none_without_network():
    client = JarvisIntelligenceClient(
        base_url="", enabled=True, limit=10, session=_ScriptedSession([_FakeResponse(200, _payload(1))])
    )
    assert client.unconfigured is True
    assert client.get_digest(["USO"]) is None


@pytest.mark.unit
def test_401_returns_none_and_does_not_retry():
    client, session = _client([_FakeResponse(401, _payload(1))])
    assert client.get_digest(["USO"]) is None
    assert len(session.meta) == 1  # no retry on 4xx


@pytest.mark.unit
def test_403_returns_none_and_does_not_retry():
    client, session = _client([_FakeResponse(403, _payload(1))])
    assert client.get_digest(["USO"]) is None
    assert len(session.meta) == 1


@pytest.mark.unit
def test_malformed_payload_returns_none():
    client, _session = _client([_FakeResponse(200, malformed=True)])
    assert client.get_digest(["USO"]) is None


@pytest.mark.unit
def test_missing_analyses_returns_none():
    client, _session = _client([_FakeResponse(200, {"generated_at": _iso_ago(1)})])
    assert client.get_digest(["USO"]) is None


# --- client: retry rules -----------------------------------------------------

@pytest.mark.unit
def test_timeout_retries_once_then_succeeds():
    steps = [
        _requests.exceptions.Timeout("boom"),
        _FakeResponse(200, _payload(1)),
    ]
    client, session = _client(steps)
    digest = client.get_digest(["USO"])
    assert digest is not None
    assert len(session.meta) == 2  # exactly one retry


@pytest.mark.unit
def test_500_retries_once_then_succeeds():
    steps = [
        _FakeResponse(500, _payload(1)),
        _FakeResponse(200, _payload(1)),
    ]
    client, session = _client(steps)
    digest = client.get_digest(["USO"])
    assert digest is not None
    assert len(session.meta) == 2  # exactly one retry


@pytest.mark.unit
def test_persistent_500_no_retry_storm():
    steps = [_FakeResponse(500, _payload(1)), _FakeResponse(500, _payload(1)), _FakeResponse(200, _payload(1))]
    client, session = _client(steps)
    assert client.get_digest(["USO"]) is None
    assert len(session.meta) == 2  # initial + 1 retry, then give up (no storm)


@pytest.mark.unit
def test_jarvis_offline_connection_error_returns_none():
    steps = [_requests.exceptions.ConnectionError("offline"), _requests.exceptions.ConnectionError("offline")]
    client, session = _client(steps)
    assert client.get_digest(["USO"]) is None
    assert len(session.meta) == 2  # initial + 1 retry, then give up


# --- digest cache / fingerprint ----------------------------------------------

@pytest.mark.unit
def test_research_state_hash_digest_aware():
    fp_a = fingerprint_analyses([_analysis(0)])
    fp_b = fingerprint_analyses([_analysis(1)])
    assert fp_a != fp_b
    # same snapshot + same fingerprint -> same research-state hash (cache reuse)
    assert research_state_hash("snap", fp_a) == research_state_hash("snap", fp_a)
    # changed digest -> different research-state hash (re-analyze)
    assert research_state_hash("snap", fp_a) != research_state_hash("snap", fp_b)
    # no digest -> plain snapshot hash preserved (existing cache-key compatibility)
    assert research_state_hash("snap", None) == "snap"


@pytest.mark.unit
def test_remember_digest_fingerprint_cache_hit_tracking():
    fp = fingerprint_analyses([_analysis(0)])
    assert remember_digest_fingerprint(fp) is False  # miss on first use
    assert remember_digest_fingerprint(fp) is True   # cache hit on reuse


# --- integration: digest injected into research context only -----------------

class _CaptureGraph:
    """Fakes TradingAgentsGraph at the class boundary; captures research config."""

    def __init__(self, config=None, **kwargs):
        self.config = config or {}
        type(self)._last_config = self.config

    def propagate(self, *a, **k):
        return {"final_trade_decision": "P(YES): 70%"}, None


@pytest.mark.unit
def test_default_research_folds_digest_into_research_context(monkeypatch):
    import tradingagents.graph.trading_graph as tg_mod
    from tradingagents.survivor.execution.pricing import MarketSnapshot as QuoteSnapshot

    monkeypatch.setattr(tg_mod, "TradingAgentsGraph", _CaptureGraph)
    snap = make_snapshot("mkt-jarv")
    candidate = type("C", (), {"snapshot": snap, "market_id": "mkt-jarv"})()
    quote = QuoteSnapshot(symbol=snap.market_id, bid_pence=snap.bid // 100,
                          ask_pence=snap.ask // 100, timestamp_utc=snap.timestamp_utc)
    digest = JarvisDigest(analyses=[_analysis(0)])
    result = _default_research(
        candidate, quote, {"_jarvis_digest": digest}, run_id="run_j1"
    )
    assert result.status == "OK"
    context = _CaptureGraph._last_config["survivor_instrument_context"]
    assert "JARVIS WORLD INTELLIGENCE DIGEST" in context
    assert "headline:" in context
    assert "fullGodsEyePayload" not in context  # context control enforced


@pytest.mark.unit
def test_default_research_without_digest_is_unchanged(monkeypatch):
    import tradingagents.graph.trading_graph as tg_mod
    from tradingagents.survivor.execution.pricing import MarketSnapshot as QuoteSnapshot

    monkeypatch.setattr(tg_mod, "TradingAgentsGraph", _CaptureGraph)
    snap = make_snapshot("mkt-plain")
    candidate = type("C", (), {"snapshot": snap, "market_id": "mkt-plain"})()
    quote = QuoteSnapshot(symbol=snap.market_id, bid_pence=snap.bid // 100,
                          ask_pence=snap.ask // 100, timestamp_utc=snap.timestamp_utc)
    result = _default_research(candidate, quote, {}, run_id="run_plain")
    assert result.status == "OK"
    context = _CaptureGraph._last_config["survivor_instrument_context"]
    assert "JARVIS WORLD INTELLIGENCE DIGEST" not in context


# --- integration: cycle fail-open when Jarvis fails --------------------------

def _config(**over):
    base = {"survivor_autonomy_enabled": True}
    base.update(over)
    return base


def _cycle(tmp_path, adapter, monkeypatch, research_fn=None, usage_ledger=None, **cfg):
    monkeypatch.delenv("SURVIVOR_DRY_RUN", raising=False)
    config = _config(**cfg)
    config["_evaluation_db_path"] = str(tmp_path / "evaluation.db")
    config["_heartbeat_path"] = str(tmp_path / "heartbeat.json")
    return run_survivor_cycle(
        config,
        adapter=adapter,
        research_fn=research_fn,
        paper_ledger_path=str(tmp_path / "paper.db"),
        runtime_state=RuntimeState(db_path=str(tmp_path / "runtime.db")),
        usage_ledger=usage_ledger or InferenceUsageLedger(db_path=str(tmp_path / "usage.db")),
        policy=SurvivorPolicy(),
        lock_path=str(tmp_path / "cycle.lock"),
        halt_path=str(tmp_path / "HALT"),
    )


def _ok_research(run_id: str) -> ResearchResult:
    return ResearchResult(
        status="OK", decision_text="BUY", run_id=run_id,
        expected_probability_bps=7000, quantity=1,
        spread_cost_bps=50, slippage_bps=50, fee_bps=50, uncertainty_penalty_bps=100,
    )


@pytest.mark.unit
def test_cycle_unchanged_when_jarvis_client_raises(tmp_path, monkeypatch):
    class _Boom(JarvisIntelligenceClient):
        def __init__(self, *a, **k):
            raise RuntimeError("jarvis totally down")

    monkeypatch.setattr(cycle_mod, "JarvisIntelligenceClient", _Boom)
    calls = {"n": 0}

    def research(candidate, quote, config, run_id):
        calls["n"] += 1
        return _ok_research(run_id)

    adapter = FakeMarketAdapter([make_snapshot("mkt-0"), make_snapshot("mkt-1")])
    report = _cycle(tmp_path, adapter, monkeypatch, research_fn=research)
    # existing behavior preserved: cycle active, all candidates researched
    assert report.status == "ACTIVE"
    assert report.research_candidates == 2
    assert report.ai_researched == 2
    assert report.candidate_failures == []
    assert calls["n"] == 2
    # jarvis observations still recorded (fail-open), no crash
    jarvis = report.details.get("jarvis", {})
    assert jarvis.get("reachable") is False


@pytest.mark.unit
def test_cycle_unchanged_when_jarvis_returns_none(tmp_path, monkeypatch):
    class _NoneClient(JarvisIntelligenceClient):
        def get_digest(self, *a, **k):
            return None

    monkeypatch.setattr(cycle_mod, "JarvisIntelligenceClient", _NoneClient)
    report = _cycle(
        tmp_path, FakeMarketAdapter([make_snapshot("mkt-0")]), monkeypatch,
        research_fn=lambda c, q, cfg, r: _ok_research(r),
    )
    assert report.status == "ACTIVE"
    assert report.ai_researched == 1
    # disabled jarvis is observable; digest absent from context default
    assert report.details["jarvis"]["reachable"] is False


@pytest.mark.unit
def test_acquire_jarvis_intel_disabled_reports_enabled_false():
    digest, metrics = _acquire_jarvis_intel(
        ["USO"], {"jarvis_intelligence_enabled": False}, _dt.datetime.now(_dt.timezone.utc)
    )
    assert digest is None
    assert metrics["enabled"] is False


# --- no live-trade path introduced -------------------------------------------

@pytest.mark.unit
def test_no_live_trade_path_introduced():
    import inspect

    import tradingagents.survivor.intelligence as m

    src = inspect.getsource(m)
    # must not reference execution / graph / broker / PaperBroker
    for banned in ("PaperBroker", "execution.", "graph.trading_graph", "broker."):
        assert banned not in src
