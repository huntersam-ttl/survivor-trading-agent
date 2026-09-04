"""Deterministic scanner: filters, ranking, top-N, zero LLMs (AST-proven)."""

import ast
import inspect

import pytest

from tests.survivor.market_helpers import FakeMarketAdapter, make_snapshot
from tradingagents.survivor.markets.filters import ScanLimits, filter_market
from tradingagents.survivor.markets.ranking import RankingWeights, rank_candidates, score_snapshot
from tradingagents.survivor.markets.scanner import MarketScanner
from tradingagents.survivor.markets.types import Candidate, ScanRejection

LIMITS = ScanLimits()


def _adapter(n=5, **over):
    return FakeMarketAdapter([make_snapshot(f"mkt-{i}", **over) for i in range(n)])


def _scanner(adapter, top=3):
    return MarketScanner(adapter, limits=LIMITS, max_candidates_per_cycle=40,
                         max_research_candidates_per_cycle=top)


@pytest.mark.unit
def test_scanner_uses_zero_llms():
    """Structural proof: the scanner module imports no LLM and calls no invoke()."""
    import tradingagents.survivor.markets.scanner as scanner_module

    tree = ast.parse(inspect.getsource(scanner_module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    for module in imported:
        low = module.lower()
        for token in ("llm", "openai", "anthropic", "requests", "urllib", "socket"):
            assert token not in low, f"forbidden import in scanner: {module}"
    assert not any(isinstance(n, ast.Attribute) and n.attr == "invoke" for n in ast.walk(tree))


@pytest.mark.unit
def test_filters_reject_unsafe_markets():
    # closed / resolved / missing close time / stale / invalid price / wide spread / low liquidity
    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    assert filter_market(make_snapshot(market_status=__import__(
        "tradingagents.survivor.markets.types", fromlist=["MarketStatus"]).MarketStatus.CLOSED), LIMITS) == ScanRejection.MARKET_CLOSED
    assert filter_market(make_snapshot(close_time_utc=None), LIMITS) == ScanRejection.MISSING_CLOSE_TIME
    assert filter_market(make_snapshot(bid=None, ask=None), LIMITS) == ScanRejection.MISSING_PRICE
    assert filter_market(make_snapshot(bid=0), LIMITS) == ScanRejection.INVALID_PRICE
    assert filter_market(make_snapshot(bid=4000, ask=5200), LIMITS) == ScanRejection.SPREAD_TOO_WIDE
    assert filter_market(make_snapshot(liquidity_usd_cents=100), LIMITS) == ScanRejection.LOW_LIQUIDITY
    assert filter_market(make_snapshot(liquidity=None), LIMITS) == ScanRejection.MISSING_LIQUIDITY
    from datetime import timedelta
    stale = make_snapshot()
    from tradingagents.survivor.markets.types import MarketSnapshot
    stale_snapshot = MarketSnapshot(**{**stale.__dict__, "source_timestamp_utc": (now - timedelta(seconds=3600)).isoformat()})
    assert filter_market(stale_snapshot, LIMITS) == ScanRejection.STALE_DATA
    assert filter_market(make_snapshot(close_hours=0.1), LIMITS) == ScanRejection.RESOLUTION_TOO_CLOSE
    assert filter_market(make_snapshot(close_hours=24 * 400), LIMITS) == ScanRejection.RESOLUTION_TOO_FAR
    assert filter_market(make_snapshot(market_type="perpetual_future"), LIMITS) == ScanRejection.UNSUPPORTED_MARKET_TYPE


@pytest.mark.unit
def test_deterministic_ranking_stable_and_ties_broken_by_id():
    a = make_snapshot("aaa", liquidity_usd_cents=200000)
    b = make_snapshot("bbb", liquidity_usd_cents=200000)
    ranked1 = rank_candidates([Candidate(b), Candidate(a)], RankingWeights())
    ranked2 = rank_candidates([Candidate(a), Candidate(b)], RankingWeights())
    # identical inputs in any order -> identical, deterministic ranking
    assert [c.market_id for c in ranked1] == [c.market_id for c in ranked2]
    assert ranked1[0].market_id == "aaa"  # exact tie -> lexicographic market_id
    assert ranked1[0].rank == 1 and ranked1[1].rank == 2


@pytest.mark.unit
def test_higher_liquidity_ranks_higher():
    small = make_snapshot("small", liquidity_usd_cents=110000)
    big = make_snapshot("big", liquidity_usd_cents=900000)
    ranked = rank_candidates([Candidate(small), Candidate(big)], RankingWeights())
    assert ranked[0].market_id == "big"
    assert score_snapshot(big, RankingWeights()) > score_snapshot(small, RankingWeights())


@pytest.mark.unit
def test_top_n_enforced():
    adapter = _adapter(8)
    scan = _scanner(adapter, top=3).scan()
    assert scan.discovered == 8
    assert len(scan.candidates) == 8
    assert len(scan.ranked) == 8
    assert len(scan.top) == 3
    # ranks are contiguous from 1 and sorted
    assert [c.rank for c in scan.ranked] == list(range(1, 9))
    assert scan.top[0].score >= scan.top[-1].score


@pytest.mark.unit
def test_existing_position_filters_candidate():
    adapter = _adapter(2)
    scan = _scanner(adapter, top=3).scan(
        open_position_symbols=frozenset({"mkt-0"})
    )
    rejected_ids = [mid for mid, reason in scan.rejected if reason == ScanRejection.EXISTING_POSITION]
    assert "mkt-0" in rejected_ids
    assert all(c.market_id != "mkt-0" for c in scan.candidates)


@pytest.mark.unit
def test_polymarket_normalization_keeps_usd_semantics():
    from tradingagents.survivor.markets.polymarket_adapter import PolymarketAdapter

    raw = {
        "id": "12345", "question": "Will it happen?", "slug": "will-it-happen",
        "outcomes": '["Yes", "No"]', "outcomePrices": '[0.62, 0.38]',
        "bestBid": 0.61, "bestAsk": 0.63, "volumeNum": 25000.0,
        "liquidityNum": 8000.0, "endDate": "2027-01-01T00:00:00Z", "closed": False,
    }
    snap = PolymarketAdapter.normalize(raw)
    assert snap is not None
    assert snap.market_probability_bps == 6200
    assert snap.bid == 6100 and snap.ask == 6300
    assert snap.liquidity.minor_units == 800000 and snap.liquidity.currency_upper == "USD"
    assert snap.volume_24h.minor_units == 2500000 and snap.volume_24h.currency_upper == "USD"
    assert snap.liquidity.currency_upper != "GBP"
