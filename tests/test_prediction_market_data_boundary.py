"""Prediction-market runs must never treat opaque pm_* IDs as stock tickers."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from tradingagents.agents.analysts.market_analyst import create_market_analyst
from tradingagents.agents.analysts.news_analyst import create_news_analyst
from tradingagents.agents.analysts.sentiment_analyst import create_sentiment_analyst
from tradingagents.agents.utils.core_stock_tools import get_stock_data
from tradingagents.agents.utils.fundamental_data_tools import get_fundamentals
from tradingagents.agents.utils.news_data_tools import get_news
from tradingagents.agents.utils.technical_indicators_tools import get_indicators
from tradingagents.dataflows import reddit, stocktwits, y_finance, yfinance_news
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.market_data_validator import build_verified_market_snapshot

PM_ID = "pm_f057cf1cde94"
STATE = {
    "company_of_interest": PM_ID,
    "asset_type": "prediction_market",
    "trade_date": "2026-09-07",
    "instrument_context": (
        "INSTRUMENT: binary prediction market\n"
        "market_id: market-1\nquestion: Will event X happen?\n"
        "YES probability: 0.60\nbid: 0.58\nask: 0.62\n"
        "liquidity: 100000\nvolume: 5000\nclose: 2026-10-01\n"
        "decision-time timestamp: 2026-09-07T12:00:00Z"
    ),
    "messages": [],
}


@pytest.mark.unit
@pytest.mark.parametrize(
    "fn,args",
    [
        (get_stock_data, (PM_ID, "2026-09-01", "2026-09-07")),
        (get_indicators, (PM_ID, "rsi", "2026-09-07")),
        (get_fundamentals, (PM_ID, "2026-09-07")),
        (get_news, (PM_ID, "2026-09-01", "2026-09-07")),
    ],
)
def test_pm_id_is_rejected_before_stock_data_routing(fn, args):
    with pytest.raises(ValueError, match="PREDICTION_MARKET_STOCK_LOOKUP_BLOCKED"):
        fn.func(*args)


@pytest.mark.unit
def test_pm_id_is_rejected_before_yfinance(monkeypatch):
    ticker = MagicMock(side_effect=AssertionError("yfinance must not be called"))
    monkeypatch.setattr(y_finance.yf, "Ticker", ticker)
    monkeypatch.setattr(yfinance_news.yf, "Ticker", ticker)
    with pytest.raises(ValueError, match="PREDICTION_MARKET_STOCK_LOOKUP_BLOCKED"):
        y_finance.get_YFin_data_online(PM_ID, "2026-09-01", "2026-09-07")
    with pytest.raises(ValueError, match="PREDICTION_MARKET_STOCK_LOOKUP_BLOCKED"):
        yfinance_news.get_news_yfinance(PM_ID, "2026-09-01", "2026-09-07")


@pytest.mark.unit
def test_pm_id_is_rejected_before_yahoo_quote_summary_or_verified_snapshot(monkeypatch):
    ticker = MagicMock(side_effect=AssertionError("Yahoo must not be called"))
    monkeypatch.setattr(y_finance.yf, "Ticker", ticker)
    with pytest.raises(ValueError, match="PREDICTION_MARKET_STOCK_LOOKUP_BLOCKED"):
        build_verified_market_snapshot(PM_ID, "2026-09-07")
    assert not ticker.called


@pytest.mark.unit
def test_pm_id_is_rejected_before_stocktwits_or_reddit(monkeypatch):
    monkeypatch.setattr(stocktwits, "urlopen", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("StockTwits must not be called")
    ))
    monkeypatch.setattr(reddit, "urlopen", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("Reddit must not be called")
    ))
    with pytest.raises(ValueError, match="PREDICTION_MARKET_STOCK_LOOKUP_BLOCKED"):
        stocktwits.fetch_stocktwits_messages(PM_ID)
    with pytest.raises(ValueError, match="PREDICTION_MARKET_STOCK_LOOKUP_BLOCKED"):
        reddit.fetch_reddit_posts(PM_ID)


class _CapturingLLM:
    def __init__(self):
        self.tools = None
        self.prompts = []

    def bind_tools(self, tools):
        self.tools = list(tools)
        return self

    def with_structured_output(self, _schema):
        raise NotImplementedError

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content="P(YES): 60%", tool_calls=[])

    __call__ = invoke


@pytest.mark.unit
def test_prediction_market_analysts_use_snapshot_and_no_stock_tools(monkeypatch):
    market_llm = _CapturingLLM()
    create_market_analyst(market_llm)(STATE)
    assert market_llm.tools is None
    assert "YES probability" in str(market_llm.prompts[0])

    fundamentals_llm = _CapturingLLM()
    create_fundamentals_analyst(fundamentals_llm)(STATE)
    assert fundamentals_llm.tools is None
    assert "NOT_APPLICABLE" in str(fundamentals_llm.prompts[0])

    news_llm = _CapturingLLM()
    create_news_analyst(news_llm)(STATE)
    assert {tool.name for tool in news_llm.tools} == {
        "get_global_news", "get_macro_indicators", "get_prediction_markets"
    }
    assert "prediction-market question" in str(news_llm.prompts[0])


@pytest.mark.unit
def test_prediction_market_sentiment_skips_ticker_specific_sources(monkeypatch):
    monkeypatch.setattr(
        "tradingagents.agents.analysts.sentiment_analyst.get_news",
        MagicMock(side_effect=AssertionError("ticker news must not be called")),
    )
    monkeypatch.setattr(
        "tradingagents.agents.analysts.sentiment_analyst.fetch_stocktwits_messages",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("StockTwits must not be called")),
    )
    monkeypatch.setattr(
        "tradingagents.agents.analysts.sentiment_analyst.fetch_reddit_posts",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Reddit must not be called")),
    )
    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError
    llm.invoke.return_value = AIMessage(content="Insufficient event evidence")
    create_sentiment_analyst(llm)(STATE)


@pytest.mark.unit
def test_stock_route_still_uses_existing_vendor_path(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "tradingagents.dataflows.interface.VENDOR_METHODS",
        {"get_stock_data": {"test": lambda *args: sentinel}},
    )
    monkeypatch.setattr(
        "tradingagents.dataflows.interface.get_vendor", lambda category, method=None: "test"
    )
    assert route_to_vendor("get_stock_data", "AAPL", "2026-09-01", "2026-09-07") is sentinel
