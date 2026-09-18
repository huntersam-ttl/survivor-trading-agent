"""Polymarket READ-ONLY market adapter over the public Gamma API.

Reuses the keyless public data client already shipped in
``tradingagents.dataflows.polymarket``. No trading API is contacted.

Currency: Gamma volumes/liquidity are USD. They are carried as
``MoneyAmount(minor_units, currency="USD")`` — never relabelled GBP.
Prices are binary-market probabilities carried in basis points (1.00 = 10000).
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import queue
import time
from datetime import datetime, timezone

import requests

from tradingagents.dataflows import polymarket as gamma
from tradingagents.survivor.markets.adapter import MarketAdapter
from tradingagents.survivor.markets.types import (
    MarketSnapshot,
    MarketStatus,
    QuoteCurrency,
    ResolutionStatus,
)

PROVIDER = "polymarket"
MARKET_TYPE = "prediction_binary"
PAGE_SIZE = 100
DEFAULT_UNIVERSE_LIMIT = 1000
PAGE_RETRIES = 1
PAGE_BACKOFF_SECONDS = 0.2
logger = logging.getLogger(__name__)


class GammaPageError(RuntimeError):
    """A required Gamma page could not be fetched safely."""

    def __init__(self, category: str, message: str, telemetry: dict):
        super().__init__(message)
        self.category = category
        self.telemetry = telemetry


def _page_deadline_seconds() -> float:
    try:
        return max(1.0, float(os.environ.get("SURVIVOR_GAMMA_PAGE_DEADLINE_SECONDS", "30")))
    except ValueError:
        return 30.0


def _gamma_page_worker(path: str, params: dict, result_queue) -> None:
    """Fetch one page in an isolated process; output is bounded and serializable."""
    started = time.monotonic()
    try:
        with requests.request(
            "GET",
            f"{gamma.GAMMA_BASE}/{path}", params=params,
            timeout=(gamma.REQUEST_TIMEOUT, gamma.REQUEST_TIMEOUT),
        ) as response:
            status = int(response.status_code)
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, list):
            raise ValueError("Gamma response was not a JSON list")
        result_queue.put({
            "ok": True, "status": status, "rows": data,
            "duration_ms": round((time.monotonic() - started) * 1000, 2),
        })
    except requests.exceptions.ConnectTimeout as exc:
        result_queue.put({"ok": False, "category": "GAMMA_CONNECT_TIMEOUT", "message": str(exc)[:200]})
    except requests.exceptions.ReadTimeout as exc:
        result_queue.put({"ok": False, "category": "GAMMA_READ_TIMEOUT", "message": str(exc)[:200]})
    except requests.exceptions.HTTPError as exc:
        result_queue.put({"ok": False, "category": "GAMMA_HTTP_ERROR", "message": str(exc)[:200]})
    except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as exc:
        result_queue.put({"ok": False, "category": "GAMMA_INVALID_JSON", "message": str(exc)[:200]})
    except Exception as exc:  # noqa: BLE001 - child reports bounded failure only
        result_queue.put({"ok": False, "category": "GAMMA_WORKER_ERROR", "message": str(exc)[:200]})


def _run_gamma_page(path: str, params: dict, deadline: float, worker=None) -> dict:
    """Run one fetch in a killable process and guarantee parent-side cleanup."""
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    target = worker or _gamma_page_worker
    process = context.Process(target=target, args=(path, params, result_queue), daemon=True)
    started = time.monotonic()
    process.start()
    result = None
    deadline_at = started + max(0.001, float(deadline))
    # Drain the result queue while the child is alive. A successful page can
    # contain hundreds of rows; joining first would let Queue.put block on its
    # pipe and falsely turn a completed HTTP request into a deadline timeout.
    while result is None and time.monotonic() < deadline_at:
        remaining = max(0.001, deadline_at - time.monotonic())
        try:
            result = result_queue.get(timeout=min(0.1, remaining))
        except queue.Empty:
            if not process.is_alive():
                break
    if result is not None:
        process.join(max(0.0, deadline_at - time.monotonic()))
    if process.is_alive():
        process.terminate()
        process.join(2.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(2.0)
        if result is None:
            result = {
                "ok": False, "category": "GAMMA_TOTAL_TIMEOUT",
                "message": f"worker exceeded {deadline:.1f}s deadline",
            }
    if result is None:
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            result = {"ok": False, "category": "GAMMA_WORKER_ERROR", "message": "worker returned no result"}
    result["worker_pid"] = process.pid
    result["duration_ms"] = round((time.monotonic() - started) * 1000, 2)
    result_queue.close()
    result_queue.join_thread()
    process.close()
    return result


def _iso_to_iso(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return None


class PolymarketAdapter(MarketAdapter):
    """Read-only Polymarket market-data adapter (public Gamma API, keyless)."""

    provider = PROVIDER
    market_type = MARKET_TYPE

    def __init__(self):
        self.last_scan_telemetry = {
            "pages_attempted": 0, "pages_completed": 0, "rows_fetched": 0,
            "scan_duration_ms": 0.0, "failure_page": None, "failure_category": None,
            "pages": [],
        }

    def _fetch_page(self, page: int, offset: int, params: dict) -> list[dict]:
        deadline = _page_deadline_seconds()
        retries = min(PAGE_RETRIES, 1)
        last_error = None
        for attempt in range(retries + 1):
            started = time.monotonic()
            status = "OK"
            category = None
            rows = 0
            try:
                result = _run_gamma_page("markets", params, deadline)
                if not result.get("ok"):
                    category = result.get("category", "GAMMA_WORKER_ERROR")
                    raise RuntimeError(result.get("message", "Gamma worker failed"))
                data = result.get("rows")
                if not isinstance(data, list):
                    category = "GAMMA_INVALID_JSON"
                    raise ValueError("Gamma worker returned non-list rows")
                rows = len(data)
                self.last_scan_telemetry["pages"].append({
                    "page": page, "offset": offset, "attempt": attempt + 1,
                    "worker_pid": result.get("worker_pid"),
                    "duration_ms": result.get("duration_ms", round((time.monotonic() - started) * 1000, 2)),
                    "rows": rows, "status": "SUCCESS", "category": None,
                })
                self.last_scan_telemetry["pages_completed"] += 1
                self.last_scan_telemetry["rows_fetched"] += rows
                return data
            except Exception as exc:  # bounded worker result or local validation
                if category is None:
                    category = "GAMMA_INVALID_JSON"
                last_error = exc
            status = "ERROR"
            self.last_scan_telemetry["pages"].append({
                "page": page, "offset": offset, "attempt": attempt + 1,
                "worker_pid": locals().get("result", {}).get("worker_pid"),
                "duration_ms": locals().get("result", {}).get("duration_ms", round((time.monotonic() - started) * 1000, 2)),
                "rows": rows, "status": status, "category": category,
            })
            if attempt < retries:
                logger.warning("Gamma page %s failed (%s); retrying once", page, category)
                time.sleep(PAGE_BACKOFF_SECONDS)
                continue
            self.last_scan_telemetry["failure_page"] = page
            self.last_scan_telemetry["failure_category"] = category
            self.last_scan_telemetry["scan_duration_ms"] = round(
                (time.monotonic() - self.last_scan_telemetry.get("_started_scan", time.monotonic())) * 1000, 2
            )
            self.last_scan_telemetry.pop("_started_scan", None)
            raise GammaPageError(
                category, f"Gamma page {page} offset {offset}: {category}",
                self.last_scan_telemetry,
            ) from last_error

    def list_markets(self, limit: int = DEFAULT_UNIVERSE_LIMIT) -> list[dict]:
        """Fetch open binary markets from the public Gamma ``/markets`` collection
        (keyless, read-only), ordered by traded volume.

        Gamma caps one page at 100 rows. Keep the universe bounded, but walk
        the provider's offset pages so the scanner is not permanently limited
        to the first volume-ranked page.
        """
        target = max(0, min(int(limit), DEFAULT_UNIVERSE_LIMIT))
        if target == 0:
            return []
        markets: list[dict] = []
        seen: set[str] = set()
        started_scan = time.monotonic()
        self.last_scan_telemetry = {
            "pages_attempted": 0, "pages_completed": 0, "rows_fetched": 0,
            "scan_duration_ms": 0.0, "failure_page": None, "failure_category": None,
            "pages": [],
        }
        self.last_scan_telemetry["_started_scan"] = started_scan
        for offset in range(0, target, PAGE_SIZE):
            page = offset // PAGE_SIZE + 1
            self.last_scan_telemetry["pages_attempted"] = page
            data = self._fetch_page(page, offset, {
                    "limit": min(PAGE_SIZE, target - offset),
                    "offset": offset,
                    "closed": "false",
                    "order": "volumeNum",
                    "ascending": "false",
                })
            for market in data:
                market_id = str(market.get("id") or "")
                if market_id and market_id in seen:
                    continue
                if market_id:
                    seen.add(market_id)
                markets.append(market)
            if len(data) < PAGE_SIZE:
                break
        self.last_scan_telemetry["scan_duration_ms"] = round((time.monotonic() - started_scan) * 1000, 2)
        self.last_scan_telemetry.pop("_started_scan", None)
        now = datetime.now(timezone.utc)
        forward = [m for m in markets if gamma._is_forward_looking(m, now)]
        forward.sort(key=lambda m: m.get("volumeNum") or 0, reverse=True)
        return forward[:target]

    @staticmethod
    def normalize(raw: dict, now: datetime | None = None) -> MarketSnapshot | None:
        """Normalize one raw Gamma market dict into a MarketSnapshot."""
        outcomes = gamma._parse_json_list(raw.get("outcomes"))
        prices = gamma._parse_json_list(raw.get("outcomePrices"))
        if len(outcomes) != 2 or len(prices) != 2:
            return None  # not a clean binary market -> unapplicable
        try:
            yes_prob = int(round(float(prices[0]) * 10000))
        except (ValueError, TypeError):
            return None
        if not 0 <= yes_prob <= 10000:
            return None

        def _bps(value) -> int | None:
            try:
                bps = int(round(float(value) * 10000))
            except (TypeError, ValueError):
                return None
            return bps if 0 <= bps <= 10000 else None

        best_bid = _bps(raw.get("bestBid"))
        best_ask = _bps(raw.get("bestAsk"))
        closed = bool(raw.get("closed"))
        # Gamma provides no boolean "resolved" flag on search results; a closed
        # market is treated as resolved for discovery purposes.
        resolution = ResolutionStatus.UNRESOLVED
        status = MarketStatus.CLOSED if closed else MarketStatus.OPEN

        now = now or datetime.now(timezone.utc)
        volume_num = raw.get("volumeNum")
        liquidity_num = raw.get("liquidityNum")
        canonical = json.dumps(
            {k: raw.get(k) for k in ("id", "question", "outcomes", "outcomePrices", "bestBid", "bestAsk")},
            sort_keys=True, default=str,
        )
        return MarketSnapshot(
            market_id=str(raw.get("id") or hashlib.sha256(canonical.encode()).hexdigest()[:16]),
            provider=PROVIDER,
            market_type=MARKET_TYPE,
            symbol_or_slug=str(raw.get("slug") or raw.get("id") or ""),
            question=str(raw.get("question") or ""),
            timestamp_utc=now.isoformat(),
            close_time_utc=_iso_to_iso(raw.get("endDate")),
            resolution_time_utc=_iso_to_iso(raw.get("endDate")),
            bid=best_bid,
            ask=best_ask,
            mid=(int((best_bid + best_ask) / 2) if best_bid is not None and best_ask is not None else None),
            yes_bid=best_bid,
            yes_ask=best_ask,
            no_bid=(10000 - best_ask) if best_ask is not None else None,
            no_ask=(10000 - best_bid) if best_bid is not None else None,
            market_probability_bps=yes_prob,
            liquidity=(
                MoneyAmountUSD(int(round(float(liquidity_num) * 100)))
                if isinstance(liquidity_num, (int, float)) else None
            ),
            volume_24h=(
                MoneyAmountUSD(int(round(float(volume_num) * 100)))
                if isinstance(volume_num, (int, float)) else None
            ),
            quote_currency=QuoteCurrency.USD.value,
            market_status=status,
            resolution_status=resolution,
            source_timestamp_utc=now.isoformat(),
        )

    def get_snapshot(self, market_id: str) -> MarketSnapshot | None:
        for raw in self.list_markets(limit=100):
            snap = self.normalize(raw)
            if snap and snap.market_id == market_id:
                return snap
        return None

    def get_resolution_status(self, market_id: str) -> ResolutionStatus:
        for raw in self.list_markets(limit=100):
            if str(raw.get("id")) == market_id:
                return ResolutionStatus.NO if raw.get("closed") else ResolutionStatus.UNRESOLVED
        return ResolutionStatus.UNKNOWN

    def get_resolved_outcome(self, market_id: str):
        """Return an outcome only when the provider exposes explicit resolution data.

        Closed markets alone are not treated as resolved: expiry and settlement
        are distinct, and this method fails closed when Gamma omits an explicit
        resolution timestamp/outcome.
        """
        for raw in self.list_markets(limit=100):
            if str(raw.get("id")) != market_id:
                continue
            if raw.get("resolved") is not True:
                return None
            prices = gamma._parse_json_list(raw.get("outcomePrices"))
            try:
                values = [float(v) for v in prices]
            except (TypeError, ValueError):
                return None
            if len(values) != 2 or values not in ([1.0, 0.0], [0.0, 1.0]):
                return None
            resolved_at = raw.get("resolvedAt") or raw.get("resolutionTime") or raw.get("closedTime")
            resolved_at = _iso_to_iso(resolved_at)
            if not resolved_at:
                return None
            return (1.0 if values[0] == 1.0 else 0.0, resolved_at, values[0])
        return None


def MoneyAmountUSD(minor_units: int):
    """Explicit USD money constructor (documents provider currency)."""
    from tradingagents.survivor.markets.types import MoneyAmount

    return MoneyAmount(minor_units, currency=QuoteCurrency.USD.value)
