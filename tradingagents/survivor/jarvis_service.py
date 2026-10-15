"""Localhost HTTP service exposing read-only SURVIVOR state and SAFE paper actions to Jarvis.

Served on 127.0.0.1 only. Bearer-token auth via ``SURVIVOR_JARVIS_TOKEN``.

Safety guarantees (enforced here and mirrored by the underlying SURVIVOR seams):

* Read-only state endpoints (health/status/budget/performance/positions).
* POST /v1/run/dry and /v1/run/paper delegate to the single tested cycle
  entrypoint ``run_survivor_cycle`` - no trading logic is duplicated.
* GET /v1/recovery-preflight performs bounded read-only market probes.
* POST /v1/halt writes the standard PAPER-autonomy HALT file. There is NO
  /v1/resume and NO auto-clear: HALT can only be lifted by an operator editing
  the filesystem, and nothing here can enable live trading.
* No live-trading, no risk/budget mutation, no wallet/withdrawal/leverage
  endpoints exist on this service at all.

The token is compared constant-time and is never logged; request logging emits
only the method/path (the stdlib handler never logs request headers).
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from tradingagents.survivor.ops.service_heartbeat import (
    ServiceLivenessHeartbeat,
    missing_service_heartbeat,
)

logger = logging.getLogger(__name__)

BIND_HOST = "127.0.0.1"  # hard-coded: localhost-only, never configurable
DEFAULT_PORT = 8765


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ServiceInjection:
    """Dependency overrides used by tests to stay hermetic (no real DBs/network).

    When left as ``None`` each handler builds its production counterpart from
    SURVIVOR's existing env-driven seams. Nothing here re-implements trading.
    """

    adapter: Any = None
    research_fn: Any = None
    paper_ledger_path: str | None = None
    runtime_state: Any = None
    usage_ledger: Any = None
    policy: Any = None
    now: Any = None
    lock_path: str | None = None
    settings: Any = None            # risk limits used to build a read-only broker
    broker: Any = None              # pre-built read-only PaperBroker
    service_heartbeat: Any = None   # in-process HTTP service liveness owner
    evaluation_db_path: str | None = None
    config_overrides: dict = field(default_factory=dict)  # extra run-cycle config keys
    acceptance_store: Any = None
    status_fn: Any = None              # deterministic local service test seam
    cycle_fn: Any = None               # hermetic single-call test seam


_CYCLE_KWARGS = (
    "adapter",
    "research_fn",
    "paper_ledger_path",
    "runtime_state",
    "usage_ledger",
    "policy",
    "now",
    "lock_path",
)


class SurvivorHandlers:
    """Handlers over SURVIVOR internals - no duplicated trading logic."""

    def __init__(self, inject: ServiceInjection | None = None):
        self.inject = inject or ServiceInjection()

    def _acceptance_store(self):
        if self.inject.acceptance_store is None:
            from tradingagents.survivor.acceptance import PaperAcceptanceStore
            self.inject.acceptance_store = PaperAcceptanceStore()
        return self.inject.acceptance_store

    def acceptance_status(self, acceptance_id: str) -> dict:
        record = self._acceptance_store().get(acceptance_id)
        if record is None:
            raise ApiError(404, "ACCEPTANCE_NOT_FOUND", "paper acceptance reservation not found")
        return {"acceptance_id": record.acceptance_id, "fingerprint": record.fingerprint,
                "status": record.status, "result": record.result}

    def acceptance_reserve(self, payload: dict) -> dict:
        if not isinstance(payload, dict) or set(payload) != {"acceptance_id", "request"}:
            raise ApiError(400, "INVALID_ACCEPTANCE_REQUEST", "acceptance_id and request are required")
        request = payload.get("request")
        if (not isinstance(request, dict) or request.get("mode") != "PAPER_RUNTIME"
                or request.get("cycle") not in {"single_provider_cycle", None}
                or set(request) - {"mode", "cycle", "symbol", "signal_timestamp"}):
            raise ApiError(400, "INVALID_ACCEPTANCE_REQUEST", "only allowlisted PAPER_RUNTIME requests are accepted")
        try:
            record, created = self._acceptance_store().reserve(payload["acceptance_id"], request)
        except (TypeError, ValueError) as exc:
            raise ApiError(409, "ACCEPTANCE_CONFLICT", str(exc)) from None
        return {"acceptance_id": record.acceptance_id, "fingerprint": record.fingerprint,
                "status": record.status, "created": created}

    def acceptance_authorize(self, payload: dict) -> dict:
        if not isinstance(payload, dict) or set(payload) - {"acceptance_id", "ttl_seconds", "operator_authorized", "operator_identity"}:
            raise ApiError(400, "INVALID_ACCEPTANCE_AUTHORIZATION", "unsupported authorization fields")
        if payload.get("operator_authorized") is not True or not isinstance(payload.get("operator_identity"), str):
            raise ApiError(403, "OPERATOR_AUTHORIZATION_REQUIRED", "operator authorization is required")
        try:
            record = self._acceptance_store().authorize(payload.get("acceptance_id"),
                ttl_seconds=payload.get("ttl_seconds", 90), operator_identity=payload["operator_identity"])
        except (TypeError, ValueError) as exc:
            raise ApiError(409, "ACCEPTANCE_CONFLICT", str(exc)) from None
        return {"acceptance_id": record.acceptance_id, "status": record.status,
                "expires_at": record.result.get("expires_at")}

    def acceptance_dispatch(self, payload: dict) -> dict:
        if not isinstance(payload, dict) or set(payload) != {"acceptance_id"}:
            raise ApiError(400, "INVALID_ACCEPTANCE_DISPATCH", "acceptance_id is required")
        acceptance_id = payload["acceptance_id"]
        store = self._acceptance_store()
        prior = store.get(acceptance_id)
        if prior is None:
            raise ApiError(404, "ACCEPTANCE_NOT_FOUND", "paper acceptance reservation not found")
        if prior.status != "AUTHORIZED":
            # Result retrieval is replay-safe; no second invocation is possible.
            return self.acceptance_status(acceptance_id)

        # Verify all current execution attestations before persisting the
        # irreversible dispatch boundary or releasing either gate.
        status = self.inject.status_fn() if self.inject.status_fn is not None else self.status()
        heartbeat = status.get("service_heartbeat") or {}
        limits_before = self._risk_limits_fingerprint()
        safe_mode = str(status.get("mode", "")).upper().replace(" ", "_") in {"PAPER", "PAPER_ONLY"}
        heartbeat_ok = (heartbeat.get("present") is True and heartbeat.get("consistent") is True
                        and heartbeat.get("age_seconds") is not None
                        and 0 <= heartbeat.get("age_seconds") <= 30)
        gates_ok = (safe_mode and status.get("broker_type") == "paper"
                    and status.get("live_broker_available") is False
                    and status.get("live_order_transport_available") is False
                    and status.get("paper_run_enabled") is False and status.get("halted") is True
                    and heartbeat_ok
                    and os.environ.get("TRADINGAGENTS_SURVIVOR_AUTONOMY_ENABLED", "").strip().lower() not in {"1", "true", "yes", "on"})
        if not gates_ok:
            record = store.transition(acceptance_id, {"AUTHORIZED"}, "REJECTED",
                                      {**prior.result, "reason": "safety_attestation_failed", "dispatch_performed": False})
            return {"acceptance_id": record.acceptance_id, "status": record.status,
                    "result": record.result, "dispatch_performed": False}
        try:
            store.mark_dispatched(acceptance_id)
        except ValueError as exc:
            raise ApiError(409, "ACCEPTANCE_CONFLICT", str(exc)) from None

        from tradingagents.survivor.autonomy.halt import clear_halt, set_halt
        from tradingagents.survivor.autonomy.cycle import run_survivor_cycle
        from tradingagents.survivor.execution.ledger import PaperLedger
        cycle_result = None
        cycle_error = None
        halt_restored = False
        restoration_status = {}
        try:
            ledger = PaperLedger(db_path=self.inject.paper_ledger_path)
            before = ledger.events()
            if not clear_halt((self.inject.config_overrides or {}).get("_halt_path")):
                raise RuntimeError("HALT could not be released for authorized attempt")
            kwargs = self._run_kwargs(dry=False)
            kwargs["config"]["survivor_paper_enabled"] = True  # in-memory, one-call permit only
            cycle = self.inject.cycle_fn or run_survivor_cycle
            report = cycle(**kwargs)
            report_counts = (report.risk_decisions, report.approved, report.rejected,
                             report.paper_trades, report.ai_cost_pence)
            if (any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in report_counts)
                    or report.approved + report.rejected != report.risk_decisions):
                raise RuntimeError("cycle returned an invalid risk or accounting result")
            # Verify production risk configuration remained unchanged and only
            # record concrete append-only ledger evidence; never synthesize fills.
            if self._risk_limits_fingerprint() != limits_before:
                raise RuntimeError("risk limits changed during acceptance")
            ledger.verify_chain()
            after = ledger.events()
            new_events = after[len(before):]
            ledger_fills = sum(event.get("event_type") == "TRADE_EXECUTED" for event in new_events)
            if report.paper_trades != ledger_fills:
                raise RuntimeError("cycle trade count does not match appended PaperLedger evidence")
            result = {"cycle_id": report.cycle_id, "cycle_status": report.status,
                      "risk_decisions": report.risk_decisions, "approved": report.approved,
                      "rejected": report.rejected, "paper_trades": report.paper_trades,
                      "provider_usage_pence": report.ai_cost_pence,
                      "ledger_event_ids": [event["event_id"] for event in new_events],
                      "ledger_event_hashes": [event["event_hash"] for event in new_events],
                      "ledger_chain_verified": True}
            cycle_result = result
        except Exception as exc:
            # A cycle may have reached provider or ledger work. Never retry.
            cycle_error = type(exc).__name__
        finally:
            try:
                set_halt((self.inject.config_overrides or {}).get("_halt_path"))
                restoration_status = self.status()
                from tradingagents.survivor.autonomy.halt import is_halted
                restored_halt = is_halted((self.inject.config_overrides or {}).get("_halt_path"))
                heartbeat_after = restoration_status.get("service_heartbeat") or {}
                halt_restored = (restored_halt is True and restoration_status.get("halted") is True
                    and restoration_status.get("paper_run_enabled") is False
                    and restoration_status.get("live_broker_available") is False
                    and restoration_status.get("live_order_transport_available") is False
                    and heartbeat_after.get("present") is True and heartbeat_after.get("consistent") is True
                    and heartbeat_after.get("halted") is True)
            except Exception:
                # DISPATCHED is recovered to RECONCILIATION_REQUIRED on boot;
                # leave the durable record intact and fail the API response.
                logger.critical("failed to restore TradingAgents HALT after acceptance")
        if not halt_restored or cycle_error:
            safe_result = {**prior.result, "reason": cycle_error or "HALT_RESTORATION_FAILED", "operator_action_required": True,
                           "halt_restored": halt_restored}
            try:
                store.transition(acceptance_id, {"DISPATCHED"}, "RECONCILIATION_REQUIRED", safe_result)
            except Exception:
                pass
            raise ApiError(503, "RECONCILIATION_REQUIRED", "acceptance outcome is uncertain; operator reconciliation required")
        record = store.transition(acceptance_id, {"DISPATCHED"}, "COMPLETED",
                                  {**prior.result, **(cycle_result or {})})
        return {"acceptance_id": record.acceptance_id, "status": record.status,
                "result": record.result, "dispatch_performed": True, "halt_restored": True,
                "restoration": {"paper_opt_in_disabled": restoration_status.get("paper_run_enabled") is False,
                    "live_broker_unavailable": restoration_status.get("live_broker_available") is False,
                    "live_order_transport_unavailable": restoration_status.get("live_order_transport_available") is False,
                    "heartbeat_consistent": (restoration_status.get("service_heartbeat") or {}).get("consistent") is True}}

    @staticmethod
    def _risk_limits_fingerprint() -> str:
        import hashlib
        import json
        from dataclasses import asdict
        from tradingagents.survivor.risk.limits import risk_limits_from_env
        return hashlib.sha256(json.dumps(asdict(risk_limits_from_env()), sort_keys=True).encode()).hexdigest()

    def recover_acceptances(self) -> None:
        recovered = self._acceptance_store().recover_interrupted()
        if recovered:
            from tradingagents.survivor.autonomy.halt import set_halt
            set_halt((self.inject.config_overrides or {}).get("_halt_path"))

    def health(self) -> dict:
        from tradingagents.survivor import is_survivor_enabled

        return {
            "status": "ok",
            "service": "survivor-jarvis",
            "time_utc": _utc_now_iso(),
            "survivor_enabled": bool(is_survivor_enabled()),
        }

    def status(self) -> dict:
        if self.inject.status_fn is not None:
            return self.inject.status_fn()
        from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger
        from tradingagents.survivor import is_survivor_enabled
        from tradingagents.survivor.autonomy import halt as halt_mod
        from tradingagents.survivor.autonomy.state import RuntimeState
        from tradingagents.survivor.execution.paper_broker import PaperBroker
        from tradingagents.survivor.markets.adapter import FORBIDDEN_METHODS, MarketAdapter
        from tradingagents.survivor.markets.polymarket_adapter import PolymarketAdapter
        from tradingagents.survivor.ops.health import heartbeat_read, watchdog_status
        from tradingagents.survivor.policy import SurvivorPolicy
        from tradingagents.survivor.pipeline import is_paper_enabled
        from tradingagents.survivor.risk.limits import risk_limits_from_env

        inj = self.inject
        broker = inj.broker or PaperBroker(limits=inj.settings or risk_limits_from_env())
        adapter = inj.adapter or PolymarketAdapter()
        # Attest only the concrete paper execution seam and the read-only
        # market-data interface used by the cycle. Custom broker/adapter
        # implementations remain unknown; absence is never guessed.
        paper_broker_confirmed = type(broker) is PaperBroker
        read_only_market_confirmed = isinstance(adapter, MarketAdapter) and not any(
            hasattr(adapter, method) for method in FORBIDDEN_METHODS
        )
        paper_execution_confirmed = paper_broker_confirmed and read_only_market_confirmed
        state = inj.runtime_state or RuntimeState()
        policy = inj.policy or SurvivorPolicy.from_env()
        usage = inj.usage_ledger or InferenceUsageLedger()
        paper = broker.status()
        last = state.last_cycle()
        heartbeat = heartbeat_read()
        service_heartbeat = (
            inj.service_heartbeat.read()
            if inj.service_heartbeat is not None
            else missing_service_heartbeat()
        )
        watchdog = watchdog_status(state)
        return {
            "mode": paper["mode"],
            "broker_type": "paper" if paper_broker_confirmed else "unknown",
            "live_broker_available": False if paper_broker_confirmed else None,
            "live_order_transport_available": False if paper_execution_confirmed else None,
            "paper_run_enabled": is_paper_enabled(),
            "halted": halt_mod.is_halted((inj.config_overrides or {}).get("_halt_path")),
            # `heartbeat` remains the cycle/trial heartbeat. This distinct
            # field proves only that this HTTP service process is alive.
            "service_heartbeat": service_heartbeat,
            "heartbeat": heartbeat,
            "watchdog": watchdog,
            "survivor_enabled": bool(is_survivor_enabled()),
            "cycle_count": state.cycle_count(),
            "last_cycle": last,
            "paper": {
                "starting_equity_pence": paper["starting_equity_pence"],
                "cash_pence": paper["cash_pence"],
                "exposure_pence": paper["exposure_pence"],
                "equity_pence": paper["equity_pence"],
                "realized_pnl_pence": paper["realized_pnl_pence"],
                "unrealized_pnl_pence": paper["unrealized_pnl_pence"],
                "high_water_mark_pence": paper["high_water_mark_pence"],
                "drawdown_bps": paper["drawdown_bps"],
                "daily_pnl_pence": paper["daily_pnl_pence"],
                "open_positions": paper["open_positions"],
                "trading_state": paper["trading_state"],
            },
            "ai_spend": {
                "daily_pence": usage.get_daily_spend_pence(None),
                "monthly_pence": usage.get_monthly_spend_pence(None),
                "daily_limit_pence": policy.global_daily_pence,
                "monthly_limit_pence": policy.global_monthly_pence,
            },
        }

    def recovery_preflight(self) -> dict:
        from tradingagents.survivor.autonomy.state import RuntimeState
        from tradingagents.survivor.execution.paper_broker import PaperBroker
        from tradingagents.survivor.markets.polymarket_adapter import PolymarketAdapter
        from tradingagents.survivor.ops.health import recovery_preflight
        from tradingagents.survivor.risk.limits import risk_limits_from_env

        inj = self.inject
        adapter = inj.adapter or PolymarketAdapter()
        broker = inj.broker or PaperBroker(limits=inj.settings or risk_limits_from_env())
        return recovery_preflight(
            runtime_state=inj.runtime_state or RuntimeState(),
            adapter=adapter,
            paper_status=broker.status(),
            heartbeat_path=(inj.config_overrides or {}).get("_heartbeat_path"),
        )

    def recovery_reconcile(self) -> dict:
        from tradingagents.survivor.autonomy.state import RuntimeState
        from tradingagents.survivor.execution.paper_broker import PaperBroker
        from tradingagents.survivor.markets.polymarket_adapter import PolymarketAdapter
        from tradingagents.survivor.ops.health import reconcile_recovery_state
        from tradingagents.survivor.risk.limits import risk_limits_from_env

        inj = self.inject
        adapter = inj.adapter or PolymarketAdapter()
        broker = inj.broker or PaperBroker(limits=inj.settings or risk_limits_from_env())
        overrides = inj.config_overrides or {}
        return reconcile_recovery_state(
            runtime_state=inj.runtime_state or RuntimeState(),
            adapter=adapter,
            heartbeat_path=overrides.get("_heartbeat_path"),
            halt_path=overrides.get("_halt_path"),
        )

    def budget(self) -> dict:
        from tradingagents.llm_clients.usage_ledger import InferenceUsageLedger
        from tradingagents.survivor.policy import SurvivorPolicy

        inj = self.inject
        ledger = inj.usage_ledger or InferenceUsageLedger()
        policy = inj.policy or SurvivorPolicy.from_env()
        daily = {p: ledger.get_daily_spend_pence(p) for p in ("openai", "deepseek", "minimax", "openrouter")}
        daily["global_pence"] = ledger.get_daily_spend_pence(None)
        daily["global_limit_pence"] = policy.global_daily_pence
        monthly = {p: ledger.get_monthly_spend_pence(p) for p in ("openai", "deepseek", "minimax", "openrouter")}
        monthly["global_pence"] = ledger.get_monthly_spend_pence(None)
        monthly["global_limit_pence"] = policy.global_monthly_pence
        return {"unit": "pence", "currency": "GBP", "daily": daily, "monthly": monthly}

    def performance(self) -> dict:
        import json as _json

        from tradingagents.survivor.evaluation.evaluate import (
            evaluate_performance,
            export_json,
        )
        from tradingagents.survivor.evaluation.store import EvaluationStore

        inj = self.inject
        store = EvaluationStore(db_path=inj.evaluation_db_path)
        report = evaluate_performance(store, paper_ledger_path=inj.paper_ledger_path)
        return _json.loads(export_json(report))

    def positions(self) -> dict:
        from tradingagents.survivor.execution.paper_broker import PaperBroker
        from tradingagents.survivor.risk.limits import risk_limits_from_env

        inj = self.inject
        broker = inj.broker or PaperBroker(limits=inj.settings or risk_limits_from_env())
        rows = []
        for symbol, pos in broker.portfolio.positions.items():
            if pos.quantity <= 0:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "quantity": pos.quantity,
                    "average_entry_price_pence": pos.average_entry_price_pence,
                    "cost_basis_pence": pos.cost_basis_pence,
                    "current_mark_pence": pos.current_mark_pence,
                    "market_value_pence": pos.market_value_pence,
                    "unrealized_pnl_pence": pos.unrealized_pnl_pence,
                    "opened_at": pos.opened_at,
                    "updated_at": pos.updated_at,
                    "stale_mark": pos.is_stale_mark,
                }
            )
        return {"positions": rows}

    def signals(self) -> dict:
        from tradingagents.survivor.markets.filters import scan_limits_from_env
        from tradingagents.survivor.markets.ranking import ranking_weights_from_env
        from tradingagents.survivor.markets.scanner import MarketScanner
        from tradingagents.survivor.markets.signals import (
            MAX_CANDIDATES,
            build_signal_summary,
        )

        inj = self.inject
        adapter = inj.adapter
        if adapter is None:
            from tradingagents.survivor.markets.polymarket_adapter import PolymarketAdapter

            adapter = PolymarketAdapter()
        scanner = MarketScanner(
            adapter,
            limits=scan_limits_from_env(),
            weights=ranking_weights_from_env(),
            max_candidates_per_cycle=MAX_CANDIDATES,
        )
        return build_signal_summary(scanner.scan())

    def history(self) -> dict:
        from tradingagents.survivor.history import HistoryStore

        return HistoryStore(read_only=True, initialize=False).stats()

    def run_dry(self) -> dict:
        from tradingagents.survivor.autonomy.cycle import run_survivor_cycle

        return asdict(run_survivor_cycle(**self._run_kwargs(dry=True)))

    def run_paper(self) -> dict:
        from tradingagents.survivor.autonomy.cycle import run_survivor_cycle
        from tradingagents.survivor.autonomy.halt import is_halted
        from tradingagents.survivor.pipeline import is_paper_enabled

        if is_halted():
            return {
                "status": "EXTERNAL_HALT",
                "reason": "external HALT is active; operator action is required",
                "paper_trades": 0,
            }
        if not is_paper_enabled():
            return {
                "status": "PAPER_OPT_IN_REQUIRED",
                "reason": "TRADINGAGENTS_SURVIVOR_PAPER_ENABLED must be explicitly enabled",
                "paper_trades": 0,
            }

        return asdict(run_survivor_cycle(**self._run_kwargs(dry=False)))

    def halt(self) -> dict:
        from tradingagents.survivor.autonomy.halt import set_halt

        set_halt()
        return {
            "halted": True,
            "note": "PAPER autonomy halted. HALT never auto-clears; only an operator "
                    "can remove the halt. Nothing here enables live trading.",
        }

    def _run_kwargs(self, *, dry: bool) -> dict:
        inj = self.inject
        config = {
            "survivor_autonomy_enabled": "true",
            "survivor_dry_run": dry,
        }
        config.update(inj.config_overrides or {})
        kwargs: dict[str, Any] = {"config": config}
        for name in _CYCLE_KWARGS:
            value = getattr(inj, name)
            if value is not None:
                kwargs[name] = value
        return kwargs
class JarvisApp:
    """Thin route table -> handler dispatch (no business logic here)."""

    ROUTES = {
        ("GET", "/v1/health"): "health",
        ("GET", "/v1/status"): "status",
        ("GET", "/v1/budget"): "budget",
        ("GET", "/v1/performance"): "performance",
        ("GET", "/v1/positions"): "positions",
        ("GET", "/v1/signals"): "signals",
        ("GET", "/v1/history"): "history",
        ("GET", "/v1/recovery-preflight"): "recovery_preflight",
        ("POST", "/v1/recovery-reconcile"): "recovery_reconcile",
        ("POST", "/v1/run/dry"): "run_dry",
        ("POST", "/v1/run/paper"): "run_paper",
        ("POST", "/v1/halt"): "halt",
        ("POST", "/v1/acceptance/reserve"): "acceptance_reserve",
        ("POST", "/v1/acceptance/authorize"): "acceptance_authorize",
        ("POST", "/v1/acceptance/dispatch"): "acceptance_dispatch",
    }

    def __init__(self, handlers: SurvivorHandlers, token: str):
        self.handlers = handlers
        self.token = token

    def dispatch(self, method: str, path: str, payload: dict | None = None):
        handler_name = self.ROUTES.get((method, path))
        if method == "GET" and path.startswith("/v1/acceptance/"):
            acceptance_id = path.removeprefix("/v1/acceptance/")
            if not acceptance_id or "/" in acceptance_id:
                raise ApiError(404, "NOT_FOUND", f"no endpoint {method} {path}")
            return self.handlers.acceptance_status(acceptance_id)
        if handler_name is None:
            raise ApiError(404, "NOT_FOUND", f"no endpoint {method} {path}")
        handler = getattr(self.handlers, handler_name)
        return handler(payload or {}) if method == "POST" and path.startswith("/v1/acceptance/") else handler()


class _JarvisRequestHandler(BaseHTTPRequestHandler):
    server_version = "SurvivorJarvis/1.0"

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        token = self.server.token
        if not token:
            return False
        expected = "Bearer " + token
        provided = self.headers.get("Authorization", "")
        return secrets.compare_digest(provided.strip(), expected)

    def _handle(self, method: str) -> None:
        if not self._authorized():
            self._send_json(
                401, {"error": {"code": "UNAUTHORIZED", "message": "missing or invalid bearer token"}}
            )
            return
        path = urlsplit(self.path).path
        try:
            payload = None
            if method == "POST":
                raw_length = self.headers.get("Content-Length", "0")
                if not raw_length.isdigit() or int(raw_length) > 16384:
                    raise ApiError(413, "REQUEST_TOO_LARGE", "request body exceeds the acceptance limit")
                if int(raw_length):
                    try:
                        payload = json.loads(self.rfile.read(int(raw_length)))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        raise ApiError(400, "INVALID_JSON", "request body must be JSON") from None
                    if not isinstance(payload, dict):
                        raise ApiError(400, "INVALID_JSON", "request body must be an object")
            payload = self.server.app.dispatch(method, path, payload)
        except ApiError as exc:
            self._send_json(exc.status, {"error": {"code": exc.code, "message": exc.message}})
            return
        except Exception:  # noqa: BLE001 - never leak internals to callers
            logger.exception("unhandled error serving %s %s", method, path)
            self._send_json(500, {"error": {"code": "INTERNAL", "message": "internal server error"}})
            return
        self._send_json(200, payload)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        self._handle("POST")


class _SurvivorServer(ThreadingHTTPServer):
    allow_reuse_address = True


def make_server(
    *,
    port: int = DEFAULT_PORT,
    token: str | None = None,
    inject: ServiceInjection | None = None,
):
    """Build (but do not start) the localhost server. Binds only to 127.0.0.1."""
    resolved_token = (token if token is not None else os.environ.get("SURVIVOR_JARVIS_TOKEN", "")).strip()
    server = _SurvivorServer((BIND_HOST, port), _JarvisRequestHandler)
    resolved_inject = inject or ServiceInjection()
    if resolved_inject.service_heartbeat is None:
        resolved_inject.service_heartbeat = ServiceLivenessHeartbeat(
            halt_path=(resolved_inject.config_overrides or {}).get("_halt_path"),
        )
    server.service_heartbeat = resolved_inject.service_heartbeat
    resolved_handlers = SurvivorHandlers(resolved_inject)
    server.app = JarvisApp(resolved_handlers, resolved_token)
    server.token = resolved_token
    server.daemon_threads = True
    return server


def serve(*, port: int = DEFAULT_PORT) -> None:
    """Blocking entrypoint. Requests are refused (fail closed) until a token is set."""
    server = make_server(port=port)
    host, bound_port = server.server_address
    token_state = "configured" if server.token else "MISSING - requests will be refused"
    sys.stderr.write(f"SURVIVOR jarvis localhost service on {host}:{bound_port} (bearer token {token_state})\n")
    sys.stderr.flush()
    try:
        server.app.handlers.recover_acceptances()
        # The heartbeat belongs to this bound HTTP service process, not a
        # trading cycle or a standalone writer. Its identity is checked again
        # by every status read, and it stops/cleans up with this lifecycle.
        server.service_heartbeat.start()
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.service_heartbeat.stop()
        server.server_close()


__all__ = [
    "ApiError",
    "BIND_HOST",
    "DEFAULT_PORT",
    "JarvisApp",
    "ServiceInjection",
    "SurvivorHandlers",
    "make_server",
    "serve",
    "_main",
]


def _main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: ``python -m tradingagents.survivor.jarvis_service --port 8765``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SURVIVOR localhost Jarvis service (read-only state + SAFE dry/paper/halt)."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SURVIVOR_JARVIS_PORT", str(DEFAULT_PORT))),
        help=f"port to bind on 127.0.0.1 (default {DEFAULT_PORT})",
    )
    args = parser.parse_args(argv)
    serve(port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
