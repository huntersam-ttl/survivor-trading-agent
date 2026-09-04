"""Wiring layer for the Survivor control plane.

``SurvivorLLM`` is a role-tagged proxy placed between TradingAgents agents and
their LLM. Every ``invoke`` is routed through the ``ModelRouter`` (health check
→ allowlist/price check → budget reservation → provider client → settlement),
so no agent can bypass the budget control plane when
``TRADINGAGENTS_SURVIVOR_ENABLED=true``.

When Survivor is disabled, agents receive the raw upstream LLM object and
upstream behaviour is preserved exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from tradingagents.llm_clients.router import ModelRouter

_run_id_var: ContextVar[str] = ContextVar("survivor_run_id", default="default_run")
_ticker_var: ContextVar[str] = ContextVar("survivor_ticker", default="GLOBAL")


def set_survivor_run(run_id: str, ticker_or_market: str) -> tuple:
    """Set the current run identity for all Survivor-guarded LLM calls."""
    return (_run_id_var.set(run_id), _ticker_var.set(ticker_or_market))


def reset_survivor_run(tokens: tuple) -> None:
    """Reset run identity tokens returned by :func:`set_survivor_run`."""
    for var, token in zip((_run_id_var, _ticker_var), tokens, strict=True):
        var.reset(token)


def get_survivor_run() -> tuple[str, str]:
    """Return the (run_id, ticker_or_market) currently in context."""
    return _run_id_var.get(), _ticker_var.get()


def estimate_input_tokens(messages: Any) -> int:
    """Cheap deterministic character-based input token estimate (len/4)."""
    if messages is None:
        return 0
    if isinstance(messages, str):
        return max(1, len(messages) // 4)
    total = 0
    try:
        iterator = list(messages)
    except TypeError:
        return 1500
    for item in iterator:
        if isinstance(item, dict):
            total += len(str(item.get("content", "")))
        else:
            total += len(str(getattr(item, "content", "") or ""))
    return max(1, total // 4)


class SurvivorLLM:
    """Role-tagged LLM proxy enforcing the Survivor control plane on every call.

    Supports the LangChain surface used by TradingAgents agents: ``invoke``,
    ``bind_tools`` (tool-calling analysts) and ``with_structured_output``
    (Research Manager / Trader / Portfolio Manager). Binding methods are lazy:
    the transform is applied to whichever provider client the router selects
    at invoke time, so fallback providers still receive bound tools.
    """

    def __init__(
        self,
        router: ModelRouter,
        role: str,
        llm_kwargs: dict[str, Any] | None = None,
        transforms: tuple[Callable[[Any], Any], ...] = (),
    ):
        self._router = router
        self._role = role
        self._llm_kwargs = dict(llm_kwargs or {})
        self._transforms = tuple(transforms)

    @property
    def role(self) -> str:
        return self._role

    def _with_transform(self, fn: Callable[[Any], Any]) -> SurvivorLLM:
        return SurvivorLLM(self._router, self._role, self._llm_kwargs, self._transforms + (fn,))

    def bind_tools(self, tools: Any, **kwargs: Any) -> SurvivorLLM:
        return self._with_transform(lambda llm: llm.bind_tools(tools, **kwargs))

    def with_structured_output(self, schema: Any, **kwargs: Any) -> SurvivorLLM:
        return self._with_transform(lambda llm: llm.with_structured_output(schema, **kwargs))

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        run_id, ticker_or_market = get_survivor_run()

        # Router-level options must not leak into the inner client's invoke()
        router_kwargs: dict[str, Any] = {
            name: kwargs.pop(name) for name in ("mock_client_factory",) if name in kwargs
        }

        transforms = self._transforms
        if kwargs:
            invoke_kwargs = dict(kwargs)

            def apply_all(llm: Any) -> Any:
                for fn in transforms:
                    llm = fn(llm)
                return llm

            # Forward invoke kwargs through a wrapping transform so the routed
            # client receives them without the router needing to know about them.
            def transform_with_kwargs(llm: Any) -> Any:
                bound = apply_all(llm)

                class _KwargsBound:
                    def __getattr__(self, name: str) -> Any:
                        return getattr(bound, name)

                    def invoke(self, msgs: Any, **kw: Any) -> Any:
                        merged = {**invoke_kwargs, **kw}
                        return bound.invoke(msgs, **merged)

                return _KwargsBound()

            return self._router.invoke_role(
                role=self._role,
                messages=messages,
                run_id=run_id,
                ticker_or_market=ticker_or_market,
                est_input_tokens=estimate_input_tokens(messages),
                llm_transform=transform_with_kwargs,
                **router_kwargs,
                **self._llm_kwargs,
            )

        return self._router.invoke_role(
            role=self._role,
            messages=messages,
            run_id=run_id,
            ticker_or_market=ticker_or_market,
            est_input_tokens=estimate_input_tokens(messages),
            llm_transform=_chain_transforms(transforms),
            **router_kwargs,
            **self._llm_kwargs,
        )


def _chain_transforms(transforms: tuple[Callable[[Any], Any], ...]) -> Callable[[Any], Any] | None:
    if not transforms:
        return None

    def chained(llm: Any) -> Any:
        for fn in transforms:
            llm = fn(llm)
        return llm

    return chained
