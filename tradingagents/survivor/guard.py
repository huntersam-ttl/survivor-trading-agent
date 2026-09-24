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

from langchain_core.runnables import Runnable

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


class SurvivorLLM(Runnable):
    """Role-tagged LLM proxy enforcing the Survivor control plane on every call.

    A real LangChain ``Runnable`` so TradingAgents agents can compose it
    directly (``prompt | llm.bind_tools(...)``) — LangGraph sequences coerce
    sequence members via :meth:`Runnable.coerce`, which requires this base
    class. ``invoke`` routes through ``ModelRouter`` (health → allowlist/price
    → budget reservation → provider client → settlement), so no agent can
    bypass the budget control plane when ``TRADINGAGENTS_SURVIVOR_ENABLED``.

    ``bind_tools`` and ``with_structured_output`` return another guarded
    ``SurvivorLLM`` — the binding transform is applied lazily to whichever
    provider client the router selects at invoke time, so the raw provider
    client NEVER escapes this wrapper and fallback providers still receive
    bound tools.
    """

    InputType = Any
    OutputType = Any

    def __init__(
        self,
        router: ModelRouter,
        role: str,
        llm_kwargs: dict[str, Any] | None = None,
        transforms: tuple[Callable[[Any], Any], ...] = (),
        transform_specs: tuple[tuple[str, Any, dict], ...] = (),
        structured: bool = False,
    ):
        self._router = router
        self._role = role
        self._llm_kwargs = dict(llm_kwargs or {})
        self._transforms = tuple(transforms)
        self._transform_specs = tuple(transform_specs)
        self._structured = structured
        # Runnable.get_name() support for LangChain tracing/composition
        self.name = f"SurvivorLLM[{role}]"

    @property
    def role(self) -> str:
        return self._role

    def _with_transform(
        self, fn: Callable[[Any], Any], spec: tuple[str, Any, dict],
        *, structured: bool | None = None,
    ) -> SurvivorLLM:
        return SurvivorLLM(
            self._router,
            self._role,
            self._llm_kwargs,
            self._transforms + (fn,),
            self._transform_specs + (spec,),
            self._structured if structured is None else structured,
        )

    def bind_tools(self, tools: Any, **kwargs: Any) -> SurvivorLLM:
        return self._with_transform(
            lambda llm: llm.bind_tools(tools, **kwargs), ("bind_tools", tools, kwargs)
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> SurvivorLLM:
        # Structured-output calls use the ModelRouter's repair path (retry the
        # same approved model once, then rotate within configured OpenRouter).
        return self._with_transform(
            lambda llm: llm.with_structured_output(schema, **kwargs),
            ("structured", schema, kwargs),
            structured=True,
        )

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Route ONE guarded inference through ModelRouter/BudgetManager.

        ``config`` is the LangChain RunnableConfig passed by sequences like
        ``prompt | llm``; it is forwarded to the routed provider client so
        tracing/callbacks keep working — never used to bypass the guard.
        """
        run_id, ticker_or_market = get_survivor_run()

        # Router-level options must not leak into the inner client's invoke()
        router_kwargs: dict[str, Any] = {
            name: kwargs.pop(name) for name in ("mock_client_factory",) if name in kwargs
        }

        transforms = self._transforms
        if config is not None or kwargs:
            invoke_kwargs = dict(kwargs)
            if config is not None:
                invoke_kwargs["config"] = config

            def apply_all(llm: Any) -> Any:
                for fn in transforms:
                    llm = fn(llm)
                return llm

            # Forward invoke kwargs/config through a wrapping transform so the
            # routed client receives them without the router needing to know.
            def transform_with_kwargs(llm: Any) -> Any:
                bound = apply_all(llm)

                class _KwargsBound:
                    def __getattr__(self, name: str) -> Any:
                        return getattr(bound, name)

                    def invoke(self, msgs: Any, **kw: Any) -> Any:
                        merged = {**invoke_kwargs, **kw}
                        return bound.invoke(msgs, **merged)

                return _KwargsBound()

            # Real provider calls cross a multiprocessing ``spawn`` boundary.
            # The serialized transform specs are sufficient there; the local
            # closure is retained only for hermetic in-process test doubles.
            routed_transform = (
                transform_with_kwargs if "mock_client_factory" in router_kwargs else None
            )
            return self._router.invoke_role(
                role=self._role,
                messages=input,
                run_id=run_id,
                ticker_or_market=ticker_or_market,
                est_input_tokens=estimate_input_tokens(input),
                purpose="structured_output" if self._structured else "standard",
                llm_transform=routed_transform,
                llm_specs=self._transform_specs,
                **router_kwargs,
                **self._llm_kwargs,
            )

        # Never pass local closures to a spawned real-provider worker.  The
        # provider reconstructs bind/structured transforms from llm_specs.
        routed_transform = (
            _chain_transforms(transforms) if "mock_client_factory" in router_kwargs else None
        )
        return self._router.invoke_role(
            role=self._role,
            messages=input,
            run_id=run_id,
            ticker_or_market=ticker_or_market,
            est_input_tokens=estimate_input_tokens(input),
            purpose="structured_output" if self._structured else "standard",
            llm_transform=routed_transform,
            llm_specs=self._transform_specs,
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
