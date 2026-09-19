"""Shared helpers for invoking an agent with structured output and a graceful fallback.

The Portfolio Manager, Trader, and Research Manager all follow the same
canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output (rare; mostly older Ollama models), the
   wrap is skipped and the agent uses free-text generation instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call itself fails for any reason
   (malformed JSON from a weak model, transient provider issue), fall
   back to a plain ``llm.invoke`` so the pipeline never blocks.

Centralising the pattern here keeps the agent factories small and ensures
all three agents log the same warnings when fallback fires.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Schema-only structured output binds exactly one tool (the schema itself), so a
# model that reaches for a search tool emits an unknown tool call and the whole
# structured attempt is discarded for a free-text retry. Agents on this path
# state the constraint explicitly rather than relying on the binding alone
# (#1130).
NO_EXTERNAL_TOOLS = (
    "Use only the evidence provided in this prompt. Do not call external tools "
    "or search the web; if something is missing, say so explicitly."
)

_JSON_WRAPPERS = frozenset({
    "data", "output", "response", "result", "structured_output",
    # Common OpenAI-compatible tool-call envelopes.  These wrappers carry the
    # model's object; they do not add or infer any decision data.
    "args", "arguments", "tool_input",
})

_telemetry: ContextVar[dict[str, int] | None] = ContextVar("structured_telemetry", default=None)


def structured_telemetry_scope(values: dict[str, int]):
    """Collect bounded structured-output outcomes for one research run."""
    token = _telemetry.set(values)
    return token


def reset_structured_telemetry(token) -> None:
    _telemetry.reset(token)


def _record_telemetry(name: str) -> None:
    values = _telemetry.get()
    if values is not None:
        values[name] = int(values.get(name, 0)) + 1


def _record_failure_details(error: Exception) -> None:
    """Persist bounded field-level details without model prompts or output."""
    fields: list[str] = []
    safe_message = type(error).__name__
    try:
        errors = error.errors() if hasattr(error, "errors") else []
        fields = sorted({
            ".".join(str(part) for part in item.get("loc", ()))
            for item in errors
            if item.get("loc")
        })[:12]
        if errors:
            safe_message = "; ".join(
                f"{'.'.join(str(part) for part in item.get('loc', ())) or '<root>'}: "
                f"{str(item.get('msg', 'validation error'))[:180]}"
                for item in errors[:12]
            )[:500]
        else:
            safe_message = str(error)[:500]
    except Exception:  # diagnostics must never affect fail-closed behavior
        pass
    values = _telemetry.get()
    if values is not None:
        # Persist only bounded structural metadata.  Never persist prompts or
        # model output: this is for diagnosing recurring schema drift safely.
        diagnostics = values.setdefault("structured_failure_diagnostics", [])
        if isinstance(diagnostics, list) and len(diagnostics) < 12:
            diagnostics.append({
                "error_type": type(error).__name__,
                "fields": fields,
                "message": safe_message,
            })


def _record_failure_category(error: Exception) -> None:
    """Record one bounded, diagnostic category for a final structured failure."""
    category = "structured_failure_provider_parse"
    try:
        errors = error.errors() if hasattr(error, "errors") else []
        types = {str(item.get("type", "")) for item in errors}
        messages = " ".join(str(item.get("msg", "")) for item in errors).lower()
        if any(t == "missing" or t.endswith(".missing") for t in types):
            category = "structured_failure_missing_field"
        elif any("enum" in t for t in types) or "enum" in messages:
            category = "structured_failure_invalid_enum"
        elif any("type" in t or "parsing" in t for t in types):
            category = "structured_failure_invalid_type"
        elif isinstance(error, json.JSONDecodeError) or any(
            marker in messages for marker in ("json", "truncat", "unclosed", "unterminated")
        ):
            category = "structured_failure_truncated_json"
    except Exception:  # diagnostics must never affect fail-closed behavior
        pass
    _record_telemetry(category)
    _record_failure_details(error)


def _response_content(value: Any) -> str | None:
    """Extract text from common LangChain response shapes without inventing data."""
    if isinstance(value, str):
        return value
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts) or None
    return None


def _extract_json_object(text: str) -> dict | None:
    """Extract one top-level JSON object from bounded prose/fenced output."""
    candidate = text.strip()
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I | re.S).strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", candidate):
        try:
            value, _ = decoder.raw_decode(candidate[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _normalise_structured_payload(value: Any) -> Any:
    """Normalise harmless provider wrappers while leaving semantics to Pydantic."""
    if isinstance(value, dict):
        if len(value) == 1:
            key, nested = next(iter(value.items()))
            if key in _JSON_WRAPPERS:
                return _normalise_structured_payload(nested)
        # Some OpenAI-compatible adapters return a tool-call object as
        # {"type": "tool_call", "args": {...}}.  Only unwrap when args is a
        # mapping; strings/lists are left to schema validation.
        for key in ("args", "arguments", "tool_input"):
            nested = value.get(key)
            if isinstance(nested, dict) and (
                value.get("type") in {"tool_call", "function"}
                or len(value) <= 3
            ):
                return _normalise_structured_payload(nested)
        return value
    text = _response_content(value)
    if text:
        payload = _extract_json_object(text)
        if payload is not None:
            return _normalise_structured_payload(payload)
    return value


def _repair_instruction(repair_prompt: Any, schema: type[BaseModel], previous: Any, error: Exception) -> str:
    """Build a bounded, schema-exact repair request with safe failure context."""
    previous_text = _response_content(previous)
    if previous_text is None:
        previous_text = "<no parseable response object was returned>"
    previous_text = previous_text[:6000]
    schema_json = json.dumps(schema.model_json_schema(), sort_keys=True)
    detail = str(error)[:2000]
    return (
        f"{repair_prompt}\n\n"
        "REPAIR REQUIRED. Return JSON only: one object, no commentary, no markdown fences.\n"
        f"Exact JSON schema: {schema_json}\n"
        f"Previous malformed response (bounded): {previous_text}\n"
        f"Parse/validation error: {detail}\n"
        "Do not invent missing evidence or approval. Return every required field with valid types."
    )


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Any | None:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None


def invoke_structured_or_freetext(
    structured_llm: Any | None,
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
    *,
    schema: type[T] | None = None,
    repair_prompt: Any | None = None,
    fail_on_structured_error: bool = False,
    on_structured_result: Callable[[T], None] | None = None,
) -> str:
    """Run the structured call and render to markdown; fall back to free-text on any failure.

    ``prompt`` is whatever the underlying LLM accepts (a string for chat
    invocations, a list of message dicts for chat models that take that
    shape). The same value is forwarded to the free-text path so the
    fallback sees the same input the structured call did.
    """
    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            if result is None:
                # A thinking model can answer in plain text instead of calling
                # the tool, leaving the parser with nothing to return. Treat it
                # as a structured miss and fall back, with a clear reason.
                raise ValueError("structured output returned no parsed result")
            result = _normalise_structured_payload(result)
            if schema is not None:
                result = schema.model_validate(result)
            _record_telemetry("structured_primary_success")
            if on_structured_result is not None:
                on_structured_result(result)
            return render(result)
        except Exception as exc:
            _record_failure_details(exc)
            if repair_prompt is not None:
                repair_instruction = repair_prompt
                if schema is not None and hasattr(exc, "errors"):
                    failed_fields = []
                    for error in exc.errors():
                        location = ".".join(str(part) for part in error.get("loc", ()))
                        if location and location not in failed_fields:
                            failed_fields.append(location)
                    if failed_fields:
                        repair_instruction = (
                            f"{repair_prompt}\n\nThe previous object is invalid.\n"
                            "Validation failures:\n"
                            + "\n".join(f"- {field}" for field in failed_fields)
                            + "\nReturn the complete object again with ALL required fields."
                        )
                if schema is not None:
                    repair_instruction = _repair_instruction(
                        repair_instruction, schema, locals().get("result"), exc
                    )
                try:
                    repaired = structured_llm.invoke(repair_instruction)
                    if repaired is None:
                        raise ValueError("structured repair returned no parsed result")
                    repaired = _normalise_structured_payload(repaired)
                    if schema is not None:
                        repaired = schema.model_validate(repaired)
                    _record_telemetry("structured_repair_success")
                    if on_structured_result is not None:
                        on_structured_result(repaired)
                    return render(repaired)
                except Exception as structured_repair_error:
                    _record_failure_details(structured_repair_error)
                    repair_error = structured_repair_error
                    # Some providers return useful JSON as ordinary content even
                    # when their structured binding yields no parsed object. A
                    # single bounded text repair preserves fail-closed validation
                    # without accepting unvalidated prose.
                    if schema is not None and plain_llm is not structured_llm:
                        try:
                            repaired_text = plain_llm.invoke(repair_instruction)
                            repaired = schema.model_validate(
                                _normalise_structured_payload(repaired_text)
                            )
                            _record_telemetry("structured_extraction_success")
                            if on_structured_result is not None:
                                on_structured_result(repaired)
                            return render(repaired)
                        except Exception as text_repair_error:
                            repair_error = text_repair_error
                    if fail_on_structured_error:
                        _record_failure_category(repair_error)
                        _record_telemetry("structured_final_failure")
                        raise ValueError(
                            f"{agent_name}: structured output and one repair were invalid"
                        ) from repair_error
            logger.warning(
                "%s: structured-output invocation failed (%s); retrying once as free text",
                agent_name, exc,
            )

    response = plain_llm.invoke(prompt)
    return response.content
