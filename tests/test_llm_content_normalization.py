"""Fail-closed handling for OpenAI-compatible provider message content."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from tradingagents.llm_clients.base_client import (
    InvalidModelResponseError,
    normalize_content,
)


@pytest.mark.unit
def test_normalizes_valid_text_content_blocks():
    message = AIMessage(content=[
        {"type": "reasoning", "summary": "private reasoning is ignored"},
        {"type": "text", "text": "Decision summary"},
    ])

    assert normalize_content(message).content == "Decision summary"


@pytest.mark.unit
@pytest.mark.parametrize("message", [
    AIMessage(content=""),
    AIMessage(content="  \n "),
    AIMessage(content=[{"type": "reasoning", "text": ""}]),
    AIMessage(content="", additional_kwargs={"reasoning_content": "hidden reasoning only"}),
])
def test_empty_or_reasoning_only_responses_fail_closed(message):
    with pytest.raises(InvalidModelResponseError, match="no usable text or tool call"):
        normalize_content(message)


@pytest.mark.unit
def test_tool_call_only_response_is_preserved_for_structured_parser():
    message = AIMessage(
        content="",
        tool_calls=[{
            "name": "trade_decision",
            "args": {"action": "hold", "reason": "insufficient evidence"},
            "id": "call-1",
            "type": "tool_call",
        }],
    )

    normalized = normalize_content(message)
    assert normalized.content == ""
    assert normalized.tool_calls[0]["name"] == "trade_decision"
    assert normalized.tool_calls[0]["args"]["action"] == "hold"


@pytest.mark.unit
@pytest.mark.parametrize("response", [None, {}, SimpleNamespace(), SimpleNamespace(content=object())])
def test_malformed_response_shapes_fail_closed(response):
    with pytest.raises(InvalidModelResponseError):
        normalize_content(response)


@pytest.mark.unit
def test_openai_compatible_response_retains_completion_metadata():
    """A length-limited reasoning-only choice remains inspectable, then fails."""
    from tradingagents.llm_clients.openai_client import NormalizedChatOpenAI

    client = NormalizedChatOpenAI(model="deepseek/deepseek-v4-flash-0731:nitro")
    result = client._create_chat_result({
        "model": "deepseek/deepseek-v4-flash-0731:nitro",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning": "redacted fixture reasoning",
            },
            "finish_reason": "length",
        }],
        "usage": {
            "prompt_tokens": 88,
            "completion_tokens": 12,
            "total_tokens": 100,
            "completion_tokens_details": {"reasoning_tokens": 12},
        },
    })
    message = result.generations[0].message

    assert result.generations[0].generation_info["finish_reason"] == "length"
    assert message.usage_metadata["output_tokens"] == 12
    assert message.usage_metadata["output_token_details"]["reasoning"] == 12
    with pytest.raises(InvalidModelResponseError):
        normalize_content(message)
