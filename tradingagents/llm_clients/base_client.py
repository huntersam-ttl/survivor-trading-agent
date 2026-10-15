import warnings
from abc import ABC, abstractmethod
from typing import Any


class InvalidModelResponseError(ValueError):
    """The provider returned no usable text or structured tool call."""


def normalize_content(response):
    """Normalize LLM response content to a plain string.

    Multiple providers (OpenAI Responses API, Google Gemini 3) return content
    as a list of typed blocks, e.g. [{'type': 'reasoning', ...}, {'type': 'text', 'text': '...'}].
    Downstream agents expect response.content to be a string. This extracts
    and joins the text blocks, discarding reasoning/metadata blocks.
    """
    if not hasattr(response, "content"):
        raise InvalidModelResponseError("Provider response has no message content.")

    content = response.content
    if isinstance(content, list):
        texts = [
            item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
            else item if isinstance(item, str) else ""
            for item in content
        ]
        response.content = "\n".join(t for t in texts if t)

    text = response.content
    if text is None:
        text = ""
    if not isinstance(text, str):
        raise InvalidModelResponseError("Provider response content has an unsupported type.")

    # Some reasoning models spend a small output budget entirely on hidden
    # reasoning and return an empty `content`. Never let that look like a
    # successful text completion. A tool-call-only AIMessage remains valid for
    # structured-output/tool-binding flows; their schema parser validates the
    # call before a domain decision is produced.
    tool_calls = getattr(response, "tool_calls", None)
    if not text.strip() and not tool_calls:
        raise InvalidModelResponseError("Provider response contained no usable text or tool call.")
    return response


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        self.model = model
        self.base_url = base_url
        self.kwargs = kwargs

    def get_provider_name(self) -> str:
        """Return the provider name used in warning messages."""
        provider = getattr(self, "provider", None)
        if provider:
            return str(provider)
        return self.__class__.__name__.removesuffix("Client").lower()

    def warn_if_unknown_model(self) -> None:
        """Warn when the model is outside the known list for the provider."""
        if self.validate_model():
            return

        warnings.warn(
            (
                f"Model '{self.model}' is not in the known model list for "
                f"provider '{self.get_provider_name()}'. Continuing anyway."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    @abstractmethod
    def get_llm(self) -> Any:
        """Return the configured LLM instance."""
        pass

    @abstractmethod
    def validate_model(self) -> bool:
        """Validate that the model is supported by this client."""
        pass
