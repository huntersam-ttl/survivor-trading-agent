"""Explicit model pricing table and cost calculation helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, InvalidOperation

from tradingagents.survivor.types import UnknownPriceError


@dataclass(frozen=True)
class ModelPrice:
    """Pricing spec for a single model in native currency per 1,000,000 tokens."""

    provider: str
    model: str
    input_cost_per_1m: Decimal
    output_cost_per_1m: Decimal
    native_currency: str = "USD"
    cached_input_cost_per_1m: Decimal | None = None


# Production price catalog for supported model IDs (costs in USD per 1M tokens)
MODEL_PRICING: dict[tuple[str, str], ModelPrice] = {
    # OpenAI Models
    ("openai", "gpt-5.6"): ModelPrice("openai", "gpt-5.6", Decimal("2.50"), Decimal("10.00")),
    ("openai", "gpt-5.6-luna"): ModelPrice("openai", "gpt-5.6-luna", Decimal("0.15"), Decimal("0.60")),
    ("openai", "gpt-4.1"): ModelPrice("openai", "gpt-4.1", Decimal("2.50"), Decimal("10.00")),
    ("openai", "gpt-4o"): ModelPrice("openai", "gpt-4o", Decimal("2.50"), Decimal("10.00")),
    ("openai", "gpt-4o-mini"): ModelPrice("openai", "gpt-4o-mini", Decimal("0.15"), Decimal("0.60")),
    ("openai", "o1"): ModelPrice("openai", "o1", Decimal("15.00"), Decimal("60.00")),
    ("openai", "o3-mini"): ModelPrice("openai", "o3-mini", Decimal("1.10"), Decimal("4.40")),

    # DeepSeek Models
    ("deepseek", "deepseek-chat"): ModelPrice("deepseek", "deepseek-chat", Decimal("0.14"), Decimal("0.28")),
    ("deepseek", "deepseek-reasoner"): ModelPrice("deepseek", "deepseek-reasoner", Decimal("0.55"), Decimal("2.19")),
    ("deepseek", "deepseek-v3"): ModelPrice("deepseek", "deepseek-v3", Decimal("0.14"), Decimal("0.28")),
    ("deepseek", "deepseek-v4"): ModelPrice("deepseek", "deepseek-v4", Decimal("0.20"), Decimal("0.80")),

    # MiniMax Models
    ("minimax", "minimax-text-01"): ModelPrice("minimax", "minimax-text-01", Decimal("0.20"), Decimal("1.10")),
    ("minimax", "abab6.5s-chat"): ModelPrice("minimax", "abab6.5s-chat", Decimal("0.20"), Decimal("1.00")),
    ("minimax", "minimax-m2"): ModelPrice("minimax", "minimax-m2", Decimal("0.30"), Decimal("1.20")),
    ("minimax-cn", "minimax-text-01"): ModelPrice("minimax-cn", "minimax-text-01", Decimal("0.20"), Decimal("1.10")),

    # Google Gemini Models
    ("google", "gemini-2.5-flash"): ModelPrice("google", "gemini-2.5-flash", Decimal("0.075"), Decimal("0.30")),
    ("google", "gemini-2.5-pro"): ModelPrice("google", "gemini-2.5-pro", Decimal("1.25"), Decimal("5.00")),

    # Anthropic Claude Models
    ("anthropic", "claude-3-5-sonnet-20241022"): ModelPrice("anthropic", "claude-3-5-sonnet-20241022", Decimal("3.00"), Decimal("15.00")),
    ("anthropic", "claude-3-5-haiku-20241022"): ModelPrice("anthropic", "claude-3-5-haiku-20241022", Decimal("0.80"), Decimal("4.00")),
}


def get_model_price(provider: str, model: str) -> ModelPrice | None:
    """Look up pricing for (provider, model). Case-insensitive match."""
    key = (provider.lower().strip(), model.lower().strip())
    return MODEL_PRICING.get(key)


def register_model_price(price: ModelPrice) -> None:
    """Register explicit pricing into the catalog (e.g. from env configuration).

    Never derived from the model name — only from explicit configuration.
    """
    MODEL_PRICING[(price.provider.lower().strip(), price.model.lower().strip())] = price


def register_openrouter_env_pricing() -> None:
    """Register explicit pricing for the configured OpenRouter model, if provided.

    ``SURVIVOR_OPENROUTER_PRICING="<input_usd_per_1m>,<output_usd_per_1m>"``
    (e.g. ``"0.14,0.28"``). Without it the model stays UNPRICED and routing
    fails closed with ``UnknownPriceError`` — price is never assumed from the
    model name. A malformed value raises (fail loudly, never guess).
    """
    raw = os.environ.get("SURVIVOR_OPENROUTER_PRICING", "").strip()
    model = os.environ.get("SURVIVOR_OPENROUTER_MODEL", "").strip()
    if not raw or not model:
        return
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError(
            "Invalid SURVIVOR_OPENROUTER_PRICING: expected '<input_usd_per_1m>,"
            f"<output_usd_per_1m>', got {raw!r}"
        )
    try:
        in_price = Decimal(parts[0])
        out_price = Decimal(parts[1])
    except InvalidOperation as exc:
        raise ValueError(f"Invalid SURVIVOR_OPENROUTER_PRICING: {raw!r}") from exc
    if in_price <= 0 or out_price <= 0:
        raise ValueError(
            f"SURVIVOR_OPENROUTER_PRICING values must be positive, got {raw!r}"
        )
    register_model_price(ModelPrice("openrouter", model, in_price, out_price))


@dataclass(frozen=True)
class CostEstimate:
    """Calculated cost breakdown for an inference request."""

    native_cost_minor: int  # e.g. cents if USD
    native_currency: str
    gbp_cost_pence: int     # integer pence


def calculate_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    usd_gbp_rate: Decimal = Decimal("1.30"),
    custom_pricing: dict[tuple[str, str], ModelPrice] | None = None,
) -> CostEstimate:
    """Calculate exact cost in native minor units and GBP pence for input & output token counts.

    Fails closed with UnknownPriceError if pricing is unknown.
    Uses Decimal arithmetic and rounds UP (ceiling) to ensure non-zero cost estimation and prevent budget undercounting.
    """
    catalog = custom_pricing if custom_pricing is not None else MODEL_PRICING
    key = (provider.lower().strip(), model.lower().strip())
    price = catalog.get(key)

    if price is None:
        raise UnknownPriceError(
            f"Pricing for provider '{provider}' and model '{model}' is UNKNOWN. "
            f"Failing closed to prevent unbudgeted API expenditure."
        )

    # Standard calculations in Decimal per token
    in_dec = Decimal(input_tokens - cached_tokens) if input_tokens > cached_tokens else Decimal(0)
    out_dec = Decimal(output_tokens)
    cache_dec = Decimal(cached_tokens)

    cost_in = (in_dec * price.input_cost_per_1m) / Decimal("1000000")
    cost_out = (out_dec * price.output_cost_per_1m) / Decimal("1000000")

    cache_price = price.cached_input_cost_per_1m or price.input_cost_per_1m
    cost_cache = (cache_dec * cache_price) / Decimal("1000000")

    total_native_currency = cost_in + cost_out + cost_cache

    if price.native_currency.upper() == "USD":
        native_minor = int((total_native_currency * Decimal("100")).quantize(Decimal("1"), rounding=ROUND_CEILING))
        if usd_gbp_rate is None or usd_gbp_rate <= 0:
            raise UnknownPriceError("FX rate USD_GBP_RATE is unavailable or invalid.")
        gbp_pounds = total_native_currency / usd_gbp_rate
        gbp_pence = int((gbp_pounds * Decimal("100")).quantize(Decimal("1"), rounding=ROUND_CEILING))
    elif price.native_currency.upper() == "GBP":
        gbp_pence = int((total_native_currency * Decimal("100")).quantize(Decimal("1"), rounding=ROUND_CEILING))
        native_minor = gbp_pence
    else:
        # Generic native currency -> convert using 1:1 fallback or raise if unknown
        native_minor = int((total_native_currency * Decimal("100")).quantize(Decimal("1"), rounding=ROUND_CEILING))
        gbp_pounds = total_native_currency / usd_gbp_rate
        gbp_pence = int((gbp_pounds * Decimal("100")).quantize(Decimal("1"), rounding=ROUND_CEILING))

    return CostEstimate(
        native_cost_minor=native_minor,
        native_currency=price.native_currency.upper(),
        gbp_cost_pence=gbp_pence,
    )
