"""Phase 4 evaluation types. Statistics use float; money stays integer pence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

STRATEGY_VERSION = "survivor-v1.0"


class SampleLabel(str, Enum):
    """Sample-size confidence label. NEVER claims profitability by itself."""

    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    VERY_LOW_CONFIDENCE = "VERY_LOW_CONFIDENCE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MODERATE_EVIDENCE = "MODERATE_EVIDENCE"
    STRONGER_SAMPLE = "STRONGER_SAMPLE"


class Warning_(str, Enum):
    PROFIT_CONCENTRATED = "PROFIT_CONCENTRATED"
    SNAPSHOT_INTEGRITY_FAILURE = "SNAPSHOT_INTEGRITY_FAILURE"
    CONFIG_HASH_MISMATCH = "CONFIG_HASH_MISMATCH"
    STRATEGY_VERSION_MISMATCH = "STRATEGY_VERSION_MISMATCH"
    OUT_OF_SAMPLE_NEGATIVE = "OUT_OF_SAMPLE_NEGATIVE"
    MARKET_BASELINE_OUTPERFORMS = "MARKET_BASELINE_OUTPERFORMS"
    AI_COST_EXCEEDS_EDGE = "AI_COST_EXCEEDS_EDGE"
    EDGE_DECAY_RISK = "EDGE_DECAY_RISK"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"


class GateState(str, Enum):
    """Evaluation gate outcome. READY_FOR_LIVE deliberately does not exist."""

    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    FAIL = "FAIL"
    CONTINUE_PAPER = "CONTINUE_PAPER"
    PROMISING_BUT_UNPROVEN = "PROMISING_BUT_UNPROVEN"


@dataclass(frozen=True)
class SampleThresholds:
    """Configurable, conservative sample-size thresholds."""

    insufficient: int = 30
    very_low: int = 100
    low: int = 300
    moderate: int = 1000


def sample_label(n: int, thresholds: SampleThresholds | None = None) -> SampleLabel:
    t = thresholds or SampleThresholds()
    if n < t.insufficient:
        return SampleLabel.INSUFFICIENT_SAMPLE
    if n < t.very_low:
        return SampleLabel.VERY_LOW_CONFIDENCE
    if n < t.low:
        return SampleLabel.LOW_CONFIDENCE
    if n < t.moderate:
        return SampleLabel.MODERATE_EVIDENCE
    return SampleLabel.STRONGER_SAMPLE


@dataclass(frozen=True)
class PredictionRecord:
    """One probability prediction with (optionally) its resolved outcome.

    Immutable once written. `outcome` is None until the market resolves:
    1.0 = YES, 0.0 = NO.
    """

    cycle_id: str
    run_id: str
    proposal_id: str
    market_id: str
    timestamp_utc: str
    strategy_version: str
    config_hash: str
    category: str
    predicted_probability: float          # SURVIVOR estimate in [0, 1]
    market_probability: float             # market price at decision time
    gross_edge_bps: int
    net_edge_bps: int
    ai_cost_pence: int = 0
    outcome: float | None = None          # None until resolved
    resolution_timestamp_utc: str | None = None
    snapshot_data_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("predicted_probability", "market_probability"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.outcome is not None and self.outcome not in (0.0, 1.0):
            raise ValueError("outcome must be None, 0.0 or 1.0")


@dataclass(frozen=True)
class TradeOutcome:
    """One resolved paper trade with full cost attribution."""

    cycle_id: str
    run_id: str
    proposal_id: str
    market_id: str
    timestamp_utc: str
    strategy_version: str
    config_hash: str
    category: str
    quantity: int
    entry_price_pence: int
    fees_pence: int
    slippage_pence: int
    gross_pnl_pence: int                  # realized P/L before AI cost
    ai_cost_pence: int
    realized_return_bps: int              # realized return vs entry (bps)
    predicted_net_edge_bps: int
    snapshot_data_hash: str = ""
    decision_latency_ms: int = 0
    outcome: float | None = None          # resolution outcome if binary


@dataclass
class CalibrationBin:
    lower_bps: int
    upper_bps: int
    count: int
    mean_predicted: float
    outcome_frequency: float
    calibration_error: float              # |mean_predicted - outcome_frequency|
