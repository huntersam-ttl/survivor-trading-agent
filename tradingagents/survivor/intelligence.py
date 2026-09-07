"""Bounded Jarvis World Intelligence digest for SURVIVOR (fail-open).

Jarvis is an *optional enrichment* for the SURVIVOR research context. It is
strictly read-only and never touches execution, risk, budgets, model routing or
autonomy semantics. If Jarvis is offline, disabled, unconfigured, times out,
returns malformed payloads, or answers 401/403, we return ``None`` and the
existing SURVIVOR research/decision flow runs unchanged.

Three hard guard rails:
* Context control — only the compact whitelisted fields ever reach the research
  context (no raw event history, full metadata, article bodies, or God's Eye
  payloads), and at most ``MAX_ANALYSES`` (10) analyses are surfaced.
* Fail-open — every failure path returns ``None`` instead of raising, so a
  digest problem can never crash or narrow a paper cycle.
* Bounded retries — at most one retry on timeout / connection- / 5xx failures,
  and never a retry on 4xx (including 401/403), so there is no retry storm.

There is **no live-trade path** here: this module imports only stdlib + the
``requests`` HTTP layer and produces a text block for the research context.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

_BOOL_TRUE = ("true", "1", "yes", "on")

# Hard upper bound on how many analyses we ever carry into the research context.
MAX_ANALYSES = 10

# The ONLY fields allowed to leave this module. Context control is enforced at
# parse and render time: anything else in the upstream payload is dropped.
_DIGEST_FIELDS = (
    "headline",
    "summary",
    "significance",
    "confidence",
    "affectedCountries",
    "affectedSectors",
    "affectedCommodities",
    "affectedSymbols",
    "impactDirection",
    "impactHorizon",
)

# Summary text is trimmed so the injected digest stays a *bounded* block.
_SUMMARY_MAX_CHARS = 300


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _BOOL_TRUE


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _parse_iso(ts: Any):
    """Best-effort ISO-8601 UTC parsing (also accepts a trailing ``Z``)."""
    try:
        parsed = datetime.fromisoformat((ts or "").replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def fingerprint_analyses(analyses: list[dict]) -> str:
    """Stable fingerprint over the *compact* digest fields only.

    Timestamps / digest age are intentionally excluded so identical intelligence
    across cycles fingerprints to the same value (enabling reuse / cache hits).
    """
    compact = [{k: a.get(k, None) for k in _DIGEST_FIELDS} for a in analyses]
    canonical = json.dumps(compact, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


# In-process track of recently-used digest fingerprints so a repeated digest in
# the same process reports a cache hit (observability) and is never re-injected.
_seen_fingerprints: set[str] = set()


def remember_digest_fingerprint(fingerprint: str) -> bool:
    """Record a digest fingerprint; returns True when it was already seen."""
    hit = fingerprint in _seen_fingerprints
    _seen_fingerprints.add(fingerprint)
    return hit


def research_state_hash(snapshot_hash: str, jarvis_fingerprint: str | None) -> str:
    """Combine the market snapshot hash with the Jarvis digest fingerprint.

    Used by the research cooldown so a candidate is only considered "already
    analyzed" when BOTH the decision-time snapshot AND the Jarvis digest are
    unchanged. That guarantees Jarvis returning unchanged intelligence never
    produces an extra LLM research call (cache reuse).

    When there is no Jarvis digest (``jarvis_fingerprint`` is falsy — disabled,
    unconfigured, or failed-open) this returns ``snapshot_hash`` unchanged so
    the existing cache-key format and behavior are preserved exactly.
    """
    if not jarvis_fingerprint:
        return snapshot_hash
    payload = json.dumps(
        {"snapshot": snapshot_hash, "jarvis": jarvis_fingerprint},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _render_analysis(analysis: dict) -> str:
    """Whitelist-render one compact analysis. No other fields are ever shown."""
    parts: list[str] = []
    headline = str(analysis.get("headline") or "").strip()
    if headline:
        parts.append(f"headline: {headline}")
    summary = str(analysis.get("summary") or "").strip()
    if summary:
        parts.append(f"summary: {summary[:_SUMMARY_MAX_CHARS]}")
    for key in (
        "significance",
        "confidence",
        "impactDirection",
        "impactHorizon",
        "affectedCountries",
        "affectedSectors",
        "affectedCommodities",
        "affectedSymbols",
    ):
        value = analysis.get(key)
        if value in (None, "", (), []):
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        parts.append(f"{key}: {value}")
    return "\n".join(parts)


@dataclass
class JarvisDigest:
    """A validated, bounded Jarvis World Intelligence digest."""

    analyses: list[dict] = field(default_factory=list)
    generated_at: datetime | None = None

    @property
    def fingerprint(self) -> str:
        return fingerprint_analyses(self.analyses)

    def age_seconds(self, now: datetime | None = None) -> int | None:
        if self.generated_at is None:
            return None
        ref = now or datetime.now(timezone.utc)
        return max(0, int((ref - self.generated_at).total_seconds()))

    def compact_context(self) -> str:
        """Render ONLY the whitelisted compact fields as a bounded text block."""
        if not self.analyses:
            return ""
        lines = ["JARVIS WORLD INTELLIGENCE DIGEST (bounded, compact)", ""]
        for idx, analysis in enumerate(self.analyses, start=1):
            rendered = _render_analysis(analysis)
            if rendered:
                lines.append(f"[{idx}] {rendered}")
            lines.append("")
        return "\n".join(lines).strip()


class JarvisIntelligenceClient:
    """Small single-purpose GET client for Jarvis World Intelligence.

    Endpoint: ``GET {base}/api/intelligence/world``.

    Environment convention (read at construction, no hard-coded secrets),
    consistent with the existing SURVIVOR ``os.environ`` config pattern:
    * ``JARVIS_BASE_URL``                     — e.g. ``http://localhost:8000``
    * ``JARVIS_INTELLIGENCE_ENABLED``          — toggle (default true)
    * ``JARVIS_INTELLIGENCE_TIMEOUT_SECONDS``  — request timeout (default 5)
    * ``JARVIS_INTELLIGENCE_LIMIT``            — analyses limit, clamped to 10

    There is no God's Eye dependency: this client talks to Jarvis directly.
    """

    def __init__(
        self,
        base_url: str | None = None,
        enabled: bool | None = None,
        timeout_seconds: int | None = None,
        limit: int | None = None,
        session: Any | None = None,
    ):
        self.base_url = (base_url or os.environ.get("JARVIS_BASE_URL", "")).strip().rstrip("/")
        self.enabled = enabled if enabled is not None else _env_bool("JARVIS_INTELLIGENCE_ENABLED", True)
        raw_timeout = os.environ.get("JARVIS_INTELLIGENCE_TIMEOUT_SECONDS", "5")
        try:
            self.timeout_seconds = timeout_seconds if timeout_seconds is not None else int(raw_timeout)
        except (TypeError, ValueError):
            self.timeout_seconds = 5
        if self.timeout_seconds < 0:
            self.timeout_seconds = 5
        raw_limit = os.environ.get("JARVIS_INTELLIGENCE_LIMIT", "10")
        try:
            limit = int(limit) if limit is not None else int(raw_limit)
        except (TypeError, ValueError):
            limit = MAX_ANALYSES
        self.limit = _clamp(limit, 1, MAX_ANALYSES)
        self.session = session or requests.Session()

    @property
    def unconfigured(self) -> bool:
        return not self.base_url

    def _url(self) -> str:
        return f"{self.base_url}/api/intelligence/world"

    def get_digest(
        self,
        symbols: Any = None,
        since: str | None = None,
        min_significance: float | None = None,
        limit: int | None = None,
    ) -> JarvisDigest | None:
        """Fetch and validate a bounded Jarvis digest.

        Fail-open: returns ``None`` on disabled / unconfigured / timeout /
        4xx (incl. 401/403) / 5xx / malformed payload. At most one retry on
        timeout- / connection- / 5xx failures; never on 4xx (no retry storm).
        """
        if not self.enabled:
            logger.info("Jarvis world intelligence disabled")
            return None
        if self.unconfigured:
            logger.info("Jarvis world intelligence unconfigured (no JARVIS_BASE_URL)")
            return None
        limit = _clamp(int(limit) if limit else self.limit, 1, MAX_ANALYSES)
        params: dict[str, Any] = {"limit": limit}
        if symbols:
            params["symbols"] = ",".join(str(_s) for _s in list(symbols)[:limit])
        if since:
            params["since"] = since
        if min_significance is not None:
            params["min_significance"] = min_significance

        retried = False
        while True:
            try:
                response = self.session.get(self._url(), params=params, timeout=self.timeout_seconds)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                if not retried:
                    retried = True
                    logger.warning("Jarvis request failed (%s); retrying once", exc)
                    continue
                logger.warning("Jarvis unreachable: %s; continuing without digest", exc)
                return None
            if response.status_code >= 500:
                if not retried:
                    retried = True
                    logger.warning("Jarvis returned %s; retrying once", response.status_code)
                    continue
                logger.warning("Jarvis returned %s; continuing without digest", response.status_code)
                return None
            if response.status_code >= 400:
                # 4xx (incl. 401/403): no retry.
                logger.warning("Jarvis returned %s; continuing without digest", response.status_code)
                return None
            try:
                payload = response.json()
            except ValueError:
                logger.warning("Jarvis returned malformed JSON; continuing without digest")
                return None
            return self._parse(payload)

    def _parse(self, payload: Any) -> JarvisDigest | None:
        if not isinstance(payload, dict):
            logger.warning("Jarvis digest: malformed payload (not an object)")
            return None
        raw = payload.get("analyses")
        if not isinstance(raw, list) or not raw:
            logger.warning("Jarvis digest: missing 'analyses' list; continuing without digest")
            return None
        analyses: list[dict] = []
        for item in raw[:MAX_ANALYSES]:
            parsed = self._parse_analysis(item)
            if parsed is not None:
                analyses.append(parsed)
            if len(analyses) >= MAX_ANALYSES:
                break
        if not analyses:
            logger.warning("Jarvis digest: no valid analyses; continuing without digest")
            return None
        generated = _parse_iso(payload.get("generated_at") or payload.get("timestamp"))
        return JarvisDigest(analyses=analyses, generated_at=generated)

    @staticmethod
    def _parse_analysis(item: Any) -> dict | None:
        if not isinstance(item, dict):
            return None
        headline = item.get("headline")
        if not isinstance(headline, str) or not headline.strip():
            return None
        # Whitelist only the compact fields (context control).
        return {k: item.get(k) for k in _DIGEST_FIELDS if k in item}
