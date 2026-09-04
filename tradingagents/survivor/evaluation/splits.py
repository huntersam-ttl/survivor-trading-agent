"""Chronological splits and walk-forward windows. Time-series order is NEVER shuffled."""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def _sort_key(item: T, timestamp_attr: str) -> str:
    return getattr(item, timestamp_attr)


def chronological_split(
    records: list[T],
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    timestamp_attr: str = "timestamp_utc",
) -> tuple[list[T], list[T], list[T]]:
    """Split chronologically: (train+dev, validation, out-of-sample test).

    Records are sorted by timestamp; no shuffling. Future observations can
    never appear in an earlier block.
    """
    ordered = sorted(records, key=lambda r: _sort_key(r, timestamp_attr))
    n = len(ordered)
    train_end = int(n * fractions[0])
    valid_end = train_end + int(n * fractions[1])
    return ordered[:train_end], ordered[train_end:valid_end], ordered[valid_end:]


def walk_forward(
    records: list[T],
    n_blocks: int = 4,
    timestamp_attr: str = "timestamp_utc",
) -> list[tuple[list[T], list[T]]]:
    """Deterministic walk-forward windows: each window trains on all data
    strictly BEFORE its evaluation block. No future leakage by construction."""
    ordered = sorted(records, key=lambda r: _sort_key(r, timestamp_attr))
    n = len(ordered)
    if n_blocks <= 0 or n < 2 * n_blocks:
        return []
    windows = []
    block_size = n // (n_blocks + 1)
    for i in range(1, n_blocks + 1):
        cut = block_size * i
        train = ordered[:cut]
        test = ordered[cut:cut + block_size]
        windows.append((train, test))
    return windows
