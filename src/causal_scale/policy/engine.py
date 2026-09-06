from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicyResult:
    membership: np.ndarray
    rank: np.ndarray
    score: np.ndarray
    eligible: np.ndarray


def capacity_policy(
    score: np.ndarray,
    capacity: float,
    *,
    tie_key: np.ndarray | None = None,
    positive_only: bool = True,
    historical_position: np.ndarray | None = None,
) -> PolicyResult:
    values = np.asarray(score, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("score must be a nonempty finite vector")
    if not np.isfinite(capacity) or not 0 < capacity <= 1:
        raise ValueError("capacity must be in (0, 1]")
    if historical_position is not None:
        key = np.asarray(historical_position)
    elif tie_key is None:
        key = np.arange(len(values), dtype=np.int64)
    else:
        key = np.asarray(tie_key)
    if key.ndim != 1 or len(key) != len(values) or len(np.unique(key)) != len(key):
        raise ValueError("tie key must be aligned and unique")
    # Preserve numeric ordering for numeric row IDs. Historical ORBIT converted
    # explicitly supplied keys to strings; the continuity path instead supplies
    # numeric historical_position, which reproduces its default positional ties.
    order = np.lexsort((key, -values))
    rank = np.empty(len(values), dtype=np.int64)
    rank[order] = np.arange(1, len(values) + 1)
    eligible = values > 0 if positive_only else np.ones(len(values), dtype=bool)
    membership = (eligible & (rank <= int(np.floor(capacity * len(values))))).astype(np.int8)
    return PolicyResult(membership, rank, values, eligible)


def policy_agreement(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape:
        raise ValueError("policies must be aligned")
    return float(np.mean(a == b))
