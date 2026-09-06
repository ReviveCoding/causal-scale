from __future__ import annotations

import numpy as np


def cumulative_uplift(
    score: np.ndarray, outcome: np.ndarray, treatment: np.ndarray, propensity: float
) -> tuple[np.ndarray, np.ndarray]:
    s, y, t = map(np.asarray, (score, outcome, treatment))
    order = np.argsort(-s, kind="stable")
    contributions = t[order] * y[order] / propensity - (1 - t[order]) * y[order] / (1 - propensity)
    return np.arange(1, len(s) + 1) / len(s), np.cumsum(contributions) / len(s)


def auuc(score: np.ndarray, outcome: np.ndarray, treatment: np.ndarray, propensity: float) -> float:
    fraction, curve = cumulative_uplift(score, outcome, treatment, propensity)
    return float(np.trapezoid(curve, fraction))


def qini(score: np.ndarray, outcome: np.ndarray, treatment: np.ndarray, propensity: float) -> float:
    fraction, curve = cumulative_uplift(score, outcome, treatment, propensity)
    baseline = fraction * curve[-1]
    return float(np.trapezoid(curve - baseline, fraction))
