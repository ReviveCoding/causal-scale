from __future__ import annotations

import numpy as np


def rct_ipw_incremental_scores(
    outcome: np.ndarray, treatment: np.ndarray, policy: np.ndarray, propensity: float
) -> np.ndarray:
    y, t, pi = (np.asarray(value, dtype=float) for value in (outcome, treatment, policy))
    if (
        len({len(y), len(t), len(pi)}) != 1
        or not np.isin(t, [0, 1]).all()
        or not np.isin(pi, [0, 1]).all()
    ):
        raise ValueError("RCT-IPW inputs must be aligned and binary")
    if not 0 < propensity < 1:
        raise ValueError("propensity must be in (0,1)")
    return pi * (t * y / propensity - (1 - t) * y / (1 - propensity))


def rct_ipw_policy_value(
    outcome: np.ndarray, treatment: np.ndarray, policy: np.ndarray, propensity: float
) -> float:
    return float(rct_ipw_incremental_scores(outcome, treatment, policy, propensity).mean())
