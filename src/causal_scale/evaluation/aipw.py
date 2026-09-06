from __future__ import annotations

import numpy as np


def dr_effect_score(
    outcome: np.ndarray,
    treatment: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    propensity: float | np.ndarray,
) -> np.ndarray:
    y, t, m0, m1 = (np.asarray(value, dtype=float) for value in (outcome, treatment, mu0, mu1))
    if (
        any(value.ndim != 1 for value in (y, t, m0, m1))
        or len({len(value) for value in (y, t, m0, m1)}) != 1
    ):
        raise ValueError("AIPW inputs must be aligned vectors")
    p = np.asarray(propensity, dtype=float)
    if p.ndim > 1 or not np.isfinite(p).all() or (p <= 0).any() or (p >= 1).any():
        raise ValueError("propensity must be in (0,1)")
    p = np.clip(p, 0.02, 0.98)
    return m1 - m0 + t * (y - m1) / p - (1 - t) * (y - m0) / (1 - p)


def aipw_policy_scores(
    outcome: np.ndarray,
    treatment: np.ndarray,
    policy: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    propensity: float | np.ndarray,
) -> np.ndarray:
    y, t, pi, m0, m1 = (
        np.asarray(value, dtype=float) for value in (outcome, treatment, policy, mu0, mu1)
    )
    if not np.isin(t, [0, 1]).all() or not np.isin(pi, [0, 1]).all():
        raise ValueError("treatment and policy must be binary")
    p = np.clip(np.asarray(propensity, dtype=float), 0.02, 0.98)
    observed_mu = np.where(t == 1, m1, m0)
    observed_probability = np.where(t == 1, p, 1 - p)

    def value(action: np.ndarray) -> np.ndarray:
        target_mu = np.where(action == 1, m1, m0)
        return target_mu + (t == action) * (y - observed_mu) / observed_probability

    return value(pi) - value(np.zeros_like(pi))


def aipw_policy_value(*args: np.ndarray | float) -> float:
    return float(np.mean(aipw_policy_scores(*args)))  # type: ignore[arg-type]
