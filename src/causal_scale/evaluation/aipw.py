from __future__ import annotations

import numpy as np


def dr_effect_score(
    outcome: np.ndarray,
    treatment: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    propensity: float | np.ndarray,
) -> np.ndarray:
    """Compute the doubly robust treatment effect score for each observation."""
    observed_outcome, observed_treatment, control_mean, treated_mean = (
        np.asarray(value, dtype=float) for value in (outcome, treatment, mu0, mu1)
    )
    if (
        any(
            value.ndim != 1
            for value in (
                observed_outcome,
                observed_treatment,
                control_mean,
                treated_mean,
            )
        )
        or len(
            {
                len(value)
                for value in (
                    observed_outcome,
                    observed_treatment,
                    control_mean,
                    treated_mean,
                )
            }
        )
        != 1
    ):
        raise ValueError("AIPW inputs must be aligned vectors")
    treatment_probability = np.asarray(propensity, dtype=float)
    if (
        treatment_probability.ndim > 1
        or not np.isfinite(treatment_probability).all()
        or (treatment_probability <= 0).any()
        or (treatment_probability >= 1).any()
    ):
        raise ValueError("propensity must be in (0,1)")
    treatment_probability = np.clip(treatment_probability, 0.02, 0.98)
    return (
        treated_mean
        - control_mean
        + observed_treatment * (observed_outcome - treated_mean) / treatment_probability
        - (1 - observed_treatment)
        * (observed_outcome - control_mean)
        / (1 - treatment_probability)
    )


def aipw_policy_scores(
    outcome: np.ndarray,
    treatment: np.ndarray,
    policy: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    propensity: float | np.ndarray,
) -> np.ndarray:
    """Compute per-observation AIPW policy gains relative to treating nobody."""
    observed_outcome, observed_treatment, policy_action, control_mean, treated_mean = (
        np.asarray(value, dtype=float)
        for value in (outcome, treatment, policy, mu0, mu1)
    )
    if (
        not np.isin(observed_treatment, [0, 1]).all()
        or not np.isin(policy_action, [0, 1]).all()
    ):
        raise ValueError("treatment and policy must be binary")
    clipped_propensity = np.clip(np.asarray(propensity, dtype=float), 0.02, 0.98)
    observed_action_mean = np.where(observed_treatment == 1, treated_mean, control_mean)
    observed_action_probability = np.where(
        observed_treatment == 1, clipped_propensity, 1 - clipped_propensity
    )

    def value(action: np.ndarray) -> np.ndarray:
        target_action_mean = np.where(action == 1, treated_mean, control_mean)
        return (
            target_action_mean
            + (observed_treatment == action)
            * (observed_outcome - observed_action_mean)
            / observed_action_probability
        )

    return value(policy_action) - value(np.zeros_like(policy_action))


def aipw_policy_value(*args: np.ndarray | float) -> float:
    return float(np.mean(aipw_policy_scores(*args)))  # type: ignore[arg-type]
