from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def causal_calibration(
    prediction: np.ndarray, pseudo_outcome: np.ndarray, bins: int = 10
) -> tuple[pd.DataFrame, float]:
    effect, gamma = np.asarray(prediction, dtype=float), np.asarray(pseudo_outcome, dtype=float)
    if effect.shape != gamma.shape or effect.ndim != 1 or len(effect) == 0:
        raise ValueError("calibration vectors must be aligned")
    codes = pd.qcut(effect, q=min(bins, len(effect)), duplicates="drop").codes
    groups = []
    squared = 0.0
    for code in range(int(codes.max()) + 1):
        mask = codes == code
        predicted, observed = float(effect[mask].mean()), float(gamma[mask].mean())
        weight = float(mask.mean())
        squared += weight * (predicted - observed) ** 2
        groups.append(
            {
                "bin": code,
                "rows": int(mask.sum()),
                "weight": weight,
                "predicted_effect": predicted,
                "dr_observed_effect": observed,
                "gap": predicted - observed,
            }
        )
    return pd.DataFrame(groups), float(np.sqrt(squared))


def calibration_comparison(
    raw: np.ndarray,
    calibrated: np.ndarray,
    gamma: np.ndarray,
    raw_policy: np.ndarray,
    calibrated_policy: np.ndarray,
) -> dict[str, float | int | None]:
    _, raw_rmse = causal_calibration(raw, gamma)
    _, cal_rmse = causal_calibration(calibrated, gamma)
    correlation = float(spearmanr(raw, calibrated).statistic)
    return {
        "raw_calibration_rmse": raw_rmse,
        "calibrated_calibration_rmse": cal_rmse,
        "rank_correlation": correlation if np.isfinite(correlation) else None,
        "policy_membership_agreement": float(
            np.mean(np.asarray(raw_policy) == np.asarray(calibrated_policy))
        ),
        "positive_eligibility_changes": int(
            np.sum((np.asarray(raw) > 0) != (np.asarray(calibrated) > 0))
        ),
    }
