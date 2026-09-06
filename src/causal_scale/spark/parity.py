from __future__ import annotations

import numpy as np
import pandas as pd


def parity_report(
    reference: pd.DataFrame,
    distributed: pd.DataFrame,
    *,
    score_tolerance: float = 1e-10,
    value_tolerance: float = 1e-12,
) -> dict[str, object]:
    required = {"source_row_id", "score", "policy"}
    if not required <= set(reference) or not required <= set(distributed):
        raise ValueError("parity frames are missing required columns")
    left = reference.sort_values("source_row_id", kind="mergesort").reset_index(drop=True)
    right = distributed.sort_values("source_row_id", kind="mergesort").reset_index(drop=True)
    exact_rows = len(left) == len(right) and np.array_equal(left.source_row_id, right.source_row_id)
    max_score_error = (
        float(np.max(np.abs(left.score - right.score))) if exact_rows else float("inf")
    )
    membership = (
        float(np.mean(left.policy.to_numpy() == right.policy.to_numpy())) if exact_rows else 0.0
    )
    value_error = abs(
        float(left.get("aipw_score", pd.Series([0])).mean())
        - float(right.get("aipw_score", pd.Series([0])).mean())
    )
    passed = (
        exact_rows
        and max_score_error <= score_tolerance
        and membership == 1.0
        and value_error <= value_tolerance
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "exact_row_set": exact_rows,
        "row_count": len(right),
        "score_max_abs_error": max_score_error,
        "policy_membership_agreement": membership,
        "aipw_abs_error": value_error,
        "score_tolerance": score_tolerance,
        "aipw_tolerance": value_tolerance,
    }
