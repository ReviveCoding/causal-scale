from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from causal_scale.baselines.random_policy import stable_random_scores
from causal_scale.evaluation.aipw import aipw_policy_scores
from causal_scale.evaluation.ipw import rct_ipw_incremental_scores
from causal_scale.evidence.manifest import atomic_json, read_json
from causal_scale.models.historical_adapter import HistoricalPortfolioAdapter
from causal_scale.policy.engine import capacity_policy


def cluster_interval(values: np.ndarray, codes: np.ndarray, cluster_count: int) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    counts = np.bincount(codes, minlength=cluster_count).astype(float)
    sums = np.bincount(codes, weights=x, minlength=cluster_count)
    estimate = float(x.mean())
    residual_sums = sums - estimate * counts
    standard_error = float(
        np.sqrt(cluster_count / (cluster_count - 1) * np.dot(residual_sums, residual_sums)) / len(x)
    )
    return {
        "lower": estimate - 1.959963984540054 * standard_error,
        "upper": estimate + 1.959963984540054 * standard_error,
        "standard_error": standard_error,
    }


def main() -> None:
    root = Path("evidence/retrospective/model_benchmark")
    data = pd.read_parquet(".local-data/marts/historical_locked_2m.parquet")
    scores = pd.read_parquet(root / "scores.parquet")
    if not np.array_equal(data.source_row_id.to_numpy(), scores.source_row_id.to_numpy()):
        raise ValueError("retrospective score/data row ordering mismatch")
    adapter = HistoricalPortfolioAdapter()
    mu0, mu1 = adapter.predict_outcomes(data)
    nuisance = joblib.load(".local-models/frozen/independent_aipw_nuisance.joblib")
    independent_mu0, independent_mu1 = nuisance.predict_outcomes(data)
    reference_source = str(Path(".reference/ORBIT-Causal/src").resolve())
    if reference_source not in sys.path:
        sys.path.insert(0, reference_source)
    from orbit_causal.data import profile_ids

    clusters = profile_ids(data, adapter.portfolio["C1"].data_config).to_numpy()
    codes, unique = pd.factorize(clusters, sort=True)
    table = pd.read_csv(root / "model_benchmark.csv")
    y, t, p = data.visit.to_numpy(), data.treatment.to_numpy(), adapter.propensity
    model_values: dict[tuple[str, str], np.ndarray] = {}
    for model_id in scores.columns.drop("source_row_id"):
        score = scores[model_id].to_numpy()
        is_response = model_id.startswith("RESPONSE_") or model_id == "RANDOM"
        if model_id == "CONSTANT_ATE" and np.all(score > 0):
            score = stable_random_scores(data.source_row_id.to_numpy(), 20260717)
            is_response = True
        policy = capacity_policy(
            score,
            0.2,
            historical_position=np.arange(len(data)),
            positive_only=not is_response,
        ).membership
        for evaluator, values in {
            "HISTORICAL_AIPW": aipw_policy_scores(y, t, policy, mu0, mu1, p),
            "RCT_IPW_HT": rct_ipw_incremental_scores(y, t, policy, p),
            "INDEPENDENT_AIPW": aipw_policy_scores(
                y, t, policy, independent_mu0, independent_mu1, p
            ),
        }.items():
            model_values[(model_id, evaluator)] = values
            interval = cluster_interval(values, codes, len(unique))
            mask = (
                (table.model == model_id)
                & (table.evaluator == evaluator)
                & np.isclose(table.capacity, 0.2)
            )
            table.loc[mask, "CI_lower"] = interval["lower"]
            table.loc[mask, "CI_upper"] = interval["upper"]
    for (model_id, evaluator), values in model_values.items():
        difference = values - model_values[("HIST_C3", evaluator)]
        interval = cluster_interval(difference, codes, len(unique))
        mask = (
            (table.model == model_id)
            & (table.evaluator == evaluator)
            & np.isclose(table.capacity, 0.2)
        )
        table.loc[mask, "delta_CI_lower"] = interval["lower"]
        table.loc[mask, "delta_CI_upper"] = interval["upper"]
    historical = read_json("evidence/historical_reference/historical_metrics.json")
    exact = {
        "HIST_E7": historical["candidate_value_two_sided_95pct"],
        "HIST_C3": historical["baseline_value_two_sided_95pct"],
    }
    for model_id, interval in exact.items():
        mask = (
            (table.model == model_id)
            & (table.evaluator == "HISTORICAL_AIPW")
            & np.isclose(table.capacity, 0.2)
        )
        table.loc[mask, "CI_lower"] = interval["lower"]
        table.loc[mask, "CI_upper"] = interval["upper"]
    mask = (
        (table.model == "HIST_E7")
        & (table.evaluator == "HISTORICAL_AIPW")
        & np.isclose(table.capacity, 0.2)
    )
    difference_interval = historical["value_difference_two_sided_95pct"]
    table.loc[mask, "delta_CI_lower"] = difference_interval["lower"]
    table.loc[mask, "delta_CI_upper"] = difference_interval["upper"]
    table.to_csv(root / "model_benchmark.csv", index=False)
    metrics = read_json(root / "metrics.json")
    metrics["cluster_inference"] = {
        "status": "PASS",
        "unit": "exact_pretreatment_profile",
        "unique_clusters": len(unique),
        "primary_capacity": 0.2,
        "method": "cluster_robust_normal; exact historical 4000-replicate bootstrap restored for E7/C3",
    }
    atomic_json(root / "metrics.json", metrics)
    print(json.dumps(metrics["cluster_inference"], indent=2))


if __name__ == "__main__":
    main()
