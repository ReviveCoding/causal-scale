from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

from causal_scale.baselines.random_policy import stable_random_scores
from causal_scale.data.canonicalize import extract_prospective_outcomes_once
from causal_scale.data.prospective_guard import assert_contract_sealed, assert_feature_only
from causal_scale.evaluation.aipw import aipw_policy_scores, dr_effect_score
from causal_scale.evaluation.ipw import rct_ipw_incremental_scores
from causal_scale.evaluation.multiple_testing import holm
from causal_scale.evidence.hashing import row_ids_sha256, sha256_file
from causal_scale.evidence.manifest import atomic_json
from causal_scale.models.historical_adapter import HistoricalPortfolioAdapter
from causal_scale.policy.engine import capacity_policy

CAPACITIES = (0.05, 0.10, 0.20, 0.30, 0.50)


def cluster_normal_inference(values: np.ndarray, clusters: np.ndarray) -> dict[str, float]:
    """Cluster-robust normal interval for predeclared secondary comparisons."""
    x = np.asarray(values, dtype=float)
    codes, uniques = pd.factorize(clusters, sort=True)
    counts = np.bincount(codes, minlength=len(uniques)).astype(float)
    sums = np.bincount(codes, weights=x, minlength=len(uniques))
    estimate = float(x.mean())
    residual_sums = sums - estimate * counts
    standard_error = float(
        np.sqrt(len(uniques) / (len(uniques) - 1) * np.dot(residual_sums, residual_sums)) / len(x)
    )
    z = estimate / standard_error if standard_error > 0 else np.inf
    return {
        "point_estimate": estimate,
        "CI_lower": estimate - 1.959963984540054 * standard_error,
        "CI_upper": estimate + 1.959963984540054 * standard_error,
        "standard_error": standard_error,
        "p_value_two_sided": float(2 * norm.sf(abs(z))),
    }


def append_attempt(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def seal_features(features_path: Path, models_path: Path, output: Path) -> dict[str, object]:
    contract = Path("evidence/prospective/scientific_contract.json")
    digest = Path("evidence/prospective/SCIENTIFIC_CONTRACT_SHA256.txt")
    contract_hash = assert_contract_sealed(contract, digest)
    features = pd.read_parquet(features_path)
    assert_feature_only(features)
    historical = HistoricalPortfolioAdapter()
    calibrators = joblib.load(models_path / "common_calibrators.joblib")
    models: dict[str, object] = {name: historical for name in ("HIST_C1", "HIST_C3", "HIST_E7")}
    for path in models_path.glob("*.joblib"):
        if path.name == "common_calibrators.joblib":
            continue
        model = joblib.load(path)
        if getattr(model, "model_id", "") != "INDEPENDENT_AIPW_NUISANCE":
            models[model.model_id] = model
    sealed = pd.DataFrame({"source_row_id": features.source_row_id})
    for model_id, model in models.items():
        if model is historical:
            raw = historical.predict_effect(model_id, features)
            causal = True
        elif hasattr(model, "predict_score"):
            raw = model.predict_score(features)
            causal = False
        else:
            raw = model.predict_effect(features)
            causal = True
        sealed[f"score__{model_id}__raw"] = raw
        variants = {"raw": raw}
        if causal and model_id in calibrators:
            variants["common_cal"] = calibrators[model_id].transform(raw)
        for variant, score in variants.items():
            sealed[f"score__{model_id}__{variant}"] = score
            bins = (
                np.zeros(len(score), dtype=np.int8)
                if np.unique(score).size == 1
                else np.asarray(
                    pd.qcut(score, q=10, labels=False, duplicates="drop"), dtype=np.int8
                )
            )
            sealed[f"bin__{model_id}__{variant}"] = bins
            for capacity in CAPACITIES:
                if model_id == "CONSTANT_ATE" and float(raw[0]) > 0:
                    ranking = stable_random_scores(features.source_row_id.to_numpy(), 20260717)
                    membership = capacity_policy(
                        ranking,
                        capacity,
                        tie_key=features.source_row_id.to_numpy(),
                        positive_only=False,
                    ).membership
                else:
                    membership = capacity_policy(
                        score,
                        capacity,
                        tie_key=features.source_row_id.to_numpy(),
                        positive_only=causal,
                    ).membership
                sealed[f"policy__{model_id}__{variant}__{capacity:.2f}"] = membership
    mu0, mu1 = historical.predict_outcomes(features)
    sealed["nuisance__historical_mu0"], sealed["nuisance__historical_mu1"] = mu0, mu1
    nuisance = joblib.load(models_path / "independent_aipw_nuisance.joblib")
    independent_mu0, independent_mu1 = nuisance.predict_outcomes(features)
    sealed["nuisance__independent_mu0"], sealed["nuisance__independent_mu1"] = (
        independent_mu0,
        independent_mu1,
    )
    output.mkdir(parents=True, exist_ok=True)
    target = output / "feature_policy_seal.parquet"
    if target.exists():
        raise PermissionError("refusing to replace an existing prospective feature/policy seal")
    sealed.to_parquet(target, index=False, compression="zstd")
    report = {
        "schema": "causal-scale-feature-policy-seal-v1",
        "state": "SEALED_FEATURES_AND_POLICIES",
        "status": "PASS",
        "scientific_contract_sha256": contract_hash,
        "rows": len(sealed),
        "row_ids_sha256": row_ids_sha256(sealed.source_row_id.to_numpy()),
        "artifact": str(target),
        "artifact_sha256": sha256_file(target),
        "feature_dataset": str(features_path),
        "feature_dataset_sha256": sha256_file(features_path),
        "columns": list(sealed.columns),
    }
    report_path = output / "feature_policy_seal.json"
    atomic_json(report_path, report)
    report_path.with_name("FEATURE_POLICY_SEAL_SHA256.txt").write_text(
        sha256_file(report_path) + "\n", encoding="ascii"
    )
    return report


def evaluate_outcomes(source: Path, features_path: Path, output: Path) -> dict[str, object]:
    contract = Path("evidence/prospective/scientific_contract.json")
    digest = Path("evidence/prospective/SCIENTIFIC_CONTRACT_SHA256.txt")
    contract_hash = assert_contract_sealed(contract, digest)
    feature_seal_path = output / "feature_policy_seal.json"
    feature_seal_digest_path = output / "FEATURE_POLICY_SEAL_SHA256.txt"
    if not feature_seal_path.is_file() or not feature_seal_digest_path.is_file():
        raise PermissionError("feature/policy seal or its digest is absent")
    expected_feature_seal_hash = feature_seal_digest_path.read_text(encoding="ascii").strip()
    if sha256_file(feature_seal_path) != expected_feature_seal_hash:
        raise PermissionError("feature/policy seal metadata hash mismatch")
    feature_seal = json.loads(feature_seal_path.read_text(encoding="utf-8"))
    sealed_artifact = Path(str(feature_seal.get("artifact", "")))
    if (
        feature_seal.get("state") != "SEALED_FEATURES_AND_POLICIES"
        or feature_seal.get("scientific_contract_sha256") != contract_hash
        or not sealed_artifact.is_file()
        or sha256_file(sealed_artifact) != feature_seal.get("artifact_sha256")
        or sha256_file(features_path) != feature_seal.get("feature_dataset_sha256")
    ):
        raise PermissionError("feature/policy seal is invalid or bound to another contract/input")
    sealed = pd.read_parquet(sealed_artifact)
    if len(sealed) != feature_seal.get("rows") or row_ids_sha256(
        sealed.source_row_id.to_numpy()
    ) != feature_seal.get("row_ids_sha256"):
        raise PermissionError("feature/policy seal row identity mismatch")
    attempt = {
        "timestamp": datetime.now(UTC).isoformat(),
        "state": "OUTCOME_ACCESS_STARTED",
        "scientific_contract_sha256": contract_hash,
        "feature_policy_sha256": feature_seal["artifact_sha256"],
        "reason": "initial_one_shot_evaluation",
    }
    append_attempt(output / "attempt_log.jsonl", attempt)
    outcomes_path = Path(".local-data/prospective-vault/outcomes.parquet")
    extraction = extract_prospective_outcomes_once(source, outcomes_path, contract, digest)
    outcomes = pd.read_parquet(outcomes_path)
    features = pd.read_parquet(features_path)
    frame = features.merge(outcomes, on="source_row_id", validate="one_to_one").merge(
        sealed, on="source_row_id", validate="one_to_one"
    )
    historical = HistoricalPortfolioAdapter()
    y, t, p = frame.visit.to_numpy(), frame.treatment.to_numpy(), historical.propensity
    mu0, mu1 = (
        frame["nuisance__historical_mu0"].to_numpy(),
        frame["nuisance__historical_mu1"].to_numpy(),
    )
    candidate_policy = frame["policy__HIST_E7__raw__0.20"].to_numpy()
    baseline_policy = frame["policy__HIST_C3__raw__0.20"].to_numpy()
    candidate_values = aipw_policy_scores(y, t, candidate_policy, mu0, mu1, p)
    baseline_values = aipw_policy_scores(y, t, baseline_policy, mu0, mu1, p)
    gamma = dr_effect_score(y, t, mu0, mu1, p)
    # Reuse the exact predecessor linearization and cluster bootstrap code.
    reference_source = str(Path(".reference/ORBIT-Causal/src").resolve())
    if reference_source not in sys.path:
        sys.path.insert(0, reference_source)
    from orbit_causal.data import profile_ids
    from orbit_causal.locked_evaluation import (
        calibration_rmse_linearization,
        paired_cluster_bootstrap,
    )

    candidate_cal = calibration_rmse_linearization(
        frame["score__HIST_E7__raw"].to_numpy(), gamma, 10
    )
    baseline_cal = calibration_rmse_linearization(
        frame["score__HIST_C3__raw"].to_numpy(), gamma, 10
    )
    clusters = profile_ids(frame, historical.portfolio["C1"].data_config).to_numpy()
    draws = paired_cluster_bootstrap(
        candidate_values,
        baseline_values,
        candidate_cal.error - baseline_cal.error,
        candidate_cal.influence - baseline_cal.influence,
        clusters,
        4000,
        20261620,
    )
    value_difference = float((candidate_values - baseline_values).mean())
    value_lower = float(draws.candidate_minus_baseline.quantile(0.05))
    calibration_difference = candidate_cal.error - baseline_cal.error
    calibration_upper = float(draws.candidate_minus_baseline_calibration_rmse.quantile(0.95))
    h1 = value_lower > -0.00025
    h2 = calibration_upper < 0
    status = (
        "REPLICATED"
        if h1 and h2
        else (
            "DIRECTIONAL_BUT_UNCERTAIN"
            if value_difference > -0.00025 and calibration_difference < 0
            else "NOT_REPLICATED"
        )
    )
    evaluator_scores: dict[str, dict[str, np.ndarray]] = {}
    policy_columns = [
        column
        for column in frame.columns
        if column.startswith("policy__") and column.endswith("__raw__0.20")
    ]
    for policy_column in policy_columns:
        model_id = policy_column.split("__")[1]
        policy = frame[policy_column].to_numpy()
        evaluator_scores[model_id] = {
            "HISTORICAL_AIPW": aipw_policy_scores(y, t, policy, mu0, mu1, p),
            "RCT_IPW_HT": rct_ipw_incremental_scores(y, t, policy, p),
            "INDEPENDENT_AIPW": aipw_policy_scores(
                y,
                t,
                policy,
                frame["nuisance__independent_mu0"].to_numpy(),
                frame["nuisance__independent_mu1"].to_numpy(),
                p,
            ),
        }
    comparison_rows: list[dict[str, object]] = []
    secondary_models = [
        model
        for model in (
            "RESPONSE_MARGINAL",
            "RESPONSE_TREATED",
            "HIST_C1",
            "GPU_T",
            "X_LEARNER",
            "R_LEARNER",
            "CAUSAL_FOREST",
        )
        if model in evaluator_scores
    ]
    for evaluator in ("HISTORICAL_AIPW", "RCT_IPW_HT", "INDEPENDENT_AIPW"):
        family: dict[str, float] = {}
        staged: list[dict[str, object]] = []
        for comparator in secondary_models:
            comparison = f"HIST_E7-{comparator}"
            inference = cluster_normal_inference(
                evaluator_scores["HIST_E7"][evaluator] - evaluator_scores[comparator][evaluator],
                clusters,
            )
            family[comparison] = inference["p_value_two_sided"]
            staged.append(
                {
                    "comparison": comparison,
                    "evaluator": evaluator,
                    "capacity": 0.2,
                    **inference,
                    "evidence_class": "PROSPECTIVE_SECONDARY",
                }
            )
        adjusted = holm(family)
        for row in staged:
            correction = adjusted[str(row["comparison"])]
            row["multiplicity_adjustment"] = "Holm_within_evaluator"
            row["holm_threshold"] = correction["holm_threshold"]
            row["gate"] = bool(correction["reject"])
            row["status"] = "SIGNIFICANT" if correction["reject"] else "NOT_SIGNIFICANT"
            comparison_rows.append(row)
    comparison_rows.insert(
        0,
        {
            "comparison": "HIST_E7-HIST_C3",
            "evaluator": "HISTORICAL_AIPW",
            "capacity": 0.2,
            "point_estimate": value_difference,
            "CI_lower": float(draws.candidate_minus_baseline.quantile(0.025)),
            "CI_upper": float(draws.candidate_minus_baseline.quantile(0.975)),
            "standard_error": float(draws.candidate_minus_baseline.std(ddof=1)),
            "p_value_two_sided": None,
            "multiplicity_adjustment": "co_primary_separate",
            "holm_threshold": None,
            "gate": h1,
            "status": "PASS" if h1 else "FAIL",
            "evidence_class": "PROSPECTIVE_PRIMARY",
        },
    )
    results = {
        "schema": "causal-scale-prospective-results-v1",
        "state": "COMPLETED",
        "status": status,
        "scientific_contract_sha256": contract_hash,
        "outcome_extraction": extraction,
        "rows": len(frame),
        "candidate_value": float(candidate_values.mean()),
        "baseline_value": float(baseline_values.mean()),
        "value_difference": value_difference,
        "value_noninferiority_one_sided_95pct_lower": value_lower,
        "candidate_calibration_rmse": candidate_cal.error,
        "baseline_calibration_rmse": baseline_cal.error,
        "calibration_rmse_difference": calibration_difference,
        "calibration_superiority_one_sided_95pct_upper": calibration_upper,
        "gates": {"H1_value_noninferiority": h1, "H2_calibration_superiority": h2},
        "secondary_comparisons": len(comparison_rows) - 1,
        "secondary_inference": "cluster_robust_normal_two_sided_Holm_within_evaluator",
    }
    draws.to_csv(output / "bootstrap_draws.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(output / "prospective_summary.csv", index=False)
    atomic_json(output / "prospective_results.json", results)
    append_attempt(
        output / "attempt_log.jsonl",
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "state": "COMPLETED",
            "status": status,
            "scientific_contract_sha256": contract_hash,
        },
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["seal-features", "evaluate"], required=True)
    parser.add_argument(
        "--source", type=Path, default=Path(".local-data/raw/criteo-research-uplift-v2.1.csv.gz")
    )
    parser.add_argument(
        "--features", type=Path, default=Path(".local-data/marts/prospective_features.parquet")
    )
    parser.add_argument("--models", type=Path, default=Path(".local-models/frozen"))
    parser.add_argument("--output", type=Path, default=Path("evidence/prospective"))
    args = parser.parse_args()
    result = (
        seal_features(args.features, args.models, args.output)
        if args.stage == "seal-features"
        else evaluate_outcomes(args.source, args.features, args.output)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
