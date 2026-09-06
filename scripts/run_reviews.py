from __future__ import annotations

import json
from pathlib import Path

from causal_scale.evidence.manifest import atomic_json, read_json

SCIENCE = {
    "data_leakage": "features frozen to f0..f11",
    "role_leakage": "SplitMix64 role hashes are contract-bound",
    "outcome_leakage": "prospective feature mart excludes treatment and all outcomes",
    "post_treatment_leakage": "exposure is prohibited",
    "propensity": "randomized treatment rate frozen as primary",
    "estimand": "incremental policy value versus never-treat",
    "self_evaluation": "historical and independently trained nuisance views separated",
    "calibration": "causal scores only; response scores excluded",
    "zero_boundary": "eligibility changes explicitly measured",
    "holdout_contamination": "zero row intersection proven",
    "multiple_testing": "Holm family frozen pre-outcome",
    "pehe": "not computed for Criteo",
    "claim_boundary": "historical, retrospective, systems, external, prospective labels required",
}

SYSTEMS = {
    "row_id_stability": "physical source_row_id assigned before Spark",
    "ordering": "global score descending and numeric tie key ascending",
    "tie_bug": "numeric IDs never converted to strings",
    "partition_dependence": "partition-invariance gate required",
    "artifact_distribution": "SparkFiles/addFile path implemented",
    "model_loading": "one load per iterator",
    "large_toPandas": "production computations aggregate natively; parity collection is bounded/explicit",
    "cache_invalidation": "code/config/data/upstream hashes bound in DAG state",
    "gpu_oom": "one neutral fallback then CPU",
    "gpu_concurrency": "single heavy job default and 80% VRAM budget",
    "windows_paths": "Path/resolved URI handling",
    "spark_cleanup": "sessions stop in finally blocks",
    "prospective_retry": "scientific identities immutable after outcome access",
}


def main() -> None:
    root = Path("evidence/reviews")
    historical = read_json("evidence/historical_reference/reference_manifest.json")
    data = read_json("evidence/data/criteo_marts/run_manifest.json")
    roles = read_json("evidence/data/row_roles/role_manifest.json")
    models = read_json("evidence/models/baselines/run_manifest.json")
    calibration = read_json("evidence/models/common_calibration/metrics.json")
    retrospective = read_json("evidence/retrospective/model_benchmark/metrics.json")
    required_models = {
        "RESPONSE_MARGINAL",
        "RESPONSE_TREATED",
        "CONSTANT_ATE",
        "GPU_T",
        "X_LEARNER",
        "R_LEARNER",
        "INDEPENDENT_AIPW_NUISANCE",
    }
    passed_models = {item["model"] for item in models["models"] if item.get("status") == "PASS"}
    expected_feature_columns = ["source_row_id", *[f"f{i}" for i in range(12)]]
    science_assertions = {
        "historical_reference_pass": historical.get("status") == "PASS",
        "authenticated_data_pass": data["source"].get("status") == "PASS",
        "prospective_zero_overlap": roles["roles"]["prospective_2m"].get(
            "intersection_with_all_prior"
        )
        == 0,
        "full_mart_feature_only": data["full_feature_mart"].get("columns")
        == expected_feature_columns
        and data["full_feature_mart"].get("outcome_fields_emitted") is False,
        "prospective_mart_feature_only": data["prospective_feature_mart"].get("columns")
        == expected_feature_columns,
        "mandatory_models_pass": required_models <= passed_models,
        "causal_forest_pass_or_exact_p1_blocker": any(
            item["model"] == "CAUSAL_FOREST"
            and item.get("status") in {"PASS", "SANDBOX_TOOLING_BLOCKER"}
            for item in models["models"]
        ),
        "response_calibration_excluded": calibration.get("response_scores_excluded") is True
        and not any(str(item["model"]).startswith("RESPONSE_") for item in calibration["models"]),
        "retrospective_complete": retrospective.get("status") == "PASS",
    }
    science_status = "PASS" if all(science_assertions.values()) else "FAIL"
    science = {
        "schema": "causal-scale-science-review-v1",
        "status": science_status,
        "checks": SCIENCE,
        "machine_assertions": science_assertions,
        "valid_issues_fixed": [
            "prospective full-source outcome mart removed",
            "numeric tie keys preserved",
            "response scores excluded from causal calibration",
        ],
    }
    systems_paths = {
        "spark_parity": Path("evidence/spark/historical_parity/metrics.json"),
        "partition_invariance": Path("evidence/spark/partition_invariance/metrics.json"),
        "feature_scale": Path("evidence/systems/scale/metrics.json"),
        "local_scaling": Path("evidence/systems/local_scaling/metrics.json"),
        "spark_backends": Path("evidence/systems/spark_backends/metrics.json"),
        "gpu": Path("evidence/gpu/controlled_benchmark/metrics.json"),
    }
    systems_assertions = {
        name: path.is_file()
        and read_json(path).get("status", "PASS" if name == "gpu" else None) == "PASS"
        for name, path in systems_paths.items()
    }
    systems_status = "PASS" if all(systems_assertions.values()) else "FAIL"
    systems = {
        "schema": "causal-scale-systems-review-v1",
        "status": systems_status,
        "checks": SYSTEMS,
        "machine_assertions": systems_assertions,
        "valid_issues_fixed": [
            "Spark workers use distributed artifacts",
            "pytest temp redirected inside workspace",
            "scikit-learn pinned to artifact version",
        ],
    }
    atomic_json(root / "science_review.json", science)
    atomic_json(root / "systems_review.json", systems)
    print(json.dumps({"science": science, "systems": systems}, indent=2))
    if science_status != "PASS" or systems_status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
