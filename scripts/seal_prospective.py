from __future__ import annotations

import json
from pathlib import Path

from causal_scale.evidence.hashing import sha256_file, tree_manifest
from causal_scale.evidence.manifest import read_json
from causal_scale.evidence.seal import freeze_contract

GATES = {
    "historical_reference": Path("evidence/historical_reference/reference_manifest.json"),
    "data": Path("evidence/data/criteo_v2_1/dataset_manifest.json"),
    "models": Path("evidence/models/baselines/run_manifest.json"),
    "retrospective": Path("evidence/retrospective/model_benchmark/metrics.json"),
    "spark_parity": Path("evidence/spark/historical_parity/metrics.json"),
    "partition_invariance": Path("evidence/spark/partition_invariance/metrics.json"),
    "engineering": Path("evidence/qualification/engineering/qualification_summary.json"),
    "science_review": Path("evidence/reviews/science_review.json"),
    "systems_review": Path("evidence/reviews/systems_review.json"),
}


def main() -> None:
    failed = {}
    receipts = {}
    for name, path in GATES.items():
        if not path.is_file():
            failed[name] = "MISSING"
            continue
        value = read_json(path)
        receipts[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "status": value.get("status"),
        }
        if value.get("status") != "PASS":
            failed[name] = value.get("status")
    if failed:
        raise RuntimeError(f"prospective seal blocked by pre-seal gates: {failed}")
    output = Path("evidence/prospective/scientific_contract.json")
    payload = {
        "state": "IMPLEMENTATION_FROZEN",
        "gate_receipts": receipts,
        "dataset_identity": read_json(GATES["data"]),
        "row_identity": read_json(Path("evidence/data/row_roles/role_manifest.json"))["roles"][
            "prospective_2m"
        ],
        "models": tree_manifest(Path(".local-models/frozen")),
        "script_manifest": tree_manifest(Path("scripts"), ("*.py",)),
        "dependency_lock_sha256": sha256_file(Path("requirements.lock")),
        "policy": {
            "primary_capacity": 0.2,
            "sensitivity_capacities": [0.05, 0.1, 0.3, 0.5],
            "tie": "score_desc_numeric_source_row_id_asc",
            "positive_only": True,
        },
        "evaluators": ["HISTORICAL_AIPW", "RCT_IPW_HT", "INDEPENDENT_AIPW"],
        "primary_tests": {
            "value": {
                "comparison": "HIST_E7-HIST_C3",
                "margin": -0.00025,
                "rule": "one_sided_95pct_lower_gt_margin",
            },
            "calibration": {
                "comparison": "RMSE_HIST_E7-RMSE_HIST_C3",
                "margin": 0.0,
                "rule": "one_sided_95pct_upper_lt_zero",
            },
        },
        "bootstrap": {
            "replicates": 4000,
            "seed": 20261620,
            "cluster": "exact_pretreatment_profile",
            "paired_stream": True,
        },
        "secondary_family": [
            "RESPONSE_MARGINAL",
            "RESPONSE_TREATED",
            "HIST_C1",
            "GPU_T",
            "X_LEARNER",
            "R_LEARNER",
            "CAUSAL_FOREST",
        ],
        "multiplicity": "Holm",
        "secondary_inference": {
            "interval": "cluster_robust_normal_95pct",
            "p_value": "two_sided_normal",
            "family": "method_comparisons_within_each_evaluator",
            "alpha": 0.05,
        },
        "operational_retry_fields": [
            "worker_count",
            "executor_memory",
            "arrow_batch_size",
            "partition_count",
        ],
    }
    digest = freeze_contract(Path.cwd(), payload, output)
    print(
        json.dumps(
            {"state": "IMPLEMENTATION_FROZEN", "scientific_contract_sha256": digest}, indent=2
        )
    )


if __name__ == "__main__":
    main()
