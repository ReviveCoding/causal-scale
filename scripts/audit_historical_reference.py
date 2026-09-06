from __future__ import annotations

import json
from pathlib import Path

from causal_scale.evidence.hashing import sha256_file, stable_hash, tree_manifest
from causal_scale.evidence.manifest import atomic_json, read_json, write_sha256s


def main() -> None:
    root = Path(".reference/ORBIT-Causal")
    output = Path("evidence/historical_reference")
    output.mkdir(parents=True, exist_ok=True)
    v38 = read_json(root / "contracts/v0.3.8-selection-contract.json")
    v39 = read_json(root / "contracts/v0.3.9-selection-contract.json")
    metrics = read_json(root / "results/metrics.json")
    replay_audit = read_json(root / "audit/orbit-causal-v039-independent-replay-audit.json")
    manifest = {
        "schema": "causal-scale-historical-reference-v1",
        "project": "ORBIT-Causal",
        "version": "0.3.9",
        "source_repository": ".reference/ORBIT-Causal",
        "committed_head": "2a129d73cab03a6c08326bc92cb79f3dd87b35bd",
        "snapshot_file_count": len(tree_manifest(root)),
        "snapshot_manifest_sha256": stable_hash(tree_manifest(root)),
        "frozen_artifact_sha256": sha256_file(root / "artifacts/frozen_candidate_portfolio.joblib"),
        "contract_v038_sha256": sha256_file(root / "contracts/v0.3.8-selection-contract.json"),
        "contract_v039_sha256": sha256_file(root / "contracts/v0.3.9-selection-contract.json"),
        "independent_replay_status": replay_audit["status"],
        "status": "PASS" if replay_audit["status"] == "PASS" else "FAIL",
    }
    contract_summary = {
        "schema": "causal-scale-historical-contract-summary-v1",
        "source": v39["source"],
        "features": [f"f{i}" for i in range(12)],
        "treatment": "treatment",
        "primary_outcome": "visit",
        "auxiliary_outcomes": ["conversion"],
        "post_treatment_not_feature": "exposure",
        "C1": "frozen TwoModelUplift T-learner",
        "C3": "frozen CrossFittedDRLearner baseline",
        "C5": "frozen ReferenceStateEffectCalibrator component",
        "E7": v39["candidate"]["effect"],
        "E7_weights": v38["candidate"]["weights"],
        "capacity": v39["evaluation"]["capacity"],
        "tie_semantics": "stable descending score; evaluation position is default tie key",
        "evaluator": "AIPW incremental value versus never-treat using frozen C1 nuisance outcomes/propensity",
        "calibration": "10 quantile-bin weighted RMSE versus DR effect pseudo-outcome",
        "bootstrap": v39["evaluation"],
        "co_primary_rules": v39["co_primary_release_rules"],
        "guardrails": v39["guardrails"],
        "verdict": metrics["verdict"],
        "limitations": [
            "offline public randomized benchmark",
            "not online revenue",
            "not production ROI",
            "no external validity claim",
        ],
    }
    row_roles = {
        "schema": "causal-scale-historical-row-roles-v1",
        **v39["data_roles"],
        "development": v38["development"],
    }
    atomic_json(output / "reference_manifest.json", manifest)
    atomic_json(output / "historical_contract_summary.json", contract_summary)
    atomic_json(output / "historical_metrics.json", metrics)
    atomic_json(output / "historical_row_roles.json", row_roles)
    paths = [path for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS"]
    write_sha256s(output, paths)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
