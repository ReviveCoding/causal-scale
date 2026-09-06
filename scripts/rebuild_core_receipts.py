from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from causal_scale.evidence.hashing import sha256_file
from causal_scale.evidence.manifest import atomic_json, read_json


def parquet_receipt(
    path: Path, expected_rows: int, expected_columns: list[str]
) -> dict[str, object]:
    metadata = pq.read_metadata(path)
    columns = metadata.schema.to_arrow_schema().names
    if metadata.num_rows != expected_rows or columns != expected_columns:
        raise ValueError(f"mart mismatch: {path}")
    return {
        "path": str(path.resolve()),
        "rows": metadata.num_rows,
        "columns": columns,
        "sha256": sha256_file(path),
    }


def rebuild_data_receipt() -> None:
    source = read_json("evidence/data/criteo_v2_1/dataset_manifest.json")
    features = ["source_row_id", *[f"f{i}" for i in range(12)]]
    outcomes = [*features, "treatment", "conversion", "visit", "exposure"]
    marts = Path(".local-data/marts")
    full = parquet_receipt(marts / "full_features.parquet", 13_979_592, features)
    full["outcome_fields_emitted"] = False
    prospective = parquet_receipt(marts / "prospective_features.parquet", 2_000_000, features)
    historical_artifacts = {}
    counts = {"development": 500_000, "historical_250k": 250_000, "historical_locked_2m": 2_000_000}
    for name, count in counts.items():
        historical_artifacts[name] = parquet_receipt(marts / f"{name}.parquet", count, outcomes)
    atomic_json(
        Path("evidence/data/criteo_marts/run_manifest.json"),
        {
            "source": source,
            "full_feature_mart": full,
            "historical_role_marts": {
                "rows_seen": 13_979_592,
                "roles": counts,
                "prospective_outcomes_emitted": False,
            },
            "prospective_feature_mart": prospective,
            "historical_artifacts": historical_artifacts,
            "status": "PASS",
        },
    )


def rebuild_model_receipt() -> None:
    model_dir = Path(".local-models/frozen")
    results = []
    devices = set()
    for manifest_path in sorted(model_dir.glob("*.joblib.manifest.json")):
        manifest = read_json(manifest_path)
        artifact = Path(str(manifest["artifact_path"]))
        if not artifact.is_file() or sha256_file(artifact) != manifest["artifact_hash"]:
            raise ValueError(f"frozen model hash mismatch: {artifact}")
        devices.add(str(manifest["device"]))
        results.append(
            {
                "model": manifest["model_id"],
                "status": "PASS",
                "artifact": str(artifact),
                "artifact_sha256": manifest["artifact_hash"],
                "manifest": str(manifest_path),
            }
        )
    forest = read_json("evidence/models/causal_forest_retry/metrics.json")
    results.append(
        {
            "model": "CAUSAL_FOREST",
            "status": forest["status"],
            "evidence": "evidence/models/causal_forest_retry/metrics.json",
        }
    )
    environment = read_json("evidence/environment/preflight/environment.json")
    atomic_json(
        Path("evidence/models/baselines/run_manifest.json"),
        {
            "status": "PASS",
            "device": "cuda" if "cuda" in devices else "cpu",
            "gpu_qualification": environment["xgboost_cuda"],
            "models": results,
            "roles": read_json(model_dir / "development_roles.json"),
            "receipt_rebuilt_from_frozen_artifacts": True,
        },
    )


def main() -> None:
    rebuild_data_receipt()
    rebuild_model_receipt()
    print(json.dumps({"status": "PASS", "receipts": ["data_marts", "models"]}, indent=2))


if __name__ == "__main__":
    main()
