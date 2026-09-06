from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd
from fit_baselines import partition_development

from causal_scale.baselines import EconMLCausalForest
from causal_scale.evidence.hashing import row_ids_sha256, sha256_file, stable_hash, tree_manifest
from causal_scale.evidence.manifest import atomic_json, read_json
from causal_scale.models.artifacts import write_model_manifest


def main() -> None:
    temp = Path(".local-cache/joblib").resolve()
    temp.mkdir(parents=True, exist_ok=True)
    os.environ["JOBLIB_TEMP_FOLDER"] = str(temp)
    train, _, _ = partition_development(pd.read_parquet(".local-data/marts/development.parquet"))
    model = EconMLCausalForest(n_estimators=400, n_jobs=4)
    started = time.perf_counter()
    model.fit(train)
    predictions = model.predict_effect(train.iloc[:10_000])
    if not pd.Series(predictions).map(pd.notna).all():
        raise ValueError("causal forest produced non-finite predictions")
    artifact = Path(".local-models/frozen/causal_forest.joblib")
    model.save(artifact)
    manifest = write_model_manifest(
        artifact,
        model_id=model.model_id,
        method=type(model).__name__,
        device="cpu",
        seed=model.seed,
        data_hash=row_ids_sha256(train.source_row_id.to_numpy()),
        hyperparameters={"n_estimators": 400, "n_jobs": 4, "cv": 3},
        code_hash=stable_hash(tree_manifest("src/causal_scale", ("*.py",))),
    )
    evidence = Path("evidence/models/baselines/run_manifest.json")
    report = read_json(evidence)
    report["models"] = [entry for entry in report["models"] if entry["model"] != "CAUSAL_FOREST"]
    report["models"].append(
        {
            "model": "CAUSAL_FOREST",
            "status": "PASS",
            "artifact": str(artifact),
            "manifest": str(manifest),
            "fit_seconds": time.perf_counter() - started,
            "retry": "workspace_local_joblib_temp",
        }
    )
    atomic_json(evidence, report)
    atomic_json(
        Path("evidence/models/causal_forest_retry/metrics.json"),
        {
            "status": "PASS",
            "artifact_sha256": sha256_file(artifact),
            "retry_reason": "WinError 5 from default joblib temp",
            "scientific_parameters_changed": False,
            "fit_seconds": report["models"][-1]["fit_seconds"],
        },
    )
    print(json.dumps(report["models"][-1], indent=2))


if __name__ == "__main__":
    main()
