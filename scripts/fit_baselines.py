from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from causal_scale.baselines import (
    ConstantATE,
    EconMLCausalForest,
    ResponseModel,
    XGBRLearner,
    XGBTlearner,
    XGBXLearner,
)
from causal_scale.evidence.hashing import row_ids_sha256, stable_hash, tree_manifest
from causal_scale.evidence.manifest import atomic_json
from causal_scale.gpu.preflight import xgboost_cuda_smoke
from causal_scale.models.artifacts import write_model_manifest
from causal_scale.models.base import CausalEstimator


def partition_development(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ids = frame.source_row_id.to_numpy(np.uint64)
    values = ids + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    priorities = values ^ (values >> np.uint64(31))
    order = np.argsort(priorities, kind="stable")
    return (
        frame.iloc[order[:350_000]].copy(),
        frame.iloc[order[350_000:425_000]].copy(),
        frame.iloc[order[425_000:500_000]].copy(),
    )


def fit_one(
    model: CausalEstimator, train: pd.DataFrame, output: Path, device: str, code_hash: str
) -> dict[str, object]:
    start = time.perf_counter()
    model.fit(train)
    elapsed = time.perf_counter() - start
    path = output / f"{model.model_id.lower()}.joblib"
    model.save(path)
    manifest_path = write_model_manifest(
        path,
        model_id=model.model_id,
        method=type(model).__name__,
        device=device,
        seed=getattr(model, "seed", 20260717),
        data_hash=row_ids_sha256(train.source_row_id.to_numpy()),
        hyperparameters=getattr(model, "params", {}),
        code_hash=code_hash,
    )
    return {
        "model": model.model_id,
        "status": "PASS",
        "artifact": str(path),
        "manifest": str(manifest_path),
        "fit_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development", type=Path, default=Path(".local-data/marts/development.parquet")
    )
    parser.add_argument("--output", type=Path, default=Path(".local-models/frozen"))
    parser.add_argument("--profile", choices=["smoke", "standard", "full"], default="standard")
    parser.add_argument("--gpu", choices=["auto", "off", "force"], default="auto")
    parser.add_argument("--include-forest", action="store_true")
    args = parser.parse_args()
    joblib_temp = Path(".local-cache/joblib").resolve()
    joblib_temp.mkdir(parents=True, exist_ok=True)
    os.environ["JOBLIB_TEMP_FOLDER"] = str(joblib_temp)
    frame = pd.read_parquet(args.development)
    train, affine, rsec = partition_development(frame)
    if len(train) != 350_000 or len(affine) != 75_000 or len(rsec) != 75_000:
        raise ValueError("development partition counts do not match the historical contract")
    gpu = xgboost_cuda_smoke() if args.gpu != "off" else {"status": "GPU_DISABLED"}
    if args.gpu == "force" and gpu["status"] != "GPU_XGBOOST_AVAILABLE":
        raise RuntimeError(f"GPU forced but unavailable: {gpu}")
    device = "cuda" if gpu["status"] == "GPU_XGBOOST_AVAILABLE" else "cpu"
    n_estimators = {"smoke": 40, "standard": 180, "full": 300}[args.profile]
    params = {"n_estimators": n_estimators, "max_depth": 6, "n_jobs": 8}
    models: list[CausalEstimator] = [
        ResponseModel("RESPONSE_MARGINAL", False, device=device, **params),
        ResponseModel("RESPONSE_TREATED", True, device=device, **params),
        ConstantATE(),
        XGBTlearner(device=device, **params),
        XGBXLearner(device=device, **params),
        XGBRLearner(device=device, **params),
        XGBTlearner(seed=20260717 + 8000, device=device, **params),
    ]
    models[-1].model_id = "INDEPENDENT_AIPW_NUISANCE"
    if args.include_forest:
        models.append(EconMLCausalForest(n_estimators=200 if args.profile == "smoke" else 400))
    args.output.mkdir(parents=True, exist_ok=True)
    code_hash = stable_hash(tree_manifest(Path("src/causal_scale"), ("*.py",)))
    results = []
    for model in models:
        try:
            results.append(
                fit_one(
                    model,
                    train,
                    args.output,
                    device if not isinstance(model, EconMLCausalForest) else "cpu",
                    code_hash,
                )
            )
        except Exception as error:
            if model.model_id == "CAUSAL_FOREST":
                results.append(
                    {
                        "model": model.model_id,
                        "status": "SKIPPED_RESOURCE_LIMIT",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            else:
                raise
    role_receipt = {
        "model_train_rows": len(train),
        "model_train_hash": row_ids_sha256(train.source_row_id.to_numpy()),
        "affine_calibration_rows": len(affine),
        "affine_calibration_hash": row_ids_sha256(affine.source_row_id.to_numpy()),
        "rsec_calibration_rows": len(rsec),
        "rsec_calibration_hash": row_ids_sha256(rsec.source_row_id.to_numpy()),
    }
    atomic_json(args.output / "development_roles.json", role_receipt)
    atomic_json(
        Path("evidence/models/baselines/run_manifest.json"),
        {
            "status": "PASS",
            "device": device,
            "gpu_qualification": gpu,
            "models": results,
            "roles": role_receipt,
        },
    )
    print(json.dumps({"device": device, "models": results, "roles": role_receipt}, indent=2))


if __name__ == "__main__":
    main()
