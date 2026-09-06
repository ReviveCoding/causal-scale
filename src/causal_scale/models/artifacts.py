from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path
from typing import Any

from causal_scale.evidence.hashing import sha256_file, stable_hash
from causal_scale.evidence.manifest import atomic_json, utc_now


def library_versions() -> dict[str, str | None]:
    names = [
        "causal-scale",
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "econml",
        "causalml",
        "pyspark",
        "pyarrow",
        "joblib",
    ]
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def model_manifest(
    artifact: Path,
    *,
    model_id: str,
    method: str,
    device: str,
    seed: int,
    data_hash: str,
    hyperparameters: dict[str, Any],
    code_hash: str,
    calibration_hash: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": "causal-scale-model-artifact-v1",
        "created_at": utc_now(),
        "model_id": model_id,
        "method": method,
        "device": device,
        "seed": seed,
        "data_hash": data_hash,
        "hyperparameters": hyperparameters,
        "hyperparameters_hash": stable_hash(hyperparameters),
        "library_versions": library_versions(),
        "python": platform.python_version(),
        "code_hash": code_hash,
        "artifact_path": str(artifact),
        "artifact_hash": sha256_file(artifact),
        "calibration_hash": calibration_hash,
    }


def write_model_manifest(artifact: Path, **kwargs: Any) -> Path:
    path = artifact.with_suffix(artifact.suffix + ".manifest.json")
    atomic_json(path, model_manifest(artifact, **kwargs))
    return path
