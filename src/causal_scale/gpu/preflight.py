from __future__ import annotations

import json
import subprocess
import time

import numpy as np


def nvidia_inventory() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=15
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    result = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        result.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "vram_mib": int(parts[2]),
                "driver": parts[3],
                "temperature_c": float(parts[4]),
                "power_w": float(parts[5]),
            }
        )
    return result


def xgboost_cuda_smoke() -> dict[str, object]:
    started = time.perf_counter()
    try:
        import xgboost as xgb

        rng = np.random.default_rng(17)
        x = rng.normal(size=(2048, 8)).astype(np.float32)
        y = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(np.int8)
        model = xgb.XGBClassifier(
            n_estimators=8,
            max_depth=3,
            tree_method="hist",
            device="cuda",
            random_state=17,
            n_jobs=1,
        ).fit(x, y)
        prediction = model.predict_proba(x[:32])[:, 1]
        if not np.isfinite(prediction).all():
            raise RuntimeError("non-finite CUDA predictions")
        status, error = "GPU_XGBOOST_AVAILABLE", None
    except Exception as exc:  # qualification must preserve exact incompatibility
        status, error = "GPU_XGBOOST_INCOMPATIBLE", f"{type(exc).__name__}: {exc}"
    return {
        "status": status,
        "elapsed_seconds": time.perf_counter() - started,
        "error": error,
        "gpus": nvidia_inventory(),
    }


def main() -> None:
    print(json.dumps(xgboost_cuda_smoke(), indent=2))
