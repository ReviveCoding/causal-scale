from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from causal_scale.baselines.tlearner_gpu import XGBTlearner
from causal_scale.evidence.manifest import atomic_json
from causal_scale.gpu.preflight import xgboost_cuda_smoke
from causal_scale.gpu.telemetry import sample_gpu
from causal_scale.policy.engine import capacity_policy


def fixture(rows: int, seed: int = 91) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 12)).astype(np.float32)
    t = rng.binomial(1, 0.85, rows)
    p = 1 / (1 + np.exp(-(-3 + 0.4 * x[:, 0] + t * (0.15 + 0.2 * (x[:, 1] > 0)))))
    y = rng.binomial(1, p)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(12)])
    frame["treatment"], frame["visit"] = t, y
    return frame


def timed(device: str, frame: pd.DataFrame) -> tuple[XGBTlearner, float, float]:
    model = XGBTlearner(device=device, n_estimators=80, max_depth=5, n_jobs=4)
    start = time.perf_counter()
    model.fit(frame)
    train = time.perf_counter() - start
    start = time.perf_counter()
    model.predict_effect(frame)
    infer = time.perf_counter() - start
    return model, train, infer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=Path("evidence/gpu/controlled_benchmark"))
    args = parser.parse_args()
    qualification = xgboost_cuda_smoke()
    frame = fixture(args.rows)
    cpu, cpu_train, cpu_infer = timed("cpu", frame)
    records = []
    if qualification["status"] == "GPU_XGBOOST_AVAILABLE":
        before = sample_gpu()
        gpu, gpu_train, gpu_infer = timed("cuda", frame)
        after = sample_gpu()
        cpu_prediction, gpu_prediction = cpu.predict_effect(frame), gpu.predict_effect(frame)
        cpu_policy = capacity_policy(cpu_prediction, 0.2).membership
        gpu_policy = capacity_policy(gpu_prediction, 0.2).membership
        records.append(
            {
                "model": "GPU_T",
                "device": "cuda",
                "train_time": gpu_train,
                "inference_time": gpu_infer,
                "cpu_gpu_speedup": cpu_infer / gpu_infer,
                "peak_vram": max(
                    [item["memory_used_mib"] for item in before + after], default=None
                ),
                "prediction_agreement": float(np.corrcoef(cpu_prediction, gpu_prediction)[0, 1]),
                "policy_agreement": float(np.mean(cpu_policy == gpu_policy)),
                "telemetry_before": before,
                "telemetry_after": after,
            }
        )
    else:
        records.append(
            {
                "model": "GPU_T",
                "device": "cpu_fallback",
                "train_time": cpu_train,
                "inference_time": cpu_infer,
                "cpu_gpu_speedup": None,
                "peak_vram": None,
                "prediction_agreement": None,
                "policy_agreement": None,
                "status": qualification["status"],
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {key: value for key, value in row.items() if not key.startswith("telemetry_")}
            for row in records
        ]
    ).to_csv(args.output / "gpu_benchmark.csv", index=False)
    atomic_json(
        args.output / "metrics.json",
        {
            "qualification": qualification,
            "benchmark": records,
            "gpu_parallelism_actually_used": 1
            if qualification["status"] == "GPU_XGBOOST_AVAILABLE"
            else 0,
        },
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
