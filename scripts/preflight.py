from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil

from causal_scale.evidence.manifest import atomic_json, utc_now
from causal_scale.gpu.preflight import nvidia_inventory, xgboost_cuda_smoke

PACKAGES = [
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "xgboost",
    "econml",
    "causalml",
    "onnx",
    "onnxruntime",
    "jax",
    "torch",
    "pyspark",
]


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def java_version() -> str | None:
    try:
        process = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=10)
        return (process.stderr or process.stdout).splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def build_report(run_cuda: bool = True) -> dict[str, object]:
    report = {
        "schema": "causal-scale-environment-v1",
        "created_at": utc_now(),
        "os": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "java": java_version(),
        "packages": {name: version(name) for name in PACKAGES},
        "cpu_physical": psutil.cpu_count(logical=False),
        "cpu_logical": psutil.cpu_count(logical=True),
        "ram_total_bytes": psutil.virtual_memory().total,
        "ram_available_bytes": psutil.virtual_memory().available,
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": nvidia_inventory(),
    }
    report["xgboost_cuda"] = (
        xgboost_cuda_smoke()
        if run_cuda and version("xgboost")
        else {"status": "GPU_XGBOOST_INCOMPATIBLE", "error": "xgboost is not installed"}
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("evidence/environment/preflight/environment.json")
    )
    parser.add_argument("--skip-cuda", action="store_true")
    args = parser.parse_args()
    report = build_report(not args.skip_cuda)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
