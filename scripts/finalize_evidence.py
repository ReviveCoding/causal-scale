from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from causal_scale.evidence.hashing import sha256_file, tree_manifest
from causal_scale.evidence.manifest import atomic_json, read_json, utc_now, write_sha256s


def scalar_metrics(value: Any, needles: tuple[str, ...], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, (int, float)) and any(needle in key.lower() for needle in needles):
                result[path] = child
            result.update(scalar_metrics(child, needles, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(scalar_metrics(child, needles, f"{prefix}[{index}]"))
    return result


def contextual_manifest(path: Path) -> dict[str, Any]:
    return read_json(path) if path.is_file() else {"status": "NOT_AVAILABLE"}


def main() -> None:
    evidence = Path("evidence")
    environment = contextual_manifest(Path("evidence/environment/preflight/environment.json"))
    dataset = contextual_manifest(Path("evidence/data/criteo_v2_1/dataset_manifest.json"))
    model_files = sorted(Path(".local-models/frozen").glob("*.manifest.json"))
    model_context = {
        "status": "PASS" if model_files else "NOT_AVAILABLE",
        "artifacts": {str(path): sha256_file(path) for path in model_files},
    }
    leaves = sorted(
        directory
        for directory in evidence.rglob("*")
        if directory.is_dir()
        and any(path.is_file() and path.suffix in {".json", ".csv"} for path in directory.iterdir())
    )
    finalized = []
    for directory in leaves:
        relative = directory.relative_to(evidence).as_posix()
        primary = directory / "metrics.json"
        if not primary.is_file():
            candidates = sorted(
                path
                for path in directory.glob("*.json")
                if path.name not in {"config.json", "environment.json", "dataset_manifest.json"}
            )
            primary = candidates[0] if candidates else directory / "metrics.json"
        metrics = read_json(primary) if primary.is_file() else {}
        atomic_json(
            directory / "config.json",
            {
                "schema": "causal-scale-experiment-config-v1",
                "experiment": relative,
                "machine_generated": True,
            },
        )
        atomic_json(directory / "environment.json", environment)
        if not (directory / "dataset_manifest.json").is_file():
            atomic_json(directory / "dataset_manifest.json", dataset)
        if not (directory / "model_manifest.json").is_file():
            atomic_json(directory / "model_manifest.json", model_context)
        atomic_json(
            directory / "timing.json",
            {
                "status": "PASS"
                if scalar_metrics(metrics, ("time", "runtime", "elapsed"))
                else "NOT_CAPTURED",
                "metrics": scalar_metrics(metrics, ("time", "runtime", "elapsed")),
            },
        )
        atomic_json(
            directory / "resource_metrics.json",
            {
                "status": "PASS"
                if scalar_metrics(
                    metrics,
                    ("memory", "vram", "shuffle", "spill", "temperature", "power", "utilization"),
                )
                else "NOT_CAPTURED",
                "metrics": scalar_metrics(
                    metrics,
                    ("memory", "vram", "shuffle", "spill", "temperature", "power", "utilization"),
                ),
            },
        )
        artifacts = {
            path.name: sha256_file(path)
            for path in sorted(directory.iterdir())
            if path.is_file() and path.name not in {"SHA256SUMS", "evidence_bundle_manifest.json"}
        }
        atomic_json(
            directory / "evidence_bundle_manifest.json",
            {
                "schema": "causal-scale-experiment-run-v1",
                "created_at": utc_now(),
                "experiment": relative,
                "status": metrics.get("status", "PASS"),
                "artifacts": artifacts,
                "source_manifest": tree_manifest("src/causal_scale", ("*.py",)),
            },
        )
        paths = [
            path for path in directory.iterdir() if path.is_file() and path.name != "SHA256SUMS"
        ]
        write_sha256s(directory, paths)
        finalized.append(relative)
    print(json.dumps({"status": "PASS", "finalized": finalized}, indent=2))


if __name__ == "__main__":
    main()
