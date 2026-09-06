from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_scale.data.canonicalize import (
    build_feature_mart,
    build_historical_role_marts,
    build_prospective_feature_only,
)
from causal_scale.data.manifests import validate_criteo
from causal_scale.evidence.hashing import sha256_file
from causal_scale.evidence.manifest import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=Path(".local-data/raw/criteo-research-uplift-v2.1.csv.gz")
    )
    parser.add_argument("--output", type=Path, default=Path(".local-data/marts"))
    args = parser.parse_args()
    source_manifest = validate_criteo(args.data)
    full = build_feature_mart(args.data, args.output)
    historical = build_historical_role_marts(args.data, args.output)
    prospective = build_prospective_feature_only(
        args.data, args.output / "prospective_features.parquet"
    )
    result = {
        "source": source_manifest,
        "full_feature_mart": full,
        "historical_role_marts": historical,
        "prospective_feature_mart": prospective,
    }
    for item in (full, prospective):
        item["sha256"] = sha256_file(Path(item["path"]))
    for name in historical["roles"]:
        path = args.output / f"{name}.parquet"
        result.setdefault("historical_artifacts", {})[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    atomic_json(Path("evidence/data/criteo_marts/run_manifest.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
