from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_scale.data.download import download_verified
from causal_scale.data.manifests import validate_criteo
from causal_scale.data.registry import CRITEO, DATASETS
from causal_scale.evidence.manifest import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--required", action="store_true")
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--dataset", choices=sorted(DATASETS))
    mode.add_argument("--verify-only", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path(".local-data/raw"))
    args = parser.parse_args()
    selected = (
        [CRITEO] if args.required or args.all or args.verify_only else [DATASETS[args.dataset]]
    )
    results = []
    for spec in selected:
        target = args.data_dir / spec.filename
        if not args.verify_only:
            download_verified(spec, target)
        try:
            manifest = (
                validate_criteo(target, verify_rows=True)
                if spec.name == CRITEO.name
                else {"dataset": spec.name, "status": "SKIPPED_RESOURCE_LIMIT"}
            )
        except FileNotFoundError:
            manifest = {
                "dataset": spec.name,
                "status": "DATA_BLOCKED_MANUAL_PROVIDER_ACTION",
                "manual_step": f"place the exact provider file at {target}",
            }
        results.append(manifest)
        atomic_json(Path("evidence/data") / spec.name / "dataset_manifest.json", manifest)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
