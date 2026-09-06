from __future__ import annotations

import json
from pathlib import Path

from causal_scale.data.roles import role_manifest
from causal_scale.evidence.manifest import atomic_json


def main() -> None:
    manifest = role_manifest()
    target = Path("evidence/data/row_roles/role_manifest.json")
    atomic_json(target, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
