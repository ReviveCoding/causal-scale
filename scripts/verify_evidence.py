from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_scale.evidence.verifier import verify_sha256s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("evidence"))
    args = parser.parse_args()
    reports = {}
    for sums in sorted(args.root.rglob("SHA256SUMS")):
        reports[str(sums.parent)] = verify_sha256s(sums.parent, sums)
    failed = {path: report for path, report in reports.items() if report["status"] != "PASS"}
    result = {"status": "PASS" if not failed else "FAIL", "manifests": reports}
    print(json.dumps(result, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
