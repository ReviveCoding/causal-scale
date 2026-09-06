from __future__ import annotations

import hmac
from pathlib import Path

from .hashing import sha256_file


def verify_bound_file(path: str | Path, expected_sha256: str, label: str) -> None:
    target = Path(path)
    if not target.is_file() or not hmac.compare_digest(
        sha256_file(target), expected_sha256.lower()
    ):
        raise ValueError(f"{label} hash mismatch")


def parse_sha256s(path: str | Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        if len(digest) != 64 or relative in result or ".." in Path(relative).parts:
            raise ValueError("malformed SHA256SUMS")
        result[relative] = digest.lower()
    return result


def verify_sha256s(root: str | Path, sums: str | Path | None = None) -> dict[str, object]:
    base = Path(root).resolve()
    sums_path = Path(sums) if sums else base / "SHA256SUMS"
    entries = parse_sha256s(sums_path)
    failures = []
    for relative, expected in entries.items():
        target = (base / relative).resolve()
        if (
            base not in target.parents
            or not target.is_file()
            or not hmac.compare_digest(sha256_file(target), expected)
        ):
            failures.append(relative)
    return {
        "status": "PASS" if not failures else "FAIL",
        "files_verified": len(entries),
        "failures": failures,
    }
