from __future__ import annotations

import csv
import gzip
from pathlib import Path

from causal_scale.evidence.hashing import sha256_file

from .registry import CRITEO, DatasetSpec

REQUIRED_COLUMNS = tuple(
    [f"f{i}" for i in range(12)] + ["treatment", "conversion", "visit", "exposure"]
)


def validate_criteo(path: str | Path, *, verify_rows: bool = True) -> dict[str, object]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    size = source.stat().st_size
    digest = sha256_file(source)
    if size != CRITEO.size_bytes or digest != CRITEO.sha256:
        raise ValueError("official Criteo v2.1 size/SHA-256 mismatch")
    rows = 0
    try:
        with gzip.open(source, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if len(header) != len(set(header)) or set(header) != set(REQUIRED_COLUMNS):
                raise ValueError("Criteo schema mismatch")
            if verify_rows:
                rows = sum(1 for _ in reader)
    except (OSError, EOFError, gzip.BadGzipFile, UnicodeDecodeError, csv.Error) as error:
        raise ValueError("Criteo gzip integrity validation failed") from error
    if verify_rows and rows != CRITEO.rows:
        raise ValueError(f"Criteo row-count mismatch: {rows}")
    return {
        "schema": "causal-scale-dataset-manifest-v1",
        "dataset": CRITEO.name,
        "source": "official_provider",
        "path": str(source.resolve()),
        "rows": rows if verify_rows else CRITEO.rows,
        "size_bytes": size,
        "sha256": digest,
        "gzip_integrity": "PASS",
        "schema_status": "PASS",
        "status": "PASS",
    }


def dataset_spec_manifest(spec: DatasetSpec) -> dict[str, object]:
    return spec.__dict__.copy()
