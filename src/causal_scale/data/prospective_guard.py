from __future__ import annotations

from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = tuple(f"f{i}" for i in range(12))
FEATURE_SIDE_COLUMNS = ("source_row_id", *FEATURE_COLUMNS)
FORBIDDEN_COLUMNS = frozenset({"visit", "conversion", "exposure", "outcome", "treatment"})


class OutcomeAccessError(PermissionError):
    pass


def assert_feature_only(frame: pd.DataFrame) -> None:
    forbidden = sorted(FORBIDDEN_COLUMNS.intersection(frame.columns))
    if forbidden:
        raise OutcomeAccessError(
            f"prospective feature-side data contains forbidden fields: {forbidden}"
        )
    missing = sorted(set(FEATURE_SIDE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"prospective feature-side data is missing fields: {missing}")


def assert_contract_sealed(contract_path: Path, digest_path: Path) -> str:
    from causal_scale.evidence.hashing import sha256_file

    if not contract_path.is_file() or not digest_path.is_file():
        raise OutcomeAccessError("scientific contract is not sealed")
    expected = digest_path.read_text(encoding="ascii").strip()
    actual = sha256_file(contract_path)
    if expected != actual:
        raise OutcomeAccessError("scientific contract hash mismatch")
    return actual
