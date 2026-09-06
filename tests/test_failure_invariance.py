from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

import causal_scale.data.manifests as manifests
from causal_scale.data.registry import DatasetSpec
from causal_scale.evidence.hashing import row_ids_sha256, sha256_file
from causal_scale.evidence.verifier import verify_bound_file, verify_sha256s
from causal_scale.orchestration.state import DurableState
from causal_scale.policy.engine import capacity_policy


def fixture_spec(path: Path, rows: int) -> DatasetSpec:
    return DatasetSpec(
        "fixture", "test", None, path.name, rows, path.stat().st_size, sha256_file(path), True
    )


def test_wrong_criteo_sha_blocks(tmp_path: Path) -> None:
    path = tmp_path / "wrong.gz"
    path.write_bytes(b"not the official payload")
    with pytest.raises(ValueError, match="size/SHA-256 mismatch"):
        manifests.validate_criteo(path)


def test_corrupt_gzip_blocks_after_identity_check(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "corrupt.gz"
    path.write_bytes(b"not-gzip")
    monkeypatch.setattr(manifests, "CRITEO", fixture_spec(path, 1))
    with pytest.raises(ValueError, match="gzip integrity"):
        manifests.validate_criteo(path)


def test_schema_mismatch_blocks(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "schema.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("wrong,columns\n1,2\n")
    monkeypatch.setattr(manifests, "CRITEO", fixture_spec(path, 1))
    with pytest.raises(ValueError, match="schema mismatch"):
        manifests.validate_criteo(path)


def test_duplicate_row_id_blocks() -> None:
    with pytest.raises(ValueError, match="unique"):
        row_ids_sha256(np.array([1, 1], dtype=np.int64))


@pytest.mark.parametrize("label", ["model", "calibrator", "config"])
def test_bound_hash_mismatch_blocks(tmp_path: Path, label: str) -> None:
    path = tmp_path / label
    path.write_text("content", encoding="utf-8")
    with pytest.raises(ValueError, match=f"{label} hash mismatch"):
        verify_bound_file(path, "0" * 64, label)


def test_partial_artifact_is_invalid(tmp_path: Path) -> None:
    (tmp_path / "SHA256SUMS").write_text(f"{'0' * 64}  missing.json\n", encoding="utf-8")
    assert verify_sha256s(tmp_path)["status"] == "FAIL"


def test_stale_cache_recomputes(tmp_path: Path) -> None:
    state = DurableState(tmp_path / "state.json")
    state.record("phase", "PASS", "old", {"artifact": "old"})
    assert not state.valid("phase", "new", {"artifact": "new"})


def test_input_order_does_not_change_numeric_tie_science() -> None:
    ids = np.arange(100, dtype=np.int64)
    scores = np.repeat(np.arange(10), 10).astype(float)
    expected = capacity_policy(scores, 0.2, tie_key=ids, positive_only=False).membership
    permutation = np.random.default_rng(7).permutation(len(ids))
    shuffled = capacity_policy(
        scores[permutation], 0.2, tie_key=ids[permutation], positive_only=False
    ).membership
    restored = np.empty_like(shuffled)
    restored[permutation] = shuffled
    np.testing.assert_array_equal(restored, expected)
