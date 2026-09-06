from __future__ import annotations

from pathlib import Path

from causal_scale.evidence.manifest import write_sha256s
from causal_scale.evidence.verifier import verify_sha256s


def test_evidence_verifier_detects_mutation(tmp_path: Path) -> None:
    evidence = tmp_path / "metrics.json"
    evidence.write_text("{}\n", encoding="utf-8")
    sums = write_sha256s(tmp_path, [evidence])
    assert verify_sha256s(tmp_path, sums)["status"] == "PASS"
    evidence.write_text('{"changed":true}\n', encoding="utf-8")
    assert verify_sha256s(tmp_path, sums)["status"] == "FAIL"
