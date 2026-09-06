from pathlib import Path

from causal_scale.evidence.hashing import sha256_file


def test_frozen_historical_artifact_identity() -> None:
    path = Path(".reference/ORBIT-Causal/artifacts/frozen_candidate_portfolio.joblib")
    assert sha256_file(path) == "047dbfd1f0ae48418b0de15e6f5da299a5d6fb575ecf81790afacf7a0a5df004"
