from __future__ import annotations

import contextlib
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from causal_scale.evidence.hashing import sha256_file

EXPECTED_ARTIFACT_SHA256 = "047dbfd1f0ae48418b0de15e6f5da299a5d6fb575ecf81790afacf7a0a5df004"


@contextlib.contextmanager
def _historical_import_path(reference: Path) -> Iterator[None]:
    source = str((reference / "src").resolve())
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, source)
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        if source in sys.path:
            sys.path.remove(source)


class HistoricalPortfolioAdapter:
    def __init__(self, reference: str | Path = ".reference/ORBIT-Causal") -> None:
        self.reference = Path(reference)
        artifact = self.reference / "artifacts/frozen_candidate_portfolio.joblib"
        if sha256_file(artifact) != EXPECTED_ARTIFACT_SHA256:
            raise ValueError("historical artifact identity mismatch")
        with _historical_import_path(self.reference), warnings.catch_warnings():
            warnings.filterwarnings("error", category=UserWarning, message=".*unpickle estimator.*")
            self.portfolio: dict[str, Any] = joblib.load(artifact)
        if set(self.portfolio) != {"C1", "C3", "C5", "E7"}:
            raise ValueError("historical portfolio members mismatch")
        if self.portfolio["E7"]["weights"] != {"C1": 0.7, "C5": 0.3}:
            raise ValueError("historical E7 weights mismatch")

    @property
    def propensity(self) -> float:
        return float(self.portfolio["C1"].propensity_)

    def predict_effect(self, model_id: str, frame: pd.DataFrame) -> np.ndarray:
        if model_id == "HIST_C1":
            return np.asarray(self.portfolio["C1"].predict_effect(frame), dtype=float)
        if model_id == "HIST_C3":
            return np.asarray(self.portfolio["C3"].predict_effect(frame), dtype=float)
        if model_id in {"HIST_C5", "REFERENCE_STATE"}:
            return np.asarray(self.portfolio["C5"].predict_effect(frame), dtype=float)
        if model_id == "HIST_E7":
            e7 = self.portfolio["E7"]
            blend = 0.7 * self.predict_effect("HIST_C1", frame) + 0.3 * self.predict_effect(
                "HIST_C5", frame
            )
            return float(e7["affine_intercept"]) + float(e7["affine_slope"]) * blend
        if model_id == "HIST_PRE_FINAL_BLEND":
            return 0.7 * self.predict_effect("HIST_C1", frame) + 0.3 * self.predict_effect(
                "HIST_C5", frame
            )
        raise KeyError(model_id)

    def predict_outcomes(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mu0, mu1 = self.portfolio["C1"].predict_outcomes(frame)
        return np.asarray(mu0, dtype=float), np.asarray(mu1, dtype=float)

    def predict_uncertainty(self, model_id: str, frame: pd.DataFrame) -> np.ndarray:
        key = {"HIST_C1": "C1", "HIST_C3": "C3", "HIST_C5": "C5"}.get(model_id)
        if key is None:
            return np.zeros(len(frame), dtype=float)
        return np.asarray(self.portfolio[key].predict_uncertainty(frame), dtype=float)
