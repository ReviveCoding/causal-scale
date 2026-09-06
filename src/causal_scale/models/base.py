from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

FEATURES = tuple(f"f{i}" for i in range(12))


class CausalEstimator(ABC):
    model_id: str

    @abstractmethod
    def fit(self, frame: pd.DataFrame, **kwargs: Any) -> CausalEstimator: ...

    @abstractmethod
    def predict_effect(self, frame: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def predict_outcomes(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]: ...

    def predict_uncertainty(self, frame: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(frame), dtype=float)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> CausalEstimator:
        result = joblib.load(path)
        if not isinstance(result, CausalEstimator):
            raise TypeError("artifact is not a CAUSAL-SCALE estimator")
        return result


def validate_training_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = set(FEATURES) | {"treatment", "visit"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"training frame is missing columns: {missing}")
    x = frame.loc[:, FEATURES].to_numpy(dtype=np.float32)
    t = frame["treatment"].to_numpy(dtype=np.int8)
    y = frame["visit"].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isin(t, [0, 1]).all():
        raise ValueError("training data must be finite with binary treatment")
    if set(np.unique(t)) != {0, 1}:
        raise ValueError("both treatment arms are required")
    return x, t, y
