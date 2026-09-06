from __future__ import annotations

import numpy as np
import pandas as pd

from causal_scale.models.base import FEATURES, CausalEstimator, validate_training_frame


class EconMLCausalForest(CausalEstimator):
    model_id = "CAUSAL_FOREST"

    def __init__(self, seed: int = 20260717, n_jobs: int = 4, n_estimators: int = 400) -> None:
        self.seed, self.n_jobs, self.n_estimators = seed, n_jobs, n_estimators

    def fit(self, frame: pd.DataFrame, **kwargs: object) -> EconMLCausalForest:
        try:
            from econml.dml import CausalForestDML
        except ImportError as error:
            raise RuntimeError("EconML is unavailable") from error
        x, t, y = validate_training_frame(frame)
        self.model_ = CausalForestDML(
            n_estimators=self.n_estimators,
            min_samples_leaf=20,
            max_depth=None,
            discrete_treatment=True,
            cv=3,
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(y, t, X=x)
        self.mu0_mean_, self.mu1_mean_ = float(y[t == 0].mean()), float(y[t == 1].mean())
        return self

    def predict_effect(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            self.model_.effect(frame.loc[:, FEATURES].to_numpy(np.float32)), dtype=float
        )

    def predict_outcomes(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return np.full(len(frame), self.mu0_mean_), np.full(len(frame), self.mu1_mean_)

    def predict_uncertainty(self, frame: pd.DataFrame) -> np.ndarray:
        inference = self.model_.effect_inference(frame.loc[:, FEATURES].to_numpy(np.float32))
        return np.asarray(inference.stderr, dtype=float)
