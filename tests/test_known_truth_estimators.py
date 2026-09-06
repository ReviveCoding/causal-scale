from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from causal_scale.baselines import XGBRLearner, XGBTlearner, XGBXLearner


def fixture(
    effect: str, propensity: float = 0.5, rows: int = 5000
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(812)
    x = rng.normal(size=(rows, 12)).astype(np.float32)
    tau = {
        "zero": np.zeros(rows),
        "constant": np.full(rows, 0.15),
        "heterogeneous": 0.12 * np.tanh(x[:, 0]),
    }[effect]
    mu0 = 1 / (1 + np.exp(-(-1.0 + 0.4 * x[:, 1])))
    mu1 = np.clip(mu0 + tau, 0.01, 0.99)
    t = rng.binomial(1, propensity, rows)
    y = rng.binomial(1, np.where(t == 1, mu1, mu0))
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(12)])
    frame["treatment"], frame["visit"] = t, y
    return frame, tau


@pytest.mark.parametrize("model_class", [XGBTlearner, XGBXLearner, XGBRLearner])
def test_custom_estimators_are_finite_repeatable_and_directional(model_class) -> None:
    frame, truth = fixture("heterogeneous")
    params = {"device": "cpu", "n_estimators": 25, "max_depth": 3, "n_jobs": 1}
    first = model_class(**params).fit(frame).predict_effect(frame)
    second = model_class(**params).fit(frame).predict_effect(frame)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second, rtol=0, atol=0)
    assert np.corrcoef(first, truth)[0, 1] > 0.1


def test_tlearner_handles_imbalanced_treatment() -> None:
    frame, _ = fixture("constant", propensity=0.85)
    prediction = (
        XGBTlearner(device="cpu", n_estimators=20, max_depth=3, n_jobs=1)
        .fit(frame)
        .predict_effect(frame)
    )
    assert np.isfinite(prediction).all() and prediction.mean() > 0
