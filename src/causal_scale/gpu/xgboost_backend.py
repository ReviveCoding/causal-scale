from __future__ import annotations

from typing import Any

import xgboost as xgb

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.9,
    "min_child_weight": 20,
    "tree_method": "hist",
    "n_jobs": 8,
}


def classifier(seed: int, device: str = "cpu", **overrides: Any) -> xgb.XGBClassifier:
    params = dict(DEFAULT_XGB_PARAMS)
    params.update(overrides)
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        device=device,
        **params,
    )


def regressor(seed: int, device: str = "cpu", **overrides: Any) -> xgb.XGBRegressor:
    params = dict(DEFAULT_XGB_PARAMS)
    params.update(overrides)
    return xgb.XGBRegressor(
        objective="reg:squarederror", eval_metric="rmse", random_state=seed, device=device, **params
    )
