from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pandas as pd

FEATURES = [f"f{i}" for i in range(12)]


class _HistoricalDistributedModel:
    def __init__(self, portfolio: dict[str, Any], model_id: str) -> None:
        self.portfolio, self.model_id = portfolio, model_id

    def predict_effect(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model_id == "HIST_C3":
            return np.asarray(self.portfolio["C3"].predict_effect(frame), dtype=float)
        if self.model_id == "HIST_E7":
            e7 = self.portfolio["E7"]
            blend = 0.7 * self.portfolio["C1"].predict_effect(frame) + 0.3 * self.portfolio[
                "C5"
            ].predict_effect(frame)
            return float(e7["affine_intercept"]) + float(e7["affine_slope"]) * blend
        raise KeyError(self.model_id)


def load_distributed_historical(filename: str, model_id: str) -> _HistoricalDistributedModel:
    import joblib
    from pyspark import SparkFiles

    portfolio = joblib.load(SparkFiles.get(filename))
    return _HistoricalDistributedModel(portfolio, model_id)


def force_cpu_model(model: Any) -> Any:
    """Keep Spark worker inference off a shared CUDA device."""
    for name in (
        "model_",
        "model0_",
        "model1_",
        "mu0_",
        "mu1_",
        "tau0_",
        "tau1_",
        "effect_model_",
    ):
        component = getattr(model, name, None)
        if component is not None and hasattr(component, "set_params"):
            with contextlib.suppress(ValueError):
                component.set_params(device="cpu")
    if hasattr(model, "device"):
        model.device = "cpu"
    return model


def load_distributed_joblib(filename: str, force_cpu: bool = True) -> Any:
    import joblib
    from pyspark import SparkFiles

    model = joblib.load(SparkFiles.get(filename))
    if force_cpu:
        model = force_cpu_model(model)
    return model


def score_map_in_pandas(frame, loader: Callable[[], Any], score_method: str = "predict_effect"):
    from pyspark.sql.types import DoubleType, LongType, StructField, StructType

    schema = StructType(
        [StructField("source_row_id", LongType(), False), StructField("score", DoubleType(), False)]
    )

    def iterator(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        model = loader()  # exactly once per worker iterator, never once per row
        method = getattr(model, score_method)
        for batch in batches:
            score = np.asarray(method(batch), dtype=float)
            yield pd.DataFrame(
                {"source_row_id": batch["source_row_id"].to_numpy(np.int64), "score": score}
            )

    return frame.select("source_row_id", *FEATURES).mapInPandas(iterator, schema)


def score_pandas_udf(frame, loader: Callable[[], Any], score_method: str = "predict_effect"):
    from pyspark.sql import functions as functions

    cache: list[Any] = []

    @functions.pandas_udf("double")  # type: ignore[call-overload]
    def score_udf(*columns: pd.Series) -> pd.Series:
        if not cache:
            cache.append(loader())
        batch = pd.DataFrame({name: column for name, column in zip(FEATURES, columns, strict=True)})
        return pd.Series(getattr(cache[0], score_method)(batch), dtype=float)

    return frame.withColumn("score", score_udf(*[functions.col(name) for name in FEATURES])).select(
        "source_row_id", "score"
    )


def score_scalar_udf(frame, predictor: Callable[[np.ndarray], float]):
    from pyspark.sql import functions as functions
    from pyspark.sql.types import DoubleType

    udf = functions.udf(
        lambda *values: float(predictor(np.asarray(values, dtype=np.float32))),
        DoubleType(),
        useArrow=False,
    )
    return frame.withColumn("score", udf(*[functions.col(name) for name in FEATURES])).select(
        "source_row_id", "score"
    )
