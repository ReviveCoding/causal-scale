from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_scale.policy.engine import capacity_policy
from causal_scale.spark.policy import exact_capacity_policy
from causal_scale.spark.session import create_session


@pytest.mark.spark
def test_pandas_spark_policy_parity(tmp_path) -> None:
    pytest.importorskip("pyspark")
    frame = pd.DataFrame(
        {
            "source_row_id": np.arange(100, dtype=np.int64),
            "score": np.repeat(np.arange(10), 10).astype(float),
        }
    )
    reference = capacity_policy(
        frame.score.to_numpy(), 0.2, tie_key=frame.source_row_id.to_numpy(), positive_only=False
    ).membership
    spark = create_session("local[2]", "causal-scale-test")
    try:
        actual = (
            exact_capacity_policy(spark.createDataFrame(frame), 0.2, positive_only=False)
            .orderBy("source_row_id")
            .select("policy")
            .toPandas()
            .policy.to_numpy()
        )
    finally:
        spark.stop()
    np.testing.assert_array_equal(actual, reference)
