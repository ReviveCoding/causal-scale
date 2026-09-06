from __future__ import annotations

import json

import numpy as np
import pandas as pd

from causal_scale.policy.engine import capacity_policy
from causal_scale.spark.policy import exact_capacity_policy
from causal_scale.spark.session import create_session


def main() -> None:
    frame = pd.DataFrame(
        {"source_row_id": np.arange(10_000), "score": np.repeat(np.arange(100), 100).astype(float)}
    )
    reference = capacity_policy(
        frame.score.to_numpy(), 0.2, tie_key=frame.source_row_id.to_numpy(), positive_only=False
    ).membership
    spark = create_session("local[2]", "causal-scale-fixture")
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
    mismatches = int(np.sum(actual != reference))
    print(
        json.dumps(
            {
                "status": "PASS" if mismatches == 0 else "FAIL",
                "rows": len(frame),
                "membership_mismatches": mismatches,
            }
        )
    )
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
