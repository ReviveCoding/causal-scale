from __future__ import annotations

import json
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from causal_scale.models.historical_adapter import HistoricalPortfolioAdapter
from causal_scale.spark.artifacts import distribute_artifact
from causal_scale.spark.scoring import load_distributed_historical, score_map_in_pandas
from causal_scale.spark.session import create_session


def main() -> None:
    pandas = pd.read_parquet(".local-data/marts/historical_locked_2m.parquet").iloc[:10_000]
    spark = create_session(
        "local[2]", "causal-scale-historical-scoring-fixture", Path(".local-spark/eventlog")
    )
    try:
        source_zip = Path(".local-spark/orbit_causal_source")
        archive = Path(shutil.make_archive(str(source_zip), "zip", ".reference/ORBIT-Causal/src"))
        spark.sparkContext.addPyFile(str(archive.resolve()))
        artifact_name = distribute_artifact(
            spark, Path(".reference/ORBIT-Causal/artifacts/frozen_candidate_portfolio.joblib")
        )
        observed = (
            score_map_in_pandas(
                spark.createDataFrame(pandas).repartition(4),
                partial(load_distributed_historical, artifact_name, "HIST_E7"),
            )
            .orderBy("source_row_id")
            .toPandas()
        )
    finally:
        spark.stop()
    expected = HistoricalPortfolioAdapter().predict_effect("HIST_E7", pandas)
    error = float(np.max(np.abs(observed.score.to_numpy() - expected)))
    report = {"status": "PASS" if error <= 1e-12 else "FAIL", "rows": len(observed), "error": error}
    print(json.dumps(report))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
