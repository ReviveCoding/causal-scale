from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import joblib
import pandas as pd

from causal_scale.evidence.manifest import atomic_json
from causal_scale.spark.artifacts import distribute_artifact
from causal_scale.spark.eventlog import event_log_for_application, summarize_event_log
from causal_scale.spark.scoring import (
    force_cpu_model,
    load_distributed_joblib,
    score_map_in_pandas,
    score_pandas_udf,
    score_scalar_udf,
)
from causal_scale.spark.session import create_session


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", type=Path, default=Path(".local-data/marts/full_features.parquet")
    )
    parser.add_argument("--model", type=Path, default=Path(".local-models/frozen/gpu_t.joblib"))
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("evidence/systems/spark_backends"))
    args = parser.parse_args()
    columns = [f"f{i}" for i in range(12)]
    pandas_frame = pd.read_parquet(args.features, columns=["source_row_id", *columns]).iloc[
        : args.rows
    ]
    model = joblib.load(args.model)
    model = force_cpu_model(model)
    records = []
    for trial in range(args.trials):
        start = time.perf_counter()
        model.predict_effect(pandas_frame)
        elapsed = time.perf_counter() - start
        records.append(
            {
                "backend": "S0_PANDAS_NUMPY",
                "trial": trial,
                "rows": args.rows,
                "runtime": elapsed,
                "rows_per_sec": args.rows / elapsed,
            }
        )
    spark = create_session("local[4]", "causal-scale-backends", Path(".local-spark/eventlog"))
    application_id = spark.sparkContext.applicationId
    try:
        frame = spark.createDataFrame(pandas_frame).repartition(8).cache()
        frame.count()
        name = distribute_artifact(spark, args.model)
        loader = partial(load_distributed_joblib, name)

        def predictor(values):
            return model.predict_effect(pd.DataFrame([values], columns=columns))[0]

        factories = {
            "S1_SCALAR_UDF": lambda: score_scalar_udf(frame, predictor),
            "S2_PANDAS_UDF": lambda: score_pandas_udf(frame, loader),
            "S3_MAP_IN_PANDAS": lambda: score_map_in_pandas(frame, loader),
        }
        for backend, factory in factories.items():
            for trial in range(args.trials):
                start = time.perf_counter()
                observed = factory().count()
                elapsed = time.perf_counter() - start
                records.append(
                    {
                        "backend": backend,
                        "trial": trial,
                        "rows": observed,
                        "runtime": elapsed,
                        "rows_per_sec": observed / elapsed,
                    }
                )
    finally:
        spark.stop()
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output / "systems_benchmark.csv", index=False)
    atomic_json(
        args.output / "metrics.json",
        {"status": "PASS", "trials": args.trials, "rows": args.rows, "records": records},
    )
    event_dir = Path(".local-spark/eventlog")
    summary = summarize_event_log(event_log_for_application(event_dir, application_id))
    atomic_json(args.output / "spark_application.json", summary)
    atomic_json(args.output / "stage_metrics.json", summary)
    atomic_json(
        args.output / "executor_metrics.json",
        {key: summary[key] for key in ("application_id", "executors", "spill_bytes")},
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
