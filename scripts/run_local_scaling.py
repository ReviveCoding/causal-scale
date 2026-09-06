from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import pandas as pd

from causal_scale.evidence.manifest import atomic_json
from causal_scale.spark.artifacts import distribute_artifact
from causal_scale.spark.eventlog import event_log_for_application, summarize_event_log
from causal_scale.spark.scoring import load_distributed_joblib, score_map_in_pandas
from causal_scale.spark.session import create_session


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", type=Path, default=Path(".local-data/marts/full_features.parquet")
    )
    parser.add_argument("--model", type=Path, default=Path(".local-models/frozen/gpu_t.joblib"))
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, default=Path("evidence/systems/local_scaling"))
    args = parser.parse_args()
    records = []
    application_ids = []
    for workers in (1, 2, 4):
        master = f"local[{workers}]"
        spark = create_session(
            master, f"causal-scale-local-{workers}", Path(".local-spark/eventlog")
        )
        application_ids.append(spark.sparkContext.applicationId)
        try:
            name = distribute_artifact(spark, args.model)
            frame = spark.read.parquet(str(args.features.resolve())).where(
                f"source_row_id < {args.rows}"
            )
            start = time.perf_counter()
            observed = score_map_in_pandas(
                frame.repartition(workers * 4), partial(load_distributed_joblib, name)
            ).count()
            elapsed = time.perf_counter() - start
            records.append(
                {
                    "backend": "MAP_IN_PANDAS",
                    "operation_class": "SCORING_ONLY",
                    "rows": observed,
                    "threads_or_workers": workers,
                    "runtime": elapsed,
                    "rows_per_sec": observed / elapsed,
                }
            )
        finally:
            spark.stop()
    baseline = records[0]["runtime"]
    for record in records:
        record["speedup"] = baseline / record["runtime"]
        record["parallel_efficiency"] = record["speedup"] / record["threads_or_workers"]
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output / "systems_benchmark.csv", index=False)
    atomic_json(
        args.output / "metrics.json",
        {"status": "PASS", "same_machine_workers_not_nodes": True, "records": records},
    )
    summaries = [
        summarize_event_log(
            event_log_for_application(Path(".local-spark/eventlog"), application_id)
        )
        for application_id in application_ids
    ]
    atomic_json(args.output / "spark_application.json", {"applications": summaries})
    atomic_json(args.output / "stage_metrics.json", {"applications": summaries})
    atomic_json(
        args.output / "executor_metrics.json",
        {
            "applications": [
                {key: summary[key] for key in ("application_id", "executors", "spill_bytes")}
                for summary in summaries
            ]
        },
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
