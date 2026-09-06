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

SIZES = (100_000, 250_000, 1_000_000, 2_000_000, 13_979_592)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", type=Path, default=Path(".local-data/marts/full_features.parquet")
    )
    parser.add_argument("--master", default="local[4]")
    parser.add_argument("--model", type=Path, default=Path(".local-models/frozen/gpu_t.joblib"))
    parser.add_argument("--output", type=Path, default=Path("evidence/systems/scale"))
    args = parser.parse_args()
    spark = create_session(
        args.master, "causal-scale-feature-only-scale", Path(".local-spark/eventlog")
    )
    application_id = spark.sparkContext.applicationId
    rows = []
    try:
        full = (
            spark.read.parquet(str(args.features.resolve()))
            .select("source_row_id", *[f"f{i}" for i in range(12)])
            .cache()
        )
        total = full.count()
        if total != SIZES[-1]:
            raise ValueError(f"authentic scale row mismatch: {total}")
        artifact_name = distribute_artifact(spark, args.model)
        loader = partial(load_distributed_joblib, artifact_name)
        for size in SIZES:
            start = time.perf_counter()
            batch = full.where(full.source_row_id < size)
            scored = score_map_in_pandas(batch, loader)
            observed = scored.count()
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "backend": "SPARK_MAP_IN_PANDAS",
                    "operation_class": "SCORING_ONLY",
                    "rows": observed,
                    "threads_or_workers": args.master,
                    "runtime": elapsed,
                    "rows_per_sec": observed / elapsed,
                    "speedup": None,
                    "parallel_efficiency": None,
                    "memory": None,
                    "shuffle": None,
                    "spill": None,
                    "parity_status": "MODEL_SCORED",
                }
            )
        args.output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output / "systems_benchmark.csv", index=False)
        atomic_json(
            args.output / "metrics.json",
            {
                "status": "PASS",
                "authentic_rows": total,
                "direct_1m": next(row for row in rows if row["rows"] == 1_000_000),
                "full_13_98m": rows[-1],
                "operation_class": "SCORING_ONLY",
                "outcomes_accessed": False,
            },
        )
        print(json.dumps(rows, indent=2))
    finally:
        spark.stop()
    event_dir = Path(".local-spark/eventlog")
    summary = summarize_event_log(event_log_for_application(event_dir, application_id))
    atomic_json(args.output / "spark_application.json", summary)
    atomic_json(args.output / "stage_metrics.json", summary)
    atomic_json(
        args.output / "executor_metrics.json",
        {key: summary[key] for key in ("application_id", "executors", "spill_bytes")},
    )


if __name__ == "__main__":
    main()
