from __future__ import annotations

import json
from pathlib import Path


def summarize_event_log(path: str | Path) -> dict[str, object]:
    events = 0
    jobs: set[int] = set()
    stages: set[int] = set()
    tasks = 0
    shuffle_read = shuffle_write = spill = 0
    application_id = spark_version = None
    executors: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            events += 1
            kind = event.get("Event")
            if kind == "SparkListenerApplicationStart":
                application_id = event.get("App ID")
            elif kind == "SparkListenerEnvironmentUpdate":
                spark_version = event.get("Spark Properties", {}).get("spark.version")
            elif kind == "SparkListenerJobStart":
                jobs.add(int(event["Job ID"]))
            elif kind == "SparkListenerExecutorAdded":
                executors.add(str(event.get("Executor ID")))
            elif kind == "SparkListenerStageCompleted":
                stages.add(int(event["Stage Info"]["Stage ID"]))
            elif kind == "SparkListenerTaskEnd":
                tasks += 1
                metrics = event.get("Task Metrics", {})
                shuffle_read += int(
                    metrics.get("Shuffle Read Metrics", {}).get("Remote Bytes Read", 0)
                ) + int(metrics.get("Shuffle Read Metrics", {}).get("Local Bytes Read", 0))
                shuffle_write += int(
                    metrics.get("Shuffle Write Metrics", {}).get("Shuffle Bytes Written", 0)
                )
                spill += int(metrics.get("Memory Bytes Spilled", 0)) + int(
                    metrics.get("Disk Bytes Spilled", 0)
                )
    return {
        "application_id": application_id,
        "spark_version": spark_version,
        "events": events,
        "jobs": len(jobs),
        "stages": len(stages),
        "tasks": tasks,
        "executors": len(executors) or 1,
        "shuffle_read_bytes": shuffle_read,
        "shuffle_write_bytes": shuffle_write,
        "spill_bytes": spill,
    }


def event_log_for_application(directory: str | Path, application_id: str) -> Path:
    candidates = sorted(
        path for path in Path(directory).glob(f"{application_id}*") if path.is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"missing Spark event log for {application_id}")
    return candidates[-1]
