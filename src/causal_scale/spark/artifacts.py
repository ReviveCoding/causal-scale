from __future__ import annotations

from pathlib import Path


def distribute_artifact(spark, path: str | Path) -> str:
    artifact = Path(path).resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    spark.sparkContext.addFile(str(artifact))
    return artifact.name


def worker_artifact_path(filename: str) -> str:
    from pyspark import SparkFiles

    return SparkFiles.get(filename)
