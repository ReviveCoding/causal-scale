from __future__ import annotations

import os
import sys
from pathlib import Path


def create_session(
    master: str = "local[2]", app_name: str = "causal-scale", event_log_dir: Path | None = None
):
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    local_hadoop = Path(".local-tools/hadoop-3.3.6").resolve()
    if (local_hadoop / "bin/winutils.exe").is_file():
        os.environ["HADOOP_HOME"] = str(local_hadoop)
        os.environ["PATH"] = f"{local_hadoop / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
    from pyspark.sql import SparkSession

    local_dir = Path(".local-spark/tmp").resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    builder = (
        SparkSession.builder.master(master)
        .appName(app_name)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.pythonUDF.arrow.enabled", "true")
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .config("spark.local.dir", str(local_dir))
    )
    if event_log_dir is not None:
        event_log_dir.mkdir(parents=True, exist_ok=True)
        builder = (
            builder.config("spark.eventLog.enabled", "true")
            .config("spark.eventLog.compress", "false")
            .config("spark.eventLog.rolling.enabled", "false")
            .config("spark.eventLog.dir", event_log_dir.resolve().as_uri())
        )
    return builder.getOrCreate()
