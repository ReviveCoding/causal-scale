from __future__ import annotations


def criteo_schema():
    from pyspark.sql.types import ByteType, DoubleType, LongType, StructField, StructType

    return StructType(
        [StructField("source_row_id", LongType(), False)]
        + [StructField(f"f{i}", DoubleType(), False) for i in range(12)]
        + [
            StructField(name, ByteType(), False)
            for name in ("treatment", "conversion", "visit", "exposure")
        ]
    )


def feature_schema():
    from pyspark.sql.types import DoubleType, LongType, StructField, StructType

    return StructType(
        [StructField("source_row_id", LongType(), False)]
        + [StructField(f"f{i}", DoubleType(), False) for i in range(12)]
    )
