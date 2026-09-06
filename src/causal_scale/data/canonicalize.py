from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .manifests import REQUIRED_COLUMNS
from .prospective_guard import FEATURE_SIDE_COLUMNS, assert_contract_sealed
from .roles import (
    PROSPECTIVE_WINDOW,
    TOTAL_ROWS,
    priority_rank_prefix,
    priority_window,
)

DTYPES = {
    **{f"f{i}": "float64" for i in range(12)},
    "treatment": "int8",
    "conversion": "int8",
    "visit": "int8",
    "exposure": "int8",
}


def build_feature_mart(source: Path, output: Path, chunk_rows: int = 250_000) -> dict[str, object]:
    """Build a full-source feature-only systems mart.

    Treatment, exposure, and outcomes are deliberately excluded. Scientific
    outcome marts are built only for already-consumed historical roles.
    """
    output.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        feature_dtypes = {f"f{i}": "float64" for i in range(12)}
        for chunk in pd.read_csv(
            source,
            compression="gzip",
            dtype=feature_dtypes,
            usecols=list(feature_dtypes),
            chunksize=chunk_rows,
        ):
            chunk.insert(0, "source_row_id", np.arange(rows, rows + len(chunk), dtype=np.int64))
            rows += len(chunk)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    output / "full_features.parquet", table.schema, compression="zstd"
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if rows != TOTAL_ROWS:
        raise ValueError(f"canonical row count mismatch: {rows}")
    return {
        "rows": rows,
        "columns": list(FEATURE_SIDE_COLUMNS),
        "path": str((output / "full_features.parquet").resolve()),
        "outcome_fields_emitted": False,
    }


def build_historical_role_marts(
    source: Path, output: Path, chunk_rows: int = 250_000
) -> dict[str, object]:
    """Extract outcomes only for development and previously consumed holdouts."""
    output.mkdir(parents=True, exist_ok=True)
    prefix = priority_rank_prefix(TOTAL_ROWS, 2_750_000)
    selections = {
        "development": np.sort(prefix[:500_000]),
        "historical_250k": np.sort(prefix[500_000:750_000]),
        "historical_locked_2m": np.sort(prefix[750_000:2_750_000]),
    }
    masks = {name: np.zeros(TOTAL_ROWS, dtype=bool) for name in selections}
    for name, ids in selections.items():
        masks[name][ids] = True
    writers: dict[str, pq.ParquetWriter] = {}
    counts = {name: 0 for name in selections}
    rows_seen = 0
    try:
        for chunk in pd.read_csv(
            source,
            compression="gzip",
            dtype=DTYPES,
            usecols=list(REQUIRED_COLUMNS),
            chunksize=chunk_rows,
        ):
            ids = np.arange(rows_seen, rows_seen + len(chunk), dtype=np.int64)
            rows_seen += len(chunk)
            chunk.insert(0, "source_row_id", ids)
            for name, bitmap in masks.items():
                selected = chunk.loc[bitmap[ids]]
                if selected.empty:
                    continue
                table = pa.Table.from_pandas(selected, preserve_index=False)
                if name not in writers:
                    writers[name] = pq.ParquetWriter(
                        output / f"{name}.parquet", table.schema, compression="zstd"
                    )
                writers[name].write_table(table)
                counts[name] += len(selected)
    finally:
        for writer in writers.values():
            writer.close()
    expected = {name: len(ids) for name, ids in selections.items()}
    if rows_seen != TOTAL_ROWS or counts != expected:
        raise ValueError(f"historical role mart mismatch: rows_seen={rows_seen}, counts={counts}")
    return {"rows_seen": rows_seen, "roles": counts, "prospective_outcomes_emitted": False}


def build_prospective_feature_only(
    source: Path, output: Path, chunk_rows: int = 250_000
) -> dict[str, object]:
    offset, count = PROSPECTIVE_WINDOW
    selected_ids = priority_window(TOTAL_ROWS, count, offset)
    bitmap = np.zeros(TOTAL_ROWS, dtype=bool)
    bitmap[selected_ids] = True
    writer: pq.ParquetWriter | None = None
    rows_seen = selected_rows = 0
    try:
        with gzip.open(source, "rt", encoding="utf-8", newline="") as handle:
            for chunk in pd.read_csv(
                handle,
                usecols=[f"f{i}" for i in range(12)],
                dtype={f"f{i}": "float64" for i in range(12)},
                chunksize=chunk_rows,
            ):
                ids = np.arange(rows_seen, rows_seen + len(chunk), dtype=np.int64)
                rows_seen += len(chunk)
                chunk.insert(0, "source_row_id", ids)
                selected = chunk.loc[bitmap[ids]]
                if selected.empty:
                    continue
                table = pa.Table.from_pandas(selected, preserve_index=False)
                if writer is None:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(output, table.schema, compression="zstd")
                writer.write_table(table)
                selected_rows += len(selected)
    finally:
        if writer is not None:
            writer.close()
    if rows_seen != TOTAL_ROWS or selected_rows != count:
        raise ValueError("prospective feature-only seal construction failed")
    return {
        "rows": selected_rows,
        "columns": list(FEATURE_SIDE_COLUMNS),
        "path": str(output.resolve()),
    }


def extract_prospective_outcomes_once(
    source: Path, output: Path, contract_path: Path, digest_path: Path, chunk_rows: int = 250_000
) -> dict[str, object]:
    """Guarded one-shot extraction after the feature/policy seal is immutable."""
    contract_hash = assert_contract_sealed(contract_path, digest_path)
    if output.exists():
        raise PermissionError("refusing to replace an existing prospective outcome artifact")
    ids = priority_window(TOTAL_ROWS, PROSPECTIVE_WINDOW[1], PROSPECTIVE_WINDOW[0])
    bitmap = np.zeros(TOTAL_ROWS, dtype=bool)
    bitmap[ids] = True
    writer: pq.ParquetWriter | None = None
    rows_seen = selected_rows = 0
    try:
        for chunk in pd.read_csv(
            source,
            compression="gzip",
            dtype={"treatment": "int8", "visit": "int8", "conversion": "int8", "exposure": "int8"},
            usecols=["treatment", "visit", "conversion", "exposure"],
            chunksize=chunk_rows,
        ):
            source_ids = np.arange(rows_seen, rows_seen + len(chunk), dtype=np.int64)
            rows_seen += len(chunk)
            chunk.insert(0, "source_row_id", source_ids)
            selected = chunk.loc[bitmap[source_ids]]
            if selected.empty:
                continue
            table = pa.Table.from_pandas(selected, preserve_index=False)
            if writer is None:
                output.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
            selected_rows += len(selected)
    finally:
        if writer is not None:
            writer.close()
    if selected_rows != PROSPECTIVE_WINDOW[1]:
        raise ValueError("prospective outcome extraction row mismatch")
    return {"rows": selected_rows, "contract_sha256": contract_hash, "path": str(output.resolve())}
