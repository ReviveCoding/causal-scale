from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    role: str
    url: str | None
    filename: str
    rows: int | None
    size_bytes: int | None
    sha256: str | None
    required: bool


CRITEO = DatasetSpec(
    name="criteo_v2_1",
    role="randomized_causal_benchmark_and_systems_workload",
    url="https://criteostorage.blob.core.windows.net/criteo-research-datasets/criteo-uplift-v2.1.csv.gz",
    filename="criteo-research-uplift-v2.1.csv.gz",
    rows=13_979_592,
    size_bytes=311_422_618,
    sha256="2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc",
    required=True,
)

DATASETS = {CRITEO.name: CRITEO}
