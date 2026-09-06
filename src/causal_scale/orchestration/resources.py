from __future__ import annotations

import os
from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class ResourcePlan:
    cpu_workers: int
    available_ram_bytes: int
    gpu_parallelism: int


def plan_resources(profile: str, max_cpu_workers: int | None, gpu_parallelism: int) -> ResourcePlan:
    logical = os.cpu_count() or 1
    cap = max_cpu_workers or logical
    fraction = {"conservative": 0.25, "auto": 0.5, "aggressive": 0.75}[profile]
    workers = max(1, min(cap, int(logical * fraction)))
    return ResourcePlan(workers, psutil.virtual_memory().available, max(1, gpu_parallelism))
