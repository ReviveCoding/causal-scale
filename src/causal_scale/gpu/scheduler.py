from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUJob:
    name: str
    expected_peak_mib: int


def schedulable_batches(
    jobs: list[GPUJob],
    vram_mib: int,
    memory_fraction: float = 0.8,
    pilot_parallel_faster: bool = False,
) -> list[list[GPUJob]]:
    if not 0 < memory_fraction <= 0.9:
        raise ValueError("GPU memory fraction must be in (0,0.9]")
    budget = int(vram_mib * memory_fraction)
    if not pilot_parallel_faster:
        return [[job] for job in jobs]
    batches: list[list[GPUJob]] = []
    for job in jobs:
        placed = False
        for batch in batches:
            if sum(item.expected_peak_mib for item in batch) + job.expected_peak_mib <= budget:
                batch.append(job)
                placed = True
                break
        if not placed:
            batches.append([job])
    return batches
