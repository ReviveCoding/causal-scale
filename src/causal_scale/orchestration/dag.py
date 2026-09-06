from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Phase:
    number: int
    name: str
    dependencies: tuple[str, ...]
    action: Callable[[], dict[str, object]]
    outcome_boundary: bool = False


class DAG:
    def __init__(self, phases: list[Phase]) -> None:
        self.phases = phases
        names = {phase.name for phase in phases}
        if len(names) != len(phases):
            raise ValueError("phase names must be unique")
        for phase in phases:
            if not set(phase.dependencies) <= names:
                raise ValueError(f"unknown dependency for {phase.name}")

    def ordered(self) -> list[Phase]:
        complete: set[str] = set()
        result: list[Phase] = []
        remaining = list(self.phases)
        while remaining:
            ready = [phase for phase in remaining if set(phase.dependencies) <= complete]
            if not ready:
                raise ValueError("DAG contains a cycle")
            for phase in sorted(ready, key=lambda item: item.number):
                result.append(phase)
                complete.add(phase.name)
                remaining.remove(phase)
        return result
