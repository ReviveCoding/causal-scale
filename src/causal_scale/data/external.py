from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExternalDatasetStatus:
    dataset: str
    status: str
    reason: str


def default_external_statuses() -> list[ExternalDatasetStatus]:
    return [
        ExternalDatasetStatus("twins", "PENDING", "required when accessible"),
        ExternalDatasetStatus("ihdp", "PENDING", "recommended replication benchmark"),
        ExternalDatasetStatus(
            "hillstrom", "OPTIONAL", "cross-domain randomized marketing benchmark"
        ),
    ]
