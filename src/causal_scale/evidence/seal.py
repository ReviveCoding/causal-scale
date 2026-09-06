from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from .hashing import sha256_file, stable_hash, tree_manifest
from .manifest import atomic_json, utc_now


class ProspectiveState(StrEnum):
    CREATED = "CREATED"
    IMPLEMENTATION_FROZEN = "IMPLEMENTATION_FROZEN"
    SEALED_FEATURES_AND_POLICIES = "SEALED_FEATURES_AND_POLICIES"
    OUTCOME_ACCESS_STARTED = "OUTCOME_ACCESS_STARTED"
    COMPLETED = "COMPLETED"
    TERMINAL_FAILURE = "TERMINAL_FAILURE"


TRANSITIONS = {
    ProspectiveState.CREATED: {ProspectiveState.IMPLEMENTATION_FROZEN},
    ProspectiveState.IMPLEMENTATION_FROZEN: {ProspectiveState.SEALED_FEATURES_AND_POLICIES},
    ProspectiveState.SEALED_FEATURES_AND_POLICIES: {ProspectiveState.OUTCOME_ACCESS_STARTED},
    ProspectiveState.OUTCOME_ACCESS_STARTED: {
        ProspectiveState.COMPLETED,
        ProspectiveState.TERMINAL_FAILURE,
    },
    ProspectiveState.COMPLETED: set(),
    ProspectiveState.TERMINAL_FAILURE: set(),
}


def freeze_contract(workspace: Path, payload: dict[str, Any], output: Path) -> str:
    bound = dict(payload)
    bound["schema"] = "causal-scale-scientific-contract-v1"
    bound["created_at"] = utc_now()
    bound["source_manifest"] = tree_manifest(workspace / "src", ("*.py",))
    bound["config_manifest"] = tree_manifest(workspace / "configs")
    bound["contract_content_sha256"] = stable_hash(
        {k: v for k, v in bound.items() if k != "created_at"}
    )
    atomic_json(output, bound)
    digest = sha256_file(output)
    output.with_name("SCIENTIFIC_CONTRACT_SHA256.txt").write_text(digest + "\n", encoding="ascii")
    return digest


def transition(current: ProspectiveState, target: ProspectiveState) -> ProspectiveState:
    if target not in TRANSITIONS[current]:
        raise ValueError(f"invalid prospective transition: {current} -> {target}")
    return target
