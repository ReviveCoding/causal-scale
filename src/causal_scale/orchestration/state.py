from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PhaseState:
    status: str = "PENDING"
    input_hash: str | None = None
    output_hashes: dict[str, str] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)


class DurableState:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.phases: dict[str, PhaseState] = {}
        if self.path.is_file():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.phases = {
                name: PhaseState(**value) for name, value in raw.get("phases", {}).items()
            }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                json.dump(
                    {
                        "schema": "causal-scale-dag-state-v1",
                        "phases": {name: asdict(value) for name, value in self.phases.items()},
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def input_hash(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()

    def valid(self, name: str, input_hash: str, outputs: dict[str, str]) -> bool:
        phase = self.phases.get(name)
        return bool(
            phase
            and phase.status == "PASS"
            and phase.input_hash == input_hash
            and phase.output_hashes == outputs
        )

    def record(
        self, name: str, status: str, input_hash: str, output_hashes: dict[str, str], **detail: Any
    ) -> None:
        self.phases[name] = PhaseState(status, input_hash, output_hashes, detail)
        self.save()
