from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from causal_scale.evidence.hashing import sha256_file

from .registry import DatasetSpec


def download_verified(spec: DatasetSpec, destination: str | Path) -> Path:
    if spec.url is None or spec.sha256 is None or spec.size_bytes is None:
        raise ValueError(f"dataset {spec.name} has no automatic authenticated source")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if (
        target.is_file()
        and target.stat().st_size == spec.size_bytes
        and sha256_file(target) == spec.sha256
    ):
        return target
    temporary: Path | None = None
    try:
        with (
            urllib.request.urlopen(spec.url, timeout=60) as response,
            tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out,
        ):
            temporary = Path(out.name)
            shutil.copyfileobj(response, out, length=1 << 20)
            out.flush()
            os.fsync(out.fileno())
        if temporary.stat().st_size != spec.size_bytes or sha256_file(temporary) != spec.sha256:
            raise ValueError(f"downloaded {spec.name} failed pinned identity")
        os.replace(temporary, target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
