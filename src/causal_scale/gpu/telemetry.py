from __future__ import annotations

import subprocess
from datetime import UTC, datetime


def sample_gpu() -> list[dict[str, object]]:
    query = "index,memory.used,utilization.gpu,temperature.gpu,power.draw"
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    now = datetime.now(UTC).isoformat()
    return [
        {
            "timestamp": now,
            "index": int(parts[0]),
            "memory_used_mib": float(parts[1]),
            "utilization_pct": float(parts[2]),
            "temperature_c": float(parts[3]),
            "power_w": float(parts[4]),
        }
        for line in output.splitlines()
        if (parts := [part.strip() for part in line.split(",")])
    ]
