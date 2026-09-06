from __future__ import annotations

import time
from collections.abc import Callable


def timed_count(factory: Callable[[], object], rows: int) -> dict[str, float | int]:
    start = time.perf_counter()
    frame = factory()
    count = frame.count()  # type: ignore[attr-defined]
    elapsed = time.perf_counter() - start
    if count != rows:
        raise ValueError(f"benchmark row mismatch: {count} != {rows}")
    return {"rows": rows, "runtime_seconds": elapsed, "rows_per_sec": rows / elapsed}
