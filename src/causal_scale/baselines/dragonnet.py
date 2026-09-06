from __future__ import annotations


class DragonNetUnavailable(RuntimeError):
    status = "SKIPPED_GPU_ENV"


def qualification_status() -> dict[str, str]:
    try:
        import jax  # noqa: F401
    except ImportError:
        return {
            "model": "DRAGONNET",
            "status": "SKIPPED_GPU_ENV",
            "reason": "optional JAX environment unavailable",
        }
    return {
        "model": "DRAGONNET",
        "status": "NOT_IMPLEMENTED_OPTIONAL",
        "reason": "optional baseline does not block P0",
    }
