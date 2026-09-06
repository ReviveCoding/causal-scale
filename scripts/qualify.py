from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from causal_scale.evidence.manifest import atomic_json


def run(command: list[str]) -> dict[str, object]:
    process = subprocess.run(command, capture_output=True, text=True)
    return {
        "command": command,
        "exit_code": process.returncode,
        "stdout": process.stdout[-10000:],
        "stderr": process.stderr[-10000:],
        "status": "PASS" if process.returncode == 0 else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("evidence/qualification/engineering"))
    args = parser.parse_args()
    python = sys.executable
    os.environ["UV_CACHE_DIR"] = str((Path(".local-cache") / "uv").resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    checks = {
        "ruff_format": run([python, "-m", "ruff", "format", "--check", "src", "tests", "scripts"]),
        "ruff": run([python, "-m", "ruff", "check", "src", "tests", "scripts"]),
        "mypy": run([python, "-m", "mypy", "src"]),
        "pytest": run(
            [python, "-m", "pytest", "-q", "--basetemp", ".local-runs/pytest-qualification"]
        ),
        "build": run(
            [python, "-m", "build", "--no-isolation", "--outdir", "build/qualification-dist"]
        ),
        "evidence": run([python, "scripts/verify_evidence.py"]),
    }
    wheels = sorted(Path("build/qualification-dist").glob("*.whl"))
    if wheels:
        smoke = Path("build/installed-wheel-smoke")
        workspace = Path.cwd().resolve()
        resolved_smoke = smoke.resolve()
        if workspace not in resolved_smoke.parents:
            raise ValueError(f"unsafe wheel smoke path: {resolved_smoke}")
        if smoke.exists():
            shutil.rmtree(smoke)
        checks["wheel_venv"] = run([python, "-m", "venv", str(smoke)])
        smoke_python = smoke / "Scripts/python.exe"
        uv = Path(".local-tools/uv/bin/uv.exe")
        install_command = (
            [str(uv), "pip", "install", "--python", str(smoke_python), "--no-deps", str(wheels[-1])]
            if uv.exists()
            else [str(smoke_python), "-m", "pip", "install", "--no-deps", str(wheels[-1])]
        )
        checks["wheel_install"] = run(install_command)
        checks["wheel_import"] = run(
            [str(smoke_python), "-c", "import causal_scale; print(causal_scale.__version__)"]
        )
    status = "PASS" if all(check["status"] == "PASS" for check in checks.values()) else "FAIL"
    report = {
        "schema": "causal-scale-engineering-qualification-v1",
        "status": status,
        "checks": checks,
    }
    atomic_json(args.output / "qualification_summary.json", report)
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
