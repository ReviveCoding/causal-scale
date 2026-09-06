"""Audit the publication candidate without traversing private/raw directories."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import unquote

MAX_PUBLIC_BYTES = 10 * 1024 * 1024

REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "CITATION.cff",
    "SECURITY.md",
    "pyproject.toml",
    "requirements.lock",
    ".github/workflows/ci.yml",
    "reports/FINAL_STATUS.md",
    "reports/TECHNICAL_REPORT.md",
    "reports/RESUME_EVIDENCE_LEDGER.md",
    "reports/RESULTS_ANALYSIS.md",
    "reports/RELEASE_NOTES_v1.0.0.md",
    "reports/PUBLIC_EVIDENCE_INDEX.md",
    "reports/PUBLICATION_READY.json",
)

PUBLIC_EVIDENCE = frozenset(
    {
        "evidence/final_audit/completion_gates.json",
        "evidence/prospective/final_audit.json",
        "evidence/spark/historical_parity/metrics.json",
        "evidence/systems/scale/metrics.json",
        "evidence/systems/local_scaling/metrics.json",
        "evidence/gpu/controlled_benchmark/metrics.json",
        "evidence/retrospective/model_benchmark/metrics.json",
        "evidence/retrospective/capacity_frontier/capacity_frontier.csv",
        "evidence/historical_reference/historical_metrics.json",
        "evidence/models/causal_forest_retry/metrics.json",
    }
)

PRIVATE_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".ci-wheel",
        ".codex_task",
        ".reference",
        ".local-data",
        ".local-cache",
        ".local-runs",
        ".local-models",
        ".local-spark",
        ".local-gpu",
        ".local-tools",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "dist",
        "htmlcov",
        "spark-warehouse",
        "metastore_db",
        ".sparkStaging",
    }
)

FORBIDDEN_FILE_SUFFIXES = (
    ".parquet",
    ".csv.gz",
    ".tsv.gz",
    ".jsonl",
    ".feather",
    ".arrow",
    ".npy",
    ".npz",
    ".joblib",
    ".pkl",
    ".pickle",
    ".onnx",
    ".dmp",
    ".core",
    ".zip",
    ".7z",
    ".tar",
    ".tgz",
    ".tar.gz",
    ".pem",
    ".key",
)

BINARY_PUBLIC_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf"})

LOCAL_ONLY_FILES = frozenset(
    {
        "reports/GIT_ISOLATION_RECEIPT.json",
        "reports/GITHUB_PUBLICATION_RECEIPT.json",
    }
)

CONTENT_PATTERNS = {
    "Windows user-specific absolute path": re.compile(
        r"(?i)(?:[a-z]:[\\/]+users[\\/]+[^\\/\s\"'<>]+)"
    ),
    "POSIX user-specific absolute path": re.compile(r"(?i)(?:/(?:home|users)/[^/\s\"'<>]+/)"),
    "explicit temporary path": re.compile(
        r"(?i)(?:[a-z]:[\\/]+users[\\/]+[^\\/]+[\\/]+appdata[\\/]+local[\\/]+temp[\\/]|"
        + "/"
        + "tmp"
        + "/)"
    ),
    "private key material": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "assigned credential": re.compile(
        r"""(?ix)
        \b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password)
        \b\s*[:=]\s*["'][^"'<>\s]{8,}["']
        """
    ),
    "email address": re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    "phone number": re.compile(
        r"(?<!\d)(?:\+?1[\s.-]?)?\(?[2-9]\d{2}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)"
    ),
    "machine hostname field": re.compile(
        r"""(?ix)
        ["'](?:host[_-]?name|computer[_-]?name)["']\s*:\s*
        ["'](?!localhost|<hostname>)[^"']+["']
        """
    ),
}

REQUIRED_IGNORE_RULES = frozenset(
    {
        ".venv/",
        "__pycache__/",
        "*.py[cod]",
        ".local-data/",
        ".local-cache/",
        ".local-runs/",
        ".local-models/",
        ".local-spark/",
        ".local-gpu/",
        ".local-tools/",
        ".reference/",
        ".codex_task/",
        "*.parquet",
        "*.csv.gz",
        "*.joblib",
        "*.pkl",
        "*.pickle",
        "*.onnx",
        "*.dmp",
    }
)


def normalized(path: Path) -> str:
    return path.as_posix().lstrip("./")


def is_private_directory(name: str) -> bool:
    lowered = name.lower()
    return (
        name in PRIVATE_DIRECTORY_NAMES
        or lowered.startswith(".venv")
        or lowered.endswith(".egg-info")
    )


def is_public_candidate(relative: Path) -> bool:
    path = normalized(relative)
    if path in LOCAL_ONLY_FILES:
        return False
    parts = relative.parts
    if any(is_private_directory(part) for part in parts[:-1]):
        return False
    if path.startswith("evidence/") and path not in PUBLIC_EVIDENCE:
        return False
    lowered = path.lower()
    if lowered.endswith(FORBIDDEN_FILE_SUFFIXES):
        return False
    if relative.name.lower() in {"derby.log", "credentials.json", "secrets.json"}:
        return False
    if relative.name.lower().startswith(("hs_err_pid", "replay_pid")):
        return False
    if relative.name == ".env" or relative.name.startswith(".env."):
        return relative.name == ".env.example"
    return True


def candidate_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for directory, names, files in os.walk(root):
        current = Path(directory)
        relative_directory = current.relative_to(root)
        names[:] = sorted(name for name in names if not is_private_directory(name))
        if normalized(relative_directory).startswith("evidence/"):
            names[:] = [
                name
                for name in names
                if any(
                    item.startswith(normalized(relative_directory / name).rstrip("/") + "/")
                    for item in PUBLIC_EVIDENCE
                )
            ]
        for name in sorted(files):
            relative = (current / name).relative_to(root)
            if is_public_candidate(relative):
                candidates.append(relative)
    return sorted(candidates, key=normalized)


def check_required(root: Path, failures: list[str]) -> None:
    for item in REQUIRED_FILES:
        if not (root / item).is_file():
            failures.append(f"required public file missing: {item}")


def check_gitignore(root: Path, failures: list[str]) -> None:
    ignore_path = root / ".gitignore"
    if not ignore_path.is_file():
        failures.append("required public file missing: .gitignore")
        return
    rules = {
        line.strip()
        for line in ignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for rule in sorted(REQUIRED_IGNORE_RULES - rules):
        failures.append(f".gitignore missing required rule: {rule}")
    for evidence_path in sorted(PUBLIC_EVIDENCE):
        if not (root / evidence_path).is_file():
            failures.append(f"selected public evidence missing: {evidence_path}")


def check_paths_and_sizes(root: Path, candidates: list[Path], failures: list[str]) -> None:
    for relative in candidates:
        path = normalized(relative)
        lowered = path.lower()
        if any(is_private_directory(part) for part in relative.parts):
            failures.append(f"forbidden directory is public: {path}")
        if lowered.endswith(FORBIDDEN_FILE_SUFFIXES):
            failures.append(f"forbidden raw/model artifact is public: {path}")
        if ("prospective" in lowered and "outcome" in lowered) and lowered.endswith(
            (".csv", ".jsonl", ".parquet", ".npy", ".npz")
        ):
            failures.append(f"prospective raw outcome candidate: {path}")
        size = (root / relative).stat().st_size
        if size > MAX_PUBLIC_BYTES:
            failures.append(f"public file exceeds 10 MiB ({size} bytes): {path}")


def check_contents(root: Path, candidates: list[Path], failures: list[str]) -> None:
    for relative in candidates:
        absolute = root / relative
        if absolute.suffix.lower() in BINARY_PUBLIC_SUFFIXES:
            continue
        try:
            text = absolute.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"unexpected non-text public file: {normalized(relative)}")
            continue
        for label, pattern in CONTENT_PATTERNS.items():
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{label} in {normalized(relative)}:{line}")


def markdown_link_targets(text: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", text)]


def check_markdown_links(root: Path, candidates: list[Path], failures: list[str]) -> None:
    for relative in candidates:
        if relative.suffix.lower() != ".md":
            continue
        text = (root / relative).read_text(encoding="utf-8")
        for target in markdown_link_targets(text):
            target = target.split(maxsplit=1)[0].strip("<>")
            lowered = target.lower()
            if (
                not target
                or target.startswith("#")
                or lowered.startswith(("http://", "https://", "mailto:"))
            ):
                continue
            target = unquote(target).split("#", 1)[0]
            resolved = (root / relative.parent / target).resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                failures.append(
                    f"Markdown link escapes repository in {normalized(relative)}: {target}"
                )
                continue
            if not resolved.exists():
                failures.append(f"broken local Markdown link in {normalized(relative)}: {target}")


def load_json(root: Path, relative: str, failures: list[str]) -> dict[str, object]:
    try:
        value = json.loads((root / relative).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"invalid JSON {relative}: {exc}")
        return {}
    if not isinstance(value, dict):
        failures.append(f"JSON root is not an object: {relative}")
        return {}
    return value


def check_evidence(root: Path, failures: list[str]) -> None:
    completion = load_json(root, "evidence/final_audit/completion_gates.json", failures)
    if completion.get("status") != "PASS":
        failures.append("completion gates are not PASS")
    gates = completion.get("gates")
    if not isinstance(gates, dict) or gates.get("PROSPECTIVE_FINAL") != "REPLICATED":
        failures.append("completion gates do not record prospective replication")

    prospective = load_json(root, "evidence/prospective/final_audit.json", failures)
    if prospective.get("status") != "PASS":
        failures.append("final prospective audit is not PASS")
    if prospective.get("result_status") != "REPLICATED":
        failures.append("final prospective audit is not REPLICATED")
    primary_gates = prospective.get("primary_gates")
    if not isinstance(primary_gates, dict) or not all(primary_gates.values()):
        failures.append("not all prospective primary gates passed")

    parity = load_json(root, "evidence/spark/historical_parity/metrics.json", failures)
    if parity.get("status") != "PASS":
        failures.append("historical Spark parity is not PASS")

    scale = load_json(root, "evidence/systems/scale/metrics.json", failures)
    if scale.get("status") != "PASS" or scale.get("outcomes_accessed") is not False:
        failures.append("systems scale is not PASS with outcomes_accessed=false")

    gpu = load_json(root, "evidence/gpu/controlled_benchmark/metrics.json", failures)
    qualification = gpu.get("qualification")
    if not isinstance(qualification, dict) or (
        qualification.get("status") != "GPU_XGBOOST_AVAILABLE"
    ):
        failures.append("GPU evidence does not record XGBoost CUDA availability")


def check_structured_files(root: Path, failures: list[str]) -> None:
    try:
        import yaml
    except ImportError:
        failures.append("PyYAML unavailable; cannot parse CI and citation files")
        return
    parsed: dict[str, dict[object, object]] = {}
    for relative in (".github/workflows/ci.yml", "CITATION.cff"):
        try:
            value = yaml.safe_load((root / relative).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            failures.append(f"invalid YAML {relative}: {exc}")
            continue
        if not isinstance(value, dict):
            failures.append(f"YAML root is not a mapping: {relative}")
            continue
        parsed[relative] = value

    workflow = parsed.get(".github/workflows/ci.yml", {})
    triggers = workflow.get("on", workflow.get(True))
    if not isinstance(triggers, dict):
        failures.append("CI triggers are not a mapping")
    else:
        for event in ("push", "pull_request", "workflow_dispatch"):
            if event not in triggers:
                failures.append(f"CI missing trigger: {event}")
        for event in ("push", "pull_request"):
            config = triggers.get(event)
            if not isinstance(config, dict) or "main" not in config.get("branches", []):
                failures.append(f"CI {event} trigger is not restricted to main")
    workflow_text = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    required_ci_fragments = (
        "requirements.lock",
        "ruff format --check",
        "ruff check",
        "mypy src",
        "test_spark_fixture.py",
        "python -m build",
        "audit_public_release.py --strict",
        'java-version: "17"',
    )
    for fragment in required_ci_fragments:
        if fragment not in workflow_text:
            failures.append(f"CI missing required public-safe step: {fragment}")
    forbidden_ci_fragments = (
        "run_prospective.py",
        "fetch_datasets.py",
        "nvidia-smi",
        ".local-data",
        ".reference",
    )
    for fragment in forbidden_ci_fragments:
        if fragment in workflow_text:
            failures.append(f"CI contains forbidden hosted operation: {fragment}")

    citation = parsed.get("CITATION.cff", {})
    citation_expected = {
        "cff-version": "1.2.0",
        "title": "CAUSAL-SCALE: Distributed Causal ML & Policy Evaluation at Scale",
        "version": "1.0.0",
        "repository-code": "https://github.com/ReviveCoding/causal-scale",
        "license": "MIT",
    }
    for key, expected_value in citation_expected.items():
        if str(citation.get(key)) != expected_value:
            failures.append(f"CITATION.cff field {key!r} is not {expected_value!r}")
    authors = citation.get("authors")
    if not isinstance(authors, list) or not authors:
        failures.append("CITATION.cff has no authors")


def check_receipt(root: Path, failures: list[str]) -> None:
    receipt = load_json(root, "reports/PUBLICATION_READY.json", failures)
    expected = {
        "status": "PASS",
        "release": "v1.0.0",
        "repository": "ReviveCoding/causal-scale",
        "scientific_results_modified": False,
        "prospective_rerun": False,
        "public_audit": "PASS",
        "qualification": "PASS",
        "evidence_verification": "PASS",
    }
    for key, expected_value in expected.items():
        if receipt.get(key) != expected_value:
            failures.append(f"publication receipt field {key!r} is not {expected_value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    failures: list[str] = []

    candidates = candidate_files(root)
    check_required(root, failures)
    check_gitignore(root, failures)
    check_paths_and_sizes(root, candidates, failures)
    check_contents(root, candidates, failures)
    check_markdown_links(root, candidates, failures)
    check_structured_files(root, failures)
    check_evidence(root, failures)
    check_receipt(root, failures)

    failures = sorted(set(failures))
    status = "PASS" if not failures else "FAIL"
    result = {
        "status": status,
        "strict": args.strict,
        "root": "<repo>",
        "public_threshold_bytes": MAX_PUBLIC_BYTES,
        "candidate_file_count": len(candidates),
        "selected_public_evidence_count": len(PUBLIC_EVIDENCE),
        "required_file_count": len(REQUIRED_FILES),
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
