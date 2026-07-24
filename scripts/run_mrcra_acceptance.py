#!/usr/bin/env python3
"""Run repository-wide acceptance and write a hash-bound evidence manifest."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "outputs" / "mrcra_acceptance_manifest.json"
# Pytest-generated parameter IDs may contain spaces.  Capture lazily up to the
# status token instead of treating the node ID as a single shell-style word.
RESULT = re.compile(r"^(tests/.+?)\s+(PASSED|SKIPPED|FAILED|ERROR)(?:\s|$)")
TRANSIENT_SOURCE_NAMES = {
    ".DS_Store",
    ".pytest_cache",
    "__pycache__",
}
TRANSIENT_SOURCE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".swp",
    ".tmp",
}


def portable_text(value: str) -> str:
    """Remove machine-specific absolute prefixes from retained public evidence."""

    return value.replace(str(ROOT), ".").replace(str(Path.home()), "~")


def node_executable() -> str:
    resolved = shutil.which("node")
    if resolved:
        return resolved
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
    if bundled.is_file():
        return str(bundled)
    raise RuntimeError("Node.js is required for frontend acceptance")


def local_javascript_tool(relative: str) -> str:
    path = ROOT / "trackio_frontend" / "node_modules" / relative
    if not path.is_file():
        raise RuntimeError(f"frontend dependency is missing: {path}")
    return str(path)


def normalized_node(node: str) -> str:
    # Function/class identifiers cannot contain ``[``; parameter renderings can
    # contain spaces or nested brackets, so removing from the first bracket is
    # both simpler and complete.
    return node.split("[", 1)[0]


def is_durable_source(path: Path) -> bool:
    """Return whether a path is a reproducible project input."""

    return (
        not any(part in TRANSIENT_SOURCE_NAMES for part in path.parts)
        and path.suffix not in TRANSIENT_SOURCE_SUFFIXES
        and not path.name.endswith("~")
    )


def run(
    command_id: str, command: list[str], *, cwd: Path = ROOT,
    environment: dict[str, str] | None = None,
) -> dict:
    completed = subprocess.run(
        command, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
        env=None if environment is None else {**os.environ, **environment},
    )
    lines = completed.stdout.splitlines()
    statuses: dict[str, list[str]] = {}
    for line in lines:
        match = RESULT.match(line.strip())
        if match:
            statuses.setdefault(normalized_node(match.group(1)), []).append(match.group(2))
    passed = sorted(
        node for node, values in statuses.items()
        if values and all(value == "PASSED" for value in values)
    )
    return {
        "id": command_id,
        "argv": [portable_text(value) for value in command],
        "cwd": str(cwd.relative_to(ROOT) or "."),
        "exit_code": completed.returncode,
        "passed_nodeids": passed,
        "result_counts": {
            status.lower(): sum(value == status for values in statuses.values() for value in values)
            for status in ("PASSED", "SKIPPED", "FAILED", "ERROR")
        },
        "output_tail": [portable_text(line) for line in lines[-40:]],
    }


def source_hashes() -> dict[str, str]:
    roots = (
        ROOT / "src", ROOT / "tests", ROOT / "scripts", ROOT / "spec",
        ROOT / "trackio_frontend" / "src", ROOT / "trackio_frontend" / "public",
    )
    files = []
    for root in roots:
        if root.exists():
            files.extend(
                path for path in root.rglob("*")
                if path.is_file() and is_durable_source(path)
            )
    files.extend((
        ROOT / "pyproject.toml", ROOT / "requirements.txt",
        ROOT / "outputs" / "multimodal_relational_continuity_resonance_architecture.md",
        ROOT / "outputs" / "mrcra_empirical_acceptance.json",
        ROOT / "outputs" / "mrcra_integrated_acceptance.json",
        ROOT / "outputs" / "mrcra_performance_acceptance.json",
        ROOT / "outputs" / "pc_rasl_empirical_acceptance.json",
    ))
    return {
        str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(files)) if path.is_file()
    }


def main() -> int:
    node = node_executable()
    commands = [
        run(
            "python-tests", [sys.executable, "-m", "pytest", "-vv"],
            environment={"MRCRA_BUILDING_ACCEPTANCE": "1"},
        ),
        run("frontend-tests", [node, local_javascript_tool("vitest/vitest.mjs"), "run"], cwd=ROOT / "trackio_frontend"),
        run("frontend-lint", [node, local_javascript_tool("eslint/bin/eslint.js"), "src/"], cwd=ROOT / "trackio_frontend"),
        run("frontend-build", [node, local_javascript_tool("vite/bin/vite.js"), "build"], cwd=ROOT / "trackio_frontend"),
        run(
            "empirical-acceptance",
            [sys.executable, "scripts/run_mrcra_empirical_acceptance.py"],
        ),
        run(
            "integrated-acceptance",
            [sys.executable, "scripts/run_mrcra_integrated_acceptance.py"],
        ),
        run(
            "performance-acceptance",
            [sys.executable, "scripts/run_mrcra_performance_acceptance.py"],
        ),
        run(
            "pc-rasl-acceptance",
            [sys.executable, "scripts/run_pc_rasl_acceptance.py"],
        ),
    ]
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
        },
        "commands": commands,
        "source_sha256": source_hashes(),
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, DESTINATION)
    failed = [command["id"] for command in commands if command["exit_code"]]
    print(json.dumps({
        "artifact": str(DESTINATION.relative_to(ROOT)),
        "failed": failed,
        "python_passed_nodeids": len(commands[0]["passed_nodeids"]),
    }, indent=2))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
