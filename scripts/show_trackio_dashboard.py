#!/usr/bin/env python3
"""Launch Trackio with the MRCRA Cognition and Spectral Network tabs installed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import trackio


ROOT = Path(__file__).resolve().parents[1]
PACKAGED_FRONTEND = ROOT / "src" / "mrrn" / "trackio_frontend"


def _discover_completed_run(search_root: Path = ROOT) -> Path | None:
    """Find a recovery source without importing the PyTorch model package."""

    candidates: list[tuple[int, int, Path]] = []
    for parent_name in ("work", "outputs"):
        parent = search_root / parent_name
        if not parent.is_dir():
            continue
        for manifest_path in parent.rglob("run_manifest.json"):
            try:
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                final = manifest["final_training_state"]
                completed = manifest["completed"] is True
                tokens = int(final["tokens_seen"])
            except (
                OSError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                continue
            if completed:
                candidates.append(
                    (
                        tokens,
                        manifest_path.stat().st_mtime_ns,
                        manifest_path.parent,
                    )
                )
    return (
        max(candidates, key=lambda item: (item[0], item[1]))[2]
        if candidates else None
    )


def _prepare_runtime_frontend(output_dir: Path) -> Path:
    """Copy the immutable UI without importing ``mrrn.__init__`` and PyTorch."""

    if not (PACKAGED_FRONTEND / "index.html").is_file():
        raise FileNotFoundError(
            f"the packaged Trackio frontend is incomplete: {PACKAGED_FRONTEND}"
        )
    destination = output_dir / ".trackio-mrrn-frontend"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(PACKAGED_FRONTEND, destination)
    return destination


def _project_has_runs(project: str) -> bool:
    try:
        return bool(list(trackio.Api().runs(project)))
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="mrcra-fineweb")
    parser.add_argument(
        "--output-dir", type=Path,
        help=(
            "Completed run used to recover an absent local Trackio project. "
            "By default the largest completed run under work/ or outputs/ is used."
        ),
    )
    parser.add_argument("--port", type=int)
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=True)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir or _discover_completed_run() or Path(
        "work/trackio-dashboard-runtime"
    )
    recovered = False
    if not _project_has_runs(arguments.project):
        # Project recovery reconstructs checkpoint-grounded diagnostics and
        # legitimately needs the model stack.  Keep that exceptional path lazy
        # so ordinary dashboard viewing remains a lightweight web observer.
        from mrrn.trackio_dashboard import ensure_trackio_project

        recovered = ensure_trackio_project(
            trackio,
            project=arguments.project,
            output_dir=output_dir,
        )
    if recovered:
        print(
            f"Recovered clean Trackio project {arguments.project!r} from {output_dir}.",
            flush=True,
        )
    frontend = _prepare_runtime_frontend(output_dir)
    trackio.show(
        project=arguments.project,
        frontend_dir=str(frontend),
        server_port=arguments.port,
        open_browser=arguments.open_browser,
        block_thread=True,
    )


if __name__ == "__main__":
    main()
