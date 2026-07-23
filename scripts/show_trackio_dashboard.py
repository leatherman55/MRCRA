#!/usr/bin/env python3
"""Launch Trackio with the MRCRA Cognition and Spectral Network tabs installed."""

from __future__ import annotations

import argparse
from pathlib import Path

import trackio

from mrrn.trackio_dashboard import (
    discover_completed_run,
    ensure_trackio_project,
    prepare_runtime_frontend,
)


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
    output_dir = arguments.output_dir or discover_completed_run() or Path(
        "work/trackio-dashboard-runtime"
    )
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
    frontend = prepare_runtime_frontend(output_dir)
    trackio.show(
        project=arguments.project,
        frontend_dir=str(frontend),
        server_port=arguments.port,
        open_browser=arguments.open_browser,
        block_thread=True,
    )


if __name__ == "__main__":
    main()
