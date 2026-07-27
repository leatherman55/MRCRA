#!/usr/bin/env python3
"""Measure bounded Trackio scalar logging against a null observer."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrrn.lm_training import LMTrainingConfig, TrackioReporter
from mrrn.observability_acceptance import (
    ObservabilitySample,
    build_observability_report,
)


class NullReporter:
    def log(self, metrics, *, step):  # noqa: ARG002
        return None

    def finish(self):
        return None


def rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value << 10


def worker(variant: str, steps: int) -> ObservabilitySample:
    if variant not in {"null", "trackio"} or steps < 10:
        raise ValueError("invalid Trackio benchmark worker request")
    temporary = tempfile.TemporaryDirectory(prefix="mrcra-trackio-overhead-")
    try:
        if variant == "trackio":
            config = LMTrainingConfig(
                output_dir=temporary.name,
                total_tokens=4,
                sequence_length=4,
                evaluation_batches=1,
                trackio_project="mrcra-observability-acceptance",
                run_name="bounded-trackio-overhead",
                show_dashboard=False,
                spectral_dashboard=False,
            )
            reporter = TrackioReporter(
                config,
                {"authority": "observability-acceptance"},
                resume=True,
            )
        else:
            reporter = NullReporter()
        times: list[float] = []
        rss: list[int] = []
        metrics = {
            f"acceptance/scalar_{index:02d}": float(index)
            for index in range(64)
        }
        for step in range(steps):
            started = perf_counter()
            reporter.log(metrics, step=step)
            times.append(perf_counter() - started)
            rss.append(rss_bytes())
        reporter.finish()
        return ObservabilitySample(variant, tuple(times), tuple(rss))
    finally:
        temporary.cleanup()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--steps", type=int, default=100)
    result.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "mrcra_trackio_overhead_acceptance.json",
    )
    result.add_argument("--worker", action="store_true")
    result.add_argument("--variant", choices=("null", "trackio"))
    return result


def main() -> None:
    args = parser().parse_args()
    if args.steps < 10:
        raise ValueError("--steps must be at least ten")
    if args.worker:
        if args.variant is None:
            raise ValueError("worker requires --variant")
        print(json.dumps(asdict(worker(args.variant, args.steps))))
        return
    samples = []
    for variant in ("null", "trackio"):
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--variant",
                variant,
                "--steps",
                str(args.steps),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        samples.append(
            ObservabilitySample(
                **json.loads(completed.stdout.strip().splitlines()[-1])
            )
        )
    report = build_observability_report(tuple(samples))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report.to_dict(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
