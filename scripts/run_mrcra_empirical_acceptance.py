#!/usr/bin/env python3
"""Run and persist the bounded MRCRA learned-behavior acceptance suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrrn.empirical_acceptance import run_empirical_acceptance  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic bounded learned-behavior gates for MRCRA."
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps-scale", type=float, default=1.0)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs" / "mrcra_empirical_acceptance.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_empirical_acceptance(seed=args.seed, steps_scale=args.steps_scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"[{status}] gate={result.gate} {result.name} "
            f"duration={result.duration_seconds:.3f}s"
        )
        for criterion in result.criteria:
            value = result.metrics[criterion.metric]
            operator = ">=" if criterion.direction == "at_least" else "<="
            print(f"  {criterion.metric}={value:.6g} {operator} {criterion.threshold:.6g}")
    print(f"artifact={args.output}")
    print(f"overall={'PASS' if report.passed else 'FAIL'}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
