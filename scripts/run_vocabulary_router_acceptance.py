#!/usr/bin/env python3
"""Run and persist exact-authority vocabulary-router acceptance evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mrrn.vocabulary_router_acceptance import run_vocabulary_router_acceptance


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--production-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exercise all GPT-2-vocabulary widths (20, 96, and 256).",
    )
    result.add_argument("--seed", type=int, default=20260725)
    result.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vocabulary_router_empirical_acceptance.json"),
    )
    return result


def main() -> None:
    args = parser().parse_args()
    report = run_vocabulary_router_acceptance(
        seed=args.seed,
        production_scale=args.production_scale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for criterion in report.criteria:
        print(f"{'PASS' if criterion.passed else 'FAIL'} {criterion.name}")
    for experiment in report.experiments:
        print(
            f"{experiment.name}: exact={experiment.exact_threshold_sets} "
            f"avoided={experiment.avoided_fraction:.1%} "
            f"dense={experiment.dense_seconds:.6f}s "
            f"routed={experiment.routed_seconds:.6f}s"
        )
    print(f"Wrote {args.output}")
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
