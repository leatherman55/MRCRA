#!/usr/bin/env python3
"""Run and persist production-path MRCRA matched ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrrn.integrated_acceptance import run_integrated_acceptance  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "mrcra_integrated_acceptance.json")
    parser.add_argument("--trials", type=int, default=16)
    args = parser.parse_args()
    report = run_integrated_acceptance(seeds=tuple(range(101, 101 + args.trials)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for result in report.results:
        print(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}: {result.successes}/{result.trials} CI={result.confidence_low:.3f}..{result.confidence_high:.3f}")
    print(f"artifact={args.output}")
    print(f"overall={'PASS' if report.passed else 'FAIL'}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
