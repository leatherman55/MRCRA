#!/usr/bin/env python3
"""Run and persist MRCRA relative performance budgets."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mrrn.performance_acceptance import run_performance_acceptance  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "mrcra_performance_acceptance.json")
    parser.add_argument("--repeats", type=int, default=41)
    args = parser.parse_args()
    report = run_performance_acceptance(repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for result in report.results:
        print(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}: {result.metric:.6g} {result.direction} {result.threshold:.6g} {result.unit}")
    print(f"artifact={args.output}")
    print(f"overall={'PASS' if report.passed else 'FAIL'}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
