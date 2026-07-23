#!/usr/bin/env python3
"""Audit a trained MRCRA checkpoint against the fail-closed serious gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrrn.serious_acceptance import audit_serious_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--evaluation-artifact", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs" / "mrcra_serious_checkpoint_acceptance.json",
    )
    args = parser.parse_args()
    report = audit_serious_checkpoint(
        args.checkpoint, args.run_manifest, args.evaluation_artifact,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for gate in report.gates:
        print(f"[{'PASS' if gate.passed else 'FAIL'}] {gate.name}: {gate.evidence}")
        if gate.failure:
            print(f"  {gate.failure}")
    print(f"artifact={args.output}")
    print(f"overall={'PASS' if report.passed else 'FAIL'}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
