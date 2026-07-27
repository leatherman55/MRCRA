#!/usr/bin/env python3
"""Run and persist integrated available-device MRCRA parity evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mrrn.device_parity_acceptance import run_device_parity_acceptance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/mrcra_device_parity_acceptance.json"),
    )
    arguments = parser.parse_args()
    report = run_device_parity_acceptance()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
