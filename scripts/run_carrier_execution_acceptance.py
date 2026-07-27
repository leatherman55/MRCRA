#!/usr/bin/env python3
"""Run and persist the fail-closed carrier execution acceptance suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mrrn.carrier_execution_acceptance import (
    run_carrier_execution_acceptance,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/carrier_execution_empirical_acceptance.json"),
    )
    arguments = parser.parse_args()
    report = run_carrier_execution_acceptance()
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
