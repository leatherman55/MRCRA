#!/usr/bin/env python3
"""Run and persist the deterministic CSTM empirical acceptance suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mrrn.cstm_acceptance import run_cstm_acceptance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Causal Spectral Target Multiplexing empirical acceptance."
    )
    parser.add_argument(
        "--output",
        default="outputs/cstm_empirical_acceptance.json",
    )
    args = parser.parse_args()
    report = run_cstm_acceptance()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
