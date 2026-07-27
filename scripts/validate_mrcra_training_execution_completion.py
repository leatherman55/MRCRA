#!/usr/bin/env python3
"""Validate all retained full-scale MRCRA repair authorities together."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrrn.retained_training_acceptance import (  # noqa: E402
    validate_retained_training_acceptance,
)


def main() -> int:
    report = validate_retained_training_acceptance(ROOT)
    destination = (
        ROOT
        / "outputs"
        / "mrcra_training_execution_completion_validation.json"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                report.to_dict(),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    print(destination)
    return int(not report.passed)


if __name__ == "__main__":
    raise SystemExit(main())
