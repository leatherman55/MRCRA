#!/usr/bin/env python3
"""Print the specification evidence audit and fail when evidence is invalid."""

from __future__ import annotations

import argparse
from pathlib import Path

from mrrn.traceability import audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="require every heading to be verified")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = audit(root, strict=args.strict)
    print(
        f"specification headings={report.total} evidenced={report.evidenced} "
        f"verified={report.verified} missing={len(report.missing)} invalid={len(report.invalid)}"
    )
    if report.missing:
        print("missing evidence:", ", ".join(report.missing))
    if report.invalid:
        print("invalid evidence:")
        print("\n".join(f"- {item}" for item in report.invalid))
    return int(bool(report.invalid or (args.strict and report.missing)))


if __name__ == "__main__":
    raise SystemExit(main())
