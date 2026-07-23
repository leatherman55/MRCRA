#!/usr/bin/env python3
"""Export checkpoint-grounded data for the four MRRN visual instruments."""

from __future__ import annotations

import argparse
from pathlib import Path

from mrrn.visualization import build_visualization_dataset, write_visualization_dataset


DEFAULT_PROMPT = (
    "Resonance lets a pattern persist across time while multiresolution pathways "
    "separate fast local detail from slow global structure."
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stable-metrics", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--maximum-tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    evidence = build_visualization_dataset(
        checkpoint=arguments.checkpoint,
        stable_metrics=arguments.stable_metrics,
        baseline_metrics=arguments.baseline_metrics,
        prompt=arguments.prompt,
        maximum_tokens=arguments.maximum_tokens,
        device=arguments.device,
    )
    write_visualization_dataset(arguments.output, evidence)
    print(
        f"wrote {arguments.output} with {len(evidence['tokens'])} tokens, "
        f"{len(evidence['poles'])} poles, and {len(evidence['triads'])} triads"
    )


if __name__ == "__main__":
    main()
