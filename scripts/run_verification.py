#!/usr/bin/env python3
"""Retain reproducible capability and matched-parameter benchmark evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from mrrn.config import MRRNConfig
from mrrn.evaluation import (
    apply_ablation,
    CausalTransformerBaseline,
    LongConvolutionBaseline,
    RealSelectiveSSMBaseline,
    benchmark_suite,
    environment_report,
    parameter_statistics,
    save_benchmark_report,
)
from mrrn.model import MRRN
from mrrn.synthetics import run_capability_suite, save_capability_report
from mrrn.traceability import audit

ROOT = Path(__file__).resolve().parents[1]


def closest(factory, target: int, widths: range):
    candidates = [(abs(parameter_statistics(factory(width))[0] - target), width) for width in widths]
    return factory(min(candidates)[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    # Small heterogeneous graphs are launch-bound on M1; one CPU thread is the
    # measured optimum for this retained shape and avoids efficiency-core jitter.
    torch.set_num_threads(1)
    config = MRRNConfig(
        input_dim=8, model_dim=32, output_dim=8, layers=2, scales=4, heads=4,
        modes=8, mimo_rank=2, attention_window=16, retrieved_items=4,
        memory_capacity=64, mixer_expansion=2, width_growth_cap=1.5,
        mode_growth_cap=1.5, width_multiple=4,
    )
    mrrn = MRRN(config)
    target = parameter_statistics(mrrn)[0]
    transformer = closest(
        lambda width: CausalTransformerBaseline(8, width, 8, 4, 2), target, range(4, 257, 4)
    )
    ssm = closest(lambda width: RealSelectiveSSMBaseline(8, width, 8, 2), target, range(4, 513))
    convolution = closest(
        lambda width: LongConvolutionBaseline(8, width, 8, 2, kernel_size=31), target, range(4, 257)
    )
    models = {
        "mrrn": mrrn,
        "mrrn_no_spectral": apply_ablation(mrrn, "no_spectral_activation"),
        "mrrn_spectral_only": apply_ablation(mrrn, "spectral_only_local"),
        "transformer": transformer,
        "real_selective_ssm": ssm, "long_convolution": convolution,
    }
    x = torch.randn(2, 64, 8)
    results = benchmark_suite(models, x, repeats=args.repeats, warmup=1)
    save_benchmark_report(
        args.output / "benchmark_report.json", results, seed=args.seed,
        notes={
            "scope": "single-thread CPU reference kernels; quality must be measured separately per task",
            "matching": "Transformer/SSM/convolution are nearest to full-MRRN trainable parameters among enumerated widths",
            "spectral_ablation": "conventional-only and spectral-only MRRN rows retain their natural structurally exact parameter counts",
        },
    )
    capabilities = run_capability_suite()
    save_capability_report(args.output / "capability_report.json", capabilities)
    traceability = audit(ROOT, strict=True)
    parameter_counts = {
        name: parameter_statistics(model)[0] for name, model in models.items()
    }
    report = {
        "environment": environment_report(), "seed": args.seed,
        "configuration": asdict(config), "parameter_counts": parameter_counts,
        "parameter_mismatch_fraction": {
            name: abs(count - target) / target for name, count in parameter_counts.items()
        },
        "capabilities_passed": all(item.passed for item in capabilities),
        "traceability": asdict(traceability),
        "benchmark_results": [asdict(item) for item in results],
        "claim_boundary": (
            "Timings characterize this machine, dtype, shape, and reference implementation only; "
            "they are not universal hardware or quality superiority claims."
        ),
    }
    (args.output / "verification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
