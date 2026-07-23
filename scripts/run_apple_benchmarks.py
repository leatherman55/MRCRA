#!/usr/bin/env python3
"""Retain synchronized, matched-parameter Apple-silicon performance evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from importlib.metadata import version
import platform
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from mrrn.config import MRRNConfig
from mrrn.evaluation import CausalTransformerBaseline, parameter_statistics
from mrrn.mlx_backend import MLXCausalTransformer, MLXMRRN, mlx_available
from mrrn.model import MRRN


ROOT = Path(__file__).resolve().parents[1]


def config() -> MRRNConfig:
    return MRRNConfig(
        input_dim=8, model_dim=32, output_dim=8, layers=2, scales=4, heads=4,
        modes=8, mimo_rank=2, attention_window=16, retrieved_items=4,
        memory_capacity=64, mixer_expansion=2, width_growth_cap=1.5,
        mode_growth_cap=1.5, width_multiple=4,
    )


def matched_transformer(target: int) -> CausalTransformerBaseline:
    candidates = (CausalTransformerBaseline(8, width, 8, 4, 2) for width in range(4, 257, 4))
    return min(candidates, key=lambda model: abs(parameter_statistics(model)[0] - target))


def timing(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    average = mean(values)
    return {
        "mean_seconds": average,
        "median_seconds": ordered[len(ordered) // 2],
        "p95_seconds": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "confidence95_seconds": 0.0 if len(values) < 2 else 1.96 * stdev(values) / len(values) ** 0.5,
    }


def worker(kind: str, length: int, repeats: int, seed: int) -> dict:
    """Run one shape in an isolated process so MLX releases compiled graphs."""

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    architecture = config()
    reference = MRRN(architecture).eval()
    target = parameter_statistics(reference)[0]
    transformer = matched_transformer(target).eval()
    if kind == "parity":
        x = torch.randn(1, 64, architecture.input_dim)
        with torch.no_grad():
            expected_mrrn = reference(x).prediction.numpy()
            expected_transformer = transformer(x).numpy()
        return {
            "mrrn_max_absolute_error": float(np.max(np.abs(np.array(MLXMRRN(reference)(x)) - expected_mrrn))),
            "transformer_max_absolute_error": float(np.max(np.abs(
                np.array(MLXCausalTransformer(transformer)(x)) - expected_transformer
            ))),
        }
    if kind == "cpu":
        x = torch.randn(2, 64, architecture.input_dim)
        rows = []
        for name, model in (("mrrn", reference), ("transformer", transformer)):
            with torch.no_grad():
                for _ in range(3):
                    model(x)
                values = []
                for _ in range(repeats):
                    start = perf_counter()
                    model(x)
                    values.append(perf_counter() - start)
            rows.append({"name": name, **timing(values)})
        return {"rows": rows}
    x = torch.randn(1, length, architecture.input_dim)
    if kind == "prefill_mrrn":
        return MLXMRRN(reference).benchmark(x, repeats=repeats, warmup=2)
    if kind == "prefill_transformer":
        return MLXCausalTransformer(transformer).benchmark(x, repeats=repeats, warmup=2)
    if kind == "training_mrrn":
        target_value = torch.randn(1, length, architecture.resolved_output_dim)
        return MLXMRRN(reference, training=True).benchmark_training(
            x, target_value, repeats=repeats, warmup=1
        )
    if kind == "training_transformer":
        target_value = torch.randn(1, length, architecture.resolved_output_dim)
        return MLXCausalTransformer(transformer, training=True).benchmark_training(
            x, target_value, repeats=repeats, warmup=1
        )
    if kind == "decode_mrrn":
        return MLXMRRN(reference).benchmark_decode(repeats=repeats, warmup_cycles=1)
    if kind == "decode_transformer":
        return MLXCausalTransformer(transformer).benchmark_decode(
            context=length, repeats=repeats, warmup=2
        )
    raise ValueError(f"unknown worker kind {kind}")


def paired_worker(kind: str, length: int, repeats: int, seed: int) -> dict:
    mrrn_result = run_worker(f"{kind}_mrrn", length, repeats, seed)
    transformer_result = run_worker(f"{kind}_transformer", length, repeats, seed)
    return {
        "batch": 1,
        "length": length,
        "mrrn": mrrn_result,
        "transformer": transformer_result,
        "speedup_over_transformer": transformer_result["mean_seconds"] / mrrn_result["mean_seconds"],
    }


def run_worker(kind: str, length: int, repeats: int, seed: int) -> dict:
    print(f"benchmark worker: {kind} length={length}", file=sys.stderr, flush=True)
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker", kind,
        "--length", str(length), "--repeats", str(repeats), "--seed", str(seed),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--training-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "apple_benchmark_report.json")
    parser.add_argument(
        "--phase", choices=("prefill", "training", "decode", "all"), default="all"
    )
    parser.add_argument("--worker", choices=(
        "parity", "cpu", "prefill_mrrn", "prefill_transformer",
        "training_mrrn", "training_transformer", "decode_mrrn", "decode_transformer",
    ),
                        help=argparse.SUPPRESS)
    parser.add_argument("--length", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.repeats, args.training_repeats) <= 0:
        raise SystemExit("repeat counts must be positive")
    if not mlx_available():
        raise SystemExit("Apple MLX is unavailable")
    import mlx
    import mlx.core as mx

    if args.worker:
        print(json.dumps(worker(args.worker, args.length, args.repeats, args.seed)))
        return

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    architecture = config()
    reference = MRRN(architecture).eval()
    target = parameter_statistics(reference)[0]
    transformer = matched_transformer(target).eval()
    transformer_parameters = parameter_statistics(transformer)[0]
    parity = run_worker("parity", 0, 1, args.seed)
    prefill = sorted([
        paired_worker("prefill", length, args.repeats, args.seed + length)
        for length in (12288, 4096, 1024, 256, 64)
    ], key=lambda row: row["length"]) if args.phase in {"prefill", "all"} else []
    training = sorted([
        paired_worker("training", length, args.training_repeats, args.seed + length)
        for length in (64, 1024, 4096)
    ], key=lambda row: row["length"]) if args.phase in {"training", "all"} else []
    decode = []
    if args.phase in {"decode", "all"}:
        mrrn_decode = run_worker("decode_mrrn", 0, args.repeats, args.seed)
        for context in (32768, 12288, 4096, 1024, 64):
            transformer_decode = run_worker(
                "decode_transformer", context, args.repeats, args.seed + context
            )
            decode.append({
                "batch": 1,
                "context": context,
                "mrrn": mrrn_decode,
                "transformer": transformer_decode,
                "speedup_over_transformer": (
                    transformer_decode["mean_seconds"] / mrrn_decode["mean_seconds"]
                ),
            })
        decode.sort(key=lambda row: row["context"])
    cpu_rows = (
        run_worker("cpu", 64, args.repeats, args.seed)["rows"]
        if args.phase in {"prefill", "all"} else []
    )

    crossover = next(
        (row["length"] for row in prefill if row["speedup_over_transformer"] >= 1), None
    )
    payload = {
        "schema": 1,
        "seed": args.seed,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mlx": version("mlx"),
            "mlx_device": str(mx.default_device()),
            "torch_threads": torch.get_num_threads(),
        },
        "configuration": asdict(architecture),
        "parameter_matching": {
            "mrrn": target,
            "transformer": transformer_parameters,
            "transformer_width": transformer.width,
            "mismatch_fraction": abs(transformer_parameters - target) / target,
        },
        "parity": parity,
        "prefill": prefill,
        "training_forward_backward": training,
        "recurrent_decode": decode,
        "cpu_short_prefill": cpu_rows,
        "first_measured_prefill_crossover_tokens": crossover,
        "first_measured_decode_crossover_tokens": next(
            (row["context"] for row in decode if row["speedup_over_transformer"] >= 1), None
        ),
        "method": {
            "synchronization": "every sample is materialized with mx.eval; CPU execution is synchronous",
            "transformer_attention": "MLX fused scaled_dot_product_attention with causal mask",
            "mrrn_attention": "exact bounded local and coarse-landmark resonant candidate attention",
            "matching": "nearest four-head, two-layer Transformer parameter count in widths 4..256",
            "compile_time": "excluded after two warmup/materialization calls per shape",
            "scope": "float32, this M1 machine, these exact batch/length/configuration points",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "parity": parity,
        "crossover": crossover,
        "longest_speedup": None if not prefill else prefill[-1]["speedup_over_transformer"],
    }, indent=2))


if __name__ == "__main__":
    main()
