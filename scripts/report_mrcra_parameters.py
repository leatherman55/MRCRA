#!/usr/bin/env python3
"""Construct the canonical serious actor and write an exact parameter audit."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrrn.config import MRCRAConfig  # noqa: E402
from mrrn.language import MRCRALanguageModel  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocabulary-size", type=int, default=50_257)
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument(
        "--lightmodel", action="store_true",
        help="Audit the integrated 8.4M light profile instead of the 120M class.",
    )
    profile.add_argument(
        "--ultralightmodel", action="store_true",
        help="Audit the integrated 1.3M ultralight profile instead of the 120M class.",
    )
    parser.add_argument(
        "--output",
        help="Destination; defaults to the selected profile's parameter report.",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    profile = (
        "mrcra_1p3m_ultralight"
        if args.ultralightmodel
        else "mrcra_8p4m_light"
        if args.lightmodel
        else "mrcra_120m_serious"
    )
    config = (
        MRCRAConfig.ultralight_1p3m(output_dim=args.vocabulary_size)
        if args.ultralightmodel
        else MRCRAConfig.light_8p4m(output_dim=args.vocabulary_size)
        if args.lightmodel
        else MRCRAConfig.serious_120m(output_dim=args.vocabulary_size)
    )
    model = MRCRALanguageModel(config, model_authority="parameter-audit")
    config.require_actor_parameter_count(model.parameter_count)
    groups: dict[str, int] = defaultdict(int)
    tensors: dict[str, int] = defaultdict(int)
    for name, parameter in model.named_parameters():
        parts = name.split(".")
        group = ".".join(parts[:2]) if parts[0] == "cognitive" else parts[0]
        groups[group] += parameter.numel()
        tensors[group] += 1
    count = model.parameter_count
    payload = {
        "artifact": "canonical MRCRA actor parameter audit",
        "model_profile": profile,
        "parameter_count": count,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "unique_parameter_tensors": sum(1 for _ in model.parameters()),
        "vocabulary_size": args.vocabulary_size,
        "tied_token_and_output_weights": (
            model.token_embedding.weight is model.cognitive.carrier.output_head.weight
        ),
        "declared_range": {
            "minimum": config.actor_parameter_minimum,
            "maximum": config.actor_parameter_maximum,
            "passed": config.actor_parameter_minimum <= count <= config.actor_parameter_maximum,
        },
        "parameter_count_by_subsystem": dict(
            sorted(groups.items(), key=lambda item: (-item[1], item[0]))
        ),
        "parameter_tensors_by_subsystem": dict(sorted(tensors.items())),
        "static_storage_gib": {
            "fp32_parameters": count * 4 / 2**30,
            "bf16_or_fp16_parameters_if_materialized": count * 2 / 2**30,
            "fp32_parameters_gradients_and_adam_moments": count * 16 / 2**30,
        },
        "storage_scope": (
            "Static tensor storage only; excludes activations, CUDA allocator reserve, "
            "checkpoint recomputation workspaces, and cognitive runtime state."
        ),
        "torch_version": str(torch.__version__),
        "configuration": asdict(config),
    }
    destination = Path(
        args.output
        or (
            "outputs/mrcra_1p3m_parameter_report.json"
            if args.ultralightmodel
            else "outputs/mrcra_8p4m_parameter_report.json"
            if args.lightmodel
            else "outputs/mrcra_120m_parameter_report.json"
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
