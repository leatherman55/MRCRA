#!/usr/bin/env python3
"""Run and record the external 32K/CUDA acceptance measurement for MRCRA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrrn.cognitive_training import MRCRANextTokenTrainer, MRCRATrainingConfig  # noqa: E402
from mrrn.config import MRCRAConfig  # noqa: E402
from mrrn.language import MRCRALanguageModel  # noqa: E402
from mrrn.lm_training import (  # noqa: E402
    HuggingFaceTextTokenizer, PackedTokenStream, SequenceTextSource,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--context-length", type=int, default=32_768)
    parser.add_argument("--execution-chunk-size", type=int, default=256)
    parser.add_argument("--tbptt-length", type=int, default=4_096)
    parser.add_argument("--vocabulary-tile-size", type=int, default=2_048)
    parser.add_argument(
        "--compile-tensor-cores", action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--tokenizer-revision", default="main")
    parser.add_argument("--output", default="outputs/mrcra-120m-32k-benchmark.json")
    parser.add_argument("--work-dir", default="work/mrcra-32k-benchmark")
    parser.add_argument("--allow-non-cuda-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available() and not args.allow_non_cuda_smoke:
        raise RuntimeError(
            "Gate M requires a real CUDA run. Use --allow-non-cuda-smoke only to test the harness."
        )
    tokenizer = HuggingFaceTextTokenizer(
        args.tokenizer, revision=args.tokenizer_revision
    )
    config = MRCRAConfig.serious_120m(output_dim=tokenizer.vocabulary_size)
    model = MRCRALanguageModel(config, model_authority="mrcra-32k-benchmark")
    config.require_actor_parameter_count(model.parameter_count)
    text = (
        "A bounded relational workspace preserves provenance while spectral recurrence "
        "maintains continuity across physical scales. "
    ) * 512
    stream = PackedTokenStream(SequenceTextSource((text,)), tokenizer)
    training = MRCRATrainingConfig(
        output_dir=args.work_dir, total_tokens=args.context_length,
        context_length=args.context_length,
        execution_chunk_size=args.execution_chunk_size,
        tbptt_length=args.tbptt_length,
        vocabulary_tile_size=args.vocabulary_tile_size,
        integrated_cognitive_path=True,
        cognitive_stride=config.cognitive.event_chunk_size,
        compile_tensor_cores=args.compile_tensor_cores,
        device=args.device if torch.cuda.is_available() else "cpu",
        precision=args.precision if torch.cuda.is_available() else "fp32",
        trackio_enabled=False, show_dashboard=False, spectral_dashboard=False,
        checkpoint_interval=10_000,
    )
    trainer = MRCRANextTokenTrainer(model, tokenizer, stream, training)
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)
        torch.cuda.synchronize(trainer.device)
    started = perf_counter()
    state = trainer.train(maximum_steps=1)
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
    elapsed = perf_counter() - started
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    payload = {
        "gate": "M",
        "passed": state.tokens_seen == args.context_length and trainer.device.type == "cuda",
        "smoke_only": trainer.device.type != "cuda",
        "device": str(trainer.device),
        "gpu": (
            torch.cuda.get_device_name(trainer.device)
            if trainer.device.type == "cuda" else None
        ),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "platform": platform.platform(),
        "parameter_count": model.parameter_count,
        "parameter_storage_gib": parameter_bytes / 2**30,
        "context_length": args.context_length,
        "execution_chunk_size": args.execution_chunk_size,
        "tbptt_length": args.tbptt_length,
        "vocabulary_tile_size": args.vocabulary_tile_size,
        "precision": training.precision,
        "integrated_cognitive_path": training.integrated_cognitive_path,
        "compiled_tensor_cores": trainer.runtime["compiled_tensor_cores"],
        "elapsed_seconds": elapsed,
        "tokens_per_second": args.context_length / elapsed,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(trainer.device) / 2**30
            if trainer.device.type == "cuda" else None
        ),
        "peak_reserved_gib": (
            torch.cuda.max_memory_reserved(trainer.device) / 2**30
            if trainer.device.type == "cuda" else None
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
