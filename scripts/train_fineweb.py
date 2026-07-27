#!/usr/bin/env python3
"""Canonical FineWeb entrypoint: integrated MRCRA by default.

The prior 4.695M sequence-only MRRN trainer remains available solely through
the explicit ``--legacy-mrrn`` compatibility switch.  A normal invocation is
delegated to :mod:`train_mrcra_fineweb`, ensuring that the familiar command can
never silently bypass the cognitive runtime, retained evaluation, or format-16
checkpoint contract.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault("PYTHONWARNINGS", "ignore:resource_tracker:UserWarning")

import torch

from mrrn.language import MRRNLanguageModel, fineweb_4p7m_config, tiny_language_config
from mrrn.lm_training import (
    ByteTextTokenizer,
    FineWebTextSource,
    HuggingFaceTextTokenizer,
    LMTrainingConfig,
    NextTokenTrainer,
    PackedTokenStream,
    SequenceTextSource,
    build_evaluation_batches,
)


def resolve_revision(repo_id: str, revision: str, *, repo_type: str) -> str:
    """Resolve mutable Hub revisions to immutable commit hashes."""

    from huggingface_hub import HfApi

    api = HfApi()
    info = (
        api.dataset_info(repo_id, revision=revision)
        if repo_type == "dataset" else api.model_info(repo_id, revision=revision)
    )
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a commit SHA for {repo_id}")
    return info.sha


def latest_checkpoint(output_dir: Path) -> Path | None:
    pointer = output_dir / "checkpoints" / "latest.json"
    if not pointer.exists():
        return None
    value = json.loads(pointer.read_text(encoding="utf-8"))
    path = pointer.parent / value["checkpoint"]
    if not path.is_file():
        raise FileNotFoundError(f"latest checkpoint pointer names missing file {path}")
    return path


def validate_fresh_output(output_dir: Path, *, resume: str | None) -> None:
    """Prevent accidental metric/checkpoint mixing between independent runs."""

    if resume is not None or not output_dir.exists():
        return
    protected = (
        output_dir / "metrics.jsonl",
        output_dir / "run_manifest.json",
        output_dir / "checkpoints",
    )
    if any(path.exists() for path in protected):
        raise FileExistsError(
            f"{output_dir} already contains a run; use --resume or select a new --output-dir"
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Stream original English FineWeb and train the tied-BPE 4.695M MRRN."
    )
    result.add_argument("--dataset-id", default="HuggingFaceFW/fineweb")
    result.add_argument("--dataset-config", default="sample-10BT")
    result.add_argument("--dataset-revision", default="main")
    result.add_argument("--tokenizer", default="gpt2")
    result.add_argument("--tokenizer-revision", default="main")
    result.add_argument("--output-dir", default="outputs/fineweb-4p7m-stable")
    result.add_argument("--total-tokens", type=int, default=20_000_000)
    result.add_argument("--sequence-length", type=int, default=2048)
    result.add_argument("--micro-batch-size", type=int, default=1)
    result.add_argument("--gradient-accumulation-steps", type=int, default=4)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--warmup-tokens", type=int, default=800_000)
    result.add_argument("--weight-decay", type=float, default=0.1)
    result.add_argument("--eval-interval", type=int, default=100)
    result.add_argument("--eval-batches", type=int, default=8)
    result.add_argument("--checkpoint-interval", type=int, default=100)
    result.add_argument("--architecture-log-interval", type=int, default=10)
    result.add_argument("--state-regularization-weight", type=float, default=1e-4)
    result.add_argument("--state-target-rms", type=float, default=8.0)
    result.add_argument("--state-warning-rms", type=float, default=16.0)
    result.add_argument("--state-abort-rms", type=float, default=32.0)
    result.add_argument("--gradient-warning-norm", type=float, default=100.0)
    result.add_argument("--gradient-abort-norm", type=float, default=1_000.0)
    result.add_argument("--stability-patience", type=int, default=3)
    result.add_argument(
        "--gradient-recovery", action=argparse.BooleanOptionalAction, default=True,
        help="Back off learning rate when persistent finite pre-clip gradients exceed the limit.",
    )
    result.add_argument("--gradient-backoff-factor", type=float, default=0.5)
    result.add_argument("--gradient-recovery-limit", type=int, default=4)
    result.add_argument(
        "--device", default="auto", metavar="DEVICE",
        help="auto, cpu, mps, cuda, or a CUDA index such as cuda:1 (default: auto).",
    )
    result.add_argument(
        "--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto",
        help="CUDA arithmetic precision; auto prefers BF16 and falls back to scaled FP16.",
    )
    result.add_argument("--seed", type=int, default=20260721)
    result.add_argument("--shuffle-buffer", type=int, default=10_000)
    result.add_argument("--eval-fraction-permyriad", type=int, default=100)
    result.add_argument("--trackio-project", default="mrrn-fineweb")
    result.add_argument(
        "--run-name",
        help="Trackio run name; defaults to a name containing the exact token budget.",
    )
    result.add_argument("--trackio-space-id")
    result.add_argument(
        "--trackio-remote-log-interval",
        type=int,
        default=4,
        help=(
            "Send one coalesced scalar row to remote Trackio every N steps; "
            "the local JSONL mirror retains every row."
        ),
    )
    result.add_argument(
        "--dashboard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Launch the local Trackio web UI inside the training process "
            "(default: disabled; logging remains enabled)."
        ),
    )
    result.add_argument(
        "--spectral-dashboard", action=argparse.BooleanOptionalAction, default=True,
        help="Publish the four spectral instruments in Trackio's Spectral Network tab.",
    )
    result.add_argument(
        "--spectral-snapshot-interval", type=int, default=100,
        help="Optimizer-step interval for checkpoint-grounded Trackio spectral artifacts.",
    )
    result.add_argument(
        "--spectral-baseline-metrics", type=Path,
        help="Optional comparison metrics.jsonl for the Stability Observatory.",
    )
    result.add_argument("--pin-revisions", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--resume", nargs="?", const="latest")
    result.add_argument(
        "--smoke-test", action="store_true",
        help="Run a tiny deterministic local end-to-end training/checkpoint/dashboard-disabled test.",
    )
    return result


def legacy_main() -> None:
    args = parser().parse_args()
    # Seed before model construction so the initial weights, not only training
    # stochasticity, are reproducible and checkpoint-comparable.
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    validate_fresh_output(output_dir, resume=args.resume)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke_test:
        tokenizer = ByteTextTokenizer()
        model = MRRNLanguageModel(tiny_language_config(tokenizer.vocabulary_size))
        training_documents = (
            "Resonance stores a compressed history of a signal.",
            "A causal language model predicts the next symbol from the symbols before it.",
            "Multiresolution processing assigns different time scales to different bands.",
            "Stable recurrent state allows context to grow without a growing key value cache.",
        )
        evaluation_documents = (
            "Spectral modes describe oscillation, phase, and decay.",
            "The evaluation stream is disjoint from the training stream.",
        )
        train_source = SequenceTextSource(training_documents)
        eval_source = SequenceTextSource(evaluation_documents)
        configuration = LMTrainingConfig(
            output_dir=str(output_dir), total_tokens=256, sequence_length=16,
            micro_batch_size=1, gradient_accumulation_steps=2,
            warmup_tokens=32, log_interval=1, architecture_log_interval=1,
            evaluation_interval=2, evaluation_batches=2, checkpoint_interval=2,
            keep_checkpoints=2, generation_tokens=4, device="cpu",
            trackio_project=args.trackio_project,
            run_name=args.run_name or "mrrn-stability-smoke",
            trackio_remote_log_interval=(
                args.trackio_remote_log_interval
            ),
            show_dashboard=False, spectral_dashboard=args.spectral_dashboard,
            spectral_snapshot_interval=2, spectral_snapshot_tokens=16, seed=args.seed,
        )
    else:
        dataset_revision = (
            resolve_revision(args.dataset_id, args.dataset_revision, repo_type="dataset")
            if args.pin_revisions else args.dataset_revision
        )
        tokenizer_revision = (
            resolve_revision(args.tokenizer, args.tokenizer_revision, repo_type="model")
            if args.pin_revisions else args.tokenizer_revision
        )
        tokenizer = HuggingFaceTextTokenizer(args.tokenizer, revision=tokenizer_revision)
        model = MRRNLanguageModel(fineweb_4p7m_config(tokenizer.vocabulary_size))
        common = dict(
            dataset_id=args.dataset_id, dataset_config=args.dataset_config,
            split="train", revision=dataset_revision,
            evaluation_fraction_permyriad=args.eval_fraction_permyriad,
            shuffle_seed=args.seed, shuffle_buffer=args.shuffle_buffer,
        )
        train_source = FineWebTextSource(partition="train", **common)
        eval_source = FineWebTextSource(partition="eval", **common)
        configuration = LMTrainingConfig(
            output_dir=str(output_dir), total_tokens=args.total_tokens,
            sequence_length=args.sequence_length, micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate, warmup_tokens=args.warmup_tokens,
            weight_decay=args.weight_decay, evaluation_interval=args.eval_interval,
            evaluation_batches=args.eval_batches, checkpoint_interval=args.checkpoint_interval,
            architecture_log_interval=args.architecture_log_interval,
            state_regularization_weight=args.state_regularization_weight,
            state_target_rms=args.state_target_rms,
            state_warning_rms=args.state_warning_rms,
            state_abort_rms=args.state_abort_rms,
            gradient_warning_norm=args.gradient_warning_norm,
            gradient_abort_norm=args.gradient_abort_norm,
            stability_patience=args.stability_patience,
            gradient_recovery=args.gradient_recovery,
            gradient_backoff_factor=args.gradient_backoff_factor,
            gradient_recovery_limit=args.gradient_recovery_limit,
            device=args.device, precision=args.precision, seed=args.seed,
            trackio_project=args.trackio_project,
            run_name=(
                args.run_name
                or f"mrrn-4p7m-fineweb-stable-{args.total_tokens}-tokens"
            ),
            trackio_space_id=args.trackio_space_id,
            trackio_remote_log_interval=(
                args.trackio_remote_log_interval
            ),
            show_dashboard=args.dashboard,
            spectral_dashboard=args.spectral_dashboard,
            spectral_snapshot_interval=args.spectral_snapshot_interval,
            spectral_baseline_metrics=(
                None
                if args.spectral_baseline_metrics is None
                else str(args.spectral_baseline_metrics.resolve())
            ),
        )
    train_stream = PackedTokenStream(train_source, tokenizer)
    evaluation_stream = PackedTokenStream(eval_source, tokenizer)
    evaluation_batches = build_evaluation_batches(
        evaluation_stream,
        count=configuration.evaluation_batches,
        batch_size=configuration.micro_batch_size,
        sequence_length=configuration.sequence_length,
    )
    trainer = NextTokenTrainer(model, tokenizer, train_stream, evaluation_batches, configuration)
    run_manifest = {
        "model_parameters": model.parameter_count,
        "trainable_parameters": model.trainable_parameter_count,
        "model_config": asdict(model.config),
        "tokenizer": tokenizer.identity(),
        "training_config": asdict(configuration),
        "training_source": train_source.state_dict(),
        "evaluation_source": eval_source.state_dict(),
        "runtime": trainer.runtime,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"MRRN language model: {model.parameter_count:,} parameters; "
        f"{configuration.total_tokens:,} training tokens; "
        f"{configuration.total_steps:,} optimizer steps.",
        flush=True,
    )
    runtime = trainer.runtime
    print(
        f"Training device: {runtime['device']} ({runtime.get('gpu_name', 'host')}); "
        f"precision: {runtime['precision']}; fused AdamW: {runtime.get('fused_adamw', False)}.",
        flush=True,
    )
    if args.resume:
        checkpoint = latest_checkpoint(output_dir) if args.resume == "latest" else Path(args.resume)
        if checkpoint is None:
            raise FileNotFoundError("--resume requested but no latest checkpoint exists")
        trainer.load_checkpoint(checkpoint)
        print(f"Resumed {checkpoint} at step {trainer.state.step}.", flush=True)
    trainer.train()


def main() -> None:
    """Dispatch to integrated MRCRA unless legacy mode is explicitly named."""

    if "--legacy-mrrn" in sys.argv[1:]:
        sys.argv.remove("--legacy-mrrn")
        legacy_main()
        return
    from train_mrcra_fineweb import main as mrcra_main

    mrcra_main()


def _run_cli_without_arrow_finalizer_deadlock() -> None:
    """Exit after durable trainer cleanup without waiting on Arrow's macOS pool.

    PyArrow 25 can deadlock in its C++ global thread-pool destructor after a
    streaming dataset has been exhausted on macOS.  At this point the trainer
    has already finished Trackio, atomically saved its checkpoint/manifest, and
    flushed its own files.  A process-level exit avoids turning a completed run
    into an indefinitely sleeping shell command.  Non-Arrow smoke and legacy
    paths retain ordinary Python teardown.
    """

    try:
        main()
    except BaseException:
        if "pyarrow" not in sys.modules:
            raise
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    if "pyarrow" in sys.modules:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    _run_cli_without_arrow_finalizer_deadlock()
