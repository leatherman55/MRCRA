#!/usr/bin/env python3
"""Run the paired three-seed MRCRA CSTM learning non-regression study.

The quick profile is a source-text-free procedural validation.  The FineWeb
profile is the learning-quality authority: it uses disjoint deterministic
FineWeb train/evaluation partitions, the production 8.4M actor, and 32K packed
contexts.  Every seed/variant executes in a fresh process and performs a
mid-run checkpoint/reconstruction/resume.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from math import isfinite, sqrt
from pathlib import Path
import os
import subprocess
import sys
import tempfile
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from mrrn.config import MRCRAConfig
from mrrn.cognitive_training import (
    MRCRANextTokenTrainer,
    MRCRATrainingConfig,
)
from mrrn.language import MRCRALanguageModel
from mrrn.learning_nonregression import (
    LEARNING_VARIANTS,
    LearningObservation,
    LearningRun,
    build_learning_nonregression_report,
    learning_run_from_dict,
    learning_study_controls_digest,
    learning_study_journal_payload,
    restore_learning_study_journal,
)
from mrrn.lm_training import (
    build_evaluation_batches,
    ByteTextTokenizer,
    DEFAULT_TOKENIZER_NAME,
    FineWebTextSource,
    PackedTokenStream,
    load_text_tokenizer,
)
from mrrn.training_execution_fixture import (
    RepeatingPackedFixtureStream,
    build_execution_fixture,
)


COGNITION_SUBSYSTEMS = (
    "event",
    "output_bridge",
    "controller",
    "workspace_router",
    "world_hypothesis",
    "memory",
    "other_cognition",
)


def _variant_policy(variant: str) -> tuple[bool, str]:
    if variant == "legacy_dense":
        return True, "legacy_dense"
    if variant == "sampled":
        return True, "sampled"
    if variant == "ce_only":
        return False, "legacy_dense"
    raise ValueError("unknown learning variant")


def _cognition_auxiliary_norm(metrics: dict[str, float]) -> float:
    return sqrt(sum(
        metrics.get(
            f"cstm/auxiliary_gradient_norm_after/{subsystem}", 0.0
        ) ** 2
        for subsystem in COGNITION_SUBSYSTEMS
    ))


def _build_quick_components(seed: int):
    tokenizer = ByteTextTokenizer()
    fixture = build_execution_fixture(
        "unit_1k",
        vocabulary_size=tokenizer.vocabulary_size,
        seed=seed,
    )
    return (
        tokenizer,
        MRCRAConfig.ultralight_2p7m(
            output_dim=tokenizer.vocabulary_size
        ),
        lambda: RepeatingPackedFixtureStream(fixture),
        (fixture.batch,),
        1_024,
        256,
        (128, 256),
        128,
    )


def _build_fineweb_components(args, seed: int):
    tokenizer = load_text_tokenizer(
        args.tokenizer,
        revision=args.tokenizer_revision,
        manifest_path=args.tokenizer_manifest,
    )

    def stream():
        source = FineWebTextSource(
            dataset_id=args.dataset_id,
            dataset_config=args.dataset_config,
            split="train",
            revision=args.dataset_revision,
            partition="train",
            evaluation_fraction_permyriad=args.eval_fraction_permyriad,
            shuffle_seed=seed,
            shuffle_buffer=args.shuffle_buffer,
        )
        return PackedTokenStream(source, tokenizer)

    evaluation_source = FineWebTextSource(
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        split="train",
        revision=args.dataset_revision,
        partition="eval",
        evaluation_fraction_permyriad=args.eval_fraction_permyriad,
        shuffle_seed=seed,
        shuffle_buffer=args.shuffle_buffer,
    )
    evaluation = build_evaluation_batches(
        PackedTokenStream(evaluation_source, tokenizer),
        count=args.eval_batches,
        batch_size=1,
        sequence_length=32_768,
    )
    return (
        tokenizer,
        MRCRAConfig.light_8p4m(output_dim=tokenizer.vocabulary_size),
        stream,
        evaluation,
        32_768,
        4_096,
        tuple(range(64, 4_096 + 1, 64)),
        4_096,
    )


def _build_trainer(
    *,
    args,
    variant: str,
    seed: int,
    output_dir: Path,
    components,
) -> MRCRANextTokenTrainer:
    (
        tokenizer,
        model_config,
        stream_factory,
        evaluation,
        context_length,
        tbptt_length,
        buckets,
        vocabulary_tile_size,
    ) = components
    enabled, execution = _variant_policy(variant)
    torch.manual_seed(seed)
    model = MRCRALanguageModel(
        model_config,
        model_authority="learning-nonregression-acceptance",
    )
    config = MRCRATrainingConfig(
        output_dir=str(output_dir),
        total_tokens=args.total_tokens,
        context_length=context_length,
        execution_chunk_size=min(256, tbptt_length),
        tbptt_length=tbptt_length,
        vocabulary_tile_size=vocabulary_tile_size,
        warmup_tokens=min(args.total_tokens - 1, context_length),
        integrated_cognitive_path=True,
        document_static_batching=True,
        document_bucket_lengths=buckets,
        document_batch_token_budget=8_192,
        document_grouping_policy="cost_aware",
        cognitive_stride=128,
        cstm_enabled=enabled,
        cstm_warmup_tokens=0,
        cstm_ramp_tokens=1,
        cstm_execution=execution,
        cstm_sampling_duty_cycle=args.cstm_sampling_duty_cycle,
        # Learning authority fixes the activation execution policy across every
        # arm. Candidate timing is intentionally excluded from this causal
        # comparison and whole-span recomputation is the memory-safe common
        # denominator for the 32K production profile.
        activation_policy="whole_span",
        activation_calibration=False,
        activation_memory_reserve_bytes=1 << 30,
        exact_loss_backend="tiled",
        device=args.device,
        precision="fp32",
        cpu_threads=args.cpu_threads,
        cpu_interop_threads=1,
        data_prefetch=False,
        checkpoint_interval=args.steps + 2,
        keep_checkpoints=2,
        # Retained data is mandatory, while interval evaluation stays outside
        # measured training so every variant pays the same explicit final cost.
        evaluation_interval=args.steps + 2,
        evaluation_batches=len(evaluation),
        require_evaluation=True,
        trackio_enabled=False,
        spectral_dashboard=False,
        phase_transition_ablation=False,
        seed=seed,
    )
    return MRCRANextTokenTrainer(
        model,
        tokenizer,
        stream_factory(),
        config,
        evaluation,
    )


def _run_worker(args) -> LearningRun:
    if args.profile == "quick":
        components = _build_quick_components(args.seed)
    else:
        components = _build_fineweb_components(args, args.seed)
    context_length = components[4]
    expected_steps = (
        args.total_tokens + context_length - 1
    ) // context_length
    if expected_steps != args.steps:
        raise ValueError(
            "--steps must equal ceil(total_tokens/context_length) so the "
            "checkpoint split and learning curve are unambiguous"
        )

    with tempfile.TemporaryDirectory(
        prefix=f"mrcra-learning-{args.variant}-{args.seed}-"
    ) as temporary:
        output_dir = Path(temporary)
        observations: list[LearningObservation] = []
        clipped_steps = 0
        carrier_participated = False
        cognition_participated = False
        cstm_loss = 0.0
        state_rms = 0.0
        feedback_rms = 0.0
        cognitive_cycles = 0
        events = 0
        finite = True
        started = perf_counter()

        def observe(_state, metrics):
            nonlocal clipped_steps, carrier_participated
            nonlocal cognition_participated, cstm_loss, state_rms
            nonlocal feedback_rms, cognitive_cycles, events, finite
            carrier_norm = metrics.get(
                "cstm/auxiliary_gradient_norm_after/carrier", 0.0
            )
            cognition_norm = _cognition_auxiliary_norm(metrics)
            clip = metrics["optimization/gradient_clip_coefficient"]
            cstm_loss = metrics.get(
                "cstm/estimated_dense_standardized_huber",
                metrics.get("cstm/standardized_huber", 0.0),
            )
            state_rms = max(
                state_rms, metrics.get("architecture/state_rms_max", 0.0)
            )
            feedback_rms = max(
                feedback_rms,
                metrics.get(
                    "architecture/cognitive_feedback_rms_max", 0.0
                ),
            )
            cycles = int(metrics.get("architecture/cognitive_cycles", 0))
            event_count = int(metrics.get("architecture/events", 0))
            cognitive_cycles += cycles
            events += event_count
            clipped_steps += int(clip < 1.0 - 1e-9)
            carrier_participated |= carrier_norm > 0
            cognition_participated |= cognition_norm > 0
            finite &= all(isfinite(value) for value in metrics.values())
            observations.append(LearningObservation(
                int(metrics["progress/step"]),
                int(metrics["progress/tokens_seen"]),
                perf_counter() - started,
                metrics["train/cross_entropy_nats_per_token"],
                cstm_loss,
                carrier_norm,
                cognition_norm,
                clip,
                state_rms,
                feedback_rms,
                cycles,
                event_count,
            ))

        trainer = _build_trainer(
            args=args,
            variant=args.variant,
            seed=args.seed,
            output_dir=output_dir,
            components=components,
        )
        midpoint = max(1, args.steps // 2)
        trainer.train(maximum_steps=midpoint, step_observer=observe)
        checkpoint = trainer.save_checkpoint()
        expected_state = (
            trainer.state.step,
            trainer.state.tokens_seen,
            trainer.state.valid_targets_seen,
        )

        resumed = _build_trainer(
            args=args,
            variant=args.variant,
            seed=args.seed,
            output_dir=output_dir,
            components=components,
        )
        resumed.load_checkpoint(checkpoint)
        checkpoint_resumable = expected_state == (
            resumed.state.step,
            resumed.state.tokens_seen,
            resumed.state.valid_targets_seen,
        )
        remaining = args.steps - midpoint
        if remaining:
            resumed.train(maximum_steps=remaining, step_observer=observe)
        evaluation = resumed.evaluate()
        training_seconds = perf_counter() - started
        checkpoint_resumable &= (
            resumed.state.step == args.steps
            and resumed.state.tokens_seen == args.total_tokens
        )
        finite &= all(isfinite(value) for value in evaluation.values())
        if args.variant == "ce_only":
            carrier_participated = cognition_participated = False
            cstm_loss = 0.0
        return LearningRun(
            args.variant,
            args.seed,
            resumed.state.tokens_seen,
            evaluation["eval/cross_entropy_nats_per_token"],
            evaluation["eval/effective_cross_entropy_nats_per_byte"],
            cstm_loss,
            carrier_participated,
            cognition_participated,
            clipped_steps / len(observations),
            state_rms,
            feedback_rms,
            cognitive_cycles,
            events,
            training_seconds,
            tuple(observations),
            finite,
            checkpoint_resumable,
        )


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value, handle, indent=2, sort_keys=True, allow_nan=False
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _source_authority_digest() -> str:
    authority = sha256()
    for relative in (
        "scripts/run_mrcra_learning_nonregression.py",
        "src/mrrn/learning_nonregression.py",
        "src/mrrn/cognitive_training.py",
        "src/mrrn/cstm.py",
        "src/mrrn/cstm_sampling.py",
        "src/mrrn/cstm_schedule.py",
        "src/mrrn/document_batching.py",
    ):
        authority.update(relative.encode("utf-8"))
        authority.update((ROOT / relative).read_bytes())
    return authority.hexdigest()


def _resolve_hugging_face_revisions(args) -> None:
    """Bind mutable user-facing revision labels to immutable Hub SHAs."""

    from huggingface_hub import HfApi

    api = HfApi()
    dataset = api.dataset_info(
        args.dataset_id, revision=args.dataset_revision
    )
    if not dataset.sha:
        raise RuntimeError(
            "FineWeb learning authority could not resolve an immutable dataset SHA"
        )
    args.dataset_revision = dataset.sha
    tokenizer_path = Path(args.tokenizer)
    if args.tokenizer != DEFAULT_TOKENIZER_NAME and not tokenizer_path.is_file():
        tokenizer = api.model_info(
            args.tokenizer, revision=args.tokenizer_revision
        )
        if not tokenizer.sha:
            raise RuntimeError(
                "FineWeb learning authority could not resolve an immutable "
                "tokenizer SHA"
            )
        args.tokenizer_revision = tokenizer.sha


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--profile",
        choices=("quick", "fineweb_8p4m_32k"),
        default="quick",
    )
    result.add_argument("--steps", type=int, default=3)
    result.add_argument("--total-tokens", type=int)
    result.add_argument("--seeds", nargs="+", type=int, default=(17, 29, 43))
    result.add_argument("--seed", type=int)
    result.add_argument("--variant", choices=LEARNING_VARIANTS)
    result.add_argument("--worker", action="store_true")
    result.add_argument("--device", default="cpu")
    result.add_argument("--cpu-threads", type=int, default=4)
    result.add_argument("--cstm-sampling-duty-cycle", type=float, default=0.25)
    result.add_argument("--dataset-id", default="HuggingFaceFW/fineweb")
    result.add_argument("--dataset-config", default="sample-10BT")
    result.add_argument("--dataset-revision", default="main")
    result.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_NAME)
    result.add_argument("--tokenizer-revision", default="main")
    result.add_argument("--tokenizer-manifest")
    result.add_argument("--eval-fraction-permyriad", type=int, default=100)
    result.add_argument("--eval-batches", type=int, default=1)
    result.add_argument("--shuffle-buffer", type=int, default=10_000)
    result.add_argument(
        "--worker-timeout-seconds",
        type=float,
        default=None,
        help=(
            "hard bound for one isolated seed/variant arm; defaults to "
            "600 seconds for quick and 14,400 seconds for FineWeb"
        ),
    )
    result.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output artifact. Defaults to a profile-specific quick artifact "
            "or the canonical FineWeb production authority."
        ),
    )
    return result


def main() -> None:
    args = parser().parse_args()
    context_length = 1_024 if args.profile == "quick" else 32_768
    if args.total_tokens is None:
        args.total_tokens = context_length * args.steps
    if (
        args.steps < 2
        or args.total_tokens < context_length * 2
        or args.cpu_threads < 0
        or not 0 < args.cstm_sampling_duty_cycle <= 1
        or (
            args.worker_timeout_seconds is not None
            and args.worker_timeout_seconds <= 0
        )
    ):
        raise ValueError("learning study controls are invalid")
    if args.worker:
        if args.variant is None or args.seed is None:
            raise ValueError("worker requires --variant and --seed")
        print(json.dumps(
            asdict(_run_worker(args)),
            sort_keys=True,
            allow_nan=False,
        ))
        return
    if len(set(args.seeds)) < 3 or min(args.seeds) < 0:
        raise ValueError("at least three distinct nonnegative seeds are required")
    if args.profile == "fineweb_8p4m_32k":
        _resolve_hugging_face_revisions(args)
    output = args.output or (
        ROOT
        / "outputs"
        / (
            "mrcra_learning_nonregression_procedure_quick.json"
            if args.profile == "quick"
            else "mrcra_learning_nonregression_procedure.json"
        )
    )
    journal_output = output.with_name(
        output.stem + "_runs.json"
    )
    controls = {
        "profile": args.profile,
        "steps": args.steps,
        "total_tokens": args.total_tokens,
        "seeds": list(args.seeds),
        "device": args.device,
        "cpu_threads": args.cpu_threads,
        "cstm_sampling_duty_cycle": args.cstm_sampling_duty_cycle,
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "tokenizer_manifest": args.tokenizer_manifest,
        "eval_fraction_permyriad": args.eval_fraction_permyriad,
        "eval_batches": args.eval_batches,
        "shuffle_buffer": args.shuffle_buffer,
    }
    controls_digest = learning_study_controls_digest(controls)
    authority_digest = _source_authority_digest()
    runs: list[LearningRun] = []
    if journal_output.exists():
        try:
            journal = json.loads(
                journal_output.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                "learning study journal is corrupt"
            ) from error
        if not isinstance(journal, dict):
            raise ValueError("learning study journal is malformed")
        runs = list(restore_learning_study_journal(
            journal,
            profile=args.profile,
            controls_digest=controls_digest,
            authority_digest=authority_digest,
        ))
    requested_keys = {
        (variant, seed)
        for seed in args.seeds
        for variant in LEARNING_VARIANTS
    }
    if any(
        (run.variant, run.seed) not in requested_keys for run in runs
    ):
        raise ValueError(
            "learning study journal contains an unrequested run"
        )
    completed_keys = {(run.variant, run.seed) for run in runs}
    for seed in args.seeds:
        for variant in LEARNING_VARIANTS:
            if (variant, seed) in completed_keys:
                continue
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--profile",
                args.profile,
                "--variant",
                variant,
                "--seed",
                str(seed),
                "--steps",
                str(args.steps),
                "--total-tokens",
                str(args.total_tokens),
                "--device",
                args.device,
                "--cpu-threads",
                str(args.cpu_threads),
                "--cstm-sampling-duty-cycle",
                str(args.cstm_sampling_duty_cycle),
                "--dataset-id",
                args.dataset_id,
                "--dataset-config",
                args.dataset_config,
                "--dataset-revision",
                args.dataset_revision,
                "--tokenizer",
                args.tokenizer,
                "--tokenizer-revision",
                args.tokenizer_revision,
                "--eval-fraction-permyriad",
                str(args.eval_fraction_permyriad),
                "--eval-batches",
                str(args.eval_batches),
                "--shuffle-buffer",
                str(args.shuffle_buffer),
            ]
            if args.tokenizer_manifest is not None:
                command.extend((
                    "--tokenizer-manifest",
                    args.tokenizer_manifest,
                ))
            timeout_seconds = (
                args.worker_timeout_seconds
                if args.worker_timeout_seconds is not None
                else (
                    600.0
                    if args.profile == "quick"
                    else 14_400.0
                )
            )
            print(
                f"Starting learning arm seed={seed} "
                f"variant={variant} (timeout={timeout_seconds:.0f}s).",
                flush=True,
            )
            arm_started = perf_counter()
            try:
                completed = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"learning arm seed={seed} variant={variant} exceeded "
                    f"its {timeout_seconds:.0f}s fail-closed budget"
                ) from error
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    f"learning arm seed={seed} variant={variant} failed "
                    f"with exit code {error.returncode}\n"
                    f"stdout:\n{error.stdout or ''}\n"
                    f"stderr:\n{error.stderr or ''}"
                ) from error
            run = learning_run_from_dict(
                json.loads(completed.stdout.strip().splitlines()[-1])
            )
            if (run.variant, run.seed) != (variant, seed):
                raise RuntimeError(
                    "learning worker returned mismatched run authority"
                )
            runs.append(run)
            completed_keys.add((variant, seed))
            print(
                f"Completed learning arm seed={seed} variant={variant} "
                f"in {perf_counter() - arm_started:.1f}s; "
                f"eval CE={run.eval_ce_nats_per_token:.6f}.",
                flush=True,
            )
            _write_json_atomic(
                journal_output,
                learning_study_journal_payload(
                    profile=args.profile,
                    controls_digest=controls_digest,
                    authority_digest=authority_digest,
                    runs=tuple(runs),
                    complete=False,
                ),
            )
    report = build_learning_nonregression_report(tuple(runs))
    payload = report.to_dict()
    payload["profile"] = args.profile
    payload["physical_token_budget"] = args.total_tokens
    payload["seeds"] = list(args.seeds)
    payload["controls"] = controls
    payload["controls_digest"] = controls_digest
    payload["source_authority_digest"] = authority_digest
    _write_json_atomic(output, payload)
    _write_json_atomic(
        journal_output,
        learning_study_journal_payload(
            profile=args.profile,
            controls_digest=controls_digest,
            authority_digest=authority_digest,
            runs=tuple(runs),
            complete=True,
        ),
    )
    print(output)
    if not report.passed:
        raise SystemExit(1)


def _run_cli_without_arrow_finalizer_deadlock() -> None:
    """Preserve worker status without entering PyArrow's macOS destructor."""

    try:
        main()
    except BaseException:
        if sys.platform != "darwin" or "pyarrow" not in sys.modules:
            raise
        # PyArrow 25 can deadlock in its macOS global thread-pool destructor
        # after a streaming dataset worker has already emitted its durable
        # machine-readable result or traceback.  Flush both pipes, preserve
        # failure status, and bypass only third-party global finalizers.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    if sys.platform == "darwin" and "pyarrow" in sys.modules:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    _run_cli_without_arrow_finalizer_deadlock()
