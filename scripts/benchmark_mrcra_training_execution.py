#!/usr/bin/env python3
"""Run matched MRCRA training-execution variants in fresh processes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import os
from statistics import median
import subprocess
import sys
import tempfile
from threading import Event, Thread
from time import perf_counter, sleep

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Exact authority of the six-arm production journal written immediately
# before compiler-aware shape coarsening.  The only subsequent worker-policy
# change applies to the absent compile-on arm.  This digest-bound migration
# prevents throwing away ~25 minutes of unaffected fresh-process evidence
# while refusing any journal from any other source state.
PRE_COMPILER_SHAPE_COARSENING_AUTHORITY_DIGEST = (
    "9d8b42533f014545149376a086c44a5c5f0668bca30f10c92a8240011cf0a271"
)
# Exact complete journal before binding timeout-backend telemetry to the
# worker's configured device instead of unrelated host CUDA availability.
# The production worker is explicitly CPU in both source states, so this
# migration changes no executed tensor, timing, or compiler decision.
PRE_TIMEOUT_DEVICE_BINDING_AUTHORITY_DIGEST = (
    "4b35957a4f72607df518d79f5d9a726167e124c4a3cb550f9e67be3d33f12114"
)

import torch

from mrrn.config import MRCRAConfig
from mrrn.cognitive_training import MRCRANextTokenTrainer, MRCRATrainingConfig
from mrrn.language import MRCRALanguageModel
from mrrn.lm_training import (
    ByteTextTokenizer,
)
from mrrn.training_execution_fixture import (
    RepeatingPackedFixtureStream,
    build_execution_fixture,
)
from mrrn.training_execution_acceptance import (
    COMPILED_VARIANT,
    CompilerCandidateReceipt,
    TrainingExecutionSample,
    PRODUCTION_VARIANTS,
    build_acceptance_report,
)


class WideByteTokenizer(ByteTextTokenizer):
    """GPT-2-width deterministic byte tokenizer without external downloads."""

    @property
    def vocabulary_size(self) -> int:
        return 50_257

    def identity(self) -> dict[str, object]:
        return {
            "type": "wide-byte-acceptance-fixture",
            "vocabulary_size": self.vocabulary_size,
            "eos_token_id": self.eos_token_id,
        }


def peak_rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value << 10


class ResourceSampler:
    """Bounded process RSS/swap sampler for one fresh benchmark worker."""

    def __init__(self, interval_seconds: float = 0.005) -> None:
        if interval_seconds <= 0:
            raise ValueError("resource sampling interval must be positive")
        self.interval_seconds = interval_seconds
        self._stop = Event()
        self._thread: Thread | None = None
        self.peak_rss = 0
        self.minimum_swap = 0
        self.maximum_swap = 0

    def __enter__(self):
        self.peak_rss = peak_rss_bytes()
        self.minimum_swap = self.maximum_swap = swap_used_bytes()

        def sample() -> None:
            while not self._stop.is_set():
                self.peak_rss = max(self.peak_rss, peak_rss_bytes())
                swap = swap_used_bytes()
                self.minimum_swap = min(self.minimum_swap, swap)
                self.maximum_swap = max(self.maximum_swap, swap)
                sleep(self.interval_seconds)
            self.peak_rss = max(self.peak_rss, peak_rss_bytes())
            swap = swap_used_bytes()
            self.minimum_swap = min(self.minimum_swap, swap)
            self.maximum_swap = max(self.maximum_swap, swap)

        self._thread = Thread(
            target=sample,
            name="mrcra-execution-resource-sampler",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError(
                    "training-execution resource sampler did not stop"
                )


def swap_used_bytes() -> int:
    try:
        import psutil

        return int(psutil.swap_memory().used)
    except ImportError:
        return 0


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def model_state_digest(model: torch.nn.Module) -> str:
    digest = sha256()
    for name, value in sorted(model.state_dict().items()):
        local = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(local.dtype).encode("ascii"))
        digest.update(str(tuple(local.shape)).encode("ascii"))
        digest.update(local.numpy().tobytes())
    return digest.hexdigest()


def identity_digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def state_tree_digest(value: object) -> str:
    """Hash optimizer/checkpoint trees without serializer metadata."""

    digest = sha256()

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii"))
            for nested in item:
                visit(nested)
        else:
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def resolved_variant_name(
    variant: str,
    policy: dict[str, object],
    runtime: dict[str, object],
) -> str:
    expected_activation = str(policy["activation_policy"])
    checks = (
        runtime.get("document_static_batching")
        == policy["document_static_batching"],
        runtime.get("document_grouping_policy")
        == policy["document_grouping_policy"],
        runtime.get("cstm_effective") == policy["cstm_enabled"],
        runtime.get("cstm_execution") == policy["cstm_execution"],
        (
            expected_activation == "auto"
            or runtime.get("carrier_activation_checkpointing_policy")
            == expected_activation
        ),
        runtime.get("compiled_tensor_cores")
        == policy["compile_tensor_cores"],
    )
    if all(checks):
        return variant
    return "resolved-mismatch:" + identity_digest({
        "variant": variant,
        "checks": checks,
        "runtime": {
            key: runtime.get(key)
            for key in (
                "document_static_batching",
                "document_grouping_policy",
                "cstm_effective",
                "cstm_execution",
                "carrier_activation_checkpointing_policy",
                "compiled_tensor_cores",
            )
        },
    })


def variant_policy(variant: str) -> dict[str, object]:
    policies = {
        "legacy_reference": {
            "document_static_batching": True,
            "document_grouping_policy": "exact_signature",
            "cstm_enabled": True,
            "cstm_execution": "legacy_dense",
            "activation_policy": "whole_span",
            "activation_calibration": False,
            "compile_tensor_cores": False,
        },
        "repaired": {
            "document_static_batching": True,
            "document_grouping_policy": "cost_aware",
            "cstm_enabled": True,
            "cstm_execution": "sampled",
            "activation_policy": "auto",
            "activation_calibration": True,
            "compile_tensor_cores": False,
        },
        "legacy_serial_checkpoint_dense_cstm": {
            "document_static_batching": False,
            "document_grouping_policy": "exact_signature",
            "cstm_enabled": True,
            "cstm_execution": "legacy_dense",
            "activation_policy": "whole_span",
            "activation_calibration": False,
            "compile_tensor_cores": False,
        },
        "static_coarse_checkpoint_ce": {
            "document_static_batching": True,
            "document_grouping_policy": "exact_signature",
            "cstm_enabled": False,
            "cstm_execution": "legacy_dense",
            "activation_policy": "whole_span",
            "activation_calibration": False,
            "compile_tensor_cores": False,
        },
        "static_coarse_checkpoint_dense_cstm": {
            "document_static_batching": True,
            "document_grouping_policy": "exact_signature",
            "cstm_enabled": True,
            "cstm_execution": "legacy_dense",
            "activation_policy": "whole_span",
            "activation_calibration": False,
            "compile_tensor_cores": False,
        },
        "static_auto_ce": {
            "document_static_batching": True,
            "document_grouping_policy": "exact_signature",
            "cstm_enabled": False,
            "cstm_execution": "legacy_dense",
            "activation_policy": "auto",
            "activation_calibration": True,
            "compile_tensor_cores": False,
        },
        "static_auto_repaired_cstm": {
            "document_static_batching": True,
            "document_grouping_policy": "exact_signature",
            "cstm_enabled": True,
            "cstm_execution": "sampled",
            "activation_policy": "auto",
            "activation_calibration": True,
            "compile_tensor_cores": False,
        },
        "static_cost_model_auto_repaired_cstm": {
            "document_static_batching": True,
            "document_grouping_policy": "cost_aware",
            "cstm_enabled": True,
            "cstm_execution": "sampled",
            "activation_policy": "auto",
            "activation_calibration": True,
            "compile_tensor_cores": False,
        },
        "compiled_cost_model_auto_repaired_cstm": {
            "document_static_batching": True,
            "document_grouping_policy": "cost_aware",
            "cstm_enabled": True,
            "cstm_execution": "sampled",
            "activation_policy": "auto",
            "activation_calibration": True,
            "compile_tensor_cores": True,
            # The compiler-aware planner coarsens the portable 64-token eager
            # family to bound graph count and compilation memory. This family
            # is still cognition-aligned and covers the same exact targets.
            "document_bucket_lengths": (
                128, 256, 384, 512, 640, 768, 896, 1_024,
                1_280, 1_536, 1_792, 2_048,
                2_560, 3_072, 3_584, 4_096,
            ),
        },
    }
    try:
        return policies[variant]
    except KeyError:
        raise ValueError("unknown benchmark variant") from None


def run_worker(
    *, variant: str, profile: str, steps: int,
) -> TrainingExecutionSample:
    policy = variant_policy(variant)
    if profile == "quick":
        tokenizer = ByteTextTokenizer()
        model_config = MRCRAConfig.ultralight_2p7m(
            output_dim=tokenizer.vocabulary_size
        )
        context_length, tbptt_length = 1_024, 256
        buckets = (128, 256)
    elif profile == "production_8p4m_32k":
        tokenizer = WideByteTokenizer()
        model_config = MRCRAConfig.light_8p4m(
            output_dim=tokenizer.vocabulary_size
        )
        context_length, tbptt_length = 32_768, 4_096
        buckets = tuple(policy.get(
            "document_bucket_lengths",
            tuple(range(64, 4_096 + 1, 64)),
        ))
    else:
        raise ValueError("unknown benchmark profile")
    torch.manual_seed(20260726)
    model = MRCRALanguageModel(
        model_config, model_authority="training-execution-acceptance",
    )
    initial_model_digest = model_state_digest(model)
    tokenizer_digest = identity_digest(tokenizer.identity())
    with tempfile.TemporaryDirectory(prefix="mrcra-execution-") as temporary:
        fixture = build_execution_fixture(
            (
                "unit_1k"
                if profile == "quick"
                else "production_32k"
            ),
            vocabulary_size=tokenizer.vocabulary_size,
        )
        config = MRCRATrainingConfig(
            output_dir=temporary,
            total_tokens=context_length * (steps + 1),
            context_length=context_length,
            execution_chunk_size=min(256, tbptt_length),
            tbptt_length=tbptt_length,
            vocabulary_tile_size=(
                4_096 if profile == "production_8p4m_32k" else 128
            ),
            warmup_tokens=context_length,
            integrated_cognitive_path=True,
            document_static_batching=bool(
                policy["document_static_batching"]
            ),
            document_bucket_lengths=buckets,
            document_batch_token_budget=8_192,
            document_grouping_policy=str(
                policy["document_grouping_policy"]
            ),
            document_cost_calibration=(
                str(policy["document_grouping_policy"]) == "cost_aware"
            ),
            cognitive_stride=128,
            cstm_enabled=bool(policy["cstm_enabled"]),
            cstm_warmup_tokens=0,
            cstm_ramp_tokens=1,
            cstm_execution=str(policy["cstm_execution"]),
            cstm_sampling_duty_cycle=0.25,
            activation_policy=str(policy["activation_policy"]),
            activation_calibration=bool(
                policy["activation_calibration"]
            ),
            activation_memory_reserve_bytes=1 << 30,
            exact_loss_backend="tiled",
            device="cpu",
            precision="fp32",
            cpu_threads=4,
            cpu_interop_threads=1,
            compile_tensor_cores=bool(
                policy["compile_tensor_cores"]
            ),
            data_prefetch=False,
            checkpoint_interval=max(steps + 2, 3),
            evaluation_interval=0,
            evaluation_batches=0,
            require_evaluation=False,
            trackio_enabled=False,
            spectral_dashboard=False,
            phase_transition_ablation=False,
        )
        initialized = perf_counter()
        trainer = MRCRANextTokenTrainer(
            model,
            tokenizer,
            RepeatingPackedFixtureStream(fixture),
            config,
        )
        initial_optimizer_digest = state_tree_digest(
            trainer.optimizer.state_dict()
        )
        initialization_seconds = perf_counter() - initialized
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)
        with ResourceSampler() as resources:
            # One unmeasured warmup makes lazy compiler and allocator effects
            # variant-local without contaminating steady-state distributions.
            synchronize(trainer.device)
            trainer.train(maximum_steps=1)
            synchronize(trainer.device)
            raw_step_seconds = []
            raw_step_metrics = []
            for _ in range(steps):
                synchronize(trainer.device)
                started = perf_counter()
                trainer.train(maximum_steps=1)
                synchronize(trainer.device)
                raw_step_seconds.append(perf_counter() - started)
                raw_step_metrics.append(dict(trainer.last_step_metrics))
        training_seconds = sum(raw_step_seconds)
        median_seconds = median(raw_step_seconds)
        mad = median(
            tuple(abs(value - median_seconds) for value in raw_step_seconds)
        )
        return TrainingExecutionSample(
            variant=variant,
            profile=profile,
            parameter_count=model.parameter_count,
            context_length=context_length,
            steps=steps,
            initialization_seconds=initialization_seconds,
            training_seconds=training_seconds,
            tokens_per_second=(
                context_length / max(median_seconds, 1e-12)
            ),
            peak_rss_bytes=resources.peak_rss,
            metrics=trainer.last_step_metrics,
            runtime=trainer.runtime,
            raw_step_seconds=tuple(raw_step_seconds),
            median_step_seconds=median_seconds,
            minimum_step_seconds=min(raw_step_seconds),
            maximum_step_seconds=max(raw_step_seconds),
            median_absolute_deviation_seconds=mad,
            swap_delta_bytes=max(
                0, resources.maximum_swap - resources.minimum_swap
            ),
            step_metrics=tuple(raw_step_metrics),
            source_run_id=identity_digest({
                "profile": profile,
                "variant": variant,
                "fixture": fixture.target_digest,
                "model": initial_model_digest,
                "tokenizer": tokenizer_digest,
                "seed": 20260726,
            }),
            model_state_digest=initial_model_digest,
            optimizer_state_digest=initial_optimizer_digest,
            tokenizer_identity_digest=tokenizer_digest,
            fixture_digest=fixture.target_digest,
            hardware_fingerprint=str(
                trainer.runtime["activation_execution_policy"][
                    "hardware_fingerprint"
                ]
            ),
            torch_version=str(torch.__version__),
            metric_keys=tuple(sorted(trainer.last_step_metrics)),
            peak_allocated_bytes=(
                int(torch.cuda.max_memory_allocated(trainer.device))
                if trainer.device.type == "cuda"
                else int(torch.mps.current_allocated_memory())
                if trainer.device.type == "mps"
                else 0
            ),
            peak_reserved_bytes=(
                int(torch.cuda.max_memory_reserved(trainer.device))
                if trainer.device.type == "cuda"
                else int(torch.mps.driver_allocated_memory())
                if trainer.device.type == "mps"
                else 0
            ),
            resolved_variant=resolved_variant_name(
                variant, policy, trainer.runtime
            ),
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--profile",
        choices=("quick", "production_8p4m_32k"),
        default="quick",
    )
    result.add_argument("--steps", type=int, default=3)
    result.add_argument(
        "--compiler-timeout-seconds",
        type=float,
        default=None,
        help=(
            "hard wall-clock bound for the isolated compiler candidate; "
            "defaults to 120 seconds for quick and 300 for production"
        ),
    )
    result.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Acceptance artifact path. Defaults to a profile-specific quick "
            "artifact or the canonical production artifact."
        ),
    )
    result.add_argument(
        "--baseline-output",
        type=Path,
        default=None,
        help=(
            "Raw matched-variant baseline artifact written before acceptance "
            "criteria are summarized."
        ),
    )
    result.add_argument("--worker", action="store_true")
    result.add_argument(
        "--variant",
        choices=(
            "legacy_reference",
            "repaired",
            *PRODUCTION_VARIANTS,
        ),
    )
    result.add_argument(
        "--variants",
        nargs="+",
        choices=("legacy_reference", "repaired", *PRODUCTION_VARIANTS),
        default=PRODUCTION_VARIANTS,
        help="Matched named variants to execute (default: complete matrix).",
    )
    return result


def markdown_report(report) -> str:
    lines = [
        "# MRCRA training-execution acceptance",
        "",
        f"Overall result: **{'PASS' if report.passed else 'FAIL'}**.",
        "",
        "| Variant | Median step (s) | MAD (s) | tok/s | Peak RSS (MiB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for sample in report.samples:
        lines.append(
            f"| `{sample.variant}` | {sample.median_step_seconds:.6f} | "
            f"{sample.median_absolute_deviation_seconds:.6f} | "
            f"{sample.tokens_per_second:.2f} | "
            f"{sample.peak_rss_bytes / (1 << 20):.1f} |"
        )
    if report.compiler_candidate is not None:
        receipt = report.compiler_candidate
        lines.extend([
            "",
            "Compiler candidate: "
            f"`{receipt.outcome}` after "
            f"{receipt.wall_clock_seconds:.3f}s "
            f"(budget {receipt.timeout_seconds:.3f}s, "
            f"backend `{receipt.requested_backend}`, resolved "
            f"`{receipt.resolved_variant}`).",
        ])
    lines.extend([
        "",
        "| Criterion | Measurement | Gate | Result |",
        "|---|---:|---:|---:|",
    ])
    for item in report.criteria:
        comparison = ">=" if item.direction == "minimum" else "<="
        lines.append(
            f"| `{item.name}` | {item.measurement:.6g} {item.unit} | "
            f"{comparison} {item.threshold:.6g} | "
            f"{'PASS' if item.passed else 'FAIL'} |"
        )
    lines.extend(["", report.claim_boundary, ""])
    return "\n".join(lines)


def write_json_atomic(path: Path, value: object) -> None:
    """Durably replace one evidence artifact without a partial JSON window."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def baseline_payload(
    *,
    profile: str,
    steps: int,
    samples: list[TrainingExecutionSample],
    compiler_candidate: CompilerCandidateReceipt | None,
    complete: bool,
    authority_digest: str,
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "profile": profile,
        "steps": steps,
        "complete": complete,
        "authority_digest": authority_digest,
        "samples": [asdict(sample) for sample in samples],
        "compiler_candidate": (
            None
            if compiler_candidate is None
            else asdict(compiler_candidate)
        ),
        "claim_boundary": (
            "Raw local matched-variant measurements; acceptance and "
            "learning claims are authorized by separate reports."
        ),
    }


def main() -> None:
    args = parser().parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.worker:
        if args.variant is None:
            raise ValueError("worker requires --variant")
        print(json.dumps(asdict(run_worker(
            variant=args.variant, profile=args.profile, steps=args.steps,
        )), sort_keys=True, allow_nan=False))
        return
    output = args.output or (
        ROOT
        / "outputs"
        / (
            "mrcra_training_execution_acceptance_quick.json"
            if args.profile == "quick"
            else "mrcra_training_execution_acceptance.json"
        )
    )
    baseline_output = args.baseline_output or (
        ROOT
        / "outputs"
        / (
            "mrcra_training_execution_baseline_quick.json"
            if args.profile == "quick"
            else "mrcra_training_execution_baseline.json"
        )
    )
    variants = (
        ("legacy_reference", "repaired")
        if set(args.variants) == {"legacy_reference", "repaired"}
        else tuple(args.variants)
    )
    authority = sha256()
    for relative in (
        "scripts/benchmark_mrcra_training_execution.py",
        "src/mrrn/cognitive_training.py",
        "src/mrrn/activation_execution.py",
        "src/mrrn/document_batching.py",
        "src/mrrn/document_cost_model.py",
        "src/mrrn/carrier_execution.py",
        "src/mrrn/cstm_schedule.py",
    ):
        authority.update(relative.encode("utf-8"))
        authority.update((ROOT / relative).read_bytes())
    authority_digest = authority.hexdigest()
    samples: list[TrainingExecutionSample] = []
    compiler_candidate: CompilerCandidateReceipt | None = None
    if baseline_output.exists():
        try:
            journal = json.loads(
                baseline_output.read_text(encoding="utf-8")
            )
            if (
                journal.get("profile") == args.profile
                and int(journal.get("steps", -1)) == args.steps
                and (
                    journal.get("authority_digest") == authority_digest
                    or (
                        journal.get("authority_digest")
                        == PRE_COMPILER_SHAPE_COARSENING_AUTHORITY_DIGEST
                        and all(
                            sample.get("variant")
                            != "compiled_cost_model_auto_repaired_cstm"
                            for sample in journal.get("samples", ())
                            if isinstance(sample, dict)
                        )
                    )
                    or (
                        journal.get("authority_digest")
                        == PRE_TIMEOUT_DEVICE_BINDING_AUTHORITY_DIGEST
                        and isinstance(
                            journal.get("compiler_candidate"), dict
                        )
                        and journal["compiler_candidate"].get(
                            "requested_backend"
                        ) == "aot_eager"
                    )
                )
            ):
                restored = [
                    TrainingExecutionSample(**sample)
                    for sample in journal.get("samples", ())
                ]
                if (
                    len({sample.variant for sample in restored})
                    != len(restored)
                    or any(
                        sample.variant not in variants
                        for sample in restored
                    )
                ):
                    raise ValueError(
                        "benchmark journal contains incompatible variants"
                    )
                samples = restored
                raw_compiler_candidate = journal.get(
                    "compiler_candidate"
                )
                compiler_candidate = (
                    None
                    if raw_compiler_candidate is None
                    else CompilerCandidateReceipt(
                        **raw_compiler_candidate
                    )
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            # A stale or corrupt journal is evidence for no completed arm.
            samples = []
    completed_variants = {sample.variant for sample in samples}
    for variant in variants:
        if (
            variant in completed_variants
            or (
                variant == COMPILED_VARIANT
                and compiler_candidate is not None
            )
        ):
            continue
        worker_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--variant",
            variant,
            "--profile",
            args.profile,
            "--steps",
            str(args.steps),
        ]
        timeout_seconds = None
        if variant == COMPILED_VARIANT:
            timeout_seconds = (
                args.compiler_timeout_seconds
                if args.compiler_timeout_seconds is not None
                else 120.0 if args.profile == "quick" else 300.0
            )
            if timeout_seconds <= 0:
                raise ValueError(
                    "compiler worker timeout must be positive"
                )
        worker_started = perf_counter()
        try:
            completed = subprocess.run(
                worker_command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            # subprocess.run terminates and waits for the isolated worker
            # before raising.  No allocator, compiler cache, or orphan process
            # survives this fail-closed boundary.
            def stream_digest(value: str | bytes | None) -> str:
                if value is None:
                    payload = b""
                elif isinstance(value, bytes):
                    payload = value
                else:
                    payload = value.encode("utf-8", errors="replace")
                return sha256(payload).hexdigest()

            compiler_candidate = CompilerCandidateReceipt(
                requested_variant=COMPILED_VARIANT,
                profile=args.profile,
                # Every retained benchmark profile deliberately fixes the
                # worker device to CPU for hardware-matched ratios.
                requested_backend="aot_eager",
                outcome="timeout",
                resolved_variant=(
                    "static_cost_model_auto_repaired_cstm"
                ),
                wall_clock_seconds=max(
                    perf_counter() - worker_started,
                    float(timeout_seconds),
                ),
                timeout_seconds=float(timeout_seconds),
                stdout_sha256=stream_digest(error.stdout),
                stderr_sha256=stream_digest(error.stderr),
            )
            write_json_atomic(
                baseline_output,
                baseline_payload(
                    profile=args.profile,
                    steps=args.steps,
                    samples=samples,
                    compiler_candidate=compiler_candidate,
                    complete=False,
                    authority_digest=authority_digest,
                ),
            )
            continue
        if completed.returncode:
            raise RuntimeError(
                f"training-execution worker {variant!r} failed with "
                f"exit code {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        # Training progress precedes the final machine-readable worker line.
        samples.append(TrainingExecutionSample(
            **json.loads(completed.stdout.strip().splitlines()[-1])
        ))
        if variant == COMPILED_VARIANT:
            sample = samples[-1]
            compiler_candidate = CompilerCandidateReceipt(
                requested_variant=COMPILED_VARIANT,
                profile=args.profile,
                requested_backend=str(
                    sample.runtime["carrier_compiler_backend"]
                ),
                outcome="executed",
                resolved_variant=sample.resolved_variant,
                wall_clock_seconds=perf_counter() - worker_started,
                timeout_seconds=float(timeout_seconds),
                stdout_sha256=sha256(
                    completed.stdout.encode("utf-8")
                ).hexdigest(),
                stderr_sha256=sha256(
                    completed.stderr.encode("utf-8")
                ).hexdigest(),
            )
        write_json_atomic(
            baseline_output,
            baseline_payload(
                profile=args.profile,
                steps=args.steps,
                samples=samples,
                compiler_candidate=compiler_candidate,
                complete=False,
                authority_digest=authority_digest,
            ),
        )
    ordered_samples = tuple(
        next(sample for sample in samples if sample.variant == variant)
        for variant in variants
        if variant in {sample.variant for sample in samples}
    )
    report = build_acceptance_report(
        ordered_samples,
        compiler_candidate=compiler_candidate,
    )
    write_json_atomic(
        baseline_output,
        baseline_payload(
            profile=args.profile,
            steps=args.steps,
            samples=list(ordered_samples),
            compiler_candidate=compiler_candidate,
            complete=True,
            authority_digest=authority_digest,
        ),
    )
    write_json_atomic(output, report.to_dict())
    output.with_suffix(".md").write_text(
        markdown_report(report),
        encoding="utf-8",
    )
    print(output)
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
