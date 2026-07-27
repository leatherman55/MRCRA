#!/usr/bin/env python3
"""Run the 100-step MRCRA resource, periodic-work, and resume soak."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
import multiprocessing as mp
from pathlib import Path
import os
import sys
import tempfile
from threading import enumerate as enumerate_threads
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
from mrrn.lm_training import ByteTextTokenizer
from mrrn.resource_soak_acceptance import (
    ResourceSoakSample,
    build_resource_soak_report,
)
from mrrn.training_execution_fixture import (
    RepeatingPackedFixtureStream,
    build_execution_fixture,
)


class WideByteTokenizer(ByteTextTokenizer):
    @property
    def vocabulary_size(self) -> int:
        return 50_257

    def identity(self) -> dict[str, object]:
        return {
            "type": "wide-byte-resource-soak",
            "vocabulary_size": self.vocabulary_size,
            "eos_token_id": self.eos_token_id,
        }


def rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value << 10


def source_authority_digest() -> str:
    authority = sha256()
    for relative in (
        "scripts/run_mrcra_resource_soak.py",
        "src/mrrn/resource_soak_acceptance.py",
        "src/mrrn/cognitive_training.py",
        "src/mrrn/cstm.py",
        "src/mrrn/cstm_sampling.py",
        "src/mrrn/cstm_schedule.py",
        "src/mrrn/document_batching.py",
    ):
        authority.update(relative.encode("utf-8"))
        authority.update((ROOT / relative).read_bytes())
    return authority.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
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


def build_trainer(
    *,
    profile: str,
    output_dir: Path,
    steps: int,
    seed: int,
) -> MRCRANextTokenTrainer:
    if profile == "quick":
        tokenizer = ByteTextTokenizer()
        model_config = MRCRAConfig.ultralight_1p3m(
            output_dim=tokenizer.vocabulary_size
        )
        fixture_profile = "unit_1k"
        context, tbptt = 1_024, 256
        buckets = (128, 256)
        evaluation_interval = checkpoint_interval = 25
    elif profile == "production_8p4m_32k":
        tokenizer = WideByteTokenizer()
        model_config = MRCRAConfig.light_8p4m(
            output_dim=tokenizer.vocabulary_size
        )
        fixture_profile = "production_32k"
        context, tbptt = 32_768, 4_096
        buckets = tuple(range(64, 4_096 + 1, 64))
        evaluation_interval = checkpoint_interval = 25
    else:
        raise ValueError("unknown resource soak profile")
    fixture = build_execution_fixture(
        fixture_profile,
        vocabulary_size=tokenizer.vocabulary_size,
        seed=seed,
    )
    torch.manual_seed(seed)
    model = MRCRALanguageModel(
        model_config, model_authority="resource-soak-acceptance"
    )
    config = MRCRATrainingConfig(
        output_dir=str(output_dir),
        total_tokens=context * steps,
        context_length=context,
        execution_chunk_size=min(256, tbptt),
        tbptt_length=tbptt,
        vocabulary_tile_size=(
            4_096 if profile == "production_8p4m_32k" else 128
        ),
        warmup_tokens=context,
        integrated_cognitive_path=True,
        document_static_batching=True,
        document_bucket_lengths=buckets,
        document_batch_token_budget=8_192,
        document_grouping_policy="cost_aware",
        cognitive_stride=128,
        cstm_enabled=True,
        cstm_warmup_tokens=0,
        cstm_ramp_tokens=1,
        cstm_execution="sampled",
        cstm_sampling_duty_cycle=0.25,
        activation_policy="auto",
        activation_calibration=True,
        activation_memory_reserve_bytes=1 << 30,
        exact_loss_backend="tiled",
        device="cpu",
        precision="fp32",
        cpu_threads=4,
        cpu_interop_threads=1,
        data_prefetch=False,
        checkpoint_interval=checkpoint_interval,
        keep_checkpoints=4,
        evaluation_interval=evaluation_interval,
        evaluation_batches=1,
        require_evaluation=True,
        trackio_enabled=False,
        spectral_dashboard=False,
        phase_transition_ablation=False,
        seed=seed,
    )
    return MRCRANextTokenTrainer(
        model,
        tokenizer,
        RepeatingPackedFixtureStream(fixture),
        config,
        (fixture.batch,),
    )


def _phase_worker(
    connection,
    *,
    profile: str,
    output_dir: str,
    total_steps: int,
    phase_steps: int,
    seed: int,
    checkpoint: str | None,
) -> None:
    """Execute one side of the resume boundary in an isolated process."""

    before_threads = {
        thread.ident
        for thread in enumerate_threads()
        if thread.name.startswith(("mrrn-", "mrcra-", "trackio"))
    }
    rss: list[int] = []
    nonfinite = 0

    def observe(_state, metrics):
        nonlocal nonfinite
        rss.append(rss_bytes())
        nonfinite += sum(not isfinite(value) for value in metrics.values())

    try:
        trainer = build_trainer(
            profile=profile,
            output_dir=Path(output_dir),
            steps=total_steps,
            seed=seed,
        )
        if checkpoint is not None:
            trainer.load_checkpoint(Path(checkpoint))
        started = perf_counter()
        trainer.train(maximum_steps=phase_steps, step_observer=observe)
        elapsed = perf_counter() - started
        saved = trainer.save_checkpoint()
        after_threads = {
            thread.ident
            for thread in enumerate_threads()
            if thread.name.startswith(("mrrn-", "mrcra-", "trackio"))
        }
        connection.send({
            "ok": True,
            "rss": rss,
            "nonfinite": nonfinite,
            "elapsed": elapsed,
            "accounted_elapsed": trainer.state.elapsed_seconds,
            "state_step": trainer.state.step,
            "checkpoint": str(saved),
            "stale_threads": len(
                (after_threads - before_threads) - {None}
            ),
            "maximum_coverage_gap": int(
                trainer.last_step_metrics.get(
                    "cstm/coverage_gap_max", 0
                )
            ),
            "declared_maximum_coverage_gap": int(
                trainer.config.cstm_maximum_coverage_gap
            ),
            "stream_batches_emitted": int(
                trainer.train_stream.state_dict()["batches_emitted"]
            ),
        })
    except BaseException:
        connection.send({
            "ok": False,
            "traceback": traceback.format_exc(),
        })
        raise
    finally:
        connection.close()


def _run_phase_process(**kwargs) -> dict:
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_phase_worker,
        kwargs={"connection": child, **kwargs},
        name="mrcra-resource-soak-phase",
    )
    process.start()
    child.close()
    result = parent.recv()
    parent.close()
    process.join()
    if process.exitcode != 0 or not result.get("ok"):
        raise RuntimeError(
            "isolated resource-soak phase failed:\n"
            + str(result.get("traceback", "child exited without traceback"))
        )
    return result


def worker(
    *, profile: str, steps: int, seed: int, run_dir: Path,
) -> ResourceSoakSample:
    if steps < 100:
        raise ValueError("resource soak requires at least 100 steps")
    run_dir.mkdir(parents=True, exist_ok=True)
    midpoint = steps // 2
    common = {
        "profile": profile,
        "output_dir": str(run_dir),
        "seed": seed,
    }
    first = _run_phase_process(
        **common,
        total_steps=steps,
        phase_steps=midpoint,
        checkpoint=None,
    )
    second = _run_phase_process(
        **common,
        total_steps=steps,
        phase_steps=steps - midpoint,
        checkpoint=first["checkpoint"],
    )
    checkpoints = list((run_dir / "checkpoints").glob("step-*.pt"))
    temporary_files = list(run_dir.rglob("*.tmp"))
    rss = tuple(first["rss"]) + tuple(second["rss"])
    return ResourceSoakSample(
        profile,
        steps,
        rss,
        float(first["elapsed"]) + float(second["elapsed"]),
        float(second["accounted_elapsed"]),
        len(checkpoints),
        int(second["state_step"]) == steps,
        len(temporary_files),
        int(first["stale_threads"]) + int(second["stale_threads"]),
        max(
            int(first["maximum_coverage_gap"]),
            int(second["maximum_coverage_gap"]),
        ),
        int(second["declared_maximum_coverage_gap"]),
        int(second["stream_batches_emitted"]) == steps,
        int(first["nonfinite"]) + int(second["nonfinite"]),
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--profile",
        choices=("quick", "production_8p4m_32k"),
        default="quick",
    )
    result.add_argument("--steps", type=int, default=100)
    result.add_argument("--seed", type=int, default=20260726)
    result.add_argument("--run-dir", type=Path)
    result.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return result


def main() -> None:
    args = parser().parse_args()
    if args.run_dir is None:
        with tempfile.TemporaryDirectory(
            prefix="mrcra-resource-soak-"
        ) as temporary:
            sample = worker(
                profile=args.profile,
                steps=args.steps,
                seed=args.seed,
                run_dir=Path(temporary),
            )
    else:
        sample = worker(
            profile=args.profile,
            steps=args.steps,
            seed=args.seed,
            run_dir=args.run_dir,
        )
    report = build_resource_soak_report(sample)
    output = args.output or (
        ROOT
        / "outputs"
        / (
            "mrcra_resource_soak_acceptance_quick.json"
            if args.profile == "quick"
            else "mrcra_resource_soak_acceptance.json"
        )
    )
    payload = report.to_dict()
    payload["source_authority_digest"] = source_authority_digest()
    payload["controls"] = {
        "profile": args.profile,
        "steps": args.steps,
        "seed": args.seed,
    }
    write_json_atomic(output, payload)
    print(output)
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
