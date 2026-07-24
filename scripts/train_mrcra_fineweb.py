#!/usr/bin/env python3
"""Train the production MRCRA actor on original English FineWeb."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault("PYTHONWARNINGS", "ignore:resource_tracker:UserWarning")

import torch

from mrrn.cognitive_training import MRCRANextTokenTrainer, MRCRATrainingConfig
from mrrn.learning_progress import LearningProgressConfig
from mrrn.config import CognitiveConfig, MRCRAConfig, MRRNConfig
from mrrn.language import MRCRALanguageModel
from mrrn.lm_training import (
    build_evaluation_batches,
    ByteTextTokenizer, FineWebTextSource, HuggingFaceTextTokenizer,
    PackedTokenStream, SequenceTextSource,
)


def resolve_revision(repo_id: str, revision: str, *, repo_type: str) -> str:
    from huggingface_hub import HfApi

    info = (
        HfApi().dataset_info(repo_id, revision=revision)
        if repo_type == "dataset" else HfApi().model_info(repo_id, revision=revision)
    )
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a commit SHA for {repo_id}")
    return info.sha


def latest_checkpoint(output_dir: Path) -> Path | None:
    pointer = output_dir / "checkpoints" / "latest.json"
    if not pointer.is_file():
        return None
    target = pointer.parent / json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"]
    if not target.is_file():
        raise FileNotFoundError(f"latest checkpoint pointer names missing file {target}")
    return target


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class GPT2WidthByteSmokeTokenizer(ByteTextTokenizer):
    """Dependency-free byte encoding with the production GPT-2 tensor width.

    This is used only to exercise a real production-profile construction in a
    deterministic offline smoke test. It does not claim GPT-2 token semantics.
    """

    vocabulary_size = 50_257
    eos_token_id = 50_256

    def identity(self) -> dict:
        return {
            "kind": "utf8-bytes-production-width-smoke",
            "vocabulary_size": self.vocabulary_size,
            "eos_token_id": self.eos_token_id,
            "semantic_tokenizer": False,
        }


def tiny_configuration(vocabulary_size: int) -> MRCRAConfig:
    carrier = MRRNConfig(
        input_dim=8, model_dim=8, output_dim=vocabulary_size, layers=1, scales=2,
        heads=2, modes=2, mimo_rank=1, attention_window=2,
        attention_query_tile_size=2, retrieved_items=1, memory_capacity=4,
        mixer_expansion=1.5, width_growth_cap=1, mode_growth_cap=1,
        width_multiple=2, spectral_modes=2, spectral_basis_order=2,
        spectral_triads_per_mode=1, enable_global_head=False,
        relational_branch=True, relational_context_dim=8,
        activation_checkpointing=True,
    )
    cognitive = CognitiveConfig(
        workspace_dim=8, provenance_features=4, uncertainty_channels=8,
        relation_heads=2, relation_modes=2, relation_adapter_rank=2,
        goal_slots=1, goal_constraint_dim=2, system_action_channels=2,
        calibration_regimes=2, active_event_capacity=4, pair_edge_capacity=8,
        hyperedge_capacity=2, maximum_hyperedge_arity=3, graph_neighbors=1,
        global_workspace_slots=2, hypothesis_slots=1, maximum_hypothesis_slots=2,
        maximum_cognitive_steps=1, event_chunk_size=2,
        event_proposals_per_chunk=1, recent_candidates=2,
        landmark_candidates=1, episodic_candidates=1, semantic_candidates=1,
        episodic_memory_capacity=4, semantic_memory_capacity=2,
        associative_depth=1, associative_budget=1, world_model_horizons=(1,),
        enable_conditional_reconstruction=True,
        enable_abstraction_validity_control=True,
        enable_post_deliberation_action_selection=True,
        enable_multi_hypothesis_planning=True,
        enable_agent_session_loop=True,
        enable_viability_gate=True,
        enable_integrated_invariant_discovery=True,
        enable_persistent_session_training=True,
        enable_metacognitive_routing=True,
    )
    return MRCRAConfig(
        carrier, cognitive, actor_parameter_minimum=1,
        actor_parameter_maximum=10_000_000,
    )


def production_configuration(
    vocabulary_size: int, *, lightmodel: bool,
    ultralightmodel: bool = False,
) -> MRCRAConfig:
    """Select the declared production actor profile without relaxing its budget."""

    if lightmodel and ultralightmodel:
        raise ValueError("lightmodel and ultralightmodel are mutually exclusive")
    factory = (
        MRCRAConfig.ultralight_1p3m
        if ultralightmodel
        else MRCRAConfig.light_8p4m
        if lightmodel
        else MRCRAConfig.serious_120m
    )
    return factory(output_dim=vocabulary_size)


def _host_memory_capacity_bytes() -> int:
    """Return stable physical host capacity without adding a runtime dependency."""

    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(
            os.sysconf("SC_PHYS_PAGES")
        )
    except (AttributeError, OSError, TypeError, ValueError):
        try:
            import psutil

            return int(psutil.virtual_memory().total)
        except (ImportError, AttributeError):
            return 8 << 30


def resolve_activation_checkpointing(
    configuration: MRCRAConfig,
    *,
    tbptt_length: int,
    device: str,
    precision: str,
    override: bool | None,
) -> tuple[MRCRAConfig, dict[str, int | float | str | bool]]:
    """Select carrier recomputation from a conservative live-memory estimate.

    The estimate intentionally models saved backward intermediates rather than
    parameter count. Shared-depth models still execute every physical scale and
    refinement pass, so their activation demand scales with TBPTT length, scale
    widths, and executed depth even when their parameter count is tiny.
    """

    use_cuda = (
        device.startswith("cuda")
        or (device == "auto" and torch.cuda.is_available())
    )
    if use_cuda:
        index = (
            int(device.split(":", 1)[1])
            if device.startswith("cuda:") else torch.cuda.current_device()
        )
        capacity = int(torch.cuda.get_device_properties(index).total_memory)
        budget = min(8 << 30, capacity // 4)
        element_bytes = 4 if precision == "fp32" else 2
        memory_kind = "cuda_device_capacity"
    else:
        capacity = _host_memory_capacity_bytes()
        budget = min(2 << 30, capacity // 8)
        element_bytes = 4
        memory_kind = "host_physical_capacity"
    carrier = configuration.carrier
    executed_width = sum(scale.width for scale in carrier.scale_configs())
    estimated = (
        tbptt_length
        * executed_width
        * carrier.layers
        * element_bytes
        * 96
    )
    if override is None:
        enabled = estimated > budget
        policy = (
            "automatic_recompute_over_budget"
            if enabled else "automatic_retain_within_budget"
        )
    else:
        enabled = override
        policy = "explicit_recompute" if enabled else "explicit_retain"
    resolved = replace(
        configuration,
        carrier=replace(carrier, activation_checkpointing=enabled),
    )
    return resolved, {
        "carrier_activation_checkpointing": enabled,
        "carrier_activation_checkpointing_policy": policy,
        "estimated_uncheckpointed_carrier_activation_bytes": estimated,
        "carrier_activation_memory_budget_bytes": budget,
        "activation_memory_capacity_bytes": capacity,
        "activation_memory_capacity_kind": memory_kind,
        "activation_estimate_element_bytes": element_bytes,
    }


def production_cognitive_stride(
    configuration: MRCRAConfig,
    *,
    ultralightmodel: bool,
    override: int | None,
) -> int:
    """Resolve the measured cognition cadence while preserving explicit control."""

    if override is not None:
        return override
    if ultralightmodel:
        return 128
    return configuration.cognitive.event_chunk_size


@dataclass(frozen=True, slots=True)
class ProductionProfile:
    """Stable names and paths associated with one declared actor profile."""

    name: str
    model_authority: str
    output_directory: str
    run_name: str


def production_profile(
    *, lightmodel: bool, ultralightmodel: bool, total_tokens: int,
) -> ProductionProfile:
    if lightmodel and ultralightmodel:
        raise ValueError("lightmodel and ultralightmodel are mutually exclusive")
    if ultralightmodel:
        return ProductionProfile(
            name="mrcra_1p3m_ultralight",
            model_authority="mrcra-ultralight-1p3m-fineweb-stage1",
            output_directory=(
                f"outputs/mrcra-1p3m-fineweb-{total_tokens}-tokens"
            ),
            run_name=(
                f"mrcra-1p3m-ultralight-integrated-fineweb-"
                f"{total_tokens}-tokens-32k"
            ),
        )
    if lightmodel:
        return ProductionProfile(
            name="mrcra_8p4m_light",
            model_authority="mrcra-light-8p4m-fineweb-stage1",
            output_directory=(
                f"outputs/mrcra-8p4m-fineweb-{total_tokens}-tokens"
            ),
            run_name=(
                f"mrcra-8p4m-light-integrated-fineweb-"
                f"{total_tokens}-tokens-32k"
            ),
        )
    return ProductionProfile(
        name="mrcra_120m_serious",
        model_authority="mrcra-fineweb-stage1",
        output_directory="outputs/mrcra-120m-fineweb-20m",
        run_name=f"mrcra-120m-fineweb-{total_tokens}-tokens-32k",
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Train a complete MRCRA actor on original English FineWeb with 32K contexts."
    )
    result.add_argument("--dataset-id", default="HuggingFaceFW/fineweb")
    result.add_argument("--dataset-config", default="sample-10BT")
    result.add_argument("--dataset-revision", default="main")
    result.add_argument("--tokenizer", default="gpt2")
    result.add_argument("--tokenizer-revision", default="main")
    result.add_argument(
        "--output-dir",
        help=(
            "Run directory; defaults to the selected 120M, 8.4M, or 1.3M "
            "FineWeb profile directory."
        ),
    )
    size = result.add_mutually_exclusive_group()
    size.add_argument(
        "--lightmodel", action="store_true",
        help=(
            "Train the complete shared-depth 8.4M MRCRA profile instead of "
            "the default 120M-class actor."
        ),
    )
    size.add_argument(
        "--ultralightmodel", action="store_true",
        help=(
            "Train the complete six-scale 1.3M MRCRA profile instead of the "
            "default 120M-class actor."
        ),
    )
    result.add_argument("--total-tokens", type=int, default=20_000_000)
    result.add_argument("--context-length", type=int, default=32_768)
    result.add_argument("--execution-chunk-size", type=int, default=256)
    result.add_argument(
        "--tbptt-length", type=int,
        help="Gradient span; defaults to 4,096 tokens.",
    )
    result.add_argument(
        "--vocabulary-tile-size", type=int, default=2_048,
        help="Exact-softmax vocabulary tile width (memory control, not an approximation).",
    )
    result.add_argument(
        "--loss-memory-policy", choices=("auto", "retain", "recompute"),
        default="auto",
        help=(
            "Retain exact-softmax tile activations, recompute them in backward, "
            "or select automatically from the declared workspace limit."
        ),
    )
    result.add_argument(
        "--maximum-retained-loss-mib", type=int, default=1_024,
        help="Auto-policy ceiling for retained exact-softmax activations.",
    )
    result.add_argument(
        "--progress-interval-tokens", type=int, default=2_048,
        help="Print progress this often inside a long 32K optimization context.",
    )
    result.add_argument(
        "--cognitive-stride", type=int,
        help=(
            "Causal event-cognition cadence. The measured ultralight default is "
            "128 tokens; larger profiles use their architectural event chunk."
        ),
    )
    result.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Recompute carrier activations in backward. By default this is "
            "selected from executed width, TBPTT span, precision, and memory "
            "capacity; either boolean flag is an explicit override."
        ),
    )
    result.add_argument(
        "--cognitive-tbptt-events", type=int, default=4,
        help="Backpropagate through this many consecutive cognitive event cycles.",
    )
    result.add_argument("--gradient-accumulation-steps", type=int, default=1)
    result.add_argument("--learning-rate", type=float, default=6e-5)
    result.add_argument("--warmup-tokens", type=int, default=1_000_000)
    result.add_argument("--weight-decay", type=float, default=0.1)
    result.add_argument("--curriculum-stage", type=int, default=1)
    result.add_argument("--training-profile", default="substrate_language_pretraining")
    result.add_argument("--trainer-mode", default="independent_packed_documents")
    result.add_argument("--checkpoint-interval", type=int, default=25)
    result.add_argument("--eval-interval", type=int, default=25)
    result.add_argument("--eval-batches", type=int, default=4)
    result.add_argument(
        "--progress-conditioned-rasl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable delayed CE-learning-progress pressure through the cognitive "
            "resonant adjoint surprise learner. This experimental subsystem is "
            "disabled by default and must be explicitly requested."
        ),
    )
    result.add_argument("--progress-probe-interval", type=int, default=5)
    result.add_argument("--progress-probe-batches", type=int, default=2)
    result.add_argument("--progress-probe-length", type=int, default=4_096)
    result.add_argument("--progress-warmup-observations", type=int, default=8)
    result.add_argument("--progress-fast-window", type=int, default=6)
    result.add_argument("--progress-baseline-min-observations", type=int, default=12)
    result.add_argument("--progress-baseline-window", type=int, default=24)
    result.add_argument("--progress-baseline-lag", type=int, default=6)
    result.add_argument("--progress-baseline-freeze", type=int, default=4)
    result.add_argument("--progress-deadband", type=float, default=0.5)
    result.add_argument("--progress-temperature", type=float, default=1.0)
    result.add_argument("--progress-slope-weight", type=float, default=0.75)
    result.add_argument("--progress-debt-weight", type=float, default=0.25)
    result.add_argument("--progress-guard-tolerance", type=float, default=0.02)
    result.add_argument("--progress-guard-patience", type=int, default=2)
    result.add_argument("--pc-rasl-trajectory-length", type=int, default=256)
    result.add_argument("--pc-rasl-candidates", type=int, default=48)
    result.add_argument("--pc-rasl-replay-batch-size", type=int, default=1)
    result.add_argument("--pc-rasl-max-interval-trajectories", type=int, default=5)
    result.add_argument(
        "--pc-rasl-captures-per-observation",
        type=int,
        default=1,
        help=(
            "Bounded behavior trajectories sampled across each progress-probe "
            "interval (default: one trajectory immediately before consequence)."
        ),
    )
    result.add_argument(
        "--pc-rasl-updates-per-observation",
        type=int,
        default=1,
        help=(
            "Critic/adjoint replay updates authorized by each new measured "
            "learning-progress consequence (default: one)."
        ),
    )
    result.add_argument(
        "--pc-rasl-critic-warmup-observations",
        type=int,
        default=4,
        help=(
            "Additional critic-only progress observations required after the "
            "causal baseline first becomes ready."
        ),
    )
    result.add_argument("--pc-rasl-consequence-weight", type=float, default=0.025)
    result.add_argument("--pc-rasl-critic-learning-rate", type=float, default=6e-5)
    result.add_argument("--pc-rasl-carrier-gradient-cap", type=float, default=0.02)
    result.add_argument("--pc-rasl-cognitive-gradient-cap", type=float, default=0.10)
    result.add_argument("--pc-rasl-controller-gradient-cap", type=float, default=0.15)
    result.add_argument("--device", default="auto")
    result.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    result.add_argument(
        "--cpu-threads", type=int, default=4,
        help="CPU intra-op workers; four is the matched Apple integrated default.",
    )
    result.add_argument(
        "--cpu-interop-threads", type=int, default=1,
        help="CPU inter-op workers; the causal training graph uses one by default.",
    )
    result.add_argument(
        "--compile-tensor-cores", action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Compile pure carrier chunk kernels. The default enables this on "
            "CUDA and leaves CPU/MPS eager."
        ),
    )
    result.add_argument(
        "--apple-mps-loss-offload", action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Offload only exact vocabulary loss to MPS while cognition remains "
            "on CPU. Disabled by default because the matched M1 integrated path "
            "is faster without the cross-device backward boundary."
        ),
    )
    result.add_argument("--seed", type=int, default=20260722)
    result.add_argument("--shuffle-buffer", type=int, default=10_000)
    result.add_argument("--eval-fraction-permyriad", type=int, default=100)
    result.add_argument("--trackio-project", default="mrcra-fineweb")
    result.add_argument(
        "--trackio", action=argparse.BooleanOptionalAction, default=True,
        help="Record the run in the single MRCRA Trackio project (default: enabled).",
    )
    result.add_argument("--run-name")
    result.add_argument("--trackio-space-id")
    result.add_argument("--dashboard", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--spectral-dashboard", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--snapshot-interval", type=int, default=25)
    result.add_argument(
        "--phase-transition-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Log event-threshold distributions, exact hard-event receipts, "
            "rolling approach rate, and subsystem gradients (default: enabled)."
        ),
    )
    result.add_argument(
        "--phase-transition-ablation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "At evaluation checkpoints compare full, soft-only, and "
            "cognition-off arms on identical retained data (default: enabled)."
        ),
    )
    result.add_argument(
        "--phase-ablation-batches", type=int, default=1,
        help="Retained batches used by each phase-transition ablation arm.",
    )
    result.add_argument("--proposal-slope-ema-decay", type=float, default=0.9)
    result.add_argument("--low-clip-coefficient-threshold", type=float, default=0.05)
    result.add_argument("--low-clip-coefficient-patience", type=int, default=10)
    result.add_argument("--pin-revisions", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--resume", nargs="?", const="latest")
    result.add_argument("--smoke-test", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if min(
        args.cpu_threads, args.cpu_interop_threads,
        args.maximum_retained_loss_mib, args.phase_ablation_batches,
        args.low_clip_coefficient_patience,
        args.progress_probe_interval, args.progress_probe_batches,
        args.progress_probe_length, args.progress_warmup_observations,
        args.progress_fast_window, args.progress_baseline_min_observations,
        args.progress_baseline_window, args.progress_baseline_freeze,
        args.progress_guard_patience,
        args.pc_rasl_trajectory_length, args.pc_rasl_candidates,
        args.pc_rasl_replay_batch_size,
        args.pc_rasl_max_interval_trajectories,
        args.pc_rasl_captures_per_observation,
        args.pc_rasl_updates_per_observation,
        args.pc_rasl_critic_warmup_observations,
    ) <= 0:
        raise ValueError(
            "thread, workspace, evaluation, and progress controls must be positive"
        )
    progress_configuration = LearningProgressConfig(
        observation_interval=args.progress_probe_interval,
        warmup_observations=args.progress_warmup_observations,
        fast_window=args.progress_fast_window,
        baseline_min_observations=args.progress_baseline_min_observations,
        baseline_window=args.progress_baseline_window,
        baseline_lag=args.progress_baseline_lag,
        baseline_freeze_observations=args.progress_baseline_freeze,
        deadband_standard_deviations=args.progress_deadband,
        pressure_temperature=args.progress_temperature,
        slope_weight=args.progress_slope_weight,
        debt_weight=args.progress_debt_weight,
        guard_regression_tolerance=args.progress_guard_tolerance,
        guard_regression_patience=args.progress_guard_patience,
        guard_recovery_patience=args.progress_guard_patience,
    )
    torch.set_num_threads(args.cpu_threads)
    if torch.get_num_interop_threads() != args.cpu_interop_threads:
        torch.set_num_interop_threads(args.cpu_interop_threads)
    torch.manual_seed(args.seed)
    selected_profile = production_profile(
        lightmodel=args.lightmodel,
        ultralightmodel=args.ultralightmodel,
        total_tokens=args.total_tokens,
    )
    model_profile = selected_profile.name
    tbptt_length = args.tbptt_length or 4_096
    output_dir = Path(
        args.output_dir or selected_profile.output_directory
    ).resolve()
    if args.resume is None and any(
        (output_dir / name).exists()
        for name in ("metrics.jsonl", "run_manifest.json", "checkpoints")
    ):
        raise FileExistsError(f"{output_dir} already contains a run; use --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke_test:
        if args.ultralightmodel:
            model_profile = selected_profile.name
            tokenizer = GPT2WidthByteSmokeTokenizer()
            smoke_model_config, activation_policy = resolve_activation_checkpointing(
                production_configuration(
                    tokenizer.vocabulary_size,
                    lightmodel=False,
                    ultralightmodel=True,
                ),
                tbptt_length=4,
                device="cpu",
                precision="fp32",
                override=args.activation_checkpointing,
            )
            model = MRCRALanguageModel(
                smoke_model_config,
                model_authority=selected_profile.model_authority,
            )
            model.config.require_actor_parameter_count(model.parameter_count)
            smoke_run_name = "mrcra-1p3m-ultralight-smoke"
            smoke_vocabulary_tile_size = 2_048
        else:
            model_profile = "mrcra_tiny_smoke"
            tokenizer = ByteTextTokenizer()
            smoke_model_config, activation_policy = resolve_activation_checkpointing(
                tiny_configuration(tokenizer.vocabulary_size),
                tbptt_length=4,
                device="cpu",
                precision="fp32",
                override=args.activation_checkpointing,
            )
            model = MRCRALanguageModel(
                smoke_model_config
            )
            smoke_run_name = "mrcra-integrated-default-smoke"
            smoke_vocabulary_tile_size = 32
        source = SequenceTextSource((
            "Events persist through typed relations.",
            "Provenance separates observation from simulation.",
        ))
        evaluation_source = SequenceTextSource((
            "Retained evidence is disjoint from optimization.",
            "Checkpoint identity binds the complete evaluation context.",
        ))
        progress_source = (
            SequenceTextSource((
                "Learning progress is measured without reading phase telemetry.",
                "Delayed consequences are assigned to earlier cognitive operations.",
            ))
            if args.progress_conditioned_rasl else None
        )
        evaluation_batches = build_evaluation_batches(
            PackedTokenStream(evaluation_source, tokenizer),
            count=1, batch_size=1, sequence_length=8,
        )
        progress_probe_batches = (
            build_evaluation_batches(
                PackedTokenStream(progress_source, tokenizer),
                count=1, batch_size=1, sequence_length=8,
            )
            if progress_source is not None else ()
        )
        configuration = MRCRATrainingConfig(
            output_dir=str(output_dir), total_tokens=16, context_length=8,
            execution_chunk_size=2, tbptt_length=4,
            vocabulary_tile_size=smoke_vocabulary_tile_size,
            integrated_cognitive_path=True, cognitive_stride=2,
            cognitive_tbptt_events=2,
            warmup_tokens=8, checkpoint_interval=2, device="cpu", precision="fp32",
            evaluation_interval=1, evaluation_batches=1,
            require_evaluation=True,
            progress_conditioned_rasl=args.progress_conditioned_rasl,
            progress_probe_batches=(
                1 if args.progress_conditioned_rasl else 0
            ),
            progress_probe_length=8,
            pc_rasl_trajectory_length=8,
            pc_rasl_candidate_count=8,
            pc_rasl_max_interval_trajectories=1,
            pc_rasl_captures_per_observation=1,
            pc_rasl_updates_per_observation=1,
            learning_progress=LearningProgressConfig(
                observation_interval=1,
                warmup_observations=4,
                fast_window=3,
                baseline_min_observations=4,
                baseline_window=8,
                baseline_lag=0,
                baseline_freeze_observations=1,
            ),
            trackio_enabled=args.trackio, show_dashboard=args.dashboard,
            spectral_dashboard=args.spectral_dashboard,
            spectral_snapshot_interval=1, spectral_snapshot_tokens=8,
            trackio_project=args.trackio_project,
            run_name=args.run_name or smoke_run_name,
            trackio_space_id=args.trackio_space_id,
            seed=args.seed,
            phase_transition_telemetry=args.phase_transition_telemetry,
            phase_transition_ablation=args.phase_transition_ablation,
            phase_transition_ablation_batches=args.phase_ablation_batches,
            proposal_slope_ema_decay=args.proposal_slope_ema_decay,
            low_clip_coefficient_threshold=args.low_clip_coefficient_threshold,
            low_clip_coefficient_patience=args.low_clip_coefficient_patience,
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
        model_config = production_configuration(
            tokenizer.vocabulary_size,
            lightmodel=args.lightmodel,
            ultralightmodel=args.ultralightmodel,
        )
        model_config, activation_policy = resolve_activation_checkpointing(
            model_config,
            tbptt_length=tbptt_length,
            device=args.device,
            precision=args.precision,
            override=args.activation_checkpointing,
        )
        cognitive_stride = production_cognitive_stride(
            model_config,
            ultralightmodel=args.ultralightmodel,
            override=args.cognitive_stride,
        )
        model = MRCRALanguageModel(
            model_config,
            model_authority=selected_profile.model_authority,
        )
        model.config.require_actor_parameter_count(model.parameter_count)
        source = FineWebTextSource(
            dataset_id=args.dataset_id, dataset_config=args.dataset_config,
            split="train", revision=dataset_revision, partition="train",
            evaluation_fraction_permyriad=args.eval_fraction_permyriad,
            shuffle_seed=args.seed, shuffle_buffer=args.shuffle_buffer,
        )
        evaluation_source = FineWebTextSource(
            dataset_id=args.dataset_id, dataset_config=args.dataset_config,
            split="train", revision=dataset_revision, partition="eval",
            evaluation_fraction_permyriad=args.eval_fraction_permyriad,
            shuffle_seed=args.seed, shuffle_buffer=args.shuffle_buffer,
        )
        progress_source = (
            FineWebTextSource(
                dataset_id=args.dataset_id, dataset_config=args.dataset_config,
                split="train", revision=dataset_revision, partition="progress",
                evaluation_fraction_permyriad=args.eval_fraction_permyriad,
                shuffle_seed=args.seed, shuffle_buffer=args.shuffle_buffer,
            )
            if args.progress_conditioned_rasl else None
        )
        evaluation_batches = build_evaluation_batches(
            PackedTokenStream(evaluation_source, tokenizer),
            count=args.eval_batches, batch_size=1,
            sequence_length=args.context_length,
        )
        progress_probe_batches = (
            build_evaluation_batches(
                PackedTokenStream(progress_source, tokenizer),
                count=args.progress_probe_batches,
                batch_size=1,
                sequence_length=args.progress_probe_length,
            )
            if args.progress_conditioned_rasl else ()
        )
        configuration = MRCRATrainingConfig(
            output_dir=str(output_dir), total_tokens=args.total_tokens,
            context_length=args.context_length,
            execution_chunk_size=args.execution_chunk_size,
            tbptt_length=tbptt_length,
            vocabulary_tile_size=args.vocabulary_tile_size,
            checkpoint_tiles=(
                None if args.loss_memory_policy == "auto"
                else args.loss_memory_policy == "recompute"
            ),
            maximum_retained_loss_bytes=args.maximum_retained_loss_mib << 20,
            integrated_cognitive_path=(
                args.training_profile == "substrate_language_pretraining"
                and args.trainer_mode == "independent_packed_documents"
            ),
            cognitive_stride=cognitive_stride,
            cognitive_tbptt_events=args.cognitive_tbptt_events,
            progress_interval_tokens=args.progress_interval_tokens,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate, warmup_tokens=args.warmup_tokens,
            weight_decay=args.weight_decay, curriculum_stage=args.curriculum_stage,
            training_profile=args.training_profile, trainer_mode=args.trainer_mode,
            checkpoint_interval=args.checkpoint_interval,
            evaluation_interval=args.eval_interval,
            evaluation_batches=args.eval_batches,
            require_evaluation=True,
            progress_conditioned_rasl=args.progress_conditioned_rasl,
            progress_probe_batches=(
                args.progress_probe_batches
                if args.progress_conditioned_rasl else 0
            ),
            progress_probe_length=args.progress_probe_length,
            learning_progress=progress_configuration,
            pc_rasl_trajectory_length=args.pc_rasl_trajectory_length,
            pc_rasl_candidate_count=args.pc_rasl_candidates,
            pc_rasl_replay_batch_size=args.pc_rasl_replay_batch_size,
            pc_rasl_max_interval_trajectories=(
                args.pc_rasl_max_interval_trajectories
            ),
            pc_rasl_captures_per_observation=(
                args.pc_rasl_captures_per_observation
            ),
            pc_rasl_updates_per_observation=(
                args.pc_rasl_updates_per_observation
            ),
            pc_rasl_critic_warmup_observations=(
                args.pc_rasl_critic_warmup_observations
            ),
            pc_rasl_consequence_weight=args.pc_rasl_consequence_weight,
            pc_rasl_critic_learning_rate=args.pc_rasl_critic_learning_rate,
            pc_rasl_carrier_gradient_cap=args.pc_rasl_carrier_gradient_cap,
            pc_rasl_cognitive_gradient_cap=args.pc_rasl_cognitive_gradient_cap,
            pc_rasl_controller_gradient_cap=args.pc_rasl_controller_gradient_cap,
            device=args.device, precision=args.precision, seed=args.seed,
            cpu_threads=args.cpu_threads,
            cpu_interop_threads=args.cpu_interop_threads,
            compile_tensor_cores=args.compile_tensor_cores,
            apple_mps_loss_offload=args.apple_mps_loss_offload,
            trackio_project=args.trackio_project,
            run_name=args.run_name or selected_profile.run_name,
            trackio_space_id=args.trackio_space_id,
            trackio_enabled=args.trackio,
            show_dashboard=args.dashboard, spectral_dashboard=args.spectral_dashboard,
            spectral_snapshot_interval=args.snapshot_interval,
            phase_transition_telemetry=args.phase_transition_telemetry,
            phase_transition_ablation=args.phase_transition_ablation,
            phase_transition_ablation_batches=args.phase_ablation_batches,
            proposal_slope_ema_decay=args.proposal_slope_ema_decay,
            low_clip_coefficient_threshold=args.low_clip_coefficient_threshold,
            low_clip_coefficient_patience=args.low_clip_coefficient_patience,
        )
    trainer = MRCRANextTokenTrainer(
        model, tokenizer, PackedTokenStream(source, tokenizer), configuration,
        evaluation_batches, progress_probe_batches=progress_probe_batches,
    )
    trainer.runtime.update(activation_policy)
    manifest = {
        "model_parameters": model.parameter_count,
        "architecture": "integrated_mrcra",
        "model_profile": model_profile,
        "model_config": asdict(model.config),
        "tokenizer": tokenizer.identity(),
        "training_config": asdict(configuration),
        "training_source": source.state_dict(),
        "evaluation_source": (
            None if evaluation_source is None else evaluation_source.state_dict()
        ),
        "evaluation_identity": trainer.evaluation_identity,
        "progress_probe_source": (
            None if progress_source is None else progress_source.state_dict()
        ),
        "progress_probe_identity": (
            trainer.progress_probe_identity
            if configuration.progress_conditioned_rasl else None
        ),
        "runtime": trainer.runtime,
        "functional_surprise_enabled": configuration.progress_conditioned_rasl,
        "functional_surprise_mode": (
            "progress_conditioned_rasl"
            if configuration.progress_conditioned_rasl else "disabled"
        ),
        "training_profile": configuration.training_profile,
        "trainer_mode": configuration.trainer_mode,
        "claim_boundary": (
            "substrate and English language modeling only; this FineWeb stage does not "
            "establish integrated cognition, agency, transfer, or deployment maturity"
        ),
        "evidence_maturity": "mechanism",
        "functional_surprise_reason": (
            "A disjoint progress-probe CE slope supplies a delayed, bounded "
            "optimization-derived meta-consequence; instantaneous task loss is "
            "never used as reward and a separate held-out guard can veto positive pressure."
            if configuration.progress_conditioned_rasl
            else "Progress-Conditioned RASL was explicitly disabled."
        ),
    }
    manifest_path = output_dir / "run_manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(
        f"MRCRA actor: {model.parameter_count:,} parameters; "
        f"{configuration.total_tokens:,} tokens; {configuration.context_length:,}-token contexts; "
        f"{configuration.total_steps:,} updates.", flush=True,
    )
    print(
        f"Device {trainer.runtime['device']} ({trainer.runtime.get('gpu_name', 'host')}), "
        f"precision {trainer.runtime['precision']}.", flush=True,
    )
    print(
        "Carrier activation policy: "
        f"{activation_policy['carrier_activation_checkpointing_policy']} "
        f"(estimated "
        f"{activation_policy['estimated_uncheckpointed_carrier_activation_bytes'] / (1 << 20):.0f} MiB, "
        f"budget "
        f"{activation_policy['carrier_activation_memory_budget_bytes'] / (1 << 20):.0f} MiB).",
        flush=True,
    )
    if args.resume:
        checkpoint = latest_checkpoint(output_dir) if args.resume == "latest" else Path(args.resume)
        if checkpoint is None:
            raise FileNotFoundError("--resume requested but no latest checkpoint exists")
        trainer.load_checkpoint(checkpoint)
        print(f"Resumed {checkpoint} at update {trainer.state.step}.", flush=True)
    trainer.train()
    if trainer.state.step % configuration.checkpoint_interval:
        trainer.save_checkpoint()
    manifest["final_training_state"] = asdict(trainer.state)
    manifest["completed"] = trainer.state.tokens_seen >= configuration.total_tokens
    write_json_atomic(manifest_path, manifest)


if __name__ == "__main__":
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
        # See the canonical wrapper's documented PyArrow 25/macOS finalizer
        # workaround. All trainer/reporting persistence has completed here.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
