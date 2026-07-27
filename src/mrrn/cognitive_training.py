"""Memory-bounded, stateful 32K training authority for MRCRA language actors.

The trainer deliberately separates three axes that are often conflated:

* an optimization context is the complete 32K token example;
* execution chunks bound live recurrent activations and vocabulary projection;
* TBPTT spans determine how far gradients cross recurrent state.

The model state and immutable provenance ledger continue across execution
chunks.  State is detached only at declared TBPTT boundaries.  Cross-document
targets are excluded and segment boundaries reset cognitive state explicitly.
"""

from __future__ import annotations

from contextlib import nullcontext
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from importlib.util import find_spec
import json
from math import ceil, isfinite, lcm, log
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .activation_execution import (
    ActivationCandidateMeasurement,
    ActivationExecutionPolicy,
    ActivationPartitionCensus,
    calibrate_activation_candidates,
    census_activation_partitions,
    fastest_safe_activation_policy,
    measure_saved_tensor_bytes,
    maximum_safe_activation_physical_tokens,
    observe_memory,
    resolve_activation_execution_policy,
    select_activation_dominant_partitions,
)
from .cognitive_checkpoint import runtime_state_dict, runtime_state_from_dict
from .carrier_execution import resolve_carrier_execution_policy
from .cognitive_diagnostics import cognitive_metrics
from .cognitive_model import HardEventTrace, MRCRAOutput, MRCRARuntimeState
from .cstm import (
    CSTMLoss,
    build_causal_spectral_targets,
    causal_spectral_target_mask,
)
from .cstm_schedule import (
    CSTMCoverageState,
    CSTMObligation,
    CSTMSamplingDecision,
    deterministic_cstm_sample,
)
from .cognitive_supervision import EvidenceBackedCognitiveSupervisor
from .cognitive_surprise import (
    CognitiveRASLConfig, CognitiveResonantAdjointSurpriseLearner,
    CognitiveTrajectoryBatch, build_language_candidate_set,
)
from .cognitive_types import CognitiveClocks, InternalAction
from .document_batching import DocumentBatchPlan, DocumentMajorBatchPlanner
from .document_cost_model import (
    DocumentExecutionCostModel,
    measured_document_cost_model,
)
from .cognitive_objectives import (
    CognitiveObjectiveSchedule, ObjectiveFamily, ObjectiveTerm,
    combine_cognitive_objectives,
)
from .language import MRCRALanguageModel, MRCRALanguageOutput
from .learning_progress import (
    LearningProgressAuthority, LearningProgressConfig, LearningProgressReport,
)
from .lm_training import (
    PackedBatch, PackedTokenStream, TextTokenizer, TrackioReporter,
    materialize_tokenizer_artifacts,
    _configure_cuda, _device_for, _memory_metrics, _precision_for,
    _runtime_details, _synchronize,
)
from .mixer import ResonantSpectralGLU
from .objectives import spectral_activation_regularization
from .optimization import (
    GradientReport, OptimizerPolicy, build_adamw, build_scheduler,
    clip_and_report_gradients, gradient_subsystem, merge_auxiliary_gradients,
)
from .provenance import ProvenanceLedger
from .training_profiles import TrainerMode, get_training_profile
from .surprise import ResonantAdjointSurpriseConfig


# Version 10 binds replay to compact pre-consequence cognitive behavior
# evidence, version 11 adds consequence-driven replay scheduling, and version
# 12 removes PC-RASL from the default production authority, version 13
# adds CSTM, version 14 binds the exact vocabulary-loss execution policy, and
# version 15 binds deterministic document-major static execution. Version 16
# separates semantic/optimization identity from interchangeable execution and
# observation policies, and records activation-policy provenance.
# Versions 3--9
# migrate conservatively and restart PC-RASL causal state when exact behavior
# evidence did not exist.
MRCRA_TRAINING_FORMAT_VERSION = 16
LEGACY_MRCRA_TRAINING_FORMAT_VERSIONS = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
}
PRE_BEHAVIOR_EVIDENCE_FORMAT_VERSIONS = {3, 4, 5, 6, 7, 8, 9}

_EXECUTION_TRAINING_FIELDS = frozenset({
    "execution_chunk_size",
    "tbptt_length",
    "vocabulary_tile_size",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "document_static_batching",
    "document_bucket_lengths",
    "document_batch_token_budget",
    "document_grouping_policy",
    "document_plan_cache_capacity",
    "document_cost_calibration",
    "checkpoint_tiles",
    "maximum_retained_loss_bytes",
    "maximum_fused_loss_bytes",
    "exact_loss_backend",
    "device",
    "precision",
    "cpu_threads",
    "cpu_interop_threads",
    "data_prefetch",
    "compile_tensor_cores",
    "performance_calibration",
    "activation_policy",
    "activation_memory_reserve_bytes",
    "activation_calibration",
    "allow_unsafe_activation_policy",
    "apple_mps_loss_offload",
})
_OBSERVATION_TRAINING_FIELDS = frozenset({
    "log_interval",
    "checkpoint_interval",
    "keep_checkpoints",
    "evaluation_interval",
    "evaluation_batches",
    "require_evaluation",
    "progress_interval_tokens",
    "trackio_project",
    "run_name",
    "trackio_space_id",
    "trackio_remote_log_interval",
    "spectral_dashboard",
    "spectral_baseline_metrics",
    "spectral_snapshot_interval",
    "spectral_snapshot_tokens",
    "spectral_dashboard_prompt",
    "phase_transition_telemetry",
    "phase_transition_ablation",
    "phase_transition_ablation_batches",
    "proposal_slope_ema_decay",
    "low_clip_coefficient_threshold",
    "low_clip_coefficient_patience",
})

_V4_COGNITIVE_DEFAULT_FIELDS = (
    "reconstruction_capacity", "action_candidate_capacity", "action_argument_dim",
    "evidence_request_capacity", "external_artifact_capacity",
    "external_artifact_digest_width", "viability_channels", "metacognitive_capacity",
    "deliberation_prediction_error_threshold", "planning_hypothesis_top_k",
    "minimum_routed_posterior_mass",
    "enable_conditional_reconstruction", "enable_abstraction_validity_control",
    "enable_post_deliberation_action_selection", "enable_multi_hypothesis_planning",
    "enable_agent_session_loop", "enable_viability_gate",
    "enable_integrated_invariant_discovery", "enable_persistent_session_training",
    "enable_metacognitive_routing",
)


@dataclass(frozen=True, slots=True)
class TiledCrossEntropy:
    """Exact likelihood statistics without a dense time-by-vocabulary tensor."""

    loss: Tensor
    nll_sum: Tensor
    token_count: int
    byte_count: int

    @property
    def nats_per_byte(self) -> Tensor:
        return self.nll_sum / max(1, self.byte_count)


def exact_tiled_cross_entropy(
    output_latent: Tensor,
    labels: Tensor,
    target_byte_lengths: Tensor,
    mask: Tensor,
    output_weight: Tensor,
    output_bias: Tensor | None,
    *,
    vocabulary_tile_size: int,
    checkpoint_tiles: bool = True,
) -> TiledCrossEntropy:
    """Compute full-softmax CE exactly by reducing vocabulary tiles.

    This is not sampled or hierarchical softmax.  Every vocabulary item enters
    the log partition.  Checkpointed tiles retain only their log-sum-exp and are
    recomputed during backward, keeping memory proportional to
    ``valid_positions + valid_positions * vocabulary_tile_size``.
    """

    if output_latent.ndim != 3 or labels.shape != output_latent.shape[:2]:
        raise ValueError("output latents and labels must have batch/time shape")
    if labels.dtype != torch.int64 or target_byte_lengths.shape != labels.shape:
        raise ValueError("labels and byte lengths must be aligned int64 tensors")
    if target_byte_lengths.dtype != torch.int64 or bool((target_byte_lengths < 0).any()):
        raise ValueError("target byte lengths must be nonnegative")
    if mask.shape != labels.shape or mask.dtype != torch.bool or not bool(mask.any()):
        raise ValueError("tiled cross entropy requires a nonempty boolean mask")
    if output_weight.ndim != 2 or output_weight.shape[1] != output_latent.shape[-1]:
        raise ValueError("output weight is incompatible with output latents")
    if output_bias is not None and output_bias.shape != (output_weight.shape[0],):
        raise ValueError("output bias is incompatible with output weight")
    if vocabulary_tile_size <= 0:
        raise ValueError("vocabulary tile size must be positive")
    selected_labels = labels[mask]
    if int(selected_labels.min()) < 0 or int(selected_labels.max()) >= output_weight.shape[0]:
        raise ValueError("labels lie outside the output vocabulary")
    hidden = output_latent[mask]
    accumulation_dtype = (
        torch.float64 if hidden.dtype == torch.float64 else torch.float32
    )
    target_weight = F.embedding(selected_labels, output_weight)
    target = (
        hidden.to(accumulation_dtype) * target_weight.to(accumulation_dtype)
    ).sum(-1)
    if output_bias is not None:
        target = target + output_bias[selected_labels].to(accumulation_dtype)
    log_partition = hidden.new_full(
        (hidden.shape[0],), -torch.inf, dtype=accumulation_dtype
    )

    def tile_partition(value: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        return F.linear(value, weight, bias).to(accumulation_dtype).logsumexp(-1)

    for start in range(0, output_weight.shape[0], vocabulary_tile_size):
        end = min(output_weight.shape[0], start + vocabulary_tile_size)
        weight = output_weight[start:end]
        bias = None if output_bias is None else output_bias[start:end]
        if checkpoint_tiles and torch.is_grad_enabled() and hidden.requires_grad:
            if bias is None:
                local = checkpoint(
                    lambda value, tile_weight: tile_partition(value, tile_weight, None),
                    hidden, weight, use_reentrant=False,
                )
            else:
                local = checkpoint(
                    tile_partition, hidden, weight, bias, use_reentrant=False,
                )
        else:
            local = tile_partition(hidden, weight, bias)
        log_partition = torch.logaddexp(log_partition, local)
    nll = log_partition - target
    if not bool(torch.isfinite(nll).all()):
        raise FloatingPointError("tiled vocabulary projection produced non-finite NLL")
    return TiledCrossEntropy(
        nll.mean(), nll.sum(), int(mask.sum()), int(target_byte_lengths[mask].sum())
    )


def exact_fused_cross_entropy(
    output_latent: Tensor,
    labels: Tensor,
    target_byte_lengths: Tensor,
    mask: Tensor,
    output_weight: Tensor,
    output_bias: Tensor | None,
) -> TiledCrossEntropy:
    """Compute the same full-softmax objective as one accelerator contraction."""

    if output_latent.ndim != 3 or labels.shape != output_latent.shape[:2]:
        raise ValueError("output latents and labels must have batch/time shape")
    if labels.dtype != torch.int64 or target_byte_lengths.shape != labels.shape:
        raise ValueError("labels and byte lengths must be aligned int64 tensors")
    if target_byte_lengths.dtype != torch.int64 or bool((target_byte_lengths < 0).any()):
        raise ValueError("target byte lengths must be nonnegative")
    if mask.shape != labels.shape or mask.dtype != torch.bool or not bool(mask.any()):
        raise ValueError("fused cross entropy requires a nonempty boolean mask")
    if output_weight.ndim != 2 or output_weight.shape[1] != output_latent.shape[-1]:
        raise ValueError("output weight is incompatible with output latents")
    if output_bias is not None and output_bias.shape != (output_weight.shape[0],):
        raise ValueError("output bias is incompatible with output weight")
    selected_labels = labels[mask]
    if int(selected_labels.min()) < 0 or int(selected_labels.max()) >= output_weight.shape[0]:
        raise ValueError("labels lie outside the output vocabulary")
    logits = F.linear(output_latent[mask], output_weight, output_bias)
    nll = F.cross_entropy(logits, selected_labels, reduction="none")
    if not bool(torch.isfinite(nll).all()):
        raise FloatingPointError("fused vocabulary projection produced non-finite NLL")
    return TiledCrossEntropy(
        nll.mean(), nll.sum(), int(mask.sum()), int(target_byte_lengths[mask].sum())
    )


def exact_cut_cross_entropy(
    output_latent: Tensor,
    labels: Tensor,
    target_byte_lengths: Tensor,
    mask: Tensor,
    output_weight: Tensor,
    output_bias: Tensor | None,
    *,
    implementation: str,
) -> TiledCrossEntropy:
    """Execute exact full-vocabulary likelihood through Apple's CCE API.

    ``cce_kahan_full_c`` is the pretraining policy: it uses stable partition
    accumulation and never filters classifier-row gradients. ``cce_exact`` is
    the no-filtering reference used for audits. The objective and NLL returned
    by every accepted implementation are full-vocabulary quantities.
    """

    if implementation not in {
        "cce_kahan_full_c",
        "cce_exact",
        "torch_compile",
    }:
        raise ValueError("unsafe or unsupported Cut Cross-Entropy implementation")
    if output_latent.ndim != 3 or labels.shape != output_latent.shape[:2]:
        raise ValueError("output latents and labels must have batch/time shape")
    if labels.dtype != torch.int64 or target_byte_lengths.shape != labels.shape:
        raise ValueError("labels and byte lengths must be aligned int64 tensors")
    if target_byte_lengths.dtype != torch.int64 or bool((target_byte_lengths < 0).any()):
        raise ValueError("target byte lengths must be nonnegative")
    if mask.shape != labels.shape or mask.dtype != torch.bool or not bool(mask.any()):
        raise ValueError("Cut Cross-Entropy requires a nonempty boolean mask")
    if output_weight.ndim != 2 or output_weight.shape[1] != output_latent.shape[-1]:
        raise ValueError("output weight is incompatible with output latents")
    if output_bias is not None and output_bias.shape != (output_weight.shape[0],):
        raise ValueError("output bias is incompatible with output weight")
    selected_labels = labels[mask]
    if int(selected_labels.min()) < 0 or int(selected_labels.max()) >= output_weight.shape[0]:
        raise ValueError("labels lie outside the output vocabulary")
    try:
        from cut_cross_entropy import linear_cross_entropy
    except ImportError as error:
        raise RuntimeError(
            "Cut Cross-Entropy was requested but is not installed; "
            "install the mrrn[cce] optional dependency"
        ) from error
    nll = linear_cross_entropy(
        output_latent[mask],
        output_weight,
        selected_labels,
        bias=output_bias,
        reduction="none",
        impl=implementation,
    )
    if nll.shape != selected_labels.shape:
        raise RuntimeError("Cut Cross-Entropy returned an unexpected NLL shape")
    nll = nll.float()
    if not bool(torch.isfinite(nll).all()):
        raise FloatingPointError("Cut Cross-Entropy produced non-finite NLL")
    return TiledCrossEntropy(
        nll.mean(), nll.sum(), int(mask.sum()), int(target_byte_lengths[mask].sum())
    )


def _recoverable_external_cce_failure(error: Exception) -> bool:
    """Identify delayed compiler failures from the optional CCE execution path.

    Cut Cross-Entropy's portable backend compiles lazily on its first real
    shape.  Installation and workspace checks can therefore succeed even when
    Inductor, clang, a precompiled header, or a generated extension later
    fails.  Input-contract and numerical failures are deliberately excluded:
    only an external backend/compiler failure may trigger an exact tiled retry.
    """

    current: BaseException | None = error
    while current is not None:
        kind = type(current)
        module = kind.__module__
        name = kind.__name__
        message = str(current).lower()
        if (
            module.startswith(("torch._inductor", "torch._dynamo"))
            or name in {
                "BackendCompilerFailed",
                "CppCompileError",
                "InductorError",
                "InvalidCxxCompiler",
            }
            or any(marker in message for marker in (
                "c++ compile error",
                "cppcompileerror",
                "inductorerror",
                "torchinductor",
                "precompiled header",
                "backend compiler failed",
            ))
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _diagnostic_snapshot_due(
    step: int, interval: int, last_attempt_step: int = -1,
) -> bool:
    """Attempt once per process immediately, then use the bounded cadence."""

    if step <= 0 or interval <= 0 or last_attempt_step < -1:
        raise ValueError("diagnostic snapshot step and interval must be positive")
    if last_attempt_step >= step:
        return False
    return last_attempt_step < 0 or step == 1 or step % interval == 0


def event_phase_metrics(
    proposal_logits: Tensor,
    end_logits: Tensor,
    *,
    proposal_threshold_logit: float = 0.0,
) -> dict[str, float]:
    """Reduce exact proposal heads into transition-boundary telemetry."""

    proposal = proposal_logits.detach().float().reshape(-1)
    ending = end_logits.detach().float().reshape(-1)
    if not proposal.numel() or proposal.shape != ending.shape:
        raise ValueError("event phase telemetry requires aligned nonempty logits")
    if not bool(torch.isfinite(proposal).all() & torch.isfinite(ending).all()):
        raise FloatingPointError("event phase logits became non-finite")
    proposal_probability = torch.sigmoid(proposal)
    end_probability = torch.sigmoid(ending)

    def quantiles(value: Tensor, prefix: str) -> dict[str, float]:
        levels = value.new_tensor((0.5, 0.9, 0.99))
        q50, q90, q99 = torch.quantile(value, levels)
        return {
            f"architecture/{prefix}_mean": float(value.mean().cpu()),
            f"architecture/{prefix}_median": float(q50.cpu()),
            f"architecture/{prefix}_p90": float(q90.cpu()),
            f"architecture/{prefix}_p99": float(q99.cpu()),
            f"architecture/{prefix}_max": float(value.max().cpu()),
        }

    result = quantiles(
        proposal_probability, "event_proposal_probability"
    )
    result.update(quantiles(end_probability, "event_end_probability"))
    maximum_logit = proposal.max()
    result.update({
        "architecture/event_proposal_logit_max": float(maximum_logit.cpu()),
        "architecture/event_phase_distance_to_threshold": float(
            proposal_threshold_logit - maximum_logit.cpu()
        ),
    })
    for probability, label in (
        (0.25, "0p25"), (0.35, "0p35"), (0.45, "0p45"), (0.50, "0p50"),
    ):
        result[
            f"architecture/event_proposal_fraction_ge_{label}"
        ] = float((proposal_probability >= probability).float().mean().cpu())
    return result


def _concatenate_event_phase_logits(
    proposal_rows: Iterable[Tensor],
    end_rows: Iterable[Tensor],
) -> tuple[Tensor, Tensor]:
    """Flatten and concatenate phase logits emitted by variable-length spans.

    Integrated packed contexts preserve document boundaries, so successive
    cognitive forwards generally have different temporal extents.  Their
    logits are samples from one telemetry population rather than rectangular
    sequence features; flattening each aligned pair before concatenation keeps
    every sample without padding, truncation, or document-length assumptions.
    """

    proposals = tuple(proposal_rows)
    endings = tuple(end_rows)
    if not proposals or len(proposals) != len(endings):
        raise ValueError("event phase telemetry requires paired nonempty rows")
    flattened_proposals: list[Tensor] = []
    flattened_endings: list[Tensor] = []
    for proposal, ending in zip(proposals, endings, strict=True):
        proposal_flat = proposal.reshape(-1)
        ending_flat = ending.reshape(-1)
        if not proposal_flat.numel() or proposal_flat.shape != ending_flat.shape:
            raise ValueError("event phase telemetry rows must be aligned and nonempty")
        flattened_proposals.append(proposal_flat)
        flattened_endings.append(ending_flat)
    return torch.cat(flattened_proposals), torch.cat(flattened_endings)


class CognitiveSupervisionProvider(Protocol):
    """Supplies only evidence-backed auxiliary targets for a training chunk."""

    def __call__(
        self, output: MRCRALanguageOutput, batch: PackedBatch, start: int, end: int,
    ) -> Iterable[ObjectiveTerm]: ...


@dataclass(frozen=True, slots=True)
class MRCRATrainingConfig:
    """Production defaults for a 120M-class actor on a 20 GiB accelerator."""

    output_dir: str = "outputs/mrcra-120m-fineweb-20m"
    total_tokens: int = 20_000_000
    context_length: int = 32_768
    execution_chunk_size: int = 256
    tbptt_length: int = 4_096
    # 4K is the measured cross-platform knee for GPT-2-scale vocabularies:
    # it halves launch/reduction overhead versus 2K without the less stable
    # workspace growth of 8K/16K tiles on unified-memory Apple systems.
    vocabulary_tile_size: int = 4_096
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 6e-5
    weight_decay: float = 0.1
    warmup_tokens: int = 1_000_000
    minimum_learning_rate_ratio: float = 0.1
    maximum_gradient_norm: float = 1.0
    spectral_regularization_weight: float = 1e-4
    state_regularization_weight: float = 1e-5
    event_compute_regularization_weight: float = 1e-5
    cstm_enabled: bool = True
    cstm_weight: float = 0.04
    cstm_warmup_tokens: int = 100_000
    cstm_ramp_tokens: int = 400_000
    cstm_carrier_gradient_cap: float = 0.10
    cstm_cognitive_gradient_cap: float = 0.05
    cstm_head_gradient_cap: float = 0.10
    cstm_execution: str = "sampled"
    cstm_sampling_duty_cycle: float = 0.25
    cstm_sampling_uniform_mixture: float = 0.05
    cstm_max_substrate_vjps: int = 1
    cstm_target_participation_budget: int = 8_192
    cstm_predictor_update_interval: int = 1
    cstm_maximum_coverage_gap: int = 4_096
    allow_cstm_execution_upgrade: bool = False
    state_target_rms: float = 8.0
    curriculum_stage: int = 1
    training_profile: str = "substrate_language_pretraining"
    trainer_mode: str = TrainerMode.INDEPENDENT_PACKED_DOCUMENTS.value
    required_auxiliary_families: tuple[int, ...] = ()
    integrated_cognitive_path: bool = False
    document_static_batching: bool = True
    document_bucket_lengths: tuple[int, ...] = tuple(
        range(64, 4_096 + 1, 64)
    )
    document_batch_token_budget: int = 8_192
    document_grouping_policy: str = "cost_aware"
    document_plan_cache_capacity: int = 128
    document_cost_calibration: bool = True
    cognitive_stride: int = 128
    cognitive_tbptt_events: int = 4
    progress_interval_tokens: int = 2_048
    checkpoint_tiles: bool | None = None
    maximum_retained_loss_bytes: int = 1 << 30
    maximum_fused_loss_bytes: int = 256 << 20
    exact_loss_backend: str = "auto"
    mlx_memory_limit_bytes: int = 1_536 << 20
    mlx_cache_limit_bytes: int = 128 << 20
    log_interval: int = 1
    checkpoint_interval: int = 25
    keep_checkpoints: int = 3
    evaluation_interval: int = 0
    evaluation_batches: int = 0
    require_evaluation: bool = False
    progress_conditioned_rasl: bool = False
    progress_probe_batches: int = 0
    progress_probe_length: int = 4_096
    learning_progress: LearningProgressConfig = LearningProgressConfig()
    pc_rasl_trajectory_length: int = 256
    pc_rasl_candidate_count: int = 48
    pc_rasl_replay_batch_size: int = 1
    pc_rasl_max_interval_trajectories: int = 5
    pc_rasl_captures_per_observation: int = 1
    pc_rasl_updates_per_observation: int = 1
    pc_rasl_critic_warmup_observations: int = 4
    pc_rasl_consequence_weight: float = 0.025
    pc_rasl_critic_learning_rate: float = 6e-5
    pc_rasl_carrier_gradient_cap: float = 0.02
    pc_rasl_cognitive_gradient_cap: float = 0.10
    pc_rasl_controller_gradient_cap: float = 0.15
    seed: int = 20260722
    device: str = "auto"
    precision: str = "auto"
    cpu_threads: int = 4
    cpu_interop_threads: int = 1
    data_prefetch: bool = True
    compile_tensor_cores: bool | None = None
    performance_calibration: bool = True
    activation_policy: str = "auto"
    activation_memory_reserve_bytes: int = 4 << 30
    activation_calibration: bool = True
    activation_calibration_cache_directory: str | None = None
    activation_calibration_reexec: bool = False
    allow_unsafe_activation_policy: bool = False
    apple_mps_loss_offload: bool = False
    trackio_enabled: bool = True
    trackio_project: str = "mrcra-fineweb"
    run_name: str = "mrcra-120m-fineweb-20m-32k"
    trackio_space_id: str | None = None
    trackio_remote_log_interval: int = 4
    # Do not colocate a polling/rendering web server with the training process
    # unless the caller explicitly requests it.
    show_dashboard: bool = False
    spectral_dashboard: bool = True
    spectral_baseline_metrics: str | None = None
    spectral_snapshot_interval: int = 25
    spectral_snapshot_tokens: int = 32
    spectral_dashboard_prompt: str = (
        "Relational continuity binds events across multiple temporal scales."
    )
    phase_transition_telemetry: bool = True
    phase_transition_ablation: bool = True
    phase_transition_ablation_batches: int = 1
    proposal_slope_ema_decay: float = 0.9
    low_clip_coefficient_threshold: float = 0.05
    low_clip_coefficient_patience: int = 10

    def __post_init__(self) -> None:
        positive = (
            self.total_tokens, self.context_length, self.execution_chunk_size,
            self.tbptt_length, self.vocabulary_tile_size, self.micro_batch_size,
            self.gradient_accumulation_steps, self.warmup_tokens,
            self.maximum_gradient_norm, self.state_target_rms,
            self.log_interval, self.checkpoint_interval, self.keep_checkpoints,
            self.spectral_snapshot_interval, self.spectral_snapshot_tokens,
            self.progress_interval_tokens, self.cognitive_stride,
            self.cognitive_tbptt_events,
            self.document_batch_token_budget,
            self.activation_memory_reserve_bytes,
            self.cpu_interop_threads,
            self.phase_transition_ablation_batches,
            self.low_clip_coefficient_patience,
            self.progress_probe_length,
            self.pc_rasl_trajectory_length,
            self.pc_rasl_candidate_count,
            self.pc_rasl_replay_batch_size,
            self.pc_rasl_max_interval_trajectories,
            self.pc_rasl_captures_per_observation,
            self.pc_rasl_updates_per_observation,
            self.pc_rasl_critic_warmup_observations,
            self.cstm_maximum_coverage_gap,
            self.cstm_max_substrate_vjps,
            self.cstm_target_participation_budget,
            self.cstm_predictor_update_interval,
            self.trackio_remote_log_interval,
            self.mlx_memory_limit_bytes,
        )
        if min(positive) <= 0:
            raise ValueError("MRCRA training sizes and intervals must be positive")
        if self.cpu_threads < 0:
            raise ValueError("CPU threads must be zero for auto or positive")
        if self.execution_chunk_size > self.tbptt_length or self.tbptt_length > self.context_length:
            raise ValueError("execution_chunk_size <= tbptt_length <= context_length is required")
        if self.tbptt_length % self.execution_chunk_size:
            raise ValueError("TBPTT length must be an integer number of execution chunks")
        if self.context_length % self.execution_chunk_size:
            raise ValueError("context length must be an integer number of execution chunks")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning rate and weight decay are invalid")
        if not 0 <= self.minimum_learning_rate_ratio <= 1:
            raise ValueError("minimum learning-rate ratio must lie in [0,1]")
        if (
            self.spectral_regularization_weight < 0
            or self.state_regularization_weight < 0
            or self.event_compute_regularization_weight < 0
        ):
            raise ValueError("regularization weights cannot be negative")
        if not isinstance(self.cstm_enabled, bool):
            raise ValueError("CSTM enablement must be boolean")
        if (
            self.cstm_weight < 0
            or self.cstm_warmup_tokens < 0
            or self.cstm_ramp_tokens <= 0
            or min(
                self.cstm_carrier_gradient_cap,
                self.cstm_cognitive_gradient_cap,
                self.cstm_head_gradient_cap,
            ) < 0
        ):
            raise ValueError("CSTM schedule, weight, and gradient caps are invalid")
        if self.cstm_enabled and self.cstm_weight <= 0:
            raise ValueError("enabled CSTM requires a positive objective weight")
        if self.cstm_execution not in {"sampled", "legacy_dense"}:
            raise ValueError("unknown CSTM execution policy")
        if not 0 < self.cstm_sampling_duty_cycle <= 1:
            raise ValueError("CSTM sampling duty cycle must lie in (0,1]")
        if not 0 <= self.cstm_sampling_uniform_mixture < 1:
            raise ValueError("CSTM sampling uniform mixture must lie in [0,1)")
        if self.cstm_max_substrate_vjps != 1:
            raise ValueError(
                "the production CSTM authority permits exactly one maximum "
                "substrate VJP per optimizer context"
            )
        if not isinstance(self.allow_cstm_execution_upgrade, bool):
            raise ValueError("CSTM execution upgrade authority must be boolean")
        if self.maximum_fused_loss_bytes < 0 or self.maximum_retained_loss_bytes < 0:
            raise ValueError("loss workspace limits cannot be negative")
        if (
            self.mlx_cache_limit_bytes < 0
            or self.mlx_cache_limit_bytes > self.mlx_memory_limit_bytes
        ):
            raise ValueError("MLX cache limit must lie within its memory limit")
        if self.exact_loss_backend not in {
            "auto",
            "cce_kahan_full_c",
            "cce_exact",
            "torch_compile",
            "mlx",
            "fused",
            "tiled",
        }:
            raise ValueError("unknown exact full-vocabulary loss backend")
        if self.exact_loss_backend == "fused" and self.maximum_fused_loss_bytes <= 0:
            raise ValueError("fused exact loss requires a positive workspace budget")
        if self.checkpoint_tiles is not None and not isinstance(self.checkpoint_tiles, bool):
            raise ValueError("checkpoint_tiles must be boolean or None for automatic selection")
        if (
            self.pc_rasl_captures_per_observation
            > min(
                self.pc_rasl_max_interval_trajectories,
                self.learning_progress.observation_interval,
            )
        ):
            raise ValueError(
                "PC-RASL captures per observation cannot exceed the progress "
                "interval or trajectory bound"
            )
        if self.total_tokens < self.micro_batch_size * self.context_length:
            raise ValueError("token budget must contain at least one full context")
        if self.evaluation_interval < 0 or self.evaluation_batches < 0:
            raise ValueError("evaluation controls cannot be negative")
        if bool(self.evaluation_interval) != bool(self.evaluation_batches):
            raise ValueError("evaluation interval and batch count must be enabled together")
        if self.require_evaluation and not self.evaluation_batches:
            raise ValueError("this training run requires retained evaluation batches")
        if not isinstance(self.progress_conditioned_rasl, bool):
            raise ValueError("progress_conditioned_rasl must be boolean")
        if self.progress_probe_batches < 0:
            raise ValueError("progress probe batch count cannot be negative")
        if self.progress_conditioned_rasl and not self.progress_probe_batches:
            raise ValueError(
                "Progress-Conditioned RASL requires disjoint progress-probe batches"
            )
        if not self.progress_conditioned_rasl and self.progress_probe_batches:
            raise ValueError(
                "progress-probe batches require Progress-Conditioned RASL"
            )
        if (
            self.progress_conditioned_rasl
            and self.progress_probe_length > self.context_length
        ):
            raise ValueError("progress probe length cannot exceed training context")
        if not 2 <= self.pc_rasl_candidate_count <= 64:
            raise ValueError("PC-RASL candidate count must lie in 2..64")
        if (
            self.progress_conditioned_rasl
            and self.pc_rasl_trajectory_length > self.context_length
        ):
            raise ValueError("PC-RASL trajectory length cannot exceed training context")
        if min(
            self.pc_rasl_consequence_weight,
            self.pc_rasl_critic_learning_rate,
        ) <= 0:
            raise ValueError("PC-RASL consequence and critic rates must be positive")
        if min(
            self.pc_rasl_carrier_gradient_cap,
            self.pc_rasl_cognitive_gradient_cap,
            self.pc_rasl_controller_gradient_cap,
        ) < 0:
            raise ValueError("PC-RASL subsystem gradient caps cannot be negative")
        if (
            self.evaluation_batches
            and self.phase_transition_ablation_batches > self.evaluation_batches
        ):
            raise ValueError("phase-transition ablation batches exceed retained evaluation")
        if not 0 <= self.proposal_slope_ema_decay < 1:
            raise ValueError("proposal slope EMA decay must lie in [0,1)")
        if not 0 < self.low_clip_coefficient_threshold <= 1:
            raise ValueError("low clip coefficient threshold must lie in (0,1]")
        CognitiveObjectiveSchedule.curriculum(self.curriculum_stage)
        profile = get_training_profile(self.training_profile)
        mode = TrainerMode(self.trainer_mode)
        if not isinstance(self.integrated_cognitive_path, bool):
            raise ValueError("integrated_cognitive_path must be boolean")
        if not isinstance(self.document_static_batching, bool):
            raise ValueError("document_static_batching must be boolean")
        if (
            self.integrated_cognitive_path
            and self.cstm_enabled
            and self.cstm_execution == "sampled"
            and not self.document_static_batching
        ):
            raise ValueError(
                "sampled CSTM requires document_static_batching so its "
                "physical-invocation sampling authority is explicit; use "
                "cstm_execution='legacy_dense' for the serial reference path"
            )
        if self.document_grouping_policy not in {
            "cost_aware", "exact_signature",
        }:
            raise ValueError("unknown document grouping policy")
        if self.document_plan_cache_capacity < 0:
            raise ValueError("document plan cache capacity cannot be negative")
        if not isinstance(self.document_cost_calibration, bool):
            raise ValueError("document cost calibration must be boolean")
        if (
            not self.document_bucket_lengths
            or tuple(sorted(set(self.document_bucket_lengths)))
            != self.document_bucket_lengths
            or min(self.document_bucket_lengths) <= 0
        ):
            raise ValueError(
                "document bucket lengths must be unique increasing positive values"
            )
        if self.tbptt_length > self.document_bucket_lengths[-1]:
            raise ValueError(
                "largest document bucket must cover the configured TBPTT span"
            )
        if not isinstance(self.apple_mps_loss_offload, bool):
            raise ValueError("apple_mps_loss_offload must be boolean")
        if not isinstance(self.allow_unsafe_activation_policy, bool):
            raise ValueError("unsafe activation policy override must be boolean")
        if not isinstance(self.data_prefetch, bool):
            raise ValueError("data_prefetch must be boolean")
        if (
            not isinstance(self.phase_transition_telemetry, bool)
            or not isinstance(self.phase_transition_ablation, bool)
        ):
            raise ValueError("phase-transition controls must be boolean")
        if (
            self.compile_tensor_cores is not None
            and not isinstance(self.compile_tensor_cores, bool)
        ):
            raise ValueError("compile_tensor_cores must be boolean or None")
        if not isinstance(self.performance_calibration, bool):
            raise ValueError("performance_calibration must be boolean")
        if self.activation_policy not in {
            "auto", "retain", "selective", "whole_span",
        }:
            raise ValueError("unknown activation execution policy")
        if not isinstance(self.activation_calibration, bool):
            raise ValueError("activation_calibration must be boolean")
        if (
            self.activation_calibration_cache_directory is not None
            and (
                not isinstance(
                    self.activation_calibration_cache_directory, str
                )
                or not self.activation_calibration_cache_directory
            )
        ):
            raise ValueError(
                "activation calibration cache directory must be a path or None"
            )
        if not isinstance(self.activation_calibration_reexec, bool):
            raise ValueError("activation calibration re-exec must be boolean")
        if (
            self.activation_calibration_reexec
            and self.activation_calibration_cache_directory is None
        ):
            raise ValueError(
                "activation calibration re-exec requires an enabled cache"
            )
        if self.integrated_cognitive_path and (
            profile.name != "substrate_language_pretraining"
            or mode != TrainerMode.INDEPENDENT_PACKED_DOCUMENTS
            or self.required_auxiliary_families
        ):
            raise ValueError(
                "the integrated multirate path is restricted to independent stage-1 language pretraining"
            )
        if self.integrated_cognitive_path and self.cognitive_stride > self.tbptt_length:
            raise ValueError("cognitive stride cannot exceed the TBPTT span")
        if self.progress_conditioned_rasl and not self.integrated_cognitive_path:
            raise ValueError(
                "Progress-Conditioned RASL requires the integrated cognitive path"
            )
        if self.curriculum_stage != profile.curriculum_stage:
            raise ValueError(
                "training curriculum stage must match the named training profile"
            )
        if (
            mode.permits_persistence
            and mode != TrainerMode.CONTINUOUS_WITHIN_DOCUMENT
            and not profile.permits_persistence
        ):
            raise ValueError("training profile does not authorize persistent trainer mode")
        families = tuple(ObjectiveFamily(value) for value in self.required_auxiliary_families)
        if len(set(families)) != len(families) or any(
            family in (ObjectiveFamily.PRIMARY_TASK, ObjectiveFamily.SPECTRAL_SUBSTRATE)
            for family in families
        ):
            raise ValueError("required auxiliary families must be unique non-core families")
        missing_profile = set(profile.required_auxiliary_families) - set(families)
        if missing_profile:
            raise ValueError(
                "training profile requires explicit auxiliary families: "
                f"{sorted(family.name for family in missing_profile)}"
            )
        cuda_index = self.device.removeprefix("cuda:")
        valid_device = self.device in {"auto", "cpu", "mps", "cuda"} or (
            self.device.startswith("cuda:") and cuda_index.isdigit()
        )
        if not valid_device or self.precision not in {"auto", "fp32", "bf16", "fp16"}:
            raise ValueError("device or precision selection is invalid")

    @property
    def tokens_per_update(self) -> int:
        return self.context_length * self.micro_batch_size * self.gradient_accumulation_steps

    @property
    def total_steps(self) -> int:
        return ceil(self.total_tokens / self.tokens_per_update)

    @property
    def warmup_steps(self) -> int:
        return max(1, min(max(1, self.total_steps - 1), ceil(self.warmup_tokens / self.tokens_per_update)))


@dataclass(slots=True)
class MRCRATrainingState:
    step: int = 0
    tokens_seen: int = 0
    valid_targets_seen: int = 0
    bytes_seen: int = 0
    elapsed_seconds: float = 0.0
    last_evaluation_step: int = 0
    last_evaluation_metrics: dict[str, float] = field(default_factory=dict)
    last_event_proposal_logit_max: float | None = None
    event_proposal_logit_slope_ema: float = 0.0
    event_proposal_observations: int = 0
    low_clip_coefficient_steps: int = 0
    first_hard_event_step: int = 0
    first_hard_event_tokens: int = 0
    first_hard_event_checkpoint: str | None = None
    last_progress_observation_step: int = 0
    last_progress_pressure: float = 0.0
    progress_observations: int = 0
    pc_rasl_updates_due: int = 0
    pc_rasl_trajectories_captured: int = 0
    pc_rasl_replay_updates: int = 0


def progress_conditioned_rasl_configuration(
    model: MRCRALanguageModel,
    config: MRCRATrainingConfig,
) -> CognitiveRASLConfig:
    """Build the compact production PC-RASL learner contract.

    The critic is intentionally much smaller than the actor, no target actor is
    retained, and the actor objective contains no duplicate task-loss term.
    This helper is public so production-profile tests can audit those resource
    and authority invariants without constructing a complete trainer.
    """

    if not config.progress_conditioned_rasl:
        raise ValueError(
            "a PC-RASL learner configuration requires progress_conditioned_rasl"
        )
    return CognitiveRASLConfig(
        core=ResonantAdjointSurpriseConfig(
            critic_width=64,
            minimum_critic_width=16,
            critic_layers=1,
            critic_scales=1,
            critic_heads=4,
            critic_modes=8,
            spectral_modes=4,
            spectral_basis_order=4,
            action_rank=4,
            latent_modes=4,
            task_weight=0.0,
            surprise_cross_entropy_weight=0.25,
            trust_region_weight=0.02,
            maximum_critic_parameter_fraction=(
                1.0 if model.parameter_count < 1_000_000 else 0.20
            ),
            require_external_reward=True,
        ),
        maximum_candidates=config.pc_rasl_candidate_count,
        maintain_target_actor=False,
    )


def _runtime_state_energy(state: MRCRARuntimeState, target_rms: float) -> tuple[Tensor, Tensor]:
    energies = torch.stack(tuple(
        resonator.value.float().square().mean()
        for carrier in state.carrier
        for block in carrier.blocks
        for resonator in block.resonators
    ))
    rms = energies.clamp_min(0).sqrt()
    penalty = (energies - target_rms**2).clamp_min(0).mean()
    return penalty, rms.max()


def _cpu_thread_calibration_worker(
    model_config,
    model_authority: str,
    state_path: str,
    output_path: str,
    maximum_length: int,
    threads: int,
    result_queue,
) -> None:
    """Measure one candidate in a spawn-isolated PyTorch runtime."""

    try:
        torch.set_num_threads(threads)
        model = MRCRALanguageModel(
            model_config, model_authority=model_authority
        )
        model.load_state_dict(
            torch.load(
                state_path,
                map_location="cpu",
                weights_only=True,
            )
        )
        model.eval()
        carrier = model.cognitive.carrier
        carrier.configure_activation_execution("retain")
        length = min(
            maximum_length,
            max(64, min(256, maximum_length)),
        )
        batch = max(1, min(4, 1_024 // length))
        values = torch.linspace(
            -0.2,
            0.2,
            steps=batch * length * carrier.config.input_dim,
            dtype=model.token_embedding.weight.dtype,
        ).reshape(batch, length, carrier.config.input_dim)
        mask = torch.ones(batch, length, dtype=torch.bool)
        samples: list[float] = []
        output = None
        input_gradient = None
        for repetition in range(3):
            model.zero_grad(set_to_none=True)
            local_values = values.detach().clone().requires_grad_(True)
            started = perf_counter()
            output = carrier.prefill(
                local_values, mask, project_output=False,
            ).latent
            output.float().square().mean().backward()
            elapsed = perf_counter() - started
            input_gradient = local_values.grad
            if input_gradient is None:
                raise RuntimeError(
                    "CPU calibration omitted the carrier input adjoint"
                )
            if repetition:
                samples.append(elapsed)
        if (
            output is None
            or input_gradient is None
            or not torch.isfinite(output).all()
            or not torch.isfinite(input_gradient).all()
        ):
            raise RuntimeError(
                "CPU calibration produced no finite carrier output/adjoint"
            )
        retained = torch.cat((
            output.detach().reshape(-1),
            input_gradient.detach().reshape(-1),
        ))
        checksum = sha256(
            retained.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        # Retain the complete calibration output outside the queue. The
        # serious profile's 64x256 tensor can exceed a multiprocessing pipe's
        # bounded buffer if the parent waits for process completion before
        # reading, whereas this temporary tensor is small, deterministic, and
        # removed with the surrounding calibration directory.
        torch.save(retained.cpu(), output_path)
        result_queue.put(
            {
                "ok": True,
                "threads": threads,
                "seconds": sum(samples) / len(samples),
                "checksum": checksum,
            }
        )
    except BaseException as error:
        result_queue.put(
            {
                "ok": False,
                "threads": threads,
                "error": f"{type(error).__name__}: {error}",
            }
        )


def _cpu_calibration_outputs_equivalent(
    reference: Tensor,
    candidate: Tensor,
) -> bool:
    """Accept only tight float32-equivalent results across thread schedules."""

    return bool(
        reference.shape == candidate.shape
        and reference.dtype == candidate.dtype
        and torch.isfinite(reference).all()
        and torch.isfinite(candidate).all()
        and torch.allclose(
            candidate,
            reference,
            rtol=3e-5,
            atol=3e-6,
        )
    )


def _calibrate_cpu_thread_count(
    model: MRCRALanguageModel,
    *,
    maximum_length: int,
) -> tuple[int, dict[int, float]]:
    """Measure the real carrier in one fresh subprocess per candidate."""

    available = max(1, os.cpu_count() or 1)
    candidates = tuple(
        value for value in (2, 4, 6, 8) if value <= available
    ) or (1,)
    rng = torch.random.get_rng_state()
    original_threads = torch.get_num_threads()
    completed = False
    timings: dict[int, float] = {}
    checksums: list[str] = []
    reference_output: Tensor | None = None
    try:
        with tempfile.TemporaryDirectory(
            prefix="mrcra-cpu-calibration-"
        ) as directory:
            state_path = str(Path(directory) / "model-state.pt")
            torch.save(model.state_dict(), state_path)
            context = multiprocessing.get_context("spawn")
            for threads in candidates:
                output_path = str(
                    Path(directory) / f"candidate-{threads}-output.pt"
                )
                result_queue = context.Queue(maxsize=1)
                process = context.Process(
                    target=_cpu_thread_calibration_worker,
                    args=(
                        model.config,
                        model.model_authority,
                        state_path,
                        output_path,
                        maximum_length,
                        threads,
                        result_queue,
                    ),
                )
                process.start()
                process.join(timeout=180)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
                    raise RuntimeError(
                        f"CPU calibration candidate {threads} timed out"
                    )
                if process.exitcode != 0:
                    raise RuntimeError(
                        "CPU calibration subprocess exited "
                        f"with code {process.exitcode}"
                    )
                result = result_queue.get(timeout=5)
                result_queue.close()
                result_queue.join_thread()
                if not result.get("ok"):
                    raise RuntimeError(
                        "CPU calibration candidate failed: "
                        + str(result.get("error"))
                    )
                timings[threads] = float(result["seconds"])
                checksums.append(str(result["checksum"]))
                candidate_output = torch.load(
                    output_path, map_location="cpu", weights_only=True,
                )
                if (
                    not isinstance(candidate_output, Tensor)
                    or not torch.isfinite(candidate_output).all()
                ):
                    raise RuntimeError(
                        "CPU calibration candidate retained invalid output"
                    )
                if reference_output is None:
                    reference_output = candidate_output
                elif (
                    result["checksum"] != checksums[0]
                    and not _cpu_calibration_outputs_equivalent(
                        reference_output, candidate_output,
                    )
                ):
                    maximum_absolute = float(
                        (candidate_output - reference_output).abs().max()
                    )
                    raise RuntimeError(
                        "CPU thread candidates materially changed carrier "
                        f"output (maximum absolute error {maximum_absolute:.8g})"
                    )
        if reference_output is None:
            raise RuntimeError("CPU thread calibration produced no output")
        selected = min(timings, key=lambda value: (timings[value], value))
        torch.set_num_threads(selected)
        completed = True
        return selected, timings
    finally:
        if not completed:
            torch.set_num_threads(original_threads)
        torch.random.set_rng_state(rng)


class _NullReporter:
    def log(self, metrics: dict[str, float], *, step: int) -> None:  # noqa: ARG002
        return None

    def alert(self, title: str, text: str, *, level: str, step: int) -> None:  # noqa: ARG002
        return None

    def log_phase_transition_trace(self, path: Path, *, step: int) -> int:  # noqa: ARG002
        return 0

    def finish(self) -> None:
        return None


class _NonAuthoritativeReporter:
    """Contain observer failures after optimization authority has begun.

    The wrapped reporter is allowed to lose dashboard delivery, but it cannot
    unwind an already valid optimizer mutation. Every contained failure is
    appended to a small local receipt using only primitive text fields. The
    receipt writer itself is best-effort because observation authority cannot
    become a second failure path.
    """

    def __init__(self, reporter: object, output_dir: str) -> None:
        self._reporter = reporter
        self._failure_path = Path(output_dir) / "observation_failures.jsonl"
        self.failure_count = 0

    def _record(self, operation: str, error: BaseException) -> None:
        self.failure_count += 1
        payload = {
            "kind": "observation_failure",
            "sequence": self.failure_count,
            "operation": operation,
            "error_type": type(error).__name__,
            "message": str(error),
        }
        try:
            self._failure_path.parent.mkdir(parents=True, exist_ok=True)
            with self._failure_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        payload,
                        sort_keys=True,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                handle.flush()
        except Exception:
            pass
        print(
            "[MRCRA OBSERVATION WARN] "
            f"{operation} failed: {type(error).__name__}: {error}",
            flush=True,
        )

    def __getattr__(self, name: str):
        target = getattr(self._reporter, name)
        if not callable(target):
            return target

        def guarded(*args, **kwargs):
            try:
                return target(*args, **kwargs)
            except ValueError:
                # Shape, dtype, device, and target-domain violations are
                # caller contract failures. Silently changing backends would
                # conceal invalid training data rather than recover a failed
                # accelerator.
                raise
            except Exception as error:
                self._record(name, error)
                return None

        return guarded


class MRCRANextTokenTrainer:
    """Exact CE trainer with bounded live state and fail-closed resume."""

    def __init__(
        self,
        model: MRCRALanguageModel,
        tokenizer: TextTokenizer,
        train_stream: PackedTokenStream,
        config: MRCRATrainingConfig,
        evaluation_batches: Sequence[PackedBatch] = (),
        *,
        progress_probe_batches: Sequence[PackedBatch] = (),
        supervision_provider: CognitiveSupervisionProvider | None = None,
    ) -> None:
        if tokenizer.vocabulary_size != model.vocabulary_size:
            raise ValueError("tokenizer vocabulary does not match MRCRA")
        if config.micro_batch_size != 1 and model.config.cognitive.maximum_cognitive_steps:
            # The implementation supports larger batches, but the serious
            # memory budget is intentionally fail-closed at microbatch one.
            raise ValueError("the serious stateful MRCRA trainer requires micro_batch_size=1")
        if config.cstm_enabled and config.cstm_execution == "sampled":
            # A selected scale must be able to execute at least one complete
            # causal coefficient row without violating the advertised hard
            # participation bound.  Sampled prediction uses the primary
            # horizon plus one rotated extra horizon when extras exist.
            horizon_count = (
                1
                if len(model.cstm_predictor.config.horizon_blocks) == 1
                else 2
            )
            coarsest_support = 2 ** (model.config.carrier.scales - 1)
            minimum_complete_row = (
                config.micro_batch_size
                * horizon_count
                * coarsest_support
            )
            if (
                config.cstm_target_participation_budget
                < minimum_complete_row
            ):
                raise ValueError(
                    "sampled CSTM target participation budget cannot fit one "
                    "complete row at the coarsest carrier scale; require at "
                    f"least {minimum_complete_row}"
                )
        if len(evaluation_batches) != config.evaluation_batches:
            raise ValueError(
                "retained evaluation batches must match the training configuration"
            )
        if len(progress_probe_batches) != config.progress_probe_batches:
            raise ValueError(
                "retained progress-probe batches must match the training configuration"
            )
        if any(
            batch.input_ids.shape
            != (config.micro_batch_size, config.context_length)
            for batch in evaluation_batches
        ):
            raise ValueError(
                "retained evaluation batches must match configured batch/context shape"
            )
        if any(
            batch.input_ids.shape
            != (config.micro_batch_size, config.progress_probe_length)
            for batch in progress_probe_batches
        ):
            raise ValueError(
                "progress-probe batches must match configured batch/probe shape"
            )
        self.model, self.tokenizer, self.train_stream = model, tokenizer, train_stream
        materialize_tokenizer_artifacts(tokenizer, config.output_dir)
        self.evaluation_batches = tuple(evaluation_batches)
        self.progress_probe_batches = tuple(progress_probe_batches)
        self.config = config
        self.learning_progress = (
            LearningProgressAuthority(config.learning_progress)
            if config.progress_conditioned_rasl else None
        )
        self.trainer_mode = TrainerMode(config.trainer_mode)
        if (
            self.trainer_mode.permits_persistence
            and not model.config.cognitive.enable_persistent_session_training
        ):
            raise ValueError(
                "persistent trainer mode requires enable_persistent_session_training"
            )
        self.supervision_provider = (
            EvidenceBackedCognitiveSupervisor()
            if supervision_provider is None else supervision_provider
        )
        self.schedule = CognitiveObjectiveSchedule.curriculum(config.curriculum_stage)
        self.integrated_cognitive_path = config.integrated_cognitive_path
        carrier_alignment = 2 ** (model.config.carrier.scales - 1)
        static_alignment = lcm(carrier_alignment, config.cognitive_stride)
        automatic_small_buckets: list[int] = []
        candidate = static_alignment
        while candidate < config.document_bucket_lengths[0]:
            automatic_small_buckets.append(candidate)
            candidate *= 2
        resolved_document_buckets = tuple(
            sorted(set(automatic_small_buckets + list(config.document_bucket_lengths)))
        )
        self.document_batch_planner = (
            DocumentMajorBatchPlanner(
                tbptt_length=config.tbptt_length,
                bucket_lengths=resolved_document_buckets,
                token_budget=config.document_batch_token_budget,
                alignment=carrier_alignment,
                cognitive_stride=config.cognitive_stride,
                padding_token_id=tokenizer.pad_token_id,
                grouping_policy=config.document_grouping_policy,
                plan_cache_capacity=config.document_plan_cache_capacity,
                actor_configuration_digest=sha256(
                    repr(model.config).encode("utf-8")
                ).hexdigest(),
                device_torch_fingerprint=sha256(
                    (
                        f"{config.device}|{config.precision}|"
                        f"{torch.__version__}"
                    ).encode("utf-8")
                ).hexdigest(),
                compiler_policy=(
                    "auto"
                    if config.compile_tensor_cores is None
                    else "on" if config.compile_tensor_cores else "off"
                ),
            )
            if config.integrated_cognitive_path
            and config.document_static_batching
            else None
        )
        self.cstm_enabled = (
            config.cstm_enabled and config.integrated_cognitive_path
        )
        resolved_device = config.device
        device_reason = "explicit selection" if config.device != "auto" else "automatic priority"
        if (
            config.device == "auto"
            and config.integrated_cognitive_path
            and not torch.cuda.is_available()
            and torch.backends.mps.is_available()
            and model.parameter_count < 20_000_000
        ):
            # The light actor's heterogeneous graph/control workload is launch
            # bound on MPS. Local matched probes show CPU is several times
            # faster once cognition participates in backward. Explicit MPS
            # remains available for hardware-specific experimentation.
            resolved_device = "cpu"
            device_reason = "light integrated cognition is faster on CPU than MPS"
        self.device = _device_for(resolved_device)
        _configure_cuda(self.device)
        self.model.to(self.device)
        interop_note = "not applicable"
        cpu_thread_calibration: dict[int, float] = {}
        resolved_cpu_threads = config.cpu_threads
        if self.device.type == "cpu":
            if config.cpu_threads == 0:
                if config.performance_calibration:
                    (
                        resolved_cpu_threads,
                        cpu_thread_calibration,
                    ) = _calibrate_cpu_thread_count(
                        model, maximum_length=config.tbptt_length,
                    )
                else:
                    resolved_cpu_threads = min(
                        4, max(1, os.cpu_count() or 1)
                    )
                    torch.set_num_threads(resolved_cpu_threads)
            else:
                torch.set_num_threads(config.cpu_threads)
            interop_note = "configured"
            if torch.get_num_interop_threads() != config.cpu_interop_threads:
                try:
                    torch.set_num_interop_threads(config.cpu_interop_threads)
                except RuntimeError:
                    # PyTorch permits this setting only before inter-op work
                    # begins. Production entrypoints configure it before model
                    # construction; embedded callers retain their established
                    # pool rather than failing after doing useful work.
                    interop_note = "existing pool retained after initialization"
        self.amp_dtype = _precision_for(self.device, config.precision)
        self.runtime = _runtime_details(self.device, self.amp_dtype)
        activation_element_bytes = (
            torch.empty(
                (),
                dtype=(
                    self.amp_dtype
                    or model.token_embedding.weight.dtype
                ),
            ).element_size()
        )
        executed_width = sum(
            scale.width
            for scale in model.config.carrier.scale_configs()
        )
        maximum_planned_physical_tokens = (
            min(
                config.document_batch_token_budget,
                config.context_length * config.micro_batch_size,
            )
            if self.document_batch_planner is not None
            else config.tbptt_length * config.micro_batch_size
        )
        estimated_activation_bytes = (
            maximum_planned_physical_tokens
            * executed_width
            * model.config.carrier.layers
            * activation_element_bytes
            * 144
        )
        # Calibration-disabled explicit selective execution checkpoints every
        # scale. Measured calibration below narrows this to the partitions
        # responsible for most retained autograd storage.
        selective_candidate_scales = tuple(
            range(model.config.carrier.scales)
        )
        activation_partition_census = ()
        calibration_measurements = ()
        selected_measurement = None
        # Capture capacity before calibration allocations enter the host or
        # device allocator cache. Candidate peaks are incremental relative to
        # that state, so this is the matching reserve authority.
        activation_memory_before_calibration = observe_memory(self.device)
        activation_cache_path: Path | None = None
        activation_cache_key: str | None = None
        activation_cache_hit = False
        cached_document_cost_model: DocumentExecutionCostModel | None = None
        cached_document_cost_observations: list[dict[str, object]] = []
        if (
            config.activation_calibration
            and config.activation_calibration_cache_directory is not None
            and self.device.type == "cpu"
        ):
            source_authority = sha256()
            for source_path in sorted(
                Path(__file__).parent.glob("*.py")
            ):
                source_authority.update(source_path.name.encode("utf-8"))
                source_authority.update(source_path.read_bytes())
            cache_identity = {
                "schema_version": 1,
                "source_authority": source_authority.hexdigest(),
                "actor_configuration": sha256(
                    repr(model.config).encode("utf-8")
                ).hexdigest(),
                "torch_version": str(torch.__version__),
                "platform": tuple(os.uname()),
                "device": str(self.device),
                "precision": str(self.amp_dtype),
                "cpu_threads": int(resolved_cpu_threads),
                "activation_request": config.activation_policy,
                "tbptt_length": config.tbptt_length,
                "maximum_physical_tokens": maximum_planned_physical_tokens,
                "document_buckets": resolved_document_buckets,
                "document_cost_calibration": config.document_cost_calibration,
            }
            activation_cache_key = sha256(
                json.dumps(
                    cache_identity,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            activation_cache_path = (
                Path(config.activation_calibration_cache_directory)
                / f"{activation_cache_key}.json"
            )
            if activation_cache_path.is_file():
                try:
                    cached = json.loads(
                        activation_cache_path.read_text(encoding="utf-8")
                    )
                    if (
                        not isinstance(cached, dict)
                        or cached.get("schema_version") != 1
                        or cached.get("cache_key") != activation_cache_key
                    ):
                        raise ValueError(
                            "activation calibration cache identity differs"
                        )
                    calibration_measurements = tuple(
                        ActivationCandidateMeasurement(**dict(item))
                        for item in cached["calibration_measurements"]
                    )
                    activation_partition_census = tuple(
                        ActivationPartitionCensus(**dict(item))
                        for item in cached["activation_partition_census"]
                    )
                    selective_candidate_scales = tuple(
                        int(value)
                        for value in cached[
                            "selective_candidate_scales"
                        ]
                    )
                    self.runtime[
                        "activation_equivalence_max_abs_error"
                    ] = float(cached["equivalence_max_abs_error"])
                    self.runtime[
                        "activation_equivalence_min_cosine"
                    ] = float(cached["equivalence_min_cosine"])
                    if cached.get("document_cost_model") is not None:
                        cached_document_cost_model = (
                            DocumentExecutionCostModel.from_dict(
                                cached["document_cost_model"]
                            )
                        )
                    if (
                        self.document_batch_planner is not None
                        and config.document_cost_calibration
                        and cached_document_cost_model is None
                    ):
                        raise ValueError(
                            "activation cache omitted its document cost model"
                        )
                    cached_document_cost_observations = [
                        dict(item)
                        for item in cached.get(
                            "document_cost_calibration_observations", ()
                        )
                    ]
                    activation_cache_hit = True
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                    OSError,
                ) as error:
                    self.runtime[
                        "activation_calibration_cache_error"
                    ] = (
                        f"{type(error).__module__}.{type(error).__name__}: "
                        f"{' '.join(str(error).split())[:2048]}"
                    )
        if config.activation_calibration and not activation_cache_hit:
            carrier = model.cognitive.carrier
            calibration_length = min(
                config.tbptt_length,
                maximum_planned_physical_tokens,
                2_048,
            )
            # Full-size retain calibration can itself exhaust a unified-memory
            # host before the safety policy has a chance to reject it. Measure
            # one bounded cohort and conservatively project its incremental
            # peak to the planner's largest authorized physical cohort.
            calibration_batch = 1
            calibration_physical_tokens = (
                calibration_batch * calibration_length
            )
            calibration_input = torch.linspace(
                -0.25,
                0.25,
                steps=(
                    calibration_batch
                    * calibration_length
                    * model.config.carrier.input_dim
                ),
                device=self.device,
                dtype=model.token_embedding.weight.dtype,
            ).reshape(
                calibration_batch,
                calibration_length,
                model.config.carrier.input_dim,
            )
            calibration_mask = torch.ones(
                calibration_input.shape[:2],
                dtype=torch.bool,
                device=self.device,
            )

            def census_candidate(
                policy_name: str,
                selected_scales: tuple[int, ...] = (),
            ):
                def execute() -> Tensor:
                    carrier.configure_activation_execution(
                        policy_name,
                        selective_scales=selected_scales,
                    )
                    local_input = (
                        calibration_input.detach().clone().requires_grad_(True)
                    )
                    return carrier.prefill(
                        local_input,
                        calibration_mask,
                        project_output=False,
                    ).latent

                return execute

            if config.activation_policy in {"auto", "selective"}:
                activation_partition_census = census_activation_partitions(
                    census_candidate("retain"),
                    {
                        f"scale:{scale}": census_candidate(
                            "selective", (scale,)
                        )
                        for scale in range(model.config.carrier.scales)
                    },
                    device=self.device,
                )
                selective_candidate_scales = tuple(
                    int(partition.split(":", 1)[1])
                    for partition in select_activation_dominant_partitions(
                        activation_partition_census,
                        reduction_fraction=0.60,
                    )
                )

            def activation_candidate(
                policy_name: str,
                candidate_input: Tensor = calibration_input,
                candidate_mask: Tensor = calibration_mask,
            ):
                def execute() -> Tensor:
                    carrier.configure_activation_execution(
                        policy_name,
                        selective_scales=(
                            selective_candidate_scales
                            if policy_name == "selective" else ()
                        ),
                    )
                    model.zero_grad(set_to_none=True)
                    local_input = (
                        candidate_input.detach().clone().requires_grad_(True)
                    )
                    result = (
                        carrier.prefill_coarse_checkpointed(
                            local_input, candidate_mask,
                        )
                        if policy_name == "whole_span"
                        else carrier.prefill(
                            local_input,
                            candidate_mask,
                            project_output=False,
                        )
                    )
                    result.latent.float().square().mean().backward()
                    if local_input.grad is None:
                        raise RuntimeError(
                            "activation calibration omitted its input adjoint"
                        )
                    output = torch.cat(
                        (
                            result.latent.detach().reshape(-1),
                            local_input.grad.detach().reshape(-1),
                        )
                    )
                    model.zero_grad(set_to_none=True)
                    carrier._last_composite_receipt = None
                    return output

                return execute

            candidate_names = (
                ("retain", "selective", "whole_span")
                if config.activation_policy == "auto"
                else (config.activation_policy,)
            )
            conservative_peak_bytes: dict[str, int] = {}
            for policy_name in candidate_names:
                def saved_graph_candidate(
                    name: str = policy_name,
                ) -> Tensor:
                    carrier.configure_activation_execution(
                        name,
                        selective_scales=(
                            selective_candidate_scales
                            if name == "selective" else ()
                        ),
                    )
                    local_input = (
                        calibration_input.detach().clone().requires_grad_(True)
                    )
                    result = (
                        carrier.prefill_coarse_checkpointed(
                            local_input, calibration_mask,
                        )
                        if name == "whole_span"
                        else carrier.prefill(
                            local_input,
                            calibration_mask,
                            project_output=False,
                        )
                    )
                    return result.latent

                saved_bytes, _, _ = measure_saved_tensor_bytes(
                    saved_graph_candidate,
                    device=self.device,
                )
                conservative_peak_bytes[policy_name] = max(
                    saved_bytes,
                    (
                        estimated_activation_bytes
                        if policy_name == "retain"
                        else 0
                    ),
                )
            equivalence_outputs = {
                name: activation_candidate(name)()
                for name in candidate_names
            }
            reference_output = equivalence_outputs[candidate_names[0]]
            equivalence_max_error = 0.0
            equivalence_min_cosine = 1.0
            for name in candidate_names[1:]:
                candidate_output = equivalence_outputs[name]
                if candidate_output.shape != reference_output.shape:
                    raise RuntimeError(
                        "activation candidates changed their output shape"
                    )
                difference = (
                    candidate_output.float()
                    - reference_output.float()
                )
                equivalence_max_error = max(
                    equivalence_max_error,
                    float(difference.abs().max().cpu()),
                )
                cosine = float(F.cosine_similarity(
                    candidate_output.float().reshape(1, -1),
                    reference_output.float().reshape(1, -1),
                ).cpu())
                equivalence_min_cosine = min(
                    equivalence_min_cosine, cosine
                )
                if (
                    not torch.allclose(
                        candidate_output,
                        reference_output,
                        atol=3e-5,
                        rtol=3e-4,
                    )
                    or cosine < 0.99999
                ):
                    raise RuntimeError(
                        "activation candidates exceed the float32 forward/"
                        "adjoint equivalence tolerance"
                    )
            equivalence_reference = (
                reference_output.detach().float().cpu().contiguous()
            )
            equivalence_digest = sha256(
                str(tuple(equivalence_reference.shape)).encode("ascii")
                + equivalence_reference.numpy().tobytes()
                + b"|float32-atol=3e-5|rtol=3e-4|cosine=0.99999"
            ).hexdigest()
            calibration_measurements = calibrate_activation_candidates(
                {
                    name: activation_candidate(name)
                    for name in candidate_names
                },
                device=self.device,
                calibration_physical_tokens=calibration_physical_tokens,
                target_physical_tokens=maximum_planned_physical_tokens,
                conservative_peak_bytes=conservative_peak_bytes,
            )
            calibration_measurements = tuple(
                replace(item, output_digest=equivalence_digest)
                for item in calibration_measurements
            )
            self.runtime["activation_equivalence_max_abs_error"] = (
                equivalence_max_error
            )
            self.runtime["activation_equivalence_min_cosine"] = (
                equivalence_min_cosine
            )
        self.activation_execution_policy = resolve_activation_execution_policy(
            requested=config.activation_policy,
            device=self.device,
            required_reserve_bytes=config.activation_memory_reserve_bytes,
            estimated_retain_bytes=estimated_activation_bytes,
            candidates=calibration_measurements,
            allow_unsafe_explicit=config.allow_unsafe_activation_policy,
            memory_observation=activation_memory_before_calibration,
        )
        self.activation_policy_token_limits = {
            policy_name: maximum_safe_activation_physical_tokens(
                self.activation_execution_policy,
                candidate_policy=policy_name,
                alignment=carrier_alignment,
                maximum_physical_tokens=maximum_planned_physical_tokens,
            )
            for policy_name in ("retain", "selective", "whole_span")
        }
        # The resolved maximum-shape policy is always authoritative for the
        # complete configured cohort, including legacy receipts that predate
        # per-candidate projected peaks.
        self.activation_policy_token_limits[
            self.activation_execution_policy.resolved
        ] = maximum_planned_physical_tokens
        activation_policy_timings = {
            item.policy: item.elapsed_seconds
            for item in calibration_measurements
            if item.finite
        }
        activation_policy_timings.setdefault(
            self.activation_execution_policy.resolved,
            0.0 if not calibration_measurements else float("inf"),
        )
        if self.document_batch_planner is not None:
            self.document_batch_planner.activation_policy = (
                self.activation_execution_policy.resolved
            )
            self.document_batch_planner.activation_policy_token_limits = dict(
                self.activation_policy_token_limits
            )
            self.document_batch_planner.activation_policy_timings = dict(
                activation_policy_timings
            )
            self.document_batch_planner.maximum_candidate_activation_bytes = (
                max(
                    1,
                    self.activation_execution_policy.memory.available_bytes
                    - config.activation_memory_reserve_bytes,
                )
            )
            self.document_batch_planner.activation_bytes_per_token = max(
                1,
                ceil(
                    estimated_activation_bytes
                    / max(1, maximum_planned_physical_tokens)
                ),
            )
            self.document_batch_planner.device_torch_fingerprint = (
                self.activation_execution_policy.hardware_fingerprint
            )
            self.document_batch_planner._group_cache.clear()
            selected_measurement = next(
                (
                    item
                    for item in calibration_measurements
                    if item.policy
                    == self.activation_execution_policy.resolved
                ),
                None,
            )
            if selected_measurement is not None:
                self.document_batch_planner.activation_bytes_per_token = max(
                    1,
                    ceil(
                        selected_measurement.reserve_peak_bytes
                        / max(1, maximum_planned_physical_tokens)
                    ),
                )
                self.document_batch_planner._group_cache.clear()
            if cached_document_cost_model is not None:
                self.document_batch_planner.cost_model = (
                    cached_document_cost_model
                )
                self.runtime[
                    "document_cost_calibration_observations"
                ] = cached_document_cost_observations
            elif (
                selected_measurement is not None
                and config.document_cost_calibration
            ):
                physical = calibration_physical_tokens
                # The activation calibration deliberately uses a single
                # bounded cohort.  A second timing observation must therefore
                # differ in physical-token count; reusing the whole cohort
                # would leave the affine launch/token fit underdetermined
                # (especially in smoke-scale configurations where the
                # calibration batch is exactly one).
                single_length = max(1, calibration_length // 2)
                single_input = calibration_input[:1, :single_length]
                single_mask = calibration_mask[:1, :single_length]
                single_measurement = calibrate_activation_candidates(
                    {
                        self.activation_execution_policy.resolved:
                        activation_candidate(
                            self.activation_execution_policy.resolved,
                            single_input,
                            single_mask,
                        )
                    },
                    device=self.device,
                    calibration_physical_tokens=single_length,
                    target_physical_tokens=single_length,
                )[0]
                activation_bytes_per_token = (
                    self.document_batch_planner.activation_bytes_per_token
                )
                length_band_observations: list[
                    tuple[int, float, int, int]
                ] = []
                eligible_buckets = tuple(
                    padded_length
                    for padded_length in resolved_document_buckets
                    if padded_length <= calibration_input.shape[1]
                )
                maximum_band_observations = 8
                if len(eligible_buckets) <= maximum_band_observations:
                    measured_buckets = eligible_buckets
                else:
                    measured_indices = tuple(sorted({
                        round(
                            index
                            * (len(eligible_buckets) - 1)
                            / (maximum_band_observations - 1)
                        )
                        for index in range(maximum_band_observations)
                    }))
                    measured_buckets = tuple(
                        eligible_buckets[index]
                        for index in measured_indices
                    )
                for padded_length in measured_buckets:
                    band_input = calibration_input[
                        :1, :padded_length
                    ]
                    band_mask = calibration_mask[
                        :1, :padded_length
                    ]
                    band_measurement = calibrate_activation_candidates(
                        {
                            self.activation_execution_policy.resolved:
                            activation_candidate(
                                self.activation_execution_policy.resolved,
                                band_input,
                                band_mask,
                            )
                        },
                        device=self.device,
                        calibration_physical_tokens=padded_length,
                        target_physical_tokens=padded_length,
                    )[0]
                    length_band_observations.append(
                        (
                            padded_length,
                            band_measurement.elapsed_seconds,
                            1,
                            band_measurement.reserve_peak_bytes,
                        )
                    )
                self.document_batch_planner.cost_model = (
                    measured_document_cost_model(
                        single_invocation_seconds=max(
                            single_measurement.elapsed_seconds, 1e-12
                        ),
                        batched_invocation_seconds=max(
                            selected_measurement.elapsed_seconds, 1e-12
                        ),
                        single_physical_tokens=single_length,
                        batched_physical_tokens=physical,
                        length_bands=resolved_document_buckets,
                        activation_policy=(
                            self.activation_execution_policy.resolved
                        ),
                        hardware_fingerprint=(
                            self.activation_execution_policy.hardware_fingerprint
                        ),
                        activation_bytes_per_token=(
                            activation_bytes_per_token
                        ),
                        length_band_observations=(
                            length_band_observations
                        ),
                    )
                )
                self.runtime[
                    "document_cost_calibration_observations"
                ] = [
                    {
                        "padded_length": length,
                        "elapsed_seconds": seconds,
                        "batch": batch,
                        "peak_bytes": peak,
                    }
                    for length, seconds, batch, peak
                    in length_band_observations
                ]
            else:
                self.runtime[
                    "document_cost_calibration_observations"
                ] = []
        self.runtime["activation_calibration_cache_hit"] = (
            activation_cache_hit
        )
        self.runtime["activation_calibration_cache_key"] = (
            activation_cache_key
        )
        self.runtime["activation_calibration_cache_path"] = (
            None
            if activation_cache_path is None
            else str(activation_cache_path)
        )
        if (
            activation_cache_path is not None
            and not activation_cache_hit
            and calibration_measurements
        ):
            activation_cache_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            cache_payload = {
                "schema_version": 1,
                "cache_key": activation_cache_key,
                "calibration_measurements": [
                    asdict(item) for item in calibration_measurements
                ],
                "activation_partition_census": [
                    asdict(item) for item in activation_partition_census
                ],
                "selective_candidate_scales": list(
                    selective_candidate_scales
                ),
                "equivalence_max_abs_error": self.runtime.get(
                    "activation_equivalence_max_abs_error", 0.0
                ),
                "equivalence_min_cosine": self.runtime.get(
                    "activation_equivalence_min_cosine", 1.0
                ),
                "document_cost_model": (
                    None
                    if self.document_batch_planner is None
                    else self.document_batch_planner.cost_model.to_dict()
                ),
                "document_cost_calibration_observations": self.runtime.get(
                    "document_cost_calibration_observations", []
                ),
            }
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{activation_cache_path.name}.",
                suffix=".tmp",
                dir=activation_cache_path.parent,
            )
            try:
                with os.fdopen(
                    descriptor, "w", encoding="utf-8"
                ) as handle:
                    json.dump(
                        cache_payload,
                        handle,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, activation_cache_path)
            finally:
                Path(temporary_name).unlink(missing_ok=True)
            if config.activation_calibration_reexec:
                print(
                    "Activation/document calibration cached; restarting once "
                    "to release calibration allocator arenas before the long "
                    "training run.",
                    flush=True,
                )
                os.execv(
                    sys.executable,
                    [sys.executable, *sys.argv],
                )
        selective_scales: tuple[int, ...] = (
            selective_candidate_scales
            if self.activation_execution_policy.resolved == "selective"
            else ()
        )
        self.activation_retain_physical_token_limit = (
            self.activation_policy_token_limits["retain"]
        )
        model.cognitive.carrier.configure_activation_execution(
            self.activation_execution_policy.resolved,
            selective_scales=selective_scales,
        )
        self.runtime["activation_execution_policy"] = (
            self.activation_execution_policy.to_dict()
        )
        self.runtime["activation_execution_policy_digest"] = (
            self.activation_execution_policy.digest
        )
        self.runtime["carrier_activation_checkpointing"] = (
            self.activation_execution_policy.resolved != "retain"
        )
        self.runtime["carrier_activation_checkpointing_policy"] = (
            self.activation_execution_policy.resolved
        )
        self.runtime["carrier_activation_memory_budget_bytes"] = (
            self.activation_execution_policy.memory.available_bytes
            - config.activation_memory_reserve_bytes
        )
        self.runtime["estimated_uncheckpointed_carrier_activation_bytes"] = (
            estimated_activation_bytes
        )
        self.runtime["carrier_selective_checkpoint_scales"] = (
            selective_scales
        )
        self.runtime["carrier_retain_physical_token_limit"] = (
            self.activation_retain_physical_token_limit
        )
        self.runtime["carrier_activation_physical_token_limits"] = dict(
            self.activation_policy_token_limits
        )
        self.runtime["carrier_activation_candidate_seconds"] = dict(
            activation_policy_timings
        )
        self.runtime["carrier_shape_conditional_activation"] = (
            any(
                policy_name != self.activation_execution_policy.resolved
                and limit > 0
                for policy_name, limit
                in self.activation_policy_token_limits.items()
            )
        )
        self.runtime["carrier_activation_partition_census"] = [
            asdict(item) for item in activation_partition_census
        ]
        self.runtime["document_static_batching"] = (
            self.document_batch_planner is not None
        )
        self.runtime["document_bucket_lengths"] = resolved_document_buckets
        self.runtime["document_batch_token_budget"] = (
            config.document_batch_token_budget
        )
        self.runtime["document_grouping_policy"] = (
            config.document_grouping_policy
        )
        self.runtime["document_cost_model_digest"] = (
            None
            if self.document_batch_planner is None
            else self.document_batch_planner.cost_model.digest
        )
        self.runtime["document_plan_cache_capacity"] = (
            config.document_plan_cache_capacity
        )
        self.runtime["document_cost_calibration"] = (
            config.document_cost_calibration
        )
        self.runtime["device_selection_reason"] = device_reason
        self.runtime["cpu_threads"] = torch.get_num_threads()
        self.runtime["cpu_threads_requested"] = config.cpu_threads
        self.runtime["cpu_threads_resolved"] = resolved_cpu_threads
        self.runtime["cpu_thread_calibration_seconds"] = {
            str(key): value
            for key, value in cpu_thread_calibration.items()
        }
        self.runtime["cpu_interop_threads"] = torch.get_num_interop_threads()
        self.runtime["cpu_interop_configuration"] = interop_note
        self.loss_device = self.device
        if (
            config.device == "auto"
            and config.integrated_cognitive_path
            and config.apple_mps_loss_offload
            and self.device.type == "cpu"
            and torch.backends.mps.is_available()
        ):
            # Cognitive authority consists of many small, branch-dependent
            # operations and remains faster on the host.  The exact tiled
            # vocabulary contraction is large and regular, so it is moved to
            # Metal through differentiable copies while CPU parameters remain
            # canonical for optimization and checkpointing.
            self.loss_device = torch.device("mps")
        self.runtime["loss_device"] = str(self.loss_device)
        self.runtime["apple_hybrid_loss_offload"] = (
            self.loss_device.type == "mps" and self.device.type == "cpu"
        )
        loss_dtype = self.amp_dtype or model.token_embedding.weight.dtype
        loss_element_size = torch.empty((), dtype=loss_dtype).element_size()
        estimated_loss_bytes = (
            min(config.tbptt_length, config.context_length)
            * model.vocabulary_size
            * loss_element_size
        )
        self._checkpoint_loss_tiles = (
            config.checkpoint_tiles
            if config.checkpoint_tiles is not None
            else estimated_loss_bytes > config.maximum_retained_loss_bytes
        )
        cce_available = find_spec("cut_cross_entropy") is not None
        mlx_cce_available = False
        if self.device.type == "cpu" and find_spec("mlx") is not None:
            try:
                from .mlx_backend import mlx_available

                mlx_cce_available = mlx_available()
            except (ImportError, RuntimeError):
                mlx_cce_available = False
        cce_capable_cuda = (
            self.loss_device.type == "cuda"
            and torch.cuda.get_device_capability(self.loss_device)[0] >= 8
        )
        compiled_cce_fits_workspace = (
            config.maximum_fused_loss_bytes > 0
            and estimated_loss_bytes <= config.maximum_fused_loss_bytes
        )
        requested_loss_backend = config.exact_loss_backend
        portable_fallback_backend = (
            "torch_compile"
            if cce_available and compiled_cce_fits_workspace
            else "tiled"
        )
        if requested_loss_backend == "auto":
            self._exact_loss_backend = (
                "cce_kahan_full_c"
                if cce_available and cce_capable_cuda
                else "torch_compile"
                if cce_available and compiled_cce_fits_workspace
                else "fused"
                if self.loss_device.type in {"cuda", "mps"}
                and compiled_cce_fits_workspace
                else "tiled"
            )
        elif requested_loss_backend in {"cce_kahan_full_c", "cce_exact"}:
            # The semantic policy is available everywhere. Triton executes it
            # on qualified CUDA; macOS/CPU use the official compiled exact
            # implementation when installed, and our native exact tiled cut
            # otherwise. The latter is stronger than Full-C because it filters
            # neither classifier nor latent gradients.
            self._exact_loss_backend = (
                requested_loss_backend
                if cce_available and cce_capable_cuda
                else "torch_compile"
                if cce_available and compiled_cce_fits_workspace
                else "tiled"
            )
        else:
            self._exact_loss_backend = requested_loss_backend
        if self._exact_loss_backend == "mlx" and not mlx_cce_available:
            raise RuntimeError(
                "the configured MLX exact-loss backend requires Apple Metal "
                "and canonical CPU model tensors"
            )
        self.runtime["mlx_memory_policy_error"] = "not required"
        self.runtime["mlx_memory_policy"] = None
        self.runtime["mlx_active_memory_bytes"] = 0
        self.runtime["mlx_cache_memory_bytes"] = 0
        self.runtime["mlx_peak_memory_bytes"] = 0
        if self._exact_loss_backend == "mlx":
            try:
                from .mlx_backend import configure_mlx_memory

                mlx_memory_policy = configure_mlx_memory(
                    memory_limit_bytes=config.mlx_memory_limit_bytes,
                    cache_limit_bytes=config.mlx_cache_limit_bytes,
                )
                self.runtime["mlx_memory_policy"] = asdict(
                    mlx_memory_policy
                )
            except Exception as error:
                if requested_loss_backend == "mlx":
                    raise RuntimeError(
                        "the explicit MLX exact-loss memory policy could not "
                        "be configured"
                    ) from error
                self._exact_loss_backend = portable_fallback_backend
                self.runtime["mlx_memory_policy_error"] = (
                    f"{type(error).__module__}.{type(error).__name__}: "
                    f"{' '.join(str(error).split())[:2048]}"
                )
        if (
            self._exact_loss_backend == "torch_compile"
            and not cce_available
        ):
            raise RuntimeError(
                "the configured Cut Cross-Entropy backend is unavailable; "
                "install the mrrn[cce] optional dependency"
            )
        self.runtime["loss_projection"] = (
            f"{self._exact_loss_backend}_exact_full_softmax"
        )
        self.runtime["cut_cross_entropy_available"] = cce_available
        self.runtime["mlx_exact_loss_available"] = mlx_cce_available
        self.runtime["mlx_exact_loss_fallback_backend"] = (
            portable_fallback_backend
        )
        self.runtime["mlx_exact_loss_runtime_fallback"] = "not required"
        self.runtime["compiled_cce_fits_workspace"] = (
            compiled_cce_fits_workspace
        )
        self.runtime["requested_exact_loss_backend"] = requested_loss_backend
        self.runtime["exact_loss_backend"] = self._exact_loss_backend
        self.runtime["compiled_cce_runtime_quarantined"] = False
        self.runtime["compiled_cce_runtime_fallbacks"] = 0
        self.runtime["compiled_cce_runtime_fallback_reason"] = "not required"
        self.runtime["estimated_fused_loss_bytes"] = estimated_loss_bytes
        self.runtime["loss_tile_checkpointing"] = self._checkpoint_loss_tiles
        self.runtime["maximum_retained_loss_bytes"] = config.maximum_retained_loss_bytes
        self.runtime["loss_memory_policy"] = (
            "explicit_recompute" if config.checkpoint_tiles is True
            else "explicit_retain" if config.checkpoint_tiles is False
            else "auto_recompute" if self._checkpoint_loss_tiles
            else "auto_retain"
        )
        self.runtime["cstm_configured"] = config.cstm_enabled
        self.runtime["cstm_effective"] = self.cstm_enabled
        self.runtime["cstm_objective_weight"] = config.cstm_weight
        self.runtime["cstm_execution"] = config.cstm_execution
        self.runtime["cstm_sampling_duty_cycle"] = (
            config.cstm_sampling_duty_cycle
        )
        self.runtime["cstm_sampling_uniform_mixture"] = (
            config.cstm_sampling_uniform_mixture
        )
        self.runtime["cstm_max_substrate_vjps"] = (
            config.cstm_max_substrate_vjps
        )
        self.runtime["cstm_target_participation_budget"] = (
            config.cstm_target_participation_budget
        )
        self.runtime["cstm_predictor_update_interval"] = (
            config.cstm_predictor_update_interval
        )
        self.runtime["cstm_maximum_coverage_gap"] = (
            config.cstm_maximum_coverage_gap
        )
        self.runtime["cstm_code_dimension"] = (
            model.cstm_predictor.config.code_dimension
        )
        self.runtime["cstm_predictor_parameters"] = sum(
            parameter.numel()
            for parameter in model.cstm_predictor.parameters()
        )
        resolved_compile_request = (
            False
            if (
                config.compile_tensor_cores is None
                and (
                    not config.performance_calibration
                    or not config.activation_calibration
                )
            )
            else config.compile_tensor_cores
        )
        carrier_execution = resolve_carrier_execution_policy(
            device_type=self.device.type,
            compile_tensor_cores=resolved_compile_request,
            integrated=config.integrated_cognitive_path,
            activation_checkpointing=(
                self.activation_execution_policy.resolved != "retain"
            ),
            activation_policy=self.activation_execution_policy.resolved,
        )
        self.runtime["carrier_execution_backend"] = carrier_execution.backend
        self.runtime["carrier_compiler_backend"] = (
            carrier_execution.compiler_backend
        )
        self.runtime["carrier_affine_scan"] = carrier_execution.affine_scan
        self.runtime["carrier_simplex_residual"] = (
            carrier_execution.simplex_residual
        )
        self.runtime["carrier_scan_saved_tensor_contract"] = (
            "transition_initial_states_mask"
        )
        self.runtime["carrier_checkpoint_granularity"] = (
            carrier_execution.checkpoint_granularity
        )
        self.runtime["carrier_nested_scale_checkpointing"] = (
            self.activation_execution_policy.resolved == "selective"
        )
        self.runtime["carrier_state_boundary"] = (
            "flat_tensor_tree_plus_immutable_static_spec"
        )
        compile_tensor_cores = carrier_execution.compiler_enabled
        compiler_first_shape_seconds = 0.0
        compiler_steady_shape_seconds = 0.0
        compiler_shape_cost_seconds = 0.0
        if compile_tensor_cores:
            try:
                model.cognitive.carrier.enable_compiled_tensor_cores(
                    backend=carrier_execution.compiler_backend
                )
                if config.activation_calibration:
                    # Measure compilation authority before optimizer
                    # construction. The same largest-planned physical cohort
                    # used for activation safety is compiled once and then
                    # executed at steady state. Their nonnegative difference
                    # is charged once per novel static shape by the planner.
                    compiled_candidate = activation_candidate(
                        self.activation_execution_policy.resolved
                    )
                    _synchronize(self.device)
                    compile_started = perf_counter()
                    compiled_candidate()
                    _synchronize(self.device)
                    compiler_first_shape_seconds = (
                        perf_counter() - compile_started
                    )
                    steady_started = perf_counter()
                    compiled_candidate()
                    _synchronize(self.device)
                    compiler_steady_shape_seconds = (
                        perf_counter() - steady_started
                    )
                    compiler_shape_cost_seconds = max(
                        0.0,
                        compiler_first_shape_seconds
                        - compiler_steady_shape_seconds,
                    )
                    if config.compile_tensor_cores is None:
                        eager_seconds = (
                            None
                            if selected_measurement is None
                            else selected_measurement.elapsed_seconds
                        )
                        amortized_compiled_seconds = (
                            compiler_steady_shape_seconds
                            + compiler_shape_cost_seconds
                            / max(1, config.total_steps)
                        )
                        if (
                            eager_seconds is None
                            or not isfinite(eager_seconds)
                            or amortized_compiled_seconds >= eager_seconds
                        ):
                            model.cognitive.carrier.disable_compiled_tensor_cores()
                            compile_tensor_cores = False
                            self.runtime[
                                "carrier_compiler_fallback_reason"
                            ] = (
                                "automatic compiler rejected: measured "
                                f"amortized={amortized_compiled_seconds:.9f}s "
                                f"eager={eager_seconds!r}s"
                            )
                            self.runtime["carrier_execution_backend"] = (
                                "portable_custom_composites"
                            )
            except Exception as error:
                if config.compile_tensor_cores is not None:
                    raise RuntimeError(
                        "explicit carrier tensor-core compilation failed"
                    ) from error
                model.cognitive.carrier.disable_compiled_tensor_cores()
                compile_tensor_cores = False
                self.runtime["carrier_compiler_fallback_reason"] = (
                    f"{type(error).__name__}: {error}"
                )
                self.runtime["carrier_execution_backend"] = (
                    "portable_custom_composites"
                )
        self.runtime.setdefault(
            "carrier_compiler_fallback_reason", "not required"
        )
        self.runtime["compiled_tensor_cores"] = compile_tensor_cores
        self.runtime["compiler_first_shape_seconds"] = (
            compiler_first_shape_seconds
        )
        self.runtime["compiler_steady_shape_seconds"] = (
            compiler_steady_shape_seconds
        )
        self.runtime["compiler_shape_cost_seconds"] = (
            compiler_shape_cost_seconds
        )
        self.runtime["compiled_tensor_core_policy"] = (
            "automatic_measured_faster"
            if config.compile_tensor_cores is None and compile_tensor_cores
            else "automatic_measured_rejected"
            if config.compile_tensor_cores is None
            else "explicit_enabled"
            if compile_tensor_cores
            else "explicit_disabled"
        )
        if self.document_batch_planner is not None:
            self.document_batch_planner.compiler_policy = (
                "on" if compile_tensor_cores else "off"
            )
            if (
                compile_tensor_cores
                and config.activation_calibration
                and config.document_cost_calibration
            ):
                self.document_batch_planner.cost_model = replace(
                    self.document_batch_planner.cost_model,
                    shape_compile_cost=compiler_shape_cost_seconds,
                    calibration_kind=(
                        self.document_batch_planner.cost_model.calibration_kind
                        + "_plus_measured_compile"
                    ),
                )
            self.document_batch_planner._group_cache.clear()
            self.runtime["document_cost_model_digest"] = (
                self.document_batch_planner.cost_model.digest
            )
        policy = OptimizerPolicy(
            learning_rate=config.learning_rate, weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            total_steps=max(config.total_steps, config.warmup_steps + 1),
            minimum_learning_rate_ratio=config.minimum_learning_rate_ratio,
        )
        self.optimizer = build_adamw(model, policy, fused=self.device.type == "cuda")
        self.scheduler = build_scheduler(self.optimizer, policy)
        self.pc_rasl: CognitiveResonantAdjointSurpriseLearner | None = None
        self.pc_rasl_critic_optimizer: torch.optim.Optimizer | None = None
        if config.progress_conditioned_rasl:
            self.pc_rasl = CognitiveResonantAdjointSurpriseLearner(
                model,
                progress_conditioned_rasl_configuration(model, config),
            ).to(self.device)
            critic_policy = OptimizerPolicy(
                learning_rate=config.pc_rasl_critic_learning_rate,
                weight_decay=config.weight_decay,
                warmup_steps=0,
                total_steps=max(1, config.total_steps),
                minimum_learning_rate_ratio=config.minimum_learning_rate_ratio,
            )
            self.pc_rasl_critic_optimizer = build_adamw(
                self.pc_rasl.critic,
                critic_policy,
                fused=self.device.type == "cuda",
            )
        self.scaler = torch.amp.GradScaler("cuda") if self.amp_dtype == torch.float16 else None
        self.state = MRCRATrainingState()
        self.cstm_coverage = CSTMCoverageState()
        self.execution_policy_history: list[dict[str, Any]] = [
            self._execution_policy_record(
                effective_step=0, reason="trainer initialization",
            )
        ]
        self._spectral_modules = tuple(
            module for module in model.modules() if isinstance(module, ResonantSpectralGLU)
        )
        if not self._spectral_modules:
            raise ValueError("MRCRA requires spectral activation modules")
        self._resumed = False
        self._last_runtime: MRCRARuntimeState | None = None
        self._last_ledger: ProvenanceLedger | None = None
        self._last_continuity_keys: tuple[str | None, ...] | None = None
        self._last_snapshot_step = -1
        self._last_snapshot_attempt_step = -1
        self._pending_first_hard_event_trace: HardEventTrace | None = None
        self._phase_update_proposal_logits: list[Tensor] = []
        self._phase_update_end_logits: list[Tensor] = []
        self._pc_rasl_pending_batches: list[CognitiveTrajectoryBatch] = []
        self._pc_rasl_finalized_batches: list[CognitiveTrajectoryBatch] = []
        self._pc_rasl_actor_gradients: dict[str, Tensor | None] = {}
        self._pc_rasl_step_metrics: dict[str, float] = {}
        self._cstm_auxiliary_gradients: dict[str, Tensor | None] = {}
        cstm_named_parameters = dict(self.model.named_parameters())
        predictor_registry = tuple(
            name
            for name in cstm_named_parameters
            if name.startswith("cstm_predictor.")
        )
        substrate_candidates = tuple(
            name
            for name in cstm_named_parameters
            if (
                name == "token_embedding.weight"
                or name.startswith("cognitive.")
            )
        )
        if (
            not predictor_registry
            or not any(
                name.startswith("cognitive.carrier.")
                for name in substrate_candidates
            )
            or not any(
                name.startswith("cognitive.")
                and not name.startswith("cognitive.carrier.")
                for name in substrate_candidates
            )
            or set(predictor_registry) & set(substrate_candidates)
        ):
            raise RuntimeError(
                "CSTM structural gradient registry is incomplete"
            )
        self._cstm_gradient_candidates = {
            "predictor": predictor_registry,
            "substrate": substrate_candidates,
        }
        self._cstm_reachable_parameter_names = {
            "predictor": predictor_registry,
            "substrate": (),
        }
        self.runtime["cstm_gradient_registry_schema_version"] = 1
        self.runtime["cstm_gradient_registry_predictor_count"] = len(
            predictor_registry
        )
        self.runtime["cstm_gradient_registry_substrate_count"] = 0
        self.runtime["cstm_gradient_registry_digest"] = (
            self._cstm_gradient_registry_digest()
        )
        self._cstm_step_metrics: dict[str, float] = {}
        self._activation_oom_retries = 0
        self.runtime["activation_oom_retries"] = 0
        self.last_step_metrics: dict[str, float] = {}
        self._prefetch_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="mrcra-data")
            if config.data_prefetch else None
        )
        self._prefetch_future: Future[PackedBatch] | None = None
        self._prefetch_request: tuple[int, int] | None = None
        self._prefetched_batch: PackedBatch | None = None

    @staticmethod
    def _packed_batch_state(batch: PackedBatch | None) -> dict[str, Any] | None:
        if batch is None:
            return None
        return {
            "input_ids": batch.input_ids,
            "labels": batch.labels,
            "target_byte_lengths": batch.target_byte_lengths,
            "segment_ids": batch.segment_ids,
            "target_segment_ids": batch.target_segment_ids,
            "source_uris_by_segment": batch.source_uris_by_segment,
            "continuity_keys": batch.continuity_keys,
        }

    @staticmethod
    def _packed_batch_from_state(value: dict[str, Any] | None) -> PackedBatch | None:
        if value is None:
            return None
        required = {
            "input_ids", "labels", "target_byte_lengths", "segment_ids",
            "target_segment_ids", "source_uris_by_segment", "continuity_keys",
        }
        if set(value) != required:
            raise ValueError("checkpoint prefetched batch schema is invalid")
        return PackedBatch(**value)

    @staticmethod
    def _trajectory_state(
        batch: CognitiveTrajectoryBatch,
    ) -> dict[str, Any]:
        return {
            name: getattr(batch, name)
            for name in CognitiveTrajectoryBatch.__dataclass_fields__
        }

    @staticmethod
    def _trajectory_from_state(
        value: dict[str, Any],
    ) -> CognitiveTrajectoryBatch:
        if not isinstance(value, dict):
            raise ValueError("checkpointed PC-RASL trajectory is malformed")
        try:
            return CognitiveTrajectoryBatch(**value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "checkpointed PC-RASL trajectory schema is invalid"
            ) from error

    def _pc_rasl_capture_due(self, next_step: int) -> bool:
        """Select bounded, deterministic behavior samples within each interval."""

        if self.pc_rasl is None:
            return False
        interval = self.config.learning_progress.observation_interval
        captures = min(
            self.config.pc_rasl_captures_per_observation,
            interval,
        )
        phase = ((next_step - 1) % interval) + 1
        positions = {
            ceil((index + 1) * interval / captures)
            for index in range(captures)
        }
        return phase in positions

    @torch.no_grad()
    def _capture_pc_rasl_trajectory(self, batch: PackedBatch) -> bool:
        """Retain compact pre-consequence behavior evidence for delayed credit."""

        if self.pc_rasl is None:
            return False
        segments = batch.segment_ids[0]
        valid = batch.loss_mask[0]
        best: tuple[int, int] | None = None
        start = 0
        while start < segments.numel():
            if not bool(valid[start]):
                start += 1
                continue
            end = start + 1
            while (
                end < segments.numel()
                and bool(valid[end])
                and segments[end] == segments[start]
            ):
                end += 1
            if end - start > self.pc_rasl.config.replay_burn_in_steps and (
                best is None or end - start > best[1] - best[0]
            ):
                best = (start, end)
            start = end
        if best is None:
            return False
        start, end = best
        end = min(end, start + self.config.pc_rasl_trajectory_length)
        retained = PackedBatch(
            batch.input_ids[:, start:end].detach().cpu().clone(),
            batch.labels[:, start:end].detach().cpu().clone(),
            batch.target_byte_lengths[:, start:end].detach().cpu().clone(),
            batch.segment_ids[:, start:end].detach().cpu().clone(),
            batch.target_segment_ids[:, start:end].detach().cpu().clone(),
            batch.source_uris_by_segment,
            tuple(None for _ in range(batch.input_ids.shape[0])),
        )
        if not bool(retained.loss_mask.all()):
            raise RuntimeError("PC-RASL retained a cross-document trajectory")
        local = retained.to(
            self.device, non_blocking=self.device.type == "cuda"
        )
        behavior_output = self.model(
            local.input_ids,
            segment_ids=local.segment_ids,
            boundary_classes=local.boundary_classes,
            source_uris=local.external_source_uris,
        )
        candidates, log_probability, sampled = build_language_candidate_set(
            behavior_output.logits,
            local.labels,
            candidate_count=self.config.pc_rasl_candidate_count,
        )
        mask = local.loss_mask
        dones = torch.zeros_like(mask)
        lengths = mask.sum(-1)
        for row, length in enumerate(lengths.tolist()):
            if length:
                dones[row, length - 1] = True
        cognitive = behavior_output.cognitive
        receipts = cognitive.action_receipts
        trajectory = CognitiveTrajectoryBatch(
            input_ids=local.input_ids,
            behavior_tokens=local.labels,
            candidate_token_ids=candidates,
            candidate_sampling_log_probabilities=log_probability,
            sampled_candidate_mask=sampled,
            rewards=torch.zeros(
                mask.shape,
                device=self.device,
                dtype=self.model.token_embedding.weight.dtype,
            ),
            dones=dones,
            mask=mask,
            segment_ids=local.segment_ids,
            boundary_classes=local.boundary_classes,
            behavior_candidate_logits=behavior_output.logits.gather(
                -1, candidates
            ),
            behavior_cognitive_features=cognitive.cognitive_features,
            behavior_workspace_features=cognitive.workspace_features,
            behavior_relation_features=cognitive.relation_features,
            behavior_relation_type_probabilities=(
                cognitive.relation_type_probabilities
            ),
            behavior_internal_actions=receipts.actions,
            behavior_internal_statuses=receipts.statuses,
            behavior_internal_mask=receipts.mask,
            reward_source="learning_progress",
            burn_in_steps=self.pc_rasl.config.replay_burn_in_steps,
        ).validated(
            vocabulary_size=self.model.vocabulary_size,
            width=self.model.config.cognitive.workspace_dim,
            maximum_candidates=self.config.pc_rasl_candidate_count,
        ).detached_cpu()
        self._pc_rasl_pending_batches.append(trajectory)
        maximum = self.config.pc_rasl_max_interval_trajectories
        if len(self._pc_rasl_pending_batches) > maximum:
            # Evenly retain the interval endpoints rather than silently keeping
            # only its most recent state.
            indices = [
                round(index * (len(self._pc_rasl_pending_batches) - 1) / (maximum - 1))
                for index in range(maximum)
            ] if maximum > 1 else [len(self._pc_rasl_pending_batches) - 1]
            self._pc_rasl_pending_batches = [
                self._pc_rasl_pending_batches[index] for index in indices
            ]
        self.state.pc_rasl_trajectories_captured += 1
        return True

    def _finalize_pc_rasl_interval(
        self, report: LearningProgressReport,
    ) -> None:
        if self.pc_rasl is None:
            return
        consequence = (
            report.pressure * self.config.pc_rasl_consequence_weight
        )
        self._pc_rasl_finalized_batches.extend(
            replace(
                batch,
                rewards=torch.full_like(batch.rewards, consequence),
            )
            for batch in self._pc_rasl_pending_batches
        )
        if self._pc_rasl_pending_batches:
            self.state.pc_rasl_updates_due += (
                self.config.pc_rasl_updates_per_observation
            )
        self._pc_rasl_pending_batches.clear()

    def _prepare_pc_rasl_gradients(self) -> None:
        """Train the detached critic and retain actor auxiliary gradients."""

        self._pc_rasl_actor_gradients.clear()
        updates_due_before = self.state.pc_rasl_updates_due
        self._pc_rasl_step_metrics = {
            "pc_rasl/replay_update_applied": 0.0,
            "pc_rasl/updates_due_before": float(updates_due_before),
            "pc_rasl/updates_due_after": float(updates_due_before),
        }
        learner = self.pc_rasl
        critic_optimizer = self.pc_rasl_critic_optimizer
        if (
            learner is None
            or critic_optimizer is None
            or self.state.pc_rasl_updates_due <= 0
        ):
            return
        ingest = self._pc_rasl_finalized_batches
        self._pc_rasl_finalized_batches = []
        for trajectory in ingest:
            signal = trajectory.rewards.abs().clamp_min(1e-6)
            learner.replay.add(
                trajectory,
                signal,
                torch.ones_like(signal),
                torch.ones_like(signal),
            )
        if not len(learner.replay):
            return
        sample = learner.replay.sample(
            self.config.pc_rasl_replay_batch_size,
            device=self.device,
        )
        critic_optimizer.zero_grad(set_to_none=True)
        losses = learner.compute_losses(sample.batch, update_calibration=True)
        losses.critic.total.backward()
        critic_gradient = clip_and_report_gradients(
            learner.critic,
            maximum_norm=learner.config.core.maximum_gradient_norm,
        )
        if not critic_gradient.finite:
            raise FloatingPointError("PC-RASL critic gradients became non-finite")
        critic_optimizer.step()
        learner.update_targets(actor_updated=False)
        actor_parameters = tuple(learner.actor.named_parameters())
        gradients = torch.autograd.grad(
            losses.actor.total,
            tuple(parameter for _, parameter in actor_parameters),
            allow_unused=True,
        )
        latest_guard_ce = (
            None
            if self.learning_progress is None
            else self.learning_progress.last_guard_ce
        )
        performance_guard_allowed = (
            False
            if latest_guard_ce is None
            else learner.performance_guard.allows(
                -latest_guard_ce,
                float(losses.actor.functional_cross_entropy.detach()),
            )
        )
        progress_ready_observation = max(
            self.config.learning_progress.warmup_observations,
            (
                self.config.learning_progress.baseline_min_observations
                + self.config.learning_progress.baseline_lag
            ),
        )
        warmup_required = (
            progress_ready_observation
            + self.config.pc_rasl_critic_warmup_observations
        )
        warmup_ready = (
            self.learning_progress is not None
            and self.learning_progress.ready
            and self.state.progress_observations >= warmup_required
        )
        actor_allowed = performance_guard_allowed and warmup_ready
        if actor_allowed:
            self._pc_rasl_actor_gradients = {
                name: gradient.detach() if gradient is not None else None
                for (name, _), gradient in zip(
                    actor_parameters, gradients, strict=True
                )
            }
        valid = sample.batch.loss_mask
        functional = losses.surprise.score.abs().mean(-1)
        learnability = losses.surprise.exploration_bonus.mean(-1)
        controllability = losses.surprise.controllability.mean(-1)
        priority_rows = (
            functional
            * learnability.clamp(0, 1)
            * controllability.clamp(0, 1)
        )
        priorities = (
            (priority_rows * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        ).clamp(1e-6, learner.config.core.replay_priority_cap)
        learner.replay.update_priorities(sample.indices, priorities.detach().cpu())
        self.state.pc_rasl_updates_due -= 1
        self.state.pc_rasl_replay_updates += 1
        self._pc_rasl_step_metrics = {
            "pc_rasl/replay_update_applied": 1.0,
            "pc_rasl/updates_due_before": float(updates_due_before),
            "pc_rasl/updates_due_after": float(
                self.state.pc_rasl_updates_due
            ),
            "pc_rasl/critic_loss": float(losses.critic.total.detach()),
            "pc_rasl/functional_cross_entropy": float(
                losses.actor.functional_cross_entropy.detach()
            ),
            "pc_rasl/internal_policy_loss": float(
                losses.actor.internal_policy.detach()
            ),
            "pc_rasl/progress_return_loss": float(
                losses.critic.progress_return.detach()
            ),
            "pc_rasl/internal_action_value_loss": float(
                losses.critic.internal_action_value.detach()
            ),
            "pc_rasl/mean_reward": float(
                sample.batch.rewards[valid].mean().detach()
            ),
            "pc_rasl/mean_absolute_surprise": float(
                losses.surprise.score.abs()[valid].mean().detach()
            ),
            "pc_rasl/critic_gradient_norm": float(
                critic_gradient.total_before_clip.detach()
            ),
            "pc_rasl/actor_auxiliary_ready": float(actor_allowed),
            "pc_rasl/performance_guard_allows_actor": float(
                performance_guard_allowed
            ),
            "pc_rasl/performance_guard_rejections": float(
                learner.performance_guard.rejections
            ),
            "pc_rasl/actor_warmup_ready": float(warmup_ready),
            "pc_rasl/replay_trajectories": float(len(learner.replay)),
            "pc_rasl/replay_transitions": float(learner.replay.transition_count),
            "pc_rasl/replay_storage_bytes": float(learner.replay.storage_bytes),
            "pc_rasl/behavior_evidence_bound": float(
                sample.batch.behavior_cognitive_features is not None
                and sample.batch.behavior_candidate_logits is not None
                and sample.batch.behavior_internal_actions is not None
            ),
        }

    def _merge_pc_rasl_gradients(self) -> dict[str, float]:
        if not self._pc_rasl_actor_gradients:
            return {
                "pc_rasl/actor_auxiliary_applied": 0.0,
                "pc_rasl/actor_auxiliary_gradient_norm_before": 0.0,
                "pc_rasl/actor_auxiliary_gradient_norm_after": 0.0,
                "pc_rasl/actor_task_gradient_norm": 0.0,
                "pc_rasl/actor_conflicting_subsystems": 0.0,
            }
        default = self.config.pc_rasl_cognitive_gradient_cap
        caps = {
            "carrier": self.config.pc_rasl_carrier_gradient_cap,
            "event": default,
            "output_bridge": default,
            "controller": self.config.pc_rasl_controller_gradient_cap,
            "workspace_router": default,
            "world_hypothesis": default,
            "memory": default,
            "other_cognition": default,
        }
        report = merge_auxiliary_gradients(
            self.model,
            self._pc_rasl_actor_gradients,
            caps,
        )
        self._pc_rasl_actor_gradients.clear()
        result = {
            "pc_rasl/actor_auxiliary_applied": float(report.applied),
            "pc_rasl/actor_auxiliary_gradient_norm_before": float(
                report.auxiliary_norm_before.cpu()
            ),
            "pc_rasl/actor_auxiliary_gradient_norm_after": float(
                report.auxiliary_norm_after.cpu()
            ),
            "pc_rasl/actor_task_gradient_norm": float(report.task_norm.cpu()),
            "pc_rasl/actor_conflicting_subsystems": float(
                len(report.conflicting_subsystems)
            ),
        }
        for subsystem, scale in report.subsystem_scales.items():
            result[f"pc_rasl/gradient_scale/{subsystem}"] = float(scale.cpu())
        return result

    def _cstm_objective_weight(self) -> float:
        """Return the causal token-scheduled CSTM coefficient for this update."""

        if not self.cstm_enabled:
            return 0.0
        progressed = self.state.tokens_seen - self.config.cstm_warmup_tokens
        if progressed < 0:
            return 0.0
        ramp = min(
            1.0,
            (progressed + max(1, self.config.context_length))
            / self.config.cstm_ramp_tokens,
        )
        return self.config.cstm_weight * ramp

    def _cstm_context_weight(self, batch: PackedBatch) -> float:
        """Count the exact valid horizon weight before TBPTT partitioning.

        Causal lifting emits scale ``s`` at deterministic document-local
        completion positions. Counting complete future blocks here makes the
        objective invariant to graph-release grouping without duplicating
        token-code lookup or Fourier target construction.
        """

        configured = self.model.cstm_predictor.config.horizon_blocks
        extras = configured[1:]
        segment_ids = batch.segment_ids[0].detach().cpu()
        length = segment_ids.numel()
        documents: list[tuple[int, int]] = []
        start = 0
        while start < length:
            end = start + 1
            while end < length and segment_ids[end] == segment_ids[start]:
                end += 1
            documents.append((start, end))
            start = end
        scale_count = self.model.config.carrier.scales
        total = 0.0
        for scale in range(scale_count):
            support = (
                2 ** (scale + 1)
                if scale < scale_count - 1
                else 2 ** (scale_count - 1)
            )
            horizons = (
                (1,)
                if not extras
                else (
                    1,
                    extras[(self.state.step + scale) % len(extras)],
                )
            )
            source_groups = [
                torch.arange(
                    document_start + support - 1,
                    document_end,
                    support,
                    dtype=torch.int64,
                    device=batch.loss_mask.device,
                )
                for document_start, document_end in documents
                if document_end - document_start >= support
            ]
            if not source_groups:
                continue
            valid = causal_spectral_target_mask(
                batch.loss_mask,
                batch.segment_ids,
                batch.target_segment_ids,
                torch.cat(source_groups),
                support=support,
                horizons=horizons,
            )
            horizon_weights = valid.new_tensor(
                (1.0,) + (0.5,) * (len(horizons) - 1),
                dtype=torch.float32,
            )
            total += float(
                (valid.to(torch.float32) * horizon_weights[None]).sum().cpu()
            )
        return total

    def _cstm_document_sampling_decision(
        self,
        plan: DocumentBatchPlan,
        *,
        duty_probability: float | None = None,
        seed_offset: int = 0,
    ) -> CSTMSamplingDecision:
        """Bind one sampled substrate VJP to a physical invocation and scale."""

        obligations: list[CSTMObligation] = []
        invocation = 0
        scale_count = self.model.config.carrier.scales
        configured = self.model.cstm_predictor.config.horizon_blocks
        extras = configured[1:]
        for cohort in plan.cohorts:
            authority = cohort.target_authority()
            physical_cursor = 0
            for physical in cohort.spans:
                invocation += 1
                for scale in range(scale_count):
                    support = (
                        2 ** (scale + 1)
                        if scale < scale_count - 1
                        else 2 ** (scale_count - 1)
                    )
                    horizons = (
                        (1,)
                        if not extras
                        else (
                            1,
                            extras[(self.state.step + scale) % len(extras)],
                        )
                    )
                    source_positions = torch.arange(
                        physical_cursor + support - 1,
                        physical_cursor + physical.padded_length,
                        support,
                        dtype=torch.int64,
                        device=authority.labels.device,
                    )
                    if not source_positions.numel():
                        continue
                    valid = causal_spectral_target_mask(
                        authority.loss_mask,
                        authority.segment_ids,
                        authority.target_segment_ids,
                        source_positions,
                        support=support,
                        horizons=horizons,
                    )
                    # This is schedule bookkeeping, not a differentiable
                    # tensor computation.  Count valid rows as exact integers
                    # on the active backend, then apply the exactly
                    # representable 1 and 1/2 horizon weights on the host.
                    # Constructing float64 on MPS is unsupported.
                    horizon_counts = tuple(
                        int(value)
                        for value in valid.sum(
                            dim=(0, 1), dtype=torch.int64
                        ).detach().cpu().tolist()
                    )
                    dense_weight = float(
                        horizon_counts[0]
                        + 0.5 * sum(horizon_counts[1:])
                    )
                    if dense_weight > 0:
                        obligations.append(
                            CSTMObligation(invocation, scale, dense_weight)
                        )
                physical_cursor += physical.padded_length
        return deterministic_cstm_sample(
            obligations,
            duty_probability=(
                self.config.cstm_sampling_duty_cycle
                if duty_probability is None else duty_probability
            ),
            uniform_mixture=self.config.cstm_sampling_uniform_mixture,
            seed=self.config.seed + seed_offset,
            optimizer_step=self.state.step,
            target_digest=plan.receipt.original_digest,
        )

    def _accumulate_cstm_gradients(
        self,
        loss: Tensor,
        *,
        objective_weight: float,
        gradient_divisor: int,
        authority: str = "all",
    ) -> None:
        """Retain CSTM gradients separately from exact next-token authority."""

        if objective_weight <= 0 or gradient_divisor <= 0:
            return
        if authority not in {"all", "substrate", "predictor"}:
            raise ValueError("unknown CSTM gradient authority")
        if loss.numel() != 1 or not bool(torch.isfinite(loss)):
            raise FloatingPointError("CSTM auxiliary loss became non-finite")
        model_parameters = dict(self.model.named_parameters())
        discovering_substrate = (
            authority in {"all", "substrate"}
            and not self._cstm_reachable_parameter_names["substrate"]
        )
        substrate_names = (
            self._cstm_gradient_candidates["substrate"]
            if discovering_substrate
            else self._cstm_reachable_parameter_names["substrate"]
        )
        parameter_names = (
            self._cstm_reachable_parameter_names["predictor"]
            if authority == "predictor"
            else substrate_names
            if authority == "substrate"
            else (
                self._cstm_reachable_parameter_names["predictor"]
                + substrate_names
            )
        )
        if (
            len(set(parameter_names)) != len(parameter_names)
            or any(name not in model_parameters for name in parameter_names)
        ):
            raise RuntimeError("CSTM gradient registry is corrupt")
        named_parameters = tuple(
            (name, model_parameters[name]) for name in parameter_names
        )
        if not named_parameters:
            raise RuntimeError("CSTM gradient authority selected no parameters")
        gradients = torch.autograd.grad(
            loss * (objective_weight / gradient_divisor),
            tuple(parameter for _, parameter in named_parameters),
            retain_graph=True,
            allow_unused=True,
        )
        if discovering_substrate:
            reachable_substrate = tuple(
                name
                for (name, _), gradient in zip(
                    named_parameters, gradients, strict=True
                )
                if (
                    gradient is not None
                    and name in self._cstm_gradient_candidates["substrate"]
                )
            )
            if (
                not any(
                    name.startswith("cognitive.carrier.")
                    for name in reachable_substrate
                )
                or not any(
                    name.startswith("cognitive.")
                    and not name.startswith("cognitive.carrier.")
                    for name in reachable_substrate
                )
            ):
                raise RuntimeError(
                    "CSTM dependency discovery omitted carrier or cognition"
                )
            self._cstm_reachable_parameter_names[
                "substrate"
            ] = reachable_substrate
            self.runtime[
                "cstm_gradient_registry_substrate_count"
            ] = len(reachable_substrate)
            self.runtime["cstm_gradient_registry_digest"] = (
                self._cstm_gradient_registry_digest()
            )
        for (name, _), gradient in zip(
            named_parameters, gradients, strict=True
        ):
            if gradient is None:
                continue
            detached = gradient.detach()
            prior = self._cstm_auxiliary_gradients.get(name)
            self._cstm_auxiliary_gradients[name] = (
                detached.clone() if prior is None else prior + detached
            )

    def _cstm_gradient_registry_digest(self) -> str:
        payload = {
            "schema_version": 1,
            "predictor": list(
                self._cstm_reachable_parameter_names["predictor"]
            ),
            "substrate": list(
                self._cstm_reachable_parameter_names["substrate"]
            ),
        }
        return sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def _cstm_gradient_registry_state(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "predictor": list(
                self._cstm_reachable_parameter_names["predictor"]
            ),
            "substrate": list(
                self._cstm_reachable_parameter_names["substrate"]
            ),
            "digest": self._cstm_gradient_registry_digest(),
        }

    def _load_cstm_gradient_registry_state(
        self, value: Mapping[str, object] | None,
    ) -> None:
        if value is None:
            # Older format-16 development checkpoints rediscover the exact
            # substrate dependency set on their next scheduled CSTM VJP.
            return
        try:
            if int(value["schema_version"]) != 1:
                raise ValueError("unknown CSTM gradient registry schema")
            predictor = tuple(str(name) for name in value["predictor"])
            substrate = tuple(str(name) for name in value["substrate"])
            if predictor != self._cstm_gradient_candidates["predictor"]:
                raise ValueError("CSTM predictor registry differs")
            candidate_set = set(
                self._cstm_gradient_candidates["substrate"]
            )
            if (
                len(set(substrate)) != len(substrate)
                or any(name not in candidate_set for name in substrate)
            ):
                raise ValueError("CSTM substrate registry differs")
            self._cstm_reachable_parameter_names = {
                "predictor": predictor,
                "substrate": substrate,
            }
            if value["digest"] != self._cstm_gradient_registry_digest():
                raise ValueError("CSTM gradient registry digest differs")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "CSTM checkpoint gradient registry is malformed"
            ) from error
        self.runtime["cstm_gradient_registry_substrate_count"] = len(
            substrate
        )
        self.runtime["cstm_gradient_registry_digest"] = (
            self._cstm_gradient_registry_digest()
        )

    def _merge_cstm_gradients(self) -> dict[str, float]:
        """Project conflicts and cap CSTM pressure relative to exact CE."""

        if not self._cstm_auxiliary_gradients:
            return {
                "cstm/auxiliary_applied": 0.0,
                "cstm/auxiliary_gradient_norm_before": 0.0,
                "cstm/auxiliary_gradient_norm_after": 0.0,
                "cstm/task_gradient_norm": 0.0,
                "cstm/conflicting_subsystems": 0.0,
            }
        cognition = self.config.cstm_cognitive_gradient_cap
        report = merge_auxiliary_gradients(
            self.model,
            self._cstm_auxiliary_gradients,
            {
                "carrier": self.config.cstm_carrier_gradient_cap,
                "event": cognition,
                "output_bridge": cognition,
                "controller": cognition,
                "workspace_router": cognition,
                "world_hypothesis": cognition,
                "memory": cognition,
                "other_cognition": cognition,
                "cstm_head": 0.0,
            },
            auxiliary_only_caps={
                "cstm_head": self.config.cstm_head_gradient_cap,
                # CSTM is an explicit learned auxiliary authority. Cognitive
                # parameters that are dormant under the current hard event
                # topology may therefore receive bounded pressure relative to
                # the *global* exact-CE norm; without this authority, the very
                # phase-forming modules CSTM is meant to prepare would be
                # permanently unable to become useful before CE already
                # traversed them. Carrier parameters remain fail-closed unless
                # exact CE supplies their direct task path.
                "event": cognition,
                "output_bridge": cognition,
                "controller": cognition,
                "workspace_router": cognition,
                "world_hypothesis": cognition,
                "memory": cognition,
                "other_cognition": cognition,
            },
        )
        self._cstm_auxiliary_gradients.clear()
        result = {
            "cstm/auxiliary_applied": float(report.applied),
            "cstm/auxiliary_gradient_norm_before": float(
                report.auxiliary_norm_before.cpu()
            ),
            "cstm/auxiliary_gradient_norm_after": float(
                report.auxiliary_norm_after.cpu()
            ),
            "cstm/task_gradient_norm": float(report.task_norm.cpu()),
            "cstm/conflicting_subsystems": float(
                len(report.conflicting_subsystems)
            ),
        }
        for subsystem, scale in report.subsystem_scales.items():
            result[f"cstm/gradient_scale/{subsystem}"] = float(scale.cpu())
        for subsystem, norm in report.subsystem_auxiliary_norms_before.items():
            result[
                f"cstm/auxiliary_gradient_norm_before/{subsystem}"
            ] = float(norm.cpu())
        for subsystem, norm in report.subsystem_auxiliary_norms_after.items():
            result[
                f"cstm/auxiliary_gradient_norm_after/{subsystem}"
            ] = float(norm.cpu())
        return result

    @staticmethod
    def _is_recoverable_out_of_memory(error: BaseException) -> bool:
        """Recognize allocator exhaustion without swallowing unrelated errors."""

        out_of_memory = getattr(torch, "OutOfMemoryError", ())
        if out_of_memory and isinstance(error, out_of_memory):
            return True
        if not isinstance(error, RuntimeError):
            return False
        message = str(error).lower()
        return any(fragment in message for fragment in (
            "out of memory",
            "mps backend out of memory",
            "cuda error: out of memory",
            "cannot allocate memory",
        ))

    def _clear_transient_device_memory(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def _advance_activation_policy_after_oom(self) -> bool:
        """Move exactly one rung toward safer recomputation and receipt it."""

        current = self.activation_execution_policy.resolved
        next_policy = {
            "retain": "selective",
            "selective": "whole_span",
            "whole_span": None,
        }[current]
        if next_policy is None:
            return False
        selective_scales = tuple(
            int(value)
            for value in self.runtime.get(
                "carrier_selective_checkpoint_scales", ()
            )
        )
        if next_policy == "selective" and not selective_scales:
            # A selective policy without a measured partition census is not a
            # real safety rung; advance directly to whole-span recomputation.
            next_policy = "whole_span"
        self.activation_execution_policy = replace(
            self.activation_execution_policy,
            resolved=next_policy,
            reason=(
                f"single recoverable pre-optimizer OOM fallback from {current}"
            ),
        )
        self.model.cognitive.carrier.configure_activation_execution(
            next_policy,
            selective_scales=(
                selective_scales if next_policy == "selective" else ()
            ),
        )
        if self.document_batch_planner is not None:
            self.document_batch_planner.activation_policy = next_policy
            self.document_batch_planner.activation_policy_token_limits = {
                name: (
                    self.config.document_batch_token_budget
                    if name == next_policy
                    else 0
                )
                for name in ("retain", "selective", "whole_span")
            }
            self.document_batch_planner._group_cache.clear()
        # The failed allocation may have occurred in a shape-local retained
        # invocation beneath a selective maximum-shape policy. Disable every
        # retained subpolicy for the single safe retry.
        self.activation_retain_physical_token_limit = 0
        self.activation_policy_token_limits = {
            name: (
                self.config.document_batch_token_budget
                if name == next_policy
                else 0
            )
            for name in ("retain", "selective", "whole_span")
        }
        self.runtime["carrier_retain_physical_token_limit"] = 0
        self.runtime["carrier_activation_physical_token_limits"] = dict(
            self.activation_policy_token_limits
        )
        self.runtime["carrier_shape_conditional_activation"] = False
        self.runtime["activation_execution_policy"] = (
            self.activation_execution_policy.to_dict()
        )
        self.runtime["activation_execution_policy_digest"] = (
            self.activation_execution_policy.digest
        )
        self.runtime["carrier_activation_checkpointing"] = (
            next_policy != "retain"
        )
        self.runtime["carrier_activation_checkpointing_policy"] = next_policy
        self._activation_oom_retries += 1
        self.runtime["activation_oom_retries"] = self._activation_oom_retries
        self.execution_policy_history.append(
            self._execution_policy_record(
                effective_step=self.state.step,
                reason=(
                    f"recoverable pre-optimizer OOM: {current} -> "
                    f"{next_policy}"
                ),
            )
        )
        return True

    def _materialize_prefetch(self) -> None:
        if self._prefetch_future is None:
            return
        self._prefetched_batch = self._prefetch_future.result()
        self._prefetch_future = None

    def _schedule_prefetch(self, batch_size: int, sequence_length: int) -> None:
        if (
            self._prefetch_executor is None or self._prefetch_future is not None
            or self._prefetched_batch is not None or sequence_length <= 0
        ):
            return
        self._prefetch_request = (batch_size, sequence_length)
        self._prefetch_future = self._prefetch_executor.submit(
            self.train_stream.next_batch, batch_size, sequence_length
        )

    def _next_training_batch(
        self, batch_size: int, sequence_length: int,
    ) -> PackedBatch:
        request = (batch_size, sequence_length)
        if self._prefetch_future is not None:
            if self._prefetch_request != request:
                raise RuntimeError(
                    "prefetched training request differs from the required context"
                )
            self._materialize_prefetch()
        if self._prefetched_batch is not None:
            batch = self._prefetched_batch
            self._prefetched_batch = None
            self._prefetch_request = None
            if batch.input_ids.shape != (batch_size, sequence_length):
                raise RuntimeError("checkpointed prefetched batch has the wrong shape")
            return batch
        return self.train_stream.next_batch(batch_size, sequence_length)

    def _autocast(self):
        return (
            nullcontext() if self.amp_dtype is None
            else torch.amp.autocast("cuda", dtype=self.amp_dtype)
        )

    def _language_statistics(
        self,
        output_latent: Tensor,
        labels: Tensor,
        target_byte_lengths: Tensor,
        mask: Tensor,
        head: nn.Linear,
        *,
        checkpoint_tiles: bool | None = None,
    ) -> TiledCrossEntropy:
        """Run exact full-softmax CE on the selected regular-work device.

        ``Tensor.to`` remains in the autograd graph, so an MPS loss contraction
        returns gradients to the canonical CPU latent and output-head
        parameters.  No model replica or independently optimized weight exists.
        """

        device = self.loss_device
        latent = output_latent.to(device)
        local_labels = labels.to(device)
        local_lengths = target_byte_lengths.to(device)
        local_mask = mask.to(device)
        weight = head.weight.to(device)
        bias = None if head.bias is None else head.bias.to(device)
        estimated_bytes = (
            local_mask.numel() * weight.shape[0] * latent.element_size()
        )
        backend = self._exact_loss_backend
        if backend == "mlx":
            try:
                from .mlx_backend import mlx_torch_exact_cross_entropy

                loss = mlx_torch_exact_cross_entropy(
                    latent,
                    weight,
                    local_labels,
                    bias,
                    mask=local_mask,
                    vocabulary_tile_size=self.config.vocabulary_tile_size,
                )
                from .mlx_backend import mlx_memory_statistics

                memory = mlx_memory_statistics()
                self.runtime["mlx_active_memory_bytes"] = memory[
                    "active_bytes"
                ]
                self.runtime["mlx_cache_memory_bytes"] = memory[
                    "cache_bytes"
                ]
                self.runtime["mlx_peak_memory_bytes"] = max(
                    int(self.runtime["mlx_peak_memory_bytes"]),
                    memory["peak_bytes"],
                )
                token_count = int(local_mask.sum())
                byte_count = int(local_lengths[local_mask].sum())
                return TiledCrossEntropy(
                    loss,
                    loss * token_count,
                    token_count,
                    byte_count,
                )
            except Exception as error:
                if self.config.exact_loss_backend == "mlx":
                    raise
                fallback = str(
                    self.runtime["mlx_exact_loss_fallback_backend"]
                )
                self._exact_loss_backend = fallback
                self.runtime["exact_loss_backend"] = fallback
                self.runtime["loss_projection"] = (
                    f"{fallback}_exact_full_softmax"
                )
                self.runtime["mlx_exact_loss_runtime_fallback"] = (
                    f"{type(error).__module__}.{type(error).__name__}: "
                    f"{' '.join(str(error).split())[:2048]}"
                )
                print(
                    "[MRCRA WARN] MLX exact loss failed; retrying the same "
                    f"batch with {fallback} and quarantining MLX loss for "
                    "the remainder of the run.",
                    flush=True,
                )
                return self._language_statistics(
                    output_latent,
                    labels,
                    target_byte_lengths,
                    mask,
                    head,
                    checkpoint_tiles=checkpoint_tiles,
                )
        if backend in {"cce_kahan_full_c", "cce_exact", "torch_compile"}:
            try:
                return exact_cut_cross_entropy(
                    latent,
                    local_labels,
                    local_lengths,
                    local_mask,
                    weight,
                    bias,
                    implementation=backend,
                )
            except Exception as error:
                if (
                    self.config.exact_loss_backend == "torch_compile"
                    or not _recoverable_external_cce_failure(error)
                ):
                    raise
                old_policy_digest = self._identity_digest(
                    self._identity()["execution"]
                )
                self._exact_loss_backend = "tiled"
                self.runtime["exact_loss_backend"] = "tiled"
                self.runtime["loss_projection"] = "tiled_exact_full_softmax"
                self.runtime["compiled_cce_runtime_quarantined"] = True
                self.runtime["compiled_cce_runtime_fallbacks"] = (
                    int(self.runtime.get("compiled_cce_runtime_fallbacks", 0))
                    + 1
                )
                compact_error = " ".join(str(error).split())
                self.runtime["compiled_cce_runtime_fallback_reason"] = (
                    f"{type(error).__module__}.{type(error).__name__}: "
                    f"{compact_error[:2048]}"
                )
                self.execution_policy_history.append(
                    self._execution_policy_record(
                        effective_step=self.state.step,
                        reason=(
                            "recoverable external exact-loss compiler failure: "
                            f"{backend} -> tiled"
                        ),
                        old_policy_digest=old_policy_digest,
                    )
                )
                print(
                    "[MRCRA WARN] Compiled exact CCE failed during lazy runtime "
                    "construction; retrying this batch with exact tiled CE and "
                    "quarantining compiled CCE for the remainder of the run.",
                    flush=True,
                )
                return exact_tiled_cross_entropy(
                    latent,
                    local_labels,
                    local_lengths,
                    local_mask,
                    weight,
                    bias,
                    vocabulary_tile_size=self.config.vocabulary_tile_size,
                    checkpoint_tiles=(
                        self._checkpoint_loss_tiles
                        if checkpoint_tiles is None else checkpoint_tiles
                    ),
                )
        if backend == "fused":
            if (
                self.config.maximum_fused_loss_bytes <= 0
                or estimated_bytes > self.config.maximum_fused_loss_bytes
            ):
                raise RuntimeError(
                    "configured fused exact loss exceeds its declared workspace budget"
                )
            return exact_fused_cross_entropy(
                latent, local_labels, local_lengths, local_mask, weight, bias,
            )
        if backend != "tiled":
            raise RuntimeError("trainer selected an unknown exact loss backend")
        return exact_tiled_cross_entropy(
            latent, local_labels, local_lengths, local_mask, weight, bias,
            vocabulary_tile_size=self.config.vocabulary_tile_size,
            checkpoint_tiles=(
                self._checkpoint_loss_tiles
                if checkpoint_tiles is None else checkpoint_tiles
            ),
        )

    def _identity(self) -> dict[str, Any]:
        return self._partition_identity(self._legacy_identity())

    def _legacy_identity(self) -> dict[str, Any]:
        """Return the monolithic identity used by formats 3--15.

        Keeping this constructor explicit is important: legacy migration first
        proves compatibility under the historical contract, then partitions
        the proven identity into format-16 semantic, optimization, execution,
        and observation authorities.
        """

        training = asdict(self.config)
        for key in (
            "output_dir",
            "total_tokens",
            "trackio_enabled",
            "show_dashboard",
            "allow_cstm_execution_upgrade",
        ):
            training.pop(key, None)
        source = {
            key: value for key, value in self.train_stream.source.state_dict().items()
            if key not in {"raw_rows_scanned", "documents_yielded"}
        }
        return {
            "model_config": asdict(self.model.config),
            "cstm_architecture": asdict(self.model.cstm_predictor.config),
            "parameter_count": self.model.parameter_count,
            "tokenizer": self.tokenizer.identity(),
            "training": training,
            "source": source,
            "evaluation": self.evaluation_identity,
            "progress_probe": (
                self.progress_probe_identity
                if self.config.progress_conditioned_rasl else None
            ),
        }

    def _partition_identity(self, legacy: Mapping[str, Any]) -> dict[str, Any]:
        """Partition a historical identity without weakening training authority."""

        value = deepcopy(dict(legacy))
        training = dict(value.pop("training"))
        model_config = deepcopy(value.pop("model_config"))
        carrier = model_config.get("carrier")
        if not isinstance(carrier, dict):
            raise ValueError("MRCRA model identity is missing its carrier contract")
        # Recompute/retention is an execution choice. It never changes the
        # carrier function, parameters, or optimizer state.
        carrier.pop("activation_checkpointing", None)

        execution_training = {
            name: training.pop(name)
            for name in sorted(_EXECUTION_TRAINING_FIELDS)
            if name in training
        }
        observation_training = {
            name: training.pop(name)
            for name in sorted(_OBSERVATION_TRAINING_FIELDS)
            if name in training
        }
        semantic = {
            "model_config": model_config,
            "cstm_architecture": value.pop("cstm_architecture"),
            "parameter_count": value.pop("parameter_count"),
            "tokenizer": value.pop("tokenizer"),
            "source": value.pop("source"),
            "evaluation": value.pop("evaluation"),
            "progress_probe": value.pop("progress_probe"),
        }
        if value:
            raise ValueError(
                "MRCRA identity contains unpartitioned semantic fields: "
                + ", ".join(sorted(value))
            )
        resolved_execution = {
            "activation": self.activation_execution_policy.to_dict(),
            "activation_digest": self.activation_execution_policy.digest,
            "selective_checkpoint_scales": list(
                self.runtime["carrier_selective_checkpoint_scales"]
            ),
            "retain_physical_token_limit": (
                self.activation_retain_physical_token_limit
            ),
            "shape_conditional_activation": bool(
                self.runtime["carrier_shape_conditional_activation"]
            ),
            "carrier_backend": self.runtime["carrier_execution_backend"],
            "carrier_compiler_backend": self.runtime[
                "carrier_compiler_backend"
            ],
            "carrier_checkpoint_granularity": self.runtime[
                "carrier_checkpoint_granularity"
            ],
            "compiled_tensor_cores": self.runtime["compiled_tensor_cores"],
            "exact_loss_backend": self._exact_loss_backend,
            "loss_device": str(self.loss_device),
            "loss_tile_checkpointing": self._checkpoint_loss_tiles,
            "document_bucket_lengths": list(
                self.runtime["document_bucket_lengths"]
            ),
        }
        equivalence_contract = {
            "schema_version": 1,
            "claim": (
                "execution policies may alter scheduling, recomputation, "
                "batch padding, device placement, or observation frequency "
                "but not valid-target order, exact objective weights, model "
                "state transitions, optimizer updates, or RNG consumption"
            ),
            "allowed_sections": ["execution", "observation"],
        }
        return {
            "schema_version": 1,
            "semantic": semantic,
            "optimization": {"training": training},
            "execution": {
                "training": execution_training,
                "resolved": resolved_execution,
                "equivalence_contract": equivalence_contract,
            },
            "observation": {"training": observation_training},
        }

    @staticmethod
    def _identity_digest(value: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def _execution_policy_record(
        self,
        *,
        effective_step: int,
        reason: str,
        old_policy_digest: str | None = None,
    ) -> dict[str, Any]:
        execution = self._identity()["execution"]
        new_policy_digest = self._identity_digest(execution)
        if old_policy_digest is None:
            history = getattr(self, "execution_policy_history", ())
            old_policy_digest = (
                history[-1]["execution_digest"]
                if history else new_policy_digest
            )
        return {
            "schema_version": 1,
            "effective_step": int(effective_step),
            "reason": str(reason),
            "old_policy_digest": old_policy_digest,
            "new_policy_digest": new_policy_digest,
            "equivalence_receipt_digest": (
                self._equivalence_receipt_digest(
                    old_policy_digest,
                    new_policy_digest,
                    execution,
                )
            ),
            "execution_digest": new_policy_digest,
            "execution": execution,
        }

    @classmethod
    def _equivalence_receipt_digest(
        cls,
        old_policy_digest: str,
        new_policy_digest: str,
        execution: Mapping[str, Any],
    ) -> str:
        if (
            len(old_policy_digest) != 64
            or len(new_policy_digest) != 64
        ):
            raise ValueError("execution transition digests are malformed")
        return cls._identity_digest({
            "schema_version": 1,
            "old_policy_digest": old_policy_digest,
            "new_policy_digest": new_policy_digest,
            "equivalence_contract": execution.get(
                "equivalence_contract"
            ),
        })

    @property
    def evaluation_identity(self) -> dict[str, int | str]:
        return {
            "batch_count": len(self.evaluation_batches),
            "sha256": self._batch_digest(self.evaluation_batches),
        }

    @property
    def progress_probe_identity(self) -> dict[str, int | str]:
        return {
            "batch_count": len(self.progress_probe_batches),
            "sha256": self._batch_digest(self.progress_probe_batches),
        }

    @staticmethod
    def _batch_digest(batches: Sequence[PackedBatch]) -> str:
        digest = sha256()
        for batch in batches:
            for tensor in (
                batch.input_ids, batch.labels, batch.target_byte_lengths,
                batch.segment_ids, batch.target_segment_ids,
            ):
                value = tensor.detach().contiguous().cpu()
                digest.update(str(value.dtype).encode("ascii"))
                digest.update(str(tuple(value.shape)).encode("ascii"))
                digest.update(value.numpy().tobytes())
            digest.update(repr(batch.source_uris_by_segment).encode("utf-8"))
            digest.update(repr(batch.continuity_keys).encode("utf-8"))
        return digest.hexdigest()

    def _checkpoint_payload(self) -> dict[str, Any]:
        # The stream state is post-prefetch.  Persist the materialized batch with
        # it so resume consumes exactly the same tokens instead of skipping them.
        self._materialize_prefetch()
        payload: dict[str, Any] = {
            "format_version": MRCRA_TRAINING_FORMAT_VERSION,
            "identity": self._identity(),
            "execution_policy": self._identity()["execution"],
            "execution_policy_history": deepcopy(
                self.execution_policy_history
            ),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": None if self.scaler is None else self.scaler.state_dict(),
            "training_state": asdict(self.state),
            "train_stream": self.train_stream.state_dict(),
            "prefetched_batch": self._packed_batch_state(self._prefetched_batch),
            "prefetch_request": self._prefetch_request,
            "torch_rng": torch.random.get_rng_state(),
            "accelerator_rng": None,
            "last_runtime": None if self._last_runtime is None else runtime_state_dict(self._last_runtime),
            "last_provenance": None if self._last_ledger is None else self._last_ledger.state_dict(),
            "last_continuity_keys": self._last_continuity_keys,
            "learning_progress": (
                None
                if self.learning_progress is None
                else self.learning_progress.state_dict()
            ),
            "cstm_sampling": (
                self.cstm_coverage.state_dict()
                if (
                    self.cstm_enabled
                    and self.config.cstm_execution == "sampled"
                )
                else None
            ),
            "cstm_gradient_registry": (
                self._cstm_gradient_registry_state()
                if self.cstm_enabled else None
            ),
            "pc_rasl": (
                None
                if self.pc_rasl is None
                else {
                    "config": asdict(self.pc_rasl.config),
                    "critic": self.pc_rasl.critic.state_dict(),
                    "target_critic": self.pc_rasl.target_critic.state_dict(),
                    "calibrator": self.pc_rasl.calibrator.state_dict(),
                    "replay": self.pc_rasl.replay.state_dict(),
                    "performance_guard": (
                        self.pc_rasl.performance_guard.state_dict()
                    ),
                    "critic_optimizer": (
                        self.pc_rasl_critic_optimizer.state_dict()
                        if self.pc_rasl_critic_optimizer is not None else None
                    ),
                    "pending_batches": [
                        self._trajectory_state(batch)
                        for batch in self._pc_rasl_pending_batches
                    ],
                    "finalized_batches": [
                        self._trajectory_state(batch)
                        for batch in self._pc_rasl_finalized_batches
                    ],
                }
            ),
        }
        if self.device.type == "cuda":
            payload["accelerator_rng"] = torch.cuda.get_rng_state(self.device)
        elif self.device.type == "mps":
            payload["accelerator_rng"] = torch.mps.get_rng_state()
        return payload

    def save_checkpoint(self, *, phase_transition: bool = False) -> Path:
        directory = Path(self.config.output_dir) / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / (
            f"phase-transition-first-event-step-{self.state.step:07d}.pt"
            if phase_transition else f"step-{self.state.step:07d}.pt"
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="mrcra-checkpoint-", suffix=".tmp", dir=directory
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with temporary.open("wb") as handle:
                torch.save(self._checkpoint_payload(), handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        if not phase_transition:
            checkpoints = sorted(directory.glob("step-*.pt"))
            for obsolete in checkpoints[:-self.config.keep_checkpoints]:
                obsolete.unlink()
        latest = directory / "latest.json"
        latest_descriptor, latest_temporary_name = tempfile.mkstemp(
            prefix="mrcra-latest-", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(
                latest_descriptor, "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {
                        "checkpoint": destination.name,
                        "step": self.state.step,
                    },
                    handle,
                    indent=2,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(latest_temporary_name, latest)
        finally:
            Path(latest_temporary_name).unlink(missing_ok=True)
        return destination

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=True)
        saved_format = payload.get("format_version")
        if saved_format not in {
            MRCRA_TRAINING_FORMAT_VERSION, *LEGACY_MRCRA_TRAINING_FORMAT_VERSIONS,
        }:
            raise ValueError("unsupported MRCRA training checkpoint format")
        expected_identity = self._identity()
        saved_identity = deepcopy(payload.get("identity"))
        retiring_pc_rasl = False
        upgrading_legacy_cstm = False
        legacy_cstm_execution = None
        if (
            saved_format == MRCRA_TRAINING_FORMAT_VERSION
            and isinstance(saved_identity, dict)
        ):
            try:
                retiring_pc_rasl = (
                    bool(
                        saved_identity["optimization"]["training"].get(
                            "progress_conditioned_rasl", False
                        )
                    )
                    and not expected_identity["optimization"]["training"][
                        "progress_conditioned_rasl"
                    ]
                )
            except (KeyError, TypeError, AttributeError):
                raise ValueError(
                    "MRCRA training checkpoint identity is malformed"
                ) from None
        elif isinstance(saved_identity, dict):
            try:
                legacy_expected_identity = self._legacy_identity()
                retiring_pc_rasl = (
                    saved_format in LEGACY_MRCRA_TRAINING_FORMAT_VERSIONS
                    and bool(
                        saved_identity["training"].get(
                            "progress_conditioned_rasl", False
                        )
                    )
                    and not legacy_expected_identity["training"][
                        "progress_conditioned_rasl"
                    ]
                )
                # Format 7 predates the execution-only tensor-core compiler
                # policy. It changes neither weights nor mathematical state,
                # so an absent field inherits the explicit current policy.
                saved_identity["training"].setdefault(
                    "compile_tensor_cores",
                    legacy_expected_identity["training"][
                        "compile_tensor_cores"
                    ],
                )
            except (KeyError, TypeError, AttributeError):
                raise ValueError("MRCRA training checkpoint identity is malformed") from None
        if saved_format in LEGACY_MRCRA_TRAINING_FORMAT_VERSIONS and isinstance(saved_identity, dict):
            legacy_expected_identity = self._legacy_identity()
            try:
                legacy_had_active_cstm = bool(
                    saved_identity["training"].get("cstm_enabled", False)
                    and saved_identity["training"].get(
                        "integrated_cognitive_path", False
                    )
                )
                legacy_cstm_execution = saved_identity["training"].get(
                    "cstm_execution", "legacy_dense"
                )
                if (
                    legacy_had_active_cstm
                    and self.config.cstm_execution == "sampled"
                    and not self.config.allow_cstm_execution_upgrade
                ):
                    raise ValueError(
                        "legacy dense CSTM checkpoint requires "
                        "cstm_execution='legacy_dense' or explicit "
                        "allow_cstm_execution_upgrade=True"
                    )
                upgrading_legacy_cstm = bool(
                    legacy_had_active_cstm
                    and legacy_cstm_execution == "legacy_dense"
                    and self.config.cstm_execution == "sampled"
                    and self.config.allow_cstm_execution_upgrade
                )
                saved_identity["cstm_architecture"] = legacy_expected_identity[
                    "cstm_architecture"
                ]
                saved_cognitive = saved_identity["model_config"]["cognitive"]
                current_cognitive = legacy_expected_identity[
                    "model_config"
                ]["cognitive"]
                saved_identity["model_config"]["carrier"][
                    "activation_checkpointing"
                ] = legacy_expected_identity["model_config"]["carrier"][
                    "activation_checkpointing"
                ]
                for name in _V4_COGNITIVE_DEFAULT_FIELDS:
                    saved_cognitive[name] = current_cognitive[name]
                saved_training = saved_identity["training"]
                current_training = legacy_expected_identity["training"]
                for name in (
                    "evaluation_interval", "evaluation_batches",
                    "require_evaluation", "event_compute_regularization_weight",
                    "checkpoint_tiles", "maximum_retained_loss_bytes",
                    "exact_loss_backend",
                    "data_prefetch", "phase_transition_telemetry",
                    "phase_transition_ablation",
                    "phase_transition_ablation_batches",
                    "proposal_slope_ema_decay",
                    "low_clip_coefficient_threshold",
                    "low_clip_coefficient_patience",
                    "progress_conditioned_rasl",
                    "progress_probe_batches",
                    "progress_probe_length",
                    "learning_progress",
                    "pc_rasl_trajectory_length",
                    "pc_rasl_candidate_count",
                    "pc_rasl_replay_batch_size",
                    "pc_rasl_max_interval_trajectories",
                    "pc_rasl_captures_per_observation",
                    "pc_rasl_updates_per_observation",
                    "pc_rasl_critic_warmup_observations",
                    "pc_rasl_consequence_weight",
                    "pc_rasl_critic_learning_rate",
                    "pc_rasl_carrier_gradient_cap",
                    "pc_rasl_cognitive_gradient_cap",
                    "pc_rasl_controller_gradient_cap",
                    "cstm_enabled",
                    "cstm_weight",
                    "cstm_warmup_tokens",
                    "cstm_ramp_tokens",
                    "cstm_carrier_gradient_cap",
                    "cstm_cognitive_gradient_cap",
                    "cstm_head_gradient_cap",
                    "cstm_execution",
                    "cstm_sampling_duty_cycle",
                    "cstm_sampling_uniform_mixture",
                    "cstm_max_substrate_vjps",
                    "cstm_target_participation_budget",
                    "cstm_predictor_update_interval",
                    "cstm_maximum_coverage_gap",
                    "document_static_batching",
                    "document_bucket_lengths",
                    "document_batch_token_budget",
                    "document_grouping_policy",
                    "document_plan_cache_capacity",
                    "trackio_remote_log_interval",
                ):
                    saved_training[name] = current_training[name]
                if (
                    saved_format == 10
                    and saved_identity["model_config"]["carrier"]["model_dim"] == 20
                    and saved_training.get("cognitive_stride") == 64
                    and current_training.get("cognitive_stride") == 128
                ):
                    # The format-11 ultralight production policy deliberately
                    # advances the causal cognition cadence from one 64-token
                    # event chunk to two. Preserve learned/optimizer state while
                    # binding every subsequent update to the new contract.
                    saved_training["cognitive_stride"] = 128
                # Pre-v6 checkpoints did not bind retained evaluation data.
                # Migration attaches the explicitly supplied current retained
                # set; the digest is subsequently enforced on every resume.
                saved_identity["evaluation"] = legacy_expected_identity["evaluation"]
                saved_identity["progress_probe"] = legacy_expected_identity[
                    "progress_probe"
                ]
                saved_identity["parameter_count"] = legacy_expected_identity[
                    "parameter_count"
                ]
            except (KeyError, TypeError):
                raise ValueError("legacy MRCRA training checkpoint identity is malformed") from None
            try:
                saved_identity = self._partition_identity(saved_identity)
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    "legacy MRCRA training checkpoint identity is malformed"
                ) from None
        if not isinstance(saved_identity, dict) or set(saved_identity) != {
            "schema_version", "semantic", "optimization", "execution",
            "observation",
        }:
            raise ValueError("MRCRA training checkpoint identity is malformed")
        if (
            saved_identity["schema_version"] != expected_identity["schema_version"]
            or saved_identity["semantic"] != expected_identity["semantic"]
            or saved_identity["optimization"] != expected_identity["optimization"]
        ):
            raise ValueError("MRCRA checkpoint model, tokenizer, data, or training contract differs")
        current_execution = expected_identity["execution"]
        current_execution_digest = self._identity_digest(current_execution)
        if saved_format == MRCRA_TRAINING_FORMAT_VERSION:
            saved_execution = saved_identity["execution"]
            if payload.get("execution_policy") != saved_execution:
                raise ValueError(
                    "MRCRA checkpoint execution policy receipt differs from "
                    "its identity"
                )
            saved_history = deepcopy(payload.get("execution_policy_history"))
            if not isinstance(saved_history, list) or not saved_history:
                raise ValueError(
                    "MRCRA checkpoint is missing execution policy history"
                )
            previous_step = -1
            previous_digest = None
            for record in saved_history:
                if (
                    not isinstance(record, dict)
                    or record.get("schema_version") != 1
                    or not isinstance(record.get("effective_step"), int)
                    or record["effective_step"] < previous_step
                    or record.get("execution_digest")
                    != self._identity_digest(record.get("execution", {}))
                ):
                    raise ValueError(
                        "MRCRA checkpoint execution policy history is malformed"
                    )
                transition_fields = (
                    "old_policy_digest",
                    "new_policy_digest",
                    "equivalence_receipt_digest",
                )
                present = tuple(
                    field in record for field in transition_fields
                )
                if any(present):
                    if (
                        not all(present)
                        or record["new_policy_digest"]
                        != record["execution_digest"]
                        or (
                            previous_digest is not None
                            and record["old_policy_digest"]
                            != previous_digest
                        )
                        or record["equivalence_receipt_digest"]
                        != self._equivalence_receipt_digest(
                            record["old_policy_digest"],
                            record["new_policy_digest"],
                            record["execution"],
                        )
                    ):
                        raise ValueError(
                            "MRCRA checkpoint execution transition receipt "
                            "is malformed"
                        )
                previous_step = record["effective_step"]
                previous_digest = record["execution_digest"]
            if (
                saved_history[-1]["execution_digest"]
                != self._identity_digest(saved_execution)
            ):
                raise ValueError(
                    "MRCRA checkpoint execution policy history does not end "
                    "at its active execution policy"
                )
            self.execution_policy_history = saved_history
        else:
            legacy_execution_digest = self._identity_digest(
                saved_identity["execution"]
            )
            self.execution_policy_history = [
                {
                    "schema_version": 1,
                    "effective_step": int(
                        payload.get("training_state", {}).get("step", 0)
                    ),
                    "reason": (
                        f"migrated legacy format-{saved_format} execution "
                        "contract"
                    ),
                    "old_policy_digest": legacy_execution_digest,
                    "new_policy_digest": legacy_execution_digest,
                    "equivalence_receipt_digest": (
                        self._equivalence_receipt_digest(
                            legacy_execution_digest,
                            legacy_execution_digest,
                            saved_identity["execution"],
                        )
                    ),
                    "execution_digest": legacy_execution_digest,
                    "execution": saved_identity["execution"],
                }
            ]
            if upgrading_legacy_cstm:
                # Sampling is an explicitly authorized, one-way estimator
                # migration rather than an implicit normalization.  Record
                # both optimization authorities even when the execution-only
                # digest is otherwise unchanged.
                self.execution_policy_history.append({
                    **self._execution_policy_record(
                        effective_step=int(
                            payload.get("training_state", {}).get("step", 0)
                        ),
                        reason=(
                            f"explicit format-{saved_format} CSTM estimator "
                            "upgrade from legacy_dense to sampled"
                        ),
                    ),
                    "cstm_execution_transition": {
                        "from": legacy_cstm_execution,
                        "to": self.config.cstm_execution,
                    },
                    "optimization_digest_after": self._identity_digest(
                        expected_identity["optimization"]
                    ),
                })
        if (
            self.execution_policy_history[-1]["execution_digest"]
            != current_execution_digest
        ):
            self.execution_policy_history.append(
                self._execution_policy_record(
                    effective_step=int(
                        payload.get("training_state", {}).get("step", 0)
                    ),
                    reason=(
                        "resume-time execution-policy change accepted under "
                        "format-16 equivalence contract"
                    ),
                )
            )
        saved_model = payload["model"]
        if saved_format in LEGACY_MRCRA_TRAINING_FORMAT_VERSIONS:
            current_model = self.model.state_dict()
            migrated_model = {}
            system_weight_suffixes = (
                "cognitive.controller.input.weight",
                "cognitive.metacognitive_router.trunk.0.weight",
                "cognitive.external_action_policy.trunk.0.weight",
                "cognitive.self_model_projection.weight",
            )
            for name, current in current_model.items():
                if name not in saved_model:
                    if name.startswith("cstm_predictor."):
                        migrated_model[name] = current
                        continue
                    raise ValueError(f"legacy MRCRA checkpoint is missing model tensor {name}")
                saved = saved_model[name]
                if saved.shape == current.shape:
                    migrated_model[name] = saved
                elif (
                    name.endswith(system_weight_suffixes)
                    and saved.ndim == current.ndim == 2
                    and saved.shape[0] == current.shape[0]
                    and saved.shape[1] + 5 * self.model.config.cognitive.system_action_channels
                    == current.shape[1]
                ):
                    action_count = self.model.config.cognitive.system_action_channels
                    new_system_width = (
                        self.model.config.cognitive.modality_count
                        + 8 * action_count + 4
                        + self.model.config.cognitive.calibration_regimes
                    )
                    prefix = current.shape[1] - new_system_width
                    old_prefix_end = (
                        prefix + self.model.config.cognitive.modality_count
                        + 3 * action_count
                    )
                    expanded = current.clone()
                    expanded[:, :old_prefix_end] = saved[:, :old_prefix_end]
                    expanded[:, old_prefix_end + 5 * action_count:] = saved[:, old_prefix_end:]
                    migrated_model[name] = expanded
                elif (
                    name.endswith("cognitive.controller.action_head.weight")
                    or name.endswith("cognitive.controller.action_head.bias")
                ) and saved.shape[0] == int(InternalAction.RECONSTRUCT_LOCAL):
                    expanded = torch.zeros_like(current)
                    expanded[: saved.shape[0]] = saved
                    migrated_model[name] = expanded
                else:
                    raise ValueError(
                        f"legacy MRCRA tensor {name} has unsupported shape "
                        f"{tuple(saved.shape)}; expected {tuple(current.shape)}"
                    )
            saved_model = migrated_model
        self.model.load_state_dict(saved_model)
        saved_optimizer = payload["optimizer"]
        if saved_format in LEGACY_MRCRA_TRAINING_FORMAT_VERSIONS:
            saved_optimizer = deepcopy(saved_optimizer)
            current_optimizer = self.optimizer.state_dict()
            parameter_names = {id(parameter): name for name, parameter in self.model.named_parameters()}
            for live_group, serialized_group in zip(
                self.optimizer.param_groups, current_optimizer["param_groups"], strict=True,
            ):
                for parameter, parameter_id in zip(
                    live_group["params"], serialized_group["params"], strict=True,
                ):
                    name = parameter_names.get(id(parameter), "")
                    if not (
                        name.endswith("cognitive.controller.action_head.weight")
                        or name.endswith("cognitive.controller.action_head.bias")
                    ):
                        continue
                    legacy_state = saved_optimizer["state"].get(parameter_id, {})
                    for state_name, state_value in tuple(legacy_state.items()):
                        if (
                            isinstance(state_value, Tensor) and state_value.ndim
                            and state_value.shape != parameter.shape
                            and state_value.shape[0] == int(InternalAction.RECONSTRUCT_LOCAL)
                        ):
                            expanded = state_value.new_zeros(parameter.shape)
                            expanded[: state_value.shape[0]] = state_value
                            legacy_state[state_name] = expanded
            saved_optimizer["param_groups"] = current_optimizer["param_groups"]
        self.optimizer.load_state_dict(saved_optimizer)
        self.scheduler.load_state_dict(payload["scheduler"])
        if self.scaler is not None:
            if payload.get("scaler") is None:
                raise ValueError("FP16 resume checkpoint is missing its gradient scaler")
            self.scaler.load_state_dict(payload["scaler"])
        training_state = deepcopy(payload["training_state"])
        for name, default in (
            ("last_progress_observation_step", 0),
            ("last_progress_pressure", 0.0),
            ("progress_observations", 0),
            ("pc_rasl_updates_due", 0),
            ("pc_rasl_trajectories_captured", 0),
            ("pc_rasl_replay_updates", 0),
        ):
            training_state.setdefault(name, default)
        self.state = MRCRATrainingState(**training_state)
        cstm_sampling_state = payload.get("cstm_sampling")
        expects_sampled_cstm = (
            self.cstm_enabled and self.config.cstm_execution == "sampled"
        )
        if saved_format == MRCRA_TRAINING_FORMAT_VERSION:
            if expects_sampled_cstm:
                if cstm_sampling_state is None:
                    raise ValueError(
                        "sampled CSTM checkpoint is missing coverage state"
                    )
                self.cstm_coverage = CSTMCoverageState.from_state_dict(
                    cstm_sampling_state
                )
            elif cstm_sampling_state is not None:
                raise ValueError(
                    "checkpoint contains sampled CSTM state but sampled "
                    "execution is disabled"
                )
        else:
            # A legacy-dense resume has no sampled schedule. An explicitly
            # authorized one-way upgrade begins deterministic coverage from
            # the restored optimizer step.
            self.cstm_coverage = CSTMCoverageState()
        if self.cstm_enabled:
            self._load_cstm_gradient_registry_state(
                payload.get("cstm_gradient_registry")
            )
        elif payload.get("cstm_gradient_registry") is not None:
            raise ValueError(
                "checkpoint contains a CSTM gradient registry while CSTM is disabled"
            )
        if retiring_pc_rasl:
            self.state.last_progress_observation_step = 0
            self.state.last_progress_pressure = 0.0
            self.state.progress_observations = 0
            self.state.pc_rasl_updates_due = 0
            self.state.pc_rasl_trajectories_captured = 0
            self.state.pc_rasl_replay_updates = 0
        if self.state.tokens_seen >= self.config.total_tokens:
            raise ValueError("resumed checkpoint already exhausted the token budget")
        self.train_stream.load_state_dict(payload["train_stream"])
        self._prefetched_batch = self._packed_batch_from_state(
            payload.get("prefetched_batch")
        )
        request = payload.get("prefetch_request")
        self._prefetch_request = None if request is None else tuple(request)
        if (
            self._prefetched_batch is not None
            and self._prefetch_request
            != tuple(self._prefetched_batch.input_ids.shape)
        ):
            raise ValueError("checkpoint prefetched request and batch shape differ")
        torch.random.set_rng_state(payload["torch_rng"].cpu())
        accelerator_rng = payload.get("accelerator_rng")
        if accelerator_rng is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state(accelerator_rng, self.device)
        elif accelerator_rng is not None and self.device.type == "mps":
            torch.mps.set_rng_state(accelerator_rng.cpu())
        if payload.get("last_runtime") is not None:
            self._last_runtime = runtime_state_from_dict(
                payload["last_runtime"], cognitive=self.model.config.cognitive,
            )
        if payload.get("last_provenance") is not None:
            ledger = ProvenanceLedger()
            ledger.load_state_dict(payload["last_provenance"])
            self._last_ledger = ledger
        saved_keys = payload.get("last_continuity_keys")
        self._last_continuity_keys = (
            None if saved_keys is None else tuple(saved_keys)
        )
        progress_state = payload.get("learning_progress")
        legacy_pc_reset = (
            saved_format in PRE_BEHAVIOR_EVIDENCE_FORMAT_VERSIONS
        )
        if self.learning_progress is None:
            if progress_state is not None and not retiring_pc_rasl:
                raise ValueError(
                    "checkpoint contains learning-progress state but PC-RASL is disabled"
                )
        elif legacy_pc_reset:
            # Pre-v10 replay did not bind the exact behavior policy and
            # cognitive receipts before the delayed outcome. Retaining only
            # part of that causal chain would create a false exact-resume
            # claim, so the complete PC-RASL authority restarts warmup.
            self.state.last_progress_observation_step = 0
            self.state.last_progress_pressure = 0.0
            self.state.progress_observations = 0
        elif progress_state is None:
            raise ValueError(
                "checkpoint is missing required learning-progress state"
            )
        else:
            self.learning_progress.load_state_dict(progress_state)
        pc_state = payload.get("pc_rasl")
        if self.pc_rasl is None:
            if pc_state is not None and not retiring_pc_rasl:
                raise ValueError(
                    "checkpoint contains PC-RASL state but the learner is disabled"
                )
        elif legacy_pc_reset:
            self._pc_rasl_pending_batches.clear()
            self._pc_rasl_finalized_batches.clear()
        elif pc_state is None:
            raise ValueError("checkpoint is missing required PC-RASL state")
        else:
            if pc_state.get("config") != asdict(self.pc_rasl.config):
                raise ValueError("checkpoint PC-RASL configuration differs")
            self.pc_rasl.critic.load_state_dict(pc_state["critic"])
            self.pc_rasl.target_critic.load_state_dict(pc_state["target_critic"])
            self.pc_rasl.calibrator.load_state_dict(pc_state["calibrator"])
            self.pc_rasl.replay.load_state_dict(pc_state["replay"])
            self.pc_rasl.performance_guard.load_state_dict(
                pc_state["performance_guard"]
            )
            if self.pc_rasl_critic_optimizer is None:
                raise RuntimeError("PC-RASL critic optimizer was not initialized")
            self.pc_rasl_critic_optimizer.load_state_dict(
                pc_state["critic_optimizer"]
            )
            self._pc_rasl_pending_batches = [
                self._trajectory_from_state(value)
                for value in pc_state["pending_batches"]
            ]
            self._pc_rasl_finalized_batches = [
                self._trajectory_from_state(value)
                for value in pc_state["finalized_batches"]
            ]
            if (
                saved_format == 10
                and self._pc_rasl_finalized_batches
                and self.state.pc_rasl_updates_due == 0
            ):
                # Format 10 already binds exact behavior evidence, but predates
                # the consequence-driven update budget. Give its outstanding
                # finalized consequence one bounded update instead of either
                # discarding it or replaying it on every future optimizer step.
                self.state.pc_rasl_updates_due = (
                    self.config.pc_rasl_updates_per_observation
                )
            for batch in (
                *self._pc_rasl_pending_batches,
                *self._pc_rasl_finalized_batches,
            ):
                batch.validated(
                    vocabulary_size=self.model.vocabulary_size,
                    width=self.model.config.cognitive.workspace_dim,
                    maximum_candidates=self.config.pc_rasl_candidate_count,
                )
        self._resumed = True

    def _auxiliary_loss(
        self, output: MRCRALanguageOutput, batch: PackedBatch, start: int, end: int,
    ) -> tuple[Tensor, dict[str, float], set[ObjectiveFamily]]:
        terms = tuple(self.supervision_provider(output, batch, start, end))
        terms = tuple(
            term for term in terms if self.schedule.weight(term.family) > 0
        )
        if not terms:
            return output.cognitive.latent.sum() * 0, {}, set()
        for term in terms:
            if term.family in (ObjectiveFamily.PRIMARY_TASK, ObjectiveFamily.SPECTRAL_SUBSTRATE):
                raise ValueError("the trainer owns primary and spectral objectives")
        breakdown = combine_cognitive_objectives(terms, self.schedule)
        metrics = {f"train/objective/{name}": float(value.detach().cpu())
                   for name, value in breakdown.terms.items()}
        return breakdown.total, metrics, {term.family for term in terms}

    @torch.no_grad()
    def _publish_cognitive_snapshot(self, reporter: TrackioReporter) -> None:
        if (
            not self.config.spectral_dashboard
            or self._last_snapshot_attempt_step == self.state.step
        ):
            return
        from .cognitive_diagnostics import cognitive_evidence
        from .visualization import model_spectral_evidence

        was_training = self.model.training
        # Attempt cadence is independent of publication success. Diagnostics
        # are non-authoritative and must not consume every optimizer step when
        # one snapshot exposes a serialization or frontend defect.
        self._last_snapshot_attempt_step = self.state.step
        try:
            evidence = model_spectral_evidence(
                self.model, self.tokenizer,
                prompt=self.config.spectral_dashboard_prompt,
                maximum_tokens=self.config.spectral_snapshot_tokens,
                step=self.state.step, tokens_seen=self.state.tokens_seen,
                source="live MRCRA training model",
                format_version=MRCRA_TRAINING_FORMAT_VERSION,
            )
            prompt_ids = self.tokenizer.encode_prompt(
                self.config.spectral_dashboard_prompt
            )[: self.config.spectral_snapshot_tokens]
            input_ids = torch.tensor(
                [prompt_ids], dtype=torch.int64, device=self.device
            )
            with self._autocast():
                cognitive_output = self.model(
                    input_ids, source_uris=("diagnostic://fixed-prompt",)
                )
            evidence["checkpoint"]["mrcra_configuration"] = asdict(self.model.config)
            evidence["cognitive"] = cognitive_evidence(
                cognitive_output.cognitive, cognitive_output.ledger
            )
            reporter.log_spectral_evidence(evidence, step=self.state.step)
            self._last_snapshot_step = self.state.step
        except Exception as error:
            reporter.alert(
                "MRCRA diagnostic snapshot failed",
                f"{type(error).__name__}: {error}. Optimization continues because diagnostics are non-authoritative.",
                level="warn", step=self.state.step,
            )
        finally:
            self.model.train(was_training)

    def _integrated_spans(self, batch: PackedBatch) -> tuple[tuple[int, int, bool], ...]:
        """Return TBPTT spans that never cross an independent document boundary."""

        if batch.input_ids.shape[0] != 1:
            raise ValueError("the integrated multirate path requires microbatch one")
        segments = batch.segment_ids[0].detach().cpu()
        length = segments.numel()
        result: list[tuple[int, int, bool]] = []
        document_start = 0
        while document_start < length:
            document_end = document_start + 1
            while document_end < length and segments[document_end] == segments[document_start]:
                document_end += 1
            span_start = document_start
            reset = True
            while span_start < document_end:
                span_end = min(document_end, span_start + self.config.tbptt_length)
                if bool(batch.loss_mask[:, span_start:span_end].any()):
                    result.append((span_start, span_end, reset))
                    reset = False
                span_start = span_end
            document_start = document_end
        if not result:
            raise ValueError("packed context has no trainable within-document span")
        return tuple(result)

    @staticmethod
    def _integrated_state_energy(state, target_rms: float) -> tuple[Tensor, Tensor]:
        penalties, maximum = MRCRANextTokenTrainer._integrated_state_energy_rows(
            state, target_rms
        )
        return penalties.mean(), maximum

    @staticmethod
    def _integrated_state_energy_rows(
        state, target_rms: float,
    ) -> tuple[Tensor, Tensor]:
        dense = tuple(
            resonator.value.float().square().flatten(1).mean(1)
            for block in state.carrier.blocks
            for resonator in block.resonators
        )
        cognitive_carriers = state.cognitive.carrier
        cognitive = tuple(
            torch.cat(
                tuple(
                    carrier.blocks[block_index]
                    .resonators[resonator_index]
                    .value.float()
                    .square()
                    .flatten(1)
                    .mean(1)
                    for carrier in cognitive_carriers
                ),
                0,
            )
            for block_index in range(len(cognitive_carriers[0].blocks))
            for resonator_index in range(
                len(cognitive_carriers[0].blocks[block_index].resonators)
            )
        )
        energies = torch.stack(dense + cognitive, 1)
        rms = energies.clamp_min(0).sqrt()
        penalty = (energies - target_rms**2).clamp_min(0).mean(1)
        return penalty, rms.max()

    def _run_integrated_context(
        self, batch: PackedBatch, *, gradient_divisor: int,
    ) -> dict[str, float]:
        """Train dense carrier and event-rate cognition under exact language loss."""

        batch = batch.to(self.device, non_blocking=self.device.type == "cuda")
        spans = self._integrated_spans(batch)
        total_valid = int(batch.loss_mask.sum())
        if total_valid <= 0:
            raise ValueError("packed context has no within-document next-token targets")
        head = self.model.cognitive.carrier.output_head
        nll_sum = 0.0
        byte_count = 0
        state_rms_max = 0.0
        cognitive_cycles = 0
        event_count = 0
        event_activation_total = 0.0
        event_opened = 0
        event_finalized = 0
        event_emitted = 0
        event_quota_rejected = 0
        event_open_after = 0
        active_nodes_total = 0.0
        active_nodes_max = 0.0
        feedback_rms_max = 0.0
        cstm_objective_weight = self._cstm_objective_weight()
        cstm_context_weight = (
            self._cstm_context_weight(batch)
            if cstm_objective_weight > 0
            else 0.0
        )
        cstm_loss_numerator = 0.0
        cstm_weighted_rows = 0.0
        cstm_valid_rows = 0
        cstm_coefficient_targets = 0
        cstm_token_participations = 0
        cstm_scale_rows = [0] * self.model.config.carrier.scales
        cstm_horizon_rows: dict[int, int] = {}
        training_state = None
        ledger: ProvenanceLedger | None = None
        last_state = None
        last_ledger = None
        group_latents: list[Tensor] = []
        group_labels: list[Tensor] = []
        group_byte_lengths: list[Tensor] = []
        group_masks: list[Tensor] = []
        group_penalties: list[Tensor] = []
        group_state_maxima: list[Tensor] = []
        group_event_activations: list[Tensor] = []
        group_cstm_sums: dict[int, Tensor] = {}
        group_cstm_weights: dict[int, Tensor] = {}
        group_tokens = 0
        document_start = 0
        proposal_logit_rows: list[Tensor] = []
        end_logit_rows: list[Tensor] = []
        processed_tokens = 0
        next_progress = self.config.progress_interval_tokens
        started = perf_counter()
        model_forward_seconds = 0.0
        loss_forward_seconds = 0.0
        backward_seconds = 0.0
        primary_backward_seconds = 0.0
        cstm_substrate_backward_seconds = 0.0

        def flush_group(*, final: bool) -> None:
            nonlocal nll_sum, byte_count, state_rms_max, group_tokens
            nonlocal loss_forward_seconds, backward_seconds
            nonlocal primary_backward_seconds
            nonlocal cstm_substrate_backward_seconds
            nonlocal cstm_loss_numerator, cstm_weighted_rows
            if not group_latents:
                return
            loss_started = perf_counter()
            with self._autocast():
                statistics = self._language_statistics(
                    torch.cat(group_latents, 1),
                    torch.cat(group_labels, 1),
                    torch.cat(group_byte_lengths, 1),
                    torch.cat(group_masks, 1),
                    head,
                )
                loss = statistics.nll_sum / total_valid
                if group_penalties:
                    loss = loss + self.config.state_regularization_weight * (
                        torch.stack(group_penalties).sum().to(loss.device) / len(spans)
                    )
                if group_event_activations:
                    loss = loss + self.config.event_compute_regularization_weight * (
                        torch.stack(group_event_activations).sum().to(loss.device)
                        / len(spans)
                    )
                if final and self.schedule.weight(ObjectiveFamily.SPECTRAL_SUBSTRATE):
                    loss = loss + self.config.spectral_regularization_weight * (
                        spectral_activation_regularization(self._spectral_modules).to(
                            loss.device
                        )
                    )
                scaled = loss / gradient_divisor
                active_cstm_scales = tuple(
                    scale
                    for scale, weight in group_cstm_weights.items()
                    if float(weight.detach()) > 0
                )
                cstm_group_loss = (
                    torch.stack([
                        group_cstm_sums[scale]
                        for scale in active_cstm_scales
                    ]).sum()
                    / max(cstm_context_weight, 1.0)
                    if active_cstm_scales
                    else loss * 0
                )
            loss_forward_seconds += perf_counter() - loss_started
            if not bool(torch.isfinite(scaled)):
                raise FloatingPointError("MRCRA integrated language loss became non-finite")
            if active_cstm_scales and cstm_objective_weight > 0:
                cstm_started = perf_counter()
                self._accumulate_cstm_gradients(
                    cstm_group_loss,
                    objective_weight=cstm_objective_weight,
                    gradient_divisor=gradient_divisor,
                )
                cstm_substrate_backward_seconds += (
                    perf_counter() - cstm_started
                )
            primary_started = perf_counter()
            if self.scaler is None:
                scaled.backward()
            else:
                self.scaler.scale(scaled).backward()
            primary_backward_seconds += perf_counter() - primary_started
            backward_seconds = (
                primary_backward_seconds
                + cstm_substrate_backward_seconds
            )
            nll_sum += float(statistics.nll_sum.detach().cpu())
            byte_count += statistics.byte_count
            if group_state_maxima:
                state_rms_max = max(
                    state_rms_max,
                    float(torch.stack(group_state_maxima).max().detach().cpu()),
                )
            if active_cstm_scales:
                local_sum = sum(
                    float(group_cstm_sums[scale].detach().cpu())
                    for scale in active_cstm_scales
                )
                local_weight = sum(
                    float(group_cstm_weights[scale].detach().cpu())
                    for scale in active_cstm_scales
                )
                cstm_loss_numerator += local_sum
                cstm_weighted_rows += local_weight
            group_latents.clear()
            group_labels.clear()
            group_byte_lengths.clear()
            group_masks.clear()
            group_penalties.clear()
            group_state_maxima.clear()
            group_event_activations.clear()
            group_cstm_sums.clear()
            group_cstm_weights.clear()
            group_tokens = 0

        for span_index, (start, end, reset) in enumerate(spans):
            if reset:
                training_state = None
                ledger = ProvenanceLedger()
                document_start = start
            if group_tokens and group_tokens + (end - start) > self.config.tbptt_length:
                flush_group(final=False)
            if ledger is None:
                raise RuntimeError("integrated document span omitted its provenance ledger")
            local_mask = batch.loss_mask[:, start:end]
            forward_started = perf_counter()
            with self._autocast():
                packet, ledger = self.model.prepare_external_input(
                    batch.input_ids[:, start:end],
                    attention_mask=torch.ones_like(local_mask),
                    segment_ids=batch.segment_ids[:, start:end],
                    boundary_classes=batch.boundary_classes[:, start:end],
                    source_uris=batch.external_source_uris,
                    ledger=ledger,
                    continuing=training_state is not None,
                    timestamp_offset=start,
                )
                output = self.model.cognitive.forward_integrated_training(
                    packet, ledger, state=training_state,
                    cognitive_stride=self.config.cognitive_stride,
                    cognitive_tbptt_events=self.config.cognitive_tbptt_events,
                )
                state_penalty, state_max = self._integrated_state_energy(
                    output.state, self.config.state_target_rms,
                )
                cstm_predictions = (
                    self.model.predict_causal_spectral_targets(
                        output,
                        extra_horizon_offset=self.state.step,
                    )
                    if cstm_objective_weight > 0 else ()
                )
            model_forward_seconds += perf_counter() - forward_started
            group_latents.append(output.output_latent)
            group_labels.append(batch.labels[:, start:end])
            group_byte_lengths.append(batch.target_byte_lengths[:, start:end])
            group_masks.append(local_mask)
            group_penalties.append(state_penalty)
            group_state_maxima.append(state_max)
            group_event_activations.append(output.event_activation_mean)
            for prediction in cstm_predictions:
                source_positions = (
                    prediction.source_positions + document_start
                )
                targets = build_causal_spectral_targets(
                    batch.labels,
                    batch.loss_mask,
                    batch.segment_ids,
                    batch.target_segment_ids,
                    self.model.cstm_predictor.token_codes,
                    source_positions,
                    support=prediction.support,
                    horizons=prediction.horizons,
                )
                report: CSTMLoss = self.model.cstm_predictor.loss(
                    prediction.values,
                    targets,
                    scale=prediction.scale,
                    update_statistics=self.model.training,
                )
                if report.valid_rows:
                    group_cstm_sums[prediction.scale] = (
                        group_cstm_sums.get(
                            prediction.scale,
                            report.standardized_huber_sum * 0,
                        )
                        + report.standardized_huber_sum
                    )
                    group_cstm_weights[prediction.scale] = (
                        group_cstm_weights.get(
                            prediction.scale,
                            report.weighted_rows * 0,
                        )
                        + report.weighted_rows
                    )
                cstm_valid_rows += report.valid_rows
                cstm_coefficient_targets += report.coefficient_targets
                cstm_token_participations += report.token_participations
                cstm_scale_rows[prediction.scale] += report.valid_rows
                for horizon, count in zip(
                    prediction.horizons,
                    report.per_horizon_rows,
                    strict=True,
                ):
                    cstm_horizon_rows[horizon] = (
                        cstm_horizon_rows.get(horizon, 0) + count
                    )
            group_tokens += end - start
            processed_tokens = end
            cognitive_cycles += int(output.cognitive_cycles.sum().detach().cpu())
            event_count += int(output.event_counts.sum().detach().cpu())
            event_activation_total += float(
                output.event_activation_mean.detach().cpu()
            )
            proposal_logit_rows.append(output.event_proposal_logits.detach())
            end_logit_rows.append(output.event_end_logits.detach())
            event_opened += int(output.event_opened.sum().detach().cpu())
            event_finalized += int(output.event_finalized.sum().detach().cpu())
            event_emitted += int(output.event_emitted.sum().detach().cpu())
            event_quota_rejected += int(
                output.event_quota_rejected.sum().detach().cpu()
            )
            event_open_after = int(output.event_open_after[:, -1].sum().detach().cpu())
            if (
                self.config.phase_transition_telemetry
                and self.state.first_hard_event_step == 0
                and self._pending_first_hard_event_trace is None
                and output.first_hard_event is not None
            ):
                self._pending_first_hard_event_trace = replace(
                    output.first_hard_event,
                    anchor_index=start + output.first_hard_event.anchor_index,
                )
            active_nodes_total += float(output.active_nodes_mean.detach().cpu())
            active_nodes_max = max(
                active_nodes_max, float(output.active_nodes_max.detach().cpu())
            )
            feedback_rms_max = max(
                feedback_rms_max, float(output.feedback_rms.detach().cpu())
            )
            training_state = output.state.detach()
            last_state, last_ledger = training_state, ledger
            final = span_index == len(spans) - 1
            if group_tokens >= self.config.tbptt_length or final:
                flush_group(final=final)
            if processed_tokens >= next_progress or final:
                _synchronize(self.device)
                elapsed = perf_counter() - started
                print(
                    f"update={self.state.step + 1} integrated_tokens={processed_tokens}/"
                    f"{batch.input_ids.shape[1]} elapsed={elapsed:.1f}s "
                    f"tok/s={processed_tokens / max(elapsed, 1e-9):.1f} "
                    f"cognitive_cycles={cognitive_cycles}",
                    flush=True,
                )
                while next_progress <= processed_tokens:
                    next_progress += self.config.progress_interval_tokens
        self._last_runtime = None if last_state is None else last_state.cognitive
        self._last_ledger = last_ledger
        self._last_continuity_keys = None
        cognitive_parameters = tuple(
            parameter for name, parameter in self.model.cognitive.named_parameters()
            if not name.startswith("carrier.")
        )
        cognitive_gradients = tuple(
            parameter.grad for parameter in cognitive_parameters
            if parameter.grad is not None
        )
        cognitive_gradient_norm = (
            torch.stack(tuple(gradient.float().square().sum() for gradient in cognitive_gradients))
            .sum().sqrt() if cognitive_gradients else torch.tensor(0.0)
        )
        metrics = {
            "train/nll_sum": nll_sum,
            "train/cross_entropy_nats_per_token": nll_sum / total_valid,
            "train/effective_cross_entropy_nats_per_byte": nll_sum / max(1, byte_count),
            "train/bits_per_byte": nll_sum / max(1, byte_count) / log(2),
            "architecture/state_rms_max": state_rms_max,
            "architecture/cognitive_feedback_rms_max": feedback_rms_max,
            "architecture/cognitive_cycles": float(cognitive_cycles),
            "architecture/events": float(event_count),
            "architecture/event_opened": float(event_opened),
            "architecture/event_finalized": float(event_finalized),
            "architecture/event_emitted": float(event_emitted),
            "architecture/event_quota_rejected": float(event_quota_rejected),
            "architecture/event_open_after": float(event_open_after),
            "architecture/event_activation_mean": (
                event_activation_total / len(spans)
            ),
            "architecture/active_nodes_mean": active_nodes_total / len(spans),
            "architecture/active_nodes_max": active_nodes_max,
            "architecture/node_capacity_utilization_max": (
                active_nodes_max
                / self.model.config.cognitive.active_event_capacity
            ),
            "optimization/cognitive_gradient_norm": float(
                cognitive_gradient_norm.detach().cpu()
            ),
            "optimization/cognitive_gradient_tensor_fraction": (
                len(cognitive_gradients) / max(1, len(cognitive_parameters))
            ),
            "train/valid_targets": float(total_valid),
            "train/utf8_bytes": float(byte_count),
            "training/integrated_cognitive_path": 1.0,
            "cstm/enabled": float(self.cstm_enabled),
            "cstm/objective_weight": cstm_objective_weight,
            "cstm/standardized_huber": (
                cstm_loss_numerator / max(1.0, cstm_weighted_rows)
            ),
            "cstm/standardized_huber_sum": cstm_loss_numerator,
            "cstm/estimated_dense_standardized_huber": (
                cstm_loss_numerator
                / max(1.0, cstm_context_weight)
            ),
            "cstm/estimated_dense_numerator": cstm_loss_numerator,
            "cstm/context_valid_weight": cstm_context_weight,
            "cstm/weighted_prediction_rows": cstm_weighted_rows,
            "cstm/spectral_target_views": float(cstm_valid_rows),
            "cstm/coefficient_targets": float(cstm_coefficient_targets),
            "cstm/raw_token_view_equivalents": float(
                cstm_token_participations
            ),
            "cstm/supervision_relations_per_primary_target": (
                cstm_token_participations / max(1, total_valid)
            ),
            "cstm/sampling_active": 0.0,
            "cstm/sampling_obligations": 0.0,
            "cstm/sampling_inclusion_probability": 1.0,
            "cstm/sampling_inverse_probability": 1.0,
            "cstm/predictor_backward_seconds": 0.0,
            "cstm/substrate_backward_seconds": (
                cstm_substrate_backward_seconds
            ),
            "performance/model_forward_seconds": model_forward_seconds,
            "performance/primary_forward_seconds": model_forward_seconds,
            "performance/loss_forward_seconds": loss_forward_seconds,
            "performance/backward_seconds": backward_seconds,
            "performance/primary_backward_seconds": (
                primary_backward_seconds
            ),
            "performance/carrier_custom_affine_adjoint": 1.0,
            "performance/carrier_custom_simplex_adjoint": 1.0,
            "performance/carrier_whole_span_checkpoint": float(
                self.model.cognitive.carrier._last_composite_receipt
                is not None
            ),
            "softmax/training/exact_full_vocabulary": 1.0,
            "softmax/training/external_cce_available": float(
                self.runtime["cut_cross_entropy_available"]
            ),
            "softmax/training/compiled_cce_fits_workspace": float(
                self.runtime["compiled_cce_fits_workspace"]
            ),
            "softmax/training/compiled_cce_runtime_quarantined": float(
                self.runtime["compiled_cce_runtime_quarantined"]
            ),
            "softmax/training/compiled_cce_runtime_fallbacks": float(
                self.runtime["compiled_cce_runtime_fallbacks"]
            ),
            "softmax/training/estimated_full_logits_mib": (
                self.runtime["estimated_fused_loss_bytes"] / (1 << 20)
            ),
            "softmax/training/backend_id": float({
                "tiled": 0,
                "fused": 1,
                "torch_compile": 2,
                "cce_kahan_full_c": 3,
                "cce_exact": 4,
                "mlx": 5,
            }[self._exact_loss_backend]),
            "softmax/training/mlx_peak_memory_mib": (
                self.runtime["mlx_peak_memory_bytes"] / (1 << 20)
            ),
            "softmax/training/mlx_cache_memory_mib": (
                self.runtime["mlx_cache_memory_bytes"] / (1 << 20)
            ),
        }
        for scale, rows in enumerate(cstm_scale_rows):
            metrics[f"cstm/scale/{scale}/valid_rows"] = float(rows)
        for horizon, rows in sorted(cstm_horizon_rows.items()):
            metrics[f"cstm/horizon/{horizon}/valid_rows"] = float(rows)
        if self.config.phase_transition_telemetry:
            if not proposal_logit_rows or not end_logit_rows:
                raise RuntimeError("integrated path omitted event phase logits")
            proposal_logits, end_logits = _concatenate_event_phase_logits(
                proposal_logit_rows, end_logit_rows,
            )
            self._phase_update_proposal_logits.append(proposal_logits.cpu())
            self._phase_update_end_logits.append(end_logits.cpu())
            metrics.update(event_phase_metrics(
                proposal_logits,
                end_logits,
                proposal_threshold_logit=self.model.cognitive.event_extractor.proposal_logit,
            ))
        return metrics

    def _run_document_major_context(
        self, batch: PackedBatch, *, gradient_divisor: int,
    ) -> dict[str, float]:
        """Execute independent documents as stable, static-shaped row cohorts.

        The planner is the sole authority for regrouping.  Its bijection
        receipt proves that every original language target appears exactly
        once.  Recurrent state and provenance are local to one cohort, rows
        never change identity between spans, and every padded token/event is
        explicitly masked.  Gradients are truncated only after one complete
        static TBPTT span, which makes this both the semantic replacement for
        the former document-at-a-time loop and the coarse unit used by the
        optimized execution backends.
        """

        planner = self.document_batch_planner
        if planner is None:
            raise RuntimeError("document-major execution requires a configured planner")
        batch = batch.to(self.device, non_blocking=self.device.type == "cuda")
        planner_started = perf_counter()
        plan = planner.plan(batch)
        planner_seconds = perf_counter() - planner_started
        total_valid = plan.original_valid_targets
        if total_valid <= 0 or not plan.receipt.passed:
            raise RuntimeError("document-major plan omitted exact target authority")

        head = self.model.cognitive.carrier.output_head
        physical_invocations = plan.physical_invocations
        logical_spans = sum(
            len(sequence.spans) for sequence in plan.sequences
        )
        if physical_invocations <= 0:
            raise RuntimeError("document-major plan contains no physical invocation")
        cstm_objective_weight = self._cstm_objective_weight()
        cstm_sampling = (
            self._cstm_document_sampling_decision(plan)
            if (
                cstm_objective_weight > 0
                and self.config.cstm_execution == "sampled"
            )
            else None
        )
        cstm_predictor_sampling = (
            self._cstm_document_sampling_decision(
                plan, duty_probability=1.0,
            )
            if (
                cstm_sampling is not None
                and self.state.step
                % self.config.cstm_predictor_update_interval
                == 0
            )
            else None
        )
        cstm_context_weight = (
            cstm_sampling.dense_weight
            if cstm_sampling is not None
            else self._cstm_context_weight(batch)
            if cstm_objective_weight > 0
            else 0.0
        )
        predictor_statistics_weight = 1.0
        if (
            cstm_predictor_sampling is not None
            and cstm_predictor_sampling.active
            and cstm_predictor_sampling.obligation is not None
        ):
            predictor_statistics_weight = (
                cstm_predictor_sampling.obligation.dense_weight
                / (
                    cstm_predictor_sampling.dense_weight
                    * cstm_predictor_sampling.conditional_probability
                )
            )

        nll_sum = 0.0
        byte_count = 0
        state_rms_max = 0.0
        cognitive_cycles = 0
        event_count = 0
        event_activation_weighted = 0.0
        valid_event_rows = 0
        event_opened = 0
        event_finalized = 0
        event_emitted = 0
        event_quota_rejected = 0
        event_open_after = 0
        active_nodes_weighted = 0.0
        active_nodes_max = 0.0
        feedback_rms_max = 0.0
        cstm_loss_numerator = 0.0
        cstm_weighted_rows = 0.0
        cstm_valid_rows = 0
        cstm_coefficient_targets = 0
        cstm_token_participations = 0
        cstm_scale_rows = [0] * self.model.config.carrier.scales
        cstm_horizon_rows: dict[int, int] = {}
        cstm_estimated_dense_numerator = 0.0
        cstm_estimated_target_views = 0.0
        cstm_estimated_token_participations = 0.0
        cstm_row_inclusion_probability_min = 1.0
        cstm_row_inverse_probability_max = 1.0
        cstm_substrate_vjp_count = 0
        proposal_logit_rows: list[Tensor] = []
        end_logit_rows: list[Tensor] = []
        last_state = None
        last_ledger = None
        processed_tokens = 0
        processed_physical_tokens = 0
        next_progress = self.config.progress_interval_tokens
        started = perf_counter()
        model_forward_seconds = 0.0
        loss_forward_seconds = 0.0
        backward_seconds = 0.0
        cstm_predictor_backward_seconds = 0.0
        cstm_substrate_backward_seconds = 0.0
        invocation_index = 0
        activation_invocations = {
            "retain": 0,
            "selective": 0,
            "whole_span": 0,
        }

        for cohort in plan.cohorts:
            authority = cohort.target_authority()
            training_state = None
            ledger = ProvenanceLedger()
            for physical in cohort.spans:
                invocation_index += 1
                physical_tokens = int(physical.input_ids.numel())
                invocation_activation_policy = fastest_safe_activation_policy(
                    self.activation_execution_policy,
                    physical_tokens=physical_tokens,
                    physical_token_limits=(
                        self.activation_policy_token_limits
                    ),
                )
                activation_invocations[invocation_activation_policy] += 1
                invocation_selective_scales = (
                    tuple(
                        self.runtime[
                            "carrier_selective_checkpoint_scales"
                        ]
                    )
                    if invocation_activation_policy == "selective"
                    else ()
                )
                self.model.cognitive.carrier.configure_activation_execution(
                    invocation_activation_policy,
                    selective_scales=invocation_selective_scales,
                )
                if physical.reset_state != (training_state is None):
                    raise RuntimeError(
                        "document-major cohort violated its declared state-reset boundary"
                    )
                token_segments = physical.segment_ids[:, None].expand(
                    -1, physical.padded_length
                ).masked_fill(~physical.token_mask, -1)
                forward_started = perf_counter()
                with self._autocast():
                    packet, ledger = self.model.prepare_external_input(
                        physical.input_ids,
                        attention_mask=physical.token_mask,
                        segment_ids=token_segments,
                        boundary_classes=physical.boundary_classes,
                        source_uris=physical.source_uris,
                        ledger=ledger,
                        continuing=training_state is not None,
                        timestamp_offset=physical.context_starts,
                    )
                    output = self.model.cognitive.forward_integrated_training(
                        packet,
                        ledger,
                        state=training_state,
                        cognitive_stride=self.config.cognitive_stride,
                        cognitive_tbptt_events=self.config.cognitive_tbptt_events,
                    )
                    if not torch.equal(output.event_mask, physical.event_mask):
                        raise RuntimeError(
                            "cognitive execution event mask differs from static plan authority"
                        )
                    state_penalty_rows, state_max = (
                        self._integrated_state_energy_rows(
                        output.state, self.config.state_target_rms,
                        )
                    )
                    selected_substrate_scale = (
                        cstm_sampling.obligation.scale
                        if (
                            cstm_sampling is not None
                            and cstm_sampling.active
                            and cstm_sampling.obligation is not None
                            and cstm_sampling.obligation.invocation
                            == invocation_index
                        )
                        else None
                    )
                    selected_predictor_scale = (
                        cstm_predictor_sampling.obligation.scale
                        if (
                            cstm_predictor_sampling is not None
                            and cstm_predictor_sampling.active
                            and cstm_predictor_sampling.obligation is not None
                            and cstm_predictor_sampling.obligation.invocation
                            == invocation_index
                        )
                        else None
                    )
                    substrate_predictions = (
                        self.model.predict_causal_spectral_targets(
                            output,
                            extra_horizon_offset=self.state.step,
                            selected_scales=(
                                None
                                if cstm_sampling is None
                                else ()
                                if selected_substrate_scale is None
                                else (selected_substrate_scale,)
                            ),
                            target_participation_budget=(
                                self.config.cstm_target_participation_budget
                                if cstm_sampling is not None else None
                            ),
                            row_sampling_digest=(
                                None
                                if cstm_sampling is None
                                else cstm_sampling.counter_digest
                            ),
                            row_sampling_stream=0,
                        )
                        if (
                            cstm_objective_weight > 0
                            and (
                                cstm_sampling is None
                                or selected_substrate_scale is not None
                            )
                        )
                        else ()
                    )
                    predictor_predictions = (
                        self.model.predict_causal_spectral_targets(
                            output,
                            extra_horizon_offset=self.state.step,
                            selected_scales=(selected_predictor_scale,),
                            detach_substrate=True,
                            target_participation_budget=(
                                self.config.cstm_target_participation_budget
                            ),
                            row_sampling_digest=(
                                cstm_predictor_sampling.counter_digest
                                if cstm_predictor_sampling is not None
                                else None
                            ),
                            row_sampling_stream=10_000,
                        )
                        if selected_predictor_scale is not None else ()
                    )
                model_forward_seconds += perf_counter() - forward_started

                loss_started = perf_counter()
                with self._autocast():
                    statistics = self._language_statistics(
                        output.output_latent,
                        physical.labels,
                        physical.target_byte_lengths,
                        physical.loss_mask,
                        head,
                    )
                    loss = statistics.nll_sum / total_valid
                    loss = loss + self.config.state_regularization_weight * (
                        state_penalty_rows.sum().to(loss.device)
                        / logical_spans
                    )
                    event_probability = torch.sigmoid(
                        output.event_proposal_logits
                    ) * (
                        0.5
                        + 0.5 * torch.sigmoid(output.event_end_logits)
                    )
                    event_rows = (
                        event_probability
                        * output.event_mask.to(event_probability.dtype)
                    ).sum(1) / output.event_mask.sum(1).clamp_min(1)
                    loss = loss + self.config.event_compute_regularization_weight * (
                        event_rows.sum().to(loss.device) / logical_spans
                    )
                    if (
                        invocation_index == physical_invocations
                        and self.schedule.weight(ObjectiveFamily.SPECTRAL_SUBSTRATE)
                    ):
                        loss = loss + self.config.spectral_regularization_weight * (
                            spectral_activation_regularization(
                                self._spectral_modules
                            ).to(loss.device)
                        )

                    substrate_sums: dict[int, Tensor] = {}
                    substrate_weights: dict[int, Tensor] = {}
                    predictor_sums: dict[int, Tensor] = {}
                    predictor_weights: dict[int, Tensor] = {}

                    def evaluate_predictions(
                        predictions,
                        sums: dict[int, Tensor],
                        weights: dict[int, Tensor],
                        *,
                        update_statistics: bool,
                        account_targets: bool,
                        statistics_importance_weight: float = 1.0,
                        obligation_inverse_probability: float = 1.0,
                    ) -> None:
                        nonlocal cstm_valid_rows
                        nonlocal cstm_coefficient_targets
                        nonlocal cstm_token_participations
                        nonlocal cstm_estimated_target_views
                        nonlocal cstm_estimated_token_participations
                        nonlocal cstm_row_inclusion_probability_min
                        nonlocal cstm_row_inverse_probability_max
                        for prediction in predictions:
                            targets = build_causal_spectral_targets(
                                authority.labels,
                                authority.loss_mask,
                                authority.segment_ids,
                                authority.target_segment_ids,
                                self.model.cstm_predictor.token_codes,
                                prediction.source_positions,
                                support=prediction.support,
                                horizons=prediction.horizons,
                            )
                            report: CSTMLoss = self.model.cstm_predictor.loss(
                                prediction.values,
                                targets,
                                scale=prediction.scale,
                                update_statistics=(
                                    self.model.training
                                    and update_statistics
                                ),
                                statistics_importance_weight=(
                                    statistics_importance_weight
                                    / prediction.row_inclusion_probability
                                ),
                            )
                            row_inverse = (
                                1.0 / prediction.row_inclusion_probability
                            )
                            cstm_row_inclusion_probability_min = min(
                                cstm_row_inclusion_probability_min,
                                prediction.row_inclusion_probability,
                            )
                            cstm_row_inverse_probability_max = max(
                                cstm_row_inverse_probability_max,
                                row_inverse,
                            )
                            if report.valid_rows:
                                sums[prediction.scale] = (
                                    sums.get(
                                        prediction.scale,
                                        report.standardized_huber_sum * 0,
                                    )
                                    + report.standardized_huber_sum
                                    * row_inverse
                                )
                                weights[prediction.scale] = (
                                    weights.get(
                                        prediction.scale,
                                        report.weighted_rows * 0,
                                    )
                                    + report.weighted_rows * row_inverse
                                )
                            if account_targets:
                                cstm_valid_rows += report.valid_rows
                                cstm_coefficient_targets += (
                                    report.coefficient_targets
                                )
                                cstm_token_participations += (
                                    report.token_participations
                                )
                                cstm_estimated_target_views += (
                                    report.valid_rows
                                    * row_inverse
                                    * obligation_inverse_probability
                                )
                                cstm_estimated_token_participations += (
                                    report.token_participations
                                    * row_inverse
                                    * obligation_inverse_probability
                                )
                                cstm_scale_rows[prediction.scale] += (
                                    report.valid_rows
                                )
                                for horizon, count in zip(
                                    prediction.horizons,
                                    report.per_horizon_rows,
                                    strict=True,
                                ):
                                    cstm_horizon_rows[horizon] = (
                                        cstm_horizon_rows.get(horizon, 0)
                                        + count
                                    )

                    if cstm_sampling is None:
                        evaluate_predictions(
                            substrate_predictions,
                            substrate_sums,
                            substrate_weights,
                            update_statistics=True,
                            account_targets=True,
                            statistics_importance_weight=1.0,
                            obligation_inverse_probability=1.0,
                        )
                    else:
                        if cstm_predictor_sampling is not None:
                            evaluate_predictions(
                                predictor_predictions,
                                predictor_sums,
                                predictor_weights,
                                update_statistics=True,
                                account_targets=True,
                                statistics_importance_weight=(
                                    predictor_statistics_weight
                                ),
                                obligation_inverse_probability=(
                                    cstm_predictor_sampling.inverse_probability
                                ),
                            )
                        evaluate_predictions(
                            substrate_predictions,
                            substrate_sums,
                            substrate_weights,
                            update_statistics=False,
                            account_targets=(
                                cstm_predictor_sampling is None
                            ),
                            obligation_inverse_probability=(
                                cstm_sampling.inverse_probability
                            ),
                        )
                    active_substrate_scales = tuple(
                        scale
                        for scale, weight in substrate_weights.items()
                        if float(weight.detach()) > 0
                    )
                    active_predictor_scales = tuple(
                        scale
                        for scale, weight in predictor_weights.items()
                        if float(weight.detach()) > 0
                    )
                    substrate_denominator = (
                        max(
                            cstm_context_weight
                            * cstm_sampling.inclusion_probability,
                            1e-30,
                        )
                        if cstm_sampling is not None and cstm_sampling.active
                        else max(cstm_context_weight, 1.0)
                    )
                    substrate_loss = (
                        torch.stack(
                            [
                                substrate_sums[scale]
                                for scale in active_substrate_scales
                            ]
                        ).sum()
                        / substrate_denominator
                        if active_substrate_scales
                        else loss * 0
                    )
                    predictor_loss = (
                        torch.stack(
                            [
                                predictor_sums[scale]
                                for scale in active_predictor_scales
                            ]
                        ).sum()
                        / max(
                            cstm_context_weight
                            * (
                                cstm_predictor_sampling.inclusion_probability
                                if cstm_predictor_sampling is not None
                                else 1.0
                            ),
                            1e-30,
                        )
                        if active_predictor_scales
                        else loss * 0
                    )
                    scaled = loss / gradient_divisor
                loss_forward_seconds += perf_counter() - loss_started
                if not bool(torch.isfinite(scaled)):
                    raise FloatingPointError(
                        "MRCRA document-major language loss became non-finite"
                    )

                backward_started = perf_counter()
                if active_predictor_scales and cstm_objective_weight > 0:
                    predictor_backward_started = perf_counter()
                    self._accumulate_cstm_gradients(
                        predictor_loss,
                        objective_weight=cstm_objective_weight,
                        gradient_divisor=gradient_divisor,
                        authority="predictor",
                    )
                    cstm_predictor_backward_seconds += (
                        perf_counter() - predictor_backward_started
                    )
                if active_substrate_scales and cstm_objective_weight > 0:
                    substrate_backward_started = perf_counter()
                    self._accumulate_cstm_gradients(
                        substrate_loss,
                        objective_weight=cstm_objective_weight,
                        gradient_divisor=gradient_divisor,
                        authority=(
                            "all"
                            if cstm_sampling is None else "substrate"
                        ),
                    )
                    cstm_substrate_backward_seconds += (
                        perf_counter() - substrate_backward_started
                    )
                    if cstm_sampling is not None:
                        cstm_substrate_vjp_count += 1
                if self.scaler is None:
                    scaled.backward()
                else:
                    self.scaler.scale(scaled).backward()
                backward_seconds += perf_counter() - backward_started
                # Keep the resolved maximum-shape policy as the public module
                # state between invocations. Checkpoint/resume and OOM fallback
                # serialize this authority; shape-local retention is a bounded
                # optimization beneath it.
                self.model.cognitive.carrier.configure_activation_execution(
                    self.activation_execution_policy.resolved,
                    selective_scales=tuple(
                        self.runtime[
                            "carrier_selective_checkpoint_scales"
                        ]
                    )
                    if self.activation_execution_policy.resolved == "selective"
                    else (),
                )

                nll_sum += float(statistics.nll_sum.detach().cpu())
                byte_count += statistics.byte_count
                state_rms_max = max(
                    state_rms_max, float(state_max.detach().cpu())
                )
                use_predictor_metrics = bool(active_predictor_scales)
                metric_scales = (
                    active_predictor_scales
                    if use_predictor_metrics else active_substrate_scales
                )
                metric_sums = (
                    predictor_sums
                    if use_predictor_metrics else substrate_sums
                )
                metric_weights = (
                    predictor_weights
                    if use_predictor_metrics else substrate_weights
                )
                metric_inverse_probability = (
                    cstm_predictor_sampling.inverse_probability
                    if (
                        use_predictor_metrics
                        and cstm_predictor_sampling is not None
                    )
                    else cstm_sampling.inverse_probability
                    if cstm_sampling is not None
                    else 1.0
                )
                if metric_scales:
                    local_cstm_sum = sum(
                        float(metric_sums[scale].detach().cpu())
                        for scale in metric_scales
                    )
                    cstm_loss_numerator += local_cstm_sum
                    cstm_estimated_dense_numerator += (
                        local_cstm_sum
                        * (
                            metric_inverse_probability
                        )
                    )
                    cstm_weighted_rows += sum(
                        float(metric_weights[scale].detach().cpu())
                        for scale in metric_scales
                    )

                local_events = int(output.event_mask.sum().detach().cpu())
                valid_event_rows += local_events
                cognitive_cycles += int(
                    output.cognitive_cycles.sum().detach().cpu()
                )
                event_count += int(output.event_counts.sum().detach().cpu())
                event_activation_weighted += (
                    float(output.event_activation_mean.detach().cpu())
                    * local_events
                )
                if local_events:
                    proposal_logit_rows.append(
                        output.event_proposal_logits[
                            output.event_mask
                        ].detach()
                    )
                    end_logit_rows.append(
                        output.event_end_logits[output.event_mask].detach()
                    )
                event_opened += int(output.event_opened.sum().detach().cpu())
                event_finalized += int(
                    output.event_finalized.sum().detach().cpu()
                )
                event_emitted += int(output.event_emitted.sum().detach().cpu())
                event_quota_rejected += int(
                    output.event_quota_rejected.sum().detach().cpu()
                )
                if bool(physical.final_rows.all()):
                    event_open_after += int(
                        output.event_open_after[:, -1].sum().detach().cpu()
                    )
                active_nodes_weighted += (
                    float(output.active_nodes_mean.detach().cpu())
                    * local_events
                )
                active_nodes_max = max(
                    active_nodes_max,
                    float(output.active_nodes_max.detach().cpu()),
                )
                feedback_rms_max = max(
                    feedback_rms_max,
                    float(output.feedback_rms.detach().cpu()),
                )
                if (
                    self.config.phase_transition_telemetry
                    and self.state.first_hard_event_step == 0
                    and self._pending_first_hard_event_trace is None
                    and output.first_hard_event is not None
                ):
                    row = output.first_hard_event.batch_index
                    if not 0 <= row < physical.batch_size:
                        raise RuntimeError(
                            "hard-event trace names a row outside its cohort"
                        )
                    self._pending_first_hard_event_trace = replace(
                        output.first_hard_event,
                        anchor_index=(
                            int(physical.context_starts[row])
                            + output.first_hard_event.anchor_index
                        ),
                        batch_index=int(physical.document_orders[row]),
                    )

                processed_tokens += physical.valid_tokens
                processed_physical_tokens += physical.token_mask.numel()
                training_state = output.state.detach()
                last_state, last_ledger = training_state, ledger
                final = invocation_index == physical_invocations
                if processed_tokens >= next_progress or final:
                    _synchronize(self.device)
                    elapsed = perf_counter() - started
                    mlx_memory = (
                        f" mlx_peak="
                        f"{self.runtime['mlx_peak_memory_bytes'] / (1 << 20):.0f}MiB"
                        if self._exact_loss_backend == "mlx"
                        else ""
                    )
                    print(
                        f"update={self.state.step + 1} "
                        f"document_tokens={processed_tokens}/"
                        f"{plan.valid_document_tokens} "
                        f"physical_tokens={processed_physical_tokens}/"
                        f"{plan.physical_tokens} "
                        f"elapsed={elapsed:.1f}s "
                        f"tok/s={processed_tokens / max(elapsed, 1e-9):.1f} "
                        f"cognitive_cycles={cognitive_cycles}"
                        f"{mlx_memory}",
                        flush=True,
                    )
                    while next_progress <= processed_tokens:
                        next_progress += self.config.progress_interval_tokens

        if nll_sum <= 0 or byte_count <= 0:
            raise RuntimeError("document-major execution produced no language authority")
        if cstm_substrate_vjp_count > self.config.cstm_max_substrate_vjps:
            raise RuntimeError(
                "CSTM substrate VJP count exceeded its checkpointed hard bound"
            )
        self._last_runtime = None if last_state is None else last_state.cognitive
        self._last_ledger = last_ledger
        self._last_continuity_keys = None
        cognitive_parameters = tuple(
            parameter
            for name, parameter in self.model.cognitive.named_parameters()
            if not name.startswith("carrier.")
        )
        cognitive_gradients = tuple(
            parameter.grad
            for parameter in cognitive_parameters
            if parameter.grad is not None
        )
        cognitive_gradient_norm = (
            torch.stack(
                tuple(
                    gradient.float().square().sum()
                    for gradient in cognitive_gradients
                )
            ).sum().sqrt()
            if cognitive_gradients
            else torch.tensor(0.0)
        )
        coverage_gap = 0
        if cstm_predictor_sampling is not None:
            predictor_obligation = cstm_predictor_sampling.obligation
            predictor_horizons: tuple[int, ...] = ()
            if predictor_obligation is not None:
                configured_horizons = (
                    self.model.cstm_predictor.config.horizon_blocks
                )
                extras = configured_horizons[1:]
                predictor_horizons = (
                    (1,)
                    if not extras
                    else (
                        1,
                        extras[
                            (
                                self.state.step
                                + predictor_obligation.scale
                            )
                            % len(extras)
                        ],
                    )
                )
            self.cstm_coverage.record_predictor(
                cstm_predictor_sampling,
                optimizer_step=self.state.step,
                horizons=tuple(
                    horizon
                    for horizon in predictor_horizons
                    if cstm_horizon_rows.get(horizon, 0) > 0
                ),
            )
            if cstm_sampling is None:
                raise RuntimeError(
                    "predictor sampling exists without substrate schedule"
                )
            self.cstm_coverage.declare_required(
                (
                    *(
                        f"scale:{scale}"
                        for scale in cstm_predictor_sampling.eligible_scales
                    ),
                    *(
                        f"horizon:{horizon}"
                        for horizon, rows in cstm_horizon_rows.items()
                        if rows > 0
                    ),
                )
            )
        if cstm_sampling is not None:
            self.cstm_coverage.record_substrate(cstm_sampling)
            self.cstm_coverage.declare_required(
                tuple(
                    f"scale:{scale}"
                    for scale in cstm_sampling.eligible_scales
                )
            )
        if cstm_sampling is not None or cstm_predictor_sampling is not None:
            coverage_gap = self.cstm_coverage.maximum_gap(
                optimizer_step=self.state.step,
            )
            if coverage_gap > self.config.cstm_maximum_coverage_gap:
                raise RuntimeError(
                    "sampled CSTM coverage exceeded its declared maximum "
                    f"gap ({coverage_gap} > "
                    f"{self.config.cstm_maximum_coverage_gap})"
                )
        predictor_inverse = (
            1.0
            if cstm_predictor_sampling is None
            else cstm_predictor_sampling.inverse_probability
        )
        auxiliary_seconds = (
            cstm_predictor_backward_seconds
            + cstm_substrate_backward_seconds
        )
        primary_backward_seconds = max(
            0.0,
            backward_seconds - auxiliary_seconds,
        )
        measured_kernel_seconds = (
            model_forward_seconds + loss_forward_seconds + backward_seconds
        )
        actual_execution_seconds = perf_counter() - started
        unique_shapes = len({
            (span.batch_size, span.padded_length)
            for cohort in plan.cohorts
            for span in cohort.spans
        })
        predicted_seconds = (
            plan.cost_receipt.selected_estimated_cost
            if planner.cost_model.calibration_kind.startswith("measured_")
            else 0.0
        )
        metrics = {
            "train/nll_sum": nll_sum,
            "train/cross_entropy_nats_per_token": nll_sum / total_valid,
            "train/effective_cross_entropy_nats_per_byte": (
                nll_sum / max(1, byte_count)
            ),
            "train/bits_per_byte": nll_sum / max(1, byte_count) / log(2),
            "train/valid_targets": float(total_valid),
            "train/utf8_bytes": float(byte_count),
            "training/integrated_cognitive_path": 1.0,
            "training/document_major_static_batches": 1.0,
            "document_batching/documents": float(len(plan.sequences)),
            "document_batching/cohorts": float(len(plan.cohorts)),
            "document_batching/physical_invocations": float(
                physical_invocations
            ),
            "document_batching/logical_spans": float(logical_spans),
            "document_batching/unique_shapes": float(unique_shapes),
            "document_batching/receipt_unique_shapes": float(
                plan.cost_receipt.unique_static_shapes
            ),
            "document_batching/physical_tokens": float(plan.physical_tokens),
            "document_batching/valid_tokens": float(
                plan.valid_document_tokens
            ),
            "document_batching/padding_efficiency": plan.padding_efficiency,
            "document_batching/target_bijection": 1.0,
            "execution/activation_invocations_retain": float(
                activation_invocations["retain"]
            ),
            "execution/activation_invocations_selective": float(
                activation_invocations["selective"]
            ),
            "execution/activation_invocations_whole_span": float(
                activation_invocations["whole_span"]
            ),
            "document_batching/estimated_cost": (
                plan.cost_receipt.selected_estimated_cost
            ),
            "document_batching/exact_signature_estimated_cost": (
                plan.cost_receipt.exact_signature_estimated_cost
            ),
            "document_batching/estimated_savings_fraction": (
                plan.cost_receipt.estimated_savings_fraction
            ),
            "document_batching/exact_signature_invocations": float(
                plan.cost_receipt.exact_signature_invocations
            ),
            "document_batching/plan_cache_hit": float(
                plan.cost_receipt.cache_hit
            ),
            "document_batching/rejected_memory_candidates": float(
                plan.cost_receipt.rejected_memory_candidates
            ),
            "document_batching/predicted_peak_memory_bytes": float(
                plan.cost_receipt.predicted_peak_memory_bytes
            ),
            "document_batching/shape_compile_cost": (
                plan.cost_receipt.shape_compile_cost
            ),
            "document_batching/planner_seconds": planner_seconds,
            "document_batching/actual_seconds": actual_execution_seconds,
            "document_batching/predicted_seconds": predicted_seconds,
            "document_batching/cost_prediction_error": (
                0.0
                if predicted_seconds <= 0
                else (
                    actual_execution_seconds - predicted_seconds
                )
                / predicted_seconds
            ),
            "architecture/state_rms_max": state_rms_max,
            "architecture/cognitive_feedback_rms_max": feedback_rms_max,
            "architecture/cognitive_cycles": float(cognitive_cycles),
            "architecture/events": float(event_count),
            "architecture/event_opened": float(event_opened),
            "architecture/event_finalized": float(event_finalized),
            "architecture/event_emitted": float(event_emitted),
            "architecture/event_quota_rejected": float(
                event_quota_rejected
            ),
            "architecture/event_open_after": float(event_open_after),
            "architecture/event_activation_mean": (
                event_activation_weighted / max(1, valid_event_rows)
            ),
            "architecture/active_nodes_mean": (
                active_nodes_weighted / max(1, valid_event_rows)
            ),
            "architecture/active_nodes_max": active_nodes_max,
            "architecture/node_capacity_utilization_max": (
                active_nodes_max
                / self.model.config.cognitive.active_event_capacity
            ),
            "optimization/cognitive_gradient_norm": float(
                cognitive_gradient_norm.detach().cpu()
            ),
            "optimization/cognitive_gradient_tensor_fraction": (
                len(cognitive_gradients) / max(1, len(cognitive_parameters))
            ),
            "cstm/enabled": float(self.cstm_enabled),
            "cstm/objective_weight": cstm_objective_weight,
            "cstm/standardized_huber": (
                cstm_loss_numerator / max(1.0, cstm_weighted_rows)
            ),
            "cstm/standardized_huber_sum": cstm_loss_numerator,
            "cstm/estimated_dense_standardized_huber": (
                cstm_estimated_dense_numerator
                / max(1.0, cstm_context_weight)
            ),
            "cstm/estimated_dense_numerator": (
                cstm_estimated_dense_numerator
            ),
            "cstm/context_valid_weight": cstm_context_weight,
            "cstm/weighted_prediction_rows": cstm_weighted_rows,
            "cstm/spectral_target_views": float(cstm_valid_rows),
            "cstm/coefficient_targets": float(cstm_coefficient_targets),
            "cstm/raw_token_view_equivalents": float(
                cstm_token_participations
            ),
            "cstm/supervision_relations_per_primary_target": (
                cstm_token_participations / max(1, total_valid)
            ),
            "cstm/sampling_active": float(
                cstm_sampling is not None and cstm_sampling.active
            ),
            "cstm/sampling_obligations": float(
                0
                if cstm_sampling is None
                else cstm_sampling.obligation_count
            ),
            "cstm/sampling_inclusion_probability": (
                1.0
                if cstm_sampling is None
                else cstm_sampling.inclusion_probability
            ),
            "cstm/sampling_inverse_probability": (
                1.0
                if cstm_sampling is None
                else cstm_sampling.inverse_probability
            ),
            "cstm/predictor_update": float(
                cstm_predictor_sampling is not None
                and cstm_predictor_sampling.active
            ),
            "cstm/substrate_update": float(
                cstm_sampling is not None and cstm_sampling.active
            ),
            "cstm/substrate_duty_probability": (
                1.0
                if cstm_sampling is None
                else cstm_sampling.duty_probability
            ),
            "cstm/inclusion_probability_min": (
                1.0
                if cstm_sampling is None or not cstm_sampling.active
                else min(
                    cstm_sampling.inclusion_probability,
                    (
                        cstm_predictor_sampling.inclusion_probability
                        if cstm_predictor_sampling is not None
                        else 1.0
                    ),
                )
            ),
            "cstm/inclusion_weight_max": (
                1.0
                if cstm_sampling is None or not cstm_sampling.active
                else max(
                    cstm_sampling.inverse_probability,
                    predictor_inverse,
                    cstm_row_inverse_probability_max,
                )
            ),
            "cstm/row_inclusion_probability_min": (
                cstm_row_inclusion_probability_min
            ),
            "cstm/row_inclusion_weight_max": (
                cstm_row_inverse_probability_max
            ),
            "cstm/actual_target_views": float(cstm_valid_rows),
            "cstm/estimated_dense_target_views": (
                cstm_estimated_target_views
            ),
            "cstm/actual_token_participations": float(
                cstm_token_participations
            ),
            "cstm/estimated_dense_token_participations": (
                cstm_estimated_token_participations
            ),
            "cstm/target_participation_budget": float(
                self.config.cstm_target_participation_budget
            ),
            "cstm/max_substrate_vjps": float(
                self.config.cstm_max_substrate_vjps
            ),
            "cstm/predictor_update_interval": float(
                self.config.cstm_predictor_update_interval
            ),
            "cstm/coverage_gap_max": float(coverage_gap),
            "cstm/predictor_updates_total": float(
                self.cstm_coverage.predictor_updates
            ),
            "cstm/substrate_updates_total": float(
                self.cstm_coverage.substrate_updates
            ),
            "cstm/selected_invocation": float(
                -1
                if cstm_sampling is None
                or cstm_sampling.obligation is None
                else cstm_sampling.obligation.invocation
            ),
            "cstm/selected_scale": float(
                -1
                if cstm_sampling is None
                or cstm_sampling.obligation is None
                else cstm_sampling.obligation.scale
            ),
            "cstm/substrate_vjp_count": float(
                cstm_substrate_vjp_count
            ),
            "cstm/predictor_backward_seconds": (
                cstm_predictor_backward_seconds
            ),
            "cstm/substrate_backward_seconds": (
                cstm_substrate_backward_seconds
            ),
            "cstm/auxiliary_time_fraction": (
                auxiliary_seconds / max(measured_kernel_seconds, 1e-30)
            ),
            "performance/model_forward_seconds": model_forward_seconds,
            "performance/primary_forward_seconds": model_forward_seconds,
            "performance/loss_forward_seconds": loss_forward_seconds,
            "performance/backward_seconds": backward_seconds,
            "performance/primary_backward_seconds": (
                primary_backward_seconds
            ),
            "performance/carrier_custom_affine_adjoint": 1.0,
            "performance/carrier_custom_simplex_adjoint": 1.0,
            "performance/carrier_whole_span_checkpoint": float(
                self.model.cognitive.carrier._last_composite_receipt
                is not None
            ),
            "softmax/training/exact_full_vocabulary": 1.0,
            "softmax/training/external_cce_available": float(
                self.runtime["cut_cross_entropy_available"]
            ),
            "softmax/training/compiled_cce_fits_workspace": float(
                self.runtime["compiled_cce_fits_workspace"]
            ),
            "softmax/training/compiled_cce_runtime_quarantined": float(
                self.runtime["compiled_cce_runtime_quarantined"]
            ),
            "softmax/training/compiled_cce_runtime_fallbacks": float(
                self.runtime["compiled_cce_runtime_fallbacks"]
            ),
            "softmax/training/estimated_full_logits_mib": (
                self.runtime["estimated_fused_loss_bytes"] / (1 << 20)
            ),
            "softmax/training/backend_id": float(
                {
                    "tiled": 0,
                    "fused": 1,
                    "torch_compile": 2,
                    "cce_kahan_full_c": 3,
                    "cce_exact": 4,
                    "mlx": 5,
                }[self._exact_loss_backend]
            ),
            "softmax/training/mlx_peak_memory_mib": (
                self.runtime["mlx_peak_memory_bytes"] / (1 << 20)
            ),
            "softmax/training/mlx_cache_memory_mib": (
                self.runtime["mlx_cache_memory_bytes"] / (1 << 20)
            ),
        }
        for scale, rows in enumerate(cstm_scale_rows):
            metrics[f"cstm/scale/{scale}/valid_rows"] = float(rows)
        for horizon, rows in sorted(cstm_horizon_rows.items()):
            metrics[f"cstm/horizon/{horizon}/valid_rows"] = float(rows)
        if self.config.phase_transition_telemetry:
            if not proposal_logit_rows or not end_logit_rows:
                raise RuntimeError("document-major path omitted event phase logits")
            proposal_logits, end_logits = _concatenate_event_phase_logits(
                proposal_logit_rows, end_logit_rows,
            )
            self._phase_update_proposal_logits.append(proposal_logits.cpu())
            self._phase_update_end_logits.append(end_logits.cpu())
            metrics.update(
                event_phase_metrics(
                    proposal_logits,
                    end_logits,
                    proposal_threshold_logit=(
                        self.model.cognitive.event_extractor.proposal_logit
                    ),
                )
            )
        return metrics

    def _run_context(
        self, batch: PackedBatch, *, gradient_divisor: int | None = None,
    ) -> dict[str, float]:
        gradient_divisor = (
            self.config.gradient_accumulation_steps
            if gradient_divisor is None else gradient_divisor
        )
        if gradient_divisor <= 0:
            raise ValueError("gradient divisor must be positive")
        if self.integrated_cognitive_path:
            if self.document_batch_planner is not None:
                return self._run_document_major_context(
                    batch, gradient_divisor=gradient_divisor,
                )
            return self._run_integrated_context(
                batch, gradient_divisor=gradient_divisor,
            )
        batch = batch.to(self.device, non_blocking=self.device.type == "cuda")
        total_valid = int(batch.loss_mask.sum())
        if total_valid <= 0:
            raise ValueError("packed context has no within-document next-token targets")
        if self.trainer_mode.permits_persistence and any(
            key is None for key in batch.continuity_keys
        ):
            raise ValueError(
                "persistent trainer batches require explicit continuity keys"
            )
        reuse = (
            self.trainer_mode.permits_persistence
            and self._last_runtime is not None and self._last_ledger is not None
            and self._last_continuity_keys == batch.continuity_keys
            and self._last_runtime.batch == batch.input_ids.shape[0]
        )
        if reuse:
            ledger = self._last_ledger
            runtime_state = self._last_runtime
        else:
            ledger = ProvenanceLedger()
            runtime_state = self.model.cognitive.initial_state(
                batch.input_ids.shape[0],
                sample_intervals=torch.ones(
                    batch.input_ids.shape[0], device=self.device,
                    dtype=self.model.token_embedding.weight.dtype,
                ),
                device=self.device, dtype=self.model.token_embedding.weight.dtype,
            )
            runtime_state = replace(
                runtime_state,
                clocks=CognitiveClocks(0, 0, self.state.step),
            )
        boundary_classes = batch.boundary_classes.clone()
        if reuse and boundary_classes.shape[1]:
            boundary_classes[:, 0] = 0
        head = self.model.cognitive.carrier.output_head
        nll_sum = 0.0
        byte_count = 0
        active_families: set[ObjectiveFamily] = set()
        auxiliary_metrics: dict[str, float] = {}
        state_rms_max = 0.0
        span_loss: Tensor | None = None
        chunks_in_span = 0
        for start in range(0, batch.input_ids.shape[1], self.config.execution_chunk_size):
            end = min(batch.input_ids.shape[1], start + self.config.execution_chunk_size)
            with self._autocast():
                output = self.model(
                    batch.input_ids[:, start:end],
                    segment_ids=batch.segment_ids[:, start:end],
                    boundary_classes=boundary_classes[:, start:end],
                    source_uris=batch.external_source_uris,
                    ledger=ledger, state=runtime_state, project_output=False,
                )
                local_mask = batch.loss_mask[:, start:end]
                if bool(local_mask.any()):
                    statistics = self._language_statistics(
                        output.cognitive.output_latent,
                        batch.labels[:, start:end],
                        batch.target_byte_lengths[:, start:end],
                        local_mask, head,
                    )
                else:
                    zero = output.cognitive.output_latent.sum() * 0
                    statistics = TiledCrossEntropy(zero, zero, 0, 0)
                auxiliary, local_metrics, families = self._auxiliary_loss(
                    output, batch, start, end
                )
                state_penalty, state_max = _runtime_state_energy(
                    output.cognitive.state, self.config.state_target_rms
                )
                chunk_loss = statistics.nll_sum / total_valid
                chunk_loss = chunk_loss + auxiliary / ceil(
                    batch.input_ids.shape[1] / self.config.execution_chunk_size
                )
                chunk_loss = chunk_loss + (
                    self.config.state_regularization_weight * state_penalty
                    / ceil(batch.input_ids.shape[1] / self.config.execution_chunk_size)
                )
            runtime_state = output.cognitive.state
            span_loss = chunk_loss if span_loss is None else span_loss + chunk_loss
            chunks_in_span += end - start
            nll_sum += float(statistics.nll_sum.detach().cpu())
            byte_count += statistics.byte_count
            state_rms_max = max(state_rms_max, float(state_max.detach().cpu()))
            active_families |= families
            auxiliary_metrics.update(local_metrics)
            at_boundary = chunks_in_span >= self.config.tbptt_length or end == batch.input_ids.shape[1]
            if at_boundary:
                if end == batch.input_ids.shape[1] and self.schedule.weight(ObjectiveFamily.SPECTRAL_SUBSTRATE):
                    spectral = spectral_activation_regularization(self._spectral_modules)
                    span_loss = span_loss + self.config.spectral_regularization_weight * spectral
                if span_loss is None or not bool(torch.isfinite(span_loss)):
                    raise FloatingPointError("MRCRA training loss became non-finite")
                scaled = span_loss / gradient_divisor
                if self.scaler is None:
                    scaled.backward()
                else:
                    self.scaler.scale(scaled).backward()
                runtime_state = runtime_state.detach()
                span_loss = None
                chunks_in_span = 0
        missing = set(ObjectiveFamily(value) for value in self.config.required_auxiliary_families) - active_families
        if missing:
            raise ValueError(f"supervision provider omitted required objective families: {sorted(x.name for x in missing)}")
        self._last_runtime, self._last_ledger = runtime_state, ledger
        self._last_continuity_keys = (
            batch.continuity_keys if self.trainer_mode.permits_persistence else None
        )
        metrics = {
            "train/nll_sum": nll_sum,
            "train/cross_entropy_nats_per_token": nll_sum / total_valid,
            "train/effective_cross_entropy_nats_per_byte": nll_sum / max(1, byte_count),
            "train/bits_per_byte": nll_sum / max(1, byte_count) / log(2),
            "architecture/state_rms_max": state_rms_max,
            "train/valid_targets": float(total_valid),
            "train/utf8_bytes": float(byte_count),
            "softmax/training/exact_full_vocabulary": 1.0,
            "softmax/training/external_cce_available": float(
                self.runtime["cut_cross_entropy_available"]
            ),
            "softmax/training/compiled_cce_fits_workspace": float(
                self.runtime["compiled_cce_fits_workspace"]
            ),
            "softmax/training/compiled_cce_runtime_quarantined": float(
                self.runtime["compiled_cce_runtime_quarantined"]
            ),
            "softmax/training/compiled_cce_runtime_fallbacks": float(
                self.runtime["compiled_cce_runtime_fallbacks"]
            ),
            "softmax/training/estimated_full_logits_mib": (
                self.runtime["estimated_fused_loss_bytes"] / (1 << 20)
            ),
            "softmax/training/backend_id": float({
                "tiled": 0,
                "fused": 1,
                "torch_compile": 2,
                "cce_kahan_full_c": 3,
                "cce_exact": 4,
                "mlx": 5,
            }[self._exact_loss_backend]),
            "softmax/training/mlx_peak_memory_mib": (
                self.runtime["mlx_peak_memory_bytes"] / (1 << 20)
            ),
            "softmax/training/mlx_cache_memory_mib": (
                self.runtime["mlx_cache_memory_bytes"] / (1 << 20)
            ),
        }
        metrics.update(cognitive_metrics(output.cognitive, ledger))
        metrics.update(auxiliary_metrics)
        return metrics

    @torch.no_grad()
    def _evaluate_integrated_arm(
        self,
        batches: Sequence[PackedBatch],
        *,
        cognition_mode: str,
    ) -> dict[str, float]:
        """Evaluate one deterministic cognition arm on an explicit retained set."""

        nll_sum = 0.0
        valid_targets = 0
        byte_count = 0
        cognitive_cycles = 0
        event_count = 0
        started = perf_counter()
        head = self.model.cognitive.carrier.output_head
        for retained in batches:
            batch = retained.to(
                self.device, non_blocking=self.device.type == "cuda"
            )
            training_state = None
            ledger: ProvenanceLedger | None = None
            group_latents: list[Tensor] = []
            group_labels: list[Tensor] = []
            group_bytes: list[Tensor] = []
            group_masks: list[Tensor] = []
            group_tokens = 0

            def flush_evaluation_group() -> None:
                nonlocal nll_sum, valid_targets, byte_count, group_tokens
                if not group_latents:
                    return
                statistics = self._language_statistics(
                    torch.cat(group_latents, 1),
                    torch.cat(group_labels, 1),
                    torch.cat(group_bytes, 1),
                    torch.cat(group_masks, 1),
                    head,
                    checkpoint_tiles=False,
                )
                nll_sum += float(statistics.nll_sum.cpu())
                valid_targets += statistics.token_count
                byte_count += statistics.byte_count
                group_latents.clear()
                group_labels.clear()
                group_bytes.clear()
                group_masks.clear()
                group_tokens = 0

            spans = self._integrated_spans(batch)
            for span_index, (start, end, reset) in enumerate(spans):
                if reset:
                    training_state = None
                    ledger = ProvenanceLedger()
                if (
                    group_tokens
                    and group_tokens + (end - start) > self.config.tbptt_length
                ):
                    flush_evaluation_group()
                if ledger is None:
                    raise RuntimeError(
                        "integrated evaluation span omitted its provenance ledger"
                    )
                local_mask = batch.loss_mask[:, start:end]
                with self._autocast():
                    packet, ledger = self.model.prepare_external_input(
                        batch.input_ids[:, start:end],
                        attention_mask=torch.ones_like(local_mask),
                        segment_ids=batch.segment_ids[:, start:end],
                        boundary_classes=batch.boundary_classes[:, start:end],
                        source_uris=batch.external_source_uris,
                        ledger=ledger,
                        continuing=training_state is not None,
                        timestamp_offset=start,
                    )
                    output = self.model.cognitive.forward_integrated_training(
                        packet, ledger, state=training_state,
                        cognitive_stride=self.config.cognitive_stride,
                        cognitive_tbptt_events=self.config.cognitive_tbptt_events,
                        cognition_mode=cognition_mode,
                    )
                group_latents.append(output.output_latent)
                group_labels.append(batch.labels[:, start:end])
                group_bytes.append(batch.target_byte_lengths[:, start:end])
                group_masks.append(local_mask)
                group_tokens += end - start
                training_state = output.state.detach()
                cognitive_cycles += int(output.cognitive_cycles.sum().cpu())
                event_count += int(output.event_counts.sum().cpu())
                if (
                    group_tokens >= self.config.tbptt_length
                    or span_index == len(spans) - 1
                ):
                    flush_evaluation_group()
        _synchronize(self.device)
        return {
            "nll_sum": nll_sum,
            "valid_targets": float(valid_targets),
            "utf8_bytes": float(byte_count),
            "cross_entropy_nats_per_token": nll_sum / max(1, valid_targets),
            "effective_cross_entropy_nats_per_byte": nll_sum / max(1, byte_count),
            "bits_per_byte": nll_sum / max(1, byte_count) / log(2),
            "cognitive_cycles": float(cognitive_cycles),
            "events": float(event_count),
            "seconds": perf_counter() - started,
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Measure exact held-out likelihood without mutating training state.

        Every retained context receives a fresh runtime and provenance ledger;
        evaluation therefore cannot leak state between held-out documents or
        into the persistent training stream.  RNG state and model mode are
        restored exactly even when evaluation fails.
        """

        if not self.evaluation_batches:
            raise ValueError("no retained evaluation batches are configured")
        was_training = self.model.training
        cpu_rng = torch.random.get_rng_state()
        accelerator_rng = None
        if self.device.type == "cuda":
            accelerator_rng = torch.cuda.get_rng_state(self.device)
        elif self.device.type == "mps":
            accelerator_rng = torch.mps.get_rng_state()
        started = perf_counter()
        nll_sum = 0.0
        valid_targets = 0
        byte_count = 0
        cognitive_cycles = 0
        event_count = 0
        try:
            self.model.eval()
            head = self.model.cognitive.carrier.output_head
            if self.integrated_cognitive_path:
                full = self._evaluate_integrated_arm(
                    self.evaluation_batches, cognition_mode="full"
                )
                _synchronize(self.device)
                elapsed = perf_counter() - started
                metrics = {
                    "eval/nll_sum": full["nll_sum"],
                    "eval/cross_entropy_nats_per_token": full[
                        "cross_entropy_nats_per_token"
                    ],
                    "eval/effective_cross_entropy_nats_per_byte": full[
                        "effective_cross_entropy_nats_per_byte"
                    ],
                    "eval/bits_per_byte": full["bits_per_byte"],
                    "eval/valid_targets": full["valid_targets"],
                    "eval/utf8_bytes": full["utf8_bytes"],
                    "eval/seconds": elapsed,
                    "eval/tokens_per_second": full["valid_targets"] / max(elapsed, 1e-9),
                    "eval/cognitive_cycles": full["cognitive_cycles"],
                    "eval/events": full["events"],
                    "eval/integrated_cognitive_path": 1.0,
                }
                if self.config.phase_transition_ablation:
                    retained = self.evaluation_batches[
                        : self.config.phase_transition_ablation_batches
                    ]
                    arms = {
                        mode: self._evaluate_integrated_arm(
                            retained, cognition_mode=mode
                        )
                        for mode in ("full", "soft_only", "off")
                    }
                    prefix = "eval/phase_ablation"
                    for mode, values in arms.items():
                        label = "cognition_off" if mode == "off" else mode
                        metrics.update({
                            f"{prefix}/{label}_ce_nats_per_token": values[
                                "cross_entropy_nats_per_token"
                            ],
                            f"{prefix}/{label}_ece_nats_per_byte": values[
                                "effective_cross_entropy_nats_per_byte"
                            ],
                            f"{prefix}/{label}_seconds": values["seconds"],
                        })
                    metrics.update({
                        f"{prefix}/valid_targets": arms["full"]["valid_targets"],
                        f"{prefix}/hard_structure_ce_gain": (
                            arms["soft_only"]["cross_entropy_nats_per_token"]
                            - arms["full"]["cross_entropy_nats_per_token"]
                        ),
                        f"{prefix}/soft_bridge_ce_gain": (
                            arms["off"]["cross_entropy_nats_per_token"]
                            - arms["soft_only"]["cross_entropy_nats_per_token"]
                        ),
                        f"{prefix}/hard_structure_ece_gain": (
                            arms["soft_only"][
                                "effective_cross_entropy_nats_per_byte"
                            ]
                            - arms["full"][
                                "effective_cross_entropy_nats_per_byte"
                            ]
                        ),
                        f"{prefix}/soft_bridge_ece_gain": (
                            arms["off"]["effective_cross_entropy_nats_per_byte"]
                            - arms["soft_only"][
                                "effective_cross_entropy_nats_per_byte"
                            ]
                        ),
                    })
                if not all(isfinite(value) for value in metrics.values()):
                    raise FloatingPointError(
                        "MRCRA integrated evaluation metrics became non-finite"
                    )
                return metrics
            for retained in self.evaluation_batches:
                batch = retained.to(
                    self.device, non_blocking=self.device.type == "cuda"
                )
                ledger = ProvenanceLedger()
                runtime_state = self.model.cognitive.initial_state(
                    batch.input_ids.shape[0],
                    sample_intervals=torch.ones(
                        batch.input_ids.shape[0], device=self.device,
                        dtype=self.model.token_embedding.weight.dtype,
                    ),
                    device=self.device,
                    dtype=self.model.token_embedding.weight.dtype,
                )
                runtime_state = replace(
                    runtime_state,
                    clocks=CognitiveClocks(0, 0, self.state.step),
                )
                for start in range(
                    0, batch.input_ids.shape[1], self.config.execution_chunk_size
                ):
                    end = min(
                        batch.input_ids.shape[1],
                        start + self.config.execution_chunk_size,
                    )
                    with self._autocast():
                        output = self.model(
                            batch.input_ids[:, start:end],
                            segment_ids=batch.segment_ids[:, start:end],
                            boundary_classes=batch.boundary_classes[:, start:end],
                            source_uris=batch.external_source_uris,
                            ledger=ledger, state=runtime_state,
                            project_output=False,
                        )
                        local_mask = batch.loss_mask[:, start:end]
                        if bool(local_mask.any()):
                            statistics = self._language_statistics(
                                output.cognitive.output_latent,
                                batch.labels[:, start:end],
                                batch.target_byte_lengths[:, start:end],
                                local_mask, head,
                                checkpoint_tiles=False,
                            )
                            nll_sum += float(statistics.nll_sum.cpu())
                            valid_targets += statistics.token_count
                            byte_count += statistics.byte_count
                    runtime_state = output.cognitive.state
                    cognitive_cycles += int(output.cognitive.cognitive_cycles.sum())
                    event_count += int(output.cognitive.event_counts.sum())
            _synchronize(self.device)
            elapsed = perf_counter() - started
            metrics = {
                "eval/nll_sum": nll_sum,
                "eval/cross_entropy_nats_per_token": nll_sum / max(1, valid_targets),
                "eval/effective_cross_entropy_nats_per_byte": nll_sum / max(1, byte_count),
                "eval/bits_per_byte": nll_sum / max(1, byte_count) / log(2),
                "eval/valid_targets": float(valid_targets),
                "eval/utf8_bytes": float(byte_count),
                "eval/seconds": elapsed,
                "eval/tokens_per_second": valid_targets / max(elapsed, 1e-9),
                "eval/cognitive_cycles": float(cognitive_cycles),
                "eval/events": float(event_count),
            }
            if not all(isfinite(value) for value in metrics.values()):
                raise FloatingPointError("MRCRA evaluation metrics became non-finite")
            return metrics
        finally:
            torch.random.set_rng_state(cpu_rng)
            if accelerator_rng is not None and self.device.type == "cuda":
                torch.cuda.set_rng_state(accelerator_rng, self.device)
            elif accelerator_rng is not None and self.device.type == "mps":
                torch.mps.set_rng_state(accelerator_rng.cpu())
            self.model.train(was_training)

    @torch.no_grad()
    def evaluate_progress_probe(self) -> dict[str, float]:
        """Measure the disjoint PC-RASL progress stream without state leakage."""

        if self.learning_progress is None or not self.progress_probe_batches:
            raise ValueError("Progress-Conditioned RASL has no progress-probe authority")
        if not self.integrated_cognitive_path:
            raise RuntimeError("progress probes require the integrated cognitive path")
        was_training = self.model.training
        cpu_rng = torch.random.get_rng_state()
        accelerator_rng = None
        if self.device.type == "cuda":
            accelerator_rng = torch.cuda.get_rng_state(self.device)
        elif self.device.type == "mps":
            accelerator_rng = torch.mps.get_rng_state()
        try:
            self.model.eval()
            values = self._evaluate_integrated_arm(
                self.progress_probe_batches, cognition_mode="full"
            )
            return {
                "pc_rasl/probe_nll_sum": values["nll_sum"],
                "pc_rasl/probe_ce_nats_per_token": values[
                    "cross_entropy_nats_per_token"
                ],
                "pc_rasl/probe_ece_nats_per_byte": values[
                    "effective_cross_entropy_nats_per_byte"
                ],
                "pc_rasl/probe_bits_per_byte": values["bits_per_byte"],
                "pc_rasl/probe_valid_targets": values["valid_targets"],
                "pc_rasl/probe_utf8_bytes": values["utf8_bytes"],
                "pc_rasl/probe_seconds": values["seconds"],
                "pc_rasl/probe_cognitive_cycles": values["cognitive_cycles"],
                "pc_rasl/probe_events": values["events"],
            }
        finally:
            torch.random.set_rng_state(cpu_rng)
            if accelerator_rng is not None and self.device.type == "cuda":
                torch.cuda.set_rng_state(accelerator_rng, self.device)
            elif accelerator_rng is not None and self.device.type == "mps":
                torch.mps.set_rng_state(accelerator_rng.cpu())
            self.model.train(was_training)

    def _record_evaluation(self, metrics: dict[str, float]) -> None:
        """Persist held-out evidence even when Trackio is intentionally off."""

        if not metrics or not all(isfinite(value) for value in metrics.values()):
            raise ValueError("evaluation evidence must be nonempty and finite")
        destination = Path(self.config.output_dir) / "evaluation_metrics.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": 2,
            "step": self.state.step,
            "tokens_seen": self.state.tokens_seen,
            "evaluation_identity": self.evaluation_identity,
            "metrics": metrics,
        }
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.state.last_evaluation_step = self.state.step
        self.state.last_evaluation_metrics = dict(metrics)

    def _record_progress_observation(
        self,
        probe_metrics: dict[str, float],
        report: LearningProgressReport,
    ) -> None:
        """Persist the complete causal progress observation independent of Trackio."""

        values = {**probe_metrics, **LearningProgressAuthority.metrics(report)}
        if not values or not all(isfinite(value) for value in values.values()):
            raise ValueError("progress evidence must be nonempty and finite")
        destination = Path(self.config.output_dir) / "progress_metrics.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": 1,
            "step": self.state.step,
            "tokens_seen": self.state.tokens_seen,
            "valid_targets_seen": self.state.valid_targets_seen,
            "progress_probe_identity": self.progress_probe_identity,
            "guard_evaluation_identity": self.evaluation_identity,
            "metrics": values,
        }
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.state.last_progress_observation_step = self.state.step
        self.state.last_progress_pressure = report.pressure
        self.state.progress_observations = report.observation_index

    def _update_phase_transition_metrics(
        self,
        metrics: dict[str, float],
        gradient: GradientReport,
    ) -> bool:
        """Attach exact update-level phase and gradient-participation evidence."""

        if self.config.phase_transition_telemetry and self.integrated_cognitive_path:
            if (
                not self._phase_update_proposal_logits
                or not self._phase_update_end_logits
            ):
                raise RuntimeError("phase telemetry was enabled but no logits were retained")
            proposal_logits, end_logits = _concatenate_event_phase_logits(
                self._phase_update_proposal_logits,
                self._phase_update_end_logits,
            )
            metrics.update(event_phase_metrics(
                proposal_logits,
                end_logits,
                proposal_threshold_logit=(
                    self.model.cognitive.event_extractor.proposal_logit
                ),
            ))
            current = metrics["architecture/event_proposal_logit_max"]
            previous = self.state.last_event_proposal_logit_max
            instantaneous = 0.0 if previous is None else current - previous
            if self.state.event_proposal_observations == 0:
                rolling = instantaneous
            else:
                decay = self.config.proposal_slope_ema_decay
                rolling = (
                    decay * self.state.event_proposal_logit_slope_ema
                    + (1 - decay) * instantaneous
                )
            self.state.last_event_proposal_logit_max = current
            self.state.event_proposal_logit_slope_ema = rolling
            self.state.event_proposal_observations += 1
            metrics.update({
                "architecture/event_proposal_logit_slope": instantaneous,
                "architecture/event_proposal_logit_slope_ema": rolling,
            })
        self._phase_update_proposal_logits.clear()
        self._phase_update_end_logits.clear()

        gradient_tensor_count = sum(gradient.subsystem_tensor_counts.values())
        for subsystem, before in gradient.subsystem_norms_before.items():
            metrics[
                f"optimization/gradient/{subsystem}_before_clip"
            ] = float(before.detach().cpu())
            metrics[
                f"optimization/gradient/{subsystem}_after_clip"
            ] = float(
                gradient.subsystem_norms_after[subsystem].detach().cpu()
            )
            metrics[
                f"optimization/gradient/{subsystem}_tensor_fraction"
            ] = (
                gradient.subsystem_tensor_counts[subsystem]
                / max(1, gradient_tensor_count)
            )
        coefficient = float(gradient.clip_coefficient.detach().cpu())
        metrics["optimization/effective_learning_rate"] = (
            metrics["optimization/learning_rate"] * coefficient
        )
        if coefficient < self.config.low_clip_coefficient_threshold:
            self.state.low_clip_coefficient_steps += 1
        else:
            self.state.low_clip_coefficient_steps = 0
        metrics["optimization/low_clip_coefficient_consecutive_steps"] = float(
            self.state.low_clip_coefficient_steps
        )
        return (
            self.state.low_clip_coefficient_steps
            == self.config.low_clip_coefficient_patience
        )

    def _alert_low_clip_pressure(
        self, reporter: TrackioReporter | _NullReporter,
    ) -> None:
        """Emit the warning only after the current step has been logged."""

        if (
            self.state.low_clip_coefficient_steps
            == self.config.low_clip_coefficient_patience
        ):
            reporter.alert(
                "Sustained strong gradient clipping",
                (
                    f"Clip coefficient remained below "
                    f"{self.config.low_clip_coefficient_threshold:.3f} for "
                    f"{self.config.low_clip_coefficient_patience} consecutive "
                    "updates. Inspect per-subsystem gradient participation and "
                    "held-out loss before changing optimizer policy."
                ),
                level="warn", step=self.state.step,
            )

    @staticmethod
    def _trace_number(value: Tensor) -> float | int:
        item = value.detach().cpu().item()
        return int(item) if not value.dtype.is_floating_point else float(item)

    def _record_first_hard_event(
        self,
        reporter: TrackioReporter | _NullReporter,
        metrics: dict[str, float],
        gradient: GradientReport,
    ) -> Path | None:
        """Atomically retain and announce the first hard-event phase crossing."""

        trace = self._pending_first_hard_event_trace
        if (
            not self.config.phase_transition_telemetry
            or trace is None
            or self.state.first_hard_event_step
        ):
            return None
        checkpoint_name = (
            f"phase-transition-first-event-step-{self.state.step:07d}.pt"
        )
        self.state.first_hard_event_step = self.state.step
        self.state.first_hard_event_tokens = self.state.tokens_seen
        self.state.first_hard_event_checkpoint = checkpoint_name
        destination = (
            Path(self.config.output_dir)
            / "diagnostics"
            / "first-hard-event.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "kind": "mrcra_first_hard_event",
            "step": self.state.step,
            "tokens_seen": self.state.tokens_seen,
            "checkpoint": checkpoint_name,
            "thresholds": {
                "proposal_logit": (
                    self.model.cognitive.event_extractor.proposal_logit
                ),
                "proposal_probability": float(torch.sigmoid(torch.tensor(
                    self.model.cognitive.event_extractor.proposal_logit
                ))),
                "end_logit": self.model.cognitive.event_extractor.end_logit,
            },
            "trigger": {
                "document_order": trace.batch_index,
                "anchor_index": trace.anchor_index,
                "timestamp": self._trace_number(trace.timestamp),
                "proposal_logit": self._trace_number(trace.proposal_logit),
                "proposal_probability": float(
                    torch.sigmoid(trace.proposal_logit.float()).cpu()
                ),
                "end_logit": self._trace_number(trace.end_logit),
                "end_probability": float(
                    torch.sigmoid(trace.end_logit.float()).cpu()
                ),
                "event_type": self._trace_number(trace.event_type),
                "confidence": self._trace_number(trace.confidence),
                "support": [
                    float(value) for value in trace.support.detach().cpu()
                ],
            },
            "state_transition": {
                "active_nodes": {
                    "before": self._trace_number(trace.active_nodes_before),
                    "after": self._trace_number(trace.active_nodes_after),
                },
                "active_relations": {
                    "before": self._trace_number(trace.active_relations_before),
                    "after": self._trace_number(trace.active_relations_after),
                },
                "workspace_residents": {
                    "before": self._trace_number(trace.workspace_before),
                    "after": self._trace_number(trace.workspace_after),
                },
            },
            "gradient_transition": {
                "global_before_clip": float(
                    gradient.total_before_clip.detach().cpu()
                ),
                "global_after_clip": float(
                    gradient.total_after_clip.detach().cpu()
                ),
                "clip_coefficient": float(
                    gradient.clip_coefficient.detach().cpu()
                ),
                "effective_learning_rate": metrics[
                    "optimization/effective_learning_rate"
                ],
                "subsystems": {
                    name: {
                        "before_clip": float(value.detach().cpu()),
                        "after_clip": float(
                            gradient.subsystem_norms_after[name].detach().cpu()
                        ),
                        "tensor_count": gradient.subsystem_tensor_counts[name],
                    }
                    for name, value in gradient.subsystem_norms_before.items()
                },
            },
            "phase": {
                key: value for key, value in metrics.items()
                if key.startswith("architecture/event_")
            },
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="first-hard-event-", suffix=".tmp",
            dir=destination.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        finally:
            Path(temporary_name).unlink(missing_ok=True)
        checkpoint = self.save_checkpoint(phase_transition=True)
        if checkpoint.name != checkpoint_name:
            raise RuntimeError("phase-transition checkpoint name is inconsistent")
        reporter.log_phase_transition_trace(destination, step=self.state.step)
        reporter.alert(
            "First MRCRA hard event",
            (
                f"Proposal crossed the hard allocation boundary at token "
                f"{self.state.tokens_seen:,}; retained {checkpoint.name} and "
                f"{destination.name}."
            ),
            level="info", step=self.state.step,
        )
        self._pending_first_hard_event_trace = None
        return checkpoint

    def train(
        self,
        *,
        maximum_steps: int | None = None,
        step_observer: Callable[
            [MRCRATrainingState, Mapping[str, float]], None
        ]
        | None = None,
    ) -> MRCRATrainingState:
        if maximum_steps is not None and maximum_steps <= 0:
            raise ValueError("maximum_steps must be positive")
        continuing = self._resumed or self.state.step > 0
        if not continuing:
            torch.manual_seed(self.config.seed)
        reporter = _NonAuthoritativeReporter(
            (
                TrackioReporter(
                    self.config,
                    self._identity(),
                    resume=continuing,
                    initial_step=self.state.step,
                    initial_tokens=self.state.tokens_seen,
                )
                if self.config.trackio_enabled else _NullReporter()
            ),
            self.config.output_dir,
        )
        started = perf_counter()
        completed_this_call = 0
        failed = True
        try:
            while self.state.tokens_seen < self.config.total_tokens:
                if maximum_steps is not None and completed_this_call >= maximum_steps:
                    break
                self.model.train()
                step_started = perf_counter()
                self.optimizer.zero_grad(set_to_none=True)
                self._cstm_auxiliary_gradients.clear()
                pc_rasl_seconds = 0.0
                if self.pc_rasl is not None:
                    pc_rasl_started = perf_counter()
                    self._prepare_pc_rasl_gradients()
                    pc_rasl_seconds = perf_counter() - pc_rasl_started
                self._phase_update_proposal_logits.clear()
                self._phase_update_end_logits.clear()
                aggregated: dict[str, float] = {}
                tokens_this_update = 0
                data_seconds = 0.0
                pc_rasl_capture_seconds = 0.0
                pc_rasl_capture_due = (
                    self.pc_rasl is not None
                    and self._pc_rasl_capture_due(self.state.step + 1)
                )
                pc_rasl_captured = False
                behavior_batch: PackedBatch | None = None
                contexts_this_update = min(
                    self.config.gradient_accumulation_steps,
                    ceil((self.config.total_tokens - self.state.tokens_seen) / self.config.context_length),
                )
                batches_for_update: list[PackedBatch] = []
                for _ in range(contexts_this_update):
                    remaining = self.config.total_tokens - self.state.tokens_seen - tokens_this_update
                    if remaining <= 0:
                        break
                    # A final partial context is legal and consumes exactly the
                    # declared token budget.
                    requested = min(self.config.context_length, remaining)
                    if self.state.step == 0 and tokens_this_update == 0:
                        print(
                            f"Preparing first {requested:,}-token packed FineWeb "
                            "context...",
                            flush=True,
                        )
                    data_started = perf_counter()
                    batch = self._next_training_batch(
                        self.config.micro_batch_size, requested
                    )
                    data_seconds += perf_counter() - data_started
                    anticipated_remaining = (
                        self.config.total_tokens - self.state.tokens_seen
                        - tokens_this_update - batch.token_count
                    )
                    if anticipated_remaining > 0:
                        self._schedule_prefetch(
                            self.config.micro_batch_size,
                            min(self.config.context_length, anticipated_remaining),
                        )
                    if self.state.step == 0 and tokens_this_update == 0:
                        path_name = (
                            "integrated multirate cognitive path"
                            if self.integrated_cognitive_path else "token-rate cognitive path"
                        )
                        print(
                            f"First context ready ({int(batch.loss_mask.sum()):,} "
                            f"valid targets); starting {path_name}.",
                            flush=True,
                        )
                    batches_for_update.append(batch)
                    tokens_this_update += batch.token_count

                # Fetch once, then execute the immutable packed batches. This
                # makes a pre-optimizer OOM retry causal: the data stream is
                # never advanced twice and every accumulated context can be
                # replayed under the next safer activation policy.
                tokens_this_update = 0

                def execute_update_batches() -> None:
                    nonlocal tokens_this_update, behavior_batch
                    aggregated.clear()
                    tokens_this_update = 0
                    behavior_batch = None
                    for local_batch in batches_for_update:
                        local = self._run_context(
                            local_batch,
                            gradient_divisor=contexts_this_update,
                        )
                        if self.pc_rasl is not None:
                            behavior_batch = local_batch
                        tokens_this_update += local_batch.token_count
                        for name, value in local.items():
                            aggregated[name] = (
                                aggregated.get(name, 0.0) + value
                            )

                # Only the ordinary CE/CSTM path is retryable. PC-RASL may
                # already have mutated its independent critic optimizer before
                # this point, so an OOM in that experimental mode must abort.
                retryable = (
                    self.pc_rasl is None
                    and self.activation_execution_policy.resolved
                    != "whole_span"
                )
                buffer_snapshot = (
                    {
                        name: value.detach().clone()
                        for name, value in self.model.named_buffers()
                        if name.startswith(
                            (
                                "cstm_predictor.target_second_moment",
                                "cstm_predictor.target_rms_initialized",
                            )
                        )
                    }
                    if retryable else {}
                )
                cpu_rng_snapshot = (
                    torch.random.get_rng_state().clone()
                    if retryable else None
                )
                accelerator_rng_snapshot = None
                if retryable and self.device.type == "cuda":
                    accelerator_rng_snapshot = torch.cuda.get_rng_state(
                        self.device
                    ).clone()
                elif (
                    retryable
                    and self.device.type == "mps"
                    and hasattr(torch.mps, "get_rng_state")
                ):
                    accelerator_rng_snapshot = (
                        torch.mps.get_rng_state().clone()
                    )
                coverage_snapshot = (
                    self.cstm_coverage.state_dict() if retryable else None
                )
                last_runtime_snapshot = self._last_runtime
                last_ledger_snapshot = self._last_ledger
                continuity_snapshot = self._last_continuity_keys
                try:
                    execute_update_batches()
                except BaseException as error:
                    if (
                        not retryable
                        or not self._is_recoverable_out_of_memory(error)
                        or not self._advance_activation_policy_after_oom()
                    ):
                        raise
                    self.optimizer.zero_grad(set_to_none=True)
                    self._cstm_auxiliary_gradients.clear()
                    self._phase_update_proposal_logits.clear()
                    self._phase_update_end_logits.clear()
                    named_buffers = dict(self.model.named_buffers())
                    if not set(buffer_snapshot).issubset(named_buffers):
                        raise RuntimeError(
                            "mutable model buffer topology changed during OOM recovery"
                        ) from error
                    with torch.no_grad():
                        for name, saved in buffer_snapshot.items():
                            named_buffers[name].copy_(saved)
                    if cpu_rng_snapshot is None or coverage_snapshot is None:
                        raise RuntimeError(
                            "OOM recovery snapshot was incomplete"
                        ) from error
                    torch.random.set_rng_state(cpu_rng_snapshot)
                    if accelerator_rng_snapshot is not None:
                        if self.device.type == "cuda":
                            torch.cuda.set_rng_state(
                                accelerator_rng_snapshot, self.device
                            )
                        elif self.device.type == "mps":
                            torch.mps.set_rng_state(
                                accelerator_rng_snapshot
                            )
                    self.cstm_coverage = CSTMCoverageState.from_state_dict(
                        coverage_snapshot
                    )
                    self._last_runtime = last_runtime_snapshot
                    self._last_ledger = last_ledger_snapshot
                    self._last_continuity_keys = continuity_snapshot
                    self._clear_transient_device_memory()
                    execute_update_batches()
                if pc_rasl_capture_due and behavior_batch is not None:
                    capture_started = perf_counter()
                    pc_rasl_captured = self._capture_pc_rasl_trajectory(
                        behavior_batch
                    )
                    pc_rasl_capture_seconds += perf_counter() - capture_started
                gradient_started = perf_counter()
                if self.loss_device != self.device:
                    _synchronize(self.loss_device)
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                cstm_merge_started = perf_counter()
                cstm_gradient_metrics = (
                    self._merge_cstm_gradients()
                    if self.cstm_enabled else {}
                )
                cstm_gradient_merge_seconds = (
                    perf_counter() - cstm_merge_started
                )
                pc_rasl_gradient_metrics = (
                    self._merge_pc_rasl_gradients()
                    if self.pc_rasl is not None else {}
                )
                gradient = clip_and_report_gradients(
                    self.model, maximum_norm=self.config.maximum_gradient_norm
                )
                if not gradient.finite:
                    raise FloatingPointError("MRCRA gradients became non-finite")
                gradient_seconds = perf_counter() - gradient_started
                optimizer_started = perf_counter()
                if self.scaler is None:
                    self.optimizer.step()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                self.scheduler.step()
                optimizer_seconds = perf_counter() - optimizer_started
                _synchronize(self.device)
                self.state.step += 1
                if self._last_runtime is not None:
                    self._last_runtime = replace(
                        self._last_runtime,
                        clocks=self._last_runtime.clocks.optimizer_tick(),
                    )
                completed_this_call += 1
                self.state.tokens_seen += tokens_this_update
                valid = int(aggregated.get("train/valid_targets", 0))
                byte_count = int(aggregated.get("train/utf8_bytes", 0))
                self.state.valid_targets_seen += valid
                self.state.bytes_seen += byte_count
                elapsed = perf_counter() - step_started
                divisor = max(1, contexts_this_update)
                metrics = {name: value / divisor for name, value in aggregated.items()}
                for phase_name in (
                    "performance/model_forward_seconds",
                    "performance/primary_forward_seconds",
                    "performance/loss_forward_seconds",
                    "performance/backward_seconds",
                    "performance/primary_backward_seconds",
                    "cstm/predictor_backward_seconds",
                    "cstm/substrate_backward_seconds",
                ):
                    metrics[phase_name] = aggregated.get(phase_name, 0.0)
                # Likelihood metrics and counters are additive, not averages of
                # per-context averages.  This matters for the final partial
                # context and for documents with different UTF-8 byte lengths.
                nll_sum = aggregated.get("train/nll_sum", 0.0)
                metrics.update({
                    "train/nll_sum": nll_sum,
                    "train/cross_entropy_nats_per_token": nll_sum / max(1, valid),
                    "train/effective_cross_entropy_nats_per_byte": nll_sum / max(1, byte_count),
                    "train/bits_per_byte": nll_sum / max(1, byte_count) / log(2),
                    "train/valid_targets": float(valid),
                    "train/utf8_bytes": float(byte_count),
                })
                cstm_weighted_rows = aggregated.get(
                    "cstm/weighted_prediction_rows", 0.0
                )
                for name in (
                    "cstm/standardized_huber_sum",
                    "cstm/estimated_dense_numerator",
                    "cstm/context_valid_weight",
                    "cstm/weighted_prediction_rows",
                    "cstm/spectral_target_views",
                    "cstm/coefficient_targets",
                    "cstm/raw_token_view_equivalents",
                ):
                    if name in aggregated:
                        metrics[name] = aggregated[name]
                if cstm_weighted_rows:
                    metrics["cstm/standardized_huber"] = (
                        aggregated.get("cstm/standardized_huber_sum", 0.0)
                        / cstm_weighted_rows
                    )
                if aggregated.get("cstm/context_valid_weight", 0.0):
                    metrics["cstm/estimated_dense_standardized_huber"] = (
                        aggregated.get(
                            "cstm/estimated_dense_numerator", 0.0
                        )
                        / aggregated["cstm/context_valid_weight"]
                    )
                metrics["cstm/supervision_relations_per_primary_target"] = (
                    aggregated.get("cstm/raw_token_view_equivalents", 0.0)
                    / max(1, valid)
                )
                metrics.update({
                    "progress/step": float(self.state.step),
                    "progress/tokens_seen": float(self.state.tokens_seen),
                    "progress/valid_targets_seen": float(self.state.valid_targets_seen),
                    "progress/utf8_bytes_seen": float(self.state.bytes_seen),
                    "progress/fraction": min(1.0, self.state.tokens_seen / self.config.total_tokens),
                    "training/cognitive_stride": float(
                        self.config.cognitive_stride
                    ),
                    "training/carrier_activation_checkpointing": float(
                        self.activation_execution_policy.resolved != "retain"
                    ),
                    "training/cpu_threads": float(
                        self.runtime["cpu_threads_resolved"]
                    ),
                    "performance/step_seconds": elapsed,
                    "performance/tokens_per_second": tokens_this_update / max(elapsed, 1e-9),
                    "performance/utf8_bytes_per_second": (
                        byte_count / max(elapsed, 1e-9)
                    ),
                    "performance/data_seconds": data_seconds,
                    "performance/data_wait_seconds": data_seconds,
                    "performance/gradient_reduction_seconds": gradient_seconds,
                    "performance/optimizer_seconds": optimizer_seconds,
                    "performance/phase_timing_synchronous": float(
                        self.device.type == "cpu" and self.loss_device.type == "cpu"
                    ),
                    "optimization/learning_rate": max(group["lr"] for group in self.optimizer.param_groups),
                    "optimization/gradient_norm_before_clip": float(gradient.total_before_clip.cpu()),
                    "optimization/gradient_norm_after_clip": float(gradient.total_after_clip.cpu()),
                    "optimization/gradient_clip_coefficient": float(gradient.clip_coefficient.cpu()),
                })
                metrics["cstm/gradient_merge_seconds"] = (
                    cstm_gradient_merge_seconds
                )
                compilation = (
                    self.model.cognitive.carrier.compilation_receipt()
                )
                selected_activation_measurement = next(
                    (
                        candidate
                        for candidate in (
                            self.activation_execution_policy.candidates
                        )
                        if candidate.policy
                        == self.activation_execution_policy.resolved
                    ),
                    None,
                )
                metrics.update({
                    "execution/policy_schema_version": float(
                        self.activation_execution_policy.schema_version
                    ),
                    "execution/activation_requested": float({
                        "auto": 0,
                        "retain": 1,
                        "selective": 2,
                        "whole_span": 3,
                    }[self.activation_execution_policy.requested]),
                    "execution/activation_resolved": float({
                        "retain": 1,
                        "selective": 2,
                        "whole_span": 3,
                    }[self.activation_execution_policy.resolved]),
                    "execution/activation_peak_bytes": float(
                        0
                        if selected_activation_measurement is None
                        else (
                            selected_activation_measurement
                            .reserve_peak_bytes
                        )
                    ),
                    "execution/activation_available_bytes": float(
                        self.activation_execution_policy.memory.available_bytes
                    ),
                    "execution/activation_reserve_bytes": float(
                        self.activation_execution_policy.required_reserve_bytes
                    ),
                    "execution/compiler_requested": float(
                        -1
                        if self.config.compile_tensor_cores is None
                        else self.config.compile_tensor_cores
                    ),
                    "execution/compiler_resolved": float(
                        self.runtime["compiled_tensor_cores"]
                    ),
                    "execution/compiler_backend": float({
                        "none": 0,
                        "aot_eager": 1,
                        "inductor": 2,
                    }[self.runtime["carrier_compiler_backend"]]),
                    "execution/compiled_shape_count": compilation[
                        "compiled_shape_count"
                    ],
                    "execution/compile_seconds": compilation[
                        "compile_seconds"
                    ],
                    "execution/fallback_count": compilation[
                        "fallback_count"
                    ],
                    "execution/graph_break_count": compilation[
                        "graph_break_count"
                    ],
                    "execution/activation_oom_retries": float(
                        self._activation_oom_retries
                    ),
                })
                metrics.update(cstm_gradient_metrics)
                if self.pc_rasl is not None:
                    metrics.update({
                        "pc_rasl/configured_captures_per_observation": float(
                            self.config.pc_rasl_captures_per_observation
                        ),
                        "pc_rasl/configured_updates_per_observation": float(
                            self.config.pc_rasl_updates_per_observation
                        ),
                        "pc_rasl/trajectory_capture_due": float(
                            pc_rasl_capture_due
                        ),
                        "pc_rasl/trajectory_captured": float(
                            pc_rasl_captured
                        ),
                        "pc_rasl/trajectories_captured_total": float(
                            self.state.pc_rasl_trajectories_captured
                        ),
                        "pc_rasl/replay_updates_total": float(
                            self.state.pc_rasl_replay_updates
                        ),
                        "performance/pc_rasl_seconds": pc_rasl_seconds,
                        "performance/pc_rasl_capture_seconds": (
                            pc_rasl_capture_seconds
                        ),
                    })
                    metrics.update(self._pc_rasl_step_metrics)
                    metrics.update(pc_rasl_gradient_metrics)
                for runtime_name, metric_name in (
                    (
                        "estimated_uncheckpointed_carrier_activation_bytes",
                        "system/estimated_uncheckpointed_carrier_activation_bytes",
                    ),
                    (
                        "carrier_activation_memory_budget_bytes",
                        "system/carrier_activation_memory_budget_bytes",
                    ),
                ):
                    if runtime_name in self.runtime:
                        metrics[metric_name] = float(
                            self.runtime[runtime_name]
                        )
                attributed = sum(metrics.get(name, 0.0) for name in (
                    "performance/data_seconds",
                    "performance/pc_rasl_seconds",
                    "performance/pc_rasl_capture_seconds",
                    "performance/model_forward_seconds",
                    "performance/loss_forward_seconds",
                    "performance/backward_seconds",
                    "performance/gradient_reduction_seconds",
                    "performance/optimizer_seconds",
                ))
                metrics["performance/unattributed_step_seconds"] = max(
                    0.0, elapsed - attributed
                )
                metrics.update({
                    "performance/training_tokens_per_second": (
                        tokens_this_update / max(elapsed, 1e-9)
                    ),
                    "performance/wall_clock_tokens_per_second": (
                        tokens_this_update / max(elapsed, 1e-9)
                    ),
                    "performance/wall_clock_step_seconds": elapsed,
                    "performance/evaluation_seconds": 0.0,
                    "performance/checkpoint_seconds": 0.0,
                    "performance/snapshot_seconds": 0.0,
                })
                memory_metrics = _memory_metrics(self.device)
                metrics.update(memory_metrics)
                metrics["data/prefetch_queue_depth"] = float(
                    int(self._prefetch_future is not None)
                    + int(self._prefetched_batch is not None)
                )
                metrics["data/prefetch_worker_shared_process_rss_bytes"] = (
                    memory_metrics.get(
                        "system/process_rss_gib",
                        memory_metrics.get(
                            "system/process_peak_rss_gib", 0.0
                        ),
                    )
                    * (1 << 30)
                    if self._prefetch_executor is not None
                    else 0.0
                )
                metrics["observation/failure_count"] = float(
                    reporter.failure_count
                )
                progress_observed = False
                if (
                    self.learning_progress is not None
                    and self.state.step
                    % self.config.learning_progress.observation_interval
                    == 0
                ):
                    probe_metrics = self.evaluate_progress_probe()
                    progress_report = self.learning_progress.observe(
                        self.state.valid_targets_seen,
                        probe_metrics["pc_rasl/probe_ce_nats_per_token"],
                        metrics["optimization/learning_rate"],
                    )
                    progress_metrics = LearningProgressAuthority.metrics(
                        progress_report
                    )
                    metrics.update(probe_metrics)
                    metrics.update(progress_metrics)
                    self._record_progress_observation(
                        probe_metrics, progress_report
                    )
                    self._finalize_pc_rasl_interval(progress_report)
                    metrics["pc_rasl/updates_due_after_consequence"] = float(
                        self.state.pc_rasl_updates_due
                    )
                    progress_observed = True
                low_clip_warning = self._update_phase_transition_metrics(
                    metrics, gradient,
                )
                if not all(isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("MRCRA metrics became non-finite")
                self.last_step_metrics = dict(metrics)
                transition_pending = (
                    self._pending_first_hard_event_trace is not None
                    and self.state.first_hard_event_step == 0
                )
                should_log = (
                    self.state.step % self.config.log_interval == 0
                    or progress_observed
                    or low_clip_warning
                    or transition_pending
                )
                if should_log:
                    reporter.log(metrics, step=self.state.step)
                if low_clip_warning:
                    self._alert_low_clip_pressure(reporter)
                exceptional_started = perf_counter()
                transition_checkpoint = self._record_first_hard_event(
                    reporter, metrics, gradient,
                )
                if transition_checkpoint is not None:
                    metrics["performance/checkpoint_seconds"] += (
                        perf_counter() - exceptional_started
                    )
                if transition_checkpoint is not None and self.config.trackio_enabled:
                    snapshot_started = perf_counter()
                    self._publish_cognitive_snapshot(reporter)
                    metrics["performance/snapshot_seconds"] += (
                        perf_counter() - snapshot_started
                    )
                if (
                    self.config.evaluation_interval
                    and self.state.step % self.config.evaluation_interval == 0
                ):
                    evaluation_started = perf_counter()
                    evaluation_metrics = self.evaluate()
                    metrics["performance/evaluation_seconds"] += (
                        perf_counter() - evaluation_started
                    )
                    if self.learning_progress is not None:
                        guard_ce = evaluation_metrics[
                            "eval/cross_entropy_nats_per_token"
                        ]
                        guard_allowed = self.learning_progress.observe_guard(
                            guard_ce
                        )
                        evaluation_metrics.update({
                            "pc_rasl/guard_ce_nats_per_token": guard_ce,
                            "pc_rasl/guard_best_ce_nats_per_token": (
                                self.learning_progress.best_guard_ce
                                if self.learning_progress.best_guard_ce is not None
                                else guard_ce
                            ),
                            "pc_rasl/guard_allows_positive_pressure": float(
                                guard_allowed
                            ),
                            "pc_rasl/guard_regressions": float(
                                self.learning_progress.guard_regressions
                            ),
                        })
                    self._record_evaluation(evaluation_metrics)
                    reporter.log(evaluation_metrics, step=self.state.step)
                if (
                    self.config.trackio_enabled
                    and self.state.step % self.config.checkpoint_interval == 0
                    and _diagnostic_snapshot_due(
                        self.state.step, self.config.spectral_snapshot_interval,
                        self._last_snapshot_attempt_step,
                    )
                ):
                    snapshot_started = perf_counter()
                    self._publish_cognitive_snapshot(reporter)
                    metrics["performance/snapshot_seconds"] += (
                        perf_counter() - snapshot_started
                    )
                if self.state.step % self.config.checkpoint_interval == 0:
                    checkpoint_started = perf_counter()
                    path = self.save_checkpoint()
                    metrics["performance/checkpoint_seconds"] += (
                        perf_counter() - checkpoint_started
                    )
                    reporter.alert(
                        "MRCRA checkpoint saved", path.name,
                        level="info", step=self.state.step,
                    )
                wall_clock_elapsed = perf_counter() - step_started
                metrics["performance/wall_clock_step_seconds"] = (
                    wall_clock_elapsed
                )
                metrics["performance/wall_clock_tokens_per_second"] = (
                    tokens_this_update / max(wall_clock_elapsed, 1e-9)
                )
                metrics["observation/failure_count"] = float(
                    reporter.failure_count
                )
                self.last_step_metrics = dict(metrics)
                if (
                    should_log
                    and wall_clock_elapsed > elapsed + 1e-9
                    and any(
                        metrics[name] > 0
                        for name in (
                            "performance/evaluation_seconds",
                            "performance/checkpoint_seconds",
                            "performance/snapshot_seconds",
                        )
                    )
                ):
                    # Same-step replacement updates the dashboard with the
                    # complete periodic-cost receipt. The append-only local
                    # mirror intentionally retains both the pre-periodic and
                    # final rows for exact wall-clock reconstruction.
                    reporter.log(metrics, step=self.state.step)
                if step_observer is not None:
                    step_observer(self.state, metrics)
            self.state.elapsed_seconds += perf_counter() - started
            failed = False
            return self.state
        finally:
            reporter.finish()
            # A bounded ``maximum_steps`` call is intentionally resumable on the
            # same trainer object.  Preserve its one-worker lookahead and RNG
            # continuity; close the worker only after the declared token budget
            # is exhausted or an exception makes the trainer unusable.
            if (
                self._prefetch_executor is not None
                and (failed or self.state.tokens_seen >= self.config.total_tokens)
            ):
                self._prefetch_executor.shutdown(
                    wait=True, cancel_futures=failed,
                )
                self._prefetch_executor = None
