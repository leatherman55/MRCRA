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
import json
from math import ceil, isfinite, log
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any, Iterable, Protocol, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .cognitive_checkpoint import runtime_state_dict, runtime_state_from_dict
from .cognitive_diagnostics import cognitive_metrics
from .cognitive_model import HardEventTrace, MRCRAOutput, MRCRARuntimeState
from .cognitive_supervision import EvidenceBackedCognitiveSupervisor
from .cognitive_types import CognitiveClocks, InternalAction
from .cognitive_objectives import (
    CognitiveObjectiveSchedule, ObjectiveFamily, ObjectiveTerm,
    combine_cognitive_objectives,
)
from .language import MRCRALanguageModel, MRCRALanguageOutput
from .lm_training import (
    PackedBatch, PackedTokenStream, TextTokenizer, TrackioReporter,
    _configure_cuda, _device_for, _memory_metrics, _precision_for,
    _runtime_details, _synchronize,
)
from .mixer import ResonantSpectralGLU
from .objectives import spectral_activation_regularization
from .optimization import (
    GradientReport, OptimizerPolicy, build_adamw, build_scheduler,
    clip_and_report_gradients,
)
from .provenance import ProvenanceLedger
from .training_profiles import TrainerMode, get_training_profile


# Version 8 adds checkpointed phase-transition state, exact hard-event receipts,
# and deterministic cognition ablations. Versions 3--7 migrate conservatively.
MRCRA_TRAINING_FORMAT_VERSION = 8
LEGACY_MRCRA_TRAINING_FORMAT_VERSIONS = {3, 4, 5, 6, 7}

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


def _diagnostic_snapshot_due(
    step: int, interval: int, last_snapshot_step: int = -1,
) -> bool:
    """Publish once per process immediately, then use the low-overhead cadence."""

    if step <= 0 or interval <= 0 or last_snapshot_step < -1:
        raise ValueError("diagnostic snapshot step and interval must be positive")
    if last_snapshot_step >= step:
        return False
    return last_snapshot_step < 0 or step == 1 or step % interval == 0


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
    vocabulary_tile_size: int = 2_048
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
    state_target_rms: float = 8.0
    curriculum_stage: int = 1
    training_profile: str = "substrate_language_pretraining"
    trainer_mode: str = TrainerMode.INDEPENDENT_PACKED_DOCUMENTS.value
    required_auxiliary_families: tuple[int, ...] = ()
    integrated_cognitive_path: bool = False
    cognitive_stride: int = 128
    cognitive_tbptt_events: int = 4
    progress_interval_tokens: int = 2_048
    checkpoint_tiles: bool | None = None
    maximum_retained_loss_bytes: int = 1 << 30
    maximum_fused_loss_bytes: int = 512 << 20
    log_interval: int = 1
    checkpoint_interval: int = 25
    keep_checkpoints: int = 3
    evaluation_interval: int = 0
    evaluation_batches: int = 0
    require_evaluation: bool = False
    seed: int = 20260722
    device: str = "auto"
    precision: str = "auto"
    cpu_threads: int = 4
    cpu_interop_threads: int = 1
    data_prefetch: bool = True
    compile_tensor_cores: bool | None = None
    apple_mps_loss_offload: bool = False
    trackio_enabled: bool = True
    trackio_project: str = "mrcra-fineweb"
    run_name: str = "mrcra-120m-fineweb-20m-32k"
    trackio_space_id: str | None = None
    show_dashboard: bool = True
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
            self.cpu_threads, self.cpu_interop_threads,
            self.phase_transition_ablation_batches,
            self.low_clip_coefficient_patience,
        )
        if min(positive) <= 0:
            raise ValueError("MRCRA training sizes and intervals must be positive")
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
        if self.maximum_fused_loss_bytes < 0 or self.maximum_retained_loss_bytes < 0:
            raise ValueError("loss workspace limits cannot be negative")
        if self.checkpoint_tiles is not None and not isinstance(self.checkpoint_tiles, bool):
            raise ValueError("checkpoint_tiles must be boolean or None for automatic selection")
        if self.total_tokens < self.micro_batch_size * self.context_length:
            raise ValueError("token budget must contain at least one full context")
        if self.evaluation_interval < 0 or self.evaluation_batches < 0:
            raise ValueError("evaluation controls cannot be negative")
        if bool(self.evaluation_interval) != bool(self.evaluation_batches):
            raise ValueError("evaluation interval and batch count must be enabled together")
        if self.require_evaluation and not self.evaluation_batches:
            raise ValueError("this training run requires retained evaluation batches")
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
        if not isinstance(self.apple_mps_loss_offload, bool):
            raise ValueError("apple_mps_loss_offload must be boolean")
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


class _NullReporter:
    def log(self, metrics: dict[str, float], *, step: int) -> None:  # noqa: ARG002
        return None

    def alert(self, title: str, text: str, *, level: str, step: int) -> None:  # noqa: ARG002
        return None

    def log_phase_transition_trace(self, path: Path, *, step: int) -> int:  # noqa: ARG002
        return 0

    def finish(self) -> None:
        return None


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
        supervision_provider: CognitiveSupervisionProvider | None = None,
    ) -> None:
        if tokenizer.vocabulary_size != model.vocabulary_size:
            raise ValueError("tokenizer vocabulary does not match MRCRA")
        if config.micro_batch_size != 1 and model.config.cognitive.maximum_cognitive_steps:
            # The implementation supports larger batches, but the serious
            # memory budget is intentionally fail-closed at microbatch one.
            raise ValueError("the serious stateful MRCRA trainer requires micro_batch_size=1")
        if len(evaluation_batches) != config.evaluation_batches:
            raise ValueError(
                "retained evaluation batches must match the training configuration"
            )
        if any(
            batch.input_ids.shape
            != (config.micro_batch_size, config.context_length)
            for batch in evaluation_batches
        ):
            raise ValueError(
                "retained evaluation batches must match configured batch/context shape"
            )
        self.model, self.tokenizer, self.train_stream = model, tokenizer, train_stream
        self.evaluation_batches = tuple(evaluation_batches)
        self.config = config
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
        interop_note = "not applicable"
        if self.device.type == "cpu":
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
        self.runtime["device_selection_reason"] = device_reason
        self.runtime["cpu_threads"] = torch.get_num_threads()
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
        self.runtime["loss_projection"] = (
            "fused_exact_full_softmax"
            if self.loss_device.type in {"cuda", "mps"}
            and config.maximum_fused_loss_bytes > 0
            and estimated_loss_bytes <= config.maximum_fused_loss_bytes
            else "tiled_exact_full_softmax"
        )
        self.runtime["estimated_fused_loss_bytes"] = estimated_loss_bytes
        self.runtime["loss_tile_checkpointing"] = self._checkpoint_loss_tiles
        self.runtime["maximum_retained_loss_bytes"] = config.maximum_retained_loss_bytes
        self.runtime["loss_memory_policy"] = (
            "explicit_recompute" if config.checkpoint_tiles is True
            else "explicit_retain" if config.checkpoint_tiles is False
            else "auto_recompute" if self._checkpoint_loss_tiles
            else "auto_retain"
        )
        self.model.to(self.device)
        compile_tensor_cores = (
            self.device.type == "cuda"
            if config.compile_tensor_cores is None
            else config.compile_tensor_cores
        )
        if compile_tensor_cores:
            model.cognitive.carrier.enable_compiled_tensor_cores()
        self.runtime["compiled_tensor_cores"] = compile_tensor_cores
        self.runtime["compiled_tensor_core_policy"] = (
            "automatic_cuda"
            if config.compile_tensor_cores is None and compile_tensor_cores
            else "automatic_disabled"
            if config.compile_tensor_cores is None
            else "explicit_enabled"
            if compile_tensor_cores
            else "explicit_disabled"
        )
        policy = OptimizerPolicy(
            learning_rate=config.learning_rate, weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            total_steps=max(config.total_steps, config.warmup_steps + 1),
            minimum_learning_rate_ratio=config.minimum_learning_rate_ratio,
        )
        self.optimizer = build_adamw(model, policy, fused=self.device.type == "cuda")
        self.scheduler = build_scheduler(self.optimizer, policy)
        self.scaler = torch.amp.GradScaler("cuda") if self.amp_dtype == torch.float16 else None
        self.state = MRCRATrainingState()
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
        self._pending_first_hard_event_trace: HardEventTrace | None = None
        self._phase_update_proposal_logits: list[Tensor] = []
        self._phase_update_end_logits: list[Tensor] = []
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
        if (
            device.type in {"cuda", "mps"}
            and self.config.maximum_fused_loss_bytes > 0
            and estimated_bytes <= self.config.maximum_fused_loss_bytes
        ):
            return exact_fused_cross_entropy(
                latent, local_labels, local_lengths, local_mask, weight, bias,
            )
        return exact_tiled_cross_entropy(
            latent, local_labels, local_lengths, local_mask, weight, bias,
            vocabulary_tile_size=self.config.vocabulary_tile_size,
            checkpoint_tiles=(
                self._checkpoint_loss_tiles
                if checkpoint_tiles is None else checkpoint_tiles
            ),
        )

    def _identity(self) -> dict[str, Any]:
        training = asdict(self.config)
        for key in ("output_dir", "total_tokens", "trackio_enabled", "show_dashboard"):
            training.pop(key, None)
        source = {
            key: value for key, value in self.train_stream.source.state_dict().items()
            if key not in {"raw_rows_scanned", "documents_yielded"}
        }
        return {
            "model_config": asdict(self.model.config),
            "parameter_count": self.model.parameter_count,
            "tokenizer": self.tokenizer.identity(),
            "training": training,
            "source": source,
            "evaluation": self.evaluation_identity,
        }

    @property
    def evaluation_identity(self) -> dict[str, int | str]:
        return {
            "batch_count": len(self.evaluation_batches),
            "sha256": self._evaluation_digest(),
        }

    def _evaluation_digest(self) -> str:
        digest = sha256()
        for batch in self.evaluation_batches:
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
            torch.save(self._checkpoint_payload(), temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        if not phase_transition:
            checkpoints = sorted(directory.glob("step-*.pt"))
            for obsolete in checkpoints[:-self.config.keep_checkpoints]:
                obsolete.unlink()
        latest = directory / "latest.json"
        latest.write_text(
            json.dumps({"checkpoint": destination.name, "step": self.state.step}, indent=2) + "\n",
            encoding="utf-8",
        )
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
        if isinstance(saved_identity, dict):
            try:
                # Format 7 predates the execution-only tensor-core compiler
                # policy. It changes neither weights nor mathematical state,
                # so an absent field inherits the explicit current policy.
                saved_identity["training"].setdefault(
                    "compile_tensor_cores",
                    expected_identity["training"]["compile_tensor_cores"],
                )
            except (KeyError, TypeError, AttributeError):
                raise ValueError("MRCRA training checkpoint identity is malformed") from None
        if saved_format in LEGACY_MRCRA_TRAINING_FORMAT_VERSIONS and isinstance(saved_identity, dict):
            try:
                saved_cognitive = saved_identity["model_config"]["cognitive"]
                current_cognitive = expected_identity["model_config"]["cognitive"]
                for name in _V4_COGNITIVE_DEFAULT_FIELDS:
                    saved_cognitive[name] = current_cognitive[name]
                saved_training = saved_identity["training"]
                current_training = expected_identity["training"]
                for name in (
                    "evaluation_interval", "evaluation_batches",
                    "require_evaluation", "event_compute_regularization_weight",
                    "checkpoint_tiles", "maximum_retained_loss_bytes",
                    "data_prefetch", "phase_transition_telemetry",
                    "phase_transition_ablation",
                    "phase_transition_ablation_batches",
                    "proposal_slope_ema_decay",
                    "low_clip_coefficient_threshold",
                    "low_clip_coefficient_patience",
                ):
                    saved_training[name] = current_training[name]
                # Pre-v6 checkpoints did not bind retained evaluation data.
                # Migration attaches the explicitly supplied current retained
                # set; the digest is subsequently enforced on every resume.
                saved_identity["evaluation"] = expected_identity["evaluation"]
                saved_identity["parameter_count"] = expected_identity["parameter_count"]
            except (KeyError, TypeError):
                raise ValueError("legacy MRCRA training checkpoint identity is malformed") from None
        if saved_identity != expected_identity:
            raise ValueError("MRCRA checkpoint model, tokenizer, data, or training contract differs")
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
        self.state = MRCRATrainingState(**payload["training_state"])
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
            or self._last_snapshot_step == self.state.step
        ):
            return
        from .cognitive_diagnostics import cognitive_evidence
        from .visualization import model_spectral_evidence

        was_training = self.model.training
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
        dense = tuple(
            resonator.value.float().square().mean()
            for block in state.carrier.blocks
            for resonator in block.resonators
        )
        cognitive = tuple(
            resonator.value.float().square().mean()
            for carrier in state.cognitive.carrier
            for block in carrier.blocks
            for resonator in block.resonators
        )
        energies = torch.stack(dense + cognitive)
        rms = energies.clamp_min(0).sqrt()
        penalty = (energies - target_rms**2).clamp_min(0).mean()
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
        group_tokens = 0
        proposal_logit_rows: list[Tensor] = []
        end_logit_rows: list[Tensor] = []
        processed_tokens = 0
        next_progress = self.config.progress_interval_tokens
        started = perf_counter()
        model_forward_seconds = 0.0
        loss_forward_seconds = 0.0
        backward_seconds = 0.0

        def flush_group(*, final: bool) -> None:
            nonlocal nll_sum, byte_count, state_rms_max, group_tokens
            nonlocal loss_forward_seconds, backward_seconds
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
            loss_forward_seconds += perf_counter() - loss_started
            if not bool(torch.isfinite(scaled)):
                raise FloatingPointError("MRCRA integrated language loss became non-finite")
            backward_started = perf_counter()
            if self.scaler is None:
                scaled.backward()
            else:
                self.scaler.scale(scaled).backward()
            backward_seconds += perf_counter() - backward_started
            nll_sum += float(statistics.nll_sum.detach().cpu())
            byte_count += statistics.byte_count
            if group_state_maxima:
                state_rms_max = max(
                    state_rms_max,
                    float(torch.stack(group_state_maxima).max().detach().cpu()),
                )
            group_latents.clear()
            group_labels.clear()
            group_byte_lengths.clear()
            group_masks.clear()
            group_penalties.clear()
            group_state_maxima.clear()
            group_event_activations.clear()
            group_tokens = 0

        for span_index, (start, end, reset) in enumerate(spans):
            if reset:
                training_state = None
                ledger = ProvenanceLedger()
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
            model_forward_seconds += perf_counter() - forward_started
            group_latents.append(output.output_latent)
            group_labels.append(batch.labels[:, start:end])
            group_byte_lengths.append(batch.target_byte_lengths[:, start:end])
            group_masks.append(local_mask)
            group_penalties.append(state_penalty)
            group_state_maxima.append(state_max)
            group_event_activations.append(output.event_activation_mean)
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
            "performance/model_forward_seconds": model_forward_seconds,
            "performance/loss_forward_seconds": loss_forward_seconds,
            "performance/backward_seconds": backward_seconds,
        }
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

    def _record_evaluation(self, metrics: dict[str, float]) -> None:
        """Persist held-out evidence even when Trackio is intentionally off."""

        if not metrics or not all(isfinite(value) for value in metrics.values()):
            raise ValueError("evaluation evidence must be nonempty and finite")
        destination = Path(self.config.output_dir) / "evaluation_metrics.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": 1,
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

    def train(self, *, maximum_steps: int | None = None) -> MRCRATrainingState:
        if maximum_steps is not None and maximum_steps <= 0:
            raise ValueError("maximum_steps must be positive")
        continuing = self._resumed or self.state.step > 0
        if not continuing:
            torch.manual_seed(self.config.seed)
        reporter = (
            TrackioReporter(self.config, self._identity(), resume=continuing)
            if self.config.trackio_enabled else _NullReporter()
        )
        started = perf_counter()
        completed_this_call = 0
        failed = True
        try:
            while self.state.tokens_seen < self.config.total_tokens:
                if maximum_steps is not None and completed_this_call >= maximum_steps:
                    break
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                self._phase_update_proposal_logits.clear()
                self._phase_update_end_logits.clear()
                step_started = perf_counter()
                aggregated: dict[str, float] = {}
                tokens_this_update = 0
                data_seconds = 0.0
                contexts_this_update = min(
                    self.config.gradient_accumulation_steps,
                    ceil((self.config.total_tokens - self.state.tokens_seen) / self.config.context_length),
                )
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
                    local = self._run_context(batch, gradient_divisor=contexts_this_update)
                    tokens_this_update += batch.token_count
                    for name, value in local.items():
                        aggregated[name] = aggregated.get(name, 0.0) + value
                gradient_started = perf_counter()
                if self.loss_device != self.device:
                    _synchronize(self.loss_device)
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
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
                    "performance/loss_forward_seconds",
                    "performance/backward_seconds",
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
                metrics.update({
                    "progress/step": float(self.state.step),
                    "progress/tokens_seen": float(self.state.tokens_seen),
                    "progress/valid_targets_seen": float(self.state.valid_targets_seen),
                    "progress/fraction": min(1.0, self.state.tokens_seen / self.config.total_tokens),
                    "performance/step_seconds": elapsed,
                    "performance/tokens_per_second": tokens_this_update / max(elapsed, 1e-9),
                    "performance/data_seconds": data_seconds,
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
                attributed = sum(metrics.get(name, 0.0) for name in (
                    "performance/data_seconds",
                    "performance/model_forward_seconds",
                    "performance/loss_forward_seconds",
                    "performance/backward_seconds",
                    "performance/gradient_reduction_seconds",
                    "performance/optimizer_seconds",
                ))
                metrics["performance/unattributed_step_seconds"] = max(
                    0.0, elapsed - attributed
                )
                metrics.update(_memory_metrics(self.device))
                low_clip_warning = self._update_phase_transition_metrics(
                    metrics, gradient,
                )
                if not all(isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("MRCRA metrics became non-finite")
                transition_pending = (
                    self._pending_first_hard_event_trace is not None
                    and self.state.first_hard_event_step == 0
                )
                if (
                    self.state.step % self.config.log_interval == 0
                    or low_clip_warning
                    or transition_pending
                ):
                    reporter.log(metrics, step=self.state.step)
                if low_clip_warning:
                    self._alert_low_clip_pressure(reporter)
                transition_checkpoint = self._record_first_hard_event(
                    reporter, metrics, gradient,
                )
                if transition_checkpoint is not None and self.config.trackio_enabled:
                    self._publish_cognitive_snapshot(reporter)
                if (
                    self.config.evaluation_interval
                    and self.state.step % self.config.evaluation_interval == 0
                ):
                    evaluation_metrics = self.evaluate()
                    self._record_evaluation(evaluation_metrics)
                    reporter.log(evaluation_metrics, step=self.state.step)
                if (
                    self.config.trackio_enabled
                    and _diagnostic_snapshot_due(
                        self.state.step, self.config.spectral_snapshot_interval,
                        self._last_snapshot_step,
                    )
                ):
                    self._publish_cognitive_snapshot(reporter)
                if self.state.step % self.config.checkpoint_interval == 0:
                    path = self.save_checkpoint()
                    reporter.alert(
                        "MRCRA checkpoint saved", path.name,
                        level="info", step=self.state.step,
                    )
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
