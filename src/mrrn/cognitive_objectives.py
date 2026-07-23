"""Masked modular MRCRA objectives, schedules, and gradient-conflict telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


class ObjectiveFamily(IntEnum):
    PRIMARY_TASK = 0
    SPECTRAL_SUBSTRATE = 1
    EVENTS_RELATIONS = 2
    MULTIMODAL_BINDING = 3
    MEMORY_COMPRESSION_INVARIANTS = 4
    WORLD_HYPOTHESES_UNCERTAINTY = 5
    CONTROLLER_CONSEQUENCE = 6
    PROVENANCE_CONSISTENCY = 7
    MEMORY_RETRIEVAL_UTILITY = 8
    COMPRESSION_ABSTRACTION_VALIDITY = 9
    RECONSTRUCTION_FIDELITY = 10
    WORLD_MODEL_HYPOTHESIS_LIKELIHOOD = 11
    ACTION_CONSEQUENCE_INFORMATION_GAIN = 12
    VIABILITY_CONSTRAINT = 13
    CONTROLLER_METACOGNITIVE_UTILITY = 14
    INVARIANT_TRANSFER = 15
    CONTINUAL_ADAPTATION_SAFETY = 16


@dataclass(frozen=True, slots=True)
class ObjectiveTerm:
    name: str
    family: ObjectiveFamily
    values: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("objective terms require a stable name")
        if self.values.shape != self.mask.shape or self.mask.dtype != torch.bool:
            raise ValueError("objective values and boolean mask must share shape")
        if not self.values.is_floating_point():
            raise ValueError("objective values must be floating point")

    @property
    def normalized(self) -> Tensor:
        finite = torch.isfinite(self.values) & self.mask
        if not bool(finite.any()):
            return self.values.sum() * 0
        return self.values.masked_fill(~finite, 0).sum() / finite.sum()


@dataclass(frozen=True, slots=True)
class CognitiveObjectiveSchedule:
    stage: int
    family_weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.stage <= 9:
            raise ValueError("MRCRA curriculum stage must lie in 0..9")
        if len(self.family_weights) != len(ObjectiveFamily) or any(value < 0 for value in self.family_weights):
            raise ValueError("one nonnegative weight is required per objective family")

    @classmethod
    def curriculum(cls, stage: int) -> "CognitiveObjectiveSchedule":
        if not 0 <= stage <= 9:
            raise ValueError("MRCRA curriculum stage must lie in 0..9")
        # Cumulative availability follows the canonical curriculum.  Later
        # stages retain earlier anchors to avoid capability erasure.
        first_stage = (
            1, 1, 2, 3, 3, 5, 6, 2,
            3, 4, 4, 5, 6, 8, 8, 7, 9,
        )
        weights = tuple(1.0 if stage >= start else 0.0 for start in first_stage)
        return cls(stage, weights)

    def weight(self, family: ObjectiveFamily) -> float:
        return self.family_weights[int(family)]


@dataclass(frozen=True, slots=True)
class CognitiveLossBreakdown:
    total: Tensor
    terms: Mapping[str, Tensor]
    family_totals: Tensor
    active_counts: Tensor


def combine_cognitive_objectives(
    terms: Iterable[ObjectiveTerm], schedule: CognitiveObjectiveSchedule,
) -> CognitiveLossBreakdown:
    terms = tuple(terms)
    if not terms:
        raise ValueError("at least one cognitive objective term is required")
    names = [term.name for term in terms]
    if len(names) != len(set(names)):
        raise ValueError("cognitive objective names must be unique")
    anchor = terms[0].values
    family_totals = anchor.new_zeros(len(ObjectiveFamily))
    counts = anchor.new_zeros(len(ObjectiveFamily))
    normalized = {}
    for term in terms:
        value = term.normalized
        if not bool(torch.isfinite(value)):
            raise FloatingPointError(f"cognitive objective {term.name!r} is non-finite")
        normalized[term.name] = value
        family_totals[int(term.family)] = family_totals[int(term.family)] + value
        counts[int(term.family)] += int(bool(term.mask.any()))
    averaged = family_totals / counts.clamp_min(1)
    weights = averaged.new_tensor(schedule.family_weights)
    total = (averaged * weights).sum()
    return CognitiveLossBreakdown(total, normalized, averaged, counts)


def masked_categorical_nll(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    if logits.shape[:-1] != targets.shape or mask.shape != targets.shape or targets.dtype != torch.int64 or mask.dtype != torch.bool:
        raise ValueError("categorical NLL requires logits, int64 targets, and matching mask")
    safe = targets.masked_fill(~mask, 0)
    return F.cross_entropy(logits.flatten(0, -2), safe.flatten(), reduction="none").reshape_as(targets)


def focal_binary_loss(logits: Tensor, targets: Tensor, mask: Tensor, *, gamma: float = 2.0) -> Tensor:
    if logits.shape != targets.shape or mask.shape != targets.shape or mask.dtype != torch.bool:
        raise ValueError("focal loss tensors must share shape and use a boolean mask")
    if gamma < 0:
        raise ValueError("focal gamma cannot be negative")
    target = targets.to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    correct_probability = torch.where(target.bool(), probability, 1 - probability)
    return (1 - correct_probability).pow(gamma) * bce


def relation_residual_decorrelation(residuals: Tensor, type_ids: Tensor, mask: Tensor) -> Tensor:
    if residuals.ndim != 3 or type_ids.shape != residuals.shape[:2] or mask.shape != type_ids.shape:
        raise ValueError("relation decorrelation inputs are incompatible")
    selected = residuals[mask]
    selected_types = type_ids[mask]
    if selected.shape[0] < 2:
        return residuals.new_zeros(())
    selected = F.normalize(selected - selected.mean(0), dim=-1)
    correlation = selected @ selected.T
    different = selected_types[:, None] != selected_types[None, :]
    if not bool(different.any()):
        return residuals.new_zeros(())
    return correlation[different].square().mean()


def contrastive_binding_loss(
    left: Tensor, right: Tensor, positive_indices: Tensor, mask: Tensor,
    *, temperature: float = 0.1,
) -> Tensor:
    if left.ndim != 3 or right.ndim != 3 or left.shape[0] != right.shape[0] or left.shape[-1] != right.shape[-1]:
        raise ValueError("contrastive binding tensors must share batch and width")
    if positive_indices.shape != left.shape[:2] or mask.shape != left.shape[:2] or positive_indices.dtype != torch.int64:
        raise ValueError("binding positives and mask must match left items")
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    logits = torch.einsum(
        "bid,bjd->bij", F.normalize(left, dim=-1), F.normalize(right, dim=-1)
    ) / temperature
    safe = positive_indices.clamp(0, max(0, right.shape[1] - 1))
    return F.cross_entropy(logits.flatten(0, 1), safe.flatten(), reduction="none").reshape_as(mask)


def hypothesis_diversity_loss(residuals: Tensor, weights: Tensor, mask: Tensor, *, margin: float = 0.2) -> Tensor:
    if residuals.ndim != 3 or weights.shape != residuals.shape[:2] or mask.shape != weights.shape:
        raise ValueError("hypothesis diversity inputs are incompatible")
    if not -1 <= margin <= 1:
        raise ValueError("hypothesis diversity margin must lie in [-1,1]")
    similarity = torch.einsum(
        "bhd,bjd->bhj", F.normalize(residuals, dim=-1), F.normalize(residuals, dim=-1)
    )
    pair_mask = mask[:, :, None] & mask[:, None, :]
    pair_mask &= ~torch.eye(mask.shape[-1], dtype=torch.bool, device=mask.device)[None]
    pair_weight = weights[:, :, None] * weights[:, None, :]
    penalty = F.relu(similarity - margin) * pair_weight
    return penalty[pair_mask].mean() if bool(pair_mask.any()) else residuals.new_zeros(())


def brier_rows(logits: Tensor, targets: Tensor) -> Tensor:
    if logits.shape[:-1] != targets.shape or targets.dtype != torch.int64:
        raise ValueError("Brier rows require logits and matching int64 targets")
    probability = torch.softmax(logits, -1)
    truth = F.one_hot(targets, logits.shape[-1]).to(logits.dtype)
    return (probability - truth).square().sum(-1)


def quantile_coverage_penalty(quantiles: Tensor, target: Tensor, levels: Tensor) -> Tensor:
    if quantiles.shape[-2] != levels.numel() or quantiles.shape[:-2] != target.shape[:-1]:
        raise ValueError("quantile coverage tensors are incompatible")
    empirical = (target.unsqueeze(-2) <= quantiles).to(quantiles.dtype)
    shape = (1,) * (empirical.ndim - 2) + (levels.numel(), 1)
    return (empirical.mean(tuple(range(empirical.ndim - 2))) - levels.view(shape[-2:])).square().mean()


@dataclass(frozen=True, slots=True)
class ReconstructionTargets:
    node_content: Tensor
    node_type_ids: Tensor
    node_mask: Tensor
    relation_content: Tensor
    relation_type_ids: Tensor
    relation_mask: Tensor
    historical_fidelity: Tensor
    structural_plausibility: Tensor
    evidence_agreement: Tensor
    applicability: Tensor
    example_mask: Tensor


def reconstruction_objectives(proposal, targets: ReconstructionTargets) -> tuple[ObjectiveTerm, ...]:
    """Supervise the actual conditional reconstruction heads and calibration."""

    if proposal.node_content.shape != targets.node_content.shape:
        raise ValueError("reconstruction node target shape differs from production proposal")
    if proposal.relation_content.shape != targets.relation_content.shape:
        raise ValueError("reconstruction relation target shape differs from production proposal")
    if targets.node_type_ids.shape != targets.node_mask.shape or targets.node_type_ids.dtype != torch.int64:
        raise ValueError("reconstruction node type targets are invalid")
    if targets.relation_type_ids.shape != targets.relation_mask.shape or targets.relation_type_ids.dtype != torch.int64:
        raise ValueError("reconstruction relation type targets are invalid")
    batch = proposal.node_content.shape[0]
    if targets.example_mask.shape != (batch,) or targets.example_mask.dtype != torch.bool:
        raise ValueError("reconstruction example mask is invalid")
    node_mask = targets.node_mask & proposal.node_mask & targets.example_mask[:, None]
    relation_mask = targets.relation_mask & proposal.relation_mask & targets.example_mask[:, None]
    node_error = (proposal.node_content - targets.node_content).square().mean(-1)
    relation_error = (proposal.relation_content - targets.relation_content).square().mean(-1)
    node_type = masked_categorical_nll(
        proposal.node_type_logits, targets.node_type_ids, node_mask
    )
    relation_type = masked_categorical_nll(
        proposal.relation_type_logits, targets.relation_type_ids, relation_mask
    )
    calibration = (
        (proposal.historical_fidelity - targets.historical_fidelity).square()
        + (proposal.structural_plausibility - targets.structural_plausibility).square()
        + (proposal.evidence_agreement - targets.evidence_agreement).square()
        + (proposal.applicability_probability - targets.applicability).square()
    )
    return (
        ObjectiveTerm("reconstruction_node_content", ObjectiveFamily.RECONSTRUCTION_FIDELITY, node_error, node_mask),
        ObjectiveTerm("reconstruction_node_type", ObjectiveFamily.RECONSTRUCTION_FIDELITY, node_type, node_mask),
        ObjectiveTerm("reconstruction_relation_content", ObjectiveFamily.RECONSTRUCTION_FIDELITY, relation_error, relation_mask),
        ObjectiveTerm("reconstruction_relation_type", ObjectiveFamily.RECONSTRUCTION_FIDELITY, relation_type, relation_mask),
        ObjectiveTerm("reconstruction_fidelity_calibration", ObjectiveFamily.RECONSTRUCTION_FIDELITY, calibration, targets.example_mask),
    )


@dataclass(frozen=True, slots=True)
class WorldModelTargets:
    latent: Tensor
    reward: Tensor
    cost: Tensor
    constraint: Tensor
    success: Tensor
    mask: Tensor


def world_model_objectives(prediction, targets: WorldModelTargets) -> tuple[ObjectiveTerm, ...]:
    """Train the production multihorizon action-conditioned consequence heads."""

    base = prediction.costs.shape
    if targets.latent.shape != prediction.latent_mean.shape:
        raise ValueError("world-model latent targets differ from production heads")
    for name in ("reward", "cost", "constraint", "success", "mask"):
        if getattr(targets, name).shape != base:
            raise ValueError(f"world-model {name} targets must match horizons")
    if targets.mask.dtype != torch.bool:
        raise ValueError("world-model target mask must be boolean")
    latent = (prediction.latent_mean - targets.latent).square().mean(-1)
    median = prediction.reward_quantiles[..., prediction.reward_quantiles.shape[-1] // 2]
    reward = (median - targets.reward).square()
    cost = (prediction.costs - targets.cost).square()
    constraint = F.binary_cross_entropy_with_logits(
        prediction.constraint_logits, targets.constraint.to(prediction.constraint_logits.dtype),
        reduction="none",
    )
    success = F.binary_cross_entropy_with_logits(
        prediction.action_success_logits, targets.success.to(prediction.action_success_logits.dtype),
        reduction="none",
    )
    return (
        ObjectiveTerm("world_multihorizon_latent", ObjectiveFamily.WORLD_MODEL_HYPOTHESIS_LIKELIHOOD, latent, targets.mask),
        ObjectiveTerm("world_multihorizon_reward", ObjectiveFamily.ACTION_CONSEQUENCE_INFORMATION_GAIN, reward, targets.mask),
        ObjectiveTerm("world_multihorizon_cost", ObjectiveFamily.ACTION_CONSEQUENCE_INFORMATION_GAIN, cost, targets.mask),
        ObjectiveTerm("world_multihorizon_constraint", ObjectiveFamily.VIABILITY_CONSTRAINT, constraint, targets.mask),
        ObjectiveTerm("world_multihorizon_success", ObjectiveFamily.ACTION_CONSEQUENCE_INFORMATION_GAIN, success, targets.mask),
    )


@dataclass(frozen=True, slots=True)
class MetacognitiveTargets:
    """Measured post-operation targets for the production self-model heads."""

    realized_error: Tensor
    operation_values: Tensor
    calibration_error: Tensor
    mask: Tensor


def metacognitive_objectives(
    predicted_values: Tensor, production_mask: Tensor,
    targets: MetacognitiveTargets,
) -> tuple[ObjectiveTerm, ...]:
    """Train routed self-predictions from measured downstream consequences.

    Columns are predicted error, five marginal operation values (compute,
    retrieval, reconstruction, simulation, evidence), and calibration error.
    Targets must be supplied by an authorized environment/evaluator; no proxy
    is fabricated here.
    """

    if predicted_values.ndim != 3 or predicted_values.shape[-1] != 7:
        raise ValueError("metacognitive predictions must be (batch,time,7)")
    if not predicted_values.is_floating_point() or not bool(torch.isfinite(predicted_values).all()):
        raise ValueError("metacognitive predictions must be finite floating point")
    base = predicted_values.shape[:2]
    if targets.realized_error.shape != base or targets.calibration_error.shape != base:
        raise ValueError("metacognitive scalar targets must align to batch/time")
    if targets.operation_values.shape != (*base, 5):
        raise ValueError("metacognitive operation targets must be (batch,time,5)")
    if targets.mask.shape != base or targets.mask.dtype != torch.bool:
        raise ValueError("metacognitive target mask must be boolean batch/time")
    if production_mask.shape != base or production_mask.dtype != torch.bool:
        raise ValueError("metacognitive production mask must be boolean batch/time")
    target_tensors = (
        targets.realized_error, targets.operation_values, targets.calibration_error,
    )
    if any(not value.is_floating_point() or not bool(torch.isfinite(value).all()) for value in target_tensors):
        raise ValueError("metacognitive targets must be finite floating point")
    mask = targets.mask & production_mask
    predicted_error = (predicted_values[..., 0] - targets.realized_error).square()
    operation_error = (
        predicted_values[..., 1:6] - targets.operation_values
    ).square().mean(-1)
    calibration_error = (
        predicted_values[..., 6] - targets.calibration_error
    ).square()
    return (
        ObjectiveTerm(
            "metacognitive_error_prediction",
            ObjectiveFamily.CONTROLLER_METACOGNITIVE_UTILITY,
            predicted_error, mask,
        ),
        ObjectiveTerm(
            "metacognitive_operation_value",
            ObjectiveFamily.CONTROLLER_METACOGNITIVE_UTILITY,
            operation_error, mask,
        ),
        ObjectiveTerm(
            "metacognitive_calibration",
            ObjectiveFamily.CONTROLLER_METACOGNITIVE_UTILITY,
            calibration_error, mask,
        ),
    )


@dataclass(frozen=True, slots=True)
class GradientConflictReport:
    names: tuple[str, ...]
    cosine: Tensor
    conflict_fraction: Tensor


def gradient_conflicts(
    named_losses: Mapping[str, Tensor], parameters: Sequence[Tensor],
) -> GradientConflictReport:
    if not named_losses or not parameters:
        raise ValueError("gradient conflict monitoring requires losses and parameters")
    names, rows = [], []
    for name, loss in named_losses.items():
        if loss.numel() != 1:
            raise ValueError("gradient-conflict losses must be scalar")
        gradients = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True,
        )
        row = torch.cat(tuple(
            torch.zeros_like(parameter).flatten() if gradient is None else gradient.flatten()
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ))
        names.append(name)
        rows.append(F.normalize(row, dim=0))
    matrix = torch.stack(rows)
    cosine = matrix @ matrix.T
    off_diagonal = ~torch.eye(len(rows), dtype=torch.bool, device=cosine.device)
    conflict = (cosine[off_diagonal] < 0).to(cosine.dtype).mean() if len(rows) > 1 else cosine.new_zeros(())
    return GradientConflictReport(tuple(names), cosine, conflict)
