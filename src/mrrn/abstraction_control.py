"""Validity state for operating at the highest reliable abstraction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .tensor_state import TensorStateMixin
from .runtime_validation import runtime_validation_enabled


@dataclass(frozen=True, slots=True)
class AbstractionValidityState(TensorStateMixin):
    applicability: Tensor
    reconstruction_distortion: Tensor
    relation_distortion: Tensor
    task_distortion: Tensor
    provenance_sufficiency: Tensor
    precision_sufficiency: Tensor
    calibrated_confidence: Tensor
    abstraction_depths: Tensor
    physical_scales: Tensor
    abstraction_node_indices: Tensor
    provenance_ids: Tensor
    last_checked_observation: Tensor
    versions: Tensor
    active: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.applicability.ndim != 2:
            raise ValueError("abstraction validity must be (batch,capacity)")
        base = self.applicability.shape
        for name in (
            "reconstruction_distortion", "relation_distortion", "task_distortion",
            "provenance_sufficiency", "precision_sufficiency", "calibrated_confidence",
        ):
            value = getattr(self, name)
            if value.shape != base or not value.is_floating_point():
                raise ValueError(f"abstraction validity {name} must match rows")
        for name in (
            "abstraction_depths", "physical_scales", "abstraction_node_indices",
            "provenance_ids", "last_checked_observation", "versions",
        ):
            value = getattr(self, name)
            if value.shape != base or value.dtype != torch.int64:
                raise ValueError(f"abstraction validity {name} must be int64 rows")
        if self.active.shape != base or self.active.dtype != torch.bool:
            raise ValueError("abstraction validity active mask is invalid")
        for name in ("applicability", "provenance_sufficiency", "precision_sufficiency", "calibrated_confidence"):
            value = getattr(self, name)
            if bool(((value < 0) | (value > 1)).any()):
                raise ValueError(f"abstraction validity {name} must lie in [0,1]")
        if bool((self.active & ((self.abstraction_node_indices < 0) | (self.provenance_ids < 0))).any()):
            raise ValueError("active abstraction validity requires node and provenance IDs")

    @classmethod
    def empty(cls, batch: int, capacity: int, *, device=None, dtype=None) -> "AbstractionValidityState":
        if min(batch, capacity) <= 0:
            raise ValueError("abstraction validity dimensions must be positive")
        base = (batch, capacity)
        zeros = lambda: torch.zeros(base, device=device, dtype=dtype)
        unknown = lambda: torch.full(base, -1, dtype=torch.int64, device=device)
        return cls(
            zeros(), zeros(), zeros(), zeros(), zeros(), zeros(), zeros(),
            unknown(), unknown(), unknown(), unknown(), unknown(),
            torch.zeros(base, dtype=torch.int64, device=device),
            torch.zeros(base, dtype=torch.bool, device=device),
        )

    @property
    def batch(self) -> int:
        return self.applicability.shape[0]


@dataclass(frozen=True, slots=True)
class AbstractionApplicability(TensorStateMixin):
    applicability: Tensor
    reconstruction_distortion: Tensor
    relation_distortion: Tensor
    task_distortion: Tensor
    supported_precision: Tensor
    calibrated_confidence: Tensor
    uncertainty: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.applicability.ndim != 1:
            raise ValueError("abstraction applicability must be per batch")
        shape = self.applicability.shape
        for name in (
            "reconstruction_distortion", "relation_distortion", "task_distortion",
            "supported_precision", "calibrated_confidence", "uncertainty",
        ):
            value = getattr(self, name)
            if value.shape != shape or not value.is_floating_point():
                raise ValueError(f"abstraction applicability {name} must be per batch")
        for name in ("applicability", "supported_precision", "calibrated_confidence"):
            value = getattr(self, name)
            if bool(((value < 0) | (value > 1)).any()):
                raise ValueError(f"abstraction applicability {name} must lie in [0,1]")
        if bool((self.uncertainty < 0).any()):
            raise ValueError("abstraction applicability uncertainty cannot be negative")


class AbstractionApplicabilityHead(nn.Module):
    """Estimate whether a particular abstraction remains valid for this task.

    This learned head is diagnostic evidence.  The selector below owns the
    hard threshold decision and therefore cannot be overridden by logits.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("abstraction applicability width must be positive")
        self.width = width
        self.encoder = nn.Sequential(
            nn.Linear(5 * width, 2 * width), nn.SiLU(),
            nn.Linear(2 * width, width), nn.SiLU(),
        )
        self.probabilities = nn.Linear(width, 3)
        self.distortions = nn.Linear(width, 3)
        self.uncertainty = nn.Linear(width, 1)

    def forward(
        self, abstraction: Tensor, observed_context: Tensor,
        relation_context: Tensor, hypothesis_context: Tensor, goal_context: Tensor,
    ) -> AbstractionApplicability:
        shape = abstraction.shape
        if abstraction.ndim != 2 or any(
            value.shape != shape for value in (
                observed_context, relation_context, hypothesis_context, goal_context,
            )
        ):
            raise ValueError("abstraction applicability contexts must share (batch,width)")
        hidden = self.encoder(torch.cat((
            abstraction, observed_context, relation_context,
            hypothesis_context, goal_context,
        ), -1))
        probabilities = torch.sigmoid(self.probabilities(hidden))
        distortions = F.softplus(self.distortions(hidden))
        return AbstractionApplicability(
            probabilities[:, 0], distortions[:, 0], distortions[:, 1],
            distortions[:, 2], probabilities[:, 1], probabilities[:, 2],
            F.softplus(self.uncertainty(hidden)).squeeze(-1),
        )


@dataclass(frozen=True, slots=True)
class AbstractionSelection(TensorStateMixin):
    validity_indices: Tensor
    abstraction_node_indices: Tensor
    abstraction_depths: Tensor
    physical_scales: Tensor
    validity_scores: Tensor
    needs_descent: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.validity_indices.ndim != 1:
            raise ValueError("abstraction selection must be per batch")
        shape = self.validity_indices.shape
        for name in (
            "validity_indices", "abstraction_node_indices",
            "abstraction_depths", "physical_scales",
        ):
            value = getattr(self, name)
            if value.shape != shape or value.dtype != torch.int64:
                raise ValueError(f"abstraction selection {name} must be int64 per batch")
        if self.validity_scores.shape != shape or not self.validity_scores.is_floating_point():
            raise ValueError("abstraction selection scores must be per batch")
        for name in ("needs_descent", "mask"):
            value = getattr(self, name)
            if value.shape != shape or value.dtype != torch.bool:
                raise ValueError(f"abstraction selection {name} must be boolean per batch")
        if bool((self.mask & (self.validity_indices < 0)).any()):
            raise ValueError("active abstraction selections require a validity record")


class AbstractionLevelSelector(nn.Module):
    """Select the deepest abstraction satisfying every hard validity clause."""

    def __init__(
        self, *, applicability_threshold: float = 0.5,
        provenance_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        for value in (
            applicability_threshold, provenance_threshold, confidence_threshold,
        ):
            if not 0 <= value <= 1:
                raise ValueError("abstraction selector thresholds must lie in [0,1]")
        self.register_buffer("applicability_threshold", torch.tensor(applicability_threshold))
        self.register_buffer("provenance_threshold", torch.tensor(provenance_threshold))
        self.register_buffer("confidence_threshold", torch.tensor(confidence_threshold))

    def forward(
        self, state: AbstractionValidityState, *, task_tolerance: Tensor,
        reconstruction_tolerance: Tensor, relation_tolerance: Tensor,
        required_precision: Tensor, candidate_mask: Tensor | None = None,
    ) -> AbstractionSelection:
        batch, capacity = state.applicability.shape
        for name, value in (
            ("task_tolerance", task_tolerance),
            ("reconstruction_tolerance", reconstruction_tolerance),
            ("relation_tolerance", relation_tolerance),
            ("required_precision", required_precision),
        ):
            if value.shape != (batch,):
                raise ValueError(f"abstraction selector {name} must be per batch")
        eligible = state.active.clone()
        if candidate_mask is not None:
            if candidate_mask.shape != eligible.shape or candidate_mask.dtype != torch.bool:
                raise ValueError("abstraction selector candidate mask is invalid")
            eligible &= candidate_mask
        eligible &= state.applicability >= self.applicability_threshold
        eligible &= state.reconstruction_distortion <= reconstruction_tolerance[:, None]
        eligible &= state.relation_distortion <= relation_tolerance[:, None]
        eligible &= state.task_distortion <= task_tolerance[:, None]
        eligible &= state.provenance_sufficiency >= self.provenance_threshold
        eligible &= state.precision_sufficiency >= required_precision[:, None]
        eligible &= state.calibrated_confidence >= self.confidence_threshold
        # Lexicographic policy: highest abstraction depth first, then confidence.
        score = (
            state.abstraction_depths.to(state.applicability.dtype) * 2
            + state.calibrated_confidence
        ).masked_fill(~eligible, -torch.inf)
        index = score.argmax(-1)
        found = eligible.any(-1)
        index = index.masked_fill(~found, -1)
        safe = index.clamp(0, capacity - 1)
        rows = torch.arange(batch, device=state.applicability.device)
        return AbstractionSelection(
            index,
            state.abstraction_node_indices[rows, safe].masked_fill(~found, -1),
            state.abstraction_depths[rows, safe].masked_fill(~found, -1),
            state.physical_scales[rows, safe].masked_fill(~found, -1),
            score[rows, safe].masked_fill(~found, 0),
            ~found, found,
        )


@dataclass(frozen=True, slots=True)
class LocalizedDescentPlan(TensorStateMixin):
    abstraction_node_indices: Tensor
    target_abstraction_depths: Tensor
    target_physical_scales: Tensor
    requested_support: Tensor
    trigger_reasons: Tensor
    precision_tolerance: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        batch = self.abstraction_node_indices.shape[0]
        for name in (
            "abstraction_node_indices", "target_abstraction_depths",
            "target_physical_scales", "trigger_reasons",
        ):
            value = getattr(self, name)
            if value.shape != (batch,) or value.dtype != torch.int64:
                raise ValueError(f"localized descent {name} must be int64 per batch")
        if self.requested_support.shape != (batch, 3):
            raise ValueError("localized descent support must be (batch,3)")
        if self.precision_tolerance.shape != (batch,):
            raise ValueError("localized descent precision must be per batch")
        if self.mask.shape != (batch,) or self.mask.dtype != torch.bool:
            raise ValueError("localized descent mask is invalid")


class LocalizedDescentPlanner(nn.Module):
    """Create one bounded region-specific descent request per affected row."""

    def forward(
        self, selection: AbstractionSelection, node_support: Tensor,
        *, fallback_node_indices: Tensor, requested_precision: Tensor,
        trigger_reasons: Tensor,
    ) -> LocalizedDescentPlan:
        if node_support.ndim != 3 or node_support.shape[-1] != 3:
            raise ValueError("localized descent node support must be (batch,nodes,3)")
        batch, capacity = node_support.shape[:2]
        for name, value in (
            ("fallback_node_indices", fallback_node_indices),
            ("trigger_reasons", trigger_reasons),
        ):
            if value.shape != (batch,) or value.dtype != torch.int64:
                raise ValueError(f"localized descent {name} must be int64 per batch")
        if requested_precision.shape != (batch,):
            raise ValueError("localized descent precision must be per batch")
        source = torch.where(
            selection.abstraction_node_indices >= 0,
            selection.abstraction_node_indices, fallback_node_indices,
        )
        valid_source = (source >= 0) & (source < capacity)
        safe = source.clamp(0, capacity - 1)
        rows = torch.arange(batch, device=node_support.device)
        active = selection.needs_descent & valid_source
        current_depth = selection.abstraction_depths.clamp_min(1)
        return LocalizedDescentPlan(
            source.masked_fill(~active, -1),
            (current_depth - 1).clamp_min(0).masked_fill(~active, -1),
            selection.physical_scales.masked_fill(~active, -1),
            node_support[rows, safe].masked_fill(~active[:, None], 0),
            trigger_reasons.masked_fill(~active, -1), requested_precision,
            active,
        )


@dataclass(frozen=True, slots=True)
class AbstractionRevisionProposal(TensorStateMixin):
    abstraction_node_indices: Tensor
    previous_provenance_ids: Tensor
    revised_applicability: Tensor
    counterexample_provenance_ids: Tensor
    reason_codes: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.abstraction_node_indices.ndim != 1:
            raise ValueError("abstraction revision must be per batch")
        batch = self.abstraction_node_indices.shape[0]
        for name in (
            "abstraction_node_indices", "previous_provenance_ids",
            "counterexample_provenance_ids", "reason_codes",
        ):
            value = getattr(self, name)
            if value.shape != (batch,) or value.dtype != torch.int64:
                raise ValueError(f"abstraction revision {name} must be int64 per batch")
        if self.revised_applicability.shape != (batch,):
            raise ValueError("abstraction revision applicability must be per batch")
        if bool(((self.revised_applicability < 0) | (self.revised_applicability > 1)).any()):
            raise ValueError("revised applicability must lie in [0,1]")
        if self.mask.shape != (batch,) or self.mask.dtype != torch.bool:
            raise ValueError("abstraction revision mask is invalid")
        if bool((self.mask & ((self.previous_provenance_ids < 0) | (self.counterexample_provenance_ids < 0))).any()):
            raise ValueError("active abstraction revisions require provenance")
