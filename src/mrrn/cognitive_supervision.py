"""Evidence-aligned supervision authority for every MRCRA objective family.

The core never invents labels.  Targets either come from immutable packed-input
facts (segment boundaries and source class), from a declared annotation resolver,
or from a causal self-supervised comparison whose target is detached.  Families
without admissible evidence emit no term; the trainer's required-family gate then
fails closed instead of silently optimizing a proxy under the wrong name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

from .cognitive_objectives import (
    MetacognitiveTargets, ObjectiveFamily, ObjectiveTerm, focal_binary_loss,
    masked_categorical_nll, metacognitive_objectives,
)
from .cognitive_types import BoundaryClass, NodeType, SourceClass, VerificationClass
from .knowledge import KnowledgeStatus

if TYPE_CHECKING:
    from .language import MRCRALanguageOutput
    from .lm_training import PackedBatch


@dataclass(frozen=True, slots=True)
class CognitiveSupervisionTargets:
    """Optional, explicitly sourced annotations aligned to a packed context.

    All sequence targets use full-context ``(batch,time,...)`` coordinates.
    A resolver may leave any family absent.  Masks are mandatory alongside
    annotations so unknown facts are never treated as negative examples.
    """

    event_start: Tensor | None = None
    event_end: Tensor | None = None
    event_type: Tensor | None = None
    event_mask: Tensor | None = None
    relation_type: Tensor | None = None
    relation_mask: Tensor | None = None
    binding_positive_indices: Tensor | None = None
    binding_mask: Tensor | None = None
    controller_action: Tensor | None = None
    controller_advantage: Tensor | None = None
    controller_mask: Tensor | None = None
    metacognitive_realized_error: Tensor | None = None
    metacognitive_operation_values: Tensor | None = None
    metacognitive_calibration_error: Tensor | None = None
    metacognitive_mask: Tensor | None = None
    external_reward: Tensor | None = None
    external_cost: Tensor | None = None
    external_constraint: Tensor | None = None
    external_success: Tensor | None = None
    external_mask: Tensor | None = None
    provenance_source: Tensor | None = None
    provenance_verification: Tensor | None = None
    provenance_mask: Tensor | None = None


class CognitiveTargetResolver(Protocol):
    def __call__(self, batch: "PackedBatch") -> CognitiveSupervisionTargets: ...


def language_evidence_targets(batch: "PackedBatch") -> CognitiveSupervisionTargets:
    """Derive only facts guaranteed by the packed language input contract."""

    valid = torch.ones_like(batch.input_ids, dtype=torch.bool)
    boundaries = batch.boundary_classes
    # A declared boundary ends the preceding event and begins a new causal span.
    event = boundaries != int(BoundaryClass.NONE)
    return CognitiveSupervisionTargets(
        event_start=event,
        event_end=event,
        event_type=torch.full_like(batch.input_ids, int(NodeType.EVENT)),
        event_mask=valid,
        provenance_source=torch.full_like(batch.input_ids, int(SourceClass.EXTERNAL)),
        provenance_verification=torch.full_like(
            batch.input_ids, int(VerificationClass.UNVERIFIED)
        ),
        provenance_mask=valid,
    )


def _slice(value: Tensor | None, start: int, end: int) -> Tensor | None:
    return None if value is None else value[:, start:end]


def _require_group(targets: CognitiveSupervisionTargets, names: tuple[str, ...]) -> bool:
    present = tuple(getattr(targets, name) is not None for name in names)
    if any(present) and not all(present):
        missing = [name for name, available in zip(names, present, strict=True) if not available]
        raise ValueError(f"partial cognitive target group is forbidden; missing {missing}")
    return all(present)


class EvidenceBackedCognitiveSupervisor:
    """Produce masked differentiable terms without fabricating missing labels."""

    def __init__(
        self, resolver: CognitiveTargetResolver = language_evidence_targets,
        *, binding_temperature: float = 0.1, controller_entropy_weight: float = 1e-3,
    ) -> None:
        if binding_temperature <= 0 or controller_entropy_weight < 0:
            raise ValueError("supervision controls are invalid")
        self.resolver = resolver
        self.binding_temperature = binding_temperature
        self.controller_entropy_weight = controller_entropy_weight

    @staticmethod
    def _aligned(value: Tensor, batch: "PackedBatch", name: str) -> None:
        if value.shape[:2] != batch.input_ids.shape:
            raise ValueError(f"{name} must align to the full packed context")

    def __call__(
        self, output: "MRCRALanguageOutput", batch: "PackedBatch", start: int, end: int,
    ) -> tuple[ObjectiveTerm, ...]:
        if not 0 <= start < end <= batch.input_ids.shape[1]:
            raise ValueError("cognitive supervision slice lies outside its packed context")
        targets = self.resolver(batch)
        terms: list[ObjectiveTerm] = []
        cognitive = output.cognitive
        local_shape = cognitive.latent.shape[:2]
        if local_shape != (batch.input_ids.shape[0], end - start):
            raise ValueError("cognitive output does not align to requested training slice")

        if _require_group(targets, ("event_start", "event_end", "event_type", "event_mask")):
            for name in ("event_start", "event_end", "event_type", "event_mask"):
                self._aligned(getattr(targets, name), batch, name)
            mask = _slice(targets.event_mask, start, end)
            if mask.dtype != torch.bool:
                raise ValueError("event mask must be boolean")
            start_target = _slice(targets.event_start, start, end)
            end_target = _slice(targets.event_end, start, end)
            type_target = _slice(targets.event_type, start, end)
            if type_target.dtype != torch.int64:
                raise ValueError("event types must be int64")
            terms.extend((
                ObjectiveTerm(
                    "event_onset_focal", ObjectiveFamily.EVENTS_RELATIONS,
                    focal_binary_loss(cognitive.event_proposal_logits, start_target, mask), mask,
                ),
                ObjectiveTerm(
                    "event_end_focal", ObjectiveFamily.EVENTS_RELATIONS,
                    focal_binary_loss(cognitive.event_end_logits, end_target, mask), mask,
                ),
                ObjectiveTerm(
                    "event_type_nll", ObjectiveFamily.EVENTS_RELATIONS,
                    masked_categorical_nll(cognitive.event_type_logits, type_target, mask & start_target.bool()),
                    mask & start_target.bool(),
                ),
            ))

        if _require_group(targets, ("relation_type", "relation_mask")):
            self._aligned(targets.relation_type, batch, "relation_type")
            self._aligned(targets.relation_mask, batch, "relation_mask")
            relation_target = _slice(targets.relation_type, start, end)
            relation_mask = _slice(targets.relation_mask, start, end)
            if relation_target.dtype != torch.int64 or relation_mask.dtype != torch.bool:
                raise ValueError("relation targets require int64 labels and boolean mask")
            relation_nll = F.nll_loss(
                cognitive.relation_type_probabilities.clamp_min(1e-8).log().flatten(0, 1),
                relation_target.flatten(), reduction="none",
            ).reshape_as(relation_target)
            terms.append(ObjectiveTerm(
                "relation_family_nll", ObjectiveFamily.EVENTS_RELATIONS,
                relation_nll, relation_mask,
            ))

        if _require_group(targets, ("binding_positive_indices", "binding_mask")):
            self._aligned(targets.binding_positive_indices, batch, "binding_positive_indices")
            self._aligned(targets.binding_mask, batch, "binding_mask")
            positive = _slice(targets.binding_positive_indices, start, end) - start
            mask = _slice(targets.binding_mask, start, end)
            if positive.dtype != torch.int64 or mask.dtype != torch.bool:
                raise ValueError("binding annotations require int64 positives and boolean mask")
            mask = mask & (positive >= 0) & (positive < end - start)
            similarity = torch.einsum(
                "bid,bjd->bij",
                F.normalize(cognitive.cognitive_features, dim=-1),
                F.normalize(cognitive.cognitive_features, dim=-1),
            ) / self.binding_temperature
            safe = positive.clamp(0, max(0, end - start - 1))
            loss = F.cross_entropy(
                similarity.flatten(0, 1), safe.flatten(), reduction="none"
            ).reshape_as(mask)
            terms.append(ObjectiveTerm(
                "crossmodal_same_event_binding", ObjectiveFamily.MULTIMODAL_BINDING,
                loss, mask,
            ))

        # Causal one-step prediction is a genuine self-supervised target.  The
        # segment mask prevents unrelated packed documents from becoming targets.
        if end - start > 1:
            same_segment = (
                batch.segment_ids[:, start + 1:end]
                == batch.segment_ids[:, start:end - 1]
            )
            prediction_error = (
                cognitive.predicted_next_latent[:, :-1]
                - cognitive.latent[:, 1:].detach()
            ).square().mean(-1)
            terms.append(ObjectiveTerm(
                "causal_next_state", ObjectiveFamily.WORLD_HYPOTHESES_UNCERTAINTY,
                prediction_error, same_segment,
            ))
            uncertainty_target = prediction_error.detach().sqrt()
            predicted_uncertainty = cognitive.uncertainty[:, :-1].mean(-1)
            terms.append(ObjectiveTerm(
                "uncertainty_matches_error", ObjectiveFamily.WORLD_HYPOTHESES_UNCERTAINTY,
                (predicted_uncertainty - uncertainty_target).square(), same_segment,
            ))

        knowledge = cognitive.knowledge
        pending = knowledge.active & (
            knowledge.status == int(KnowledgeStatus.PENDING)
        )
        if bool(pending.any()):
            # Distortion is optimized, while positive code gain is rewarded with
            # a bounded log utility.  Promotion still requires the separate
            # held-out authority gate in KnowledgeProposalBank.validate.
            compression = (
                knowledge.reconstruction_distortion + knowledge.relation_distortion
                - torch.log1p(knowledge.code_gain_bits.clamp_min(0))
            )
            terms.append(ObjectiveTerm(
                "validated_compression_candidate",
                ObjectiveFamily.MEMORY_COMPRESSION_INVARIANTS,
                compression, pending,
            ))

        if _require_group(
            targets, ("controller_action", "controller_advantage", "controller_mask")
        ):
            for name in ("controller_action", "controller_advantage", "controller_mask"):
                self._aligned(getattr(targets, name), batch, name)
            action = _slice(targets.controller_action, start, end)
            advantage = _slice(targets.controller_advantage, start, end)
            mask = _slice(targets.controller_mask, start, end)
            expected = cognitive.action_receipts.actions.shape
            if action.shape != expected or advantage.shape != expected or mask.shape != expected:
                raise ValueError("controller annotations must align to (batch,time,microstep)")
            if action.dtype != torch.int64 or mask.dtype != torch.bool:
                raise ValueError("controller actions require int64 labels and boolean mask")
            safe = action.masked_fill(~mask, 0)
            nll = F.cross_entropy(
                cognitive.action_receipts.action_logits.flatten(0, -2),
                safe.flatten(), reduction="none",
            ).reshape_as(action)
            probability = torch.softmax(cognitive.action_receipts.action_logits, -1)
            entropy = -(probability.clamp_min(1e-8) * probability.clamp_min(1e-8).log()).sum(-1)
            policy = advantage.detach() * nll - self.controller_entropy_weight * entropy
            terms.append(ObjectiveTerm(
                "functional_surprise_policy", ObjectiveFamily.CONTROLLER_CONSEQUENCE,
                policy, mask & cognitive.action_receipts.mask,
            ))

        if _require_group(targets, (
            "metacognitive_realized_error", "metacognitive_operation_values",
            "metacognitive_calibration_error", "metacognitive_mask",
        )):
            for name in (
                "metacognitive_realized_error", "metacognitive_operation_values",
                "metacognitive_calibration_error", "metacognitive_mask",
            ):
                self._aligned(getattr(targets, name), batch, name)
            mask = _slice(targets.metacognitive_mask, start, end)
            if mask.dtype != torch.bool:
                raise ValueError("metacognitive mask must be boolean")
            terms.extend(metacognitive_objectives(
                cognitive.metacognitive_values,
                cognitive.metacognitive_mask,
                MetacognitiveTargets(
                    _slice(targets.metacognitive_realized_error, start, end),
                    _slice(targets.metacognitive_operation_values, start, end),
                    _slice(targets.metacognitive_calibration_error, start, end),
                    mask,
                ),
            ))

        if _require_group(
            targets,
            ("external_reward", "external_cost", "external_constraint", "external_success", "external_mask"),
        ):
            for name in (
                "external_reward", "external_cost", "external_constraint",
                "external_success", "external_mask",
            ):
                self._aligned(getattr(targets, name), batch, name)
            # External decisions are sparse and the output retains the most
            # recent decision.  Supervise it only from the final annotated row.
            index = end - 1
            mask = targets.external_mask[:, index] & cognitive.external_action.active
            selected = cognitive.external_action.selected_action.clamp_min(0)
            batch_index = torch.arange(selected.shape[0], device=selected.device)
            reward = cognitive.external_action.expected_reward[batch_index, selected]
            cost = cognitive.external_action.expected_cost[batch_index, selected]
            constraint = cognitive.external_action.constraint_probability[batch_index, selected]
            success = cognitive.external_action.expected_success[batch_index, selected]
            consequence = (
                (reward - targets.external_reward[:, index]).square()
                + (cost - targets.external_cost[:, index]).square()
                + F.binary_cross_entropy(
                    constraint.clamp(1e-6, 1 - 1e-6),
                    targets.external_constraint[:, index].to(constraint.dtype), reduction="none",
                )
                + F.binary_cross_entropy(
                    success.clamp(1e-6, 1 - 1e-6),
                    targets.external_success[:, index].to(success.dtype), reduction="none",
                )
            )
            terms.append(ObjectiveTerm(
                "external_consequence_model", ObjectiveFamily.CONTROLLER_CONSEQUENCE,
                consequence, mask,
            ))

        if _require_group(
            targets, ("provenance_source", "provenance_verification", "provenance_mask")
        ):
            for name in ("provenance_source", "provenance_verification", "provenance_mask"):
                self._aligned(getattr(targets, name), batch, name)
            source = _slice(targets.provenance_source, start, end)
            verification = _slice(targets.provenance_verification, start, end)
            mask = _slice(targets.provenance_mask, start, end)
            if source.dtype != torch.int64 or verification.dtype != torch.int64 or mask.dtype != torch.bool:
                raise ValueError("provenance targets require int64 classes and boolean mask")
            terms.extend((
                ObjectiveTerm(
                    "provenance_source_reconstruction", ObjectiveFamily.PROVENANCE_CONSISTENCY,
                    masked_categorical_nll(cognitive.provenance_source_logits, source, mask), mask,
                ),
                ObjectiveTerm(
                    "provenance_verification_reconstruction", ObjectiveFamily.PROVENANCE_CONSISTENCY,
                    masked_categorical_nll(
                        cognitive.provenance_verification_logits, verification, mask
                    ), mask,
                ),
            ))
        return tuple(terms)
