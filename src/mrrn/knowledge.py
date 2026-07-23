"""Bounded two-phase authority for abstractions, invariants, and symbols.

Neural modules may propose a reusable representation, but a proposal is not an
authoritative abstraction or invariant.  This module keeps the proposal in a
bounded tensor ring, records the evidence needed to judge it, and permits
promotion only after held-out utility, distortion, provenance-diversity, and
counterexample checks have all been supplied.  The separation is deliberate:
it prevents an untrained compressor from silently turning its own latent state
into semantic memory.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum
from typing import TypeVar

import torch
from .runtime_validation import runtime_validation_enabled
from torch import Tensor

from .provenance import ProvenanceLedger


class KnowledgeKind(IntEnum):
    ABSTRACTION = 0
    INVARIANT = 1
    SYMBOL = 2


class KnowledgeStatus(IntEnum):
    EMPTY = 0
    PENDING = 1
    ACCEPTED = 2
    REJECTED = 3
    REVOKED = 4


T = TypeVar("T")


def _tensor_map(instance: T, method: str, *args, **kwargs) -> T:
    values = {}
    for field in fields(instance):
        value = getattr(instance, field.name)
        values[field.name] = (
            getattr(value, method)(*args, **kwargs)
            if isinstance(value, Tensor) else value
        )
    return type(instance)(**values)


@dataclass(frozen=True, slots=True)
class KnowledgeProposalBatch:
    """Candidate proposals created by one bounded cognitive operation."""

    latent: Tensor
    kind: Tensor
    code_gain_bits: Tensor
    reconstruction_distortion: Tensor
    relation_distortion: Tensor
    predictive_utility: Tensor
    action_utility: Tensor
    confidence: Tensor
    provenance_ids: Tensor
    supporting_provenance_ids: Tensor
    supporting_mask: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.latent.ndim != 2:
            raise ValueError("knowledge proposal latent must be (batch,width)")
        batch = self.latent.shape[0]
        for name in (
            "kind", "provenance_ids",
        ):
            value = getattr(self, name)
            if value.shape != (batch,) or value.dtype != torch.int64:
                raise ValueError(f"knowledge proposal {name} must be int64 per batch")
        for name in (
            "code_gain_bits", "reconstruction_distortion", "relation_distortion",
            "predictive_utility", "action_utility", "confidence",
        ):
            if getattr(self, name).shape != (batch,):
                raise ValueError(f"knowledge proposal {name} must be scalar per batch")
        if self.supporting_provenance_ids.ndim != 2 or self.supporting_provenance_ids.shape[0] != batch:
            raise ValueError("knowledge supporters must be (batch,supporters)")
        if self.supporting_mask.shape != self.supporting_provenance_ids.shape or self.supporting_mask.dtype != torch.bool:
            raise ValueError("knowledge supporting mask must match supporter IDs")
        if self.mask.shape != (batch,) or self.mask.dtype != torch.bool:
            raise ValueError("knowledge proposal mask must be boolean per batch")
        if bool((self.supporting_mask & (self.supporting_provenance_ids < 0)).any()):
            raise ValueError("active knowledge supporters require provenance IDs")
        if bool((self.mask & (self.provenance_ids < 0)).any()):
            raise ValueError("active knowledge proposals require derived provenance")
        if bool((self.mask & ((self.kind < 0) | (self.kind >= len(KnowledgeKind)))).any()):
            raise ValueError("knowledge proposal kind lies outside the ontology")
        if bool((self.mask & ((self.confidence < 0) | (self.confidence > 1))).any()):
            raise ValueError("knowledge confidence must lie in [0,1]")
        if bool((self.mask & ((self.reconstruction_distortion < 0) | (self.relation_distortion < 0))).any()):
            raise ValueError("knowledge distortion cannot be negative")


@dataclass(frozen=True, slots=True)
class KnowledgeValidationBatch:
    """Empirical evidence supplied after a proposal has been evaluated."""

    proposal_indices: Tensor
    held_out_loss_before: Tensor
    held_out_loss_after: Tensor
    predictive_utility: Tensor
    action_utility: Tensor
    calibrated_confidence: Tensor
    counterexample_search_completed: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        batch = self.proposal_indices.shape[0]
        if self.proposal_indices.shape != (batch,) or self.proposal_indices.dtype != torch.int64:
            raise ValueError("one int64 proposal index is required per batch")
        for name in (
            "held_out_loss_before", "held_out_loss_after", "predictive_utility",
            "action_utility", "calibrated_confidence",
        ):
            value = getattr(self, name)
            if value.shape != (batch,) or not value.is_floating_point():
                raise ValueError(f"knowledge validation {name} must be floating point per batch")
        for name in ("counterexample_search_completed", "mask"):
            value = getattr(self, name)
            if value.shape != (batch,) or value.dtype != torch.bool:
                raise ValueError(f"knowledge validation {name} must be boolean per batch")
        if bool((self.mask & ((self.calibrated_confidence < 0) | (self.calibrated_confidence > 1))).any()):
            raise ValueError("validated confidence must lie in [0,1]")


@dataclass(frozen=True, slots=True)
class KnowledgeValidationResult:
    accepted: Tensor
    code_gain_pass: Tensor
    distortion_pass: Tensor
    held_out_pass: Tensor
    utility_pass: Tensor
    provenance_pass: Tensor
    counterexample_pass: Tensor


@dataclass(frozen=True, slots=True)
class KnowledgeProposalState:
    """Fixed-capacity, versioned proposal ring carried by MRCRA runtime state."""

    latent: Tensor
    kind: Tensor
    status: Tensor
    code_gain_bits: Tensor
    reconstruction_distortion: Tensor
    relation_distortion: Tensor
    predictive_utility: Tensor
    action_utility: Tensor
    confidence: Tensor
    provenance_ids: Tensor
    supporting_provenance_ids: Tensor
    supporting_mask: Tensor
    held_out_loss_before: Tensor
    held_out_loss_after: Tensor
    counterexample_search_completed: Tensor
    versions: Tensor
    active: Tensor
    write_cursor: Tensor
    clock: int

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.latent.ndim != 3:
            raise ValueError("knowledge state latent must be (batch,capacity,width)")
        base = self.latent.shape[:2]
        for name in ("kind", "status", "provenance_ids", "versions"):
            value = getattr(self, name)
            if value.shape != base or value.dtype != torch.int64:
                raise ValueError(f"knowledge state {name} must be int64 with capacity shape")
        for name in (
            "code_gain_bits", "reconstruction_distortion", "relation_distortion",
            "predictive_utility", "action_utility", "confidence",
            "held_out_loss_before", "held_out_loss_after",
        ):
            if getattr(self, name).shape != base:
                raise ValueError(f"knowledge state {name} must have capacity shape")
        if self.supporting_provenance_ids.ndim != 3 or self.supporting_provenance_ids.shape[:2] != base:
            raise ValueError("knowledge state supporters must be (batch,capacity,supporters)")
        if self.supporting_mask.shape != self.supporting_provenance_ids.shape or self.supporting_mask.dtype != torch.bool:
            raise ValueError("knowledge state supporter mask is invalid")
        if self.counterexample_search_completed.shape != base or self.counterexample_search_completed.dtype != torch.bool:
            raise ValueError("knowledge counterexample status must have capacity shape")
        if self.active.shape != base or self.active.dtype != torch.bool:
            raise ValueError("knowledge active mask is invalid")
        if self.write_cursor.shape != (base[0],) or self.write_cursor.dtype != torch.int64:
            raise ValueError("knowledge write cursor must be int64 per batch")
        if not torch.equal(self.active, self.provenance_ids >= 0):
            raise ValueError("only active knowledge rows may retain provenance")
        if bool((~self.active & (self.status != int(KnowledgeStatus.EMPTY))).any()):
            raise ValueError("inactive knowledge rows must have EMPTY status")
        if bool((self.active & (self.status == int(KnowledgeStatus.EMPTY))).any()):
            raise ValueError("active knowledge rows cannot have EMPTY status")
        if bool((self.supporting_mask & (self.supporting_provenance_ids < 0)).any()):
            raise ValueError("active knowledge supporters require provenance")
        if self.clock < 0:
            raise ValueError("knowledge clock cannot be negative")

    @classmethod
    def empty(
        cls, batch: int, capacity: int, width: int, supporter_capacity: int, *,
        device=None, dtype=None,
    ) -> "KnowledgeProposalState":
        if min(batch, capacity, width, supporter_capacity) <= 0:
            raise ValueError("knowledge state dimensions must be positive")
        base = (batch, capacity)
        options = dict(device=device, dtype=dtype)
        return cls(
            torch.zeros(*base, width, **options),
            torch.full(base, -1, dtype=torch.int64, device=device),
            torch.zeros(base, dtype=torch.int64, device=device),
            torch.zeros(*base, **options), torch.zeros(*base, **options),
            torch.zeros(*base, **options), torch.zeros(*base, **options),
            torch.zeros(*base, **options), torch.zeros(*base, **options),
            torch.full(base, -1, dtype=torch.int64, device=device),
            torch.full((*base, supporter_capacity), -1, dtype=torch.int64, device=device),
            torch.zeros(*base, supporter_capacity, dtype=torch.bool, device=device),
            torch.full(base, torch.inf, **options),
            torch.full(base, torch.inf, **options),
            torch.zeros(base, dtype=torch.bool, device=device),
            torch.zeros(base, dtype=torch.int64, device=device),
            torch.zeros(base, dtype=torch.bool, device=device),
            torch.zeros(batch, dtype=torch.int64, device=device), 0,
        )

    @property
    def batch(self) -> int:
        return self.latent.shape[0]

    @property
    def capacity(self) -> int:
        return self.latent.shape[1]

    def detach(self) -> "KnowledgeProposalState":
        return _tensor_map(self, "detach")

    def to(self, *args, **kwargs) -> "KnowledgeProposalState":
        return _tensor_map(self, "to", *args, **kwargs)


class KnowledgeProposalBank:
    """Pure state transitions for proposal, validation, rejection, and revocation."""

    @staticmethod
    def propose(
        state: KnowledgeProposalState, proposals: KnowledgeProposalBatch,
    ) -> tuple[KnowledgeProposalState, Tensor]:
        if proposals.latent.shape[0] != state.batch or proposals.latent.shape[-1] != state.latent.shape[-1]:
            raise ValueError("knowledge proposal batch does not match state")
        if proposals.supporting_provenance_ids.shape[-1] > state.supporting_provenance_ids.shape[-1]:
            raise ValueError("knowledge proposal exceeds supporter capacity")
        values = {
            field.name: getattr(state, field.name).clone()
            if isinstance(getattr(state, field.name), Tensor) else getattr(state, field.name)
            for field in fields(state)
        }
        written = torch.full(
            (state.batch,), -1, dtype=torch.int64, device=state.latent.device
        )
        for row in torch.nonzero(proposals.mask, as_tuple=False).flatten().tolist():
            slot = int(state.write_cursor[row] % state.capacity)
            written[row] = slot
            values["latent"][row, slot] = proposals.latent[row]
            values["kind"][row, slot] = proposals.kind[row]
            values["status"][row, slot] = int(KnowledgeStatus.PENDING)
            for name in (
                "code_gain_bits", "reconstruction_distortion", "relation_distortion",
                "predictive_utility", "action_utility", "confidence",
            ):
                values[name][row, slot] = getattr(proposals, name)[row]
            values["provenance_ids"][row, slot] = proposals.provenance_ids[row]
            values["supporting_provenance_ids"][row, slot].fill_(-1)
            values["supporting_mask"][row, slot].zero_()
            count = proposals.supporting_provenance_ids.shape[-1]
            values["supporting_provenance_ids"][row, slot, :count] = proposals.supporting_provenance_ids[row]
            values["supporting_mask"][row, slot, :count] = proposals.supporting_mask[row]
            values["held_out_loss_before"][row, slot] = torch.inf
            values["held_out_loss_after"][row, slot] = torch.inf
            values["counterexample_search_completed"][row, slot] = False
            values["versions"][row, slot] += 1
            values["active"][row, slot] = True
            values["write_cursor"][row] = (slot + 1) % state.capacity
        values["clock"] = state.clock + int(proposals.mask.any())
        return KnowledgeProposalState(**values), written

    @staticmethod
    def validate(
        state: KnowledgeProposalState, evidence: KnowledgeValidationBatch,
        ledger: ProvenanceLedger, *, minimum_code_gain_bits: float,
        maximum_reconstruction_distortion: float,
        maximum_relation_distortion: float,
        held_out_tolerance: float = 0.0,
        require_independent_roots_for_invariants: int = 2,
    ) -> tuple[KnowledgeProposalState, KnowledgeValidationResult]:
        if min(
            minimum_code_gain_bits, maximum_reconstruction_distortion,
            maximum_relation_distortion, held_out_tolerance,
        ) < 0 or require_independent_roots_for_invariants < 1:
            raise ValueError("knowledge validation thresholds are invalid")
        if evidence.proposal_indices.shape[0] != state.batch:
            raise ValueError("knowledge validation batch does not match state")
        device = state.latent.device
        result_fields = [torch.zeros(state.batch, dtype=torch.bool, device=device) for _ in range(7)]
        accepted, code_pass, distortion_pass, held_pass, utility_pass, provenance_pass, counter_pass = result_fields
        values = {
            field.name: getattr(state, field.name).clone()
            if isinstance(getattr(state, field.name), Tensor) else getattr(state, field.name)
            for field in fields(state)
        }
        for row in torch.nonzero(evidence.mask, as_tuple=False).flatten().tolist():
            slot = int(evidence.proposal_indices[row])
            if not 0 <= slot < state.capacity or not bool(state.active[row, slot]):
                raise ValueError("knowledge validation references an inactive proposal")
            if int(state.status[row, slot]) != int(KnowledgeStatus.PENDING):
                raise ValueError("only pending knowledge proposals may be validated")
            code_pass[row] = state.code_gain_bits[row, slot] >= minimum_code_gain_bits
            distortion_pass[row] = (
                state.reconstruction_distortion[row, slot] <= maximum_reconstruction_distortion
                and state.relation_distortion[row, slot] <= maximum_relation_distortion
            )
            held_pass[row] = (
                evidence.held_out_loss_after[row]
                <= evidence.held_out_loss_before[row] + held_out_tolerance
            )
            utility_pass[row] = (
                evidence.predictive_utility[row] > 0 or evidence.action_utility[row] > 0
            )
            kind = KnowledgeKind(int(state.kind[row, slot]))
            supporters = state.supporting_provenance_ids[row, slot][state.supporting_mask[row, slot]]
            roots: set[int] = set()
            valid_support = True
            for provenance_id in supporters.tolist():
                try:
                    roots.update(ledger.independent_roots(int(provenance_id)))
                except KeyError:
                    valid_support = False
            required_roots = (
                require_independent_roots_for_invariants
                if kind == KnowledgeKind.INVARIANT else 1
            )
            provenance_pass[row] = valid_support and len(roots) >= required_roots
            counter_pass[row] = (
                bool(evidence.counterexample_search_completed[row])
                if kind == KnowledgeKind.INVARIANT else True
            )
            accepted[row] = (
                code_pass[row] and distortion_pass[row] and held_pass[row]
                and utility_pass[row] and provenance_pass[row] and counter_pass[row]
            )
            values["held_out_loss_before"][row, slot] = evidence.held_out_loss_before[row]
            values["held_out_loss_after"][row, slot] = evidence.held_out_loss_after[row]
            values["predictive_utility"][row, slot] = evidence.predictive_utility[row]
            values["action_utility"][row, slot] = evidence.action_utility[row]
            values["confidence"][row, slot] = evidence.calibrated_confidence[row]
            values["counterexample_search_completed"][row, slot] = evidence.counterexample_search_completed[row]
            values["status"][row, slot] = int(
                KnowledgeStatus.ACCEPTED if accepted[row] else KnowledgeStatus.REJECTED
            )
            values["versions"][row, slot] += 1
        values["clock"] = state.clock + int(evidence.mask.any())
        return KnowledgeProposalState(**values), KnowledgeValidationResult(*result_fields)

    @staticmethod
    def revoke(
        state: KnowledgeProposalState, proposal_indices: Tensor, mask: Tensor,
    ) -> KnowledgeProposalState:
        if proposal_indices.shape != (state.batch,) or proposal_indices.dtype != torch.int64:
            raise ValueError("knowledge revocation requires one int64 index per batch")
        if mask.shape != (state.batch,) or mask.dtype != torch.bool:
            raise ValueError("knowledge revocation mask must be boolean per batch")
        values = {
            field.name: getattr(state, field.name).clone()
            if isinstance(getattr(state, field.name), Tensor) else getattr(state, field.name)
            for field in fields(state)
        }
        for row in torch.nonzero(mask, as_tuple=False).flatten().tolist():
            slot = int(proposal_indices[row])
            if not 0 <= slot < state.capacity or not bool(state.active[row, slot]):
                raise ValueError("knowledge revocation references an inactive proposal")
            if int(state.status[row, slot]) != int(KnowledgeStatus.ACCEPTED):
                raise ValueError("only accepted knowledge may be revoked")
            values["status"][row, slot] = int(KnowledgeStatus.REVOKED)
            values["versions"][row, slot] += 1
        values["clock"] = state.clock + int(mask.any())
        return KnowledgeProposalState(**values)
