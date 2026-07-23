"""Typed requests for evidence that can discriminate live hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor

from .tensor_state import TensorStateMixin
from .runtime_validation import runtime_validation_enabled


@dataclass(frozen=True, slots=True)
class EvidenceRequestState(TensorStateMixin):
    proposition: Tensor
    requested_modalities: Tensor
    tool_schema_ids: Tensor
    hypothesis_indices: Tensor
    hypothesis_mask: Tensor
    expected_information_gain: Tensor
    maximum_cost: Tensor
    maximum_latency: Tensor
    required_precision: Tensor
    supporting_provenance_ids: Tensor
    supporting_mask: Tensor
    status: Tensor
    active: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.proposition.ndim != 3:
            raise ValueError("evidence request propositions must be (batch,capacity,width)")
        base = self.proposition.shape[:2]
        for name in ("requested_modalities", "tool_schema_ids", "status"):
            value = getattr(self, name)
            if value.shape != base or value.dtype != torch.int64:
                raise ValueError(f"evidence request {name} must be int64 rows")
        if self.hypothesis_indices.ndim != 3 or self.hypothesis_indices.shape[:2] != base or self.hypothesis_indices.dtype != torch.int64:
            raise ValueError("evidence request hypotheses must be int64 rows")
        if self.hypothesis_mask.shape != self.hypothesis_indices.shape or self.hypothesis_mask.dtype != torch.bool:
            raise ValueError("evidence request hypothesis mask is invalid")
        for name in ("expected_information_gain", "maximum_cost", "maximum_latency", "required_precision"):
            value = getattr(self, name)
            if value.shape != base or not value.is_floating_point():
                raise ValueError(f"evidence request {name} must match rows")
        if self.supporting_provenance_ids.ndim != 3 or self.supporting_provenance_ids.shape[:2] != base or self.supporting_provenance_ids.dtype != torch.int64:
            raise ValueError("evidence request supporters must be int64 rows")
        if self.supporting_mask.shape != self.supporting_provenance_ids.shape or self.supporting_mask.dtype != torch.bool:
            raise ValueError("evidence request supporter mask is invalid")
        if self.active.shape != base or self.active.dtype != torch.bool:
            raise ValueError("evidence request active mask is invalid")
        if bool((self.hypothesis_mask & (self.hypothesis_indices < 0)).any()):
            raise ValueError("active evidence request hypotheses require indices")
        if bool((self.supporting_mask & (self.supporting_provenance_ids < 0)).any()):
            raise ValueError("active evidence request supporters require provenance")
        if bool((self.maximum_cost < 0).any() | (self.maximum_latency < 0).any() | (self.required_precision < 0).any()):
            raise ValueError("evidence request limits cannot be negative")

    @classmethod
    def empty(
        cls, batch: int, capacity: int, width: int, hypothesis_capacity: int,
        supporter_capacity: int, *, device=None, dtype=None,
    ) -> "EvidenceRequestState":
        if min(batch, capacity, width, hypothesis_capacity, supporter_capacity) <= 0:
            raise ValueError("evidence request dimensions must be positive")
        base = (batch, capacity)
        return cls(
            torch.zeros(*base, width, device=device, dtype=dtype),
            torch.full(base, -1, dtype=torch.int64, device=device),
            torch.full(base, -1, dtype=torch.int64, device=device),
            torch.full((*base, hypothesis_capacity), -1, dtype=torch.int64, device=device),
            torch.zeros(*base, hypothesis_capacity, dtype=torch.bool, device=device),
            torch.zeros(base, device=device, dtype=dtype),
            torch.zeros(base, device=device, dtype=dtype),
            torch.zeros(base, device=device, dtype=dtype),
            torch.zeros(base, device=device, dtype=dtype),
            torch.full((*base, supporter_capacity), -1, dtype=torch.int64, device=device),
            torch.zeros(*base, supporter_capacity, dtype=torch.bool, device=device),
            torch.zeros(base, dtype=torch.int64, device=device),
            torch.zeros(base, dtype=torch.bool, device=device),
        )


class EvidenceRequestStatus(IntEnum):
    PENDING = 0
    DISPATCHED = 1
    RESOLVED = 2
    UNAVAILABLE = 3
    EXPIRED = 4
    REJECTED = 5


def create_evidence_request(
    state: EvidenceRequestState, *, proposition: Tensor,
    requested_modality: Tensor, tool_schema_id: Tensor,
    hypothesis_indices: Tensor, hypothesis_mask: Tensor,
    expected_information_gain: Tensor, maximum_cost: Tensor,
    maximum_latency: Tensor, required_precision: Tensor,
    supporting_provenance_ids: Tensor, supporting_mask: Tensor,
    create_mask: Tensor,
) -> tuple[EvidenceRequestState, Tensor]:
    """Insert one bounded typed request per active row."""

    batch, capacity = state.active.shape
    if proposition.shape != (batch, state.proposition.shape[-1]):
        raise ValueError("evidence request proposition has invalid shape")
    for name, value in (
        ("requested_modality", requested_modality),
        ("tool_schema_id", tool_schema_id),
    ):
        if value.shape != (batch,) or value.dtype != torch.int64:
            raise ValueError(f"evidence request {name} must be int64 per batch")
    if hypothesis_indices.shape != (batch, state.hypothesis_indices.shape[-1]) or hypothesis_indices.dtype != torch.int64:
        raise ValueError("evidence request hypothesis indices are invalid")
    if hypothesis_mask.shape != hypothesis_indices.shape or hypothesis_mask.dtype != torch.bool:
        raise ValueError("evidence request hypothesis mask is invalid")
    for name, value in (
        ("expected_information_gain", expected_information_gain),
        ("maximum_cost", maximum_cost), ("maximum_latency", maximum_latency),
        ("required_precision", required_precision),
    ):
        if value.shape != (batch,):
            raise ValueError(f"evidence request {name} must be per batch")
    if supporting_provenance_ids.shape != (batch, state.supporting_provenance_ids.shape[-1]) or supporting_provenance_ids.dtype != torch.int64:
        raise ValueError("evidence request supporters are invalid")
    if supporting_mask.shape != supporting_provenance_ids.shape or supporting_mask.dtype != torch.bool:
        raise ValueError("evidence request supporter mask is invalid")
    if create_mask.shape != (batch,) or create_mask.dtype != torch.bool:
        raise ValueError("evidence request create mask is invalid")
    if bool((create_mask & ~supporting_mask.any(-1)).any()):
        raise ValueError("active evidence requests require supporting provenance")
    values = {
        name: getattr(state, name).clone()
        for name in state.__dataclass_fields__
    }
    written = torch.full((batch,), -1, dtype=torch.int64, device=state.active.device)
    for row in torch.nonzero(create_mask, as_tuple=False).flatten().tolist():
        free = torch.nonzero(~state.active[row], as_tuple=False).flatten()
        if not free.numel():
            continue
        slot = int(free[0])
        written[row] = slot
        values["proposition"][row, slot] = proposition[row]
        values["requested_modalities"][row, slot] = requested_modality[row]
        values["tool_schema_ids"][row, slot] = tool_schema_id[row]
        values["hypothesis_indices"][row, slot] = hypothesis_indices[row]
        values["hypothesis_mask"][row, slot] = hypothesis_mask[row]
        values["expected_information_gain"][row, slot] = expected_information_gain[row]
        values["maximum_cost"][row, slot] = maximum_cost[row]
        values["maximum_latency"][row, slot] = maximum_latency[row]
        values["required_precision"][row, slot] = required_precision[row]
        values["supporting_provenance_ids"][row, slot] = supporting_provenance_ids[row]
        values["supporting_mask"][row, slot] = supporting_mask[row]
        values["status"][row, slot] = int(EvidenceRequestStatus.PENDING)
        values["active"][row, slot] = True
    return EvidenceRequestState(**values), written


def transition_evidence_requests(
    state: EvidenceRequestState, request_indices: Tensor, next_status: Tensor,
    mask: Tensor,
) -> EvidenceRequestState:
    batch, capacity = state.active.shape
    if request_indices.shape != (batch,) or request_indices.dtype != torch.int64:
        raise ValueError("evidence request transition indices are invalid")
    if next_status.shape != (batch,) or next_status.dtype != torch.int64:
        raise ValueError("evidence request transition statuses are invalid")
    if mask.shape != (batch,) or mask.dtype != torch.bool:
        raise ValueError("evidence request transition mask is invalid")
    if bool((mask & ((request_indices < 0) | (request_indices >= capacity))).any()):
        raise ValueError("evidence request transition index is out of bounds")
    allowed = {
        EvidenceRequestStatus.PENDING: {
            EvidenceRequestStatus.DISPATCHED, EvidenceRequestStatus.REJECTED,
            EvidenceRequestStatus.EXPIRED,
        },
        EvidenceRequestStatus.DISPATCHED: {
            EvidenceRequestStatus.RESOLVED, EvidenceRequestStatus.UNAVAILABLE,
            EvidenceRequestStatus.EXPIRED,
        },
    }
    status = state.status.clone()
    active = state.active.clone()
    for row in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        slot = int(request_indices[row])
        if not bool(state.active[row, slot]):
            raise ValueError("cannot transition an inactive evidence request")
        current = EvidenceRequestStatus(int(state.status[row, slot]))
        target = EvidenceRequestStatus(int(next_status[row]))
        if target not in allowed.get(current, set()):
            raise ValueError(f"invalid evidence request transition {current.name}->{target.name}")
        status[row, slot] = int(target)
        if target in {
            EvidenceRequestStatus.RESOLVED, EvidenceRequestStatus.UNAVAILABLE,
            EvidenceRequestStatus.EXPIRED, EvidenceRequestStatus.REJECTED,
        }:
            active[row, slot] = False
    values = {name: getattr(state, name) for name in state.__dataclass_fields__}
    values.update(status=status, active=active)
    return EvidenceRequestState(**values)
