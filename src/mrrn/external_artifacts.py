"""Bounded live references to application-owned external cognitive artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .tensor_state import TensorStateMixin
from .cognitive_types import ModalityClass, SourceClass, SupportInterval
from .provenance import ProvenanceLedger
from .runtime_validation import runtime_validation_enabled


@dataclass(frozen=True, slots=True)
class ExternalArtifactState(TensorStateMixin):
    artifact_ids: Tensor
    content_digests: Tensor
    versions: Tensor
    creator_action_ids: Tensor
    provenance_ids: Tensor
    expected_persistence: Tensor
    estimated_cost: Tensor
    last_verified_time: Tensor
    readable: Tensor
    writable: Tensor
    active: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.artifact_ids.ndim != 2 or self.artifact_ids.dtype != torch.int64:
            raise ValueError("external artifact IDs must be (batch,capacity)")
        base = self.artifact_ids.shape
        if self.content_digests.ndim != 3 or self.content_digests.shape[:2] != base:
            raise ValueError("external artifact digests must match rows")
        for name in ("versions", "creator_action_ids", "provenance_ids"):
            value = getattr(self, name)
            if value.shape != base or value.dtype != torch.int64:
                raise ValueError(f"external artifact {name} must be int64 rows")
        for name in ("expected_persistence", "estimated_cost", "last_verified_time"):
            value = getattr(self, name)
            if value.shape != base or not value.is_floating_point():
                raise ValueError(f"external artifact {name} must match rows")
        for name in ("readable", "writable", "active"):
            value = getattr(self, name)
            if value.shape != base or value.dtype != torch.bool:
                raise ValueError(f"external artifact {name} must be boolean rows")
        if bool((self.active & ((self.artifact_ids < 0) | (self.provenance_ids < 0))).any()):
            raise ValueError("active external artifacts require IDs and provenance")
        if bool((self.expected_persistence < 0).any() | (self.estimated_cost < 0).any()):
            raise ValueError("external artifact persistence and cost cannot be negative")

    @classmethod
    def empty(
        cls, batch: int, capacity: int, digest_width: int = 32,
        *, device=None, dtype=None,
    ) -> "ExternalArtifactState":
        if min(batch, capacity, digest_width) <= 0:
            raise ValueError("external artifact dimensions must be positive")
        base = (batch, capacity)
        ids = lambda fill=-1: torch.full(base, fill, dtype=torch.int64, device=device)
        floats = lambda: torch.zeros(base, device=device, dtype=dtype)
        flags = lambda: torch.zeros(base, dtype=torch.bool, device=device)
        return cls(
            ids(), torch.zeros(*base, digest_width, dtype=torch.uint8, device=device),
            ids(0), ids(), ids(), floats(), floats(), floats(), flags(), flags(), flags(),
        )


def record_external_artifact(
    state: ExternalArtifactState, ledger: ProvenanceLedger, *,
    artifact_ids: Tensor, content_digests: Tensor, versions: Tensor,
    creator_action_ids: Tensor, parent_provenance_ids: Tensor,
    parent_mask: Tensor, expected_persistence: Tensor, estimated_cost: Tensor,
    timestamp: Tensor, readable: Tensor, writable: Tensor, create_mask: Tensor,
    model_authority: str,
) -> tuple[ExternalArtifactState, Tensor]:
    """Record application-owned artifacts without claiming they were observed."""

    batch, capacity = state.artifact_ids.shape
    integer_rows = (
        ("artifact_ids", artifact_ids), ("versions", versions),
        ("creator_action_ids", creator_action_ids),
    )
    for name, value in integer_rows:
        if value.shape != (batch,) or value.dtype != torch.int64:
            raise ValueError(f"external artifact {name} must be int64 per batch")
    if content_digests.shape != (batch, state.content_digests.shape[-1]) or content_digests.dtype != torch.uint8:
        raise ValueError("external artifact content digests are invalid")
    if parent_provenance_ids.ndim != 2 or parent_provenance_ids.shape[0] != batch or parent_provenance_ids.dtype != torch.int64:
        raise ValueError("external artifact parents must be int64 rows")
    if parent_mask.shape != parent_provenance_ids.shape or parent_mask.dtype != torch.bool:
        raise ValueError("external artifact parent mask is invalid")
    for name, value in (
        ("expected_persistence", expected_persistence),
        ("estimated_cost", estimated_cost), ("timestamp", timestamp),
    ):
        if value.shape != (batch,):
            raise ValueError(f"external artifact {name} must be per batch")
    for name, value in (
        ("readable", readable), ("writable", writable), ("create_mask", create_mask),
    ):
        if value.shape != (batch,) or value.dtype != torch.bool:
            raise ValueError(f"external artifact {name} must be boolean per batch")
    if bool((create_mask & ~parent_mask.any(-1)).any()):
        raise ValueError("created external artifacts require action provenance")
    values = {
        name: getattr(state, name).clone()
        for name in state.__dataclass_fields__
    }
    written = torch.full((batch,), -1, dtype=torch.int64, device=state.artifact_ids.device)
    for row in torch.nonzero(create_mask, as_tuple=False).flatten().tolist():
        duplicate = torch.nonzero(
            state.active[row] & (state.artifact_ids[row] == artifact_ids[row]),
            as_tuple=False,
        ).flatten()
        free = torch.nonzero(~state.active[row], as_tuple=False).flatten()
        if duplicate.numel():
            slot = int(duplicate[0])
            if int(versions[row]) <= int(state.versions[row, slot]):
                raise ValueError("external artifact versions must increase")
        elif free.numel():
            slot = int(free[0])
        else:
            continue
        parents = parent_provenance_ids[row][parent_mask[row]].tolist()
        provenance = ledger.derive(
            parents, source_class=SourceClass.EXTERNAL_ARTIFACT,
            operator="mrcra:external_artifact:create:v1",
            support=SupportInterval(float(timestamp[row]), float(timestamp[row]), float(timestamp[row])),
            modality=ModalityClass.MEMORY, scenario_id=0,
            model_authority=model_authority,
        )
        written[row] = slot
        values["artifact_ids"][row, slot] = artifact_ids[row]
        values["content_digests"][row, slot] = content_digests[row]
        values["versions"][row, slot] = versions[row]
        values["creator_action_ids"][row, slot] = creator_action_ids[row]
        values["provenance_ids"][row, slot] = provenance
        values["expected_persistence"][row, slot] = expected_persistence[row]
        values["estimated_cost"][row, slot] = estimated_cost[row]
        values["last_verified_time"][row, slot] = timestamp[row]
        values["readable"][row, slot] = readable[row]
        values["writable"][row, slot] = writable[row]
        values["active"][row, slot] = True
    return ExternalArtifactState(**values), written


def verify_external_artifact(
    state: ExternalArtifactState, *, artifact_ids: Tensor,
    content_digests: Tensor, versions: Tensor, timestamp: Tensor,
) -> tuple[ExternalArtifactState, Tensor]:
    """Detect stale or modified external artifacts by exact version and digest."""

    batch = state.artifact_ids.shape[0]
    if artifact_ids.shape != (batch,) or artifact_ids.dtype != torch.int64:
        raise ValueError("artifact verification IDs are invalid")
    if versions.shape != (batch,) or versions.dtype != torch.int64:
        raise ValueError("artifact verification versions are invalid")
    if content_digests.shape != (batch, state.content_digests.shape[-1]) or content_digests.dtype != torch.uint8:
        raise ValueError("artifact verification digests are invalid")
    if timestamp.shape != (batch,):
        raise ValueError("artifact verification timestamps are invalid")
    matches = torch.zeros(batch, dtype=torch.bool, device=state.artifact_ids.device)
    verified = state.last_verified_time.clone()
    for row in range(batch):
        slots = torch.nonzero(
            state.active[row] & (state.artifact_ids[row] == artifact_ids[row]),
            as_tuple=False,
        ).flatten()
        if not slots.numel():
            continue
        slot = int(slots[0])
        matches[row] = (
            int(state.versions[row, slot]) == int(versions[row])
            and torch.equal(state.content_digests[row, slot], content_digests[row])
        )
        if bool(matches[row]):
            verified[row, slot] = timestamp[row]
    values = {name: getattr(state, name) for name in state.__dataclass_fields__}
    values["last_verified_time"] = verified
    return ExternalArtifactState(**values), matches
