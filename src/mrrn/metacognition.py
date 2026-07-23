"""Bounded technical self-prediction and reflective decision history."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cognitive_types import CognitiveTrigger
from .tensor_state import TensorStateMixin
from .runtime_validation import runtime_validation_enabled


@dataclass(frozen=True, slots=True)
class MetacognitiveState(TensorStateMixin):
    predicted_error: Tensor
    realized_error: Tensor
    value_of_compute: Tensor
    value_of_retrieval: Tensor
    value_of_reconstruction: Tensor
    value_of_simulation: Tensor
    value_of_evidence: Tensor
    calibration_error: Tensor
    decision_actions: Tensor
    trigger_classes: Tensor
    provenance_ids: Tensor
    reflection_depth: Tensor
    versions: Tensor
    active: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.predicted_error.ndim != 2:
            raise ValueError("metacognitive history must be (batch,capacity)")
        base = self.predicted_error.shape
        for name in (
            "realized_error", "value_of_compute", "value_of_retrieval",
            "value_of_reconstruction", "value_of_simulation", "value_of_evidence",
            "calibration_error",
        ):
            value = getattr(self, name)
            if value.shape != base or not value.is_floating_point():
                raise ValueError(f"metacognitive {name} must match rows")
        for name in (
            "decision_actions", "trigger_classes", "provenance_ids",
            "reflection_depth", "versions",
        ):
            value = getattr(self, name)
            if value.shape != base or value.dtype != torch.int64:
                raise ValueError(f"metacognitive {name} must be int64 rows")
        if self.active.shape != base or self.active.dtype != torch.bool:
            raise ValueError("metacognitive active mask is invalid")
        if bool((self.active & (self.provenance_ids < 0)).any()):
            raise ValueError("active metacognitive records require provenance")
        active_triggers = self.trigger_classes[self.active]
        if active_triggers.numel() and bool(((active_triggers < 0) | (active_triggers >= len(CognitiveTrigger))).any()):
            raise ValueError("metacognitive trigger is outside the ontology")
        if bool((self.reflection_depth < 0).any()):
            raise ValueError("metacognitive reflection depth cannot be negative")

    @classmethod
    def empty(cls, batch: int, capacity: int, *, device=None, dtype=None) -> "MetacognitiveState":
        if min(batch, capacity) <= 0:
            raise ValueError("metacognitive dimensions must be positive")
        base = (batch, capacity)
        floats = lambda: torch.zeros(base, device=device, dtype=dtype)
        ids = lambda fill=-1: torch.full(base, fill, dtype=torch.int64, device=device)
        return cls(
            floats(), floats(), floats(), floats(), floats(), floats(), floats(), floats(),
            ids(), ids(), ids(), ids(0), ids(0),
            torch.zeros(base, dtype=torch.bool, device=device),
        )


@dataclass(frozen=True, slots=True)
class MetacognitivePrediction(TensorStateMixin):
    predicted_error: Tensor
    value_of_compute: Tensor
    value_of_retrieval: Tensor
    value_of_reconstruction: Tensor
    value_of_simulation: Tensor
    value_of_evidence: Tensor
    calibration_error: Tensor
    trigger_logits: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.predicted_error.ndim != 1:
            raise ValueError("metacognitive predictions must be per batch")
        shape = self.predicted_error.shape
        for name in (
            "value_of_compute", "value_of_retrieval", "value_of_reconstruction",
            "value_of_simulation", "value_of_evidence", "calibration_error",
        ):
            if getattr(self, name).shape != shape:
                raise ValueError(f"metacognitive prediction {name} must be per batch")
        if self.trigger_logits.shape != (shape[0], len(CognitiveTrigger)):
            raise ValueError("metacognitive trigger logits are invalid")


class MetacognitiveRouter(nn.Module):
    """Predict the marginal utility of bounded internal operations."""

    def __init__(self, width: int, system_feature_dim: int, uncertainty_channels: int) -> None:
        super().__init__()
        if min(width, system_feature_dim, uncertainty_channels) <= 0:
            raise ValueError("metacognitive router dimensions must be positive")
        self.trunk = nn.Sequential(
            nn.Linear(2 * width + system_feature_dim + uncertainty_channels, width),
            nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
        )
        self.values = nn.Linear(width, 7)
        self.triggers = nn.Linear(width, len(CognitiveTrigger))

    def forward(
        self, workspace: Tensor, relational_context: Tensor,
        uncertainty: Tensor, system_features: Tensor,
    ) -> MetacognitivePrediction:
        if workspace.ndim != 2 or relational_context.shape != workspace.shape:
            raise ValueError("metacognitive workspace contexts are invalid")
        batch = workspace.shape[0]
        if uncertainty.shape[0] != batch or system_features.shape[0] != batch:
            raise ValueError("metacognitive uncertainty/system batches are invalid")
        hidden = self.trunk(torch.cat((
            workspace, relational_context, uncertainty, system_features,
        ), -1))
        values = self.values(hidden)
        positive = F.softplus(values[:, :6])
        return MetacognitivePrediction(
            positive[:, 0], positive[:, 1], positive[:, 2], positive[:, 3],
            positive[:, 4], positive[:, 5], torch.sigmoid(values[:, 6]),
            self.triggers(hidden),
        )


def append_metacognitive_record(
    state: MetacognitiveState, prediction: MetacognitivePrediction, *,
    realized_error: Tensor, decision_actions: Tensor, trigger_classes: Tensor,
    provenance_ids: Tensor, mask: Tensor, maximum_reflection_depth: int = 1,
) -> MetacognitiveState:
    """Append one bounded reflective receipt; it never edits source evidence."""

    batch, capacity = state.active.shape
    for name, value in (
        ("realized_error", realized_error), ("decision_actions", decision_actions),
        ("trigger_classes", trigger_classes), ("provenance_ids", provenance_ids),
    ):
        if value.shape != (batch,):
            raise ValueError(f"metacognitive record {name} must be per batch")
    if decision_actions.dtype != torch.int64 or trigger_classes.dtype != torch.int64 or provenance_ids.dtype != torch.int64:
        raise ValueError("metacognitive record IDs must be int64")
    if mask.shape != (batch,) or mask.dtype != torch.bool:
        raise ValueError("metacognitive record mask is invalid")
    if maximum_reflection_depth <= 0:
        raise ValueError("metacognitive reflection depth must be positive")
    values = {name: getattr(state, name).clone() for name in state.__dataclass_fields__}
    for row in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        free = torch.nonzero(~state.active[row], as_tuple=False).flatten()
        slot = int(free[0]) if free.numel() else int(state.versions[row].argmin())
        values["predicted_error"][row, slot] = prediction.predicted_error[row]
        values["realized_error"][row, slot] = realized_error[row]
        values["value_of_compute"][row, slot] = prediction.value_of_compute[row]
        values["value_of_retrieval"][row, slot] = prediction.value_of_retrieval[row]
        values["value_of_reconstruction"][row, slot] = prediction.value_of_reconstruction[row]
        values["value_of_simulation"][row, slot] = prediction.value_of_simulation[row]
        values["value_of_evidence"][row, slot] = prediction.value_of_evidence[row]
        values["calibration_error"][row, slot] = prediction.calibration_error[row]
        values["decision_actions"][row, slot] = decision_actions[row]
        values["trigger_classes"][row, slot] = trigger_classes[row]
        values["provenance_ids"][row, slot] = provenance_ids[row]
        values["reflection_depth"][row, slot] = min(maximum_reflection_depth, 1)
        values["versions"][row, slot] += 1
        values["active"][row, slot] = True
    return MetacognitiveState(**values)
