"""Scoped continuity boundaries for document-safe persistent cognition."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .cognitive_types import BoundaryClass, BoundaryScope
from .tensor_state import TensorStateMixin
from .runtime_validation import runtime_validation_enabled


@dataclass(frozen=True, slots=True)
class BoundaryContextState(TensorStateMixin):
    scope: Tensor
    continuity_ids: Tensor
    environment_ids: Tensor
    session_ids: Tensor
    sequence_numbers: Tensor
    reset_counts: Tensor
    discontinuity: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        batch = self.scope.shape[0]
        for name in (
            "scope", "continuity_ids", "environment_ids", "session_ids",
            "sequence_numbers", "reset_counts",
        ):
            value = getattr(self, name)
            if value.shape != (batch,) or value.dtype != torch.int64:
                raise ValueError(f"boundary {name} must be int64 per batch")
        if self.discontinuity.shape != (batch,) or self.discontinuity.dtype != torch.bool:
            raise ValueError("boundary discontinuity must be boolean per batch")
        if bool(((self.scope < 0) | (self.scope >= len(BoundaryScope))).any()):
            raise ValueError("boundary scope is outside the ontology")
        if bool((self.sequence_numbers < 0).any() | (self.reset_counts < 0).any()):
            raise ValueError("boundary counters cannot be negative")

    @classmethod
    def empty(cls, batch: int, *, device=None) -> "BoundaryContextState":
        if batch <= 0:
            raise ValueError("boundary batch must be positive")
        integers = torch.zeros(batch, dtype=torch.int64, device=device)
        unknown = torch.full((batch,), -1, dtype=torch.int64, device=device)
        return cls(
            integers.clone(), unknown.clone(), unknown.clone(), unknown.clone(),
            integers.clone(), integers.clone(),
            torch.zeros(batch, dtype=torch.bool, device=device),
        )

    @property
    def batch(self) -> int:
        return self.scope.shape[0]

    def transition(
        self, scopes: Tensor, *, continuity_ids: Tensor | None = None,
        environment_ids: Tensor | None = None, session_ids: Tensor | None = None,
    ) -> "BoundaryContextState":
        if scopes.shape != (self.batch,) or scopes.dtype != torch.int64:
            raise ValueError("boundary transition scopes must be int64 per batch")
        if bool(((scopes < 0) | (scopes >= len(BoundaryScope))).any()):
            raise ValueError("boundary transition scope is outside the ontology")
        active = scopes != int(BoundaryScope.NONE)
        replacements = {
            "continuity_ids": continuity_ids,
            "environment_ids": environment_ids,
            "session_ids": session_ids,
        }
        values = {}
        for name, replacement in replacements.items():
            current = getattr(self, name)
            if replacement is None:
                values[name] = current
            else:
                if replacement.shape != (self.batch,) or replacement.dtype != torch.int64:
                    raise ValueError(f"boundary {name} update must be int64 per batch")
                values[name] = torch.where(active, replacement, current)
        discontinuity = scopes == int(BoundaryScope.STREAM_DISCONTINUITY)
        return BoundaryContextState(
            torch.where(active, scopes, self.scope),
            values["continuity_ids"], values["environment_ids"], values["session_ids"],
            self.sequence_numbers + 1,
            self.reset_counts + active.to(torch.int64),
            discontinuity,
        )


def legacy_scope(boundary: Tensor) -> Tensor:
    """Map legacy boundary strength to conservative explicit reset scope."""

    if boundary.dtype != torch.int64:
        raise ValueError("legacy boundary class must be int64")
    if bool(((boundary < 0) | (boundary > int(BoundaryClass.HARD))).any()):
        raise ValueError("legacy boundary class is outside the ontology")
    result = torch.full_like(boundary, int(BoundaryScope.NONE))
    result = torch.where(
        boundary == int(BoundaryClass.SOFT),
        torch.full_like(result, int(BoundaryScope.EVENT)), result,
    )
    result = torch.where(
        boundary == int(BoundaryClass.SEGMENT),
        torch.full_like(result, int(BoundaryScope.SEGMENT)), result,
    )
    return torch.where(
        boundary == int(BoundaryClass.HARD),
        torch.full_like(result, int(BoundaryScope.DOCUMENT)), result,
    )
