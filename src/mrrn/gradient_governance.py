"""Per-objective gradient telemetry and bounded conflict projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from .cognitive_objectives import ObjectiveFamily


@dataclass(frozen=True, slots=True)
class GradientGovernanceReport:
    families: tuple[str, ...]
    norms: Tensor
    cosine_similarity: Tensor
    conflict_mask: Tensor
    finite: bool


def objective_gradient_report(
    losses: Mapping[ObjectiveFamily, Tensor], parameters: Sequence[nn.Parameter],
) -> GradientGovernanceReport:
    """Measure family gradients without mutating ``.grad`` authority."""

    ordered = tuple(sorted(losses, key=int))
    if not ordered or not parameters:
        raise ValueError("gradient governance requires losses and parameters")
    flattened: list[Tensor] = []
    for family in ordered:
        loss = losses[family]
        if loss.ndim or not bool(torch.isfinite(loss)):
            raise ValueError(f"objective {family.name} must be one finite scalar")
        gradients = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True,
        )
        flattened.append(torch.cat([
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None else gradient.reshape(-1)
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]))
    matrix = torch.stack(flattened)
    norms = matrix.norm(dim=-1)
    normalized = matrix / norms[:, None].clamp_min(1e-12)
    cosine = normalized @ normalized.T
    active = norms > 0
    cosine = cosine * (active[:, None] & active[None])
    conflicts = (cosine < 0) & ~torch.eye(
        len(ordered), dtype=torch.bool, device=cosine.device
    )
    return GradientGovernanceReport(
        tuple(family.name for family in ordered), norms, cosine, conflicts,
        bool(torch.isfinite(matrix).all()),
    )


def project_conflicting_gradients(gradients: Tensor) -> Tensor:
    """Deterministic PCGrad-style projection over family gradient rows."""

    if gradients.ndim != 2 or gradients.shape[0] == 0:
        raise ValueError("gradient projection requires (families,parameters)")
    projected = gradients.clone()
    for left in range(projected.shape[0]):
        for right in range(projected.shape[0]):
            if left == right:
                continue
            dot = torch.dot(projected[left], gradients[right])
            denominator = gradients[right].square().sum()
            if bool(dot < 0) and bool(denominator > 0):
                projected[left] -= dot / denominator * gradients[right]
    return projected
