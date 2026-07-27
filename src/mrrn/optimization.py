"""Numerically conservative optimizer groups, schedules, and gradient controls."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, sqrt
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class OptimizerPolicy:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    pole_learning_rate_multiplier: float = 0.25
    warmup_steps: int = 1000
    total_steps: int = 100_000
    minimum_learning_rate_ratio: float = 0.1
    schedule: str = "cosine"

    def __post_init__(self) -> None:
        if min(self.learning_rate, self.pole_learning_rate_multiplier) <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer rates and decay are invalid")
        if self.warmup_steps < 0 or self.total_steps <= self.warmup_steps:
            raise ValueError("total steps must exceed nonnegative warmup")
        if not 0 <= self.minimum_learning_rate_ratio <= 1 or self.schedule not in {"cosine", "inverse_sqrt"}:
            raise ValueError("optimizer schedule controls are invalid")


@dataclass(frozen=True, slots=True)
class GradientReport:
    total_before_clip: Tensor
    total_after_clip: Tensor
    clip_coefficient: Tensor
    phase_norm: Tensor
    amplitude_norm: Tensor
    subsystem_norms_before: dict[str, Tensor]
    subsystem_norms_after: dict[str, Tensor]
    subsystem_tensor_counts: dict[str, int]
    finite: bool


@dataclass(frozen=True, slots=True)
class AuxiliaryGradientMergeReport:
    """Evidence from conflict projection and subsystem-relative norm caps."""

    applied: bool
    auxiliary_norm_before: Tensor
    auxiliary_norm_after: Tensor
    task_norm: Tensor
    conflicting_subsystems: tuple[str, ...]
    subsystem_scales: dict[str, Tensor]
    subsystem_auxiliary_norms_before: dict[str, Tensor]
    subsystem_auxiliary_norms_after: dict[str, Tensor]


def gradient_subsystem(name: str) -> str:
    """Map a parameter to one stable, low-cardinality architecture family."""

    if name.startswith("cstm_predictor."):
        return "cstm_head"
    if name.startswith("token_embedding.") or name.startswith("cognitive.carrier."):
        return "carrier"
    if name.startswith((
        "cognitive.event_extractor.", "cognitive.event_allocator.",
    )):
        return "event"
    if name.startswith("cognitive.output_context_adapter."):
        return "output_bridge"
    if name.startswith((
        "cognitive.controller.", "cognitive.operational_schemas.",
        "cognitive.metacognitive_router.", "cognitive.self_model_projection.",
        "cognitive.external_action_policy.", "cognitive.action_candidate",
        "cognitive.viability_",
    )):
        return "controller"
    if name.startswith((
        "cognitive.workspace_graph.", "cognitive.relation_writer.",
        "cognitive.cognitive_state_projection.", "cognitive.relational_",
        "cognitive.symbol_activator.", "cognitive.invariant_discoverer.",
    )):
        return "workspace_router"
    if name.startswith((
        "cognitive.hypothesis_bank.", "cognitive.world_model.",
        "cognitive.next_latent_predictor.", "cognitive.distributional_head.",
        "cognitive.uncertainty_", "cognitive.online_calibration.",
    )):
        return "world_hypothesis"
    if name.startswith((
        "cognitive.memory.", "cognitive.memory_key.",
        "cognitive.memory_signature.", "cognitive.memory_value.",
        "cognitive.memory_write_policy.",
    )):
        return "memory"
    if name.startswith("cognitive."):
        return "other_cognition"
    return "carrier"


@torch.no_grad()
def merge_auxiliary_gradients(
    model: nn.Module,
    auxiliary_gradients: Mapping[str, Tensor | None],
    subsystem_caps: Mapping[str, float],
    *,
    auxiliary_only_caps: Mapping[str, float] | None = None,
    epsilon: float = 1e-12,
) -> AuxiliaryGradientMergeReport:
    """Project task-conflicting auxiliary gradients and apply relative caps.

    The live ``parameter.grad`` tensors are treated as the primary task
    authority.  Auxiliary tensors are never allowed to introduce a negative
    aggregate dot product with that authority inside an architecture
    subsystem, and their post-projection norm is capped relative to the task
    norm before being added.
    """

    auxiliary_only_caps = auxiliary_only_caps or {}
    if epsilon <= 0 or any(
        not isinstance(value, (int, float))
        or not isfinite(float(value))
        or value < 0
        for value in (*subsystem_caps.values(), *auxiliary_only_caps.values())
    ):
        raise ValueError("auxiliary gradient caps must be finite and nonnegative")
    named = dict(model.named_parameters())
    unknown = set(auxiliary_gradients) - set(named)
    if unknown:
        raise ValueError(f"auxiliary gradients name unknown parameters: {sorted(unknown)}")
    grouped: dict[str, list[tuple[nn.Parameter, Tensor]]] = {}
    auxiliary_only: dict[str, list[tuple[nn.Parameter, Tensor]]] = {}
    anchor = next(model.parameters())
    task_total = sum(
        (
            parameter.grad.detach().float().square().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        ),
        start=anchor.new_zeros((), dtype=torch.float32),
    )
    if not bool(torch.isfinite(task_total)):
        raise FloatingPointError("primary task gradients became non-finite")
    aux_total = anchor.new_zeros((), dtype=torch.float32)
    for name, auxiliary in auxiliary_gradients.items():
        if auxiliary is None:
            continue
        parameter = named[name]
        if auxiliary.shape != parameter.shape or auxiliary.device != parameter.device:
            raise ValueError("auxiliary gradients must match their live parameters")
        if not bool(torch.isfinite(auxiliary).all()):
            raise FloatingPointError("auxiliary gradients became non-finite")
        aux_total += auxiliary.detach().float().square().sum()
        if parameter.grad is None:
            subsystem = gradient_subsystem(name)
            if subsystem in auxiliary_only_caps:
                auxiliary_only.setdefault(subsystem, []).append(
                    (parameter, auxiliary.detach())
                )
            # Ordinary actor parameters still fail closed when the primary task
            # exposes no path. Explicit auxiliary-only heads are the sole
            # exception and receive their own global-task-relative cap below.
            continue
        grouped.setdefault(gradient_subsystem(name), []).append(
            (parameter, auxiliary.detach())
        )
    conflicts: list[str] = []
    scales: dict[str, Tensor] = {}
    subsystem_before: dict[str, Tensor] = {}
    subsystem_after: dict[str, Tensor] = {}
    after_total = anchor.new_zeros((), dtype=torch.float32)
    for subsystem, rows in grouped.items():
        cap = float(subsystem_caps.get(subsystem, 0.0))
        task_norm_sq = sum(
            (parameter.grad.detach().float().square().sum() for parameter, _ in rows),
            start=anchor.new_zeros((), dtype=torch.float32),
        )
        dot = sum(
            (
                parameter.grad.detach().float()
                .mul(auxiliary.float())
                .sum()
                for parameter, auxiliary in rows
            ),
            start=anchor.new_zeros((), dtype=torch.float32),
        )
        projection = min(0.0, float(dot.cpu())) / max(
            float(task_norm_sq.cpu()), epsilon
        )
        if projection < 0:
            conflicts.append(subsystem)
        projected = [
            auxiliary - projection * parameter.grad.detach()
            for parameter, auxiliary in rows
        ]
        projected_norm = torch.stack([
            value.float().square().sum() for value in projected
        ]).sum().sqrt()
        subsystem_before[subsystem] = torch.stack([
            auxiliary.float().square().sum() for _, auxiliary in rows
        ]).sum().sqrt().detach()
        task_norm = task_norm_sq.sqrt()
        allowed = cap * task_norm
        scale = torch.clamp(
            allowed / projected_norm.clamp_min(epsilon), max=1.0
        )
        if cap == 0:
            scale = scale * 0
        scales[subsystem] = scale.detach()
        contributed_norm_squared = anchor.new_zeros((), dtype=torch.float32)
        for (parameter, _), value in zip(rows, projected, strict=True):
            contribution = value.to(parameter.grad.dtype) * scale.to(
                parameter.grad.dtype
            )
            parameter.grad.add_(contribution)
            contribution_squared = contribution.float().square().sum()
            contributed_norm_squared += contribution_squared
            after_total += contribution_squared
        subsystem_after[subsystem] = contributed_norm_squared.sqrt().detach()
    global_task_norm = task_total.sqrt()
    for subsystem, rows in auxiliary_only.items():
        cap = float(auxiliary_only_caps[subsystem])
        norm = torch.stack([
            value.float().square().sum() for _, value in rows
        ]).sum().sqrt()
        allowed = cap * global_task_norm
        scale = torch.clamp(allowed / norm.clamp_min(epsilon), max=1.0)
        if cap == 0:
            scale = scale * 0
        scale_key = (
            f"{subsystem}_auxiliary_only"
            if subsystem in scales else subsystem
        )
        scales[scale_key] = scale.detach()
        prior_before = subsystem_before.get(subsystem)
        subsystem_before[subsystem] = (
            norm.detach()
            if prior_before is None
            else (
                prior_before.float().square() + norm.float().square()
            ).sqrt().detach()
        )
        contributed_norm_squared = anchor.new_zeros((), dtype=torch.float32)
        for parameter, value in rows:
            contribution = value.to(parameter.dtype) * scale.to(parameter.dtype)
            parameter.grad = contribution.clone()
            contribution_squared = contribution.float().square().sum()
            contributed_norm_squared += contribution_squared
            after_total += contribution_squared
        contributed_norm = contributed_norm_squared.sqrt()
        prior_after = subsystem_after.get(subsystem)
        subsystem_after[subsystem] = (
            contributed_norm.detach()
            if prior_after is None
            else (
                prior_after.float().square()
                + contributed_norm.float().square()
            ).sqrt().detach()
        )
    return AuxiliaryGradientMergeReport(
        bool(grouped or auxiliary_only),
        aux_total.sqrt(),
        after_total.sqrt(),
        task_total.sqrt(),
        tuple(sorted(conflicts)),
        scales,
        subsystem_before,
        subsystem_after,
    )


def build_adamw(
    model: nn.Module,
    policy: OptimizerPolicy = OptimizerPolicy(),
    *,
    fused: bool = False,
) -> torch.optim.AdamW:
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {
        (False, False): [], (False, True): [], (True, False): [], (True, True): []
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        pole = any(token in name for token in ("raw_alpha", "raw_omega", "omega_projection", "alpha_projection", "raw_frequency"))
        no_decay = parameter.ndim <= 1 or any(token in name.lower() for token in ("norm", "bias", "phase", "raw_alpha", "raw_omega", "frequency"))
        groups[(pole, no_decay)].append(parameter)
    parameter_groups = []
    for (pole, no_decay), parameters in groups.items():
        if parameters:
            parameter_groups.append({
                "params": parameters,
                "lr": policy.learning_rate * (policy.pole_learning_rate_multiplier if pole else 1.0),
                "weight_decay": 0.0 if no_decay else policy.weight_decay,
                "pole_group": pole,
            })
    return torch.optim.AdamW(
        parameter_groups, betas=(0.9, 0.95), eps=1e-8, fused=fused
    )


def learning_rate_multiplier(step: int, policy: OptimizerPolicy) -> float:
    if step < 0:
        raise ValueError("optimizer step cannot be negative")
    if step < policy.warmup_steps:
        return (step + 1) / max(1, policy.warmup_steps)
    progress = min(1.0, (step - policy.warmup_steps) / (policy.total_steps - policy.warmup_steps))
    if policy.schedule == "cosine":
        decay = 0.5 * (1 + cos(pi * progress))
    else:
        decay = sqrt(max(1, policy.warmup_steps) / max(step + 1, policy.warmup_steps))
    return policy.minimum_learning_rate_ratio + (1 - policy.minimum_learning_rate_ratio) * decay


def build_scheduler(optimizer: torch.optim.Optimizer, policy: OptimizerPolicy) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_multiplier(step, policy))


def clip_and_report_gradients(model: nn.Module, *, maximum_norm: float) -> GradientReport:
    """Measure, validate, and clip gradients with one norm reduction.

    ``torch.nn.utils.clip_grad_norm_`` would calculate the global norm again
    after the phase/amplitude diagnostic traversal.  More importantly, the old
    finite check synchronized once per parameter on accelerators.  This routine
    reduces every diagnostic on-device, performs one finite-status read, and
    reuses the resulting coefficient for clipping.
    """

    if maximum_norm <= 0:
        raise ValueError("maximum gradient norm must be positive")
    named_gradients = tuple(
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    )
    if not named_gradients:
        zero = torch.tensor(0.0)
        return GradientReport(
            zero, zero, zero.new_tensor(1.0), zero, zero, {}, {}, {}, True
        )
    device = named_gradients[0][1].device
    phase_terms, amplitude_terms = [], []
    subsystem_terms: dict[str, list[Tensor]] = {}
    subsystem_tensor_counts: dict[str, int] = {}
    for name, gradient in named_gradients:
        norm_squared = gradient.float().square().sum()
        subsystem = gradient_subsystem(name)
        subsystem_terms.setdefault(subsystem, []).append(norm_squared)
        subsystem_tensor_counts[subsystem] = (
            subsystem_tensor_counts.get(subsystem, 0) + 1
        )
        if any(token in name for token in ("omega", "frequency", "phase")):
            phase_terms.append(norm_squared)
        else:
            amplitude_terms.append(norm_squared)
    zero = torch.zeros((), device=device)
    phase_squared = torch.stack(phase_terms).sum() if phase_terms else zero
    amplitude_squared = torch.stack(amplitude_terms).sum() if amplitude_terms else zero
    total = (phase_squared + amplitude_squared).sqrt()
    finite = bool(torch.isfinite(total))
    coefficient = (total.new_tensor(maximum_norm) / (total + 1e-6)).clamp_max(1.0)
    subsystem_before = {
        name: torch.stack(terms).sum().sqrt()
        for name, terms in subsystem_terms.items()
    }
    subsystem_after = {
        name: value * coefficient for name, value in subsystem_before.items()
    }
    if finite:
        # Grouping avoids foreach dtype/device restrictions while retaining a
        # single fused scaling launch for the usual homogeneous model.
        groups: dict[tuple[torch.device, torch.dtype], list[Tensor]] = {}
        for _, gradient in named_gradients:
            groups.setdefault((gradient.device, gradient.dtype), []).append(gradient)
        for gradients in groups.values():
            torch._foreach_mul_(gradients, coefficient.to(gradients[0].dtype))
    return GradientReport(
        total, total * coefficient, coefficient,
        phase_squared.sqrt(), amplitude_squared.sqrt(),
        subsystem_before, subsystem_after, subsystem_tensor_counts, finite,
    )
