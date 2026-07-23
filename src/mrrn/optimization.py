"""Numerically conservative optimizer groups, schedules, and gradient controls."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sqrt

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


def gradient_subsystem(name: str) -> str:
    """Map a parameter to one stable, low-cardinality architecture family."""

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
