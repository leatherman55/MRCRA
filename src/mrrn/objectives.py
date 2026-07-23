"""Composable task, predictive, retrieval, pole, energy, routing, and physics losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from . import complex_ops as c
from .mixer import ResonantSpectralGLU


@dataclass(frozen=True, slots=True)
class LossWeights:
    predictive: float = 0.0
    retrieval: float = 0.0
    pole: float = 0.0
    energy: float = 0.0
    routing: float = 0.0
    physical: float = 0.0
    spectral: float = 0.0

    def __post_init__(self) -> None:
        if min(self.predictive, self.retrieval, self.pole, self.energy, self.routing, self.physical, self.spectral) < 0:
            raise ValueError("loss weights cannot be negative")


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    total: Tensor
    task: Tensor
    predictive: Tensor
    retrieval: Tensor
    pole: Tensor
    energy: Tensor
    routing: Tensor
    physical: Tensor
    spectral: Tensor


def spectral_activation_regularization(
    modules: Sequence[ResonantSpectralGLU], *, smoothness: float = 1.0,
    phase: float = 0.1, triad: float = 0.1,
) -> Tensor:
    """Penalize jagged mode responses, excessive phase response, and dense triad use."""

    if not modules or min(smoothness, phase, triad) < 0:
        raise ValueError("spectral modules must be nonempty and regularization weights nonnegative")
    losses = []
    for module in modules:
        gain_delta = module.gain_coefficients[:, 1:] - module.gain_coefficients[:, :-1]
        phase_delta = module.phase_coefficients[:, 1:] - module.phase_coefficients[:, :-1]
        zero = module.gain_coefficients.sum() * 0
        roughness = (
            gain_delta.square().mean() + phase_delta.square().mean()
            if gain_delta.numel() else zero
        )
        triad_penalty = (
            torch.tanh(module.raw_triad_weight).square().mean()
            if module.raw_triad_weight.numel() else zero
        )
        context_phase = module.context.weight.unflatten(0, (module.heads, module.modes, 2))[:, :, 1]
        losses.append(
            smoothness * roughness
            + phase * (module.phase_coefficients.square().mean() + context_phase.square().mean())
            + triad * triad_penalty
        )
    return torch.stack(losses).mean()


def supervised_task_loss(prediction: Tensor, target: Tensor, *, kind: str = "mse", mask: Tensor | None = None) -> Tensor:
    if kind == "cross_entropy":
        if prediction.ndim < 2 or target.shape != prediction.shape[:-1]:
            raise ValueError("classification targets must match prediction without the class axis")
        loss = F.cross_entropy(prediction.flatten(0, -2), target.flatten(), reduction="none").reshape(target.shape)
    elif kind in {"mse", "reconstruction"}:
        if prediction.shape != target.shape:
            raise ValueError("regression/reconstruction target must match prediction")
        loss = (prediction - target).square().mean(-1) if prediction.ndim > 1 else (prediction - target).square()
    elif kind == "l1":
        if prediction.shape != target.shape:
            raise ValueError("L1 target must match prediction")
        loss = (prediction - target).abs().mean(-1) if prediction.ndim > 1 else (prediction - target).abs()
    else:
        raise ValueError("unsupported task loss kind")
    if mask is None:
        return loss.mean()
    if mask.shape != loss.shape or mask.dtype != torch.bool:
        raise ValueError("task mask must be boolean and match the unreduced loss")
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def predictive_state_loss(
    predictions: Sequence[Tensor], targets: Sequence[Tensor], masks: Sequence[Tensor] | None = None,
    scale_weights: Sequence[float] | None = None,
) -> Tensor:
    if not predictions or len(predictions) != len(targets):
        raise ValueError("predictive state sequences must be nonempty and paired")
    if masks is not None and len(masks) != len(predictions):
        raise ValueError("predictive masks must name every scale")
    weights = [1.0] * len(predictions) if scale_weights is None else list(scale_weights)
    if len(weights) != len(predictions) or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("scale weights must be nonnegative, aligned, and nonzero")
    losses = []
    for index, (prediction, target, weight) in enumerate(zip(predictions, targets, weights, strict=True)):
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError("state predictions/targets must share (batch,time,width)")
        error = (prediction[:, :-1] - target[:, 1:].detach()).square().mean(-1)
        if masks is not None:
            mask = masks[index]
            if mask.shape != prediction.shape[:2] or mask.dtype != torch.bool:
                raise ValueError("state masks must match batch/time")
            valid = mask[:, :-1] & mask[:, 1:]
            value = (error * valid).sum() / valid.sum().clamp_min(1)
        else:
            value = error.mean() if error.numel() else prediction.sum() * 0
        losses.append(weight * value)
    return sum(losses) / sum(weights)


def retrieval_contrastive_loss(positive: Tensor, negatives: Tensor, *, temperature: float = 0.1) -> Tensor:
    if positive.ndim != 1 or negatives.ndim != 2 or negatives.shape[0] != positive.shape[0] or temperature <= 0:
        raise ValueError("retrieval scores must be positive=(batch), negatives=(batch,count), temperature>0")
    logits = torch.cat((positive[:, None], negatives), 1) / temperature
    return F.cross_entropy(logits, torch.zeros(positive.shape[0], dtype=torch.long, device=positive.device))


def pole_coverage_loss(alpha: Tensor, omega: Tensor, *, sigma_tau: float = 1.0, sigma_omega: float = 1.0) -> Tensor:
    if alpha.shape != omega.shape or alpha.ndim < 1 or min(sigma_tau, sigma_omega) <= 0 or not bool((alpha > 0).all()):
        raise ValueError("positive aligned poles and positive bandwidths are required")
    if alpha.shape[-1] < 2:
        return alpha.sum() * 0
    log_tau = -alpha.log()
    tau_delta = log_tau.unsqueeze(-1) - log_tau.unsqueeze(-2)
    omega_delta = omega.unsqueeze(-1) - omega.unsqueeze(-2)
    similarity = torch.exp(-(tau_delta / sigma_tau).square() - (omega_delta / sigma_omega).square())
    off_diagonal = ~torch.eye(alpha.shape[-1], dtype=torch.bool, device=alpha.device)
    return similarity[..., off_diagonal].mean()


def state_energy_loss(states: Sequence[Tensor] | Tensor, *, maximum_rms: float = 10.0) -> Tensor:
    if maximum_rms <= 0:
        raise ValueError("maximum state RMS must be positive")
    states = (states,) if isinstance(states, Tensor) else tuple(states)
    if not states:
        raise ValueError("at least one state tensor is required")
    energies = []
    for state in states:
        c.validate(state)
        energy = c.abs_squared(state).mean()
        energies.append(energy + F.relu(energy.sqrt() - maximum_rms).square())
    return torch.stack(energies).mean()


def router_balance_loss(
    probabilities: Tensor, *, entropy_floor: float = 0.25, capacity_factor: float = 1.5
) -> Tensor:
    if probabilities.ndim < 2 or not 0 <= entropy_floor <= 1 or capacity_factor < 1:
        raise ValueError("router probabilities and balance limits are invalid")
    if bool((probabilities < 0).any()):
        raise ValueError("router probabilities cannot be negative")
    probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    normalized_entropy = entropy / torch.tensor(probabilities.shape[-1], device=probabilities.device).log().clamp_min(1)
    entropy_penalty = F.relu(entropy_floor - normalized_entropy).square().mean()
    load = probabilities.reshape(-1, probabilities.shape[-1]).mean(0)
    capacity = capacity_factor / probabilities.shape[-1]
    return entropy_penalty + F.relu(load - capacity).square().sum()


def physical_constraint_loss(terms: Mapping[str, Tensor | Callable[[], Tensor]]) -> Tensor:
    if not terms:
        raise ValueError("physical constraints must be supplied by the domain")
    values = []
    for name, term in terms.items():
        value = term() if callable(term) else term
        if not isinstance(value, Tensor) or value.numel() != 1 or not bool(torch.isfinite(value)):
            raise ValueError(f"physical constraint {name!r} must be a finite scalar tensor")
        values.append(value)
    return torch.stack(values).sum()


def sobolev_spectral_loss(
    prediction: Tensor, target: Tensor, *, spacing: Sequence[float], order: float = 1.0,
    periodic: bool,
) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim < 3 or len(spacing) != prediction.ndim - 2:
        raise ValueError("field shapes and one spacing per spatial axis are required")
    if min(spacing) <= 0 or order < 0:
        raise ValueError("spacing must be positive and Sobolev order nonnegative")
    if not periodic:
        raise ValueError("FFT Sobolev weighting requires an explicitly periodic domain")
    spatial_axes = tuple(range(1, prediction.ndim - 1))
    error = prediction - target
    spectrum = torch.fft.rfftn(error, dim=spatial_axes)
    frequencies = [
        (torch.fft.rfftfreq(size, d=step, device=prediction.device) if axis == spatial_axes[-1]
         else torch.fft.fftfreq(size, d=step, device=prediction.device))
        for axis, (size, step) in enumerate(zip(prediction.shape[1:-1], spacing, strict=True), start=1)
    ]
    mesh = torch.meshgrid(*frequencies, indexing="ij")
    weight = (1 + sum((2 * torch.pi * frequency).square() for frequency in mesh)).pow(order)
    return (spectrum.abs().square() * weight.unsqueeze(0).unsqueeze(-1)).mean()


def combine_losses(
    task: Tensor,
    weights: LossWeights,
    *,
    predictive: Tensor | None = None,
    retrieval: Tensor | None = None,
    pole: Tensor | None = None,
    energy: Tensor | None = None,
    routing: Tensor | None = None,
    physical: Tensor | None = None,
    spectral: Tensor | None = None,
) -> LossBreakdown:
    if task.numel() != 1:
        raise ValueError("task loss must be scalar")
    zero = task * 0
    components = tuple(zero if value is None else value for value in (predictive, retrieval, pole, energy, routing, physical, spectral))
    if any(value.numel() != 1 for value in components):
        raise ValueError("every auxiliary loss must be scalar")
    total = task + sum(weight * value for weight, value in zip(
        (weights.predictive, weights.retrieval, weights.pole, weights.energy, weights.routing, weights.physical, weights.spectral),
        components, strict=True,
    ))
    return LossBreakdown(total, task, *components)
