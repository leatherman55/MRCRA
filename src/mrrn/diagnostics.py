"""Interpretable mode, scale, attention, memory, and numerical-stability measurements."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Callable, Sequence

import torch
from torch import Tensor

from . import complex_ops as c
from .lifting import ScaleTensor
from .memory import EideticMemory, MemoryHandle
from .mixer import HybridMixerDiagnostics, ResonantSpectralGLU
from .resonance import ResonatorParameters, ResonatorState


@dataclass(frozen=True, slots=True)
class ModeReport:
    frequency: Tensor
    decay: Tensor
    quality_factor: Tensor
    half_life: Tensor
    occupancy: Tensor
    phase_entropy: Tensor
    phase_locking: Tensor
    dead_modes: Tensor
    transition_max: Tensor
    gradient_norm: Tensor | None


@dataclass(frozen=True, slots=True)
class ScaleReport:
    energy_fraction: Tensor
    valid_coefficients: Tensor
    supports: Tensor
    cross_scale_gate_magnitude: Tensor
    reconstruction_error: Tensor
    ablation_delta: Tensor | None


@dataclass(frozen=True, slots=True)
class AttentionReport:
    entropy: Tensor
    maximum_weight: Tensor
    kind_mass: Tensor
    band_mass: Tensor | None
    selected_lag_error: Tensor | None
    candidate_set_miss_rate: Tensor | None


@dataclass(frozen=True, slots=True)
class MemoryReport:
    occupancy: float
    writes_per_thousand: float
    eviction_rate: float
    router_recall: float
    retrieved_age_mean: float


@dataclass(frozen=True, slots=True)
class StabilityReport:
    maximum: Tensor
    median: Tensor
    p95: Tensor
    rms: Tensor
    all_finite: bool
    phase_drift: Tensor | None


@dataclass(frozen=True, slots=True)
class SpectralActivationReport:
    spectral_fraction_mean: Tensor
    amplitude_gate_mean: Tensor
    amplitude_gate_maximum: Tensor
    phase_utilization: Tensor
    triad_rms: Tensor
    gain_transfer_roughness: Tensor
    phase_transfer_roughness: Tensor
    triad_frequency_error: Tensor
    all_finite: bool


def alias_energy_fraction(signal: Tensor, *, cutoff_fraction: float = 0.75) -> Tensor:
    """Fraction of temporal energy above a chosen fraction of the Nyquist frequency."""

    if signal.ndim < 2 or signal.shape[1] < 2 or not 0 < cutoff_fraction < 1:
        raise ValueError("alias diagnostic requires time length >=2 and cutoff in (0,1)")
    spectrum = torch.fft.rfft(signal.float(), dim=1)
    power = spectrum.abs().square()
    normalized_frequency = torch.linspace(0, 1, power.shape[1], device=signal.device)
    high = normalized_frequency >= cutoff_fraction
    return power[:, high].sum() / power.sum().clamp_min(torch.finfo(power.dtype).tiny)


def spectral_activation_diagnostics(
    diagnostics: HybridMixerDiagnostics, module: ResonantSpectralGLU
) -> SpectralActivationReport:
    gate, phase, triad = (
        diagnostics.spectral.amplitude_gate,
        diagnostics.spectral.phase_rotation,
        diagnostics.spectral.triad,
    )
    if gate.shape != phase.shape or triad.shape[:-1] != gate.shape or triad.shape[-1] != 2:
        raise ValueError("spectral activation diagnostic tensors are inconsistent")
    if diagnostics.spectral_fraction.shape[:-1] != gate.shape[:2]:
        raise ValueError("spectral blend diagnostics do not share batch/time axes")
    gain_delta = module.gain_coefficients[:, 1:] - module.gain_coefficients[:, :-1]
    phase_delta = module.phase_coefficients[:, 1:] - module.phase_coefficients[:, :-1]
    zero = gate.new_zeros(())
    values = (diagnostics.spectral_fraction, gate, phase, triad)
    return SpectralActivationReport(
        diagnostics.spectral_fraction.mean(), gate.mean(), gate.max(),
        phase.abs().mean() / max(module.maximum_phase, torch.finfo(phase.dtype).eps),
        c.abs_squared(triad).mean().sqrt(),
        gain_delta.square().mean() if gain_delta.numel() else zero,
        phase_delta.square().mean() if phase_delta.numel() else zero,
        module.triad_frequency_error(),
        all(bool(torch.isfinite(value).all()) for value in values),
    )


def mode_diagnostics(
    parameters: ResonatorParameters,
    state: ResonatorState,
    *,
    gradient: Tensor | None = None,
    dead_threshold: float = 1e-5,
    phase_bins: int = 16,
) -> ModeReport:
    if dead_threshold < 0 or phase_bins < 2:
        raise ValueError("mode diagnostic thresholds are invalid")
    c.validate(state.value)
    decay = parameters.alpha.mean((0, 1))
    frequency = parameters.omega.mean((0, 1))
    quality = frequency.abs() / (2 * decay).clamp_min(1e-12)
    half_life = log(2.0) / decay.clamp_min(1e-12)
    amplitude = c.magnitude(state.value)
    occupancy = amplitude.mean((0, 3))
    phase = torch.atan2(c.imag(state.value), c.real(state.value))
    phase_locking = torch.sqrt(torch.cos(phase).mean((0, 3)).square() + torch.sin(phase).mean((0, 3)).square())
    entropy = []
    for head in range(phase.shape[1]):
        per_head = []
        for mode in range(phase.shape[2]):
            histogram = torch.histc(phase[:, head, mode].float(), bins=phase_bins, min=-torch.pi, max=torch.pi)
            probability = histogram / histogram.sum().clamp_min(1)
            per_head.append(-(probability * probability.clamp_min(1e-12).log()).sum() / log(phase_bins))
        entropy.append(torch.stack(per_head))
    phase_entropy = torch.stack(entropy).to(state.value)
    gradient_norm = None
    if gradient is not None:
        if gradient.shape != state.value.shape:
            raise ValueError("state gradient shape does not match state")
        gradient_norm = c.magnitude(gradient).square().mean((0, 3)).sqrt()
    return ModeReport(
        frequency, decay, quality, half_life, occupancy, phase_entropy, phase_locking,
        occupancy < dead_threshold, c.magnitude(parameters.transition).amax((0, 1, 4)), gradient_norm,
    )


def scale_diagnostics(
    bands: Sequence[ScaleTensor],
    *,
    reconstruction_error: Tensor,
    fine_gains: Tensor,
    coarse_gains: Tensor,
    ablated_losses: Tensor | None = None,
    full_loss: Tensor | None = None,
) -> ScaleReport:
    if not bands or reconstruction_error.numel() != 1:
        raise ValueError("scale bands and scalar reconstruction error are required")
    energy, valid = [], []
    for band in bands:
        weight = band.mask.unsqueeze(-1)
        valid.append(band.mask.sum())
        energy.append((band.data.square() * weight).sum() / weight.sum().clamp_min(1))
    energy = torch.stack(energy)
    energy_fraction = energy / energy.sum().clamp_min(1e-12)
    gains = torch.cat((fine_gains.abs().flatten(), coarse_gains.abs().flatten()))
    delta = None
    if ablated_losses is not None:
        if full_loss is None or ablated_losses.numel() != len(bands) or full_loss.numel() != 1:
            raise ValueError("scale ablations require one loss per band and a scalar full loss")
        delta = ablated_losses.flatten() - full_loss
    return ScaleReport(
        energy_fraction, torch.stack(valid),
        torch.tensor([band.support for band in bands], device=bands[0].data.device),
        gains.mean() if gains.numel() else bands[0].data.new_zeros(()), reconstruction_error, delta,
    )


def attention_diagnostics(
    weights: Tensor,
    *,
    kinds: Tensor | None = None,
    selected_lag: Tensor | None = None,
    true_lag: Tensor | None = None,
    candidate_indices: Tensor | None = None,
    dense_weights: Tensor | None = None,
) -> AttentionReport:
    if weights.ndim not in {4, 5}:
        raise ValueError("attention weights must be (batch,query,candidate,head[,band])")
    collapsed = weights.mean(-1) if weights.ndim == 5 else weights
    probability = collapsed / collapsed.sum(2, keepdim=True).clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(2).mean()
    maximum = probability.max(2).values.mean()
    if kinds is None:
        kind_mass = probability.new_tensor([1.0, 0.0, 0.0])
    else:
        if kinds.shape != weights.shape[:3]:
            raise ValueError("candidate kinds must match batch/query/candidate")
        kind_mass = torch.stack([
            (probability * (kinds == kind).unsqueeze(-1)).sum(2).mean() for kind in range(3)
        ])
    band_mass = None
    if weights.ndim == 5:
        contribution = weights.clamp_min(0)
        band_mass = contribution.sum(2).mean((0, 1, 2))
        band_mass = band_mass / band_mass.sum().clamp_min(1e-12)
    lag_error = None
    if selected_lag is not None or true_lag is not None:
        if selected_lag is None or true_lag is None or selected_lag.shape != true_lag.shape:
            raise ValueError("selected and true lags must be paired")
        lag_error = (selected_lag - true_lag).abs().float().mean()
    miss_rate = None
    if candidate_indices is not None or dense_weights is not None:
        if candidate_indices is None or dense_weights is None:
            raise ValueError("candidate indices and dense weights must be paired")
        if candidate_indices.ndim != 3 or dense_weights.ndim not in {3, 4}:
            raise ValueError("dense comparison expects (batch,query,candidate[,head]) weights and candidate indices")
        if candidate_indices.shape[:2] != dense_weights.shape[:2]:
            raise ValueError("candidate and dense query axes must match")
        dense_score = dense_weights.mean(-1) if dense_weights.ndim == 4 else dense_weights
        oracle = dense_score.argmax(-1, keepdim=True)
        miss_rate = (~(candidate_indices == oracle).any(-1)).float().mean()
    return AttentionReport(entropy, maximum, kind_mass, band_mass, lag_error, miss_rate)


def memory_diagnostics(
    memory: EideticMemory,
    *,
    processed_positions: int,
    retrieved: list[MemoryHandle] | None = None,
    oracle: list[MemoryHandle] | None = None,
    query_time: int | None = None,
) -> MemoryReport:
    if processed_positions < 0:
        raise ValueError("processed positions cannot be negative")
    recall = 1.0
    if retrieved is not None or oracle is not None:
        if retrieved is None or oracle is None:
            raise ValueError("retrieved and oracle handles must be paired")
        recall = EideticMemory.recall(retrieved, oracle, max(1, len(oracle)))
    ages = []
    if query_time is not None:
        for item in memory.items():
            ages.append(max(0, query_time - item.timestamp))
    return MemoryReport(
        len(memory) / memory.capacity,
        1000 * memory.writes / max(1, processed_positions),
        memory.evictions / max(1, memory.writes),
        recall,
        sum(ages) / len(ages) if ages else 0.0,
    )


def stability_diagnostics(states: Sequence[Tensor], reference_phases: Sequence[Tensor] | None = None) -> StabilityReport:
    if not states:
        raise ValueError("stability diagnostics require states")
    magnitudes = []
    for state in states:
        c.validate(state)
        magnitudes.append(c.magnitude(state).flatten())
    values = torch.cat(magnitudes)
    phase_drift = None
    if reference_phases is not None:
        if len(reference_phases) != len(states):
            raise ValueError("reference phases must name every state")
        errors = []
        for state, reference in zip(states, reference_phases, strict=True):
            phase = torch.atan2(c.imag(state), c.real(state))
            if phase.shape != reference.shape:
                raise ValueError("reference phase shape mismatch")
            errors.append(torch.atan2(torch.sin(phase - reference), torch.cos(phase - reference)).abs().mean())
        phase_drift = torch.stack(errors).mean()
    return StabilityReport(
        values.max(), values.median(), torch.quantile(values.float(), 0.95).to(values),
        values.square().mean().sqrt(), bool(torch.isfinite(values).all()), phase_drift,
    )


def estimate_local_lipschitz(
    function: Callable[[Tensor], Tensor], x: Tensor, *, epsilon: float = 1e-4, directions: int = 4,
) -> Tensor:
    if epsilon <= 0 or directions <= 0:
        raise ValueError("finite-difference controls must be positive")
    baseline = function(x)
    estimates = []
    for index in range(directions):
        generator = torch.Generator(device=x.device).manual_seed(index)
        direction = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        direction = direction / direction.norm().clamp_min(1e-12)
        estimates.append((function(x + epsilon * direction) - baseline).norm() / epsilon)
    return torch.stack(estimates).max()
