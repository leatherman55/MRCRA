"""Deterministic synthetic capability probes for the full verification gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import pi
from pathlib import Path

import torch
from torch import Tensor

from . import complex_ops as c
from .attention import linear_cross_correlation
from .lifting import LiftingAnalysisBank
from .memory import EideticMemory, MemoryItem
from .mixer import ResonantSpectralGLU
from .resonance import associative_affine_scan


@dataclass(frozen=True, slots=True)
class CapabilityResult:
    name: str
    metric: float
    threshold: float
    higher_is_better: bool
    passed: bool
    detail: str


def _result(name: str, metric: Tensor | float, threshold: float, higher: bool, detail: str) -> CapabilityResult:
    value = float(metric.detach()) if isinstance(metric, Tensor) else float(metric)
    return CapabilityResult(name, value, threshold, higher, value >= threshold if higher else value <= threshold, detail)


def multisine_recovery(
    length: int = 512, frequencies: tuple[int, ...] = (5, 17, 41),
    amplitudes: tuple[float, ...] = (1.0, 0.6, 0.3), phases: tuple[float, ...] = (0.2, -0.7, 1.1),
) -> tuple[Tensor, Tensor, Tensor]:
    if length <= 0 or not frequencies or not (len(frequencies) == len(amplitudes) == len(phases)):
        raise ValueError("aligned nonempty multisine parameters and positive length are required")
    time = torch.arange(length, dtype=torch.float64)
    signal = sum(amplitude * torch.cos(2 * pi * frequency * time / length + phase) for frequency, amplitude, phase in zip(frequencies, amplitudes, phases, strict=True))
    spectrum = torch.fft.rfft(signal) * (2 / length)
    recovered_amplitude = spectrum.abs()[list(frequencies)]
    recovered_phase = torch.angle(spectrum[list(frequencies)])
    return signal, recovered_amplitude, recovered_phase


def affine_state_tracking(events: Tensor, *, modulus: int | None = None) -> Tensor:
    """Represent depth accumulation or a modular counter in the same complex affine scan algebra."""

    if events.ndim != 2 or (modulus is not None and modulus < 2):
        raise ValueError("events must be (batch,time) and modulus at least two")
    if modulus is None:
        transition = c.pair(torch.ones_like(events), torch.zeros_like(events))
        drive = c.pair(events, torch.zeros_like(events))
        initial = c.pair(torch.zeros(events.shape[0], device=events.device, dtype=events.dtype), torch.zeros(events.shape[0], device=events.device, dtype=events.dtype))
        return c.real(associative_affine_scan(transition, drive, initial))
    angle = events * (2 * pi / modulus)
    transition = c.pair(torch.cos(angle), torch.sin(angle))
    drive = torch.zeros_like(transition)
    initial = c.pair(torch.ones(events.shape[0], device=events.device, dtype=events.dtype), torch.zeros(events.shape[0], device=events.device, dtype=events.dtype))
    return associative_affine_scan(transition, drive, initial)


def transient_trend_separation(length: int = 128, impulse_position: int = 63) -> tuple[Tensor, Tensor]:
    if not 0 <= impulse_position < length:
        raise ValueError("impulse must lie inside the sequence")
    bank = LiftingAnalysisBank(1, 4).double()
    time = torch.linspace(-1, 1, length, dtype=torch.float64)
    trend = (0.5 * time + 0.1).view(1, length, 1)
    transient = trend.clone()
    transient[:, impulse_position] += 5
    trend_bands, _ = bank(trend)
    transient_bands, _ = bank(transient)
    fine_increase = transient_bands[0].data.square().sum() - trend_bands[0].data.square().sum()
    coarse_trend_energy = trend_bands[-1].data.square().mean()
    return fine_increase, coarse_trend_energy


def delayed_match(query: Tensor, motif: Tensor) -> tuple[int, Tensor]:
    correlation, lags = linear_cross_correlation(query, motif)
    index = correlation.argmax(-1)
    if index.numel() != 1:
        raise ValueError("delayed_match expects one signal pair")
    return int(lags[index]), correlation


def phase_collision_margin(signal: Tensor) -> Tensor:
    if signal.ndim != 1 or signal.numel() < 3:
        raise ValueError("phase collision probe requires one nontrivial signal")
    collision = signal.flip(0)
    magnitude_error = (torch.fft.fft(signal).abs() - torch.fft.fft(collision).abs()).abs().max()
    own = (signal * signal).sum()
    wrong = (signal * collision).sum()
    return (own - wrong) / own.clamp_min(1e-12) - magnitude_error


def selective_copy_accuracy(distance: int = 10_000, symbols: int = 16) -> float:
    if distance <= 0 or symbols <= 1:
        raise ValueError("copy distance and symbol count are invalid")
    memory = EideticMemory(symbols, symbols, symbols, symbols)
    handles = []
    for symbol in range(symbols):
        value = torch.nn.functional.one_hot(torch.tensor(symbol), symbols).float()
        handles.append(memory.write(MemoryItem(value, value, value, symbol, 0, 1.0)))
    correct = 0
    for symbol, handle in enumerate(handles):
        value = memory.get(memory.rerank(memory.get(handle).key, memory.retrieve(memory.get(handle).signature, 4, query_time=distance), 1)[0]).value
        correct += int(value.argmax() == symbol)
    return correct / symbols


def regime_switch_adaptation(length: int = 128, switch: int = 64) -> tuple[Tensor, Tensor]:
    if not 0 < switch < length:
        raise ValueError("regime switch must lie inside the rollout")
    time = torch.arange(length, dtype=torch.float64)
    first = torch.sin(0.15 * time)
    second = torch.sin(0.75 * time)
    target = torch.where(time < switch, first, second)
    fixed, selective = torch.zeros((), dtype=torch.float64), torch.zeros((), dtype=torch.float64)
    fixed_trace, selective_trace = [], []
    for index in range(length):
        fixed = 0.98 * fixed + 0.02 * target[index]
        decay = 0.5 if switch <= index < switch + 16 else 0.98
        selective = decay * selective + (1 - decay) * target[index]
        fixed_trace.append(fixed)
        selective_trace.append(selective)
    fixed_error = (torch.stack(fixed_trace)[switch : switch + 16] - target[switch : switch + 16]).square().mean()
    selective_error = (torch.stack(selective_trace)[switch : switch + 16] - target[switch : switch + 16]).square().mean()
    return fixed_error, selective_error


def bounded_noise_rollout(length: int = 100_000, transition_magnitude: float = 0.999, drive_bound: float = 0.01) -> Tensor:
    if length <= 0 or not 0 <= transition_magnitude < 1 or drive_bound < 0:
        raise ValueError("stable noise rollout controls are invalid")
    generator = torch.Generator().manual_seed(2718)
    state, maximum = torch.zeros(()), torch.zeros(())
    for _ in range(length):
        drive = (2 * torch.rand((), generator=generator) - 1) * drive_bound
        state = transition_magnitude * state + drive
        maximum = torch.maximum(maximum, state.abs())
    theoretical_bound = drive_bound / (1 - transition_magnitude)
    return maximum / theoretical_bound


def learned_spectral_activation_separation() -> Tensor:
    """Deterministic margin proving two modes can learn distinct nonlinear transfer curves."""

    module = ResonantSpectralGLU(
        2, 1, 2, 1, basis_order=2, maximum_phase=0, triads_per_mode=0
    ).double()
    with torch.no_grad():
        module.gain_coefficients[0, 0, 0] = 2
        module.gain_coefficients[0, 1, 0] = -2
    real = torch.ones(1, 1, 1, 2, 1, dtype=torch.float64)
    modal = c.pair(real, torch.zeros_like(real))
    _, diagnostics = module.modal_activation(modal, modal)
    first = diagnostics.amplitude_gate[..., 0, :].mean()
    second = diagnostics.amplitude_gate[..., 1, :].mean()
    return first - second


def run_capability_suite() -> tuple[CapabilityResult, ...]:
    _, amplitudes, phases = multisine_recovery()
    expected_amplitudes = torch.tensor([1.0, 0.6, 0.3], dtype=amplitudes.dtype)
    expected_phases = torch.tensor([0.2, -0.7, 1.1], dtype=phases.dtype)
    amplitude_error = (amplitudes - expected_amplitudes).abs().max()
    phase_error = torch.atan2(torch.sin(phases - expected_phases), torch.cos(phases - expected_phases)).abs().max()
    events = torch.tensor([[1.0, 1.0, -1.0, 1.0, -1.0]], dtype=torch.float64)
    depth = affine_state_tracking(events)
    modular = affine_state_tracking(torch.ones_like(events), modulus=3)
    fine, coarse = transient_trend_separation()
    query = torch.zeros(1, 64)
    motif = torch.tensor([[1.0, -2.0, 3.0, 0.5]])
    query[:, 37:41] = motif
    lag, _ = delayed_match(query, motif)
    fixed_error, selective_error = regime_switch_adaptation()
    return (
        _result("multi_sine_amplitude", amplitude_error, 1e-10, False, "maximum recovered amplitude error"),
        _result("multi_sine_phase", phase_error, 1e-10, False, "maximum recovered phase error"),
        _result("nested_state_depth", (depth - events.cumsum(1)).abs().max(), 1e-12, False, "complex affine depth trace error"),
        _result("modular_counter", c.magnitude(modular).sub(1).abs().max(), 1e-12, False, "unit-circle modular state error"),
        _result("transient_detection", fine, 1.0, True, "fine-band impulse energy increase"),
        _result("trend_preservation", coarse, 1e-3, True, "coarsest trend energy"),
        _result("delayed_match", abs(lag - 37), 0, False, "known motif lag error"),
        _result("spectral_collision", phase_collision_margin(torch.tensor([1.0, 3.0, -2.0, 0.5, 4.0])), 0.1, True, "phase/order discrimination margin"),
        _result("selective_copy", selective_copy_accuracy(), 1.0, True, "exact remote symbol copy accuracy"),
        _result("regime_switch", selective_error / fixed_error, 0.9, False, "selective/fixed early-regime MSE ratio"),
        _result("noise_stability", bounded_noise_rollout(10_000), 1.0, False, "observed/theoretical BIBS bound"),
        _result(
            "learned_spectral_activation", learned_spectral_activation_separation(), 0.5, True,
            "mode-specific learned nonlinear gain separation",
        ),
    )


def save_capability_report(path: str | Path, results: tuple[CapabilityResult, ...]) -> None:
    Path(path).write_text(json.dumps({"all_passed": all(item.passed for item in results), "results": [asdict(item) for item in results]}, indent=2) + "\n")
