"""Hard viability envelopes and measured resource state."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .tensor_state import TensorStateMixin
from .runtime_validation import runtime_validation_enabled


@dataclass(frozen=True, slots=True)
class ViabilityState(TensorStateMixin):
    values: Tensor
    target_low: Tensor
    target_high: Tensor
    hard_low: Tensor
    hard_high: Tensor
    trend: Tensor
    uncertainty: Tensor
    reserve: Tensor
    recovery_priority: Tensor
    authority_mask: Tensor
    provenance_ids: Tensor
    active: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.values.ndim != 2:
            raise ValueError("viability values must be (batch,channels)")
        base = self.values.shape
        for name in (
            "target_low", "target_high", "hard_low", "hard_high", "trend",
            "uncertainty", "reserve", "recovery_priority",
        ):
            value = getattr(self, name)
            if value.shape != base or not value.is_floating_point():
                raise ValueError(f"viability {name} must match values")
        for name in ("authority_mask", "active"):
            value = getattr(self, name)
            if value.shape != base or value.dtype != torch.bool:
                raise ValueError(f"viability {name} must be boolean channels")
        if self.provenance_ids.shape != base or self.provenance_ids.dtype != torch.int64:
            raise ValueError("viability provenance must match channels")
        if bool((self.target_low > self.target_high).any() | (self.hard_low > self.hard_high).any()):
            raise ValueError("viability intervals are inverted")
        if bool((self.active & ((self.target_low < self.hard_low) | (self.target_high > self.hard_high))).any()):
            raise ValueError("active viability targets must lie inside hard bounds")
        if bool((self.active & self.authority_mask & (self.provenance_ids < 0)).any()):
            raise ValueError("authoritative viability channels require provenance")
        if bool((self.uncertainty < 0).any() | (self.reserve < 0).any() | (self.recovery_priority < 0).any()):
            raise ValueError("viability uncertainty, reserve, and recovery priority cannot be negative")

    @classmethod
    def empty(cls, batch: int, channels: int, *, device=None, dtype=None) -> "ViabilityState":
        if min(batch, channels) <= 0:
            raise ValueError("viability dimensions must be positive")
        base = (batch, channels)
        zeros = lambda: torch.zeros(base, device=device, dtype=dtype)
        return cls(
            zeros(), zeros(), zeros(), zeros(), zeros(), zeros(), zeros(), zeros(), zeros(),
            torch.zeros(base, dtype=torch.bool, device=device),
            torch.full(base, -1, dtype=torch.int64, device=device),
            torch.zeros(base, dtype=torch.bool, device=device),
        )

    @property
    def batch(self) -> int:
        return self.values.shape[0]

    @property
    def hard_violation(self) -> Tensor:
        return self.active & ((self.values < self.hard_low) | (self.values > self.hard_high))

    @property
    def within_target(self) -> Tensor:
        return self.active & ((self.values >= self.target_low) & (self.values <= self.target_high))


@dataclass(frozen=True, slots=True)
class ViabilityForecast(TensorStateMixin):
    values: Tensor
    uncertainty: Tensor
    candidate_mask: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.values.ndim != 3:
            raise ValueError("viability forecast must be (batch,candidates,channels)")
        if self.uncertainty.shape != self.values.shape or not self.uncertainty.is_floating_point():
            raise ValueError("viability forecast uncertainty must match values")
        if self.candidate_mask.shape != self.values.shape[:2] or self.candidate_mask.dtype != torch.bool:
            raise ValueError("viability forecast candidate mask is invalid")
        if bool((self.uncertainty < 0).any()):
            raise ValueError("viability forecast uncertainty cannot be negative")


@dataclass(frozen=True, slots=True)
class ViabilityDecision(TensorStateMixin):
    authorized: Tensor
    violation_probability: Tensor
    minimum_hard_margin: Tensor
    recovery_required: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.authorized.ndim != 2 or self.authorized.dtype != torch.bool:
            raise ValueError("viability decisions must be boolean candidate rows")
        if self.violation_probability.shape != self.authorized.shape:
            raise ValueError("viability violation probability must match candidates")
        if self.minimum_hard_margin.shape != self.authorized.shape:
            raise ValueError("viability margins must match candidates")
        if self.recovery_required.shape != (self.authorized.shape[0],) or self.recovery_required.dtype != torch.bool:
            raise ValueError("viability recovery flag must be per batch")
        if bool(((self.violation_probability < 0) | (self.violation_probability > 1)).any()):
            raise ValueError("viability violation probability must lie in [0,1]")


class CandidateViabilityForecaster(nn.Module):
    """Predict bounded candidate effects while retaining measured state authority."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("viability forecaster channels must be positive")
        self.channels = channels
        self.delta = nn.Linear(6, channels)
        self.uncertainty = nn.Linear(6, channels)

    def forward(
        self, state: ViabilityState, *, expected_reward: Tensor,
        expected_cost: Tensor, constraint_probability: Tensor,
        expected_success: Tensor, expected_energy: Tensor, tail_risk: Tensor,
        candidate_mask: Tensor,
    ) -> ViabilityForecast:
        base = expected_reward.shape
        if state.values.shape[0] != base[0] or state.values.shape[-1] != self.channels:
            raise ValueError("viability state does not match forecaster")
        for name, value in (
            ("expected_cost", expected_cost),
            ("constraint_probability", constraint_probability),
            ("expected_success", expected_success),
            ("expected_energy", expected_energy), ("tail_risk", tail_risk),
        ):
            if value.shape != base:
                raise ValueError(f"viability candidate {name} must match reward")
        if candidate_mask.shape != base or candidate_mask.dtype != torch.bool:
            raise ValueError("viability candidate mask is invalid")
        features = torch.stack((
            expected_reward, expected_cost, constraint_probability,
            expected_success, expected_energy, tail_risk,
        ), -1)
        learned_delta = self.delta(features)
        # Resource costs have an explicit conservative effect on the first
        # practical reserve channels even before the learned model calibrates.
        authority_delta = torch.zeros_like(learned_delta)
        if self.channels > 0:
            authority_delta[..., 0] = -expected_energy
        if self.channels > 1:
            authority_delta[..., 1] = -expected_cost
        if self.channels > 2:
            authority_delta[..., 2] = -constraint_probability
        values = state.values.detach()[:, None] + learned_delta + authority_delta
        uncertainty = F.softplus(self.uncertainty(features)) + state.uncertainty.detach()[:, None]
        return ViabilityForecast(
            values.masked_fill(~candidate_mask[..., None], 0),
            uncertainty.masked_fill(~candidate_mask[..., None], 0), candidate_mask,
        )


class ViabilityGate(nn.Module):
    """A non-learned conservative hard-envelope authority."""

    def __init__(self, *, maximum_violation_probability: float = 0.05, sigma_multiplier: float = 2.0) -> None:
        super().__init__()
        if not 0 <= maximum_violation_probability <= 1 or sigma_multiplier < 0:
            raise ValueError("viability gate controls are invalid")
        self.maximum_violation_probability = maximum_violation_probability
        self.sigma_multiplier = sigma_multiplier

    def forward(self, state: ViabilityState, forecast: ViabilityForecast) -> ViabilityDecision:
        if forecast.values.shape[0] != state.batch or forecast.values.shape[-1] != state.values.shape[-1]:
            raise ValueError("viability forecast and state dimensions differ")
        active = state.active.detach()[:, None]
        authoritative = state.authority_mask.detach()[:, None]
        controlled = active & authoritative
        lower_margin = forecast.values - state.hard_low.detach()[:, None]
        upper_margin = state.hard_high.detach()[:, None] - forecast.values
        conservative_margin = torch.minimum(lower_margin, upper_margin) - (
            self.sigma_multiplier * forecast.uncertainty
        )
        safe_channel = ~controlled | (conservative_margin >= 0)
        # Gaussian tail is used only as calibrated telemetry.  Authorization is
        # based on the conservative hard margin above.
        scale = forecast.uncertainty.clamp_min(1e-6)
        lower_z = lower_margin / scale
        upper_z = upper_margin / scale
        violation = torch.maximum(
            torch.special.ndtr(-lower_z), torch.special.ndtr(-upper_z)
        ).masked_fill(~controlled, 0)
        maximum_violation = violation.amax(-1)
        configured = controlled.any(-1)
        authorized = (
            forecast.candidate_mask & configured
            & safe_channel.all(-1)
            & (maximum_violation <= self.maximum_violation_probability)
        )
        minimum = conservative_margin.masked_fill(~controlled, torch.inf).amin(-1)
        minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))
        return ViabilityDecision(
            authorized, maximum_violation, minimum,
            configured.any(-1) & ~authorized.any(-1),
        )


def update_measured_viability(
    state: ViabilityState, *, measurements: Tensor, measurement_mask: Tensor,
    provenance_ids: Tensor, replenishment: Tensor | None = None,
) -> ViabilityState:
    """Apply application measurements exactly; learned estimates cannot alter limits."""

    if measurements.shape != state.values.shape or measurement_mask.shape != state.active.shape:
        raise ValueError("viability measurements must match state channels")
    if measurement_mask.dtype != torch.bool:
        raise ValueError("viability measurement mask must be boolean")
    if provenance_ids.shape != state.provenance_ids.shape or provenance_ids.dtype != torch.int64:
        raise ValueError("viability measurement provenance must match channels")
    if bool((measurement_mask & (provenance_ids < 0)).any()):
        raise ValueError("measured viability channels require provenance")
    replenishment = torch.zeros_like(measurements) if replenishment is None else replenishment
    if replenishment.shape != measurements.shape or bool((replenishment < 0).any()):
        raise ValueError("viability replenishment must be nonnegative channel values")
    previous = state.values
    updated = measurements + replenishment
    values = torch.where(measurement_mask, updated, previous)
    return ViabilityState(
        values, state.target_low, state.target_high, state.hard_low, state.hard_high,
        torch.where(measurement_mask, values - previous, state.trend),
        state.uncertainty, (state.hard_high - values).clamp_min(0),
        state.recovery_priority, state.authority_mask,
        torch.where(measurement_mask, provenance_ids, state.provenance_ids),
        state.active | measurement_mask,
    )
