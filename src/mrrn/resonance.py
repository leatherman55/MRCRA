"""Stable complex selective MIMO state-space resonance."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi

import torch
from torch import Tensor, nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from . import complex_ops as c
from .runtime_validation import runtime_validation_enabled


@dataclass(frozen=True, slots=True)
class ResonatorState:
    value: Tensor
    previous_drive: Tensor
    steps: int = 0

    def detach(self) -> "ResonatorState":
        return ResonatorState(self.value.detach(), self.previous_drive.detach(), self.steps)


@dataclass(frozen=True, slots=True)
class ResonatorParameters:
    transition: Tensor
    affine_drive: Tensor
    drive: Tensor
    readout: Tensor
    alpha: Tensor
    omega: Tensor
    delta: Tensor
    final_drive: Tensor


def _inverse_softplus(value: Tensor) -> Tensor:
    return value + torch.log(-torch.expm1(-value))


def half_life_steps(alpha: Tensor, delta: Tensor | float = 1.0) -> Tensor:
    delta = torch.as_tensor(delta, dtype=alpha.dtype, device=alpha.device)
    if not bool((alpha > 0).all()) or not bool((delta > 0).all()):
        raise ValueError("decay and step must be positive")
    return log(2.0) / (alpha * delta)


def effective_horizon_steps(alpha: Tensor, tolerance: float, delta: Tensor | float = 1.0) -> Tensor:
    if not 0 < tolerance < 1:
        raise ValueError("memory tolerance must lie in (0,1)")
    delta = torch.as_tensor(delta, dtype=alpha.dtype, device=alpha.device)
    if not bool((alpha > 0).all()) or not bool((delta > 0).all()):
        raise ValueError("decay and step must be positive")
    return torch.log(alpha.new_tensor(1 / tolerance)) / (alpha * delta)


def uniform_state_bound(transition_bound: float, drive_bound: float, initial_magnitude: float = 0.0) -> float:
    if not 0 <= transition_bound < 1 or min(drive_bound, initial_magnitude) < 0:
        raise ValueError("BIBS bounds require rho in [0,1) and nonnegative magnitudes")
    return initial_magnitude + drive_bound / (1 - transition_bound)


def _affine_prefix_tree(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Pure O(T)-work/O(log T)-depth prefix used by forward and adjoint."""

    length = a.shape[1]
    if length <= 1:
        return a, b
    pairs = length // 2
    even_a, odd_a = a[:, : 2 * pairs : 2], a[:, 1 : 2 * pairs : 2]
    even_b, odd_b = b[:, : 2 * pairs : 2], b[:, 1 : 2 * pairs : 2]
    pair_a = c.multiply(odd_a, even_a)
    pair_b = odd_b + c.multiply(odd_a, even_b)
    scanned_a, scanned_b = _affine_prefix_tree(pair_a, pair_b)
    if pairs > 1:
        remaining_a = c.multiply(even_a[:, 1:], scanned_a[:, :-1])
        remaining_b = (
            even_b[:, 1:] + c.multiply(even_a[:, 1:], scanned_b[:, :-1])
        )
        prefix_even_a = torch.cat((even_a[:, :1], remaining_a), 1)
        prefix_even_b = torch.cat((even_b[:, :1], remaining_b), 1)
    else:
        prefix_even_a, prefix_even_b = even_a, even_b
    prefix_a = torch.stack((prefix_even_a, scanned_a), 2).flatten(1, 2)
    prefix_b = torch.stack((prefix_even_b, scanned_b), 2).flatten(1, 2)
    if length % 2:
        tail_a = c.multiply(a[:, -1:], scanned_a[:, -1:])
        tail_b = b[:, -1:] + c.multiply(a[:, -1:], scanned_b[:, -1:])
        prefix_a = torch.cat((prefix_a, tail_a), 1)
        prefix_b = torch.cat((prefix_b, tail_b), 1)
    return prefix_a, prefix_b


def _associative_affine_scan_composite(
    transition: Tensor, drive: Tensor, initial: Tensor
) -> Tensor:
    """Differentiable reference composite retained for compilation and audit."""

    prefix_a, prefix_b = _affine_prefix_tree(transition, drive)
    return c.multiply(prefix_a, initial.unsqueeze(1)) + prefix_b


class _ComplexAffineScanAdjoint(torch.autograd.Function):
    """Memory-bounded exact first-order adjoint for a diagonal complex scan.

    The recursive prefix implementation is ideal for parallel forward
    execution but, when exposed directly to autograd, it retains every
    intermediate pair, slice, stack, and concatenation.  This custom boundary
    retains only the transition, initial state, and output states.  Backward
    computes all state cotangents with one reverse associative scan and then
    forms the transition/drive gradients pointwise.
    """

    @staticmethod
    def forward(ctx, transition: Tensor, drive: Tensor, initial: Tensor) -> Tensor:
        states = _associative_affine_scan_composite(
            transition, drive, initial
        )
        ctx.save_for_backward(transition, initial, states)
        return states

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_states: Tensor):
        transition, initial, states = ctx.saved_tensors
        length = transition.shape[1]
        if length == 0:
            return (
                torch.zeros_like(transition),
                torch.zeros_like(transition),
                torch.zeros_like(initial),
            )

        # For y_t = a_t y_{t-1} + b_t, the cotangent recurrence is
        # q_t = g_t + conj(a_{t+1}) q_{t+1}.  Reverse time converts it into the
        # same inclusive affine-prefix primitive used by forward.
        reverse_transition = torch.empty_like(transition)
        reverse_transition[:, 0] = 0
        reverse_transition[:, 0, ..., 0] = 1
        if length > 1:
            reverse_transition[:, 1:] = c.conjugate(
                transition[:, 1:].flip(1)
            )
        zero = torch.zeros_like(initial)
        reverse_cotangent = _associative_affine_scan_composite(
            reverse_transition,
            grad_states.flip(1),
            zero,
        )
        cotangent = reverse_cotangent.flip(1)
        previous = torch.cat((initial.unsqueeze(1), states[:, :-1]), 1)
        grad_transition = c.multiply(cotangent, c.conjugate(previous))
        grad_drive = cotangent
        grad_initial = c.multiply(
            c.conjugate(transition[:, 0]), cotangent[:, 0]
        )
        return grad_transition, grad_drive, grad_initial


def _masked_scan_composite(
    transition: Tensor,
    drive: Tensor,
    initial: Tensor,
    mask: Tensor,
) -> Tensor:
    active = mask.view(*mask.shape, *([1] * (transition.ndim - 2)))
    identity = torch.zeros_like(transition)
    identity[..., 0] = 1
    return _associative_affine_scan_composite(
        torch.where(active, transition, identity),
        torch.where(active, drive, torch.zeros_like(drive)),
        initial,
    )


class _MaskedComplexAffineScanAdjoint(torch.autograd.Function):
    """Mask-aware scan/adjoint that keeps identity padding out of autograd."""

    @staticmethod
    def forward(
        ctx,
        transition: Tensor,
        drive: Tensor,
        initial: Tensor,
        mask: Tensor,
    ) -> Tensor:
        states = _masked_scan_composite(transition, drive, initial, mask)
        ctx.save_for_backward(transition, initial, states, mask)
        return states

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_states: Tensor):
        transition, initial, states, mask = ctx.saved_tensors
        length = transition.shape[1]
        if length == 0:
            return (
                torch.zeros_like(transition),
                torch.zeros_like(transition),
                torch.zeros_like(initial),
                None,
            )
        active = mask.view(*mask.shape, *([1] * (transition.ndim - 2)))
        adjoint_transition = c.conjugate(transition)
        identity = torch.zeros_like(adjoint_transition)
        identity[..., 0] = 1
        adjoint_transition = torch.where(
            active, adjoint_transition, identity
        )
        reverse_transition = torch.empty_like(transition)
        reverse_transition[:, 0] = 0
        reverse_transition[:, 0, ..., 0] = 1
        if length > 1:
            reverse_transition[:, 1:] = adjoint_transition[:, 1:].flip(1)
        cotangent = _associative_affine_scan_composite(
            reverse_transition,
            grad_states.flip(1),
            torch.zeros_like(initial),
        ).flip(1)
        previous = torch.cat((initial.unsqueeze(1), states[:, :-1]), 1)
        grad_transition = (
            c.multiply(cotangent, c.conjugate(previous)) * active
        )
        grad_drive = cotangent * active
        grad_initial = c.multiply(
            adjoint_transition[:, 0], cotangent[:, 0]
        )
        return grad_transition, grad_drive, grad_initial, None


def associative_affine_scan(
    transition: Tensor,
    drive: Tensor,
    initial: Tensor,
    *,
    implementation: str = "auto",
) -> Tensor:
    """Work-efficient inclusive prefix of diagonal complex affine transforms.

    ``auto`` uses a custom first-order adjoint during eager training and the
    pure composite under inference or graph compilation.  ``composite`` is the
    transparent reference path used for numerical and compiler validation;
    ``custom_adjoint`` explicitly requests the bounded saved-tensor path.
    """

    if transition.shape != drive.shape or transition.ndim < 3:
        raise ValueError(
            "transition and drive must share a batched time/complex shape"
        )
    c.validate(transition)
    if initial.shape != transition.shape[:1] + transition.shape[2:]:
        raise ValueError("initial state shape does not match scan state shape")
    if implementation not in {"auto", "composite", "custom_adjoint"}:
        raise ValueError("unknown affine scan implementation")
    custom = implementation == "custom_adjoint" or (
        implementation == "auto"
        and torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and (
            transition.requires_grad
            or drive.requires_grad
            or initial.requires_grad
        )
    )
    if custom:
        return _ComplexAffineScanAdjoint.apply(transition, drive, initial)
    return _associative_affine_scan_composite(transition, drive, initial)


def masked_associative_affine_scan(
    transition: Tensor,
    drive: Tensor,
    initial: Tensor,
    mask: Tensor,
    *,
    implementation: str = "auto",
) -> Tensor:
    """Apply a complex affine scan while invalid positions are exact identity.

    The custom path does not expose the full identity-transition and
    zero-drive tensors to autograd, which is particularly important for
    document-major static batches with right padding.
    """

    if (
        transition.shape != drive.shape
        or transition.ndim < 3
        or mask.shape != transition.shape[:2]
        or mask.dtype != torch.bool
    ):
        raise ValueError("masked scan tensors have incompatible shapes or dtypes")
    c.validate(transition)
    if initial.shape != transition.shape[:1] + transition.shape[2:]:
        raise ValueError("masked scan initial state has incompatible shape")
    if implementation not in {"auto", "composite", "custom_adjoint"}:
        raise ValueError("unknown masked affine scan implementation")
    custom = implementation == "custom_adjoint" or (
        implementation == "auto"
        and torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and (
            transition.requires_grad
            or drive.requires_grad
            or initial.requires_grad
        )
    )
    if custom:
        return _MaskedComplexAffineScanAdjoint.apply(
            transition, drive, initial, mask
        )
    return _masked_scan_composite(transition, drive, initial, mask)


class ComplexResonator(nn.Module):
    """Content-selective damped oscillators with low-rank MIMO drive/readout."""

    def __init__(
        self,
        width: int,
        heads: int,
        modes: int,
        mimo_rank: int,
        *,
        alpha_min: float = 1e-4,
        delta_min: float = 1e-4,
        omega_max: float = pi,
        maximum_half_life: float = 1024.0,
        decay_normalized_drive: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(width, heads, modes, mimo_rank) <= 0:
            raise ValueError("width, heads, modes, and mimo_rank must be positive")
        if min(alpha_min, delta_min, omega_max, maximum_half_life, eps) <= 0:
            raise ValueError("stability constants must be positive")
        if omega_max > pi:
            raise ValueError("omega_max must not exceed pi")
        self.width, self.heads, self.modes, self.mimo_rank = width, heads, modes, mimo_rank
        self.alpha_min, self.delta_min, self.omega_max, self.eps = (
            alpha_min,
            delta_min,
            omega_max,
            eps,
        )
        self.decay_normalized_drive = decay_normalized_drive
        lanes, mode_lanes = heads * mimo_rank, heads * modes * mimo_rank
        self.input_projection = nn.Linear(width, lanes)
        self.delta_projection = nn.Linear(width, heads)
        self.alpha_projection = nn.Linear(width, heads * modes)
        self.omega_projection = nn.Linear(width, heads * modes)
        self.input_direction = nn.Linear(width, 2 * mode_lanes)
        self.readout_direction = nn.Linear(width, 2 * mode_lanes)
        self.output_projection = nn.Linear(lanes, width)
        self.output_gate = nn.Linear(width, width)

        half_lives = torch.logspace(log(2.0), log(maximum_half_life), modes, base=torch.e)
        alpha = log(2.0) / half_lives
        raw_alpha = _inverse_softplus((alpha - alpha_min).clamp_min(alpha_min))
        self.raw_alpha = nn.Parameter(raw_alpha.repeat(heads, 1))
        if modes == 1:
            frequencies = torch.zeros(1)
        else:
            positive = torch.logspace(log(2 * pi / maximum_half_life), log(0.9 * omega_max), modes - 1, base=torch.e)
            frequencies = torch.cat((torch.zeros(1), positive))
        normalized = (frequencies / omega_max).clamp(-0.999, 0.999)
        self.raw_omega = nn.Parameter(torch.atanh(normalized).repeat(heads, 1))
        self.reset_selective_parameters()

    def reset_selective_parameters(self) -> None:
        for layer in (self.delta_projection, self.alpha_projection, self.omega_projection):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.output_gate.bias, -1.0)

    @property
    def state_shape(self) -> tuple[int, int, int, int]:
        return self.heads, self.modes, self.mimo_rank, 2

    @staticmethod
    def _precision_dtype(dtype: torch.dtype | None) -> torch.dtype | None:
        return torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype

    def initial_state(self, batch: int, *, device=None, dtype=None) -> ResonatorState:
        if batch <= 0:
            raise ValueError("batch must be positive")
        shape = (batch, *self.state_shape)
        value = torch.zeros(shape, device=device, dtype=self._precision_dtype(dtype))
        return ResonatorState(value, torch.zeros_like(value))

    def _validate_input(self, u: Tensor, mask: Tensor | None) -> Tensor:
        if u.ndim != 3 or u.shape[-1] != self.width:
            raise ValueError(f"expected input shape (batch, time, {self.width})")
        if mask is None:
            return torch.ones(u.shape[:2], dtype=torch.bool, device=u.device)
        if mask.shape != u.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape (batch, time)")
        return mask

    def parameterize(
        self, u: Tensor, previous_drive: Tensor | None = None, mask: Tensor | None = None,
        *, sample_interval: float | Tensor = 1.0,
    ) -> ResonatorParameters:
        batch, length, _ = u.shape
        h, n, r = self.heads, self.modes, self.mimo_rank
        precision = self._precision_dtype(u.dtype)
        interval = torch.as_tensor(sample_interval, dtype=precision, device=u.device)
        if interval.ndim == 0:
            interval = interval.expand(batch, length)
        if interval.shape != (batch, length):
            raise ValueError("sample_interval must be positive and scalar or shaped (batch,time)")
        if (
            runtime_validation_enabled()
            and not torch.compiler.is_compiling()
            and not bool((interval > 0).all())
        ):
            raise ValueError("sample_interval must be positive and scalar or shaped (batch,time)")
        projections = (
            self.input_projection, self.delta_projection, self.alpha_projection,
            self.omega_projection, self.input_direction, self.readout_direction,
        )
        packed = F.linear(
            u,
            torch.cat(tuple(layer.weight for layer in projections)),
            torch.cat(tuple(layer.bias for layer in projections)),
        ).to(precision)
        amplitudes, delta_content, alpha_content, omega_content, input_direction, readout = packed.split(
            tuple(layer.out_features for layer in projections), -1
        )
        delta = (
            self.delta_min + F.softplus(delta_content).view(batch, length, h, 1, 1)
        ) * interval[..., None, None, None]
        alpha_content = alpha_content.view(batch, length, h, n)
        alpha = self.alpha_min + F.softplus(self.raw_alpha.to(precision) + alpha_content)
        omega_content = omega_content.view(batch, length, h, n)
        omega = self.omega_max * torch.tanh(self.raw_omega.to(precision) + omega_content)
        lambda_pair = c.pair(-alpha, omega).unsqueeze(-2)
        q = c.scale(lambda_pair, delta)
        transition = c.exponential(q).expand(batch, length, h, n, r, 2)
        phi1, phi2 = c.phi_functions(q)

        amplitudes = amplitudes.view(batch, length, h, r)
        input_direction = input_direction.view(batch, length, h, n, r, 2)
        readout = readout.view(batch, length, h, n, r, 2)
        input_direction = c.rms_normalize(input_direction, dim=-3, eps=self.eps)
        readout = c.rms_normalize(readout, dim=-3, eps=self.eps)
        drive = c.scale(input_direction, amplitudes.unsqueeze(-2))
        if self.decay_normalized_drive:
            # A continuous resonator has impulse kernel exp((-alpha+i*omega)t),
            # whose absolute integral is 1/alpha.  Scaling its drive by alpha
            # therefore gives every mode an induced L-infinity gain no larger
            # than one, independent of half-life, without changing the linear
            # recurrence or its exact associative scan.  Long-lived modes keep
            # their memory but cannot obtain unbounded gain merely by reducing
            # damping.
            drive = c.scale(drive, alpha.unsqueeze(-1))
        expected = (batch, h, n, r, 2)
        if previous_drive is None:
            carried = torch.zeros(expected, dtype=precision, device=u.device)
        elif previous_drive.shape != expected:
            raise ValueError(f"previous_drive must have shape {expected}")
        else:
            carried = previous_drive
        valid = torch.ones(batch, length, dtype=torch.bool, device=u.device) if mask is None else mask
        previous, carried = self._previous_and_final_drive(drive, carried, valid)
        affine = c.scale(
            c.multiply(phi1 - phi2, previous) + c.multiply(phi2, drive), delta
        )
        return ResonatorParameters(transition, affine, drive, readout, alpha, omega, delta, carried)

    @staticmethod
    def _previous_and_final_drive(
        drive: Tensor, initial: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch, length = drive.shape[:2]
        if length == 0:
            return drive[:, :0], initial
        active = mask.view(batch, length, *([1] * (drive.ndim - 2)))
        if length == 1:
            return initial.unsqueeze(1), torch.where(active[:, 0], drive[:, 0], initial)
        positions = torch.arange(1, length + 1, device=drive.device).expand(batch, -1)
        inclusive = torch.where(mask, positions, 0).cummax(1).values
        previous_indices = torch.cat((torch.zeros_like(inclusive[:, :1]), inclusive[:, :-1]), 1)
        sources = torch.cat((initial.unsqueeze(1), drive), 1)
        gather = previous_indices.view(batch, length, *([1] * (drive.ndim - 2))).expand_as(drive)
        previous = sources.gather(1, gather)
        final_index = inclusive[:, -1].view(batch, *([1] * (drive.ndim - 2))).unsqueeze(1)
        final = sources.gather(
            1, final_index.expand(batch, 1, *drive.shape[2:])
        )[:, 0]
        return previous, final

    def _readout(self, u: Tensor, states: Tensor, readout: Tensor, mask: Tensor) -> Tensor:
        projected = c.real(c.multiply(c.conjugate(readout), states)).sum(dim=3)
        output = self.output_projection(projected.flatten(-2).to(u.dtype))
        output = torch.sigmoid(self.output_gate(u)) * output
        return output * mask.unsqueeze(-1)

    def sequential(
        self, u: Tensor, state: ResonatorState | None = None, mask: Tensor | None = None,
        *, sample_interval: float | Tensor = 1.0,
    ) -> tuple[Tensor, ResonatorState, ResonatorParameters]:
        mask = self._validate_input(u, mask)
        state = self.initial_state(u.shape[0], device=u.device, dtype=u.dtype) if state is None else state
        expected = (u.shape[0], *self.state_shape)
        if state.value.shape != expected or state.previous_drive.shape != expected:
            raise ValueError(f"state tensors must have shape {expected}")

        parameters = self.parameterize(u, state.previous_drive, mask, sample_interval=sample_interval)
        current = state.value
        states = []
        for time in range(u.shape[1]):
            active = mask[:, time].view(u.shape[0], 1, 1, 1, 1)
            proposed = c.multiply(parameters.transition[:, time], current) + parameters.affine_drive[:, time]
            current = torch.where(active, proposed, current)
            states.append(current)
        stacked = torch.stack(states, dim=1) if states else u.new_empty((u.shape[0], 0, *self.state_shape))
        output = self._readout(u, stacked, parameters.readout, mask)
        return output, ResonatorState(current, parameters.final_drive, state.steps + u.shape[1]), parameters

    def parallel(
        self, u: Tensor, state: ResonatorState | None = None, mask: Tensor | None = None,
        *, sample_interval: float | Tensor = 1.0,
    ) -> tuple[Tensor, ResonatorState, ResonatorParameters]:
        mask = self._validate_input(u, mask)
        state = self.initial_state(u.shape[0], device=u.device, dtype=u.dtype) if state is None else state
        expected = (u.shape[0], *self.state_shape)
        if state.value.shape != expected or state.previous_drive.shape != expected:
            raise ValueError(f"state tensors must have shape {expected}")
        parameters = self.parameterize(u, state.previous_drive, mask, sample_interval=sample_interval)
        states = masked_associative_affine_scan(
            parameters.transition,
            parameters.affine_drive,
            state.value,
            mask,
        )
        output = self._readout(u, states, parameters.readout, mask)
        current = states[:, -1] if u.shape[1] else state.value
        return output, ResonatorState(current, parameters.final_drive, state.steps + u.shape[1]), parameters

    def euler(
        self, u: Tensor, state: ResonatorState | None = None, mask: Tensor | None = None,
        *, sample_interval: float | Tensor = 1.0,
    ) -> tuple[Tensor, ResonatorState, ResonatorParameters]:
        """Explicit-Euler ablation; intentionally not used by the stable baseline."""

        mask = self._validate_input(u, mask)
        state = self.initial_state(u.shape[0], device=u.device, dtype=u.dtype) if state is None else state
        expected = (u.shape[0], *self.state_shape)
        if state.value.shape != expected or state.previous_drive.shape != expected:
            raise ValueError(f"state tensors must have shape {expected}")
        exact = self.parameterize(u, state.previous_drive, mask, sample_interval=sample_interval)
        lambda_delta = c.pair(-exact.alpha, exact.omega).unsqueeze(-2)
        lambda_delta = c.scale(lambda_delta, exact.delta)
        transition = lambda_delta.clone()
        transition[..., 0] += 1
        affine = c.scale(exact.drive, exact.delta)
        parameters = ResonatorParameters(
            transition, affine, exact.drive, exact.readout,
            exact.alpha, exact.omega, exact.delta, exact.final_drive,
        )
        current, states = state.value, []
        for time in range(u.shape[1]):
            active = mask[:, time].view(u.shape[0], 1, 1, 1, 1)
            proposed = c.multiply(transition[:, time], current) + affine[:, time]
            current = torch.where(active, proposed, current)
            states.append(current)
        stacked = torch.stack(states, 1) if states else u.new_empty((u.shape[0], 0, *self.state_shape))
        output = self._readout(u, stacked, exact.readout, mask)
        return output, ResonatorState(current, exact.final_drive, state.steps + u.shape[1]), parameters

    def forward(
        self, u: Tensor, state: ResonatorState | None = None, mask: Tensor | None = None,
        *, sample_interval: float | Tensor = 1.0,
    ) -> tuple[Tensor, ResonatorState, ResonatorParameters]:
        return self.parallel(u, state, mask, sample_interval=sample_interval)
