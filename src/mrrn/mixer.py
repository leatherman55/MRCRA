"""Local nonlinear, structured, triadic, MoE, and anti-aliased mixing paths."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import complex_ops as c
from .packed_projection import PackedCache, packed_linear


class LowRankStructuredLinear(nn.Module):
    """Diagonal, cyclic, and low-rank channel mixing in O(d r) work."""

    def __init__(self, width: int, rank: int, bias: bool = True) -> None:
        super().__init__()
        if width <= 0 or rank <= 0:
            raise ValueError("width and rank must be positive")
        self.width = width
        self.diagonal = nn.Parameter(torch.ones(width))
        self.cyclic = nn.Parameter(torch.zeros(width))
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        self.bias = nn.Parameter(torch.zeros(width)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.width:
            raise ValueError(f"expected final dimension {self.width}")
        output = self.diagonal * x + self.cyclic * x.roll(1, dims=-1) + self.up(self.down(x))
        return output if self.bias is None else output + self.bias


class GatedLocalMixer(nn.Module):
    """Dense or structured SwiGLU-like transient path."""

    def __init__(
        self, width: int, expansion: float = 3.0, *, structured_rank: int | None = None
    ) -> None:
        super().__init__()
        if width <= 0 or expansion <= 0:
            raise ValueError("width and expansion must be positive")
        hidden = max(1, round(width * expansion))
        projection = (
            (lambda source, target: nn.Linear(source, target))
            if structured_rank is None
            else (lambda source, target: _StructuredProjection(source, target, structured_rank))
        )
        self.width = width
        self.a, self.b, self.output = projection(width, hidden), projection(width, hidden), projection(hidden, width)
        self._packed_projection_cache: PackedCache = {}

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.width:
            raise ValueError(f"expected final dimension {self.width}")
        if isinstance(self.a, nn.Linear) and isinstance(self.b, nn.Linear):
            projected = packed_linear(
                x, (self.a, self.b), self._packed_projection_cache, "gate",
            )
            a, b = projected.split((self.a.out_features, self.b.out_features), -1)
        else:
            a, b = self.a(x), self.b(x)
        return self.forward_projected(a, b)

    def forward_projected(self, a: Tensor, b: Tensor) -> Tensor:
        """Finish the mixer from already fused input projections."""

        return self.output(F.silu(a) * b)


def chebyshev_basis(x: Tensor, order: int) -> Tensor:
    """Evaluate ``T_0..T_(order-1)`` without powers or a Vandermonde matrix."""

    if order <= 0:
        raise ValueError("Chebyshev order must be positive")
    terms = [torch.ones_like(x)]
    if order > 1:
        terms.append(x)
    for _ in range(2, order):
        terms.append(2 * x * terms[-1] - terms[-2])
    return torch.stack(terms, -1)


@dataclass(frozen=True, slots=True)
class SpectralActivationDiagnostics:
    amplitude_gate: Tensor
    phase_rotation: Tensor
    triad: Tensor


class ResonantSpectralGLU(nn.Module):
    """Learned mode-wise amplitude/phase gate with sparse frequency-legal triads."""

    def __init__(
        self,
        width: int,
        heads: int,
        modes: int,
        mimo_rank: int,
        *,
        basis_order: int = 6,
        maximum_gain: float = 2.0,
        maximum_phase: float = pi / 8,
        triads_per_mode: int = 2,
        maximum_triad_gain: float = 0.25,
        frequency_max: float = pi,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(width, heads, modes, mimo_rank, basis_order) <= 0:
            raise ValueError("spectral activation dimensions and basis order must be positive")
        if maximum_gain <= 1 or not 0 <= maximum_phase <= pi:
            raise ValueError("maximum gain must exceed one and phase must lie in [0,pi]")
        if triads_per_mode < 0 or maximum_triad_gain < 0 or not 0 < frequency_max <= pi or eps <= 0:
            raise ValueError("spectral triad, frequency, and epsilon controls are invalid")
        self.width, self.heads, self.modes, self.mimo_rank = width, heads, modes, mimo_rank
        self.basis_order, self.maximum_gain, self.maximum_phase = basis_order, maximum_gain, maximum_phase
        self.maximum_triad_gain, self.eps = maximum_triad_gain, eps
        size = 2 * heads * modes * mimo_rank
        self.control = nn.Linear(width, size, bias=False)
        self.carrier = nn.Linear(width, size, bias=False)
        self.context = nn.Linear(width, 2 * heads * modes, bias=False)
        self.output = nn.Linear(size, width, bias=False)
        self._packed_projection_cache: PackedCache = {}
        nn.init.zeros_(self.context.weight)
        self.gain_coefficients = nn.Parameter(torch.zeros(heads, modes, basis_order))
        self.phase_coefficients = nn.Parameter(torch.zeros(heads, modes, basis_order))
        self.register_buffer("frequencies", torch.linspace(0, 0.9 * frequency_max, modes))
        target, left, right, conjugate = self._make_triads(modes, triads_per_mode)
        self.register_buffer("triad_target", target)
        self.register_buffer("triad_left", left)
        self.register_buffer("triad_right", right)
        self.register_buffer("triad_conjugate", conjugate)
        self.raw_triad_weight = nn.Parameter(torch.zeros(heads, target.numel(), mimo_rank))
        self._gain_logit = log(1 / (maximum_gain - 1))

    @staticmethod
    def _make_triads(modes: int, per_mode: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        target, left, right, conjugate = [], [], [], []
        for destination in range(modes):
            candidates = [
                (source, destination - source, False) for source in range(destination + 1)
            ] + [
                (destination + source, source, True) for source in range(modes - destination)
            ]
            for first, second, difference in candidates[:per_mode]:
                target.append(destination)
                left.append(first)
                right.append(second)
                conjugate.append(difference)
        return (
            torch.tensor(target, dtype=torch.long),
            torch.tensor(left, dtype=torch.long),
            torch.tensor(right, dtype=torch.long),
            torch.tensor(conjugate, dtype=torch.bool),
        )

    def _project(self, layer: nn.Linear, x: Tensor) -> Tensor:
        return layer(x).unflatten(-1, (self.heads, self.modes, self.mimo_rank, 2))

    @staticmethod
    def _precision_dtype(dtype: torch.dtype) -> torch.dtype:
        return torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype

    def _magnitude(self, z: Tensor) -> Tensor:
        """Zero-preserving magnitude with a finite derivative at the complex origin."""

        return (c.abs_squared(z) + self.eps).sqrt() - self.eps**0.5

    def triad_frequency_error(self) -> Tensor:
        """Maximum routing error for sum/difference interactions in physical frequency."""

        if not self.triad_target.numel():
            return self.frequencies.new_zeros(())
        expected = torch.where(
            self.triad_conjugate,
            self.frequencies[self.triad_left] - self.frequencies[self.triad_right],
            self.frequencies[self.triad_left] + self.frequencies[self.triad_right],
        )
        return (expected - self.frequencies[self.triad_target]).abs().max()

    def modal_activation(
        self, control: Tensor, carrier: Tensor, context: Tensor | None = None
    ) -> tuple[Tensor, SpectralActivationDiagnostics]:
        expected = (self.heads, self.modes, self.mimo_rank, 2)
        c.validate(control)
        c.validate(carrier)
        if control.shape != carrier.shape or control.ndim < 5 or control.shape[-4:] != expected:
            raise ValueError(f"modal inputs must share (...,{self.heads},{self.modes},{self.mimo_rank},2)")
        leading = control.shape[:-4]
        if context is None:
            context = control.new_zeros(*leading, self.heads, self.modes, 2)
        if context.shape != (*leading, self.heads, self.modes, 2):
            raise ValueError("spectral context must match leading/head/mode axes and end in gain/phase")
        precision = self._precision_dtype(control.dtype)
        control, carrier, context = control.to(precision), carrier.to(precision), context.to(precision)
        amplitude = self._magnitude(control)
        coordinate = 2 * amplitude / (1 + amplitude) - 1
        basis = chebyshev_basis(coordinate, self.basis_order)
        gain_response = torch.einsum(
            "...hnrk,hnk->...hnr", basis, self.gain_coefficients.to(precision)
        )
        phase_response = torch.einsum(
            "...hnrk,hnk->...hnr", basis, self.phase_coefficients.to(precision)
        )
        multiplier = self.maximum_gain * torch.sigmoid(
            self._gain_logit + gain_response + context[..., 0].unsqueeze(-1)
        )
        amplitude_gate = (2 * torch.sigmoid(amplitude) - 1) * multiplier
        phase = self.maximum_phase * torch.tanh(
            phase_response + context[..., 1].unsqueeze(-1)
        )
        gated = c.scale(c.rotate(carrier, phase), amplitude_gate)
        triad = torch.zeros_like(gated)
        if self.triad_target.numel() and self.maximum_triad_gain:
            left = control[..., self.triad_left, :, :]
            right = carrier[..., self.triad_right, :, :]
            right = torch.where(
                self.triad_conjugate.view(*((1,) * len(leading)), 1, -1, 1, 1),
                c.conjugate(right), right,
            )
            interaction = c.multiply(left, right)
            product_amplitude = self._magnitude(left) * self._magnitude(right)
            interaction = c.scale(interaction, (1 + product_amplitude).reciprocal())
            weight = self.maximum_triad_gain * torch.tanh(self.raw_triad_weight.to(precision))
            interaction = c.scale(
                interaction, weight.view(*((1,) * len(leading)), self.heads, -1, self.mimo_rank)
            )
            triad.index_add_(-3, self.triad_target, interaction)
        return gated + triad, SpectralActivationDiagnostics(amplitude_gate, phase, triad)

    def forward_with_diagnostics(self, x: Tensor) -> tuple[Tensor, SpectralActivationDiagnostics]:
        if x.shape[-1] != self.width:
            raise ValueError(f"expected final dimension {self.width}")
        projected = packed_linear(
            x, (self.control, self.carrier, self.context),
            self._packed_projection_cache, "spectral",
        )
        control, carrier, context = projected.split(
            (self.control.out_features, self.carrier.out_features, self.context.out_features), -1
        )
        return self.forward_projected(control, carrier, context, x.dtype)

    def forward_projected(
        self, control: Tensor, carrier: Tensor, context: Tensor, dtype: torch.dtype
    ) -> tuple[Tensor, SpectralActivationDiagnostics]:
        """Finish RSGLU from a single packed real projection."""

        control = control.unflatten(-1, (self.heads, self.modes, self.mimo_rank, 2))
        carrier = carrier.unflatten(-1, (self.heads, self.modes, self.mimo_rank, 2))
        context = context.unflatten(-1, (self.heads, self.modes, 2))
        modal, diagnostics = self.modal_activation(control, carrier, context)
        return self.output(modal.flatten(-4).to(dtype)), diagnostics

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_diagnostics(x)[0]


@dataclass(frozen=True, slots=True)
class HybridMixerDiagnostics:
    spectral_fraction: Tensor
    spectral: SpectralActivationDiagnostics


class HybridSpectralMixer(nn.Module):
    """Conventional SwiGLU and RSGLU blended per position and output channel."""

    def __init__(
        self, width: int, expansion: float, heads: int, modes: int, mimo_rank: int, *,
        structured_rank: int | None = None, spectral_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.width = width
        self.conventional = GatedLocalMixer(width, expansion, structured_rank=structured_rank)
        self.spectral = ResonantSpectralGLU(width, heads, modes, mimo_rank, **(spectral_kwargs or {}))
        self.blend = nn.Linear(width, width)
        self._packed_projection_cache: PackedCache = {}
        nn.init.zeros_(self.blend.weight)
        nn.init.constant_(self.blend.bias, -2.0)

    def forward_with_diagnostics(self, x: Tensor) -> tuple[Tensor, HybridMixerDiagnostics]:
        if x.shape[-1] != self.width:
            raise ValueError(f"expected final dimension {self.width}")
        if isinstance(self.conventional.a, nn.Linear) and isinstance(self.conventional.b, nn.Linear):
            layers = (
                self.conventional.a, self.conventional.b,
                self.spectral.control, self.spectral.carrier, self.spectral.context, self.blend,
            )
            projected = packed_linear(
                x, layers, self._packed_projection_cache, "hybrid",
            )
            a, b, control, carrier, context, blend = projected.split(
                tuple(layer.out_features for layer in layers), -1
            )
            ordinary = self.conventional.forward_projected(a, b)
            spectral, diagnostics = self.spectral.forward_projected(
                control, carrier, context, x.dtype
            )
            fraction = torch.sigmoid(blend)
        else:
            ordinary = self.conventional(x)
            spectral, diagnostics = self.spectral.forward_with_diagnostics(x)
            fraction = torch.sigmoid(self.blend(x))
        return torch.lerp(ordinary, spectral, fraction), HybridMixerDiagnostics(fraction, diagnostics)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_diagnostics(x)[0]


class _StructuredProjection(nn.Module):
    def __init__(self, source: int, target: int, rank: int) -> None:
        super().__init__()
        self.source, self.target = source, target
        self.down = nn.Linear(source, rank, bias=False)
        self.up = nn.Linear(rank, target)
        self.skip = nn.Linear(source, target, bias=False) if source != target else None

    def forward(self, x: Tensor) -> Tensor:
        result = self.up(self.down(x))
        if self.skip is None:
            overlap = min(self.source, self.target)
            result[..., :overlap] += x[..., :overlap]
        else:
            result += self.skip(x)
        return result


class ComplexTriadMixer(nn.Module):
    """Optional low-rank complex product that adds and subtracts phases."""

    def __init__(self, channels: int, rank: int) -> None:
        super().__init__()
        if min(channels, rank) <= 0:
            raise ValueError("channels and rank must be positive")
        self.channels, self.rank = channels, rank
        self.left = nn.Linear(2 * channels, 2 * rank)
        self.right = nn.Linear(2 * channels, 2 * rank)
        self.output = nn.Linear(2 * rank, 2 * channels)

    def forward(self, z: Tensor) -> Tensor:
        c.validate(z)
        if z.shape[-2] != self.channels:
            raise ValueError(f"expected {self.channels} complex channels")
        flat = z.flatten(-2)
        left = self.left(flat).unflatten(-1, (self.rank, 2))
        right = self.right(flat).unflatten(-1, (self.rank, 2))
        return self.output(c.multiply(left, right).flatten(-2)).unflatten(-1, (self.channels, 2))


class SparseMixtureOfExperts(nn.Module):
    """Top-k local experts with deterministic routing and load diagnostics."""

    def __init__(self, width: int, experts: int, top_k: int = 2, expansion: float = 2.0) -> None:
        super().__init__()
        if min(width, experts, top_k) <= 0 or top_k > experts:
            raise ValueError("width/experts/top_k must be positive and top_k <= experts")
        self.width, self.expert_count, self.top_k = width, experts, top_k
        self.router = nn.Linear(width, experts)
        self.experts = nn.ModuleList(GatedLocalMixer(width, expansion) for _ in range(experts))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.shape[-1] != self.width:
            raise ValueError(f"expected final dimension {self.width}")
        logits = self.router(x)
        values, indices = torch.topk(logits, self.top_k, dim=-1, sorted=True)
        weights = torch.softmax(values, dim=-1)
        all_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        selected = torch.gather(
            all_outputs,
            -2,
            indices.unsqueeze(-1).expand(*indices.shape, self.width),
        )
        output = (selected * weights.unsqueeze(-1)).sum(-2)
        routed = F.one_hot(indices, self.expert_count).to(x.dtype)
        load = routed.mean(tuple(range(indices.ndim - 1))).sum(dim=0)
        return output, load


@dataclass(slots=True)
class AntiAliasState:
    pre_history: Tensor
    post_history: Tensor

    def detach(self) -> "AntiAliasState":
        return AntiAliasState(self.pre_history.detach(), self.post_history.detach())


class AntiAliasActivation(nn.Module):
    """Oversample, activate, low-pass, and return to the original rate."""

    def __init__(self, channels: int, factor: int = 2, *, causal: bool = False) -> None:
        super().__init__()
        if channels <= 0 or factor < 2:
            raise ValueError("channels must be positive and factor must be at least two")
        self.channels, self.factor, self.causal = channels, factor, causal
        kernel = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        self.register_buffer("kernel", kernel.view(1, 1, -1).repeat(channels, 1, 1))

    def _filter(self, x: Tensor) -> Tensor:
        padded = F.pad(x.transpose(1, 2), (2, 2), mode="replicate")
        return F.conv1d(padded, self.kernel, groups=self.channels).transpose(1, 2)

    def _causal_filter(self, x: Tensor) -> Tensor:
        padded = F.pad(x.transpose(1, 2), (self.kernel.shape[-1] - 1, 0))
        return F.conv1d(padded, self.kernel, groups=self.channels).transpose(1, 2)

    def initial_state(self, batch: int, *, device=None, dtype=None) -> AntiAliasState:
        if batch <= 0 or not self.causal:
            raise ValueError("a positive batch and causal anti-alias module are required")
        history = torch.zeros(batch, self.kernel.shape[-1] - 1, self.channels, device=device, dtype=dtype)
        return AntiAliasState(history, history.clone())

    def _stateful_filter(self, x: Tensor, history: Tensor) -> tuple[Tensor, Tensor]:
        sequence = torch.cat((history, x), 1)
        output = F.conv1d(sequence.transpose(1, 2), self.kernel, groups=self.channels).transpose(1, 2)
        return output, sequence[:, -(self.kernel.shape[-1] - 1) :]

    def step(self, x: Tensor, state: AntiAliasState) -> tuple[Tensor, AntiAliasState]:
        if not self.causal or x.ndim != 2 or x.shape[-1] != self.channels:
            raise ValueError("causal anti-alias step expects (batch,channels)")
        if state.pre_history.shape != (x.shape[0], self.kernel.shape[-1] - 1, self.channels):
            raise ValueError("anti-alias stream state shape mismatch")
        up = x.new_zeros(x.shape[0], self.factor, self.channels)
        up[:, 0] = self.factor * x
        pre, pre_history = self._stateful_filter(up, state.pre_history)
        post, post_history = self._stateful_filter(F.silu(pre), state.post_history)
        return post[:, -1], AntiAliasState(pre_history, post_history)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(f"expected (batch, length, {self.channels})")
        if x.shape[1] == 0:
            return x.clone()
        if self.causal:
            up = x.new_zeros(x.shape[0], x.shape[1] * self.factor, self.channels)
            up[:, :: self.factor] = self.factor * x
            return self._causal_filter(F.silu(self._causal_filter(up)))[:, self.factor - 1 :: self.factor]
        up = F.interpolate(
            x.transpose(1, 2), size=x.shape[1] * self.factor, mode="linear", align_corners=False
        ).transpose(1, 2)
        filtered = self._filter(F.silu(self._filter(up)))
        return filtered[:, self.factor // 2 :: self.factor][:, : x.shape[1]]
