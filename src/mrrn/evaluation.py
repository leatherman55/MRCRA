"""Reproducible capability, ablation, extrapolation, and end-to-end efficiency measurements."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
from math import sqrt
import json
import platform
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
from typing import Callable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import MRRNConfig
from .mixer import GatedLocalMixer, HybridSpectralMixer
from .model import MRRN


@dataclass(frozen=True, slots=True)
class AttentionDecodeState:
    keys: list[Tensor]
    values: list[Tensor]


class CausalTransformerBaseline(nn.Module):
    """Exact causal self-attention baseline with a real KV-cached decode path."""

    def __init__(self, input_dim: int, width: int, output_dim: int, heads: int, layers: int) -> None:
        super().__init__()
        if width % heads or min(input_dim, width, output_dim, heads, layers) <= 0:
            raise ValueError("Transformer dimensions must be positive and width divisible by heads")
        self.width, self.heads, self.layers = width, heads, layers
        self.encoder, self.head = nn.Linear(input_dim, width), nn.Linear(width, output_dim)
        self.norm1 = nn.ModuleList(nn.RMSNorm(width) for _ in range(layers))
        self.qkv = nn.ModuleList(nn.Linear(width, 3 * width) for _ in range(layers))
        self.attention_out = nn.ModuleList(nn.Linear(width, width) for _ in range(layers))
        self.norm2 = nn.ModuleList(nn.RMSNorm(width) for _ in range(layers))
        self.mixers = nn.ModuleList(GatedLocalMixer(width, 3) for _ in range(layers))

    def _split(self, value: Tensor) -> Tensor:
        return value.unflatten(-1, (self.heads, self.width // self.heads)).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.encoder(x)
        length = x.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        for norm1, qkv, output, norm2, mixer in zip(
            self.norm1, self.qkv, self.attention_out, self.norm2, self.mixers, strict=True
        ):
            q, k, v = (self._split(item) for item in qkv(norm1(hidden)).chunk(3, -1))
            scores = torch.matmul(q, k.transpose(-1, -2)) / sqrt(q.shape[-1])
            weights = torch.softmax(scores.masked_fill(~causal, -torch.inf), -1)
            attended = torch.matmul(weights, v).transpose(1, 2).flatten(-2)
            hidden = hidden + output(attended)
            hidden = hidden + mixer(norm2(hidden))
        return self.head(hidden)

    def initial_decode_state(self, batch: int, *, device=None, dtype=None) -> AttentionDecodeState:
        empty = [torch.empty(batch, self.heads, 0, self.width // self.heads, device=device, dtype=dtype) for _ in range(self.layers)]
        return AttentionDecodeState(empty, [item.clone() for item in empty])

    def step(self, x: Tensor, state: AttentionDecodeState) -> tuple[Tensor, AttentionDecodeState]:
        hidden = self.encoder(x).unsqueeze(1)
        for index, (norm1, qkv, output, norm2, mixer) in enumerate(zip(
            self.norm1, self.qkv, self.attention_out, self.norm2, self.mixers, strict=True
        )):
            q, k, v = (self._split(item) for item in qkv(norm1(hidden)).chunk(3, -1))
            state.keys[index] = torch.cat((state.keys[index], k), 2)
            state.values[index] = torch.cat((state.values[index], v), 2)
            weights = torch.softmax(torch.matmul(q, state.keys[index].transpose(-1, -2)) / sqrt(q.shape[-1]), -1)
            hidden = hidden + output(torch.matmul(weights, state.values[index]).transpose(1, 2).flatten(-2))
            hidden = hidden + mixer(norm2(hidden))
        return self.head(hidden[:, 0]), state


@dataclass(frozen=True, slots=True)
class RealSSMState:
    values: list[Tensor]


class RealSelectiveSSMBaseline(nn.Module):
    """Stable input-selective real diagonal SSM baseline with recurrent decode."""

    def __init__(self, input_dim: int, width: int, output_dim: int, layers: int) -> None:
        super().__init__()
        if min(input_dim, width, output_dim, layers) <= 0:
            raise ValueError("SSM dimensions must be positive")
        self.width, self.layers = width, layers
        self.encoder, self.head = nn.Linear(input_dim, width), nn.Linear(width, output_dim)
        self.norms = nn.ModuleList(nn.RMSNorm(width) for _ in range(layers))
        self.decays = nn.ModuleList(nn.Linear(width, width) for _ in range(layers))
        self.drives = nn.ModuleList(nn.Linear(width, width) for _ in range(layers))
        self.readouts = nn.ModuleList(nn.Linear(width, width) for _ in range(layers))
        self.gates = nn.ModuleList(nn.Linear(width, width) for _ in range(layers))

    def initial_decode_state(self, batch: int, *, device=None, dtype=None) -> RealSSMState:
        return RealSSMState([torch.zeros(batch, self.width, device=device, dtype=dtype) for _ in range(self.layers)])

    def step(self, x: Tensor, state: RealSSMState) -> tuple[Tensor, RealSSMState]:
        hidden = self.encoder(x)
        for index, (norm, decay, drive, readout, gate) in enumerate(zip(
            self.norms, self.decays, self.drives, self.readouts, self.gates, strict=True
        )):
            normalized = norm(hidden)
            transition = torch.exp(-F.softplus(decay(normalized)))
            state.values[index] = transition * state.values[index] + (1 - transition) * drive(normalized)
            hidden = hidden + torch.sigmoid(gate(normalized)) * readout(state.values[index])
        return self.head(hidden), state

    def forward(self, x: Tensor) -> Tensor:
        state = self.initial_decode_state(x.shape[0], device=x.device, dtype=x.dtype)
        output = []
        for position in range(x.shape[1]):
            value, state = self.step(x[:, position], state)
            output.append(value.unsqueeze(1))
        return torch.cat(output, 1) if output else x.new_empty(x.shape[0], 0, self.head.out_features)


@dataclass(frozen=True, slots=True)
class ConvolutionState:
    histories: list[Tensor]


class LongConvolutionBaseline(nn.Module):
    """Causal depthwise long-convolution baseline with a bounded rolling decode cache."""

    def __init__(self, input_dim: int, width: int, output_dim: int, layers: int, kernel_size: int = 31) -> None:
        super().__init__()
        if min(input_dim, width, output_dim, layers, kernel_size) <= 0:
            raise ValueError("convolution dimensions must be positive")
        self.width, self.layers, self.kernel_size = width, layers, kernel_size
        self.encoder, self.head = nn.Linear(input_dim, width), nn.Linear(width, output_dim)
        self.convolutions = nn.ModuleList(nn.Conv1d(width, width, kernel_size, groups=width) for _ in range(layers))
        self.mixers = nn.ModuleList(GatedLocalMixer(width, 2) for _ in range(layers))

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.encoder(x)
        for convolution, mixer in zip(self.convolutions, self.mixers, strict=True):
            filtered = convolution(F.pad(hidden.transpose(1, 2), (self.kernel_size - 1, 0))).transpose(1, 2)
            hidden = hidden + mixer(filtered)
        return self.head(hidden)

    def initial_decode_state(self, batch: int, *, device=None, dtype=None) -> ConvolutionState:
        return ConvolutionState([
            torch.zeros(batch, self.kernel_size - 1, self.width, device=device, dtype=dtype)
            for _ in range(self.layers)
        ])

    def step(self, x: Tensor, state: ConvolutionState) -> tuple[Tensor, ConvolutionState]:
        hidden = self.encoder(x)
        for index, (convolution, mixer) in enumerate(zip(self.convolutions, self.mixers, strict=True)):
            sequence = torch.cat((state.histories[index], hidden.unsqueeze(1)), 1)
            filtered = convolution(sequence.transpose(1, 2)).squeeze(-1)
            state.histories[index] = sequence[:, 1:]
            hidden = hidden + mixer(filtered)
        return self.head(hidden), state


@dataclass(frozen=True, slots=True)
class Timing:
    mean_seconds: float
    p95_seconds: float
    confidence95_seconds: float
    samples_per_second: float


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    phase: str
    parameters: int
    parameter_bytes: int
    timing: Timing
    peak_device_bytes: int | None
    dtype: str
    batch: int
    length: int
    input_width: int


def parameter_statistics(model: nn.Module) -> tuple[int, int]:
    parameters = sum(value.numel() for value in model.parameters() if value.requires_grad)
    byte_count = sum(value.numel() * value.element_size() for value in model.parameters())
    return parameters, byte_count


def _timing(values: list[float], batch: int, length: int) -> Timing:
    ordered = sorted(values)
    average = mean(values)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    confidence = 0.0 if len(values) < 2 else 1.96 * stdev(values) / sqrt(len(values))
    return Timing(average, p95, confidence, batch * length / average)


def _synchronize(device: torch.device) -> None:
    """Make lazy/asynchronous accelerator work visible to wall-clock timing."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark(
    name: str,
    model: nn.Module,
    x: Tensor,
    *,
    phase: str,
    repeats: int = 5,
    warmup: int = 1,
) -> BenchmarkResult:
    if phase not in {"prefill", "training", "decode"} or repeats <= 0 or warmup < 0:
        raise ValueError("benchmark phase/repetition controls are invalid")
    model.train(phase == "training")
    if x.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(x.device)

    def execute() -> None:
        if phase == "decode":
            state = model.initial_stream_state(x.shape[0], device=x.device, dtype=x.dtype) if isinstance(model, MRRN) else model.initial_decode_state(x.shape[0], device=x.device, dtype=x.dtype)
            for position in range(x.shape[1]):
                if isinstance(model, MRRN):
                    state = model.step(x[:, position], state).state
                else:
                    _, state = model.step(x[:, position], state)
        elif phase == "training":
            model.zero_grad(set_to_none=True)
            output = model(x)
            prediction = output.prediction if isinstance(model, MRRN) else output
            prediction.square().mean().backward()
        else:
            with torch.no_grad():
                model(x)
        _synchronize(x.device)

    for _ in range(warmup):
        execute()
    values = []
    for _ in range(repeats):
        _synchronize(x.device)
        start = perf_counter()
        execute()
        values.append(perf_counter() - start)
    parameters, parameter_bytes = parameter_statistics(model)
    peak = torch.cuda.max_memory_allocated(x.device) if x.device.type == "cuda" else None
    return BenchmarkResult(
        name, phase, parameters, parameter_bytes, _timing(values, x.shape[0], x.shape[1]), peak,
        str(x.dtype), x.shape[0], x.shape[1], x.shape[2],
    )


def benchmark_suite(
    models: Mapping[str, nn.Module], x: Tensor, *, repeats: int = 5, warmup: int = 1
) -> tuple[BenchmarkResult, ...]:
    return tuple(
        benchmark(name, model, x, phase=phase, repeats=repeats, warmup=warmup)
        for name, model in models.items() for phase in ("training", "prefill", "decode")
    )


def matched_width(factory: Callable[[int], nn.Module], target_parameters: int, *, minimum: int = 4, maximum: int = 4096) -> int:
    if target_parameters <= 0 or minimum <= 0 or maximum < minimum:
        raise ValueError("parameter target and width range are invalid")
    best, best_error = minimum, float("inf")
    low, high = minimum, maximum
    while low <= high:
        width = (low + high) // 2
        count, _ = parameter_statistics(factory(width))
        error = abs(count - target_parameters)
        if error < best_error:
            best, best_error = width, error
        if count < target_parameters:
            low = width + 1
        else:
            high = width - 1
    return best


def environment_report() -> dict[str, str]:
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = "Apple Metal (MPS)"
    else:
        device = "cpu"
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "platform": platform.platform(), "processor": platform.processor(),
        "cuda": str(torch.version.cuda), "device": device,
    }


def save_benchmark_report(
    path: str | Path, results: tuple[BenchmarkResult, ...], *, seed: int, notes: Mapping[str, str] | None = None
) -> None:
    payload = {
        "environment": environment_report(), "seed": seed,
        "results": [asdict(result) for result in results], "notes": dict(notes or {}),
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


ABLATION_MATRIX = (
    "local_mixer_only", "real_decay_vs_complex_resonance", "euler_vs_exponential_trapezoid",
    "single_vs_fixed_vs_learned_multiscale", "no_vs_one_way_vs_bidirectional_exchange",
    "dot_product_vs_resonant_attention", "no_vs_recent_vs_distant_memory",
    "magnitude_vs_amplitude_phase_keys", "siso_vs_mimo2_vs_mimo4",
    "fixed_vs_content_dependent_poles", "dense_vs_structured_mixer", "ordinary_vs_alias_controlled",
    "fixed_swiglu_vs_learned_spectral_activation",
)


def local_path_ablation(model: MRRN) -> None:
    """In-place diagnostic ablation retaining only the exact local nonlinear branch."""

    with torch.no_grad():
        for block in model.blocks:
            block.exchange.fine_gain.zero_()
            block.exchange.coarse_gain.zero_()
            for gate in block.branch_gates:
                gate.weight.zero_()
                gate.bias.copy_(gate.bias.new_tensor([-30.0, 30.0, -30.0, -30.0]))


def apply_ablation(model: MRRN, variant: str) -> MRRN:
    """Return an isolated, executable ablation without mutating the supplied full model."""

    supported = {
        "local_mixer_only", "no_cross_scale", "fine_to_coarse_only",
        "coarse_to_fine_only", "no_attention", "fixed_poles",
        "fixed_haar", "magnitude_only_keys",
        "no_spectral_activation", "spectral_only_local", "fixed_spectral_activation",
    }
    if variant not in supported:
        raise ValueError(f"unsupported in-place ablation {variant!r}")
    result = deepcopy(model)
    with torch.no_grad():
        if variant == "local_mixer_only":
            local_path_ablation(result)
        for block in result.blocks:
            if variant in {"no_cross_scale", "local_mixer_only"}:
                block.exchange.fine_gain.zero_()
                block.exchange.coarse_gain.zero_()
            elif variant == "fine_to_coarse_only":
                block.exchange.coarse_gain.zero_()
            elif variant == "coarse_to_fine_only":
                block.exchange.fine_gain.zero_()
            if variant == "no_attention":
                for gate in block.branch_gates:
                    gate.bias[2] = -30
            if variant == "fixed_poles":
                for resonator in block.resonators:
                    for projection in (
                        resonator.delta_projection, resonator.alpha_projection,
                        resonator.omega_projection,
                    ):
                        projection.weight.zero_()
                        projection.bias.zero_()
                        projection.requires_grad_(False)
            if variant == "magnitude_only_keys":
                for attention in block.attentions:
                    for projection in (attention.query_projection, attention.key_projection):
                        projection.weight[1::2].zero_()
                        projection.bias[1::2].zero_()
            if variant in {"no_spectral_activation", "spectral_only_local", "fixed_spectral_activation"}:
                for index, mixer in enumerate(list(block.mixers)):
                    if not isinstance(mixer, HybridSpectralMixer):
                        continue
                    if variant in {"no_spectral_activation", "spectral_only_local"}:
                        block.mixers[index] = (
                            mixer.conventional if variant == "no_spectral_activation" else mixer.spectral
                        )
                    else:
                        spectral = mixer.spectral
                        for parameter in (
                            spectral.gain_coefficients, spectral.phase_coefficients,
                            spectral.raw_triad_weight, spectral.context.weight,
                        ):
                            parameter.zero_()
                            parameter.requires_grad_(False)
        if variant == "fixed_haar":
            result.analysis.requires_grad_(False)
    return result


def make_reference_baselines(config: MRRNConfig) -> dict[str, nn.Module]:
    output = config.resolved_output_dim
    return {
        "transformer": CausalTransformerBaseline(config.input_dim, config.model_dim, output, config.heads, config.layers),
        "real_selective_ssm": RealSelectiveSSMBaseline(config.input_dim, config.model_dim, output, config.layers),
        "long_convolution": LongConvolutionBaseline(config.input_dim, config.model_dim, output, config.layers),
    }
