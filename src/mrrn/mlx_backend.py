"""Compiled Apple-silicon inference for the exact causal MRRN batch graph.

The PyTorch implementation remains the portable reference and checkpoint
authority.  This module imports its parameters without transposition or
approximation and expresses the same sequence path in MLX so Metal can fuse
the many small spectral operations that dominate eager execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi, sqrt
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .config import MRRNConfig

if TYPE_CHECKING:
    from mlx import core as mx
    from .evaluation import CausalTransformerBaseline
    from .model import MRRN


def mlx_available() -> bool:
    """Return whether MLX and an Apple Metal device are usable."""

    try:
        import mlx.core as mx
    except ImportError:
        return False
    return bool(mx.metal.is_available())


def _mlx():
    try:
        import mlx.core as mx
    except ImportError as error:  # pragma: no cover - exercised on non-Apple CI
        raise RuntimeError("MLX is optional; install mrrn[apple] on Apple silicon") from error
    if not mx.metal.is_available():  # pragma: no cover - hardware dependent
        raise RuntimeError("MLX requires an available Apple Metal device")
    return mx


def _array(tensor: torch.Tensor):
    mx = _mlx()
    return mx.array(tensor.detach().cpu().numpy())


def _softplus(x):
    mx = _mlx()
    return mx.maximum(x, 0) + mx.log1p(mx.exp(-mx.abs(x)))


def _linear(parameters: dict[str, Any], buffers: dict[str, Any], x, prefix: str):
    mx = _mlx()
    weight = parameters.get(f"{prefix}.weight", buffers.get(f"{prefix}.weight"))
    if weight is None:
        raise KeyError(f"missing MLX weight {prefix}.weight")
    output = x @ mx.swapaxes(weight, -1, -2)
    bias = parameters.get(f"{prefix}.bias", buffers.get(f"{prefix}.bias"))
    return output if bias is None else output + bias


def _structured_linear(parameters: dict[str, Any], buffers: dict[str, Any], x, prefix: str):
    """Exact MLX form of ``_StructuredProjection`` from the PyTorch graph."""

    result = _linear(
        parameters, buffers,
        _linear(parameters, buffers, x, f"{prefix}.down"), f"{prefix}.up",
    )
    skip_name = f"{prefix}.skip.weight"
    if skip_name in parameters or skip_name in buffers:
        return result + _linear(parameters, buffers, x, f"{prefix}.skip")
    # The implicit identity is used only when source and target widths match.
    if result.shape[-1] != x.shape[-1]:
        raise ValueError(f"structured projection {prefix} lacks its required skip")
    return result + x


def _packed_linear(parameters: dict[str, Any], buffers: dict[str, Any], x, prefixes: tuple[str, ...]):
    """Evaluate independent projections of one tensor with one matrix multiplication."""

    mx = _mlx()
    cache = "__packed__." + "|".join(prefixes)
    if f"{cache}.weight" in buffers:
        weight = buffers[f"{cache}.weight"]
        bias = buffers[f"{cache}.bias"]
        sizes = buffers[f"{cache}.sizes"]
        projected = x @ mx.swapaxes(weight, -1, -2) + bias
        return tuple(mx.split(projected, np.cumsum(sizes[:-1]).tolist(), axis=-1))
    weights, biases, sizes = [], [], []
    for prefix in prefixes:
        weight = parameters.get(f"{prefix}.weight", buffers.get(f"{prefix}.weight"))
        if weight is None:
            raise KeyError(f"missing MLX weight {prefix}.weight")
        bias = parameters.get(f"{prefix}.bias", buffers.get(f"{prefix}.bias"))
        weights.append(weight)
        biases.append(mx.zeros((weight.shape[0],), dtype=weight.dtype) if bias is None else bias)
        sizes.append(weight.shape[0])
    projected = x @ mx.swapaxes(mx.concatenate(weights, axis=0), -1, -2)
    projected = projected + mx.concatenate(biases, axis=0)
    cuts = np.cumsum(sizes[:-1]).tolist()
    return tuple(mx.split(projected, cuts, axis=-1))


def _cache_packed(parameters: dict[str, Any], buffers: dict[str, Any], prefixes: tuple[str, ...]) -> None:
    """Materialize immutable inference packs; training temporarily removes them."""

    mx = _mlx()
    cache = "__packed__." + "|".join(prefixes)
    weights, biases, sizes = [], [], []
    for prefix in prefixes:
        weight = parameters.get(f"{prefix}.weight", buffers.get(f"{prefix}.weight"))
        bias = parameters.get(f"{prefix}.bias", buffers.get(f"{prefix}.bias"))
        weights.append(weight)
        biases.append(mx.zeros((weight.shape[0],), dtype=weight.dtype) if bias is None else bias)
        sizes.append(weight.shape[0])
    buffers[f"{cache}.weight"] = mx.concatenate(weights, axis=0)
    buffers[f"{cache}.bias"] = mx.concatenate(biases, axis=0)
    buffers[f"{cache}.sizes"] = tuple(sizes)


def _rms_norm(parameters: dict[str, Any], buffers: dict[str, Any], x, prefix: str):
    mx = _mlx()
    weight = parameters.get(f"{prefix}.weight", buffers.get(f"{prefix}.weight"))
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + np.finfo(np.float32).eps) * weight


def _cmul(a, b):
    mx = _mlx()
    return mx.stack((a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
                     a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]), axis=-1)


def _cconj(z):
    mx = _mlx()
    return mx.stack((z[..., 0], -z[..., 1]), axis=-1)


def _cabs2(z):
    return z[..., 0] * z[..., 0] + z[..., 1] * z[..., 1]


def _cscale(z, factor):
    return z * factor[..., None]


def _cexp(z):
    mx = _mlx()
    magnitude = mx.exp(z[..., 0])
    return mx.stack((magnitude * mx.cos(z[..., 1]), magnitude * mx.sin(z[..., 1])), axis=-1)


def _crotate(z, angle):
    mx = _mlx()
    return _cmul(z, mx.stack((mx.cos(angle), mx.sin(angle)), axis=-1))


def _cdivide(a, b):
    mx = _mlx()
    denominator = mx.maximum(_cabs2(b), np.finfo(np.float32).tiny)
    return _cscale(_cmul(a, _cconj(b)), mx.reciprocal(denominator))


def _phi_functions(q):
    mx = _mlx()
    one = mx.stack((mx.ones_like(q[..., 0]), mx.zeros_like(q[..., 0])), axis=-1)
    half = one * mx.array(0.5, dtype=q.dtype)
    q2, q3 = _cmul(q, q), None
    q3 = _cmul(q2, q)
    exp_q = _cexp(q)
    phi1 = _cdivide(exp_q - one, q)
    phi2 = _cdivide(exp_q - one - q, q2)
    series1 = one + 0.5 * q + (1 / 6) * q2 + (1 / 24) * q3
    series2 = half + (1 / 6) * q + (1 / 24) * q2 + (1 / 120) * q3
    small = (_cabs2(q) < 1e-6)[..., None]
    return mx.where(small, series1, phi1), mx.where(small, series2, phi2)


def _complex_rms(z, axis: int, eps: float):
    mx = _mlx()
    return _cscale(z, mx.rsqrt(mx.mean(_cabs2(z), axis=axis, keepdims=True) + eps))


def _safe_softmax(scores, axis: int):
    mx = _mlx()
    valid = mx.isfinite(scores)
    safe = mx.where(valid, scores, -mx.array(np.finfo(np.float32).max, dtype=scores.dtype))
    weights = mx.softmax(safe, axis=axis) * valid
    return weights / mx.maximum(mx.sum(weights, axis=axis, keepdims=True), np.finfo(np.float32).tiny)


@dataclass(slots=True)
class MLXMRRNStreamState:
    """Fixed-shape Metal-resident state for constant-memory recurrent decode."""

    tensors: dict[str, Any]
    position: int
    batch: int


@dataclass(slots=True)
class MLXTransformerStreamState:
    """Growing exact KV cache for the matched causal Transformer."""

    keys: tuple[Any, ...]
    values: tuple[Any, ...]
    position: int
    batch: int
    capacity: int


class MLXMRRN:
    """Exact-weight MLX executor for causal sequence-mode MRRN inference/training."""

    def __init__(self, model: "MRRN", *, compile: bool = True, training: bool = False) -> None:
        mx = _mlx()
        if not model.config.causal:
            raise ValueError("the optimized MLX executor currently requires a causal MRRN")
        self.config = model.config
        parameter_names = {name for name, _ in model.named_parameters(remove_duplicate=False)}
        state = model.state_dict()
        self.parameters = {name: _array(value) for name, value in state.items() if name in parameter_names}
        self.buffers = {name: _array(value) for name, value in state.items() if name not in parameter_names}
        self.training = training
        if not training:
            self._pack_inference_weights()
        self._forward = (
            mx.compile(self._forward_impl, inputs=[self.parameters, self.buffers])
            if compile else self._forward_impl
        )
        self._loss_grad = (
            mx.compile(mx.value_and_grad(self._loss_impl), inputs=self.buffers)
            if compile and training else None
        )
        self._compile_streaming = compile
        self._stream_functions: dict[int, Any] = {}

    def _pack_inference_weights(self) -> None:
        for block in range(self.config.layers):
            for scale in range(self.config.scales):
                resonator = f"blocks.{block}.resonators.{scale}"
                _cache_packed(self.parameters, self.buffers, tuple(
                    f"{resonator}.{name}" for name in (
                        "input_projection", "delta_projection", "alpha_projection",
                        "omega_projection", "input_direction", "readout_direction",
                    )
                ))
                mixer = f"blocks.{block}.mixers.{scale}"
                if self.config.structured_mixer_rank is None:
                    _cache_packed(self.parameters, self.buffers, (
                        f"{mixer}.conventional.a", f"{mixer}.conventional.b",
                        f"{mixer}.spectral.control", f"{mixer}.spectral.carrier",
                        f"{mixer}.spectral.context", f"{mixer}.blend",
                    ))
                else:
                    _cache_packed(self.parameters, self.buffers, (
                        f"{mixer}.spectral.control", f"{mixer}.spectral.carrier",
                        f"{mixer}.spectral.context", f"{mixer}.blend",
                    ))
                attention = f"blocks.{block}.attentions.{scale}"
                _cache_packed(self.parameters, self.buffers, (
                    f"{attention}.key_projection", f"{attention}.value_projection",
                ))
                _cache_packed(self.parameters, self.buffers, (
                    f"blocks.{block}.identity.{scale}", f"blocks.{block}.branch_gates.{scale}",
                ))
        if self.config.structured_mixer_rank is None:
            _cache_packed(self.parameters, self.buffers, ("raw_mixer.a", "raw_mixer.b"))

    def initial_stream_state(self, batch: int) -> MLXMRRNStreamState:
        """Allocate all recurrent, lifting, exchange, and candidate caches once."""

        if self.training:
            raise RuntimeError("streaming decode requires the inference executor")
        if batch <= 0:
            raise ValueError("batch must be positive")
        mx = _mlx()
        state: dict[str, Any] = {}
        kernel_history = self.config.lifting_kernel - 1
        for level in range(self.config.scales - 1):
            state[f"lift.pending_value.{level}"] = mx.zeros(
                (batch, self.config.model_dim), dtype=mx.float32
            )
            state[f"lift.pending_mask.{level}"] = mx.zeros((batch,), dtype=mx.bool_)
            state[f"lift.even_history.{level}"] = mx.zeros(
                (batch, kernel_history, self.config.model_dim), dtype=mx.float32
            )
            state[f"lift.detail_history.{level}"] = mx.zeros(
                (batch, kernel_history, self.config.model_dim), dtype=mx.float32
            )
        scales = self.config.scale_configs()
        for block in range(self.config.layers):
            for edge in range(self.config.scales - 1):
                coarse_width = scales[edge + 1].width
                state[f"exchange.sum.{block}.{edge}"] = mx.zeros(
                    (batch, coarse_width), dtype=mx.float32
                )
                state[f"exchange.count.{block}.{edge}"] = mx.zeros(
                    (batch, 1), dtype=mx.float32
                )
                state[f"exchange.latest.{block}.{edge}"] = mx.zeros(
                    (batch, coarse_width), dtype=mx.float32
                )
                state[f"exchange.latest_mask.{block}.{edge}"] = mx.zeros(
                    (batch,), dtype=mx.bool_
                )
            for scale, scale_config in enumerate(scales):
                shape = (
                    batch, scale_config.heads, scale_config.modes,
                    scale_config.mimo_rank, 2,
                )
                state[f"res.value.{block}.{scale}"] = mx.zeros(shape, dtype=mx.float32)
                state[f"res.drive.{block}.{scale}"] = mx.zeros(shape, dtype=mx.float32)
                if self.config.continuous_signal:
                    width = scale_config.width
                    state[f"alias.pre.{block}.{scale}"] = mx.zeros(
                        (batch, 4, width), dtype=mx.float32
                    )
                    state[f"alias.post.{block}.{scale}"] = mx.zeros(
                        (batch, 4, width), dtype=mx.float32
                    )
                state[f"recent.features.{block}.{scale}"] = mx.zeros(
                    (batch, self.config.attention_window, scale_config.width), dtype=mx.float32
                )
                state[f"recent.mask.{block}.{scale}"] = mx.zeros(
                    (batch, self.config.attention_window), dtype=mx.bool_
                )
                state[f"recent.times.{block}.{scale}"] = mx.zeros(
                    (batch, self.config.attention_window), dtype=mx.float32
                )
        for scale, scale_config in enumerate(scales):
            state[f"latest.data.{scale}"] = mx.zeros(
                (batch, scale_config.width), dtype=mx.float32
            )
            state[f"latest.mask.{scale}"] = mx.zeros((batch,), dtype=mx.bool_)
        mx.eval(state)
        return MLXMRRNStreamState(state, 0, batch)

    def _lifting_history_step(self, parameters, history, value, prefix: str):
        mx = _mlx()
        complete = mx.concatenate((history, value[:, None]), axis=1)
        output = self._lifting_filter(parameters, complete, prefix)[:, -1]
        return output, complete[:, 1:]

    def _lifting_stream_step(self, parameters, state, encoded, mask, phase: int):
        next_state = dict(state)
        active: list[dict[str, Any] | None] = [None] * self.config.scales
        value, valid = encoded, mask
        for level in range(self.config.scales - 1):
            arrival_period = 2**level
            if phase % arrival_period != arrival_period - 1:
                break
            completion_period = 2 ** (level + 1)
            if phase % completion_period != completion_period - 1:
                next_state[f"lift.pending_value.{level}"] = value
                next_state[f"lift.pending_mask.{level}"] = valid
                break
            even = state[f"lift.pending_value.{level}"]
            even_valid = state[f"lift.pending_mask.{level}"]
            prefix = f"analysis.levels.{level}"
            prediction, even_history = self._lifting_history_step(
                parameters, state[f"lift.even_history.{level}"], even, f"{prefix}.predict"
            )
            detail = value - prediction
            update, detail_history = self._lifting_history_step(
                parameters, state[f"lift.detail_history.{level}"], detail, f"{prefix}.update"
            )
            next_state[f"lift.even_history.{level}"] = even_history
            next_state[f"lift.detail_history.{level}"] = detail_history
            coefficient_mask = even_valid & valid
            active[level] = {
                "data": detail[:, None], "mask": coefficient_mask[:, None],
                "scale": level, "support": 2 ** (level + 1), "kind": 0,
            }
            value, valid = even + update, coefficient_mask
        else:
            active[-1] = {
                "data": value[:, None], "mask": valid[:, None],
                "scale": self.config.scales - 1,
                "support": 2 ** (self.config.scales - 1), "kind": 1,
            }
        for scale, band in enumerate(active):
            if band is not None:
                projected = _linear(
                    parameters, self.buffers, band["data"], f"analysis_adapters.{scale}"
                )
                band["data"] = projected * band["mask"][..., None]
        return active, next_state

    def _exchange_stream_step(self, parameters, bands, state, block: int):
        mx = _mlx()
        next_state = dict(state)
        conditioned: list[Any | None] = []
        prefix = f"blocks.{block}.exchange"
        for scale, band in enumerate(bands):
            if band is None:
                conditioned.append(None)
                continue
            metadata = mx.array(
                [log(float(2**scale)), log(float(band["support"]))], dtype=band["data"].dtype
            )
            data = band["data"] + self._p(parameters, f"{prefix}.scale_codes.{scale}")
            data = data + _linear(parameters, self.buffers, metadata, f"{prefix}.metadata.{scale}")
            conditioned.append(data * band["mask"][..., None])
        fine_messages: list[Any | None] = [None] * self.config.scales
        coarse_messages: list[Any | None] = [None] * self.config.scales
        for edge in range(self.config.scales - 1):
            fine, coarse = bands[edge], bands[edge + 1]
            sum_key, count_key = f"exchange.sum.{block}.{edge}", f"exchange.count.{block}.{edge}"
            sum_value, count = state[sum_key], state[count_key]
            if fine is not None:
                selected = mx.sigmoid(_linear(
                    parameters, self.buffers, conditioned[edge], f"{prefix}.fine_gate.{edge}"
                )) * conditioned[edge]
                projected = _linear(
                    parameters, self.buffers, selected, f"{prefix}.fine_value.{edge}"
                )[:, 0]
                active = fine["mask"][:, 0, None]
                sum_value = sum_value + projected * active
                count = count + active
            if coarse is not None:
                fine_messages[edge + 1] = (
                    sum_value / mx.maximum(count, 1)
                )[:, None]
                next_state[sum_key] = mx.zeros_like(sum_value)
                next_state[count_key] = mx.zeros_like(count)
                next_state[f"exchange.latest.{block}.{edge}"] = conditioned[edge + 1][:, 0]
                next_state[f"exchange.latest_mask.{block}.{edge}"] = coarse["mask"][:, 0]
            else:
                next_state[sum_key], next_state[count_key] = sum_value, count
            if fine is not None:
                if coarse is not None:
                    context = conditioned[edge + 1]
                else:
                    context = state[f"exchange.latest.{block}.{edge}"][:, None]
                modulation = _linear(
                    parameters, self.buffers, context, f"{prefix}.coarse_modulation.{edge}"
                )
                gamma, beta = mx.split(modulation, 2, axis=-1)
                normalized = conditioned[edge] * mx.rsqrt(
                    mx.mean(conditioned[edge] ** 2, axis=-1, keepdims=True)
                    + np.finfo(np.float32).eps
                )
                coarse_messages[edge] = mx.tanh(gamma) * normalized + beta
        fine_gain = self._p(parameters, f"{prefix}.fine_gain")
        coarse_gain = self._p(parameters, f"{prefix}.coarse_gain")
        output = []
        for scale, band in enumerate(bands):
            if band is None:
                output.append(None)
                continue
            updated = conditioned[scale]
            if fine_messages[scale] is not None:
                updated = updated + fine_gain[scale - 1] * fine_messages[scale]
            if coarse_messages[scale] is not None:
                updated = updated + coarse_gain[scale] * coarse_messages[scale]
            output.append({**band, "data": updated * band["mask"][..., None]})
        return output, next_state

    def _resonator_stream_step(self, parameters, u, mask, state, block: int, scale: int, interval: float):
        mx = _mlx()
        config = self.config.scale_configs()[scale]
        h, n, r = config.heads, config.modes, config.mimo_rank
        prefix = f"blocks.{block}.resonators.{scale}"
        names = (
            "input_projection", "delta_projection", "alpha_projection", "omega_projection",
            "input_direction", "readout_direction",
        )
        amplitudes, delta_content, alpha_content, omega_content, direction, readout = _packed_linear(
            parameters, self.buffers, u, tuple(f"{prefix}.{name}" for name in names)
        )
        batch = u.shape[0]
        delta = (self.config.delta_min + _softplus(delta_content).reshape(batch, 1, h, 1, 1)) * interval
        alpha = self.config.alpha_min + _softplus(
            self._p(parameters, f"{prefix}.raw_alpha") + alpha_content.reshape(batch, 1, h, n)
        )
        omega_max = self.config.omega_max / (
            2 ** (scale + 1) if scale < self.config.scales - 1 else 2**scale
        )
        omega = omega_max * mx.tanh(
            self._p(parameters, f"{prefix}.raw_omega") + omega_content.reshape(batch, 1, h, n)
        )
        q = mx.stack((-alpha, omega), axis=-1)[:, :, :, :, None, :] * delta[..., None]
        transition = mx.broadcast_to(_cexp(q), (batch, 1, h, n, r, 2))
        phi1, phi2 = _phi_functions(q)
        direction = _complex_rms(direction.reshape(batch, 1, h, n, r, 2), 3, 1e-6)
        readout = _complex_rms(readout.reshape(batch, 1, h, n, r, 2), 3, 1e-6)
        drive = direction * amplitudes.reshape(batch, 1, h, r)[:, :, :, None, :, None]
        if self.config.decay_normalized_resonance:
            drive = drive * alpha[..., None, None]
        previous = state[f"res.drive.{block}.{scale}"][:, None]
        affine = (_cmul(phi1 - phi2, previous) + _cmul(phi2, drive)) * delta[..., None]
        old_value = state[f"res.value.{block}.{scale}"][:, None]
        proposed = _cmul(transition, old_value) + affine
        active = mask[..., None, None, None, None]
        value = mx.where(active, proposed, old_value)
        next_drive = mx.where(active, drive, previous)
        projected = (readout[..., 0] * value[..., 0] + readout[..., 1] * value[..., 1]).sum(axis=3)
        output = _linear(
            parameters, self.buffers, projected.reshape(batch, 1, h * r),
            f"{prefix}.output_projection",
        )
        gate = mx.sigmoid(_linear(parameters, self.buffers, u, f"{prefix}.output_gate"))
        return gate * output * mask[..., None], value[:, 0], next_drive[:, 0]

    def _block_stream_step(self, parameters, bands, state, block: int):
        mx = _mlx()
        bands, next_state = self._exchange_stream_step(parameters, bands, state, block)
        schedule = tuple(
            (scale == self.config.scales - 1)
            or (scale == 0 and block % 2 == 0)
            or (0 < scale < self.config.scales - 1 and block % 3 == 0)
            for scale in range(self.config.scales)
        )
        outputs = []
        for scale, band in enumerate(bands):
            if band is None:
                outputs.append(None)
                continue
            feature_key = f"recent.features.{block}.{scale}"
            mask_key = f"recent.mask.{block}.{scale}"
            time_key = f"recent.times.{block}.{scale}"
            next_state[feature_key] = mx.concatenate(
                (state[feature_key][:, 1:], band["data"]), axis=1
            )
            next_state[mask_key] = mx.concatenate(
                (state[mask_key][:, 1:], band["mask"]), axis=1
            )
            next_state[time_key] = mx.concatenate((
                state[time_key][:, 1:],
                mx.zeros((band["data"].shape[0], 1), dtype=band["data"].dtype),
            ), axis=1)
            normalized = _rms_norm(
                parameters, self.buffers, band["data"], f"blocks.{block}.norms.{scale}"
            ) * band["mask"][..., None]
            resonant, value, drive = self._resonator_stream_step(
                parameters, normalized, band["mask"], state, block, scale, float(band["support"])
            )
            next_state[f"res.value.{block}.{scale}"] = value
            next_state[f"res.drive.{block}.{scale}"] = drive
            local = self._hybrid_mixer(parameters, normalized, block, scale) * band["mask"][..., None]
            if self.config.continuous_signal:
                local_value, pre_history, post_history = self._anti_alias_stream(
                    parameters, local[:, 0], state, block, scale
                )
                local = local_value[:, None]
                next_state[f"alias.pre.{block}.{scale}"] = pre_history
                next_state[f"alias.post.{block}.{scale}"] = post_history
                local = local * band["mask"][..., None]
            if schedule[scale]:
                candidates = [
                    next_state[feature_key], -next_state[time_key],
                    mx.full(next_state[time_key].shape, float(scale), dtype=band["data"].dtype),
                    next_state[mask_key],
                ]
                if scale + 1 < self.config.scales:
                    count = max(1, min(8, self.config.attention_window // 4))
                    coarse_features = state[f"recent.features.{block}.{scale + 1}"][:, -count:]
                    landmark = _linear(
                        parameters, self.buffers, coarse_features,
                        f"blocks.{block}.landmark_values.{scale}",
                    )
                    coarse_ages = state[f"recent.times.{block}.{scale + 1}"][:, -count:]
                    coarse_times = -coarse_ages
                    coarse_mask = state[f"recent.mask.{block}.{scale + 1}"][:, -count:]
                    candidates = [
                        mx.concatenate((candidates[0], landmark), axis=1),
                        mx.concatenate((candidates[1], coarse_times), axis=1),
                        mx.concatenate((
                            candidates[2], mx.full(coarse_times.shape, float(scale + 1),
                                                   dtype=band["data"].dtype)
                        ), axis=1),
                        mx.concatenate((
                            candidates[3], coarse_mask
                        ), axis=1),
                    ]
                query_times = mx.zeros((band["data"].shape[0], 1), dtype=band["data"].dtype)
                query_scales = mx.full_like(query_times, float(scale))
                attended = self._attention(
                    parameters, band["data"], candidates, query_times, query_scales, block, scale
                ) * band["mask"][..., None]
            else:
                attended = mx.zeros_like(band["data"])
            identity, branch_logits = _packed_linear(
                parameters, self.buffers, normalized,
                (f"blocks.{block}.identity.{scale}", f"blocks.{block}.branch_gates.{scale}"),
            )
            branch = mx.softmax(branch_logits, axis=-1)
            delta = (branch[..., :1] * resonant + branch[..., 1:2] * local
                     + branch[..., 2:3] * attended + branch[..., 3:4] * identity)
            layer_scale = self._p(parameters, f"blocks.{block}.layer_scale")[scale]
            outputs.append({
                **band,
                "data": (band["data"] + layer_scale * delta) * band["mask"][..., None],
            })
        return outputs, next_state

    def _stream_step_impl(self, state, x, mask, phase: int):
        mx = _mlx()
        aged_state = dict(state)
        for block in range(self.config.layers):
            for scale in range(self.config.scales):
                key = f"recent.times.{block}.{scale}"
                mask_key = f"recent.mask.{block}.{scale}"
                aged_state[key] = state[key] + state[mask_key]
        encoded = _linear(self.parameters, self.buffers, x, "encoder") * mask[:, None]
        bands, next_state = self._lifting_stream_step(
            self.parameters, aged_state, encoded, mask, phase
        )
        for block in range(self.config.layers):
            bands, next_state = self._block_stream_step(
                self.parameters, bands, next_state, block
            )
        for scale, band in enumerate(bands):
            if band is not None:
                next_state[f"latest.data.{scale}"] = band["data"][:, 0]
                next_state[f"latest.mask.{scale}"] = band["mask"][:, 0]
        raw = _rms_norm(self.parameters, self.buffers, encoded, "raw_norm")
        if self.config.structured_mixer_rank is None:
            a, b = _packed_linear(
                self.parameters, self.buffers, raw, ("raw_mixer.a", "raw_mixer.b")
            )
            mixed = _linear(
                self.parameters, self.buffers, a * mx.sigmoid(a) * b,
                "raw_mixer.output",
            )
        else:
            a = _structured_linear(self.parameters, self.buffers, raw, "raw_mixer.a")
            b = _structured_linear(self.parameters, self.buffers, raw, "raw_mixer.b")
            mixed = _structured_linear(
                self.parameters, self.buffers, a * mx.sigmoid(a) * b,
                "raw_mixer.output",
            )
        latent = encoded + self._p(self.parameters, "raw_gain") * mixed
        gains = self._p(self.parameters, "scale_output_gain")
        for scale in range(self.config.scales):
            contribution = _linear(
                self.parameters, self.buffers, next_state[f"latest.data.{scale}"],
                f"synthesis_adapters.{scale}",
            )
            latent = latent + gains[scale] * contribution * next_state[f"latest.mask.{scale}"][:, None]
        prediction = _linear(self.parameters, self.buffers, latent, "output_head") * mask[:, None]
        return prediction, next_state

    def stream_step(self, x, state: MLXMRRNStreamState, mask=None):
        """Advance one token with constant-size state and a phase-specialized graph."""

        mx = _mlx()
        x = _array(x) if isinstance(x, torch.Tensor) else x
        if x.ndim == 3 and x.shape[1] == 1:
            x = x[:, 0]
        if x.shape != (state.batch, self.config.input_dim):
            raise ValueError("stream input shape does not match the state")
        if mask is None:
            mask = mx.ones((state.batch,), dtype=mx.bool_)
        elif isinstance(mask, torch.Tensor):
            mask = _array(mask)
        cycle = 2 ** (self.config.scales - 1)
        phase = state.position % cycle
        function = self._stream_functions.get(phase)
        if function is None:
            function = lambda tensors, value, valid: self._stream_step_impl(
                tensors, value, valid, phase
            )
            if self._compile_streaming:
                function = mx.compile(function, inputs=[self.parameters, self.buffers])
            self._stream_functions[phase] = function
        prediction, tensors = function(state.tensors, x, mask)
        mx.eval(prediction, tensors)
        return prediction, MLXMRRNStreamState(tensors, state.position + 1, state.batch)

    def decode(self, x, state: MLXMRRNStreamState | None = None):
        """Decode a complete sequence through the exact recurrent path."""

        mx = _mlx()
        x = _array(x) if isinstance(x, torch.Tensor) else x
        state = self.initial_stream_state(x.shape[0]) if state is None else state
        outputs = []
        for position in range(x.shape[1]):
            prediction, state = self.stream_step(x[:, position], state)
            outputs.append(prediction[:, None])
        return mx.concatenate(outputs, axis=1) if outputs else mx.zeros(
            (x.shape[0], 0, self.config.resolved_output_dim), dtype=x.dtype
        ), state

    def benchmark_decode(self, *, repeats: int = 32, warmup_cycles: int = 1) -> dict[str, float]:
        """Measure phase-averaged recurrent latency after bounded caches are warm."""

        if repeats <= 0 or warmup_cycles < 0:
            raise ValueError("decode benchmark controls are invalid")
        mx = _mlx()
        cycle = 2 ** (self.config.scales - 1)
        state = self.initial_stream_state(1)
        x = mx.zeros((1, self.config.input_dim), dtype=mx.float32)
        for _ in range((warmup_cycles + 1) * cycle):
            _, state = self.stream_step(x, state)
        values = []
        for _ in range(repeats):
            start = perf_counter()
            _, state = self.stream_step(x, state)
            values.append(perf_counter() - start)
        ordered = sorted(values)
        return {
            "mean_seconds": sum(values) / len(values),
            "p95_seconds": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        }

    @property
    def parameter_count(self) -> int:
        return sum(array.size for array in self.parameters.values())

    def _p(self, parameters: dict[str, Any], name: str):
        value = parameters.get(name)
        return self.buffers[name] if value is None else value

    def _lifting_filter(self, parameters, x, prefix: str):
        mx = _mlx()
        point = _linear(parameters, self.buffers, x, f"{prefix}.point")
        weight = self._p(parameters, f"{prefix}.depth.weight")
        weight = mx.transpose(weight, (0, 2, 1))
        kernel = weight.shape[1]
        padded = mx.pad(x, ((0, 0), (kernel - 1, 0), (0, 0)))
        temporal = mx.conv_general(padded, weight, groups=x.shape[-1])
        return point + temporal + self._p(parameters, f"{prefix}.depth.bias")

    def _analysis(self, parameters, encoded, mask):
        mx = _mlx()
        current, current_mask = encoded, mask
        bands = []
        for scale in range(self.config.scales - 1):
            even, odd = current[:, 0::2], current[:, 1::2]
            even_mask, odd_mask = current_mask[:, 0::2], current_mask[:, 1::2]
            paired = odd.shape[1]
            even_pair = even[:, :paired]
            prefix = f"analysis.levels.{scale}"
            detail = odd - self._lifting_filter(parameters, even_pair, f"{prefix}.predict")
            approximation = even_pair + self._lifting_filter(parameters, detail, f"{prefix}.update")
            valid = even_mask[:, :paired] & odd_mask
            approximation_mask = valid
            if even.shape[1] > paired:
                approximation = mx.concatenate((approximation, even[:, -1:]), axis=1)
                approximation_mask = mx.concatenate((approximation_mask, even_mask[:, -1:]), axis=1)
            bands.append({"data": detail, "mask": valid, "scale": scale,
                          "support": 2 ** (scale + 1), "kind": 0})
            current, current_mask = approximation, approximation_mask
        bands.append({"data": current, "mask": current_mask, "scale": self.config.scales - 1,
                      "support": 2 ** (self.config.scales - 1), "kind": 1})
        output = []
        for scale, band in enumerate(bands):
            data = _linear(parameters, self.buffers, band["data"], f"analysis_adapters.{scale}")
            output.append({**band, "data": data * band["mask"][..., None]})
        return output

    @staticmethod
    def _fine_to_coarse(x, mask, target_length: int, fine_support: int, coarse_support: int):
        mx = _mlx()
        pieces = []
        for target in range(target_length):
            start = target * coarse_support // fine_support
            stop = min((target + 1) * coarse_support // fine_support, x.shape[1])
            if start >= stop:
                pieces.append(mx.zeros_like(x[:, :1]))
            else:
                valid = mask[:, start:stop, None]
                count = mx.maximum(mx.sum(valid, axis=1, keepdims=True), 1)
                pieces.append(mx.sum(x[:, start:stop] * valid, axis=1, keepdims=True) / count)
        return mx.concatenate(pieces, axis=1) if pieces else x[:, :0]

    @staticmethod
    def _coarse_to_fine(x, target_length: int, coarse_support: int, fine_support: int):
        mx = _mlx()
        position = mx.arange(target_length)
        index = ((position + 1) * fine_support) // coarse_support - 1
        completed = index >= 0
        index = mx.clip(index, 0, max(0, x.shape[1] - 1))
        return mx.take(x, index, axis=1) * completed[None, :, None]

    def _exchange(self, parameters, bands, block: int):
        mx = _mlx()
        conditioned = []
        for scale, band in enumerate(bands):
            metadata = mx.array([log(float(2**scale)), log(float(band["support"]))], dtype=band["data"].dtype)
            prefix = f"blocks.{block}.exchange"
            data = band["data"] + self._p(parameters, f"{prefix}.scale_codes.{scale}")
            data = data + _linear(parameters, self.buffers, metadata, f"{prefix}.metadata.{scale}")
            conditioned.append(data * band["mask"][..., None])
        fine_messages = [None] * len(bands)
        coarse_messages = [None] * len(bands)
        for scale in range(len(bands) - 1):
            prefix = f"blocks.{block}.exchange"
            selected = mx.sigmoid(_linear(
                parameters, self.buffers, conditioned[scale], f"{prefix}.fine_gate.{scale}"
            )) * conditioned[scale]
            projected = _linear(parameters, self.buffers, selected, f"{prefix}.fine_value.{scale}")
            fine_messages[scale + 1] = self._fine_to_coarse(
                projected, bands[scale]["mask"], conditioned[scale + 1].shape[1],
                bands[scale]["support"], bands[scale + 1]["support"],
            )
            context = self._coarse_to_fine(
                conditioned[scale + 1], conditioned[scale].shape[1],
                bands[scale + 1]["support"], bands[scale]["support"],
            )
            modulation = _linear(
                parameters, self.buffers, context, f"{prefix}.coarse_modulation.{scale}"
            )
            gamma, beta = mx.split(modulation, 2, axis=-1)
            normalized = conditioned[scale] * mx.rsqrt(
                mx.mean(conditioned[scale] ** 2, axis=-1, keepdims=True) + np.finfo(np.float32).eps
            )
            coarse_messages[scale] = mx.tanh(gamma) * normalized + beta
        fine_gain = self._p(parameters, f"blocks.{block}.exchange.fine_gain")
        coarse_gain = self._p(parameters, f"blocks.{block}.exchange.coarse_gain")
        output = []
        for scale, band in enumerate(bands):
            updated = conditioned[scale]
            if fine_messages[scale] is not None:
                updated = updated + fine_gain[scale - 1] * fine_messages[scale]
            if coarse_messages[scale] is not None:
                updated = updated + coarse_gain[scale] * coarse_messages[scale]
            output.append({**band, "data": updated * band["mask"][..., None]})
        return output

    @staticmethod
    def _affine_scan(transition, drive, initial, *, work_efficient: bool = False):
        mx = _mlx()

        if not work_efficient:
            prefix_a, prefix_b = transition, drive
            offset = 1
            while offset < transition.shape[1]:
                current_a, previous_a = prefix_a[:, offset:], prefix_a[:, :-offset]
                current_b, previous_b = prefix_b[:, offset:], prefix_b[:, :-offset]
                prefix_a = mx.concatenate(
                    (prefix_a[:, :offset], _cmul(current_a, previous_a)), axis=1
                )
                prefix_b = mx.concatenate(
                    (prefix_b[:, :offset], current_b + _cmul(current_a, previous_b)), axis=1
                )
                offset *= 2
            return _cmul(prefix_a, initial[:, None]) + prefix_b

        def prefix(a, b):
            length = a.shape[1]
            if length <= 1:
                return a, b
            pairs = length // 2
            even_a, odd_a = a[:, : 2 * pairs : 2], a[:, 1 : 2 * pairs : 2]
            even_b, odd_b = b[:, : 2 * pairs : 2], b[:, 1 : 2 * pairs : 2]
            pair_a = _cmul(odd_a, even_a)
            pair_b = odd_b + _cmul(odd_a, even_b)
            scanned_a, scanned_b = prefix(pair_a, pair_b)
            if pairs > 1:
                remaining_a = _cmul(even_a[:, 1:], scanned_a[:, :-1])
                remaining_b = even_b[:, 1:] + _cmul(even_a[:, 1:], scanned_b[:, :-1])
                prefix_even_a = mx.concatenate((even_a[:, :1], remaining_a), axis=1)
                prefix_even_b = mx.concatenate((even_b[:, :1], remaining_b), axis=1)
            else:
                prefix_even_a, prefix_even_b = even_a, even_b
            prefix_a = mx.stack((prefix_even_a, scanned_a), axis=2).reshape(
                a.shape[0], 2 * pairs, *a.shape[2:]
            )
            prefix_b = mx.stack((prefix_even_b, scanned_b), axis=2).reshape(
                b.shape[0], 2 * pairs, *b.shape[2:]
            )
            if length % 2:
                tail_a = _cmul(a[:, -1:], scanned_a[:, -1:])
                tail_b = b[:, -1:] + _cmul(a[:, -1:], scanned_b[:, -1:])
                prefix_a = mx.concatenate((prefix_a, tail_a), axis=1)
                prefix_b = mx.concatenate((prefix_b, tail_b), axis=1)
            return prefix_a, prefix_b

        prefix_a, prefix_b = prefix(transition, drive)
        return _cmul(prefix_a, initial[:, None]) + prefix_b

    def _resonator(self, parameters, u, mask, block: int, scale: int, interval: float):
        mx = _mlx()
        config = self.config.scale_configs()[scale]
        h, n, r = config.heads, config.modes, config.mimo_rank
        prefix = f"blocks.{block}.resonators.{scale}"
        names = ("input_projection", "delta_projection", "alpha_projection", "omega_projection",
                 "input_direction", "readout_direction")
        sizes = (h * r, h, h * n, h * n, 2 * h * n * r, 2 * h * n * r)
        amplitudes, delta_content, alpha_content, omega_content, direction, readout = _packed_linear(
            parameters, self.buffers, u, tuple(f"{prefix}.{name}" for name in names)
        )
        batch, length = u.shape[:2]
        delta = (self.config.delta_min + _softplus(delta_content).reshape(batch, length, h, 1, 1)) * interval
        alpha = self.config.alpha_min + _softplus(
            self._p(parameters, f"{prefix}.raw_alpha") + alpha_content.reshape(batch, length, h, n)
        )
        omega_max = self.config.omega_max / (
            2 ** (scale + 1) if scale < self.config.scales - 1 else 2**scale
        )
        omega = omega_max * mx.tanh(
            self._p(parameters, f"{prefix}.raw_omega") + omega_content.reshape(batch, length, h, n)
        )
        q = mx.stack((-alpha, omega), axis=-1)[:, :, :, :, None, :] * delta[..., None]
        transition = mx.broadcast_to(_cexp(q), (batch, length, h, n, r, 2))
        phi1, phi2 = _phi_functions(q)
        amplitudes = amplitudes.reshape(batch, length, h, r)
        direction = _complex_rms(direction.reshape(batch, length, h, n, r, 2), 3, 1e-6)
        readout = _complex_rms(readout.reshape(batch, length, h, n, r, 2), 3, 1e-6)
        drive = direction * amplitudes[:, :, :, None, :, None]
        if self.config.decay_normalized_resonance:
            drive = drive * alpha[..., None, None]
        initial = mx.zeros((batch, h, n, r, 2), dtype=u.dtype)
        if length:
            positions = mx.broadcast_to(mx.arange(1, length + 1)[None], (batch, length))
            inclusive = mx.cummax(mx.where(mask, positions, 0), axis=1)
            previous_indices = mx.concatenate((mx.zeros_like(inclusive[:, :1]), inclusive[:, :-1]), axis=1)
            sources = mx.concatenate((initial[:, None], drive), axis=1)
            indices = mx.broadcast_to(previous_indices[:, :, None, None, None, None], drive.shape)
            previous = mx.take_along_axis(sources, indices, axis=1)
        else:
            previous = drive
        affine = ((_cmul(phi1 - phi2, previous) + _cmul(phi2, drive)) * delta[..., None])
        active = mask[:, :, None, None, None, None]
        identity = mx.zeros_like(transition)
        identity = mx.stack((mx.ones_like(identity[..., 0]), identity[..., 1]), axis=-1)
        transitions = mx.where(active, transition, identity)
        drives = mx.where(active, affine, mx.zeros_like(affine))
        states = self._affine_scan(
            transitions, drives, initial, work_efficient=self.training or length >= 4096
        ) if length else drives
        projected = (readout[..., 0] * states[..., 0] + readout[..., 1] * states[..., 1]).sum(axis=3)
        output = _linear(parameters, self.buffers, projected.reshape(batch, length, h * r),
                         f"{prefix}.output_projection")
        gate = mx.sigmoid(_linear(parameters, self.buffers, u, f"{prefix}.output_gate"))
        return gate * output * mask[..., None]

    @staticmethod
    def _chebyshev(x, order: int):
        mx = _mlx()
        terms = [mx.ones_like(x)]
        if order > 1:
            terms.append(x)
        for _ in range(2, order):
            terms.append(2 * x * terms[-1] - terms[-2])
        return mx.stack(terms, axis=-1)

    def _causal_alias_filter(self, parameters, x, prefix: str):
        """Depthwise five-tap causal filter without layout-dependent conv calls."""

        mx = _mlx()
        kernel = self._p(parameters, f"{prefix}.kernel")[:, 0]
        padded = mx.pad(x, ((0, 0), (4, 0), (0, 0)))
        result = mx.zeros_like(x)
        for tap in range(5):
            result = result + padded[:, tap:tap + x.shape[1]] * kernel[:, tap]
        return result

    def _anti_alias_sequence(self, parameters, x, block: int, scale: int):
        mx = _mlx()
        prefix = f"blocks.{block}.anti_aliases.{scale}"
        up = mx.stack((2 * x, mx.zeros_like(x)), axis=2).reshape(
            x.shape[0], 2 * x.shape[1], x.shape[2]
        )
        pre = self._causal_alias_filter(parameters, up, prefix)
        post = self._causal_alias_filter(parameters, pre * mx.sigmoid(pre), prefix)
        return post[:, 1::2]

    def _anti_alias_stream(self, parameters, x, state, block: int, scale: int):
        mx = _mlx()
        prefix = f"blocks.{block}.anti_aliases.{scale}"
        kernel = self._p(parameters, f"{prefix}.kernel")[:, 0]

        def filter_step(value, history):
            sequence = mx.concatenate((history, value), axis=1)
            rows = []
            for offset in range(value.shape[1]):
                window = sequence[:, offset:offset + 5]
                rows.append(mx.sum(window * mx.swapaxes(kernel, 0, 1)[None], axis=1))
            return mx.stack(rows, axis=1), sequence[:, -4:]

        up = mx.stack((2 * x, mx.zeros_like(x)), axis=1)
        pre, pre_history = filter_step(up, state[f"alias.pre.{block}.{scale}"])
        activated = pre * mx.sigmoid(pre)
        post, post_history = filter_step(
            activated, state[f"alias.post.{block}.{scale}"]
        )
        return post[:, -1], pre_history, post_history

    def _hybrid_mixer(self, parameters, x, block: int, scale: int):
        mx = _mlx()
        prefix = f"blocks.{block}.mixers.{scale}"
        conventional = f"{prefix}.conventional"
        spectral = f"{prefix}.spectral"
        if self.config.structured_mixer_rank is None:
            a, b, control, carrier, context, blend_logits = _packed_linear(
                parameters, self.buffers, x,
                (
                    f"{conventional}.a", f"{conventional}.b", f"{spectral}.control",
                    f"{spectral}.carrier", f"{spectral}.context", f"{prefix}.blend",
                ),
            )
            ordinary = _linear(
                parameters, self.buffers, mx.sigmoid(a) * a * b,
                f"{conventional}.output",
            )
        else:
            a = _structured_linear(parameters, self.buffers, x, f"{conventional}.a")
            b = _structured_linear(parameters, self.buffers, x, f"{conventional}.b")
            control, carrier, context, blend_logits = _packed_linear(
                parameters, self.buffers, x,
                (
                    f"{spectral}.control", f"{spectral}.carrier",
                    f"{spectral}.context", f"{prefix}.blend",
                ),
            )
            ordinary = _structured_linear(
                parameters, self.buffers, mx.sigmoid(a) * a * b,
                f"{conventional}.output",
            )
        config = self.config.scale_configs()[scale]
        h, n, r = config.heads, min(config.modes, self.config.spectral_modes), config.mimo_rank
        control = control.reshape(*x.shape[:-1], h, n, r, 2)
        carrier = carrier.reshape(*x.shape[:-1], h, n, r, 2)
        context = context.reshape(*x.shape[:-1], h, n, 2)
        amplitude = mx.sqrt(_cabs2(control) + 1e-6) - 1e-3
        coordinate = 2 * amplitude / (1 + amplitude) - 1
        basis = self._chebyshev(coordinate, self.config.spectral_basis_order)
        gain_response = mx.einsum("...hnrk,hnk->...hnr", basis, self._p(parameters, f"{spectral}.gain_coefficients"))
        phase_response = mx.einsum("...hnrk,hnk->...hnr", basis, self._p(parameters, f"{spectral}.phase_coefficients"))
        gain_logit = log(1 / (self.config.spectral_maximum_gain - 1))
        multiplier = self.config.spectral_maximum_gain * mx.sigmoid(
            gain_logit + gain_response + context[..., 0, None]
        )
        amplitude_gate = (2 * mx.sigmoid(amplitude) - 1) * multiplier
        phase = self.config.spectral_maximum_phase * mx.tanh(
            phase_response + context[..., 1, None]
        )
        gated = _cscale(_crotate(carrier, phase), amplitude_gate)
        triad = mx.zeros_like(gated)
        target = self.buffers[f"{spectral}.triad_target"]
        if target.size and self.config.spectral_maximum_triad_gain:
            left_index = self.buffers[f"{spectral}.triad_left"]
            right_index = self.buffers[f"{spectral}.triad_right"]
            left = mx.take(control, left_index, axis=-3)
            right = mx.take(carrier, right_index, axis=-3)
            conjugate = self.buffers[f"{spectral}.triad_conjugate"]
            right = mx.where(conjugate[None, None, None, :, None, None], _cconj(right), right)
            interaction = _cmul(left, right)
            product_amplitude = mx.sqrt(_cabs2(left) + 1e-6) - 1e-3
            product_amplitude *= mx.sqrt(_cabs2(right) + 1e-6) - 1e-3
            interaction = _cscale(interaction, mx.reciprocal(1 + product_amplitude))
            weight = self.config.spectral_maximum_triad_gain * mx.tanh(
                self._p(parameters, f"{spectral}.raw_triad_weight")
            )
            interaction = _cscale(interaction, weight[None, None, :, :, :])
            routing = mx.eye(n, dtype=x.dtype)[target]
            triad = mx.einsum("bthkrz,kn->bthnrz", interaction, routing)
        modal = gated + triad
        spectral_output = _linear(
            parameters, self.buffers, modal.reshape(*x.shape[:-1], -1), f"{spectral}.output"
        )
        blend = mx.sigmoid(blend_logits)
        return ordinary + blend * (spectral_output - ordinary)

    @staticmethod
    def _local_candidates(band, window: int):
        mx = _mlx()
        data, mask = band["data"], band["mask"]
        batch, length = data.shape[:2]
        positions = mx.arange(length)
        offsets = mx.arange(window - 1, -1, -1)
        indices = positions[:, None] - offsets[None]
        valid_index = (indices >= 0) & (indices < length)
        safe = mx.clip(indices, 0, max(0, length - 1))
        features = mx.take(data, safe, axis=1).reshape(batch * length, window, data.shape[-1])
        candidate_mask = (mx.take(mask, safe, axis=1) & valid_index[None]).reshape(batch * length, window)
        times = mx.broadcast_to((safe + 1) * band["support"] - 1, (batch, length, window)).reshape(batch * length, window)
        scales = mx.full(times.shape, float(band["scale"]), dtype=data.dtype)
        query = data.reshape(batch * length, 1, data.shape[-1])
        query_times = mx.broadcast_to((positions + 1) * band["support"] - 1, (batch, length)).reshape(batch * length, 1)
        query_scales = mx.full(query_times.shape, float(band["scale"]), dtype=data.dtype)
        return query, [features, times.astype(data.dtype), scales, candidate_mask], query_times.astype(data.dtype), query_scales

    def _landmarks(self, parameters, query_band, coarse_band, block: int, scale: int, count: int):
        mx = _mlx()
        batch, query_length = query_band["data"].shape[:2]
        query_positions = (mx.arange(query_length) + 1) * query_band["support"] - 1
        latest = (query_positions + 1) // coarse_band["support"] - 1
        indices = latest[:, None] - mx.arange(count - 1, -1, -1)[None]
        valid_index = (indices >= 0) & (indices < coarse_band["data"].shape[1])
        safe = mx.clip(indices, 0, max(0, coarse_band["data"].shape[1] - 1))
        selected = mx.take(coarse_band["data"], safe, axis=1)
        features = _linear(parameters, self.buffers, selected, f"blocks.{block}.landmark_values.{scale}")
        features = features.reshape(batch * query_length, count, -1)
        mask = (mx.take(coarse_band["mask"], safe, axis=1) & valid_index[None]).reshape(batch * query_length, count)
        times = mx.broadcast_to((safe + 1) * coarse_band["support"] - 1,
                                (batch, query_length, count)).reshape(batch * query_length, count)
        scales = mx.full(times.shape, float(coarse_band["scale"]), dtype=features.dtype)
        return [features, times.astype(features.dtype), scales, mask]

    def _attention(self, parameters, query, candidates, query_times, query_scales, block: int, scale: int):
        mx = _mlx()
        features, times, scales, mask = candidates
        prefix = f"blocks.{block}.attentions.{scale}"
        config = self.config.scale_configs()[scale]
        h, n = config.heads, min(config.modes, 16)
        delta = query_times[:, :, None] - times[:, None, :]
        scale_delta = query_scales[:, :, None] - scales[:, None, :]
        valid = mask[:, None, :] & (delta >= 0)
        q = _linear(parameters, self.buffers, query, f"{prefix}.query_projection").reshape(*query.shape[:-1], h, n, 2)
        key, values = _packed_linear(
            parameters, self.buffers, features,
            (f"{prefix}.key_projection", f"{prefix}.value_projection"),
        )
        key = key.reshape(*features.shape[:-1], h, n, 2)
        values = values.reshape(*features.shape[:-1], h, n, 2)
        cross = _cmul(q[:, :, None], _cconj(key[:, None]))
        q_norm = mx.sqrt(mx.sum(_cabs2(q), axis=-1) + 1e-6)
        k_norm = mx.sqrt(mx.sum(_cabs2(key), axis=-1) + 1e-6)
        cross = _cscale(
            cross, mx.reciprocal(q_norm[:, :, None] * k_norm[:, None])[..., None]
        )
        interval_multiplier = 2 ** (scale + 1) if scale < self.config.scales - 1 else 2**scale
        frequency_max = self.config.omega_max / interval_multiplier
        frequency = frequency_max * mx.tanh(self._p(parameters, f"{prefix}.raw_frequency"))
        aligned = _crotate(cross, -delta[..., None, None] * frequency)
        band_weight = mx.softmax(self._p(parameters, f"{prefix}.band_logits"), axis=-1)
        coherence = mx.sum(aligned[..., 0] * band_weight, axis=-1) / sqrt(n)
        amplitude = mx.sum(mx.sqrt(_cabs2(q))[:, :, None] * mx.sqrt(_cabs2(key))[:, None], axis=-1)
        score = coherence + _softplus(self._p(parameters, f"{prefix}.raw_amplitude_weight")) * mx.log(1e-6 + amplitude)
        score -= _softplus(self._p(parameters, f"{prefix}.raw_distance_decay")) * mx.log1p(mx.abs(delta))[..., None]
        score -= _softplus(self._p(parameters, f"{prefix}.raw_scale_decay")) * mx.abs(scale_delta)[..., None]
        score = mx.where(valid[..., None], score, -mx.inf)
        weights = _safe_softmax(score, 2)
        value_frequency = frequency_max * mx.tanh(self._p(parameters, f"{prefix}.raw_value_frequency"))
        aligned_values = _crotate(values[:, None], -delta[..., None, None] * value_frequency)
        aggregate = mx.sum(aligned_values * weights[..., None, None], axis=2)
        return _linear(parameters, self.buffers, aggregate.reshape(*aggregate.shape[:-3], -1),
                       f"{prefix}.output_projection")

    def _block(self, parameters, bands, block: int):
        mx = _mlx()
        bands = self._exchange(parameters, bands, block)
        output = []
        schedule = tuple(
            (scale == self.config.scales - 1)
            or (scale == 0 and block % 2 == 0)
            or (0 < scale < self.config.scales - 1 and block % 3 == 0)
            for scale in range(self.config.scales)
        )
        for scale, band in enumerate(bands):
            normalized = _rms_norm(parameters, self.buffers, band["data"], f"blocks.{block}.norms.{scale}")
            normalized *= band["mask"][..., None]
            resonant = self._resonator(parameters, normalized, band["mask"], block, scale, float(band["support"]))
            local = self._hybrid_mixer(parameters, normalized, block, scale) * band["mask"][..., None]
            if self.config.continuous_signal:
                local = self._anti_alias_sequence(parameters, local, block, scale)
                local = local * band["mask"][..., None]
            if schedule[scale] and band["data"].shape[1]:
                window = min(self.config.attention_window, max(1, band["data"].shape[1]))
                query, candidates, query_times, query_scales = self._local_candidates(band, window)
                if scale + 1 < len(bands) and bands[scale + 1]["data"].shape[1]:
                    count = min(max(1, min(8, self.config.attention_window // 4)),
                                max(1, bands[scale + 1]["data"].shape[1]))
                    landmark = self._landmarks(parameters, band, bands[scale + 1], block, scale, count)
                    candidates = [mx.concatenate((left, right), axis=1)
                                  for left, right in zip(candidates, landmark, strict=True)]
                attended = self._attention(
                    parameters, query, candidates, query_times, query_scales, block, scale
                ).reshape(band["data"].shape) * band["mask"][..., None]
            else:
                attended = mx.zeros_like(band["data"])
            identity, branch_logits = _packed_linear(
                parameters, self.buffers, normalized,
                (f"blocks.{block}.identity.{scale}", f"blocks.{block}.branch_gates.{scale}"),
            )
            branch = mx.softmax(branch_logits, axis=-1)
            delta = (branch[..., :1] * resonant + branch[..., 1:2] * local
                     + branch[..., 2:3] * attended + branch[..., 3:4] * identity)
            layer_scale = self._p(parameters, f"blocks.{block}.layer_scale")[scale]
            output.append({**band, "data": (band["data"] + layer_scale * delta) * band["mask"][..., None]})
        return output

    @staticmethod
    def _causal_expand(data, target_length: int, support: int):
        mx = _mlx()
        positions = mx.arange(target_length)
        completed = positions >= support - 1
        indices = mx.clip((mx.maximum(positions - (support - 1), 0)) // support,
                          0, max(0, data.shape[1] - 1))
        return mx.take(data, indices, axis=1) * completed[None, :, None]

    def _forward_impl(self, parameters, x, mask):
        encoded = _linear(parameters, self.buffers, x, "encoder") * mask[..., None]
        bands = self._analysis(parameters, encoded, mask)
        for block in range(self.config.layers):
            bands = self._block(parameters, bands, block)
        raw = _rms_norm(parameters, self.buffers, encoded, "raw_norm")
        if self.config.structured_mixer_rank is None:
            a, b = _packed_linear(parameters, self.buffers, raw, ("raw_mixer.a", "raw_mixer.b"))
            mixed = _linear(
                parameters, self.buffers, a * _mlx().sigmoid(a) * b,
                "raw_mixer.output",
            )
        else:
            a = _structured_linear(parameters, self.buffers, raw, "raw_mixer.a")
            b = _structured_linear(parameters, self.buffers, raw, "raw_mixer.b")
            mixed = _structured_linear(
                parameters, self.buffers, a * _mlx().sigmoid(a) * b,
                "raw_mixer.output",
            )
        fused = encoded + self._p(parameters, "raw_gain") * mixed
        gains = self._p(parameters, "scale_output_gain")
        for scale, band in enumerate(bands):
            contribution = _linear(parameters, self.buffers, band["data"], f"synthesis_adapters.{scale}")
            fused = fused + gains[scale] * self._causal_expand(
                contribution, encoded.shape[1], band["support"]
            )
        return _linear(parameters, self.buffers, fused, "output_head") * mask[..., None]

    def _loss_impl(self, parameters, x, target, mask):
        mx = _mlx()
        prediction = self._forward_impl(parameters, x, mask)
        return mx.mean((prediction - target) ** 2)

    def __call__(self, x, mask=None, *, evaluate: bool = True):
        mx = _mlx()
        if isinstance(x, torch.Tensor):
            x = _array(x)
        if mask is None:
            mask = mx.ones(x.shape[:2], dtype=mx.bool_)
        elif isinstance(mask, torch.Tensor):
            mask = _array(mask)
        output = self._forward(self.parameters, x, mask)
        if evaluate:
            mx.eval(output)
        return output

    def loss_and_grad(self, x, target, mask=None):
        """Return exact MSE and gradients for every trainable imported parameter."""

        mx = _mlx()
        x = _array(x) if isinstance(x, torch.Tensor) else x
        target = _array(target) if isinstance(target, torch.Tensor) else target
        if mask is None:
            mask = mx.ones(x.shape[:2], dtype=mx.bool_)
        elif isinstance(mask, torch.Tensor):
            mask = _array(mask)

        if self._loss_grad is not None:
            value, gradients = self._loss_grad(self.parameters, x, target, mask)
            mx.eval(value, gradients)
            return value, gradients
        packed = {
            name: value for name, value in self.buffers.items() if name.startswith("__packed__.")
        }
        for name in packed:
            del self.buffers[name]
        try:
            value, gradients = mx.value_and_grad(self._loss_impl)(self.parameters, x, target, mask)
            mx.eval(value, gradients)
            return value, gradients
        finally:
            self.buffers.update(packed)

    def apply_gradients(self, gradients, learning_rate: float) -> None:
        """Apply an in-place SGD update while preserving compiled parameter capture."""

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        for name in self.parameters:
            self.parameters[name] = self.parameters[name] - learning_rate * gradients[name]

    def benchmark_training(self, x, target, *, repeats: int = 5, warmup: int = 1) -> dict[str, float]:
        """Synchronously time differentiable MLX forward/backward execution."""

        if not self.training:
            raise RuntimeError("construct MLXMRRN(training=True) for a stable compiled training graph")
        if repeats <= 0 or warmup < 0:
            raise ValueError("benchmark repeat controls are invalid")
        for _ in range(warmup):
            self.loss_and_grad(x, target)
        values = []
        for _ in range(repeats):
            start = perf_counter()
            self.loss_and_grad(x, target)
            values.append(perf_counter() - start)
        ordered = sorted(values)
        return {
            "mean_seconds": sum(values) / len(values),
            "p95_seconds": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        }

    def benchmark(self, x, *, repeats: int = 20, warmup: int = 3) -> dict[str, float]:
        """Synchronously time compiled MLX prefill on the supplied batch."""

        if repeats <= 0 or warmup < 0:
            raise ValueError("benchmark repeat controls are invalid")
        mx = _mlx()
        x = _array(x) if isinstance(x, torch.Tensor) else x
        mask = mx.ones(x.shape[:2], dtype=mx.bool_)
        for _ in range(warmup):
            mx.eval(self._forward(self.parameters, x, mask))
        values = []
        for _ in range(repeats):
            start = perf_counter()
            mx.eval(self._forward(self.parameters, x, mask))
            values.append(perf_counter() - start)
        ordered = sorted(values)
        return {
            "mean_seconds": sum(values) / len(values),
            "p95_seconds": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        }


class MLXCausalTransformer:
    """MLX mirror of the matched PyTorch Transformer used by the benchmark suite."""

    def __init__(
        self, model: "CausalTransformerBaseline", *, compile: bool = True, training: bool = False
    ) -> None:
        mx = _mlx()
        self.width, self.heads, self.layers = model.width, model.heads, model.layers
        self.parameters = {name: _array(value) for name, value in model.state_dict().items()}
        self.buffers: dict[str, Any] = {}
        self.training = training
        if not training:
            for layer in range(self.layers):
                _cache_packed(
                    self.parameters, self.buffers, (f"mixers.{layer}.a", f"mixers.{layer}.b")
                )
        self._forward = (
            mx.compile(self._forward_impl, inputs=[self.parameters, self.buffers])
            if compile else self._forward_impl
        )
        self._loss_grad = (
            mx.compile(mx.value_and_grad(self._loss_impl), inputs=self.buffers)
            if compile and training else None
        )
        self._compile_streaming = compile
        self._stream_function = None

    @property
    def parameter_count(self) -> int:
        return sum(array.size for array in self.parameters.values())

    def _forward_impl(self, parameters, x):
        mx = _mlx()
        hidden = _linear(parameters, {}, x, "encoder")
        length, head_width = x.shape[1], self.width // self.heads
        for layer in range(self.layers):
            normalized = _rms_norm(parameters, {}, hidden, f"norm1.{layer}")
            qkv = _linear(parameters, {}, normalized, f"qkv.{layer}")
            q, k, v = mx.split(qkv, 3, axis=-1)
            q = mx.transpose(q.reshape(x.shape[0], length, self.heads, head_width), (0, 2, 1, 3))
            k = mx.transpose(k.reshape(x.shape[0], length, self.heads, head_width), (0, 2, 1, 3))
            v = mx.transpose(v.reshape(x.shape[0], length, self.heads, head_width), (0, 2, 1, 3))
            attended = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=1 / sqrt(head_width), mask="causal"
            )
            attended = mx.transpose(attended, (0, 2, 1, 3)).reshape(x.shape[0], length, self.width)
            hidden = hidden + _linear(parameters, {}, attended, f"attention_out.{layer}")
            normalized = _rms_norm(parameters, {}, hidden, f"norm2.{layer}")
            a, b = _packed_linear(
                parameters, self.buffers, normalized, (f"mixers.{layer}.a", f"mixers.{layer}.b")
            )
            mixed = _linear(parameters, {}, a * mx.sigmoid(a) * b, f"mixers.{layer}.output")
            hidden = hidden + mixed
        return _linear(parameters, {}, hidden, "head")

    def _loss_impl(self, parameters, x, target):
        mx = _mlx()
        return mx.mean((self._forward_impl(parameters, x) - target) ** 2)

    def __call__(self, x, *, evaluate: bool = True):
        mx = _mlx()
        x = _array(x) if isinstance(x, torch.Tensor) else x
        output = self._forward(self.parameters, x)
        if evaluate:
            mx.eval(output)
        return output

    def initial_stream_state(self, batch: int, capacity: int = 1) -> MLXTransformerStreamState:
        if min(batch, capacity) <= 0:
            raise ValueError("batch and KV capacity must be positive")
        mx = _mlx()
        shape = (batch, self.heads, capacity, self.width // self.heads)
        keys = tuple(mx.zeros(shape, dtype=mx.float32) for _ in range(self.layers))
        values = tuple(mx.zeros(shape, dtype=mx.float32) for _ in range(self.layers))
        return MLXTransformerStreamState(keys, values, 0, batch, capacity)

    def _stream_step_impl(self, keys, values, x, slot, valid):
        mx = _mlx()
        hidden = _linear(self.parameters, {}, x, "encoder")[:, None]
        head_width = self.width // self.heads
        next_keys, next_values = [], []
        for layer in range(self.layers):
            normalized = _rms_norm(self.parameters, {}, hidden, f"norm1.{layer}")
            qkv = _linear(self.parameters, {}, normalized, f"qkv.{layer}")
            q, k, v = mx.split(qkv, 3, axis=-1)
            q = mx.transpose(q.reshape(x.shape[0], 1, self.heads, head_width), (0, 2, 1, 3))
            k = mx.transpose(k.reshape(x.shape[0], 1, self.heads, head_width), (0, 2, 1, 3))
            v = mx.transpose(v.reshape(x.shape[0], 1, self.heads, head_width), (0, 2, 1, 3))
            key = keys[layer] + k * slot[None, None, :, None]
            value = values[layer] + v * slot[None, None, :, None]
            attended = mx.fast.scaled_dot_product_attention(
                q, key, value, scale=1 / sqrt(head_width), mask=valid[None, None, None]
            )
            attended = mx.transpose(attended, (0, 2, 1, 3)).reshape(x.shape[0], 1, self.width)
            hidden = hidden + _linear(self.parameters, {}, attended, f"attention_out.{layer}")
            normalized = _rms_norm(self.parameters, {}, hidden, f"norm2.{layer}")
            packed = normalized @ mx.swapaxes(
                self.buffers[f"__packed__.mixers.{layer}.a|mixers.{layer}.b.weight"], -1, -2
            ) + self.buffers[f"__packed__.mixers.{layer}.a|mixers.{layer}.b.bias"]
            hidden_size = self.parameters[f"mixers.{layer}.a.weight"].shape[0]
            a, b = packed[..., :hidden_size], packed[..., hidden_size:]
            hidden = hidden + _linear(
                self.parameters, {}, a * mx.sigmoid(a) * b, f"mixers.{layer}.output"
            )
            next_keys.append(key)
            next_values.append(value)
        return _linear(self.parameters, {}, hidden[:, 0], "head"), tuple(next_keys), tuple(next_values)

    def stream_step(self, x, state: MLXTransformerStreamState):
        mx = _mlx()
        x = _array(x) if isinstance(x, torch.Tensor) else x
        if x.shape != (state.batch, self.parameters["encoder.weight"].shape[1]):
            raise ValueError("stream input shape does not match the Transformer state")
        if state.position >= state.capacity:
            raise ValueError("Transformer KV cache capacity exceeded")
        slot = (mx.arange(state.capacity) == state.position).astype(x.dtype)
        valid = mx.arange(state.capacity) <= state.position
        if self._stream_function is None:
            self._stream_function = self._stream_step_impl
            if self._compile_streaming:
                self._stream_function = mx.compile(
                    self._stream_function, inputs=[self.parameters, self.buffers]
                )
        output, keys, values = self._stream_function(state.keys, state.values, x, slot, valid)
        mx.eval(output, keys, values)
        return output, MLXTransformerStreamState(
            keys, values, state.position + 1, state.batch, state.capacity
        )

    def decode(self, x, state: MLXTransformerStreamState | None = None):
        mx = _mlx()
        x = _array(x) if isinstance(x, torch.Tensor) else x
        state = self.initial_stream_state(x.shape[0], max(1, x.shape[1])) if state is None else state
        output = []
        for position in range(x.shape[1]):
            value, state = self.stream_step(x[:, position], state)
            output.append(value[:, None])
        return mx.concatenate(output, axis=1) if output else mx.zeros(
            (x.shape[0], 0, self.parameters["head.weight"].shape[0]), dtype=x.dtype
        ), state

    def benchmark_decode(
        self, *, context: int, repeats: int = 32, warmup: int = 2
    ) -> dict[str, float]:
        """Measure cached single-token latency at an explicit context length."""

        if context < 0 or repeats <= 0 or warmup < 0:
            raise ValueError("decode benchmark controls are invalid")
        mx = _mlx()
        state = self.initial_stream_state(1, max(1, context + warmup + repeats))
        state.position = context
        x = mx.zeros((1, self.parameters["encoder.weight"].shape[1]), dtype=mx.float32)
        for _ in range(warmup):
            _, state = self.stream_step(x, state)
        values = []
        for _ in range(repeats):
            start = perf_counter()
            _, state = self.stream_step(x, state)
            values.append(perf_counter() - start)
        ordered = sorted(values)
        return {
            "mean_seconds": sum(values) / len(values),
            "p95_seconds": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        }

    def loss_and_grad(self, x, target):
        mx = _mlx()
        x = _array(x) if isinstance(x, torch.Tensor) else x
        target = _array(target) if isinstance(target, torch.Tensor) else target
        function = self._loss_grad or mx.value_and_grad(self._loss_impl)
        value, gradients = function(self.parameters, x, target)
        mx.eval(value, gradients)
        return value, gradients

    def benchmark_training(self, x, target, *, repeats: int = 5, warmup: int = 1) -> dict[str, float]:
        if not self.training:
            raise RuntimeError("construct MLXCausalTransformer(training=True) for training benchmarks")
        if repeats <= 0 or warmup < 0:
            raise ValueError("benchmark repeat controls are invalid")
        for _ in range(warmup):
            self.loss_and_grad(x, target)
        values = []
        for _ in range(repeats):
            start = perf_counter()
            self.loss_and_grad(x, target)
            values.append(perf_counter() - start)
        ordered = sorted(values)
        return {
            "mean_seconds": sum(values) / len(values),
            "p95_seconds": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        }

    def benchmark(self, x, *, repeats: int = 20, warmup: int = 3) -> dict[str, float]:
        if repeats <= 0 or warmup < 0:
            raise ValueError("benchmark repeat controls are invalid")
        mx = _mlx()
        x = _array(x) if isinstance(x, torch.Tensor) else x
        for _ in range(warmup):
            mx.eval(self._forward(self.parameters, x))
        values = []
        for _ in range(repeats):
            start = perf_counter()
            mx.eval(self._forward(self.parameters, x))
            values.append(perf_counter() - start)
        ordered = sorted(values)
        return {
            "mean_seconds": sum(values) / len(values),
            "p95_seconds": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        }
