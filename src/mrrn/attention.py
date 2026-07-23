"""Exact candidate-set resonant coherence attention and lag routing."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import complex_ops as c
from .packed_projection import PackedCache, packed_linear


@dataclass(frozen=True, slots=True)
class AttentionCandidates:
    features: Tensor
    times: Tensor
    scales: Tensor
    mask: Tensor
    kinds: Tensor | None = None

    def validate(self, width: int) -> None:
        if self.features.ndim != 3 or self.features.shape[-1] != width:
            raise ValueError(f"candidate features must have shape (batch, count, {width})")
        expected = self.features.shape[:2]
        if self.times.shape != expected or self.scales.shape != expected or self.mask.shape != expected:
            raise ValueError("candidate metadata must have shape (batch, count)")
        if self.mask.dtype != torch.bool:
            raise ValueError("candidate mask must be boolean")
        if self.kinds is not None and self.kinds.shape != expected:
            raise ValueError("candidate kinds must have shape (batch, count)")


def linear_cross_correlation(query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
    """All non-circular lags for real signals, ordered -(K-1) through Q-1."""

    if query.ndim < 1 or key.ndim != query.ndim or query.shape[:-1] != key.shape[:-1]:
        raise ValueError("query and key must share leading dimensions")
    q_length, k_length = query.shape[-1], key.shape[-1]
    if q_length == 0 or k_length == 0:
        raise ValueError("correlation inputs cannot be empty")
    fft_length = q_length + k_length - 1
    spectrum = torch.fft.rfft(query, fft_length) * torch.fft.rfft(key, fft_length).conj()
    circular = torch.fft.irfft(spectrum, fft_length)
    correlation = torch.cat((circular[..., -(k_length - 1) :], circular[..., :q_length]), -1) if k_length > 1 else circular
    lags = torch.arange(-(k_length - 1), q_length, device=query.device)
    return correlation, lags


class ResonantAttention(nn.Module):
    """Amplitude/phase/delay-aware exact attention over an explicit candidate set."""

    def __init__(
        self, width: int, heads: int, bands: int, *, max_scale: int = 16,
        frequency_max: float = pi, eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(width, heads, bands, max_scale) <= 0 or min(frequency_max, eps) <= 0 or frequency_max > pi:
            raise ValueError("attention dimensions and epsilon must be positive")
        self.width, self.heads, self.bands, self.max_scale, self.frequency_max, self.eps = width, heads, bands, max_scale, frequency_max, eps
        projection_size = 2 * heads * bands
        self.query_projection = nn.Linear(width, projection_size)
        self.key_projection = nn.Linear(width, projection_size)
        self.value_projection = nn.Linear(width, projection_size)
        self.output_projection = nn.Linear(projection_size, width)
        self._packed_projection_cache: PackedCache = {}
        self.band_logits = nn.Parameter(torch.zeros(heads, bands))
        self.raw_amplitude_weight = nn.Parameter(torch.zeros(heads))
        self.raw_distance_decay = nn.Parameter(torch.zeros(heads))
        self.raw_scale_decay = nn.Parameter(torch.zeros(heads))
        frequencies = torch.linspace(0, 0.9 * frequency_max, bands).repeat(heads, 1)
        self.raw_frequency = nn.Parameter(torch.atanh((frequencies / frequency_max).clamp(-0.999, 0.999)))
        self.raw_value_frequency = nn.Parameter(self.raw_frequency.detach().clone())

    def _project(self, layer: nn.Linear, x: Tensor) -> Tensor:
        return layer(x).unflatten(-1, (self.heads, self.bands, 2))

    def _metadata(
        self, query: Tensor, candidates: AttentionCandidates, query_times: Tensor,
        query_scales: Tensor, causal: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        candidates.validate(self.width)
        expected = query.shape[:2]
        if query.ndim != 3 or query.shape[-1] != self.width:
            raise ValueError(f"query must have shape (batch, count, {self.width})")
        if query_times.shape != expected or query_scales.shape != expected:
            raise ValueError("query metadata must have shape (batch, query_count)")
        if query.shape[0] != candidates.features.shape[0]:
            raise ValueError("query and candidates must share batch size")
        delta = query_times.unsqueeze(2) - candidates.times.unsqueeze(1)
        scale_delta = query_scales.unsqueeze(2) - candidates.scales.unsqueeze(1)
        valid = candidates.mask.unsqueeze(1).expand(-1, query.shape[1], -1)
        if causal:
            valid = valid & (delta >= 0)
        return delta, scale_delta, valid

    def _projected_scores(
        self, query: Tensor, key: Tensor, delta: Tensor, scale_delta: Tensor, valid: Tensor
    ) -> Tensor:
        cross = c.multiply(query.unsqueeze(2), c.conjugate(key).unsqueeze(1))
        query_norm = (c.abs_squared(query).sum(-1) + self.eps).sqrt()
        key_norm = (c.abs_squared(key).sum(-1) + self.eps).sqrt()
        denominator = query_norm.unsqueeze(2) * key_norm.unsqueeze(1)
        cross = c.scale(cross, denominator.reciprocal().unsqueeze(-1))
        frequency = self.frequency_max * torch.tanh(self.raw_frequency)
        aligned = c.rotate(cross, -delta[..., None, None] * frequency)
        band_weight = torch.softmax(self.band_logits, -1)
        coherence = (c.real(aligned) * band_weight).sum(-1) / sqrt(self.bands)
        amplitude = (
            c.magnitude(query).unsqueeze(2) * c.magnitude(key).unsqueeze(1)
        ).sum(-1)
        score = coherence + F.softplus(self.raw_amplitude_weight) * torch.log(self.eps + amplitude)
        score = score - F.softplus(self.raw_distance_decay) * torch.log1p(delta.abs()).unsqueeze(-1)
        score = score - F.softplus(self.raw_scale_decay) * scale_delta.abs().unsqueeze(-1)
        return score.masked_fill(~valid.unsqueeze(-1), -torch.inf)

    def scores(
        self, query: Tensor, candidates: AttentionCandidates, query_times: Tensor,
        query_scales: Tensor, *, causal: bool = True
    ) -> Tensor:
        delta, scale_delta, valid = self._metadata(
            query, candidates, query_times, query_scales, causal
        )
        return self._projected_scores(
            self._project(self.query_projection, query),
            self._project(self.key_projection, candidates.features),
            delta,
            scale_delta,
            valid,
        )

    def _aligned_values(self, values: Tensor, delta: Tensor) -> Tensor:
        frequency = self.frequency_max * torch.tanh(self.raw_value_frequency)
        return c.rotate(values.unsqueeze(1), -delta[..., None, None] * frequency)

    @staticmethod
    def _safe_softmax(scores: Tensor, dim: int) -> Tensor:
        valid = torch.isfinite(scores)
        safe = torch.where(valid, scores, torch.full_like(scores, -torch.finfo(scores.dtype).max))
        weights = torch.softmax(safe, dim=dim) * valid
        return weights / weights.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(scores.dtype).tiny)

    def attend(
        self, query: Tensor, candidates: AttentionCandidates, query_times: Tensor,
        query_scales: Tensor, *, causal: bool = True, bandwise: bool = False
    ) -> tuple[Tensor, Tensor]:
        delta, scale_delta, valid = self._metadata(
            query, candidates, query_times, query_scales, causal
        )
        query_projected = self._project(self.query_projection, query)
        key_values = packed_linear(
            candidates.features,
            (self.key_projection, self.value_projection),
            self._packed_projection_cache, "key_value",
        )
        key_flat, value_flat = key_values.split(
            (self.key_projection.out_features, self.value_projection.out_features), -1
        )
        key = key_flat.unflatten(-1, (self.heads, self.bands, 2))
        values = value_flat.unflatten(-1, (self.heads, self.bands, 2))
        return self._attend_projected(
            query_projected, key, values, delta, scale_delta, valid,
            bandwise=bandwise,
        )

    def _attend_projected(
        self, query: Tensor, key: Tensor, values: Tensor, delta: Tensor,
        scale_delta: Tensor, valid: Tensor, *, bandwise: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Finish exact attention from reusable projected Q/K/V tensors."""

        scores = self._projected_scores(query, key, delta, scale_delta, valid)
        aligned_values = self._aligned_values(values, delta)
        if bandwise:
            cross = c.real(
                c.rotate(
                    c.multiply(query.unsqueeze(2), c.conjugate(key).unsqueeze(1)),
                    -delta[..., None, None] * (self.frequency_max * torch.tanh(self.raw_frequency)),
                )
            )
            band_scores = cross.masked_fill(~valid[..., None, None], -torch.inf)
            weights = self._safe_softmax(band_scores, dim=2)
            aggregated = (aligned_values * weights.unsqueeze(-1)).sum(2)
        else:
            weights = self._safe_softmax(scores, dim=2)
            aggregated = (aligned_values * weights.unsqueeze(-1).unsqueeze(-1)).sum(2)
        return self.output_projection(aggregated.flatten(-3)), weights

    def tiled_attend(
        self, query: Tensor, candidates: AttentionCandidates, query_times: Tensor,
        query_scales: Tensor, *, tile_size: int, causal: bool = True
    ) -> Tensor:
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        candidates.validate(self.width)
        batch, queries = query.shape[:2]
        accumulator = query.new_zeros(batch, queries, self.heads, self.bands, 2)
        normalizer = query.new_zeros(batch, queries, self.heads)
        maximum = query.new_full((batch, queries, self.heads), -torch.inf)
        query_projected = self._project(self.query_projection, query)
        for start in range(0, candidates.features.shape[1], tile_size):
            stop = min(start + tile_size, candidates.features.shape[1])
            tile = AttentionCandidates(
                candidates.features[:, start:stop], candidates.times[:, start:stop],
                candidates.scales[:, start:stop], candidates.mask[:, start:stop],
                None if candidates.kinds is None else candidates.kinds[:, start:stop],
            )
            delta, scale_delta, valid = self._metadata(query, tile, query_times, query_scales, causal)
            key_values = packed_linear(
                tile.features,
                (self.key_projection, self.value_projection),
                self._packed_projection_cache, "key_value",
            )
            key_flat, value_flat = key_values.split(
                (self.key_projection.out_features, self.value_projection.out_features), -1
            )
            key = key_flat.unflatten(-1, (self.heads, self.bands, 2))
            scores = self._projected_scores(query_projected, key, delta, scale_delta, valid)
            values = self._aligned_values(
                value_flat.unflatten(-1, (self.heads, self.bands, 2)), delta
            )
            tile_maximum = scores.max(2).values
            tile_has_values = torch.isfinite(tile_maximum)
            safe_tile_maximum = torch.where(tile_has_values, tile_maximum, torch.zeros_like(tile_maximum))
            exponentials = torch.where(
                torch.isfinite(scores), torch.exp(scores - safe_tile_maximum.unsqueeze(2)), torch.zeros_like(scores)
            )
            tile_normalizer = exponentials.sum(2)
            tile_accumulator = (values * exponentials.unsqueeze(-1).unsqueeze(-1)).sum(2)
            new_maximum = torch.maximum(maximum, tile_maximum)
            old_scale = torch.where(torch.isfinite(maximum), torch.exp(maximum - new_maximum), torch.zeros_like(maximum))
            tile_scale = torch.where(tile_has_values, torch.exp(tile_maximum - new_maximum), torch.zeros_like(tile_maximum))
            accumulator = accumulator * old_scale[..., None, None] + tile_accumulator * tile_scale[..., None, None]
            normalizer = normalizer * old_scale + tile_normalizer * tile_scale
            maximum = new_maximum
        normalized = accumulator / normalizer.clamp_min(torch.finfo(query.dtype).tiny)[..., None, None]
        return self.output_projection(normalized.flatten(-3))

    def sliding_window_attend(
        self, features: Tensor, mask: Tensor, *, window: int,
        query_tile_size: int, scale: int, sample_interval: float,
        coefficient_interval: float, causal: bool = True,
        additional_candidates: AttentionCandidates | None = None,
        history: AttentionCandidates | None = None,
        time_offset: float = 0.0,
        return_weights: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        """Exact local attention without a sequence-wide ``[T,w,d]`` tensor.

        Query tiles are only an execution partition.  Every query sees the same
        exact window and optional landmark/memory candidates as the materialized
        implementation, so tiling changes peak memory rather than semantics.
        """

        if features.ndim != 3 or features.shape[-1] != self.width:
            raise ValueError("sliding attention features must be (batch,time,width)")
        if mask.shape != features.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("sliding attention mask must be boolean with batch/time shape")
        if min(window, query_tile_size) <= 0 or min(sample_interval, coefficient_interval) <= 0:
            raise ValueError("sliding attention execution controls must be positive")
        if time_offset < 0:
            raise ValueError("sliding attention time offset cannot be negative")
        batch, length = features.shape[:2]
        if additional_candidates is not None:
            additional_candidates.validate(self.width)
            if additional_candidates.features.shape[0] != batch * length:
                raise ValueError("additional candidates must have one row per sequence query")
        current_times = (
            time_offset
            + (torch.arange(length, device=features.device, dtype=features.dtype) + 1)
            * coefficient_interval
            - sample_interval
        ).expand(batch, -1)
        current_scales = torch.full_like(current_times, float(scale))
        if history is not None:
            history.validate(self.width)
            if not causal:
                raise ValueError("attention history is defined only for causal execution")
            if history.features.shape[0] != batch:
                raise ValueError("attention history must have one row per batch item")
            history_length = history.features.shape[1]
            local_features = torch.cat((history.features, features), 1)
            local_times_all = torch.cat((history.times, current_times), 1)
            local_scales_all = torch.cat((history.scales, current_scales), 1)
            local_mask_all = torch.cat((history.mask, mask), 1)
        else:
            history_length = 0
            local_features = features
            local_times_all = current_times
            local_scales_all = current_scales
            local_mask_all = mask
        total_local_length = local_features.shape[1]
        # Local tokens occur in as many as ``window`` overlapping candidate
        # sets.  Project the scale once, then gather projected K/V windows.
        # This is algebraically identical to projecting every gathered window
        # and removes the dominant redundant linear work.
        query_projected = self._project(self.query_projection, features)
        local_key_values = packed_linear(
            local_features,
            (self.key_projection, self.value_projection),
            self._packed_projection_cache, "key_value",
        )
        local_key_flat, local_value_flat = local_key_values.split(
            (self.key_projection.out_features, self.value_projection.out_features), -1
        )
        local_keys = local_key_flat.unflatten(-1, (self.heads, self.bands, 2))
        local_values = local_value_flat.unflatten(-1, (self.heads, self.bands, 2))
        extra_keys = extra_values = None
        if additional_candidates is not None:
            extra_key_values = packed_linear(
                additional_candidates.features,
                (self.key_projection, self.value_projection),
                self._packed_projection_cache, "key_value",
            )
            extra_key_flat, extra_value_flat = extra_key_values.split(
                (self.key_projection.out_features, self.value_projection.out_features), -1
            )
            extra_keys = extra_key_flat.unflatten(-1, (self.heads, self.bands, 2))
            extra_values = extra_value_flat.unflatten(-1, (self.heads, self.bands, 2))
        output_tiles, weight_tiles = [], []
        for start in range(0, length, query_tile_size):
            stop = min(start + query_tile_size, length)
            positions = torch.arange(
                history_length + start, history_length + stop,
                device=features.device,
            )
            if causal:
                offsets = torch.arange(window - 1, -1, -1, device=features.device)
                indices = positions[:, None] - offsets[None]
            else:
                offsets = torch.arange(window, device=features.device) - window // 2
                indices = positions[:, None] + offsets[None]
            valid_index = (indices >= 0) & (indices < total_local_length)
            safe = indices.clamp(0, max(0, total_local_length - 1))
            tile_length = stop - start
            local_mask = (
                local_mask_all[:, safe] & valid_index
            ).reshape(batch * tile_length, window)
            local_times = local_times_all[:, safe].reshape(
                batch * tile_length, window
            )
            local_scales = local_scales_all[:, safe].reshape(
                batch * tile_length, window
            )
            keys = local_keys[:, safe].reshape(
                batch * tile_length, window, self.heads, self.bands, 2
            )
            values = local_values[:, safe].reshape(
                batch * tile_length, window, self.heads, self.bands, 2
            )
            if additional_candidates is not None:
                count = additional_candidates.features.shape[1]

                def select(value: Tensor) -> Tensor:
                    tail = value.shape[2:]
                    return value.reshape(batch, length, count, *tail)[:, start:stop].reshape(
                        batch * tile_length, count, *tail
                    )

                if extra_keys is None or extra_values is None:
                    raise RuntimeError("additional candidate projections were not prepared")
                keys = torch.cat((keys, select(extra_keys)), 1)
                values = torch.cat((values, select(extra_values)), 1)
                local_times = torch.cat(
                    (local_times, select(additional_candidates.times)), 1
                )
                local_scales = torch.cat(
                    (local_scales, select(additional_candidates.scales)), 1
                )
                local_mask = torch.cat(
                    (local_mask, select(additional_candidates.mask)), 1
                )
            query = query_projected[:, start:stop].reshape(
                batch * tile_length, 1, self.heads, self.bands, 2
            )
            query_times = current_times[:, start:stop].reshape(
                batch * tile_length, 1
            )
            query_scales = torch.full_like(query_times, float(scale))
            delta = query_times.unsqueeze(2) - local_times.unsqueeze(1)
            scale_delta = query_scales.unsqueeze(2) - local_scales.unsqueeze(1)
            valid = local_mask.unsqueeze(1)
            if causal:
                valid = valid & (delta >= 0)
            attended, weights = self._attend_projected(
                query, keys, values, delta, scale_delta, valid,
            )
            output_tiles.append(attended.reshape(batch, tile_length, self.width))
            if return_weights:
                weight_tiles.append(
                    weights.reshape(
                        batch, tile_length, 1, weights.shape[2], weights.shape[3]
                    )
                )
        if not output_tiles:
            weights = (
                features.new_zeros(batch * 0, 1, window, self.heads)
                if return_weights else None
            )
            return features[:, :0], weights
        output = torch.cat(output_tiles, 1)
        if not return_weights:
            return output, None
        weights = torch.cat(weight_tiles, 1).reshape(
            batch * length, 1, weight_tiles[0].shape[-2], self.heads
        )
        return output, weights


class DotProductCandidateAttention(nn.Module):
    """Ordinary exact dot-product attention on the identical bounded candidate contract."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads:
            raise ValueError("width must be positive and divisible by positive heads")
        self.width, self.heads, self.head_width = width, heads, width // heads
        self.query, self.key, self.value, self.output = (
            nn.Linear(width, width) for _ in range(4)
        )

    def _split(self, value: Tensor) -> Tensor:
        return value.unflatten(-1, (self.heads, self.head_width))

    def attend(
        self, query: Tensor, candidates: AttentionCandidates, query_times: Tensor,
        query_scales: Tensor, *, causal: bool = True,
    ) -> tuple[Tensor, Tensor]:
        candidates.validate(self.width)
        if query.ndim != 3 or query.shape[-1] != self.width or query_times.shape != query.shape[:2] or query_scales.shape != query.shape[:2]:
            raise ValueError("dot-product query and metadata contracts are invalid")
        delta = query_times.unsqueeze(2) - candidates.times.unsqueeze(1)
        valid = candidates.mask.unsqueeze(1) & ((delta >= 0) if causal else torch.ones_like(delta, dtype=torch.bool))
        scores = torch.einsum(
            "bqhd,bkhd->bqkh", self._split(self.query(query)), self._split(self.key(candidates.features))
        ) / sqrt(self.head_width)
        weights = ResonantAttention._safe_softmax(scores.masked_fill(~valid.unsqueeze(-1), -torch.inf), 2)
        aggregated = torch.einsum("bqkh,bkhd->bqhd", weights, self._split(self.value(candidates.features)))
        return self.output(aggregated.flatten(-2)), weights
