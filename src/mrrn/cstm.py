"""Causal Spectral Target Multiplexing for multiresolution language training.

CSTM does not manufacture or recount corpus tokens.  It turns one causal
carrier state into several additional prediction obligations by asking each
emitted MRRN band coefficient to forecast a strictly future block of token
codes.  Targets retain the block's DC component and first order-sensitive
Fourier harmonic.  This gives the carrier and the bounded cognitive residual a
cheap, scale-native learning signal without another vocabulary projection or
another carrier forward pass.

The target side is deliberately fixed and non-trainable.  A deterministic
Rademacher code maps vocabulary identities into a compact space; targets are
stopped before loss construction.  Packed-document boundaries and incomplete
future blocks fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class CSTMArchitectureConfig:
    """Shape and deterministic-identity contract of the CSTM prediction head."""

    code_dimension: int = 64
    predictor_rank: int = 8
    horizon_blocks: tuple[int, ...] = (1, 2, 4, 8)
    token_code_seed: int = 20260725
    target_rms_decay: float = 0.99
    minimum_target_rms: float = 1e-4
    huber_delta: float = 1.0

    def __post_init__(self) -> None:
        if self.code_dimension < 8:
            raise ValueError("CSTM code dimension must be at least eight")
        if self.predictor_rank <= 0:
            raise ValueError("CSTM predictor rank must be positive")
        if (
            not self.horizon_blocks
            or self.horizon_blocks[0] != 1
            or any(value <= 0 for value in self.horizon_blocks)
            or tuple(sorted(set(self.horizon_blocks))) != self.horizon_blocks
        ):
            raise ValueError(
                "CSTM horizons must be unique, increasing positive blocks beginning with one"
            )
        if not 0 <= self.target_rms_decay < 1:
            raise ValueError("CSTM target RMS decay must lie in [0,1)")
        if self.minimum_target_rms <= 0 or self.huber_delta <= 0:
            raise ValueError("CSTM numerical scales must be positive")


@dataclass(frozen=True, slots=True)
class CSTMTargetBatch:
    """One scale's fixed spectral targets aligned to causal source states."""

    values: Tensor
    mask: Tensor
    horizons: tuple[int, ...]
    support: int
    source_positions: Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 5 or self.values.shape[3] != 3:
            raise ValueError(
                "CSTM target values must have shape "
                "(batch,rows,horizons,3,code_dimension)"
            )
        if self.mask.shape != self.values.shape[:3] or self.mask.dtype != torch.bool:
            raise ValueError(
                "CSTM target mask must be boolean with shape (batch,rows,horizons)"
            )
        if len(self.horizons) != self.values.shape[2]:
            raise ValueError("CSTM horizon metadata does not match the target tensor")
        if self.source_positions.shape != self.values.shape[1:2]:
            raise ValueError("CSTM source positions must name every target row")
        if self.source_positions.dtype != torch.int64 or self.support < 2:
            raise ValueError("CSTM source positions and support are invalid")

    @property
    def valid_rows(self) -> int:
        return int(self.mask.sum())

    @property
    def token_participations(self) -> int:
        return self.valid_rows * self.support


@dataclass(frozen=True, slots=True)
class CSTMPredictionBatch:
    """One scale's predictions and the exact source positions that produced them."""

    values: Tensor
    source_positions: Tensor
    horizons: tuple[int, ...]
    support: int
    scale: int
    kind: str
    row_inclusion_probability: float = 1.0
    row_sampling_digest: str | None = None

    def __post_init__(self) -> None:
        if self.values.ndim != 5 or self.values.shape[0] <= 0:
            raise ValueError(
                "CSTM predictions must have shape "
                "(batch,rows,horizons,3,code_dimension)"
            )
        if self.source_positions.shape != self.values.shape[1:2]:
            raise ValueError("CSTM prediction source positions must name every row")
        if self.source_positions.dtype != torch.int64:
            raise ValueError("CSTM prediction source positions must be int64")
        if len(self.horizons) != self.values.shape[2] or self.support < 2:
            raise ValueError("CSTM prediction horizon or support metadata is invalid")
        if self.scale < 0 or self.kind not in {"detail", "approximation"}:
            raise ValueError("CSTM prediction band metadata is invalid")
        if (
            not 0 < self.row_inclusion_probability <= 1
            or (
                self.row_sampling_digest is not None
                and len(self.row_sampling_digest) != 64
            )
        ):
            raise ValueError("CSTM prediction row-sampling metadata is invalid")


@dataclass(frozen=True, slots=True)
class CSTMLoss:
    """Differentiable loss plus honest supervision-accounting receipts."""

    loss: Tensor
    standardized_huber_sum: Tensor
    weighted_rows: Tensor
    valid_rows: int
    coefficient_targets: int
    token_participations: int
    per_horizon_rows: tuple[int, ...]


def deterministic_token_codes(
    vocabulary_size: int,
    code_dimension: int,
    *,
    seed: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return a fixed normalized Rademacher codebook without global RNG mutation."""

    if vocabulary_size < 2 or code_dimension < 8:
        raise ValueError("CSTM codebook dimensions are invalid")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    bits = torch.randint(
        0,
        2,
        (vocabulary_size, code_dimension),
        generator=generator,
        dtype=torch.int8,
    )
    codes = bits.to(torch.float32).mul_(2).sub_(1).div_(code_dimension**0.5)
    return codes.to(device=device, dtype=dtype)


def _validate_target_inputs(
    labels: Tensor,
    loss_mask: Tensor,
    segment_ids: Tensor,
    target_segment_ids: Tensor,
    token_codes: Tensor,
    source_positions: Tensor,
) -> None:
    if labels.ndim != 2 or labels.shape[0] <= 0 or labels.dtype != torch.int64:
        raise ValueError("CSTM labels must be int64 with shape (batch,time)")
    for name, value in (
        ("loss_mask", loss_mask),
        ("segment_ids", segment_ids),
        ("target_segment_ids", target_segment_ids),
    ):
        if value.shape != labels.shape:
            raise ValueError(f"CSTM {name} must match labels")
    if loss_mask.dtype != torch.bool:
        raise ValueError("CSTM loss mask must be boolean")
    if segment_ids.dtype != torch.int64 or target_segment_ids.dtype != torch.int64:
        raise ValueError("CSTM segment identifiers must be int64")
    if token_codes.ndim != 2 or labels.numel() and (
        int(labels.min()) < 0 or int(labels.max()) >= token_codes.shape[0]
    ):
        raise ValueError("CSTM token codebook does not cover the supplied labels")
    if source_positions.ndim != 1 or source_positions.dtype != torch.int64:
        raise ValueError("CSTM source positions must be a one-dimensional int64 tensor")
    if source_positions.numel() and (
        int(source_positions.min()) < 0 or int(source_positions.max()) >= labels.shape[1]
    ):
        raise ValueError("CSTM source positions fall outside the packed context")


def causal_spectral_target_mask(
    loss_mask: Tensor,
    segment_ids: Tensor,
    target_segment_ids: Tensor,
    source_positions: Tensor,
    *,
    support: int,
    horizons: Sequence[int],
) -> Tensor:
    """Return the authoritative complete-block validity mask.

    This performs no token-code lookup or Fourier work. The trainer uses it to
    determine the exact context-wide loss denominator before any TBPTT group is
    released, while target construction uses the same authority to keep
    accounting and mathematics identical.
    """

    horizons = tuple(int(value) for value in horizons)
    if support < 2 or not horizons or any(value <= 0 for value in horizons):
        raise ValueError("CSTM support and horizons must be positive")
    if len(set(horizons)) != len(horizons):
        raise ValueError("CSTM horizons must be unique")
    if (
        loss_mask.ndim != 2
        or loss_mask.shape[0] <= 0
        or loss_mask.dtype != torch.bool
        or segment_ids.shape != loss_mask.shape
        or target_segment_ids.shape != loss_mask.shape
        or segment_ids.dtype != torch.int64
        or target_segment_ids.dtype != torch.int64
    ):
        raise ValueError("CSTM validity tensors have incompatible shape or dtype")
    if source_positions.ndim != 1 or source_positions.dtype != torch.int64:
        raise ValueError("CSTM source positions must be a one-dimensional int64 tensor")
    if source_positions.numel() and (
        int(source_positions.min()) < 0
        or int(source_positions.max()) >= loss_mask.shape[1]
    ):
        raise ValueError("CSTM source positions fall outside the packed context")

    batch, rows = loss_mask.shape[0], source_positions.shape[0]
    valid = torch.zeros(
        batch, rows, len(horizons), dtype=torch.bool, device=loss_mask.device
    )
    offsets = torch.arange(support, device=loss_mask.device, dtype=torch.int64)
    source_segments = segment_ids[:, source_positions]
    time = loss_mask.shape[1]
    for horizon_index, horizon in enumerate(horizons):
        starts = source_positions + (horizon - 1) * support
        indices = starts[:, None] + offsets[None]
        in_range = indices[:, -1] < time
        safe = indices.clamp(0, max(0, time - 1))
        valid[:, :, horizon_index] = (
            in_range[None]
            & loss_mask[:, safe].all(-1)
            & (
                target_segment_ids[:, safe] == source_segments[:, :, None]
            ).all(-1)
        )
    return valid


def build_causal_spectral_targets(
    labels: Tensor,
    loss_mask: Tensor,
    segment_ids: Tensor,
    target_segment_ids: Tensor,
    token_codes: Tensor,
    source_positions: Tensor,
    *,
    support: int,
    horizons: Sequence[int],
) -> CSTMTargetBatch:
    """Build DC and first-harmonic targets from strictly future token blocks.

    ``labels[:, p]`` is the token immediately after input position ``p``.
    Therefore the next block after a source ending at ``p`` begins at label
    index ``p``.  Horizon ``h`` begins ``(h-1)*support`` blocks farther ahead.
    A row is valid only when the complete target block exists, every constituent
    is an ordinary same-document language target, and every target segment
    matches the source segment.
    """

    horizons = tuple(int(value) for value in horizons)
    if support < 2 or not horizons or any(value <= 0 for value in horizons):
        raise ValueError("CSTM support and horizons must be positive")
    if len(set(horizons)) != len(horizons):
        raise ValueError("CSTM horizons must be unique")
    _validate_target_inputs(
        labels,
        loss_mask,
        segment_ids,
        target_segment_ids,
        token_codes,
        source_positions,
    )
    batch, rows = labels.shape[0], source_positions.shape[0]
    code_dimension = token_codes.shape[1]
    values = token_codes.new_zeros(
        batch, rows, len(horizons), 3, code_dimension
    )
    valid = causal_spectral_target_mask(
        loss_mask,
        segment_ids,
        target_segment_ids,
        source_positions,
        support=support,
        horizons=horizons,
    )
    offsets = torch.arange(support, device=labels.device, dtype=torch.int64)
    phase = 2 * pi * offsets.to(token_codes.dtype) / support
    cosine = phase.cos()
    negative_sine = -phase.sin()
    normalizer = support**0.5
    time = labels.shape[1]

    for horizon_index, horizon in enumerate(horizons):
        starts = source_positions + (horizon - 1) * support
        indices = starts[:, None] + offsets[None]
        safe = indices.clamp(0, max(0, time - 1))
        row_valid = valid[:, :, horizon_index]
        codes = token_codes[labels[:, safe]]
        dc = codes.sum(2) / normalizer
        real = (codes * cosine[None, None, :, None]).sum(2) / normalizer
        imaginary = (
            codes * negative_sine[None, None, :, None]
        ).sum(2) / normalizer
        spectral = torch.stack((dc, real, imaginary), 2)
        values[:, :, horizon_index] = (
            spectral * row_valid[:, :, None, None]
        )
    return CSTMTargetBatch(
        values.detach(),
        valid,
        horizons,
        support,
        source_positions.detach(),
    )


class CausalSpectralTargetPredictor(nn.Module):
    """Shared low-rank predictor for every MRRN scale and CSTM horizon."""

    def __init__(
        self,
        model_dimension: int,
        scale_count: int,
        vocabulary_size: int,
        config: CSTMArchitectureConfig = CSTMArchitectureConfig(),
    ) -> None:
        super().__init__()
        if min(model_dimension, scale_count, vocabulary_size) <= 0:
            raise ValueError("CSTM predictor dimensions must be positive")
        self.config = config
        self.model_dimension = model_dimension
        self.scale_count = scale_count
        rank = min(config.predictor_rank, model_dimension)
        self.rank = rank
        self.carrier_projection = nn.Linear(model_dimension, rank, bias=False)
        self.cognitive_projection = nn.Linear(model_dimension, rank, bias=False)
        self.scale_embedding = nn.Parameter(torch.zeros(scale_count, rank))
        self.horizon_embedding = nn.Parameter(
            torch.zeros(len(config.horizon_blocks), rank)
        )
        self.output_projection = nn.Linear(
            rank, 3 * config.code_dimension, bias=True
        )
        self.cognitive_gate = nn.Parameter(torch.zeros(scale_count))
        self.raw_harmonic_phase = nn.Parameter(
            torch.zeros(scale_count, len(config.horizon_blocks))
        )
        self.register_buffer(
            "token_codes",
            deterministic_token_codes(
                vocabulary_size,
                config.code_dimension,
                seed=config.token_code_seed,
            ),
            persistent=False,
        )
        self.register_buffer(
            "target_second_moment",
            torch.zeros(scale_count, 3, config.code_dimension),
        )
        self.register_buffer(
            "target_rms_initialized",
            torch.zeros(scale_count, 3, dtype=torch.bool),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.carrier_projection.weight)
        nn.init.xavier_uniform_(self.cognitive_projection.weight)
        nn.init.normal_(self.scale_embedding, std=0.02)
        nn.init.normal_(self.horizon_embedding, std=0.02)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        nn.init.zeros_(self.cognitive_gate)
        nn.init.zeros_(self.raw_harmonic_phase)

    def _horizon_indices(self, horizons: Sequence[int], device: torch.device) -> Tensor:
        lookup = {value: index for index, value in enumerate(self.config.horizon_blocks)}
        try:
            indices = [lookup[int(value)] for value in horizons]
        except KeyError as error:
            raise ValueError(f"CSTM horizon {error.args[0]} is not configured") from None
        return torch.tensor(indices, dtype=torch.int64, device=device)

    def forward(
        self,
        carrier_features: Tensor,
        cognitive_residual: Tensor,
        *,
        scale: int,
        horizons: Sequence[int],
    ) -> Tensor:
        if (
            carrier_features.ndim != 3
            or carrier_features.shape[-1] != self.model_dimension
            or cognitive_residual.shape != carrier_features.shape
        ):
            raise ValueError(
                "CSTM carrier and cognitive features must share (batch,rows,model_dimension)"
            )
        if not 0 <= scale < self.scale_count:
            raise ValueError("CSTM scale index is out of range")
        horizon_indices = self._horizon_indices(horizons, carrier_features.device)
        carrier = self.carrier_projection(carrier_features)
        cognitive = self.cognitive_projection(cognitive_residual)
        gate = torch.tanh(self.cognitive_gate[scale])
        hidden = torch.tanh(
            carrier[:, :, None, :]
            + gate * cognitive[:, :, None, :]
            + self.scale_embedding[scale][None, None, None, :]
            + self.horizon_embedding[horizon_indices][None, None, :, :]
        )
        prediction = self.output_projection(hidden).unflatten(
            -1, (3, self.config.code_dimension)
        )
        phase = pi * torch.tanh(
            self.raw_harmonic_phase[scale, horizon_indices]
        )
        cosine = phase.cos()[None, None, :, None]
        sine = phase.sin()[None, None, :, None]
        real = prediction[..., 1, :]
        imaginary = prediction[..., 2, :]
        rotated_real = cosine * real - sine * imaginary
        rotated_imaginary = sine * real + cosine * imaginary
        return torch.stack(
            (prediction[..., 0, :], rotated_real, rotated_imaginary),
            -2,
        )

    @torch.no_grad()
    def update_target_statistics(
        self,
        scale: int,
        targets: CSTMTargetBatch,
        *,
        importance_weight: float = 1.0,
    ) -> None:
        if not 0 <= scale < self.scale_count:
            raise ValueError("CSTM scale index is out of range")
        if not isfinite(importance_weight) or importance_weight <= 0:
            raise ValueError(
                "CSTM target-statistics importance weight must be finite and positive"
            )
        for component in range(3):
            selected = targets.values[:, :, :, component][targets.mask]
            if not selected.numel():
                continue
            moment = (
                selected.float().square().mean(0) * importance_weight
            )
            if bool(self.target_rms_initialized[scale, component]):
                self.target_second_moment[scale, component].lerp_(
                    moment.to(self.target_second_moment),
                    1 - self.config.target_rms_decay,
                )
            else:
                self.target_second_moment[scale, component].copy_(
                    moment.to(self.target_second_moment)
                )
                self.target_rms_initialized[scale, component] = True

    def target_rms(self, scale: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if not 0 <= scale < self.scale_count:
            raise ValueError("CSTM scale index is out of range")
        return self.target_second_moment[scale].clamp_min(
            self.config.minimum_target_rms**2
        ).sqrt().to(device=device, dtype=dtype)

    def loss(
        self,
        prediction: Tensor,
        targets: CSTMTargetBatch,
        *,
        scale: int,
        horizon_weights: Sequence[float] | None = None,
        update_statistics: bool,
        statistics_importance_weight: float = 1.0,
    ) -> CSTMLoss:
        if prediction.ndim != 5 or prediction.shape[0] <= 0:
            raise ValueError(
                "CSTM predictions must have shape "
                "(batch,rows,horizons,3,code_dimension)"
            )
        if prediction.shape != targets.values.shape:
            raise ValueError("CSTM prediction and target shapes differ")
        if update_statistics:
            self.update_target_statistics(
                scale,
                targets,
                importance_weight=statistics_importance_weight,
            )
        weights = (
            (1.0,) + (0.5,) * (len(targets.horizons) - 1)
            if horizon_weights is None else tuple(float(value) for value in horizon_weights)
        )
        if len(weights) != len(targets.horizons) or any(value < 0 for value in weights):
            raise ValueError("CSTM horizon weights must be nonnegative and aligned")
        weight = prediction.new_tensor(weights)[None, None, :]
        valid_weight = weight * targets.mask.to(prediction.dtype)
        target = targets.values.to(device=prediction.device, dtype=prediction.dtype)
        rms = self.target_rms(scale, device=prediction.device, dtype=prediction.dtype)
        error = (prediction - target) / rms[None, None, None]
        element = F.huber_loss(
            error,
            torch.zeros_like(error),
            reduction="none",
            delta=self.config.huber_delta,
        )
        row_loss = element.mean((-1, -2))
        weighted_sum = (row_loss * valid_weight).sum()
        weighted_rows = valid_weight.sum()
        differentiable_zero = prediction.sum() * 0
        loss = (
            weighted_sum / weighted_rows.clamp_min(1)
            if bool(targets.mask.any())
            else differentiable_zero
        )
        per_horizon = tuple(
            int(targets.mask[:, :, index].sum())
            for index in range(targets.mask.shape[2])
        )
        return CSTMLoss(
            loss,
            weighted_sum,
            weighted_rows,
            targets.valid_rows,
            targets.valid_rows * 3 * self.config.code_dimension,
            targets.token_participations,
            per_horizon,
        )
