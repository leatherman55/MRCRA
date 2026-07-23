"""Neighbor-only fine/coarse context exchange with explicit causal alignment."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .lifting import ScaleTensor


@dataclass(slots=True)
class ScaleExchangeStreamState:
    fine_values: list[list[Tensor]]
    fine_masks: list[list[Tensor]]
    latest_coarse: list[Tensor | None]

    def detach(self) -> "ScaleExchangeStreamState":
        return ScaleExchangeStreamState(
            [[value.detach() for value in group] for group in self.fine_values],
            [[value.detach() for value in group] for group in self.fine_masks],
            [None if value is None else value.detach() for value in self.latest_coarse],
        )


def _downsample(x: Tensor, target_length: int) -> Tensor:
    if target_length == 0:
        return x[:, :0]
    if x.shape[1] == 0:
        return x.new_zeros(x.shape[0], target_length, x.shape[2])
    result = F.avg_pool1d(x.transpose(1, 2), 2, 2, ceil_mode=True).transpose(1, 2)
    if result.shape[1] < target_length:
        result = torch.cat((result, result[:, -1:].expand(-1, target_length - result.shape[1], -1)), 1)
    return result[:, :target_length]


def _upsample(x: Tensor, target_length: int, causal: bool) -> Tensor:
    if target_length == 0:
        return x[:, :0]
    if x.shape[1] == 0:
        return x.new_zeros(x.shape[0], target_length, x.shape[2])
    repeated = x.repeat_interleave(2, dim=1)
    if causal:
        repeated = torch.cat((torch.zeros_like(repeated[:, :1]), repeated), dim=1)
    if repeated.shape[1] < target_length:
        repeated = torch.cat(
            (repeated, repeated[:, -1:].expand(-1, target_length - repeated.shape[1], -1)), 1
        )
    return repeated[:, :target_length]


def _fine_to_coarse(
    x: Tensor,
    mask: Tensor,
    target_length: int,
    *,
    fine_support: int,
    coarse_support: int,
) -> Tensor:
    """Aggregate only fine coefficients completed inside each coarse support interval."""

    if fine_support <= 0 or coarse_support < fine_support:
        raise ValueError("coarse support must be at least fine support")
    if target_length == 0:
        return x[:, :0]
    if x.shape[1] == 0:
        return x.new_zeros(x.shape[0], target_length, x.shape[2])
    valid = mask.unsqueeze(-1)
    weighted_prefix = torch.cat(
        (
            x.new_zeros(x.shape[0], 1, x.shape[2]),
            (x * valid).cumsum(1),
        ),
        1,
    )
    count_prefix = torch.cat(
        (
            x.new_zeros(x.shape[0], 1, 1),
            valid.to(x.dtype).cumsum(1),
        ),
        1,
    )
    targets = torch.arange(target_length, device=x.device)
    starts = (targets * coarse_support).div(
        fine_support, rounding_mode="floor"
    ).clamp(0, x.shape[1])
    stops = ((targets + 1) * coarse_support).div(
        fine_support, rounding_mode="floor"
    ).clamp(0, x.shape[1])
    sums = weighted_prefix[:, stops] - weighted_prefix[:, starts]
    counts = count_prefix[:, stops] - count_prefix[:, starts]
    populated = (stops > starts).view(1, -1, 1)
    return torch.where(
        populated,
        sums / counts.clamp_min(1),
        torch.zeros_like(sums),
    )


def _coarse_to_fine(
    x: Tensor,
    target_length: int,
    *,
    coarse_support: int,
    fine_support: int,
    causal: bool,
) -> Tensor:
    """Align coarse context by physical completion time rather than array index."""

    if fine_support <= 0 or coarse_support < fine_support:
        raise ValueError("coarse support must be at least fine support")
    if target_length == 0:
        return x[:, :0]
    if x.shape[1] == 0:
        return x.new_zeros(x.shape[0], target_length, x.shape[2])
    position = torch.arange(target_length, device=x.device)
    if causal:
        # A coarse item j may be used only once (j + 1) * coarse_support has elapsed.
        index = ((position + 1) * fine_support).div(coarse_support, rounding_mode="floor") - 1
        completed = index >= 0
    else:
        index = (position * fine_support).div(coarse_support, rounding_mode="floor")
        completed = torch.ones_like(index, dtype=torch.bool)
    index = index.clamp(0, x.shape[1] - 1)
    return x[:, index] * completed.view(1, -1, 1)


class ScaleExchange(nn.Module):
    """Fine innovation moves upward; coarse context modulates completed fine positions."""

    def __init__(self, widths: list[int] | tuple[int, ...], *, causal: bool = True) -> None:
        super().__init__()
        if not widths or any(width <= 0 for width in widths):
            raise ValueError("widths must contain positive values")
        self.widths, self.causal = tuple(widths), causal
        self.scale_codes = nn.ParameterList(nn.Parameter(torch.zeros(width)) for width in widths)
        self.metadata = nn.ModuleList(nn.Linear(2, width) for width in widths)
        self.fine_gate = nn.ModuleList(nn.Linear(widths[s], widths[s]) for s in range(len(widths) - 1))
        self.fine_value = nn.ModuleList(
            nn.Linear(widths[s], widths[s + 1]) for s in range(len(widths) - 1)
        )
        self.coarse_modulation = nn.ModuleList(
            nn.Linear(widths[s + 1], 2 * widths[s]) for s in range(len(widths) - 1)
        )
        self.fine_gain = nn.Parameter(torch.full((len(widths) - 1,), 1e-2))
        self.coarse_gain = nn.Parameter(torch.full((len(widths) - 1,), 1e-2))

    def _condition(self, band: ScaleTensor, index: int) -> Tensor:
        metadata = band.data.new_tensor([log(band.sample_interval), log(float(band.support))])
        return band.data + self.scale_codes[index] + self.metadata[index](metadata)

    def initial_stream_state(self) -> ScaleExchangeStreamState:
        edges = len(self.widths) - 1
        return ScaleExchangeStreamState([[] for _ in range(edges)], [[] for _ in range(edges)], [None] * edges)

    def step(
        self,
        bands: tuple[ScaleTensor | None, ...],
        state: ScaleExchangeStreamState,
    ) -> tuple[tuple[ScaleTensor | None, ...], ScaleExchangeStreamState]:
        """Exchange context among coefficients completing at the current original-domain step."""

        edges = len(self.widths) - 1
        if len(bands) != len(self.widths) or any(
            len(values) != edges for values in (state.fine_values, state.fine_masks, state.latest_coarse)
        ):
            raise ValueError("stream exchange state or bands do not match configured scales")
        conditioned: list[Tensor | None] = []
        for scale, (band, width) in enumerate(zip(bands, self.widths, strict=True)):
            if band is None:
                conditioned.append(None)
            elif band.data.shape[1:] != (1, width):
                raise ValueError("active stream bands must contain exactly one coefficient")
            else:
                conditioned.append(self._condition(band, scale) * band.mask.unsqueeze(-1))

        fine_messages: list[Tensor | None] = [None] * len(bands)
        coarse_messages: list[Tensor | None] = [None] * len(bands)
        for edge in range(edges):
            fine, coarse = bands[edge], bands[edge + 1]
            if fine is not None:
                selected = torch.sigmoid(self.fine_gate[edge](conditioned[edge])) * conditioned[edge]
                state.fine_values[edge].append(self.fine_value[edge](selected))
                state.fine_masks[edge].append(fine.mask)
            if coarse is not None:
                values, masks = state.fine_values[edge], state.fine_masks[edge]
                if values:
                    stacked, valid = torch.cat(values, 1), torch.cat(masks, 1).unsqueeze(-1)
                    fine_messages[edge + 1] = (
                        (stacked * valid).sum(1, keepdim=True)
                        / valid.sum(1, keepdim=True).clamp_min(1)
                    )
                else:
                    fine_messages[edge + 1] = torch.zeros_like(conditioned[edge + 1])
                state.fine_values[edge].clear()
                state.fine_masks[edge].clear()
                state.latest_coarse[edge] = conditioned[edge + 1]
            if fine is not None:
                context = conditioned[edge + 1] if coarse is not None else state.latest_coarse[edge]
                if context is None:
                    context = conditioned[edge].new_zeros(
                        conditioned[edge].shape[0], 1, self.widths[edge + 1]
                    )
                gamma, beta = self.coarse_modulation[edge](context).chunk(2, -1)
                normalized = F.rms_norm(conditioned[edge], (self.widths[edge],))
                coarse_messages[edge] = torch.tanh(gamma) * normalized + beta

        output: list[ScaleTensor | None] = []
        for scale, band in enumerate(bands):
            if band is None:
                output.append(None)
                continue
            updated = conditioned[scale]
            if fine_messages[scale] is not None:
                updated = updated + self.fine_gain[scale - 1] * fine_messages[scale]
            if coarse_messages[scale] is not None:
                updated = updated + self.coarse_gain[scale] * coarse_messages[scale]
            output.append(ScaleTensor(
                updated * band.mask.unsqueeze(-1), band.mask, band.scale,
                band.sample_interval, band.support, band.kind
            ))
        return tuple(output), state

    def forward_aligned_chunk(
        self,
        bands: tuple[ScaleTensor, ...] | list[ScaleTensor],
        state: ScaleExchangeStreamState,
    ) -> tuple[tuple[ScaleTensor, ...], ScaleExchangeStreamState]:
        """Vectorize an exact complete-support stream transition.

        Every adjacent fine/coarse pair must cover the same original-domain
        interval and the incoming fine accumulator must be empty.  Under that
        contract all upward messages are ordinary batched reductions.  The
        only cross-chunk dependency is the most recently completed coarse
        coefficient, which supplies causal context before this chunk completes
        its first new coarse coefficient.
        """

        edges = len(self.widths) - 1
        if len(bands) != len(self.widths) or any(
            len(values) != edges
            for values in (state.fine_values, state.fine_masks, state.latest_coarse)
        ):
            raise ValueError("stream exchange state or bands do not match configured scales")
        if any(values or masks for values, masks in zip(
            state.fine_values, state.fine_masks, strict=True
        )):
            raise ValueError("aligned exchange chunks require empty fine accumulators")

        data: list[Tensor] = []
        for index, (band, width) in enumerate(zip(bands, self.widths, strict=True)):
            if band.data.ndim != 3 or band.data.shape[-1] != width:
                raise ValueError(f"scale {index} expected batched width {width}")
            data.append(self._condition(band, index) * band.mask.unsqueeze(-1))

        fine_messages: list[Tensor | None] = [None] * len(data)
        coarse_messages: list[Tensor | None] = [None] * len(data)
        for edge in range(edges):
            fine_band, coarse_band = bands[edge], bands[edge + 1]
            fine_duration = fine_band.data.shape[1] * fine_band.support
            coarse_duration = coarse_band.data.shape[1] * coarse_band.support
            if fine_duration != coarse_duration:
                raise ValueError(
                    "aligned exchange bands must cover the same original-domain interval"
                )
            selected = torch.sigmoid(self.fine_gate[edge](data[edge])) * data[edge]
            fine_messages[edge + 1] = _fine_to_coarse(
                self.fine_value[edge](selected),
                fine_band.mask,
                data[edge + 1].shape[1],
                fine_support=fine_band.support,
                coarse_support=coarse_band.support,
            )

            fine_length = data[edge].shape[1]
            if fine_length:
                positions = torch.arange(fine_length, device=data[edge].device)
                if self.causal:
                    indices = (
                        ((positions + 1) * fine_band.support).div(
                            coarse_band.support, rounding_mode="floor"
                        )
                        - 1
                    )
                    has_current = indices >= 0
                else:
                    indices = (positions * fine_band.support).div(
                        coarse_band.support, rounding_mode="floor"
                    )
                    has_current = torch.ones_like(indices, dtype=torch.bool)
                indices = indices.clamp(0, max(0, data[edge + 1].shape[1] - 1))
                current = data[edge + 1][:, indices]
                prior = state.latest_coarse[edge]
                if prior is None:
                    prior = data[edge].new_zeros(
                        data[edge].shape[0], 1, self.widths[edge + 1]
                    )
                if prior.shape != (
                    data[edge].shape[0], 1, self.widths[edge + 1]
                ):
                    raise ValueError("latest coarse exchange state has an incompatible shape")
                context = torch.where(
                    has_current.view(1, -1, 1), current,
                    prior.expand(-1, fine_length, -1),
                )
            else:
                context = data[edge].new_zeros(
                    data[edge].shape[0], 0, self.widths[edge + 1]
                )
            gamma, beta = self.coarse_modulation[edge](context).chunk(2, -1)
            normalized = F.rms_norm(data[edge], (self.widths[edge],))
            coarse_messages[edge] = torch.tanh(gamma) * normalized + beta
            if data[edge + 1].shape[1]:
                state.latest_coarse[edge] = data[edge + 1][:, -1:]

        output = []
        for scale, band in enumerate(bands):
            updated = data[scale]
            if fine_messages[scale] is not None:
                updated = updated + self.fine_gain[scale - 1] * fine_messages[scale]
            if coarse_messages[scale] is not None:
                updated = updated + self.coarse_gain[scale] * coarse_messages[scale]
            output.append(ScaleTensor(
                updated * band.mask.unsqueeze(-1),
                band.mask,
                band.scale,
                band.sample_interval,
                band.support,
                band.kind,
            ))
        return tuple(output), state

    def forward(self, bands: tuple[ScaleTensor, ...] | list[ScaleTensor]) -> tuple[ScaleTensor, ...]:
        if len(bands) != len(self.widths):
            raise ValueError("band count must match configured scale widths")
        data = []
        for index, (band, width) in enumerate(zip(bands, self.widths, strict=True)):
            if band.data.shape[-1] != width:
                raise ValueError(f"scale {index} expected width {width}")
            data.append(self._condition(band, index) * band.mask.unsqueeze(-1))

        fine_messages = [None] * len(data)
        coarse_messages = [None] * len(data)
        for scale in range(len(data) - 1):
            selected = torch.sigmoid(self.fine_gate[scale](data[scale])) * data[scale]
            fine_messages[scale + 1] = _fine_to_coarse(
                self.fine_value[scale](selected),
                bands[scale].mask,
                data[scale + 1].shape[1],
                fine_support=bands[scale].support,
                coarse_support=bands[scale + 1].support,
            )
            context = _coarse_to_fine(
                data[scale + 1],
                data[scale].shape[1],
                coarse_support=bands[scale + 1].support,
                fine_support=bands[scale].support,
                causal=self.causal,
            )
            gamma, beta = self.coarse_modulation[scale](context).chunk(2, -1)
            normalized = F.rms_norm(data[scale], (self.widths[scale],))
            coarse_messages[scale] = torch.tanh(gamma) * normalized + beta

        output = []
        for scale, band in enumerate(bands):
            updated = data[scale]
            if fine_messages[scale] is not None:
                updated = updated + self.fine_gain[scale - 1] * fine_messages[scale]
            if coarse_messages[scale] is not None:
                updated = updated + self.coarse_gain[scale] * coarse_messages[scale]
            updated = updated * band.mask.unsqueeze(-1)
            output.append(
                ScaleTensor(
                    updated,
                    band.mask,
                    band.scale,
                    band.sample_interval,
                    band.support,
                    band.kind,
                )
            )
        return tuple(output)
