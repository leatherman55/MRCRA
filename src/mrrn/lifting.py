"""Exact learned one-dimensional lifting analysis and synthesis."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class ScaleTensor:
    """One detail or approximation band plus its validity and physical support."""

    data: Tensor
    mask: Tensor
    scale: int
    sample_interval: float
    support: int
    kind: str = "detail"

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError("ScaleTensor.data must have shape (batch, length, channels)")
        if self.mask.shape != self.data.shape[:2] or self.mask.dtype != torch.bool:
            raise ValueError("ScaleTensor.mask must be boolean with shape (batch, length)")
        if self.scale < 0 or self.sample_interval <= 0 or self.support <= 0:
            raise ValueError("scale, sample_interval, and support must be valid positive metadata")
        if self.kind not in {"detail", "approximation"}:
            raise ValueError("kind must be 'detail' or 'approximation'")

    @property
    def coefficient_interval(self) -> float:
        """Physical spacing of adjacent coefficients (detail bands decimate once when formed)."""

        return self.sample_interval * (2.0 if self.kind == "detail" else 1.0)

    @property
    def base_interval(self) -> float:
        return self.sample_interval / 2**self.scale


@dataclass(frozen=True, slots=True)
class ReconstructionLevel:
    length: int
    paired: int

    @property
    def has_tail(self) -> bool:
        return self.length % 2 == 1


@dataclass(frozen=True, slots=True)
class ReconstructionContext:
    levels: tuple[ReconstructionLevel, ...]
    original_length: int
    sample_interval: float
    boundary: str


@dataclass(slots=True)
class LiftingStreamState:
    """Binary-carry state for exact completed causal lifting coefficients."""

    pending: list[tuple[Tensor, Tensor] | None]
    even_history: list[Tensor]
    detail_history: list[Tensor]
    emitted: list[int]
    steps: int
    sample_interval: float

    def detach(self) -> "LiftingStreamState":
        return LiftingStreamState(
            [None if item is None else (item[0].detach(), item[1].detach()) for item in self.pending],
            [item.detach() for item in self.even_history],
            [item.detach() for item in self.detail_history],
            list(self.emitted),
            self.steps,
            self.sample_interval,
        )


class CausalDepthwiseAffine(nn.Module):
    """Short causal depthwise filter plus a pointwise channel map."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        if channels <= 0 or kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("channels must be positive and kernel_size must be positive and odd")
        self.channels, self.kernel_size = channels, kernel_size
        self.point = nn.Linear(channels, channels, bias=False)
        self.depth = nn.Conv1d(
            channels, channels, kernel_size, groups=channels, bias=True
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.point.weight)
        nn.init.zeros_(self.depth.weight)
        nn.init.zeros_(self.depth.bias)

    def forward(self, x: Tensor, *, boundary: str = "causal") -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(f"expected (batch, length, {self.channels}), got {tuple(x.shape)}")
        if boundary not in {"causal", "reflect", "physical"}:
            raise ValueError("unsupported filter boundary")
        if x.shape[1] == 0:
            return x.clone()
        values = x.transpose(1, 2)
        if boundary == "causal":
            padded = F.pad(values, (self.kernel_size - 1, 0))
        else:
            half = self.kernel_size // 2
            mode = "reflect" if boundary == "reflect" and x.shape[1] > half else "replicate"
            padded = F.pad(values, (half, half), mode=mode)
        temporal = self.depth(padded).transpose(1, 2)
        return self.point(x) + temporal


class LiftingLevel(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.predict = CausalDepthwiseAffine(channels, kernel_size)
        self.update = CausalDepthwiseAffine(channels, kernel_size)
        with torch.no_grad():
            self.predict.point.weight.copy_(torch.eye(channels))
            self.update.point.weight.copy_(0.5 * torch.eye(channels))

    def analysis(
        self, x: Tensor, mask: Tensor, *, boundary: str = "causal"
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        even, odd = x[:, 0::2], x[:, 1::2]
        even_mask, odd_mask = mask[:, 0::2], mask[:, 1::2]
        paired = odd.shape[1]
        even_pair = even[:, :paired]
        detail = odd - self.predict(even_pair, boundary=boundary)
        approx_pair = even_pair + self.update(detail, boundary=boundary)
        detail_mask = even_mask[:, :paired] & odd_mask
        approx_mask = detail_mask
        if even.shape[1] > paired:
            approx_pair = torch.cat((approx_pair, even[:, -1:]), dim=1)
            approx_mask = torch.cat((approx_mask, even_mask[:, -1:]), dim=1)
        return detail, detail_mask, approx_pair, approx_mask

    def synthesis(
        self, detail: Tensor, approx: Tensor, original_length: int, *, boundary: str = "causal"
    ) -> Tensor:
        paired = original_length // 2
        detail = detail[:, :paired]
        approx_pair = approx[:, :paired]
        even = approx_pair - self.update(detail, boundary=boundary)
        odd = detail + self.predict(even, boundary=boundary)
        output = approx.new_empty(approx.shape[0], original_length, approx.shape[-1])
        output[:, 0 : 2 * paired : 2] = even
        output[:, 1 : 2 * paired : 2] = odd
        if original_length % 2:
            output[:, -1:] = approx[:, paired : paired + 1]
        return output


class LiftingAnalysisBank(nn.Module):
    """Transform once into detail bands and a final approximation, then invert exactly."""

    def __init__(self, channels: int, levels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if levels <= 0:
            raise ValueError("levels must be positive")
        self.channels, self.level_count = channels, levels
        self.levels = nn.ModuleList(LiftingLevel(channels, kernel_size) for _ in range(levels))

    def forward(
        self, x: Tensor, mask: Tensor | None = None, *, sample_interval: float = 1.0,
        boundary: str = "causal"
    ) -> tuple[tuple[ScaleTensor, ...], ReconstructionContext]:
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(f"expected (batch, length, {self.channels}), got {tuple(x.shape)}")
        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive")
        if boundary not in {"causal", "reflect", "physical"}:
            raise ValueError("unsupported boundary mode")
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        elif mask.shape != x.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape (batch, length)")

        current, current_mask = x, mask
        bands: list[ScaleTensor] = []
        contexts: list[ReconstructionLevel] = []
        for scale, level in enumerate(self.levels):
            contexts.append(ReconstructionLevel(current.shape[1], current.shape[1] // 2))
            detail, detail_mask, current, current_mask = level.analysis(
                current, current_mask, boundary=boundary
            )
            bands.append(
                ScaleTensor(detail, detail_mask, scale, sample_interval * 2**scale, 2 ** (scale + 1))
            )
        bands.append(
            ScaleTensor(
                current,
                current_mask,
                self.level_count,
                sample_interval * 2**self.level_count,
                2**self.level_count,
                "approximation",
            )
        )
        return tuple(bands), ReconstructionContext(
            tuple(contexts), x.shape[1], sample_interval, boundary
        )

    def inverse(
        self, bands: tuple[ScaleTensor, ...] | list[ScaleTensor], context: ReconstructionContext
    ) -> Tensor:
        if len(bands) != self.level_count + 1 or len(context.levels) != self.level_count:
            raise ValueError("band/context count does not match this analysis bank")
        current = bands[-1].data
        for index in range(self.level_count - 1, -1, -1):
            current = self.levels[index].synthesis(
                bands[index].data, current, context.levels[index].length,
                boundary=context.boundary,
            )
        if current.shape[1] != context.original_length:
            raise RuntimeError("internal reconstruction length mismatch")
        return current

    def initial_stream_state(
        self, batch: int, *, sample_interval: float = 1.0, device=None, dtype=None
    ) -> LiftingStreamState:
        if batch <= 0 or sample_interval <= 0:
            raise ValueError("batch and sample_interval must be positive")
        empty = [torch.empty(batch, 0, self.channels, device=device, dtype=dtype) for _ in self.levels]
        return LiftingStreamState(
            [None] * self.level_count,
            empty,
            [item.clone() for item in empty],
            [0] * (self.level_count + 1),
            0,
            sample_interval,
        )

    @staticmethod
    def _history_step(module: nn.Module, history: Tensor, value: Tensor, length: int) -> tuple[Tensor, Tensor]:
        history = torch.cat((history, value.unsqueeze(1)), 1)[:, -length:]
        return module(history)[:, -1], history

    def push(
        self,
        x: Tensor,
        state: LiftingStreamState,
        mask: Tensor | None = None,
    ) -> tuple[tuple[ScaleTensor | None, ...], LiftingStreamState]:
        """Push one position and emit each coefficient whose full support just completed."""

        if x.ndim != 2 or x.shape[-1] != self.channels:
            raise ValueError(f"expected one step with shape (batch, {self.channels})")
        if len(state.pending) != self.level_count or x.shape[0] != state.even_history[0].shape[0]:
            raise ValueError("stream state does not match this bank or batch")
        if mask is None:
            mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        elif mask.shape != x.shape[:1] or mask.dtype != torch.bool:
            raise ValueError("step mask must be boolean with shape (batch,)")
        active: list[ScaleTensor | None] = [None] * (self.level_count + 1)

        def carry(level_index: int, value: Tensor, valid: Tensor) -> None:
            if level_index == self.level_count:
                active[level_index] = ScaleTensor(
                    value.unsqueeze(1), valid.unsqueeze(1), level_index,
                    state.sample_interval * 2**level_index, 2**level_index, "approximation"
                )
                state.emitted[level_index] += 1
                return
            pending = state.pending[level_index]
            if pending is None:
                state.pending[level_index] = (value, valid)
                return
            even, even_valid = pending
            level = self.levels[level_index]
            prediction, state.even_history[level_index] = self._history_step(
                level.predict, state.even_history[level_index], even, level.predict.kernel_size
            )
            detail = value - prediction
            update, state.detail_history[level_index] = self._history_step(
                level.update, state.detail_history[level_index], detail, level.update.kernel_size
            )
            approximation = even + update
            coefficient_valid = even_valid & valid
            active[level_index] = ScaleTensor(
                detail.unsqueeze(1), coefficient_valid.unsqueeze(1), level_index,
                state.sample_interval * 2**level_index, 2 ** (level_index + 1), "detail"
            )
            state.emitted[level_index] += 1
            state.pending[level_index] = None
            carry(level_index + 1, approximation, coefficient_valid)

        carry(0, x, mask)
        state.steps += 1
        return tuple(active), state

    def push_aligned_chunk(
        self,
        x: Tensor,
        state: LiftingStreamState,
        mask: Tensor | None = None,
    ) -> tuple[tuple[ScaleTensor, ...], LiftingStreamState]:
        """Vectorize a stream-aligned chunk while preserving every carry.

        Intermediate chunks must align to the coarsest binary support. An
        arbitrary final tail can subsequently be processed through ``push``.
        """

        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(
                f"expected an aligned chunk shaped (batch,time,{self.channels})"
            )
        alignment = 2**self.level_count
        if (
            x.shape[1] <= 0
            or x.shape[1] % alignment
            or state.steps % alignment
            or any(item is not None for item in state.pending)
        ):
            raise ValueError(
                "chunk and stream position must align to the coarsest lifting support"
            )
        if x.shape[0] != state.even_history[0].shape[0]:
            raise ValueError("aligned chunk batch does not match lifting state")
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        elif mask.shape != x.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("chunk mask must be boolean with batch/time shape")

        current, current_mask = x, mask
        bands: list[ScaleTensor] = []
        for scale, level in enumerate(self.levels):
            even, odd = current[:, 0::2], current[:, 1::2]
            even_mask, odd_mask = current_mask[:, 0::2], current_mask[:, 1::2]
            if even.shape[1] != odd.shape[1]:
                raise RuntimeError("aligned lifting chunk produced an unmatched pair")

            even_history = state.even_history[scale]
            prediction_input = torch.cat((even_history, even), 1)
            prediction = level.predict(prediction_input)[
                :, even_history.shape[1] :
            ]
            detail = odd - prediction
            state.even_history[scale] = prediction_input[
                :, -level.predict.kernel_size :
            ]

            detail_history = state.detail_history[scale]
            update_input = torch.cat((detail_history, detail), 1)
            update = level.update(update_input)[:, detail_history.shape[1] :]
            state.detail_history[scale] = update_input[
                :, -level.update.kernel_size :
            ]

            current = even + update
            current_mask = even_mask & odd_mask
            bands.append(ScaleTensor(
                detail, current_mask, scale,
                state.sample_interval * 2**scale, 2 ** (scale + 1),
            ))
            state.emitted[scale] += detail.shape[1]

        bands.append(ScaleTensor(
            current, current_mask, self.level_count,
            state.sample_interval * 2**self.level_count,
            2**self.level_count, "approximation",
        ))
        state.emitted[self.level_count] += current.shape[1]
        state.steps += x.shape[1]
        return tuple(bands), state

    def roundtrip_error(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        bands, context = self(x, mask)
        return (self.inverse(bands, context) - x).norm() / x.norm().clamp_min(1e-12)
