"""Complete multiscale block and batch MRRN topology."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .attention import AttentionCandidates, ResonantAttention
from .config import MRRNConfig
from .lifting import LiftingAnalysisBank, LiftingStreamState, ReconstructionContext, ScaleTensor
from .mixer import (
    AntiAliasActivation, AntiAliasState, GatedLocalMixer,
    HybridMixerDiagnostics, HybridSpectralMixer,
)
from .memory import EideticMemory, MemoryItem, MemoryWritePolicy
from .packed_projection import PackedCache, packed_linear
from .resonance import ComplexResonator, ResonatorParameters, ResonatorState
from .scale_exchange import ScaleExchange, ScaleExchangeStreamState


@dataclass(frozen=True, slots=True)
class BlockState:
    resonators: tuple[ResonatorState, ...]


@dataclass(frozen=True, slots=True)
class MRRNState:
    blocks: tuple[BlockState, ...]

    def detach(self) -> "MRRNState":
        return MRRNState(tuple(BlockState(tuple(state.detach() for state in block.resonators)) for block in self.blocks))


@dataclass(frozen=True, slots=True)
class BlockDiagnostics:
    branch_weights: tuple[Tensor, ...]
    resonance: tuple[ResonatorParameters, ...]
    attention_weights: tuple[Tensor | None, ...]
    spectral_mixers: tuple[HybridMixerDiagnostics | None, ...]


@dataclass(frozen=True, slots=True)
class MRRNOutput:
    prediction: Tensor
    bands: tuple[ScaleTensor, ...]
    state: MRRNState
    diagnostics: tuple[BlockDiagnostics, ...]
    reconstruction: ReconstructionContext
    latent: Tensor | None = None


@dataclass(slots=True)
class BlockStreamState:
    resonators: list[ResonatorState]
    exchange: ScaleExchangeStreamState
    recent_features: list[list[Tensor]]
    recent_masks: list[list[Tensor]]
    recent_times: list[list[Tensor]]
    scale_steps: list[int]
    anti_alias: list[AntiAliasState | None]

    def detach(self) -> "BlockStreamState":
        return BlockStreamState(
            [item.detach() for item in self.resonators], self.exchange.detach(),
            [[item.detach() for item in values] for values in self.recent_features],
            [[item.detach() for item in values] for values in self.recent_masks],
            [[item.detach() for item in values] for values in self.recent_times],
            list(self.scale_steps),
            [None if item is None else item.detach() for item in self.anti_alias],
        )


@dataclass(slots=True)
class MRRNStreamState:
    lifting: LiftingStreamState
    blocks: list[BlockStreamState]
    latest_bands: list[ScaleTensor | None]
    position: int
    batch: int
    sample_interval: float

    def detach(self) -> "MRRNStreamState":
        return MRRNStreamState(
            self.lifting.detach(), [block.detach() for block in self.blocks],
            [
                None if band is None else ScaleTensor(
                    band.data.detach(), band.mask.detach(), band.scale,
                    band.sample_interval, band.support, band.kind
                )
                for band in self.latest_bands
            ],
            self.position, self.batch, self.sample_interval,
        )


@dataclass(frozen=True, slots=True)
class MRRNStepOutput:
    prediction: Tensor
    active_bands: tuple[ScaleTensor | None, ...]
    state: MRRNStreamState
    latent: Tensor | None = None


@dataclass(frozen=True, slots=True)
class MRRNChunkOutput:
    """Exact vectorized transition over a complete-support causal chunk."""

    prediction: Tensor
    bands: tuple[ScaleTensor, ...]
    state: MRRNStreamState
    latent: Tensor


@dataclass(frozen=True, slots=True)
class MRRNPrefillOutput:
    """Sequence outputs and exact continuation state from causal prefill."""

    prediction: Tensor
    state: MRRNStreamState
    latent: Tensor


def _local_candidates(
    band: ScaleTensor, window: int, *, causal: bool = True
) -> tuple[Tensor, AttentionCandidates, Tensor, Tensor]:
    data, batch, length = band.data, band.data.shape[0], band.data.shape[1]
    positions = torch.arange(length, device=data.device)
    if causal:
        offsets = torch.arange(window - 1, -1, -1, device=data.device)
        indices = positions[:, None] - offsets[None, :]
    else:
        offsets = torch.arange(window, device=data.device) - window // 2
        indices = positions[:, None] + offsets[None, :]
    valid_index = (indices >= 0) & (indices < length)
    safe = indices.clamp(0, max(0, length - 1))
    features = data[:, safe].reshape(batch * length, window, data.shape[-1])
    mask = (band.mask[:, safe] & valid_index).reshape(batch * length, window)
    times = (
        (safe.to(data.dtype) + 1) * band.coefficient_interval - band.base_interval
    ).expand(batch, -1, -1).reshape(batch * length, window)
    scales = torch.full_like(times, float(band.scale))
    candidates = AttentionCandidates(features, times, scales, mask)
    query = data.reshape(batch * length, 1, data.shape[-1])
    query_times = (
        (positions.to(data.dtype) + 1) * band.coefficient_interval - band.base_interval
    ).expand(batch, -1).reshape(batch * length, 1)
    query_scales = torch.full_like(query_times, float(band.scale))
    return query, candidates, query_times, query_scales


def _memory_candidates(
    band: ScaleTensor,
    memories: tuple[EideticMemory, ...] | list[EideticMemory],
    key_projection: nn.Module,
    signature_projection: nn.Module,
    value_projection: nn.Module,
    count: int,
    *,
    absolute_positions: Tensor | None = None,
    query_events: Tensor | None = None,
    resonant_attention: ResonantAttention | None = None,
) -> AttentionCandidates:
    """Route cheaply, rerank exactly, and materialize a fixed-size causal candidate tensor."""

    batch, length, _ = band.data.shape
    if len(memories) != batch or count <= 0:
        raise ValueError("one compatible memory per batch item and a positive count are required")
    keys = key_projection(band.data)
    signatures = signature_projection(band.data)
    if absolute_positions is None:
        positions = (torch.arange(length, device=band.data.device) + 1) * band.support - 1
        absolute_positions = positions.expand(batch, -1)
    if absolute_positions.shape != (batch, length):
        raise ValueError("absolute_positions must have shape (batch,time)")
    if query_events is None:
        query_events = torch.ones(batch, length, dtype=torch.bool, device=band.data.device)
    if query_events.shape != (batch, length) or query_events.dtype != torch.bool:
        raise ValueError("query_events must be boolean with shape (batch,time)")
    rows, row_times, row_scales, row_masks = [], [], [], []
    for batch_index, memory in enumerate(memories):
        if (memory.key_dim, memory.value_dim, memory.signature_dim) != (
            keys.shape[-1], value_projection.in_features, signatures.shape[-1]
        ):
            raise ValueError("memory dimensions do not match the model memory interface")
        for time_index in range(length):
            query_time = int(absolute_positions[batch_index, time_index])
            routed = []
            if bool(query_events[batch_index, time_index]):
                routed = memory.retrieve(
                    signatures[batch_index, time_index].detach().cpu(), max(count, 4 * count),
                    query_time=query_time,
                )
            if resonant_attention is None:
                selected = memory.rerank(
                    keys[batch_index, time_index].detach().cpu(), routed, count
                )
                pool_count = count
            else:
                selected, pool_count = routed, max(count, 4 * count)
            items = [memory.get(handle) for handle in selected]
            values = band.data.new_zeros(pool_count, value_projection.in_features)
            times = band.data.new_zeros(pool_count)
            scales = band.data.new_zeros(pool_count)
            valid = torch.zeros(pool_count, dtype=torch.bool, device=band.data.device)
            for slot, item in enumerate(items[:pool_count]):
                values[slot] = item.value.to(device=band.data.device, dtype=band.data.dtype)
                times[slot] = item.timestamp * band.base_interval
                scales[slot] = item.scale
                valid[slot] = True
            projected = value_projection(values)
            if resonant_attention is not None:
                pool = AttentionCandidates(
                    projected.unsqueeze(0), times.unsqueeze(0), scales.unsqueeze(0), valid.unsqueeze(0)
                )
                query = band.data[batch_index, time_index].view(1, 1, -1)
                query_time_tensor = band.data.new_tensor([[query_time * band.base_interval]])
                query_scale = band.data.new_tensor([[float(band.scale)]])
                score = resonant_attention.scores(
                    query, pool, query_time_tensor, query_scale, causal=True
                ).mean(-1).flatten()
                order = torch.argsort(score, descending=True, stable=True)[:count]
                projected, times, scales, valid = (
                    projected[order], times[order], scales[order], valid[order]
                )
                for index in order[valid].tolist():
                    memory.get(selected[index]).use_count += 1
            rows.append(projected)
            row_times.append(times)
            row_scales.append(scales)
            row_masks.append(valid)
    return AttentionCandidates(
        torch.stack(rows), torch.stack(row_times), torch.stack(row_scales), torch.stack(row_masks),
        torch.full((batch * length, count), 2, dtype=torch.long, device=band.data.device),
    )


def _join_candidates(left: AttentionCandidates, right: AttentionCandidates) -> AttentionCandidates:
    if left.features.shape[0] != right.features.shape[0]:
        raise ValueError("candidate groups must share batch/query rows")
    left_kinds = left.kinds
    if left_kinds is None:
        left_kinds = torch.zeros(left.mask.shape, dtype=torch.long, device=left.mask.device)
    right_kinds = right.kinds
    if right_kinds is None:
        right_kinds = torch.zeros(right.mask.shape, dtype=torch.long, device=right.mask.device)
    return AttentionCandidates(
        torch.cat((left.features, right.features), 1), torch.cat((left.times, right.times), 1),
        torch.cat((left.scales, right.scales), 1), torch.cat((left.mask, right.mask), 1),
        torch.cat((left_kinds, right_kinds), 1),
    )


def _landmark_candidates(
    query_band: ScaleTensor,
    coarse_band: ScaleTensor,
    projection: nn.Module,
    count: int,
    *,
    causal: bool,
) -> AttentionCandidates:
    """Select a bounded time-aligned window of explicit coarser-scale landmarks."""

    if count <= 0 or coarse_band.scale <= query_band.scale:
        raise ValueError("landmarks require a positive count and a genuinely coarser band")
    batch, query_length = query_band.data.shape[:2]
    query_positions = (torch.arange(query_length, device=query_band.data.device) + 1) * query_band.support - 1
    coarse_positions = (torch.arange(coarse_band.data.shape[1], device=query_band.data.device) + 1) * coarse_band.support - 1
    if causal:
        latest = torch.searchsorted(coarse_positions, query_positions, right=True) - 1
        offsets = torch.arange(count - 1, -1, -1, device=query_band.data.device)
        indices = latest[:, None] - offsets
    else:
        centers = torch.searchsorted(coarse_positions, query_positions).clamp_max(max(0, coarse_positions.numel() - 1))
        indices = centers[:, None] + torch.arange(count, device=query_band.data.device) - count // 2
    valid_index = (indices >= 0) & (indices < coarse_band.data.shape[1])
    safe = indices.clamp(0, max(0, coarse_band.data.shape[1] - 1))
    features = projection(coarse_band.data[:, safe]).reshape(batch * query_length, count, -1)
    mask = (coarse_band.mask[:, safe] & valid_index).reshape(batch * query_length, count)
    times = (
        (safe.to(query_band.data.dtype) + 1) * coarse_band.coefficient_interval - coarse_band.base_interval
    ).expand(batch, -1, -1).reshape(batch * query_length, count)
    scales = torch.full_like(times, float(coarse_band.scale))
    return AttentionCandidates(
        features, times, scales, mask,
        torch.ones(batch * query_length, count, dtype=torch.long, device=query_band.data.device),
    )


def _stream_landmark_candidates(
    query_band: ScaleTensor,
    query_times: Tensor,
    coarse_features: Tensor,
    coarse_mask: Tensor,
    coarse_times: Tensor,
    coarse_scale: int,
    projection: nn.Module,
    count: int,
) -> AttentionCandidates:
    """Gather bounded coarser landmarks from absolute stream time."""

    batch, query_length = query_band.data.shape[:2]
    if (
        count <= 0
        or coarse_scale <= query_band.scale
        or query_times.shape != (batch, query_length)
        or coarse_features.ndim != 3
        or coarse_features.shape[:2] != coarse_mask.shape
        or coarse_mask.shape != coarse_times.shape
        or coarse_features.shape[0] != batch
    ):
        raise ValueError("stream landmark candidate contracts are incompatible")
    coarse_length = coarse_features.shape[1]
    if coarse_length == 0:
        return AttentionCandidates(
            query_band.data.new_zeros(batch * query_length, count, query_band.data.shape[-1]),
            query_band.data.new_zeros(batch * query_length, count),
            query_band.data.new_full((batch * query_length, count), float(coarse_scale)),
            torch.zeros(
                batch * query_length, count, dtype=torch.bool,
                device=query_band.data.device,
            ),
            torch.ones(
                batch * query_length, count, dtype=torch.long,
                device=query_band.data.device,
            ),
        )
    latest = torch.searchsorted(
        coarse_times.contiguous(), query_times.contiguous(), right=True
    ) - 1
    offsets = torch.arange(
        count - 1, -1, -1, device=query_band.data.device
    )
    indices = latest.unsqueeze(-1) - offsets
    valid_index = (indices >= 0) & (indices < coarse_length)
    safe = indices.clamp(0, coarse_length - 1)
    projected = projection(coarse_features)
    feature_indices = safe.unsqueeze(-1).expand(-1, -1, -1, projected.shape[-1])
    features = projected.unsqueeze(1).expand(
        -1, query_length, -1, -1
    ).gather(2, feature_indices)
    gathered_times = coarse_times.unsqueeze(1).expand(
        -1, query_length, -1
    ).gather(2, safe)
    gathered_mask = coarse_mask.unsqueeze(1).expand(
        -1, query_length, -1
    ).gather(2, safe)
    mask = gathered_mask & valid_index & (
        gathered_times <= query_times.unsqueeze(-1)
    )
    return AttentionCandidates(
        features.reshape(batch * query_length, count, -1),
        gathered_times.reshape(batch * query_length, count),
        torch.full_like(
            gathered_times.reshape(batch * query_length, count),
            float(coarse_scale),
        ),
        mask.reshape(batch * query_length, count),
        torch.ones(
            batch * query_length, count, dtype=torch.long,
            device=query_band.data.device,
        ),
    )


def _causal_expand(data: Tensor, target_length: int, support: int) -> Tensor:
    """Expose coefficient j only at/after its original-domain completion time."""

    if target_length == 0:
        return data[:, :0]
    positions = torch.arange(target_length, device=data.device)
    completed = positions >= support - 1
    indices = ((positions - (support - 1)).clamp_min(0) // support).clamp_max(max(0, data.shape[1] - 1))
    if data.shape[1] == 0:
        return data.new_zeros(data.shape[0], target_length, data.shape[-1])
    return data[:, indices] * completed.view(1, -1, 1)


def _causal_expand_chunk(
    data: Tensor,
    target_length: int,
    support: int,
    prior: Tensor | None,
) -> Tensor:
    """Expand current coefficients while carrying the prior completed value."""

    if target_length == 0:
        return data[:, :0]
    if support <= 0 or data.ndim != 3:
        raise ValueError("chunk synthesis support and coefficient tensor are invalid")
    batch, _, width = data.shape
    if prior is None:
        prior = data.new_zeros(batch, 1, width)
    elif prior.shape != (batch, 1, width):
        raise ValueError("prior synthesis band has an incompatible shape")
    positions = torch.arange(target_length, device=data.device)
    indices = (positions + 1).div(support, rounding_mode="floor") - 1
    has_current = indices >= 0
    if data.shape[1] == 0:
        current = prior.expand(-1, target_length, -1)
    else:
        current = data[:, indices.clamp(0, data.shape[1] - 1)]
    return torch.where(
        has_current.view(1, -1, 1),
        current,
        prior.expand(-1, target_length, -1),
    )


class MRRNBlock(nn.Module):
    """Scale exchange, resonance, local mixing, bounded attention, and simplex fusion."""

    def __init__(
        self, config: MRRNConfig, *, layer_index: int, attention_enabled: tuple[bool, ...]
    ) -> None:
        super().__init__()
        scales = config.scale_configs()
        widths = [scale.width for scale in scales]
        interval_multipliers = [
            2 ** (scale + 1) if scale < len(scales) - 1 else 2**scale
            for scale in range(len(scales))
        ]
        if len(attention_enabled) != len(scales):
            raise ValueError("attention schedule must name every scale")
        self.widths, self.attention_enabled, self.causal = tuple(widths), attention_enabled, config.causal
        self.exchange = ScaleExchange(widths, causal=config.causal)
        self.norms = nn.ModuleList(nn.RMSNorm(width) for width in widths)
        self.resonators = nn.ModuleList(
            ComplexResonator(
                scale.width,
                scale.heads,
                scale.modes,
                scale.mimo_rank,
                alpha_min=config.alpha_min,
                delta_min=config.delta_min,
                omega_max=config.omega_max / interval_multipliers[index],
                decay_normalized_drive=config.decay_normalized_resonance,
            )
            for index, scale in enumerate(scales)
        )
        self.reverse_resonators = (
            None
            if config.causal
            else nn.ModuleList(
                ComplexResonator(
                    scale.width, scale.heads, scale.modes, scale.mimo_rank,
                    alpha_min=config.alpha_min, delta_min=config.delta_min,
                    omega_max=config.omega_max / interval_multipliers[index],
                    decay_normalized_drive=config.decay_normalized_resonance,
                )
                for index, scale in enumerate(scales)
            )
        )
        self.bidirectional_gates = nn.ModuleList(nn.Linear(width, width) for width in widths)
        self.mixers = nn.ModuleList([
            HybridSpectralMixer(
                width, config.mixer_expansion, scale.heads,
                min(scale.modes, config.spectral_modes), scale.mimo_rank,
                structured_rank=config.structured_mixer_rank,
                spectral_kwargs={
                    "basis_order": config.spectral_basis_order,
                    "maximum_gain": config.spectral_maximum_gain,
                    "maximum_phase": config.spectral_maximum_phase,
                    "triads_per_mode": config.spectral_triads_per_mode,
                    "maximum_triad_gain": config.spectral_maximum_triad_gain,
                    "frequency_max": config.omega_max / interval_multipliers[index],
                },
            )
            if config.spectral_activation else
            GatedLocalMixer(width, config.mixer_expansion, structured_rank=config.structured_mixer_rank)
            for index, (width, scale) in enumerate(zip(widths, scales, strict=True))
        ])
        self.anti_aliases = (
            nn.ModuleList(AntiAliasActivation(width, causal=config.causal) for width in widths)
            if config.continuous_signal else None
        )
        self.attentions = nn.ModuleList(
            ResonantAttention(
                scale.width, scale.heads, min(scale.modes, 16),
                max_scale=config.scales + 1,
                frequency_max=config.omega_max / interval_multipliers[index],
            )
            for index, scale in enumerate(scales)
        )
        self.identity = nn.ModuleList(nn.Linear(width, width) for width in widths)
        branch_count = 5 if config.relational_branch else 4
        self.branch_gates = nn.ModuleList(nn.Linear(width, branch_count) for width in widths)
        self._packed_projection_cache: PackedCache = {}
        self._compiled_chunk_cores: dict[int, object] = {}
        self.relational_projections = (
            nn.ModuleList(
                nn.Linear(config.relational_context_dim, width, bias=False) for width in widths
            )
            if config.relational_branch else None
        )
        self.layer_scale = nn.Parameter(torch.full((len(scales),), config.residual_scale))
        self.windows = tuple(scale.attention_window for scale in scales)
        self.retrieved_items = config.retrieved_items
        self.memory_keys = nn.ModuleList(nn.Linear(width, config.model_dim) for width in widths)
        self.memory_signatures = nn.ModuleList(nn.Linear(width, config.model_dim) for width in widths)
        self.memory_values = nn.ModuleList(nn.Linear(config.model_dim, width) for width in widths)
        self.memory_query_gates = nn.ModuleList(nn.Linear(width, 1) for width in widths)
        self.landmark_values = nn.ModuleList(
            nn.Linear(widths[scale + 1], widths[scale]) for scale in range(len(widths) - 1)
        )
        self.landmark_count = max(1, min(8, config.attention_window // 4))
        self.attention_query_tile_size = config.attention_query_tile_size
        self.activation_checkpointing = config.activation_checkpointing
        for gate in self.branch_gates:
            nn.init.zeros_(gate.weight)
            bias = [-2.0, 0.0, -2.0, 1.0]
            if config.relational_branch:
                bias.append(-6.0)
            gate.bias.data.copy_(gate.bias.new_tensor(bias))
        for gate in self.memory_query_gates:
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, -2.0)
        self.layer_index = layer_index

    def initial_state(self, batch: int, *, device=None, dtype=None) -> BlockState:
        return BlockState(tuple(module.initial_state(batch, device=device, dtype=dtype) for module in self.resonators))

    def enable_compiled_tensor_cores(self, *, mode: str = "default") -> None:
        """Compile pure per-scale compute while leaving state commits in Python."""

        if not hasattr(torch, "compile"):
            raise RuntimeError("this PyTorch build does not provide torch.compile")
        compiled: dict[int, object] = {}
        for scale, (resonator, mixer) in enumerate(zip(
            self.resonators, self.mixers, strict=True
        )):
            def tensor_core(
                normalized: Tensor,
                state_value: Tensor,
                previous_drive: Tensor,
                mask: Tensor,
                coefficient_interval: Tensor,
                *,
                resonator_module=resonator,
                mixer_module=mixer,
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
                resonant, next_state, _ = resonator_module.parallel(
                    normalized,
                    ResonatorState(state_value, previous_drive, 0),
                    mask,
                    sample_interval=coefficient_interval,
                )
                return (
                    resonant,
                    next_state.value,
                    next_state.previous_drive,
                    mixer_module(normalized),
                )

            compiled[scale] = torch.compile(
                tensor_core, fullgraph=False, dynamic=False, mode=mode
            )
        self._compiled_chunk_cores = compiled

    def disable_compiled_tensor_cores(self) -> None:
        self._compiled_chunk_cores.clear()

    def _identity_and_branch(self, scale: int, normalized: Tensor) -> tuple[Tensor, Tensor]:
        """Share one matrix launch for two independent projections of the same input."""

        identity, gate = self.identity[scale], self.branch_gates[scale]
        packed = packed_linear(
            normalized, (identity, gate), self._packed_projection_cache,
            f"identity_branch_{scale}",
        )
        projected, logits = packed.split((identity.out_features, gate.out_features), -1)
        return projected, torch.softmax(logits, -1)

    def initial_stream_state(self, batch: int, *, device=None, dtype=None) -> BlockStreamState:
        return BlockStreamState(
            [module.initial_state(batch, device=device, dtype=dtype) for module in self.resonators],
            self.exchange.initial_stream_state(),
            [[] for _ in self.widths], [[] for _ in self.widths], [[] for _ in self.widths],
            [0] * len(self.widths),
            [
                None if self.anti_aliases is None else self.anti_aliases[scale].initial_state(
                    batch, device=device, dtype=dtype
                )
                for scale in range(len(self.widths))
            ],
        )

    def step(
        self, bands: tuple[ScaleTensor | None, ...], state: BlockStreamState,
        memories: tuple[EideticMemory, ...] | list[EideticMemory] | None = None,
        *, absolute_position: int = 0, attention_enabled: tuple[bool, ...] | None = None,
        relational_context: Tensor | None = None,
    ) -> tuple[tuple[ScaleTensor | None, ...], BlockStreamState]:
        """Process only coefficients completing at one original-domain position."""

        if len(bands) != len(self.widths) or len(state.resonators) != len(self.widths):
            raise ValueError("stream block state or bands have the wrong number of scales")
        scheduled = self.attention_enabled if attention_enabled is None else attention_enabled
        if len(scheduled) != len(self.widths):
            raise ValueError("stream attention schedule has the wrong number of scales")
        exchanged, state.exchange = self.exchange.step(bands, state.exchange)
        step_times: list[Tensor | None] = []
        for scale, band in enumerate(exchanged):
            if band is None:
                step_times.append(None)
                continue
            time = band.data.new_full(band.mask.shape, absolute_position * band.base_interval)
            step_times.append(time)
            state.recent_features[scale].append(band.data)
            state.recent_masks[scale].append(band.mask)
            state.recent_times[scale].append(time)
            for cache in (
                state.recent_features[scale], state.recent_masks[scale], state.recent_times[scale]
            ):
                del cache[:-self.windows[scale]]
        output: list[ScaleTensor | None] = []
        for scale, band in enumerate(exchanged):
            if band is None:
                output.append(None)
                continue
            normalized = self.norms[scale](band.data) * band.mask.unsqueeze(-1)
            resonator_state = state.resonators[scale]
            resonator_module = self.resonators[scale]
            mixer_module = self.mixers[scale]
            band_mask = band.mask
            coefficient_interval = band.coefficient_interval
            prior_steps = resonator_state.steps

            def resonance_and_mixer(
                normalized_value: Tensor, state_value: Tensor, previous_drive: Tensor,
                *, resonator=resonator_module, mixer=mixer_module,
                mask=band_mask, interval=coefficient_interval, steps=prior_steps,
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
                # Bind every scale-dependent value in the closure.  Checkpoint
                # recomputation occurs during backward, after this loop has
                # advanced to later scales.
                resonant_value, next_resonator, _ = resonator.sequential(
                    normalized_value,
                    ResonatorState(state_value, previous_drive, steps),
                    mask, sample_interval=interval,
                )
                local_value = mixer(normalized_value)
                return (
                    resonant_value, next_resonator.value,
                    next_resonator.previous_drive, local_value,
                )

            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                resonant, next_value, next_drive, local = checkpoint(
                    resonance_and_mixer, normalized, resonator_state.value,
                    resonator_state.previous_drive, use_reentrant=False,
                )
            else:
                resonant, next_value, next_drive, local = resonance_and_mixer(
                    normalized, resonator_state.value, resonator_state.previous_drive
                )
            state.resonators[scale] = ResonatorState(
                next_value, next_drive, resonator_state.steps + normalized.shape[1]
            )
            if self.anti_aliases is not None:
                local_step, state.anti_alias[scale] = self.anti_aliases[scale].step(
                    local[:, 0], state.anti_alias[scale]
                )
                local = local_step.unsqueeze(1)
            local = local * band.mask.unsqueeze(-1)
            time = step_times[scale]
            if scheduled[scale]:
                candidates = AttentionCandidates(
                    torch.cat(state.recent_features[scale], 1),
                    torch.cat(state.recent_times[scale], 1),
                    torch.full_like(torch.cat(state.recent_times[scale], 1), float(band.scale)),
                    torch.cat(state.recent_masks[scale], 1),
                )
                if scale + 1 < len(self.widths) and state.recent_features[scale + 1]:
                    landmark_times = torch.cat(state.recent_times[scale + 1], 1)[:, -self.landmark_count :]
                    landmark_mask = torch.cat(state.recent_masks[scale + 1], 1)[:, -self.landmark_count :]
                    landmark = AttentionCandidates(
                        self.landmark_values[scale](
                            torch.cat(state.recent_features[scale + 1], 1)[:, -self.landmark_count :]
                        ),
                        landmark_times,
                        torch.full_like(landmark_times, float(scale + 1)),
                        landmark_mask & (landmark_times <= time),
                        torch.ones_like(landmark_mask, dtype=torch.long),
                    )
                    candidates = _join_candidates(candidates, landmark)
                if memories is not None:
                    query_event = (
                        torch.sigmoid(self.memory_query_gates[scale](band.data)).squeeze(-1) > 0.5
                    ) | (((state.scale_steps[scale] + 1) % self.windows[scale]) == 0)
                    query_event = query_event & band.mask
                    distant = _memory_candidates(
                        band, memories, self.memory_keys[scale], self.memory_signatures[scale],
                        self.memory_values[scale], self.retrieved_items,
                        absolute_positions=torch.full(
                            band.mask.shape, absolute_position, dtype=torch.long, device=band.data.device
                        ),
                        query_events=query_event,
                        resonant_attention=self.attentions[scale],
                    )
                    candidates = _join_candidates(candidates, distant)
                attended, _ = self.attentions[scale].attend(
                    band.data, candidates, time, torch.full_like(time, float(band.scale)), causal=True
                )
                attended = attended * band.mask.unsqueeze(-1)
            else:
                attended = torch.zeros_like(band.data)
            identity, branch = self._identity_and_branch(scale, normalized)
            delta = (
                branch[..., 0:1] * resonant + branch[..., 1:2] * local
                + branch[..., 2:3] * attended + branch[..., 3:4] * identity
            )
            if self.relational_projections is not None:
                if relational_context is None:
                    relational = torch.zeros_like(band.data)
                else:
                    if relational_context.ndim != 2 or relational_context.shape[0] != band.data.shape[0]:
                        raise ValueError("stream relational_context must have shape (batch,features)")
                    relational = self.relational_projections[scale](relational_context).unsqueeze(1)
                    relational = relational * band.mask.unsqueeze(-1)
                delta = delta + branch[..., 4:5] * relational
            updated = (band.data + self.layer_scale[scale] * delta) * band.mask.unsqueeze(-1)
            output.append(ScaleTensor(
                updated, band.mask, band.scale, band.sample_interval, band.support, band.kind
            ))
            state.scale_steps[scale] += 1
        return tuple(output), state

    def forward_aligned_chunk(
        self,
        bands: tuple[ScaleTensor, ...],
        state: BlockStreamState,
        *,
        absolute_start: int,
        attention_enabled: tuple[bool, ...] | None = None,
        relational_context: Tensor | None = None,
    ) -> tuple[tuple[ScaleTensor, ...], BlockStreamState]:
        """Process a complete-support chunk with exact stream continuation."""

        if (
            absolute_start < 0
            or len(bands) != len(self.widths)
            or len(state.resonators) != len(self.widths)
        ):
            raise ValueError("aligned block state, bands, or absolute start are invalid")
        scheduled = self.attention_enabled if attention_enabled is None else attention_enabled
        if len(scheduled) != len(self.widths):
            raise ValueError("stream attention schedule has the wrong number of scales")
        exchanged, state.exchange = self.exchange.forward_aligned_chunk(
            bands, state.exchange
        )

        histories: list[AttentionCandidates | None] = []
        current_times: list[Tensor] = []
        combined_features_by_scale: list[Tensor] = []
        combined_masks_by_scale: list[Tensor] = []
        combined_times_by_scale: list[Tensor] = []
        for scale, band in enumerate(exchanged):
            time = (
                absolute_start * band.base_interval
                + (
                    torch.arange(
                        band.data.shape[1], device=band.data.device,
                        dtype=band.data.dtype,
                    )
                    + 1
                )
                * band.coefficient_interval
                - band.base_interval
            ).expand(band.data.shape[0], -1)
            current_times.append(time)
            if state.recent_features[scale]:
                prior_features = torch.cat(state.recent_features[scale], 1)
                prior_masks = torch.cat(state.recent_masks[scale], 1)
                prior_times = torch.cat(state.recent_times[scale], 1)
                histories.append(AttentionCandidates(
                    prior_features,
                    prior_times,
                    torch.full_like(prior_times, float(band.scale)),
                    prior_masks,
                ))
                combined_features = torch.cat((prior_features, band.data), 1)
                combined_masks = torch.cat((prior_masks, band.mask), 1)
                combined_times = torch.cat((prior_times, time), 1)
            else:
                histories.append(None)
                combined_features, combined_masks, combined_times = (
                    band.data, band.mask, time,
                )
            keep = self.windows[scale]
            combined_features_by_scale.append(combined_features)
            combined_masks_by_scale.append(combined_masks)
            combined_times_by_scale.append(combined_times)
            state.recent_features[scale] = [combined_features[:, -keep:]]
            state.recent_masks[scale] = [combined_masks[:, -keep:]]
            state.recent_times[scale] = [combined_times[:, -keep:]]

        output: list[ScaleTensor] = []
        for scale, band in enumerate(exchanged):
            normalized = self.norms[scale](band.data) * band.mask.unsqueeze(-1)
            resonator_state = state.resonators[scale]
            resonator_module = self.resonators[scale]
            mixer_module = self.mixers[scale]
            band_mask = band.mask
            coefficient_interval = band.coefficient_interval
            prior_steps = resonator_state.steps

            def resonance_and_mixer(
                normalized_value: Tensor,
                state_value: Tensor,
                previous_drive: Tensor,
                *,
                resonator=resonator_module,
                mixer=mixer_module,
                mask=band_mask,
                interval=coefficient_interval,
                steps=prior_steps,
                compiled_core=self._compiled_chunk_cores.get(scale),
            ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
                if compiled_core is not None:
                    return compiled_core(
                        normalized_value,
                        state_value,
                        previous_drive,
                        mask,
                        normalized_value.new_tensor(interval),
                    )
                resonant_value, next_resonator, _ = resonator.parallel(
                    normalized_value,
                    ResonatorState(state_value, previous_drive, steps),
                    mask,
                    sample_interval=interval,
                )
                local_value = mixer(normalized_value)
                return (
                    resonant_value,
                    next_resonator.value,
                    next_resonator.previous_drive,
                    local_value,
                )

            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                resonant, next_value, next_drive, local = checkpoint(
                    resonance_and_mixer,
                    normalized,
                    resonator_state.value,
                    resonator_state.previous_drive,
                    use_reentrant=False,
                )
            else:
                resonant, next_value, next_drive, local = resonance_and_mixer(
                    normalized, resonator_state.value, resonator_state.previous_drive
                )
            state.resonators[scale] = ResonatorState(
                next_value,
                next_drive,
                resonator_state.steps + normalized.shape[1],
            )
            if self.anti_aliases is not None:
                anti_alias = self.anti_aliases[scale]
                anti_state = state.anti_alias[scale]
                if anti_state is None:
                    raise RuntimeError("causal anti-alias state is missing")
                up = local.new_zeros(
                    local.shape[0], local.shape[1] * anti_alias.factor, local.shape[2]
                )
                up[:, :: anti_alias.factor] = anti_alias.factor * local
                pre, pre_history = anti_alias._stateful_filter(
                    up, anti_state.pre_history
                )
                post, post_history = anti_alias._stateful_filter(
                    F.silu(pre), anti_state.post_history
                )
                local = post[:, anti_alias.factor - 1 :: anti_alias.factor]
                state.anti_alias[scale] = AntiAliasState(
                    pre_history, post_history
                )
            local = local * band.mask.unsqueeze(-1)

            if scheduled[scale]:
                additional_candidates = None
                if scale + 1 < len(self.widths):
                    additional_candidates = _stream_landmark_candidates(
                        band,
                        current_times[scale],
                        combined_features_by_scale[scale + 1],
                        combined_masks_by_scale[scale + 1],
                        combined_times_by_scale[scale + 1],
                        exchanged[scale + 1].scale,
                        self.landmark_values[scale],
                        self.landmark_count,
                    )
                attended, _ = self.attentions[scale].sliding_window_attend(
                    band.data,
                    band.mask,
                    window=self.windows[scale],
                    query_tile_size=self.attention_query_tile_size,
                    scale=band.scale,
                    sample_interval=band.base_interval,
                    coefficient_interval=band.coefficient_interval,
                    causal=True,
                    additional_candidates=additional_candidates,
                    history=histories[scale],
                    time_offset=absolute_start * band.base_interval,
                    return_weights=False,
                )
                attended = attended * band.mask.unsqueeze(-1)
            else:
                attended = torch.zeros_like(band.data)
            identity, branch = self._identity_and_branch(scale, normalized)
            delta = (
                branch[..., 0:1] * resonant
                + branch[..., 1:2] * local
                + branch[..., 2:3] * attended
                + branch[..., 3:4] * identity
            )
            if self.relational_projections is not None:
                if relational_context is None:
                    relational = torch.zeros_like(band.data)
                elif relational_context.ndim == 2:
                    if relational_context.shape[0] != band.data.shape[0]:
                        raise ValueError("relational context batch does not match bands")
                    relational = self.relational_projections[scale](
                        relational_context
                    ).unsqueeze(1).expand(-1, band.data.shape[1], -1)
                elif relational_context.ndim == 3:
                    if relational_context.shape[0] != band.data.shape[0]:
                        raise ValueError("relational context batch does not match bands")
                    context = self.relational_projections[scale](relational_context)
                    completion = (
                        (
                            torch.arange(
                                band.data.shape[1], device=band.data.device
                            )
                            + 1
                        )
                        * band.support
                        - 1
                    )
                    if context.shape[1] == 0 or int(completion.max()) >= context.shape[1]:
                        raise ValueError(
                            "sequence relational context does not cover chunk completions"
                        )
                    relational = context[:, completion]
                else:
                    raise ValueError(
                        "relational context must be (batch,features) or (batch,time,features)"
                    )
                relational = relational * band.mask.unsqueeze(-1)
                delta = delta + branch[..., 4:5] * relational
            updated = (
                band.data + self.layer_scale[scale] * delta
            ) * band.mask.unsqueeze(-1)
            output.append(ScaleTensor(
                updated,
                band.mask,
                band.scale,
                band.sample_interval,
                band.support,
                band.kind,
            ))
            state.scale_steps[scale] += band.data.shape[1]
        return tuple(output), state

    def forward(
        self, bands: tuple[ScaleTensor, ...], state: BlockState | None = None,
        memories: tuple[EideticMemory, ...] | list[EideticMemory] | None = None,
        attention_enabled: tuple[bool, ...] | None = None,
        relational_context: Tensor | None = None,
        collect_diagnostics: bool = True,
    ) -> tuple[tuple[ScaleTensor, ...], BlockState, BlockDiagnostics | None]:
        if len(bands) != len(self.widths):
            raise ValueError("block received the wrong number of scales")
        scheduled = self.attention_enabled if attention_enabled is None else attention_enabled
        if len(scheduled) != len(self.widths):
            raise ValueError("attention schedule has the wrong number of scales")
        exchanged = self.exchange(bands)
        if state is None:
            state = self.initial_state(bands[0].data.shape[0], device=bands[0].data.device, dtype=bands[0].data.dtype)
        if len(state.resonators) != len(bands):
            raise ValueError("block state received the wrong number of scales")

        outputs, states, branches, resonance_diagnostics, attention_diagnostics, mixer_diagnostics = [], [], [], [], [], []
        for scale, band in enumerate(exchanged):
            normalized = self.norms[scale](band.data) * band.mask.unsqueeze(-1)
            resonant, next_state, parameters = self.resonators[scale](
                normalized, state.resonators[scale], band.mask,
                sample_interval=band.coefficient_interval,
            )
            if self.reverse_resonators is not None:
                reverse, _, _ = self.reverse_resonators[scale](
                    normalized.flip(1), None, band.mask.flip(1),
                    sample_interval=band.coefficient_interval,
                )
                reverse = reverse.flip(1)
                gate = torch.sigmoid(self.bidirectional_gates[scale](normalized))
                resonant = gate * resonant + (1 - gate) * reverse
            mixer = self.mixers[scale]
            if isinstance(mixer, HybridSpectralMixer):
                local, mixer_diagnostic = mixer.forward_with_diagnostics(normalized)
            else:
                local, mixer_diagnostic = mixer(normalized), None
            if self.anti_aliases is not None:
                local = self.anti_aliases[scale](local)
            local = local * band.mask.unsqueeze(-1)
            if scheduled[scale] and band.data.shape[1]:
                additional_candidates = None
                if scale + 1 < len(exchanged) and exchanged[scale + 1].data.shape[1]:
                    additional_candidates = _landmark_candidates(
                        band, exchanged[scale + 1], self.landmark_values[scale],
                        min(self.landmark_count, max(1, exchanged[scale + 1].data.shape[1])),
                        causal=self.causal,
                    )
                if memories is not None:
                    positions = torch.arange(band.data.shape[1], device=band.data.device)
                    query_events = (
                        torch.sigmoid(self.memory_query_gates[scale](band.data)).squeeze(-1) > 0.5
                    ) | (((positions + 1) % self.windows[scale]) == 0).unsqueeze(0)
                    query_events = query_events & band.mask
                    distant = _memory_candidates(
                        band, memories, self.memory_keys[scale], self.memory_signatures[scale],
                        self.memory_values[scale], self.retrieved_items,
                        query_events=query_events,
                        resonant_attention=self.attentions[scale],
                    )
                    additional_candidates = (
                        distant if additional_candidates is None
                        else _join_candidates(additional_candidates, distant)
                    )
                attended, weights = self.attentions[scale].sliding_window_attend(
                    band.data, band.mask,
                    window=min(self.windows[scale], max(1, band.data.shape[1])),
                    query_tile_size=self.attention_query_tile_size,
                    scale=band.scale, sample_interval=band.base_interval,
                    coefficient_interval=band.coefficient_interval, causal=self.causal,
                    additional_candidates=additional_candidates,
                    return_weights=collect_diagnostics,
                )
                attended = attended * band.mask.unsqueeze(-1)
            else:
                attended, weights = torch.zeros_like(band.data), None
            identity, branch = self._identity_and_branch(scale, normalized)
            delta = (
                branch[..., 0:1] * resonant
                + branch[..., 1:2] * local
                + branch[..., 2:3] * attended
                + branch[..., 3:4] * identity
            )
            if self.relational_projections is not None:
                if relational_context is None:
                    relational = torch.zeros_like(band.data)
                elif relational_context.ndim == 2:
                    if relational_context.shape[0] != band.data.shape[0]:
                        raise ValueError("relational_context batch does not match bands")
                    relational = self.relational_projections[scale](relational_context).unsqueeze(1)
                    relational = relational.expand(-1, band.data.shape[1], -1)
                elif relational_context.ndim == 3:
                    if relational_context.shape[0] != band.data.shape[0]:
                        raise ValueError("relational_context batch does not match bands")
                    context = self.relational_projections[scale](relational_context)
                    completion = (
                        (torch.arange(band.data.shape[1], device=band.data.device) + 1)
                        * band.support - 1
                    ).clamp_max(max(0, context.shape[1] - 1))
                    relational = context[:, completion]
                else:
                    raise ValueError("relational_context must be (batch,features) or (batch,time,features)")
                relational = relational * band.mask.unsqueeze(-1)
                delta = delta + branch[..., 4:5] * relational
            updated = (band.data + self.layer_scale[scale] * delta) * band.mask.unsqueeze(-1)
            outputs.append(
                ScaleTensor(updated, band.mask, band.scale, band.sample_interval, band.support, band.kind)
            )
            states.append(next_state)
            if collect_diagnostics:
                branches.append(branch)
                resonance_diagnostics.append(parameters)
                attention_diagnostics.append(weights)
                mixer_diagnostics.append(mixer_diagnostic)
        return (
            tuple(outputs),
            BlockState(tuple(states)),
            None if not collect_diagnostics else BlockDiagnostics(
                tuple(branches), tuple(resonance_diagnostics), tuple(attention_diagnostics),
                tuple(mixer_diagnostics),
            ),
        )


class MRRN(nn.Module):
    """Reference transform-once multiresolution resonance network."""

    def __init__(self, config: MRRNConfig) -> None:
        super().__init__()
        if config.scales < 2:
            raise ValueError("the full MRRN requires at least two scales; use a local ablation for one scale")
        self.config = config
        self.encoder = nn.Linear(config.input_dim, config.model_dim)
        self.analysis = LiftingAnalysisBank(config.model_dim, config.scales - 1, config.lifting_kernel)
        scales = config.scale_configs()
        self.analysis_adapters = nn.ModuleList(nn.Linear(config.model_dim, scale.width) for scale in scales)
        self.synthesis_adapters = nn.ModuleList(nn.Linear(scale.width, config.model_dim) for scale in scales)
        blocks, schedules = [], []
        for layer in range(config.layers):
            schedule = tuple(
                (scale == config.scales - 1)
                or (scale == 0 and layer % 2 == 0)
                or (0 < scale < config.scales - 1 and layer % 3 == 0)
                for scale in range(config.scales)
            )
            schedules.append(schedule)
            blocks.append(
                blocks[0] if config.share_depth_parameters and blocks
                else MRRNBlock(config, layer_index=layer, attention_enabled=schedule)
            )
        self.blocks = nn.ModuleList(blocks)
        self.attention_schedules = tuple(schedules)
        self.raw_norm = nn.RMSNorm(config.model_dim)
        self.raw_mixer = GatedLocalMixer(
            config.model_dim, config.mixer_expansion, structured_rank=config.structured_mixer_rank
        )
        self.raw_gain = nn.Parameter(torch.tensor(config.residual_scale))
        self.scale_output_gain = nn.Parameter(torch.full((config.scales,), config.residual_scale))
        self.output_head = nn.Linear(config.model_dim, config.resolved_output_dim)
        self.global_head = (
            nn.Linear(scales[-1].width, config.resolved_output_dim)
            if config.enable_global_head else None
        )
        self.global_memory_adapter = (
            nn.Linear(config.model_dim, scales[-1].width)
            if config.enable_global_head else None
        )
        self.global_memory_gate = (
            nn.Linear(2 * scales[-1].width, scales[-1].width)
            if config.enable_global_head else None
        )
        self.boundary_embedding = nn.Parameter(torch.zeros(config.model_dim))
        self.memory_key = nn.Linear(config.model_dim, config.model_dim)
        self.memory_signature = nn.Linear(config.model_dim, config.model_dim)
        self.memory_value = nn.Linear(config.model_dim, config.model_dim)
        self.memory_write_policy = MemoryWritePolicy()

    def create_memories(self, batch: int) -> list[EideticMemory]:
        if batch <= 0:
            raise ValueError("batch must be positive")
        return [
            EideticMemory(
                self.config.memory_capacity, self.config.model_dim,
                self.config.model_dim, self.config.model_dim,
            )
            for _ in range(batch)
        ]

    def enable_compiled_tensor_cores(self, *, mode: str = "default") -> None:
        """Compile each unique block's pure chunk kernels exactly once."""

        seen: set[int] = set()
        for block in self.blocks:
            identity = id(block)
            if identity not in seen:
                block.enable_compiled_tensor_cores(mode=mode)
                seen.add(identity)

    def disable_compiled_tensor_cores(self) -> None:
        seen: set[int] = set()
        for block in self.blocks:
            identity = id(block)
            if identity not in seen:
                block.disable_compiled_tensor_cores()
                seen.add(identity)

    def write_memory_step(
        self,
        encoded: Tensor,
        features: Tensor,
        memories: tuple[EideticMemory, ...] | list[EideticMemory],
        *,
        timestamp: int,
        threshold: float = 0.5,
        scale: int = 0,
    ) -> Tensor:
        if encoded.ndim != 2 or encoded.shape[-1] != self.config.model_dim:
            raise ValueError("encoded memory values must have shape (batch,model_dim)")
        if features.shape != (encoded.shape[0], MemoryWritePolicy.feature_count):
            raise ValueError("write features must have shape (batch,5)")
        if len(memories) != encoded.shape[0] or timestamp < 0 or scale < 0:
            raise ValueError("memory batch, timestamp, and scale must be valid")
        scores = self.memory_write_policy(features)
        selected = scores >= threshold
        keys, signatures, values = self.memory_key(encoded), self.memory_signature(encoded), self.memory_value(encoded)
        for batch_index in selected.nonzero(as_tuple=False).flatten().tolist():
            memories[batch_index].write(MemoryItem(
                keys[batch_index], values[batch_index], signatures[batch_index], timestamp,
                scale, float(scores[batch_index].detach()),
            ))
        return selected

    def initial_state(self, batch: int, *, device=None, dtype=None) -> MRRNState:
        return MRRNState(tuple(block.initial_state(batch, device=device, dtype=dtype) for block in self.blocks))

    def initial_stream_state(
        self, batch: int, *, sample_interval: float = 1.0, device=None, dtype=None
    ) -> MRRNStreamState:
        if not self.config.causal:
            raise ValueError("incremental streaming is available only for a causal model")
        return MRRNStreamState(
            self.analysis.initial_stream_state(
                batch, sample_interval=sample_interval, device=device, dtype=dtype
            ),
            [block.initial_stream_state(batch, device=device, dtype=dtype) for block in self.blocks],
            [None] * self.config.scales,
            0, batch, sample_interval,
        )

    def _adapt_analysis(self, bands: tuple[ScaleTensor, ...]) -> tuple[ScaleTensor, ...]:
        return tuple(
            ScaleTensor(
                adapter(band.data) * band.mask.unsqueeze(-1),
                band.mask,
                band.scale,
                band.sample_interval,
                band.support,
                band.kind,
            )
            for adapter, band in zip(self.analysis_adapters, bands, strict=True)
        )

    def _adapt_active(
        self, bands: tuple[ScaleTensor | None, ...]
    ) -> tuple[ScaleTensor | None, ...]:
        return tuple(
            None if band is None else ScaleTensor(
                adapter(band.data) * band.mask.unsqueeze(-1), band.mask, band.scale,
                band.sample_interval, band.support, band.kind
            )
            for adapter, band in zip(self.analysis_adapters, bands, strict=True)
        )

    def step(
        self,
        x: Tensor,
        state: MRRNStreamState,
        mask: Tensor | None = None,
        *,
        memories: tuple[EideticMemory, ...] | list[EideticMemory] | None = None,
        write_features: Tensor | None = None,
        write_threshold: float = 0.5,
        soft_boundary: Tensor | None = None,
        relational_context: Tensor | None = None,
        project_output: bool = True,
    ) -> MRRNStepOutput:
        """Constant-state causal decode step, exactly matching batch output for the prefix."""

        if x.ndim == 3 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 2 or x.shape != (state.batch, self.config.input_dim):
            raise ValueError(f"expected stream input shape ({state.batch}, {self.config.input_dim})")
        if mask is None:
            mask = torch.ones(state.batch, dtype=torch.bool, device=x.device)
        elif mask.shape != (state.batch,) or mask.dtype != torch.bool:
            raise ValueError("stream mask must be boolean with shape (batch,)")
        if soft_boundary is None:
            soft_boundary = torch.zeros(state.batch, dtype=torch.bool, device=x.device)
        elif soft_boundary.shape != (state.batch,) or soft_boundary.dtype != torch.bool:
            raise ValueError("soft_boundary must be boolean with shape (batch,)")
        if bool(soft_boundary.any()):
            keep = ~soft_boundary.unsqueeze(1)
            for block_state in state.blocks:
                for scale_masks in block_state.recent_masks:
                    for index in range(len(scale_masks)):
                        scale_masks[index] = scale_masks[index] & keep
        encoded = (
            self.encoder(x) + soft_boundary.to(x.dtype).unsqueeze(-1) * self.boundary_embedding
        ) * mask.unsqueeze(-1)
        active, state.lifting = self.analysis.push(encoded, state.lifting, mask)
        active = self._adapt_active(active)
        for index, (block, schedule) in enumerate(
            zip(self.blocks, self.attention_schedules, strict=True)
        ):
            active, state.blocks[index] = block.step(
                active, state.blocks[index], memories, absolute_position=state.position,
                attention_enabled=schedule, relational_context=relational_context,
            )
        for scale, band in enumerate(active):
            if band is not None:
                state.latest_bands[scale] = band
        latent = encoded + self.raw_gain * self.raw_mixer(self.raw_norm(encoded))
        for scale, (adapter, band) in enumerate(
            zip(self.synthesis_adapters, state.latest_bands, strict=True)
        ):
            if band is not None:
                latent = latent + self.scale_output_gain[scale] * adapter(band.data[:, 0])
        prediction = (
            self.output_head(latent) * mask.unsqueeze(-1)
            if project_output else latent.new_zeros(state.batch, 0)
        )
        if write_features is not None:
            if memories is None:
                raise ValueError("write_features require memories")
            self.write_memory_step(
                encoded, write_features, memories, timestamp=state.position,
                threshold=write_threshold,
            )
        state.position += 1
        return MRRNStepOutput(prediction, active, state, latent)

    def forward_aligned_chunk(
        self,
        x: Tensor,
        state: MRRNStreamState,
        mask: Tensor | None = None,
        *,
        relational_context: Tensor | None = None,
        project_output: bool = True,
    ) -> MRRNChunkOutput:
        """Run the largest efficient causal unit without resetting local state."""

        if not self.config.causal:
            raise ValueError("aligned chunks require a causal model")
        if x.ndim != 3 or x.shape[0] != state.batch or x.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"expected aligned input shape ({state.batch}, time, {self.config.input_dim})"
            )
        alignment = 2 ** (self.config.scales - 1)
        if (
            x.shape[1] <= 0
            or x.shape[1] % alignment
            or state.position % alignment
        ):
            raise ValueError(
                f"aligned chunks and their stream position must align to {alignment} samples"
            )
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        elif mask.shape != x.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("chunk mask must be boolean with shape (batch,time)")
        if relational_context is not None:
            if relational_context.ndim == 2:
                if relational_context.shape[0] != state.batch:
                    raise ValueError("relational context batch does not match chunk")
            elif relational_context.ndim == 3:
                if relational_context.shape[:2] != x.shape[:2]:
                    raise ValueError(
                        "sequence relational context must match chunk batch/time"
                    )
            else:
                raise ValueError(
                    "relational context must be (batch,features) or (batch,time,features)"
                )
        encoded = self.encoder(x) * mask.unsqueeze(-1)
        bands, state.lifting = self.analysis.push_aligned_chunk(
            encoded, state.lifting, mask
        )
        active = self._adapt_analysis(bands)
        absolute_start = state.position
        for index, (block, schedule) in enumerate(
            zip(self.blocks, self.attention_schedules, strict=True)
        ):
            active, state.blocks[index] = block.forward_aligned_chunk(
                active,
                state.blocks[index],
                absolute_start=absolute_start,
                attention_enabled=schedule,
                relational_context=relational_context,
            )

        prior_bands = tuple(state.latest_bands)
        latent = encoded + self.raw_gain * self.raw_mixer(self.raw_norm(encoded))
        for scale, (adapter, band, prior) in enumerate(zip(
            self.synthesis_adapters, active, prior_bands, strict=True
        )):
            prior_data = None if prior is None else adapter(prior.data)
            contribution = _causal_expand_chunk(
                adapter(band.data), x.shape[1], band.support, prior_data
            )
            latent = latent + self.scale_output_gain[scale] * contribution
            state.latest_bands[scale] = ScaleTensor(
                band.data[:, -1:],
                band.mask[:, -1:],
                band.scale,
                band.sample_interval,
                band.support,
                band.kind,
            )
        prediction = (
            self.output_head(latent) * mask.unsqueeze(-1)
            if project_output else latent.new_zeros(state.batch, x.shape[1], 0)
        )
        state.position += x.shape[1]
        return MRRNChunkOutput(prediction, active, state, latent)

    def prefill(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        *,
        state: MRRNStreamState | None = None,
        sample_interval: float = 1.0,
        relational_context: Tensor | None = None,
        project_output: bool = True,
    ) -> MRRNPrefillOutput:
        """Prefill vectorially where aligned and exactly stream every remainder."""

        if x.ndim != 3 or x.shape[-1] != self.config.input_dim:
            raise ValueError(f"expected input shape (batch,time,{self.config.input_dim})")
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        elif mask.shape != x.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("prefill mask must be boolean with shape (batch,time)")
        if state is None:
            state = self.initial_stream_state(
                x.shape[0],
                sample_interval=sample_interval,
                device=x.device,
                dtype=x.dtype,
            )
        elif state.batch != x.shape[0]:
            raise ValueError("prefill stream state batch does not match input")
        if relational_context is not None and (
            relational_context.ndim not in (2, 3)
            or relational_context.shape[0] != x.shape[0]
            or (
                relational_context.ndim == 3
                and relational_context.shape[1] != x.shape[1]
            )
        ):
            raise ValueError("prefill relational context has an incompatible shape")

        alignment = 2 ** (self.config.scales - 1)
        cursor = 0
        predictions: list[Tensor] = []
        latents: list[Tensor] = []

        def token_context(position: int) -> Tensor | None:
            if relational_context is None or relational_context.ndim == 2:
                return relational_context
            return relational_context[:, position]

        while cursor < x.shape[1] and state.position % alignment:
            result = self.step(
                x[:, cursor],
                state,
                mask[:, cursor],
                relational_context=token_context(cursor),
                project_output=project_output,
            )
            state = result.state
            predictions.append(result.prediction.unsqueeze(1))
            if result.latent is None:
                raise RuntimeError("stream prefill step omitted its latent")
            latents.append(result.latent.unsqueeze(1))
            cursor += 1
        aligned_length = ((x.shape[1] - cursor) // alignment) * alignment
        if aligned_length:
            chunk_context = (
                relational_context
                if relational_context is None or relational_context.ndim == 2
                else relational_context[:, cursor : cursor + aligned_length]
            )
            chunk = self.forward_aligned_chunk(
                x[:, cursor : cursor + aligned_length],
                state,
                mask[:, cursor : cursor + aligned_length],
                relational_context=chunk_context,
                project_output=project_output,
            )
            state = chunk.state
            predictions.append(chunk.prediction)
            latents.append(chunk.latent)
            cursor += aligned_length
        while cursor < x.shape[1]:
            result = self.step(
                x[:, cursor],
                state,
                mask[:, cursor],
                relational_context=token_context(cursor),
                project_output=project_output,
            )
            state = result.state
            predictions.append(result.prediction.unsqueeze(1))
            if result.latent is None:
                raise RuntimeError("stream prefill step omitted its latent")
            latents.append(result.latent.unsqueeze(1))
            cursor += 1
        prediction_width = self.config.resolved_output_dim if project_output else 0
        prediction = (
            torch.cat(predictions, 1)
            if predictions
            else x.new_zeros(x.shape[0], 0, prediction_width)
        )
        latent = (
            torch.cat(latents, 1)
            if latents
            else x.new_zeros(x.shape[0], 0, self.config.model_dim)
        )
        return MRRNPrefillOutput(prediction, state, latent)

    def _offline_synthesis(
        self, bands: tuple[ScaleTensor, ...], context: ReconstructionContext
    ) -> Tensor:
        restored = tuple(
            ScaleTensor(
                adapter(band.data), band.mask, band.scale, band.sample_interval, band.support, band.kind
            )
            for adapter, band in zip(self.synthesis_adapters, bands, strict=True)
        )
        return self.analysis.inverse(restored, context)

    def _causal_synthesis(self, encoded: Tensor, bands: tuple[ScaleTensor, ...]) -> Tensor:
        fused = encoded + self.raw_gain * self.raw_mixer(self.raw_norm(encoded))
        for scale, (adapter, band) in enumerate(zip(self.synthesis_adapters, bands, strict=True)):
            contribution = _causal_expand(adapter(band.data), encoded.shape[1], band.support)
            fused = fused + self.scale_output_gain[scale] * contribution
        return fused

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        *,
        sample_interval: float = 1.0,
        state: MRRNState | None = None,
        output_mode: str = "sequence",
        memories: tuple[EideticMemory, ...] | list[EideticMemory] | None = None,
        relational_context: Tensor | None = None,
        project_output: bool = True,
        collect_diagnostics: bool = True,
    ) -> MRRNOutput:
        if x.ndim != 3 or x.shape[-1] != self.config.input_dim:
            raise ValueError(f"expected input shape (batch, time, {self.config.input_dim})")
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        elif mask.shape != x.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape (batch,time)")
        if output_mode not in {"sequence", "global", "operator"}:
            raise ValueError("output_mode must be sequence, global, or operator")
        if output_mode == "global" and self.global_head is None:
            raise ValueError("global output is disabled for this sequence-only model")
        encoded = self.encoder(x) * mask.unsqueeze(-1)
        raw_bands, context = self.analysis(
            encoded, mask, sample_interval=sample_interval, boundary="causal" if self.config.causal else "reflect"
        )
        bands = self._adapt_analysis(raw_bands)
        state = self.initial_state(x.shape[0], device=x.device, dtype=x.dtype) if state is None else state
        if len(state.blocks) != len(self.blocks):
            raise ValueError("model state has the wrong number of blocks")
        next_states, diagnostics = [], []
        for block, block_state, schedule in zip(
            self.blocks, state.blocks, self.attention_schedules, strict=True
        ):
            bands, next_state, diagnostic = block(
                bands, block_state, memories, attention_enabled=schedule,
                relational_context=relational_context,
                collect_diagnostics=collect_diagnostics,
            )
            next_states.append(next_state)
            if diagnostic is not None:
                diagnostics.append(diagnostic)

        if output_mode == "global":
            coarsest = bands[-1]
            denominator = coarsest.mask.sum(1, keepdim=True).clamp_min(1)
            pooled = (coarsest.data * coarsest.mask.unsqueeze(-1)).sum(1) / denominator
            if memories is not None:
                contexts = []
                for batch_index, memory in enumerate(memories):
                    items = memory.items(query_time=x.shape[1] - 1)
                    if items:
                        values = torch.stack([
                            item.value.to(device=x.device, dtype=x.dtype) for item in items
                        ])
                        contexts.append(self.global_memory_adapter(values).mean(0))
                    else:
                        contexts.append(torch.zeros_like(pooled[batch_index]))
                memory_context = torch.stack(contexts)
                pooled = pooled + torch.sigmoid(
                    self.global_memory_gate(torch.cat((pooled, memory_context), -1))
                ) * memory_context
            prediction = self.global_head(pooled)
        else:
            latent = (
                self._offline_synthesis(bands, context)
                if output_mode == "operator" or not self.config.causal
                else self._causal_synthesis(encoded, bands)
            )
            prediction = (
                self.output_head(latent) * mask.unsqueeze(-1)
                if project_output else latent.new_zeros(*latent.shape[:-1], 0)
            )
        return MRRNOutput(
            prediction, bands, MRRNState(tuple(next_states)), tuple(diagnostics), context,
            None if output_mode == "global" else latent,
        )
