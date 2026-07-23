"""Resonant Adjoint Surprise Learning for causal MRRN actors.

The critic reuses detached actor bands.  Its forward resonators estimate causal
consequences; separate outcome-conditioned resonators run from the end of a
completed trajectory toward its beginning and estimate adjoint credit.  No
critic, target-network, return-target, or replay gradient is allowed to enter
the actor graph.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from math import pi, sqrt
from pathlib import Path
from typing import Literal, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import MRRNConfig
from .lifting import ScaleTensor
from .mixer import HybridSpectralMixer, ResonantSpectralGLU
from .model import MRRN, MRRNOutput, _causal_expand
from .objectives import spectral_activation_regularization
from .optimization import OptimizerPolicy, build_adamw, clip_and_report_gradients
from .resonance import ComplexResonator
from .scale_exchange import ScaleExchange


@dataclass(frozen=True, slots=True)
class ResonantAdjointSurpriseConfig:
    """Validated training-system controls; defaults are the implemented design."""

    critic_width: int = 16
    minimum_critic_width: int = 4
    critic_layers: int = 1
    critic_scales: int = 3
    critic_heads: int = 2
    critic_modes: int = 4
    critic_mimo_rank: int = 1
    spectral_modes: int = 3
    spectral_basis_order: int = 4
    spectral_triads_per_mode: int = 1
    action_rank: int = 4
    latent_modes: int = 3
    horizons: tuple[int, ...] = (1, 4, 16)
    quantiles: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    bootstrap_heads: int = 4
    discount: float = 0.99
    ema_decay: float = 0.995
    calibration_decay: float = 0.98
    task_weight: float = 1.0
    surprise_cross_entropy_weight: float = 0.25
    trust_region_weight: float = 0.02
    spectral_regularization_weight: float = 1e-4
    critic_return_weight: float = 1.0
    critic_latent_weight: float = 0.25
    critic_reward_weight: float = 0.25
    critic_termination_weight: float = 0.1
    critic_adjoint_weight: float = 0.25
    critic_calibration_weight: float = 0.05
    critic_ranking_weight: float = 0.1
    return_surprise_weight: float = 1.0
    advantage_weight: float = 1.0
    adjoint_credit_weight: float = 0.5
    exploration_weight: float = 0.15
    surprise_temperature: float = 0.7
    maximum_surprise: float = 4.0
    maximum_gradient_norm: float = 1.0
    maximum_critic_parameter_fraction: float = 0.20
    replay_capacity: int = 32_768
    replay_priority_cap: float = 10.0
    replay_priority_alpha: float = 0.6
    replay_priority_fraction: float = 0.5
    performance_tolerance: float = 0.02
    require_external_reward: bool = True

    def __post_init__(self) -> None:
        positive_integers = (
            "critic_width", "minimum_critic_width", "critic_layers", "critic_scales",
            "critic_heads", "critic_modes", "critic_mimo_rank", "spectral_modes",
            "spectral_basis_order", "action_rank", "latent_modes", "bootstrap_heads",
            "replay_capacity",
        )
        if any(getattr(self, name) <= 0 for name in positive_integers):
            raise ValueError("critic, spectral, action, bootstrap, and replay dimensions must be positive")
        if self.minimum_critic_width > self.critic_width:
            raise ValueError("minimum_critic_width cannot exceed critic_width")
        if self.spectral_triads_per_mode < 0:
            raise ValueError("spectral_triads_per_mode cannot be negative")
        if not self.horizons or any(value <= 0 for value in self.horizons):
            raise ValueError("horizons must be a nonempty tuple of positive steps")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be unique and strictly increasing")
        if not self.quantiles or any(not 0 < value < 1 for value in self.quantiles):
            raise ValueError("quantiles must lie strictly inside (0,1)")
        if tuple(sorted(set(self.quantiles))) != self.quantiles:
            raise ValueError("quantiles must be unique and strictly increasing")
        if not 0 < self.discount <= 1 or not 0 <= self.ema_decay < 1:
            raise ValueError("discount and EMA decay are invalid")
        if not 0 <= self.calibration_decay < 1:
            raise ValueError("calibration decay must lie in [0,1)")
        nonnegative = (
            "task_weight", "surprise_cross_entropy_weight", "trust_region_weight",
            "spectral_regularization_weight", "critic_return_weight", "critic_latent_weight",
            "critic_reward_weight", "critic_termination_weight", "critic_adjoint_weight",
            "critic_calibration_weight", "critic_ranking_weight", "return_surprise_weight",
            "advantage_weight", "adjoint_credit_weight", "exploration_weight",
            "performance_tolerance",
        )
        if any(getattr(self, name) < 0 for name in nonnegative):
            raise ValueError("loss, surprise, and performance weights cannot be negative")
        if min(
            self.surprise_temperature, self.maximum_surprise, self.maximum_gradient_norm,
            self.maximum_critic_parameter_fraction, self.replay_priority_cap,
            self.replay_priority_alpha,
        ) <= 0:
            raise ValueError("temperature, limits, parameter fraction, and priority controls must be positive")
        if not 0 <= self.replay_priority_fraction <= 1:
            raise ValueError("replay_priority_fraction must lie in [0,1]")


@dataclass(frozen=True, slots=True)
class TrajectoryBatch:
    """One padded trajectory batch with transition-aligned rewards.

    ``rewards[:, t]`` and ``dones[:, t]`` describe the consequence of
    ``actions[:, t]``.  A mask may change from true to false once, but cannot
    reactivate, which makes padding and return construction unambiguous.
    """

    inputs: Tensor
    actions: Tensor
    rewards: Tensor
    dones: Tensor
    mask: Tensor | None = None
    task_targets: Tensor | None = None
    behavior_logits: Tensor | None = None
    reward_source: Literal["environment", "human", "verifier", "task_loss"] = "environment"
    importance_weights: Tensor | None = None

    def validated(self, *, input_dim: int, action_dim: int) -> "TrajectoryBatch":
        if self.inputs.ndim != 3 or self.inputs.shape[-1] != input_dim:
            raise ValueError(f"inputs must have shape (batch,time,{input_dim})")
        shape = self.inputs.shape[:2]
        if self.actions.shape != shape or self.actions.dtype != torch.long:
            raise ValueError("actions must be int64 with shape (batch,time)")
        if self.rewards.shape != shape or not self.rewards.is_floating_point():
            raise ValueError("rewards must be floating point with shape (batch,time)")
        if self.dones.shape != shape or self.dones.dtype != torch.bool:
            raise ValueError("dones must be boolean with shape (batch,time)")
        mask = self.mask
        if mask is None:
            mask = torch.ones(shape, dtype=torch.bool, device=self.inputs.device)
        elif mask.shape != shape or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape (batch,time)")
        if shape[1] > 1 and bool((mask[:, 1:] & ~mask[:, :-1]).any()):
            raise ValueError("trajectory padding mask cannot reactivate")
        valid_actions = self.actions[mask]
        if valid_actions.numel() and (
            int(valid_actions.min()) < 0 or int(valid_actions.max()) >= action_dim
        ):
            raise ValueError("valid actions lie outside the actor action space")
        if self.task_targets is not None and self.task_targets.shape != shape:
            raise ValueError("task_targets must match batch/time")
        if self.behavior_logits is not None and self.behavior_logits.shape != (*shape, action_dim):
            raise ValueError("behavior_logits must have shape (batch,time,actions)")
        if self.importance_weights is not None:
            if (
                self.importance_weights.shape != (shape[0],)
                or not self.importance_weights.is_floating_point()
                or not bool(torch.isfinite(self.importance_weights).all())
                or bool((self.importance_weights <= 0).any())
            ):
                raise ValueError("importance_weights must be finite positive floats with shape (batch,)")
        if self.reward_source not in {"environment", "human", "verifier", "task_loss"}:
            raise ValueError("unknown reward source")
        if not bool(torch.isfinite(self.inputs).all()) or not bool(torch.isfinite(self.rewards).all()):
            raise ValueError("trajectory inputs and rewards must be finite")
        if self.behavior_logits is not None and not bool(torch.isfinite(self.behavior_logits).all()):
            raise ValueError("behavior logits must be finite")
        tensors = (self.actions, self.rewards, self.dones, mask)
        optional = (self.task_targets, self.behavior_logits, self.importance_weights)
        if any(value.device != self.inputs.device for value in tensors) or any(
            value is not None and value.device != self.inputs.device for value in optional
        ):
            raise ValueError("every trajectory tensor must be on the same device")
        return replace(self, mask=mask)

    def detached_cpu(self) -> "TrajectoryBatch":
        def move(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.detach().cpu().clone()

        return TrajectoryBatch(
            move(self.inputs), move(self.actions), move(self.rewards), move(self.dones),
            move(self.mask), move(self.task_targets), move(self.behavior_logits), self.reward_source,
            move(self.importance_weights),
        )


def _select_scale_indices(total: int, requested: int) -> tuple[int, ...]:
    if total <= 0 or requested <= 0:
        raise ValueError("scale counts must be positive")
    count = min(total, requested)
    if count == 1:
        return (total - 1,)
    return tuple(round(index * (total - 1) / (count - 1)) for index in range(count))


def _support_expand(data: Tensor, target_length: int, support: int) -> Tensor:
    """Map completed-trajectory coefficients across their physical support."""

    if target_length < 0 or support <= 0:
        raise ValueError("target length and support are invalid")
    if target_length == 0:
        return data[:, :0]
    if data.shape[1] == 0:
        return data.new_zeros(data.shape[0], target_length, *data.shape[2:])
    index = torch.arange(target_length, device=data.device).div(support, rounding_mode="floor")
    return data[:, index.clamp_max(data.shape[1] - 1)]


class _FrozenSpectralProjection(nn.Module):
    """Non-collapsible deterministic map from a real actor band to complex modes."""

    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        row = torch.arange(1, 2 * modes + 1, dtype=torch.float64)[:, None]
        column = torch.arange(1, width + 1, dtype=torch.float64)[None, :]
        basis = torch.cos(pi * row * column / (width + 1))
        basis[1::2] = torch.sin(pi * row[1::2] * column / (width + 1))
        basis = F.normalize(basis, dim=-1).float()
        self.register_buffer("basis", basis)
        self.width, self.modes = width, modes

    def forward(self, value: Tensor) -> Tensor:
        if value.shape[-1] != self.width:
            raise ValueError("spectral target width mismatch")
        normalized = F.rms_norm(value, (self.width,))
        return F.linear(normalized, self.basis.to(value.dtype)).unflatten(-1, (self.modes, 2))


class _AdjointResonanceLayer(nn.Module):
    """Causal consequence update plus a disjoint reverse outcome adjoint."""

    def __init__(self, widths: Sequence[int], action_dim: int, config: ResonantAdjointSurpriseConfig) -> None:
        super().__init__()
        self.widths = tuple(widths)
        self.exchange = ScaleExchange(self.widths, causal=True)
        self.norms = nn.ModuleList(nn.RMSNorm(width) for width in self.widths)
        self.forward_resonators = nn.ModuleList(
            ComplexResonator(
                width, config.critic_heads, config.critic_modes, config.critic_mimo_rank,
                omega_max=pi / (2 ** (scale + 1)),
            )
            for scale, width in enumerate(self.widths)
        )
        self.adjoint_resonators = nn.ModuleList(
            ComplexResonator(
                width, config.critic_heads, config.critic_modes, config.critic_mimo_rank,
                omega_max=pi / (2 ** (scale + 1)),
            )
            for scale, width in enumerate(self.widths)
        )
        self.mixers = nn.ModuleList(
            HybridSpectralMixer(
                width, 1.5, config.critic_heads,
                min(config.spectral_modes, config.critic_modes), config.critic_mimo_rank,
                spectral_kwargs={
                    "basis_order": config.spectral_basis_order,
                    "triads_per_mode": config.spectral_triads_per_mode,
                    "frequency_max": pi / (2 ** (scale + 1)),
                },
            )
            for scale, width in enumerate(self.widths)
        )
        self.outcome_projections = nn.ModuleList(
            nn.Linear(action_dim + 2, width) for width in self.widths
        )
        self.forward_gates = nn.ModuleList(nn.Linear(width, 2) for width in self.widths)
        self.adjoint_gates = nn.ModuleList(nn.Linear(width, width) for width in self.widths)
        self.forward_scale = nn.Parameter(torch.full((len(self.widths),), 1e-2))
        self.adjoint_scale = nn.Parameter(torch.full((len(self.widths),), 1e-2))

    def forward(
        self,
        bands: tuple[ScaleTensor, ...],
        outcomes: tuple[Tensor, ...] | None,
    ) -> tuple[tuple[ScaleTensor, ...], tuple[ScaleTensor, ...]]:
        if outcomes is not None and len(outcomes) != len(bands):
            raise ValueError("one outcome grid is required per critic scale")
        exchanged = self.exchange(bands)
        causal_bands, adjoint_bands = [], []
        for scale, band in enumerate(exchanged):
            normalized = self.norms[scale](band.data) * band.mask.unsqueeze(-1)
            resonant, _, _ = self.forward_resonators[scale](
                normalized, mask=band.mask, sample_interval=band.coefficient_interval
            )
            local = self.mixers[scale](normalized) * band.mask.unsqueeze(-1)
            weights = torch.softmax(self.forward_gates[scale](normalized), -1)
            causal = (
                band.data + self.forward_scale[scale]
                * (weights[..., :1] * resonant + weights[..., 1:] * local)
            ) * band.mask.unsqueeze(-1)
            causal_band = ScaleTensor(
                causal, band.mask, band.scale, band.sample_interval, band.support, band.kind
            )
            causal_bands.append(causal_band)
            if outcomes is None:
                adjoint = torch.zeros_like(causal)
            else:
                aligned = outcomes[scale]
                if aligned.shape[:2] != causal.shape[:2]:
                    raise ValueError("outcome grid must match its critic scale")
                drive = self.norms[scale](
                    causal + self.outcome_projections[scale](aligned)
                ) * band.mask.unsqueeze(-1)
                reverse, _, _ = self.adjoint_resonators[scale](
                    drive.flip(1), mask=band.mask.flip(1),
                    sample_interval=band.coefficient_interval,
                )
                reverse = reverse.flip(1)
                gate = torch.sigmoid(self.adjoint_gates[scale](causal))
                adjoint = (
                    causal + self.adjoint_scale[scale] * gate * reverse
                ) * band.mask.unsqueeze(-1)
            adjoint_bands.append(ScaleTensor(
                adjoint, band.mask, band.scale, band.sample_interval, band.support, band.kind
            ))
        return tuple(causal_bands), tuple(adjoint_bands)


@dataclass(frozen=True, slots=True)
class AdjointCriticOutput:
    """Memory-efficient distributional output (action shifts are not duplicated per quantile)."""

    value_quantiles: Tensor
    action_values: Tensor
    reward_prediction: Tensor
    termination_logits: Tensor
    latent_predictions: tuple[Tensor, ...]
    latent_targets: tuple[Tensor, ...]
    adjoint_credit: Tensor
    forward_features: Tensor
    adjoint_features: Tensor
    epistemic_uncertainty: Tensor
    aleatoric_uncertainty: Tensor
    mask: Tensor

    def quantiles_for(self, actions: Tensor) -> Tensor:
        """Return (batch,time,bootstrap,horizon,quantile) for selected actions."""

        if actions.shape != self.mask.shape or actions.dtype != torch.long:
            raise ValueError("actions must be int64 and match critic batch/time")
        valid = actions[self.mask]
        if valid.numel() and (int(valid.min()) < 0 or int(valid.max()) >= self.action_values.shape[-1]):
            raise ValueError("valid critic actions are outside the action space")
        safe_actions = torch.where(self.mask, actions, 0)
        gather = safe_actions[:, :, None, None, None].expand(
            -1, -1, self.action_values.shape[2], self.action_values.shape[3], 1
        )
        shift = self.action_values.gather(-1, gather).squeeze(-1)
        return self.value_quantiles + shift.unsqueeze(-1)

    def mean_action_values(self) -> Tensor:
        return self.action_values.mean(2) + self.value_quantiles.mean((2, 4)).unsqueeze(-1)


class ResonantAdjointCritic(nn.Module):
    """Compact multiscale critic that consumes actor bands without relifting."""

    def __init__(
        self,
        actor_config: MRRNConfig,
        config: ResonantAdjointSurpriseConfig = ResonantAdjointSurpriseConfig(),
        *,
        width: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.action_dim = actor_config.resolved_output_dim
        if self.action_dim < 2:
            raise ValueError("surprise learning requires at least two actor actions")
        self.scale_indices = _select_scale_indices(actor_config.scales, config.critic_scales)
        actor_widths = tuple(actor_config.scale_configs()[index].width for index in self.scale_indices)
        critic_width = config.critic_width if width is None else width
        if critic_width < config.minimum_critic_width:
            raise ValueError("critic width is below the configured minimum")
        self.width = critic_width
        widths = (critic_width,) * len(self.scale_indices)
        self.input_adapters = nn.ModuleList(
            nn.Linear(source, critic_width) for source in actor_widths
        )
        self.layers = nn.ModuleList(
            _AdjointResonanceLayer(widths, self.action_dim, config)
            for _ in range(config.critic_layers)
        )
        self.forward_fusion = nn.ModuleList(nn.Linear(critic_width, critic_width) for _ in widths)
        self.adjoint_fusion = nn.ModuleList(nn.Linear(critic_width, critic_width) for _ in widths)
        self.policy_projection = nn.Linear(self.action_dim, critic_width)
        self.fusion_norm = nn.RMSNorm(critic_width)
        self.fusion_mixer = HybridSpectralMixer(
            critic_width, 1.5, config.critic_heads,
            min(config.spectral_modes, config.critic_modes), config.critic_mimo_rank,
            spectral_kwargs={
                "basis_order": config.spectral_basis_order,
                "triads_per_mode": config.spectral_triads_per_mode,
                "frequency_max": pi / 2,
            },
        )
        bootstrap, horizons, quantiles = (
            config.bootstrap_heads, len(config.horizons), len(config.quantiles)
        )
        self.quantile_head = nn.Linear(critic_width, bootstrap * horizons * quantiles)
        self.action_feature_head = nn.Linear(
            critic_width, bootstrap * horizons * config.action_rank
        )
        self.action_embeddings = nn.Parameter(torch.empty(
            bootstrap, horizons, self.action_dim, config.action_rank
        ))
        nn.init.normal_(self.action_embeddings, std=1 / sqrt(config.action_rank))
        self.reward_base = nn.Linear(critic_width, 1)
        self.reward_action = nn.Linear(critic_width, config.action_rank)
        self.reward_embeddings = nn.Parameter(torch.empty(self.action_dim, config.action_rank))
        self.termination_base = nn.Linear(critic_width, 1)
        self.termination_action = nn.Linear(critic_width, config.action_rank)
        self.termination_embeddings = nn.Parameter(torch.empty(self.action_dim, config.action_rank))
        nn.init.normal_(self.reward_embeddings, std=1 / sqrt(config.action_rank))
        nn.init.normal_(self.termination_embeddings, std=1 / sqrt(config.action_rank))
        self.latent_projections = nn.ModuleList(
            _FrozenSpectralProjection(source, config.latent_modes) for source in actor_widths
        )
        self.latent_heads = nn.ModuleList(
            nn.Linear(
                critic_width,
                len(config.horizons) * config.latent_modes * 2,
            )
            for _ in widths
        )
        self.latent_action_embeddings = nn.ParameterList(
            nn.Parameter(torch.zeros(
                self.action_dim, len(config.horizons), config.latent_modes, 2
            ))
            for _ in widths
        )
        self.adjoint_head = nn.Linear(critic_width, self.action_dim)

    @property
    def spectral_modules(self) -> tuple[ResonantSpectralGLU, ...]:
        return tuple(module for module in self.modules() if isinstance(module, ResonantSpectralGLU))

    @staticmethod
    def _align_outcome(outcome: Tensor, band: ScaleTensor) -> Tensor:
        if band.data.shape[1] == 0:
            return outcome[:, :0]
        completion = (torch.arange(band.data.shape[1], device=outcome.device) + 1) * band.support - 1
        return outcome[:, completion.clamp_max(outcome.shape[1] - 1)]

    def _prepare_bands(self, bands: Sequence[ScaleTensor]) -> tuple[ScaleTensor, ...]:
        if len(bands) <= max(self.scale_indices):
            raise ValueError("actor did not provide every selected critic scale")
        result = []
        for adapter, index in zip(self.input_adapters, self.scale_indices, strict=True):
            source = bands[index]
            data = adapter(source.data.detach()) * source.mask.unsqueeze(-1)
            result.append(ScaleTensor(
                data, source.mask.detach(), source.scale, source.sample_interval,
                source.support, source.kind,
            ))
        return tuple(result)

    def value_distribution(self, forward_features: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Apply the compact distributional readout to an existing critic representation."""

        if forward_features.ndim != 3 or forward_features.shape[-1] != self.width:
            raise ValueError("critic value features must have shape (batch,time,critic_width)")
        k, h, q, rank = (
            self.config.bootstrap_heads, len(self.config.horizons),
            len(self.config.quantiles), self.config.action_rank,
        )
        raw_quantiles = self.quantile_head(forward_features).unflatten(-1, (k, h, q))
        value_quantiles = torch.sort(raw_quantiles, dim=-1).values
        action_features = self.action_feature_head(forward_features).unflatten(-1, (k, h, rank))
        action_values = torch.einsum("btkhr,khar->btkha", action_features, self.action_embeddings)
        head_means = value_quantiles.mean(-1).unsqueeze(-1) + action_values
        epistemic = head_means.std(2, correction=0)
        aleatoric = (
            value_quantiles[..., -1] - value_quantiles[..., 0]
        ).mean(2).unsqueeze(-1)
        return value_quantiles, action_values, epistemic, aleatoric

    def forward(
        self,
        bands: Sequence[ScaleTensor],
        policy_logits: Tensor,
        *,
        actions: Tensor | None = None,
        rewards: Tensor | None = None,
        dones: Tensor | None = None,
        mask: Tensor | None = None,
        include_adjoint: bool = True,
    ) -> AdjointCriticOutput:
        if policy_logits.ndim != 3 or policy_logits.shape[-1] != self.action_dim:
            raise ValueError("policy_logits must have shape (batch,time,actions)")
        batch, length = policy_logits.shape[:2]
        if mask is None:
            mask = torch.ones((batch, length), dtype=torch.bool, device=policy_logits.device)
        if mask.shape != (batch, length) or mask.dtype != torch.bool:
            raise ValueError("critic mask must be boolean with shape (batch,time)")
        supplied = (actions is not None, rewards is not None, dones is not None)
        if include_adjoint and not all(supplied):
            raise ValueError("adjoint execution requires actions, rewards, and dones")
        if not include_adjoint and any(supplied):
            raise ValueError("outcomes must be omitted when include_adjoint is false")
        if include_adjoint:
            if actions.shape != (batch, length) or actions.dtype != torch.long:
                raise ValueError("critic actions must be int64 with shape (batch,time)")
            if rewards.shape != (batch, length) or dones.shape != (batch, length):
                raise ValueError("critic rewards and dones must match batch/time")
            if dones.dtype != torch.bool:
                raise ValueError("critic dones must be boolean")
            valid_actions = actions[mask]
            if valid_actions.numel() and (
                int(valid_actions.min()) < 0 or int(valid_actions.max()) >= self.action_dim
            ):
                raise ValueError("valid critic actions are outside the action space")
            safe_actions = torch.where(mask, actions, 0)
            one_hot = F.one_hot(safe_actions, self.action_dim).to(policy_logits.dtype)
            outcome_full = torch.cat(
                (
                    one_hot * mask.unsqueeze(-1),
                    (rewards.detach() * mask).unsqueeze(-1),
                    (dones.detach() & mask).to(policy_logits.dtype).unsqueeze(-1),
                ),
                -1,
            )
        else:
            outcome_full = None

        selected_actor_bands = tuple(bands[index] for index in self.scale_indices)
        causal_bands = self._prepare_bands(bands)
        adjoint_bands = tuple(
            ScaleTensor(torch.zeros_like(band.data), band.mask, band.scale, band.sample_interval, band.support, band.kind)
            for band in causal_bands
        )
        for layer in self.layers:
            outcomes = None
            if outcome_full is not None:
                outcomes = tuple(
                    self._align_outcome(outcome_full.detach(), band) for band in causal_bands
                )
            causal_bands, adjoint_bands = layer(causal_bands, outcomes)

        forward_parts, adjoint_parts = [], []
        for scale, band in enumerate(causal_bands):
            forward_parts.append(self.forward_fusion[scale](
                _causal_expand(band.data, length, band.support)
            ))
            adjoint_parts.append(self.adjoint_fusion[scale](
                _support_expand(adjoint_bands[scale].data, length, band.support)
            ))
        normalizer = sqrt(len(causal_bands) + 1)
        policy_context = self.policy_projection(policy_logits.detach())
        forward_features = self.fusion_norm(
            (sum(forward_parts) + policy_context) / normalizer
        ) * mask.unsqueeze(-1)
        forward_features = (
            forward_features + 1e-2 * self.fusion_mixer(forward_features)
        ) * mask.unsqueeze(-1)
        adjoint_features = (
            torch.zeros_like(forward_features)
            if outcome_full is None
            else (sum(adjoint_parts) / sqrt(max(1, len(adjoint_parts)))) * mask.unsqueeze(-1)
        )

        h = len(self.config.horizons)
        value_quantiles, action_values, epistemic, aleatoric = self.value_distribution(
            forward_features
        )
        reward_prediction = self.reward_base(forward_features) + torch.einsum(
            "btr,ar->bta", self.reward_action(forward_features), self.reward_embeddings
        )
        termination_logits = self.termination_base(forward_features) + torch.einsum(
            "btr,ar->bta", self.termination_action(forward_features), self.termination_embeddings
        )
        latent_predictions, latent_targets = [], []
        if actions is None:
            latent_actions = torch.zeros((batch, length), dtype=torch.long, device=policy_logits.device)
        else:
            latent_actions = torch.where(mask, actions, 0)
        for scale, (head, projection, source) in enumerate(zip(
            self.latent_heads, self.latent_projections, selected_actor_bands, strict=True
        )):
            base = head(forward_features).unflatten(
                -1, (h, self.config.latent_modes, 2)
            )
            action_offset = self.latent_action_embeddings[scale][latent_actions]
            latent_predictions.append(base + action_offset)
            signature = projection(source.data.detach())
            latent_targets.append(_support_expand(signature, length, source.support))
        adjoint_credit = self.adjoint_head(adjoint_features) * mask.unsqueeze(-1)
        return AdjointCriticOutput(
            value_quantiles, action_values, reward_prediction, termination_logits,
            tuple(latent_predictions), tuple(latent_targets), adjoint_credit,
            forward_features, adjoint_features, epistemic, aleatoric, mask,
        )


def multihorizon_returns(
    rewards: Tensor,
    dones: Tensor,
    mask: Tensor,
    horizons: Sequence[int],
    *,
    discount: float,
    bootstrap: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Construct masked n-step returns, optionally bootstrapped at each horizon."""

    if rewards.ndim != 2 or dones.shape != rewards.shape or mask.shape != rewards.shape:
        raise ValueError("rewards, dones, and mask must share (batch,time)")
    if dones.dtype != torch.bool or mask.dtype != torch.bool or not rewards.is_floating_point():
        raise ValueError("returns require floating rewards and boolean dones/mask")
    if not horizons or any(value <= 0 for value in horizons) or not 0 < discount <= 1:
        raise ValueError("return horizons and discount are invalid")
    if bootstrap is not None and bootstrap.shape != (*rewards.shape, len(horizons)):
        raise ValueError("bootstrap values must have shape (batch,time,horizons)")
    batch, length = rewards.shape
    returns, validity = [], []
    for horizon_index, horizon in enumerate(horizons):
        value = torch.zeros_like(rewards)
        alive = torch.ones_like(mask)
        observed = torch.zeros_like(mask)
        for offset in range(horizon):
            if offset >= length:
                break
            source = torch.arange(length, device=rewards.device) + offset
            in_range = source < length
            safe = source.clamp_max(max(0, length - 1))
            step_valid = mask[:, safe] & in_range.unsqueeze(0) & alive
            value = value + (discount**offset) * rewards[:, safe] * step_valid
            observed = observed | step_valid
            alive = alive & ~(dones[:, safe] & step_valid) & in_range.unsqueeze(0)
        if bootstrap is not None and horizon < length:
            source = torch.arange(length, device=rewards.device) + horizon
            in_range = source < length
            safe = source.clamp_max(max(0, length - 1))
            can_bootstrap = alive & mask[:, safe] & in_range.unsqueeze(0)
            value = value + (discount**horizon) * bootstrap[:, safe, horizon_index] * can_bootstrap
        returns.append(value)
        validity.append(observed & mask)
    return torch.stack(returns, -1), torch.stack(validity, -1)


def phase_aware_latent_error(
    prediction: Tensor,
    target: Tensor,
    *,
    amplitude_weight: float = 1.0,
    phase_weight: float = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    """Amplitude error plus circular phase error for paired-real modes."""

    if prediction.shape != target.shape or prediction.shape[-1] != 2:
        raise ValueError("phase-aware tensors must share shape and end in paired real/imag")
    if min(amplitude_weight, phase_weight) < 0 or eps <= 0:
        raise ValueError("phase-aware loss weights and epsilon are invalid")
    pred_energy = prediction.square().sum(-1)
    target_energy = target.square().sum(-1)
    pred_amplitude = pred_energy.clamp_min(eps).sqrt()
    target_amplitude = target_energy.clamp_min(eps).sqrt()
    amplitude = (pred_amplitude - target_amplitude).square()
    dot = (prediction * target).sum(-1)
    cosine = dot / (pred_amplitude * target_amplitude).clamp_min(eps)
    phase = 1 - cosine.clamp(-1, 1)
    target_present = target_energy > eps
    return amplitude_weight * amplitude + phase_weight * phase * target_present


def quantile_huber_loss(
    prediction: Tensor,
    target: Tensor,
    quantiles: Sequence[float],
    mask: Tensor,
    *,
    bootstrap_mask: Tensor | None = None,
    sample_weights: Tensor | None = None,
    kappa: float = 1.0,
) -> Tensor:
    """Bootstrap-distributional quantile regression with an exact mask contract."""

    if prediction.ndim != 5 or target.shape != prediction.shape[:2] + prediction.shape[3:4]:
        raise ValueError("quantile prediction/target shapes must be (B,T,K,H,Q)/(B,T,H)")
    if mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError("quantile mask must be boolean with shape (B,T,H)")
    if len(quantiles) != prediction.shape[-1] or kappa <= 0:
        raise ValueError("quantile coordinates and positive Huber kappa are required")
    if any(not 0 < value < 1 for value in quantiles):
        raise ValueError("quantiles must lie inside (0,1)")
    if bootstrap_mask is None:
        bootstrap_mask = torch.ones(
            prediction.shape[:3], dtype=torch.bool, device=prediction.device
        )
    if bootstrap_mask.shape != prediction.shape[:3] or bootstrap_mask.dtype != torch.bool:
        raise ValueError("bootstrap mask must be boolean with shape (B,T,K)")
    if sample_weights is None:
        sample_weights = prediction.new_ones(prediction.shape[0])
    if (
        sample_weights.shape != (prediction.shape[0],)
        or not sample_weights.is_floating_point()
        or not bool(torch.isfinite(sample_weights).all())
        or bool((sample_weights <= 0).any())
    ):
        raise ValueError("quantile sample weights must be finite positive floats with shape (batch,)")
    residual = target[:, :, None, :, None] - prediction
    absolute = residual.abs()
    huber = torch.where(
        absolute <= kappa, 0.5 * residual.square(), kappa * (absolute - 0.5 * kappa)
    )
    tau = prediction.new_tensor(quantiles).view(1, 1, 1, 1, -1)
    loss = (tau - (residual.detach() < 0).to(prediction.dtype)).abs() * huber / kappa
    valid = mask[:, :, None, :, None] & bootstrap_mask[:, :, :, None, None]
    weighted = valid * sample_weights[:, None, None, None, None]
    return (loss * weighted).sum() / weighted.sum().clamp_min(1) / prediction.shape[-1]


def differentiable_quantile_calibration_loss(
    prediction: Tensor,
    target: Tensor,
    quantiles: Sequence[float],
    mask: Tensor,
    *,
    smoothing: float = 0.1,
    sample_weights: Tensor | None = None,
) -> Tensor:
    """Smooth empirical-CDF calibration penalty for each horizon and quantile."""

    if smoothing <= 0:
        raise ValueError("calibration smoothing must be positive")
    if prediction.ndim != 5 or target.shape != prediction.shape[:2] + prediction.shape[3:4]:
        raise ValueError("calibration prediction/target shapes are incompatible")
    if mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError("calibration mask must be boolean with shape (B,T,H)")
    if len(quantiles) != prediction.shape[-1]:
        raise ValueError("calibration quantile coordinates are incomplete")
    if sample_weights is None:
        sample_weights = prediction.new_ones(prediction.shape[0])
    if (
        sample_weights.shape != (prediction.shape[0],)
        or not sample_weights.is_floating_point()
        or not bool(torch.isfinite(sample_weights).all())
        or bool((sample_weights <= 0).any())
    ):
        raise ValueError("calibration sample weights must be positive with shape (batch,)")
    target_expanded = target[:, :, None, :, None]
    soft_coverage = torch.sigmoid((prediction - target_expanded) / smoothing)
    valid = mask[:, :, None, :, None] * sample_weights[:, None, None, None, None]
    denominator = valid.sum((0, 1, 2)).clamp_min(1)
    coverage = (soft_coverage * valid).sum((0, 1, 2)) / denominator
    expected = prediction.new_tensor(quantiles).view(1, -1)
    return (coverage - expected).square().mean()


@dataclass(frozen=True, slots=True)
class CriticLossBreakdown:
    total: Tensor
    return_distribution: Tensor
    latent_phase: Tensor
    reward: Tensor
    termination: Tensor
    adjoint_consistency: Tensor
    calibration: Tensor
    counterfactual_ranking: Tensor


def critic_losses(
    output: AdjointCriticOutput,
    actions: Tensor,
    rewards: Tensor,
    dones: Tensor,
    returns: Tensor,
    return_mask: Tensor,
    target_policy: Tensor,
    config: ResonantAdjointSurpriseConfig,
    *,
    bootstrap_mask: Tensor | None = None,
    sample_weights: Tensor | None = None,
) -> tuple[CriticLossBreakdown, Tensor]:
    """All critic objectives and per-time/per-scale phase surprise."""

    if sample_weights is None:
        sample_weights = rewards.new_ones(rewards.shape[0])
    if (
        sample_weights.shape != (rewards.shape[0],)
        or not sample_weights.is_floating_point()
        or not bool(torch.isfinite(sample_weights).all())
        or bool((sample_weights <= 0).any())
    ):
        raise ValueError("critic sample weights must be positive with shape (batch,)")

    def weighted_mean(value: Tensor, valid: Tensor) -> Tensor:
        weight = valid * sample_weights[:, None]
        return (value * weight).sum() / weight.sum().clamp_min(1)

    selected_quantiles = output.quantiles_for(actions)
    safe_actions = torch.where(output.mask, actions, 0)
    distribution = quantile_huber_loss(
        selected_quantiles, returns, config.quantiles, return_mask,
        bootstrap_mask=bootstrap_mask, sample_weights=sample_weights,
    )
    time_mask = output.mask
    phase_rows, phase_losses = [], []
    for prediction, target in zip(
        output.latent_predictions, output.latent_targets, strict=True
    ):
        horizon_errors = []
        for horizon_index, horizon in enumerate(config.horizons):
            future = torch.arange(target.shape[1], device=target.device) + horizon
            in_range = future < target.shape[1]
            safe = future.clamp_max(max(0, target.shape[1] - 1))
            actual = target[:, safe]
            error = phase_aware_latent_error(
                prediction[:, :, horizon_index], actual
            ).mean(-1)
            valid = time_mask & time_mask[:, safe] & in_range.unsqueeze(0)
            horizon_errors.append(error * valid)
            phase_losses.append(weighted_mean(error, valid))
        phase_rows.append(torch.stack(horizon_errors, -1).mean(-1))
    phase_by_scale = torch.stack(phase_rows, -1)
    latent_phase = torch.stack(phase_losses).mean() if phase_losses else distribution * 0

    action_index = safe_actions.unsqueeze(-1)
    predicted_reward = output.reward_prediction.gather(-1, action_index).squeeze(-1)
    predicted_done = output.termination_logits.gather(-1, action_index).squeeze(-1)
    reward_loss = weighted_mean((predicted_reward - rewards).square(), time_mask)
    termination_loss = F.binary_cross_entropy_with_logits(
        predicted_done, dones.to(predicted_done.dtype), reduction="none"
    )
    termination_loss = weighted_mean(termination_loss, time_mask)

    mean_values = output.mean_action_values()
    expected = (target_policy.unsqueeze(2) * mean_values).sum(-1)
    value_action_index = safe_actions[:, :, None, None].expand(
        -1, -1, mean_values.shape[2], 1
    )
    chosen = mean_values.gather(-1, value_action_index).squeeze(-1)
    longest_return = returns[..., -1]
    credit_target = longest_return - expected[..., -1].detach()
    scale = credit_target[time_mask].std(correction=0).clamp_min(1e-3)
    credit_target = (credit_target / scale).clamp(-config.maximum_surprise, config.maximum_surprise)
    credit = output.adjoint_credit.gather(-1, action_index).squeeze(-1)
    adjoint = weighted_mean((credit - credit_target.detach()).square(), time_mask)

    calibration = differentiable_quantile_calibration_loss(
        selected_quantiles, returns, config.quantiles, return_mask,
        sample_weights=sample_weights,
    )
    realized_sign = torch.sign(longest_return - expected[..., -1].detach())
    counterfactual_margin = chosen[..., -1] - expected[..., -1]
    ranking = F.softplus(-realized_sign * counterfactual_margin)
    informative = time_mask & (realized_sign != 0)
    ranking = weighted_mean(ranking, informative)
    total = (
        config.critic_return_weight * distribution
        + config.critic_latent_weight * latent_phase
        + config.critic_reward_weight * reward_loss
        + config.critic_termination_weight * termination_loss
        + config.critic_adjoint_weight * adjoint
        + config.critic_calibration_weight * calibration
        + config.critic_ranking_weight * ranking
    )
    return CriticLossBreakdown(
        total, distribution, latent_phase, reward_loss, termination_loss,
        adjoint, calibration, ranking,
    ), phase_by_scale


class FunctionalSurpriseCalibrator(nn.Module):
    """EMA calibration authority for signed return and multiscale phase surprise."""

    def __init__(
        self, horizons: int, scales: int, *, decay: float = 0.98, eps: float = 1e-5
    ) -> None:
        super().__init__()
        if min(horizons, scales) <= 0 or not 0 <= decay < 1 or eps <= 0:
            raise ValueError("calibrator dimensions, decay, and epsilon are invalid")
        self.decay, self.eps = decay, eps
        self.register_buffer("return_mean", torch.zeros(horizons))
        self.register_buffer("return_variance", torch.ones(horizons))
        self.register_buffer("scale_error", torch.ones(scales))
        self.register_buffer("scale_calibration", torch.ones(scales))
        self.register_buffer("previous_scale_error", torch.ones(scales))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(
        self, residual: Tensor, scale_error: Tensor, mask: Tensor,
        sample_weights: Tensor | None = None,
    ) -> None:
        if residual.shape[:2] != mask.shape or residual.shape[-1] != self.return_mean.numel():
            raise ValueError("return residual does not match calibrator")
        if scale_error.shape[:2] != mask.shape or scale_error.shape[-1] != self.scale_error.numel():
            raise ValueError("scale error does not match calibrator")
        if mask.dtype != torch.bool:
            raise ValueError("calibration mask must be boolean")
        if sample_weights is None:
            sample_weights = residual.new_ones(residual.shape[0])
        if (
            sample_weights.shape != (residual.shape[0],)
            or not sample_weights.is_floating_point()
            or not bool(torch.isfinite(sample_weights).all())
            or bool((sample_weights <= 0).any())
        ):
            raise ValueError("calibration sample weights must be positive with shape (batch,)")
        if not bool(mask.any()):
            return
        expanded = mask.unsqueeze(-1) * sample_weights[:, None, None]
        count = expanded.sum((0, 1)).clamp_min(1)
        mean = (residual * expanded).sum((0, 1)) / count
        variance = ((residual - mean).square() * expanded).sum((0, 1)) / count
        current_scale = (scale_error * expanded).sum((0, 1)) / count
        calibration = ((scale_error - self.scale_error).abs() * expanded).sum((0, 1)) / count
        if int(self.updates) == 0:
            self.return_mean.copy_(mean)
            self.return_variance.copy_(variance.clamp_min(self.eps))
            self.scale_error.copy_(current_scale)
            self.previous_scale_error.copy_(current_scale)
            self.scale_calibration.copy_(calibration.clamp_min(self.eps))
        else:
            self.previous_scale_error.copy_(self.scale_error)
            self.return_mean.lerp_(mean, 1 - self.decay)
            self.return_variance.lerp_(variance.clamp_min(self.eps), 1 - self.decay)
            self.scale_error.lerp_(current_scale, 1 - self.decay)
            self.scale_calibration.lerp_(calibration.clamp_min(self.eps), 1 - self.decay)
        self.updates.add_(1)

    def standardize_returns(self, residual: Tensor) -> Tensor:
        return (residual - self.return_mean) / self.return_variance.add(self.eps).sqrt()

    def scale_weights(self) -> Tensor:
        reliability = self.scale_calibration.add(self.eps).reciprocal()
        return reliability / reliability.sum().clamp_min(self.eps)

    def learning_progress(self) -> Tensor:
        if int(self.updates) < 2:
            return self.scale_error.new_zeros(self.scale_error.shape)
        return (
            (self.previous_scale_error - self.scale_error).clamp_min(0)
            / self.previous_scale_error.abs().clamp_min(self.eps)
        ).clamp_max(1)


@dataclass(frozen=True, slots=True)
class FunctionalSurpriseTarget:
    distribution: Tensor
    score: Tensor
    signed_return_surprise: Tensor
    counterfactual_advantage: Tensor
    adjoint_credit: Tensor
    exploration_bonus: Tensor
    phase_surprise: Tensor
    scale_weights: Tensor
    learning_progress: Tensor
    controllability: Tensor


def functional_surprise_target(
    actor_logits: Tensor,
    target_logits: Tensor,
    critic: AdjointCriticOutput,
    actions: Tensor,
    returns: Tensor,
    phase_error: Tensor,
    calibrator: FunctionalSurpriseCalibrator,
    config: ResonantAdjointSurpriseConfig,
    *,
    update_calibration: bool = False,
    sample_weights: Tensor | None = None,
) -> FunctionalSurpriseTarget:
    """Build a bounded stop-gradient policy target from functional surprise."""

    if actor_logits.shape != target_logits.shape or actor_logits.ndim != 3:
        raise ValueError("actor and target logits must share (batch,time,actions)")
    if returns.shape[:2] != actions.shape or returns.shape[-1] != len(config.horizons):
        raise ValueError("returns do not match actions and configured horizons")
    if phase_error.shape[:2] != actions.shape or phase_error.shape[-1] != calibrator.scale_error.numel():
        raise ValueError("phase errors do not match the surprise calibrator")
    mask = critic.mask
    valid_actions = actions[mask]
    if valid_actions.numel() and (
        int(valid_actions.min()) < 0 or int(valid_actions.max()) >= actor_logits.shape[-1]
    ):
        raise ValueError("valid surprise actions are outside the action space")
    safe_actions = torch.where(mask, actions, 0)
    target_policy = torch.softmax(target_logits.detach(), -1)
    values = critic.mean_action_values().detach().mean(2)
    expected = (target_policy * values).sum(-1, keepdim=True)
    advantage = values - expected
    aleatoric = critic.aleatoric_uncertainty.detach().mean(2)
    advantage_scale = (
        advantage.square().mean(-1, keepdim=True).sqrt() + aleatoric.mean(-1, keepdim=True)
    ).clamp_min(1e-3)
    normalized_advantage = (advantage / advantage_scale).clamp(
        -config.maximum_surprise, config.maximum_surprise
    )

    gather = safe_actions.unsqueeze(-1)
    chosen = values.gather(-1, gather).squeeze(-1)
    residual = returns.detach() - chosen.unsqueeze(-1)
    standardized = calibrator.standardize_returns(residual)
    signed_return = standardized.mean(-1).clamp(
        -config.maximum_surprise, config.maximum_surprise
    )
    weights = calibrator.scale_weights()
    phase_surprise = (phase_error.detach() * weights).sum(-1)
    phase_normalizer = calibrator.scale_error.mul(weights).sum().clamp_min(1e-3)
    phase_surprise = (phase_surprise / phase_normalizer).clamp(0, config.maximum_surprise)

    one_hot = F.one_hot(safe_actions, actor_logits.shape[-1]).to(actor_logits.dtype)
    adjoint = critic.adjoint_credit.detach()
    adjoint_scale = adjoint.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-3)
    normalized_adjoint = (adjoint / adjoint_scale).clamp(
        -config.maximum_surprise, config.maximum_surprise
    ) * one_hot
    progress_by_scale = calibrator.learning_progress()
    progress = progress_by_scale.mul(weights).sum()
    epistemic = critic.epistemic_uncertainty.detach().mean(2)
    uncertainty_ratio = epistemic / (epistemic + aleatoric + 1e-5)
    controllability = torch.sigmoid(normalized_advantage.abs() - 1)
    exploration = uncertainty_ratio * progress * controllability
    realized = one_hot * signed_return.unsqueeze(-1) * (1 + phase_surprise.unsqueeze(-1))
    score = (
        config.return_surprise_weight * realized
        + config.advantage_weight * normalized_advantage
        + config.adjoint_credit_weight * normalized_adjoint
        + config.exploration_weight * exploration
    ).clamp(-config.maximum_surprise, config.maximum_surprise)
    log_target = F.log_softmax(target_logits.detach(), -1)
    distribution = torch.softmax(
        (log_target + score) / config.surprise_temperature, -1
    )
    distribution = torch.where(
        mask.unsqueeze(-1), distribution, target_policy
    ).detach()
    if not bool(torch.isfinite(distribution).all()) or not bool(torch.isfinite(score).all()):
        raise FloatingPointError("functional surprise target became non-finite")
    if update_calibration:
        calibrator.update(
            residual.detach(), phase_error.detach(), mask, sample_weights=sample_weights
        )
    return FunctionalSurpriseTarget(
        distribution, score.detach(), signed_return.detach(), normalized_advantage.detach(),
        normalized_adjoint.detach(), exploration.detach(), phase_surprise.detach(),
        weights.detach(), progress_by_scale.detach(), controllability.detach(),
    )


@dataclass(frozen=True, slots=True)
class ActorLossBreakdown:
    total: Tensor
    task: Tensor
    functional_cross_entropy: Tensor
    trust_region: Tensor
    spectral_regularization: Tensor


def actor_losses(
    actor: MRRN,
    actor_logits: Tensor,
    target_logits: Tensor,
    surprise: FunctionalSurpriseTarget,
    mask: Tensor,
    config: ResonantAdjointSurpriseConfig,
    *,
    task_targets: Tensor | None = None,
    task_loss: Tensor | None = None,
    sample_weights: Tensor | None = None,
) -> ActorLossBreakdown:
    """Task + FSCE + trust region + spectral regularization actor objective."""

    if mask.shape != actor_logits.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("actor loss mask must match batch/time")
    if task_targets is not None and task_loss is not None:
        raise ValueError("supply task_targets or an explicit task_loss, not both")
    if sample_weights is None:
        sample_weights = actor_logits.new_ones(actor_logits.shape[0])
    if (
        sample_weights.shape != (actor_logits.shape[0],)
        or not sample_weights.is_floating_point()
        or not bool(torch.isfinite(sample_weights).all())
        or bool((sample_weights <= 0).any())
    ):
        raise ValueError("actor sample weights must be positive with shape (batch,)")

    def weighted_mean(value: Tensor) -> Tensor:
        weight = mask * sample_weights[:, None]
        return (value * weight).sum() / weight.sum().clamp_min(1)
    if task_loss is not None:
        if task_loss.numel() != 1 or not bool(torch.isfinite(task_loss)):
            raise ValueError("explicit task loss must be a finite scalar")
        task = task_loss
    elif task_targets is not None:
        if task_targets.shape != mask.shape or task_targets.dtype != torch.long:
            raise ValueError("classification task targets must be int64 and match batch/time")
        safe_targets = torch.where(mask, task_targets, 0)
        raw = F.cross_entropy(
            actor_logits.flatten(0, 1), safe_targets.flatten(), reduction="none"
        ).reshape_as(mask)
        task = weighted_mean(raw)
    else:
        task = actor_logits.sum() * 0
    log_policy = F.log_softmax(actor_logits, -1)
    fsce_rows = -(surprise.distribution * log_policy).sum(-1)
    fsce = weighted_mean(fsce_rows)
    target_policy = torch.softmax(target_logits.detach(), -1)
    kl_rows = (
        target_policy * (target_policy.clamp_min(1e-8).log() - log_policy)
    ).sum(-1)
    trust = weighted_mean(kl_rows)
    spectral_modules = tuple(
        module for module in actor.modules() if isinstance(module, ResonantSpectralGLU)
    )
    spectral = (
        spectral_activation_regularization(spectral_modules)
        if spectral_modules else actor_logits.sum() * 0
    )
    total = (
        config.task_weight * task
        + config.surprise_cross_entropy_weight * fsce
        + config.trust_region_weight * trust
        + config.spectral_regularization_weight * spectral
    )
    return ActorLossBreakdown(total, task, fsce, trust, spectral)


@dataclass(frozen=True, slots=True)
class ReplaySample:
    batch: TrajectoryBatch
    indices: tuple[int, ...]
    importance_weights: Tensor


@dataclass(slots=True)
class _ReplayItem:
    trajectory: TrajectoryBatch
    priority: float
    sequence: int


class PrioritizedTrajectoryReplay:
    """Bounded CPU replay with capped surprise/learnability/controllability priority."""

    def __init__(
        self,
        capacity: int,
        *,
        priority_cap: float = 10.0,
        priority_alpha: float = 0.6,
        prioritized_fraction: float = 0.5,
    ) -> None:
        if min(capacity, priority_cap, priority_alpha) <= 0:
            raise ValueError("replay capacity, cap, and alpha must be positive")
        if not 0 <= prioritized_fraction <= 1:
            raise ValueError("prioritized replay fraction must lie in [0,1]")
        self.capacity = capacity
        self.priority_cap = priority_cap
        self.priority_alpha = priority_alpha
        self.prioritized_fraction = prioritized_fraction
        self._items: list[_ReplayItem] = []
        self._transitions = 0
        self._sequence = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def transition_count(self) -> int:
        return self._transitions

    @property
    def priorities(self) -> tuple[float, ...]:
        return tuple(item.priority for item in self._items)

    def add(
        self,
        batch: TrajectoryBatch,
        functional_surprise: Tensor,
        learnability: Tensor,
        controllability: Tensor,
    ) -> tuple[int, ...]:
        if batch.mask is None:
            raise ValueError("replay requires a validated trajectory mask")
        shape = batch.inputs.shape[:2]
        for name, value in (
            ("functional surprise", functional_surprise),
            ("learnability", learnability),
            ("controllability", controllability),
        ):
            if value.shape != shape:
                raise ValueError(f"{name} must match trajectory batch/time")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        inserted = []
        cpu = batch.detached_cpu()
        for row in range(shape[0]):
            length = int(cpu.mask[row].sum())
            if length == 0:
                continue
            def part(value: Tensor | None) -> Tensor | None:
                return None if value is None else value[row : row + 1, :length].clone()

            trajectory = TrajectoryBatch(
                part(cpu.inputs), part(cpu.actions), part(cpu.rewards), part(cpu.dones),
                part(cpu.mask), part(cpu.task_targets), part(cpu.behavior_logits),
                cpu.reward_source,
            )
            valid = batch.mask[row]
            raw_priority = (
                functional_surprise[row, valid].abs()
                * learnability[row, valid].clamp(0, 1)
                * controllability[row, valid].clamp(0, 1)
            ).mean()
            priority = float(raw_priority.detach().clamp(1e-6, self.priority_cap).cpu())
            self._items.append(_ReplayItem(trajectory, priority, self._sequence))
            self._transitions += length
            inserted.append(len(self._items) - 1)
            self._sequence += 1
        while self._transitions > self.capacity and self._items:
            removed = self._items.pop(0)
            self._transitions -= removed.trajectory.inputs.shape[1]
            inserted = [index - 1 for index in inserted if index > 0]
        return tuple(inserted)

    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> ReplaySample:
        if batch_size <= 0 or not self._items:
            raise ValueError("positive sample size and nonempty replay are required")
        count = min(batch_size, len(self._items))
        priorities = torch.tensor([item.priority for item in self._items], dtype=torch.float64)
        probabilities = priorities.pow(self.priority_alpha)
        probabilities /= probabilities.sum()
        prioritized_count = min(count, round(count * self.prioritized_fraction))
        chosen: list[int] = []
        if prioritized_count:
            chosen.extend(torch.multinomial(
                probabilities, prioritized_count, replacement=False, generator=generator
            ).tolist())
        remaining = [index for index in range(len(self._items)) if index not in chosen]
        uniform_count = count - len(chosen)
        if uniform_count:
            order = torch.randperm(len(remaining), generator=generator)[:uniform_count]
            chosen.extend(remaining[index] for index in order.tolist())
        rows = [self._items[index].trajectory for index in chosen]
        maximum = max(row.inputs.shape[1] for row in rows)

        def padded(name: str, fill=0):
            values = [getattr(row, name) for row in rows]
            if any(value is None for value in values):
                return None
            suffix = values[0].shape[2:]
            result = values[0].new_full((count, maximum, *suffix), fill)
            for index, value in enumerate(values):
                result[index, : value.shape[1]] = value[0]
            return result.to(device=device) if device is not None else result

        sources = {row.reward_source for row in rows}
        source = next(iter(sources)) if len(sources) == 1 else "environment"
        sampled = TrajectoryBatch(
            padded("inputs"), padded("actions"), padded("rewards"), padded("dones", False),
            padded("mask", False), padded("task_targets"), padded("behavior_logits"), source,
        )
        selected_probability = probabilities[torch.tensor(chosen)]
        importance = (len(self._items) * selected_probability).pow(-1)
        importance /= importance.max().clamp_min(1e-12)
        if device is not None:
            importance = importance.to(device)
        importance = importance.to(torch.float32)
        sampled = replace(sampled, importance_weights=importance)
        return ReplaySample(sampled, tuple(chosen), importance)

    def update_priorities(self, indices: Sequence[int], priorities: Tensor) -> None:
        if priorities.ndim != 1 or priorities.numel() != len(indices):
            raise ValueError("one new priority is required per replay index")
        if not bool(torch.isfinite(priorities).all()) or bool((priorities < 0).any()):
            raise ValueError("replay priorities must be finite and nonnegative")
        for index, priority in zip(indices, priorities.tolist(), strict=True):
            if not 0 <= index < len(self._items):
                raise ValueError("replay index is out of range")
            self._items[index].priority = min(self.priority_cap, max(1e-6, float(priority)))

    def state_dict(self) -> dict:
        def encode(batch: TrajectoryBatch) -> dict:
            return {
                "inputs": batch.inputs, "actions": batch.actions, "rewards": batch.rewards,
                "dones": batch.dones, "mask": batch.mask,
                "task_targets": batch.task_targets, "behavior_logits": batch.behavior_logits,
                "reward_source": batch.reward_source, "importance_weights": None,
            }

        return {
            "capacity": self.capacity, "priority_cap": self.priority_cap,
            "priority_alpha": self.priority_alpha,
            "prioritized_fraction": self.prioritized_fraction,
            "transitions": self._transitions, "sequence": self._sequence,
            "items": [
                {"trajectory": encode(item.trajectory), "priority": item.priority,
                 "sequence": item.sequence}
                for item in self._items
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        expected = (
            self.capacity, self.priority_cap, self.priority_alpha, self.prioritized_fraction
        )
        actual = (
            state.get("capacity"), state.get("priority_cap"),
            state.get("priority_alpha"), state.get("prioritized_fraction"),
        )
        if actual != expected:
            raise ValueError("replay checkpoint controls do not match this learner")
        items = []
        transitions = 0
        for encoded in state.get("items", []):
            trajectory = TrajectoryBatch(**encoded["trajectory"])
            length = trajectory.inputs.shape[1]
            if length <= 0 or trajectory.inputs.shape[0] != 1:
                raise ValueError("replay checkpoint contains an invalid trajectory")
            priority = float(encoded["priority"])
            if not 0 < priority <= self.priority_cap:
                raise ValueError("replay checkpoint contains an invalid priority")
            items.append(_ReplayItem(trajectory, priority, int(encoded["sequence"])))
            transitions += length
        if transitions != int(state.get("transitions", -1)) or transitions > self.capacity:
            raise ValueError("replay checkpoint transition count is inconsistent")
        self._items = items
        self._transitions = transitions
        self._sequence = int(state.get("sequence", 0))


class PerformanceGuard:
    """Reject actor updates when proxy surprise improves while realized reward regresses."""

    def __init__(self, tolerance: float = 0.02) -> None:
        if tolerance < 0:
            raise ValueError("performance tolerance cannot be negative")
        self.tolerance = tolerance
        self.reference_performance: float | None = None
        self.best_surprise_loss: float | None = None
        self.rejections = 0

    def allows(self, performance: float, surprise_loss: float) -> bool:
        if not torch.isfinite(torch.tensor([performance, surprise_loss])).all():
            raise ValueError("performance guard inputs must be finite")
        if self.reference_performance is None:
            self.reference_performance = performance
            self.best_surprise_loss = surprise_loss
            return True
        tolerance = self.tolerance * max(1.0, abs(self.reference_performance))
        proxy_improved = surprise_loss < self.best_surprise_loss
        reward_regressed = performance < self.reference_performance - tolerance
        if proxy_improved and reward_regressed:
            self.rejections += 1
            return False
        if performance >= self.reference_performance - tolerance:
            self.reference_performance = max(self.reference_performance, performance)
            self.best_surprise_loss = min(self.best_surprise_loss, surprise_loss)
        return True

    def state_dict(self) -> dict[str, float | int | None]:
        return {
            "tolerance": self.tolerance,
            "reference_performance": self.reference_performance,
            "best_surprise_loss": self.best_surprise_loss,
            "rejections": self.rejections,
        }

    def load_state_dict(self, state: dict) -> None:
        if float(state.get("tolerance", -1)) != self.tolerance:
            raise ValueError("performance-guard tolerance does not match this learner")
        reference = state.get("reference_performance")
        surprise = state.get("best_surprise_loss")
        if (reference is None) != (surprise is None):
            raise ValueError("performance-guard checkpoint is incomplete")
        if reference is not None and not bool(torch.isfinite(torch.tensor([reference, surprise])).all()):
            raise ValueError("performance-guard checkpoint is non-finite")
        rejections = int(state.get("rejections", -1))
        if rejections < 0:
            raise ValueError("performance-guard rejection count is invalid")
        self.reference_performance = None if reference is None else float(reference)
        self.best_surprise_loss = None if surprise is None else float(surprise)
        self.rejections = rejections


@dataclass(frozen=True, slots=True)
class RASLParameterReport:
    actor: int
    critic: int
    target_actor: int
    target_critic: int
    critic_fraction: float
    selected_width: int
    selected_scales: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RASLLosses:
    actor: ActorLossBreakdown
    critic: CriticLossBreakdown
    surprise: FunctionalSurpriseTarget
    returns: Tensor
    return_mask: Tensor
    actor_output: MRRNOutput
    target_logits: Tensor


@dataclass(frozen=True, slots=True)
class RASLStepReport:
    actor_loss: float
    critic_loss: float
    task_loss: float
    functional_cross_entropy: float
    mean_reward: float
    mean_absolute_surprise: float
    actor_update_applied: bool
    actor_gradient_norm: float
    critic_gradient_norm: float
    replay_size: int


class ResonantAdjointSurpriseLearner(nn.Module):
    """Complete actor/critic/targets/calibration/replay training authority."""

    def __init__(
        self,
        actor: MRRN,
        config: ResonantAdjointSurpriseConfig = ResonantAdjointSurpriseConfig(),
    ) -> None:
        super().__init__()
        if not isinstance(actor, MRRN) or not actor.config.causal:
            raise ValueError("RASL requires a causal MRRN actor")
        if actor.config.resolved_output_dim < 2:
            raise ValueError("RASL actor output must represent at least two actions")
        self.actor = actor
        self.config = config
        actor_parameters = sum(parameter.numel() for parameter in actor.parameters())
        critic = None
        for width in range(config.critic_width, config.minimum_critic_width - 1, -1):
            candidate = ResonantAdjointCritic(actor.config, config, width=width)
            candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())
            if candidate_parameters / actor_parameters <= config.maximum_critic_parameter_fraction:
                critic = candidate
                break
        if critic is None:
            raise ValueError(
                "the minimum critic exceeds the parameter budget; increase actor capacity, "
                "raise maximum_critic_parameter_fraction, or reduce action/critic dimensions"
            )
        self.critic = critic
        self.target_actor = deepcopy(actor).requires_grad_(False).eval()
        self.target_critic = deepcopy(critic).requires_grad_(False).eval()
        self.calibrator = FunctionalSurpriseCalibrator(
            len(config.horizons), len(critic.scale_indices), decay=config.calibration_decay
        )
        self.replay = PrioritizedTrajectoryReplay(
            config.replay_capacity, priority_cap=config.replay_priority_cap,
            priority_alpha=config.replay_priority_alpha,
            prioritized_fraction=config.replay_priority_fraction,
        )
        self.performance_guard = PerformanceGuard(config.performance_tolerance)

    def train(self, mode: bool = True):
        super().train(mode)
        # EMA targets are always deterministic inference authorities.
        self.target_actor.eval()
        self.target_critic.eval()
        return self

    def parameter_report(self) -> RASLParameterReport:
        count = lambda module: sum(parameter.numel() for parameter in module.parameters())
        actor, critic = count(self.actor), count(self.critic)
        return RASLParameterReport(
            actor, critic, count(self.target_actor), count(self.target_critic),
            critic / actor, self.critic.width, self.critic.scale_indices,
        )

    def make_optimizers(
        self,
        *,
        actor_policy: OptimizerPolicy | None = None,
        critic_policy: OptimizerPolicy | None = None,
    ) -> tuple[torch.optim.AdamW, torch.optim.AdamW]:
        actor_policy = OptimizerPolicy() if actor_policy is None else actor_policy
        critic_policy = (
            OptimizerPolicy(learning_rate=actor_policy.learning_rate)
            if critic_policy is None else critic_policy
        )
        return build_adamw(self.actor, actor_policy), build_adamw(self.critic, critic_policy)

    @torch.no_grad()
    def rollout_policy(
        self, inputs: Tensor, mask: Tensor | None = None
    ) -> MRRNOutput:
        """Run the stable EMA actor used to collect behavior logits and actions."""

        self.target_actor.eval()
        return self.target_actor(inputs, mask)

    @torch.no_grad()
    def update_targets(self, *, actor_updated: bool = True) -> None:
        def update(target: nn.Module, online: nn.Module, decay: float) -> None:
            target_parameters = dict(target.named_parameters())
            for name, parameter in online.named_parameters():
                target_parameters[name].mul_(decay).add_(parameter, alpha=1 - decay)
            target_buffers = dict(target.named_buffers())
            for name, buffer in online.named_buffers():
                if name in target_buffers:
                    target_buffers[name].copy_(buffer)

        if actor_updated:
            update(self.target_actor, self.actor, self.config.ema_decay)
        update(self.target_critic, self.critic, self.config.ema_decay)

    def _bootstrap_mask(self, batch: int, length: int, device) -> Tensor:
        result = torch.rand(
            batch, length, self.config.bootstrap_heads, device=device
        ) >= 0.2
        result[..., 0] = True
        return result

    def compute_losses(
        self,
        batch: TrajectoryBatch,
        *,
        task_loss: Tensor | None = None,
        update_calibration: bool = False,
        bootstrap_mask: Tensor | None = None,
    ) -> RASLLosses:
        batch = batch.validated(
            input_dim=self.actor.config.input_dim,
            action_dim=self.actor.config.resolved_output_dim,
        )
        if self.config.require_external_reward and batch.reward_source == "task_loss":
            raise ValueError(
                "functional surprise requires downstream external consequences; "
                "task-loss-as-reward would collapse to hard-example weighting"
            )
        actor_output = self.actor(batch.inputs, batch.mask)
        # Rollouts should retain EMA behavior logits.  When they are absent
        # (for example in fixed offline data), the detached pre-update actor is
        # the exact trust anchor.
        target_logits = (
            actor_output.prediction.detach()
            if batch.behavior_logits is None else batch.behavior_logits.detach()
        )
        critic_output = self.critic(
            actor_output.bands, actor_output.prediction,
            actions=batch.actions, rewards=batch.rewards, dones=batch.dones,
            mask=batch.mask, include_adjoint=True,
        )
        with torch.no_grad():
            # Semi-gradient target: reuse the detached online multiscale critic
            # representation but read it through EMA distributional heads.  It
            # preserves a stable target without a second resonant backbone pass.
            target_quantiles, target_action_values, _, _ = self.target_critic.value_distribution(
                critic_output.forward_features.detach()
            )
            target_values = (
                target_action_values.mean(2)
                + target_quantiles.mean((2, 4)).unsqueeze(-1)
            )
            target_policy = torch.softmax(target_logits, -1)
            bootstrap = (target_policy.unsqueeze(2) * target_values).sum(-1)
            returns, return_mask = multihorizon_returns(
                batch.rewards, batch.dones, batch.mask, self.config.horizons,
                discount=self.config.discount, bootstrap=bootstrap,
            )
        if bootstrap_mask is None:
            bootstrap_mask = self._bootstrap_mask(
                batch.inputs.shape[0], batch.inputs.shape[1], batch.inputs.device
            )
        critic_breakdown, phase_error = critic_losses(
            critic_output, batch.actions, batch.rewards, batch.dones, returns,
            return_mask, target_policy, self.config, bootstrap_mask=bootstrap_mask,
            sample_weights=batch.importance_weights,
        )
        surprise = functional_surprise_target(
            actor_output.prediction, target_logits, critic_output,
            batch.actions, returns, phase_error, self.calibrator, self.config,
            update_calibration=update_calibration,
            sample_weights=batch.importance_weights,
        )
        actor_breakdown = actor_losses(
            self.actor, actor_output.prediction, target_logits,
            surprise, batch.mask, self.config, task_targets=batch.task_targets,
            task_loss=task_loss, sample_weights=batch.importance_weights,
        )
        return RASLLosses(
            actor_breakdown, critic_breakdown, surprise, returns, return_mask,
            actor_output, target_logits,
        )

    def train_step(
        self,
        batch: TrajectoryBatch,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        *,
        task_loss: Tensor | None = None,
        performance: float | None = None,
        add_to_replay: bool = True,
        replay_indices: Sequence[int] | None = None,
    ) -> RASLStepReport:
        if add_to_replay and replay_indices is not None:
            raise ValueError("a step cannot both append replay items and update sampled items")
        self.train(True)
        batch = batch.validated(
            input_dim=self.actor.config.input_dim,
            action_dim=self.actor.config.resolved_output_dim,
        )
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        losses = self.compute_losses(
            batch, task_loss=task_loss, update_calibration=True
        )
        losses.critic.total.backward()
        critic_report = clip_and_report_gradients(
            self.critic, maximum_norm=self.config.maximum_gradient_norm
        )
        if not critic_report.finite:
            raise FloatingPointError("critic gradients became non-finite")
        critic_optimizer.step()
        losses.actor.total.backward()
        actor_report = clip_and_report_gradients(
            self.actor, maximum_norm=self.config.maximum_gradient_norm
        )
        if not actor_report.finite:
            raise FloatingPointError("actor gradients became non-finite")
        valid = batch.mask
        mean_reward = float((batch.rewards * valid).sum().detach() / valid.sum().clamp_min(1))
        observed_performance = mean_reward if performance is None else float(performance)
        actor_allowed = self.performance_guard.allows(
            observed_performance, float(losses.actor.functional_cross_entropy.detach())
        )
        if actor_allowed:
            actor_optimizer.step()
        else:
            actor_optimizer.zero_grad(set_to_none=True)
        self.update_targets(actor_updated=actor_allowed)
        learnability = losses.surprise.exploration_bonus.mean(-1)
        controllability = losses.surprise.controllability.mean(-1)
        functional = losses.surprise.score.abs().mean(-1)
        if add_to_replay:
            self.replay.add(batch, functional, learnability, controllability)
        elif replay_indices is not None:
            if len(replay_indices) != batch.inputs.shape[0]:
                raise ValueError("one replay index is required per sampled trajectory")
            priority_rows = functional * learnability.clamp(0, 1) * controllability.clamp(0, 1)
            priorities = (
                (priority_rows * valid).sum(-1) / valid.sum(-1).clamp_min(1)
            ).clamp(1e-6, self.config.replay_priority_cap)
            self.replay.update_priorities(replay_indices, priorities.detach().cpu())
        return RASLStepReport(
            float(losses.actor.total.detach()), float(losses.critic.total.detach()),
            float(losses.actor.task.detach()),
            float(losses.actor.functional_cross_entropy.detach()), mean_reward,
            float(losses.surprise.score.abs()[valid].mean().detach()), actor_allowed,
            float(actor_report.total_before_clip.detach()),
            float(critic_report.total_before_clip.detach()), len(self.replay),
        )

    def train_replay_step(
        self,
        batch_size: int,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        *,
        device: torch.device | str | None = None,
        performance: float | None = None,
        generator: torch.Generator | None = None,
    ) -> RASLStepReport:
        """Sample stratified replay, apply importance correction, and refresh priorities."""

        if device is None:
            device = next(self.actor.parameters()).device
        sample = self.replay.sample(batch_size, device=device, generator=generator)
        return self.train_step(
            sample.batch, actor_optimizer, critic_optimizer,
            performance=performance, add_to_replay=False,
            replay_indices=sample.indices,
        )


RASL_CHECKPOINT_VERSION = 1


def save_rasl_checkpoint(
    path: str | Path,
    learner: ResonantAdjointSurpriseLearner,
    *,
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
) -> None:
    """Save every authority required for deterministic training continuation."""

    if step < 0:
        raise ValueError("checkpoint step cannot be negative")
    payload = {
        "format_version": RASL_CHECKPOINT_VERSION,
        "actor_config": asdict(learner.actor.config),
        "surprise_config": asdict(learner.config),
        "learner": learner.state_dict(),
        "replay": learner.replay.state_dict(),
        "performance_guard": learner.performance_guard.state_dict(),
        "actor_optimizer": None if actor_optimizer is None else actor_optimizer.state_dict(),
        "critic_optimizer": None if critic_optimizer is None else critic_optimizer.state_dict(),
        "torch_rng": torch.random.get_rng_state(),
        "mps_rng": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "step": step,
    }
    torch.save(payload, Path(path))


def load_rasl_checkpoint(
    path: str | Path,
    learner: ResonantAdjointSurpriseLearner,
    *,
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
) -> int:
    """Restore model, targets, calibration, replay, guard, optimizers, and RNG."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if payload.get("format_version") != RASL_CHECKPOINT_VERSION:
        raise ValueError("unsupported RASL checkpoint version")
    if payload.get("actor_config") != asdict(learner.actor.config):
        raise ValueError("RASL checkpoint actor configuration does not match")
    if payload.get("surprise_config") != asdict(learner.config):
        raise ValueError("RASL checkpoint surprise configuration does not match")
    if (payload.get("actor_optimizer") is None) != (actor_optimizer is None):
        raise ValueError("actor optimizer presence does not match checkpoint")
    if (payload.get("critic_optimizer") is None) != (critic_optimizer is None):
        raise ValueError("critic optimizer presence does not match checkpoint")
    learner.load_state_dict(payload["learner"], strict=True)
    learner.replay.load_state_dict(payload["replay"])
    learner.performance_guard.load_state_dict(payload["performance_guard"])
    if actor_optimizer is not None:
        actor_optimizer.load_state_dict(payload["actor_optimizer"])
    if critic_optimizer is not None:
        critic_optimizer.load_state_dict(payload["critic_optimizer"])
    torch.random.set_rng_state(payload["torch_rng"].cpu())
    if payload.get("mps_rng") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(payload["mps_rng"].cpu())
    if payload.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    learner.target_actor.eval()
    learner.target_critic.eval()
    step = int(payload.get("step", -1))
    if step < 0:
        raise ValueError("RASL checkpoint step is invalid")
    return step
