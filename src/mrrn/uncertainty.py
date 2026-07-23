"""Decomposed uncertainty, distributional heads, calibration, and abstention."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .runtime_validation import runtime_validation_enabled


class UncertaintyChannel(IntEnum):
    ALEATORIC = 0
    EPISTEMIC = 1
    HYPOTHESIS_ENTROPY = 2
    ROUTER = 3
    RETRIEVAL = 4
    SOURCE = 5
    STRUCTURAL_CONFLICT = 6
    CALIBRATION_ERROR = 7


@dataclass(frozen=True, slots=True)
class DistributionalOutput:
    categorical_logits: Tensor
    continuous_quantiles: Tensor
    quantile_levels: Tensor
    ensemble_values: Tensor
    aleatoric: Tensor
    epistemic: Tensor


class DistributionalPredictionHead(nn.Module):
    """Categorical logits, ordered quantiles, and compact bootstrap heads."""

    def __init__(
        self, width: int, categories: int, continuous_dims: int, *,
        quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9), ensemble_heads: int = 4,
    ) -> None:
        super().__init__()
        if min(width, categories, continuous_dims, ensemble_heads) <= 0:
            raise ValueError("distributional head dimensions must be positive")
        if not quantile_levels or tuple(sorted(set(quantile_levels))) != quantile_levels:
            raise ValueError("quantile levels must be unique and increasing")
        if any(not 0 < value < 1 for value in quantile_levels):
            raise ValueError("quantile levels must lie in (0,1)")
        self.categories = categories
        self.continuous_dims = continuous_dims
        self.ensemble_heads = ensemble_heads
        self.register_buffer("quantile_levels", torch.tensor(quantile_levels), persistent=True)
        self.categorical = nn.Linear(width, categories)
        self.quantile_base = nn.Linear(width, continuous_dims)
        self.quantile_increments = nn.Linear(
            width, continuous_dims * (len(quantile_levels) - 1)
        )
        self.ensemble = nn.Linear(width, continuous_dims * ensemble_heads)

    def forward(self, features: Tensor) -> DistributionalOutput:
        if features.ndim < 2:
            raise ValueError("distributional features require a final feature dimension")
        base = self.quantile_base(features)
        if self.quantile_levels.numel() == 1:
            quantiles = base.unsqueeze(-2)
        else:
            increments = F.softplus(self.quantile_increments(features)).reshape(
                *features.shape[:-1], self.quantile_levels.numel() - 1, self.continuous_dims
            )
            quantiles = torch.cat((base.unsqueeze(-2), base.unsqueeze(-2) + increments.cumsum(-2)), -2)
        ensemble = self.ensemble(features).reshape(
            *features.shape[:-1], self.ensemble_heads, self.continuous_dims
        )
        aleatoric = quantiles[..., -1, :] - quantiles[..., 0, :]
        epistemic = ensemble.var(-2, correction=0)
        return DistributionalOutput(
            self.categorical(features), quantiles, self.quantile_levels,
            ensemble, aleatoric, epistemic,
        )


@dataclass(frozen=True, slots=True)
class UncertaintyInputs:
    aleatoric: Tensor
    ensemble_values: Tensor
    hypothesis_weights: Tensor
    router_posterior: Tensor
    retrieval_scores: Tensor
    retrieval_mask: Tensor
    retrieval_oracle_gap: Tensor
    source_reliability: Tensor
    structural_conflict: Tensor
    calibration_error: Tensor


class UncertaintyEstimator(nn.Module):
    """Keep semantically distinct uncertainty causes in separate channels."""

    def __init__(self, output_channels: int = 8) -> None:
        super().__init__()
        if output_channels != len(UncertaintyChannel):
            raise ValueError("uncertainty output must contain all controlled channels")
        self.calibration = nn.Parameter(torch.ones(output_channels))

    def forward(self, inputs: UncertaintyInputs) -> Tensor:
        base = inputs.aleatoric.shape[:-1]
        if inputs.ensemble_values.shape[:-2] != base:
            raise ValueError("ensemble values do not align with uncertainty rows")
        if inputs.hypothesis_weights.shape[:-1] != base:
            raise ValueError("hypothesis weights do not align with uncertainty rows")
        if inputs.router_posterior.shape[:-1] != base:
            raise ValueError("router posterior does not align with uncertainty rows")
        if inputs.retrieval_scores.shape != inputs.retrieval_mask.shape or inputs.retrieval_scores.shape[:-1] != base:
            raise ValueError("retrieval scores and mask do not align with uncertainty rows")
        epistemic = inputs.ensemble_values.var(-2, correction=0).mean(-1)
        hypothesis = inputs.hypothesis_weights.clamp_min(1e-8)
        hypothesis_entropy = -(hypothesis * hypothesis.log()).sum(-1) / torch.log(
            hypothesis.new_tensor(max(2, hypothesis.shape[-1]))
        )
        router = inputs.router_posterior.clamp_min(1e-8)
        router_entropy = -(router * router.log()).sum(-1) / torch.log(
            router.new_tensor(max(2, router.shape[-1]))
        )
        retrieval_probability = torch.softmax(
            inputs.retrieval_scores.masked_fill(~inputs.retrieval_mask, -torch.inf), -1
        )
        retrieval_probability = torch.nan_to_num(retrieval_probability)
        retrieval_entropy = -(retrieval_probability.clamp_min(1e-8) * retrieval_probability.clamp_min(1e-8).log()).sum(-1)
        retrieval = retrieval_entropy + inputs.retrieval_oracle_gap.clamp_min(0)
        raw = torch.stack((
            inputs.aleatoric.mean(-1), epistemic, hypothesis_entropy, router_entropy,
            retrieval, 1 - inputs.source_reliability.clamp(0, 1),
            inputs.structural_conflict.clamp_min(0), inputs.calibration_error.clamp_min(0),
        ), -1)
        return raw * F.softplus(self.calibration)


@dataclass(frozen=True, slots=True)
class CalibrationState:
    counts: Tensor
    confidence_sum: Tensor
    accuracy_sum: Tensor
    squared_error_sum: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.counts.ndim != 2:
            raise ValueError("calibration state must be (groups,bins)")
        for name in ("confidence_sum", "accuracy_sum", "squared_error_sum"):
            if getattr(self, name).shape != self.counts.shape:
                raise ValueError(f"calibration {name} shape mismatch")
        if bool((self.counts < 0).any()):
            raise ValueError("calibration counts cannot be negative")

    def detach(self) -> "CalibrationState":
        return CalibrationState(
            self.counts.detach(), self.confidence_sum.detach(),
            self.accuracy_sum.detach(), self.squared_error_sum.detach(),
        )

    def to(self, *args, **kwargs) -> "CalibrationState":
        return CalibrationState(
            self.counts.to(*args, **kwargs),
            self.confidence_sum.to(*args, **kwargs),
            self.accuracy_sum.to(*args, **kwargs),
            self.squared_error_sum.to(*args, **kwargs),
        )


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    expected_calibration_error: Tensor
    brier_score: Tensor
    bin_confidence: Tensor
    bin_accuracy: Tensor
    counts: Tensor


class OnlineCalibration:
    """Source/task/modality/scale groups with streaming reliability bins."""

    def __init__(self, groups: int, bins: int = 15) -> None:
        if min(groups, bins) <= 0:
            raise ValueError("calibration groups and bins must be positive")
        self.groups = groups
        self.bins = bins

    def initial_state(self, *, device=None, dtype=None) -> CalibrationState:
        shape = (self.groups, self.bins)
        return CalibrationState(*(
            torch.zeros(shape, device=device, dtype=dtype) for _ in range(4)
        ))

    def update(
        self, state: CalibrationState, probabilities: Tensor, targets: Tensor,
        group_ids: Tensor, mask: Tensor,
    ) -> CalibrationState:
        if probabilities.ndim != 2 or targets.shape != probabilities.shape[:1]:
            raise ValueError("calibration probabilities require (items,classes) and item targets")
        if group_ids.shape != targets.shape or mask.shape != targets.shape or mask.dtype != torch.bool:
            raise ValueError("calibration groups and mask must match targets")
        confidence, prediction = probabilities.max(-1)
        correct = (prediction == targets).to(probabilities.dtype)
        squared = (probabilities - F.one_hot(targets, probabilities.shape[-1]).to(probabilities.dtype)).square().sum(-1)
        bins = (confidence * self.bins).long().clamp_max(self.bins - 1)
        values = [value.clone() for value in (
            state.counts, state.confidence_sum, state.accuracy_sum, state.squared_error_sum,
        )]
        for index in torch.nonzero(mask, as_tuple=False).flatten().tolist():
            group, bin_index = int(group_ids[index]), int(bins[index])
            if not 0 <= group < self.groups:
                raise ValueError("calibration group lies outside configured groups")
            values[0][group, bin_index] += 1
            values[1][group, bin_index] += confidence[index]
            values[2][group, bin_index] += correct[index]
            values[3][group, bin_index] += squared[index]
        return CalibrationState(*values)

    @staticmethod
    def report(state: CalibrationState) -> CalibrationReport:
        denominator = state.counts.clamp_min(1)
        confidence = state.confidence_sum / denominator
        accuracy = state.accuracy_sum / denominator
        totals = state.counts.sum(-1).clamp_min(1)
        ece = (state.counts * (confidence - accuracy).abs()).sum(-1) / totals
        brier = state.squared_error_sum.sum(-1) / totals
        return CalibrationReport(ece, brier, confidence, accuracy, state.counts)


def pinball_loss(predicted_quantiles: Tensor, target: Tensor, levels: Tensor) -> Tensor:
    if predicted_quantiles.shape[-2] != levels.numel() or predicted_quantiles.shape[:-2] != target.shape[:-1]:
        raise ValueError("pinball loss tensor shapes are incompatible")
    error = target.unsqueeze(-2) - predicted_quantiles
    view = (1,) * (error.ndim - 2) + (levels.numel(), 1)
    level = levels.view(view)
    return torch.maximum(level * error, (level - 1) * error).mean()


def selective_abstention(
    utility: Tensor, uncertainty: Tensor, risk_limit: Tensor, *, uncertainty_weight: float = 1.0,
) -> Tensor:
    if utility.shape != risk_limit.shape or uncertainty.shape[:-1] != utility.shape:
        raise ValueError("abstention inputs are incompatible")
    if uncertainty_weight < 0:
        raise ValueError("uncertainty weight cannot be negative")
    risk = uncertainty.mean(-1) * uncertainty_weight
    return (risk > risk_limit) | (utility < 0)
