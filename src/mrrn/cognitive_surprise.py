"""Candidate-bounded consequence learning for the cognitive language actor.

The ordinary vocabulary projection remains the next-token task authority.  This
module evaluates only a small, explicit candidate set for functional-surprise
learning and feeds detached cognitive state into the critic.  Consequently a
critic update cannot leak gradients into the actor and a 50k vocabulary never
becomes a ``time x vocabulary x critic`` tensor.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from math import log, sqrt
from pathlib import Path
import tempfile
from typing import Literal, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cognitive_model import CognitiveActionReceipts, MRCRAOutput
from .cognitive_types import BoundaryClass, InternalAction, RelationFamily
from .language import MRCRALanguageModel, MRCRALanguageOutput
from .optimization import OptimizerPolicy, build_adamw, clip_and_report_gradients
from .provenance import ProvenanceLedger
from .surprise import (
    FunctionalSurpriseCalibrator, FunctionalSurpriseTarget, PerformanceGuard,
    ResonantAdjointSurpriseConfig, functional_surprise_target,
    multihorizon_returns, quantile_huber_loss,
)


RewardSource = Literal["environment", "human", "verifier", "task_loss"]


@dataclass(frozen=True, slots=True)
class CognitiveRASLConfig:
    core: ResonantAdjointSurpriseConfig = ResonantAdjointSurpriseConfig(
        critic_width=64, minimum_critic_width=16, critic_layers=1,
        critic_scales=1, critic_heads=4, critic_modes=8,
    )
    maximum_candidates: int = 64
    relation_transition_weight: float = 0.15
    memory_utility_weight: float = 0.15
    cognitive_transition_weight: float = 0.25
    termination_weight: float = 0.10
    reverse_credit_weight: float = 0.25
    replay_burn_in_steps: int = 1

    def __post_init__(self) -> None:
        if not 2 <= self.maximum_candidates <= 64:
            raise ValueError("cognitive RASL candidate bound must lie in 2..64")
        if min(
            self.relation_transition_weight, self.memory_utility_weight,
            self.cognitive_transition_weight, self.termination_weight,
            self.reverse_credit_weight,
        ) < 0:
            raise ValueError("cognitive RASL loss weights cannot be negative")
        if self.replay_burn_in_steps <= 0:
            raise ValueError("cognitive replay requires a positive recurrent burn-in")


@dataclass(frozen=True, slots=True)
class CognitiveTrajectoryBatch:
    input_ids: Tensor
    behavior_tokens: Tensor
    candidate_token_ids: Tensor
    candidate_sampling_log_probabilities: Tensor
    sampled_candidate_mask: Tensor
    rewards: Tensor
    dones: Tensor
    mask: Tensor
    segment_ids: Tensor | None = None
    boundary_classes: Tensor | None = None
    task_targets: Tensor | None = None
    behavior_candidate_logits: Tensor | None = None
    goal_features: Tensor | None = None
    relation_transition_targets: Tensor | None = None
    memory_utility_targets: Tensor | None = None
    cognitive_transition_targets: Tensor | None = None
    reward_source: RewardSource = "environment"
    burn_in_steps: int = 0
    importance_weights: Tensor | None = None

    @property
    def candidate_count(self) -> int:
        return self.candidate_token_ids.shape[-1]

    @property
    def loss_mask(self) -> Tensor:
        result = self.mask.clone()
        if self.burn_in_steps:
            result[:, : self.burn_in_steps] = False
        return result

    def validated(
        self, *, vocabulary_size: int, width: int,
        maximum_candidates: int,
    ) -> "CognitiveTrajectoryBatch":
        if self.input_ids.ndim != 2 or self.input_ids.dtype != torch.int64:
            raise ValueError("cognitive trajectories require int64 input_ids (batch,time)")
        base = self.input_ids.shape
        if base[1] == 0:
            raise ValueError("cognitive trajectories cannot be empty")
        for name in ("behavior_tokens",):
            value = getattr(self, name)
            if value.shape != base or value.dtype != torch.int64:
                raise ValueError(f"{name} must be int64 with batch/time shape")
        if self.candidate_token_ids.ndim != 3 or self.candidate_token_ids.shape[:2] != base:
            raise ValueError("candidate token IDs must be (batch,time,candidates)")
        if self.candidate_token_ids.dtype != torch.int64:
            raise ValueError("candidate token IDs must be int64")
        candidates = self.candidate_count
        if not 2 <= candidates <= maximum_candidates:
            raise ValueError("trajectory candidate count lies outside the configured bound")
        for name in ("candidate_sampling_log_probabilities", "sampled_candidate_mask"):
            if getattr(self, name).shape != self.candidate_token_ids.shape:
                raise ValueError(f"{name} must match candidate IDs")
        if not self.candidate_sampling_log_probabilities.is_floating_point():
            raise ValueError("candidate sampling probabilities must be floating point")
        if self.sampled_candidate_mask.dtype != torch.bool:
            raise ValueError("sampled candidate mask must be boolean")
        if self.rewards.shape != base or not self.rewards.is_floating_point():
            raise ValueError("rewards must be floating point with batch/time shape")
        if self.dones.shape != base or self.dones.dtype != torch.bool:
            raise ValueError("dones must be boolean with batch/time shape")
        if self.mask.shape != base or self.mask.dtype != torch.bool:
            raise ValueError("trajectory mask must be boolean with batch/time shape")
        if base[1] > 1 and bool((self.mask[:, 1:] & ~self.mask[:, :-1]).any()):
            raise ValueError("trajectory padding cannot reactivate")
        if not 0 <= self.burn_in_steps < max(1, base[1]):
            raise ValueError("burn-in must leave at least one trajectory step")
        valid_ids = torch.cat((
            self.input_ids[self.mask], self.behavior_tokens[self.mask],
            self.candidate_token_ids[self.mask].flatten(),
        ))
        if valid_ids.numel() and (
            int(valid_ids.min()) < 0 or int(valid_ids.max()) >= vocabulary_size
        ):
            raise ValueError("trajectory contains token IDs outside the vocabulary")
        candidates_for_valid = self.candidate_token_ids[self.mask]
        if candidates_for_valid.numel():
            sorted_candidates = candidates_for_valid.sort(-1).values
            if bool((sorted_candidates[:, 1:] == sorted_candidates[:, :-1]).any()):
                raise ValueError("each candidate set must contain unique token IDs")
            behavior_count = (
                candidates_for_valid == self.behavior_tokens[self.mask, None]
            ).sum(-1)
            if not bool((behavior_count == 1).all()):
                raise ValueError("every candidate set must contain the behavior token exactly once")
        if not bool(torch.isfinite(self.rewards).all()):
            raise ValueError("trajectory rewards must be finite")
        if not bool(torch.isfinite(self.candidate_sampling_log_probabilities).all()):
            raise ValueError("candidate proposal log probabilities must be finite")
        if self.segment_ids is not None and (
            self.segment_ids.shape != base or self.segment_ids.dtype != torch.int64
        ):
            raise ValueError("segment IDs must be int64 with batch/time shape")
        if self.boundary_classes is not None and (
            self.boundary_classes.shape != base or self.boundary_classes.dtype != torch.int64
        ):
            raise ValueError("boundary classes must be int64 with batch/time shape")
        if self.task_targets is not None and (
            self.task_targets.shape != base or self.task_targets.dtype != torch.int64
        ):
            raise ValueError("task targets must be int64 with batch/time shape")
        if self.behavior_candidate_logits is not None and self.behavior_candidate_logits.shape != self.candidate_token_ids.shape:
            raise ValueError("behavior candidate logits must match candidate IDs")
        if self.goal_features is not None and self.goal_features.shape != (*base, width):
            raise ValueError("goal features must be (batch,time,cognitive_width)")
        relation_shape = (*base, len(RelationFamily), 3)
        if self.relation_transition_targets is not None and self.relation_transition_targets.shape != relation_shape:
            raise ValueError("relation transition targets must be (batch,time,families,3)")
        if self.memory_utility_targets is not None and self.memory_utility_targets.shape != base:
            raise ValueError("memory utility targets must match batch/time")
        if self.cognitive_transition_targets is not None and self.cognitive_transition_targets.shape != (*base, width):
            raise ValueError("cognitive transition targets must be (batch,time,cognitive_width)")
        if self.reward_source not in {"environment", "human", "verifier", "task_loss"}:
            raise ValueError("unknown cognitive trajectory reward source")
        if self.importance_weights is not None and (
            self.importance_weights.shape != (base[0],)
            or not self.importance_weights.is_floating_point()
            or not bool(torch.isfinite(self.importance_weights).all())
            or bool((self.importance_weights <= 0).any())
        ):
            raise ValueError("importance weights must be finite positive values per trajectory")
        return self

    def detached_cpu(self) -> "CognitiveTrajectoryBatch":
        def move(value):
            return None if value is None else value.detach().cpu().clone()

        return CognitiveTrajectoryBatch(
            move(self.input_ids), move(self.behavior_tokens),
            move(self.candidate_token_ids),
            move(self.candidate_sampling_log_probabilities),
            move(self.sampled_candidate_mask), move(self.rewards), move(self.dones),
            move(self.mask), move(self.segment_ids), move(self.boundary_classes),
            move(self.task_targets), move(self.behavior_candidate_logits),
            move(self.goal_features), move(self.relation_transition_targets),
            move(self.memory_utility_targets), move(self.cognitive_transition_targets),
            self.reward_source, self.burn_in_steps, move(self.importance_weights),
        )


@dataclass(frozen=True, slots=True)
class CognitiveReplaySample:
    batch: CognitiveTrajectoryBatch
    indices: tuple[int, ...]
    importance_weights: Tensor


@dataclass(slots=True)
class _CognitiveReplayItem:
    trajectory: CognitiveTrajectoryBatch
    priority: float
    sequence: int


class CognitiveTrajectoryReplay:
    """Bounded CPU replay that preserves a valid recurrent burn-in prefix."""

    _tensor_names = (
        "input_ids", "behavior_tokens", "candidate_token_ids",
        "candidate_sampling_log_probabilities", "sampled_candidate_mask",
        "rewards", "dones", "mask", "segment_ids", "boundary_classes",
        "task_targets", "behavior_candidate_logits", "goal_features",
        "relation_transition_targets", "memory_utility_targets",
        "cognitive_transition_targets",
    )

    def __init__(
        self, capacity: int, *, burn_in_steps: int,
        priority_cap: float = 10.0, priority_alpha: float = 0.6,
        prioritized_fraction: float = 0.5,
    ) -> None:
        if min(capacity, burn_in_steps, priority_cap, priority_alpha) <= 0:
            raise ValueError("cognitive replay capacity, burn-in, cap, and alpha must be positive")
        if not 0 <= prioritized_fraction <= 1:
            raise ValueError("prioritized replay fraction must lie in [0,1]")
        self.capacity, self.burn_in_steps = capacity, burn_in_steps
        self.priority_cap, self.priority_alpha = priority_cap, priority_alpha
        self.prioritized_fraction = prioritized_fraction
        self._items: list[_CognitiveReplayItem] = []
        self._transitions = 0
        self._sequence = 0
        self._candidate_count: int | None = None
        self._reward_source: RewardSource | None = None
        self._optional_schema: tuple[bool, ...] | None = None

    def __len__(self) -> int:
        return len(self._items)

    @property
    def transition_count(self) -> int:
        return self._transitions

    @property
    def priorities(self) -> tuple[float, ...]:
        return tuple(item.priority for item in self._items)

    @staticmethod
    def _schema(batch: CognitiveTrajectoryBatch) -> tuple[bool, ...]:
        return tuple(getattr(batch, name) is not None for name in (
            "segment_ids", "boundary_classes", "task_targets",
            "behavior_candidate_logits", "goal_features",
            "relation_transition_targets", "memory_utility_targets",
            "cognitive_transition_targets",
        ))

    def add(
        self, batch: CognitiveTrajectoryBatch, functional_surprise: Tensor,
        learnability: Tensor, controllability: Tensor,
    ) -> tuple[int, ...]:
        if batch.burn_in_steps != self.burn_in_steps:
            raise ValueError(
                "replay trajectories must carry the configured recurrent burn-in"
            )
        shape = batch.input_ids.shape
        for name, value in (
            ("functional surprise", functional_surprise),
            ("learnability", learnability), ("controllability", controllability),
        ):
            if value.shape != shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite and match batch/time")
        if self._candidate_count not in (None, batch.candidate_count):
            raise ValueError("one cognitive replay must use a fixed candidate count")
        if self._reward_source not in (None, batch.reward_source):
            raise ValueError("one cognitive replay cannot mix reward authorities")
        schema = self._schema(batch)
        if self._optional_schema not in (None, schema):
            raise ValueError("one cognitive replay must use a fixed optional-target schema")
        self._candidate_count = batch.candidate_count
        self._reward_source = batch.reward_source
        self._optional_schema = schema
        cpu = batch.detached_cpu()
        inserted = []
        for row in range(shape[0]):
            length = int(cpu.mask[row].sum())
            if length <= self.burn_in_steps:
                continue

            def part(name: str):
                value = getattr(cpu, name)
                return None if value is None else value[row : row + 1, :length].clone()

            trajectory = CognitiveTrajectoryBatch(
                input_ids=part("input_ids"), behavior_tokens=part("behavior_tokens"),
                candidate_token_ids=part("candidate_token_ids"),
                candidate_sampling_log_probabilities=part("candidate_sampling_log_probabilities"),
                sampled_candidate_mask=part("sampled_candidate_mask"),
                rewards=part("rewards"), dones=part("dones"), mask=part("mask"),
                segment_ids=part("segment_ids"), boundary_classes=part("boundary_classes"),
                task_targets=part("task_targets"),
                behavior_candidate_logits=part("behavior_candidate_logits"),
                goal_features=part("goal_features"),
                relation_transition_targets=part("relation_transition_targets"),
                memory_utility_targets=part("memory_utility_targets"),
                cognitive_transition_targets=part("cognitive_transition_targets"),
                reward_source=cpu.reward_source, burn_in_steps=self.burn_in_steps,
            )
            valid = batch.loss_mask[row]
            raw = (
                functional_surprise[row, valid].abs()
                * learnability[row, valid].clamp(0, 1)
                * controllability[row, valid].clamp(0, 1)
            ).mean()
            priority = float(raw.detach().clamp(1e-6, self.priority_cap).cpu())
            self._items.append(_CognitiveReplayItem(trajectory, priority, self._sequence))
            self._transitions += length
            inserted.append(len(self._items) - 1)
            self._sequence += 1
        while self._transitions > self.capacity and self._items:
            removed = self._items.pop(0)
            self._transitions -= removed.trajectory.input_ids.shape[1]
            inserted = [index - 1 for index in inserted if index > 0]
        return tuple(inserted)

    def sample(
        self, batch_size: int, *, device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> CognitiveReplaySample:
        if batch_size <= 0 or not self._items:
            raise ValueError("positive sample size and nonempty cognitive replay are required")
        count = min(batch_size, len(self._items))
        priorities = torch.tensor([item.priority for item in self._items], dtype=torch.float64)
        probability = priorities.pow(self.priority_alpha)
        probability /= probability.sum()
        priority_count = min(count, round(count * self.prioritized_fraction))
        chosen: list[int] = []
        if priority_count:
            chosen.extend(torch.multinomial(
                probability, priority_count, replacement=False, generator=generator
            ).tolist())
        remaining = [index for index in range(len(self._items)) if index not in chosen]
        if len(chosen) < count:
            order = torch.randperm(len(remaining), generator=generator)[: count - len(chosen)]
            chosen.extend(remaining[index] for index in order.tolist())
        rows = [self._items[index].trajectory for index in chosen]
        maximum = max(row.input_ids.shape[1] for row in rows)

        def padded(name: str, fill=0):
            values = [getattr(row, name) for row in rows]
            if values[0] is None:
                return None
            suffix = values[0].shape[2:]
            result = values[0].new_full((count, maximum, *suffix), fill)
            for index, value in enumerate(values):
                result[index, : value.shape[1]] = value[0]
            return result.to(device=device) if device is not None else result

        sampled = CognitiveTrajectoryBatch(
            input_ids=padded("input_ids"), behavior_tokens=padded("behavior_tokens"),
            candidate_token_ids=padded("candidate_token_ids"),
            candidate_sampling_log_probabilities=padded("candidate_sampling_log_probabilities"),
            sampled_candidate_mask=padded("sampled_candidate_mask", False),
            rewards=padded("rewards"), dones=padded("dones", False),
            mask=padded("mask", False), segment_ids=padded("segment_ids"),
            boundary_classes=padded("boundary_classes"), task_targets=padded("task_targets"),
            behavior_candidate_logits=padded("behavior_candidate_logits"),
            goal_features=padded("goal_features"),
            relation_transition_targets=padded("relation_transition_targets"),
            memory_utility_targets=padded("memory_utility_targets"),
            cognitive_transition_targets=padded("cognitive_transition_targets"),
            reward_source=self._reward_source or "environment",
            burn_in_steps=self.burn_in_steps,
        )
        selected_probability = probability[torch.tensor(chosen)]
        importance = (len(self._items) * selected_probability).pow(-1)
        importance /= importance.max().clamp_min(1e-12)
        importance = importance.to(dtype=torch.float32, device=device)
        sampled = CognitiveTrajectoryBatch(
            **{
                name: getattr(sampled, name)
                for name in CognitiveTrajectoryBatch.__dataclass_fields__
                if name != "importance_weights"
            },
            importance_weights=importance,
        )
        return CognitiveReplaySample(sampled, tuple(chosen), importance)

    def update_priorities(self, indices: Sequence[int], priorities: Tensor) -> None:
        if priorities.ndim != 1 or priorities.numel() != len(indices):
            raise ValueError("one priority is required per cognitive replay index")
        if not bool(torch.isfinite(priorities).all()) or bool((priorities < 0).any()):
            raise ValueError("cognitive replay priorities must be finite and nonnegative")
        for index, value in zip(indices, priorities.tolist(), strict=True):
            if not 0 <= index < len(self._items):
                raise ValueError("cognitive replay index is out of range")
            self._items[index].priority = min(self.priority_cap, max(1e-6, float(value)))

    def state_dict(self) -> dict:
        def encode(batch: CognitiveTrajectoryBatch) -> dict:
            return {
                name: getattr(batch, name)
                for name in CognitiveTrajectoryBatch.__dataclass_fields__
                if name != "importance_weights"
            } | {"importance_weights": None}

        return {
            "capacity": self.capacity, "burn_in_steps": self.burn_in_steps,
            "priority_cap": self.priority_cap, "priority_alpha": self.priority_alpha,
            "prioritized_fraction": self.prioritized_fraction,
            "transitions": self._transitions, "sequence": self._sequence,
            "candidate_count": self._candidate_count,
            "reward_source": self._reward_source,
            "optional_schema": self._optional_schema,
            "items": [
                {"trajectory": encode(item.trajectory), "priority": item.priority,
                 "sequence": item.sequence}
                for item in self._items
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        expected = (
            self.capacity, self.burn_in_steps, self.priority_cap,
            self.priority_alpha, self.prioritized_fraction,
        )
        actual = tuple(state.get(name) for name in (
            "capacity", "burn_in_steps", "priority_cap",
            "priority_alpha", "prioritized_fraction",
        ))
        if actual != expected:
            raise ValueError("cognitive replay checkpoint controls do not match")
        items, transitions = [], 0
        for encoded in state.get("items", []):
            trajectory = CognitiveTrajectoryBatch(**encoded["trajectory"])
            length = trajectory.input_ids.shape[1]
            if trajectory.input_ids.shape[0] != 1 or length <= self.burn_in_steps:
                raise ValueError("cognitive replay checkpoint contains an invalid trajectory")
            priority = float(encoded["priority"])
            if not 0 < priority <= self.priority_cap:
                raise ValueError("cognitive replay checkpoint contains an invalid priority")
            items.append(_CognitiveReplayItem(trajectory, priority, int(encoded["sequence"])))
            transitions += length
        if transitions != int(state.get("transitions", -1)) or transitions > self.capacity:
            raise ValueError("cognitive replay transition count is inconsistent")
        self._items, self._transitions = items, transitions
        self._sequence = int(state.get("sequence", 0))
        self._candidate_count = state.get("candidate_count")
        self._reward_source = state.get("reward_source")
        schema = state.get("optional_schema")
        self._optional_schema = None if schema is None else tuple(schema)


def build_language_candidate_set(
    target_logits: Tensor, behavior_tokens: Tensor, *, candidate_count: int = 48,
    verifier_alternatives: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build unique behavior/top-policy/verifier/random candidate sets.

    Proposal probabilities for random negatives are recorded at the exact
    without-replacement draw step, enabling sampled-policy correction.
    """

    if target_logits.ndim != 3 or behavior_tokens.shape != target_logits.shape[:2]:
        raise ValueError("target logits and behavior tokens must share batch/time")
    if behavior_tokens.dtype != torch.int64:
        raise ValueError("behavior tokens must be int64")
    vocabulary = target_logits.shape[-1]
    if not 2 <= candidate_count <= min(64, vocabulary):
        raise ValueError("candidate_count must lie in 2..min(64,vocabulary)")
    if verifier_alternatives is not None and (
        verifier_alternatives.ndim != 3
        or verifier_alternatives.shape[:2] != behavior_tokens.shape
        or verifier_alternatives.dtype != torch.int64
    ):
        raise ValueError("verifier alternatives must be int64 (batch,time,count)")
    batch, length = behavior_tokens.shape
    candidates = torch.empty(
        batch, length, candidate_count, dtype=torch.int64, device=target_logits.device
    )
    log_probability = target_logits.new_zeros(batch, length, candidate_count)
    sampled = torch.zeros_like(candidates, dtype=torch.bool)
    top_count = max(1, candidate_count // 2)
    top = target_logits.detach().topk(min(vocabulary, top_count + 1), -1).indices
    for row in range(batch):
        for time in range(length):
            chosen: list[int] = []
            behavior = int(behavior_tokens[row, time])
            if not 0 <= behavior < vocabulary:
                raise ValueError("behavior token lies outside vocabulary")
            chosen.append(behavior)
            if verifier_alternatives is not None:
                for token in verifier_alternatives[row, time].tolist():
                    if 0 <= int(token) < vocabulary and int(token) not in chosen:
                        chosen.append(int(token))
                        if len(chosen) == candidate_count:
                            break
            if len(chosen) < candidate_count:
                for token in top[row, time].tolist():
                    if int(token) not in chosen:
                        chosen.append(int(token))
                        if len(chosen) == candidate_count:
                            break
            while len(chosen) < candidate_count:
                available = torch.ones(vocabulary, dtype=torch.bool, device=target_logits.device)
                available[torch.tensor(chosen, device=target_logits.device)] = False
                remaining = int(available.sum())
                draw = int(torch.multinomial(
                    available.to(target_logits.dtype), 1, generator=generator
                ).item())
                position = len(chosen)
                chosen.append(draw)
                sampled[row, time, position] = True
                log_probability[row, time, position] = -log(remaining)
            candidates[row, time] = torch.tensor(chosen, device=target_logits.device)
    return candidates, log_probability, sampled


@dataclass(frozen=True, slots=True)
class CognitiveCriticOutput:
    value_quantiles: Tensor
    action_values: Tensor
    reward_prediction: Tensor
    termination_logits: Tensor
    relation_transition_logits: Tensor
    memory_utility: Tensor
    cognitive_transition: Tensor
    adjoint_credit: Tensor
    forward_features: Tensor
    adjoint_features: Tensor
    epistemic_uncertainty: Tensor
    aleatoric_uncertainty: Tensor
    mask: Tensor

    def quantiles_for(self, actions: Tensor) -> Tensor:
        if actions.shape != self.mask.shape or actions.dtype != torch.int64:
            raise ValueError("local actions must be int64 with batch/time shape")
        gather = torch.where(self.mask, actions, 0)[:, :, None, None, None].expand(
            -1, -1, self.action_values.shape[2], self.action_values.shape[3], 1
        )
        shift = self.action_values.gather(-1, gather).squeeze(-1)
        return self.value_quantiles + shift.unsqueeze(-1)

    def mean_action_values(self) -> Tensor:
        return self.action_values.mean(2) + self.value_quantiles.mean((2, 4)).unsqueeze(-1)


class CognitiveAdjointCritic(nn.Module):
    """Causal cognitive critic with a separate reverse consequence adjoint."""

    def __init__(
        self, cognitive_width: int, relation_families: int,
        config: CognitiveRASLConfig, *, width: int | None = None,
    ) -> None:
        super().__init__()
        core = config.core
        self.config = config
        self.width = core.critic_width if width is None else width
        if self.width < core.minimum_critic_width:
            raise ValueError("cognitive critic width is below configured minimum")
        self.cognitive_width = cognitive_width
        self.relation_families = relation_families
        self.relation_types = nn.Linear(relation_families, self.width)
        self.receipt_action = nn.Embedding(len(InternalAction) + 1, self.width)
        self.receipt_status = nn.Embedding(9, self.width)
        self.input = nn.Linear(4 * cognitive_width + 2 * self.width, self.width)
        self.forward_cell = nn.GRUCell(self.width, self.width)
        self.outcome = nn.Linear(cognitive_width + 2, self.width)
        self.adjoint_cell = nn.GRUCell(self.width, self.width)
        self.candidate = nn.Linear(cognitive_width, self.width)
        self.candidate_key = nn.Linear(self.width, core.action_rank, bias=False)
        bootstrap, horizons, quantiles = (
            core.bootstrap_heads, len(core.horizons), len(core.quantiles)
        )
        self.quantile_head = nn.Linear(self.width, bootstrap * horizons * quantiles)
        self.action_query = nn.Linear(
            self.width, bootstrap * horizons * core.action_rank
        )
        self.reward_query = nn.Linear(self.width, core.action_rank)
        self.termination_query = nn.Linear(self.width, core.action_rank)
        self.memory_query = nn.Linear(self.width, core.action_rank)
        self.adjoint_query = nn.Linear(self.width, core.action_rank)
        self.reward_base = nn.Linear(self.width, 1)
        self.termination_base = nn.Linear(self.width, 1)
        self.memory_base = nn.Linear(self.width, 1)
        self.relation_transition = nn.Linear(
            self.width + core.action_rank,
            len(core.horizons) * relation_families * 3,
        )
        self.cognitive_transition = nn.Linear(
            self.width + core.action_rank,
            len(core.horizons) * cognitive_width,
        )

    def _receipt_summary(self, receipts: CognitiveActionReceipts) -> Tensor:
        actions = receipts.actions.clamp_min(-1) + 1
        statuses = receipts.statuses.clamp(0, self.receipt_status.num_embeddings - 1)
        value = self.receipt_action(actions) + self.receipt_status(statuses)
        weight = receipts.mask.to(value.dtype)
        return (value * weight.unsqueeze(-1)).sum(2) / weight.sum(2, keepdim=True).clamp_min(1)

    def _causal_features(
        self, output: MRCRAOutput, goal_features: Tensor, mask: Tensor,
        boundary_classes: Tensor | None,
    ) -> Tensor:
        receipt = self._receipt_summary(output.action_receipts).detach()
        features = F.silu(self.input(torch.cat((
            output.cognitive_features.detach(), output.workspace_features.detach(),
            output.relation_features.detach(), goal_features.detach(),
            self.relation_types(output.relation_type_probabilities.detach()), receipt,
        ), -1)))
        batch, length = mask.shape
        hidden = features.new_zeros(batch, self.width)
        rows = []
        for index in range(length):
            if boundary_classes is not None:
                reset = (
                    (boundary_classes[:, index] == int(BoundaryClass.HARD))
                    | (boundary_classes[:, index] == int(BoundaryClass.SEGMENT))
                )
                hidden = hidden.masked_fill(reset[:, None], 0)
            proposal = self.forward_cell(features[:, index], hidden)
            hidden = torch.where(mask[:, index, None], proposal, hidden)
            rows.append(hidden * mask[:, index, None])
        return torch.stack(rows, 1) if rows else features[:, :0]

    def value_distribution(
        self, forward_features: Tensor, candidate_embeddings: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch, length, candidates = candidate_embeddings.shape[:3]
        core = self.config.core
        bootstrap, horizons, quantiles, rank = (
            core.bootstrap_heads, len(core.horizons), len(core.quantiles), core.action_rank
        )
        candidate = self.candidate(candidate_embeddings.detach())
        keys = self.candidate_key(candidate)
        raw = self.quantile_head(forward_features).reshape(
            batch, length, bootstrap, horizons, quantiles
        )
        quantile_values = torch.sort(raw, -1).values
        query = self.action_query(forward_features).reshape(
            batch, length, bootstrap, horizons, rank
        )
        action_values = torch.einsum("btkhr,btcr->btkhc", query, keys) / sqrt(rank)
        head_mean = quantile_values.mean(-1).unsqueeze(-1) + action_values
        epistemic = head_mean.std(2, correction=0)
        aleatoric = (
            quantile_values[..., -1] - quantile_values[..., 0]
        ).mean(2).unsqueeze(-1)
        return quantile_values, action_values, epistemic, aleatoric, keys

    def forward(
        self, output: MRCRAOutput, candidate_embeddings: Tensor,
        local_actions: Tensor, rewards: Tensor, dones: Tensor, mask: Tensor,
        *, goal_features: Tensor | None = None,
        boundary_classes: Tensor | None = None,
    ) -> CognitiveCriticOutput:
        if candidate_embeddings.ndim != 4 or candidate_embeddings.shape[:2] != mask.shape:
            raise ValueError("candidate embeddings must be (batch,time,candidates,width)")
        if candidate_embeddings.shape[-1] != self.cognitive_width:
            raise ValueError("candidate embedding width does not match cognitive critic")
        if local_actions.shape != mask.shape or local_actions.dtype != torch.int64:
            raise ValueError("local actions must be int64 with batch/time shape")
        if rewards.shape != mask.shape or dones.shape != mask.shape or dones.dtype != torch.bool:
            raise ValueError("critic outcomes must match batch/time")
        goal_features = (
            candidate_embeddings.new_zeros(*mask.shape, self.cognitive_width)
            if goal_features is None else goal_features
        )
        forward = self._causal_features(output, goal_features, mask, boundary_classes)
        quantiles, action_values, epistemic, aleatoric, keys = self.value_distribution(
            forward, candidate_embeddings
        )
        safe = torch.where(mask, local_actions, 0)
        selected_key = keys.gather(
            2, safe[:, :, None, None].expand(-1, -1, 1, keys.shape[-1])
        ).squeeze(2)
        selected_embedding = candidate_embeddings.detach().gather(
            2, safe[:, :, None, None].expand(-1, -1, 1, self.cognitive_width)
        ).squeeze(2)
        reverse_hidden = forward.new_zeros(forward.shape[0], self.width)
        reverse_rows: list[Tensor] = [reverse_hidden] * forward.shape[1]
        for index in range(forward.shape[1] - 1, -1, -1):
            drive = F.silu(self.outcome(torch.cat((
                selected_embedding[:, index], rewards[:, index, None].detach(),
                dones[:, index, None].to(rewards.dtype),
            ), -1)))
            proposal = self.adjoint_cell(drive, reverse_hidden)
            reverse_hidden = torch.where(mask[:, index, None], proposal, reverse_hidden)
            reverse_rows[index] = reverse_hidden * mask[:, index, None]
            boundary = dones[:, index].clone()
            if boundary_classes is not None:
                boundary |= (
                    (boundary_classes[:, index] == int(BoundaryClass.HARD))
                    | (boundary_classes[:, index] == int(BoundaryClass.SEGMENT))
                )
            reverse_hidden = reverse_hidden.masked_fill(boundary[:, None], 0)
        adjoint = torch.stack(reverse_rows, 1) if reverse_rows else forward[:, :0]
        reward = self.reward_base(forward) + torch.einsum(
            "btr,btcr->btc", self.reward_query(forward), keys
        ).unsqueeze(-1).squeeze(-1) / sqrt(keys.shape[-1])
        termination = self.termination_base(forward) + torch.einsum(
            "btr,btcr->btc", self.termination_query(forward), keys
        ) / sqrt(keys.shape[-1])
        memory_utility = self.memory_base(forward) + torch.einsum(
            "btr,btcr->btc", self.memory_query(forward), keys
        ) / sqrt(keys.shape[-1])
        conditioned = torch.cat((forward, selected_key), -1)
        relation = self.relation_transition(conditioned).reshape(
            *mask.shape, len(self.config.core.horizons), self.relation_families, 3
        )
        cognitive = self.cognitive_transition(conditioned).reshape(
            *mask.shape, len(self.config.core.horizons), self.cognitive_width
        )
        adjoint_credit = torch.einsum(
            "btr,btcr->btc", self.adjoint_query(adjoint), keys
        ) / sqrt(keys.shape[-1])
        return CognitiveCriticOutput(
            quantiles, action_values, reward, termination, relation,
            memory_utility, cognitive, adjoint_credit, forward, adjoint,
            epistemic, aleatoric, mask,
        )


@dataclass(frozen=True, slots=True)
class CognitiveCriticLosses:
    total: Tensor
    returns: Tensor
    return_mask: Tensor
    distribution: Tensor
    reward: Tensor
    termination: Tensor
    relation_transition: Tensor
    memory_utility: Tensor
    cognitive_transition: Tensor
    reverse_credit: Tensor


@dataclass(frozen=True, slots=True)
class CognitiveActorLosses:
    total: Tensor
    task: Tensor
    functional_cross_entropy: Tensor
    trust_region: Tensor


@dataclass(frozen=True, slots=True)
class CognitiveRASLLosses:
    actor: CognitiveActorLosses
    critic: CognitiveCriticLosses
    surprise: FunctionalSurpriseTarget
    actor_output: MRCRALanguageOutput
    candidate_logits: Tensor
    corrected_candidate_logits: Tensor
    local_actions: Tensor


def _weighted_mean(value: Tensor, mask: Tensor, sample_weights: Tensor | None) -> Tensor:
    weights = mask.to(value.dtype)
    if sample_weights is not None:
        weights = weights * sample_weights[:, None]
    while weights.ndim < value.ndim:
        weights = weights.unsqueeze(-1)
    expanded = weights.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1)


def _local_action_indices(batch: CognitiveTrajectoryBatch) -> Tensor:
    matches = batch.candidate_token_ids == batch.behavior_tokens.unsqueeze(-1)
    return matches.to(torch.int64).argmax(-1)


def cognitive_critic_losses(
    output: CognitiveCriticOutput, batch: CognitiveTrajectoryBatch,
    local_actions: Tensor, target_values: Tensor,
    config: CognitiveRASLConfig,
) -> CognitiveCriticLosses:
    core = config.core
    target_policy = torch.softmax(target_values.detach(), -1)
    bootstrap = (target_policy * target_values.detach()).sum(-1)
    bootstrap = bootstrap[:, :, None].expand(-1, -1, len(core.horizons))
    returns, return_mask = multihorizon_returns(
        batch.rewards, batch.dones, batch.mask, core.horizons,
        discount=core.discount, bootstrap=bootstrap,
    )
    selected = output.quantiles_for(local_actions)
    distribution = quantile_huber_loss(
        selected, returns, core.quantiles,
        return_mask & batch.loss_mask.unsqueeze(-1),
        sample_weights=batch.importance_weights,
    )
    chosen = local_actions.unsqueeze(-1)
    reward_prediction = output.reward_prediction.gather(-1, chosen).squeeze(-1)
    termination_prediction = output.termination_logits.gather(-1, chosen).squeeze(-1)
    memory_prediction = output.memory_utility.gather(-1, chosen).squeeze(-1)
    loss_mask = batch.loss_mask
    reward = _weighted_mean(
        (reward_prediction - batch.rewards).square(), loss_mask,
        batch.importance_weights,
    )
    termination = _weighted_mean(
        F.binary_cross_entropy_with_logits(
            termination_prediction, batch.dones.to(termination_prediction.dtype), reduction="none"
        ), loss_mask, batch.importance_weights,
    )
    relation = output.relation_transition_logits.sum() * 0
    if batch.relation_transition_targets is not None:
        target = batch.relation_transition_targets[:, :, None].expand_as(
            output.relation_transition_logits
        )
        relation = _weighted_mean(
            F.binary_cross_entropy_with_logits(
                output.relation_transition_logits, target, reduction="none"
            ), loss_mask, batch.importance_weights,
        )
    memory = output.memory_utility.sum() * 0
    if batch.memory_utility_targets is not None:
        memory = _weighted_mean(
            (memory_prediction - batch.memory_utility_targets).square(),
            loss_mask, batch.importance_weights,
        )
    cognitive = output.cognitive_transition.sum() * 0
    if batch.cognitive_transition_targets is not None:
        target = batch.cognitive_transition_targets[:, :, None].expand_as(
            output.cognitive_transition
        )
        cognitive = _weighted_mean(
            (output.cognitive_transition - target).square(),
            loss_mask, batch.importance_weights,
        )
    expected = output.mean_action_values().detach().mean(2)
    chosen_value = expected.gather(-1, chosen).squeeze(-1)
    credit_target = (
        returns[..., -1] - chosen_value
    ).clamp(-core.maximum_surprise, core.maximum_surprise)
    credit = output.adjoint_credit.gather(-1, chosen).squeeze(-1)
    reverse = _weighted_mean(
        (credit - credit_target.detach()).square(), loss_mask,
        batch.importance_weights,
    )
    total = (
        core.critic_return_weight * distribution
        + core.critic_reward_weight * reward
        + config.termination_weight * termination
        + config.relation_transition_weight * relation
        + config.memory_utility_weight * memory
        + config.cognitive_transition_weight * cognitive
        + config.reverse_credit_weight * reverse
    )
    return CognitiveCriticLosses(
        total, returns, return_mask, distribution, reward, termination,
        relation, memory, cognitive, reverse,
    )


def cognitive_actor_losses(
    actor_logits: Tensor, corrected_candidate_logits: Tensor,
    target_candidate_logits: Tensor, surprise: FunctionalSurpriseTarget,
    batch: CognitiveTrajectoryBatch, config: CognitiveRASLConfig,
    *, task_loss: Tensor | None = None,
) -> CognitiveActorLosses:
    loss_mask = batch.loss_mask
    if task_loss is not None and batch.task_targets is not None:
        raise ValueError("supply task targets or explicit task loss, not both")
    if task_loss is not None:
        if task_loss.numel() != 1 or not bool(torch.isfinite(task_loss)):
            raise ValueError("explicit cognitive task loss must be a finite scalar")
        task = task_loss
    elif batch.task_targets is not None:
        task_rows = F.cross_entropy(
            actor_logits.flatten(0, 1).float(), batch.task_targets.flatten(), reduction="none"
        ).reshape_as(loss_mask)
        task = _weighted_mean(task_rows, loss_mask, batch.importance_weights)
    else:
        task = actor_logits.sum() * 0
    log_policy = F.log_softmax(corrected_candidate_logits, -1)
    fsce = _weighted_mean(
        -(surprise.distribution * log_policy).sum(-1), loss_mask,
        batch.importance_weights,
    )
    target_policy = torch.softmax(target_candidate_logits.detach(), -1)
    trust = _weighted_mean(
        (target_policy * (target_policy.clamp_min(1e-8).log() - log_policy)).sum(-1),
        loss_mask, batch.importance_weights,
    )
    total = (
        config.core.task_weight * task
        + config.core.surprise_cross_entropy_weight * fsce
        + config.core.trust_region_weight * trust
    )
    return CognitiveActorLosses(total, task, fsce, trust)


@dataclass(frozen=True, slots=True)
class CognitiveRASLStepReport:
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


class CognitiveResonantAdjointSurpriseLearner(nn.Module):
    """Integrated MRCRA consequence learner with strict gradient firewalls."""

    def __init__(
        self, actor: MRCRALanguageModel,
        config: CognitiveRASLConfig = CognitiveRASLConfig(),
    ) -> None:
        super().__init__()
        if not isinstance(actor, MRCRALanguageModel):
            raise ValueError("cognitive RASL requires an MRCRALanguageModel actor")
        self.actor = actor
        self.config = config
        actor_parameters = sum(parameter.numel() for parameter in actor.parameters())
        critic = None
        cognitive = actor.config.cognitive
        for width in range(
            config.core.critic_width, config.core.minimum_critic_width - 1, -1
        ):
            candidate = CognitiveAdjointCritic(
                cognitive.workspace_dim, cognitive.relation_family_count,
                config, width=width,
            )
            fraction = sum(parameter.numel() for parameter in candidate.parameters()) / actor_parameters
            if fraction <= config.core.maximum_critic_parameter_fraction:
                critic = candidate
                break
        if critic is None:
            raise ValueError("minimum cognitive critic exceeds configured actor fraction")
        self.critic = critic
        self.target_actor = deepcopy(actor).requires_grad_(False).eval()
        self.target_critic = deepcopy(critic).requires_grad_(False).eval()
        self.calibrator = FunctionalSurpriseCalibrator(
            len(config.core.horizons), 1, decay=config.core.calibration_decay
        )
        self.replay = CognitiveTrajectoryReplay(
            config.core.replay_capacity,
            burn_in_steps=config.replay_burn_in_steps,
            priority_cap=config.core.replay_priority_cap,
            priority_alpha=config.core.replay_priority_alpha,
            prioritized_fraction=config.core.replay_priority_fraction,
        )
        self.performance_guard = PerformanceGuard(config.core.performance_tolerance)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_actor.eval()
        self.target_critic.eval()
        return self

    def make_optimizers(
        self, *, actor_policy: OptimizerPolicy | None = None,
        critic_policy: OptimizerPolicy | None = None,
    ) -> tuple[torch.optim.AdamW, torch.optim.AdamW]:
        actor_policy = OptimizerPolicy() if actor_policy is None else actor_policy
        critic_policy = (
            OptimizerPolicy(learning_rate=actor_policy.learning_rate)
            if critic_policy is None else critic_policy
        )
        fused = next(self.parameters()).device.type == "cuda"
        return (
            build_adamw(self.actor, actor_policy, fused=fused),
            build_adamw(self.critic, critic_policy, fused=fused),
        )

    @torch.no_grad()
    def update_targets(self, *, actor_updated: bool) -> None:
        def update(target: nn.Module, online: nn.Module) -> None:
            target_parameters = dict(target.named_parameters())
            for name, parameter in online.named_parameters():
                target_parameters[name].mul_(self.config.core.ema_decay).add_(
                    parameter, alpha=1 - self.config.core.ema_decay
                )
            target_buffers = dict(target.named_buffers())
            for name, buffer in online.named_buffers():
                if name in target_buffers:
                    target_buffers[name].copy_(buffer)

        if actor_updated:
            update(self.target_actor, self.actor)
        update(self.target_critic, self.critic)

    def compute_losses(
        self, batch: CognitiveTrajectoryBatch, *, task_loss: Tensor | None = None,
        update_calibration: bool = False,
    ) -> CognitiveRASLLosses:
        batch = batch.validated(
            vocabulary_size=self.actor.vocabulary_size,
            width=self.actor.config.cognitive.workspace_dim,
            maximum_candidates=self.config.maximum_candidates,
        )
        if self.config.core.require_external_reward and batch.reward_source == "task_loss":
            raise ValueError(
                "cognitive functional surprise requires an external downstream consequence"
            )
        ledger = ProvenanceLedger()
        actor_output = self.actor(
            batch.input_ids, attention_mask=batch.mask,
            segment_ids=batch.segment_ids, boundary_classes=batch.boundary_classes,
            ledger=ledger,
        )
        actor_logits = actor_output.logits
        candidate_logits = actor_logits.gather(-1, batch.candidate_token_ids)
        correction = torch.where(
            batch.sampled_candidate_mask,
            batch.candidate_sampling_log_probabilities, torch.zeros_like(candidate_logits),
        )
        corrected = candidate_logits - correction
        target_candidate_logits = (
            candidate_logits.detach()
            if batch.behavior_candidate_logits is None
            else batch.behavior_candidate_logits.detach()
        ) - correction
        local_actions = _local_action_indices(batch)
        candidate_embeddings = self.actor.token_embedding(
            batch.candidate_token_ids
        ).detach()
        critic_output = self.critic(
            actor_output.cognitive, candidate_embeddings, local_actions,
            batch.rewards, batch.dones, batch.mask,
            goal_features=batch.goal_features,
            boundary_classes=batch.boundary_classes,
        )
        with torch.no_grad():
            target_quantiles, target_action_values, _, _, _ = self.target_critic.value_distribution(
                critic_output.forward_features.detach(), candidate_embeddings
            )
            target_values = (
                target_action_values.mean(2)
                + target_quantiles.mean((2, 4)).unsqueeze(-1)
            ).mean(2)
        critic_losses = cognitive_critic_losses(
            critic_output, batch, local_actions, target_values, self.config
        )
        predicted_cognitive = critic_output.cognitive_transition[..., 0, :]
        actual_next = torch.cat((
            actor_output.cognitive.cognitive_features[:, 1:].detach(),
            actor_output.cognitive.cognitive_features[:, -1:].detach(),
        ), 1)
        phase_error = (
            predicted_cognitive.detach() - actual_next
        ).square().mean(-1, keepdim=True)
        surprise = functional_surprise_target(
            corrected, target_candidate_logits, critic_output, local_actions,
            critic_losses.returns, phase_error, self.calibrator,
            self.config.core, update_calibration=update_calibration,
            sample_weights=batch.importance_weights,
        )
        actor_losses = cognitive_actor_losses(
            actor_logits, corrected, target_candidate_logits, surprise,
            batch, self.config, task_loss=task_loss,
        )
        return CognitiveRASLLosses(
            actor_losses, critic_losses, surprise, actor_output,
            candidate_logits, corrected, local_actions,
        )

    def train_step(
        self, batch: CognitiveTrajectoryBatch,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer, *, task_loss: Tensor | None = None,
        performance: float | None = None,
        add_to_replay: bool = True,
        replay_indices: Sequence[int] | None = None,
    ) -> CognitiveRASLStepReport:
        if add_to_replay and replay_indices is not None:
            raise ValueError("a cognitive step cannot append and reprioritize replay simultaneously")
        self.train(True)
        batch = batch.validated(
            vocabulary_size=self.actor.vocabulary_size,
            width=self.actor.config.cognitive.workspace_dim,
            maximum_candidates=self.config.maximum_candidates,
        )
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        losses = self.compute_losses(
            batch, task_loss=task_loss, update_calibration=True
        )
        losses.critic.total.backward()
        critic_gradient = clip_and_report_gradients(
            self.critic, maximum_norm=self.config.core.maximum_gradient_norm
        )
        if not critic_gradient.finite:
            raise FloatingPointError("cognitive critic gradients became non-finite")
        critic_optimizer.step()
        losses.actor.total.backward()
        actor_gradient = clip_and_report_gradients(
            self.actor, maximum_norm=self.config.core.maximum_gradient_norm
        )
        if not actor_gradient.finite:
            raise FloatingPointError("cognitive actor gradients became non-finite")
        valid = batch.loss_mask
        mean_reward = float(
            (batch.rewards * valid).sum().detach() / valid.sum().clamp_min(1)
        )
        measured = mean_reward if performance is None else float(performance)
        actor_allowed = self.performance_guard.allows(
            measured, float(losses.actor.functional_cross_entropy.detach())
        )
        if actor_allowed:
            actor_optimizer.step()
        else:
            actor_optimizer.zero_grad(set_to_none=True)
        self.update_targets(actor_updated=actor_allowed)
        functional = losses.surprise.score.abs().mean(-1)
        learnability = losses.surprise.exploration_bonus.mean(-1)
        controllability = losses.surprise.controllability.mean(-1)
        if add_to_replay:
            self.replay.add(batch, functional, learnability, controllability)
        elif replay_indices is not None:
            if len(replay_indices) != batch.input_ids.shape[0]:
                raise ValueError("one replay index is required per sampled trajectory")
            priority_rows = functional * learnability.clamp(0, 1) * controllability.clamp(0, 1)
            priority = (
                (priority_rows * valid).sum(-1) / valid.sum(-1).clamp_min(1)
            ).clamp(1e-6, self.config.core.replay_priority_cap)
            self.replay.update_priorities(replay_indices, priority.detach().cpu())
        return CognitiveRASLStepReport(
            float(losses.actor.total.detach()), float(losses.critic.total.detach()),
            float(losses.actor.task.detach()),
            float(losses.actor.functional_cross_entropy.detach()), mean_reward,
            float(losses.surprise.score.abs()[valid].mean().detach()), actor_allowed,
            float(actor_gradient.total_before_clip.detach()),
            float(critic_gradient.total_before_clip.detach()),
            len(self.replay),
        )

    def train_replay_step(
        self, batch_size: int, actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer, *,
        device: torch.device | str | None = None,
        performance: float | None = None,
        generator: torch.Generator | None = None,
    ) -> CognitiveRASLStepReport:
        """Train from recurrently valid prioritized replay and refresh priority."""

        if device is None:
            device = next(self.actor.parameters()).device
        sample = self.replay.sample(batch_size, device=device, generator=generator)
        return self.train_step(
            sample.batch, actor_optimizer, critic_optimizer,
            performance=performance, add_to_replay=False,
            replay_indices=sample.indices,
        )


COGNITIVE_RASL_CHECKPOINT_VERSION = 2


def save_cognitive_rasl_checkpoint(
    path: str | Path, learner: CognitiveResonantAdjointSurpriseLearner, *,
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None, step: int = 0,
) -> None:
    if step < 0:
        raise ValueError("checkpoint step cannot be negative")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": COGNITIVE_RASL_CHECKPOINT_VERSION,
        "mrcra_config": asdict(learner.actor.config),
        "rasl_config": asdict(learner.config),
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
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with open(descriptor, "wb", closefd=True) as handle:
            torch.save(payload, handle)
        Path(temporary_name).replace(destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_cognitive_rasl_checkpoint(
    path: str | Path, learner: CognitiveResonantAdjointSurpriseLearner, *,
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
) -> int:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if payload.get("format_version") != COGNITIVE_RASL_CHECKPOINT_VERSION:
        raise ValueError("unsupported cognitive RASL checkpoint version")
    if payload.get("mrcra_config") != asdict(learner.actor.config):
        raise ValueError("cognitive RASL actor configuration does not match")
    if payload.get("rasl_config") != asdict(learner.config):
        raise ValueError("cognitive RASL configuration does not match")
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
        raise ValueError("cognitive RASL checkpoint step is invalid")
    return step
