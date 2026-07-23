"""Action-conditioned distributional world model and sandboxed rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cognitive_types import SourceClass


@dataclass(frozen=True, slots=True)
class WorldModelPrediction:
    horizons: Tensor
    latent_mean: Tensor
    latent_log_scale: Tensor
    relation_logits: Tensor
    event_logits: Tensor
    observation_mean: Tensor
    observation_log_scale: Tensor
    reward_quantiles: Tensor
    costs: Tensor
    constraint_logits: Tensor
    action_success_logits: Tensor


@dataclass(frozen=True, slots=True)
class CounterfactualRollout:
    latent_states: Tensor
    reward_quantiles: Tensor
    costs: Tensor
    constraint_probabilities: Tensor
    uncertainty: Tensor
    scenario_ids: Tensor
    source_classes: Tensor
    valid_mask: Tensor


@dataclass(frozen=True, slots=True)
class CandidateRollout:
    """Bounded hypothesis x action x horizon consequence lattice."""

    latent_states: Tensor
    relation_logits: Tensor
    event_logits: Tensor
    observation_mean: Tensor
    reward_quantiles: Tensor
    costs: Tensor
    constraint_probabilities: Tensor
    action_success_probabilities: Tensor
    termination_probabilities: Tensor
    uncertainty: Tensor
    scenario_ids: Tensor
    source_classes: Tensor
    valid_mask: Tensor

    def __post_init__(self) -> None:
        if self.latent_states.ndim != 5:
            raise ValueError("candidate rollout latent must be (batch,hypotheses,actions,horizons,width)")
        base = self.latent_states.shape[:4]
        for name in (
            "costs", "constraint_probabilities", "action_success_probabilities",
            "termination_probabilities", "uncertainty", "scenario_ids",
            "source_classes", "valid_mask",
        ):
            if getattr(self, name).shape != base:
                raise ValueError(f"candidate rollout {name} must match lattice")
        if self.reward_quantiles.shape[:4] != base:
            raise ValueError("candidate rollout reward quantiles must match lattice")
        if self.relation_logits.shape[:4] != base or self.event_logits.shape[:4] != base:
            raise ValueError("candidate rollout graph/event outputs must match lattice")
        if self.observation_mean.shape[:4] != base:
            raise ValueError("candidate rollout observations must match lattice")
        if self.scenario_ids.dtype != torch.int64 or self.source_classes.dtype != torch.int64:
            raise ValueError("candidate rollout scenario/source IDs must be int64")
        if self.valid_mask.dtype != torch.bool:
            raise ValueError("candidate rollout mask must be boolean")


class ActionConditionedWorldModel(nn.Module):
    """Predict state, graph events, observations, and consequence distributions."""

    def __init__(
        self, width: int, action_dim: int, relation_families: int,
        observation_dim: int, *, horizons: tuple[int, ...] = (1, 4, 16, 64),
        reward_quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> None:
        super().__init__()
        if min(width, action_dim, relation_families, observation_dim) <= 0:
            raise ValueError("world model dimensions must be positive")
        if not horizons or tuple(sorted(set(horizons))) != horizons or any(value <= 0 for value in horizons):
            raise ValueError("world-model horizons must be unique, positive, and increasing")
        if not reward_quantiles or tuple(sorted(set(reward_quantiles))) != reward_quantiles:
            raise ValueError("reward quantiles must be unique and increasing")
        if any(not 0 < value < 1 for value in reward_quantiles):
            raise ValueError("reward quantiles must lie in (0,1)")
        self.width = width
        self.action_dim = action_dim
        self.relation_families = relation_families
        self.observation_dim = observation_dim
        self.horizon_values = horizons
        self.register_buffer("horizons", torch.tensor(horizons, dtype=torch.int64), persistent=True)
        self.register_buffer("reward_levels", torch.tensor(reward_quantiles), persistent=True)
        self.action = nn.Linear(action_dim, width)
        self.graph = nn.Linear(width, width)
        self.transition = nn.GRUCell(2 * width, width)
        self.latent_mean = nn.Linear(width, width)
        self.latent_log_scale = nn.Linear(width, width)
        self.relation = nn.Linear(width, 3 * relation_families)  # persist/create/delete
        self.event = nn.Linear(width, 3)  # continue/terminate/identity-change
        self.observation_mean = nn.Linear(width, observation_dim)
        self.observation_log_scale = nn.Linear(width, observation_dim)
        self.reward_base = nn.Linear(width, 1)
        self.reward_increment = nn.Linear(width, len(reward_quantiles) - 1)
        self.cost = nn.Linear(width, 1)
        self.constraint = nn.Linear(width, 1)
        self.success = nn.Linear(width, 1)

    def _heads(self, hidden: Tensor):
        reward_base = self.reward_base(hidden)
        if self.reward_levels.numel() > 1:
            reward = torch.cat((
                reward_base,
                reward_base + F.softplus(self.reward_increment(hidden)).cumsum(-1),
            ), -1)
        else:
            reward = reward_base
        return (
            self.latent_mean(hidden), self.latent_log_scale(hidden).clamp(-10, 5),
            self.relation(hidden).reshape(*hidden.shape[:-1], self.relation_families, 3),
            self.event(hidden), self.observation_mean(hidden),
            self.observation_log_scale(hidden).clamp(-10, 5), reward,
            F.softplus(self.cost(hidden)).squeeze(-1), self.constraint(hidden).squeeze(-1),
            self.success(hidden).squeeze(-1),
        )

    def forward(self, latent: Tensor, graph_summary: Tensor, action: Tensor) -> WorldModelPrediction:
        if latent.ndim != 2 or latent.shape[-1] != self.width or graph_summary.shape != latent.shape:
            raise ValueError("world latent and graph summary must be (batch,width)")
        if action.shape != (latent.shape[0], self.action_dim):
            raise ValueError("world action has invalid shape")
        transition_input = torch.cat((self.action(action), self.graph(graph_summary)), -1)
        hidden = latent
        outputs = [[] for _ in range(10)]
        selected = set(self.horizon_values)
        for step in range(1, self.horizon_values[-1] + 1):
            hidden = self.transition(transition_input, hidden)
            if step in selected:
                for collection, value in zip(outputs, self._heads(hidden), strict=True):
                    collection.append(value)
        stacked = [torch.stack(values, 1) for values in outputs]
        return WorldModelPrediction(self.horizons, *stacked)

    def rollout(
        self, shared_latent: Tensor, graph_summary: Tensor, hypothesis_residuals: Tensor,
        scenario_ids: Tensor, actions: Tensor, action_mask: Tensor,
    ) -> CounterfactualRollout:
        if shared_latent.ndim != 2 or shared_latent.shape[-1] != self.width:
            raise ValueError("shared rollout latent must be (batch,width)")
        if graph_summary.shape != shared_latent.shape:
            raise ValueError("rollout graph summary must match shared latent")
        if hypothesis_residuals.ndim != 3 or hypothesis_residuals.shape[0] != shared_latent.shape[0] or hypothesis_residuals.shape[-1] != self.width:
            raise ValueError("hypothesis residuals must be (batch,hypotheses,width)")
        base = hypothesis_residuals.shape[:2]
        if scenario_ids.shape != base or scenario_ids.dtype != torch.int64 or bool((scenario_ids <= 0).any()):
            raise ValueError("each rollout hypothesis requires a positive scenario ID")
        if actions.ndim != 4 or actions.shape[:2] != base or actions.shape[-1] != self.action_dim:
            raise ValueError("rollout actions must be (batch,hypotheses,steps,action_dim)")
        if action_mask.shape != actions.shape[:3] or action_mask.dtype != torch.bool:
            raise ValueError("rollout action mask must be boolean with action sequence shape")
        batch, hypotheses, steps = actions.shape[:3]
        hidden = shared_latent[:, None] + hypothesis_residuals
        graph = self.graph(graph_summary)[:, None].expand(-1, hypotheses, -1)
        latent_rows, reward_rows, cost_rows, constraint_rows, uncertainty_rows = [], [], [], [], []
        for step in range(steps):
            transition_input = torch.cat((self.action(actions[:, :, step]), graph), -1)
            proposed = self.transition(
                transition_input.reshape(batch * hypotheses, -1), hidden.reshape(batch * hypotheses, -1)
            ).reshape(batch, hypotheses, self.width)
            valid = action_mask[:, :, step, None]
            hidden = torch.where(valid, proposed, hidden)
            heads = self._heads(hidden)
            latent_mean, latent_log_scale = heads[:2]
            latent_rows.append(latent_mean)
            reward_rows.append(heads[6])
            cost_rows.append(heads[7])
            constraint_rows.append(torch.sigmoid(heads[8]))
            uncertainty_rows.append(torch.exp(latent_log_scale).mean(-1))
        source = torch.full(
            (batch, hypotheses, steps), int(SourceClass.SIMULATED),
            dtype=torch.int64, device=shared_latent.device,
        )
        return CounterfactualRollout(
            torch.stack(latent_rows, 2), torch.stack(reward_rows, 2),
            torch.stack(cost_rows, 2), torch.stack(constraint_rows, 2),
            torch.stack(uncertainty_rows, 2),
            scenario_ids[:, :, None].expand(-1, -1, steps), source, action_mask,
        )

    def rollout_candidates(
        self, shared_latent: Tensor, graph_summary: Tensor,
        hypothesis_residuals: Tensor, scenario_ids: Tensor,
        hypothesis_mask: Tensor, candidate_actions: Tensor,
        candidate_mask: Tensor,
    ) -> CandidateRollout:
        """Evaluate every routed hypothesis/action pair with shared parameters.

        The configured horizon set is used directly, so compute is strictly
        bounded by ``batch * hypotheses * candidates * max(horizons)``.
        """

        if shared_latent.ndim != 2 or shared_latent.shape[-1] != self.width:
            raise ValueError("candidate rollout shared latent must be (batch,width)")
        batch = shared_latent.shape[0]
        if graph_summary.shape != shared_latent.shape:
            raise ValueError("candidate rollout graph summary must match latent")
        if hypothesis_residuals.ndim != 3 or hypothesis_residuals.shape[0] != batch or hypothesis_residuals.shape[-1] != self.width:
            raise ValueError("candidate rollout hypotheses must be (batch,hypotheses,width)")
        hypotheses = hypothesis_residuals.shape[1]
        if scenario_ids.shape != (batch, hypotheses) or scenario_ids.dtype != torch.int64:
            raise ValueError("candidate rollout scenario IDs are invalid")
        if hypothesis_mask.shape != (batch, hypotheses) or hypothesis_mask.dtype != torch.bool:
            raise ValueError("candidate rollout hypothesis mask is invalid")
        if bool((hypothesis_mask & (scenario_ids <= 0)).any()):
            raise ValueError("active candidate-rollout hypotheses require positive scenarios")
        if candidate_actions.ndim != 3 or candidate_actions.shape[0] != batch or candidate_actions.shape[-1] != self.action_dim:
            raise ValueError("candidate actions must be (batch,candidates,action_dim)")
        candidates = candidate_actions.shape[1]
        if candidate_mask.shape != (batch, candidates) or candidate_mask.dtype != torch.bool:
            raise ValueError("candidate action mask is invalid")
        hidden = (
            shared_latent[:, None, None]
            + hypothesis_residuals[:, :, None]
        ).expand(-1, -1, candidates, -1).contiguous()
        graph = self.graph(graph_summary)[:, None, None].expand(
            -1, hypotheses, candidates, -1
        )
        action = self.action(candidate_actions)[:, None].expand(
            -1, hypotheses, -1, -1
        )
        valid_pair = hypothesis_mask[:, :, None] & candidate_mask[:, None, :]
        selected_horizons = set(self.horizon_values)
        rows: list[list[Tensor]] = [[] for _ in range(10)]
        for step in range(1, self.horizon_values[-1] + 1):
            proposed = self.transition(
                torch.cat((action, graph), -1).reshape(-1, 2 * self.width),
                hidden.reshape(-1, self.width),
            ).reshape(batch, hypotheses, candidates, self.width)
            hidden = torch.where(valid_pair[..., None], proposed, hidden)
            if step in selected_horizons:
                for collection, value in zip(rows, self._heads(hidden), strict=True):
                    collection.append(value)
        heads = [torch.stack(values, 3) for values in rows]
        latent_mean, latent_log_scale, relation, event, observation = heads[:5]
        reward, costs, constraint_logits, success_logits = heads[6:]
        horizons = len(self.horizon_values)
        valid = valid_pair[..., None].expand(-1, -1, -1, horizons)
        scenario = scenario_ids[:, :, None, None].expand(
            -1, -1, candidates, horizons
        ).masked_fill(~valid, 0)
        source = torch.full_like(scenario, int(SourceClass.SIMULATED)).masked_fill(
            ~valid, int(SourceClass.PREDICTED)
        )
        return CandidateRollout(
            latent_mean, relation, event, observation, reward, costs,
            torch.sigmoid(constraint_logits), torch.sigmoid(success_logits),
            torch.softmax(event, -1)[..., 1],
            torch.exp(latent_log_scale).mean(-1), scenario, source, valid,
        )


@dataclass(frozen=True, slots=True)
class InterventionResult:
    node_values: Tensor
    incoming_causal_edges: Tensor
    causal_intervention: Tensor
    conditional_simulation: Tensor


def apply_intervention(
    node_values: Tensor, incoming_causal_edges: Tensor, target_indices: Tensor,
    replacement_values: Tensor, causal_authority: Tensor,
) -> InterventionResult:
    """Apply ``do`` only where explicit causal parents and authority are present."""

    if node_values.ndim != 3 or incoming_causal_edges.shape != (
        node_values.shape[0], node_values.shape[1], node_values.shape[1]
    ):
        raise ValueError("intervention graph tensors are incompatible")
    batch = node_values.shape[0]
    if target_indices.shape != (batch,) or target_indices.dtype != torch.int64:
        raise ValueError("one int64 intervention target is required per batch")
    if replacement_values.shape != (batch, node_values.shape[-1]):
        raise ValueError("intervention replacement values have invalid shape")
    if causal_authority.shape != (batch,) or causal_authority.dtype != torch.bool:
        raise ValueError("causal authority must be boolean per batch")
    if bool(((target_indices < 0) | (target_indices >= node_values.shape[1])).any()):
        raise ValueError("intervention target lies outside node set")
    values = node_values.clone()
    edges = incoming_causal_edges.clone()
    batch_indices = torch.arange(batch, device=node_values.device)
    values[batch_indices, target_indices] = replacement_values
    has_parents = edges[batch_indices, :, target_indices].any(-1)
    causal = causal_authority & has_parents
    for index in torch.nonzero(causal, as_tuple=False).flatten().tolist():
        edges[index, :, target_indices[index]] = False
    return InterventionResult(values, edges, causal, ~causal)
