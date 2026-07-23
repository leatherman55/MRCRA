"""Bounded structured candidate actions prior to authoritative selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .tensor_state import TensorStateMixin
from .provenance import ProvenanceLedger
from .runtime_validation import runtime_validation_enabled


@dataclass(frozen=True, slots=True)
class ActionCandidateState(TensorStateMixin):
    schema_ids: Tensor
    arguments: Tensor
    argument_mask: Tensor
    proposal_logits: Tensor
    expected_reward: Tensor
    expected_cost: Tensor
    constraint_probability: Tensor
    expected_success: Tensor
    information_gain: Tensor
    tail_risk: Tensor
    expected_energy: Tensor
    normalized_utility: Tensor
    available: Tensor
    permitted: Tensor
    provenance_authorized: Tensor
    viability_authorized: Tensor
    supporting_provenance_ids: Tensor
    supporting_mask: Tensor
    selected: Tensor
    active: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.schema_ids.ndim != 2 or self.schema_ids.dtype != torch.int64:
            raise ValueError("action candidate schema IDs must be (batch,candidates)")
        base = self.schema_ids.shape
        if self.arguments.ndim != 3 or self.arguments.shape[:2] != base:
            raise ValueError("action candidate arguments must match candidate rows")
        if self.argument_mask.shape != self.arguments.shape or self.argument_mask.dtype != torch.bool:
            raise ValueError("action candidate argument mask must match arguments")
        for name in (
            "proposal_logits", "expected_reward", "expected_cost",
            "constraint_probability", "expected_success", "information_gain",
            "tail_risk", "expected_energy", "normalized_utility",
        ):
            value = getattr(self, name)
            if value.shape != base or not value.is_floating_point():
                raise ValueError(f"action candidate {name} must match rows")
        for name in (
            "available", "permitted", "provenance_authorized", "viability_authorized",
            "selected", "active",
        ):
            value = getattr(self, name)
            if value.shape != base or value.dtype != torch.bool:
                raise ValueError(f"action candidate {name} must be boolean rows")
        if self.supporting_provenance_ids.ndim != 3 or self.supporting_provenance_ids.shape[:2] != base or self.supporting_provenance_ids.dtype != torch.int64:
            raise ValueError("action candidate supporters must be int64 candidate rows")
        if self.supporting_mask.shape != self.supporting_provenance_ids.shape or self.supporting_mask.dtype != torch.bool:
            raise ValueError("action candidate supporter mask is invalid")
        if bool((self.active & (self.schema_ids < 0)).any()):
            raise ValueError("active action candidates require a schema")
        if bool((self.supporting_mask & (self.supporting_provenance_ids < 0)).any()):
            raise ValueError("active action supporters require provenance")
        if bool((self.selected.sum(-1) > 1).any()):
            raise ValueError("at most one action candidate may be selected")
        if bool((self.selected & ~self.active).any()):
            raise ValueError("only active action candidates may be selected")
        for name in ("constraint_probability", "expected_success"):
            value = getattr(self, name)
            if bool(((value < 0) | (value > 1)).any()):
                raise ValueError(f"action candidate {name} must lie in [0,1]")
        if bool((self.expected_cost < 0).any() | (self.tail_risk < 0).any() | (self.expected_energy < 0).any()):
            raise ValueError("action candidate cost, risk, and energy cannot be negative")

    @classmethod
    def empty(
        cls, batch: int, capacity: int, argument_dim: int, supporter_capacity: int,
        *, device=None, dtype=None,
    ) -> "ActionCandidateState":
        if min(batch, capacity, argument_dim, supporter_capacity) <= 0:
            raise ValueError("action candidate dimensions must be positive")
        base = (batch, capacity)
        zeros = lambda: torch.zeros(base, device=device, dtype=dtype)
        flags = lambda: torch.zeros(base, dtype=torch.bool, device=device)
        return cls(
            torch.full(base, -1, dtype=torch.int64, device=device),
            torch.zeros(*base, argument_dim, device=device, dtype=dtype),
            torch.zeros(*base, argument_dim, dtype=torch.bool, device=device),
            zeros(), zeros(), zeros(), zeros(), zeros(), zeros(), zeros(), zeros(), zeros(),
            flags(), flags(), flags(), flags(),
            torch.full((*base, supporter_capacity), -1, dtype=torch.int64, device=device),
            torch.zeros(*base, supporter_capacity, dtype=torch.bool, device=device),
            flags(), flags(),
        )

    @property
    def batch(self) -> int:
        return self.schema_ids.shape[0]

    @property
    def capacity(self) -> int:
        return self.schema_ids.shape[1]


def build_action_candidates(
    *, proposal_logits: Tensor, expected_reward: Tensor, expected_cost: Tensor,
    constraint_probability: Tensor, expected_success: Tensor,
    available: Tensor, permission_mask: Tensor,
    supporting_provenance_ids: Tensor, supporting_mask: Tensor,
    capacity: int, argument_dim: int,
) -> ActionCandidateState:
    """Route a bounded, diverse subset of schema proposals into candidate state."""

    if proposal_logits.ndim != 2 or capacity <= 0 or argument_dim <= 0:
        raise ValueError("candidate proposal dimensions are invalid")
    batch, schemas = proposal_logits.shape
    for name, value in (
        ("expected_reward", expected_reward), ("expected_cost", expected_cost),
        ("constraint_probability", constraint_probability),
        ("expected_success", expected_success),
    ):
        if value.shape != proposal_logits.shape:
            raise ValueError(f"candidate proposal {name} must match logits")
    if available.shape != proposal_logits.shape or available.dtype != torch.bool:
        raise ValueError("candidate proposal availability is invalid")
    if permission_mask.shape != proposal_logits.shape or permission_mask.dtype != torch.bool:
        raise ValueError("candidate proposal permissions are invalid")
    if supporting_provenance_ids.ndim != 2 or supporting_provenance_ids.shape[0] != batch:
        raise ValueError("candidate proposal supporters must be (batch,supporters)")
    if supporting_mask.shape != supporting_provenance_ids.shape or supporting_mask.dtype != torch.bool:
        raise ValueError("candidate proposal supporter mask is invalid")
    routed = min(capacity, schemas)
    score = proposal_logits + expected_success + expected_reward - expected_cost
    # Permissions do not influence proposal diversity, but unavailable schemas
    # are excluded because they cannot be meaningfully evaluated in this state.
    top = score.masked_fill(~available, -torch.inf).topk(routed, -1)
    indices = top.indices
    valid = torch.isfinite(top.values)
    gather = lambda value: torch.gather(value, 1, indices)
    state = ActionCandidateState.empty(
        batch, capacity, argument_dim, supporting_provenance_ids.shape[1],
        device=proposal_logits.device, dtype=proposal_logits.dtype,
    )
    values = {
        name: getattr(state, name).clone()
        for name in state.__dataclass_fields__
    }
    values["schema_ids"][:, :routed] = indices.masked_fill(~valid, -1)
    values["proposal_logits"][:, :routed] = top.values.masked_fill(~valid, 0)
    values["expected_reward"][:, :routed] = gather(expected_reward).masked_fill(~valid, 0)
    values["expected_cost"][:, :routed] = gather(expected_cost).masked_fill(~valid, 0)
    values["constraint_probability"][:, :routed] = gather(constraint_probability).masked_fill(~valid, 0)
    values["expected_success"][:, :routed] = gather(expected_success).masked_fill(~valid, 0)
    values["available"][:, :routed] = valid & gather(available)
    values["permitted"][:, :routed] = valid & gather(permission_mask)
    values["supporting_provenance_ids"][:, :routed] = supporting_provenance_ids[:, None]
    values["supporting_mask"][:, :routed] = supporting_mask[:, None] & valid[..., None]
    values["active"][:, :routed] = valid
    return ActionCandidateState(**values)


def _replace(state: ActionCandidateState, **changes) -> ActionCandidateState:
    values = {name: getattr(state, name) for name in state.__dataclass_fields__}
    values.update(changes)
    return ActionCandidateState(**values)


def authorize_candidate_provenance(
    state: ActionCandidateState, ledger: ProvenanceLedger,
) -> ActionCandidateState:
    """Authorize candidate support using immutable ledger records only."""

    authorized = torch.zeros_like(state.active)
    for row, candidate in torch.nonzero(state.active, as_tuple=False).tolist():
        supporters = state.supporting_provenance_ids[row, candidate][
            state.supporting_mask[row, candidate]
        ]
        if supporters.numel() == 0:
            continue
        valid = True
        for record_id in supporters.tolist():
            try:
                valid = valid and ledger.can_justify_external_action(int(record_id))
            except KeyError:
                valid = False
        authorized[row, candidate] = valid
    return _replace(state, provenance_authorized=authorized)


def evaluate_candidate_rollout(
    state: ActionCandidateState, *, reward_quantiles: Tensor, costs: Tensor,
    constraint_probabilities: Tensor, success_probabilities: Tensor,
    uncertainty: Tensor, hypothesis_weights: Tensor, rollout_mask: Tensor,
) -> ActionCandidateState:
    """Posterior-average the consequence lattice without collapsing tail risk."""

    # Rollout tensors are (batch,hypotheses,candidates,horizons[,quantiles]).
    base = reward_quantiles.shape[:4]
    batch, hypotheses, candidates, _ = base
    if (batch, candidates) != state.schema_ids.shape:
        raise ValueError("candidate rollout does not match candidate state")
    if reward_quantiles.ndim != 5:
        raise ValueError("candidate reward must include quantiles")
    for name, value in (
        ("costs", costs), ("constraint_probabilities", constraint_probabilities),
        ("success_probabilities", success_probabilities), ("uncertainty", uncertainty),
    ):
        if value.shape != base:
            raise ValueError(f"candidate rollout {name} must share its lattice")
    if rollout_mask.shape != base or rollout_mask.dtype != torch.bool:
        raise ValueError("candidate rollout mask is invalid")
    if hypothesis_weights.shape != (batch, hypotheses):
        raise ValueError("candidate hypothesis weights are invalid")
    final_mask = rollout_mask[..., -1]
    weights = hypothesis_weights[:, :, None] * final_mask.to(hypothesis_weights.dtype)
    weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
    median_index = reward_quantiles.shape[-1] // 2
    final_reward = reward_quantiles[..., -1, median_index]
    lower_reward = reward_quantiles[..., -1, 0]
    expected_reward = (final_reward * weights).sum(1)
    expected_cost = (costs[..., -1] * weights).sum(1)
    constraint = (constraint_probabilities[..., -1] * weights).sum(1)
    success = (success_probabilities[..., -1] * weights).sum(1)
    tail_risk = ((final_reward - lower_reward).clamp_min(0) * weights).sum(1)
    expected_energy = (uncertainty[..., -1] * weights).sum(1)
    prior_entropy = -(
        hypothesis_weights.clamp_min(1e-8)
        * hypothesis_weights.clamp_min(1e-8).log()
    ).sum(-1, keepdim=True)
    likelihood = torch.exp(-uncertainty[..., -1]).masked_fill(~final_mask, 0)
    posterior = hypothesis_weights[:, :, None] * likelihood
    posterior = posterior / posterior.sum(1, keepdim=True).clamp_min(1e-8)
    posterior_entropy = -(
        posterior.clamp_min(1e-8) * posterior.clamp_min(1e-8).log()
    ).sum(1)
    information_gain = (prior_entropy - posterior_entropy).clamp_min(0)
    return _replace(
        state, expected_reward=expected_reward, expected_cost=expected_cost,
        constraint_probability=constraint, expected_success=success,
        information_gain=information_gain, tail_risk=tail_risk,
        expected_energy=expected_energy,
    )


def _unit_normalize(value: Tensor, mask: Tensor) -> Tensor:
    minimum = value.masked_fill(~mask, torch.inf).amin(-1, keepdim=True)
    maximum = value.masked_fill(~mask, -torch.inf).amax(-1, keepdim=True)
    minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))
    maximum = torch.where(torch.isfinite(maximum), maximum, minimum)
    return ((value - minimum) / (maximum - minimum).clamp_min(1e-6)).masked_fill(~mask, 0)


def select_action_candidate(
    state: ActionCandidateState, *, information_gain_weight: float = 0.25,
    cost_weight: float = 1.0, energy_weight: float = 0.25,
    risk_weight: float = 1.0, maximum_constraint_probability: float = 0.5,
) -> ActionCandidateState:
    """Apply all hard gates before comparing normalized consequence utility."""

    if min(information_gain_weight, cost_weight, energy_weight, risk_weight) < 0:
        raise ValueError("candidate utility weights cannot be negative")
    if not 0 <= maximum_constraint_probability <= 1:
        raise ValueError("candidate constraint threshold must lie in [0,1]")
    eligible = (
        state.active & state.available & state.permitted
        & state.provenance_authorized & state.viability_authorized
        & (state.constraint_probability <= maximum_constraint_probability)
    )
    utility = (
        _unit_normalize(state.expected_reward, eligible)
        + _unit_normalize(state.expected_success, eligible)
        + information_gain_weight * _unit_normalize(state.information_gain, eligible)
        - cost_weight * _unit_normalize(state.expected_cost, eligible)
        - energy_weight * _unit_normalize(state.expected_energy, eligible)
        - risk_weight * _unit_normalize(state.tail_risk, eligible)
        - _unit_normalize(state.constraint_probability, eligible)
    ).masked_fill(~eligible, -torch.inf)
    has_choice = eligible.any(-1)
    chosen = utility.argmax(-1)
    selected = torch.zeros_like(eligible)
    rows = torch.nonzero(has_choice, as_tuple=False).flatten()
    if rows.numel():
        selected[rows, chosen[rows]] = True
    return _replace(
        state, normalized_utility=utility.masked_fill(~eligible, 0),
        selected=selected,
    )
