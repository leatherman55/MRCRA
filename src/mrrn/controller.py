"""Budgeted factorized internal-action controller and system-model state."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cognitive_types import InternalAction, RelationFamily
from .memory_v2 import MemoryTier
from .runtime_validation import runtime_validation_enabled


@dataclass(frozen=True, slots=True)
class GoalState:
    desired_outcomes: Tensor
    constraints: Tensor
    priorities: Tensor
    horizons: Tensor
    authority: Tensor
    termination: Tensor
    mask: Tensor
    provenance_ids: Tensor | None = None
    status: Tensor | None = None
    progress: Tensor | None = None
    conflict_mask: Tensor | None = None

    def __post_init__(self) -> None:
        if self.desired_outcomes.ndim != 3 or self.constraints.ndim != 3:
            raise ValueError("goals and constraints must be (batch,goals,features)")
        base = self.desired_outcomes.shape[:2]
        if self.constraints.shape[:2] != base:
            raise ValueError("goal constraints must share goal slots")
        for name in ("priorities", "horizons", "authority", "termination"):
            if getattr(self, name).shape != base:
                raise ValueError(f"goal {name} has invalid shape")
        if self.mask.shape != base or self.mask.dtype != torch.bool:
            raise ValueError("goal mask must be boolean with slot shape")
        if self.provenance_ids is None:
            object.__setattr__(
                self, "provenance_ids",
                torch.full(base, -1, dtype=torch.int64, device=self.mask.device),
            )
        if self.status is None:
            object.__setattr__(
                self, "status", self.mask.to(torch.int64),
            )
        if self.progress is None:
            object.__setattr__(
                self, "progress", torch.zeros(
                    base, device=self.desired_outcomes.device,
                    dtype=self.desired_outcomes.dtype,
                ),
            )
        if self.conflict_mask is None:
            object.__setattr__(
                self, "conflict_mask", torch.zeros(
                    *base, base[1], dtype=torch.bool, device=self.mask.device,
                ),
            )
        if not runtime_validation_enabled():
            return
        if self.provenance_ids.shape != base or self.provenance_ids.dtype != torch.int64:
            raise ValueError("goal provenance IDs must be int64 with slot shape")
        if self.status.shape != base or self.status.dtype != torch.int64:
            raise ValueError("goal status must be int64 with slot shape")
        if self.progress.shape != base or not self.progress.is_floating_point():
            raise ValueError("goal progress must be floating point with slot shape")
        if self.conflict_mask.shape != (*base, base[1]) or self.conflict_mask.dtype != torch.bool:
            raise ValueError("goal conflicts must be boolean (batch,goals,goals)")
        if not torch.equal(self.conflict_mask, self.conflict_mask.transpose(-1, -2)):
            raise ValueError("goal conflict relation must be symmetric")
        if bool(self.conflict_mask.diagonal(dim1=-2, dim2=-1).any()):
            raise ValueError("a goal cannot conflict with itself")
        if bool(((self.progress < 0) | (self.progress > 1)).any()):
            raise ValueError("goal progress must lie in [0,1]")
        if bool((self.mask & ((self.priorities < 0) | (self.horizons <= 0))).any()):
            raise ValueError("active goals require nonnegative priority and positive horizon")

    def summary(self) -> Tensor:
        """Return the dominant authorized goal without averaging conflicts away."""

        eligible = self.mask & (self.authority > 0) & (self.status == 1)
        score = (
            self.priorities.clamp_min(0) * self.authority.clamp_min(0)
            / self.horizons.clamp_min(1e-8)
        ).masked_fill(~eligible, -torch.inf)
        has_goal = eligible.any(-1)
        selected = score.argmax(-1)
        rows = torch.arange(self.desired_outcomes.shape[0], device=selected.device)
        summary = self.desired_outcomes[rows, selected]
        return summary.masked_fill(~has_goal[:, None], 0)

    def detach(self) -> "GoalState":
        return GoalState(*(
            getattr(self, field.name).detach()
            if isinstance(getattr(self, field.name), Tensor)
            else getattr(self, field.name)
            for field in fields(self)
        ))

    def to(self, *args, **kwargs) -> "GoalState":
        return GoalState(*(
            getattr(self, field.name).to(*args, **kwargs)
            if isinstance(getattr(self, field.name), Tensor)
            else getattr(self, field.name)
            for field in fields(self)
        ))


@dataclass(frozen=True, slots=True)
class SystemModelState:
    modality_availability: Tensor
    action_availability: Tensor
    action_success: Tensor
    action_latency: Tensor
    action_reward: Tensor
    action_cost: Tensor
    action_constraint_violation: Tensor
    action_reversibility: Tensor
    executor_reliability: Tensor
    memory_reliability: Tensor
    router_reliability: Tensor
    remaining_compute: Tensor
    remaining_memory: Tensor
    permission_mask: Tensor
    calibration_regime: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.modality_availability.ndim != 2 or self.action_availability.ndim != 2:
            raise ValueError("system availability tensors must be (batch,channels)")
        batch = self.modality_availability.shape[0]
        action_shape = self.action_availability.shape
        for name in (
            "action_success", "action_latency", "action_reward", "action_cost",
            "action_constraint_violation", "action_reversibility",
            "executor_reliability", "permission_mask",
        ):
            value = getattr(self, name)
            if value.shape != action_shape:
                raise ValueError(f"system {name} must match action availability")
        if self.permission_mask.dtype != torch.bool:
            raise ValueError("system permission mask must be boolean")
        for name in (
            "memory_reliability", "router_reliability", "remaining_compute",
            "remaining_memory", "calibration_regime",
        ):
            value = getattr(self, name)
            if value.ndim != 2 or value.shape[0] != batch:
                raise ValueError(f"system {name} must have a batch dimension")
        if bool((self.remaining_compute < 0).any() | (self.remaining_memory < 0).any()):
            raise ValueError("system budgets cannot be negative")
        bounded = (
            self.action_success, self.action_constraint_violation,
            self.action_reversibility, self.executor_reliability,
        )
        if any(bool(((value < 0) | (value > 1)).any()) for value in bounded):
            raise ValueError("bounded system consequence estimates must lie in [0,1]")
        if bool((self.action_latency < 0).any() | (self.action_cost < 0).any()):
            raise ValueError("system latency and cost estimates cannot be negative")

    def features(self) -> Tensor:
        return torch.cat((
            self.modality_availability, self.action_availability,
            self.action_success, self.action_latency, self.action_reward,
            self.action_cost, self.action_constraint_violation,
            self.action_reversibility, self.executor_reliability,
            self.memory_reliability, self.router_reliability,
            self.remaining_compute, self.remaining_memory,
            self.calibration_regime,
        ), -1)

    def detach(self) -> "SystemModelState":
        return SystemModelState(*(
            getattr(self, field.name).detach() for field in fields(self)
        ))

    def to(self, *args, **kwargs) -> "SystemModelState":
        return SystemModelState(*(
            getattr(self, field.name).to(*args, **kwargs) for field in fields(self)
        ))


@dataclass(frozen=True, slots=True)
class ControllerState:
    hidden: Tensor
    remaining_steps: Tensor
    halted: Tensor
    abstained: Tensor
    action_history: Tensor
    history_mask: Tensor
    step: int

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.hidden.ndim != 2:
            raise ValueError("controller hidden state must be (batch,width)")
        batch = self.hidden.shape[0]
        for name in ("remaining_steps",):
            value = getattr(self, name)
            if value.shape != (batch,) or value.dtype != torch.int64:
                raise ValueError(f"controller {name} must be int64 per batch")
        for name in ("halted", "abstained"):
            value = getattr(self, name)
            if value.shape != (batch,) or value.dtype != torch.bool:
                raise ValueError(f"controller {name} must be boolean per batch")
        if self.action_history.ndim != 2 or self.action_history.shape[0] != batch or self.action_history.dtype != torch.int64:
            raise ValueError("controller action history must be int64 (batch,steps)")
        if self.history_mask.shape != self.action_history.shape or self.history_mask.dtype != torch.bool:
            raise ValueError("controller history mask must match history")
        if self.step < 0 or bool((self.remaining_steps < 0).any()):
            raise ValueError("controller steps cannot be negative")

    def detach(self) -> "ControllerState":
        return ControllerState(
            self.hidden.detach(), self.remaining_steps.detach(), self.halted.detach(),
            self.abstained.detach(), self.action_history.detach(),
            self.history_mask.detach(), self.step,
        )


@dataclass(frozen=True, slots=True)
class ControllerDecision:
    action_logits: Tensor
    action: Tensor
    node_pointer_logits: Tensor
    node_pointer: Tensor
    relation_pointer_logits: Tensor
    relation_pointer: Tensor
    relation_logits: Tensor
    relation_family: Tensor
    horizon_logits: Tensor
    horizon_index: Tensor
    memory_tier_logits: Tensor
    memory_tier: Tensor
    halt_probability: Tensor
    active: Tensor
    secondary_node_pointer: Tensor
    secondary_relation_pointer: Tensor
    abstraction_pointer: Tensor
    requested_physical_scale: Tensor
    requested_abstraction_depth: Tensor
    precision_tolerance: Tensor
    argument_schema_id: Tensor
    arguments: Tensor
    argument_mask: Tensor
    trigger_class: Tensor
    expected_operation_cost: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.action_logits.ndim != 2:
            raise ValueError("controller action logits must be (batch,actions)")
        batch = self.action_logits.shape[0]
        for name in (
            "action", "node_pointer", "relation_pointer", "relation_family",
            "horizon_index", "memory_tier", "secondary_node_pointer",
            "secondary_relation_pointer", "abstraction_pointer",
            "requested_physical_scale", "requested_abstraction_depth",
            "argument_schema_id", "trigger_class",
        ):
            value = getattr(self, name)
            if value.shape != (batch,) or value.dtype != torch.int64:
                raise ValueError(f"controller decision {name} must be int64 per batch")
        for name in (
            "node_pointer_logits", "relation_pointer_logits", "relation_logits",
            "horizon_logits", "memory_tier_logits",
        ):
            value = getattr(self, name)
            if value.ndim != 2 or value.shape[0] != batch:
                raise ValueError(f"controller decision {name} must have a batch dimension")
        for name in ("halt_probability", "precision_tolerance", "expected_operation_cost"):
            value = getattr(self, name)
            if value.shape != (batch,) or not value.is_floating_point():
                raise ValueError(f"controller decision {name} must be floating point per batch")
        if self.active.shape != (batch,) or self.active.dtype != torch.bool:
            raise ValueError("controller decision active mask must be boolean per batch")
        if self.arguments.ndim != 2 or self.arguments.shape[0] != batch:
            raise ValueError("controller decision arguments must be (batch,arguments)")
        if self.argument_mask.shape != self.arguments.shape or self.argument_mask.dtype != torch.bool:
            raise ValueError("controller decision argument mask must match arguments")
        if bool((self.precision_tolerance < 0).any() | (self.expected_operation_cost < 0).any()):
            raise ValueError("controller precision and operation cost cannot be negative")


@dataclass(frozen=True, slots=True)
class ControllerRollout:
    decisions: tuple[ControllerDecision, ...]
    state: ControllerState
    expected_steps: Tensor
    ponder_cost: Tensor


class AdaptiveController(nn.Module):
    """Choose action type and arguments separately under a hard microstep cap."""

    def __init__(
        self, width: int, goal_dim: int, uncertainty_channels: int,
        system_feature_dim: int, maximum_nodes: int, *, maximum_steps: int = 4,
        horizon_count: int = 4, maximum_relations: int | None = None,
        action_argument_dim: int = 8,
    ) -> None:
        super().__init__()
        if min(
            width, goal_dim, uncertainty_channels, system_feature_dim,
            maximum_nodes, maximum_steps, horizon_count, action_argument_dim,
        ) <= 0:
            raise ValueError("controller dimensions must be positive")
        self.width = width
        self.maximum_nodes = maximum_nodes
        self.maximum_relations = maximum_nodes if maximum_relations is None else maximum_relations
        if self.maximum_relations <= 0:
            raise ValueError("controller maximum_relations must be positive")
        self.maximum_steps = maximum_steps
        self.action_argument_dim = action_argument_dim
        input_dim = width + goal_dim + uncertainty_channels + system_feature_dim
        self.input = nn.Linear(input_dim, width)
        self.recurrence = nn.GRUCell(width, width)
        self.action_head = nn.Linear(width, len(InternalAction))
        self.node_query = nn.Linear(width, width, bias=False)
        self.node_key = nn.Linear(width, width, bias=False)
        self.relation_query = nn.Linear(width, width, bias=False)
        self.relation_key = nn.Linear(width, width, bias=False)
        self.relation_head = nn.Linear(width, len(RelationFamily))
        self.horizon_head = nn.Linear(width, horizon_count)
        self.memory_tier_head = nn.Linear(width, len(MemoryTier))
        self.halt_head = nn.Linear(width, 1)
        nn.init.constant_(self.halt_head.bias, -1.0)

    def initial_state(self, batch: int, *, device=None, dtype=None) -> ControllerState:
        if batch <= 0:
            raise ValueError("controller batch must be positive")
        return ControllerState(
            torch.zeros(batch, self.width, device=device, dtype=dtype),
            torch.full((batch,), self.maximum_steps, dtype=torch.int64, device=device),
            torch.zeros(batch, dtype=torch.bool, device=device),
            torch.zeros(batch, dtype=torch.bool, device=device),
            torch.full((batch, self.maximum_steps), -1, dtype=torch.int64, device=device),
            torch.zeros(batch, self.maximum_steps, dtype=torch.bool, device=device), 0,
        )

    def begin_cycle(
        self, previous: ControllerState, active_rows: Tensor,
    ) -> ControllerState:
        """Start a new bounded microstep cycle while preserving learned fast state.

        ``ControllerState.step`` and its fixed-size history are cycle-local.  The
        recurrent hidden state and episode-level abstention flag are persistent.
        Reinitializing the entire controller here would erase precisely the
        self-model/control continuity represented by the runtime contract.
        """

        batch = previous.hidden.shape[0]
        if active_rows.shape != (batch,) or active_rows.dtype != torch.bool:
            raise ValueError("controller cycle mask must be boolean per batch")
        history = torch.full_like(previous.action_history, -1)
        history_mask = torch.zeros_like(previous.history_mask)
        return ControllerState(
            previous.hidden,
            torch.where(
                active_rows,
                torch.full_like(previous.remaining_steps, self.maximum_steps),
                torch.zeros_like(previous.remaining_steps),
            ),
            ~active_rows,
            previous.abstained,
            history,
            history_mask,
            0,
        )

    def step(
        self, state: ControllerState, workspace: Tensor, goals: Tensor,
        uncertainty: Tensor, system_features: Tensor, nodes: Tensor,
        node_mask: Tensor, *, relations: Tensor | None = None,
        relation_mask: Tensor | None = None, action_mask: Tensor | None = None,
        action_bias: Tensor | None = None,
    ) -> tuple[ControllerDecision, ControllerState]:
        batch = state.hidden.shape[0]
        if workspace.shape != (batch, self.width) or goals.shape[0] != batch:
            raise ValueError("controller workspace or goals have invalid shape")
        if uncertainty.shape[0] != batch or system_features.shape[0] != batch:
            raise ValueError("controller uncertainty or system features have invalid shape")
        if nodes.shape != (batch, self.maximum_nodes, self.width):
            raise ValueError("controller node tensor has invalid shape")
        if node_mask.shape != (batch, self.maximum_nodes) or node_mask.dtype != torch.bool:
            raise ValueError("controller node mask is invalid")
        if relations is None:
            relations = nodes.new_zeros(batch, self.maximum_relations, self.width)
        if relation_mask is None:
            relation_mask = torch.zeros(
                batch, self.maximum_relations, dtype=torch.bool, device=nodes.device
            )
        if relations.shape != (batch, self.maximum_relations, self.width):
            raise ValueError("controller relation tensor has invalid shape")
        if relation_mask.shape != (batch, self.maximum_relations) or relation_mask.dtype != torch.bool:
            raise ValueError("controller relation mask is invalid")
        if state.step >= self.maximum_steps:
            raise ValueError("controller microstep budget exhausted")
        active = ~state.halted & (state.remaining_steps > 0)
        input_features = torch.cat((
            workspace, goals, uncertainty.detach(), system_features,
        ), -1)
        hidden_proposal = self.recurrence(F.silu(self.input(input_features)), state.hidden)
        hidden = torch.where(active[:, None], hidden_proposal, state.hidden)
        action_logits = self.action_head(hidden)
        if action_bias is not None:
            if action_bias.shape != action_logits.shape or not action_bias.is_floating_point():
                raise ValueError("controller action bias must be floating point with action-logit shape")
            if not bool(torch.isfinite(action_bias).all()):
                raise ValueError("controller action bias must be finite")
            action_logits = action_logits + action_bias
        if action_mask is not None:
            if action_mask.shape != action_logits.shape or action_mask.dtype != torch.bool:
                raise ValueError("controller action mask must be boolean with action-logit shape")
            if bool((active & ~action_mask.any(-1)).any()):
                raise ValueError("every active controller row requires an allowed action")
            action_logits = action_logits.masked_fill(~action_mask, -torch.inf)
        action = action_logits.argmax(-1)
        halt_probability = torch.sigmoid(self.halt_head(hidden)).squeeze(-1)
        action = torch.where(
            halt_probability >= 0.5,
            torch.full_like(action, int(InternalAction.HALT)), action,
        )
        action = torch.where(active, action, torch.full_like(action, int(InternalAction.HALT)))
        node_logits = torch.einsum(
            "bd,bnd->bn", self.node_query(hidden), self.node_key(nodes)
        ) / self.width ** 0.5
        node_logits = node_logits.masked_fill(~node_mask, -torch.inf)
        node_pointer = node_logits.argmax(-1).masked_fill(~node_mask.any(-1), -1)
        if self.maximum_nodes > 1:
            secondary_node_pointer = node_logits.topk(2, -1).indices[:, 1]
            secondary_node_pointer = secondary_node_pointer.masked_fill(node_mask.sum(-1) < 2, -1)
        else:
            secondary_node_pointer = torch.full_like(node_pointer, -1)
        relation_pointer_logits = torch.einsum(
            "bd,bed->be", self.relation_query(hidden), self.relation_key(relations)
        ) / self.width ** 0.5
        relation_pointer_logits = relation_pointer_logits.masked_fill(~relation_mask, -torch.inf)
        relation_pointer = relation_pointer_logits.argmax(-1).masked_fill(~relation_mask.any(-1), -1)
        if self.maximum_relations > 1:
            secondary_relation_pointer = relation_pointer_logits.topk(2, -1).indices[:, 1]
            secondary_relation_pointer = secondary_relation_pointer.masked_fill(
                relation_mask.sum(-1) < 2, -1
            )
        else:
            secondary_relation_pointer = torch.full_like(relation_pointer, -1)
        relation_logits = self.relation_head(hidden)
        horizon_logits = self.horizon_head(hidden)
        memory_logits = self.memory_tier_head(hidden)
        halted = state.halted | (active & (action == int(InternalAction.HALT)))
        abstained = state.abstained | (
            active & (action == int(InternalAction.ABSTAIN_OR_REQUEST_EXTERNAL_EVIDENCE))
        )
        remaining = (state.remaining_steps - active.to(torch.int64)).clamp_min(0)
        halted = halted | (remaining == 0)
        history = state.action_history.clone()
        history_mask = state.history_mask.clone()
        history[:, state.step] = action
        history_mask[:, state.step] = active
        decision = ControllerDecision(
            action_logits, action, node_logits, node_pointer,
            relation_pointer_logits, relation_pointer, relation_logits,
            relation_logits.argmax(-1), horizon_logits, horizon_logits.argmax(-1),
            memory_logits, memory_logits.argmax(-1), halt_probability, active,
            secondary_node_pointer, secondary_relation_pointer, node_pointer,
            torch.zeros(batch, dtype=torch.int64, device=workspace.device),
            torch.zeros(batch, dtype=torch.int64, device=workspace.device),
            workspace.new_zeros(batch),
            torch.full((batch,), -1, dtype=torch.int64, device=workspace.device),
            workspace.new_zeros(batch, self.action_argument_dim),
            torch.zeros(
                batch, self.action_argument_dim, dtype=torch.bool, device=workspace.device
            ),
            torch.zeros(batch, dtype=torch.int64, device=workspace.device),
            workspace.new_zeros(batch),
        )
        return decision, ControllerState(
            hidden, remaining, halted, abstained, history, history_mask, state.step + 1,
        )

    def forward(
        self, workspace: Tensor, goals: Tensor, uncertainty: Tensor,
        system_features: Tensor, nodes: Tensor, node_mask: Tensor,
        *, relations: Tensor | None = None, relation_mask: Tensor | None = None,
        state: ControllerState | None = None, action_mask: Tensor | None = None,
        action_bias: Tensor | None = None,
        ponder_weight: float = 0.01,
    ) -> ControllerRollout:
        if ponder_weight < 0:
            raise ValueError("ponder weight cannot be negative")
        batch = workspace.shape[0]
        if state is None:
            state = self.initial_state(batch, device=workspace.device, dtype=workspace.dtype)
        decisions = []
        survival = torch.ones(batch, device=workspace.device, dtype=workspace.dtype)
        expected_steps = torch.zeros_like(survival)
        while state.step < self.maximum_steps and bool((~state.halted).any()):
            decision, state = self.step(
                state, workspace, goals, uncertainty, system_features,
                nodes, node_mask, relations=relations, relation_mask=relation_mask,
                action_mask=action_mask, action_bias=action_bias,
            )
            decisions.append(decision)
            expected_steps = expected_steps + survival * decision.active.to(survival.dtype)
            survival = survival * (1 - decision.halt_probability)
        return ControllerRollout(
            tuple(decisions), state, expected_steps, ponder_weight * expected_steps.mean(),
        )


@dataclass(frozen=True, slots=True)
class OperationalSchemaState:
    probabilities: Tensor
    active_schema: Tensor
    dwell_steps: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.probabilities.ndim != 2:
            raise ValueError("schema probabilities must be (batch,schemas)")
        batch = self.probabilities.shape[0]
        for name in ("active_schema", "dwell_steps"):
            value = getattr(self, name)
            if value.shape != (batch,) or value.dtype != torch.int64:
                raise ValueError(f"schema {name} must be int64 per batch")
        if bool((self.probabilities < 0).any()):
            raise ValueError("schema probabilities cannot be negative")

    def detach(self) -> "OperationalSchemaState":
        return OperationalSchemaState(
            self.probabilities.detach(), self.active_schema.detach(),
            self.dwell_steps.detach(),
        )

    def to(self, *args, **kwargs) -> "OperationalSchemaState":
        dtype = kwargs.get("dtype")
        integer_kwargs = dict(kwargs)
        integer_kwargs.pop("dtype", None)
        return OperationalSchemaState(
            self.probabilities.to(*args, **kwargs),
            self.active_schema.to(*args, **integer_kwargs),
            self.dwell_steps.to(*args, **integer_kwargs),
        )


class OperationalSchemas(nn.Module):
    """Sparse context priors with entropy floor and switching hysteresis."""

    def __init__(
        self, width: int, schema_count: int, *, entropy_floor: float = 0.2,
        switching_margin: float = 0.1, minimum_dwell: int = 2,
    ) -> None:
        super().__init__()
        if min(width, schema_count, minimum_dwell) <= 0:
            raise ValueError("operational schema dimensions must be positive")
        if not 0 <= entropy_floor <= 1 or switching_margin < 0:
            raise ValueError("operational schema controls are invalid")
        self.schema_count = schema_count
        self.entropy_floor = entropy_floor
        self.switching_margin = switching_margin
        self.minimum_dwell = minimum_dwell
        self.logits = nn.Linear(width, schema_count)
        self.adapters = nn.Parameter(torch.empty(schema_count, width))
        nn.init.normal_(self.adapters, std=width ** -0.5)

    def initial_state(self, batch: int, *, device=None, dtype=None) -> OperationalSchemaState:
        probability = torch.full(
            (batch, self.schema_count), 1 / self.schema_count, device=device, dtype=dtype
        )
        return OperationalSchemaState(
            probability, torch.zeros(batch, dtype=torch.int64, device=device),
            torch.zeros(batch, dtype=torch.int64, device=device),
        )

    def forward(
        self, context: Tensor, state: OperationalSchemaState | None = None,
    ) -> tuple[Tensor, OperationalSchemaState]:
        if context.ndim != 2:
            raise ValueError("schema context must be (batch,width)")
        if state is None:
            state = self.initial_state(context.shape[0], device=context.device, dtype=context.dtype)
        learned = torch.softmax(self.logits(context), -1)
        uniform = torch.full_like(learned, 1 / self.schema_count)
        probability = (1 - self.entropy_floor) * learned + self.entropy_floor * uniform
        proposal = probability.argmax(-1)
        current_score = probability.gather(-1, state.active_schema[:, None]).squeeze(-1)
        proposal_score = probability.gather(-1, proposal[:, None]).squeeze(-1)
        switch = (
            (proposal != state.active_schema)
            & (state.dwell_steps >= self.minimum_dwell)
            & (proposal_score >= current_score + self.switching_margin)
        )
        active = torch.where(switch, proposal, state.active_schema)
        dwell = torch.where(switch, torch.zeros_like(state.dwell_steps), state.dwell_steps + 1)
        modulation = torch.einsum("bs,sd->bd", probability, self.adapters)
        return modulation, OperationalSchemaState(probability, active, dwell)
