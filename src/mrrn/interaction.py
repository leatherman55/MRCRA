"""Closed-loop external action contracts for MRCRA.

Internal cognitive actions and environment-changing actions are intentionally
different ontologies.  The former edit bounded internal state; the latter are
proposals that require capability, permission, provenance, and an external
executor.  This module supplies the learned proposal policy, hard authorization
checks, feedback transition, and executor protocol without pretending that a
neural tensor has itself changed the environment.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterable, Protocol, TypeVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .runtime_validation import runtime_validation_enabled

from .controller import GoalState, SystemModelState
from .action_candidates import ActionCandidateState
from .provenance import ProvenanceLedger


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ActionParameterSpec:
    name: str
    lower: float
    upper: float
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name or self.lower > self.upper:
            raise ValueError("action parameter specification is invalid")


@dataclass(frozen=True, slots=True)
class ActionSchema:
    schema_id: int
    name: str
    parameters: tuple[ActionParameterSpec, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    required_permissions: tuple[str, ...] = ()
    expected_modalities: tuple[int, ...] = ()
    cost_unit: str = "normalized"
    reversible: bool = False
    timeout_seconds: float = 30.0
    safety_class: int = 0
    requires_provenance: bool = True
    information_gathering: bool = False

    def __post_init__(self) -> None:
        if self.schema_id < 0 or not self.name or not self.cost_unit:
            raise ValueError("action schema identity is invalid")
        if self.timeout_seconds <= 0 or self.safety_class < 0:
            raise ValueError("action schema timeout and safety class are invalid")
        names = tuple(parameter.name for parameter in self.parameters)
        if len(names) != len(set(names)):
            raise ValueError("action schema parameter names must be unique")
        if len(self.required_capabilities) != len(set(self.required_capabilities)):
            raise ValueError("action schema capabilities must be unique")
        if len(self.required_permissions) != len(set(self.required_permissions)):
            raise ValueError("action schema permissions must be unique")


@dataclass(frozen=True, slots=True)
class StructuredExternalAction:
    schema_id: int
    schema_name: str
    parameters: tuple[float, ...]
    supporter_provenance_ids: tuple[int, ...]
    expected_cost: float
    timeout_seconds: float
    reversible: bool
    information_gathering: bool
    batch_index: int

    def __post_init__(self) -> None:
        if self.schema_id < 0 or not self.schema_name or self.batch_index < 0:
            raise ValueError("structured action identity is invalid")
        if self.expected_cost < 0 or self.timeout_seconds <= 0:
            raise ValueError("structured action cost or timeout is invalid")
        if any(record_id < 0 for record_id in self.supporter_provenance_ids):
            raise ValueError("structured action supporters require provenance")


class ActionSchemaRegistry:
    """Application-owned action ontology, capabilities, and permissions."""

    def __init__(
        self, schemas: Iterable[ActionSchema] = (), *,
        capabilities: Iterable[str] = (), permissions: Iterable[str] = (),
    ) -> None:
        self._schemas: dict[int, ActionSchema] = {}
        self.capabilities = frozenset(capabilities)
        self.permissions = frozenset(permissions)
        for schema in schemas:
            self.register(schema)

    def register(self, schema: ActionSchema) -> None:
        if schema.schema_id in self._schemas:
            raise ValueError(f"duplicate action schema ID {schema.schema_id}")
        if any(existing.name == schema.name for existing in self._schemas.values()):
            raise ValueError(f"duplicate action schema name {schema.name!r}")
        self._schemas[schema.schema_id] = schema

    def get(self, schema_id: int) -> ActionSchema:
        try:
            return self._schemas[int(schema_id)]
        except KeyError as error:
            raise KeyError(f"unregistered action schema {schema_id}") from error

    @property
    def schema_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._schemas))

    def availability_mask(self, batch: int, action_count: int, *, device=None) -> Tensor:
        if min(batch, action_count) <= 0:
            raise ValueError("action registry mask dimensions must be positive")
        mask = torch.zeros(batch, action_count, dtype=torch.bool, device=device)
        for schema_id, schema in self._schemas.items():
            if schema_id >= action_count:
                continue
            allowed = (
                set(schema.required_capabilities) <= self.capabilities
                and set(schema.required_permissions) <= self.permissions
            )
            mask[:, schema_id] = allowed
        return mask

    def materialize(
        self, decision: "ExternalActionDecision", *, batch_index: int,
        parameters: Tensor | None = None, parameter_mask: Tensor | None = None,
    ) -> StructuredExternalAction:
        if not 0 <= batch_index < decision.logits.shape[0]:
            raise ValueError("structured action batch index is invalid")
        if not bool(decision.authorized[batch_index]):
            raise PermissionError("external action is not authority-approved")
        schema_id = int(decision.selected_action[batch_index])
        schema = self.get(schema_id)
        if not set(schema.required_capabilities) <= self.capabilities:
            raise PermissionError("required action capability is not registered")
        if not set(schema.required_permissions) <= self.permissions:
            raise PermissionError("required action permission is not granted")
        if parameters is None:
            parameters = decision.logits.new_zeros(len(schema.parameters))
        if parameter_mask is None:
            parameter_mask = torch.ones_like(parameters, dtype=torch.bool)
        if parameters.shape != (len(schema.parameters),) or parameter_mask.shape != parameters.shape:
            raise ValueError("structured action arguments do not match schema")
        values: list[float] = []
        for index, spec in enumerate(schema.parameters):
            if spec.required and not bool(parameter_mask[index]):
                raise ValueError(f"required action parameter {spec.name!r} is absent")
            value = float(parameters[index])
            if bool(parameter_mask[index]) and not spec.lower <= value <= spec.upper:
                raise ValueError(f"action parameter {spec.name!r} is out of bounds")
            values.append(value)
        supporters = decision.supporting_provenance_ids[batch_index][
            decision.supporting_mask[batch_index]
        ].tolist()
        if schema.requires_provenance and not supporters:
            raise PermissionError("action schema requires supporting provenance")
        return StructuredExternalAction(
            schema_id, schema.name, tuple(values), tuple(int(item) for item in supporters),
            float(decision.expected_cost[batch_index, schema_id].detach()), schema.timeout_seconds,
            schema.reversible, schema.information_gathering, batch_index,
        )


def _tensor_map(instance: T, method: str, *args, **kwargs) -> T:
    values = {}
    for field in fields(instance):
        value = getattr(instance, field.name)
        values[field.name] = (
            getattr(value, method)(*args, **kwargs)
            if isinstance(value, Tensor) else value
        )
    return type(instance)(**values)


@dataclass(frozen=True, slots=True)
class ExternalActionDecision:
    logits: Tensor
    probabilities: Tensor
    utility: Tensor
    expected_reward: Tensor
    expected_cost: Tensor
    constraint_probability: Tensor
    expected_success: Tensor
    available: Tensor
    selected_action: Tensor
    supporting_provenance_ids: Tensor
    supporting_mask: Tensor
    active: Tensor
    abstained: Tensor
    authorized: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.logits.ndim != 2:
            raise ValueError("external action logits must be (batch,actions)")
        shape = self.logits.shape
        for name in (
            "probabilities", "utility", "expected_reward", "expected_cost",
            "constraint_probability", "expected_success",
        ):
            if getattr(self, name).shape != shape:
                raise ValueError(f"external action {name} must match logits")
        if self.available.shape != shape or self.available.dtype != torch.bool:
            raise ValueError("external action availability must be boolean")
        batch = shape[0]
        if self.selected_action.shape != (batch,) or self.selected_action.dtype != torch.int64:
            raise ValueError("external selected action must be int64 per batch")
        if self.supporting_provenance_ids.ndim != 2 or self.supporting_provenance_ids.shape[0] != batch:
            raise ValueError("external action supporters must be (batch,supporters)")
        if self.supporting_mask.shape != self.supporting_provenance_ids.shape or self.supporting_mask.dtype != torch.bool:
            raise ValueError("external action supporting mask is invalid")
        for name in ("active", "abstained", "authorized"):
            value = getattr(self, name)
            if value.shape != (batch,) or value.dtype != torch.bool:
                raise ValueError(f"external action {name} must be boolean per batch")
        if bool((self.supporting_mask & (self.supporting_provenance_ids < 0)).any()):
            raise ValueError("external action supporters require provenance")
        if bool((self.active & ((self.selected_action < 0) | (self.selected_action >= shape[1]))).any()):
            raise ValueError("active external actions require an in-range selection")
        if bool((self.authorized & (~self.active | self.abstained)).any()):
            raise ValueError("only active non-abstained actions may be authorized")

    @classmethod
    def empty(
        cls, batch: int, actions: int, supporters: int, *, device=None, dtype=None,
    ) -> "ExternalActionDecision":
        if min(batch, actions, supporters) <= 0:
            raise ValueError("external action dimensions must be positive")
        matrix = torch.zeros(batch, actions, device=device, dtype=dtype)
        return cls(
            matrix, matrix.clone(), matrix.clone(), matrix.clone(), matrix.clone(),
            matrix.clone(), matrix.clone(),
            torch.zeros(batch, actions, dtype=torch.bool, device=device),
            torch.full((batch,), -1, dtype=torch.int64, device=device),
            torch.full((batch, supporters), -1, dtype=torch.int64, device=device),
            torch.zeros(batch, supporters, dtype=torch.bool, device=device),
            torch.zeros(batch, dtype=torch.bool, device=device),
            torch.ones(batch, dtype=torch.bool, device=device),
            torch.zeros(batch, dtype=torch.bool, device=device),
        )

    def detach(self) -> "ExternalActionDecision":
        return _tensor_map(self, "detach")

    def to(self, *args, **kwargs) -> "ExternalActionDecision":
        return _tensor_map(self, "to", *args, **kwargs)


@dataclass(frozen=True, slots=True)
class ExternalActionFeedback:
    selected_action: Tensor
    success: Tensor
    latency: Tensor
    reward: Tensor
    cost: Tensor
    constraint_violation: Tensor
    provenance_ids: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        batch = self.selected_action.shape[0]
        if self.selected_action.shape != (batch,) or self.selected_action.dtype != torch.int64:
            raise ValueError("external feedback action must be int64 per batch")
        for name in ("success", "latency", "reward", "cost", "constraint_violation"):
            value = getattr(self, name)
            if value.shape != (batch,) or not value.is_floating_point():
                raise ValueError(f"external feedback {name} must be floating point per batch")
        if self.provenance_ids.shape != (batch,) or self.provenance_ids.dtype != torch.int64:
            raise ValueError("external feedback provenance must be int64 per batch")
        if self.mask.shape != (batch,) or self.mask.dtype != torch.bool:
            raise ValueError("external feedback mask must be boolean per batch")
        if bool((self.mask & ((self.success < 0) | (self.success > 1))).any()):
            raise ValueError("external feedback success must lie in [0,1]")
        if bool((self.mask & ((self.latency < 0) | (self.cost < 0))).any()):
            raise ValueError("external feedback latency and cost cannot be negative")
        if bool((self.mask & (self.provenance_ids < 0)).any()):
            raise ValueError("active external feedback requires provenance")


class ExternalActionPolicy(nn.Module):
    """Factorized proposal and consequence heads under hard capability masks."""

    def __init__(
        self, width: int, goal_dim: int, uncertainty_channels: int,
        system_feature_dim: int, action_count: int,
    ) -> None:
        super().__init__()
        if min(width, goal_dim, uncertainty_channels, system_feature_dim, action_count) <= 0:
            raise ValueError("external action policy dimensions must be positive")
        self.action_count = action_count
        self.trunk = nn.Sequential(
            nn.Linear(width + goal_dim + uncertainty_channels + system_feature_dim, width),
            nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
        )
        self.logits = nn.Linear(width, action_count)
        self.consequences = nn.Linear(width, 4 * action_count)

    def forward(
        self, context: Tensor, goals: GoalState, uncertainty: Tensor,
        system: SystemModelState, *, supporting_provenance_ids: Tensor,
        supporting_mask: Tensor, active_mask: Tensor,
        controller_abstained: Tensor | None = None,
    ) -> ExternalActionDecision:
        batch = context.shape[0]
        if context.ndim != 2 or goals.desired_outcomes.shape[0] != batch:
            raise ValueError("external action context and goals have incompatible batches")
        if uncertainty.shape[0] != batch or system.action_availability.shape != (batch, self.action_count):
            raise ValueError("external action uncertainty or system action count is invalid")
        if supporting_provenance_ids.ndim != 2 or supporting_provenance_ids.shape[0] != batch:
            raise ValueError("external action supporters must be (batch,supporters)")
        if supporting_mask.shape != supporting_provenance_ids.shape or supporting_mask.dtype != torch.bool:
            raise ValueError("external action supporter mask is invalid")
        if active_mask.shape != (batch,) or active_mask.dtype != torch.bool:
            raise ValueError("external action active mask must be boolean per batch")
        controller_abstained = (
            torch.zeros(batch, dtype=torch.bool, device=context.device)
            if controller_abstained is None else controller_abstained
        )
        if controller_abstained.shape != (batch,) or controller_abstained.dtype != torch.bool:
            raise ValueError("controller abstention must be boolean per batch")
        features = torch.cat((
            context, goals.summary(), uncertainty.detach(), system.features(),
        ), -1)
        hidden = self.trunk(features)
        raw_logits = self.logits(hidden)
        consequence = self.consequences(hidden).reshape(batch, self.action_count, 4)
        reward = consequence[..., 0]
        cost = F.softplus(consequence[..., 1])
        constraint = torch.sigmoid(consequence[..., 2])
        success = torch.sigmoid(consequence[..., 3])
        # Empirical system estimates are authority-side priors.  They remain
        # detached so policy gradients cannot rewrite measured capabilities.
        system_success = system.action_success.detach().clamp(0, 1)
        system_latency = system.action_latency.detach().clamp_min(0)
        empirical_utility = (
            system.action_reward.detach()
            + system.action_reversibility.detach().clamp(0, 1)
            + system.executor_reliability.detach().clamp(0, 1)
            - system.action_cost.detach().clamp_min(0)
            - system.action_constraint_violation.detach().clamp(0, 1)
        )
        utility = (
            reward + success + system_success + empirical_utility
            - cost - constraint - system_latency
        )
        available = (
            (system.action_availability > 0) & system.permission_mask
            & active_mask[:, None]
        )
        scored = (raw_logits + utility).masked_fill(~available, -torch.inf)
        has_action = available.any(-1)
        probabilities = torch.zeros_like(scored)
        if bool(has_action.any()):
            probabilities[has_action] = torch.softmax(scored[has_action], -1)
        selected = scored.argmax(-1).masked_fill(~has_action, -1)
        active = active_mask & has_action & ~controller_abstained
        abstained = active_mask & (~has_action | controller_abstained)
        selected = selected.masked_fill(~active, -1)
        return ExternalActionDecision(
            raw_logits, probabilities, utility, reward, cost, constraint, success,
            available, selected, supporting_provenance_ids, supporting_mask,
            active, abstained, torch.zeros_like(active),
        )


def authorize_external_actions(
    decision: ExternalActionDecision, ledger: ProvenanceLedger,
) -> ExternalActionDecision:
    """Apply the immutable provenance gate to otherwise valid proposals."""

    authorized = torch.zeros_like(decision.active)
    for row in torch.nonzero(decision.active, as_tuple=False).flatten().tolist():
        supporters = decision.supporting_provenance_ids[row][decision.supporting_mask[row]]
        if supporters.numel() == 0:
            continue
        valid = True
        for record_id in supporters.tolist():
            try:
                valid = valid and ledger.can_justify_external_action(int(record_id))
            except KeyError:
                valid = False
        authorized[row] = valid
    values = {field.name: getattr(decision, field.name) for field in fields(decision)}
    values["authorized"] = authorized
    return ExternalActionDecision(**values)


def decision_from_candidates(
    candidates: ActionCandidateState, *, action_count: int,
    active_mask: Tensor,
) -> ExternalActionDecision:
    """Project one authority-gated candidate back into the executor contract."""

    if action_count <= 0 or active_mask.shape != (candidates.batch,) or active_mask.dtype != torch.bool:
        raise ValueError("candidate decision action count or active mask is invalid")
    batch = candidates.batch
    matrix = candidates.arguments.new_zeros(batch, action_count)
    logits = matrix.clone()
    utility = matrix.clone()
    reward = matrix.clone()
    cost = matrix.clone()
    constraint = matrix.clone()
    success = matrix.clone()
    available = torch.zeros(batch, action_count, dtype=torch.bool, device=matrix.device)
    for row, slot in torch.nonzero(candidates.active, as_tuple=False).tolist():
        schema = int(candidates.schema_ids[row, slot])
        if not 0 <= schema < action_count:
            continue
        logits[row, schema] = candidates.proposal_logits[row, slot]
        utility[row, schema] = candidates.normalized_utility[row, slot]
        reward[row, schema] = candidates.expected_reward[row, slot]
        cost[row, schema] = candidates.expected_cost[row, slot]
        constraint[row, schema] = candidates.constraint_probability[row, slot]
        success[row, schema] = candidates.expected_success[row, slot]
        available[row, schema] = (
            candidates.available[row, slot] & candidates.permitted[row, slot]
            & candidates.provenance_authorized[row, slot]
            & candidates.viability_authorized[row, slot]
        )
    selected_slot = candidates.selected.to(torch.int64).argmax(-1)
    has_selected = candidates.selected.any(-1) & active_mask
    rows = torch.arange(batch, device=matrix.device)
    selected = candidates.schema_ids[rows, selected_slot].masked_fill(~has_selected, -1)
    supporters = candidates.supporting_provenance_ids[rows, selected_slot]
    supporter_mask = candidates.supporting_mask[rows, selected_slot] & has_selected[:, None]
    probabilities = matrix.clone()
    for row in torch.nonzero(available.any(-1), as_tuple=False).flatten().tolist():
        probabilities[row] = torch.softmax(
            (logits[row] + utility[row]).masked_fill(~available[row], -torch.inf), -1
        )
    return ExternalActionDecision(
        logits, probabilities, utility, reward, cost, constraint, success,
        available, selected, supporters, supporter_mask,
        has_selected, active_mask & ~has_selected, has_selected,
    )


def update_system_model_from_feedback(
    state: SystemModelState, feedback: ExternalActionFeedback,
    ledger: ProvenanceLedger, *, momentum: float = 0.95,
) -> SystemModelState:
    """Update measured consequence statistics without changing host authority."""

    if not 0 <= momentum < 1:
        raise ValueError("system feedback momentum must lie in [0,1)")
    if feedback.selected_action.shape[0] != state.action_availability.shape[0]:
        raise ValueError("external feedback batch does not match system model")
    success = state.action_success.clone()
    latency = state.action_latency.clone()
    reward = state.action_reward.clone()
    cost = state.action_cost.clone()
    constraint = state.action_constraint_violation.clone()
    for row in torch.nonzero(feedback.mask, as_tuple=False).flatten().tolist():
        action = int(feedback.selected_action[row])
        if not 0 <= action < state.action_availability.shape[1]:
            raise ValueError("external feedback action lies outside the system model")
        record_id = int(feedback.provenance_ids[row])
        ledger.get(record_id)
        success[row, action] = (
            momentum * success[row, action]
            + (1 - momentum) * feedback.success[row]
        )
        latency[row, action] = (
            momentum * latency[row, action]
            + (1 - momentum) * feedback.latency[row]
        )
        reward[row, action] = momentum * reward[row, action] + (1 - momentum) * feedback.reward[row]
        cost[row, action] = momentum * cost[row, action] + (1 - momentum) * feedback.cost[row]
        constraint[row, action] = (
            momentum * constraint[row, action]
            + (1 - momentum) * feedback.constraint_violation[row]
        )
    return SystemModelState(
        state.modality_availability, state.action_availability, success, latency,
        reward, cost, constraint, state.action_reversibility,
        state.executor_reliability,
        state.memory_reliability, state.router_reliability,
        state.remaining_compute, state.remaining_memory,
        state.permission_mask, state.calibration_regime,
    )


class EnvironmentExecutor(Protocol):
    """Application-owned authority that can actually change an environment."""

    def execute(self, action_index: int, batch_index: int) -> object: ...


def execute_authorized_actions(
    decision: ExternalActionDecision, executor: EnvironmentExecutor,
) -> tuple[object | None, ...]:
    """Execute only authorized rows; return one result slot per batch item."""

    results: list[object | None] = [None] * decision.logits.shape[0]
    for row in torch.nonzero(decision.authorized, as_tuple=False).flatten().tolist():
        results[row] = executor.execute(int(decision.selected_action[row]), row)
    return tuple(results)
