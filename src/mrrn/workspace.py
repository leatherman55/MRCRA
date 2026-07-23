"""Bounded active graph, global competition, and low-gain carrier broadcast."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cognitive_types import NodeSlots, NodeType, RelationSlots
from .relational_router import (
    NodeCandidateBuilder, RelationalResonanceRouter, RelationalRouterOutput,
)
from .runtime_validation import runtime_validation_enabled


@dataclass(frozen=True, slots=True)
class RelationProposals:
    content: Tensor
    family_ids: Tensor
    participant_indices: Tensor
    participant_roles: Tensor
    participant_mask: Tensor
    support: Tensor
    confidence: Tensor
    parent_provenance_ids: Tensor
    provenance_ids: Tensor
    scenario_ids: Tensor
    active: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.content.ndim != 4:
            raise ValueError("relation proposals must be (batch,queries,degree,width)")
        base = self.content.shape[:3]
        for name in ("family_ids", "provenance_ids", "scenario_ids"):
            value = getattr(self, name)
            if value.shape != base or value.dtype != torch.int64:
                raise ValueError(f"{name} must be int64 with proposal shape")
        if self.participant_indices.ndim != 4 or self.participant_indices.shape[:3] != base:
            raise ValueError("relation participants must be (batch,queries,degree,arity)")
        if self.participant_indices.shape[-1] < 2:
            raise ValueError("relation proposals require room for at least two participants")
        if self.participant_roles.shape != self.participant_indices.shape:
            raise ValueError("participant roles must match participant pointers")
        if self.participant_mask.shape != self.participant_indices.shape or self.participant_mask.dtype != torch.bool:
            raise ValueError("participant mask must be boolean and match pointers")
        if self.support.shape != (*base, 3):
            raise ValueError("relation proposal support has invalid shape")
        if self.confidence.shape != base:
            raise ValueError("relation proposal confidence has invalid shape")
        if self.parent_provenance_ids.shape != self.participant_indices.shape:
            raise ValueError("parent provenance must match relation participants")
        if self.active.shape != base or self.active.dtype != torch.bool:
            raise ValueError("relation proposal active mask is invalid")
        if bool((self.active & (self.family_ids < 0)).any()):
            raise ValueError("active relation proposals require a family")
        active_participants = self.active.unsqueeze(-1) & self.participant_mask
        if bool((self.active & (self.participant_mask.sum(-1) < 2)).any()):
            raise ValueError("active relation proposals require at least two participants")
        if bool((active_participants & (self.participant_indices < 0)).any()):
            raise ValueError("active relation participant pointers cannot be negative")
        if bool((active_participants & (self.parent_provenance_ids < 0)).any()):
            raise ValueError("active relation proposals require parent provenance")
        if bool((
            self.active.unsqueeze(-1) & ~self.participant_mask
            & (self.participant_indices >= 0)
        ).any()):
            raise ValueError("masked relation participant slots cannot retain pointers")

    def with_provenance(self, provenance_ids: Tensor) -> "RelationProposals":
        if provenance_ids.shape != self.active.shape or provenance_ids.dtype != torch.int64:
            raise ValueError("relation provenance IDs must be int64 with proposal shape")
        if bool((self.active & (provenance_ids < 0)).any()):
            raise ValueError("active relation proposals require derived provenance")
        return RelationProposals(
            self.content, self.family_ids, self.participant_indices,
            self.participant_roles, self.participant_mask, self.support,
            self.confidence, self.parent_provenance_ids, provenance_ids,
            self.scenario_ids, self.active,
        )


def relation_proposals(nodes: NodeSlots, output: RelationalRouterOutput) -> RelationProposals:
    batch, queries, degree = output.selected_node_indices.shape
    if nodes.content.shape[:2] != (batch, queries):
        raise ValueError("router output does not address this node ring")
    relation_index = output.selected_relation_families.clamp_min(0)
    message_index = relation_index[..., None].expand(-1, -1, -1, nodes.content.shape[-1])
    messages = output.relation_messages[:, :, None].expand(-1, -1, degree, -1, -1)
    content = torch.gather(messages, 3, message_index.unsqueeze(3)).squeeze(3)
    query_indices = torch.arange(queries, device=nodes.content.device)[None, :, None].expand(batch, -1, degree)
    participants = torch.stack((query_indices, output.selected_node_indices), -1)
    roles = torch.tensor([0, 1], device=nodes.content.device, dtype=torch.int64).view(1, 1, 1, 2)
    roles = roles.expand_as(participants)
    participant_mask = output.selected_mask[..., None].expand_as(participants)
    safe_candidates = output.selected_node_indices.clamp_min(0)
    batch_indices = torch.arange(batch, device=nodes.content.device)[:, None, None]
    candidate_support = nodes.support[batch_indices, safe_candidates]
    query_support = nodes.support[:, :, None].expand(-1, -1, degree, -1)
    support = torch.stack((
        torch.minimum(query_support[..., 0], candidate_support[..., 0]),
        torch.maximum(query_support[..., 1], candidate_support[..., 1]),
        torch.maximum(query_support[..., 2], candidate_support[..., 2]),
    ), -1)
    candidate_provenance = nodes.provenance_ids[batch_indices, safe_candidates]
    parent_provenance = torch.stack((
        nodes.provenance_ids[:, :, None].expand(-1, -1, degree), candidate_provenance,
    ), -1)
    scenarios = nodes.scenario_ids[:, :, None].expand(-1, -1, degree)
    inactive = torch.full_like(output.selected_relation_families, -1)
    return RelationProposals(
        content * output.selected_mask.unsqueeze(-1),
        torch.where(output.selected_mask, output.selected_relation_families, inactive),
        participants.masked_fill(~participant_mask, -1),
        roles,
        participant_mask,
        support * output.selected_mask.unsqueeze(-1),
        output.selected_scores * output.selected_mask,
        parent_provenance.masked_fill(~participant_mask, -1),
        inactive,
        scenarios.masked_fill(~output.selected_mask, -1),
        output.selected_mask,
    )


class RelationSlotWriter(nn.Module):
    """Materialize typed pointers with independently bounded pair/hyperedge regions."""

    def __init__(
        self, relation_families: int, uncertainty_channels: int, *,
        pair_capacity: int | None = None, hyperedge_capacity: int = 0,
    ) -> None:
        super().__init__()
        if min(relation_families, uncertainty_channels) <= 0:
            raise ValueError("relation writer dimensions must be positive")
        if pair_capacity is not None and pair_capacity <= 0:
            raise ValueError("pair relation capacity must be positive")
        if hyperedge_capacity < 0:
            raise ValueError("hyperedge capacity cannot be negative")
        self.relation_families = relation_families
        self.uncertainty_channels = uncertainty_channels
        self.pair_capacity = pair_capacity
        self.hyperedge_capacity = hyperedge_capacity

    def forward(
        self, slots: RelationSlots, proposals: RelationProposals, nodes: NodeSlots,
    ) -> RelationSlots:
        if slots.batch != proposals.content.shape[0] or slots.content.shape[-1] != proposals.content.shape[-1]:
            raise ValueError("relation proposals and slots are incompatible")
        if nodes.batch != slots.batch or nodes.content.shape[-1] != slots.content.shape[-1]:
            raise ValueError("relation writer node ring is incompatible")
        if bool((proposals.active & (proposals.provenance_ids < 0)).any()):
            raise ValueError("relations require derived provenance before materialization")
        if proposals.participant_indices.shape[-1] > slots.participant_indices.shape[-1]:
            raise ValueError("relation proposal arity exceeds the slot contract")
        if self.pair_capacity is not None and slots.capacity != (
            self.pair_capacity + self.hyperedge_capacity
        ):
            raise ValueError("relation ring does not match its reserved pair/hyperedge capacities")
        values = {name: getattr(slots, name).clone() for name in slots.__dataclass_fields__}
        batch = slots.batch
        for batch_index in range(batch):
            active_flat = torch.nonzero(proposals.active[batch_index].flatten(), as_tuple=False).flatten()
            for flat_index in active_flat.tolist():
                degree = proposals.active.shape[-1]
                query_index, edge_index = divmod(flat_index, degree)
                participants = proposals.participant_indices[batch_index, query_index, edge_index]
                participant_mask = proposals.participant_mask[batch_index, query_index, edge_index]
                participant_count = int(participant_mask.sum().item())
                if self.pair_capacity is None:
                    region = torch.arange(slots.capacity, device=slots.content.device)
                elif participant_count == 2:
                    region = torch.arange(self.pair_capacity, device=slots.content.device)
                else:
                    if self.hyperedge_capacity == 0:
                        raise ValueError("higher-arity relation has no reserved hyperedge capacity")
                    region = torch.arange(
                        self.pair_capacity, self.pair_capacity + self.hyperedge_capacity,
                        device=slots.content.device,
                    )
                family = proposals.family_ids[batch_index, query_index, edge_index]
                padded_participants = torch.full(
                    (slots.participant_indices.shape[-1],), -1,
                    dtype=torch.int64, device=slots.content.device,
                )
                padded_mask = torch.zeros_like(padded_participants, dtype=torch.bool)
                proposal_arity = participants.shape[-1]
                padded_participants[:proposal_arity] = participants
                padded_mask[:proposal_arity] = participant_mask
                existing = (
                    values["active"][batch_index, region]
                    & (values["type_logits"][batch_index, region].argmax(-1) == family)
                    & (values["participant_mask"][batch_index, region] == padded_mask).all(-1)
                    & (
                        values["participant_indices"][batch_index, region]
                        == padded_participants
                    ).all(-1)
                )
                matching = torch.nonzero(existing, as_tuple=False).flatten()
                if matching.numel():
                    slot = int(region[matching[0]].item())
                else:
                    free = region[~values["active"][batch_index, region]]
                    if free.numel():
                        slot = int(free[0].item())
                    else:
                        local = values["confidence"][batch_index, region, 0].argmin()
                        slot = int(region[local].item())
                values["content"][batch_index, slot] = proposals.content[batch_index, query_index, edge_index]
                values["type_logits"][batch_index, slot].zero_()
                values["type_logits"][batch_index, slot, family] = 1
                values["participant_indices"][batch_index, slot].fill_(-1)
                values["participant_roles"][batch_index, slot].zero_()
                values["participant_versions"][batch_index, slot].fill_(-1)
                values["participant_weights"][batch_index, slot].zero_()
                values["participant_mask"][batch_index, slot].zero_()
                values["participant_indices"][batch_index, slot, :proposal_arity] = participants
                values["participant_roles"][batch_index, slot, :proposal_arity] = proposals.participant_roles[
                    batch_index, query_index, edge_index
                ]
                safe_participants = participants.clamp_min(0)
                versions = nodes.versions[batch_index, safe_participants]
                values["participant_versions"][batch_index, slot, :proposal_arity] = torch.where(
                    participant_mask, versions, torch.full_like(versions, -1)
                )
                values["participant_weights"][batch_index, slot, :proposal_arity] = participant_mask
                values["participant_mask"][batch_index, slot, :proposal_arity] = participant_mask
                values["support"][batch_index, slot] = proposals.support[batch_index, query_index, edge_index]
                values["confidence"][batch_index, slot].zero_()
                values["confidence"][batch_index, slot, 0] = proposals.confidence[
                    batch_index, query_index, edge_index
                ]
                values["provenance_ids"][batch_index, slot] = proposals.provenance_ids[
                    batch_index, query_index, edge_index
                ]
                values["scenario_ids"][batch_index, slot] = proposals.scenario_ids[
                    batch_index, query_index, edge_index
                ]
                values["versions"][batch_index, slot] += 1
                values["active"][batch_index, slot] = True
        return RelationSlots(**values)


def invalidate_stale_relations(slots: RelationSlots, nodes: NodeSlots) -> RelationSlots:
    """Remove edges whose slot/version pointer no longer names the same node."""

    if slots.batch != nodes.batch:
        raise ValueError("relation and node batches must match")
    safe = slots.participant_indices.clamp_min(0)
    batch = torch.arange(slots.batch, device=nodes.content.device)[:, None, None]
    current_versions = nodes.versions[batch, safe]
    current_active = nodes.active[batch, safe]
    stale_participant = slots.participant_mask & (
        ~current_active | (current_versions != slots.participant_versions)
    )
    stale = slots.active & stale_participant.any(-1)
    if not bool(stale.any()):
        return slots
    values = {name: getattr(slots, name).clone() for name in slots.__dataclass_fields__}
    values["active"][stale] = False
    values["provenance_ids"][stale] = -1
    values["scenario_ids"][stale] = -1
    values["participant_indices"][stale] = -1
    values["participant_roles"][stale] = 0
    values["participant_versions"][stale] = -1
    values["participant_weights"][stale] = 0
    values["participant_mask"][stale] = False
    values["content"][stale] = 0
    values["type_logits"][stale] = 0
    values["support"][stale] = 0
    values["confidence"][stale] = 0
    values["hypothesis_membership"][stale] = 0
    return RelationSlots(**values)


@dataclass(frozen=True, slots=True)
class GlobalWorkspaceState:
    slots: Tensor
    node_pointers: Tensor
    pointer_scores: Tensor
    active: Tensor
    ages: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.slots.ndim != 3:
            raise ValueError("global workspace slots must be (batch,slots,width)")
        base = self.slots.shape[:2]
        if self.node_pointers.shape != base or self.node_pointers.dtype != torch.int64:
            raise ValueError("workspace node pointers must be int64 with slot shape")
        if self.pointer_scores.shape != base:
            raise ValueError("workspace pointer scores have invalid shape")
        if self.active.shape != base or self.active.dtype != torch.bool:
            raise ValueError("workspace active mask is invalid")
        if self.ages.shape != base or self.ages.dtype != torch.int64:
            raise ValueError("workspace ages must be int64 with slot shape")
        if bool((self.active & (self.node_pointers < 0)).any()):
            raise ValueError("active workspace slots require a node pointer")

    def detach(self) -> "GlobalWorkspaceState":
        return GlobalWorkspaceState(
            self.slots.detach(), self.node_pointers.detach(), self.pointer_scores.detach(),
            self.active.detach(), self.ages.detach(),
        )


@dataclass(frozen=True, slots=True)
class WorkspaceCompetitionOutput:
    state: GlobalWorkspaceState
    assignment: Tensor
    update_gates: Tensor


class GlobalWorkspace(nn.Module):
    """Competitive fixed-capacity broadcast workspace over the active graph."""

    def __init__(self, width: int, slots: int, *, update_maximum: float = 0.5) -> None:
        super().__init__()
        if min(width, slots) <= 0 or not 0 < update_maximum < 1:
            raise ValueError("global workspace controls are invalid")
        self.width = width
        self.slot_count = slots
        self.update_maximum = update_maximum
        self.initial_slots = nn.Parameter(torch.empty(slots, width))
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.update_gate = nn.Linear(2 * width, 1)
        self.priority = nn.Linear(width + 2, 1)
        nn.init.normal_(self.initial_slots, std=width ** -0.5)

    def initial_state(self, batch: int, *, device=None, dtype=None) -> GlobalWorkspaceState:
        if batch <= 0:
            raise ValueError("workspace batch must be positive")
        slots = self.initial_slots.to(device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1).clone()
        shape = (batch, self.slot_count)
        return GlobalWorkspaceState(
            slots,
            torch.full(shape, -1, dtype=torch.int64, device=device),
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, dtype=torch.bool, device=device),
            torch.zeros(shape, dtype=torch.int64, device=device),
        )

    def forward(self, nodes: NodeSlots, state: GlobalWorkspaceState | None = None) -> WorkspaceCompetitionOutput:
        batch, node_count, width = nodes.content.shape
        if width != self.width:
            raise ValueError("node width does not match global workspace")
        if state is None:
            state = self.initial_state(batch, device=nodes.content.device, dtype=nodes.content.dtype)
        if state.slots.shape != (batch, self.slot_count, width):
            raise ValueError("global workspace state shape mismatch")
        query = F.normalize(self.query(state.slots), dim=-1)
        key = F.normalize(self.key(nodes.content), dim=-1)
        logits = torch.einsum("bsd,bnd->bsn", query, key) / width ** 0.5
        priority_input = torch.cat((
            nodes.content, nodes.importance.unsqueeze(-1), nodes.activity.unsqueeze(-1),
        ), -1)
        logits = logits + self.priority(priority_input).transpose(1, 2)
        valid = nodes.active[:, None].expand(-1, self.slot_count, -1)
        logits = logits.masked_fill(~valid, -torch.inf)
        # Alternating normalization approximates a bounded rectangular transport:
        # nodes compete across slots, then each slot receives a normalized read.
        assignment = torch.exp(logits - torch.where(
            torch.isfinite(logits.amax((-2, -1), keepdim=True)),
            logits.amax((-2, -1), keepdim=True), torch.zeros_like(logits[:, :1, :1]),
        )) * valid
        for _ in range(3):
            assignment = assignment / assignment.sum(1, keepdim=True).clamp_min(1e-8)
            assignment = assignment / assignment.sum(2, keepdim=True).clamp_min(1e-8)
            assignment = assignment * valid
        values = self.value(nodes.content)
        proposed = torch.einsum("bsn,bnd->bsd", assignment, values)
        gate = torch.sigmoid(self.update_gate(torch.cat((state.slots, proposed), -1)))
        gate = gate * self.update_maximum
        has_candidate = valid.any(-1)
        gate = gate * has_candidate.unsqueeze(-1)
        slots = (1 - gate) * state.slots + gate * proposed
        scores, pointers = assignment.max(-1)
        active = has_candidate & (scores > 0)
        pointers = pointers.masked_fill(~active, -1)
        ages = torch.where(active, torch.zeros_like(state.ages), state.ages + 1)
        return WorkspaceCompetitionOutput(
            GlobalWorkspaceState(slots, pointers, scores, active, ages), assignment, gate,
        )


@dataclass(frozen=True, slots=True)
class WorkspaceBroadcastOutput:
    scale_and_bias: tuple[tuple[Tensor, Tensor], ...]
    attention_query: Tensor
    memory_query: Tensor
    event_bias: Tensor
    output_context: Tensor
    controller_context: Tensor
    residual_gain: Tensor


class WorkspaceBroadcast(nn.Module):
    """Low-rank, norm-budgeted conditioning of carrier and cognitive heads."""

    def __init__(
        self, width: int, scale_widths: tuple[int, ...], *, rank: int = 32,
        maximum_gain: float = 0.1,
    ) -> None:
        super().__init__()
        if min(width, rank, *scale_widths) <= 0 or not 0 < maximum_gain <= 1:
            raise ValueError("workspace broadcast dimensions are invalid")
        self.maximum_gain = maximum_gain
        self.summary = nn.Linear(width, rank)
        self.scale_heads = nn.ModuleList(nn.Linear(rank, 2 * value) for value in scale_widths)
        self.attention = nn.Linear(rank, width)
        self.memory = nn.Linear(rank, width)
        self.event = nn.Linear(rank, 1)
        self.output = nn.Linear(rank, width)
        self.controller = nn.Linear(rank, width)
        self.gain_raw = nn.Parameter(torch.tensor(-6.0))
        for module in (*self.scale_heads, self.attention, self.memory, self.event, self.output, self.controller):
            nn.init.normal_(module.weight, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state: GlobalWorkspaceState) -> WorkspaceBroadcastOutput:
        weights = state.pointer_scores * state.active.to(state.slots.dtype)
        pooled = (state.slots * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)
        hidden = F.silu(self.summary(pooled))
        gain = self.maximum_gain * torch.sigmoid(self.gain_raw)
        scale_and_bias = []
        for head in self.scale_heads:
            scale, bias = head(hidden).chunk(2, -1)
            scale_and_bias.append((gain * torch.tanh(scale), gain * bias))
        return WorkspaceBroadcastOutput(
            tuple(scale_and_bias), gain * self.attention(hidden), gain * self.memory(hidden),
            gain * self.event(hidden), gain * self.output(hidden), gain * self.controller(hidden), gain,
        )


@dataclass(frozen=True, slots=True)
class WorkspaceGraphOutput:
    nodes: NodeSlots
    router: RelationalRouterOutput
    relation_proposals: RelationProposals
    workspace: WorkspaceCompetitionOutput
    broadcast: WorkspaceBroadcastOutput


class WorkspaceGraph(nn.Module):
    """One bounded relation-construction and global-access microstep."""

    def __init__(
        self, candidate_builder: NodeCandidateBuilder,
        router: RelationalResonanceRouter, workspace: GlobalWorkspace,
        broadcast: WorkspaceBroadcast, *, compact_active: bool = True,
    ) -> None:
        super().__init__()
        self.candidate_builder = candidate_builder
        self.router = router
        self.workspace = workspace
        self.broadcast = broadcast
        self.compact_active = compact_active

    @staticmethod
    def _gather_nodes(nodes: NodeSlots, order: Tensor) -> NodeSlots:
        batch, count = order.shape
        values = {}
        for name in nodes.__dataclass_fields__:
            value = getattr(nodes, name)
            tail = value.shape[2:]
            index = order.reshape(batch, count, *([1] * len(tail))).expand(
                batch, count, *tail
            )
            values[name] = torch.gather(value, 1, index)
        return NodeSlots(**values)

    @staticmethod
    def _scatter_queries(value: Tensor, order: Tensor, size: int, *, fill=0) -> Tensor:
        batch, count = order.shape
        result = value.new_full((batch, size, *value.shape[2:]), fill)
        index = order.reshape(batch, count, *([1] * (value.ndim - 2))).expand_as(value)
        return result.scatter(1, index, value)

    @staticmethod
    def _map_compact_indices(indices: Tensor, order: Tensor, mask: Tensor) -> Tensor:
        batch = indices.shape[0]
        safe = indices.clamp_min(0)
        batch_index = torch.arange(
            batch, device=indices.device
        ).reshape(batch, *([1] * (indices.ndim - 1)))
        mapped = order[batch_index, safe]
        return mapped.masked_fill(~mask, -1)

    def _dense_forward(
        self, nodes: NodeSlots, workspace_state: GlobalWorkspaceState | None,
        goal_context: Tensor | None,
    ) -> WorkspaceGraphOutput:
        candidates = self.candidate_builder(nodes)
        routed = self.router(nodes, candidates, goal_context=goal_context)
        values = {name: getattr(nodes, name) for name in nodes.__dataclass_fields__}
        values["content"] = nodes.content + routed.update
        updated_nodes = NodeSlots(**values)
        proposals = relation_proposals(updated_nodes, routed)
        competition = self.workspace(updated_nodes, workspace_state)
        return WorkspaceGraphOutput(
            updated_nodes, routed, proposals, competition,
            self.broadcast(competition.state),
        )

    def _empty_forward(
        self, nodes: NodeSlots, workspace_state: GlobalWorkspaceState | None,
    ) -> WorkspaceGraphOutput:
        """Advance empty authority state without executing impossible relations."""

        batch, capacity, width = nodes.content.shape
        if workspace_state is None:
            workspace_state = self.workspace.initial_state(
                batch, device=nodes.content.device, dtype=nodes.content.dtype,
            )
        slot_shape = workspace_state.active.shape
        empty_workspace = GlobalWorkspaceState(
            workspace_state.slots,
            torch.full(
                slot_shape, -1, dtype=torch.int64, device=nodes.content.device,
            ),
            torch.zeros(
                slot_shape, dtype=nodes.content.dtype, device=nodes.content.device,
            ),
            torch.zeros(
                slot_shape, dtype=torch.bool, device=nodes.content.device,
            ),
            workspace_state.ages + 1,
        )
        assignment = nodes.content.new_zeros(
            batch, self.workspace.slot_count, capacity,
        )
        competition = WorkspaceCompetitionOutput(
            empty_workspace, assignment,
            nodes.content.new_zeros(batch, self.workspace.slot_count, 1),
        )

        # Preserve the public bounded tensor contract.  One masked candidate is
        # sufficient because an empty ring has no semantic candidate dimension.
        relation_count = self.router.relation_count
        retained = min(self.router.retained_edges, relation_count)
        routed = RelationalRouterOutput(
            nodes.content.new_zeros(batch, capacity, width),
            nodes.content.new_zeros(batch, capacity, relation_count, width),
            nodes.content.new_zeros(batch, capacity, 1, relation_count),
            nodes.content.new_zeros(batch, capacity, 1, relation_count),
            nodes.content.new_zeros(batch, capacity, 1, relation_count),
            torch.full(
                (batch, capacity, retained), -1, dtype=torch.int64,
                device=nodes.content.device,
            ),
            torch.full(
                (batch, capacity, retained), -1, dtype=torch.int64,
                device=nodes.content.device,
            ),
            nodes.content.new_zeros(batch, capacity, retained),
            torch.zeros(
                batch, capacity, retained, dtype=torch.bool,
                device=nodes.content.device,
            ),
            nodes.content.new_zeros(batch, capacity, 1, relation_count),
        )
        # A single compact query keeps proposal storage proportional to live
        # cognition while all authority masks remain false.
        order = torch.zeros(batch, 1, dtype=torch.int64, device=nodes.content.device)
        compact = self._gather_nodes(nodes, order)
        compact_router = RelationalRouterOutput(
            routed.update[:, :1],
            routed.relation_messages[:, :1],
            routed.relation_posterior[:, :1],
            routed.candidate_posterior[:, :1],
            routed.joint_posterior[:, :1],
            routed.selected_node_indices[:, :1],
            routed.selected_relation_families[:, :1],
            routed.selected_scores[:, :1],
            routed.selected_mask[:, :1],
            routed.phase_coherence[:, :1],
        )
        proposals = relation_proposals(compact, compact_router)
        return WorkspaceGraphOutput(
            nodes, routed, proposals, competition,
            self.broadcast(empty_workspace),
        )

    def forward(
        self, nodes: NodeSlots, workspace_state: GlobalWorkspaceState | None = None,
        *, goal_context: Tensor | None = None,
    ) -> WorkspaceGraphOutput:
        if not self.compact_active:
            return self._dense_forward(nodes, workspace_state, goal_context)

        # Learned graph work should scale with live cognition rather than the
        # maximum authority-ring capacity.  Stable active-first compaction keeps
        # the original slot order, while at least one masked row preserves the
        # well-defined empty-workspace behavior.
        live_count = int(nodes.active.sum(-1).max().item())
        if live_count == 0:
            return self._empty_forward(nodes, workspace_state)
        if live_count == nodes.capacity:
            return self._dense_forward(nodes, workspace_state, goal_context)
        live_count = max(1, live_count)
        order = torch.argsort(
            (~nodes.active).to(torch.int8), dim=1, stable=True,
        )[:, :live_count]
        compact = self._gather_nodes(nodes, order)
        compact_output = self._dense_forward(
            compact, workspace_state, goal_context,
        )

        # Relation proposals are authority-bearing: translate both participants
        # from compact execution coordinates back to persistent ring slots.
        proposals = compact_output.relation_proposals
        participant_indices = self._map_compact_indices(
            proposals.participant_indices, order, proposals.participant_mask,
        )
        proposals = RelationProposals(
            proposals.content, proposals.family_ids, participant_indices,
            proposals.participant_roles, proposals.participant_mask,
            proposals.support, proposals.confidence,
            proposals.parent_provenance_ids, proposals.provenance_ids,
            proposals.scenario_ids, proposals.active,
        )

        full_content = nodes.content.scatter(
            1,
            order[:, :, None].expand(-1, -1, nodes.content.shape[-1]),
            compact_output.nodes.content,
        )
        node_values = {
            name: getattr(nodes, name) for name in nodes.__dataclass_fields__
        }
        node_values["content"] = full_content
        updated_nodes = NodeSlots(**node_values)

        routed = compact_output.router
        selected_nodes = self._map_compact_indices(
            routed.selected_node_indices, order, routed.selected_mask,
        )
        public_router = RelationalRouterOutput(
            self._scatter_queries(routed.update, order, nodes.capacity),
            self._scatter_queries(routed.relation_messages, order, nodes.capacity),
            self._scatter_queries(routed.relation_posterior, order, nodes.capacity),
            self._scatter_queries(routed.candidate_posterior, order, nodes.capacity),
            self._scatter_queries(routed.joint_posterior, order, nodes.capacity),
            self._scatter_queries(selected_nodes, order, nodes.capacity, fill=-1),
            self._scatter_queries(
                routed.selected_relation_families, order, nodes.capacity, fill=-1,
            ),
            self._scatter_queries(routed.selected_scores, order, nodes.capacity),
            self._scatter_queries(routed.selected_mask, order, nodes.capacity),
            self._scatter_queries(routed.phase_coherence, order, nodes.capacity),
        )
        compact_competition = compact_output.workspace
        assignment = nodes.content.new_zeros(
            nodes.batch, compact_competition.assignment.shape[1], nodes.capacity
        )
        assignment.scatter_(
            2,
            order[:, None].expand(
                -1, compact_competition.assignment.shape[1], -1
            ),
            compact_competition.assignment,
        )
        competition = WorkspaceCompetitionOutput(
            compact_competition.state, assignment,
            compact_competition.update_gates,
        )
        return WorkspaceGraphOutput(
            updated_nodes, public_router, proposals, competition,
            compact_output.broadcast,
        )
