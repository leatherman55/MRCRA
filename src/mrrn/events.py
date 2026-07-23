"""Causal multiscale event proposal, span finalization, and bounded allocation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cognitive_types import BoundaryClass, NodeSlots, NodeType, SourceClass
from .runtime_validation import runtime_validation_enabled


@dataclass(frozen=True, slots=True)
class EventEvidence:
    features: Tensor
    detail_energy: Tensor
    prediction_error: Tensor
    resonator_change: Tensor
    uncertainty: Tensor
    goal_context: Tensor
    boundary_classes: Tensor
    timestamps: Tensor
    modality_ids: Tensor
    parent_provenance_ids: Tensor
    scenario_ids: Tensor
    spectral: Tensor
    valid_mask: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.features.ndim != 3:
            raise ValueError("event features must have shape (batch,time,width)")
        batch, length, _ = self.features.shape
        base = (batch, length)
        for name in ("detail_energy", "prediction_error", "resonator_change"):
            value = getattr(self, name)
            if value.shape != (*base, 1):
                raise ValueError(f"{name} must have shape (batch,time,1)")
        if self.uncertainty.ndim != 3 or self.uncertainty.shape[:2] != base:
            raise ValueError("event uncertainty must have shape (batch,time,channels)")
        if self.goal_context.ndim != 2 or self.goal_context.shape[0] != batch:
            raise ValueError("goal_context must have shape (batch,features)")
        for name in ("boundary_classes", "modality_ids", "parent_provenance_ids", "scenario_ids"):
            value = getattr(self, name)
            if value.shape != base or value.dtype != torch.int64:
                raise ValueError(f"{name} must be int64 with shape (batch,time)")
        if self.timestamps.shape != base:
            raise ValueError("timestamps must have shape (batch,time)")
        if self.spectral.ndim != 5 or self.spectral.shape[:2] != base or self.spectral.shape[-1] != 2:
            raise ValueError("spectral evidence must be (batch,time,heads,modes,2)")
        if self.valid_mask.shape != base or self.valid_mask.dtype != torch.bool:
            raise ValueError("event valid_mask must be boolean with shape (batch,time)")
        if bool((self.valid_mask & (self.parent_provenance_ids < 0)).any()):
            raise ValueError("valid event evidence requires parent provenance")


@dataclass(frozen=True, slots=True)
class EventProposalOutput:
    proposal_logits: Tensor
    end_logits: Tensor
    identity_keys: Tensor
    content: Tensor
    node_type_logits: Tensor


@dataclass(frozen=True, slots=True)
class EventTransitionReceipts:
    """Exact hard-threshold decisions made by the causal event extractor.

    These receipts are observational: they expose every threshold crossing and
    quota decision without changing allocation authority.  Temporal masks have
    shape ``(batch,time)`` and therefore remain unambiguous when the extractor
    is called on more than one evidence position.
    """

    opened: Tensor
    finalized: Tensor
    emitted: Tensor
    quota_rejected: Tensor
    open_after: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.opened.ndim != 2:
            raise ValueError("event transition receipts must have shape (batch,time)")
        for name in ("finalized", "emitted", "quota_rejected", "open_after"):
            value = getattr(self, name)
            if value.shape != self.opened.shape or value.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean and match opened")
        if self.opened.dtype != torch.bool:
            raise ValueError("opened must be boolean")
        if bool((self.emitted & ~self.finalized).any()):
            raise ValueError("only finalized events may be emitted")
        if bool((self.quota_rejected & ~self.finalized).any()):
            raise ValueError("only finalized events may be rejected by quota")
        if bool((self.emitted & self.quota_rejected).any()):
            raise ValueError("an event cannot be emitted and quota-rejected")


class EventProposalNetwork(nn.Module):
    """Pointwise proposal head over causal carrier evidence.

    The network never convolves across time.  Temporal context has already been
    summarized causally by the carrier, so a proposal at ``t`` cannot observe a
    later coefficient even in batched execution.
    """

    def __init__(
        self, width: int, uncertainty_channels: int, goal_dim: int,
        node_type_count: int, *, hidden: int | None = None,
    ) -> None:
        super().__init__()
        if min(width, uncertainty_channels, goal_dim, node_type_count) <= 0:
            raise ValueError("event proposal dimensions must be positive")
        hidden = width if hidden is None else hidden
        if hidden <= 0:
            raise ValueError("event proposal hidden width must be positive")
        evidence_dim = width + 3 + uncertainty_channels + goal_dim + 4
        self.normalization = nn.LayerNorm(evidence_dim)
        self.input = nn.Linear(evidence_dim, hidden)
        self.gate = nn.Linear(evidence_dim, hidden)
        self.proposal = nn.Linear(hidden, 1)
        self.end = nn.Linear(hidden, 1)
        self.identity = nn.Linear(hidden, width)
        self.content = nn.Linear(hidden, width)
        self.node_type = nn.Linear(hidden, node_type_count)
        nn.init.constant_(self.proposal.bias, -2.0)
        nn.init.constant_(self.end.bias, -2.0)

    def forward(self, evidence: EventEvidence) -> EventProposalOutput:
        batch, length = evidence.features.shape[:2]
        goal = evidence.goal_context[:, None, :].expand(batch, length, -1)
        boundary = F.one_hot(
            evidence.boundary_classes.clamp(0, len(BoundaryClass) - 1), len(BoundaryClass)
        ).to(evidence.features.dtype)
        inputs = torch.cat((
            evidence.features, evidence.detail_energy, evidence.prediction_error,
            evidence.resonator_change, evidence.uncertainty, goal, boundary,
        ), -1)
        normalized = self.normalization(inputs)
        hidden = F.silu(self.input(normalized)) * torch.sigmoid(self.gate(normalized))
        mask = evidence.valid_mask.unsqueeze(-1)
        type_logits = self.node_type(hidden)
        # Event is the default proposal type, but learning may refine it.
        type_bias = torch.zeros_like(type_logits)
        type_bias[..., int(NodeType.EVENT)] = 1.0
        return EventProposalOutput(
            self.proposal(hidden).squeeze(-1).masked_fill(~evidence.valid_mask, -torch.inf),
            self.end(hidden).squeeze(-1).masked_fill(~evidence.valid_mask, -torch.inf),
            F.normalize(self.identity(hidden), dim=-1) * mask,
            self.content(hidden) * mask,
            (type_logits + type_bias) * mask,
        )


@dataclass(frozen=True, slots=True)
class EventExtractorState:
    open_start_times: Tensor
    open: Tensor
    emitted_in_chunk: Tensor
    position: int

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.open_start_times.ndim != 1 or self.open.shape != self.open_start_times.shape:
            raise ValueError("event extractor state must be one-dimensional per batch")
        if self.open.dtype != torch.bool:
            raise ValueError("event open mask must be boolean")
        if self.emitted_in_chunk.shape != self.open.shape or self.emitted_in_chunk.dtype != torch.int64:
            raise ValueError("event chunk counters must be int64 per batch")
        if self.position < 0 or bool((self.emitted_in_chunk < 0).any()):
            raise ValueError("event extractor state cannot be negative")

    def detach(self) -> "EventExtractorState":
        return EventExtractorState(
            self.open_start_times.detach(), self.open.detach(),
            self.emitted_in_chunk.detach(), self.position,
        )


@dataclass(frozen=True, slots=True)
class EventCandidates:
    content: Tensor
    identity_keys: Tensor
    spectral: Tensor
    type_logits: Tensor
    support: Tensor
    modality_ids: Tensor
    parent_provenance_ids: Tensor
    provenance_ids: Tensor
    source_classes: Tensor
    scenario_ids: Tensor
    uncertainty: Tensor
    score: Tensor
    active: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.content.ndim != 3:
            raise ValueError("event candidates must have shape (batch,count,width)")
        base = self.content.shape[:2]
        if self.identity_keys.shape != self.content.shape:
            raise ValueError("event identity keys must match event content")
        if self.spectral.ndim != 5 or self.spectral.shape[:2] != base or self.spectral.shape[-1] != 2:
            raise ValueError("event spectral signatures have invalid shape")
        if self.type_logits.ndim != 3 or self.type_logits.shape[:2] != base:
            raise ValueError("event type logits have invalid shape")
        if self.support.shape != (*base, 3):
            raise ValueError("event support must contain start, end, and completion")
        for name in (
            "modality_ids", "parent_provenance_ids", "provenance_ids",
            "source_classes", "scenario_ids",
        ):
            value = getattr(self, name)
            if value.shape != base or value.dtype != torch.int64:
                raise ValueError(f"{name} must be int64 with candidate shape")
        if self.uncertainty.ndim != 3 or self.uncertainty.shape[:2] != base:
            raise ValueError("event uncertainty has invalid shape")
        if self.score.shape != base:
            raise ValueError("event score has invalid shape")
        if self.active.shape != base or self.active.dtype != torch.bool:
            raise ValueError("event active mask has invalid shape")
        if bool((self.active & (self.parent_provenance_ids < 0)).any()):
            raise ValueError("active events require parent provenance")

    def with_provenance(self, provenance_ids: Tensor) -> "EventCandidates":
        if provenance_ids.shape != self.active.shape or provenance_ids.dtype != torch.int64:
            raise ValueError("derived provenance IDs must be int64 with candidate shape")
        if bool((self.active & (provenance_ids < 0)).any()):
            raise ValueError("active events require a derived provenance record")
        return EventCandidates(
            self.content, self.identity_keys, self.spectral, self.type_logits, self.support,
            self.modality_ids, self.parent_provenance_ids, provenance_ids,
            self.source_classes, self.scenario_ids, self.uncertainty, self.score, self.active,
        )


class CausalEventExtractor(nn.Module):
    """Finalize event spans online with a hard per-chunk write quota."""

    def __init__(
        self, proposal: EventProposalNetwork, *, chunk_size: int = 256,
        proposals_per_chunk: int = 8, proposal_threshold: float = 0.5,
        end_threshold: float = 0.5, maximum_span: int = 256,
    ) -> None:
        super().__init__()
        if min(chunk_size, proposals_per_chunk, maximum_span) <= 0:
            raise ValueError("event extractor capacities must be positive")
        if proposals_per_chunk > chunk_size:
            raise ValueError("event proposal quota cannot exceed chunk size")
        if not 0 < proposal_threshold < 1 or not 0 < end_threshold < 1:
            raise ValueError("event thresholds must lie in (0,1)")
        self.proposal_network = proposal
        self.chunk_size = chunk_size
        self.proposals_per_chunk = proposals_per_chunk
        self.proposal_logit = torch.logit(torch.tensor(proposal_threshold)).item()
        self.end_logit = torch.logit(torch.tensor(end_threshold)).item()
        self.maximum_span = maximum_span

    def initial_state(self, batch: int, *, device=None, dtype=None) -> EventExtractorState:
        if batch <= 0:
            raise ValueError("event extractor batch must be positive")
        return EventExtractorState(
            torch.zeros(batch, device=device, dtype=dtype),
            torch.zeros(batch, device=device, dtype=torch.bool),
            torch.zeros(batch, device=device, dtype=torch.int64),
            0,
        )

    def forward(
        self, evidence: EventEvidence, state: EventExtractorState | None = None,
    ) -> tuple[EventCandidates, EventExtractorState]:
        events, next_state, _, _ = self.extract_with_proposals(evidence, state)
        return events, next_state

    def extract_with_proposals(
        self, evidence: EventEvidence, state: EventExtractorState | None = None,
    ) -> tuple[
        EventCandidates, EventExtractorState, EventProposalOutput,
        EventTransitionReceipts,
    ]:
        """Extract events and retain differentiable heads plus exact decisions."""

        proposal = self.proposal_network(evidence)
        batch, length, width = proposal.content.shape
        if state is None:
            state = self.initial_state(batch, device=evidence.features.device, dtype=evidence.timestamps.dtype)
        if state.open.shape[0] != batch:
            raise ValueError("event extractor state batch does not match evidence")
        starts = state.open_start_times.clone()
        opened = state.open.clone()
        counters = state.emitted_in_chunk.clone()
        rows: list[list[tuple[int, float]]] = [[] for _ in range(batch)]
        opened_rows = torch.zeros(
            batch, length, dtype=torch.bool, device=evidence.features.device
        )
        finalized_rows = torch.zeros_like(opened_rows)
        emitted_rows = torch.zeros_like(opened_rows)
        rejected_rows = torch.zeros_like(opened_rows)
        open_after_rows = torch.zeros_like(opened_rows)
        for time_index in range(length):
            absolute_position = state.position + time_index
            if absolute_position and absolute_position % self.chunk_size == 0:
                counters.zero_()
            valid = evidence.valid_mask[:, time_index]
            proposal_now = proposal.proposal_logits[:, time_index] >= self.proposal_logit
            start_now = valid & ~opened & proposal_now
            opened_rows[:, time_index] = start_now
            starts = torch.where(start_now, evidence.timestamps[:, time_index], starts)
            opened = opened | start_now
            span_steps = evidence.timestamps[:, time_index] - starts
            boundary = evidence.boundary_classes[:, time_index] != int(BoundaryClass.NONE)
            end_now = opened & valid & (
                (proposal.end_logits[:, time_index] >= self.end_logit)
                | boundary | (span_steps >= self.maximum_span)
            )
            allowed = end_now & (counters < self.proposals_per_chunk)
            finalized_rows[:, time_index] = end_now
            emitted_rows[:, time_index] = allowed
            rejected_rows[:, time_index] = end_now & ~allowed
            for batch_index in torch.nonzero(allowed, as_tuple=False).flatten().tolist():
                rows[batch_index].append((time_index, float(starts[batch_index].item())))
            counters = counters + allowed.to(torch.int64)
            # A finalized span closes even if the chunk quota prevents emission;
            # otherwise an old open event could leak across arbitrary boundaries.
            opened = opened & ~end_now
            open_after_rows[:, time_index] = opened
        maximum = max(1, max(map(len, rows), default=0))
        options = dict(device=evidence.features.device, dtype=evidence.features.dtype)
        content = torch.zeros(batch, maximum, width, **options)
        identity = torch.zeros_like(content)
        spectral = torch.zeros(
            batch, maximum, *evidence.spectral.shape[2:], **options
        )
        types = torch.zeros(batch, maximum, proposal.node_type_logits.shape[-1], **options)
        support = torch.zeros(batch, maximum, 3, **options)
        modalities = torch.full((batch, maximum), -1, dtype=torch.int64, device=evidence.features.device)
        parents = torch.full_like(modalities, -1)
        provenance = torch.full_like(modalities, -1)
        sources = torch.full_like(modalities, int(SourceClass.INFERRED))
        scenarios = torch.full_like(modalities, -1)
        uncertainty = torch.zeros(batch, maximum, evidence.uncertainty.shape[-1], **options)
        scores = torch.full((batch, maximum), -torch.inf, **options)
        active = torch.zeros(batch, maximum, dtype=torch.bool, device=evidence.features.device)
        for batch_index, emitted in enumerate(rows):
            for slot, (time_index, start_time) in enumerate(emitted):
                content[batch_index, slot] = proposal.content[batch_index, time_index]
                identity[batch_index, slot] = proposal.identity_keys[batch_index, time_index]
                spectral[batch_index, slot] = evidence.spectral[batch_index, time_index]
                types[batch_index, slot] = proposal.node_type_logits[batch_index, time_index]
                end_time = evidence.timestamps[batch_index, time_index]
                support[batch_index, slot] = torch.stack((
                    end_time.new_tensor(start_time), end_time, end_time,
                ))
                modalities[batch_index, slot] = evidence.modality_ids[batch_index, time_index]
                parents[batch_index, slot] = evidence.parent_provenance_ids[batch_index, time_index]
                scenarios[batch_index, slot] = evidence.scenario_ids[batch_index, time_index]
                uncertainty[batch_index, slot] = evidence.uncertainty[batch_index, time_index]
                scores[batch_index, slot] = proposal.proposal_logits[batch_index, time_index]
                active[batch_index, slot] = True
        events = EventCandidates(
            content, identity, spectral, types, support, modalities, parents,
            provenance, sources, scenarios, uncertainty, scores, active,
        )
        next_state = EventExtractorState(starts, opened, counters, state.position + length)
        receipts = EventTransitionReceipts(
            opened_rows, finalized_rows, emitted_rows, rejected_rows,
            open_after_rows,
        )
        return events, next_state, proposal, receipts


class PersistentEventAllocator(nn.Module):
    """Update identity-compatible slots, otherwise allocate/evict by utility."""

    def __init__(
        self, width: int, *, identity_threshold: float = 0.75,
        update_maximum: float = 0.5,
    ) -> None:
        super().__init__()
        if width <= 0 or not -1 < identity_threshold < 1 or not 0 < update_maximum < 1:
            raise ValueError("event allocator controls are invalid")
        self.identity_threshold = identity_threshold
        self.update_maximum = update_maximum
        self.identity_projection = nn.Linear(width, width, bias=False)
        self.update_gate = nn.Linear(2 * width + 2, 1)

    def allocate_with_receipts(
        self, slots: NodeSlots, events: EventCandidates,
    ) -> tuple[NodeSlots, Tensor]:
        if slots.batch != events.content.shape[0] or slots.content.shape[-1] != events.content.shape[-1]:
            raise ValueError("event candidates and node slots are incompatible")
        if events.provenance_ids.shape != events.active.shape or bool(
            (events.active & (events.provenance_ids < 0)).any()
        ):
            raise ValueError("allocate only after deriving event provenance records")
        values = {name: getattr(slots, name).clone() for name in slots.__dataclass_fields__}
        batch, count = events.active.shape
        receipts = torch.full(
            (batch, count), -1, dtype=torch.int64, device=slots.content.device
        )
        for batch_index in range(batch):
            claimed: set[int] = set()
            for event_index in torch.nonzero(events.active[batch_index], as_tuple=False).flatten().tolist():
                current_active = values["active"][batch_index]
                similarity = F.cosine_similarity(
                    self.identity_projection(values["content"][batch_index]),
                    events.identity_keys[batch_index, event_index].unsqueeze(0), dim=-1,
                )
                same_scenario = values["scenario_ids"][batch_index] == events.scenario_ids[batch_index, event_index]
                overlap_or_after = (
                    values["support"][batch_index, :, 2]
                    <= events.support[batch_index, event_index, 2]
                )
                eligible = current_active & same_scenario & overlap_or_after
                if claimed:
                    claimed_tensor = torch.tensor(sorted(claimed), device=similarity.device)
                    eligible[claimed_tensor] = False
                masked = similarity.masked_fill(~eligible, -torch.inf)
                best_value, best_index = masked.max(0)
                if bool(torch.isfinite(best_value) & (best_value >= self.identity_threshold)):
                    slot = int(best_index.item())
                    gate_input = torch.cat((
                        values["content"][batch_index, slot], events.content[batch_index, event_index],
                        values["importance"][batch_index, slot].view(1),
                        events.score[batch_index, event_index].sigmoid().view(1),
                    ))
                    gate = torch.sigmoid(self.update_gate(gate_input)).squeeze() * self.update_maximum
                    values["content"][batch_index, slot] = (
                        (1 - gate) * values["content"][batch_index, slot]
                        + gate * events.content[batch_index, event_index]
                    )
                    values["spectral"][batch_index, slot] = (
                        (1 - gate) * values["spectral"][batch_index, slot]
                        + gate * events.spectral[batch_index, event_index]
                    )
                    values["support"][batch_index, slot, 1:] = events.support[batch_index, event_index, 1:]
                    values["uncertainty"][batch_index, slot] = events.uncertainty[batch_index, event_index]
                    values["provenance_ids"][batch_index, slot] = events.provenance_ids[batch_index, event_index]
                    values["source_classes"][batch_index, slot] = events.source_classes[batch_index, event_index]
                    values["importance"][batch_index, slot] = torch.maximum(
                        values["importance"][batch_index, slot], events.score[batch_index, event_index].sigmoid()
                    )
                    values["versions"][batch_index, slot] += 1
                else:
                    free = torch.nonzero(~current_active, as_tuple=False).flatten()
                    if free.numel():
                        slot = int(free[0].item())
                    else:
                        utility = (
                            values["importance"][batch_index]
                            + 0.25 * values["activity"][batch_index]
                            - 0.01 * values["age"][batch_index]
                        )
                        if claimed:
                            utility[torch.tensor(sorted(claimed), device=utility.device)] = torch.inf
                        slot = int(utility.argmin().item())
                    values["content"][batch_index, slot] = events.content[batch_index, event_index]
                    values["spectral"][batch_index, slot] = events.spectral[batch_index, event_index]
                    values["type_logits"][batch_index, slot] = events.type_logits[batch_index, event_index]
                    values["support"][batch_index, slot] = events.support[batch_index, event_index]
                    values["modality_presence"][batch_index, slot].zero_()
                    modality = int(events.modality_ids[batch_index, event_index].item())
                    if not 0 <= modality < values["modality_presence"].shape[-1]:
                        raise ValueError("event modality lies outside node-slot capacity")
                    values["modality_presence"][batch_index, slot, modality] = 1
                    values["uncertainty"][batch_index, slot] = events.uncertainty[batch_index, event_index]
                    values["provenance_features"][batch_index, slot].zero_()
                    values["provenance_ids"][batch_index, slot] = events.provenance_ids[batch_index, event_index]
                    values["source_classes"][batch_index, slot] = events.source_classes[batch_index, event_index]
                    values["scenario_ids"][batch_index, slot] = events.scenario_ids[batch_index, event_index]
                    values["hypothesis_membership"][batch_index, slot].zero_()
                    values["activity"][batch_index, slot] = 1
                    values["age"][batch_index, slot] = 0
                    values["importance"][batch_index, slot] = events.score[batch_index, event_index].sigmoid()
                    values["versions"][batch_index, slot] += 1
                    values["active"][batch_index, slot] = True
                claimed.add(slot)
                receipts[batch_index, event_index] = slot
            values["age"][batch_index, values["active"][batch_index]] += 1
            values["activity"][batch_index] *= 0.95
        return NodeSlots(**values), receipts

    def forward(self, slots: NodeSlots, events: EventCandidates) -> NodeSlots:
        updated, _ = self.allocate_with_receipts(slots, events)
        return updated
