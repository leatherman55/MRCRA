"""Evidence-conditioned localized relational reconstruction contracts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cognitive_types import SourceClass
from .tensor_state import TensorStateMixin
from .runtime_validation import runtime_validation_enabled


def _base(name: str, value: Tensor, shape: tuple[int, ...], dtype=None) -> None:
    if value.shape != shape or (dtype is not None and value.dtype != dtype):
        raise ValueError(f"reconstruction {name} must have shape {shape}")


@dataclass(frozen=True, slots=True)
class ReconstructionQuery(TensorStateMixin):
    abstraction_indices: Tensor
    seed_node_indices: Tensor
    requested_support: Tensor
    requested_node_count: Tensor
    requested_relation_count: Tensor
    target_scale: Tensor
    target_abstraction_depth: Tensor
    precision_tolerance: Tensor
    goal_context: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        batch = self.abstraction_indices.shape[0]
        for name in (
            "abstraction_indices", "seed_node_indices", "requested_node_count",
            "requested_relation_count", "target_scale", "target_abstraction_depth",
        ):
            _base(name, getattr(self, name), (batch,), torch.int64)
        _base("requested_support", self.requested_support, (batch, 3))
        _base("precision_tolerance", self.precision_tolerance, (batch,))
        if self.goal_context.ndim != 2 or self.goal_context.shape[0] != batch:
            raise ValueError("reconstruction goal context must be (batch,width)")
        _base("mask", self.mask, (batch,), torch.bool)
        if bool((self.mask & (self.abstraction_indices < 0)).any()):
            raise ValueError("active reconstruction queries require an abstraction")
        if bool((self.mask & ((self.requested_node_count <= 0) | (self.requested_relation_count < 0))).any()):
            raise ValueError("active reconstruction query sizes are invalid")
        if bool((self.mask & (self.precision_tolerance < 0)).any()):
            raise ValueError("reconstruction precision tolerance cannot be negative")


@dataclass(frozen=True, slots=True)
class ReconstructionEvidence(TensorStateMixin):
    abstraction_latent: Tensor
    trace_content: Tensor
    trace_mask: Tensor
    trace_provenance_ids: Tensor
    observed_context: Tensor
    observed_provenance_ids: Tensor
    current_relations: Tensor
    hypothesis_context: Tensor
    goal_context: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.abstraction_latent.ndim != 2:
            raise ValueError("reconstruction abstraction latent must be (batch,width)")
        batch, width = self.abstraction_latent.shape
        if self.trace_content.ndim != 3 or self.trace_content.shape[0] != batch or self.trace_content.shape[-1] != width:
            raise ValueError("reconstruction traces must be (batch,traces,width)")
        trace_shape = self.trace_content.shape[:2]
        _base("trace_mask", self.trace_mask, trace_shape, torch.bool)
        _base("trace_provenance_ids", self.trace_provenance_ids, trace_shape, torch.int64)
        if bool((self.trace_mask & (self.trace_provenance_ids < 0)).any()):
            raise ValueError("active reconstruction traces require provenance")
        for name in ("observed_context", "hypothesis_context", "goal_context"):
            _base(name, getattr(self, name), (batch, width))
        if self.current_relations.ndim != 3 or self.current_relations.shape[0] != batch or self.current_relations.shape[-1] != width:
            raise ValueError("reconstruction relations must be (batch,relations,width)")
        if self.observed_provenance_ids.ndim != 2 or self.observed_provenance_ids.shape[0] != batch or self.observed_provenance_ids.dtype != torch.int64:
            raise ValueError("reconstruction observed provenance must be (batch,evidence)")


@dataclass(frozen=True, slots=True)
class ReconstructionResult(TensorStateMixin):
    node_content: Tensor
    node_type_logits: Tensor
    node_mask: Tensor
    relation_content: Tensor
    relation_type_logits: Tensor
    participant_indices: Tensor
    relation_mask: Tensor
    historical_fidelity: Tensor
    structural_plausibility: Tensor
    evidence_agreement: Tensor
    epistemic_uncertainty: Tensor
    aleatoric_uncertainty: Tensor
    applicability_probability: Tensor
    provenance_ids: Tensor
    relation_provenance_ids: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.node_content.ndim != 3 or self.relation_content.ndim != 3:
            raise ValueError("reconstruction nodes and relations must be batched sequences")
        batch, nodes = self.node_content.shape[:2]
        relations = self.relation_content.shape[1]
        if self.relation_content.shape[0] != batch:
            raise ValueError("reconstruction node and relation batches differ")
        if self.node_type_logits.ndim != 3 or self.node_type_logits.shape[:2] != (batch, nodes):
            raise ValueError("reconstruction node types must match node rows")
        if self.relation_type_logits.ndim != 3 or self.relation_type_logits.shape[:2] != (batch, relations):
            raise ValueError("reconstruction relation types must match relation rows")
        _base("node_mask", self.node_mask, (batch, nodes), torch.bool)
        _base("relation_mask", self.relation_mask, (batch, relations), torch.bool)
        if self.participant_indices.ndim != 3 or self.participant_indices.shape[:2] != (batch, relations) or self.participant_indices.dtype != torch.int64:
            raise ValueError("reconstruction participants must be int64 relation rows")
        for name in (
            "historical_fidelity", "structural_plausibility", "evidence_agreement",
            "epistemic_uncertainty", "aleatoric_uncertainty", "applicability_probability",
        ):
            _base(name, getattr(self, name), (batch,))
        if self.provenance_ids.shape != (batch, nodes) or self.provenance_ids.dtype != torch.int64:
            raise ValueError("reconstruction provenance must match node rows")
        if self.relation_provenance_ids.shape != (batch, relations) or self.relation_provenance_ids.dtype != torch.int64:
            raise ValueError("reconstruction relation provenance must match relation rows")
        if bool((self.node_mask & (self.provenance_ids < 0)).any()):
            raise ValueError("active reconstructed nodes require provenance")
        if bool((self.relation_mask & (self.relation_provenance_ids < 0)).any()):
            raise ValueError("active reconstructed relations require provenance")
        for name in ("historical_fidelity", "structural_plausibility", "evidence_agreement", "applicability_probability"):
            value = getattr(self, name)
            if bool(((value < 0) | (value > 1)).any()):
                raise ValueError(f"reconstruction {name} must lie in [0,1]")
        if bool((self.epistemic_uncertainty < 0).any() | (self.aleatoric_uncertainty < 0).any()):
            raise ValueError("reconstruction uncertainty cannot be negative")


@dataclass(frozen=True, slots=True)
class ReconstructionState(TensorStateMixin):
    latent: Tensor
    historical_fidelity: Tensor
    structural_plausibility: Tensor
    evidence_agreement: Tensor
    uncertainty: Tensor
    abstraction_indices: Tensor
    provenance_ids: Tensor
    source_classes: Tensor
    scenario_ids: Tensor
    physical_scales: Tensor
    abstraction_depths: Tensor
    support: Tensor
    versions: Tensor
    active: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.latent.ndim != 3:
            raise ValueError("reconstruction state latent must be (batch,capacity,width)")
        base = self.latent.shape[:2]
        for name in ("historical_fidelity", "structural_plausibility", "evidence_agreement"):
            _base(name, getattr(self, name), base)
        if self.uncertainty.ndim != 3 or self.uncertainty.shape[:2] != base:
            raise ValueError("reconstruction state uncertainty must share rows")
        for name in (
            "abstraction_indices", "provenance_ids", "source_classes", "scenario_ids",
            "physical_scales", "abstraction_depths", "versions",
        ):
            _base(name, getattr(self, name), base, torch.int64)
        _base("support", self.support, (*base, 3))
        _base("active", self.active, base, torch.bool)
        if bool((self.active & (self.provenance_ids < 0)).any()):
            raise ValueError("active reconstruction records require provenance")
        if bool((self.active & (self.source_classes != int(SourceClass.RECONSTRUCTED))).any()):
            raise ValueError("active reconstruction records require reconstructed source class")

    @classmethod
    def empty(
        cls, batch: int, capacity: int, width: int, uncertainty_channels: int,
        *, device=None, dtype=None,
    ) -> "ReconstructionState":
        if min(batch, capacity, width, uncertainty_channels) <= 0:
            raise ValueError("reconstruction state dimensions must be positive")
        base = (batch, capacity)
        floats = dict(device=device, dtype=dtype)
        ids = lambda fill=-1: torch.full(base, fill, dtype=torch.int64, device=device)
        return cls(
            torch.zeros(*base, width, **floats),
            torch.zeros(base, **floats), torch.zeros(base, **floats),
            torch.zeros(base, **floats),
            torch.zeros(*base, uncertainty_channels, **floats),
            ids(), ids(), ids(), ids(), ids(), ids(),
            torch.zeros(*base, 3, **floats), ids(0),
            torch.zeros(base, dtype=torch.bool, device=device),
        )

    @property
    def batch(self) -> int:
        return self.latent.shape[0]

    @property
    def capacity(self) -> int:
        return self.latent.shape[1]


@dataclass(frozen=True, slots=True)
class ReconstructionProposal(TensorStateMixin):
    """Neural reconstruction before the authority layer derives provenance."""

    node_content: Tensor
    node_type_logits: Tensor
    node_mask: Tensor
    relation_content: Tensor
    relation_type_logits: Tensor
    participant_indices: Tensor
    relation_mask: Tensor
    historical_fidelity: Tensor
    structural_plausibility: Tensor
    evidence_agreement: Tensor
    epistemic_uncertainty: Tensor
    aleatoric_uncertainty: Tensor
    applicability_probability: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.node_content.ndim != 3 or self.relation_content.ndim != 3:
            raise ValueError("reconstruction proposal content must be batched sequences")
        batch, nodes = self.node_content.shape[:2]
        relations = self.relation_content.shape[1]
        if self.relation_content.shape[0] != batch:
            raise ValueError("reconstruction proposal batches differ")
        if self.node_type_logits.ndim != 3 or self.node_type_logits.shape[:2] != (batch, nodes):
            raise ValueError("reconstruction proposal node types must match nodes")
        if self.relation_type_logits.ndim != 3 or self.relation_type_logits.shape[:2] != (batch, relations):
            raise ValueError("reconstruction proposal relation types must match relations")
        _base("proposal node_mask", self.node_mask, (batch, nodes), torch.bool)
        _base("proposal relation_mask", self.relation_mask, (batch, relations), torch.bool)
        if self.participant_indices.ndim != 3 or self.participant_indices.shape[:2] != (batch, relations) or self.participant_indices.dtype != torch.int64:
            raise ValueError("reconstruction proposal participants are invalid")
        for name in (
            "historical_fidelity", "structural_plausibility", "evidence_agreement",
            "epistemic_uncertainty", "aleatoric_uncertainty", "applicability_probability",
        ):
            _base(name, getattr(self, name), (batch,))
        for name in ("historical_fidelity", "structural_plausibility", "evidence_agreement", "applicability_probability"):
            value = getattr(self, name)
            if bool(((value < 0) | (value > 1)).any()):
                raise ValueError(f"reconstruction proposal {name} must lie in [0,1]")

    def finalize(
        self, provenance_ids: Tensor, relation_provenance_ids: Tensor,
    ) -> ReconstructionResult:
        return ReconstructionResult(
            self.node_content, self.node_type_logits, self.node_mask,
            self.relation_content, self.relation_type_logits,
            self.participant_indices, self.relation_mask,
            self.historical_fidelity, self.structural_plausibility,
            self.evidence_agreement, self.epistemic_uncertainty,
            self.aleatoric_uncertainty, self.applicability_probability,
            provenance_ids, relation_provenance_ids,
        )


class ConditionalGraphReconstructor(nn.Module):
    """Decode a bounded local graph from abstraction, traces, and current evidence."""

    def __init__(
        self, width: int, node_types: int, relation_families: int,
        maximum_nodes: int, maximum_relations: int, *, arity: int = 2,
    ) -> None:
        super().__init__()
        if min(width, node_types, relation_families, maximum_nodes, maximum_relations, arity) <= 0:
            raise ValueError("conditional reconstructor dimensions must be positive")
        if arity < 2:
            raise ValueError("reconstructed relations require at least binary arity")
        self.width = width
        self.maximum_nodes = maximum_nodes
        self.maximum_relations = maximum_relations
        self.arity = arity
        self.trace_query = nn.Linear(width, width, bias=False)
        self.trace_key = nn.Linear(width, width, bias=False)
        self.trace_value = nn.Linear(width, width, bias=False)
        self.context = nn.Sequential(
            nn.Linear(7 * width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width),
        )
        self.node_queries = nn.Parameter(torch.empty(maximum_nodes, width))
        self.relation_queries = nn.Parameter(torch.empty(maximum_relations, width))
        nn.init.normal_(self.node_queries, std=width ** -0.5)
        nn.init.normal_(self.relation_queries, std=width ** -0.5)
        self.node_decoder = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.node_type = nn.Linear(width, node_types)
        self.relation_decoder = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.relation_type = nn.Linear(width, relation_families)
        self.fidelity_head = nn.Linear(width, 3)
        self.uncertainty_head = nn.Linear(width, 2)
        self.applicability_head = nn.Linear(width, 1)

    def _trace_summary(self, abstraction: Tensor, evidence: ReconstructionEvidence) -> Tensor:
        logits = torch.einsum(
            "bd,bkd->bk", self.trace_query(abstraction), self.trace_key(evidence.trace_content)
        ) / self.width ** 0.5
        logits = logits.masked_fill(~evidence.trace_mask, -torch.inf)
        present = evidence.trace_mask.any(-1)
        weights = torch.zeros_like(logits)
        if bool(present.any()):
            weights[present] = torch.softmax(logits[present], -1)
        return torch.einsum("bk,bkd->bd", weights, self.trace_value(evidence.trace_content))

    def forward(
        self, query: ReconstructionQuery, evidence: ReconstructionEvidence,
    ) -> ReconstructionProposal:
        batch = query.mask.shape[0]
        if evidence.abstraction_latent.shape != (batch, self.width):
            raise ValueError("reconstruction query and evidence batches are incompatible")
        relation_summary = (
            evidence.current_relations.mean(1)
            if evidence.current_relations.shape[1]
            else evidence.abstraction_latent.new_zeros(batch, self.width)
        )
        trace_summary = self._trace_summary(evidence.abstraction_latent, evidence)
        combined = torch.cat((
            evidence.abstraction_latent, trace_summary, evidence.observed_context,
            relation_summary, evidence.hypothesis_context, evidence.goal_context,
            query.goal_context,
        ), -1)
        context = self.context(combined)
        node_hidden = context[:, None] + self.node_queries[None]
        node_content = self.node_decoder(node_hidden)
        node_types = self.node_type(node_hidden)
        node_range = torch.arange(self.maximum_nodes, device=context.device)[None]
        node_count = query.requested_node_count.clamp(0, self.maximum_nodes)
        node_mask = query.mask[:, None] & (node_range < node_count[:, None])
        node_content = node_content * node_mask.unsqueeze(-1)

        relation_hidden = context[:, None] + self.relation_queries[None]
        relation_content = self.relation_decoder(relation_hidden)
        relation_types = self.relation_type(relation_hidden)
        relation_range = torch.arange(self.maximum_relations, device=context.device)[None]
        relation_count = query.requested_relation_count.clamp(0, self.maximum_relations)
        relation_mask = (
            query.mask[:, None] & (relation_range < relation_count[:, None])
            & (node_count[:, None] >= 2)
        )
        relation_content = relation_content * relation_mask.unsqueeze(-1)
        safe_count = node_count.clamp_min(1)
        source = relation_range.remainder(safe_count[:, None])
        target = (source + 1).remainder(safe_count[:, None])
        participants = torch.full(
            (batch, self.maximum_relations, self.arity), -1,
            dtype=torch.int64, device=context.device,
        )
        participants[..., 0] = source
        participants[..., 1] = target
        participants = participants.masked_fill(~relation_mask.unsqueeze(-1), -1)

        fidelity = torch.sigmoid(self.fidelity_head(context))
        uncertainty = F.softplus(self.uncertainty_head(context))
        applicability = torch.sigmoid(self.applicability_head(context)).squeeze(-1)
        active = query.mask.to(context.dtype)
        return ReconstructionProposal(
            node_content, node_types, node_mask, relation_content, relation_types,
            participants, relation_mask,
            fidelity[:, 0] * active, fidelity[:, 1] * active,
            fidelity[:, 2] * active, uncertainty[:, 0] * active,
            uncertainty[:, 1] * active, applicability * active,
        )
