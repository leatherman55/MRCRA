"""Role normalization, bounded structural matching, and conditional invariants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .compression import GraphFragment


@dataclass(frozen=True, slots=True)
class NormalizedGraph:
    content: Tensor
    role_logits: Tensor
    role_ids: Tensor
    node_type_ids: Tensor
    node_provenance_ids: Tensor
    node_mask: Tensor
    adjacency: Tensor
    relation_mask: Tensor

    def __post_init__(self) -> None:
        if self.content.ndim != 3 or self.role_logits.ndim != 3:
            raise ValueError("normalized graph content and roles must be (batch,nodes,features)")
        base = self.content.shape[:2]
        if self.role_logits.shape[:2] != base:
            raise ValueError("role logits must share graph node shape")
        for name in ("role_ids", "node_type_ids", "node_provenance_ids"):
            value = getattr(self, name)
            if value.shape != base or value.dtype != torch.int64:
                raise ValueError(f"normalized graph {name} must be int64 with node shape")
        if self.node_mask.shape != base or self.node_mask.dtype != torch.bool:
            raise ValueError("normalized graph node mask is invalid")
        if self.adjacency.ndim != 4 or self.adjacency.shape[0] != base[0] or self.adjacency.shape[2:] != (base[1], base[1]):
            raise ValueError("normalized adjacency must be (batch,families,nodes,nodes)")
        if self.relation_mask.ndim != 2 or self.relation_mask.shape[0] != base[0]:
            raise ValueError("normalized relation mask has invalid shape")


class StructuralNormalizer(nn.Module):
    """Add a role-variable view without replacing identities or provenance."""

    def __init__(self, width: int, node_types: int, role_count: int, relation_families: int) -> None:
        super().__init__()
        if min(width, node_types, role_count, relation_families) <= 0:
            raise ValueError("structural normalizer dimensions must be positive")
        self.node_types = node_types
        self.role_count = role_count
        self.relation_families = relation_families
        self.type_embedding = nn.Embedding(node_types, width)
        self.role_head = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, role_count))
        self.content = nn.Linear(2 * width, width)

    def forward(self, fragment: GraphFragment) -> NormalizedGraph:
        if fragment.node_content.shape[-1] != self.type_embedding.embedding_dim:
            raise ValueError("fragment width does not match structural normalizer")
        type_embedding = self.type_embedding(fragment.node_type_ids.clamp(0, self.node_types - 1))
        combined = torch.cat((fragment.node_content, type_embedding), -1)
        role_logits = self.role_head(combined)
        role_ids = role_logits.argmax(-1).masked_fill(~fragment.node_mask, -1)
        content = self.content(combined) * fragment.node_mask.unsqueeze(-1)
        batch, nodes = fragment.node_mask.shape
        adjacency = content.new_zeros(batch, self.relation_families, nodes, nodes)
        relation_mask = torch.zeros(
            batch, fragment.relation_content.shape[1], dtype=torch.bool,
            device=fragment.node_content.device,
        )
        for batch_index in range(batch):
            for relation in torch.nonzero(fragment.relation_mask[batch_index], as_tuple=False).flatten().tolist():
                participants = fragment.participant_indices[batch_index, relation]
                if participants.numel() < 2:
                    continue
                source, target = int(participants[0]), int(participants[1])
                family = int(fragment.relation_family_ids[batch_index, relation])
                if not 0 <= family < self.relation_families:
                    raise ValueError("fragment relation family lies outside the ontology")
                adjacency[batch_index, family, source, target] = 1
                relation_mask[batch_index, relation] = True
        return NormalizedGraph(
            content, role_logits, role_ids, fragment.node_type_ids,
            fragment.node_provenance_ids, fragment.node_mask, adjacency, relation_mask,
        )


@dataclass(frozen=True, slots=True)
class GraphMatch:
    assignment: Tensor
    node_cost: Tensor
    relation_cost: Tensor
    total_cost: Tensor
    matched_mask: Tensor


class BoundedGraphMatcher(nn.Module):
    """Permutation-aware Sinkhorn matching on routed bounded subgraphs."""

    def __init__(
        self, maximum_nodes: int, *, sinkhorn_iterations: int = 8,
        temperature: float = 0.1, content_weight: float = 1.0,
        relation_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if min(maximum_nodes, sinkhorn_iterations) <= 0 or temperature <= 0:
            raise ValueError("graph matcher controls are invalid")
        if min(content_weight, relation_weight) < 0:
            raise ValueError("graph matcher weights cannot be negative")
        self.maximum_nodes = maximum_nodes
        self.sinkhorn_iterations = sinkhorn_iterations
        self.temperature = temperature
        self.content_weight = content_weight
        self.relation_weight = relation_weight

    def forward(self, left: NormalizedGraph, right: NormalizedGraph) -> GraphMatch:
        if left.content.shape[0] != right.content.shape[0] or left.content.shape[-1] != right.content.shape[-1]:
            raise ValueError("matched graphs must share batch and content width")
        left_nodes, right_nodes = left.content.shape[1], right.content.shape[1]
        if max(left_nodes, right_nodes) > self.maximum_nodes:
            raise ValueError("graph matcher node bound exceeded")
        if left.adjacency.shape[1] != right.adjacency.shape[1]:
            raise ValueError("relation ontologies must match exactly; relabeling is forbidden")
        similarity = torch.einsum(
            "bid,bjd->bij", F.normalize(left.content, dim=-1), F.normalize(right.content, dim=-1)
        )
        same_role = left.role_ids[:, :, None] == right.role_ids[:, None, :]
        same_type = left.node_type_ids[:, :, None] == right.node_type_ids[:, None, :]
        valid = left.node_mask[:, :, None] & right.node_mask[:, None, :]
        logits = (similarity + same_role + same_type) / self.temperature
        assignment = torch.exp(logits - logits.amax((-2, -1), keepdim=True)) * valid
        for _ in range(self.sinkhorn_iterations):
            assignment = assignment / assignment.sum(-1, keepdim=True).clamp_min(1e-8)
            assignment = assignment / assignment.sum(-2, keepdim=True).clamp_min(1e-8)
            assignment = assignment * valid
        node_error = 1 - similarity
        node_cost = (assignment * node_error).sum((-2, -1)) / assignment.sum((-2, -1)).clamp_min(1)
        # Map right adjacency into left coordinates, preserving each family.
        mapped = torch.einsum("bij,brjk,bkl->bril", assignment, right.adjacency, assignment.transpose(-1, -2))
        relation_cost = (left.adjacency - mapped).square().mean((-1, -2, -3))
        total = self.content_weight * node_cost + self.relation_weight * relation_cost
        return GraphMatch(assignment, node_cost, relation_cost, total, valid)


@dataclass(frozen=True, slots=True)
class InvariantDiscoveryProposal:
    latent: Tensor
    code_gain_bits: Tensor
    reconstruction_distortion: Tensor
    relation_distortion: Tensor
    match_cost: Tensor
    counterexample_cost: Tensor
    predictive_utility: Tensor
    applicability_probability: Tensor
    supporting_provenance_ids: Tensor
    supporting_mask: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        if self.latent.ndim != 2:
            raise ValueError("invariant proposal latent must be (batch,width)")
        batch = self.latent.shape[0]
        for name in (
            "code_gain_bits", "reconstruction_distortion", "relation_distortion",
            "match_cost", "counterexample_cost", "predictive_utility",
            "applicability_probability",
        ):
            if getattr(self, name).shape != (batch,):
                raise ValueError(f"invariant proposal {name} must be per batch")
        if self.supporting_provenance_ids.ndim != 2 or self.supporting_provenance_ids.shape[0] != batch:
            raise ValueError("invariant proposal supporters must be (batch,supporters)")
        if self.supporting_mask.shape != self.supporting_provenance_ids.shape or self.supporting_mask.dtype != torch.bool:
            raise ValueError("invariant proposal supporter mask is invalid")
        if self.mask.shape != (batch,) or self.mask.dtype != torch.bool:
            raise ValueError("invariant proposal mask is invalid")
        if bool((self.supporting_mask & (self.supporting_provenance_ids < 0)).any()):
            raise ValueError("invariant proposal supporters require provenance")


class IntegratedInvariantDiscoverer(nn.Module):
    """Role-normalized, permutation-aware, counterexample-tested discovery."""

    def __init__(
        self, width: int, node_types: int, role_count: int,
        relation_families: int, maximum_nodes: int,
        *, maximum_match_cost: float = 0.35,
        minimum_counterexample_margin: float = 0.05,
    ) -> None:
        super().__init__()
        if maximum_match_cost < 0 or minimum_counterexample_margin < 0:
            raise ValueError("invariant discovery thresholds cannot be negative")
        self.normalizer = StructuralNormalizer(
            width, node_types, role_count, relation_families
        )
        self.matcher = BoundedGraphMatcher(maximum_nodes)
        self.maximum_match_cost = maximum_match_cost
        self.minimum_counterexample_margin = minimum_counterexample_margin
        self.pattern = nn.Linear(width, width)
        self.applicability = nn.Sequential(
            nn.Linear(3, max(4, width // 4)), nn.SiLU(), nn.Linear(max(4, width // 4), 1)
        )

    @staticmethod
    def _supporters(left: NormalizedGraph, right: NormalizedGraph) -> tuple[Tensor, Tensor]:
        ids = torch.cat((left.node_provenance_ids, right.node_provenance_ids), -1)
        mask = torch.cat((left.node_mask, right.node_mask), -1)
        return ids.masked_fill(~mask, -1), mask

    def forward(
        self, left_fragment: GraphFragment, right_fragment: GraphFragment,
    ) -> InvariantDiscoveryProposal:
        left = self.normalizer(left_fragment)
        right = self.normalizer(right_fragment)
        match = self.matcher(left, right)
        aligned_right = torch.einsum("bij,bjd->bid", match.assignment, right.content)
        shared = 0.5 * (left.content + aligned_right)
        latent = self.pattern(
            (shared * left.node_mask.unsqueeze(-1)).sum(1)
            / left.node_mask.sum(1, keepdim=True).clamp_min(1)
        )
        # A declared structural near-match changes every relation family while
        # preserving node identities.  A useful invariant must distinguish it.
        altered = NormalizedGraph(
            right.content, right.role_logits, right.role_ids, right.node_type_ids,
            right.node_provenance_ids, right.node_mask,
            right.adjacency.roll(1, dims=1), right.relation_mask,
        )
        counterexample = self.matcher(left, altered)
        node_count = torch.minimum(
            left.node_mask.sum(-1), right.node_mask.sum(-1)
        ).to(latent.dtype)
        relation_count = torch.minimum(
            left.relation_mask.sum(-1), right.relation_mask.sum(-1)
        ).to(latent.dtype)
        baseline_bits = (node_count + relation_count) * latent.shape[-1] * 16
        invariant_bits = latent.shape[-1] * 16 + (node_count + relation_count) * 2
        gain = (baseline_bits - invariant_bits).clamp_min(0)
        counterexample_margin = counterexample.total_cost - match.total_cost
        utility = (counterexample_margin - match.total_cost).tanh()
        applicability_features = torch.stack((
            match.node_cost, match.relation_cost, counterexample_margin,
        ), -1)
        applicability = torch.sigmoid(
            self.applicability(applicability_features).squeeze(-1)
        )
        supporters, supporter_mask = self._supporters(left, right)
        mask = (
            (left.node_mask.sum(-1) >= 2) & (right.node_mask.sum(-1) >= 2)
            & (match.total_cost <= self.maximum_match_cost)
            & (counterexample_margin >= self.minimum_counterexample_margin)
            & (gain > 0) & supporter_mask.any(-1)
        )
        return InvariantDiscoveryProposal(
            latent, gain, match.node_cost, match.relation_cost,
            match.total_cost, counterexample.total_cost, utility,
            applicability, supporters, supporter_mask, mask,
        )


@dataclass(frozen=True, slots=True)
class InvariantEvidence:
    pattern: tuple[float, ...]
    applicability_conditions: tuple[str, ...]
    known_failures: tuple[str, ...]
    procedure: tuple[float, ...]
    residual_decoder: tuple[float, ...]
    episode_ids: tuple[int, ...]
    transformation_ids: tuple[str, ...]
    supporting_provenance_ids: tuple[int, ...]
    contradicting_provenance_ids: tuple[int, ...]
    independent_source_roots: tuple[int, ...]
    predictive_utility: float
    action_utility: float
    code_gain_bits: float
    reconstruction_distortion: float
    relation_distortion: float
    calibrated_confidence: float
    counterexample_search_completed: bool


@dataclass(frozen=True, slots=True)
class ConditionalInvariantRecord:
    record_id: int
    revision_of: int | None
    evidence: InvariantEvidence
    provenance_id: int


class InvariantLedger:
    """Append-only semantic authority with mandatory conditions and failures."""

    def __init__(self) -> None:
        self._records: list[ConditionalInvariantRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    def get(self, record_id: int) -> ConditionalInvariantRecord:
        if not 0 <= record_id < len(self._records):
            raise KeyError(f"unknown invariant {record_id}")
        return self._records[record_id]

    def promote(
        self, evidence: InvariantEvidence, *, provenance_id: int,
        maximum_reconstruction_distortion: float,
        maximum_relation_distortion: float,
        require_source_diversity: bool = True,
        revision_of: int | None = None,
    ) -> int:
        if provenance_id < 0 or min(maximum_reconstruction_distortion, maximum_relation_distortion) < 0:
            raise ValueError("invariant provenance and thresholds are invalid")
        if revision_of is not None:
            self.get(revision_of)
        if len(set(evidence.episode_ids)) < 2 and len(set(evidence.transformation_ids)) < 2:
            raise ValueError("invariant coverage requires multiple episodes or transformations")
        if require_source_diversity and len(set(evidence.independent_source_roots)) < 2:
            raise ValueError("claimed source diversity requires independent provenance roots")
        if evidence.predictive_utility <= 0 and evidence.action_utility <= 0:
            raise ValueError("an invariant requires predictive or action utility")
        if evidence.code_gain_bits <= 0:
            raise ValueError("an invariant requires measured compression gain")
        if evidence.reconstruction_distortion > maximum_reconstruction_distortion:
            raise ValueError("invariant reconstruction distortion exceeds its bound")
        if evidence.relation_distortion > maximum_relation_distortion:
            raise ValueError("invariant relation distortion exceeds its bound")
        if not evidence.counterexample_search_completed:
            raise ValueError("invariant promotion requires explicit counterexample search")
        if not evidence.applicability_conditions:
            raise ValueError("conditional invariants require applicability conditions")
        if not 0 <= evidence.calibrated_confidence <= 1:
            raise ValueError("invariant confidence must lie in [0,1]")
        record_id = len(self._records)
        self._records.append(ConditionalInvariantRecord(
            record_id, revision_of, evidence, provenance_id,
        ))
        return record_id

    def add_counterexample(
        self, record_id: int, *, failure_condition: str,
        provenance_id: int, calibrated_confidence: float,
    ) -> int:
        current = self.get(record_id)
        if not failure_condition or provenance_id < 0 or not 0 <= calibrated_confidence <= 1:
            raise ValueError("counterexample revision fields are invalid")
        old = current.evidence
        revised = InvariantEvidence(
            old.pattern, old.applicability_conditions,
            tuple(dict.fromkeys((*old.known_failures, failure_condition))),
            old.procedure, old.residual_decoder, old.episode_ids, old.transformation_ids,
            old.supporting_provenance_ids,
            tuple(dict.fromkeys((*old.contradicting_provenance_ids, provenance_id))),
            old.independent_source_roots, old.predictive_utility, old.action_utility,
            old.code_gain_bits, old.reconstruction_distortion, old.relation_distortion,
            calibrated_confidence, True,
        )
        new_id = len(self._records)
        self._records.append(ConditionalInvariantRecord(new_id, record_id, revised, provenance_id))
        return new_id


class SymbolActivator(nn.Module):
    """Context-conditioned activation of persistent invariant identifiers."""

    def __init__(self, invariant_capacity: int, width: int, context_width: int) -> None:
        super().__init__()
        if min(invariant_capacity, width, context_width) <= 0:
            raise ValueError("symbol activator dimensions must be positive")
        self.embedding = nn.Embedding(invariant_capacity, width)
        self.context = nn.Linear(context_width, width)
        self.gate = nn.Linear(2 * width, 1)

    def forward(self, invariant_ids: Tensor, context: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if invariant_ids.ndim != 2 or invariant_ids.dtype != torch.int64:
            raise ValueError("invariant IDs must be int64 with shape (batch,symbols)")
        if context.ndim != 2 or context.shape[0] != invariant_ids.shape[0]:
            raise ValueError("symbol context must be (batch,features)")
        if mask.shape != invariant_ids.shape or mask.dtype != torch.bool:
            raise ValueError("symbol mask must match invariant IDs")
        embedding = self.embedding(invariant_ids.clamp_min(0))
        conditioned = torch.tanh(embedding + self.context(context)[:, None])
        gate = torch.sigmoid(self.gate(torch.cat((embedding, conditioned), -1))).squeeze(-1)
        gate = gate * mask
        return conditioned * gate.unsqueeze(-1), gate
