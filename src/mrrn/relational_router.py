"""Bounded phase-aware typed relational resonance routing."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cognitive_types import NodeSlots, RELATION_COMPATIBILITY, RelationFamily
from .runtime_validation import runtime_validation_enabled


def _masked_softmax(logits: Tensor, mask: Tensor, dim: int) -> Tensor:
    masked = logits.masked_fill(~mask, -torch.inf)
    maximum = masked.amax(dim=dim, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    exponential = torch.exp(masked - maximum) * mask.to(logits.dtype)
    return exponential / exponential.sum(dim=dim, keepdim=True).clamp_min(
        torch.finfo(logits.dtype).tiny
    )


@dataclass(frozen=True, slots=True)
class RelationalCandidates:
    content: Tensor
    spectral: Tensor
    type_logits: Tensor
    support: Tensor
    modality_presence: Tensor
    uncertainty: Tensor
    provenance_features: Tensor
    source_classes: Tensor
    scenario_ids: Tensor
    node_indices: Tensor
    tier_ids: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.content.ndim != 4:
            raise ValueError("relational candidate content must be (batch,queries,k,width)")
        base = self.content.shape[:3]
        if self.spectral.ndim != 6 or self.spectral.shape[:3] != base or self.spectral.shape[-1] != 2:
            raise ValueError("candidate spectral state must be (batch,queries,k,heads,modes,2)")
        for name in ("type_logits", "modality_presence", "uncertainty", "provenance_features"):
            value = getattr(self, name)
            if value.ndim != 4 or value.shape[:3] != base:
                raise ValueError(f"candidate {name} has invalid shape")
        if self.support.shape != (*base, 3):
            raise ValueError("candidate support has invalid shape")
        for name in ("source_classes", "scenario_ids", "node_indices", "tier_ids"):
            value = getattr(self, name)
            if value.shape != base or value.dtype != torch.int64:
                raise ValueError(f"candidate {name} must be int64 with candidate shape")
        if self.mask.shape != base or self.mask.dtype != torch.bool:
            raise ValueError("candidate mask must be boolean with candidate shape")
        if bool((self.mask & (self.node_indices < 0)).any()):
            raise ValueError("valid in-workspace candidates require authoritative node indices")


class NodeCandidateBuilder(nn.Module):
    """Cheap low-dimensional router over the bounded active node ring."""

    def __init__(self, width: int, maximum_candidates: int, *, router_dim: int = 32) -> None:
        super().__init__()
        if min(width, maximum_candidates, router_dim) <= 0:
            raise ValueError("candidate router dimensions must be positive")
        self.maximum_candidates = maximum_candidates
        self.query = nn.Linear(width, router_dim, bias=False)
        self.key = nn.Linear(width, router_dim, bias=False)
        self.recency_weight = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _gather(value: Tensor, indices: Tensor) -> Tensor:
        batch, queries, count = indices.shape
        tail = value.shape[2:]
        expanded = value[:, None].expand(batch, queries, *value.shape[1:])
        gather_index = indices.clamp_min(0).reshape(batch, queries, count, *([1] * len(tail)))
        gather_index = gather_index.expand(batch, queries, count, *tail)
        return torch.gather(expanded, 2, gather_index)

    def forward(
        self, nodes: NodeSlots, *, query_mask: Tensor | None = None,
        include_self: bool = False,
    ) -> RelationalCandidates:
        batch, node_count, _ = nodes.content.shape
        if query_mask is None:
            query_mask = nodes.active
        if query_mask.shape != nodes.active.shape or query_mask.dtype != torch.bool:
            raise ValueError("query_mask must be boolean with node-ring shape")
        query = F.normalize(self.query(nodes.content), dim=-1)
        key = F.normalize(self.key(nodes.content), dim=-1)
        score = torch.einsum("bqd,bkd->bqk", query, key)
        query_completion = nodes.support[:, :, 2, None]
        candidate_completion = nodes.support[:, None, :, 2]
        causal = candidate_completion <= query_completion
        same_scenario = nodes.scenario_ids[:, :, None] == nodes.scenario_ids[:, None, :]
        valid = query_mask[:, :, None] & nodes.active[:, None, :] & causal & same_scenario
        if not include_self:
            diagonal = torch.eye(node_count, dtype=torch.bool, device=nodes.content.device)[None]
            valid = valid & ~diagonal
        age = (query_completion - candidate_completion).clamp_min(0)
        score = score - F.softplus(self.recency_weight) * torch.log1p(age)
        score = score.masked_fill(~valid, -torch.inf)
        count = min(self.maximum_candidates, node_count)
        selected_score, indices = score.topk(count, dim=-1)
        mask = torch.isfinite(selected_score)
        # Fill invalid indices with zero for safe gathers; the mask remains authority.
        indices = indices.masked_fill(~mask, 0)
        return RelationalCandidates(
            self._gather(nodes.content, indices),
            self._gather(nodes.spectral, indices),
            self._gather(nodes.type_logits, indices),
            self._gather(nodes.support, indices),
            self._gather(nodes.modality_presence, indices),
            self._gather(nodes.uncertainty, indices),
            self._gather(nodes.provenance_features, indices),
            self._gather(nodes.source_classes, indices),
            self._gather(nodes.scenario_ids, indices),
            indices,
            torch.zeros_like(indices),
            mask,
        )


class FactorizedTypedValue(nn.Module):
    """Shared value projection plus bounded low-rank relation adapters."""

    def __init__(self, width: int, relation_count: int, rank: int) -> None:
        super().__init__()
        if min(width, relation_count, rank) <= 0:
            raise ValueError("typed value dimensions must be positive")
        self.base = nn.Linear(width, width, bias=False)
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        self.relation_scale = nn.Parameter(torch.empty(relation_count, rank))
        nn.init.normal_(self.relation_scale, std=rank ** -0.5)

    def aggregate(self, values: Tensor, weights: Tensor) -> Tensor:
        """Aggregate before expanding adapters to avoid a ``K*R*D`` tensor."""

        if values.ndim != 4 or weights.shape[:3] != values.shape[:3]:
            raise ValueError("typed value aggregation received incompatible tensors")
        base = self.base(values)
        low = self.down(values)
        base_message = torch.einsum("bqkr,bqkd->bqrd", weights, base)
        low_message = torch.einsum("bqkr,bqka->bqra", weights, low)
        low_message = low_message * self.relation_scale[None, None]
        return base_message + self.up(low_message)


@dataclass(frozen=True, slots=True)
class RelationalRouterOutput:
    update: Tensor
    relation_messages: Tensor
    relation_posterior: Tensor
    candidate_posterior: Tensor
    joint_posterior: Tensor
    selected_node_indices: Tensor
    selected_relation_families: Tensor
    selected_scores: Tensor
    selected_mask: Tensor
    phase_coherence: Tensor


class RelationalResonanceRouter(nn.Module):
    """Jointly route participants and typed relations under a hard candidate cap."""

    def __init__(
        self, width: int, heads: int, modes: int, relation_count: int,
        uncertainty_channels: int, provenance_features: int, *,
        adapter_rank: int = 8, retained_edges: int = 8,
    ) -> None:
        super().__init__()
        if min(
            width, heads, modes, relation_count, uncertainty_channels,
            provenance_features, adapter_rank, retained_edges,
        ) <= 0:
            raise ValueError("relational router dimensions must be positive")
        if relation_count != len(RelationFamily):
            raise ValueError("router relation count must exactly match the controlled ontology")
        self.width = width
        self.heads = heads
        self.modes = modes
        self.relation_count = relation_count
        self.retained_edges = retained_edges
        pair_dim = 4 * width + 3 + 2 * uncertainty_channels + 2 * provenance_features + 4
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.shared_pair = nn.Sequential(
            nn.Linear(pair_dim, width), nn.SiLU(), nn.Linear(width, adapter_rank),
        )
        self.relation_score_scale = nn.Parameter(torch.empty(relation_count, adapter_rank))
        self.relation_score_bias = nn.Parameter(torch.zeros(relation_count))
        self.type_bias = nn.Parameter(torch.zeros(relation_count))
        self.goal_score = nn.Linear(width, relation_count, bias=False)
        self.source_gate = nn.Embedding(8, relation_count)
        self.frequency_raw = nn.Parameter(torch.zeros(relation_count, heads, modes))
        self.coherence_scale_raw = nn.Parameter(torch.zeros(relation_count))
        self.delay_penalty_raw = nn.Parameter(torch.tensor(-2.0))
        self.scale_penalty_raw = nn.Parameter(torch.tensor(-2.0))
        self.value = FactorizedTypedValue(width, relation_count, adapter_rank)
        self.relation_mix = nn.Linear(width, relation_count)
        self.output = nn.Linear(width, width)
        self.output_gain = nn.Parameter(torch.tensor(-4.0))
        nn.init.normal_(self.relation_score_scale, std=adapter_rank ** -0.5)
        nn.init.zeros_(self.source_gate.weight)

    def _coherence(self, query: Tensor, candidate: Tensor, delay: Tensor) -> Tensor:
        if query.shape[-3:] != (self.heads, self.modes, 2):
            raise ValueError("query spectral dimensions do not match the router")
        if candidate.shape[-3:] != (self.heads, self.modes, 2):
            raise ValueError("candidate spectral dimensions do not match the router")
        frequency = pi * torch.sigmoid(self.frequency_raw)
        angle = delay[..., None, None, None] * frequency[None, None, None]
        cosine, sine = torch.cos(angle), torch.sin(angle)
        candidate_real = candidate[..., 0][:, :, :, None] * cosine + candidate[..., 1][:, :, :, None] * sine
        candidate_imag = -candidate[..., 0][:, :, :, None] * sine + candidate[..., 1][:, :, :, None] * cosine
        query_real = query[..., 0][:, :, None, None]
        query_imag = query[..., 1][:, :, None, None]
        numerator = (query_real * candidate_real + query_imag * candidate_imag).sum((-2, -1))
        query_norm = query.square().sum((-1, -2, -3)).sqrt()[:, :, None, None]
        candidate_norm = candidate.square().sum((-1, -2, -3)).sqrt()[..., None]
        return numerator / (query_norm * candidate_norm).clamp_min(1e-6)

    def forward(
        self, queries: NodeSlots, candidates: RelationalCandidates,
        *, goal_context: Tensor | None = None,
    ) -> RelationalRouterOutput:
        batch, query_count, width = queries.content.shape
        if candidates.content.shape[:2] != (batch, query_count) or width != self.width:
            raise ValueError("queries and relational candidates are incompatible")
        candidate_count = candidates.content.shape[2]
        query_content = queries.content[:, :, None].expand(-1, -1, candidate_count, -1)
        delta = candidates.content - query_content
        product = candidates.content * query_content
        support_delta = candidates.support - queries.support[:, :, None]
        uncertainty = torch.cat((
            queries.uncertainty[:, :, None].expand(-1, -1, candidate_count, -1),
            candidates.uncertainty,
        ), -1)
        provenance = torch.cat((
            queries.provenance_features[:, :, None].expand(-1, -1, candidate_count, -1),
            candidates.provenance_features,
        ), -1)
        same_modality = (
            queries.modality_presence[:, :, None] * candidates.modality_presence
        ).sum(-1, keepdim=True).clamp_max(1)
        same_source = (
            queries.source_classes[:, :, None] == candidates.source_classes
        ).to(queries.content.dtype).unsqueeze(-1)
        same_scenario = (
            queries.scenario_ids[:, :, None] == candidates.scenario_ids
        ).to(queries.content.dtype).unsqueeze(-1)
        existing = torch.zeros_like(same_scenario)
        pair = torch.cat((
            query_content, candidates.content, delta, product, support_delta,
            uncertainty, provenance, same_modality, same_source, same_scenario, existing,
        ), -1)
        shared = self.shared_pair(self.pair_norm(pair))
        typed_score = torch.einsum("bqka,ra->bqkr", shared, self.relation_score_scale)
        delay = (queries.support[:, :, None, 2] - candidates.support[..., 2]).clamp_min(0)
        coherence = self._coherence(queries.spectral, candidates.spectral, delay)
        score = typed_score + self.relation_score_bias + self.type_bias
        score = score + F.softplus(self.coherence_scale_raw)[None, None, None] * coherence
        score = score - F.softplus(self.delay_penalty_raw) * torch.log1p(delay)[..., None]
        # Physical scale is not currently an authoritative node field.  The term
        # is retained explicitly at zero until scale metadata is supplied.
        score = score - F.softplus(self.scale_penalty_raw) * 0
        source = candidates.source_classes.clamp(0, self.source_gate.num_embeddings - 1)
        score = score + self.source_gate(source)
        if goal_context is not None:
            if goal_context.shape != (batch, self.width):
                raise ValueError("goal_context must have shape (batch,width)")
            score = score + self.goal_score(goal_context)[:, None, None]
        query_types = queries.type_logits.argmax(-1)
        candidate_types = candidates.type_logits.argmax(-1)
        compatibility = RELATION_COMPATIBILITY.to(score.device)[
            query_types[:, :, None].expand(-1, -1, candidate_count), candidate_types
        ]
        valid = candidates.mask[..., None] & queries.active[:, :, None, None] & compatibility
        relation_posterior = _masked_softmax(score, valid, -1)
        candidate_posterior = _masked_softmax(score, valid, -2)
        joint = relation_posterior * candidate_posterior
        joint = joint * valid.to(joint.dtype)
        joint = joint / joint.sum((-2, -1), keepdim=True).clamp_min(
            torch.finfo(joint.dtype).tiny
        )
        messages = self.value.aggregate(candidates.content, joint)
        relation_mix = torch.softmax(self.relation_mix(queries.content), -1)
        update = self.output((messages * relation_mix[..., None]).sum(-2))
        update = update * torch.sigmoid(self.output_gain) * queries.active.unsqueeze(-1)
        flat = joint.flatten(-2)
        retained = min(self.retained_edges, flat.shape[-1])
        selected_scores, flat_indices = flat.topk(retained, -1)
        selected_mask = selected_scores > 0
        selected_candidate = torch.div(flat_indices, self.relation_count, rounding_mode="floor")
        selected_relation = flat_indices.remainder(self.relation_count)
        selected_nodes = torch.gather(candidates.node_indices, -1, selected_candidate)
        selected_nodes = selected_nodes.masked_fill(~selected_mask, -1)
        selected_relation = selected_relation.masked_fill(~selected_mask, -1)
        return RelationalRouterOutput(
            update, messages, relation_posterior, candidate_posterior, joint,
            selected_nodes, selected_relation, selected_scores, selected_mask, coherence,
        )
