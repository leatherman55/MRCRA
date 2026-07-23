"""Batched causal episodic/semantic tensor memory with controlled spread."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum
from math import pi
from typing import TypeVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cognitive_types import SourceClass
from .provenance import ProvenanceLedger
from .runtime_validation import runtime_validation_enabled


class MemoryTier(IntEnum):
    EPISODIC = 0
    SEMANTIC = 1


T = TypeVar("T")


def _tensor_map(instance: T, method: str, *args, **kwargs) -> T:
    values = {}
    for field in fields(instance):
        value = getattr(instance, field.name)
        values[field.name] = getattr(value, method)(*args, **kwargs) if isinstance(value, Tensor) else value
    return type(instance)(**values)


@dataclass(frozen=True, slots=True)
class TensorMemoryState:
    keys: Tensor
    values: Tensor
    signatures: Tensor
    spectral: Tensor
    support: Tensor
    type_ids: Tensor
    provenance_ids: Tensor
    source_classes: Tensor
    scenario_ids: Tensor
    uncertainty: Tensor
    consequence: Tensor
    utility: Tensor
    use_count: Tensor
    last_access: Tensor
    versions: Tensor
    association_indices: Tensor
    association_families: Tensor
    association_weights: Tensor
    association_mask: Tensor
    active: Tensor
    clock: int

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.keys.ndim != 3 or self.values.ndim != 3 or self.signatures.ndim != 3:
            raise ValueError("memory keys, values, and signatures must be (batch,capacity,width)")
        base = self.keys.shape[:2]
        if self.values.shape[:2] != base or self.signatures.shape[:2] != base:
            raise ValueError("memory tensors must share batch and capacity")
        if self.spectral.ndim != 5 or self.spectral.shape[:2] != base or self.spectral.shape[-1] != 2:
            raise ValueError("memory spectral state must be (batch,capacity,heads,modes,2)")
        if self.support.shape != (*base, 3):
            raise ValueError("memory support must contain start, end, completion")
        for name in (
            "type_ids", "provenance_ids", "source_classes", "scenario_ids",
            "use_count", "last_access", "versions",
        ):
            value = getattr(self, name)
            if value.shape != base or value.dtype != torch.int64:
                raise ValueError(f"memory {name} must be int64 with capacity shape")
        if self.uncertainty.ndim != 3 or self.uncertainty.shape[:2] != base:
            raise ValueError("memory uncertainty has invalid shape")
        if self.consequence.ndim != 3 or self.consequence.shape[:2] != base:
            raise ValueError("memory consequence has invalid shape")
        if self.utility.shape != base:
            raise ValueError("memory utility has invalid shape")
        association_shape = self.association_indices.shape
        if len(association_shape) != 3 or association_shape[:2] != base:
            raise ValueError("memory associations must be (batch,capacity,degree)")
        for name in ("association_families",):
            value = getattr(self, name)
            if value.shape != association_shape or value.dtype != torch.int64:
                raise ValueError(f"{name} must be int64 with association shape")
        if self.association_weights.shape != association_shape:
            raise ValueError("association weights must match association pointers")
        if self.association_mask.shape != association_shape or self.association_mask.dtype != torch.bool:
            raise ValueError("association mask must be boolean and match pointers")
        if self.active.shape != base or self.active.dtype != torch.bool:
            raise ValueError("memory active mask is invalid")
        if not torch.equal(self.active, self.provenance_ids >= 0):
            raise ValueError("only active memory rows may have provenance IDs")
        if bool((self.association_mask & (self.association_indices < 0)).any()):
            raise ValueError("active association pointers cannot be negative")
        if self.clock < 0:
            raise ValueError("memory clock cannot be negative")

    @classmethod
    def empty(
        cls, batch: int, capacity: int, key_dim: int, value_dim: int,
        signature_dim: int, *, heads: int, modes: int, uncertainty_channels: int,
        consequence_dim: int, association_degree: int, device=None, dtype=None,
    ) -> "TensorMemoryState":
        if min(
            batch, capacity, key_dim, value_dim, signature_dim, heads, modes,
            uncertainty_channels, consequence_dim, association_degree,
        ) <= 0:
            raise ValueError("all tensor-memory dimensions must be positive")
        options = dict(device=device, dtype=dtype)
        base = (batch, capacity)
        association_shape = (*base, association_degree)
        integer_empty = lambda: torch.full(base, -1, dtype=torch.int64, device=device)
        return cls(
            torch.zeros(*base, key_dim, **options),
            torch.zeros(*base, value_dim, **options),
            torch.zeros(*base, signature_dim, **options),
            torch.zeros(*base, heads, modes, 2, **options),
            torch.zeros(*base, 3, **options),
            integer_empty(), integer_empty(), integer_empty(), integer_empty(),
            torch.zeros(*base, uncertainty_channels, **options),
            torch.zeros(*base, consequence_dim, **options),
            torch.zeros(*base, **options),
            torch.zeros(base, dtype=torch.int64, device=device),
            torch.full(base, -1, dtype=torch.int64, device=device),
            torch.zeros(base, dtype=torch.int64, device=device),
            torch.full(association_shape, -1, dtype=torch.int64, device=device),
            torch.full(association_shape, -1, dtype=torch.int64, device=device),
            torch.zeros(*association_shape, **options),
            torch.zeros(association_shape, dtype=torch.bool, device=device),
            torch.zeros(base, dtype=torch.bool, device=device),
            0,
        )

    @property
    def batch(self) -> int:
        return self.keys.shape[0]

    @property
    def capacity(self) -> int:
        return self.keys.shape[1]

    def detach(self) -> "TensorMemoryState":
        return _tensor_map(self, "detach")

    def to(self, *args, **kwargs) -> "TensorMemoryState":
        return _tensor_map(self, "to", *args, **kwargs)


@dataclass(frozen=True, slots=True)
class MemoryWriteEvidence:
    innovation: Tensor
    prediction_error: Tensor
    boundary: Tensor
    novelty: Tensor
    goal_relevance: Tensor
    outcome_magnitude: Tensor
    epistemic_uncertainty: Tensor
    controllability: Tensor
    redundancy: Tensor
    irreducible_noise: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        base = self.innovation.shape
        if len(base) != 2:
            raise ValueError("memory write evidence must be (batch,candidates)")
        for name in (
            "prediction_error", "boundary", "novelty", "goal_relevance",
            "outcome_magnitude", "epistemic_uncertainty", "controllability",
            "redundancy", "irreducible_noise",
        ):
            if getattr(self, name).shape != base:
                raise ValueError(f"memory write evidence {name} has invalid shape")
        if self.mask.shape != base or self.mask.dtype != torch.bool:
            raise ValueError("memory write mask must be boolean with evidence shape")


class MemoryWritePolicyV2(nn.Module):
    """Learned write value with an explicit irreducible-noise penalty."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("memory write hidden width must be positive")
        self.network = nn.Sequential(nn.Linear(10, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, evidence: MemoryWriteEvidence) -> Tensor:
        features = torch.stack((
            evidence.innovation, evidence.prediction_error, evidence.boundary,
            evidence.novelty, evidence.goal_relevance, evidence.outcome_magnitude,
            evidence.epistemic_uncertainty, evidence.controllability,
            -evidence.redundancy, -evidence.irreducible_noise,
        ), -1)
        return self.network(features).squeeze(-1).masked_fill(~evidence.mask, -torch.inf)


@dataclass(frozen=True, slots=True)
class MemoryWriteBatch:
    keys: Tensor
    values: Tensor
    signatures: Tensor
    spectral: Tensor
    support: Tensor
    type_ids: Tensor
    provenance_ids: Tensor
    source_classes: Tensor
    scenario_ids: Tensor
    uncertainty: Tensor
    consequence: Tensor
    utility: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.keys.ndim != 3:
            raise ValueError("memory write keys must be (batch,candidates,key_dim)")
        base = self.keys.shape[:2]
        for name in ("values", "signatures", "uncertainty", "consequence"):
            value = getattr(self, name)
            if value.ndim != 3 or value.shape[:2] != base:
                raise ValueError(f"memory write {name} has invalid shape")
        if self.spectral.ndim != 5 or self.spectral.shape[:2] != base or self.spectral.shape[-1] != 2:
            raise ValueError("memory write spectral state has invalid shape")
        if self.support.shape != (*base, 3):
            raise ValueError("memory write support has invalid shape")
        for name in ("type_ids", "provenance_ids", "source_classes", "scenario_ids"):
            value = getattr(self, name)
            if value.shape != base or value.dtype != torch.int64:
                raise ValueError(f"memory write {name} must be int64 with candidate shape")
        if self.utility.shape != base:
            raise ValueError("memory write utility has invalid shape")
        if self.mask.shape != base or self.mask.dtype != torch.bool:
            raise ValueError("memory write mask is invalid")
        if bool((self.mask & (self.provenance_ids < 0)).any()):
            raise ValueError("active memory writes require provenance")


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    keys: Tensor
    signatures: Tensor
    spectral: Tensor
    timestamps: Tensor
    type_ids: Tensor
    source_classes: Tensor
    scenario_ids: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.keys.ndim != 3 or self.signatures.ndim != 3 or self.keys.shape[:2] != self.signatures.shape[:2]:
            raise ValueError("memory queries must be (batch,queries,width)")
        base = self.keys.shape[:2]
        if self.spectral.ndim != 5 or self.spectral.shape[:2] != base or self.spectral.shape[-1] != 2:
            raise ValueError("memory query spectral state has invalid shape")
        if self.timestamps.shape != base:
            raise ValueError("memory query timestamps have invalid shape")
        for name in ("type_ids", "source_classes", "scenario_ids"):
            value = getattr(self, name)
            if value.shape != base or value.dtype != torch.int64:
                raise ValueError(f"memory query {name} must be int64 with query shape")
        if self.mask.shape != base or self.mask.dtype != torch.bool:
            raise ValueError("memory query mask is invalid")


@dataclass(frozen=True, slots=True)
class MemoryRetrieval:
    values: Tensor
    indices: Tensor
    scores: Tensor
    provenance_ids: Tensor
    source_classes: Tensor
    support: Tensor
    uncertainty: Tensor
    mask: Tensor
    router_recall: Tensor
    oracle_indices: Tensor


@dataclass(frozen=True, slots=True)
class AssociativeExpansion:
    indices: Tensor
    scores: Tensor
    mask: Tensor
    depths: Tensor


class BatchedTensorMemory(nn.Module):
    """GPU-resident exact ring with signature routing and typed reranking."""

    def __init__(
        self, key_dim: int, signature_dim: int, heads: int, modes: int, *,
        route_candidates: int = 32, retrieved_items: int = 8,
        duplicate_threshold: float = 0.995,
    ) -> None:
        super().__init__()
        if min(key_dim, signature_dim, heads, modes, route_candidates, retrieved_items) <= 0:
            raise ValueError("tensor memory dimensions must be positive")
        if retrieved_items > route_candidates:
            raise ValueError("retrieved_items cannot exceed route_candidates")
        if not -1 < duplicate_threshold <= 1:
            raise ValueError("duplicate threshold must lie in (-1,1]")
        self.key_dim = key_dim
        self.signature_dim = signature_dim
        self.heads = heads
        self.modes = modes
        self.route_candidates = route_candidates
        self.retrieved_items = retrieved_items
        self.duplicate_threshold = duplicate_threshold
        self.frequency_raw = nn.Parameter(torch.zeros(heads, modes))
        self.key_weight_raw = nn.Parameter(torch.tensor(1.0))
        self.signature_weight_raw = nn.Parameter(torch.tensor(1.0))
        self.phase_weight_raw = nn.Parameter(torch.tensor(0.0))
        self.type_bonus = nn.Parameter(torch.tensor(0.1))
        self.source_bonus = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _gather(value: Tensor, indices: Tensor) -> Tensor:
        batch, queries, count = indices.shape
        tail = value.shape[2:]
        expanded = value[:, None].expand(batch, queries, *value.shape[1:])
        gather_index = indices.clamp_min(0).reshape(batch, queries, count, *([1] * len(tail)))
        return torch.gather(expanded, 2, gather_index.expand(batch, queries, count, *tail))

    def _phase_coherence(self, query: Tensor, memory: Tensor, delay: Tensor) -> Tensor:
        frequency = pi * torch.sigmoid(self.frequency_raw)
        angle = delay[..., None, None] * frequency
        cosine, sine = torch.cos(angle), torch.sin(angle)
        real = memory[..., 0] * cosine + memory[..., 1] * sine
        imag = -memory[..., 0] * sine + memory[..., 1] * cosine
        q_real, q_imag = query[..., 0][:, :, None], query[..., 1][:, :, None]
        numerator = (q_real * real + q_imag * imag).sum((-2, -1))
        q_norm = query.square().sum((-1, -2, -3)).sqrt()[:, :, None]
        m_norm = memory.square().sum((-1, -2, -3)).sqrt()
        return numerator / (q_norm * m_norm).clamp_min(1e-6)

    def retrieve(
        self, state: TensorMemoryState, query: MemoryQuery, *, compute_oracle: bool = False,
    ) -> MemoryRetrieval:
        if state.batch != query.keys.shape[0] or state.keys.shape[-1] != self.key_dim:
            raise ValueError("memory state and query dimensions are incompatible")
        if state.signatures.shape[-1] != self.signature_dim:
            raise ValueError("memory state signature dimension is incompatible")
        if state.spectral.shape[-3:-1] != (self.heads, self.modes):
            raise ValueError("memory spectral dimensions are incompatible")
        q_signature = F.normalize(query.signatures, dim=-1)
        signatures = F.normalize(state.signatures, dim=-1)
        route_score = torch.einsum("bqd,bmd->bqm", q_signature, signatures)
        causal = state.support[:, None, :, 2] <= query.timestamps[:, :, None]
        scenario = (
            (state.scenario_ids[:, None, :] == 0)
            | (state.scenario_ids[:, None, :] == query.scenario_ids[:, :, None])
        )
        valid = query.mask[:, :, None] & state.active[:, None, :] & causal & scenario
        route_score = route_score.masked_fill(~valid, -torch.inf)
        route_count = min(self.route_candidates, state.capacity)
        routed_score, routed_indices = route_score.topk(route_count, -1)
        routed_mask = torch.isfinite(routed_score)
        routed_indices = routed_indices.masked_fill(~routed_mask, 0)
        routed_keys = self._gather(state.keys, routed_indices)
        routed_signatures = self._gather(state.signatures, routed_indices)
        routed_spectral = self._gather(state.spectral, routed_indices)
        routed_support = self._gather(state.support, routed_indices)
        routed_types = self._gather(state.type_ids, routed_indices)
        routed_sources = self._gather(state.source_classes, routed_indices)
        key_score = F.cosine_similarity(query.keys[:, :, None], routed_keys, dim=-1)
        signature_score = F.cosine_similarity(query.signatures[:, :, None], routed_signatures, dim=-1)
        delay = (query.timestamps[:, :, None] - routed_support[..., 2]).clamp_min(0)
        phase_score = self._phase_coherence(query.spectral, routed_spectral, delay)
        exact = (
            F.softplus(self.key_weight_raw) * key_score
            + F.softplus(self.signature_weight_raw) * signature_score
            + F.softplus(self.phase_weight_raw) * phase_score
            + self.type_bonus * (routed_types == query.type_ids[:, :, None])
            + self.source_bonus * (routed_sources == query.source_classes[:, :, None])
        ).masked_fill(~routed_mask, -torch.inf)
        count = min(self.retrieved_items, route_count)
        scores, rerank = exact.topk(count, -1)
        indices = torch.gather(routed_indices, -1, rerank)
        mask = torch.isfinite(scores)
        indices = indices.masked_fill(~mask, 0)
        # Full exact oracle is opt-in validation telemetry, never the production
        # retrieval path, because it materializes query-by-capacity score rows.
        if compute_oracle:
            all_key_score = F.cosine_similarity(query.keys[:, :, None], state.keys[:, None], dim=-1)
            all_delay = (query.timestamps[:, :, None] - state.support[:, None, :, 2]).clamp_min(0)
            all_spectral = state.spectral[:, None].expand(-1, query.keys.shape[1], -1, -1, -1, -1)
            all_phase = self._phase_coherence(query.spectral, all_spectral, all_delay)
            oracle_score = (
                F.softplus(self.key_weight_raw) * all_key_score
                + F.softplus(self.signature_weight_raw) * route_score.masked_fill(~valid, 0)
                + F.softplus(self.phase_weight_raw) * all_phase
                + self.type_bonus * (state.type_ids[:, None] == query.type_ids[:, :, None])
                + self.source_bonus * (state.source_classes[:, None] == query.source_classes[:, :, None])
            ).masked_fill(~valid, -torch.inf)
            oracle_indices = oracle_score.argmax(-1).masked_fill(~valid.any(-1), -1)
            recall = (
                (routed_indices == oracle_indices[..., None]) & routed_mask
            ).any(-1).to(query.keys.dtype)
        else:
            oracle_indices = torch.full(
                query.mask.shape, -1, dtype=torch.int64, device=query.keys.device
            )
            recall = torch.full_like(query.timestamps, -1)
        provenance = self._gather(state.provenance_ids, indices)
        sources = self._gather(state.source_classes, indices)
        support = self._gather(state.support, indices)
        uncertainty = self._gather(state.uncertainty, indices)
        values = self._gather(state.values, indices) * mask.unsqueeze(-1)
        return MemoryRetrieval(
            values, indices.masked_fill(~mask, -1), scores.masked_fill(~mask, 0),
            provenance.masked_fill(~mask, -1), sources.masked_fill(~mask, -1),
            support * mask.unsqueeze(-1), uncertainty * mask.unsqueeze(-1), mask,
            recall, oracle_indices,
        )

    def write(
        self, state: TensorMemoryState, batch: MemoryWriteBatch, scores: Tensor,
        *, quota: int, tier: MemoryTier, ledger: ProvenanceLedger | None = None,
    ) -> TensorMemoryState:
        if quota <= 0 or scores.shape != batch.mask.shape:
            raise ValueError("memory write quota and score shape are invalid")
        if state.batch != batch.keys.shape[0] or state.keys.shape[-1] != batch.keys.shape[-1]:
            raise ValueError("memory write batch is incompatible with state")
        if tier == MemoryTier.SEMANTIC and ledger is None:
            raise ValueError("semantic writes require the authoritative provenance ledger")
        valid_scores = scores.masked_fill(~batch.mask, -torch.inf)
        selected_count = min(quota, batch.mask.shape[-1])
        selected_scores, selected = valid_scores.topk(selected_count, -1)
        selected_mask = torch.isfinite(selected_scores)
        values = {name: getattr(state, name).clone() if isinstance(getattr(state, name), Tensor) else getattr(state, name) for name in state.__dataclass_fields__}
        for batch_index in range(state.batch):
            for rank in torch.nonzero(selected_mask[batch_index], as_tuple=False).flatten().tolist():
                item = int(selected[batch_index, rank].item())
                provenance_id = int(batch.provenance_ids[batch_index, item].item())
                source_class = SourceClass(int(batch.source_classes[batch_index, item].item()))
                if tier == MemoryTier.SEMANTIC:
                    if source_class in (
                        SourceClass.SIMULATED, SourceClass.PREDICTED,
                        SourceClass.RECONSTRUCTED,
                    ):
                        raise ValueError(
                            "simulated, predicted, or reconstructed-only items cannot enter semantic memory"
                        )
                    if not ledger.can_consolidate(provenance_id):
                        raise ValueError("semantic write lacks verified consolidation authority")
                normalized = F.normalize(batch.signatures[batch_index, item], dim=-1)
                similarity = F.cosine_similarity(
                    normalized[None], F.normalize(values["signatures"][batch_index], dim=-1), dim=-1,
                )
                duplicate = (
                    values["active"][batch_index]
                    & (similarity >= self.duplicate_threshold)
                    & (values["type_ids"][batch_index] == batch.type_ids[batch_index, item])
                    & (values["scenario_ids"][batch_index] == batch.scenario_ids[batch_index, item])
                )
                matches = torch.nonzero(duplicate, as_tuple=False).flatten()
                if matches.numel():
                    slot = int(matches[similarity[matches].argmax()].item())
                else:
                    free = torch.nonzero(~values["active"][batch_index], as_tuple=False).flatten()
                    if free.numel():
                        slot = int(free[0].item())
                    else:
                        age = (state.clock - values["last_access"][batch_index]).clamp_min(0).to(state.keys.dtype)
                        retention = (
                            values["utility"][batch_index]
                            + 0.05 * torch.log1p(values["use_count"][batch_index].to(state.keys.dtype))
                            - 0.001 * age
                        )
                        slot = int(retention.argmin().item())
                for name in ("keys", "values", "signatures", "spectral", "support", "uncertainty", "consequence"):
                    values[name][batch_index, slot] = getattr(batch, name)[batch_index, item].detach()
                for name in ("type_ids", "provenance_ids", "source_classes", "scenario_ids"):
                    values[name][batch_index, slot] = getattr(batch, name)[batch_index, item]
                values["utility"][batch_index, slot] = batch.utility[batch_index, item].detach()
                values["use_count"][batch_index, slot] = 0
                values["last_access"][batch_index, slot] = state.clock
                values["versions"][batch_index, slot] += 1
                values["association_indices"][batch_index, slot].fill_(-1)
                values["association_families"][batch_index, slot].fill_(-1)
                values["association_weights"][batch_index, slot].zero_()
                values["association_mask"][batch_index, slot].zero_()
                values["active"][batch_index, slot] = True
        values["clock"] = state.clock + 1
        return TensorMemoryState(**values)

    @staticmethod
    def mark_accessed(state: TensorMemoryState, retrieval: MemoryRetrieval) -> TensorMemoryState:
        values = {name: getattr(state, name).clone() if isinstance(getattr(state, name), Tensor) else getattr(state, name) for name in state.__dataclass_fields__}
        for batch_index in range(state.batch):
            indices = retrieval.indices[batch_index][retrieval.mask[batch_index]].unique()
            values["use_count"][batch_index, indices] += 1
            values["last_access"][batch_index, indices] = state.clock
        return TensorMemoryState(**values)

    @staticmethod
    def link_associations(
        state: TensorMemoryState, source_indices: Tensor, target_indices: Tensor,
        family_ids: Tensor, weights: Tensor, mask: Tensor,
    ) -> TensorMemoryState:
        """Insert directed typed links under the fixed per-record degree budget."""

        shape = source_indices.shape
        if not all(value.shape == shape for value in (target_indices, family_ids, weights, mask)):
            raise ValueError("association link tensors must share shape")
        if source_indices.ndim != 2 or source_indices.shape[0] != state.batch or mask.dtype != torch.bool:
            raise ValueError("association links must have shape (batch,links) and a boolean mask")
        if any(value.dtype != torch.int64 for value in (source_indices, target_indices, family_ids)):
            raise ValueError("association source, target, and family must be int64")
        values = {name: getattr(state, name).clone() if isinstance(getattr(state, name), Tensor) else getattr(state, name) for name in state.__dataclass_fields__}
        for batch_index in range(state.batch):
            for link in torch.nonzero(mask[batch_index], as_tuple=False).flatten().tolist():
                source = int(source_indices[batch_index, link].item())
                target = int(target_indices[batch_index, link].item())
                if not 0 <= source < state.capacity or not 0 <= target < state.capacity:
                    raise ValueError("association pointer lies outside memory capacity")
                if not bool(state.active[batch_index, source] & state.active[batch_index, target]):
                    raise ValueError("associations may only link active records")
                existing = (
                    values["association_mask"][batch_index, source]
                    & (values["association_indices"][batch_index, source] == target)
                    & (values["association_families"][batch_index, source] == family_ids[batch_index, link])
                )
                found = torch.nonzero(existing, as_tuple=False).flatten()
                if found.numel():
                    slot = int(found[0].item())
                else:
                    free = torch.nonzero(
                        ~values["association_mask"][batch_index, source], as_tuple=False
                    ).flatten()
                    slot = int(free[0].item()) if free.numel() else int(
                        values["association_weights"][batch_index, source].argmin().item()
                    )
                values["association_indices"][batch_index, source, slot] = target
                values["association_families"][batch_index, source, slot] = family_ids[batch_index, link]
                values["association_weights"][batch_index, source, slot] = weights[batch_index, link]
                values["association_mask"][batch_index, source, slot] = True
        return TensorMemoryState(**values)

    @staticmethod
    def expand_associations(
        state: TensorMemoryState, seed_indices: Tensor, seed_mask: Tensor, *,
        maximum_depth: int, budget: int,
    ) -> AssociativeExpansion:
        if seed_indices.shape != seed_mask.shape or seed_indices.ndim != 2 or seed_mask.dtype != torch.bool:
            raise ValueError("association seeds require matching (batch,seeds) tensors")
        if seed_indices.shape[0] != state.batch or min(maximum_depth, budget) <= 0:
            raise ValueError("association expansion controls are invalid")
        indices = torch.full((state.batch, budget), -1, dtype=torch.int64, device=state.keys.device)
        scores = torch.zeros(state.batch, budget, device=state.keys.device, dtype=state.keys.dtype)
        mask = torch.zeros(state.batch, budget, dtype=torch.bool, device=state.keys.device)
        depths = torch.zeros(state.batch, budget, dtype=torch.int64, device=state.keys.device)
        for batch_index in range(state.batch):
            visited: set[int] = set()
            frontier = [
                (int(index), 1.0, 0)
                for index in seed_indices[batch_index, seed_mask[batch_index]].tolist()
                if 0 <= int(index) < state.capacity and bool(state.active[batch_index, int(index)])
            ]
            emitted = 0
            while frontier and emitted < budget:
                frontier.sort(key=lambda item: (-item[1], item[2], item[0]))
                current, score, depth = frontier.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                indices[batch_index, emitted] = current
                scores[batch_index, emitted] = score
                depths[batch_index, emitted] = depth
                mask[batch_index, emitted] = True
                emitted += 1
                if depth >= maximum_depth:
                    continue
                neighbors = state.association_indices[batch_index, current]
                neighbor_scores = state.association_weights[batch_index, current]
                neighbor_mask = state.association_mask[batch_index, current]
                for neighbor, weight in zip(
                    neighbors[neighbor_mask].tolist(), neighbor_scores[neighbor_mask].tolist(), strict=True
                ):
                    if neighbor not in visited and 0 <= neighbor < state.capacity:
                        frontier.append((int(neighbor), score * float(torch.sigmoid(torch.tensor(weight))), depth + 1))
        return AssociativeExpansion(indices, scores, mask, depths)
