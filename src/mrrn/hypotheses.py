"""Factorized copy-on-write hypothesis bank with stable evidence updates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .runtime_validation import runtime_validation_enabled


@dataclass(frozen=True, slots=True)
class RoutedHypotheses:
    indices: Tensor
    mask: Tensor
    posterior_mass: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.indices.ndim != 2 or self.indices.dtype != torch.int64:
            raise ValueError("routed hypothesis indices must be int64 rows")
        if self.mask.shape != self.indices.shape or self.mask.dtype != torch.bool:
            raise ValueError("routed hypothesis mask is invalid")
        if self.posterior_mass.shape != (self.indices.shape[0],):
            raise ValueError("routed hypothesis posterior mass must be per batch")


@dataclass(frozen=True, slots=True)
class HypothesisState:
    residuals: Tensor
    relation_overrides: Tensor
    log_weights: Tensor
    predicted_outcomes: Tensor
    uncertainty: Tensor
    supporting_evidence: Tensor
    contradicting_evidence: Tensor
    latest_supporting_provenance_ids: Tensor
    latest_contradicting_provenance_ids: Tensor
    unknown: Tensor
    scenario_ids: Tensor
    weak_steps: Tensor
    versions: Tensor
    active: Tensor
    next_scenario_id: int

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.residuals.ndim != 3:
            raise ValueError("hypothesis residuals must be (batch,hypotheses,width)")
        base = self.residuals.shape[:2]
        if self.relation_overrides.ndim != 4 or self.relation_overrides.shape[:2] != base:
            raise ValueError("relation overrides must be (batch,hypotheses,relations,families)")
        for name in ("predicted_outcomes", "uncertainty"):
            value = getattr(self, name)
            if value.ndim != 3 or value.shape[:2] != base:
                raise ValueError(f"hypothesis {name} has invalid shape")
        for name in ("log_weights", "supporting_evidence", "contradicting_evidence"):
            if getattr(self, name).shape != base:
                raise ValueError(f"hypothesis {name} has invalid shape")
        for name in (
            "scenario_ids", "weak_steps", "versions",
            "latest_supporting_provenance_ids",
            "latest_contradicting_provenance_ids",
        ):
            value = getattr(self, name)
            if value.shape != base or value.dtype != torch.int64:
                raise ValueError(f"hypothesis {name} must be int64 with slot shape")
        if self.active.shape != base or self.active.dtype != torch.bool:
            raise ValueError("hypothesis active mask is invalid")
        if self.unknown.shape != base or self.unknown.dtype != torch.bool:
            raise ValueError("hypothesis unknown mask is invalid")
        if bool((self.unknown & ~self.active).any()):
            raise ValueError("unknown hypothesis slots must be active")
        if not torch.equal(self.active, self.scenario_ids > 0):
            raise ValueError("active hypotheses require unique positive scenario IDs")
        if self.next_scenario_id <= 0:
            raise ValueError("next scenario ID must be positive")
        active_ids = self.scenario_ids[self.active].tolist()
        if len(active_ids) != len(set(active_ids)):
            raise ValueError("hypothesis scenario IDs must be unique")

    @property
    def batch(self) -> int:
        return self.residuals.shape[0]

    @property
    def capacity(self) -> int:
        return self.residuals.shape[1]

    @property
    def weights(self) -> Tensor:
        logits = self.log_weights.masked_fill(~self.active, -torch.inf)
        maximum = logits.amax(-1, keepdim=True)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        exponential = torch.exp(logits - maximum) * self.active
        return exponential / exponential.sum(-1, keepdim=True).clamp_min(1e-8)

    @property
    def effective_count(self) -> Tensor:
        weights = self.weights.clamp_min(1e-8)
        return torch.exp(-(weights * weights.log() * self.active).sum(-1))

    def detach(self) -> "HypothesisState":
        return HypothesisState(
            self.residuals.detach(), self.relation_overrides.detach(), self.log_weights.detach(),
            self.predicted_outcomes.detach(), self.uncertainty.detach(),
            self.supporting_evidence.detach(), self.contradicting_evidence.detach(),
            self.latest_supporting_provenance_ids.detach(),
            self.latest_contradicting_provenance_ids.detach(), self.unknown.detach(),
            self.scenario_ids.detach(), self.weak_steps.detach(), self.versions.detach(),
            self.active.detach(), self.next_scenario_id,
        )


class HypothesisBank(nn.Module):
    """Maintain alternatives as residual state rather than backbone copies."""

    def __init__(
        self, width: int, capacity: int, relation_slots: int, relation_families: int,
        outcome_dim: int, uncertainty_channels: int, *, prune_threshold: float = 0.02,
        prune_hysteresis: int = 3, merge_similarity: float = 0.995,
    ) -> None:
        super().__init__()
        if min(
            width, capacity, relation_slots, relation_families,
            outcome_dim, uncertainty_channels, prune_hysteresis,
        ) <= 0:
            raise ValueError("hypothesis bank dimensions must be positive")
        if not 0 <= prune_threshold < 1 or not -1 < merge_similarity <= 1:
            raise ValueError("hypothesis thresholds are invalid")
        self.width = width
        self.capacity = capacity
        self.relation_slots = relation_slots
        self.relation_families = relation_families
        self.outcome_dim = outcome_dim
        self.uncertainty_channels = uncertainty_channels
        self.prune_threshold = prune_threshold
        self.prune_hysteresis = prune_hysteresis
        self.merge_similarity = merge_similarity
        self.proposal = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.outcome = nn.Linear(width, outcome_dim)
        self.uncertainty_head = nn.Linear(width, uncertainty_channels)

    def initial_state(self, batch: int, *, device=None, dtype=None) -> HypothesisState:
        if batch <= 0:
            raise ValueError("hypothesis batch must be positive")
        options = dict(device=device, dtype=dtype)
        base = (batch, self.capacity)
        return HypothesisState(
            torch.zeros(*base, self.width, **options),
            torch.zeros(*base, self.relation_slots, self.relation_families, **options),
            torch.full(base, -torch.inf, **options),
            torch.zeros(*base, self.outcome_dim, **options),
            torch.zeros(*base, self.uncertainty_channels, **options),
            torch.zeros(*base, **options), torch.zeros(*base, **options),
            torch.full(base, -1, dtype=torch.int64, device=device),
            torch.full(base, -1, dtype=torch.int64, device=device),
            torch.zeros(base, dtype=torch.bool, device=device),
            torch.zeros(base, dtype=torch.int64, device=device),
            torch.zeros(base, dtype=torch.int64, device=device),
            torch.zeros(base, dtype=torch.int64, device=device),
            torch.zeros(base, dtype=torch.bool, device=device), 1,
        )

    @staticmethod
    def route(
        state: HypothesisState, top_k: int, *, diversity_penalty: float = 0.2,
    ) -> RoutedHypotheses:
        """Bound deliberation with posterior-aware diverse routing.

        The explicit unknown alternative is retained whenever it is active, so
        high-confidence named hypotheses cannot erase open-world uncertainty.
        """

        if not 0 < top_k <= state.capacity or diversity_penalty < 0:
            raise ValueError("hypothesis routing controls are invalid")
        indices = torch.full(
            (state.batch, top_k), -1, dtype=torch.int64,
            device=state.residuals.device,
        )
        mask = torch.zeros_like(indices, dtype=torch.bool)
        mass = state.log_weights.new_zeros(state.batch)
        normalized = F.normalize(state.residuals, dim=-1)
        for row in range(state.batch):
            active = torch.nonzero(state.active[row], as_tuple=False).flatten().tolist()
            if not active:
                continue
            selected: list[int] = []
            unknown = torch.nonzero(
                state.unknown[row] & state.active[row], as_tuple=False
            ).flatten().tolist()
            if unknown:
                selected.append(max(
                    unknown,
                    key=lambda slot: float(state.weights[row, slot].detach()),
                ))
            while len(selected) < min(top_k, len(active)):
                candidates = [slot for slot in active if slot not in selected]
                def score(slot: int) -> float:
                    redundancy = 0.0 if not selected else max(
                        float((normalized[row, slot] * normalized[row, other]).sum().detach())
                        for other in selected
                    )
                    return float(state.weights[row, slot].detach()) - diversity_penalty * max(0.0, redundancy)
                selected.append(max(candidates, key=score))
            count = len(selected)
            indices[row, :count] = torch.tensor(
                selected, dtype=torch.int64, device=indices.device
            )
            mask[row, :count] = True
            mass[row] = state.weights[row, selected].sum()
        return RoutedHypotheses(indices, mask, mass)

    def create(
        self, state: HypothesisState, context: Tensor, create_mask: Tensor,
        *, relation_overrides: Tensor | None = None,
    ) -> HypothesisState:
        if context.shape != (state.batch, self.width) or create_mask.shape != (state.batch,) or create_mask.dtype != torch.bool:
            raise ValueError("hypothesis creation context and mask are incompatible")
        if relation_overrides is not None and relation_overrides.shape != (
            state.batch, self.relation_slots, self.relation_families
        ):
            raise ValueError("hypothesis relation overrides have invalid shape")
        values = {
            name: getattr(state, name).clone() if isinstance(getattr(state, name), Tensor) else getattr(state, name)
            for name in state.__dataclass_fields__
        }
        next_id = state.next_scenario_id
        residual = self.proposal(context)
        for batch_index in torch.nonzero(create_mask, as_tuple=False).flatten().tolist():
            was_empty = not bool(state.active[batch_index].any())
            free = torch.nonzero(~state.active[batch_index], as_tuple=False).flatten()
            if free.numel():
                slot = int(free[0].item())
            else:
                slot = int(state.weights[batch_index].argmin().item())
            values["residuals"][batch_index, slot] = residual[batch_index]
            values["relation_overrides"][batch_index, slot].zero_()
            if relation_overrides is not None:
                values["relation_overrides"][batch_index, slot] = relation_overrides[batch_index]
            values["predicted_outcomes"][batch_index, slot] = self.outcome(residual[batch_index])
            values["uncertainty"][batch_index, slot] = F.softplus(self.uncertainty_head(residual[batch_index]))
            values["supporting_evidence"][batch_index, slot] = 0
            values["contradicting_evidence"][batch_index, slot] = 0
            values["latest_supporting_provenance_ids"][batch_index, slot] = -1
            values["latest_contradicting_provenance_ids"][batch_index, slot] = -1
            values["unknown"][batch_index, slot] = was_empty
            values["scenario_ids"][batch_index, slot] = next_id
            values["weak_steps"][batch_index, slot] = 0
            values["versions"][batch_index, slot] += 1
            values["active"][batch_index, slot] = True
            active_count = int(values["active"][batch_index].sum().item())
            values["log_weights"][batch_index, values["active"][batch_index]] = -torch.log(
                context.new_tensor(float(active_count))
            )
            next_id += 1
        values["next_scenario_id"] = next_id
        return HypothesisState(**values)

    def update_evidence(
        self, state: HypothesisState, log_likelihood: Tensor,
        support: Tensor | None = None, contradiction: Tensor | None = None,
        provenance_ids: Tensor | None = None,
    ) -> HypothesisState:
        if log_likelihood.shape != state.active.shape:
            raise ValueError("hypothesis likelihood must match slot shape")
        support = torch.zeros_like(log_likelihood) if support is None else support
        contradiction = torch.zeros_like(log_likelihood) if contradiction is None else contradiction
        if support.shape != state.active.shape or contradiction.shape != state.active.shape:
            raise ValueError("hypothesis evidence tensors must match slot shape")
        if provenance_ids is not None and (
            provenance_ids.shape != (state.batch,)
            or provenance_ids.dtype != torch.int64
        ):
            raise ValueError("hypothesis evidence provenance must be int64 per batch")
        log_weights = (state.log_weights + log_likelihood).masked_fill(~state.active, -torch.inf)
        normalizer = torch.logsumexp(log_weights, -1, keepdim=True)
        normalizer = torch.where(torch.isfinite(normalizer), normalizer, torch.zeros_like(normalizer))
        log_weights = (log_weights - normalizer).masked_fill(~state.active, -torch.inf)
        weights = torch.exp(log_weights).masked_fill(~state.active, 0)
        weak = torch.where(
            state.active & (weights < self.prune_threshold), state.weak_steps + 1,
            torch.zeros_like(state.weak_steps),
        )
        latest_support = state.latest_supporting_provenance_ids.clone()
        latest_contradiction = state.latest_contradicting_provenance_ids.clone()
        if provenance_ids is not None:
            evidence_ids = provenance_ids[:, None].expand_as(state.active)
            latest_support = torch.where(
                state.active & (support > 0), evidence_ids, latest_support
            )
            latest_contradiction = torch.where(
                state.active & (contradiction > 0), evidence_ids,
                latest_contradiction,
            )
        return HypothesisState(
            state.residuals, state.relation_overrides, log_weights,
            state.predicted_outcomes, state.uncertainty,
            state.supporting_evidence + support * state.active,
            state.contradicting_evidence + contradiction * state.active,
            latest_support, latest_contradiction, state.unknown,
            state.scenario_ids, weak, state.versions, state.active,
            state.next_scenario_id,
        )

    def prune(self, state: HypothesisState, logically_impossible: Tensor | None = None) -> HypothesisState:
        impossible = torch.zeros_like(state.active) if logically_impossible is None else logically_impossible
        if impossible.shape != state.active.shape or impossible.dtype != torch.bool:
            raise ValueError("logical-impossibility mask must be boolean with slot shape")
        remove = state.active & ((state.weak_steps >= self.prune_hysteresis) | impossible)
        # Retain one explanation whenever a batch has any active hypothesis.
        for batch_index in range(state.batch):
            if bool(
                state.active[batch_index].any()
                and (remove[batch_index].sum() == state.active[batch_index].sum())
            ):
                remove[batch_index, state.weights[batch_index].argmax()] = False
        values = {
            name: getattr(state, name).clone() if isinstance(getattr(state, name), Tensor) else getattr(state, name)
            for name in state.__dataclass_fields__
        }
        values["active"][remove] = False
        values["scenario_ids"][remove] = 0
        values["log_weights"][remove] = -torch.inf
        values["weak_steps"][remove] = 0
        values["unknown"][remove] = False
        for batch_index in range(state.batch):
            remaining = torch.nonzero(
                values["active"][batch_index], as_tuple=False
            ).flatten()
            if remaining.numel() and not bool(values["unknown"][batch_index].any()):
                values["unknown"][batch_index, remaining[0]] = True
        weights = HypothesisState(**values).weights
        values["log_weights"] = torch.where(
            values["active"], weights.clamp_min(1e-8).log(), values["log_weights"]
        )
        return HypothesisState(**values)

    def merge_duplicates(self, state: HypothesisState) -> HypothesisState:
        values = {
            name: getattr(state, name).clone() if isinstance(getattr(state, name), Tensor) else getattr(state, name)
            for name in state.__dataclass_fields__
        }
        for batch_index in range(state.batch):
            active = torch.nonzero(values["active"][batch_index], as_tuple=False).flatten().tolist()
            for position, left in enumerate(active):
                if not bool(values["active"][batch_index, left]):
                    continue
                for right in active[position + 1:]:
                    if not bool(values["active"][batch_index, right]):
                        continue
                    similarity = F.cosine_similarity(
                        values["residuals"][batch_index, left][None],
                        values["residuals"][batch_index, right][None], dim=-1,
                    ).item()
                    relation_equal = torch.allclose(
                        values["relation_overrides"][batch_index, left],
                        values["relation_overrides"][batch_index, right], atol=1e-5, rtol=1e-5,
                    )
                    if similarity >= self.merge_similarity and relation_equal:
                        pair_log = torch.stack((
                            values["log_weights"][batch_index, left],
                            values["log_weights"][batch_index, right],
                        ))
                        pair_weights = torch.softmax(pair_log, 0)
                        values["residuals"][batch_index, left] = (
                            pair_weights[0] * values["residuals"][batch_index, left]
                            + pair_weights[1] * values["residuals"][batch_index, right]
                        )
                        values["log_weights"][batch_index, left] = torch.logsumexp(pair_log, 0)
                        values["supporting_evidence"][batch_index, left] += values["supporting_evidence"][batch_index, right]
                        values["contradicting_evidence"][batch_index, left] += values["contradicting_evidence"][batch_index, right]
                        values["unknown"][batch_index, left] |= values["unknown"][batch_index, right]
                        values["versions"][batch_index, left] += 1
                        values["active"][batch_index, right] = False
                        values["unknown"][batch_index, right] = False
                        values["scenario_ids"][batch_index, right] = 0
                        values["log_weights"][batch_index, right] = -torch.inf
        normalized = HypothesisState(**values).weights
        values["log_weights"] = torch.where(
            values["active"], normalized.clamp_min(1e-8).log(), values["log_weights"]
        )
        return HypothesisState(**values)

    @staticmethod
    def match_slots(previous: HypothesisState, proposed_residuals: Tensor, proposed_mask: Tensor) -> Tensor:
        """Greedy stable matching prevents harmless slot permutations looking new."""

        if proposed_residuals.shape != previous.residuals.shape or proposed_mask.shape != previous.active.shape:
            raise ValueError("proposed hypotheses must match bank shape")
        batch, count = previous.active.shape
        assignment = torch.full((batch, count), -1, dtype=torch.int64, device=previous.residuals.device)
        similarity = torch.einsum(
            "bhd,bjd->bhj", F.normalize(previous.residuals, dim=-1),
            F.normalize(proposed_residuals, dim=-1),
        )
        for batch_index in range(batch):
            pairs = []
            for old in torch.nonzero(previous.active[batch_index], as_tuple=False).flatten().tolist():
                for new in torch.nonzero(proposed_mask[batch_index], as_tuple=False).flatten().tolist():
                    pairs.append((float(similarity[batch_index, old, new].detach()), old, new))
            used_old: set[int] = set()
            used_new: set[int] = set()
            for _, old, new in sorted(pairs, reverse=True):
                if old not in used_old and new not in used_new:
                    assignment[batch_index, new] = old
                    used_old.add(old)
                    used_new.add(new)
        return assignment
