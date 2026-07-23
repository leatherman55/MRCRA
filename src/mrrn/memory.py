"""Bounded deterministic eidetic memory, write policy, routing, and exact reranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log1p

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(slots=True)
class MemoryItem:
    key: Tensor
    value: Tensor
    signature: Tensor
    timestamp: int
    scale: int
    priority: float
    version: int = 0
    use_count: int = 0


@dataclass(frozen=True, slots=True)
class MemoryHandle:
    slot: int
    version: int


class MemoryWritePolicy(nn.Module):
    """Deployable five-feature innovation/energy/surprise/boundary/novelty gate."""

    feature_count = 5

    def __init__(self, initial_bias: float = -2.0) -> None:
        super().__init__()
        self.linear = nn.Linear(self.feature_count, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, initial_bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.feature_count:
            raise ValueError("write features must end in five deployable values")
        return torch.sigmoid(self.linear(features).squeeze(-1))

    @staticmethod
    def select(scores: Tensor, *, quota: int, threshold: float = 0.5) -> Tensor:
        if scores.ndim != 2 or quota < 0 or not 0 <= threshold <= 1:
            raise ValueError("scores must be (batch,time), quota nonnegative, threshold in [0,1]")
        selected = torch.zeros_like(scores, dtype=torch.bool)
        if quota == 0 or scores.shape[1] == 0:
            return selected
        count = min(quota, scores.shape[1])
        values, indices = torch.topk(scores, count, dim=1, largest=True, sorted=True)
        selected.scatter_(1, indices, values >= threshold)
        return selected

    @classmethod
    def straight_through_select(
        cls, scores: Tensor, *, quota: int, threshold: float = 0.5, temperature: float = 0.1
    ) -> Tensor:
        if temperature <= 0:
            raise ValueError("straight-through temperature must be positive")
        hard = cls.select(scores, quota=quota, threshold=threshold).to(scores.dtype)
        soft = torch.sigmoid((scores - threshold) / temperature)
        # Parenthesize the zero-valued surrogate so the forward pass remains
        # bit-exactly binary instead of accumulating a rounding residue.
        return hard + (soft - soft.detach())


class EideticMemory:
    """Fixed-capacity exact items with cheap signature routing and versioned handles."""

    def __init__(
        self,
        capacity: int,
        key_dim: int,
        value_dim: int,
        signature_dim: int,
        *,
        age_weight: float = 0.01,
        redundancy_weight: float = 0.1,
        use_weight: float = 0.05,
    ) -> None:
        if min(capacity, key_dim, value_dim, signature_dim) <= 0:
            raise ValueError("memory dimensions and capacity must be positive")
        if min(age_weight, redundancy_weight, use_weight) < 0:
            raise ValueError("eviction weights cannot be negative")
        self.capacity, self.key_dim, self.value_dim, self.signature_dim = (
            capacity,
            key_dim,
            value_dim,
            signature_dim,
        )
        self.age_weight, self.redundancy_weight, self.use_weight = (
            age_weight,
            redundancy_weight,
            use_weight,
        )
        self._items: list[MemoryItem | None] = [None] * capacity
        self._versions = [0] * capacity
        self.writes = 0
        self.evictions = 0

    def __len__(self) -> int:
        return sum(item is not None for item in self._items)

    def items(self, *, query_time: int | None = None) -> tuple[MemoryItem, ...]:
        return tuple(
            item for item in self._items
            if item is not None and (query_time is None or item.timestamp <= query_time)
        )

    def _validate_item(self, item: MemoryItem) -> None:
        expected = (("key", self.key_dim), ("value", self.value_dim), ("signature", self.signature_dim))
        for name, dimension in expected:
            tensor = getattr(item, name)
            if tensor.ndim != 1 or tensor.numel() != dimension or not tensor.is_floating_point():
                raise ValueError(f"{name} must be a floating vector of length {dimension}")
        if item.timestamp < 0 or item.scale < 0:
            raise ValueError("timestamp and scale cannot be negative")

    @staticmethod
    def _cosine(a: Tensor, b: Tensor) -> float:
        return float(F.cosine_similarity(a.double(), b.double(), dim=0, eps=1e-12))

    def _eviction_score(self, slot: int, now: int) -> float:
        item = self._items[slot]
        if item is None:
            return -float("inf")
        redundancy = 0.0
        for other_slot, other in enumerate(self._items):
            if other is not None and other_slot != slot:
                redundancy = max(redundancy, self._cosine(item.key, other.key))
        return (
            item.priority
            - self.age_weight * log1p(max(0, now - item.timestamp))
            - self.redundancy_weight * redundancy
            + self.use_weight * log1p(item.use_count)
        )

    def _slot_for_write(self, now: int) -> int:
        for slot, item in enumerate(self._items):
            if item is None:
                return slot
        self.evictions += 1
        return min(range(self.capacity), key=lambda slot: (self._eviction_score(slot, now), slot))

    def write(self, item: MemoryItem) -> MemoryHandle:
        self._validate_item(item)
        slot = self._slot_for_write(item.timestamp)
        self._versions[slot] += 1
        stored = MemoryItem(
            item.key.detach().clone(),
            item.value.detach().clone(),
            item.signature.detach().clone(),
            item.timestamp,
            item.scale,
            float(item.priority),
            self._versions[slot],
            int(item.use_count),
        )
        self._items[slot] = stored
        self.writes += 1
        return MemoryHandle(slot, stored.version)

    def get(self, handle: MemoryHandle) -> MemoryItem:
        if not 0 <= handle.slot < self.capacity:
            raise KeyError("memory slot is out of range")
        item = self._items[handle.slot]
        if item is None or item.version != handle.version:
            raise KeyError("memory handle is stale or empty")
        return item

    def retrieve(
        self, signature: Tensor, k_prime: int, *, query_time: int | None = None
    ) -> list[MemoryHandle]:
        if signature.ndim != 1 or signature.numel() != self.signature_dim:
            raise ValueError(f"signature must have length {self.signature_dim}")
        if k_prime <= 0:
            raise ValueError("k_prime must be positive")
        eligible = [
            (slot, item) for slot, item in enumerate(self._items)
            if item is not None and (query_time is None or item.timestamp <= query_time)
        ]
        if not eligible:
            return []
        matrix = torch.stack([item.signature for _, item in eligible]).double()
        query = signature.to(matrix).double().expand_as(matrix)
        scores = F.cosine_similarity(query, matrix, dim=1, eps=1e-12)
        order = torch.argsort(scores, descending=True, stable=True)[:k_prime].tolist()
        return [MemoryHandle(eligible[index][0], eligible[index][1].version) for index in order]

    def rerank(self, query_key: Tensor, handles: list[MemoryHandle], k: int) -> list[MemoryHandle]:
        if query_key.ndim != 1 or query_key.numel() != self.key_dim or k <= 0:
            raise ValueError("query_key shape and k must be valid")
        handles = sorted(handles, key=lambda handle: handle.slot)
        if not handles:
            return []
        matrix = torch.stack([self.get(handle).key for handle in handles]).double()
        query = query_key.to(matrix).double().expand_as(matrix)
        scores = F.cosine_similarity(query, matrix, dim=1, eps=1e-12)
        selected = [handles[index] for index in torch.argsort(scores, descending=True, stable=True)[:k].tolist()]
        for handle in selected:
            self.get(handle).use_count += 1
        return selected

    def clear(self) -> None:
        self._versions = [version + 1 for version in self._versions]
        self._items = [None] * self.capacity
        self.writes = 0
        self.evictions = 0

    def state_dict(self) -> dict:
        items = []
        for item in self._items:
            items.append(None if item is None else asdict(item))
        return {
            "capacity": self.capacity,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "signature_dim": self.signature_dim,
            "items": items,
            "versions": list(self._versions),
            "writes": self.writes,
            "evictions": self.evictions,
        }

    def load_state_dict(self, state: dict) -> None:
        identity = (state["capacity"], state["key_dim"], state["value_dim"], state["signature_dim"])
        if identity != (self.capacity, self.key_dim, self.value_dim, self.signature_dim):
            raise ValueError("serialized memory dimensions do not match")
        if len(state["items"]) != self.capacity or len(state["versions"]) != self.capacity:
            raise ValueError("serialized memory slot count does not match")
        self._items = [None if item is None else MemoryItem(**item) for item in state["items"]]
        self._versions = list(state["versions"])
        self.writes, self.evictions = int(state["writes"]), int(state["evictions"])

    @staticmethod
    def recall(retrieved: list[MemoryHandle], oracle: list[MemoryHandle], k: int) -> float:
        if k <= 0:
            raise ValueError("k must be positive")
        expected = {(item.slot, item.version) for item in oracle[:k]}
        if not expected:
            return 1.0
        actual = {(item.slot, item.version) for item in retrieved[:k]}
        return len(actual & expected) / len(expected)
