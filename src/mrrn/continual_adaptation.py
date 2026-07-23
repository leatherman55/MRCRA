"""Isolated, replay-bounded continual adaptation with exact rollback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ReplayItem:
    inputs: Tensor
    targets: Tensor
    provenance_id: int
    continuity_key: str

    def __post_init__(self) -> None:
        if self.provenance_id < 0 or not self.continuity_key:
            raise ValueError("continual replay items require provenance and continuity")
        if self.inputs.shape[0] != self.targets.shape[0]:
            raise ValueError("continual replay input and target batches differ")


class ContinualReplayBuffer:
    """A deterministic bounded ring; archival storage remains application-owned."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("continual replay capacity must be positive")
        self.capacity = capacity
        self._items: list[ReplayItem] = []
        self._cursor = 0

    def append(self, item: ReplayItem) -> None:
        detached = ReplayItem(
            item.inputs.detach().cpu(), item.targets.detach().cpu(),
            item.provenance_id, item.continuity_key,
        )
        if len(self._items) < self.capacity:
            self._items.append(detached)
        else:
            self._items[self._cursor] = detached
        self._cursor = (self._cursor + 1) % self.capacity

    def items(self) -> tuple[ReplayItem, ...]:
        return tuple(self._items)

    def __len__(self) -> int:
        return len(self._items)


@dataclass(frozen=True, slots=True)
class AdaptationReceipt:
    adapter_parameter_names: tuple[str, ...]
    baseline_retention_metric: float
    candidate_retention_metric: float
    maximum_allowed_regression: float
    committed: bool
    rolled_back: bool
    update_steps: int


class IsolatedContinualAdapter:
    """Own a reversible transaction over an explicit parameter allowlist.

    Base weights are never placed in the optimizer.  A candidate is committed
    only after an application-supplied retention evaluator passes; otherwise
    both parameters and optimizer state are discarded exactly.
    """

    def __init__(
        self, model: nn.Module, parameter_names: Iterable[str], *,
        learning_rate: float = 1e-4, maximum_gradient_norm: float = 1.0,
    ) -> None:
        if learning_rate <= 0 or maximum_gradient_norm <= 0:
            raise ValueError("continual optimizer controls must be positive")
        named = dict(model.named_parameters())
        names = tuple(dict.fromkeys(parameter_names))
        if not names or any(name not in named for name in names):
            raise ValueError("continual adaptation requires an exact nonempty parameter allowlist")
        self.model, self.names = model, names
        self.parameters = tuple(named[name] for name in names)
        self.maximum_gradient_norm = maximum_gradient_norm
        self.optimizer = torch.optim.AdamW(self.parameters, lr=learning_rate)
        self._baseline = {name: named[name].detach().clone() for name in names}
        self._base_fingerprints = {
            name: parameter.detach().clone()
            for name, parameter in named.items() if name not in names
        }
        self._steps = 0

    def step(self, loss: Tensor) -> float:
        if loss.ndim or not bool(torch.isfinite(loss)):
            raise ValueError("continual adaptation loss must be one finite scalar")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.maximum_gradient_norm)
        if not bool(torch.isfinite(norm)):
            self.rollback()
            raise FloatingPointError("continual adapter gradients became non-finite")
        self.optimizer.step()
        self._assert_base_unchanged()
        self._steps += 1
        return float(norm.detach())

    def _assert_base_unchanged(self) -> None:
        named = dict(self.model.named_parameters())
        for name, baseline in self._base_fingerprints.items():
            if not torch.equal(named[name].detach(), baseline):
                self.rollback()
                raise RuntimeError(f"continual adaptation changed base parameter {name}")

    def rollback(self) -> None:
        named = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, baseline in self._baseline.items():
                named[name].copy_(baseline)
        self.optimizer.state.clear()

    def retention_gate(
        self, evaluator: Callable[[nn.Module], float], *,
        baseline_metric: float, maximum_allowed_regression: float,
        higher_is_better: bool = True,
    ) -> AdaptationReceipt:
        if maximum_allowed_regression < 0:
            raise ValueError("retention regression allowance cannot be negative")
        candidate = float(evaluator(self.model))
        if not torch.isfinite(torch.tensor(candidate)):
            passed = False
        elif higher_is_better:
            passed = candidate >= baseline_metric - maximum_allowed_regression
        else:
            passed = candidate <= baseline_metric + maximum_allowed_regression
        if not passed:
            self.rollback()
        self._assert_base_unchanged()
        return AdaptationReceipt(
            self.names, float(baseline_metric), candidate,
            float(maximum_allowed_regression), passed, not passed, self._steps,
        )
