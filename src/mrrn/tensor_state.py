"""Shared mechanics for immutable bounded tensor-state contracts."""

from __future__ import annotations

from dataclasses import fields
from typing import TypeVar

import torch
from torch import Tensor


T = TypeVar("T")


def map_tensor_fields(instance: T, method: str, *args, **kwargs) -> T:
    """Apply a tensor method while preserving integer authority dtypes on ``to``.

    PyTorch's ``Tensor.to(dtype=...)`` converts integer identifiers and masks as
    well as floating state.  Authority tensors must retain their exact dtypes,
    so non-floating values move only to the target device inferred by a probe.
    """

    values = {}
    for field in fields(instance):
        value = getattr(instance, field.name)
        if not isinstance(value, Tensor):
            values[field.name] = value
            continue
        if method != "to" or value.is_floating_point() or value.is_complex():
            values[field.name] = getattr(value, method)(*args, **kwargs)
            continue
        probe = torch.empty((), device=value.device).to(*args, **kwargs)
        values[field.name] = value.to(device=probe.device)
    return type(instance)(**values)


class TensorStateMixin:
    """Detach/device conversion shared by frozen tensor-state dataclasses."""

    def detach(self):
        return map_tensor_fields(self, "detach")

    def to(self, *args, **kwargs):
        return map_tensor_fields(self, "to", *args, **kwargs)
