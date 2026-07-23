"""Zero-staleness packed projections for training and repeated inference."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PackedCache = dict[str, tuple[tuple[Any, ...], Tensor, Tensor | None]]


def _signature(layers: Sequence[nn.Linear]) -> tuple[Any, ...]:
    result: list[Any] = []
    for layer in layers:
        for parameter in (layer.weight, layer.bias):
            result.append(
                None if parameter is None else (
                    parameter._version, parameter.device, parameter.dtype,
                    parameter.data_ptr(),
                )
            )
    return tuple(result)


def packed_linear(
    x: Tensor, layers: Sequence[nn.Linear], cache: PackedCache, key: str,
) -> Tensor:
    """Apply independent linear heads through one GEMM.

    The packed tensor remains in the live graph during training. Inference
    reuses an immutable copy until any source parameter changes version,
    storage, device, or precision.
    """

    if not layers or any(
        layer.in_features != layers[0].in_features for layer in layers
    ):
        raise ValueError("packed projections require a shared positive input width")

    retained = None if torch.is_grad_enabled() else cache.get(key)
    signature = None if torch.is_grad_enabled() else _signature(layers)
    if retained is None or retained[0] != signature:
        weight = torch.cat(tuple(layer.weight for layer in layers))
        bias = (
            None
            if all(layer.bias is None for layer in layers)
            else torch.cat(tuple(
                layer.bias
                if layer.bias is not None
                else layer.weight.new_zeros(layer.out_features)
                for layer in layers
            ))
        )
        if signature is not None:
            cache[key] = (signature, weight, bias)
    else:
        _, weight, bias = retained
    return F.linear(x, weight, bias)
