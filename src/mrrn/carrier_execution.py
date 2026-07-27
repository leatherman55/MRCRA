"""Pure tensor-tree and execution contracts for coarse carrier composites.

The streaming carrier state intentionally uses descriptive dataclasses and
small Python containers at its public boundary.  PyTorch checkpoint and
compiled regions, however, are most reliable when their complete authority is
expressed as a flat tuple of tensors plus immutable static metadata.  This
module provides that reversible bridge without knowing any MRRN class and
without copying tensor storage.

Only tensors cross an autograd/checkpoint boundary.  Scalars, strings, keys,
container kinds, and ``None`` values remain in a fail-closed template.  The
same codec is used for incoming state, outgoing state, and emitted band
histories, so a recomputation cannot silently omit a newly introduced state
field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from threading import Lock
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable


@dataclass(frozen=True, slots=True)
class TensorLeaf:
    """Immutable position of one tensor in a flattened execution tuple."""

    index: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("tensor leaf indices cannot be negative")


@dataclass(frozen=True, slots=True)
class MappingTemplate:
    """Ordered mapping representation that preserves key identity."""

    kind: type
    items: tuple[tuple[Any, Any], ...]


@dataclass(frozen=True, slots=True)
class SequenceTemplate:
    """List/tuple representation that preserves the original container kind."""

    kind: type
    items: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class TensorTreeSpec:
    """Static metadata needed to reconstruct one flattened tensor tree."""

    template: Any
    tensor_count: int

    def __post_init__(self) -> None:
        if self.tensor_count < 0:
            raise ValueError("tensor tree count cannot be negative")

    @property
    def digest(self) -> str:
        return sha256(repr(self.template).encode("utf-8")).hexdigest()


def flatten_tensor_tree(value: Any) -> tuple[tuple[Tensor, ...], TensorTreeSpec]:
    """Extract tensors recursively while preserving every non-tensor leaf."""

    tensors: list[Tensor] = []

    def visit(item: Any) -> Any:
        if isinstance(item, Tensor):
            leaf = TensorLeaf(len(tensors))
            tensors.append(item)
            return leaf
        if isinstance(item, Mapping):
            return MappingTemplate(
                type(item),
                tuple((key, visit(child)) for key, child in item.items()),
            )
        if isinstance(item, (list, tuple)):
            return SequenceTemplate(type(item), tuple(visit(child) for child in item))
        if item is None or isinstance(item, (bool, int, float, str)):
            return item
        raise TypeError(
            "carrier tensor trees support tensors, mappings, lists, tuples, "
            f"scalars, strings, and None; received {type(item).__name__}"
        )

    template = visit(value)
    return tuple(tensors), TensorTreeSpec(template, len(tensors))


def unflatten_tensor_tree(
    tensors: tuple[Tensor, ...] | list[Tensor],
    spec: TensorTreeSpec,
) -> Any:
    """Reconstruct a tensor tree and reject missing, extra, or reused leaves."""

    if len(tensors) != spec.tensor_count:
        raise ValueError(
            f"tensor tree expected {spec.tensor_count} tensors, got {len(tensors)}"
        )
    consumed: set[int] = set()

    def restore(item: Any) -> Any:
        if isinstance(item, TensorLeaf):
            if item.index >= len(tensors) or item.index in consumed:
                raise ValueError("tensor tree contains an invalid or duplicate leaf")
            consumed.add(item.index)
            return tensors[item.index]
        if isinstance(item, MappingTemplate):
            values = [(key, restore(child)) for key, child in item.items]
            if item.kind is dict:
                return dict(values)
            try:
                return item.kind(values)
            except TypeError as error:
                raise TypeError(
                    f"cannot reconstruct mapping kind {item.kind.__name__}"
                ) from error
        if isinstance(item, SequenceTemplate):
            values = tuple(restore(child) for child in item.items)
            if item.kind is tuple:
                return values
            if item.kind is list:
                return list(values)
            try:
                return item.kind(values)
            except TypeError as error:
                raise TypeError(
                    f"cannot reconstruct sequence kind {item.kind.__name__}"
                ) from error
        return item

    result = restore(spec.template)
    if consumed != set(range(spec.tensor_count)):
        raise ValueError("tensor tree template failed to consume every tensor exactly once")
    return result


@dataclass(frozen=True, slots=True)
class CarrierCompositeReceipt:
    """Auditable static identity for one coarse checkpoint invocation."""

    input_state_digest: str
    output_state_digest: str
    state_tensor_count: int
    history_tensor_count: int
    recomputation_granularity: str = "whole_carrier_span"

    def __post_init__(self) -> None:
        if (
            len(self.input_state_digest) != 64
            or len(self.output_state_digest) != 64
            or min(self.state_tensor_count, self.history_tensor_count) < 0
            or self.recomputation_granularity != "whole_carrier_span"
        ):
            raise ValueError("carrier composite receipt is malformed")


@dataclass(frozen=True, slots=True)
class CarrierExecutionPolicy:
    """Resolved cross-platform backend and recomputation contract."""

    device_type: str
    backend: str
    compiler_enabled: bool
    compiler_requested: bool | None
    compiler_backend: str
    affine_scan: str
    simplex_residual: str
    checkpoint_granularity: str
    fallback: str

    def __post_init__(self) -> None:
        if (
            self.device_type not in {"cpu", "mps", "cuda"}
            or self.backend not in {
                "portable_custom_composites",
                "compiled_tensor_cores_with_portable_custom_composites",
            }
            or self.compiler_backend not in {
                "none", "aot_eager", "inductor",
            }
            or (
                self.compiler_enabled
                != (self.compiler_backend != "none")
            )
            or self.affine_scan != "custom_paired_real_adjoint"
            or self.simplex_residual != "custom_simplex_residual_adjoint"
            or self.checkpoint_granularity
            not in {"none", "whole_carrier_span", "per_scale_legacy"}
            or not self.fallback
        ):
            raise ValueError("carrier execution policy is malformed")


@dataclass(frozen=True, slots=True)
class CompiledCarrierShapeKey:
    """Complete identity of one static compiled carrier specialization."""

    model_semantic_digest: str
    device: str
    dtype: str
    batch: int
    padded_length: int
    scale: int
    activation_policy: str
    torch_version: str

    def __post_init__(self) -> None:
        if (
            len(self.model_semantic_digest) != 64
            or not self.device
            or not self.dtype
            or min(self.batch, self.padded_length) <= 0
            or self.scale < 0
            or self.activation_policy
            not in {"retain", "selective", "whole_span"}
            or not self.torch_version
        ):
            raise ValueError("compiled carrier shape key is malformed")

    @property
    def digest(self) -> str:
        return sha256(
            json.dumps(
                asdict(self),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


class CarrierCompilationRegistry:
    """Thread-safe bounded registry for compiler specialization authority."""

    def __init__(self, *, maximum_shapes: int = 128) -> None:
        if maximum_shapes <= 0:
            raise ValueError("compiled carrier registry must be bounded")
        self.maximum_shapes = maximum_shapes
        self._entries: dict[str, CompiledCarrierShapeKey] = {}
        self._first_execution_seconds: dict[str, float] = {}
        self.fallback_count = 0
        self._fallback_reasons: dict[str, int] = {}
        self._lock = Lock()

    def __deepcopy__(self, memo: dict[int, object]):
        """Copy registry authority while always constructing a fresh mutex."""

        existing = memo.get(id(self))
        if existing is not None:
            return existing
        copied = type(self).from_state_dict(self.state_dict())
        memo[id(self)] = copied
        return copied

    def __getstate__(self) -> dict[str, object]:
        """Serialize logical registry state, never process-local lock state."""

        return self.state_dict()

    def __setstate__(self, value: Mapping[str, object]) -> None:
        restored = type(self).from_state_dict(value)
        self.maximum_shapes = restored.maximum_shapes
        self._entries = restored._entries
        self._first_execution_seconds = (
            restored._first_execution_seconds
        )
        self.fallback_count = restored.fallback_count
        self._fallback_reasons = restored._fallback_reasons
        self._lock = Lock()

    def register(self, key: CompiledCarrierShapeKey) -> bool:
        digest = key.digest
        with self._lock:
            existing = self._entries.get(digest)
            if existing is not None:
                if existing != key:
                    raise RuntimeError(
                        "compiled carrier shape digest collision"
                    )
                return False
            if len(self._entries) >= self.maximum_shapes:
                raise RuntimeError(
                    "compiled carrier shape registry exceeded its bound"
                )
            self._entries[digest] = key
            return True

    def record_first_execution(
        self, key: CompiledCarrierShapeKey, seconds: float,
    ) -> None:
        if seconds < 0:
            raise ValueError("compiled carrier timing cannot be negative")
        with self._lock:
            if self._entries.get(key.digest) != key:
                raise RuntimeError(
                    "cannot time an unregistered compiled carrier shape"
                )
            self._first_execution_seconds.setdefault(key.digest, seconds)

    def record_fallback(self, reason: str = "compiler_execution_error") -> None:
        if (
            not reason
            or len(reason) > 160
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_." for character in reason)
        ):
            raise ValueError("compiled fallback reason is malformed")
        with self._lock:
            self.fallback_count += 1
            self._fallback_reasons[reason] = (
                self._fallback_reasons.get(reason, 0) + 1
            )

    @property
    def fallback_reasons(self) -> dict[str, int]:
        with self._lock:
            return dict(self._fallback_reasons)

    @property
    def shape_count(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def compile_seconds(self) -> float:
        with self._lock:
            return sum(self._first_execution_seconds.values())

    def state_dict(self) -> dict[str, object]:
        with self._lock:
            return {
                "schema_version": 2,
                "maximum_shapes": self.maximum_shapes,
                "entries": [
                    {
                        "digest": digest,
                        "key": asdict(key),
                        "first_execution_seconds": (
                            self._first_execution_seconds.get(digest)
                        ),
                    }
                    for digest, key in sorted(self._entries.items())
                ],
                "fallback_count": self.fallback_count,
                "fallback_reasons": dict(sorted(self._fallback_reasons.items())),
            }

    @classmethod
    def from_state_dict(
        cls, value: Mapping[str, object],
    ) -> "CarrierCompilationRegistry":
        try:
            schema_version = int(value["schema_version"])
            if schema_version not in {1, 2}:
                raise ValueError("unsupported compilation registry schema")
            registry = cls(maximum_shapes=int(value["maximum_shapes"]))
            entries = value["entries"]
            if not isinstance(entries, (list, tuple)):
                raise TypeError("compiled registry entries are malformed")
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise TypeError("compiled registry entry is malformed")
                key_value = entry["key"]
                if not isinstance(key_value, Mapping):
                    raise TypeError("compiled shape key is malformed")
                key = CompiledCarrierShapeKey(**dict(key_value))
                if entry["digest"] != key.digest:
                    raise ValueError(
                        "compiled carrier registry digest is corrupt"
                    )
                registry.register(key)
                seconds = entry["first_execution_seconds"]
                if seconds is not None:
                    registry.record_first_execution(key, float(seconds))
            fallback_count = int(value["fallback_count"])
            if fallback_count < 0:
                raise ValueError("compiled fallback count is invalid")
            registry.fallback_count = fallback_count
            if schema_version == 1:
                registry._fallback_reasons = (
                    {"legacy_unspecified": fallback_count}
                    if fallback_count else {}
                )
            else:
                raw_reasons = value["fallback_reasons"]
                if not isinstance(raw_reasons, Mapping):
                    raise TypeError("compiled fallback reasons are malformed")
                reasons = {
                    str(reason): int(count)
                    for reason, count in raw_reasons.items()
                }
                if (
                    any(
                        not reason
                        or len(reason) > 160
                        or count <= 0
                        or any(
                            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_."
                            for character in reason
                        )
                        for reason, count in reasons.items()
                    )
                    or sum(reasons.values()) != fallback_count
                ):
                    raise ValueError(
                        "compiled fallback reason counts are corrupt"
                    )
                registry._fallback_reasons = reasons
            return registry
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "serialized carrier compilation registry is malformed"
            ) from error


def resolve_carrier_execution_policy(
    *,
    device_type: str,
    compile_tensor_cores: bool | None,
    integrated: bool,
    activation_checkpointing: bool,
    activation_policy: str | None = None,
) -> CarrierExecutionPolicy:
    """Resolve a deterministic portable baseline plus optional compilation."""

    if device_type not in {"cpu", "mps", "cuda"}:
        raise ValueError("unsupported carrier execution device")
    if compile_tensor_cores is not None and not isinstance(
        compile_tensor_cores, bool
    ):
        raise ValueError("carrier compiler request must be boolean or None")
    if activation_policy is not None and activation_policy not in {
        "retain", "selective", "whole_span",
    }:
        raise ValueError("unknown carrier activation policy")
    compiler = (
        device_type == "cuda"
        if compile_tensor_cores is None
        else compile_tensor_cores
    )
    # Inductor is the native kernel compiler on CUDA.  On CPU and MPS the
    # carrier's custom paired-real adjoints and explicit state boundary make
    # Inductor's native lowering path pathologically expensive to compile.
    # AOTAutograd still captures and compiles the complete forward/backward
    # graphs on those portable devices while leaving operator execution with
    # PyTorch's proven eager kernels.  This is an explicit backend authority,
    # not an implicit graph-break fallback.
    compiler_backend = (
        "none"
        if not compiler
        else "inductor"
        if device_type == "cuda"
        else "aot_eager"
    )
    effective_activation = (
        "whole_span" if integrated and activation_checkpointing
        else "selective" if activation_checkpointing
        else "retain"
    ) if activation_policy is None else activation_policy
    checkpoint_granularity = {
        "whole_span": "whole_carrier_span",
        "selective": "per_scale_legacy",
        "retain": "none",
    }[effective_activation]
    return CarrierExecutionPolicy(
        device_type=device_type,
        backend=(
            "compiled_tensor_cores_with_portable_custom_composites"
            if compiler
            else "portable_custom_composites"
        ),
        compiler_enabled=compiler,
        compiler_requested=compile_tensor_cores,
        compiler_backend=compiler_backend,
        affine_scan="custom_paired_real_adjoint",
        simplex_residual="custom_simplex_residual_adjoint",
        checkpoint_granularity=checkpoint_granularity,
        fallback=(
            "portable custom adjoints remain authoritative if optional "
            "compiler execution is unavailable"
        ),
    )


class _SimplexResidualAdjoint(torch.autograd.Function):
    """One custom boundary for branch mixing, residual scaling, and masking."""

    @staticmethod
    def forward(
        ctx,
        band: Tensor,
        weights: Tensor,
        layer_scale: Tensor,
        mask: Tensor,
        *branches: Tensor,
    ) -> Tensor:
        delta = torch.zeros_like(band)
        for index, branch in enumerate(branches):
            delta = delta + weights[..., index : index + 1] * branch
        active = mask.unsqueeze(-1).to(band.dtype)
        ctx.save_for_backward(
            weights, layer_scale, active, delta, *branches
        )
        return (band + layer_scale * delta) * active

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        weights, layer_scale, active, delta, *branches = ctx.saved_tensors
        grad_active = grad_output * active
        grad_delta = grad_active * layer_scale
        grad_band = grad_active
        grad_weights = torch.stack(
            tuple(
                (grad_delta * branch).sum(-1)
                for branch in branches
            ),
            -1,
        )
        grad_scale = (grad_active * delta).sum().reshape_as(layer_scale)
        grad_branches = tuple(
            grad_delta * weights[..., index : index + 1]
            for index in range(len(branches))
        )
        return (
            grad_band,
            grad_weights,
            grad_scale,
            None,
            *grad_branches,
        )


def fused_simplex_residual(
    band: Tensor,
    weights: Tensor,
    layer_scale: Tensor,
    mask: Tensor,
    *branches: Tensor,
) -> Tensor:
    """Fuse branch simplex mixing, residual scaling, and padding authority.

    Branch probabilities remain an ordinary softmax output, so their exact
    Jacobian stays under PyTorch's tested implementation.  The much wider
    per-feature weighted sum and residual path becomes one custom autograd
    node with a direct adjoint on CPU, MPS, and CUDA.
    """

    if (
        band.ndim < 2
        or mask.shape != band.shape[:-1]
        or mask.dtype != torch.bool
        or layer_scale.numel() != 1
        or weights.shape != band.shape[:-1] + (len(branches),)
        or not branches
        or any(branch.shape != band.shape for branch in branches)
    ):
        raise ValueError("fused simplex residual tensors are misaligned")
    return _SimplexResidualAdjoint.apply(
        band, weights, layer_scale, mask, *branches
    )
