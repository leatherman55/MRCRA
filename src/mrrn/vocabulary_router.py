"""Certified balanced vocabulary routing for exact-authority generation.

The router accelerates the *execution* of a tied vocabulary projection without
changing its logits or its sampling authority.  A deterministic, balanced
hierarchical partition groups nearby classifier rows.  Every group stores a
centroid, an enclosing Euclidean radius, and its maximum output bias.  For a
query ``h`` this gives the rigorous upper bound

    max_{v in cluster k} h @ W[v] + b[v]
        <= h @ centroid[k] + ||h||_2 radius[k] + max_bias[k].

Clusters are refined in descending upper-bound order.  Exact token logits are
evaluated inside each refined cluster.  The search stops only when conservative
floating-point lower bounds for the selected tokens exceed every unrefined
cluster upper bound.  Otherwise it expands the search or fails closed to the
dense vocabulary projection.

The index is inference metadata, not a trainable module.  It is cryptographically
bound to the classifier weight and bias and also records PyTorch mutation
versions for cheap per-token stale-index detection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import functional as F


ROUTER_INDEX_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class VocabularyRouterConfig:
    """Execution, clustering, and numerical policy for certified routing."""

    enabled: bool = True
    cluster_size: int = 16
    clustering_iterations: int = 3
    initial_refinement_clusters: int = 64
    refinement_growth: float = 2.0
    maximum_refinement_clusters: int = 512
    certificate_absolute_tolerance: float = 1e-6
    stale_index_policy: Literal["rebuild", "error", "dense"] = "rebuild"
    minimum_vocabulary_size: int = 512
    minimum_model_dimension: int = 32
    adaptive_fallback_window: int = 4
    maximum_adaptive_fallback_fraction: float = 0.50
    computation_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.cluster_size < 2:
            raise ValueError("vocabulary router cluster_size must be at least two")
        if self.clustering_iterations <= 0:
            raise ValueError("vocabulary router clustering_iterations must be positive")
        if self.initial_refinement_clusters <= 0:
            raise ValueError("initial_refinement_clusters must be positive")
        if self.refinement_growth <= 1:
            raise ValueError("vocabulary router refinement_growth must exceed one")
        if self.maximum_refinement_clusters < self.initial_refinement_clusters:
            raise ValueError(
                "maximum_refinement_clusters cannot be smaller than the initial count"
            )
        if self.certificate_absolute_tolerance < 0:
            raise ValueError("certificate tolerance cannot be negative")
        if self.stale_index_policy not in {"rebuild", "error", "dense"}:
            raise ValueError("unknown stale-index policy")
        if self.minimum_vocabulary_size < 2:
            raise ValueError("minimum_vocabulary_size must be at least two")
        if self.minimum_model_dimension <= 0:
            raise ValueError("minimum_model_dimension must be positive")
        if self.adaptive_fallback_window <= 0:
            raise ValueError("adaptive_fallback_window must be positive")
        if not 0 <= self.maximum_adaptive_fallback_fraction <= 1:
            raise ValueError("adaptive fallback fraction must lie in [0,1]")
        if self.computation_dtype not in {torch.float32, torch.float64}:
            raise ValueError("certified routing requires float32 or float64 computation")

    def serializable(self) -> dict[str, Any]:
        values = asdict(self)
        values["computation_dtype"] = str(self.computation_dtype).removeprefix("torch.")
        return values

    @classmethod
    def from_serializable(cls, values: dict[str, Any]) -> "VocabularyRouterConfig":
        copied = dict(values)
        dtype_name = copied.pop("computation_dtype")
        dtype = getattr(torch, dtype_name, None)
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError("router index contains an unsupported computation dtype")
        return cls(**copied, computation_dtype=dtype)


@dataclass(frozen=True, slots=True)
class VocabularyParameterSignature:
    """Content identity plus cheap in-process mutation identity."""

    weight_shape: tuple[int, int]
    bias_shape: tuple[int, ...]
    weight_dtype: str
    bias_dtype: str
    content_digest: str
    weight_data_pointer: int
    bias_data_pointer: int
    weight_version: int
    bias_version: int

    def serializable(self) -> dict[str, Any]:
        return {
            "weight_shape": self.weight_shape,
            "bias_shape": self.bias_shape,
            "weight_dtype": self.weight_dtype,
            "bias_dtype": self.bias_dtype,
            "content_digest": self.content_digest,
        }


@dataclass(frozen=True, slots=True)
class VocabularyRouterIndex:
    """Immutable balanced clusters and their exact certification metadata."""

    config: VocabularyRouterConfig
    token_ids: Tensor
    token_mask: Tensor
    centroids: Tensor
    radii: Tensor
    maximum_bias: Tensor
    centroid_l1: Tensor
    token_l1: Tensor
    signature: VocabularyParameterSignature
    build_seconds: float

    def __post_init__(self) -> None:
        clusters, cluster_size = self.token_ids.shape
        vocabulary, dimension = self.signature.weight_shape
        if self.token_ids.dtype != torch.int64 or self.token_mask.dtype != torch.bool:
            raise ValueError("router token index has an invalid dtype")
        if self.token_mask.shape != self.token_ids.shape:
            raise ValueError("router token mask must match the token index")
        if self.centroids.shape != (clusters, dimension):
            raise ValueError("router centroids have an invalid shape")
        for name, value in (
            ("radii", self.radii),
            ("maximum_bias", self.maximum_bias),
            ("centroid_l1", self.centroid_l1),
        ):
            if value.shape != (clusters,):
                raise ValueError(f"router {name} has an invalid shape")
        if self.token_l1.shape != (vocabulary,):
            raise ValueError("router token L1 norms have an invalid shape")
        if cluster_size != self.config.cluster_size:
            raise ValueError("router physical cluster width does not match its config")
        active = self.token_ids[self.token_mask]
        if active.numel() != vocabulary:
            raise ValueError("router index does not contain every vocabulary token exactly once")
        if not torch.equal(
            active.sort().values,
            torch.arange(vocabulary, dtype=torch.int64, device=active.device),
        ):
            raise ValueError("router index contains duplicate or missing token identifiers")
        if bool((self.token_ids[~self.token_mask] != -1).any()):
            raise ValueError("router padding slots must use token identifier -1")
        if not all(
            bool(torch.isfinite(value).all())
            for value in (
                self.centroids,
                self.radii,
                self.maximum_bias,
                self.centroid_l1,
                self.token_l1,
            )
        ):
            raise ValueError("router index contains non-finite certification metadata")
        if bool((self.radii < 0).any()) or bool((self.token_l1 < 0).any()):
            raise ValueError("router norms and radii cannot be negative")

    @property
    def vocabulary_size(self) -> int:
        return self.signature.weight_shape[0]

    @property
    def model_dimension(self) -> int:
        return self.signature.weight_shape[1]

    @property
    def cluster_count(self) -> int:
        return self.token_ids.shape[0]

    @staticmethod
    def _digest(weight: Tensor, bias: Tensor) -> str:
        digest = sha256()
        for name, value in (("weight", weight), ("bias", bias)):
            contiguous = value.detach().to(device="cpu").contiguous()
            digest.update(name.encode("ascii"))
            digest.update(str(tuple(contiguous.shape)).encode("ascii"))
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(contiguous.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    @classmethod
    def parameter_signature(
        cls, weight: Tensor, bias: Tensor, *, compute_digest: bool = True
    ) -> VocabularyParameterSignature:
        if weight.ndim != 2 or bias.shape != weight.shape[:1]:
            raise ValueError("vocabulary weight and bias shapes are incompatible")
        if not weight.is_floating_point() or not bias.is_floating_point():
            raise ValueError("vocabulary weight and bias must be floating point")
        return VocabularyParameterSignature(
            tuple(weight.shape),
            tuple(bias.shape),
            str(weight.dtype),
            str(bias.dtype),
            cls._digest(weight, bias) if compute_digest else "",
            weight.data_ptr(),
            bias.data_ptr(),
            int(weight._version),
            int(bias._version),
        )

    @staticmethod
    def _unit_rows(weight: Tensor) -> Tensor:
        norms = weight.norm(dim=1, keepdim=True)
        return torch.where(norms > 0, weight / norms.clamp_min(1e-30), weight)

    @classmethod
    def _balanced_spherical_partition(
        cls, weight: Tensor, cluster_count: int, iterations: int
    ) -> tuple[Tensor, ...]:
        """Deterministic capacity-balanced hierarchical spherical two-means.

        Each recursive split is assigned an exact number of descendant
        clusters.  The split cardinality is therefore chosen before geometry,
        making every final cluster differ in size by at most one while repeated
        spherical two-means updates keep nearby directions together.
        """

        unit = cls._unit_rows(weight.float().cpu())
        leaves: list[Tensor] = []
        stack: list[tuple[Tensor, int]] = [
            (torch.arange(unit.shape[0], dtype=torch.int64), cluster_count)
        ]
        while stack:
            indices, descendants = stack.pop()
            if descendants == 1:
                leaves.append(indices)
                continue
            left_descendants = descendants // 2
            right_descendants = descendants - left_descendants
            left_size = round(indices.numel() * left_descendants / descendants)
            left_size = min(
                indices.numel() - right_descendants,
                max(left_descendants, left_size),
            )
            rows = unit[indices]
            mean = rows.mean(0)
            first = (rows - mean).square().sum(-1).argmax()
            second = (rows - rows[first]).square().sum(-1).argmax()
            left_center = rows[first]
            right_center = rows[second]
            ordering = torch.arange(indices.numel(), dtype=torch.int64)
            for _ in range(iterations):
                direction = left_center - right_center
                if float(direction.square().sum()) <= 1e-20:
                    ordering = torch.arange(indices.numel(), dtype=torch.int64)
                else:
                    projection = rows @ direction
                    ordering = torch.argsort(projection, descending=True, stable=True)
                left_rows = rows[ordering[:left_size]]
                right_rows = rows[ordering[left_size:]]
                left_center = cls._unit_rows(left_rows.mean(0, keepdim=True))[0]
                right_center = cls._unit_rows(right_rows.mean(0, keepdim=True))[0]
            left = indices[ordering[:left_size]]
            right = indices[ordering[left_size:]]
            # LIFO insertion preserves deterministic left-to-right leaf order.
            stack.append((right, right_descendants))
            stack.append((left, left_descendants))
        if len(leaves) != cluster_count:
            raise RuntimeError("balanced vocabulary partition produced the wrong leaf count")
        return tuple(leaves)

    @classmethod
    def build(
        cls,
        weight: Tensor,
        bias: Tensor,
        config: VocabularyRouterConfig = VocabularyRouterConfig(),
    ) -> "VocabularyRouterIndex":
        """Build a deterministic, content-bound inference index on the CPU."""

        started = perf_counter()
        signature = cls.parameter_signature(weight, bias)
        vocabulary, dimension = weight.shape
        cluster_count = ceil(vocabulary / config.cluster_size)
        weight_cpu = weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
        bias_cpu = bias.detach().to(device="cpu", dtype=torch.float32).contiguous()
        leaves = cls._balanced_spherical_partition(
            weight_cpu, cluster_count, config.clustering_iterations
        )
        token_ids = torch.full(
            (cluster_count, config.cluster_size), -1, dtype=torch.int64
        )
        token_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        centroids = torch.empty(cluster_count, dimension, dtype=torch.float32)
        radii = torch.empty(cluster_count, dtype=torch.float32)
        maximum_bias = torch.empty(cluster_count, dtype=torch.float32)
        for cluster, indices in enumerate(leaves):
            count = indices.numel()
            if count > config.cluster_size:
                raise RuntimeError("balanced partition exceeded physical cluster capacity")
            token_ids[cluster, :count] = indices
            token_mask[cluster, :count] = True
            rows = weight_cpu[indices]
            centroid = rows.mean(0)
            centroids[cluster] = centroid
            radii[cluster] = (rows - centroid).norm(dim=-1).max()
            maximum_bias[cluster] = bias_cpu[indices].max()
        return cls(
            config,
            token_ids,
            token_mask,
            centroids,
            radii,
            maximum_bias,
            centroids.abs().sum(-1),
            weight_cpu.abs().sum(-1),
            signature,
            perf_counter() - started,
        )

    def fast_compatible(self, weight: Tensor, bias: Tensor) -> bool:
        current = self.parameter_signature(weight, bias, compute_digest=False)
        expected = self.signature
        return (
            current.weight_shape == expected.weight_shape
            and current.bias_shape == expected.bias_shape
            and current.weight_dtype == expected.weight_dtype
            and current.bias_dtype == expected.bias_dtype
            and current.weight_data_pointer == expected.weight_data_pointer
            and current.bias_data_pointer == expected.bias_data_pointer
            and current.weight_version == expected.weight_version
            and current.bias_version == expected.bias_version
        )

    def assert_compatible(
        self, weight: Tensor, bias: Tensor, *, verify_content: bool = False
    ) -> None:
        if self.fast_compatible(weight, bias) and not verify_content:
            return
        current = self.parameter_signature(weight, bias, compute_digest=True)
        if current.content_digest != self.signature.content_digest:
            raise RuntimeError(
                "vocabulary router index is stale: classifier weight or bias changed"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": ROUTER_INDEX_SCHEMA_VERSION,
            "config": self.config.serializable(),
            "token_ids": self.token_ids.cpu(),
            "token_mask": self.token_mask.cpu(),
            "centroids": self.centroids.cpu(),
            "radii": self.radii.cpu(),
            "maximum_bias": self.maximum_bias.cpu(),
            "centroid_l1": self.centroid_l1.cpu(),
            "token_l1": self.token_l1.cpu(),
            "signature": self.signature.serializable(),
            "build_seconds": self.build_seconds,
        }

    def save(self, destination: str | Path) -> None:
        torch.save(self.to_payload(), Path(destination))

    @classmethod
    def load(
        cls,
        source: str | Path,
        weight: Tensor,
        bias: Tensor,
    ) -> "VocabularyRouterIndex":
        payload = torch.load(Path(source), map_location="cpu", weights_only=True)
        if payload.get("schema_version") != ROUTER_INDEX_SCHEMA_VERSION:
            raise ValueError("unsupported vocabulary router index schema")
        stored = payload["signature"]
        current = cls.parameter_signature(weight, bias)
        if stored["content_digest"] != current.content_digest:
            raise RuntimeError(
                "vocabulary router index does not belong to this classifier"
            )
        signature = VocabularyParameterSignature(
            tuple(stored["weight_shape"]),
            tuple(stored["bias_shape"]),
            stored["weight_dtype"],
            stored["bias_dtype"],
            stored["content_digest"],
            current.weight_data_pointer,
            current.bias_data_pointer,
            current.weight_version,
            current.bias_version,
        )
        return cls(
            VocabularyRouterConfig.from_serializable(payload["config"]),
            payload["token_ids"],
            payload["token_mask"],
            payload["centroids"],
            payload["radii"],
            payload["maximum_bias"],
            payload["centroid_l1"],
            payload["token_l1"],
            signature,
            float(payload["build_seconds"]),
        )


@dataclass(frozen=True, slots=True)
class VocabularyRoutingMetrics:
    """One routed query's auditable execution receipt."""

    queries: int
    certified_queries: int
    dense_fallback_queries: int
    stale_index_events: int
    clusters_total: int
    clusters_refined: int
    token_logits_evaluated: int
    output_vectors_avoided: int
    bound_rounds: int
    routing_seconds: float
    minimum_certificate_margin: float

    @property
    def certificate_rate(self) -> float:
        return self.certified_queries / max(1, self.queries)

    @property
    def dense_fallback_rate(self) -> float:
        return self.dense_fallback_queries / max(1, self.queries)

    def as_trackio_metrics(self, prefix: str = "softmax/router") -> dict[str, float]:
        return {
            f"{prefix}/queries": float(self.queries),
            f"{prefix}/certified_queries": float(self.certified_queries),
            f"{prefix}/certificate_rate": self.certificate_rate,
            f"{prefix}/dense_fallback_queries": float(self.dense_fallback_queries),
            f"{prefix}/dense_fallback_rate": self.dense_fallback_rate,
            f"{prefix}/stale_index_events": float(self.stale_index_events),
            f"{prefix}/clusters_total": float(self.clusters_total),
            f"{prefix}/clusters_refined": float(self.clusters_refined),
            f"{prefix}/token_logits_evaluated": float(self.token_logits_evaluated),
            f"{prefix}/output_vectors_avoided": float(self.output_vectors_avoided),
            f"{prefix}/bound_rounds": float(self.bound_rounds),
            f"{prefix}/routing_seconds": self.routing_seconds,
            f"{prefix}/minimum_certificate_margin": self.minimum_certificate_margin,
        }


@dataclass(frozen=True, slots=True)
class RoutedVocabularyCandidates:
    """Exact eligible logits after top-k thresholding, padded by ``mask``."""

    token_ids: Tensor
    logits: Tensor
    mask: Tensor
    metrics: VocabularyRoutingMetrics

    def __post_init__(self) -> None:
        if (
            self.token_ids.ndim != 2
            or self.logits.shape != self.token_ids.shape
            or self.mask.shape != self.token_ids.shape
        ):
            raise ValueError("routed candidates must have matching rank-two tensors")
        if self.token_ids.dtype != torch.int64 or self.mask.dtype != torch.bool:
            raise ValueError("routed candidate identifiers or mask have an invalid dtype")
        if bool((self.token_ids[self.mask] < 0).any()):
            raise ValueError("active routed candidate identifiers cannot be negative")
        if not bool(torch.isfinite(self.logits[self.mask]).all()):
            raise ValueError("active routed logits must be finite")

    def to_dense(self, vocabulary_size: int) -> Tensor:
        if vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive")
        dense = self.logits.new_full(
            (self.logits.shape[0], vocabulary_size), -torch.inf
        )
        for row in range(self.logits.shape[0]):
            active = self.mask[row]
            dense[row, self.token_ids[row, active]] = self.logits[row, active]
        return dense


class CertifiedBalancedVocabularyRouter:
    """Exact top-k retrieval with conservative certification and dense fallback."""

    def __init__(
        self,
        weight: Tensor,
        bias: Tensor,
        config: VocabularyRouterConfig = VocabularyRouterConfig(),
        *,
        index: VocabularyRouterIndex | None = None,
    ) -> None:
        self.weight = weight
        self.bias = bias
        self.config = config
        if index is not None and index.config != config:
            raise ValueError(
                "vocabulary router index configuration does not match the router"
            )
        self.index = (
            VocabularyRouterIndex.build(weight, bias, config)
            if index is None else index
        )
        self.index.assert_compatible(weight, bias, verify_content=index is not None)
        if index is not None and not self.index.fast_compatible(weight, bias):
            current = VocabularyRouterIndex.parameter_signature(
                weight, bias, compute_digest=False
            )
            self.index = replace(
                self.index,
                signature=replace(
                    current,
                    content_digest=self.index.signature.content_digest,
                ),
            )
        # MPS is excellent at large regular contractions but incurs a device
        # synchronization for each branch in this bounded certificate search.
        # A read-only FP32 CPU shadow keeps the control-heavy search fast while
        # the recurrent network remains resident on the Apple GPU.
        self._execution_device = self._select_execution_device()
        self._execution_weight: Tensor
        self._execution_bias: Tensor
        self._refresh_execution_parameters()
        self._totals = {
            "queries": 0,
            "certified_queries": 0,
            "dense_fallback_queries": 0,
            "stale_index_events": 0,
            "clusters_refined": 0,
            "token_logits_evaluated": 0,
            "output_vectors_avoided": 0,
            "bound_rounds": 0,
            "routing_seconds": 0.0,
        }
        self._adaptive_queries = 0
        self._adaptive_fallbacks = 0
        self._adaptively_disabled = False
        self._device_index_cache: dict[
            tuple[str, torch.dtype],
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        ] = {}

    @property
    def enabled_for_vocabulary(self) -> bool:
        return (
            self.config.enabled
            and self.weight.shape[0] >= self.config.minimum_vocabulary_size
            and self.weight.shape[1] >= self.config.minimum_model_dimension
            and not self._adaptively_disabled
        )

    @property
    def execution_device(self) -> torch.device:
        """Device selected for the branch-heavy exact certificate search."""

        return self._execution_device

    def _select_execution_device(self) -> torch.device:
        return (
            torch.device("cpu")
            if self.weight.device.type == "mps"
            else self.weight.device
        )

    def _refresh_execution_parameters(self) -> None:
        if self._execution_device == self.weight.device:
            self._execution_weight = self.weight
            self._execution_bias = self.bias
        else:
            self._execution_weight = self.weight.detach().to(
                device=self._execution_device,
                dtype=self.config.computation_dtype,
            ).contiguous()
            self._execution_bias = self.bias.detach().to(
                device=self._execution_device,
                dtype=self.config.computation_dtype,
            ).contiguous()

    def rebuild(self) -> None:
        self.index = VocabularyRouterIndex.build(self.weight, self.bias, self.config)
        self._execution_device = self._select_execution_device()
        self._refresh_execution_parameters()
        self._device_index_cache.clear()
        self._adaptive_queries = 0
        self._adaptive_fallbacks = 0
        self._adaptively_disabled = False

    def _device_index(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return immutable certificate metadata resident on the query device."""

        key = (str(device), dtype)
        cached = self._device_index_cache.get(key)
        if cached is None:
            cached = (
                self.index.token_ids.to(device=device),
                self.index.token_mask.to(device=device),
                self.index.centroids.to(device=device, dtype=dtype),
                self.index.radii.to(device=device, dtype=dtype),
                self.index.maximum_bias.to(device=device, dtype=dtype),
                self.index.centroid_l1.to(device=device, dtype=dtype),
                self.index.token_l1.to(device=device, dtype=dtype),
            )
            self._device_index_cache[key] = cached
        return cached

    def _check_staleness(self) -> Literal["valid", "rebuilt", "dense"]:
        if self.index.fast_compatible(self.weight, self.bias):
            return "valid"
        if self.config.stale_index_policy == "rebuild":
            self.rebuild()
            return "rebuilt"
        if self.config.stale_index_policy == "dense":
            return "dense"
        self._totals["stale_index_events"] += 1
        raise RuntimeError(
            "vocabulary router index is stale and its policy forbids automatic rebuilding"
        )

    def _round_sizes(self, cluster_count: int) -> tuple[int, ...]:
        maximum = min(cluster_count, self.config.maximum_refinement_clusters)
        sizes: list[int] = []
        current = min(maximum, self.config.initial_refinement_clusters)
        while current < maximum:
            sizes.append(current)
            current = min(maximum, max(current + 1, ceil(current * self.config.refinement_growth)))
        sizes.append(maximum)
        return tuple(dict.fromkeys(sizes))

    def _gamma(self) -> float:
        dimension = self.weight.shape[1]
        epsilon = torch.finfo(self.config.computation_dtype).eps
        product = dimension * epsilon
        if product >= 1:
            raise RuntimeError("router numerical error bound is undefined at this dimension")
        return product / (1 - product)

    def _adjust_repetition(
        self,
        logits: Tensor,
        token_ids: Tensor,
        seen_token_ids: Tensor | None,
        seen_token_mask: Tensor | None,
        repetition_penalty: float,
    ) -> Tensor:
        if repetition_penalty == 1:
            return logits
        if seen_token_mask is not None:
            seen = seen_token_mask.index_select(0, token_ids)
        elif seen_token_ids is not None and seen_token_ids.numel():
            seen = torch.isin(token_ids, seen_token_ids.to(token_ids.device))
        else:
            return logits
        penalized = torch.where(
            logits < 0, logits * repetition_penalty, logits / repetition_penalty
        )
        return torch.where(seen, penalized, logits)

    def _dense_row(
        self,
        hidden: Tensor,
        seen_token_ids: Tensor | None,
        seen_token_mask: Tensor | None,
        repetition_penalty: float,
    ) -> Tensor:
        dtype = self.config.computation_dtype
        logits = F.linear(
            hidden.to(dtype),
            self._execution_weight.to(dtype),
            self._execution_bias.to(dtype),
        )[0]
        all_ids = torch.arange(
            self._execution_weight.shape[0], device=hidden.device
        )
        return self._adjust_repetition(
            logits, all_ids, seen_token_ids, seen_token_mask, repetition_penalty
        )

    def _fallback_row(
        self,
        hidden: Tensor,
        top_k: int,
        seen_token_ids: Tensor | None,
        seen_token_mask: Tensor | None,
        repetition_penalty: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits = self._dense_row(
            hidden, seen_token_ids, seen_token_mask, repetition_penalty
        )
        ids = torch.arange(logits.numel(), device=logits.device)
        threshold = logits.topk(top_k).values[-1]
        eligible = logits >= threshold
        ids = ids[eligible]
        logits = logits[eligible]
        order = torch.argsort(logits, descending=True, stable=True)
        ids, logits = ids[order], logits[order]
        return ids, logits, torch.ones_like(ids, dtype=torch.bool)

    @torch.no_grad()
    def exact_top_k(
        self,
        hidden: Tensor,
        top_k: int,
        *,
        seen_token_ids: Tensor | None = None,
        seen_token_mask: Tensor | None = None,
        repetition_penalty: float = 1.0,
        return_on_input_device: bool = True,
    ) -> RoutedVocabularyCandidates:
        """Return every token tied at or above the exact top-k threshold.

        Returning all threshold ties exactly preserves the prior generation
        implementation, which masked values *below* the kth value rather than
        selecting an arbitrary set of exactly k identifiers.
        """

        if hidden.ndim == 1:
            hidden = hidden.unsqueeze(0)
        if hidden.ndim != 2 or hidden.shape[1] != self.weight.shape[1]:
            raise ValueError("router hidden states must have shape (batch,model_dimension)")
        if not hidden.is_floating_point() or not bool(torch.isfinite(hidden).all()):
            raise ValueError("router hidden states must be finite floating-point values")
        if not 0 < top_k <= self.weight.shape[0]:
            raise ValueError("router top_k lies outside the vocabulary")
        if repetition_penalty < 1:
            raise ValueError("repetition_penalty must be at least one")
        if seen_token_ids is not None and seen_token_mask is not None:
            raise ValueError("supply seen_token_ids or seen_token_mask, not both")
        if seen_token_ids is not None:
            if seen_token_ids.ndim != 1 or seen_token_ids.dtype != torch.int64:
                raise ValueError("seen_token_ids must be one-dimensional int64")
            if seen_token_ids.numel() and (
                int(seen_token_ids.min()) < 0
                or int(seen_token_ids.max()) >= self.weight.shape[0]
            ):
                raise ValueError("seen_token_ids fall outside the vocabulary")
        if seen_token_mask is not None and (
            seen_token_mask.ndim != 1
            or seen_token_mask.dtype != torch.bool
            or seen_token_mask.shape[0] != self.weight.shape[0]
        ):
            raise ValueError(
                "seen_token_mask must be a boolean vocabulary mask"
            )

        started = perf_counter()
        return_device = (
            hidden.device if return_on_input_device else self._execution_device
        )
        stale = self._check_staleness()
        if stale == "dense":
            self._execution_device = self._select_execution_device()
            self._refresh_execution_parameters()
        hidden = hidden.to(self._execution_device)
        if seen_token_ids is not None:
            seen_token_ids = seen_token_ids.to(self._execution_device)
        if seen_token_mask is not None:
            seen_token_mask = seen_token_mask.to(self._execution_device)
        batch = hidden.shape[0]
        rows: list[tuple[Tensor, Tensor, Tensor]] = []
        certified_queries = 0
        fallback_queries = 0
        clusters_refined = 0
        token_evaluations = 0
        rounds = 0
        minimum_margin = float("inf")
        index = self.index
        dtype = self.config.computation_dtype
        device = hidden.device
        (
            token_ids,
            token_mask,
            centroids,
            radii,
            maximum_bias,
            centroid_l1,
            token_l1,
        ) = self._device_index(device, dtype)
        gamma = self._gamma()
        tolerance = self.config.certificate_absolute_tolerance

        for batch_index in range(batch):
            query = hidden[batch_index : batch_index + 1].to(dtype)
            if stale == "dense" or not self.enabled_for_vocabulary:
                rows.append(self._fallback_row(
                    query, top_k, seen_token_ids, seen_token_mask,
                    repetition_penalty,
                ))
                fallback_queries += 1
                token_evaluations += self.weight.shape[0]
                continue

            query_norm = query.norm(dim=-1)[0]
            query_infinity = query.abs().max()
            centroid_dot = (query @ centroids.T)[0]
            radial = query_norm * radii
            arithmetic_error = (
                gamma * query_infinity * centroid_l1
                + 4 * torch.finfo(dtype).eps * radial.abs()
                + tolerance
            )
            upper = centroid_dot + radial + maximum_bias + arithmetic_error
            round_sizes = self._round_sizes(index.cluster_count)
            # The certificate can inspect at most the configured refinement
            # budget plus the largest omitted bound.  Sorting every cluster
            # therefore performs work that cannot affect the decision.
            order_count = min(
                index.cluster_count,
                round_sizes[-1] + (round_sizes[-1] < index.cluster_count),
            )
            order = torch.topk(
                upper, order_count, largest=True, sorted=True
            ).indices
            evaluated_ids: list[Tensor] = []
            evaluated_logits: list[Tensor] = []
            previous = 0
            certified = False
            threshold = query.new_tensor(-torch.inf)
            for target_count in round_sizes:
                rounds += 1
                selected_clusters = order[previous:target_count]
                cluster_ids = token_ids.index_select(0, selected_clusters)
                cluster_mask = token_mask.index_select(0, selected_clusters)
                ids = cluster_ids[cluster_mask]
                if ids.numel():
                    rows_weight = self._execution_weight.index_select(
                        0, ids
                    ).to(dtype)
                    logits = (rows_weight * query).sum(-1)
                    logits = logits + self._execution_bias.index_select(
                        0, ids
                    ).to(dtype)
                    logits = self._adjust_repetition(
                        logits, ids, seen_token_ids, seen_token_mask,
                        repetition_penalty,
                    )
                    evaluated_ids.append(ids)
                    evaluated_logits.append(logits)
                    token_evaluations += ids.numel()
                previous = target_count
                clusters_refined += selected_clusters.numel()
                complete_ids = torch.cat(evaluated_ids)
                complete_logits = torch.cat(evaluated_logits)
                if complete_logits.numel() < top_k:
                    continue
                selected_values, selected_positions = complete_logits.topk(top_k)
                selected_ids = complete_ids[selected_positions]
                selected_error = (
                    gamma
                    * query_infinity
                    * token_l1.index_select(0, selected_ids)
                    + tolerance
                )
                if repetition_penalty > 1:
                    selected_error = selected_error * repetition_penalty
                kth_lower = (selected_values - selected_error).min()
                remaining_upper = (
                    upper[order[target_count]]
                    if target_count < index.cluster_count
                    else query.new_tensor(-torch.inf)
                )
                margin = float((kth_lower - remaining_upper).detach().cpu())
                minimum_margin = min(minimum_margin, margin)
                if margin > 0:
                    threshold = selected_values.min()
                    certified = True
                    break

            if certified:
                eligible = complete_logits >= threshold
                eligible_ids = complete_ids[eligible]
                eligible_logits = complete_logits[eligible]
                sorting = torch.argsort(eligible_logits, descending=True, stable=True)
                eligible_ids = eligible_ids[sorting]
                eligible_logits = eligible_logits[sorting]
                rows.append((
                    eligible_ids,
                    eligible_logits,
                    torch.ones_like(eligible_ids, dtype=torch.bool),
                ))
                certified_queries += 1
            else:
                rows.append(self._fallback_row(
                    query, top_k, seen_token_ids, seen_token_mask,
                    repetition_penalty,
                ))
                fallback_queries += 1
                token_evaluations += self.weight.shape[0]

        maximum_candidates = max(row[0].numel() for row in rows)
        token_ids = torch.full(
            (batch, maximum_candidates), -1, dtype=torch.int64, device=device
        )
        logits = hidden.new_full(
            (batch, maximum_candidates), -torch.inf, dtype=dtype
        )
        mask = torch.zeros_like(token_ids, dtype=torch.bool)
        for row_index, (row_ids, row_logits, row_mask) in enumerate(rows):
            count = row_ids.numel()
            token_ids[row_index, :count] = row_ids
            logits[row_index, :count] = row_logits
            mask[row_index, :count] = row_mask

        vectors_avoided = max(0, batch * self.weight.shape[0] - token_evaluations)
        metrics = VocabularyRoutingMetrics(
            batch,
            certified_queries,
            fallback_queries,
            1 if stale != "valid" else 0,
            index.cluster_count * batch,
            clusters_refined,
            token_evaluations,
            vectors_avoided,
            rounds,
            perf_counter() - started,
            0.0 if minimum_margin == float("inf") else minimum_margin,
        )
        for name in (
            "queries",
            "certified_queries",
            "dense_fallback_queries",
            "stale_index_events",
            "clusters_refined",
            "token_logits_evaluated",
            "output_vectors_avoided",
            "bound_rounds",
        ):
            self._totals[name] += getattr(metrics, name)
        self._totals["routing_seconds"] += metrics.routing_seconds
        if stale != "dense" and not self._adaptively_disabled:
            self._adaptive_queries += batch
            self._adaptive_fallbacks += fallback_queries
            if (
                self._adaptive_queries >= self.config.adaptive_fallback_window
                and self._adaptive_fallbacks / self._adaptive_queries
                > self.config.maximum_adaptive_fallback_fraction
            ):
                self._adaptively_disabled = True
        return RoutedVocabularyCandidates(
            token_ids.to(return_device),
            logits.to(return_device),
            mask.to(return_device),
            metrics,
        )

    def cumulative_metrics(self) -> dict[str, float]:
        queries = int(self._totals["queries"])
        metrics = VocabularyRoutingMetrics(
            queries,
            int(self._totals["certified_queries"]),
            int(self._totals["dense_fallback_queries"]),
            int(self._totals["stale_index_events"]),
            self.index.cluster_count * queries,
            int(self._totals["clusters_refined"]),
            int(self._totals["token_logits_evaluated"]),
            int(self._totals["output_vectors_avoided"]),
            int(self._totals["bound_rounds"]),
            float(self._totals["routing_seconds"]),
            0.0,
        )
        values = metrics.as_trackio_metrics()
        values["softmax/router/adaptively_enabled"] = float(
            not self._adaptively_disabled
        )
        values["softmax/router/adaptive_probe_queries"] = float(
            self._adaptive_queries
        )
        values["softmax/router/cpu_shadow_execution"] = float(
            self.weight.device.type == "mps"
            and self._execution_device.type == "cpu"
        )
        return values
