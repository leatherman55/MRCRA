"""Empirical acceptance suite for the MRCRA exact-authority softmax stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import torch
from torch.nn import functional as F

from .vocabulary_router import (
    CertifiedBalancedVocabularyRouter,
    VocabularyRouterConfig,
    VocabularyRouterIndex,
)


@dataclass(frozen=True, slots=True)
class VocabularyRouterCriterion:
    name: str
    passed: bool
    evidence: dict[str, float | int | str | bool]


@dataclass(frozen=True, slots=True)
class VocabularyRouterExperiment:
    name: str
    vocabulary_size: int
    model_dimension: int
    top_k: int
    exact_threshold_sets: bool
    maximum_logit_error: float
    certificate_rate: float
    dense_fallback_rate: float
    token_vectors_evaluated: int
    output_vectors_avoided: int
    avoided_fraction: float
    dense_seconds: float
    routed_seconds: float
    index_build_seconds: float


@dataclass(frozen=True, slots=True)
class VocabularyRouterAcceptanceReport:
    schema_version: int
    seed: int
    production_scale: bool
    criteria: tuple[VocabularyRouterCriterion, ...]
    experiments: tuple[VocabularyRouterExperiment, ...]
    mlx: dict[str, float | int | bool | str]
    exact_training: dict[str, float | int | bool | str]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _threshold_dense(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    threshold = logits.topk(top_k, -1).values[:, -1:]
    return logits.masked_fill(logits < threshold, -torch.inf)


def _maximum_active_error(
    actual: torch.Tensor, expected: torch.Tensor
) -> float:
    active = torch.isfinite(expected)
    if not bool(active.any()):
        return 0.0
    return float((actual[active] - expected[active]).abs().max())


def _time(function, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    values = []
    for _ in range(repeats):
        started = perf_counter()
        function()
        values.append(perf_counter() - started)
    return sum(values) / len(values)


def _structured_classifier(
    vocabulary_size: int,
    model_dimension: int,
    cluster_size: int,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create separated but nontrivial semantic neighborhoods for work probes."""

    clusters = (vocabulary_size + cluster_size - 1) // cluster_size
    centers = 0.05 * torch.randn(
        clusters, model_dimension, generator=generator
    )
    # Four isolated high-logit neighborhoods make the intended early-exit
    # behavior measurable without pretending that random untrained embeddings
    # should already possess useful semantic geometry.
    centers[:, 0] = -6
    priority = min(4, clusters)
    centers[:priority, 0] = 6
    assignments = torch.arange(vocabulary_size) // cluster_size
    weight = centers[assignments] + 0.003 * torch.randn(
        vocabulary_size, model_dimension, generator=generator
    )
    bias = 0.002 * torch.randn(vocabulary_size, generator=generator)
    hidden = torch.zeros(1, model_dimension)
    hidden[:, 0] = 3
    return weight, bias, hidden


def _experiment(
    vocabulary_size: int,
    model_dimension: int,
    *,
    top_k: int,
    seed: int,
    maximum_refinement_clusters: int,
) -> VocabularyRouterExperiment:
    generator = torch.Generator().manual_seed(seed)
    config = VocabularyRouterConfig(
        cluster_size=16,
        clustering_iterations=3,
        initial_refinement_clusters=32,
        refinement_growth=2,
        maximum_refinement_clusters=maximum_refinement_clusters,
        minimum_vocabulary_size=2,
        minimum_model_dimension=1,
    )
    weight, bias, hidden = _structured_classifier(
        vocabulary_size,
        model_dimension,
        config.cluster_size,
        generator=generator,
    )
    router = CertifiedBalancedVocabularyRouter(weight, bias, config)
    seen = torch.tensor(
        [0, vocabulary_size // 3, vocabulary_size - 1], dtype=torch.int64
    )
    penalty = 1.3

    def dense_call():
        logits = F.linear(hidden, weight, bias)
        selected = logits[:, seen]
        logits[:, seen] = torch.where(
            selected < 0, selected * penalty, selected / penalty
        )
        return _threshold_dense(logits, top_k)

    def routed_call():
        return router.exact_top_k(
            hidden,
            top_k,
            seen_token_ids=seen,
            repetition_penalty=penalty,
        )

    dense = dense_call()
    routed = routed_call()
    routed_dense = routed.to_dense(vocabulary_size)
    exact = torch.equal(torch.isfinite(routed_dense), torch.isfinite(dense))
    error = _maximum_active_error(routed_dense, dense)
    dense_seconds = _time(dense_call, warmup=1, repeats=3)
    routed_seconds = _time(routed_call, warmup=1, repeats=3)
    possible = hidden.shape[0] * vocabulary_size
    return VocabularyRouterExperiment(
        f"structured-v{vocabulary_size}-d{model_dimension}",
        vocabulary_size,
        model_dimension,
        top_k,
        exact,
        error,
        routed.metrics.certificate_rate,
        routed.metrics.dense_fallback_rate,
        routed.metrics.token_logits_evaluated,
        routed.metrics.output_vectors_avoided,
        routed.metrics.output_vectors_avoided / possible,
        dense_seconds,
        routed_seconds,
        router.index.build_seconds,
    )


def _random_and_adversarial_correctness(seed: int) -> dict[str, float | int | bool]:
    generator = torch.Generator().manual_seed(seed)
    cases = 0
    maximum_error = 0.0
    exact_sets = True
    for vocabulary, dimension, top_k in (
        (31, 5, 1),
        (67, 9, 7),
        (257, 20, 50),
        (521, 32, 17),
    ):
        weight = torch.randn(vocabulary, dimension, generator=generator)
        bias = torch.randn(vocabulary, generator=generator)
        hidden = torch.randn(3, dimension, generator=generator)
        router = CertifiedBalancedVocabularyRouter(
            weight,
            bias,
            VocabularyRouterConfig(
                cluster_size=8,
                clustering_iterations=2,
                initial_refinement_clusters=2,
                refinement_growth=2,
                maximum_refinement_clusters=1024,
                minimum_vocabulary_size=2,
                minimum_model_dimension=1,
            ),
        )
        actual = router.exact_top_k(hidden, top_k).to_dense(vocabulary)
        expected = _threshold_dense(F.linear(hidden, weight, bias), top_k)
        exact_sets &= torch.equal(torch.isfinite(actual), torch.isfinite(expected))
        maximum_error = max(maximum_error, _maximum_active_error(actual, expected))
        cases += hidden.shape[0]

    # Every token is tied. Correct behavior is to retain every threshold tie.
    weight = torch.zeros(33, 7)
    bias = torch.zeros(33)
    router = CertifiedBalancedVocabularyRouter(
        weight,
        bias,
        VocabularyRouterConfig(
            cluster_size=4,
            initial_refinement_clusters=1,
            maximum_refinement_clusters=64,
            minimum_vocabulary_size=2,
            minimum_model_dimension=1,
        ),
    )
    tie = router.exact_top_k(torch.randn(1, 7, generator=generator), 3)
    ties_retained = int(tie.mask.sum()) == 33
    return {
        "queries": cases + 1,
        "maximum_logit_error": maximum_error,
        "exact_threshold_sets": exact_sets,
        "all_threshold_ties_retained": ties_retained,
    }


def _fallback_and_staleness(seed: int) -> dict[str, float | int | bool]:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(64, 8, generator=generator)
    bias = torch.randn(64, generator=generator)
    hidden = torch.randn(1, 8, generator=generator)
    config = VocabularyRouterConfig(
        cluster_size=4,
        initial_refinement_clusters=1,
        maximum_refinement_clusters=1,
        minimum_vocabulary_size=2,
        minimum_model_dimension=1,
        stale_index_policy="dense",
    )
    router = CertifiedBalancedVocabularyRouter(weight, bias, config)
    fallback = router.exact_top_k(hidden, 8)
    fallback_exact = torch.equal(
        torch.isfinite(fallback.to_dense(64)),
        torch.isfinite(_threshold_dense(F.linear(hidden, weight, bias), 8)),
    )
    with torch.no_grad():
        weight[0, 0].add_(100)
    stale = router.exact_top_k(hidden, 8)
    stale_exact = torch.equal(
        torch.isfinite(stale.to_dense(64)),
        torch.isfinite(_threshold_dense(F.linear(hidden, weight, bias), 8)),
    )
    return {
        "bounded_search_dense_fallback": fallback.metrics.dense_fallback_queries == 1,
        "bounded_search_fallback_exact": fallback_exact,
        "stale_index_detected": stale.metrics.stale_index_events == 1,
        "stale_index_dense_fallback": stale.metrics.dense_fallback_queries == 1,
        "stale_index_fallback_exact": stale_exact,
    }


def _serialization(seed: int) -> dict[str, bool]:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(47, 6, generator=generator)
    bias = torch.randn(47, generator=generator)
    index = VocabularyRouterIndex.build(
        weight,
        bias,
        VocabularyRouterConfig(
            cluster_size=7,
            minimum_vocabulary_size=2,
            minimum_model_dimension=1,
        ),
    )
    with TemporaryDirectory() as directory:
        path = Path(directory) / "router.pt"
        index.save(path)
        restored = VocabularyRouterIndex.load(path, weight, bias)
        round_trip = (
            restored.signature.content_digest == index.signature.content_digest
            and torch.equal(restored.token_ids, index.token_ids)
        )
        altered = weight.clone()
        altered[0, 0] += 1
        rejected = False
        try:
            VocabularyRouterIndex.load(path, altered, bias)
        except RuntimeError:
            rejected = True
    return {
        "round_trip_exact": round_trip,
        "wrong_checkpoint_rejected": rejected,
    }


def _mlx_acceptance(seed: int) -> dict[str, float | int | bool | str]:
    try:
        from .mlx_backend import MLXCertifiedBalancedVocabularyRouter, mlx_available
    except ImportError as error:  # pragma: no cover - defensive
        return {"available": False, "reason": str(error)}
    if not mlx_available():
        return {"available": False, "reason": "Apple MLX Metal device unavailable"}
    import mlx.core as mx
    import numpy as np

    generator = torch.Generator().manual_seed(seed)
    weight, bias, hidden = _structured_classifier(
        1024, 32, 16, generator=generator
    )
    config = VocabularyRouterConfig(
        cluster_size=16,
        clustering_iterations=2,
        initial_refinement_clusters=8,
        maximum_refinement_clusters=128,
        minimum_vocabulary_size=2,
        minimum_model_dimension=1,
    )
    index = VocabularyRouterIndex.build(weight, bias, config)
    router = MLXCertifiedBalancedVocabularyRouter(
        index,
        mx.array(weight.numpy()),
        mx.array(bias.numpy()),
    )
    actual = router.exact_top_k(mx.array(hidden.numpy()), 50)
    expected = _threshold_dense(F.linear(hidden, weight, bias), 50)
    actual_dense = np.array(actual.to_dense())
    exact = np.array_equal(np.isfinite(actual_dense), np.isfinite(expected.numpy()))
    error = float(
        np.max(np.abs(actual_dense[np.isfinite(actual_dense)] - expected.numpy()[
            np.isfinite(expected.numpy())
        ]))
    )
    result: dict[str, float | int | bool | str] = {
        "available": True,
        "exact_threshold_sets": exact,
        "maximum_logit_error": error,
        "certificate_rate": actual.metrics.certificate_rate,
        "dense_fallback_rate": actual.metrics.dense_fallback_rate,
        "output_vectors_avoided": actual.metrics.output_vectors_avoided,
    }
    pytorch_mps_available = torch.backends.mps.is_available()
    result["pytorch_mps_router_available"] = pytorch_mps_available
    if pytorch_mps_available:
        weight_mps = weight.to("mps")
        bias_mps = bias.to("mps")
        hidden_mps = hidden.to("mps")
        mps_router = CertifiedBalancedVocabularyRouter(
            weight_mps, bias_mps, config
        )
        mps_actual = mps_router.exact_top_k(hidden_mps, 50)
        # A second call proves that immutable certificate tensors remain on
        # the selected control device.  PyTorch MPS intentionally uses a CPU
        # shadow because its branch-heavy scalar certificate synchronizations
        # are slower than one bounded latent/token transfer.
        mps_router.exact_top_k(hidden_mps, 50)
        mps_dense = mps_actual.to_dense(weight.shape[0]).cpu()
        mps_exact = torch.equal(
            torch.isfinite(mps_dense), torch.isfinite(expected)
        )
        mps_error = _maximum_active_error(mps_dense, expected)
        cached = mps_router._device_index_cache[
            (str(mps_router.execution_device), torch.float32)
        ]
        result.update({
            "pytorch_mps_router_exact": mps_exact and mps_error <= 2e-5,
            "pytorch_mps_router_maximum_logit_error": mps_error,
            "pytorch_mps_router_certificate_rate": (
                mps_actual.metrics.certificate_rate
            ),
            "pytorch_mps_router_metadata_resident": all(
                value.device == mps_router.execution_device for value in cached
            ),
            "pytorch_mps_router_control_device": str(
                mps_router.execution_device
            ),
        })
    return result


def _maximum_gradient_error(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
) -> float:
    return max(
        float((left - right).abs().max())
        for left, right in zip(actual, expected, strict=True)
    )


def _exact_training_acceptance(
    seed: int, *, production_scale: bool
) -> dict[str, float | int | bool | str]:
    """Certify the portable exact objective and every differentiated input."""

    from .cognitive_training import (
        exact_cut_cross_entropy,
        exact_tiled_cross_entropy,
    )

    generator = torch.Generator().manual_seed(seed)
    vocabulary_size = 50_257 if production_scale else 97
    model_dimension = 96 if production_scale else 11
    shape = (1, 16) if production_scale else (2, 5)
    hidden_reference = torch.randn(
        *shape, model_dimension, generator=generator, requires_grad=True
    )
    weight_reference = torch.randn(
        vocabulary_size,
        model_dimension,
        generator=generator,
        requires_grad=True,
    )
    bias_reference = torch.randn(
        vocabulary_size, generator=generator, requires_grad=True
    )
    labels = torch.randint(
        0, vocabulary_size, shape, generator=generator
    )
    lengths = torch.randint(0, 4, shape, generator=generator)
    mask = torch.ones(shape, dtype=torch.bool)
    mask.reshape(-1)[::5] = False
    started = perf_counter()
    dense_loss = F.cross_entropy(
        F.linear(hidden_reference, weight_reference, bias_reference)[mask],
        labels[mask],
    )
    dense_loss.backward()
    dense_seconds = perf_counter() - started
    loss_tolerance = 2e-5 + 2e-6 * abs(float(dense_loss.detach()))
    expected_gradients = (
        hidden_reference.grad.detach(),
        weight_reference.grad.detach(),
        bias_reference.grad.detach(),
    )

    hidden_native = hidden_reference.detach().clone().requires_grad_(True)
    weight_native = weight_reference.detach().clone().requires_grad_(True)
    bias_native = bias_reference.detach().clone().requires_grad_(True)
    started = perf_counter()
    native = exact_tiled_cross_entropy(
        hidden_native,
        labels,
        lengths,
        mask,
        weight_native,
        bias_native,
        vocabulary_tile_size=2048 if production_scale else 13,
        checkpoint_tiles=True,
    )
    native.loss.backward()
    native_seconds = perf_counter() - started
    native_gradients = (
        hidden_native.grad,
        weight_native.grad,
        bias_native.grad,
    )
    native_loss_error = float(
        (native.loss.detach() - dense_loss.detach()).abs()
    )
    native_gradient_error = _maximum_gradient_error(
        native_gradients, expected_gradients
    )
    result: dict[str, float | int | bool | str] = {
        "tokens": int(mask.sum()),
        "vocabulary_size": weight_reference.shape[0],
        "model_dimension": model_dimension,
        "dense_reference_seconds": dense_seconds,
        "float32_loss_tolerance": loss_tolerance,
        "native_tiled_exact": (
            native_loss_error <= loss_tolerance
            and native_gradient_error <= 2e-5
        ),
        "native_tiled_seconds": native_seconds,
        "native_tiled_loss_error": native_loss_error,
        "native_tiled_maximum_gradient_error": native_gradient_error,
    }
    native_mps_available = torch.backends.mps.is_available()
    result["native_tiled_mps_available"] = native_mps_available
    if native_mps_available:
        hidden_native_mps = (
            hidden_reference.detach().to("mps").requires_grad_(True)
        )
        weight_native_mps = (
            weight_reference.detach().to("mps").requires_grad_(True)
        )
        bias_native_mps = (
            bias_reference.detach().to("mps").requires_grad_(True)
        )
        started = perf_counter()
        native_mps = exact_tiled_cross_entropy(
            hidden_native_mps,
            labels.to("mps"),
            lengths.to("mps"),
            mask.to("mps"),
            weight_native_mps,
            bias_native_mps,
            vocabulary_tile_size=2048 if production_scale else 13,
            checkpoint_tiles=True,
        )
        native_mps.loss.backward()
        torch.mps.synchronize()
        native_mps_seconds = perf_counter() - started
        native_mps_loss_error = abs(
            float(native_mps.loss.detach().cpu()) - float(dense_loss.detach())
        )
        native_mps_gradient_error = _maximum_gradient_error(
            (
                hidden_native_mps.grad.cpu(),
                weight_native_mps.grad.cpu(),
                bias_native_mps.grad.cpu(),
            ),
            expected_gradients,
        )
        result.update({
            "native_tiled_mps_exact": (
                native_mps_loss_error <= loss_tolerance
                and native_mps_gradient_error <= 2e-5
            ),
            "native_tiled_mps_seconds": native_mps_seconds,
            "native_tiled_mps_loss_error": native_mps_loss_error,
            "native_tiled_mps_maximum_gradient_error": (
                native_mps_gradient_error
            ),
        })

    try:
        from importlib.util import find_spec

        cce_available = find_spec("cut_cross_entropy") is not None
    except (ImportError, ValueError):
        cce_available = False
    result["official_cce_available"] = cce_available
    if cce_available:
        hidden_cce = hidden_reference.detach().clone().requires_grad_(True)
        weight_cce = weight_reference.detach().clone().requires_grad_(True)
        bias_cce = bias_reference.detach().clone().requires_grad_(True)
        started = perf_counter()
        cce = exact_cut_cross_entropy(
            hidden_cce,
            labels,
            lengths,
            mask,
            weight_cce,
            bias_cce,
            implementation="torch_compile",
        )
        cce.loss.backward()
        cce_seconds = perf_counter() - started
        cce_loss_error = float(
            (cce.loss.detach() - dense_loss.detach()).abs()
        )
        cce_gradient_error = _maximum_gradient_error(
            (hidden_cce.grad, weight_cce.grad, bias_cce.grad),
            expected_gradients,
        )
        result.update({
            "official_cce_implementation": "torch_compile",
            "official_cce_exact": (
                cce_loss_error <= loss_tolerance
                and cce_gradient_error <= 2e-5
            ),
            "official_cce_seconds": cce_seconds,
            "official_cce_loss_error": cce_loss_error,
            "official_cce_maximum_gradient_error": cce_gradient_error,
        })
        mps_available = torch.backends.mps.is_available()
        result["official_cce_mps_available"] = mps_available
        if mps_available:
            hidden_mps = (
                hidden_reference.detach().to("mps").requires_grad_(True)
            )
            weight_mps = (
                weight_reference.detach().to("mps").requires_grad_(True)
            )
            bias_mps = bias_reference.detach().to("mps").requires_grad_(True)
            started = perf_counter()
            cce_mps = exact_cut_cross_entropy(
                hidden_mps,
                labels.to("mps"),
                lengths.to("mps"),
                mask.to("mps"),
                weight_mps,
                bias_mps,
                implementation="torch_compile",
            )
            cce_mps.loss.backward()
            torch.mps.synchronize()
            cce_mps_seconds = perf_counter() - started
            mps_loss_error = abs(
                float(cce_mps.loss.detach().cpu()) - float(dense_loss.detach())
            )
            mps_gradient_error = _maximum_gradient_error(
                (
                    hidden_mps.grad.cpu(),
                    weight_mps.grad.cpu(),
                    bias_mps.grad.cpu(),
                ),
                expected_gradients,
            )
            result.update({
                "official_cce_mps_exact": (
                    mps_loss_error <= loss_tolerance
                    and mps_gradient_error <= 2e-5
                ),
                "official_cce_mps_seconds": cce_mps_seconds,
                "official_cce_mps_loss_error": mps_loss_error,
                "official_cce_mps_maximum_gradient_error": mps_gradient_error,
            })

    try:
        from .mlx_backend import (
            mlx_available,
            mlx_exact_tiled_cross_entropy,
        )

        mlx_is_available = mlx_available()
    except ImportError:
        mlx_is_available = False
    result["mlx_native_cce_available"] = mlx_is_available
    if mlx_is_available:
        import mlx.core as mx
        import numpy as np

        # PyTorch MPS and MLX share the same Metal device but maintain separate
        # allocators. Release completed PyTorch probe storage before timing MLX
        # so the benchmark measures the executor rather than allocator pressure
        # left by the preceding cross-framework parity checks.
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
        mx.clear_cache()
        hidden_mlx = mx.array(hidden_reference.detach().numpy())
        weight_mlx = mx.array(weight_reference.detach().numpy())
        bias_mlx = mx.array(bias_reference.detach().numpy())
        labels_mlx = mx.array(labels.numpy().astype(np.int32))
        mask_mlx = mx.array(mask.numpy())

        def objective(hidden, weight, bias):
            return mlx_exact_tiled_cross_entropy(
                hidden,
                weight,
                labels_mlx,
                bias,
                mask=mask_mlx,
                vocabulary_tile_size=2048 if production_scale else 13,
            )

        mlx_loss_and_grad = mx.compile(mx.value_and_grad(
            objective, argnums=(0, 1, 2)
        ))
        started = perf_counter()
        mlx_loss, mlx_gradients = mlx_loss_and_grad(
            hidden_mlx, weight_mlx, bias_mlx
        )
        mx.eval(mlx_loss, mlx_gradients)
        mlx_cold_seconds = perf_counter() - started
        started = perf_counter()
        mlx_loss, mlx_gradients = mlx_loss_and_grad(
            hidden_mlx, weight_mlx, bias_mlx
        )
        mx.eval(mlx_loss, mlx_gradients)
        mlx_warm_seconds = perf_counter() - started
        mlx_loss_error = abs(
            float(np.array(mlx_loss)) - float(dense_loss.detach())
        )
        mlx_gradient_error = max(
            float(np.max(np.abs(np.array(actual) - expected.numpy())))
            for actual, expected in zip(
                mlx_gradients, expected_gradients, strict=True
            )
        )
        result.update({
            "mlx_native_cce_exact": (
                mlx_loss_error <= loss_tolerance
                and mlx_gradient_error <= 2e-5
            ),
            "mlx_native_cce_cold_seconds": mlx_cold_seconds,
            "mlx_native_cce_warm_seconds": mlx_warm_seconds,
            "mlx_native_cce_loss_error": mlx_loss_error,
            "mlx_native_cce_maximum_gradient_error": mlx_gradient_error,
        })
    return result


def run_vocabulary_router_acceptance(
    *,
    seed: int = 20260725,
    production_scale: bool = False,
) -> VocabularyRouterAcceptanceReport:
    """Run correctness, authority, work-reduction, and MLX empirical gates."""

    correctness = _random_and_adversarial_correctness(seed)
    fallback = _fallback_and_staleness(seed + 1)
    serialization = _serialization(seed + 2)
    profiles = (
        ((50_257, 20), (50_257, 96), (50_257, 256))
        if production_scale
        else ((2_048, 20), (4_096, 96), (4_096, 256))
    )
    experiments = tuple(
        _experiment(
            vocabulary,
            dimension,
            top_k=min(50, vocabulary),
            seed=seed + 10 + index,
            maximum_refinement_clusters=512,
        )
        for index, (vocabulary, dimension) in enumerate(profiles)
    )
    mlx = _mlx_acceptance(seed + 20)
    exact_training = _exact_training_acceptance(
        seed + 30, production_scale=production_scale
    )
    criteria = (
        VocabularyRouterCriterion(
            "dense_reference_exactness",
            bool(correctness["exact_threshold_sets"])
            and bool(correctness["all_threshold_ties_retained"])
            and float(correctness["maximum_logit_error"]) <= 2e-5,
            correctness,
        ),
        VocabularyRouterCriterion(
            "fail_closed_fallback_and_staleness",
            all(bool(value) for value in fallback.values()),
            fallback,
        ),
        VocabularyRouterCriterion(
            "content_bound_serialization",
            all(serialization.values()),
            serialization,
        ),
        VocabularyRouterCriterion(
            "certified_work_reduction",
            all(
                experiment.exact_threshold_sets
                and experiment.maximum_logit_error <= 2e-5
                and experiment.certificate_rate == 1
                and experiment.dense_fallback_rate == 0
                and experiment.avoided_fraction >= 0.50
                for experiment in experiments
            ),
            {
                "minimum_avoided_fraction": min(
                    experiment.avoided_fraction for experiment in experiments
                ),
                "maximum_logit_error": max(
                    experiment.maximum_logit_error for experiment in experiments
                ),
                "profiles": len(experiments),
            },
        ),
        VocabularyRouterCriterion(
            "mlx_exact_authority",
            (
                not bool(mlx["available"])
                or (
                    bool(mlx["exact_threshold_sets"])
                    and float(mlx["maximum_logit_error"]) <= 2e-5
                    and float(mlx["certificate_rate"]) == 1
                    and (
                        not bool(mlx["pytorch_mps_router_available"])
                        or (
                            bool(mlx["pytorch_mps_router_exact"])
                            and bool(
                                mlx[
                                    "pytorch_mps_router_metadata_resident"
                                ]
                            )
                        )
                    )
                )
            ),
            mlx,
        ),
        VocabularyRouterCriterion(
            "portable_exact_training_authority",
            bool(exact_training["native_tiled_exact"])
            and (
                not bool(exact_training["native_tiled_mps_available"])
                or bool(exact_training["native_tiled_mps_exact"])
            )
            and (
                not bool(exact_training["official_cce_available"])
                or (
                    bool(exact_training["official_cce_exact"])
                    and (
                        not bool(
                            exact_training.get(
                                "official_cce_mps_available", False
                            )
                        )
                        or bool(exact_training["official_cce_mps_exact"])
                    )
                )
            )
            and (
                not bool(exact_training["mlx_native_cce_available"])
                or bool(exact_training["mlx_native_cce_exact"])
            ),
            exact_training,
        ),
    )
    return VocabularyRouterAcceptanceReport(
        2,
        seed,
        production_scale,
        criteria,
        experiments,
        mlx,
        exact_training,
        all(criterion.passed for criterion in criteria),
    )
