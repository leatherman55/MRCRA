from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from mrrn.vocabulary_router import (
    CertifiedBalancedVocabularyRouter,
    VocabularyRouterConfig,
    VocabularyRouterIndex,
)


def router_config(**overrides) -> VocabularyRouterConfig:
    values = {
        "cluster_size": 8,
        "clustering_iterations": 2,
        "initial_refinement_clusters": 1,
        "refinement_growth": 2,
        "maximum_refinement_clusters": 64,
        "minimum_vocabulary_size": 2,
        "minimum_model_dimension": 1,
    }
    values.update(overrides)
    return VocabularyRouterConfig(**values)


def dense_adjusted(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    seen: torch.Tensor | None = None,
    penalty: float = 1.0,
) -> torch.Tensor:
    logits = torch.nn.functional.linear(hidden.float(), weight.float(), bias.float())
    if seen is not None and penalty > 1:
        selected = logits[:, seen]
        logits[:, seen] = torch.where(
            selected < 0, selected * penalty, selected / penalty
        )
    return logits


def assert_same_threshold_set(
    result, dense: torch.Tensor, top_k: int
) -> None:
    for row in range(dense.shape[0]):
        threshold = dense[row].topk(top_k).values[-1]
        expected = torch.nonzero(dense[row] >= threshold).flatten()
        active = result.mask[row]
        actual = result.token_ids[row, active].sort().values
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            result.logits[row, active].sort().values,
            dense[row, expected].sort().values,
        )
    routed_dense = result.to_dense(dense.shape[-1])
    expected_dense = dense.clone()
    for row in range(dense.shape[0]):
        threshold = dense[row].topk(top_k).values[-1]
        expected_dense[row, dense[row] < threshold] = -torch.inf
    torch.testing.assert_close(routed_dense, expected_dense)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cluster_size", 1),
        ("clustering_iterations", 0),
        ("initial_refinement_clusters", 0),
        ("refinement_growth", 1.0),
        ("maximum_refinement_clusters", 0),
        ("certificate_absolute_tolerance", -1),
        ("minimum_vocabulary_size", 1),
        ("computation_dtype", torch.float16),
    ],
)
def test_router_configuration_fails_closed(field, value):
    with pytest.raises(ValueError):
        VocabularyRouterConfig(**{field: value})


def test_balanced_index_is_deterministic_complete_and_geometrically_sound():
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(67, 9, generator=generator)
    bias = torch.randn(67, generator=generator)
    config = router_config(cluster_size=8)
    first = VocabularyRouterIndex.build(weight, bias, config)
    second = VocabularyRouterIndex.build(weight, bias, config)

    assert first.cluster_count == 9
    sizes = first.token_mask.sum(-1)
    assert int(sizes.max() - sizes.min()) <= 1
    assert int(sizes.max()) <= config.cluster_size
    torch.testing.assert_close(first.token_ids, second.token_ids)
    torch.testing.assert_close(first.centroids, second.centroids)
    assert first.signature.content_digest == second.signature.content_digest

    for cluster in range(first.cluster_count):
        ids = first.token_ids[cluster, first.token_mask[cluster]]
        rows = weight[ids].float()
        distances = (rows - first.centroids[cluster]).norm(dim=-1)
        assert bool((distances <= first.radii[cluster] + 1e-6).all())
        assert first.maximum_bias[cluster] == bias[ids].max()


@pytest.mark.parametrize(("vocabulary", "dimension", "top_k"), [(31, 5, 1), (67, 9, 7)])
def test_certified_router_matches_dense_random_topk_exactly(
    vocabulary, dimension, top_k
):
    generator = torch.Generator().manual_seed(vocabulary + dimension)
    weight = torch.randn(vocabulary, dimension, generator=generator)
    bias = torch.randn(vocabulary, generator=generator)
    hidden = torch.randn(3, dimension, generator=generator)
    config = router_config(
        cluster_size=7,
        initial_refinement_clusters=2,
        maximum_refinement_clusters=128,
    )
    router = CertifiedBalancedVocabularyRouter(weight, bias, config)
    result = router.exact_top_k(hidden, top_k)
    dense = dense_adjusted(hidden, weight, bias)
    assert result.metrics.certified_queries == hidden.shape[0]
    assert result.metrics.dense_fallback_queries == 0
    assert_same_threshold_set(result, dense, top_k)


def test_router_preserves_every_kth_threshold_tie_instead_of_arbitrarily_dropping_tokens():
    weight = torch.zeros(20, 4)
    bias = torch.zeros(20)
    hidden = torch.randn(1, 4)
    router = CertifiedBalancedVocabularyRouter(
        weight, bias, router_config(cluster_size=4)
    )
    result = router.exact_top_k(hidden, 3)
    assert result.metrics.certified_queries == 1
    assert result.mask.sum() == 20
    assert set(result.token_ids[result.mask].tolist()) == set(range(20))
    assert_same_threshold_set(result, torch.zeros(1, 20), 3)


def test_router_can_certify_early_and_empirically_avoid_most_token_dot_products():
    generator = torch.Generator().manual_seed(19)
    positive = torch.randn(8, 6, generator=generator) * 0.01
    positive[:, 0] += 10
    negative = torch.randn(56, 6, generator=generator) * 0.01
    negative[:, 0] -= 10
    weight = torch.cat((positive, negative))
    bias = torch.zeros(64)
    hidden = torch.tensor([[1.0, 0, 0, 0, 0, 0]])
    router = CertifiedBalancedVocabularyRouter(
        weight,
        bias,
        router_config(
            cluster_size=8,
            initial_refinement_clusters=1,
            maximum_refinement_clusters=4,
        ),
    )
    result = router.exact_top_k(hidden, 4)
    assert result.metrics.certified_queries == 1
    assert result.metrics.dense_fallback_queries == 0
    assert result.metrics.output_vectors_avoided >= 48
    assert result.metrics.token_logits_evaluated <= 16
    assert_same_threshold_set(
        result, dense_adjusted(hidden, weight, bias), 4
    )


def test_repetition_penalty_is_part_of_exact_candidate_authority():
    generator = torch.Generator().manual_seed(23)
    weight = torch.randn(43, 7, generator=generator)
    bias = torch.randn(43, generator=generator)
    hidden = torch.randn(1, 7, generator=generator)
    dense = dense_adjusted(hidden, weight, bias)
    seen = dense.topk(9).indices.flatten().to(torch.int64)
    penalty = 2.5
    router = CertifiedBalancedVocabularyRouter(
        weight, bias, router_config(cluster_size=6)
    )
    result = router.exact_top_k(
        hidden, 5, seen_token_ids=seen, repetition_penalty=penalty
    )
    assert_same_threshold_set(
        result, dense_adjusted(hidden, weight, bias, seen, penalty), 5
    )


def test_device_local_seen_mask_is_exact_and_avoids_pairwise_membership(monkeypatch):
    generator = torch.Generator().manual_seed(24)
    weight = torch.randn(67, 9, generator=generator)
    bias = torch.randn(67, generator=generator)
    hidden = torch.randn(2, 9, generator=generator)
    seen = torch.tensor([0, 3, 7, 11, 19, 43, 66], dtype=torch.int64)
    seen_mask = torch.zeros(67, dtype=torch.bool)
    seen_mask[seen] = True
    penalty = 1.8
    router = CertifiedBalancedVocabularyRouter(
        weight, bias, router_config(cluster_size=7)
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("pairwise membership was used despite a seen mask")

    monkeypatch.setattr(torch, "isin", forbidden)
    result = router.exact_top_k(
        hidden,
        9,
        seen_token_mask=seen_mask,
        repetition_penalty=penalty,
    )
    assert_same_threshold_set(
        result,
        dense_adjusted(hidden, weight, bias, seen, penalty),
        9,
    )


def test_uncertifiable_bounded_search_fails_closed_to_dense_logits():
    generator = torch.Generator().manual_seed(29)
    weight = torch.randn(48, 8, generator=generator)
    bias = torch.randn(48, generator=generator)
    hidden = torch.randn(1, 8, generator=generator)
    config = router_config(
        cluster_size=4,
        initial_refinement_clusters=1,
        maximum_refinement_clusters=1,
    )
    result = CertifiedBalancedVocabularyRouter(weight, bias, config).exact_top_k(
        hidden, 5
    )
    assert result.metrics.certified_queries == 0
    assert result.metrics.dense_fallback_queries == 1
    assert result.mask.sum() == 5
    assert_same_threshold_set(
        result, dense_adjusted(hidden, weight, bias), 5
    )


def test_repeated_uncertifiable_queries_adaptively_disable_routing_overhead():
    generator = torch.Generator().manual_seed(30)
    weight = torch.randn(48, 8, generator=generator)
    bias = torch.randn(48, generator=generator)
    hidden = torch.randn(1, 8, generator=generator)
    router = CertifiedBalancedVocabularyRouter(
        weight,
        bias,
        router_config(
            cluster_size=4,
            initial_refinement_clusters=1,
            maximum_refinement_clusters=1,
            adaptive_fallback_window=2,
            maximum_adaptive_fallback_fraction=0.5,
        ),
    )
    first = router.exact_top_k(hidden, 5)
    second = router.exact_top_k(hidden, 5)
    third = router.exact_top_k(hidden, 5)
    assert first.metrics.bound_rounds == second.metrics.bound_rounds == 1
    assert third.metrics.bound_rounds == 0
    assert third.metrics.dense_fallback_queries == 1
    assert router.cumulative_metrics()["softmax/router/adaptively_enabled"] == 0


def test_stale_index_policies_rebuild_error_and_dense_are_enforced():
    generator = torch.Generator().manual_seed(31)
    hidden = torch.randn(1, 5, generator=generator)

    for policy in ("rebuild", "error", "dense"):
        weight = torch.randn(32, 5, generator=generator)
        bias = torch.randn(32, generator=generator)
        config = router_config(stale_index_policy=policy)
        router = CertifiedBalancedVocabularyRouter(weight, bias, config)
        old_digest = router.index.signature.content_digest
        with torch.no_grad():
            weight[0, 0].add_(100)
        if policy == "error":
            with pytest.raises(RuntimeError, match="stale"):
                router.exact_top_k(hidden, 3)
            continue
        result = router.exact_top_k(hidden, 3)
        assert result.metrics.stale_index_events == 1
        assert_same_threshold_set(
            result, dense_adjusted(hidden, weight, bias), 3
        )
        if policy == "rebuild":
            assert router.index.signature.content_digest != old_digest
        else:
            assert result.metrics.dense_fallback_queries == 1
            assert router.index.signature.content_digest == old_digest


def test_index_serialization_is_content_bound_and_round_trips(tmp_path):
    generator = torch.Generator().manual_seed(37)
    weight = torch.randn(35, 6, generator=generator)
    bias = torch.randn(35, generator=generator)
    index = VocabularyRouterIndex.build(weight, bias, router_config())
    path = tmp_path / "router.pt"
    index.save(path)
    restored = VocabularyRouterIndex.load(path, weight, bias)
    torch.testing.assert_close(restored.token_ids, index.token_ids)
    torch.testing.assert_close(restored.radii, index.radii)
    assert restored.signature.content_digest == index.signature.content_digest

    altered = weight.clone()
    altered[0, 0] += 1
    with pytest.raises(RuntimeError, match="does not belong"):
        VocabularyRouterIndex.load(path, altered, bias)


def test_loaded_index_requires_matching_router_configuration_and_parameters(tmp_path):
    weight = torch.randn(24, 4)
    bias = torch.randn(24)
    index = VocabularyRouterIndex.build(weight, bias, router_config())
    restored = VocabularyRouterIndex(
        replace(index.config),
        index.token_ids,
        index.token_mask,
        index.centroids,
        index.radii,
        index.maximum_bias,
        index.centroid_l1,
        index.token_l1,
        index.signature,
        index.build_seconds,
    )
    router = CertifiedBalancedVocabularyRouter(
        weight, bias, restored.config, index=restored
    )
    assert router.index.fast_compatible(weight, bias)

    cloned_weight = weight.clone()
    cloned_bias = bias.clone()
    rebound = CertifiedBalancedVocabularyRouter(
        cloned_weight, cloned_bias, restored.config, index=restored
    )
    assert rebound.index.fast_compatible(cloned_weight, cloned_bias)
    result = rebound.exact_top_k(torch.randn(1, 4), 3)
    assert result.metrics.stale_index_events == 0
    with pytest.raises(ValueError, match="configuration"):
        CertifiedBalancedVocabularyRouter(
            weight,
            bias,
            replace(restored.config, initial_refinement_clusters=2),
            index=restored,
        )


def test_router_rejects_nonfinite_queries_bad_seen_ids_and_bad_topk():
    weight = torch.randn(12, 3)
    bias = torch.randn(12)
    router = CertifiedBalancedVocabularyRouter(weight, bias, router_config())
    with pytest.raises(ValueError, match="finite"):
        router.exact_top_k(torch.tensor([[float("nan"), 0, 0]]), 1)
    with pytest.raises(ValueError, match="top_k"):
        router.exact_top_k(torch.zeros(1, 3), 0)
    with pytest.raises(ValueError, match="outside"):
        router.exact_top_k(
            torch.zeros(1, 3), 1, seen_token_ids=torch.tensor([12])
        )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS is unavailable"
)
def test_mps_router_uses_persistent_cpu_certificate_shadow_and_returns_exact_mps_results():
    generator = torch.Generator().manual_seed(41)
    weight_cpu = torch.randn(521, 32, generator=generator)
    bias_cpu = torch.randn(521, generator=generator)
    hidden_cpu = torch.randn(2, 32, generator=generator)
    weight = weight_cpu.to("mps")
    bias = bias_cpu.to("mps")
    hidden = hidden_cpu.to("mps")
    router = CertifiedBalancedVocabularyRouter(
        weight,
        bias,
        router_config(
            cluster_size=8,
            initial_refinement_clusters=4,
            maximum_refinement_clusters=128,
        ),
    )

    first = router.exact_top_k(hidden, 17)
    cache = router._device_index_cache[("cpu", torch.float32)]
    pointers = tuple(value.data_ptr() for value in cache)
    second = router.exact_top_k(hidden, 17)

    assert router.execution_device.type == "cpu"
    assert router._execution_weight.device.type == "cpu"
    assert first.logits.device.type == first.token_ids.device.type == "mps"
    assert all(value.device.type == "cpu" for value in cache)
    assert pointers == tuple(
        value.data_ptr()
        for value in router._device_index_cache[("cpu", torch.float32)]
    )
    expected = dense_adjusted(hidden, weight, bias)
    assert_same_threshold_set(first, expected, 17)
    assert_same_threshold_set(second, expected, 17)
