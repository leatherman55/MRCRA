import numpy as np
import pytest
import torch

from mrrn.config import MRRNConfig
from mrrn.evaluation import CausalTransformerBaseline, parameter_statistics
from mrrn.mlx_backend import (
    MLXCausalTransformer,
    MLXMRRN,
    configure_mlx_memory,
    mlx_available,
    mlx_exact_tiled_cross_entropy,
    mlx_memory_statistics,
    mlx_torch_exact_cross_entropy,
)
from mrrn.model import MRRN
from mrrn.vocabulary_router import VocabularyRouterConfig


pytestmark = pytest.mark.skipif(not mlx_available(), reason="Apple MLX is unavailable")


def tiny_config(**overrides):
    values = dict(
        input_dim=3, model_dim=4, output_dim=2, layers=1, scales=2, heads=1,
        modes=2, mimo_rank=1, attention_window=2, retrieved_items=1,
        memory_capacity=2, mixer_expansion=1, width_growth_cap=1,
        mode_growth_cap=1, width_multiple=1,
    )
    values.update(overrides)
    return MRRNConfig(**values)


def test_mlx_mrrn_imports_exact_weights_and_matches_reference_prediction():
    torch.manual_seed(307)
    reference = MRRN(tiny_config()).eval()
    x = torch.randn(1, 5, 3)
    with torch.no_grad():
        expected = reference(x).prediction.numpy()
    optimized = MLXMRRN(reference)
    actual = np.array(optimized(x))
    assert optimized.parameter_count == parameter_statistics(reference)[0]
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_mlx_training_graph_has_finite_gradients_and_supports_updates():
    import mlx.core as mx

    torch.manual_seed(311)
    optimized = MLXMRRN(MRRN(tiny_config()), compile=False, training=True)
    x, target = torch.randn(1, 4, 3), torch.randn(1, 4, 2)
    loss, gradients = optimized.loss_and_grad(x, target)
    assert mx.isfinite(loss).item()
    assert gradients.keys() == optimized.parameters.keys()
    assert all(mx.all(mx.isfinite(value)).item() for value in gradients.values())
    before = np.array(optimized.parameters["output_head.weight"])
    optimized.apply_gradients(gradients, 1e-3)
    assert not np.array_equal(before, np.array(optimized.parameters["output_head.weight"]))


def test_mlx_exact_tiled_cross_entropy_matches_dense_loss_and_every_gradient():
    import mlx.core as mx

    generator = np.random.default_rng(20260725)
    hidden_np = generator.standard_normal((2, 5, 7), dtype=np.float32)
    weight_np = generator.standard_normal((29, 7), dtype=np.float32)
    bias_np = generator.standard_normal((29,), dtype=np.float32)
    labels_np = generator.integers(0, 29, size=(2, 5), dtype=np.int32)
    mask_np = np.array([
        [True, True, False, True, False],
        [True, False, True, True, True],
    ])

    hidden_torch = torch.tensor(hidden_np, requires_grad=True)
    weight_torch = torch.tensor(weight_np, requires_grad=True)
    bias_torch = torch.tensor(bias_np, requires_grad=True)
    labels_torch = torch.tensor(labels_np, dtype=torch.int64)
    mask_torch = torch.tensor(mask_np)
    expected = torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(hidden_torch, weight_torch, bias_torch)[
            mask_torch
        ],
        labels_torch[mask_torch],
    )
    expected.backward()

    hidden = mx.array(hidden_np)
    weight = mx.array(weight_np)
    bias = mx.array(bias_np)
    labels = mx.array(labels_np)
    mask = mx.array(mask_np)

    def objective(h, w, b):
        return mlx_exact_tiled_cross_entropy(
            h,
            w,
            labels,
            b,
            mask=mask,
            vocabulary_tile_size=6,
        )

    actual, gradients = mx.value_and_grad(
        objective, argnums=(0, 1, 2)
    )(hidden, weight, bias)
    mx.eval(actual, gradients)
    np.testing.assert_allclose(
        np.array(actual), expected.detach().numpy(), atol=3e-6, rtol=3e-6
    )
    for mlx_gradient, torch_gradient in zip(
        gradients,
        (hidden_torch.grad, weight_torch.grad, bias_torch.grad),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.array(mlx_gradient),
            torch_gradient.numpy(),
            atol=8e-6,
            rtol=8e-6,
        )
    assert np.count_nonzero(np.array(gradients[1])) == weight_np.size
    assert np.count_nonzero(np.array(gradients[2])) == bias_np.size


def test_mlx_torch_bridge_preserves_exact_loss_and_complete_autograd():
    torch.manual_seed(20260727)
    hidden_a = torch.randn(2, 5, 7, requires_grad=True)
    weight_a = torch.randn(29, 7, requires_grad=True)
    bias_a = torch.randn(29, requires_grad=True)
    labels = torch.randint(0, 29, (2, 5))
    mask = torch.tensor([
        [True, True, False, True, False],
        [True, False, True, True, True],
    ])
    expected = torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(
            hidden_a, weight_a, bias_a,
        )[mask],
        labels[mask],
    )
    expected.backward()

    hidden_b = hidden_a.detach().clone().requires_grad_(True)
    weight_b = weight_a.detach().clone().requires_grad_(True)
    bias_b = bias_a.detach().clone().requires_grad_(True)
    actual = mlx_torch_exact_cross_entropy(
        hidden_b,
        weight_b,
        labels,
        bias_b,
        mask=mask,
        vocabulary_tile_size=8,
    )
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    for actual_gradient, expected_gradient in (
        (hidden_b.grad, hidden_a.grad),
        (weight_b.grad, weight_a.grad),
        (bias_b.grad, bias_a.grad),
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            atol=8e-6,
            rtol=8e-6,
        )
    assert int(torch.count_nonzero(weight_b.grad)) == weight_b.numel()
    assert int(torch.count_nonzero(bias_b.grad)) == bias_b.numel()


def test_mlx_torch_bridge_rejects_invalid_targets_before_accelerator_work():
    hidden = torch.randn(1, 3, 4, requires_grad=True)
    weight = torch.randn(7, 4, requires_grad=True)
    labels = torch.tensor([[0, 7, 2]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    with pytest.raises(ValueError, match="within the vocabulary"):
        mlx_torch_exact_cross_entropy(
            hidden,
            weight,
            labels,
            mask=mask,
            vocabulary_tile_size=4,
        )


def test_mlx_exact_loss_obeys_bounded_cache_policy_and_releases_free_buffers():
    policy = configure_mlx_memory(
        memory_limit_bytes=512 << 20,
        cache_limit_bytes=32 << 20,
    )
    try:
        hidden = torch.randn(2, 16, 7, requires_grad=True)
        weight = torch.randn(257, 7, requires_grad=True)
        bias = torch.randn(257, requires_grad=True)
        labels = torch.randint(0, 257, (2, 16))
        loss = mlx_torch_exact_cross_entropy(
            hidden,
            weight,
            labels,
            bias,
            mask=torch.ones_like(labels, dtype=torch.bool),
            vocabulary_tile_size=64,
        )
        loss.backward()
        memory = mlx_memory_statistics()
        assert memory["peak_bytes"] > 0
        assert memory["cache_bytes"] == 0
        assert memory["cache_bytes"] <= 32 << 20
    finally:
        configure_mlx_memory(
            memory_limit_bytes=policy.previous_memory_limit_bytes,
            cache_limit_bytes=policy.previous_cache_limit_bytes,
        )


def test_mlx_mrrn_compiled_exact_cce_method_preserves_mask_and_reductions():
    torch.manual_seed(312)
    reference = MRRN(
        tiny_config(input_dim=6, model_dim=6, output_dim=37)
    ).eval()
    optimized = MLXMRRN(reference, compile=True)
    hidden = torch.randn(2, 4, 6)
    labels = torch.randint(0, 37, (2, 4))
    mask = torch.tensor([
        [True, True, False, True],
        [False, True, True, True],
    ])
    expected_nll = torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(
            hidden,
            reference.output_head.weight,
            reference.output_head.bias,
        )[mask],
        labels[mask],
        reduction="none",
    )
    actual_mean = optimized.linear_cross_entropy(
        hidden,
        labels,
        mask=mask,
        vocabulary_tile_size=8,
    )
    actual_sum = optimized.linear_cross_entropy(
        hidden,
        labels,
        mask=mask,
        vocabulary_tile_size=8,
        reduction="sum",
    )
    actual_none = np.array(optimized.linear_cross_entropy(
        hidden,
        labels,
        mask=mask,
        vocabulary_tile_size=8,
        reduction="none",
    ))
    np.testing.assert_allclose(
        np.array(actual_mean), expected_nll.mean().detach().numpy(), atol=3e-6
    )
    np.testing.assert_allclose(
        np.array(actual_sum), expected_nll.sum().detach().numpy(), atol=3e-6
    )
    np.testing.assert_allclose(
        actual_none[mask.numpy()], expected_nll.detach().numpy(), atol=3e-6
    )
    assert np.count_nonzero(actual_none[~mask.numpy()]) == 0


def test_mlx_mrrn_compiled_exact_cce_gradient_api_matches_dense_reference():
    torch.manual_seed(314)
    reference = MRRN(
        tiny_config(input_dim=6, model_dim=6, output_dim=37)
    ).eval()
    optimized = MLXMRRN(reference, compile=True)
    hidden = torch.randn(2, 4, 6, requires_grad=True)
    labels = torch.randint(0, 37, (2, 4))
    mask = torch.tensor([
        [True, True, False, True],
        [False, True, True, True],
    ])
    weight = reference.output_head.weight.detach().clone().requires_grad_(True)
    bias = reference.output_head.bias.detach().clone().requires_grad_(True)
    expected = torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(hidden, weight, bias)[mask],
        labels[mask],
    )
    expected.backward()

    actual, gradients = optimized.linear_cross_entropy_and_grad(
        hidden.detach(),
        labels,
        mask=mask,
        vocabulary_tile_size=8,
    )
    np.testing.assert_allclose(
        np.array(actual), expected.detach().numpy(), atol=3e-6, rtol=3e-6
    )
    for mlx_gradient, torch_gradient in zip(
        gradients, (hidden.grad, weight.grad, bias.grad), strict=True
    ):
        np.testing.assert_allclose(
            np.array(mlx_gradient),
            torch_gradient.numpy(),
            atol=8e-6,
            rtol=8e-6,
        )


def test_mlx_transformer_uses_the_same_baseline_weights_and_outputs():
    torch.manual_seed(313)
    reference = CausalTransformerBaseline(3, 4, 2, 1, 1).eval()
    x = torch.randn(1, 5, 3)
    with torch.no_grad():
        expected = reference(x).numpy()
    optimized = MLXCausalTransformer(reference)
    np.testing.assert_allclose(np.array(optimized(x)), expected, atol=2e-6, rtol=2e-6)
    assert optimized.parameter_count == parameter_statistics(reference)[0]


def test_mlx_recurrent_paths_match_reference_streaming_exactly():
    torch.manual_seed(317)
    x = torch.randn(1, 8, 3)
    reference = MRRN(tiny_config()).eval()
    state, expected = reference.initial_stream_state(1), []
    with torch.no_grad():
        for position in range(x.shape[1]):
            step = reference.step(x[:, position], state)
            state = step.state
            expected.append(step.prediction.unsqueeze(1))
    optimized = MLXMRRN(reference)
    actual, state = optimized.decode(x)
    np.testing.assert_allclose(
        np.array(actual), torch.cat(expected, 1).numpy(), atol=2e-6, rtol=2e-6
    )
    assert state.position == x.shape[1]

    transformer = CausalTransformerBaseline(3, 4, 2, 1, 1).eval()
    state, expected = transformer.initial_decode_state(1), []
    with torch.no_grad():
        for position in range(x.shape[1]):
            value, state = transformer.step(x[:, position], state)
            expected.append(value.unsqueeze(1))
    optimized_transformer = MLXCausalTransformer(transformer)
    actual, state = optimized_transformer.decode(x)
    np.testing.assert_allclose(
        np.array(actual), torch.cat(expected, 1).numpy(), atol=2e-6, rtol=2e-6
    )
    assert state.position == state.capacity == x.shape[1]


@pytest.mark.parametrize("topology", [
    {"structured_mixer_rank": 1},
    {"continuous_signal": True},
    {"structured_mixer_rank": 1, "continuous_signal": True},
])
def test_mlx_backend_matches_structured_and_alias_controlled_topologies(topology):
    torch.manual_seed(319)
    reference = MRRN(tiny_config(**topology)).eval()
    x = torch.randn(1, 8, 3)
    with torch.no_grad():
        expected = reference(x).prediction.numpy()
    optimized = MLXMRRN(reference)
    np.testing.assert_allclose(
        np.array(optimized(x)), expected, atol=3e-6, rtol=3e-6
    )
    actual_stream, _ = optimized.decode(x)
    state, expected_stream = reference.initial_stream_state(1), []
    with torch.no_grad():
        for position in range(x.shape[1]):
            step = reference.step(x[:, position], state)
            state = step.state
            expected_stream.append(step.prediction[:, None])
    np.testing.assert_allclose(
        np.array(actual_stream), torch.cat(expected_stream, 1).numpy(),
        atol=3e-6, rtol=3e-6,
    )


def test_mlx_backend_fails_closed_on_bad_benchmarks():
    with pytest.raises(ValueError, match="causal"):
        MLXMRRN(MRRN(tiny_config(causal=False)))
    optimized = MLXMRRN(MRRN(tiny_config()))
    with pytest.raises(ValueError):
        optimized.benchmark(torch.randn(1, 3, 3), repeats=0)
    with pytest.raises(RuntimeError, match="training=True"):
        optimized.benchmark_training(torch.randn(1, 3, 3), torch.randn(1, 3, 2))
    with pytest.raises(ValueError, match="learning_rate"):
        optimized.apply_gradients({}, 0)
    stream = optimized.initial_stream_state(1)
    with pytest.raises(ValueError, match="shape"):
        optimized.stream_step(torch.randn(2, 3), stream)
    transformer = MLXCausalTransformer(CausalTransformerBaseline(3, 4, 2, 1, 1))
    transformer_state = transformer.initial_stream_state(1, 1)
    _, transformer_state = transformer.stream_step(torch.randn(1, 3), transformer_state)
    with pytest.raises(ValueError, match="capacity"):
        transformer.stream_step(torch.randn(1, 3), transformer_state)


def test_mlx_projection_free_decode_matches_reference_latents():
    torch.manual_seed(331)
    reference = MRRN(tiny_config(input_dim=4, model_dim=4, output_dim=29)).eval()
    x = torch.randn(1, 7, 4)
    state = reference.initial_stream_state(1)
    expected = []
    with torch.no_grad():
        for position in range(x.shape[1]):
            step = reference.step(
                x[:, position], state, project_output=False
            )
            state = step.state
            expected.append(step.latent[:, None])
            assert step.prediction.shape[-1] == 0
    optimized = MLXMRRN(reference)
    actual, mlx_state = optimized.decode_latents(x)
    np.testing.assert_allclose(
        np.array(actual),
        torch.cat(expected, 1).numpy(),
        atol=3e-6,
        rtol=3e-6,
    )
    assert mlx_state.position == x.shape[1]


def test_mlx_certified_router_matches_dense_topk_and_repetition_penalty():
    torch.manual_seed(337)
    reference = MRRN(tiny_config(input_dim=6, model_dim=6, output_dim=67)).eval()
    config = VocabularyRouterConfig(
        cluster_size=7,
        clustering_iterations=2,
        initial_refinement_clusters=2,
        refinement_growth=2,
        maximum_refinement_clusters=128,
        minimum_vocabulary_size=2,
        minimum_model_dimension=1,
    )
    optimized = MLXMRRN(
        reference,
        compile=False,
        vocabulary_router_config=config,
    )
    latent = torch.randn(2, 6)
    seen = torch.tensor([1, 3, 5, 7], dtype=torch.int64)
    penalty = 1.7
    actual = optimized.routed_top_k(
        latent,
        9,
        seen_token_ids=seen,
        repetition_penalty=penalty,
    )
    seen_mask = torch.zeros(67, dtype=torch.bool)
    seen_mask[seen] = True
    masked = optimized.routed_top_k(
        latent,
        9,
        seen_token_mask=seen_mask,
        repetition_penalty=penalty,
    )
    dense = torch.nn.functional.linear(
        latent.float(),
        reference.output_head.weight.float(),
        reference.output_head.bias.float(),
    )
    selected = dense[:, seen]
    dense[:, seen] = torch.where(
        selected < 0, selected * penalty, selected / penalty
    )
    expected = dense.clone()
    for row in range(dense.shape[0]):
        threshold = dense[row].topk(9).values[-1]
        expected[row, dense[row] < threshold] = -torch.inf
    np.testing.assert_allclose(
        np.array(actual.to_dense()),
        expected.detach().numpy(),
        atol=2e-5,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        np.array(masked.to_dense()),
        expected.detach().numpy(),
        atol=2e-5,
        rtol=2e-5,
    )
    assert actual.metrics.certified_queries == latent.shape[0]
    assert actual.metrics.dense_fallback_queries == 0


def test_mlx_router_is_explicitly_inference_only_and_fails_closed_when_absent():
    reference = MRRN(tiny_config(input_dim=4, model_dim=4, output_dim=17))
    config = VocabularyRouterConfig(
        minimum_vocabulary_size=2,
        minimum_model_dimension=1,
    )
    with pytest.raises(ValueError, match="inference-only"):
        MLXMRRN(
            reference,
            compile=False,
            training=True,
            vocabulary_router_config=config,
        )
    optimized = MLXMRRN(reference, compile=False)
    with pytest.raises(RuntimeError, match="vocabulary_router_config"):
        optimized.routed_top_k(torch.zeros(1, 4), 2)


def test_mlx_router_is_enabled_by_default_for_qualifying_inference_models():
    reference = MRRN(tiny_config(input_dim=32, model_dim=32, output_dim=521))
    optimized = MLXMRRN(reference, compile=False)
    assert optimized.vocabulary_router is not None

    disabled = MLXMRRN(
        reference,
        compile=False,
        vocabulary_router_config=VocabularyRouterConfig(enabled=False),
    )
    assert disabled.vocabulary_router is None
