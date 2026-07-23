import numpy as np
import pytest
import torch

from mrrn.config import MRRNConfig
from mrrn.evaluation import CausalTransformerBaseline, parameter_statistics
from mrrn.mlx_backend import MLXCausalTransformer, MLXMRRN, mlx_available
from mrrn.model import MRRN


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
    optimized = MLXMRRN(reference, compile=False)
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
