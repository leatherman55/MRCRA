import pytest
import torch

from mrrn import complex_ops as c
from mrrn.mixer import (
    AntiAliasActivation,
    ComplexTriadMixer,
    GatedLocalMixer,
    LowRankStructuredLinear,
    SparseMixtureOfExperts,
)


def test_dense_and_structured_local_mixers_have_expected_shapes_and_gradients():
    x = torch.randn(2, 9, 8, requires_grad=True)
    for mixer in (GatedLocalMixer(8, 2.5), GatedLocalMixer(8, 2.5, structured_rank=3)):
        output = mixer(x)
        assert output.shape == x.shape
        output.square().mean().backward(retain_graph=True)
    assert torch.isfinite(x.grad).all()


def test_low_rank_structured_linear_identity_initial_path_and_bias_option():
    layer = LowRankStructuredLinear(5, 2, bias=False)
    layer.down.weight.data.zero_()
    layer.up.weight.data.zero_()
    x = torch.randn(3, 5)
    torch.testing.assert_close(layer(x), x)
    with_bias = LowRankStructuredLinear(5, 2, bias=True)
    assert with_bias(x).shape == x.shape


def test_complex_triad_multiplies_projected_complex_factors():
    mixer = ComplexTriadMixer(4, 3).double()
    z = torch.randn(2, 5, 4, 2, dtype=torch.float64, requires_grad=True)
    output = mixer(z)
    assert output.shape == z.shape
    assert torch.isfinite(c.magnitude(output)).all()
    output.square().mean().backward()
    assert torch.isfinite(z.grad).all()


def test_sparse_moe_is_exact_weighted_topk_and_reports_load():
    torch.manual_seed(2)
    moe = SparseMixtureOfExperts(6, 4, top_k=2)
    x = torch.randn(2, 3, 6)
    output, load = moe(x)
    logits = moe.router(x)
    values, indices = logits.topk(2, dim=-1)
    expert_outputs = torch.stack([expert(x) for expert in moe.experts], -2)
    selected = torch.gather(expert_outputs, -2, indices.unsqueeze(-1).expand(2, 3, 2, 6))
    expected = (selected * values.softmax(-1).unsqueeze(-1)).sum(-2)
    torch.testing.assert_close(output, expected)
    assert load.shape == (4,)
    torch.testing.assert_close(load.sum(), torch.tensor(2.0))


def test_antialias_activation_preserves_shape_constant_smoothness_and_empty_input():
    module = AntiAliasActivation(2)
    constant = torch.ones(1, 12, 2)
    output = module(constant)
    assert output.shape == constant.shape
    assert output.std(dim=1).max() < 1e-6
    assert module(torch.empty(1, 0, 2)).shape == (1, 0, 2)


def test_causal_antialias_stream_matches_batch_and_has_no_future_leakage():
    module = AntiAliasActivation(2, factor=3, causal=True).double()
    x = torch.randn(2, 11, 2, dtype=torch.float64)
    batch = module(x)
    state = module.initial_state(2, dtype=torch.float64)
    streamed = []
    for position in range(x.shape[1]):
        value, state = module.step(x[:, position], state)
        streamed.append(value.unsqueeze(1))
    torch.testing.assert_close(torch.cat(streamed, 1), batch, atol=1e-12, rtol=1e-12)
    changed = x.clone()
    changed[:, 7:] += 100
    torch.testing.assert_close(module(changed)[:, :7], batch[:, :7], atol=1e-12, rtol=1e-12)
    assert not state.detach().pre_history.requires_grad
    with pytest.raises(ValueError):
        AntiAliasActivation(2, causal=False).initial_state(1)
    with pytest.raises(ValueError):
        module.step(torch.randn(2, 3), state)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LowRankStructuredLinear(0, 1),
        lambda: GatedLocalMixer(0),
        lambda: ComplexTriadMixer(1, 0),
        lambda: SparseMixtureOfExperts(2, 1, 2),
        lambda: AntiAliasActivation(1, 1),
    ],
)
def test_invalid_mixer_configurations(factory):
    with pytest.raises(ValueError):
        factory()


def test_mixer_input_contracts_fail_closed():
    x = torch.randn(2, 3, 4)
    with pytest.raises(ValueError):
        LowRankStructuredLinear(3, 1)(x)
    with pytest.raises(ValueError):
        GatedLocalMixer(3)(x)
    with pytest.raises(ValueError):
        ComplexTriadMixer(3, 1)(torch.randn(2, 4, 2))
    with pytest.raises(ValueError):
        SparseMixtureOfExperts(3, 2)(x)
    with pytest.raises(ValueError):
        AntiAliasActivation(3)(x)
