import torch
from torch import nn

from mrrn.packed_projection import packed_linear


def test_inference_packing_is_reused_and_invalidates_after_parameter_change():
    torch.manual_seed(811)
    layers = (nn.Linear(5, 7), nn.Linear(5, 3))
    cache = {}
    x = torch.randn(2, 5)
    with torch.no_grad():
        first = packed_linear(x, layers, cache, "heads")
        retained_weight = cache["heads"][1]
        second = packed_linear(x, layers, cache, "heads")
        assert cache["heads"][1] is retained_weight
        torch.testing.assert_close(second, first, atol=0, rtol=0)
        layers[0].weight.add_(0.25)
        changed = packed_linear(x, layers, cache, "heads")
        assert cache["heads"][1] is not retained_weight
        assert not torch.equal(changed, first)


def test_training_packing_preserves_exact_outputs_and_gradients():
    torch.manual_seed(821)
    layers = (nn.Linear(4, 6).double(), nn.Linear(4, 2).double())
    reference = (
        nn.Linear(4, 6).double(),
        nn.Linear(4, 2).double(),
    )
    for target, source in zip(reference, layers, strict=True):
        target.load_state_dict(source.state_dict())
    x = torch.randn(3, 4, dtype=torch.float64)
    packed = packed_linear(x, layers, {}, "heads")
    expected = torch.cat(tuple(layer(x) for layer in reference), -1)
    torch.testing.assert_close(packed, expected, atol=1e-14, rtol=1e-14)
    packed.square().sum().backward()
    expected.square().sum().backward()
    for actual, wanted in zip(
        (parameter for layer in layers for parameter in layer.parameters()),
        (parameter for layer in reference for parameter in layer.parameters()),
        strict=True,
    ):
        torch.testing.assert_close(
            actual.grad, wanted.grad, atol=1e-14, rtol=1e-14,
        )
