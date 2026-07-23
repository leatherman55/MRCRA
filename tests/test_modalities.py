import pytest
import torch

from mrrn.modalities import (
    ConservationProjector,
    ContinuousSignalEncoder,
    DomainSpec,
    DirectionalResonator2D,
    FactorizedVideoLifting,
    GraphPolynomialEncoder,
    InducingPointSetEncoder,
    PhysicalFieldEncoder,
    SeparableLifting2D,
    SpatialResonanceNetwork,
    TokenEncoder,
    ImageBand,
    require_meaningful_order,
)


@pytest.mark.parametrize("height,width", [(1, 1), (4, 6), (5, 7), (8, 3)])
def test_separable_2d_lifting_is_exact_for_even_odd_and_degenerate_images(height, width):
    torch.manual_seed(height * 10 + width)
    transform = SeparableLifting2D(2, 2).double()
    for parameter in transform.parameters():
        parameter.data.normal_(std=0.08)
    image = torch.randn(2, height, width, 2, dtype=torch.float64, requires_grad=True)
    mask = torch.rand(2, height, width) > 0.1
    bands, context = transform(image, mask)
    restored = transform.inverse(bands, context)
    torch.testing.assert_close(restored, image, atol=3e-12, rtol=3e-12)
    assert [item.orientation for item in bands[-4:]] == ["LH", "HL", "HH", "LL"]
    restored.square().mean().backward()
    assert torch.isfinite(image.grad).all()


def test_factorized_video_spatial_temporal_transform_roundtrips_exactly():
    torch.manual_seed(117)
    transform = FactorizedVideoLifting(2, spatial_levels=1, temporal_levels=2).double()
    for parameter in transform.parameters():
        parameter.data.normal_(std=0.05)
    video = torch.randn(2, 5, 5, 4, 2, dtype=torch.float64)
    mask = torch.rand(2, 5, 5, 4) > 0.1
    bands, context = transform(video, mask)
    restored = transform.inverse(bands, context)
    torch.testing.assert_close(restored, video, atol=3e-12, rtol=3e-12)
    assert len(bands) == 4 * 3


def test_token_and_continuous_encoders_preserve_masks_calibration_and_metadata():
    tokens = torch.tensor([[1, 2, 0], [3, 0, 0]])
    encoded = TokenEncoder(8, 4, padding_index=0)(tokens)
    assert encoded.mask.tolist() == [[True, True, False], [True, False, False]]
    assert (encoded.values[~encoded.mask] == 0).all() and encoded.domain.kind == "sequence"

    signal = torch.randn(2, 9, 3)
    mask = torch.ones(2, 9, dtype=torch.bool)
    sensor = ContinuousSignalEncoder(3, 6)(
        signal, mask, sample_interval=0.01, amplitude_scale=2.5, kind="audio"
    )
    assert sensor.values.shape == (2, 9, 6)
    assert sensor.domain.sample_interval == 0.01 and sensor.domain.amplitude_scale == 2.5


def test_field_encoder_and_conservation_projection_enforce_domain_contracts():
    encoder = PhysicalFieldEncoder(2, 2, 1, 1, 8)
    shape = (2, 4, 3)
    values = torch.randn(*shape, 2)
    coordinates = torch.randn(*shape, 2)
    coefficients = torch.randn(*shape, 1)
    forcing = torch.randn(*shape, 1)
    valid = torch.ones(shape, dtype=torch.bool)
    boundary = torch.zeros(shape, dtype=torch.bool)
    boundary[:, 0] = True
    encoded = encoder(values, coordinates, coefficients, forcing, boundary, valid, torch.tensor([0.1, 0.2]))
    assert encoded.values.shape == (*shape, 8) and encoded.domain.boundary == "physical"
    projected = ConservationProjector()(torch.randn(*shape, 2), valid, boundary)
    assert (projected[:, 0] == 0).all()
    # With no boundary clamp the projector has exactly zero masked spatial integral.
    free = ConservationProjector()(torch.randn(*shape, 2), valid)
    torch.testing.assert_close(free.sum((1, 2)), torch.zeros(2, 2), atol=1e-6, rtol=1e-6)


def test_graph_polynomial_encoder_is_permutation_equivariant_without_eigendecomposition():
    torch.manual_seed(130)
    encoder = GraphPolynomialEncoder(3, 5, order=3).double()
    nodes = torch.randn(2, 6, 3, dtype=torch.float64)
    adjacency = torch.randint(0, 2, (2, 6, 6), dtype=torch.float64)
    adjacency = ((adjacency + adjacency.transpose(1, 2)) > 0).double()
    adjacency.diagonal(dim1=1, dim2=2).zero_()
    mask = torch.tensor([[True] * 6, [True] * 5 + [False]])
    reference = encoder(nodes, adjacency, mask).values
    permutation = torch.tensor([3, 0, 5, 2, 1, 4])
    permuted = encoder(
        nodes[:, permutation], adjacency[:, permutation][:, :, permutation], mask[:, permutation]
    ).values
    torch.testing.assert_close(permuted, reference[:, permutation], atol=1e-12, rtol=1e-12)


def test_set_encoder_is_permutation_invariant_and_rejects_false_frequency_order():
    torch.manual_seed(141)
    encoder = InducingPointSetEncoder(3, 6, 4).double()
    items = torch.randn(2, 7, 3, dtype=torch.float64)
    mask = torch.tensor([[True] * 7, [True] * 5 + [False] * 2])
    reference = encoder(items, mask)
    permutation = torch.tensor([5, 2, 0, 6, 1, 4, 3])
    actual = encoder(items[:, permutation], mask[:, permutation])
    torch.testing.assert_close(actual.values, reference.values, atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError):
        require_meaningful_order(actual.domain)
    require_meaningful_order(DomainSpec("sequence"))


def test_directional_spatial_resonance_and_full_field_network_keep_native_2d_shape_masks_and_gradients():
    torch.manual_seed(151)
    field = torch.randn(2, 5, 7, 4, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(2, 5, 7, dtype=torch.bool)
    mask[0, -1, -1] = False
    directional = DirectionalResonator2D(4, 1, 2, 1).double()
    mixed = directional(field, mask, spacing=(0.2, 0.1))
    assert mixed.shape == field.shape and (mixed[0, -1, -1] == 0).all()
    network = SpatialResonanceNetwork(4, 4, 3, levels=1, layers=1, heads=1, modes=2, mimo_rank=1).double()
    output = network(field, mask, spacing=(0.2, 0.1))
    assert output.shape == (2, 5, 7, 3) and (output[0, -1, -1] == 0).all()
    output.square().mean().backward()
    assert torch.isfinite(field.grad).all()


def test_spatial_resonance_contracts_fail_closed():
    with pytest.raises(ValueError):
        DirectionalResonator2D(0, 1, 1, 1)
    directional = DirectionalResonator2D(2, 1, 1, 1)
    with pytest.raises(ValueError):
        directional(torch.randn(1, 3, 3, 3), torch.ones(1, 3, 3, dtype=torch.bool))
    with pytest.raises(ValueError):
        directional(torch.randn(1, 3, 3, 2), torch.ones(1, 3, 3, dtype=torch.bool), spacing=(0, 1))
    with pytest.raises(ValueError):
        SpatialResonanceNetwork(0, 2, 1)
    network = SpatialResonanceNetwork(2, 2, 1, levels=1, layers=1, heads=1, modes=1)
    with pytest.raises(ValueError):
        network(torch.randn(1, 3, 3, 1))
    with pytest.raises(ValueError):
        network(torch.randn(1, 3, 3, 2), torch.ones(1, 3, 3))


def test_modality_contracts_fail_closed():
    with pytest.raises(ValueError):
        DomainSpec("unknown")
    with pytest.raises(ValueError):
        DomainSpec("audio", sample_interval=0)
    with pytest.raises(ValueError):
        DomainSpec("image", boundary="circular")
    with pytest.raises(ValueError):
        DomainSpec("set")
    with pytest.raises(ValueError):
        TokenEncoder(0, 2)
    with pytest.raises(ValueError):
        TokenEncoder(3, 2)(torch.randn(1, 2))
    with pytest.raises(ValueError):
        TokenEncoder(3, 2)(torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2))
    assert TokenEncoder(3, 2)(torch.ones(1, 2, dtype=torch.long)).mask.all()
    with pytest.raises(ValueError):
        ContinuousSignalEncoder(0, 2)
    with pytest.raises(ValueError):
        ContinuousSignalEncoder(2, 3)(torch.randn(1, 4, 3), torch.ones(1, 4, dtype=torch.bool), sample_interval=1)
    with pytest.raises(ValueError):
        ContinuousSignalEncoder(2, 3)(torch.randn(1, 4, 2), torch.ones(1, 4), sample_interval=1)
    assert ContinuousSignalEncoder(2, 3, anti_alias=False)(
        torch.randn(1, 4, 2), torch.ones(1, 4, dtype=torch.bool), sample_interval=1
    ).values.shape == (1, 4, 3)
    with pytest.raises(ValueError):
        ImageBand(torch.randn(1, 2, 3), torch.ones(1, 2, dtype=torch.bool), 0, "LH")
    with pytest.raises(ValueError):
        ImageBand(torch.randn(1, 2, 3, 1), torch.ones(1, 2, 3, dtype=torch.bool), -1, "bad")
    with pytest.raises(ValueError):
        SeparableLifting2D(0, 1)
    image = SeparableLifting2D(1, 1)
    with pytest.raises(ValueError):
        image(torch.randn(1, 3, 3, 2))
    bands, context = image(torch.randn(1, 3, 3, 1))
    with pytest.raises(ValueError):
        image(torch.randn(1, 3, 3, 1), torch.ones(1, 3, 3))
    with pytest.raises(ValueError):
        image.inverse(bands[:-1], context)
    wrong_order = list(bands)
    wrong_order[0] = ImageBand(wrong_order[0].data, wrong_order[0].mask, 0, "HH")
    with pytest.raises(ValueError):
        image.inverse(wrong_order, context)
    video = FactorizedVideoLifting(1, 1, 1)
    with pytest.raises(ValueError):
        video(torch.randn(1, 3, 3, 1))
    video_data = torch.randn(1, 2, 3, 3, 1)
    video_bands, video_context = video(video_data)
    with pytest.raises(ValueError):
        video(video_data, torch.ones(1, 2, 3, 3))
    with pytest.raises(ValueError):
        video.inverse(video_bands[:-1], video_context)
    with pytest.raises(ValueError):
        PhysicalFieldEncoder(0, 1, 0, 0, 2)
    graph = GraphPolynomialEncoder(2, 3)
    with pytest.raises(ValueError):
        graph(torch.randn(1, 3, 4), torch.eye(3), torch.ones(1, 3, dtype=torch.bool))
    field = PhysicalFieldEncoder(1, 1, 0, 0, 3)
    with pytest.raises(ValueError):
        field(
            torch.randn(1, 2, 1), torch.randn(1, 2, 1), torch.empty(1, 2, 0),
            torch.empty(1, 2, 0), torch.zeros(1, 2, dtype=torch.bool),
            torch.ones(1, 2, dtype=torch.bool), torch.tensor([-1.0]),
        )
    with pytest.raises(ValueError):
        field(torch.randn(1, 2, 2), torch.randn(1, 2, 1), torch.empty(1, 2, 0), torch.empty(1, 2, 0), torch.zeros(1, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool), torch.tensor([1.0]))
    with pytest.raises(ValueError):
        ConservationProjector()(torch.randn(1, 2, 1), torch.ones(1, 2))
    with pytest.raises(ValueError):
        ConservationProjector()(torch.randn(1, 2, 1), torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2))
    with pytest.raises(ValueError):
        GraphPolynomialEncoder(0, 2)
    graph2 = GraphPolynomialEncoder(2, 3)
    assert graph2(torch.randn(2, 3, 2), torch.eye(3), torch.ones(2, 3, dtype=torch.bool)).values.shape == (2, 3, 3)
    with pytest.raises(ValueError):
        graph2(torch.randn(1, 3, 2), torch.eye(2), torch.ones(1, 3, dtype=torch.bool))
    with pytest.raises(ValueError):
        InducingPointSetEncoder(0, 2, 1)
    set_encoder = InducingPointSetEncoder(2, 3, 2)
    empty_set = set_encoder(torch.randn(1, 4, 2), torch.zeros(1, 4, dtype=torch.bool))
    assert not empty_set.mask.any() and (empty_set.values == 0).all()
    with pytest.raises(ValueError):
        set_encoder(torch.randn(1, 4, 3), torch.ones(1, 4, dtype=torch.bool))
