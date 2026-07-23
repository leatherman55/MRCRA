import pytest
import torch

from mrrn.lifting import ScaleTensor
from mrrn.scale_exchange import (
    ScaleExchange,
    _coarse_to_fine,
    _downsample,
    _fine_to_coarse,
    _upsample,
)
from mrrn.lifting import LiftingAnalysisBank


def band(length, width, scale, *, valid=True):
    return ScaleTensor(
        torch.randn(2, length, width),
        torch.full((2, length), valid, dtype=torch.bool),
        scale,
        float(2**scale),
        2 ** (scale + 1),
        "approximation" if scale == 2 else "detail",
    )


def test_neighbor_exchange_preserves_contracts_masks_and_has_gradients():
    module = ScaleExchange([4, 6, 8])
    bands = list((band(9, 4, 0), band(5, 6, 1), band(3, 8, 2)))
    bands[0] = ScaleTensor(
        bands[0].data.requires_grad_(),
        torch.tensor([[True] * 8 + [False], [True] * 9]),
        0,
        1.0,
        2,
    )
    result = module(bands)
    assert [item.data.shape for item in result] == [(2, 9, 4), (2, 5, 6), (2, 3, 8)]
    assert (result[0].data[0, -1] == 0).all()
    sum(item.data.square().mean() for item in result).backward()
    assert torch.isfinite(bands[0].data.grad).all()


def test_causal_coarse_context_never_changes_earlier_fine_position():
    module = ScaleExchange([3, 3], causal=True)
    fine = band(8, 3, 0)
    coarse = band(4, 3, 1)
    baseline = module((fine, coarse))[0].data
    changed_data = coarse.data.clone()
    changed_data[:, 2:] += 100
    changed = ScaleTensor(changed_data, coarse.mask, coarse.scale, coarse.sample_interval, coarse.support)
    actual = module((fine, changed))[0].data
    torch.testing.assert_close(actual[:, :5], baseline[:, :5])


def test_resampling_helpers_cover_empty_padding_and_noncausal_alignment():
    x = torch.arange(3.0).view(1, 3, 1)
    assert _downsample(x, 4).shape == (1, 4, 1)
    assert _downsample(x[:, :0], 2).shape == (1, 2, 1)
    assert _downsample(x, 0).shape == (1, 0, 1)
    torch.testing.assert_close(_upsample(x, 6, False), x.repeat_interleave(2, 1))
    causal = _upsample(x, 7, True)
    assert causal[0, 0, 0] == 0 and causal.shape == (1, 7, 1)
    assert _upsample(x[:, :0], 2, True).shape == (1, 2, 1)
    assert _upsample(x, 0, True).shape == (1, 0, 1)


def test_support_aware_resampling_uses_completion_times_and_masks():
    fine = torch.arange(1.0, 6.0).view(1, 5, 1)
    mask = torch.tensor([[True, False, True, True, True]])
    aggregated = _fine_to_coarse(
        fine, mask, 3, fine_support=2, coarse_support=4
    )
    torch.testing.assert_close(aggregated.flatten(), torch.tensor([1.0, 3.5, 5.0]))
    same_support = _fine_to_coarse(
        fine, mask, 5, fine_support=4, coarse_support=4
    )
    torch.testing.assert_close(same_support.flatten(), torch.tensor([1.0, 0.0, 3.0, 4.0, 5.0]))

    coarse = torch.tensor([[[10.0], [20.0], [30.0]]])
    causal = _coarse_to_fine(
        coarse, 6, coarse_support=4, fine_support=2, causal=True
    )
    torch.testing.assert_close(causal.flatten(), torch.tensor([0.0, 10.0, 10.0, 20.0, 20.0, 30.0]))
    noncausal = _coarse_to_fine(
        coarse, 6, coarse_support=4, fine_support=2, causal=False
    )
    torch.testing.assert_close(noncausal.flatten(), torch.tensor([10.0, 10.0, 20.0, 20.0, 30.0, 30.0]))


def test_support_aware_resampling_empty_zero_and_invalid_contracts():
    x = torch.ones(1, 2, 1)
    mask = torch.ones(1, 2, dtype=torch.bool)
    assert _fine_to_coarse(x, mask, 0, fine_support=2, coarse_support=4).shape[1] == 0
    assert _fine_to_coarse(x[:, :0], mask[:, :0], 2, fine_support=2, coarse_support=4).shape[1] == 2
    assert _coarse_to_fine(x, 0, coarse_support=4, fine_support=2, causal=True).shape[1] == 0
    assert _coarse_to_fine(x[:, :0], 2, coarse_support=4, fine_support=2, causal=True).shape[1] == 2
    with pytest.raises(ValueError):
        _fine_to_coarse(x, mask, 2, fine_support=4, coarse_support=2)
    with pytest.raises(ValueError):
        _coarse_to_fine(x, 2, coarse_support=2, fine_support=4, causal=True)


def test_equal_support_final_approximation_does_not_leak_future_index():
    module = ScaleExchange([3, 3], causal=True).eval()
    fine = ScaleTensor(torch.randn(1, 3, 3), torch.ones(1, 3, dtype=torch.bool), 1, 2.0, 4)
    coarse = ScaleTensor(
        torch.randn(1, 3, 3), torch.ones(1, 3, dtype=torch.bool), 2, 4.0, 4, "approximation"
    )
    baseline = module((fine, coarse))
    changed_fine = fine.data.clone()
    changed_fine[:, 2] += 100
    changed = module((ScaleTensor(changed_fine, fine.mask, 1, 2.0, 4), coarse))
    torch.testing.assert_close(changed[1].data[:, :2], baseline[1].data[:, :2])


def test_stream_exchange_matches_batch_at_every_completed_coefficient():
    torch.manual_seed(51)
    lifting = LiftingAnalysisBank(3, 2).double()
    exchange = ScaleExchange([3, 3, 3], causal=True).double()
    x = torch.randn(2, 13, 3, dtype=torch.float64)
    mask = torch.rand(2, 13) > 0.15
    batch_bands, _ = lifting(x, mask)
    reference = exchange(batch_bands)
    lift_state = lifting.initial_stream_state(2, dtype=torch.float64)
    exchange_state = exchange.initial_stream_state()
    collected = [[] for _ in reference]
    for position in range(x.shape[1]):
        active, lift_state = lifting.push(x[:, position], lift_state, mask[:, position])
        active, exchange_state = exchange.step(active, exchange_state)
        for scale, band_value in enumerate(active):
            if band_value is not None:
                collected[scale].append(band_value.data)
    for scale, band_value in enumerate(reference):
        completed = x.shape[1] // band_value.support
        actual = torch.cat(collected[scale], 1) if collected[scale] else x[:, :0]
        torch.testing.assert_close(actual, band_value.data[:, :completed], atol=1e-12, rtol=1e-12)
    detached = exchange_state.detach()
    assert all(value is None or not value.requires_grad for value in detached.latest_coarse)


def test_aligned_chunk_exchange_matches_stream_across_chunk_boundaries():
    torch.manual_seed(771)
    lifting = LiftingAnalysisBank(3, 2).double()
    exchange = ScaleExchange([3, 3, 3], causal=True).double()
    for module in (lifting, exchange):
        for parameter in module.parameters():
            parameter.data.normal_(std=0.09)
    x = torch.randn(2, 32, 3, dtype=torch.float64)
    mask = torch.rand(2, 32) > 0.15
    lift_chunk = lifting.initial_stream_state(2, dtype=torch.float64)
    exchange_chunk = exchange.initial_stream_state()
    lift_stream = lifting.initial_stream_state(2, dtype=torch.float64)
    exchange_stream = exchange.initial_stream_state()
    chunk_rows = [[] for _ in range(3)]
    stream_rows = [[] for _ in range(3)]

    for start in (0, 16):
        bands, lift_chunk = lifting.push_aligned_chunk(
            x[:, start : start + 16], lift_chunk, mask[:, start : start + 16],
        )
        exchanged, exchange_chunk = exchange.forward_aligned_chunk(
            bands, exchange_chunk,
        )
        for scale, band_value in enumerate(exchanged):
            chunk_rows[scale].append(band_value.data)

    for position in range(x.shape[1]):
        active, lift_stream = lifting.push(
            x[:, position], lift_stream, mask[:, position],
        )
        active, exchange_stream = exchange.step(active, exchange_stream)
        for scale, band_value in enumerate(active):
            if band_value is not None:
                stream_rows[scale].append(band_value.data)

    for chunk_values, stream_values in zip(chunk_rows, stream_rows, strict=True):
        torch.testing.assert_close(
            torch.cat(chunk_values, 1), torch.cat(stream_values, 1),
            atol=2e-12, rtol=2e-12,
        )
    for chunk_latest, stream_latest in zip(
        exchange_chunk.latest_coarse, exchange_stream.latest_coarse, strict=True,
    ):
        torch.testing.assert_close(chunk_latest, stream_latest, atol=2e-12, rtol=2e-12)
    assert all(not values for values in exchange_chunk.fine_values)
    assert all(not masks for masks in exchange_chunk.fine_masks)


def test_aligned_chunk_exchange_rejects_partial_support_or_pending_fine_state():
    module = ScaleExchange([2, 2])
    fine = ScaleTensor(
        torch.randn(1, 4, 2), torch.ones(1, 4, dtype=torch.bool), 0, 1.0, 2,
    )
    coarse = ScaleTensor(
        torch.randn(1, 1, 2), torch.ones(1, 1, dtype=torch.bool), 1, 2.0, 4,
        "approximation",
    )
    with pytest.raises(ValueError, match="same original-domain"):
        module.forward_aligned_chunk((fine, coarse), module.initial_stream_state())
    state = module.initial_stream_state()
    state.fine_values[0].append(torch.randn(1, 1, 2))
    state.fine_masks[0].append(torch.ones(1, 1, dtype=torch.bool))
    complete_coarse = ScaleTensor(
        torch.randn(1, 2, 2), torch.ones(1, 2, dtype=torch.bool), 1, 2.0, 4,
        "approximation",
    )
    with pytest.raises(ValueError, match="empty fine"):
        module.forward_aligned_chunk((fine, complete_coarse), state)


def test_stream_exchange_contract_validation():
    module = ScaleExchange([2, 2])
    state = module.initial_stream_state()
    with pytest.raises(ValueError):
        module.step((None,), state)
    wrong = module.initial_stream_state()
    wrong.fine_values.clear()
    with pytest.raises(ValueError):
        module.step((None, None), wrong)
    invalid = ScaleTensor(torch.randn(1, 2, 2), torch.ones(1, 2, dtype=torch.bool), 0, 1.0, 2)
    with pytest.raises(ValueError):
        module.step((invalid, None), state)
    coarse = ScaleTensor(torch.randn(1, 1, 2), torch.ones(1, 1, dtype=torch.bool), 1, 2.0, 4, "approximation")
    active, _ = module.step((None, coarse), module.initial_stream_state())
    assert active[1] is not None


def test_invalid_scale_exchange_contracts():
    with pytest.raises(ValueError):
        ScaleExchange([])
    module = ScaleExchange([2, 3])
    with pytest.raises(ValueError):
        module((band(4, 2, 0),))
    with pytest.raises(ValueError):
        module((band(4, 4, 0), band(2, 3, 1)))
