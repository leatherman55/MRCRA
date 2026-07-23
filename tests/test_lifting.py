import pytest
import torch

from mrrn.lifting import CausalDepthwiseAffine, LiftingAnalysisBank, ScaleTensor
from mrrn.lifting import ReconstructionLevel


@pytest.mark.parametrize("length", [1, 2, 3, 8, 15, 32])
def test_exact_roundtrip_for_odd_even_and_short_lengths(length):
    torch.manual_seed(length)
    bank = LiftingAnalysisBank(channels=3, levels=4, kernel_size=3).double()
    for parameter in bank.parameters():
        parameter.data.normal_(std=0.08)
    x = torch.randn(2, length, 3, dtype=torch.float64)
    bands, context = bank(x)
    reconstructed = bank.inverse(bands, context)
    torch.testing.assert_close(reconstructed, x, atol=2e-12, rtol=2e-12)
    assert len(bands) == 5
    assert bands[-1].kind == "approximation"
    assert bands[0].coefficient_interval == 2.0
    assert context.original_length == length


def test_impulses_at_every_boundary_roundtrip_and_causal_filter_does_not_look_ahead():
    bank = LiftingAnalysisBank(channels=1, levels=3, kernel_size=5).double()
    for parameter in bank.parameters():
        parameter.data.uniform_(-0.2, 0.2)
    for location in range(9):
        impulse = torch.zeros(1, 9, 1, dtype=torch.float64)
        impulse[0, location, 0] = 1
        bands, context = bank(impulse)
        torch.testing.assert_close(bank.inverse(bands, context), impulse, atol=1e-12, rtol=1e-12)

    layer = CausalDepthwiseAffine(1, 3).double()
    layer.depth.weight.data.fill_(1)
    prefix = torch.randn(1, 5, 1, dtype=torch.float64)
    changed_future = torch.cat((prefix, torch.randn(1, 4, 1, dtype=torch.float64)), 1)
    original_future = torch.cat((prefix, torch.zeros(1, 4, 1, dtype=torch.float64)), 1)
    torch.testing.assert_close(layer(changed_future)[:, :5], layer(original_future)[:, :5])


def test_initial_analysis_is_the_known_perfect_reconstruction_haar_split():
    bank = LiftingAnalysisBank(1, 1).double()
    x = torch.tensor([[[2.0], [4.0], [6.0], [10.0]]], dtype=torch.float64)
    bands, context = bank(x)
    torch.testing.assert_close(bands[0].data.flatten(), torch.tensor([2.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(bands[1].data.flatten(), torch.tensor([3.0, 8.0], dtype=torch.float64))
    torch.testing.assert_close(bank.inverse(bands, context), x)


@pytest.mark.parametrize("boundary", ["reflect", "physical"])
def test_offline_boundary_filters_are_real_and_still_invert_exactly(boundary):
    torch.manual_seed(17)
    bank = LiftingAnalysisBank(2, 3, kernel_size=5).double()
    for parameter in bank.parameters():
        parameter.data.normal_(std=0.1)
    x = torch.randn(1, 11, 2, dtype=torch.float64)
    bands, context = bank(x, boundary=boundary)
    torch.testing.assert_close(bank.inverse(bands, context), x, atol=2e-12, rtol=2e-12)
    causal, _ = bank(x, boundary="causal")
    assert not torch.allclose(causal[0].data, bands[0].data)


def test_masks_and_metadata_are_explicitly_propagated():
    bank = LiftingAnalysisBank(2, 2)
    x = torch.randn(1, 5, 2)
    mask = torch.tensor([[True, True, False, True, True]])
    bands, context = bank(x, mask, sample_interval=0.25, boundary="physical")
    assert bands[0].mask.tolist() == [[True, False]]
    assert bands[1].mask.tolist() == [[False]]
    assert bands[-1].mask.tolist() == [[False, True]]
    assert [band.sample_interval for band in bands] == [0.25, 0.5, 1.0]
    assert context.boundary == "physical"
    torch.testing.assert_close(bank.inverse(bands, context), x)


def test_roundtrip_error_and_gradients():
    bank = LiftingAnalysisBank(2, 3).double()
    x = torch.randn(2, 11, 2, dtype=torch.float64, requires_grad=True)
    error = bank.roundtrip_error(x)
    assert error < 1e-12
    bands, context = bank(x)
    loss = bank.inverse(bands, context).square().mean()
    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_invalid_inputs_and_context_fail_closed():
    with pytest.raises(ValueError):
        LiftingAnalysisBank(2, 0)
    with pytest.raises(ValueError):
        CausalDepthwiseAffine(0, 3)
    layer = CausalDepthwiseAffine(2, 3)
    with pytest.raises(ValueError):
        layer(torch.randn(2, 4, 3))
    with pytest.raises(ValueError):
        layer(torch.randn(2, 4, 2), boundary="periodic")
    assert layer(torch.empty(2, 0, 2)).shape == (2, 0, 2)

    bank = LiftingAnalysisBank(2, 2)
    with pytest.raises(ValueError):
        bank(torch.randn(2, 4, 3))
    with pytest.raises(ValueError):
        bank(torch.randn(2, 4, 2), sample_interval=0)
    with pytest.raises(ValueError):
        bank(torch.randn(2, 4, 2), boundary="circular")
    with pytest.raises(ValueError):
        bank(torch.randn(2, 4, 2), torch.ones(2, 4))
    bands, context = bank(torch.randn(2, 4, 2))
    with pytest.raises(ValueError):
        bank.inverse(bands[:-1], context)
    assert ReconstructionLevel(3, 1).has_tail and not ReconstructionLevel(4, 2).has_tail


def test_scale_tensor_validates_contract():
    data = torch.randn(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)
    ScaleTensor(data, mask, 0, 1.0, 2)
    bad_cases = [
        (data[0], mask, 0, 1.0, 2, "detail"),
        (data, mask.float(), 0, 1.0, 2, "detail"),
        (data, mask, -1, 1.0, 2, "detail"),
        (data, mask, 0, 0.0, 2, "detail"),
        (data, mask, 0, 1.0, 2, "other"),
    ]
    for args in bad_cases:
        with pytest.raises(ValueError):
            ScaleTensor(*args)


@pytest.mark.parametrize("length", [1, 2, 3, 7, 8, 17])
def test_binary_carry_stream_emits_exact_completed_batch_coefficients(length):
    torch.manual_seed(31 + length)
    bank = LiftingAnalysisBank(2, 3, kernel_size=3).double()
    for parameter in bank.parameters():
        parameter.data.normal_(std=0.1)
    x = torch.randn(2, length, 2, dtype=torch.float64)
    mask = torch.rand(2, length) > 0.2
    batch, _ = bank(x, mask, sample_interval=0.25)
    state = bank.initial_stream_state(2, sample_interval=0.25, dtype=torch.float64)
    emitted = [[] for _ in batch]
    emitted_masks = [[] for _ in batch]
    for index in range(length):
        updates, state = bank.push(x[:, index], state, mask[:, index])
        for scale, update in enumerate(updates):
            if update is not None:
                emitted[scale].append(update.data)
                emitted_masks[scale].append(update.mask)
    for scale, reference in enumerate(batch):
        completed = length // reference.support
        actual = torch.cat(emitted[scale], 1) if emitted[scale] else x[:, :0]
        actual_mask = torch.cat(emitted_masks[scale], 1) if emitted_masks[scale] else mask[:, :0]
        torch.testing.assert_close(actual, reference.data[:, :completed])
        assert torch.equal(actual_mask, reference.mask[:, :completed])
    detached = state.detach()
    assert detached.steps == length and all(not item.requires_grad for item in detached.even_history)


def test_aligned_chunk_lifting_matches_every_stream_emission_and_carry():
    torch.manual_seed(911)
    bank = LiftingAnalysisBank(5, 3, kernel_size=3).double()
    for parameter in bank.parameters():
        parameter.data.normal_(std=0.08)
    values = torch.randn(2, 32, 5, dtype=torch.float64)
    mask = torch.ones(2, 32, dtype=torch.bool)
    chunk_state = bank.initial_stream_state(2, dtype=torch.float64)
    stream_state = bank.initial_stream_state(2, dtype=torch.float64)

    chunk_bands, chunk_state = bank.push_aligned_chunk(
        values, chunk_state, mask,
    )
    streamed = [[] for _ in chunk_bands]
    for index in range(values.shape[1]):
        active, stream_state = bank.push(
            values[:, index], stream_state, mask[:, index],
        )
        for scale, band in enumerate(active):
            if band is not None:
                streamed[scale].append(band.data)

    for scale, band in enumerate(chunk_bands):
        torch.testing.assert_close(
            band.data, torch.cat(streamed[scale], 1),
            atol=1e-12, rtol=1e-12,
        )
    for chunk_history, stream_history in zip(
        chunk_state.even_history, stream_state.even_history, strict=True,
    ):
        torch.testing.assert_close(
            chunk_history, stream_history, atol=1e-12, rtol=1e-12,
        )
    for chunk_history, stream_history in zip(
        chunk_state.detail_history, stream_state.detail_history, strict=True,
    ):
        torch.testing.assert_close(
            chunk_history, stream_history, atol=1e-12, rtol=1e-12,
        )
    assert chunk_state.emitted == stream_state.emitted
    assert chunk_state.steps == stream_state.steps == 32
    assert chunk_state.pending == stream_state.pending == [None, None, None]


def test_aligned_chunk_lifting_rejects_incomplete_intermediate_spans():
    bank = LiftingAnalysisBank(3, 3)
    state = bank.initial_stream_state(1)
    with pytest.raises(ValueError, match="align"):
        bank.push_aligned_chunk(torch.randn(1, 7, 3), state)


def test_stream_lifting_contracts_fail_closed():
    bank = LiftingAnalysisBank(2, 2)
    with pytest.raises(ValueError):
        bank.initial_stream_state(0)
    state = bank.initial_stream_state(2)
    with pytest.raises(ValueError):
        bank.push(torch.randn(2, 1), state)
    with pytest.raises(ValueError):
        bank.push(torch.randn(2, 2), state, torch.ones(2))
    wrong = bank.initial_stream_state(1)
    with pytest.raises(ValueError):
        bank.push(torch.randn(2, 2), wrong)
