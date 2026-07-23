import pytest
import torch

from mrrn import complex_ops as c
from mrrn.config import MRRNConfig
from mrrn.lifting import LiftingAnalysisBank
from mrrn.modalities import SeparableLifting2D
from mrrn.model import MRRN
from mrrn.resonance import ComplexResonator, ResonatorState


def extrapolation_config():
    return MRRNConfig(
        input_dim=2, model_dim=4, output_dim=2, layers=1, scales=2, heads=1,
        modes=2, mimo_rank=1, attention_window=3, retrieved_items=1,
        memory_capacity=2, mixer_expansion=1, width_growth_cap=1,
        mode_growth_cap=1, width_multiple=1,
    )


@pytest.mark.parametrize("length", [1, 3, 16, 65, 257])
@pytest.mark.parametrize("sample_interval", [0.01, 1.0, 10.0])
def test_network_lengths_and_sampling_intervals_outside_nominal_grid_remain_finite_and_stream_exact(length, sample_interval):
    torch.manual_seed(length)
    model = MRRN(extrapolation_config()).double().eval()
    x = torch.randn(1, length, 2, dtype=torch.float64)
    with torch.no_grad():
        batch = model(x, sample_interval=sample_interval).prediction
        state = model.initial_stream_state(1, sample_interval=sample_interval, dtype=torch.float64)
        streamed = []
        for position in range(length):
            result = model.step(x[:, position], state)
            state = result.state
            streamed.append(result.prediction.unsqueeze(1))
    assert torch.isfinite(batch).all()
    torch.testing.assert_close(torch.cat(streamed, 1), batch, atol=2e-10, rtol=2e-10)


def test_homogeneous_resonator_respects_physical_duration_under_resampling():
    torch.manual_seed(301)
    model = ComplexResonator(3, 1, 2, 1).double()
    for layer in (model.input_projection, model.input_direction):
        layer.weight.data.zero_()
        layer.bias.data.zero_()
    initial = model.initial_state(1, dtype=torch.float64)
    initial = ResonatorState(torch.randn_like(initial.value), initial.previous_drive)
    coarse = model.sequential(torch.zeros(1, 10, 3, dtype=torch.float64), initial, sample_interval=1.0)[1]
    fine = model.sequential(torch.zeros(1, 20, 3, dtype=torch.float64), initial, sample_interval=0.5)[1]
    torch.testing.assert_close(fine.value, coarse.value, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("height,width", [(3, 19), (17, 5), (32, 33)])
def test_spatial_transform_generalizes_across_unseen_aspect_ratios_and_boundaries(height, width):
    transform = SeparableLifting2D(1, 3).double()
    image = torch.randn(1, height, width, 1, dtype=torch.float64)
    bands, context = transform(image)
    torch.testing.assert_close(transform.inverse(bands, context), image, atol=2e-12, rtol=2e-12)


def test_chunk_boundaries_do_not_change_lifting_coefficients_or_model_outputs():
    torch.manual_seed(317)
    bank = LiftingAnalysisBank(2, 3).double()
    x = torch.randn(1, 73, 2, dtype=torch.float64)
    reference, _ = bank(x)
    state = bank.initial_stream_state(1, dtype=torch.float64)
    emitted = [[] for _ in reference]
    for chunk in x.split([7, 13, 1, 31, 21], dim=1):
        for position in range(chunk.shape[1]):
            active, state = bank.push(chunk[:, position], state)
            for scale, item in enumerate(active):
                if item is not None:
                    emitted[scale].append(item.data)
    for scale, band in enumerate(reference):
        count = x.shape[1] // band.support
        actual = torch.cat(emitted[scale], 1) if emitted[scale] else band.data[:, :0]
        torch.testing.assert_close(actual, band.data[:, :count], atol=1e-12, rtol=1e-12)


def test_long_zero_input_and_bounded_noise_do_not_grow_or_produce_nonfinite_state():
    torch.manual_seed(331)
    model = ComplexResonator(4, 1, 3, 1, alpha_min=1e-3).float()
    initial = model.initial_state(1)
    initial = ResonatorState(torch.randn_like(initial.value), initial.previous_drive)
    zero = torch.zeros(1, 4096, 4)
    for layer in (model.input_projection, model.input_direction):
        layer.weight.data.zero_()
        layer.bias.data.zero_()
    _, state, _ = model.sequential(zero, initial)
    assert torch.isfinite(state.value).all()
    assert c.magnitude(state.value).max() <= c.magnitude(initial.value).max()


def test_invalid_extrapolation_metadata_fails_instead_of_guessing():
    model = MRRN(extrapolation_config())
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 2), sample_interval=0)
