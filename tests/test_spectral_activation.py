from dataclasses import replace

import pytest
import torch

from mrrn import complex_ops as c
from mrrn.config import MRRNConfig
from mrrn.diagnostics import alias_energy_fraction, spectral_activation_diagnostics
from mrrn.evaluation import apply_ablation, parameter_statistics
from mrrn.mixer import (
    GatedLocalMixer,
    AntiAliasActivation,
    HybridMixerDiagnostics,
    HybridSpectralMixer,
    ResonantSpectralGLU,
    SpectralActivationDiagnostics,
    chebyshev_basis,
)
from mrrn.model import MRRN
from mrrn.objectives import spectral_activation_regularization


def tiny_config(**overrides):
    values = dict(
        input_dim=3, model_dim=8, output_dim=2, layers=1, scales=3,
        heads=2, modes=4, mimo_rank=2, attention_window=3,
        retrieved_items=1, memory_capacity=2, mixer_expansion=1,
        width_growth_cap=1, mode_growth_cap=1, width_multiple=2,
        spectral_modes=3, spectral_basis_order=4,
    )
    values.update(overrides)
    return MRRNConfig(**values)


def test_chebyshev_recurrence_matches_closed_form_terms_and_is_differentiable():
    x = torch.tensor([-1.0, -0.25, 0.5, 1.0], requires_grad=True)
    basis = chebyshev_basis(x, 5)
    expected = torch.stack((
        torch.ones_like(x), x, 2 * x.square() - 1,
        4 * x.pow(3) - 3 * x, 8 * x.pow(4) - 8 * x.square() + 1,
    ), -1)
    torch.testing.assert_close(basis, expected)
    basis.sum().backward()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(chebyshev_basis(x.detach(), 1), torch.ones(4, 1))


def test_default_spectral_gate_is_zero_preserving_bounded_phase_neutral_and_frequency_legal():
    torch.manual_seed(401)
    module = ResonantSpectralGLU(8, 2, 5, 2).double()
    zero = torch.zeros(2, 7, 8, dtype=torch.float64, requires_grad=True)
    output, report = module.forward_with_diagnostics(zero)
    assert output.count_nonzero() == 0
    assert report.amplitude_gate.count_nonzero() == 0
    assert report.phase_rotation.count_nonzero() == 0
    assert report.triad.count_nonzero() == 0
    output.sum().backward()
    assert torch.isfinite(zero.grad).all()
    assert module.triad_frequency_error() < 2e-7


def test_learned_transfer_is_mode_specific_bounded_and_receives_finite_gradients():
    torch.manual_seed(409)
    module = ResonantSpectralGLU(
        6, 2, 4, 1, basis_order=5, maximum_gain=2.5,
        maximum_phase=0.3, triads_per_mode=3,
    ).double()
    module.gain_coefficients.data[0, 1, 1] = 1.5
    module.phase_coefficients.data[1, 2, 2] = -1.0
    module.raw_triad_weight.data.fill_(0.25)
    x = torch.randn(2, 9, 6, dtype=torch.float64, requires_grad=True)
    output, diagnostic = module.forward_with_diagnostics(x)
    assert output.shape == x.shape and torch.isfinite(output).all()
    assert diagnostic.amplitude_gate.min() >= 0
    assert diagnostic.amplitude_gate.max() <= module.maximum_gain + 1e-12
    assert diagnostic.phase_rotation.abs().max() <= module.maximum_phase + 1e-12
    assert diagnostic.triad.abs().sum() > 0
    output.square().mean().backward()
    for parameter in module.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_activations_keep_transfer_phase_and_triad_math_in_fp32(dtype):
    module = ResonantSpectralGLU(4, 1, 3, 1).to(dtype)
    x = torch.randn(1, 5, 4).to(dtype)
    output, diagnostic = module.forward_with_diagnostics(x)
    assert output.dtype == dtype and torch.isfinite(output).all()
    assert diagnostic.amplitude_gate.dtype == torch.float32
    assert diagnostic.phase_rotation.dtype == torch.float32
    assert diagnostic.triad.dtype == torch.float32


def test_modal_gate_is_equivariant_to_carrier_phase_and_invariant_to_control_phase_without_triads():
    torch.manual_seed(419)
    module = ResonantSpectralGLU(4, 1, 3, 2, triads_per_mode=2).double()
    control = torch.randn(2, 5, 1, 3, 2, 2, dtype=torch.float64)
    carrier = torch.randn_like(control)
    angle = torch.tensor(0.73, dtype=torch.float64)
    baseline, _ = module.modal_activation(control, carrier)
    shifted, _ = module.modal_activation(c.rotate(control, -1.2 * angle), c.rotate(carrier, angle))
    torch.testing.assert_close(shifted, c.rotate(baseline, angle), atol=2e-12, rtol=2e-12)


def test_sparse_triads_obey_sum_or_difference_frequency_and_are_amplitude_bounded():
    module = ResonantSpectralGLU(
        3, 1, 6, 1, triads_per_mode=4, maximum_triad_gain=0.2
    ).double()
    module.raw_triad_weight.data.fill_(20)
    control = c.pair(torch.full((1, 1, 1, 6, 1), 1e6), torch.zeros(1, 1, 1, 6, 1)).double()
    carrier = control.clone()
    _, diagnostics = module.modal_activation(control, carrier)
    target = module.frequencies[module.triad_target]
    routed = torch.where(
        module.triad_conjugate,
        module.frequencies[module.triad_left] - module.frequencies[module.triad_right],
        module.frequencies[module.triad_left] + module.frequencies[module.triad_right],
    )
    torch.testing.assert_close(routed, target, atol=2e-7, rtol=2e-7)
    assert torch.isfinite(diagnostics.triad).all()
    assert c.magnitude(diagnostics.triad).max().item() <= 4 * module.maximum_triad_gain + 1e-6


def test_nonzero_triads_are_equivariant_to_physical_time_translation_by_constructed_frequency():
    torch.manual_seed(420)
    module = ResonantSpectralGLU(4, 1, 5, 1, triads_per_mode=4).double()
    module.raw_triad_weight.data.uniform_(-0.7, 0.7)
    control = torch.randn(2, 3, 1, 5, 1, 2, dtype=torch.float64)
    carrier = torch.randn_like(control)
    baseline, _ = module.modal_activation(control, carrier)
    angle = (module.frequencies * 1.37).view(1, 1, 1, 5, 1)
    shifted, _ = module.modal_activation(c.rotate(control, angle), c.rotate(carrier, angle))
    torch.testing.assert_close(shifted, c.rotate(baseline, angle), atol=3e-7, rtol=3e-7)


def test_no_triad_configuration_is_exact_and_regularizable_without_empty_reductions():
    module = ResonantSpectralGLU(
        4, 1, 1, 1, triads_per_mode=0, maximum_phase=0,
    ).double()
    assert module.triad_frequency_error() == 0
    output, diagnostic = module.forward_with_diagnostics(torch.randn(1, 3, 4, dtype=torch.float64))
    assert output.shape == (1, 3, 4) and diagnostic.triad.count_nonzero() == 0
    assert spectral_activation_regularization([module]) == 0
    hybrid = HybridSpectralMixer(
        4, 1, 1, 1, 1,
        spectral_kwargs={"triads_per_mode": 0, "maximum_phase": 0},
    ).double()
    _, details = hybrid.forward_with_diagnostics(torch.randn(1, 3, 4, dtype=torch.float64))
    report = spectral_activation_diagnostics(details, hybrid.spectral)
    assert report.phase_utilization == 0
    assert report.gain_transfer_roughness == 0


def test_hybrid_mixer_can_select_exact_conventional_or_spectral_endpoint():
    torch.manual_seed(421)
    mixer = HybridSpectralMixer(6, 2, 2, 3, 1).double()
    x = torch.randn(2, 5, 6, dtype=torch.float64)
    with torch.no_grad():
        mixer.blend.weight.zero_()
        mixer.blend.bias.fill_(-100)
        ordinary = mixer(x)
        expected_ordinary = mixer.conventional(x)
        mixer.blend.bias.fill_(100)
        spectral = mixer(x)
        expected_spectral = mixer.spectral(x)
    torch.testing.assert_close(ordinary, expected_ordinary, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(spectral, expected_spectral, atol=1e-14, rtol=1e-14)


def test_continuous_signal_alias_filter_suppresses_high_frequency_hybrid_activation_energy():
    torch.manual_seed(423)
    mixer = HybridSpectralMixer(4, 2, 1, 4, 1)
    alternating = (torch.arange(128) % 2 * 2 - 1).float().view(1, 128, 1).repeat(1, 1, 4)
    raw = mixer(alternating)
    filtered = AntiAliasActivation(4)(raw)
    before, after = alias_energy_fraction(raw), alias_energy_fraction(filtered)
    assert before > 1e-3 and after < before / 20


def test_network_integrates_scale_limited_spectral_modes_diagnostics_and_exact_streaming():
    torch.manual_seed(431)
    model = MRRN(tiny_config()).double().eval()
    assert all(isinstance(mixer, HybridSpectralMixer) for mixer in model.blocks[0].mixers)
    assert all(mixer.spectral.modes == 3 for mixer in model.blocks[0].mixers)
    for mixer, resonator in zip(model.blocks[0].mixers, model.blocks[0].resonators, strict=True):
        assert mixer.spectral.frequencies.max() <= resonator.omega_max
    x = torch.randn(1, 13, 3, dtype=torch.float64)
    with torch.no_grad():
        batch = model(x)
        state = model.initial_stream_state(1, dtype=torch.float64)
        streamed = []
        for position in range(x.shape[1]):
            step = model.step(x[:, position], state)
            state = step.state
            streamed.append(step.prediction.unsqueeze(1))
    torch.testing.assert_close(torch.cat(streamed, 1), batch.prediction, atol=2e-10, rtol=2e-10)
    diagnostic = batch.diagnostics[0].spectral_mixers[0]
    assert diagnostic is not None
    report = spectral_activation_diagnostics(diagnostic, model.blocks[0].mixers[0].spectral)
    assert report.all_finite and 0 < report.spectral_fraction_mean < 1
    assert report.triad_frequency_error < 2e-7


def test_spectral_regularizer_prefers_smooth_inactive_transfer_and_backpropagates():
    first = ResonantSpectralGLU(4, 1, 4, 1)
    second = ResonantSpectralGLU(4, 1, 4, 1)
    baseline = spectral_activation_regularization([first, second])
    assert baseline == 0
    first.gain_coefficients.data[:, 1::2] = 2
    first.phase_coefficients.data[:, 2] = 1
    first.raw_triad_weight.data.fill_(1)
    penalized = spectral_activation_regularization([first, second])
    assert penalized > baseline
    penalized.backward()
    assert first.gain_coefficients.grad is not None
    assert first.phase_coefficients.grad is not None
    assert first.raw_triad_weight.grad is not None


@pytest.mark.parametrize(
    "variant", ["no_spectral_activation", "spectral_only_local", "fixed_spectral_activation"]
)
def test_spectral_ablation_variants_are_executable_and_isolated(variant):
    full = MRRN(tiny_config())
    ablated = apply_ablation(full, variant)
    assert ablated is not full
    assert ablated(torch.randn(1, 7, 3)).prediction.shape == (1, 7, 2)
    mixer = ablated.blocks[0].mixers[0]
    if variant == "no_spectral_activation":
        assert isinstance(mixer, GatedLocalMixer)
    elif variant == "spectral_only_local":
        assert isinstance(mixer, ResonantSpectralGLU)
    else:
        assert not mixer.spectral.gain_coefficients.requires_grad
        assert not mixer.spectral.phase_coefficients.requires_grad


def test_structural_path_ablations_remove_unused_parameters_instead_of_masking_them():
    full = MRRN(tiny_config())
    conventional = apply_ablation(full, "no_spectral_activation")
    spectral = apply_ablation(full, "spectral_only_local")
    full_count = parameter_statistics(full)[0]
    assert parameter_statistics(conventional)[0] < full_count
    assert parameter_statistics(spectral)[0] < full_count


def test_disabling_extension_retains_original_local_mixer_contract():
    model = MRRN(replace(tiny_config(), spectral_activation=False))
    assert all(isinstance(mixer, GatedLocalMixer) for mixer in model.blocks[0].mixers)
    output = model(torch.randn(1, 6, 3))
    assert output.prediction.shape == (1, 6, 2)
    assert all(item is None for item in output.diagnostics[0].spectral_mixers)


def test_spectral_activation_contracts_fail_closed():
    with pytest.raises(ValueError):
        chebyshev_basis(torch.ones(2), 0)
    for constructor in (
        lambda: ResonantSpectralGLU(0, 1, 1, 1),
        lambda: ResonantSpectralGLU(2, 1, 1, 1, maximum_gain=1),
        lambda: ResonantSpectralGLU(2, 1, 1, 1, maximum_phase=4),
        lambda: ResonantSpectralGLU(2, 1, 1, 1, triads_per_mode=-1),
        lambda: ResonantSpectralGLU(2, 1, 1, 1, frequency_max=4),
    ):
        with pytest.raises(ValueError):
            constructor()
    module = ResonantSpectralGLU(4, 1, 2, 1)
    with pytest.raises(ValueError):
        module(torch.ones(2, 3))
    control = torch.ones(1, 2, 1, 2, 1, 2)
    with pytest.raises(ValueError):
        module.modal_activation(control, control[..., :1])
    wrong_modes = torch.ones(1, 2, 1, 1, 1, 2)
    with pytest.raises(ValueError):
        module.modal_activation(wrong_modes, wrong_modes)
    with pytest.raises(ValueError):
        module.modal_activation(control, control, torch.ones(1, 2, 1, 2, 3))
    with pytest.raises(ValueError):
        spectral_activation_regularization([])
    with pytest.raises(ValueError):
        alias_energy_fraction(torch.ones(1, 1, 2))
    with pytest.raises(ValueError):
        alias_energy_fraction(torch.ones(1, 3, 2), cutoff_fraction=1)
    hybrid = HybridSpectralMixer(4, 1, 1, 2, 1)
    with pytest.raises(ValueError):
        hybrid.forward_with_diagnostics(torch.ones(2, 3))
    _, valid = hybrid.forward_with_diagnostics(torch.ones(1, 2, 4))
    malformed = HybridMixerDiagnostics(
        valid.spectral_fraction,
        SpectralActivationDiagnostics(
            valid.spectral.amplitude_gate[..., :1, :], valid.spectral.phase_rotation,
            valid.spectral.triad,
        ),
    )
    with pytest.raises(ValueError):
        spectral_activation_diagnostics(malformed, hybrid.spectral)
    malformed_blend = HybridMixerDiagnostics(valid.spectral_fraction[:, :1], valid.spectral)
    with pytest.raises(ValueError):
        spectral_activation_diagnostics(malformed_blend, hybrid.spectral)
    disabled = MRRN(replace(tiny_config(), spectral_activation=False))
    assert apply_ablation(disabled, "no_spectral_activation")(
        torch.randn(1, 4, 3)
    ).prediction.shape == (1, 4, 2)
