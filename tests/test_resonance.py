import math

import mpmath
import pytest
import torch

from mrrn import complex_ops as c
from mrrn.resonance import (
    ComplexResonator,
    ResonatorState,
    associative_affine_scan,
    effective_horizon_steps,
    half_life_steps,
    uniform_state_bound,
)


def make_resonator(dtype=torch.float64):
    torch.manual_seed(4)
    return ComplexResonator(8, 2, 4, 2, maximum_half_life=128).to(dtype=dtype)


def test_parameter_contract_stability_bounds_and_mimo_normalization():
    model = make_resonator()
    u = torch.randn(3, 7, 8, dtype=torch.float64)
    params = model.parameterize(u)
    assert params.transition.shape == (3, 7, 2, 4, 2, 2)
    assert params.drive.shape == params.transition.shape
    assert params.readout.shape == params.transition.shape
    assert (params.alpha >= model.alpha_min).all()
    assert (params.omega.abs() <= model.omega_max).all()
    assert (params.delta >= model.delta_min).all()
    torch.testing.assert_close(
        c.abs_squared(params.readout).mean(dim=3),
        torch.ones(3, 7, 2, 2, dtype=torch.float64),
        atol=1e-5,
        rtol=1e-5,
    )
    assert c.magnitude(params.transition).max() < 1


def test_physical_sample_interval_scales_effective_step_and_accepts_per_position_values():
    model = make_resonator()
    u = torch.randn(2, 3, 8, dtype=torch.float64)
    unit = model.parameterize(u)
    doubled = model.parameterize(u, sample_interval=2.0)
    torch.testing.assert_close(doubled.delta, 2 * unit.delta)
    varying = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]], dtype=torch.float64)
    params = model.parameterize(u, sample_interval=varying)
    expected = varying[..., None, None, None].expand_as(params.delta)
    torch.testing.assert_close(params.delta / unit.delta, expected)
    with pytest.raises(ValueError):
        model.parameterize(u, sample_interval=0.0)
    with pytest.raises(ValueError):
        model.parameterize(u, sample_interval=torch.ones(3))


def test_parallel_affine_scan_matches_direct_native_complex_recurrence():
    torch.manual_seed(9)
    transition = torch.randn(2, 11, 3, 2, dtype=torch.float64) * 0.2
    transition[..., 0] += 0.7
    drive = torch.randn_like(transition) * 0.1
    initial = torch.randn(2, 3, 2, dtype=torch.float64)
    actual = associative_affine_scan(transition, drive, initial)
    current = c.to_native(initial)
    reference = []
    for index in range(transition.shape[1]):
        current = c.to_native(transition[:, index]) * current + c.to_native(drive[:, index])
        reference.append(current)
    torch.testing.assert_close(c.to_native(actual), torch.stack(reference, 1), atol=1e-12, rtol=1e-12)


def test_exponential_trapezoid_scalar_step_matches_high_precision_continuous_integration():
    decay, frequency, delta = 0.3, 1.2, 0.7
    initial, previous_drive, current_drive = 0.2 - 0.1j, 0.4 + 0.2j, -0.1 + 0.5j
    lam = -decay + 1j * frequency
    q = torch.tensor([delta * lam.real, delta * lam.imag], dtype=torch.float64)
    phi1, phi2 = c.phi_functions(q)
    affine = delta * ((c.to_native(phi1 - phi2).item() * previous_drive) + c.to_native(phi2).item() * current_drive)
    actual = mpmath.exp(delta * lam) * initial + affine
    reference = mpmath.exp(delta * lam) * initial + mpmath.quad(
        lambda s: mpmath.exp((delta - s) * lam)
        * (previous_drive + (s / delta) * (current_drive - previous_drive)),
        [0, delta],
    )
    assert abs(actual - reference) < 1e-13


def test_euler_ablation_is_available_but_less_accurate_than_exponential_homogeneous_step():
    model = make_resonator()
    for layer in (model.input_projection, model.input_direction):
        layer.weight.data.zero_()
        layer.bias.data.zero_()
    u = torch.zeros(1, 1, 8, dtype=torch.float64)
    initial = model.initial_state(1, dtype=torch.float64)
    initial = ResonatorState(torch.randn_like(initial.value), initial.previous_drive)
    _, exact, exact_parameters = model.sequential(u, initial, sample_interval=1.5)
    _, euler, _ = model.euler(u, initial, sample_interval=1.5)
    analytic = c.multiply(exact_parameters.transition[:, 0], initial.value)
    exact_error = (exact.value - analytic).abs().max()
    euler_error = (euler.value - analytic).abs().max()
    assert exact_error < 1e-12 and euler_error > exact_error + 1e-6


@pytest.mark.parametrize("masked", [False, True])
def test_parallel_and_sequential_resonators_are_equivalent(masked):
    model = make_resonator()
    u = torch.randn(2, 13, 8, dtype=torch.float64)
    initial = model.initial_state(2, dtype=torch.float64)
    initial = ResonatorState(torch.randn_like(initial.value) * 0.1, torch.randn_like(initial.value) * 0.1, 5)
    mask = None
    if masked:
        mask = torch.tensor([[True] * 9 + [False] * 4, [True, False] * 6 + [True]])
    sequential = model.sequential(u, initial, mask)
    parallel = model.parallel(u, initial, mask)
    torch.testing.assert_close(parallel[0], sequential[0], atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(parallel[1].value, sequential[1].value, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(parallel[1].previous_drive, sequential[1].previous_drive)
    assert parallel[1].steps == sequential[1].steps == 18


def test_masked_steps_do_not_change_state_and_have_zero_output():
    model = make_resonator()
    u = torch.randn(1, 4, 8, dtype=torch.float64)
    initial = model.initial_state(1, dtype=torch.float64)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    output, state, _ = model(u, initial, mask)
    torch.testing.assert_close(output, torch.zeros_like(output))
    torch.testing.assert_close(state.value, initial.value)
    torch.testing.assert_close(state.previous_drive, initial.previous_drive)


def test_zero_drive_stable_rotation_never_increases_state_magnitude():
    model = make_resonator()
    for layer in (model.input_projection, model.input_direction):
        layer.weight.data.zero_()
        layer.bias.data.zero_()
    u = torch.zeros(1, 64, 8, dtype=torch.float64)
    state = model.initial_state(1, dtype=torch.float64)
    state = ResonatorState(torch.randn_like(state.value), state.previous_drive)
    _, final, params = model.sequential(u, state)
    assert c.magnitude(final.value).max() <= c.magnitude(state.value).max()
    assert c.magnitude(params.transition).max() < 1


def test_decay_normalized_drive_bounds_long_lived_mode_gain_without_changing_scan():
    normalized = ComplexResonator(
        1, 1, 1, 1, decay_normalized_drive=True
    )
    legacy = ComplexResonator(
        1, 1, 1, 1, decay_normalized_drive=False
    )
    with torch.no_grad():
        normalized.raw_alpha.fill_(-10)
        normalized.raw_omega.zero_()
        normalized.input_projection.weight.fill_(1)
        normalized.input_projection.bias.zero_()
        normalized.input_direction.weight.zero_()
        normalized.input_direction.bias.copy_(torch.tensor([1.0, 0.0]))
    legacy.load_state_dict(normalized.state_dict())
    u = torch.ones(1, 4096, 1)
    _, normalized_state, _ = normalized(u)
    _, legacy_state, _ = legacy(u)
    assert c.magnitude(normalized_state.value).max() < 1
    assert c.magnitude(legacy_state.value).max() > 1_000
    sequential, sequential_state, _ = normalized.sequential(u)
    parallel, parallel_state, _ = normalized.parallel(u)
    torch.testing.assert_close(sequential, parallel, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(sequential_state.value, parallel_state.value, rtol=2e-4, atol=2e-5)


def test_half_life_effective_horizon_and_uniform_bibs_bound_match_closed_forms():
    alpha = torch.tensor([0.1, 0.2], dtype=torch.float64)
    torch.testing.assert_close(half_life_steps(alpha, 0.5), torch.log(torch.tensor(2.0)) / (alpha * 0.5))
    horizon = effective_horizon_steps(alpha, 1e-3, 0.5)
    torch.testing.assert_close(torch.exp(-alpha * 0.5 * horizon), torch.full_like(alpha, 1e-3))
    assert uniform_state_bound(0.9, 0.2, 1.0) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        half_life_steps(torch.tensor([0.0]))
    with pytest.raises(ValueError):
        effective_horizon_steps(alpha, 1.0)
    with pytest.raises(ValueError):
        uniform_state_bound(1.0, 1.0)


def test_content_changes_poles_and_all_gradients_are_finite():
    model = make_resonator()
    model.alpha_projection.weight.data.normal_(std=0.1)
    model.omega_projection.weight.data.normal_(std=0.1)
    u = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    output, state, params = model(u)
    assert not torch.allclose(params.alpha[:, 0], params.alpha[:, 1])
    assert not torch.allclose(params.omega[:, 0], params.omega[:, 1])
    (output.square().mean() + state.value.square().mean()).backward()
    assert torch.isfinite(u.grad).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_low_precision_projections_keep_poles_phase_normalization_and_recurrent_state_in_fp32():
    model = ComplexResonator(4, 1, 3, 1).to(dtype=torch.bfloat16)
    u = torch.randn(1, 32, 4, dtype=torch.bfloat16)
    output, state, parameters = model(u)
    assert output.dtype == torch.bfloat16
    assert state.value.dtype == parameters.transition.dtype == parameters.alpha.dtype == torch.float32
    assert torch.isfinite(state.value).all() and torch.isfinite(output.float()).all()


def test_streamed_chunks_equal_one_batch_and_state_detach():
    model = make_resonator()
    u = torch.randn(2, 12, 8, dtype=torch.float64)
    whole_output, whole_state, _ = model.sequential(u)
    state = None
    chunks = []
    for part in u.split([3, 5, 4], dim=1):
        output, state, _ = model.sequential(part, state)
        chunks.append(output)
    torch.testing.assert_close(torch.cat(chunks, 1), whole_output, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(state.value, whole_state.value, atol=2e-11, rtol=2e-11)
    detached = state.detach()
    assert not detached.value.requires_grad and detached.steps == state.steps


def test_invalid_resonator_and_scan_inputs_fail_closed():
    with pytest.raises(ValueError):
        ComplexResonator(0, 1, 1, 1)
    with pytest.raises(ValueError):
        ComplexResonator(1, 1, 1, 1, omega_max=math.pi + 0.1)
    model = make_resonator()
    with pytest.raises(ValueError):
        model.initial_state(0)
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 7, dtype=torch.float64))
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 8, dtype=torch.float64), mask=torch.ones(2, 3))
    with pytest.raises(ValueError):
        model.parameterize(torch.randn(2, 3, 8, dtype=torch.float64), torch.randn(2, 3))
    bad_state = ResonatorState(torch.randn(2, 3, 2), torch.randn(2, 3, 2))
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 8, dtype=torch.float64), bad_state)
    with pytest.raises(ValueError):
        associative_affine_scan(torch.randn(2, 3, 2), torch.randn(2, 4, 2), torch.randn(2, 2))
    with pytest.raises(ValueError):
        associative_affine_scan(torch.randn(2, 3, 2), torch.randn(2, 3, 2), torch.randn(3, 2))
