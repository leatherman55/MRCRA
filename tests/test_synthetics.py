import json

import pytest
import torch

from mrrn import complex_ops as c
from mrrn.synthetics import (
    affine_state_tracking,
    bounded_noise_rollout,
    delayed_match,
    learned_spectral_activation_separation,
    multisine_recovery,
    phase_collision_margin,
    regime_switch_adaptation,
    run_capability_suite,
    save_capability_report,
    selective_copy_accuracy,
    transient_trend_separation,
)


def test_every_required_synthetic_capability_probe_passes_its_explicit_threshold(tmp_path):
    results = run_capability_suite()
    assert len(results) == 12
    assert all(result.passed for result in results), [result for result in results if not result.passed]
    path = tmp_path / "capabilities.json"
    save_capability_report(path, results)
    payload = json.loads(path.read_text())
    assert payload["all_passed"] and len(payload["results"]) == 12


def test_multisine_amplitude_phase_and_slow_frequency_recovery_are_numerically_exact():
    signal, amplitude, phase = multisine_recovery()
    torch.testing.assert_close(amplitude, torch.tensor([1.0, 0.6, 0.3], dtype=torch.float64), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(phase, torch.tensor([0.2, -0.7, 1.1], dtype=torch.float64), atol=1e-12, rtol=1e-12)
    assert signal.shape == (512,)


def test_affine_complex_scan_represents_nested_depth_parity_and_modular_counters():
    events = torch.tensor([[1.0, 1.0, -1.0, 1.0, -1.0]], dtype=torch.float64)
    torch.testing.assert_close(affine_state_tracking(events), events.cumsum(1), atol=1e-12, rtol=1e-12)
    parity = affine_state_tracking(torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0]], dtype=torch.float64), modulus=2)
    expected = torch.tensor([[1.0, -1.0, 1.0, 1.0, -1.0]], dtype=torch.float64)
    torch.testing.assert_close(c.real(parity), expected, atol=1e-12, rtol=1e-12)
    modulo_three = affine_state_tracking(torch.ones(1, 12, dtype=torch.float64), modulus=3)
    torch.testing.assert_close(c.magnitude(modulo_three), torch.ones(1, 12, dtype=torch.float64), atol=1e-12, rtol=1e-12)


def test_transients_delays_phase_collisions_copy_regime_switch_and_long_noise_are_distinct_capabilities():
    fine, coarse = transient_trend_separation()
    assert fine > 1 and coarse > 0
    motif = torch.tensor([[1.0, -2.0, 3.0, 0.5]])
    query = torch.zeros(1, 64)
    query[:, 37:41] = motif
    lag, correlation = delayed_match(query, motif)
    assert lag == 37 and correlation.numel() == 67
    assert phase_collision_margin(torch.tensor([1.0, 3.0, -2.0, 0.5, 4.0])) > 0.1
    assert selective_copy_accuracy(distance=1_000_000) == 1
    fixed, selective = regime_switch_adaptation()
    assert selective < fixed
    assert bounded_noise_rollout(length=20_000) <= 1
    assert learned_spectral_activation_separation() > 0.5


def test_synthetic_probe_contracts_fail_closed():
    with pytest.raises(ValueError):
        multisine_recovery(0)
    with pytest.raises(ValueError):
        affine_state_tracking(torch.ones(3), modulus=2)
    with pytest.raises(ValueError):
        affine_state_tracking(torch.ones(1, 2), modulus=1)
    with pytest.raises(ValueError):
        transient_trend_separation(4, 5)
    with pytest.raises(ValueError):
        phase_collision_margin(torch.ones(2))
    with pytest.raises(ValueError):
        selective_copy_accuracy(0)
    with pytest.raises(ValueError):
        regime_switch_adaptation(4, 4)
    with pytest.raises(ValueError):
        bounded_noise_rollout(0)
