import pytest
import torch

from mrrn import complex_ops as c
from mrrn.diagnostics import (
    attention_diagnostics,
    estimate_local_lipschitz,
    memory_diagnostics,
    mode_diagnostics,
    scale_diagnostics,
    stability_diagnostics,
)
from mrrn.lifting import ScaleTensor
from mrrn.memory import EideticMemory, MemoryItem
from mrrn.resonance import ComplexResonator, ResonatorState


def test_mode_report_contains_physical_frequency_decay_quality_phase_and_occupancy():
    torch.manual_seed(160)
    resonator = ComplexResonator(4, 2, 3, 1).double()
    x = torch.randn(2, 6, 4, dtype=torch.float64)
    _, state, parameters = resonator(x)
    gradient = torch.ones_like(state.value)
    report = mode_diagnostics(parameters, state, gradient=gradient)
    assert report.frequency.shape == report.decay.shape == (2, 3)
    assert (report.half_life > 0).all() and (report.quality_factor >= 0).all()
    assert ((report.phase_entropy >= 0) & (report.phase_entropy <= 1 + 1e-6)).all()
    assert ((report.phase_locking >= 0) & (report.phase_locking <= 1 + 1e-6)).all()
    assert report.gradient_norm is not None and report.transition_max.max() < 1
    with pytest.raises(ValueError):
        mode_diagnostics(parameters, state, gradient=torch.ones(1, 2))


def test_scale_report_energy_fractions_gates_reconstruction_and_ablation_delta():
    bands = tuple(
        ScaleTensor(
            torch.full((1, 3 - index, 2), float(index + 1)),
            torch.ones(1, 3 - index, dtype=torch.bool), index, float(2**index), 2 ** (index + 1),
        )
        for index in range(2)
    )
    report = scale_diagnostics(
        bands, reconstruction_error=torch.tensor(1e-7), fine_gains=torch.tensor([0.1]),
        coarse_gains=torch.tensor([0.2]), ablated_losses=torch.tensor([2.0, 3.0]),
        full_loss=torch.tensor(1.0),
    )
    torch.testing.assert_close(report.energy_fraction.sum(), torch.tensor(1.0))
    torch.testing.assert_close(report.ablation_delta, torch.tensor([1.0, 2.0]))
    assert report.supports.tolist() == [2, 4]


def test_attention_report_separates_candidate_tiers_and_lag_error():
    weights = torch.tensor([[[[0.6], [0.3], [0.1]]]])
    kinds = torch.tensor([[[0, 1, 2]]])
    report = attention_diagnostics(
        weights, kinds=kinds, selected_lag=torch.tensor([4]), true_lag=torch.tensor([3]),
        candidate_indices=torch.tensor([[[0, 2]]]), dense_weights=weights.squeeze(-1),
    )
    torch.testing.assert_close(report.kind_mass, torch.tensor([0.6, 0.3, 0.1]))
    assert report.selected_lag_error == 1 and report.maximum_weight == 0.6
    assert report.candidate_set_miss_rate == 0 and report.band_mass is None
    with pytest.raises(ValueError):
        attention_diagnostics(torch.ones(2, 3))


def test_memory_report_measures_capacity_eviction_recall_write_rate_and_age():
    memory = EideticMemory(2, 2, 2, 2)
    handles = []
    for timestamp in range(3):
        vector = torch.tensor([1.0, float(timestamp)])
        handles.append(memory.write(MemoryItem(vector, vector, vector, timestamp, 0, 1.0)))
    query = torch.tensor([1.0, 2.0])
    retrieved = memory.retrieve(query, 2, query_time=5)
    report = memory_diagnostics(
        memory, processed_positions=100, retrieved=retrieved, oracle=retrieved, query_time=5
    )
    assert report.occupancy == 1 and report.eviction_rate == pytest.approx(1 / 3)
    assert report.writes_per_thousand == 30 and report.router_recall == 1


def test_stability_report_and_local_lipschitz_detect_finite_gain_and_phase_drift():
    state = c.pair(torch.ones(2, 3), torch.zeros(2, 3))
    report = stability_diagnostics([state], [torch.zeros(2, 3)])
    assert report.all_finite and report.maximum == 1 and report.phase_drift == 0
    matrix = torch.diag(torch.tensor([2.0, 0.5]))
    estimate = estimate_local_lipschitz(lambda value: value @ matrix, torch.ones(2, 2), directions=12)
    assert 1.5 < estimate <= 2.01
    with pytest.raises(ValueError):
        stability_diagnostics([])
    with pytest.raises(ValueError):
        estimate_local_lipschitz(lambda value: value, torch.ones(1), epsilon=0)


def test_diagnostic_contract_failures():
    band = ScaleTensor(torch.ones(1, 2, 1), torch.ones(1, 2, dtype=torch.bool), 0, 1, 2)
    with pytest.raises(ValueError):
        scale_diagnostics([], reconstruction_error=torch.tensor(0.0), fine_gains=torch.tensor([]), coarse_gains=torch.tensor([]))
    with pytest.raises(ValueError):
        scale_diagnostics([band], reconstruction_error=torch.tensor(0.0), fine_gains=torch.tensor([]), coarse_gains=torch.tensor([]), ablated_losses=torch.ones(2), full_loss=torch.tensor(1.0))
    report = scale_diagnostics([band], reconstruction_error=torch.tensor(0.0), fine_gains=torch.tensor([]), coarse_gains=torch.tensor([]))
    assert report.cross_scale_gate_magnitude == 0
    memory = EideticMemory(1, 1, 1, 1)
    assert memory_diagnostics(memory, processed_positions=0).retrieved_age_mean == 0
    with pytest.raises(ValueError):
        memory_diagnostics(memory, processed_positions=-1)
    with pytest.raises(ValueError):
        memory_diagnostics(memory, processed_positions=1, retrieved=[])
    resonator = ComplexResonator(2, 1, 1, 1)
    _, state, parameters = resonator(torch.randn(1, 2, 2))
    with pytest.raises(ValueError):
        mode_diagnostics(parameters, state, dead_threshold=-1)
    band_weights = torch.ones(1, 1, 2, 1, 3) / 2
    band_report = attention_diagnostics(band_weights)
    assert band_report.kind_mass[0] == 1
    torch.testing.assert_close(band_report.band_mass, torch.full((3,), 1 / 3))
    with pytest.raises(ValueError):
        attention_diagnostics(torch.ones(1, 1, 2, 1), kinds=torch.ones(1, 1, dtype=torch.long))
    with pytest.raises(ValueError):
        attention_diagnostics(torch.ones(1, 1, 2, 1), selected_lag=torch.ones(1))
    with pytest.raises(ValueError):
        attention_diagnostics(torch.ones(1, 1, 2, 1), candidate_indices=torch.zeros(1, 1, 1, dtype=torch.long))
    with pytest.raises(ValueError):
        attention_diagnostics(
            torch.ones(1, 1, 2, 1), candidate_indices=torch.zeros(2, 1, 1, dtype=torch.long),
            dense_weights=torch.ones(1, 1, 2),
        )
    state_value = c.pair(torch.ones(1, 2), torch.zeros(1, 2))
    with pytest.raises(ValueError):
        stability_diagnostics([state_value], [])
    with pytest.raises(ValueError):
        stability_diagnostics([state_value], [torch.zeros(1, 3)])
