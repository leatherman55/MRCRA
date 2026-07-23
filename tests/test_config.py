import pytest

from mrrn.config import (
    MRRNConfig,
    ScaleConfig,
    choose_scale_count,
    decay_limits,
    memory_capacity,
    multiresolution_cost_factor,
    usable_frequency_limit,
)


def test_default_configuration_and_geometric_scale_allocation():
    config = MRRNConfig(input_dim=7, model_dim=32, scales=4, width_multiple=8)
    scales = config.scale_configs()
    assert len(scales) == 4
    assert [s.width for s in scales] == [32, 48, 64, 64]
    assert [s.modes for s in scales] == [16, 23, 32, 32]
    assert all(s.heads == 4 and s.mimo_rank == 2 for s in scales)
    assert config.resolved_output_dim == 7
    assert MRRNConfig(input_dim=7, output_dim=3).resolved_output_dim == 3


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"input_dim": 0}, "input_dim"),
        ({"input_dim": 1, "layers": -1}, "layers"),
        ({"input_dim": 1, "output_dim": 0}, "output_dim"),
        ({"input_dim": 1, "lifting_kernel": 4}, "odd"),
        ({"input_dim": 1, "memory_capacity": 3, "retrieved_items": 4}, "memory_capacity"),
        ({"input_dim": 1, "omega_max": 4.0}, "Nyquist"),
        ({"input_dim": 1, "alpha_min": 0.0}, "alpha_min"),
        ({"input_dim": 1, "spectral_modes": 0}, "spectral_modes"),
        ({"input_dim": 1, "spectral_maximum_gain": 1.0}, "maximum_gain"),
        ({"input_dim": 1, "spectral_maximum_phase": 4.0}, "maximum_phase"),
        ({"input_dim": 1, "spectral_triads_per_mode": -1}, "triad"),
        ({"input_dim": 1, "spectral_maximum_triad_gain": -0.1}, "triad"),
    ],
)
def test_invalid_model_configurations_fail_closed(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MRRNConfig(**kwargs)


def test_invalid_scale_configuration():
    with pytest.raises(ValueError, match="modes"):
        ScaleConfig(width=8, heads=1, modes=0, mimo_rank=1, attention_window=1)


def test_research_capability_and_efficiency_profiles_encode_the_stated_tradeoffs():
    research = MRRNConfig.research(5, model_dim=128)
    capability = MRRNConfig.capability_first(5)
    efficient = MRRNConfig.efficiency_first(5)
    assert research.model_dim == 128 and research.memory_capacity == 2048
    assert capability.mimo_rank == 4 and capability.scales == 7 and capability.spectral_modes == 16
    assert efficient.share_depth_parameters and efficient.structured_mixer_rank == 32
    assert efficient.spectral_basis_order == 4 and efficient.spectral_triads_per_mode == 1


def test_hyperparameter_decision_rules_follow_physical_span_half_life_nyquist_cost_and_write_rate():
    assert choose_scale_count(1, 128, 4096, maximum_scales=8) == 6
    minimum, maximum = decay_limits(2, 1024, 0.5)
    assert minimum < maximum
    assert usable_frequency_limit(0.5, 0) == pytest.approx(2 * 3.141592653589793)
    assert memory_capacity(0.1, 1000, burst_factor=1.5) == 150
    assert multiresolution_cost_factor(0.5) < 3.42
    with pytest.raises(ValueError):
        choose_scale_count(2, 1, 10, maximum_scales=3)
    with pytest.raises(ValueError):
        decay_limits(10, 2, 1)
    with pytest.raises(ValueError):
        usable_frequency_limit(1, 1)
    with pytest.raises(ValueError):
        memory_capacity(-1, 2)
    with pytest.raises(ValueError):
        multiresolution_cost_factor(1)
