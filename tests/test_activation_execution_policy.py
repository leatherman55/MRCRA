from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import mrrn.activation_execution as activation
from mrrn.activation_execution import (
    ActivationCandidateMeasurement,
    ActivationExecutionPolicy,
    MemoryObservation,
    calibrate_activation_candidates,
    census_activation_partitions,
    maximum_safe_retain_physical_tokens,
    resolve_activation_execution_policy,
    select_activation_dominant_partitions,
)


def measurement(
    policy: str,
    *,
    seconds: float,
    peak: int,
    finite: bool = True,
) -> ActivationCandidateMeasurement:
    return ActivationCandidateMeasurement(
        policy,
        seconds,
        peak,
        peak + 100,
        "a" * 64,
        finite,
    )


@pytest.fixture
def fixed_memory(monkeypatch):
    observed = MemoryObservation(
        "cpu",
        16 << 30,
        10 << 30,
        "test_available_memory",
    )
    monkeypatch.setattr(activation, "observe_memory", lambda _device: observed)
    monkeypatch.setattr(activation, "hardware_fingerprint", lambda _device: "b" * 64)
    return observed


def test_auto_policy_selects_fastest_finite_candidate_within_reserve(fixed_memory):
    result = resolve_activation_execution_policy(
        requested="auto",
        device=torch.device("cpu"),
        required_reserve_bytes=2 << 30,
        estimated_retain_bytes=5 << 30,
        candidates=(
            measurement("retain", seconds=1.0, peak=7 << 30),
            measurement("selective", seconds=1.4, peak=4 << 30),
            measurement("whole_span", seconds=2.0, peak=1 << 30),
        ),
    )
    assert result.resolved == "retain"
    assert "fastest measured" in result.reason
    assert result.memory == fixed_memory
    assert len(result.digest) == 64
    assert ActivationExecutionPolicy.from_dict(result.to_dict()) == result


def test_auto_policy_selects_selective_before_whole_span_when_retain_exceeds_reserve(
    fixed_memory,
):
    result = resolve_activation_execution_policy(
        requested="auto",
        device=torch.device("cpu"),
        required_reserve_bytes=4 << 30,
        estimated_retain_bytes=8 << 30,
        candidates=(
            measurement("retain", seconds=1.0, peak=7 << 30),
            measurement("selective", seconds=1.5, peak=5 << 30),
            measurement("whole_span", seconds=2.0, peak=1 << 30),
        ),
    )
    assert result.resolved == "selective"


def test_auto_policy_falls_back_to_whole_span_when_no_candidate_fits(
    fixed_memory,
):
    result = resolve_activation_execution_policy(
        requested="auto",
        device=torch.device("cpu"),
        required_reserve_bytes=9 << 30,
        estimated_retain_bytes=8 << 30,
        candidates=(
            measurement("retain", seconds=1.0, peak=7 << 30),
            measurement("selective", seconds=1.5, peak=5 << 30),
            measurement("whole_span", seconds=2.0, peak=2 << 30),
        ),
    )
    assert result.resolved == "whole_span"
    assert "safe fallback" in result.reason


def test_precalibration_memory_observation_prevents_allocator_cache_false_fallback(
    monkeypatch,
):
    before = MemoryObservation(
        "cpu", 16 << 30, 10 << 30, "precalibration_available_memory"
    )
    after = MemoryObservation(
        "cpu", 16 << 30, 2 << 30, "allocator_cache_depressed_memory"
    )
    monkeypatch.setattr(activation, "observe_memory", lambda _device: after)
    monkeypatch.setattr(
        activation, "hardware_fingerprint", lambda _device: "b" * 64
    )
    result = resolve_activation_execution_policy(
        requested="auto",
        device=torch.device("cpu"),
        required_reserve_bytes=2 << 30,
        estimated_retain_bytes=12 << 30,
        candidates=(
            measurement("retain", seconds=1.0, peak=9 << 30),
            measurement("selective", seconds=1.4, peak=5 << 30),
            measurement("whole_span", seconds=2.0, peak=1 << 30),
        ),
        memory_observation=before,
    )
    assert result.resolved == "selective"
    assert result.memory == before


def test_precalibration_memory_observation_is_device_bound(fixed_memory):
    with pytest.raises(ValueError, match="different device"):
        resolve_activation_execution_policy(
            requested="auto",
            device=torch.device("cpu"),
            required_reserve_bytes=0,
            estimated_retain_bytes=0,
            memory_observation=MemoryObservation(
                "mps", 16 << 30, 8 << 30, "wrong-device"
            ),
        )


def test_nonfinite_fast_candidate_is_not_authorized(fixed_memory):
    result = resolve_activation_execution_policy(
        requested="auto",
        device=torch.device("cpu"),
        required_reserve_bytes=1 << 30,
        estimated_retain_bytes=2 << 30,
        candidates=(
            measurement("retain", seconds=0.5, peak=1 << 30, finite=False),
            measurement("whole_span", seconds=2.0, peak=1 << 30),
        ),
    )
    assert result.resolved == "whole_span"


def test_measured_candidates_must_be_forward_equivalent(fixed_memory):
    candidates = (
        measurement("retain", seconds=1.0, peak=10),
        replace(
            measurement("whole_span", seconds=2.0, peak=5),
            output_digest="1" * 64,
        ),
    )
    with pytest.raises(RuntimeError, match="forward-equivalent"):
        resolve_activation_execution_policy(
            requested="auto",
            device=torch.device("cpu"),
            required_reserve_bytes=0,
            estimated_retain_bytes=0,
            candidates=candidates,
        )


def test_explicit_measured_policy_over_reserve_fails_closed(fixed_memory):
    with pytest.raises(MemoryError, match="exceeds"):
        resolve_activation_execution_policy(
            requested="retain",
            device=torch.device("cpu"),
            required_reserve_bytes=5 << 30,
            estimated_retain_bytes=7 << 30,
            candidates=(
                measurement("retain", seconds=1.0, peak=6 << 30),
            ),
        )


def test_estimate_only_policy_uses_live_available_memory(fixed_memory):
    retain = resolve_activation_execution_policy(
        requested="auto",
        device=torch.device("cpu"),
        required_reserve_bytes=2 << 30,
        estimated_retain_bytes=5 << 30,
    )
    whole = resolve_activation_execution_policy(
        requested="auto",
        device=torch.device("cpu"),
        required_reserve_bytes=6 << 30,
        estimated_retain_bytes=5 << 30,
    )
    assert retain.resolved == "retain"
    assert whole.resolved == "whole_span"


@pytest.mark.parametrize("requested", ("retain", "selective"))
def test_uncalibrated_explicit_memory_intensive_policy_fails_when_estimate_is_unsafe(
    fixed_memory, requested,
):
    with pytest.raises(MemoryError, match="conservative activation estimate"):
        resolve_activation_execution_policy(
            requested=requested,
            device=torch.device("cpu"),
            required_reserve_bytes=4 << 30,
            estimated_retain_bytes=7 << 30,
        )


def test_uncalibrated_explicit_whole_span_remains_safe_fallback(fixed_memory):
    result = resolve_activation_execution_policy(
        requested="whole_span",
        device=torch.device("cpu"),
        required_reserve_bytes=9 << 30,
        estimated_retain_bytes=9 << 30,
    )
    assert result.resolved == "whole_span"
    assert "recomputation" in result.reason


def test_explicit_unsafe_override_is_named_and_never_implicit(fixed_memory):
    result = resolve_activation_execution_policy(
        requested="retain",
        device=torch.device("cpu"),
        required_reserve_bytes=4 << 30,
        estimated_retain_bytes=7 << 30,
        allow_unsafe_explicit=True,
    )
    assert result.resolved == "retain"
    assert "unsafe override" in result.reason


def test_candidate_measurement_restores_global_rng_and_reports_finite_digest():
    torch.manual_seed(1701)
    before = torch.random.get_rng_state().clone()

    def candidate():
        return torch.randn(32).square().mean().reshape(1)

    report = calibrate_activation_candidates(
        {"retain": candidate},
        device=torch.device("cpu"),
    )[0]
    after = torch.random.get_rng_state()
    assert torch.equal(after, before)
    assert report.policy == "retain"
    assert report.finite
    assert report.elapsed_seconds >= 0
    assert report.absolute_peak_bytes > 0
    assert len(report.output_digest) == 64


def test_peak_memory_measurement_is_finite_nonnegative_and_device_named():
    report = calibrate_activation_candidates(
        {"retain": lambda: torch.ones(4096).square().sum().reshape(1)},
        device=torch.device("cpu"),
    )[0]
    policy = resolve_activation_execution_policy(
        requested="retain",
        device=torch.device("cpu"),
        required_reserve_bytes=0,
        estimated_retain_bytes=0,
        candidates=(report,),
    )
    assert report.finite
    assert report.incremental_peak_bytes >= 0
    assert report.absolute_peak_bytes >= 0
    assert policy.memory.device_type == "cpu"
    assert policy.memory.observation_kind == "host_available_memory"
    assert policy.memory.available_bytes >= 0


def test_candidate_peak_is_projected_to_largest_planned_physical_cohort(
    fixed_memory,
):
    measured = ActivationCandidateMeasurement(
        "retain",
        0.25,
        64 << 20,
        512 << 20,
        "4" * 64,
        True,
        calibration_physical_tokens=512,
        target_physical_tokens=8192,
        projected_incremental_peak_bytes=9 << 30,
    )
    assert measured.reserve_peak_bytes == 9 << 30
    with pytest.raises(MemoryError, match="reserve"):
        resolve_activation_execution_policy(
            requested="retain",
            device=torch.device("cpu"),
            required_reserve_bytes=3 << 30,
            estimated_retain_bytes=64 << 20,
            candidates=(measured,),
        )


def test_candidate_measurement_records_calibration_and_target_shape_projection():
    report = calibrate_activation_candidates(
        {"retain": lambda: torch.ones(1024).square()},
        device=torch.device("cpu"),
        calibration_physical_tokens=128,
        target_physical_tokens=1024,
        conservative_peak_bytes={"retain": 96 << 20},
    )[0]
    assert report.calibration_physical_tokens == 128
    assert report.target_physical_tokens == 1024
    assert report.projected_incremental_peak_bytes >= 96 << 20
    restored = ActivationExecutionPolicy.from_dict(
        resolve_activation_execution_policy(
            requested="retain",
            device=torch.device("cpu"),
            required_reserve_bytes=0,
            estimated_retain_bytes=0,
            candidates=(report,),
        ).to_dict()
    )
    assert restored.candidates[0] == report


def test_partition_census_measures_saved_tensors_instead_of_parameter_size():
    value = torch.linspace(-1, 1, 4096, dtype=torch.float64).requires_grad_()

    def retained():
        return value.square().sin()

    def checkpointed():
        return torch.utils.checkpoint.checkpoint(
            lambda item: item.square().sin(),
            value,
            use_reentrant=False,
        )

    before = torch.random.get_rng_state().clone()
    report = census_activation_partitions(
        retained,
        {"scale:0": checkpointed},
        device=torch.device("cpu"),
    )[0]
    assert torch.equal(torch.random.get_rng_state(), before)
    assert report.partition == "scale:0"
    assert report.saved_byte_reduction > 0
    assert (
        report.partition_checkpointed_saved_bytes
        < report.retained_saved_bytes
    )
    assert report.elapsed_seconds >= 0


def test_selective_partition_target_uses_reducible_bytes_not_retained_graph():
    reports = (
        activation.ActivationPartitionCensus(
            "scale:0", 3_000, 2_500, 500, 0.3, "a" * 64
        ),
        activation.ActivationPartitionCensus(
            "scale:1", 3_000, 2_700, 300, 0.2, "a" * 64
        ),
        activation.ActivationPartitionCensus(
            "scale:2", 3_000, 2_900, 100, 0.1, "a" * 64
        ),
    )
    # Sixty percent of the 900 reducible bytes is 540, so the two dominant
    # partitions are sufficient. Sixty percent of retained bytes (1,800)
    # would be impossible and would incorrectly select all three.
    assert select_activation_dominant_partitions(
        reports, reduction_fraction=0.60
    ) == ("scale:0", "scale:1")


def test_zero_reduction_census_retains_one_real_selective_candidate():
    reports = (
        activation.ActivationPartitionCensus(
            "scale:0", 100, 100, 0, 0.2, "a" * 64
        ),
        activation.ActivationPartitionCensus(
            "scale:1", 100, 100, 0, 0.1, "a" * 64
        ),
    )
    assert select_activation_dominant_partitions(reports) == ("scale:1",)


def test_shape_conditional_retain_limit_is_aligned_and_reserve_bounded(
    fixed_memory,
):
    policy = resolve_activation_execution_policy(
        requested="auto",
        device=torch.device("cpu"),
        required_reserve_bytes=2 << 30,
        estimated_retain_bytes=12 << 30,
        candidates=(
            replace(
                measurement("retain", seconds=1.0, peak=12 << 30),
                calibration_physical_tokens=2_048,
                target_physical_tokens=8_192,
                projected_incremental_peak_bytes=12 << 30,
            ),
            measurement("selective", seconds=1.5, peak=2 << 30),
        ),
    )
    assert policy.resolved == "selective"
    limit = maximum_safe_retain_physical_tokens(
        policy, alignment=128, maximum_physical_tokens=8_192
    )
    assert limit == 5_376
    assert limit % 128 == 0
    assert (
        (12 << 30) * limit / 8_192 + (2 << 30)
        <= fixed_memory.available_bytes
    )


def test_retain_limit_is_full_for_retain_and_zero_without_measurement(
    fixed_memory,
):
    retained = resolve_activation_execution_policy(
        requested="retain",
        device=torch.device("cpu"),
        required_reserve_bytes=0,
        estimated_retain_bytes=0,
    )
    assert maximum_safe_retain_physical_tokens(
        retained, alignment=128, maximum_physical_tokens=8_192
    ) == 8_192
    fallback = resolve_activation_execution_policy(
        requested="auto",
        device=torch.device("cpu"),
        required_reserve_bytes=9 << 30,
        estimated_retain_bytes=12 << 30,
    )
    assert maximum_safe_retain_physical_tokens(
        fallback, alignment=128, maximum_physical_tokens=8_192
    ) == 0


@pytest.mark.parametrize(
    "field,value",
    (
        ("requested", "invalid"),
        ("resolved", "invalid"),
        ("required_reserve_bytes", -1),
        ("hardware_fingerprint", "short"),
    ),
)
def test_serialized_policy_validation_fails_closed(
    fixed_memory, field, value,
):
    valid = resolve_activation_execution_policy(
        requested="retain",
        device=torch.device("cpu"),
        required_reserve_bytes=0,
        estimated_retain_bytes=0,
    )
    with pytest.raises(ValueError, match="malformed"):
        replace(valid, **{field: value})


def test_unknown_candidate_name_is_rejected_before_execution():
    called = False

    def candidate():
        nonlocal called
        called = True
        return torch.zeros(1)

    with pytest.raises(ValueError, match="unknown"):
        calibrate_activation_candidates(
            {"not-a-policy": candidate},
            device=torch.device("cpu"),
        )
    assert not called
