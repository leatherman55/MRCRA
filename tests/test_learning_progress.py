from dataclasses import replace
import copy

import pytest

from mrrn.learning_progress import (
    LearningProgressAuthority, LearningProgressConfig, robust_line,
)


def config(**updates):
    base = LearningProgressConfig(
        observation_interval=1,
        warmup_observations=4,
        fast_window=3,
        baseline_min_observations=4,
        baseline_window=12,
        baseline_lag=0,
        baseline_freeze_observations=2,
        deadband_standard_deviations=0.0,
        minimum_slope_noise=1e-5,
        minimum_ce_noise=1e-5,
        guard_regression_patience=2,
        guard_recovery_patience=2,
    )
    return replace(base, **updates)


def feed(authority, losses, *, start=1_000_000, stride=1_000_000):
    reports = []
    for index, loss in enumerate(losses):
        reports.append(
            authority.observe(start + index * stride, loss, learning_rate=1e-4)
        )
    return reports


def test_robust_line_resists_one_large_outlier():
    _, slope, scale = robust_line(
        [0, 1, 2, 3, 4, 5],
        [5, 4, 30, 2, 1, 0],
        huber_delta=1.5,
        iterations=12,
        scale_floor=1e-6,
    )
    assert slope == pytest.approx(-1, abs=0.08)
    assert scale < 0.2


def test_progress_authority_rewards_faster_decline_and_penalizes_plateau():
    improving = LearningProgressAuthority(config())
    improving.observe_guard(5.1)
    improving_reports = feed(
        improving,
        [5.0, 4.7, 4.45, 4.20, 3.82, 3.45, 3.10, 2.80],
    )
    assert improving_reports[-1].baseline_ready
    assert improving_reports[-1].observed_slope_per_million_tokens < 0
    assert improving_reports[-1].pressure > 0

    plateau = LearningProgressAuthority(config())
    plateau_reports = feed(
        plateau,
        [5.0, 4.7, 4.45, 4.20, 4.19, 4.205, 4.20, 4.21],
    )
    assert plateau_reports[-1].pressure < 0
    assert plateau_reports[-1].progress_debt_nats_per_token > 0


def test_rising_ce_can_never_receive_positive_pressure():
    authority = LearningProgressAuthority(config())
    reports = feed(
        authority,
        [5.0, 4.7, 4.45, 4.20, 4.25, 4.30, 4.35, 4.40],
    )
    assert reports[-1].observed_slope_per_million_tokens > 0
    assert reports[-1].pressure <= 0


def test_positive_pressure_is_fail_closed_until_guard_reference_exists():
    authority = LearningProgressAuthority(config())
    reports = feed(
        authority,
        [5.0, 4.7, 4.45, 4.20, 3.82, 3.45, 3.10, 2.80],
    )
    assert reports[-1].raw_pressure <= 0
    assert reports[-1].pressure <= 0
    assert not reports[-1].guard_allows_positive_pressure
    assert authority.observe_guard(3.0)
    report = authority.observe(9_000_000, 2.55, learning_rate=1e-4)
    assert report.guard_allows_positive_pressure
    assert report.pressure > 0


def test_slope_and_debt_prevent_regress_then_drop_gaming():
    authority = LearningProgressAuthority(config(
        baseline_lag=4, debt_weight=0.75, slope_weight=0.25,
    ))
    reports = feed(
        authority,
        [5.0, 4.7, 4.45, 4.20, 4.65, 4.90, 4.75, 4.55],
    )
    # The final local slope improves, but CE remains far behind the causal
    # baseline.  Progress debt prevents a positive consequence.
    assert reports[-1].observed_slope_per_million_tokens < 0
    assert reports[-1].progress_debt_nats_per_token > 0
    assert reports[-1].pressure < 0


def test_guard_vetoes_positive_pressure_and_recovers_with_patience():
    authority = LearningProgressAuthority(config())
    feed(
        authority,
        [5.0, 4.7, 4.45, 4.20, 3.82, 3.45, 3.10],
    )
    assert authority.observe_guard(3.0)
    assert authority.observe_guard(3.2)
    assert not authority.observe_guard(3.3)
    report = authority.observe(8_000_000, 2.80, learning_rate=1e-4)
    assert report.raw_pressure <= 0
    assert report.pressure <= 0
    assert not authority.observe_guard(3.01)
    assert authority.observe_guard(2.99)


def test_progress_checkpoint_resume_is_exact_and_rejects_contract_drift():
    original = LearningProgressAuthority(config())
    feed(original, [5.0, 4.7, 4.45, 4.20, 3.95, 3.70])
    original.observe_guard(3.8)
    state = copy.deepcopy(original.state_dict())
    restored = LearningProgressAuthority(config())
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    left = original.observe(7_000_000, 3.40, learning_rate=8e-5)
    right = restored.observe(7_000_000, 3.40, learning_rate=8e-5)
    assert left == right
    with pytest.raises(ValueError, match="configuration differs"):
        LearningProgressAuthority(
            config(deadband_standard_deviations=0.25)
        ).load_state_dict(state)


def test_version1_progress_checkpoint_migrates_latest_guard_from_best():
    authority = LearningProgressAuthority(config())
    feed(authority, [5.0, 4.7, 4.45, 4.20])
    authority.observe_guard(3.8)
    legacy = copy.deepcopy(authority.state_dict())
    legacy["format_version"] = 1
    legacy.pop("last_guard_ce")
    restored = LearningProgressAuthority(config())
    restored.load_state_dict(legacy)
    assert restored.best_guard_ce == 3.8
    assert restored.last_guard_ce == 3.8


def test_progress_authority_rejects_noncausal_or_invalid_observations():
    authority = LearningProgressAuthority(config())
    authority.observe(100, 4.0, 1e-4)
    with pytest.raises(ValueError, match="increase strictly"):
        authority.observe(100, 3.9, 1e-4)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        authority.observe(200, float("nan"), 1e-4)


def test_metrics_and_state_contain_no_phase_transition_authority():
    authority = LearningProgressAuthority(config())
    report = feed(authority, [5.0, 4.7, 4.45, 4.20])[-1]
    metrics = authority.metrics(report)
    serialized = repr((metrics, authority.state_dict())).lower()
    assert metrics
    assert all(name.startswith("pc_rasl/") for name in metrics)
    assert "event_proposal" not in serialized
    assert "phase_transition" not in serialized
