from __future__ import annotations

from dataclasses import replace

import pytest

from mrrn.observability_acceptance import (
    ObservabilitySample,
    build_observability_report,
)


def sample(variant: str, *, seconds: float, rss: int) -> ObservabilitySample:
    return ObservabilitySample(
        variant,
        tuple(seconds for _ in range(100)),
        tuple(rss for _ in range(100)),
    )


def test_observability_acceptance_enforces_time_and_memory_budgets():
    report = build_observability_report((
        sample("null", seconds=0.00001, rss=100 << 20),
        sample("trackio", seconds=0.00021, rss=120 << 20),
    ))
    assert report.passed
    assert all(item.passed for item in report.criteria)
    assert all(item.unit for item in report.criteria)


def test_observability_acceptance_fails_each_excessive_resource_axis():
    null = sample("null", seconds=0.00001, rss=100 << 20)
    excessive = ObservabilitySample(
        "trackio",
        tuple(0.00041 for _ in range(100)),
        tuple((100 + index * 4) << 20 for index in range(100)),
    )
    report = build_observability_report((null, excessive))
    assert not report.passed
    assert {
        item.name for item in report.criteria if not item.passed
    } == {
        "trackio_steady_state_step_overhead",
        "trackio_additional_peak_rss",
        "trackio_steady_state_rss_range",
    }


def test_observability_acceptance_rejects_unmatched_samples():
    with pytest.raises(ValueError, match="matched"):
        build_observability_report((
            sample("null", seconds=1.0, rss=1),
            replace(
                sample("null", seconds=1.0, rss=1),
                step_seconds=tuple(1.0 for _ in range(99)),
                rss_bytes=tuple(1 for _ in range(99)),
            ),
        ))
