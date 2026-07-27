from __future__ import annotations

from dataclasses import replace

from mrrn.resource_soak_acceptance import (
    ResourceSoakSample,
    build_resource_soak_report,
)


def passing_sample() -> ResourceSoakSample:
    return ResourceSoakSample(
        profile="quick",
        steps=100,
        rss_bytes=tuple((500 << 20) + index * 1024 for index in range(100)),
        measured_elapsed_seconds=100.0,
        accounted_elapsed_seconds=100.5,
        checkpoint_count=4,
        resume_succeeded=True,
        temporary_file_count=0,
        stale_thread_count=0,
        maximum_coverage_gap=12,
        declared_maximum_coverage_gap=4_096,
        data_cursor_exact=True,
        nonfinite_metric_count=0,
    )


def test_resource_soak_acceptance_passes_bounded_resumable_run():
    report = build_resource_soak_report(passing_sample())
    assert report.passed
    assert all(item.passed for item in report.criteria)


def test_resource_soak_acceptance_rejects_growth_leaks_and_accounting_drift():
    sample = replace(
        passing_sample(),
        rss_bytes=tuple(
            (500 << 20) + index * (4 << 20)
            for index in range(100)
        ),
        accounted_elapsed_seconds=105.0,
        resume_succeeded=False,
        temporary_file_count=1,
        stale_thread_count=1,
        maximum_coverage_gap=5_000,
        data_cursor_exact=False,
        nonfinite_metric_count=1,
    )
    report = build_resource_soak_report(sample)
    assert not report.passed
    failed = {item.name for item in report.criteria if not item.passed}
    assert {
        "rss_growth_slope",
        "rss_range",
        "resume_succeeded",
        "temporary_files_remaining",
        "stale_threads_remaining",
        "sampled_cstm_coverage_gap",
        "resume_data_cursor_exactness",
        "wall_clock_accounting_error",
        "nonfinite_metrics",
    }.issubset(failed)


def test_resource_soak_acceptance_does_not_call_process_resume_baseline_a_leak():
    first = tuple(
        (1_500 << 20) - index * (10 << 20)
        for index in range(50)
    )
    second = tuple(
        (2_800 << 20) + min(index, 8) * (2 << 20)
        for index in range(50)
    )
    report = build_resource_soak_report(
        replace(passing_sample(), rss_bytes=first + second)
    )
    assert report.passed
    measurements = {
        item.name: item.measurement for item in report.criteria
    }
    assert measurements["rss_growth_slope"] <= 1 << 20
    assert measurements["rss_range"] == 16 << 20
