"""Long-duration resource and resume acceptance for MRCRA training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class ResourceSoakSample:
    profile: str
    steps: int
    rss_bytes: tuple[int, ...]
    measured_elapsed_seconds: float
    accounted_elapsed_seconds: float
    checkpoint_count: int
    resume_succeeded: bool
    temporary_file_count: int
    stale_thread_count: int
    maximum_coverage_gap: int
    declared_maximum_coverage_gap: int
    data_cursor_exact: bool
    nonfinite_metric_count: int

    def __post_init__(self) -> None:
        if (
            self.profile not in {"quick", "production_8p4m_32k"}
            or self.steps <= 0
            or len(self.rss_bytes) != self.steps
            or any(value <= 0 for value in self.rss_bytes)
            or min(
                self.measured_elapsed_seconds,
                self.accounted_elapsed_seconds,
            )
            <= 0
            or min(
                self.checkpoint_count,
                self.temporary_file_count,
                self.stale_thread_count,
                self.maximum_coverage_gap,
                self.declared_maximum_coverage_gap,
                self.nonfinite_metric_count,
            )
            < 0
        ):
            raise ValueError("resource soak sample is malformed")


@dataclass(frozen=True, slots=True)
class ResourceSoakCriterion:
    name: str
    measurement: float
    threshold: float
    direction: str
    unit: str
    passed: bool


@dataclass(frozen=True, slots=True)
class ResourceSoakReport:
    schema_version: int
    sample: ResourceSoakSample
    criteria: tuple[ResourceSoakCriterion, ...]
    passed: bool
    claim_boundary: str

    def to_dict(self) -> dict:
        return asdict(self)


def _criterion(
    name: str,
    measurement: float,
    threshold: float,
    direction: str,
    unit: str,
) -> ResourceSoakCriterion:
    if direction == "maximum":
        passed = measurement <= threshold
    elif direction == "minimum":
        passed = measurement >= threshold
    else:
        raise ValueError("resource soak criterion direction is invalid")
    return ResourceSoakCriterion(
        name, measurement, threshold, direction, unit, passed
    )


def build_resource_soak_report(
    sample: ResourceSoakSample,
) -> ResourceSoakReport:
    x_mean = (sample.steps - 1) / 2
    y_mean = sum(sample.rss_bytes) / sample.steps
    denominator = sum(
        (index - x_mean) ** 2 for index in range(sample.steps)
    )
    slope = (
        sum(
            (index - x_mean) * (value - y_mean)
            for index, value in enumerate(sample.rss_bytes)
        )
        / max(denominator, 1)
    )
    rss_range = max(sample.rss_bytes) - min(sample.rss_bytes)
    elapsed_error = abs(
        sample.accounted_elapsed_seconds
        - sample.measured_elapsed_seconds
    ) / sample.measured_elapsed_seconds
    criteria = (
        _criterion(
            "optimizer_steps",
            float(sample.steps),
            100.0,
            "minimum",
            "steps",
        ),
        _criterion(
            "rss_growth_slope",
            max(0.0, slope),
            float(1 << 20),
            "maximum",
            "bytes/step",
        ),
        _criterion(
            "rss_range",
            float(rss_range),
            float(256 << 20),
            "maximum",
            "bytes",
        ),
        _criterion(
            "resume_succeeded",
            float(sample.resume_succeeded),
            1.0,
            "minimum",
            "boolean",
        ),
        _criterion(
            "checkpoint_count",
            float(sample.checkpoint_count),
            2.0,
            "minimum",
            "files",
        ),
        _criterion(
            "temporary_files_remaining",
            float(sample.temporary_file_count),
            0.0,
            "maximum",
            "files",
        ),
        _criterion(
            "stale_threads_remaining",
            float(sample.stale_thread_count),
            0.0,
            "maximum",
            "threads",
        ),
        _criterion(
            "sampled_cstm_coverage_gap",
            float(sample.maximum_coverage_gap),
            float(sample.declared_maximum_coverage_gap),
            "maximum",
            "steps",
        ),
        _criterion(
            "resume_data_cursor_exactness",
            float(sample.data_cursor_exact),
            1.0,
            "minimum",
            "boolean",
        ),
        _criterion(
            "wall_clock_accounting_error",
            elapsed_error,
            0.01,
            "maximum",
            "fraction",
        ),
        _criterion(
            "nonfinite_metrics",
            float(sample.nonfinite_metric_count),
            0.0,
            "maximum",
            "values",
        ),
    )
    return ResourceSoakReport(
        1,
        sample,
        criteria,
        all(item.passed for item in criteria),
        (
            "This validates bounded resources, periodic checkpoint/evaluation "
            "behavior, and a mid-run exact resume for the named local profile. "
            "Only production_8p4m_32k evidence supports a full-size claim."
        ),
    )
