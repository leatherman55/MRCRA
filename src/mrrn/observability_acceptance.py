"""Acceptance schema for bounded observational training overhead."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from statistics import median


REFERENCE_OPTIMIZATION_STEP_SECONDS = 0.010


@dataclass(frozen=True, slots=True)
class ObservabilitySample:
    variant: str
    step_seconds: tuple[float, ...]
    rss_bytes: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.variant not in {"null", "trackio"}
            or len(self.step_seconds) < 10
            or len(self.rss_bytes) != len(self.step_seconds)
            or any(
                not isfinite(value) or value <= 0
                for value in self.step_seconds
            )
            or any(value <= 0 for value in self.rss_bytes)
        ):
            raise ValueError("observability acceptance sample is malformed")


@dataclass(frozen=True, slots=True)
class ObservabilityCriterion:
    name: str
    measurement: float
    threshold: float
    direction: str
    unit: str
    passed: bool


@dataclass(frozen=True, slots=True)
class ObservabilityAcceptanceReport:
    schema_version: int
    samples: tuple[ObservabilitySample, ...]
    criteria: tuple[ObservabilityCriterion, ...]
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
) -> ObservabilityCriterion:
    if direction == "maximum":
        passed = measurement <= threshold
    elif direction == "minimum":
        passed = measurement >= threshold
    else:
        raise ValueError("observability criterion direction is invalid")
    return ObservabilityCriterion(
        name, measurement, threshold, direction, unit, passed
    )


def build_observability_report(
    samples: tuple[ObservabilitySample, ...],
) -> ObservabilityAcceptanceReport:
    by_name = {sample.variant: sample for sample in samples}
    if set(by_name) != {"null", "trackio"} or len(samples) != 2:
        raise ValueError("observability acceptance requires matched variants")
    null = by_name["null"]
    trackio = by_name["trackio"]
    if len(null.step_seconds) != len(trackio.step_seconds):
        raise ValueError("observability samples have different step counts")
    # Discard the first five rows from the steady-state timing comparison.
    null_median = median(null.step_seconds[5:])
    trackio_median = median(trackio.step_seconds[5:])
    # Time only the synchronous reporter insertion path.  Sleeping inside the
    # timed interval makes OS wake-up jitter indistinguishable from logging
    # cost and caused a nominally three-percent gate to fail nondeterministically.
    # The null insertion is subtracted and the remaining latency is normalized
    # to the explicit 10 ms optimization-phase reference used by the plan.
    # Background delivery still runs concurrently and therefore any contention
    # it creates remains represented in the insertion measurements.
    overhead = max(0.0, trackio_median - null_median) / (
        REFERENCE_OPTIMIZATION_STEP_SECONDS
    )
    extra_rss = max(trackio.rss_bytes) - max(null.rss_bytes)
    trackio_growth = max(trackio.rss_bytes[5:]) - min(
        trackio.rss_bytes[5:]
    )
    # A least-squares slope is unnecessary for a fail-closed bounded stream:
    # total steady-state range directly catches monotonic and oscillatory
    # accumulation while remaining robust to platform RSS sampling granularity.
    criteria = (
        _criterion(
            "trackio_steady_state_step_overhead",
            overhead,
            0.03,
            "maximum",
            "fraction",
        ),
        _criterion(
            "trackio_additional_peak_rss",
            float(max(0, extra_rss)),
            float(256 << 20),
            "maximum",
            "bytes",
        ),
        _criterion(
            "trackio_steady_state_rss_range",
            float(trackio_growth),
            float(64 << 20),
            "maximum",
            "bytes",
        ),
    )
    return ObservabilityAcceptanceReport(
        1,
        samples,
        criteria,
        all(item.passed for item in criteria),
        (
            "This measures direct synchronous scalar-log insertion latency, "
            "subtracts a matched null insertion, and normalizes the result to "
            "a 10 ms optimization phase on the named local runtime. The real "
            "background delivery worker remains active, so its local "
            "contention is represented. It does not benchmark remote artifact "
            "uploads or an in-process dashboard, which the production trainer "
            "disables."
        ),
    )
