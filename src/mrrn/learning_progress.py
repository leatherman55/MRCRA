"""Causal CE-learning-progress authority for Progress-Conditioned RASL.

This module is deliberately independent of architecture and phase-transition
telemetry.  Its complete observation contract is a monotonically increasing
count of valid target tokens, exact cross entropy in nats per token, and the
current learning rate.  The authority never enters the actor forward pass; it
produces a delayed, bounded training consequence after comparing observed
learning progress with a causal forecast.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite, log, sqrt, tanh
from statistics import median
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class LearningProgressConfig:
    """Validated controls for the production learning-progress authority."""

    observation_interval: int = 5
    warmup_observations: int = 8
    fast_window: int = 6
    baseline_min_observations: int = 12
    baseline_window: int = 24
    baseline_lag: int = 6
    baseline_freeze_observations: int = 4
    huber_delta: float = 1.5
    robust_iterations: int = 8
    deadband_standard_deviations: float = 0.5
    pressure_temperature: float = 1.0
    slope_weight: float = 0.75
    debt_weight: float = 0.25
    maximum_pressure: float = 1.0
    minimum_slope_noise: float = 1e-4
    minimum_ce_noise: float = 1e-4
    positive_pressure_requires_decreasing_ce: bool = True
    guard_regression_tolerance: float = 0.02
    guard_regression_patience: int = 2
    guard_recovery_patience: int = 2

    def __post_init__(self) -> None:
        positive_ints = (
            self.observation_interval,
            self.warmup_observations,
            self.fast_window,
            self.baseline_min_observations,
            self.baseline_window,
            self.baseline_freeze_observations,
            self.robust_iterations,
            self.guard_regression_patience,
            self.guard_recovery_patience,
        )
        if min(positive_ints) <= 0 or self.baseline_lag < 0:
            raise ValueError("learning-progress windows and patience must be positive")
        if self.fast_window < 3:
            raise ValueError("a robust progress slope requires at least three observations")
        if self.baseline_min_observations < 4:
            raise ValueError("a progress baseline requires at least four observations")
        if self.baseline_window < self.baseline_min_observations:
            raise ValueError("baseline window cannot be shorter than its minimum")
        if self.warmup_observations < self.fast_window:
            raise ValueError("progress warmup cannot be shorter than the fast window")
        if min(
            self.huber_delta,
            self.pressure_temperature,
            self.maximum_pressure,
            self.minimum_slope_noise,
            self.minimum_ce_noise,
        ) <= 0:
            raise ValueError("learning-progress scales and limits must be positive")
        if self.deadband_standard_deviations < 0:
            raise ValueError("learning-progress deadband cannot be negative")
        if self.slope_weight < 0 or self.debt_weight < 0:
            raise ValueError("learning-progress component weights cannot be negative")
        if self.slope_weight + self.debt_weight <= 0:
            raise ValueError("at least one learning-progress component must be active")
        if self.guard_regression_tolerance < 0:
            raise ValueError("guard regression tolerance cannot be negative")


@dataclass(frozen=True, slots=True)
class ProgressObservation:
    """One immutable exact-CE observation."""

    valid_tokens: int
    ce_nats_per_token: float
    learning_rate: float

    def __post_init__(self) -> None:
        if self.valid_tokens <= 0:
            raise ValueError("progress observations require positive valid-token positions")
        if (
            not isfinite(self.ce_nats_per_token)
            or self.ce_nats_per_token < 0
            or not isfinite(self.learning_rate)
            or self.learning_rate < 0
        ):
            raise ValueError("progress observation values must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class PowerLawBaseline:
    """Causal shifted-power-law forecast, locally interpreted as a line."""

    asymptote: float
    amplitude: float
    token_offset: float
    exponent: float
    fitted_through_tokens: int
    residual_scale: float

    def __post_init__(self) -> None:
        values = (
            self.asymptote,
            self.amplitude,
            self.token_offset,
            self.exponent,
            self.residual_scale,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("progress baseline values must be finite")
        if (
            self.asymptote < 0
            or self.amplitude <= 0
            or self.token_offset < 0
            or self.exponent <= 0
            or self.fitted_through_tokens <= 0
            or self.residual_scale <= 0
        ):
            raise ValueError("progress baseline values lie outside their valid domains")

    def expected_ce(self, valid_tokens: int) -> float:
        if valid_tokens <= 0:
            raise ValueError("baseline evaluation requires positive valid tokens")
        return self.asymptote + self.amplitude * (
            valid_tokens + self.token_offset
        ) ** (-self.exponent)

    def expected_slope_per_token(self, valid_tokens: int) -> float:
        if valid_tokens <= 0:
            raise ValueError("baseline slope requires positive valid tokens")
        return -self.exponent * self.amplitude * (
            valid_tokens + self.token_offset
        ) ** (-self.exponent - 1)


@dataclass(frozen=True, slots=True)
class LearningProgressReport:
    """Bounded signed consequence and all evidence used to derive it."""

    observation_index: int
    valid_tokens: int
    ce_nats_per_token: float
    pressure: float
    raw_pressure: float
    confidence: float
    observed_slope_per_million_tokens: float
    expected_slope_per_million_tokens: float
    slope_noise_per_million_tokens: float
    slope_advantage_z: float
    expected_ce_nats_per_token: float
    progress_debt_nats_per_token: float
    debt_z: float
    baseline_ready: bool
    warmup_complete: bool
    guard_allows_positive_pressure: bool
    interval_start_tokens: int
    interval_end_tokens: int

    def __post_init__(self) -> None:
        numeric = tuple(
            value for value in asdict(self).values()
            if isinstance(value, float)
        )
        if not all(isfinite(value) for value in numeric):
            raise ValueError("learning-progress report must be finite")
        if not -1e-12 <= self.confidence <= 1 + 1e-12:
            raise ValueError("learning-progress confidence must lie in [0,1]")
        if self.interval_start_tokens < 0 or self.interval_end_tokens != self.valid_tokens:
            raise ValueError("learning-progress report interval is malformed")


def _median_absolute_deviation(values: Sequence[float], *, floor: float) -> float:
    if not values:
        return floor
    center = median(values)
    # 1.4826 makes MAD consistent with Gaussian standard deviation.
    return max(floor, 1.4826 * median(abs(value - center) for value in values))


def _weighted_line(
    x: Sequence[float],
    y: Sequence[float],
    weights: Sequence[float],
) -> tuple[float, float]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("robust regression weights must have positive mass")
    mean_x = sum(weight * value for weight, value in zip(weights, x, strict=True)) / total
    mean_y = sum(weight * value for weight, value in zip(weights, y, strict=True)) / total
    variance = sum(
        weight * (value - mean_x) ** 2
        for weight, value in zip(weights, x, strict=True)
    )
    if variance <= 0:
        raise ValueError("regression coordinates must not be constant")
    covariance = sum(
        weight * (left - mean_x) * (right - mean_y)
        for weight, left, right in zip(weights, x, y, strict=True)
    )
    slope = covariance / variance
    return mean_y - slope * mean_x, slope


def robust_line(
    x: Sequence[float],
    y: Sequence[float],
    *,
    huber_delta: float,
    iterations: int,
    scale_floor: float,
) -> tuple[float, float, float]:
    """Deterministic Huber IRLS line with a robust residual scale."""

    if len(x) != len(y) or len(x) < 3:
        raise ValueError("robust line fitting requires at least three paired points")
    if any(not isfinite(value) for value in (*x, *y)):
        raise ValueError("robust line inputs must be finite")
    weights = [1.0] * len(x)
    intercept, slope = _weighted_line(x, y, weights)
    scale = scale_floor
    for _ in range(iterations):
        residuals = [
            target - (intercept + slope * coordinate)
            for coordinate, target in zip(x, y, strict=True)
        ]
        scale = _median_absolute_deviation(residuals, floor=scale_floor)
        cutoff = huber_delta * scale
        weights = [
            1.0 if abs(residual) <= cutoff else cutoff / max(abs(residual), scale_floor)
            for residual in residuals
        ]
        intercept, slope = _weighted_line(x, y, weights)
    residuals = [
        target - (intercept + slope * coordinate)
        for coordinate, target in zip(x, y, strict=True)
    ]
    return intercept, slope, _median_absolute_deviation(
        residuals, floor=scale_floor
    )


def _fit_power_law(
    observations: Sequence[ProgressObservation],
    config: LearningProgressConfig,
) -> PowerLawBaseline:
    """Fit a robust shifted power law by deterministic bounded grid search."""

    if len(observations) < config.baseline_min_observations:
        raise ValueError("insufficient observations for the progress baseline")
    rows = observations[-config.baseline_window:]
    tokens = [float(row.valid_tokens) for row in rows]
    losses = [row.ce_nats_per_token for row in rows]
    minimum = min(losses)
    loss_span = max(max(losses) - minimum, config.minimum_ce_noise)
    token_span = max(tokens[-1] - tokens[0], 1.0)
    # The asymptote must remain below every observed CE.  Offset candidates
    # cover an unshifted curve through a history-scale shift without an
    # unconstrained nonlinear optimizer.
    asymptotes = tuple(
        max(0.0, minimum - loss_span * fraction)
        for fraction in (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5)
    )
    offsets = tuple(
        token_span * fraction
        for fraction in (0.0, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 4.0)
    )
    best: tuple[float, PowerLawBaseline] | None = None
    for asymptote in asymptotes:
        shifted_loss = [value - asymptote for value in losses]
        if min(shifted_loss) <= 0:
            continue
        for offset in offsets:
            x = [log(value + offset) for value in tokens]
            y = [log(value) for value in shifted_loss]
            intercept, logarithmic_slope, _ = robust_line(
                x,
                y,
                huber_delta=config.huber_delta,
                iterations=config.robust_iterations,
                scale_floor=1e-8,
            )
            exponent = -logarithmic_slope
            if not 1e-4 <= exponent <= 5.0:
                continue
            amplitude = exp(intercept)
            predictions = [
                asymptote + amplitude * (token + offset) ** (-exponent)
                for token in tokens
            ]
            residuals = [
                actual - prediction
                for actual, prediction in zip(losses, predictions, strict=True)
            ]
            scale = _median_absolute_deviation(
                residuals, floor=config.minimum_ce_noise
            )
            robust_error = median(abs(value) for value in residuals) + 0.1 * scale
            candidate = PowerLawBaseline(
                asymptote,
                amplitude,
                offset,
                exponent,
                rows[-1].valid_tokens,
                scale,
            )
            if best is None or robust_error < best[0]:
                best = (robust_error, candidate)
    if best is None:
        raise RuntimeError("no valid causal power-law progress baseline could be fitted")
    return best[1]


class LearningProgressAuthority:
    """Stateful, checkpointable authority for delayed CE progress pressure."""

    FORMAT_VERSION = 2

    def __init__(self, config: LearningProgressConfig = LearningProgressConfig()) -> None:
        self.config = config
        self.observations: list[ProgressObservation] = []
        self.baseline: PowerLawBaseline | None = None
        self.baseline_observation_index = 0
        self.last_report: LearningProgressReport | None = None
        self.best_guard_ce: float | None = None
        self.last_guard_ce: float | None = None
        self.guard_regressions = 0
        self.guard_recoveries = 0
        # Positive pressure is fail-closed until the independent guard has
        # established a real held-out performance reference.
        self.guard_allows_positive_pressure = False

    @property
    def ready(self) -> bool:
        return (
            len(self.observations) >= self.config.warmup_observations
            and self.baseline is not None
        )

    def _maybe_fit_baseline(self) -> None:
        available = len(self.observations) - self.config.baseline_lag
        if available < self.config.baseline_min_observations:
            return
        if (
            self.baseline is not None
            and available - self.baseline_observation_index
            < self.config.baseline_freeze_observations
        ):
            return
        fit_rows = self.observations[:available]
        self.baseline = _fit_power_law(fit_rows, self.config)
        self.baseline_observation_index = available

    def observe_guard(self, ce_nats_per_token: float) -> bool:
        """Update the disjoint guard; return whether positive pressure is allowed."""

        if not isfinite(ce_nats_per_token) or ce_nats_per_token < 0:
            raise ValueError("guard CE must be finite and nonnegative")
        self.last_guard_ce = ce_nats_per_token
        if self.best_guard_ce is None:
            self.best_guard_ce = ce_nats_per_token
            self.guard_allows_positive_pressure = True
            return self.guard_allows_positive_pressure
        tolerance = self.config.guard_regression_tolerance * max(
            1.0, abs(self.best_guard_ce)
        )
        if ce_nats_per_token > self.best_guard_ce + tolerance:
            self.guard_regressions += 1
            self.guard_recoveries = 0
            if self.guard_regressions >= self.config.guard_regression_patience:
                self.guard_allows_positive_pressure = False
        else:
            self.guard_regressions = 0
            self.guard_recoveries += 1
            self.best_guard_ce = min(self.best_guard_ce, ce_nats_per_token)
            if self.guard_recoveries >= self.config.guard_recovery_patience:
                self.guard_allows_positive_pressure = True
        return self.guard_allows_positive_pressure

    def observe(
        self,
        valid_tokens: int,
        ce_nats_per_token: float,
        learning_rate: float,
    ) -> LearningProgressReport:
        observation = ProgressObservation(
            valid_tokens, ce_nats_per_token, learning_rate
        )
        if self.observations and valid_tokens <= self.observations[-1].valid_tokens:
            raise ValueError("progress valid-token positions must increase strictly")
        interval_start = (
            0 if not self.observations else self.observations[-1].valid_tokens
        )
        self.observations.append(observation)
        self._maybe_fit_baseline()
        enough_fast = len(self.observations) >= self.config.fast_window
        warm = len(self.observations) >= self.config.warmup_observations
        observed_slope = 0.0
        slope_noise = self.config.minimum_slope_noise / 1_000_000
        if enough_fast:
            fast = self.observations[-self.config.fast_window:]
            origin = float(fast[-1].valid_tokens)
            x = [
                (row.valid_tokens - origin) / 1_000_000
                for row in fast
            ]
            y = [row.ce_nats_per_token for row in fast]
            _, slope_per_million, residual_scale = robust_line(
                x,
                y,
                huber_delta=self.config.huber_delta,
                iterations=self.config.robust_iterations,
                scale_floor=self.config.minimum_ce_noise,
            )
            observed_slope = slope_per_million / 1_000_000
            x_span = max(x) - min(x)
            history = self.observations[-self.config.baseline_window:]
            incremental_slopes = [
                (right.ce_nats_per_token - left.ce_nats_per_token)
                / ((right.valid_tokens - left.valid_tokens) / 1_000_000)
                for left, right in zip(history, history[1:])
            ]
            historical_slope_noise = _median_absolute_deviation(
                incremental_slopes,
                floor=self.config.minimum_slope_noise,
            )
            slope_noise = max(
                self.config.minimum_slope_noise / 1_000_000,
                residual_scale / max(abs(x_span), 1e-9) / 1_000_000,
                historical_slope_noise / 1_000_000,
            )
        baseline = self.baseline
        baseline_ready = baseline is not None
        expected_slope = (
            0.0 if baseline is None
            else baseline.expected_slope_per_token(valid_tokens)
        )
        expected_ce = (
            ce_nats_per_token if baseline is None
            else baseline.expected_ce(valid_tokens)
        )
        debt = ce_nats_per_token - expected_ce
        slope_z = (
            (expected_slope - observed_slope) / slope_noise
            if enough_fast and baseline_ready else 0.0
        )
        ce_noise = (
            self.config.minimum_ce_noise
            if baseline is None else baseline.residual_scale
        )
        debt_z = -debt / max(ce_noise, self.config.minimum_ce_noise)
        weight_sum = self.config.slope_weight + self.config.debt_weight
        combined = (
            self.config.slope_weight * slope_z
            + self.config.debt_weight * debt_z
        ) / weight_sum
        deadband = self.config.deadband_standard_deviations
        after_deadband = (
            0.0
            if abs(combined) <= deadband
            else (1 if combined > 0 else -1) * (abs(combined) - deadband)
        )
        sample_confidence = min(
            1.0,
            max(0.0, (len(self.observations) - self.config.fast_window + 1) / 4),
        )
        signal_confidence = 1 - exp(-abs(after_deadband))
        confidence = sample_confidence * signal_confidence
        raw = tanh(after_deadband / self.config.pressure_temperature)
        if not (warm and baseline_ready):
            raw = 0.0
            confidence = 0.0
        if (
            self.config.positive_pressure_requires_decreasing_ce
            and observed_slope >= 0
        ):
            raw = min(0.0, raw)
        if not self.guard_allows_positive_pressure:
            raw = min(0.0, raw)
        pressure = self.config.maximum_pressure * raw * confidence
        report = LearningProgressReport(
            len(self.observations),
            valid_tokens,
            ce_nats_per_token,
            pressure,
            raw,
            confidence,
            observed_slope * 1_000_000,
            expected_slope * 1_000_000,
            slope_noise * 1_000_000,
            slope_z,
            expected_ce,
            debt,
            debt_z,
            baseline_ready,
            warm,
            self.guard_allows_positive_pressure,
            interval_start,
            valid_tokens,
        )
        self.last_report = report
        return report

    @staticmethod
    def metrics(report: LearningProgressReport) -> dict[str, float]:
        """Stable Trackio metric names, intentionally free of phase metrics."""

        return {
            "pc_rasl/progress_pressure": report.pressure,
            "pc_rasl/raw_progress_pressure": report.raw_pressure,
            "pc_rasl/progress_confidence": report.confidence,
            "pc_rasl/observed_ce_slope_per_million_tokens": (
                report.observed_slope_per_million_tokens
            ),
            "pc_rasl/expected_ce_slope_per_million_tokens": (
                report.expected_slope_per_million_tokens
            ),
            "pc_rasl/slope_noise_per_million_tokens": (
                report.slope_noise_per_million_tokens
            ),
            "pc_rasl/slope_advantage_z": report.slope_advantage_z,
            "pc_rasl/expected_ce_nats_per_token": report.expected_ce_nats_per_token,
            "pc_rasl/progress_debt_nats_per_token": (
                report.progress_debt_nats_per_token
            ),
            "pc_rasl/debt_z": report.debt_z,
            "pc_rasl/baseline_ready": float(report.baseline_ready),
            "pc_rasl/warmup_complete": float(report.warmup_complete),
            "pc_rasl/guard_allows_positive_pressure": float(
                report.guard_allows_positive_pressure
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.FORMAT_VERSION,
            "config": asdict(self.config),
            "observations": [asdict(row) for row in self.observations],
            "baseline": None if self.baseline is None else asdict(self.baseline),
            "baseline_observation_index": self.baseline_observation_index,
            "last_report": (
                None if self.last_report is None else asdict(self.last_report)
            ),
            "best_guard_ce": self.best_guard_ce,
            "last_guard_ce": self.last_guard_ce,
            "guard_regressions": self.guard_regressions,
            "guard_recoveries": self.guard_recoveries,
            "guard_allows_positive_pressure": self.guard_allows_positive_pressure,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        format_version = state.get("format_version")
        if format_version not in {1, self.FORMAT_VERSION}:
            raise ValueError("unsupported learning-progress checkpoint format")
        if state.get("config") != asdict(self.config):
            raise ValueError("learning-progress checkpoint configuration differs")
        restored = [ProgressObservation(**row) for row in state["observations"]]
        if any(
            right.valid_tokens <= left.valid_tokens
            for left, right in zip(restored, restored[1:])
        ):
            raise ValueError("checkpointed progress positions are not strictly increasing")
        baseline = (
            None
            if state["baseline"] is None
            else PowerLawBaseline(**state["baseline"])
        )
        report = (
            None
            if state["last_report"] is None
            else LearningProgressReport(**state["last_report"])
        )
        baseline_index = int(state["baseline_observation_index"])
        regressions = int(state["guard_regressions"])
        recoveries = int(state["guard_recoveries"])
        if (
            not 0 <= baseline_index <= len(restored)
            or regressions < 0
            or recoveries < 0
        ):
            raise ValueError("learning-progress checkpoint counters are invalid")
        best = state["best_guard_ce"]
        if best is not None and (not isfinite(float(best)) or float(best) < 0):
            raise ValueError("checkpointed guard CE is invalid")
        last = state.get("last_guard_ce", best)
        if last is not None and (not isfinite(float(last)) or float(last) < 0):
            raise ValueError("checkpointed latest guard CE is invalid")
        self.observations = restored
        self.baseline = baseline
        self.baseline_observation_index = baseline_index
        self.last_report = report
        self.best_guard_ce = None if best is None else float(best)
        self.last_guard_ce = None if last is None else float(last)
        self.guard_regressions = regressions
        self.guard_recoveries = recoveries
        self.guard_allows_positive_pressure = (
            best is not None
            and bool(state["guard_allows_positive_pressure"])
        )
