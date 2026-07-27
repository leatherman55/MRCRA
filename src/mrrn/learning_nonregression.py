"""Three-seed learning-quality authority for sampled CSTM activation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from hashlib import sha256
from math import isfinite, sqrt
from statistics import mean, stdev
from typing import Mapping


LEARNING_VARIANTS = ("legacy_dense", "sampled", "ce_only")


@dataclass(frozen=True, slots=True)
class LearningObservation:
    """One matched-token point on the wall-clock and token learning curves."""

    step: int
    physical_tokens: int
    wall_clock_seconds: float
    train_ce_nats_per_token: float
    cstm_standardized_huber: float
    carrier_auxiliary_norm_after: float
    cognition_auxiliary_norm_after: float
    gradient_clip_coefficient: float
    state_rms_max: float
    feedback_rms_max: float
    cognitive_cycles: int
    events: int

    def __post_init__(self) -> None:
        numeric = (
            self.wall_clock_seconds,
            self.train_ce_nats_per_token,
            self.cstm_standardized_huber,
            self.carrier_auxiliary_norm_after,
            self.cognition_auxiliary_norm_after,
            self.gradient_clip_coefficient,
            self.state_rms_max,
            self.feedback_rms_max,
        )
        if (
            self.step <= 0
            or self.physical_tokens <= 0
            or self.wall_clock_seconds < 0
            or not all(isfinite(value) for value in numeric)
            or not 0 <= self.gradient_clip_coefficient <= 1
            or min(self.cognitive_cycles, self.events) < 0
        ):
            raise ValueError("learning observation is malformed")


@dataclass(frozen=True, slots=True)
class LearningRun:
    variant: str
    seed: int
    physical_tokens: int
    eval_ce_nats_per_token: float
    eval_ece_nats_per_byte: float
    cstm_standardized_huber: float
    carrier_auxiliary_participation: bool
    cognition_auxiliary_participation: bool
    gradient_clip_frequency: float
    state_rms_max: float
    feedback_rms_max: float
    cognitive_cycles: int
    events: int
    training_seconds: float
    observations: tuple[LearningObservation, ...]
    finite: bool
    checkpoint_resumable: bool

    def __post_init__(self) -> None:
        numeric = (
            self.eval_ce_nats_per_token,
            self.eval_ece_nats_per_byte,
            self.cstm_standardized_huber,
            self.gradient_clip_frequency,
            self.state_rms_max,
            self.feedback_rms_max,
            self.training_seconds,
        )
        if (
            self.variant not in LEARNING_VARIANTS
            or self.seed < 0
            or self.physical_tokens <= 0
            or not all(isfinite(value) for value in numeric)
            or not 0 <= self.gradient_clip_frequency <= 1
            or min(self.cognitive_cycles, self.events) < 0
            or self.training_seconds < 0
            or not self.observations
            or self.observations[-1].physical_tokens != self.physical_tokens
            or any(
                right.step <= left.step
                or right.physical_tokens <= left.physical_tokens
                or right.wall_clock_seconds < left.wall_clock_seconds
                for left, right in zip(
                    self.observations, self.observations[1:], strict=False
                )
            )
        ):
            raise ValueError("learning non-regression run is malformed")


@dataclass(frozen=True, slots=True)
class LearningCriterion:
    name: str
    measurement: float
    threshold: float
    direction: str
    unit: str
    passed: bool


@dataclass(frozen=True, slots=True)
class LearningNonRegressionReport:
    schema_version: int
    runs: tuple[LearningRun, ...]
    paired_sampled_minus_legacy: tuple[float, ...]
    mean_difference: float
    confidence_interval_95: tuple[float, float]
    criteria: tuple[LearningCriterion, ...]
    passed: bool
    claim_boundary: str

    def to_dict(self) -> dict:
        return asdict(self)


def learning_run_from_dict(value: Mapping[str, object]) -> LearningRun:
    """Decode one journal row through every public evidence validator."""

    decoded = dict(value)
    raw_observations = decoded.get("observations")
    if not isinstance(raw_observations, (list, tuple)):
        raise ValueError("learning journal observations are malformed")
    decoded["observations"] = tuple(
        LearningObservation(**dict(item))
        for item in raw_observations
        if isinstance(item, Mapping)
    )
    if len(decoded["observations"]) != len(raw_observations):
        raise ValueError("learning journal contains a non-mapping observation")
    try:
        return LearningRun(**decoded)
    except (TypeError, ValueError) as error:
        raise ValueError("learning journal run is malformed") from error


def learning_study_controls_digest(
    controls: Mapping[str, object],
) -> str:
    return sha256(
        json.dumps(
            controls,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def learning_study_journal_payload(
    *,
    profile: str,
    controls_digest: str,
    authority_digest: str,
    runs: tuple[LearningRun, ...],
    complete: bool,
) -> dict[str, object]:
    if (
        profile not in {"quick", "fineweb_8p4m_32k"}
        or len(controls_digest) != 64
        or len(authority_digest) != 64
        or len({(run.variant, run.seed) for run in runs}) != len(runs)
    ):
        raise ValueError("learning journal authority is malformed")
    return {
        "schema_version": 1,
        "profile": profile,
        "controls_digest": controls_digest,
        "authority_digest": authority_digest,
        "complete": bool(complete),
        "runs": [asdict(run) for run in runs],
    }


def restore_learning_study_journal(
    value: Mapping[str, object],
    *,
    profile: str,
    controls_digest: str,
    authority_digest: str,
) -> tuple[LearningRun, ...]:
    """Restore an exact journal or reject stale/corrupt evidence."""

    if (
        value.get("schema_version") != 1
        or value.get("profile") != profile
        or value.get("controls_digest") != controls_digest
        or value.get("authority_digest") != authority_digest
    ):
        return ()
    raw_runs = value.get("runs")
    if not isinstance(raw_runs, list):
        raise ValueError("learning journal run collection is malformed")
    runs = tuple(learning_run_from_dict(item) for item in raw_runs)
    if len({(run.variant, run.seed) for run in runs}) != len(runs):
        raise ValueError("learning journal contains duplicate run authority")
    return runs


def _criterion(
    name: str,
    measurement: float,
    threshold: float,
    direction: str,
    unit: str,
) -> LearningCriterion:
    if direction == "maximum":
        passed = measurement <= threshold
    elif direction == "minimum":
        passed = measurement >= threshold
    else:
        raise ValueError("learning criterion direction is invalid")
    return LearningCriterion(
        name, measurement, threshold, direction, unit, passed
    )


def build_learning_nonregression_report(
    runs: tuple[LearningRun, ...],
) -> LearningNonRegressionReport:
    if len(runs) < 9:
        raise ValueError(
            "learning acceptance requires three variants across at least three seeds"
        )
    by_key = {(run.variant, run.seed): run for run in runs}
    seeds = sorted({run.seed for run in runs})
    if (
        len(by_key) != len(runs)
        or len(seeds) < 3
        or set(by_key)
        != {
            (variant, seed)
            for variant in LEARNING_VARIANTS
            for seed in seeds
        }
    ):
        raise ValueError(
            "learning acceptance matrix is incomplete or duplicated"
        )
    token_counts = {run.physical_tokens for run in runs}
    if len(token_counts) != 1:
        raise ValueError("learning acceptance token budgets are not matched")
    differences = tuple(
        by_key[("sampled", seed)].eval_ce_nats_per_token
        - by_key[("legacy_dense", seed)].eval_ce_nats_per_token
        for seed in seeds
    )
    difference_mean = mean(differences)
    standard_error = (
        stdev(differences) / sqrt(len(differences))
        if len(differences) > 1
        else 0.0
    )
    # Student-t critical values for the small seed counts used here; 1.96 is
    # the conservative asymptotic fallback.
    t_critical = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(
        len(differences) - 1,
        1.96,
    )
    interval = (
        difference_mean - t_critical * standard_error,
        difference_mean + t_critical * standard_error,
    )
    sampled = tuple(by_key[("sampled", seed)] for seed in seeds)
    criteria = (
        _criterion(
            "sampled_mean_ce_regression",
            difference_mean,
            0.02,
            "maximum",
            "nats/token",
        ),
        _criterion(
            "all_runs_finite",
            float(all(run.finite for run in runs)),
            1.0,
            "minimum",
            "boolean",
        ),
        _criterion(
            "all_runs_checkpoint_resumable",
            float(all(run.checkpoint_resumable for run in runs)),
            1.0,
            "minimum",
            "boolean",
        ),
        _criterion(
            "sampled_carrier_auxiliary_participation",
            float(
                all(
                    run.carrier_auxiliary_participation
                    for run in sampled
                )
            ),
            1.0,
            "minimum",
            "boolean",
        ),
        _criterion(
            "sampled_cognition_auxiliary_participation",
            float(
                all(
                    run.cognition_auxiliary_participation
                    for run in sampled
                )
            ),
            1.0,
            "minimum",
            "boolean",
        ),
    )
    return LearningNonRegressionReport(
        1,
        runs,
        differences,
        difference_mean,
        interval,
        criteria,
        all(item.passed for item in criteria),
        (
            "This is a matched-token, paired-seed non-regression test. "
            "Only a FineWeb production profile with the declared retained "
            "evaluation set supports a public learning-quality claim; quick "
            "fixtures validate the procedure and resume mechanics only."
        ),
    )
