"""Fresh-process empirical acceptance schema for MRCRA training execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from statistics import median
from typing import Mapping


TRAINING_EXECUTION_ACCEPTANCE_FORMAT = 3
PRODUCTION_VARIANTS = (
    "legacy_serial_checkpoint_dense_cstm",
    "static_coarse_checkpoint_ce",
    "static_coarse_checkpoint_dense_cstm",
    "static_auto_ce",
    "static_auto_repaired_cstm",
    "static_cost_model_auto_repaired_cstm",
    "compiled_cost_model_auto_repaired_cstm",
)
PRODUCTION_REQUIRED_VARIANTS = PRODUCTION_VARIANTS[:-1]
COMPILED_VARIANT = PRODUCTION_VARIANTS[-1]


@dataclass(frozen=True, slots=True)
class CompilerCandidateReceipt:
    """Fail-closed outcome of the isolated optional compiler candidate."""

    requested_variant: str
    profile: str
    requested_backend: str
    outcome: str
    resolved_variant: str
    wall_clock_seconds: float
    timeout_seconds: float
    stdout_sha256: str
    stderr_sha256: str

    def __post_init__(self) -> None:
        if (
            self.requested_variant != COMPILED_VARIANT
            or self.profile not in {"quick", "production_8p4m_32k"}
            or self.requested_backend not in {"aot_eager", "inductor"}
            or self.outcome not in {"executed", "timeout"}
            or self.resolved_variant not in {
                COMPILED_VARIANT,
                "static_cost_model_auto_repaired_cstm",
            }
            or not isfinite(self.wall_clock_seconds)
            or self.wall_clock_seconds <= 0
            or not isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
            or len(self.stdout_sha256) != 64
            or len(self.stderr_sha256) != 64
            or (
                self.outcome == "executed"
                and self.resolved_variant != COMPILED_VARIANT
            )
            or (
                self.outcome == "timeout"
                and (
                    self.resolved_variant
                    != "static_cost_model_auto_repaired_cstm"
                    or self.wall_clock_seconds + 0.1 < self.timeout_seconds
                )
            )
        ):
            raise ValueError("compiler candidate receipt is malformed")


@dataclass(frozen=True, slots=True)
class TrainingExecutionSample:
    variant: str
    profile: str
    parameter_count: int
    context_length: int
    steps: int
    initialization_seconds: float
    training_seconds: float
    tokens_per_second: float
    peak_rss_bytes: int
    metrics: dict[str, float]
    runtime: dict[str, object]
    raw_step_seconds: tuple[float, ...] = ()
    median_step_seconds: float = 0.0
    minimum_step_seconds: float = 0.0
    maximum_step_seconds: float = 0.0
    median_absolute_deviation_seconds: float = 0.0
    swap_delta_bytes: int = 0
    step_metrics: tuple[dict[str, float], ...] = ()
    source_run_id: str = ""
    model_state_digest: str = ""
    optimizer_state_digest: str = ""
    tokenizer_identity_digest: str = ""
    fixture_digest: str = ""
    hardware_fingerprint: str = ""
    torch_version: str = ""
    metric_keys: tuple[str, ...] = ()
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    resolved_variant: str = ""

    def __post_init__(self) -> None:
        if (
            not self.variant
            or self.profile not in {"quick", "production_8p4m_32k"}
            or min(
                self.parameter_count,
                self.context_length,
                self.steps,
                self.peak_rss_bytes,
            )
            <= 0
            or min(self.initialization_seconds, self.training_seconds) < 0
            or self.swap_delta_bytes < 0
            or min(
                self.peak_allocated_bytes,
                self.peak_reserved_bytes,
            ) < 0
            or not isfinite(self.tokens_per_second)
            or self.tokens_per_second <= 0
            or not all(isfinite(value) for value in self.metrics.values())
        ):
            raise ValueError("training execution sample is malformed")
        if self.raw_step_seconds:
            if (
                len(self.raw_step_seconds) != self.steps
                or any(
                    not isfinite(value) or value <= 0
                    for value in self.raw_step_seconds
                )
                or min(
                    self.median_step_seconds,
                    self.minimum_step_seconds,
                    self.maximum_step_seconds,
                )
                <= 0
                or self.median_absolute_deviation_seconds < 0
                or abs(
                    self.median_step_seconds
                    - median(self.raw_step_seconds)
                )
                > 1e-9
                or (
                    self.step_metrics
                    and len(self.step_metrics) != self.steps
                )
                or any(
                    not all(isfinite(value) for value in row.values())
                    for row in self.step_metrics
                )
            ):
                raise ValueError(
                    "training execution timing distribution is malformed"
                )
        bound_strings = (
            self.source_run_id,
            self.model_state_digest,
            self.optimizer_state_digest,
            self.tokenizer_identity_digest,
            self.fixture_digest,
            self.hardware_fingerprint,
            self.torch_version,
            self.resolved_variant,
        )
        if any(bound_strings) and (
            any(not value for value in bound_strings)
            or any(
                len(value) != 64
                for value in (
                    self.source_run_id,
                    self.model_state_digest,
                    self.optimizer_state_digest,
                    self.tokenizer_identity_digest,
                    self.fixture_digest,
                    self.hardware_fingerprint,
                )
            )
            or not self.metric_keys
            or tuple(sorted(set(self.metric_keys))) != tuple(
                self.metric_keys
            )
        ):
            raise ValueError(
                "training execution evidence identity is incomplete"
            )


@dataclass(frozen=True, slots=True)
class TrainingExecutionCriterion:
    name: str
    measurement: float
    threshold: float
    direction: str
    unit: str
    passed: bool


@dataclass(frozen=True, slots=True)
class TrainingExecutionAcceptanceReport:
    format_version: int
    suite: str
    samples: tuple[TrainingExecutionSample, ...]
    compiler_candidate: CompilerCandidateReceipt | None
    criteria: tuple[TrainingExecutionCriterion, ...]
    passed: bool
    claim_boundary: str

    def to_dict(self) -> dict:
        return asdict(self)


def _criterion(
    name: str,
    measurement: float,
    threshold: float,
    direction: str,
    unit: str = "ratio",
) -> TrainingExecutionCriterion:
    if direction == "minimum":
        passed = measurement >= threshold
    elif direction == "maximum":
        passed = measurement <= threshold
    else:
        raise ValueError("acceptance criterion direction is unknown")
    return TrainingExecutionCriterion(
        name, measurement, threshold, direction, unit, passed,
    )


def build_acceptance_report(
    samples: tuple[TrainingExecutionSample, ...],
    *,
    compiler_candidate: CompilerCandidateReceipt | None = None,
) -> TrainingExecutionAcceptanceReport:
    by_name = {sample.variant: sample for sample in samples}
    if len(by_name) != len(samples):
        raise ValueError("acceptance sample variant names must be unique")
    aliases = set(by_name) == {"legacy_reference", "repaired"}
    production = (
        set(by_name) == set(PRODUCTION_REQUIRED_VARIANTS)
        or set(by_name) == set(PRODUCTION_VARIANTS)
    )
    if not aliases and not production:
        raise ValueError(
            "acceptance requires either the two compatibility variants or "
            "the complete named production eager matrix, plus a compiler "
            "candidate receipt"
        )
    if production and compiler_candidate is None:
        raise ValueError(
            "production acceptance requires a compiler candidate receipt"
        )
    if compiler_candidate is not None and (
        not production
        or compiler_candidate.profile != samples[0].profile
    ):
        raise ValueError(
            "compiler candidate receipt does not match the sample profile"
        )
    first = samples[0]
    if any(
        sample.profile != first.profile
        or sample.context_length != first.context_length
        or sample.steps != first.steps
        or sample.parameter_count != first.parameter_count
        for sample in samples[1:]
    ):
        raise ValueError("acceptance variants are not matched")
    if aliases:
        legacy = by_name["legacy_reference"]
        repaired = by_name["repaired"]
        speedup = repaired.tokens_per_second / legacy.tokens_per_second
        metrics: Mapping[str, float] = repaired.metrics
        criteria = (
            _criterion("repaired_speedup", speedup, 1.10, "minimum"),
            _criterion(
                "matched_initial_model_optimizer_fixture",
                float(
                    legacy.model_state_digest
                    == repaired.model_state_digest
                    and legacy.optimizer_state_digest
                    == repaired.optimizer_state_digest
                    and legacy.tokenizer_identity_digest
                    == repaired.tokenizer_identity_digest
                    and legacy.fixture_digest == repaired.fixture_digest
                    and legacy.hardware_fingerprint
                    == repaired.hardware_fingerprint
                ),
                1.0,
                "minimum",
                "boolean",
            ),
            _criterion(
                "target_bijection",
                metrics.get("document_batching/target_bijection", 0.0),
                1.0,
                "minimum",
                "boolean",
            ),
            _criterion(
                "sampled_cstm_substrate_vjp_count",
                metrics.get("cstm/substrate_vjp_count", 0.0),
                1.0,
                "maximum",
                "VJPs/context",
            ),
            _criterion(
                "finite_cross_entropy",
                float(
                    isfinite(
                        metrics.get(
                            "train/cross_entropy_nats_per_token",
                            float("nan"),
                        )
                    )
                ),
                1.0,
                "minimum",
                "boolean",
            ),
            _criterion(
                "execution_policy_history_present",
                float(
                    bool(
                        repaired.runtime.get(
                            "activation_execution_policy_digest"
                        )
                    )
                ),
                1.0,
                "minimum",
                "boolean",
            ),
        )
    else:
        legacy = by_name["legacy_serial_checkpoint_dense_cstm"]
        coarse_ce = by_name["static_coarse_checkpoint_ce"]
        repaired_ce = by_name["static_auto_ce"]
        fixed_repaired_cstm = by_name["static_auto_repaired_cstm"]
        repaired_cstm = by_name["static_cost_model_auto_repaired_cstm"]
        production_profile = first.profile == "production_8p4m_32k"
        # The coarse arm executes the already-repaired portable custom
        # adjoints, so requiring the historical 828.5/501 activation-only
        # ratio would count those carrier gains twice. Preserve a strict
        # matched activation non-regression gate and place the large speedup
        # requirement on the complete repaired default versus the immutable
        # fragmented/dense-CSTM reference.
        ce_threshold = 1.00 if production_profile else 0.90
        auxiliary_threshold = 0.85 if production_profile else 0.60
        default_threshold = 2.50 if production_profile else 1.05
        padding_threshold = 0.85 if production_profile else 0.70
        repaired_metrics = repaired_cstm.metrics
        measured_vjps = tuple(
            row.get("cstm/substrate_vjp_count", 0.0)
            for row in (
                repaired_cstm.step_metrics
                if repaired_cstm.step_metrics
                else (repaired_metrics,)
            )
        )
        mean_vjps = sum(measured_vjps) / len(measured_vjps)
        duty = float(
            repaired_cstm.runtime.get(
                "cstm_sampling_duty_cycle", 0.25
            )
        )
        finite_sample_tolerance = 1.0 / max(1, len(measured_vjps))
        padding = repaired_metrics.get(
            "document_batching/padding_efficiency", 0.0
        )
        cost_savings = repaired_metrics.get(
            "document_batching/estimated_savings_fraction", 0.0
        )
        required_phase_metrics = {
            "performance/primary_forward_seconds",
            "performance/loss_forward_seconds",
            "performance/primary_backward_seconds",
            "cstm/predictor_backward_seconds",
            "cstm/substrate_backward_seconds",
            "cstm/gradient_merge_seconds",
            "performance/gradient_reduction_seconds",
            "performance/optimizer_seconds",
            "performance/unattributed_step_seconds",
        }
        assert compiler_candidate is not None
        compiled_sample = by_name.get(COMPILED_VARIANT)
        compiler_resolution_valid = (
            compiler_candidate.outcome == "executed"
            and compiled_sample is not None
            and compiled_sample.resolved_variant == COMPILED_VARIANT
            and bool(
                compiled_sample.runtime.get("compiled_tensor_cores")
            )
            and compiled_sample.runtime.get("carrier_compiler_backend")
            == compiler_candidate.requested_backend
        ) or (
            compiler_candidate.outcome == "timeout"
            and compiled_sample is None
            and not bool(
                repaired_cstm.runtime.get("compiled_tensor_cores", False)
            )
            and compiler_candidate.resolved_variant
            == repaired_cstm.variant
        )
        criteria = (
            _criterion(
                "ce_repaired_vs_coarse_speedup",
                repaired_ce.tokens_per_second
                / coarse_ce.tokens_per_second,
                ce_threshold,
                "minimum",
            ),
            _criterion(
                "repaired_cstm_vs_repaired_ce_throughput",
                repaired_cstm.tokens_per_second
                / repaired_ce.tokens_per_second,
                auxiliary_threshold,
                "minimum",
            ),
            _criterion(
                "repaired_default_vs_legacy_speedup",
                repaired_cstm.tokens_per_second
                / legacy.tokens_per_second,
                default_threshold,
                "minimum",
            ),
            _criterion(
                "padding_or_measured_cost_advantage",
                max(
                    padding / padding_threshold,
                    1.0 + cost_savings if cost_savings > 0 else 0.0,
                    repaired_cstm.tokens_per_second
                    / fixed_repaired_cstm.tokens_per_second,
                ),
                1.0,
                "minimum",
                "normalized gate",
            ),
            _criterion(
                "target_bijection",
                repaired_metrics.get(
                    "document_batching/target_bijection", 0.0
                ),
                1.0,
                "minimum",
                "boolean",
            ),
            _criterion(
                "sampled_cstm_substrate_vjp_count",
                max(measured_vjps),
                1.0,
                "maximum",
                "VJPs/context",
            ),
            _criterion(
                "sampled_cstm_mean_substrate_vjps",
                mean_vjps,
                duty + finite_sample_tolerance,
                "maximum",
                "VJPs/context",
            ),
            _criterion(
                "finite_cross_entropy",
                float(
                    all(
                        isfinite(
                            sample.metrics.get(
                                "train/cross_entropy_nats_per_token",
                                float("nan"),
                            )
                        )
                        for sample in samples
                    )
                ),
                1.0,
                "minimum",
                "boolean",
            ),
            _criterion(
                "timing_distributions_complete",
                float(
                    all(
                        len(sample.raw_step_seconds) == sample.steps
                        for sample in samples
                    )
                ),
                1.0,
                "minimum",
                "boolean",
            ),
            _criterion(
                "phase_timing_contract_complete",
                float(all(
                    required_phase_metrics.issubset(sample.metric_keys)
                    and all(
                        required_phase_metrics.issubset(row)
                        for row in sample.step_metrics
                    )
                    for sample in samples
                )),
                1.0,
                "minimum",
                "boolean",
            ),
            _criterion(
                "evidence_identity_complete",
                float(all(
                    len(sample.source_run_id) == 64
                    and len(sample.model_state_digest) == 64
                    and len(sample.optimizer_state_digest) == 64
                    and len(sample.tokenizer_identity_digest) == 64
                    and len(sample.fixture_digest) == 64
                    and len(sample.hardware_fingerprint) == 64
                    and bool(sample.torch_version)
                    for sample in samples
                )),
                1.0,
                "minimum",
                "boolean",
            ),
            _criterion(
                "matched_initial_model_optimizer_fixture",
                float(
                    len({sample.model_state_digest for sample in samples}) == 1
                    and len({
                        sample.optimizer_state_digest for sample in samples
                    }) == 1
                    and len({
                        sample.tokenizer_identity_digest for sample in samples
                    }) == 1
                    and len({sample.fixture_digest for sample in samples}) == 1
                    and len({
                        sample.hardware_fingerprint for sample in samples
                    }) == 1
                ),
                1.0,
                "minimum",
                "boolean",
            ),
            _criterion(
                "compiler_candidate_bounded_and_truthfully_resolved",
                float(compiler_resolution_valid),
                1.0,
                "minimum",
                "boolean",
            ),
            _criterion(
                "resolved_variant_names_match",
                float(all(
                    sample.resolved_variant == sample.variant
                    for sample in samples
                )),
                1.0,
                "minimum",
                "boolean",
            ),
            _criterion(
                "production_profile_contract",
                float(
                    not production_profile
                    or (
                        first.context_length == 32_768
                        and first.parameter_count == 8_416_803
                        and first.steps >= 3
                    )
                ),
                1.0,
                "minimum",
                "boolean",
            ),
        )
    return TrainingExecutionAcceptanceReport(
        TRAINING_EXECUTION_ACCEPTANCE_FORMAT,
        "mrcra_training_execution_repair",
        samples,
        compiler_candidate,
        criteria,
        all(item.passed for item in criteria),
        (
            "This report proves matched local execution behavior, complete "
            "timing distributions, and measured throughput for the named "
            "hardware/profile. Learning-quality and long-duration resource "
            "acceptance remain separate evidence authorities."
        ),
    )
