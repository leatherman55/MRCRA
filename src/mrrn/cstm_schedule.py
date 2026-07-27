"""Checkpoint-stable scheduling and importance sampling for CSTM obligations.

Sampling changes execution frequency, not the defined dense objective.  For an
obligation with dense numerator ``S_j``, context denominator ``W``, duty
probability ``q``, and conditional selection probability ``p_j``, the emitted
pre-governance estimator is ``S_j / (W q p_j)``.  Its exact expectation is the
dense normalized objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import gcd, isfinite
from typing import Mapping, Sequence


CSTM_SAMPLING_SCHEMA_VERSION = 1
CSTM_COVERAGE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CSTMRowSamplingDecision:
    """Uniform-without-replacement coefficient-row selection receipt."""

    population: int
    selected_indices: tuple[int, ...]
    inclusion_probability: float
    inverse_probability: float
    counter_digest: str

    def __post_init__(self) -> None:
        selected = len(self.selected_indices)
        if (
            self.population <= 0
            or selected <= 0
            or selected > self.population
            or tuple(sorted(set(self.selected_indices)))
            != self.selected_indices
            or self.selected_indices[-1] >= self.population
            or not 0 < self.inclusion_probability <= 1
            or abs(
                self.inclusion_probability - selected / self.population
            ) > 1e-12
            or abs(
                self.inverse_probability
                - 1.0 / self.inclusion_probability
            ) > 1e-9
            or len(self.counter_digest) != 64
        ):
            raise ValueError("CSTM row sampling decision is malformed")


@dataclass(frozen=True, slots=True)
class CSTMObligation:
    invocation: int
    scale: int
    dense_weight: float

    def __post_init__(self) -> None:
        if self.invocation <= 0 or self.scale < 0 or not (
            isfinite(self.dense_weight) and self.dense_weight > 0
        ):
            raise ValueError("CSTM obligation is malformed")


@dataclass(frozen=True, slots=True)
class CSTMSamplingDecision:
    schema_version: int
    active: bool
    obligation: CSTMObligation | None
    duty_probability: float
    conditional_probability: float
    inclusion_probability: float
    inverse_probability: float
    obligation_count: int
    dense_weight: float
    counter_digest: str
    eligible_scales: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != CSTM_SAMPLING_SCHEMA_VERSION
            or not 0 < self.duty_probability <= 1
            or self.obligation_count < 0
            or self.dense_weight < 0
            or len(self.counter_digest) != 64
            or tuple(sorted(set(self.eligible_scales)))
            != self.eligible_scales
            or any(scale < 0 for scale in self.eligible_scales)
        ):
            raise ValueError("CSTM sampling decision is malformed")
        if self.active:
            if (
                self.obligation is None
                or not 0 < self.conditional_probability <= 1
                or not 0 < self.inclusion_probability <= 1
                or abs(
                    self.inclusion_probability
                    - self.duty_probability * self.conditional_probability
                )
                > 1e-12
                or abs(
                    self.inverse_probability
                    - 1.0 / self.inclusion_probability
                )
                > 1e-9
            ):
                raise ValueError("active CSTM sampling receipt is inconsistent")
        elif (
            self.obligation is not None
            or self.conditional_probability != 0
            or self.inclusion_probability != 0
            or self.inverse_probability != 0
        ):
            raise ValueError("inactive CSTM sampling receipt carries an obligation")


@dataclass(slots=True)
class CSTMCoverageState:
    """Checkpointable coverage and update accounting for sampled CSTM."""

    predictor_updates: int = 0
    substrate_updates: int = 0
    coverage_counts: dict[str, int] = field(default_factory=dict)
    last_selected_step: dict[str, int] = field(default_factory=dict)
    required_keys: set[str] = field(default_factory=set)
    last_obligation_digest: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if (
            min(self.predictor_updates, self.substrate_updates) < 0
            or any(
                not isinstance(key, str)
                or not key
                or not isinstance(value, int)
                or value < 0
                for mapping in (
                    self.coverage_counts,
                    self.last_selected_step,
                )
                for key, value in mapping.items()
            )
            or set(self.last_selected_step) - set(self.coverage_counts)
            or any(not isinstance(key, str) or not key for key in self.required_keys)
            or (
                self.last_obligation_digest is not None
                and len(self.last_obligation_digest) != 64
            )
        ):
            raise ValueError("CSTM coverage state is malformed")

    def record_predictor(
        self,
        decision: CSTMSamplingDecision,
        *,
        optimizer_step: int,
        horizons: Sequence[int],
    ) -> None:
        if optimizer_step < 0:
            raise ValueError("CSTM coverage step cannot be negative")
        if not decision.active or decision.obligation is None:
            return
        self.predictor_updates += 1
        keys = (
            f"scale:{decision.obligation.scale}",
            *(f"horizon:{int(value)}" for value in horizons),
        )
        for key in keys:
            self.coverage_counts[key] = self.coverage_counts.get(key, 0) + 1
            self.last_selected_step[key] = optimizer_step
        self.last_obligation_digest = decision.counter_digest
        self._validate()

    def declare_required(self, keys: Sequence[str]) -> None:
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("CSTM required coverage keys are malformed")
        self.required_keys.update(keys)
        self._validate()

    def record_substrate(self, decision: CSTMSamplingDecision) -> None:
        if decision.active:
            if decision.obligation is None:
                raise ValueError("active CSTM substrate decision lacks obligation")
            self.substrate_updates += 1
            self.last_obligation_digest = decision.counter_digest
        self._validate()

    def maximum_gap(
        self,
        *,
        optimizer_step: int,
        required_keys: Sequence[str] | None = None,
    ) -> int:
        keys = tuple(self.required_keys if required_keys is None else required_keys)
        if optimizer_step < 0 or any(not key for key in keys):
            raise ValueError("CSTM coverage query is malformed")
        return max(
            (
                optimizer_step
                - self.last_selected_step.get(key, -1)
                for key in keys
            ),
            default=0,
        )

    def state_dict(self) -> dict[str, object]:
        self._validate()
        return {
            "schema_version": CSTM_COVERAGE_SCHEMA_VERSION,
            "predictor_updates": self.predictor_updates,
            "substrate_updates": self.substrate_updates,
            "coverage_counts": dict(sorted(self.coverage_counts.items())),
            "last_selected_step": dict(
                sorted(self.last_selected_step.items())
            ),
            "required_keys": sorted(self.required_keys),
            "last_obligation_digest": self.last_obligation_digest,
        }

    @classmethod
    def from_state_dict(
        cls, value: Mapping[str, object],
    ) -> "CSTMCoverageState":
        try:
            if int(value["schema_version"]) != CSTM_COVERAGE_SCHEMA_VERSION:
                raise ValueError("unsupported CSTM coverage schema")
            counts = value["coverage_counts"]
            last = value["last_selected_step"]
            required = value["required_keys"]
            if (
                not isinstance(counts, Mapping)
                or not isinstance(last, Mapping)
                or not isinstance(required, (list, tuple))
            ):
                raise TypeError("CSTM coverage mappings are malformed")
            return cls(
                predictor_updates=int(value["predictor_updates"]),
                substrate_updates=int(value["substrate_updates"]),
                coverage_counts={
                    str(key): int(item) for key, item in counts.items()
                },
                last_selected_step={
                    str(key): int(item) for key, item in last.items()
                },
                required_keys={str(key) for key in required},
                last_obligation_digest=(
                    None
                    if value["last_obligation_digest"] is None
                    else str(value["last_obligation_digest"])
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("serialized CSTM coverage state is malformed") from error


def _counter_uniform(
    *, seed: int, optimizer_step: int, target_digest: str, stream: int,
) -> tuple[float, str]:
    if seed < 0 or optimizer_step < 0 or stream < 0 or len(target_digest) != 64:
        raise ValueError("CSTM counter identity is malformed")
    digest = sha256(
        (
            f"cstm-sampling-v{CSTM_SAMPLING_SCHEMA_VERSION}|{seed}|"
            f"{optimizer_step}|{target_digest}|{stream}"
        ).encode("ascii")
    ).hexdigest()
    value = int(digest[:16], 16)
    return (value + 0.5) / float(1 << 64), digest


def deterministic_cstm_sample(
    obligations: Sequence[CSTMObligation],
    *,
    duty_probability: float,
    seed: int,
    optimizer_step: int,
    target_digest: str,
    uniform_mixture: float = 0.05,
    uniform_override: tuple[float, float] | None = None,
) -> CSTMSamplingDecision:
    """Select at most one physical-invocation/scale obligation.

    Duty decisions use randomized-phase systematic sampling.  For any
    contiguous window of ``n`` optimizer steps, the number of active substrate
    updates is one of ``floor(n*q)`` or ``ceil(n*q)``.  Across the hashed phase,
    every step still has first-order inclusion probability ``q``; inverse-duty
    weighting therefore remains unbiased while short-run compute is tightly
    bounded.  The categorical obligation choice remains counter-randomized per
    step.
    """

    obligations = tuple(obligations)
    if not 0 < duty_probability <= 1:
        raise ValueError("CSTM duty probability must lie in (0,1]")
    if not 0 <= uniform_mixture < 1:
        raise ValueError("CSTM uniform mixture must lie in [0,1)")
    if len({
        (item.invocation, item.scale) for item in obligations
    }) != len(obligations):
        raise ValueError("CSTM obligation identities must be unique")
    total = sum(item.dense_weight for item in obligations)
    eligible_scales = tuple(sorted({item.scale for item in obligations}))
    phase_digest = sha256(
        f"cstm-duty-phase-v1|{seed}".encode("ascii")
    ).hexdigest()
    phase_integer = int(phase_digest[:16], 16)
    duty_numerator, duty_denominator = duty_probability.as_integer_ratio()
    phase_numerator = 2 * phase_integer + 1
    phase_denominator = 1 << 65

    def systematic_count(step: int) -> int:
        numerator = (
            step
            * duty_numerator
            * phase_denominator
            + phase_numerator * duty_denominator
        )
        return numerator // (duty_denominator * phase_denominator)

    duty_active = (
        systematic_count(optimizer_step + 1)
        > systematic_count(optimizer_step)
    )
    choice_uniform, choice_digest = _counter_uniform(
        seed=seed,
        optimizer_step=optimizer_step,
        target_digest=target_digest,
        stream=1,
    )
    if uniform_override is not None:
        duty_uniform, choice_uniform = uniform_override
        if not (
            0 <= duty_uniform < 1 and 0 <= choice_uniform < 1
        ):
            raise ValueError("CSTM uniform overrides must lie in [0,1)")
        duty_active = duty_uniform < duty_probability
    receipt_digest = sha256(
        (phase_digest + choice_digest).encode("ascii")
    ).hexdigest()
    if not obligations or total <= 0 or not duty_active:
        return CSTMSamplingDecision(
            CSTM_SAMPLING_SCHEMA_VERSION,
            False,
            None,
            duty_probability,
            0.0,
            0.0,
            0.0,
            len(obligations),
            float(total),
            receipt_digest,
            eligible_scales,
        )
    probabilities = tuple(
        (1 - uniform_mixture) * item.dense_weight / total
        + uniform_mixture / len(obligations)
        for item in obligations
    )
    threshold = choice_uniform
    cumulative = 0.0
    selected = obligations[-1]
    selected_probability = probabilities[-1]
    for obligation, probability in zip(
        obligations, probabilities, strict=True,
    ):
        cumulative += probability
        if threshold < cumulative:
            selected = obligation
            selected_probability = probability
            break
    conditional = selected_probability
    inclusion = duty_probability * conditional
    return CSTMSamplingDecision(
        CSTM_SAMPLING_SCHEMA_VERSION,
        True,
        selected,
        duty_probability,
        conditional,
        inclusion,
        1.0 / inclusion,
        len(obligations),
        float(total),
        receipt_digest,
        eligible_scales,
    )


def deterministic_cstm_rows(
    population: int,
    budget: int,
    *,
    counter_digest: str,
    stream: int,
) -> CSTMRowSamplingDecision:
    """Select a bounded uniform subset by a content-bound cyclic permutation.

    Across uniformly distributed counter digests every row has exact
    first-order inclusion probability ``min(1, budget / population)``.  Hash
    selected start/stride parameters define a permutation without replacement
    and never consume a Torch or Python RNG. Sorted returned indices preserve
    causal row order during tensor gathering.
    """

    if (
        population <= 0
        or budget <= 0
        or stream < 0
        or len(counter_digest) != 64
    ):
        raise ValueError("CSTM row sampling request is malformed")
    selected_count = min(population, budget)
    authority = sha256(
        f"cstm-row-v1|{counter_digest}|{stream}".encode("ascii")
    ).digest()
    start = int.from_bytes(authority[:8], "big") % population
    stride = int.from_bytes(authority[8:16], "big") % population
    stride = max(1, stride)
    while gcd(stride, population) != 1:
        stride = stride % population + 1
    selected = tuple(sorted(
        (start + offset * stride) % population
        for offset in range(selected_count)
    ))
    digest = sha256(
        (
            f"{counter_digest}|{stream}|{population}|{budget}|"
            + ",".join(map(str, selected))
        ).encode("ascii")
    ).hexdigest()
    inclusion = selected_count / population
    return CSTMRowSamplingDecision(
        population,
        selected,
        inclusion,
        1.0 / inclusion,
        digest,
    )
