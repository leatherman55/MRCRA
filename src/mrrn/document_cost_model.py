"""Portable cost authority for MRCRA document-major execution.

The model is deliberately small, deterministic, and serializable.  It does not
guess semantic behavior: it estimates only physical work for candidate static
cohorts, allowing the document planner to trade a little padding for fewer
carrier/cognition/autograd launches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Mapping, Sequence


DOCUMENT_COST_MODEL_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class DocumentExecutionCostModel:
    """Calibratable affine estimate of one cohort's execution cost."""

    schema_version: int = DOCUMENT_COST_MODEL_SCHEMA_VERSION
    launch_cost: float = 20_000.0
    token_forward_cost: float = 1.0
    padding_cost: float = 0.35
    cognitive_anchor_cost: float = 48.0
    backward_multiplier_retain: float = 2.15
    backward_multiplier_selective: float = 2.65
    backward_multiplier_whole_span: float = 3.35
    calibration_kind: str = "portable_analytic"
    hardware_fingerprint: str = "portable"
    # Upper-bound length -> measured seconds per physical token. Empty retains
    # the portable scalar token cost for format-16 compatibility.
    token_seconds_by_length_band: tuple[tuple[int, float], ...] = ()
    shape_compile_cost: float = 0.0
    activation_bytes_per_token: int = 0
    memory_cost_per_byte: float = 0.0
    # Explicit measured overrides: (batch, padded_length, peak_bytes).
    memory_bytes_by_shape: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        if (
            self.schema_version != DOCUMENT_COST_MODEL_SCHEMA_VERSION
            or min(
                self.launch_cost,
                self.token_forward_cost,
                self.padding_cost,
                self.cognitive_anchor_cost,
                self.backward_multiplier_retain,
                self.backward_multiplier_selective,
                self.backward_multiplier_whole_span,
                self.shape_compile_cost,
                self.activation_bytes_per_token,
                self.memory_cost_per_byte,
            )
            < 0
            or not self.calibration_kind
            or not self.hardware_fingerprint
            or tuple(
                sorted(
                    self.token_seconds_by_length_band,
                    key=lambda item: item[0],
                )
            )
            != self.token_seconds_by_length_band
            or len({
                upper for upper, _ in self.token_seconds_by_length_band
            })
            != len(self.token_seconds_by_length_band)
            or any(
                upper <= 0 or not isinstance(seconds, (int, float))
                or not isfinite(float(seconds)) or seconds < 0
                for upper, seconds in self.token_seconds_by_length_band
            )
            or len({
                (batch, length)
                for batch, length, _ in self.memory_bytes_by_shape
            })
            != len(self.memory_bytes_by_shape)
            or any(
                min(batch, length) <= 0 or peak < 0
                for batch, length, peak in self.memory_bytes_by_shape
            )
        ):
            raise ValueError("document execution cost model is malformed")

    @property
    def digest(self) -> str:
        return sha256(
            json.dumps(
                asdict(self), sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-safe checkpoint authority."""

        return asdict(self)

    def backward_multiplier(self, activation_policy: str) -> float:
        try:
            return {
                "retain": self.backward_multiplier_retain,
                "selective": self.backward_multiplier_selective,
                "whole_span": self.backward_multiplier_whole_span,
            }[activation_policy]
        except KeyError:
            raise ValueError("unknown activation policy in document cost") from None

    def token_cost(self, padded_length: int) -> float:
        if padded_length <= 0:
            raise ValueError("document cost length must be positive")
        for upper, seconds in self.token_seconds_by_length_band:
            if padded_length <= upper:
                return float(seconds)
        if self.token_seconds_by_length_band:
            return float(self.token_seconds_by_length_band[-1][1])
        return self.token_forward_cost

    def shape_memory_bytes(self, batch: int, padded_length: int) -> int:
        if min(batch, padded_length) <= 0:
            raise ValueError("document cost shape must be positive")
        measured = {
            (candidate_batch, candidate_length): peak
            for candidate_batch, candidate_length, peak
            in self.memory_bytes_by_shape
        }.get((batch, padded_length))
        if measured is not None:
            return measured
        return batch * padded_length * self.activation_bytes_per_token

    def estimate(
        self,
        *,
        padded_lengths: tuple[int, ...],
        valid_lengths_by_row: tuple[tuple[int, ...], ...],
        cognitive_stride: int,
        activation_policy: str,
        known_shapes: frozenset[tuple[int, int]] = frozenset(),
        compiler_enabled: bool = False,
    ) -> float:
        """Estimate forward+backward work for one stable-row cohort."""

        if (
            not padded_lengths
            or not valid_lengths_by_row
            or cognitive_stride <= 0
            or any(len(row) != len(padded_lengths) for row in valid_lengths_by_row)
            or any(length <= 0 for length in padded_lengths)
            or any(length <= 0 for row in valid_lengths_by_row for length in row)
            or any(
                min(batch, length) <= 0
                for batch, length in known_shapes
            )
        ):
            raise ValueError("document cost candidate has malformed dimensions")
        rows = len(valid_lengths_by_row)
        physical = rows * sum(padded_lengths)
        valid = sum(sum(row) for row in valid_lengths_by_row)
        padding = physical - valid
        anchors = rows * sum(
            (length + cognitive_stride - 1) // cognitive_stride
            for length in padded_lengths
        )
        shapes = frozenset((rows, length) for length in padded_lengths)
        token_seconds = sum(
            rows * length * self.token_cost(length)
            for length in padded_lengths
        )
        peak_memory = max(
            self.shape_memory_bytes(rows, length)
            for length in padded_lengths
        )
        forward = (
            len(padded_lengths) * self.launch_cost
            + token_seconds
            + padding * self.padding_cost
            + anchors * self.cognitive_anchor_cost
            + (
                self.shape_compile_cost * len(shapes - known_shapes)
                if compiler_enabled else 0.0
            )
            + self.memory_cost_per_byte * peak_memory
        )
        return forward * self.backward_multiplier(activation_policy)

    @classmethod
    def from_dict(
        cls, value: Mapping[str, object],
    ) -> "DocumentExecutionCostModel":
        try:
            payload = dict(value)
            payload["token_seconds_by_length_band"] = tuple(
                (int(bound), float(seconds))
                for bound, seconds in payload.get(
                    "token_seconds_by_length_band", ()
                )
            )
            payload["memory_bytes_by_shape"] = tuple(
                (int(batch), int(length), int(memory))
                for batch, length, memory in payload.get(
                    "memory_bytes_by_shape", ()
                )
            )
            return cls(**payload)
        except (TypeError, ValueError) as error:
            raise ValueError("serialized document cost model is malformed") from error


@dataclass(frozen=True, slots=True)
class DocumentPlanCostReceipt:
    """Auditable comparison between selected and legacy-exact grouping."""

    schema_version: int
    policy: str
    cost_model_digest: str
    selected_estimated_cost: float
    exact_signature_estimated_cost: float
    selected_invocations: int
    exact_signature_invocations: int
    rejected_memory_candidates: int
    cache_hit: bool
    unique_static_shapes: int = 0
    predicted_peak_memory_bytes: int = 0
    shape_compile_cost: float = 0.0

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.policy not in {"cost_aware", "exact_signature"}
            or len(self.cost_model_digest) != 64
            or min(
                self.selected_estimated_cost,
                self.exact_signature_estimated_cost,
                self.selected_invocations,
                self.exact_signature_invocations,
                self.rejected_memory_candidates,
                self.unique_static_shapes,
                self.predicted_peak_memory_bytes,
                self.shape_compile_cost,
            )
            < 0
        ):
            raise ValueError("document plan cost receipt is malformed")

    @property
    def estimated_savings_fraction(self) -> float:
        if self.exact_signature_estimated_cost == 0:
            return 0.0
        return 1.0 - (
            self.selected_estimated_cost / self.exact_signature_estimated_cost
        )


def measured_document_cost_model(
    *,
    single_invocation_seconds: float,
    batched_invocation_seconds: float,
    single_physical_tokens: int,
    batched_physical_tokens: int,
    length_bands: Sequence[int],
    activation_policy: str,
    hardware_fingerprint: str,
    activation_bytes_per_token: int,
    shape_compile_cost: float = 0.0,
    length_band_observations: Sequence[
        tuple[int, float, int, int]
    ] = (),
) -> DocumentExecutionCostModel:
    """Fit a nonnegative affine cost authority from two real carrier probes.

    The fit uses no guessed split between launch and token work. With
    ``T(p)=a+b p``, two measured physical-token counts determine ``a`` and
    ``b`` exactly; a negative intercept caused by timing noise is clamped to
    zero and the token coefficient is recomputed from the larger observation.
    Optional per-band observations are ``(padded_length, elapsed_seconds,
    batch_size, peak_bytes)``. They refine the token coefficient at every
    authorized length and bind exact measured memory for that static shape.
    """

    if (
        not all(isfinite(value) for value in (
            single_invocation_seconds,
            batched_invocation_seconds,
            shape_compile_cost,
        ))
        or min(single_invocation_seconds, batched_invocation_seconds) <= 0
        or min(single_physical_tokens, batched_physical_tokens) <= 0
        or batched_physical_tokens <= single_physical_tokens
        or not length_bands
        or tuple(sorted(set(int(value) for value in length_bands)))
        != tuple(length_bands)
        or activation_policy not in {"retain", "selective", "whole_span"}
        or len(hardware_fingerprint) != 64
        or activation_bytes_per_token <= 0
        or shape_compile_cost < 0
        or (
            length_band_observations
            and (
                tuple(
                    sorted(
                        observation[0]
                        for observation in length_band_observations
                    )
                )
                != tuple(
                    observation[0]
                    for observation in length_band_observations
                )
                or len({
                    observation[0]
                    for observation in length_band_observations
                })
                != len(length_band_observations)
                or any(
                    observation[0] not in length_bands
                    for observation in length_band_observations
                )
            )
        )
        or any(
            len(observation) != 4
            or observation[0] <= 0
            or not isfinite(float(observation[1]))
            or observation[1] <= 0
            or observation[2] <= 0
            or observation[3] < 0
            for observation in length_band_observations
        )
    ):
        raise ValueError("measured document cost calibration is malformed")
    token_cost = max(
        0.0,
        (
            batched_invocation_seconds - single_invocation_seconds
        )
        / (batched_physical_tokens - single_physical_tokens),
    )
    launch = max(
        0.0,
        single_invocation_seconds
        - token_cost * single_physical_tokens,
    )
    if launch == 0.0:
        token_cost = (
            batched_invocation_seconds / batched_physical_tokens
        )
    multipliers = {
        "retain": (1.0, 1.0, 1.0),
        "selective": (1.0, 1.0, 1.0),
        "whole_span": (1.0, 1.0, 1.0),
    }[activation_policy]
    if length_band_observations:
        observed_band_costs = tuple(
            (
                int(length),
                max(
                    0.0,
                    (float(seconds) - launch)
                    / (int(batch) * int(length)),
                ),
            )
            for length, seconds, batch, _ in length_band_observations
        )
        # Timing noise can place a short observation below the fitted launch
        # intercept. A zero token coefficient would make padding free, so use
        # the global measured coefficient only for that degenerate band.
        observed_band_costs = tuple(
            (length, cost if cost > 0 else token_cost)
            for length, cost in observed_band_costs
        )
        observed_by_length = dict(observed_band_costs)
        band_costs = tuple(
            (
                int(length),
                observed_by_length.get(
                    int(length),
                    next(
                        (
                            cost
                            for observed_length, cost
                            in observed_band_costs
                            if observed_length >= int(length)
                        ),
                        observed_band_costs[-1][1],
                    ),
                ),
            )
            for length in length_bands
        )
        measured_memory = tuple(
            (int(batch), int(length), int(peak))
            for length, _, batch, peak in length_band_observations
        )
        calibration_kind = (
            "measured_affine_plus_per_length_carrier_forward_backward"
        )
    else:
        band_costs = tuple(
            (int(upper), token_cost) for upper in length_bands
        )
        measured_memory = ()
        calibration_kind = "measured_affine_carrier_forward_backward"
    return DocumentExecutionCostModel(
        launch_cost=launch,
        token_forward_cost=token_cost,
        padding_cost=token_cost,
        cognitive_anchor_cost=0.0,
        backward_multiplier_retain=multipliers[0],
        backward_multiplier_selective=multipliers[1],
        backward_multiplier_whole_span=multipliers[2],
        calibration_kind=calibration_kind,
        hardware_fingerprint=hardware_fingerprint,
        token_seconds_by_length_band=band_costs,
        shape_compile_cost=shape_compile_cost,
        activation_bytes_per_token=activation_bytes_per_token,
        memory_bytes_by_shape=measured_memory,
    )
