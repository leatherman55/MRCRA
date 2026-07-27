"""Measured activation-execution policy for MRCRA training.

Activation retention is an execution decision, not part of the learned
function.  This module keeps that decision explicit and serializable, measures
candidate callables without consuming the global RNG, and chooses the fastest
candidate that leaves a declared memory reserve.

The module deliberately knows nothing about MRCRA model objects.  The carrier
supplies one callable per semantically matched execution policy; tests can
therefore validate policy selection independently from a large model and the
trainer can calibrate the exact physical cohort shapes it will execute.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
from hashlib import sha256
import json
from math import ceil
import os
import platform
import resource
import sys
from threading import Event, Thread
from time import perf_counter, sleep
from typing import Callable, Mapping

import torch
from torch import Tensor


ACTIVATION_EXECUTION_POLICY_SCHEMA_VERSION = 1
ACTIVATION_POLICIES = ("retain", "selective", "whole_span")
ACTIVATION_REQUESTS = ("auto", *ACTIVATION_POLICIES)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def hardware_fingerprint(device: torch.device) -> str:
    """Return a stable execution-hardware identity without user-local paths."""

    device_name = platform.processor() or platform.machine()
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        device_name = f"Apple-{platform.machine()}"
    payload = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": device_name,
        "python": platform.python_version(),
        # Newer PyTorch releases expose ``TorchVersion``, a ``str`` subclass
        # that the weights-only checkpoint loader intentionally rejects as a
        # non-primitive global. Normalize it at the policy boundary.
        "torch": str(torch.__version__),
        "device_type": device.type,
        "device_index": device.index,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _host_total_memory() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(
            os.sysconf("SC_PHYS_PAGES")
        )
    except (AttributeError, OSError, TypeError, ValueError):
        try:
            import psutil

            return int(psutil.virtual_memory().total)
        except (ImportError, AttributeError):
            return 8 << 30


def _host_available_memory() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except (ImportError, AttributeError):
        # ``SC_AVPHYS_PAGES`` is available on Linux but not every macOS/Python
        # combination.  Falling back to one half of installed capacity is
        # intentionally conservative and is named in the receipt.
        try:
            return int(os.sysconf("SC_PAGE_SIZE")) * int(
                os.sysconf("SC_AVPHYS_PAGES")
            )
        except (AttributeError, OSError, TypeError, ValueError):
            return _host_total_memory() // 2


@dataclass(frozen=True, slots=True)
class MemoryObservation:
    """Available capacity at calibration time."""

    device_type: str
    total_bytes: int
    available_bytes: int
    observation_kind: str

    def __post_init__(self) -> None:
        if (
            self.device_type not in {"cpu", "mps", "cuda"}
            or min(self.total_bytes, self.available_bytes) <= 0
            or self.available_bytes > self.total_bytes
            or not self.observation_kind
        ):
            raise ValueError("activation memory observation is malformed")


def observe_memory(device: torch.device) -> MemoryObservation:
    """Observe usable memory for CPU, unified-memory MPS, or CUDA."""

    if device.type == "cuda":
        index = (
            torch.cuda.current_device()
            if device.index is None else int(device.index)
        )
        free, total = torch.cuda.mem_get_info(index)
        return MemoryObservation(
            "cuda", int(total), int(free), "cuda_mem_get_info"
        )
    total = _host_total_memory()
    available = min(total, _host_available_memory())
    return MemoryObservation(
        device.type,
        total,
        available,
        (
            "host_available_unified_memory"
            if device.type == "mps"
            else "host_available_memory"
        ),
    )


@dataclass(frozen=True, slots=True)
class ActivationCandidateMeasurement:
    """One semantically equivalent policy's measured resource cost."""

    policy: str
    elapsed_seconds: float
    incremental_peak_bytes: int
    absolute_peak_bytes: int
    output_digest: str
    finite: bool
    calibration_physical_tokens: int = 1
    target_physical_tokens: int = 1
    projected_incremental_peak_bytes: int = 0

    def __post_init__(self) -> None:
        if (
            self.policy not in ACTIVATION_POLICIES
            or self.elapsed_seconds < 0
            or min(self.incremental_peak_bytes, self.absolute_peak_bytes) < 0
            or len(self.output_digest) != 64
            or self.calibration_physical_tokens <= 0
            or self.target_physical_tokens < self.calibration_physical_tokens
            or self.projected_incremental_peak_bytes < 0
        ):
            raise ValueError("activation candidate measurement is malformed")

    @property
    def reserve_peak_bytes(self) -> int:
        """Conservative peak used for production reserve authorization."""

        return max(
            self.incremental_peak_bytes,
            self.projected_incremental_peak_bytes,
        )


@dataclass(frozen=True, slots=True)
class ActivationPartitionCensus:
    """Saved-tensor census for one independently checkpointable partition."""

    partition: str
    retained_saved_bytes: int
    partition_checkpointed_saved_bytes: int
    saved_byte_reduction: int
    elapsed_seconds: float
    output_digest: str

    def __post_init__(self) -> None:
        if (
            not self.partition
            or min(
                self.retained_saved_bytes,
                self.partition_checkpointed_saved_bytes,
                self.saved_byte_reduction,
            )
            < 0
            or self.saved_byte_reduction
            != max(
                0,
                self.retained_saved_bytes
                - self.partition_checkpointed_saved_bytes,
            )
            or self.elapsed_seconds < 0
            or len(self.output_digest) != 64
        ):
            raise ValueError("activation partition census is malformed")


@dataclass(frozen=True, slots=True)
class ActivationExecutionPolicy:
    """Resolved policy plus the evidence that authorized it."""

    schema_version: int
    requested: str
    resolved: str
    calibration_kind: str
    memory: MemoryObservation
    required_reserve_bytes: int
    estimated_retain_bytes: int
    candidates: tuple[ActivationCandidateMeasurement, ...]
    hardware_fingerprint: str
    torch_version: str
    reason: str

    def __post_init__(self) -> None:
        if (
            self.schema_version != ACTIVATION_EXECUTION_POLICY_SCHEMA_VERSION
            or self.requested not in ACTIVATION_REQUESTS
            or self.resolved not in ACTIVATION_POLICIES
            or self.required_reserve_bytes < 0
            or self.estimated_retain_bytes < 0
            or len(self.hardware_fingerprint) != 64
            or not self.calibration_kind
            or not self.torch_version
            or not self.reason
            or len({item.policy for item in self.candidates})
            != len(self.candidates)
        ):
            raise ValueError("activation execution policy is malformed")

    @property
    def digest(self) -> str:
        return sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ActivationExecutionPolicy":
        try:
            memory = MemoryObservation(**dict(value["memory"]))
            candidates = tuple(
                ActivationCandidateMeasurement(**dict(item))
                for item in value["candidates"]
            )
            return cls(
                schema_version=int(value["schema_version"]),
                requested=str(value["requested"]),
                resolved=str(value["resolved"]),
                calibration_kind=str(value["calibration_kind"]),
                memory=memory,
                required_reserve_bytes=int(value["required_reserve_bytes"]),
                estimated_retain_bytes=int(value["estimated_retain_bytes"]),
                candidates=candidates,
                hardware_fingerprint=str(value["hardware_fingerprint"]),
                torch_version=str(value["torch_version"]),
                reason=str(value["reason"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "serialized activation execution policy is malformed"
            ) from error


class _ResidentMemorySampler:
    """Bounded background RSS sampler used only during calibration."""

    def __init__(self, interval_seconds: float = 0.002) -> None:
        if interval_seconds <= 0:
            raise ValueError("RSS sampling interval must be positive")
        self.interval_seconds = interval_seconds
        self._stop = Event()
        self._thread: Thread | None = None
        self.baseline = 0
        self.peak = 0

    @staticmethod
    def _rss() -> int:
        try:
            import psutil

            return int(psutil.Process().memory_info().rss)
        except (ImportError, AttributeError):
            # macOS reports bytes and Linux reports KiB for ru_maxrss.
            value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return value if sys.platform == "darwin" else value << 10

    def __enter__(self) -> "_ResidentMemorySampler":
        self.baseline = self.peak = self._rss()

        def sample() -> None:
            while not self._stop.is_set():
                self.peak = max(self.peak, self._rss())
                sleep(self.interval_seconds)
            self.peak = max(self.peak, self._rss())

        self._thread = Thread(
            target=sample, name="mrcra-activation-rss-sampler", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                raise RuntimeError("activation RSS sampler did not stop")


def _tensor_digest(value: Tensor) -> str:
    local = value.detach().float().cpu().contiguous()
    digest = sha256()
    digest.update(str(tuple(local.shape)).encode("ascii"))
    digest.update(local.numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def _preserve_global_rng(device: torch.device):
    """Restore every RNG owned by the selected execution process."""

    cpu_state = torch.random.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    )
    mps_state = (
        torch.mps.get_rng_state()
        if device.type == "mps" and hasattr(torch.mps, "get_rng_state")
        else None
    )
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def measure_activation_candidate(
    policy: str,
    function: Callable[[], Tensor],
    *,
    device: torch.device,
    calibration_physical_tokens: int = 1,
    target_physical_tokens: int = 1,
    conservative_peak_bytes: int = 0,
) -> ActivationCandidateMeasurement:
    """Measure one candidate and bind its finite output to a digest."""

    if (
        policy not in ACTIVATION_POLICIES
        or calibration_physical_tokens <= 0
        or target_physical_tokens < calibration_physical_tokens
        or conservative_peak_bytes < 0
    ):
        raise ValueError("unknown activation execution candidate")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = int(torch.cuda.memory_allocated(device))
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
        baseline = int(torch.mps.current_allocated_memory())
    else:
        baseline = 0
    _synchronize(device)
    with _preserve_global_rng(device), _ResidentMemorySampler() as rss:
        started = perf_counter()
        output = function()
        _synchronize(device)
        elapsed = perf_counter() - started
    if not isinstance(output, Tensor) or output.numel() <= 0:
        raise ValueError("activation calibration callable must return a tensor")
    finite = bool(torch.isfinite(output).all())
    if device.type == "cuda":
        absolute_peak = int(torch.cuda.max_memory_allocated(device))
        incremental = max(0, absolute_peak - baseline)
    elif device.type == "mps":
        absolute_peak = int(torch.mps.driver_allocated_memory())
        incremental = max(
            0, int(torch.mps.current_allocated_memory()) - baseline
        )
    else:
        absolute_peak = rss.peak
        incremental = max(0, rss.peak - rss.baseline)
    projected = max(
        conservative_peak_bytes,
        ceil(
            incremental
            * target_physical_tokens
            / calibration_physical_tokens
        ),
    )
    return ActivationCandidateMeasurement(
        policy,
        elapsed,
        incremental,
        absolute_peak,
        _tensor_digest(output),
        finite,
        calibration_physical_tokens,
        target_physical_tokens,
        projected,
    )


def resolve_activation_execution_policy(
    *,
    requested: str,
    device: torch.device,
    required_reserve_bytes: int,
    estimated_retain_bytes: int,
    candidates: tuple[ActivationCandidateMeasurement, ...] = (),
    allow_unsafe_explicit: bool = False,
    memory_observation: MemoryObservation | None = None,
) -> ActivationExecutionPolicy:
    """Choose the fastest finite candidate satisfying the memory reserve.

    ``memory_observation`` may be captured immediately before candidate
    calibration.  This matters on host allocators that retain freed calibration
    arenas: observing memory only after all candidates would count those
    reusable arenas as unavailable and could force a spurious safer/slower
    policy.  The supplied observation is still a real live-memory measurement;
    it is not a configured capacity or installed-memory heuristic.
    """

    if requested not in ACTIVATION_REQUESTS:
        raise ValueError("unknown activation execution request")
    if min(required_reserve_bytes, estimated_retain_bytes) < 0:
        raise ValueError("activation memory sizes cannot be negative")
    if not isinstance(allow_unsafe_explicit, bool):
        raise ValueError("unsafe activation override must be boolean")
    memory = (
        observe_memory(device)
        if memory_observation is None
        else memory_observation
    )
    if memory.device_type != device.type:
        raise ValueError(
            "activation memory observation belongs to a different device"
        )
    fingerprints = hardware_fingerprint(device)
    measured = {item.policy: item for item in candidates}
    if any(not item.finite for item in candidates):
        measured = {
            name: item for name, item in measured.items() if item.finite
        }
    output_digests = {item.output_digest for item in measured.values()}
    if len(output_digests) > 1:
        raise RuntimeError(
            "activation candidates are not forward-equivalent; refusing "
            "performance-based policy selection"
        )

    def fits(item: ActivationCandidateMeasurement) -> bool:
        return (
            item.reserve_peak_bytes + required_reserve_bytes
            <= memory.available_bytes
        )

    if requested != "auto":
        selected = measured.get(requested)
        if (
            selected is not None
            and not fits(selected)
            and not allow_unsafe_explicit
        ):
            raise MemoryError(
                f"requested activation policy {requested!r} exceeds the "
                "measured available-memory reserve"
            )
        if (
            selected is None
            and requested in {"retain", "selective"}
            and estimated_retain_bytes + required_reserve_bytes
            > memory.available_bytes
            and not allow_unsafe_explicit
        ):
            # Without a measured selective census there is no defensible
            # reduction factor. Treat the full-retain estimate as the
            # conservative upper bound and refuse an explicitly unsafe policy;
            # callers can select whole_span or enable calibration.
            raise MemoryError(
                f"requested activation policy {requested!r} lacks a measured "
                "candidate and its conservative activation estimate exceeds "
                "the observed available-memory reserve"
            )
        resolved = requested
        reason = (
            "explicit unsafe override accepted outside measured reserve"
            if (
                allow_unsafe_explicit
                and (
                    (selected is not None and not fits(selected))
                    or (
                        selected is None
                        and requested in {"retain", "selective"}
                        and estimated_retain_bytes + required_reserve_bytes
                        > memory.available_bytes
                    )
                )
            )
            else "explicit measured policy within reserve"
            if selected is not None
            else (
                "explicit whole-span recomputation without candidate calibration"
                if requested == "whole_span"
                else "explicit conservative estimate within reserve"
            )
        )
    elif measured:
        feasible = tuple(item for item in measured.values() if fits(item))
        if feasible:
            selected = min(
                feasible,
                key=lambda item: (
                    item.elapsed_seconds,
                    item.incremental_peak_bytes,
                    ACTIVATION_POLICIES.index(item.policy),
                ),
            )
            resolved = selected.policy
            reason = "fastest measured finite policy satisfying reserve"
        else:
            resolved = "whole_span"
            reason = "no measured candidate satisfied reserve; safe fallback"
    elif estimated_retain_bytes + required_reserve_bytes <= memory.available_bytes:
        resolved = "retain"
        reason = "retain estimate satisfies observed available-memory reserve"
    else:
        resolved = "whole_span"
        reason = "retain estimate exceeds observed available-memory reserve"
    return ActivationExecutionPolicy(
        ACTIVATION_EXECUTION_POLICY_SCHEMA_VERSION,
        requested,
        resolved,
        "measured_candidates" if candidates else "estimate_plus_live_memory",
        memory,
        required_reserve_bytes,
        estimated_retain_bytes,
        tuple(candidates),
        fingerprints,
        str(torch.__version__),
        reason,
    )


def calibrate_activation_candidates(
    candidates: Mapping[str, Callable[[], Tensor]],
    *,
    device: torch.device,
    calibration_physical_tokens: int = 1,
    target_physical_tokens: int = 1,
    conservative_peak_bytes: Mapping[str, int] | None = None,
) -> tuple[ActivationCandidateMeasurement, ...]:
    """Measure candidates in deterministic policy order.

    The caller owns model/RNG restoration around each callable because only it
    knows which buffers and external states are authoritative.
    """

    unknown = set(candidates) - set(ACTIVATION_POLICIES)
    if unknown:
        raise ValueError(
            f"unknown activation calibration policies: {sorted(unknown)}"
        )
    conservative = {} if conservative_peak_bytes is None else dict(
        conservative_peak_bytes
    )
    if set(conservative) - set(candidates) or any(
        not isinstance(value, int) or value < 0
        for value in conservative.values()
    ):
        raise ValueError("activation conservative peaks are malformed")
    return tuple(
        measure_activation_candidate(
            name,
            candidates[name],
            device=device,
            calibration_physical_tokens=calibration_physical_tokens,
            target_physical_tokens=target_physical_tokens,
            conservative_peak_bytes=conservative.get(name, 0),
        )
        for name in ACTIVATION_POLICIES
        if name in candidates
    )


def _saved_tensor_bytes(function: Callable[[], Tensor]) -> tuple[int, float, str]:
    """Measure tensors retained by one forward graph without retaining copies."""

    saved_bytes = 0

    def pack(value: Tensor) -> Tensor:
        nonlocal saved_bytes
        saved_bytes += value.numel() * value.element_size()
        return value

    started = perf_counter()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        output = function()
    elapsed = perf_counter() - started
    if not isinstance(output, Tensor) or output.numel() <= 0:
        raise ValueError("activation census callable must return a tensor")
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError("activation census produced non-finite output")
    return saved_bytes, elapsed, _tensor_digest(output)


def measure_saved_tensor_bytes(
    function: Callable[[], Tensor],
    *,
    device: torch.device,
) -> tuple[int, float, str]:
    """Public, RNG-neutral saved-tensor census for one execution candidate."""

    with _preserve_global_rng(device):
        return _saved_tensor_bytes(function)


def census_activation_partitions(
    retained: Callable[[], Tensor],
    checkpointed_partitions: Mapping[str, Callable[[], Tensor]],
    *,
    device: torch.device,
) -> tuple[ActivationPartitionCensus, ...]:
    """Measure saved-tensor reduction from checkpointing each partition alone.

    This is deliberately a forward-graph census rather than a parameter-count
    heuristic. The returned digest also proves that every independently
    checkpointed candidate preserved the calibrated forward value.
    """

    if not checkpointed_partitions or any(
        not isinstance(name, str) or not name
        for name in checkpointed_partitions
    ):
        raise ValueError("activation census requires named partitions")
    with _preserve_global_rng(device):
        retained_bytes, _, retained_digest = _saved_tensor_bytes(retained)
    receipts: list[ActivationPartitionCensus] = []
    for name in sorted(checkpointed_partitions):
        with _preserve_global_rng(device):
            saved_bytes, elapsed, digest = _saved_tensor_bytes(
                checkpointed_partitions[name]
            )
        if digest != retained_digest:
            raise RuntimeError(
                f"activation partition {name!r} changed the forward value"
            )
        receipts.append(
            ActivationPartitionCensus(
                name,
                retained_bytes,
                saved_bytes,
                max(0, retained_bytes - saved_bytes),
                elapsed,
                digest,
            )
        )
    return tuple(receipts)


def select_activation_dominant_partitions(
    census: tuple[ActivationPartitionCensus, ...],
    *,
    reduction_fraction: float = 0.60,
) -> tuple[str, ...]:
    """Choose the smallest fast set covering a fraction of reducible storage.

    The denominator is the sum of independently measured byte reductions, not
    the retained graph size.  Using retained bytes would make the target
    unreachable whenever no single partition can eliminate most of the whole
    graph and would silently collapse selective execution into checkpointing
    every partition.
    """

    if (
        not census
        or not 0 < reduction_fraction <= 1
        or len({item.partition for item in census}) != len(census)
    ):
        raise ValueError("activation partition selection request is malformed")
    ordered = sorted(
        census,
        key=lambda item: (
            -item.saved_byte_reduction,
            item.elapsed_seconds,
            item.partition,
        ),
    )
    total_reducible = sum(item.saved_byte_reduction for item in census)
    if total_reducible <= 0:
        # A backend retaining no partition-local tensors still needs one real
        # selective candidate for semantic/performance comparison.
        return (ordered[0].partition,)
    target = reduction_fraction * total_reducible
    selected: list[str] = []
    cumulative = 0
    for item in ordered:
        if item.saved_byte_reduction <= 0:
            continue
        selected.append(item.partition)
        cumulative += item.saved_byte_reduction
        if cumulative >= target:
            break
    if not selected:
        return (ordered[0].partition,)
    return tuple(sorted(selected))


def maximum_safe_retain_physical_tokens(
    policy: ActivationExecutionPolicy,
    *,
    alignment: int,
    maximum_physical_tokens: int,
) -> int:
    """Return the largest aligned cohort authorized for full retention.

    Candidate reserve peaks are projected to a declared target shape. Scaling
    that target by the available activation capacity gives a conservative
    shape boundary. The boundary is rounded down to the carrier alignment and
    never exceeds the planner's maximum physical cohort.
    """

    if alignment <= 0 or maximum_physical_tokens <= 0:
        raise ValueError("activation retain-shape request is malformed")
    if policy.resolved == "retain":
        return maximum_physical_tokens
    retain = next(
        (item for item in policy.candidates if item.policy == "retain"),
        None,
    )
    capacity = (
        policy.memory.available_bytes - policy.required_reserve_bytes
    )
    if retain is None or capacity <= 0 or retain.reserve_peak_bytes <= 0:
        return 0
    raw = (
        capacity
        * retain.target_physical_tokens
        // retain.reserve_peak_bytes
    )
    return min(
        maximum_physical_tokens,
        max(0, raw // alignment * alignment),
    )
