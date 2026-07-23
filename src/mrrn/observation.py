"""Validated multimodal observation packets and authoritative source creation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping, Sequence, TypeVar

import torch
from torch import Tensor

from .cognitive_types import (
    BoundaryClass, ModalityClass, SourceClass, SupportInterval,
)
from .provenance import ProvenanceLedger
from .runtime_validation import runtime_validation_enabled


T = TypeVar("T")


def _map_tensors(instance: T, method: str, *args, **kwargs) -> T:
    values = {}
    for field in fields(instance):
        value = getattr(instance, field.name)
        values[field.name] = getattr(value, method)(*args, **kwargs) if isinstance(value, Tensor) else value
    return type(instance)(**values)


@dataclass(frozen=True, slots=True)
class ObservationPacket:
    """The only legal ingress contract from a modality adapter into MRCRA.

    Timestamps are physical or logical completion times.  ``valid_mask`` marks
    positions that exist; ``observed_mask`` is a strict subset indicating direct
    measurement.  Unobserved-but-valid values must carry uncertainty and are not
    allowed to masquerade as external observations downstream.
    """

    values: Tensor
    valid_mask: Tensor
    observed_mask: Tensor
    timestamps: Tensor
    coordinates: Tensor
    sample_intervals: Tensor
    boundary_classes: Tensor
    modality_ids: Tensor
    source_record_ids: Tensor
    uncertainty_seed: Tensor
    segment_ids: Tensor
    clock_units: str = "logical_step"
    coordinate_frame: str = "sequence"

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.values.ndim != 3:
            raise ValueError("observation values must have shape (batch,time,width)")
        batch, length, _ = self.values.shape
        base = (batch, length)
        for name in ("valid_mask", "observed_mask"):
            value = getattr(self, name)
            if value.shape != base or value.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean with shape (batch,time)")
        if bool((self.observed_mask & ~self.valid_mask).any()):
            raise ValueError("observed positions must also be valid")
        if self.timestamps.shape != base or not self.timestamps.is_floating_point():
            raise ValueError("timestamps must be floating point with shape (batch,time)")
        if self.coordinates.ndim != 3 or self.coordinates.shape[:2] != base:
            raise ValueError("coordinates must have shape (batch,time,coordinate_dims)")
        if self.coordinates.shape[-1] == 0:
            raise ValueError("at least one coordinate dimension is required")
        if self.sample_intervals.shape != (batch,) or not self.sample_intervals.is_floating_point():
            raise ValueError("sample_intervals must be floating point with shape (batch,)")
        if bool((self.sample_intervals <= 0).any()):
            raise ValueError("sample intervals must be positive")
        for name in ("boundary_classes", "modality_ids", "source_record_ids", "segment_ids"):
            value = getattr(self, name)
            if value.shape != base or value.dtype != torch.int64:
                raise ValueError(f"{name} must be int64 with shape (batch,time)")
        if self.uncertainty_seed.ndim != 3 or self.uncertainty_seed.shape[:2] != base:
            raise ValueError("uncertainty_seed must have shape (batch,time,channels)")
        if self.uncertainty_seed.shape[-1] == 0 or bool((self.uncertainty_seed < 0).any()):
            raise ValueError("uncertainty seed requires nonnegative channels")
        if not self.clock_units or not self.coordinate_frame:
            raise ValueError("clock units and coordinate frame must be declared")
        valid_boundaries = (self.boundary_classes >= 0) & (self.boundary_classes < len(BoundaryClass))
        valid_modalities = (self.modality_ids >= 0) & (self.modality_ids < len(ModalityClass))
        if bool((self.valid_mask & ~valid_boundaries).any()):
            raise ValueError("valid positions contain an unknown boundary class")
        if bool((self.valid_mask & ~valid_modalities).any()):
            raise ValueError("valid positions contain an unknown modality class")
        if bool((self.valid_mask & (self.source_record_ids < 0)).any()):
            raise ValueError("every valid position requires an authoritative provenance reference")
        if bool((~self.valid_mask & (self.source_record_ids >= 0)).any()):
            raise ValueError("padding positions cannot claim provenance")
        if bool((self.valid_mask & (self.segment_ids < 0)).any()):
            raise ValueError("every valid position requires a nonnegative segment ID")
        if bool((~self.valid_mask & (self.segment_ids >= 0)).any()):
            raise ValueError("padding positions cannot belong to a segment")
        for batch_index in range(batch):
            valid = self.valid_mask[batch_index]
            times = self.timestamps[batch_index, valid]
            if times.numel() > 1 and bool((times[1:] < times[:-1]).any()):
                raise ValueError("timestamps must be monotonic within a packet")
            valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
            if valid_indices.numel() and not torch.equal(
                valid_indices, torch.arange(valid_indices.numel(), device=valid_indices.device)
            ):
                raise ValueError("valid packet positions must be a prefix; split sparse streams into packets")
        # Neural ingress must not accumulate arbitrary data in padding rows.
        if bool((self.values[~self.valid_mask] != 0).any()):
            raise ValueError("padding values must be exactly zero")

    @property
    def batch(self) -> int:
        return self.values.shape[0]

    @property
    def length(self) -> int:
        return self.values.shape[1]

    @property
    def hard_reset_mask(self) -> Tensor:
        return self.valid_mask & (self.boundary_classes == int(BoundaryClass.HARD))

    @property
    def segment_reset_mask(self) -> Tensor:
        return self.valid_mask & (self.boundary_classes == int(BoundaryClass.SEGMENT))

    @property
    def soft_boundary_mask(self) -> Tensor:
        return self.valid_mask & (self.boundary_classes == int(BoundaryClass.SOFT))

    def assert_ledger_consistent(
        self, ledger: ProvenanceLedger, *, allow_internal: bool = False,
    ) -> None:
        for record_id in torch.unique(self.source_record_ids[self.valid_mask]).tolist():
            record = ledger.get(int(record_id))
            if not allow_internal and record.source_class not in (
                SourceClass.EXTERNAL, SourceClass.BODILY,
            ):
                raise ValueError("observation-only packet points to an internally generated source")
        valid_modalities = self.modality_ids[self.valid_mask]
        valid_records = self.source_record_ids[self.valid_mask]
        valid_observed = self.observed_mask[self.valid_mask]
        for modality, record_id, observed in zip(
            valid_modalities.tolist(), valid_records.tolist(), valid_observed.tolist(), strict=True
        ):
            record = ledger.get(int(record_id))
            if int(record.modality) != int(modality):
                raise ValueError("packet modality disagrees with authoritative provenance")
            if observed and record.source_class not in (SourceClass.EXTERNAL, SourceClass.BODILY):
                raise ValueError("internally generated input cannot be marked directly observed")

    def detach(self) -> "ObservationPacket":
        return _map_tensors(self, "detach")

    def to(self, *args, **kwargs) -> "ObservationPacket":
        return _map_tensors(self, "to", *args, **kwargs)


def register_external_observations(
    values: Tensor,
    valid_mask: Tensor,
    *,
    observed_mask: Tensor | None,
    timestamps: Tensor,
    coordinates: Tensor,
    sample_intervals: Tensor,
    boundary_classes: Tensor,
    modality_ids: Tensor,
    uncertainty_seed: Tensor,
    segment_ids: Tensor,
    source_uris: Sequence[str | Mapping[int, str]],
    ledger: ProvenanceLedger,
    model_authority: str,
    source_class: SourceClass = SourceClass.EXTERNAL,
    source_reliability: float = 1.0,
    clock_units: str = "logical_step",
    coordinate_frame: str = "sequence",
) -> ObservationPacket:
    """Atomically register packet segments and return a validated tensor packet.

    One immutable source record is created for every ``(batch, segment)`` pair,
    avoiding a provenance-ledger entry per token while retaining exact segment
    support and references at every position.
    """

    if values.ndim != 3:
        raise ValueError("values must have shape (batch,time,width)")
    if source_class not in (SourceClass.EXTERNAL, SourceClass.BODILY):
        raise ValueError("observation registration requires an external or bodily source class")
    if not model_authority or not 0 <= source_reliability <= 1:
        raise ValueError("model authority and source reliability are invalid")
    batch, length = values.shape[:2]
    if len(source_uris) != batch:
        raise ValueError("one source URI declaration is required per batch item")
    if valid_mask.shape != (batch, length) or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean with shape (batch,time)")
    if timestamps.shape != (batch, length) or modality_ids.shape != (batch, length):
        raise ValueError("timestamps and modality_ids must match values")
    observed_mask = valid_mask.clone() if observed_mask is None else observed_mask
    source_record_ids = torch.full(
        (batch, length), -1, dtype=torch.int64, device=values.device
    )
    # Validate masks and shapes before mutating the append-only ledger.
    preview = ObservationPacket(
        values, valid_mask, observed_mask, timestamps, coordinates,
        sample_intervals, boundary_classes, modality_ids, source_record_ids.masked_fill(valid_mask, 0),
        uncertainty_seed, segment_ids, clock_units, coordinate_frame,
    )
    pending: list[tuple[int, Tensor, int, SupportInterval, ModalityClass]] = []
    for batch_index in range(batch):
        segments = torch.unique(segment_ids[batch_index, valid_mask[batch_index]], sorted=True)
        declaration = source_uris[batch_index]
        if isinstance(declaration, str):
            if not declaration:
                raise ValueError("source URIs must be nonempty")
        elif not isinstance(declaration, Mapping):
            raise ValueError("source declarations must be a URI or segment-to-URI mapping")
        for segment in segments.tolist():
            if isinstance(declaration, Mapping):
                uri = declaration.get(int(segment))
                if not isinstance(uri, str) or not uri:
                    raise ValueError("every valid segment requires a nonempty source URI")
            selection = valid_mask[batch_index] & (segment_ids[batch_index] == int(segment))
            positions = torch.nonzero(selection, as_tuple=False).flatten()
            selected_modalities = torch.unique(modality_ids[batch_index, selection])
            if selected_modalities.numel() != 1:
                raise ValueError("one packet segment cannot mix authoritative modalities")
            start = float(timestamps[batch_index, positions[0]].item())
            end = float(timestamps[batch_index, positions[-1]].item())
            completion = end
            pending.append((
                batch_index, positions, int(segment),
                SupportInterval(start, end, completion),
                ModalityClass(int(selected_modalities.item())),
            ))
    for batch_index, positions, segment, support, modality in pending:
        declaration = source_uris[batch_index]
        source_uri = declaration if isinstance(declaration, str) else declaration[segment]
        record_id = ledger.append(
            source_class=source_class,
            source_uri_or_episode=f"{source_uri}#segment={segment}",
            support=support,
            modality=modality,
            operator="observation_adapter",
            scenario_id=0,
            model_authority=model_authority,
            source_reliability=source_reliability,
            calibration_context=clock_units,
        )
        source_record_ids[batch_index, positions] = record_id
    packet = ObservationPacket(
        preview.values, preview.valid_mask, preview.observed_mask, preview.timestamps,
        preview.coordinates, preview.sample_intervals, preview.boundary_classes,
        preview.modality_ids, source_record_ids, preview.uncertainty_seed,
        preview.segment_ids, preview.clock_units, preview.coordinate_frame,
    )
    packet.assert_ledger_consistent(ledger)
    return packet


def register_internal_inputs(
    values: Tensor,
    valid_mask: Tensor,
    *,
    parent_record_ids: Tensor,
    timestamps: Tensor,
    coordinates: Tensor,
    sample_intervals: Tensor,
    boundary_classes: Tensor,
    modality_ids: Tensor,
    uncertainty_seed: Tensor,
    segment_ids: Tensor,
    ledger: ProvenanceLedger,
    source_class: SourceClass,
    operator: str,
    scenario_ids: Tensor,
    model_authority: str,
    clock_units: str = "logical_step",
    coordinate_frame: str = "sequence",
) -> ObservationPacket:
    """Register retrieved/predicted/simulated/abstracted/goal-derived inputs."""

    if source_class in (SourceClass.EXTERNAL, SourceClass.BODILY):
        raise ValueError("use register_external_observations for observed sources")
    if values.ndim != 3:
        raise ValueError("internal values must be (batch,time,width)")
    batch, length = values.shape[:2]
    base = (batch, length)
    if parent_record_ids.ndim != 3 or parent_record_ids.shape[:2] != base:
        raise ValueError("internal inputs require (batch,time,parents) provenance pointers")
    if scenario_ids.shape != base or scenario_ids.dtype != torch.int64:
        raise ValueError("internal scenario IDs must be int64 with packet shape")
    source_ids = torch.full(base, -1, dtype=torch.int64, device=values.device)
    # Validate the complete tensor contract before mutating the ledger.
    preview_ids = source_ids.masked_fill(valid_mask, 0)
    preview = ObservationPacket(
        values, valid_mask, torch.zeros_like(valid_mask), timestamps, coordinates,
        sample_intervals, boundary_classes, modality_ids, preview_ids,
        uncertainty_seed, segment_ids, clock_units, coordinate_frame,
    )
    pending = []
    for batch_index, time_index in torch.nonzero(valid_mask, as_tuple=False).tolist():
        parents = parent_record_ids[batch_index, time_index]
        parents = parents[parents >= 0].tolist()
        if not parents:
            raise ValueError("every internally generated input requires parent provenance")
        timestamp = float(timestamps[batch_index, time_index].item())
        pending.append((
            batch_index, time_index, parents,
            SupportInterval(timestamp, timestamp, timestamp),
            ModalityClass(int(modality_ids[batch_index, time_index].item())),
            int(scenario_ids[batch_index, time_index].item()),
        ))
    for batch_index, time_index, parents, support, modality, scenario in pending:
        source_ids[batch_index, time_index] = ledger.derive(
            parents, source_class=source_class, operator=operator, support=support,
            modality=modality, scenario_id=scenario, model_authority=model_authority,
        )
    packet = ObservationPacket(
        preview.values, preview.valid_mask, preview.observed_mask, preview.timestamps,
        preview.coordinates, preview.sample_intervals, preview.boundary_classes,
        preview.modality_ids, source_ids, preview.uncertainty_seed,
        preview.segment_ids, preview.clock_units, preview.coordinate_frame,
    )
    packet.assert_ledger_consistent(ledger, allow_internal=True)
    return packet
