"""Production multimodal preparation and causal packet fusion for MRCRA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from .cognitive_types import BoundaryClass, ModalityClass, SourceClass
from .modalities import DomainSpec, EncodedDomain
from .observation import ObservationPacket, register_external_observations
from .provenance import ProvenanceLedger


_DOMAIN_MODALITIES = {
    "sequence": {ModalityClass.TEXT, ModalityClass.SYMBOLIC, ModalityClass.ACTION,
                 ModalityClass.REWARD, ModalityClass.MEMORY, ModalityClass.PREDICTION,
                 ModalityClass.SIMULATION, ModalityClass.GOAL},
    "audio": {ModalityClass.AUDIO},
    "sensor": {ModalityClass.SENSOR},
    "image": {ModalityClass.IMAGE},
    "video": {ModalityClass.VIDEO},
    "field": {ModalityClass.FIELD},
    "graph": {ModalityClass.GRAPH, ModalityClass.MESH},
    "set": {ModalityClass.SET},
}


@dataclass(frozen=True, slots=True)
class PreparedModality:
    packet: ObservationPacket
    modality: ModalityClass
    domain: DomainSpec


class MultimodalPacketAssembler(nn.Module):
    """Project domain-faithful encodings into one authoritative packet format.

    Each adapter remains responsible for the geometry of its domain.  This
    assembler performs only the shared operations: width projection, flattening
    of physical positions, coordinate construction, timing, uncertainty seeds,
    immutable provenance registration, and stable time-ordered fusion.
    """

    def __init__(
        self, input_dimensions: Mapping[ModalityClass | int, int], width: int,
        uncertainty_channels: int,
    ) -> None:
        super().__init__()
        if min(width, uncertainty_channels) <= 0 or not input_dimensions:
            raise ValueError("multimodal assembler dimensions must be positive")
        normalized = {ModalityClass(int(key)): int(value) for key, value in input_dimensions.items()}
        if any(value <= 0 for value in normalized.values()):
            raise ValueError("every modality input dimension must be positive")
        self.width = width
        self.uncertainty_channels = uncertainty_channels
        self.input_dimensions = normalized
        self.projections = nn.ModuleDict({
            str(int(modality)): (
                nn.Identity() if dimension == width else nn.Linear(dimension, width)
            )
            for modality, dimension in normalized.items()
        })

    @staticmethod
    def _flatten(encoded: EncodedDomain) -> tuple[Tensor, Tensor, tuple[int, ...]]:
        if encoded.values.ndim < 3 or encoded.mask.shape != encoded.values.shape[:-1]:
            raise ValueError("encoded domain values require batch, positions, and features")
        batch = encoded.values.shape[0]
        spatial_shape = tuple(encoded.values.shape[1:-1])
        count = 1
        for value in spatial_shape:
            count *= value
        return (
            encoded.values.reshape(batch, count, encoded.values.shape[-1]),
            encoded.mask.reshape(batch, count), spatial_shape,
        )

    @staticmethod
    def _default_coordinates(
        batch: int, spatial_shape: tuple[int, ...], *, device, dtype,
    ) -> Tensor:
        axes = [
            torch.arange(length, device=device, dtype=dtype)
            / max(1, length - 1)
            for length in spatial_shape
        ]
        if not axes:
            return torch.zeros(batch, 1, 1, device=device, dtype=dtype)
        mesh = torch.meshgrid(*axes, indexing="ij")
        coordinates = torch.stack(mesh, -1).reshape(-1, len(axes))
        return coordinates.unsqueeze(0).expand(batch, -1, -1).clone()

    def prepare_external(
        self, encoded: EncodedDomain, modality: ModalityClass, *,
        source_uris: Sequence[str | Mapping[int, str]], ledger: ProvenanceLedger,
        model_authority: str, timestamps: Tensor | None = None,
        coordinates: Tensor | None = None, observed_mask: Tensor | None = None,
        uncertainty_seed: Tensor | None = None, segment_ids: Tensor | None = None,
        boundary_classes: Tensor | None = None, source_class: SourceClass = SourceClass.EXTERNAL,
        source_reliability: float = 1.0, clock_units: str = "seconds",
        coordinate_frame: str | None = None,
    ) -> PreparedModality:
        modality = ModalityClass(modality)
        if modality not in self.input_dimensions:
            raise ValueError(f"modality {modality.name} has no configured projection")
        if modality not in _DOMAIN_MODALITIES[encoded.domain.kind]:
            raise ValueError(
                f"domain {encoded.domain.kind!r} cannot claim modality {modality.name}"
            )
        values, valid, spatial_shape = self._flatten(encoded)
        if values.shape[-1] != self.input_dimensions[modality]:
            raise ValueError("encoded feature width does not match its modality projection")
        values = self.projections[str(int(modality))](values)
        values = values * valid.unsqueeze(-1)
        batch, length = valid.shape
        if len(source_uris) != batch:
            raise ValueError("one source declaration is required per multimodal batch row")
        if coordinates is None:
            coordinates = self._default_coordinates(
                batch, spatial_shape, device=values.device, dtype=values.dtype
            )
        elif coordinates.shape[:-1] == encoded.mask.shape:
            coordinates = coordinates.reshape(batch, length, coordinates.shape[-1])
        if coordinates.ndim != 3 or coordinates.shape[:2] != (batch, length):
            raise ValueError("multimodal coordinates do not align with flattened positions")
        if timestamps is None:
            if encoded.domain.ordered:
                timestamps = (
                    torch.arange(length, device=values.device, dtype=values.dtype)
                    * encoded.domain.sample_interval
                ).expand(batch, -1)
            else:
                timestamps = torch.zeros(batch, length, device=values.device, dtype=values.dtype)
        elif timestamps.shape == encoded.mask.shape:
            timestamps = timestamps.reshape(batch, length)
        if timestamps.shape != (batch, length):
            raise ValueError("multimodal timestamps do not align with flattened positions")
        observed_mask = valid.clone() if observed_mask is None else observed_mask.reshape(batch, length)
        segment_ids = (
            torch.zeros(batch, length, dtype=torch.int64, device=values.device)
            if segment_ids is None else segment_ids.reshape(batch, length)
        )
        segment_ids = segment_ids.masked_fill(~valid, -1)
        if boundary_classes is None:
            boundary_classes = torch.zeros(
                batch, length, dtype=torch.int64, device=values.device
            )
            for row in range(batch):
                if bool(valid[row].any()):
                    boundary_classes[row, 0] = int(BoundaryClass.HARD)
        else:
            boundary_classes = boundary_classes.reshape(batch, length)
        uncertainty_seed = (
            values.new_zeros(batch, length, self.uncertainty_channels)
            if uncertainty_seed is None else uncertainty_seed.reshape(
                batch, length, self.uncertainty_channels
            )
        )
        modality_ids = torch.full(
            (batch, length), int(modality), dtype=torch.int64, device=values.device
        )
        packet = register_external_observations(
            values, valid, observed_mask=observed_mask,
            timestamps=timestamps, coordinates=coordinates,
            sample_intervals=values.new_full(
                (batch,), encoded.domain.sample_interval
            ),
            boundary_classes=boundary_classes, modality_ids=modality_ids,
            uncertainty_seed=uncertainty_seed, segment_ids=segment_ids,
            source_uris=source_uris, ledger=ledger,
            model_authority=model_authority, source_class=source_class,
            source_reliability=source_reliability, clock_units=clock_units,
            coordinate_frame=coordinate_frame or encoded.domain.kind,
        )
        return PreparedModality(packet, modality, encoded.domain)

    @staticmethod
    def fuse(
        prepared: Sequence[PreparedModality], ledger: ProvenanceLedger, *,
        coordinate_frame: str = "multimodal",
    ) -> ObservationPacket:
        """Stable-sort packets by completion time without altering provenance."""

        if not prepared:
            raise ValueError("multimodal fusion requires at least one prepared packet")
        packets = [item.packet for item in prepared]
        batch = packets[0].batch
        width = packets[0].values.shape[-1]
        device, dtype = packets[0].values.device, packets[0].values.dtype
        if any(
            packet.batch != batch or packet.values.shape[-1] != width
            or packet.values.device != device or packet.values.dtype != dtype
            for packet in packets
        ):
            raise ValueError("fused packets must share batch, width, device, and dtype")
        if len({packet.clock_units for packet in packets}) != 1:
            raise ValueError("fused packets must use one declared clock unit")
        coordinate_dims = max(packet.coordinates.shape[-1] for packet in packets)
        row_lengths = [
            sum(int(packet.valid_mask[row].sum()) for packet in packets)
            for row in range(batch)
        ]
        maximum = max(row_lengths)
        values = torch.zeros(batch, maximum, width, device=device, dtype=dtype)
        valid = torch.zeros(batch, maximum, dtype=torch.bool, device=device)
        observed = torch.zeros_like(valid)
        timestamps = torch.zeros(batch, maximum, device=device, dtype=dtype)
        coordinates = torch.zeros(
            batch, maximum, coordinate_dims, device=device, dtype=dtype
        )
        boundaries = torch.zeros(batch, maximum, dtype=torch.int64, device=device)
        modality_ids = torch.zeros(batch, maximum, dtype=torch.int64, device=device)
        source_ids = torch.full((batch, maximum), -1, dtype=torch.int64, device=device)
        uncertainty_channels = packets[0].uncertainty_seed.shape[-1]
        if any(packet.uncertainty_seed.shape[-1] != uncertainty_channels for packet in packets):
            raise ValueError("fused packets must share uncertainty width")
        uncertainty = torch.zeros(
            batch, maximum, uncertainty_channels, device=device, dtype=dtype
        )
        segments = torch.full((batch, maximum), -1, dtype=torch.int64, device=device)
        sample_intervals = torch.stack(tuple(packet.sample_intervals for packet in packets)).amin(0)
        for row in range(batch):
            entries = []
            segment_offset = 0
            order = 0
            for packet in packets:
                count = int(packet.valid_mask[row].sum())
                for position in range(count):
                    entries.append((
                        float(packet.timestamps[row, position]), order, packet,
                        position, segment_offset + int(packet.segment_ids[row, position]),
                    ))
                    order += 1
                active_segments = packet.segment_ids[row, packet.valid_mask[row]]
                segment_offset += (
                    int(active_segments.max()) + 1 if active_segments.numel() else 0
                )
            entries.sort(key=lambda item: (item[0], item[1]))
            for output_index, (_, _, packet, position, segment) in enumerate(entries):
                values[row, output_index] = packet.values[row, position]
                valid[row, output_index] = True
                observed[row, output_index] = packet.observed_mask[row, position]
                timestamps[row, output_index] = packet.timestamps[row, position]
                coordinate_count = packet.coordinates.shape[-1]
                coordinates[row, output_index, :coordinate_count] = packet.coordinates[row, position]
                boundary = int(packet.boundary_classes[row, position])
                # Independently prepared modalities each declare their first
                # sample HARD.  When several begin at the same physical time,
                # only the first opens the episode; the others are synchronized
                # evidence and must not erase the just-ingested modality state.
                if (
                    output_index > 0
                    and boundary == int(BoundaryClass.HARD)
                    and timestamps[row, output_index] == timestamps[row, output_index - 1]
                ):
                    boundary = int(BoundaryClass.SOFT)
                boundaries[row, output_index] = boundary
                modality_ids[row, output_index] = packet.modality_ids[row, position]
                source_ids[row, output_index] = packet.source_record_ids[row, position]
                uncertainty[row, output_index] = packet.uncertainty_seed[row, position]
                segments[row, output_index] = segment
        result = ObservationPacket(
            values, valid, observed, timestamps, coordinates, sample_intervals,
            boundaries, modality_ids, source_ids, uncertainty, segments,
            packets[0].clock_units, coordinate_frame,
        )
        result.assert_ledger_consistent(ledger)
        return result
