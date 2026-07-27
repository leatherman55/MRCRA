"""Deterministic source-text-free fixtures for training-execution acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping

import torch
from torch import Tensor

from .lm_training import PackedBatch


FIXTURE_SCHEMA_VERSION = 1


def _tensor_digest(values: tuple[Tensor, ...]) -> str:
    digest = sha256()
    for value in values:
        local = value.detach().cpu().contiguous()
        digest.update(str(local.dtype).encode("ascii"))
        digest.update(str(tuple(local.shape)).encode("ascii"))
        digest.update(local.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PackedExecutionFixture:
    schema_version: int
    profile: str
    vocabulary_size: int
    batch: PackedBatch
    target_digest: str
    document_length_histogram: tuple[tuple[int, int], ...]
    document_count: int
    boundary_count: int

    def __post_init__(self) -> None:
        if (
            self.schema_version != FIXTURE_SCHEMA_VERSION
            or self.profile not in {"unit_1k", "frequent_8k", "production_32k"}
            or self.vocabulary_size < 2
            or self.batch.input_ids.shape != (1, {
                "unit_1k": 1_024,
                "frequent_8k": 8_192,
                "production_32k": 32_768,
            }[self.profile])
            or len(self.target_digest) != 64
            or self.document_count <= 0
            or self.boundary_count != self.document_count - 1
            or sum(count for _, count in self.document_length_histogram)
            != self.document_count
        ):
            raise ValueError("training execution fixture is malformed")

    def metadata(self) -> dict[str, object]:
        """Return JSON-safe evidence without any recoverable source text."""

        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "vocabulary_size": self.vocabulary_size,
            "shape": list(self.batch.input_ids.shape),
            "target_digest": self.target_digest,
            "document_length_histogram": [
                [length, count]
                for length, count in self.document_length_histogram
            ],
            "document_count": self.document_count,
            "boundary_count": self.boundary_count,
        }


def build_execution_fixture(
    profile: str,
    *,
    vocabulary_size: int,
    seed: int = 20260726,
) -> PackedExecutionFixture:
    """Construct deterministic token authority from counters, never text."""

    lengths = {
        "unit_1k": 1_024,
        "frequent_8k": 8_192,
        "production_32k": 32_768,
    }
    if profile not in lengths or vocabulary_size < 2 or seed < 0:
        raise ValueError("training execution fixture request is invalid")
    context = lengths[profile]
    # A deterministic heavy-tailed-like repeating pattern approximates the
    # short-document pressure of packed FineWeb without embedding corpus text.
    pattern = (37, 61, 89, 131, 197, 293, 431, 641, 953, 1429)
    document_lengths: list[int] = []
    remaining = context
    index = 0
    while remaining:
        proposed = pattern[(index * 7 + seed) % len(pattern)]
        length = min(remaining, proposed)
        document_lengths.append(length)
        remaining -= length
        index += 1

    positions = torch.arange(context, dtype=torch.int64)
    # Integer mixing is deterministic on every backend and does not touch RNG.
    tokens = (
        (
            positions * 6364136223846793005
            + (seed * 1442695040888963407) % ((1 << 63) - 1)
        )
        .remainder(vocabulary_size)
        .unsqueeze(0)
    )
    labels = (
        (
            (positions + 1) * 2862933555777941757
            + ((seed + 17) * 3037000493) % ((1 << 63) - 1)
        )
        .remainder(vocabulary_size)
        .unsqueeze(0)
    )
    byte_lengths = (positions.remainder(4) + 1).unsqueeze(0)
    segment_values = torch.cat(
        tuple(
            torch.full((length,), segment, dtype=torch.int64)
            for segment, length in enumerate(document_lengths)
        )
    )
    target_segments = segment_values.clone()
    if context > 1:
        target_segments[:-1] = segment_values[1:]
    declarations = (
        tuple(
            (
                segment,
                (
                    "fixture://mrcra-training-execution/"
                    f"{profile}/document-{segment:05d}"
                ),
            )
            for segment in range(len(document_lengths))
        ),
    )
    batch = PackedBatch(
        tokens,
        labels,
        byte_lengths,
        segment_values.unsqueeze(0),
        target_segments.unsqueeze(0),
        declarations,
    )
    digest = _tensor_digest((
        batch.labels,
        batch.target_byte_lengths,
        batch.segment_ids,
        batch.target_segment_ids,
        batch.loss_mask,
    ))
    histogram: dict[int, int] = {}
    for length in document_lengths:
        histogram[length] = histogram.get(length, 0) + 1
    return PackedExecutionFixture(
        FIXTURE_SCHEMA_VERSION,
        profile,
        vocabulary_size,
        batch,
        digest,
        tuple(sorted(histogram.items())),
        len(document_lengths),
        len(document_lengths) - 1,
    )


class _FixtureSource:
    def __init__(self, fixture: PackedExecutionFixture) -> None:
        self.fixture = fixture

    def state_dict(self) -> dict[str, object]:
        return {
            "type": "source-text-free-training-execution-fixture",
            **self.fixture.metadata(),
        }


class RepeatingPackedFixtureStream:
    """Minimal checkpointable packed stream for isolated execution kernels."""

    def __init__(self, fixture: PackedExecutionFixture) -> None:
        self.fixture = fixture
        self.source = _FixtureSource(fixture)
        self.batches_emitted = 0

    def next_batch(self, batch_size: int, sequence_length: int) -> PackedBatch:
        expected = self.fixture.batch.input_ids.shape
        if (batch_size, sequence_length) != expected:
            raise ValueError(
                "fixture stream request differs from its immutable shape"
            )
        self.batches_emitted += 1
        return self.fixture.batch

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "fixture": self.fixture.metadata(),
            "batches_emitted": self.batches_emitted,
        }

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        try:
            if (
                int(value["schema_version"]) != 1
                or value["fixture"] != self.fixture.metadata()
                or int(value["batches_emitted"]) < 0
            ):
                raise ValueError("fixture stream identity differs")
            self.batches_emitted = int(value["batches_emitted"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "serialized fixture stream state is malformed"
            ) from error
