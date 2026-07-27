"""Streaming next-token training authority for MRRN language models.

The optimized objective is ordinary causal cross entropy plus the bounded MRRN
spectral-activation regularizer.  Functional-surprise/RASL is intentionally not
used here: FineWeb supplies text, not an external downstream reward, and using
task loss as reward is explicitly rejected by the RASL contract.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from math import ceil, exp, isfinite, log
import os
from pathlib import Path
import platform
from queue import Full, Queue
import tempfile
from threading import Lock, Thread
from time import monotonic, perf_counter, sleep
from typing import Any, Iterable, Iterator, Protocol, Sequence

try:
    import resource
except ImportError:  # pragma: no cover - exercised by the Windows import smoke test
    resource = None

import torch
from torch import Tensor
from torch.nn import functional as F

from .cognitive_types import BoundaryClass
from .language import LanguageModelOutput, MRRNLanguageModel
from .mixer import ResonantSpectralGLU
from .objectives import spectral_activation_regularization
from .optimization import (
    GradientReport,
    OptimizerPolicy,
    build_adamw,
    build_scheduler,
    clip_and_report_gradients,
)
from .trackio_dashboard import (
    SPECTRAL_ARTIFACT_NAME,
    SPECTRAL_ARTIFACT_TYPE,
    SPECTRAL_DATA_FILENAME,
    attach_training_evidence,
    prepare_runtime_frontend,
    write_evidence_atomically,
)


CHECKPOINT_FORMAT_VERSION = 2


@dataclass(frozen=True, slots=True)
class TokenizedDocument:
    token_ids: tuple[int, ...]
    byte_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.token_ids or len(self.token_ids) != len(self.byte_lengths):
            raise ValueError("tokenized documents require aligned nonempty token and byte lengths")
        if min(self.token_ids) < 0 or min(self.byte_lengths) < 0:
            raise ValueError("token ids and byte lengths cannot be negative")


class TextTokenizer(Protocol):
    vocabulary_size: int
    eos_token_id: int

    def encode_document(self, text: str) -> TokenizedDocument: ...
    def encode_prompt(self, text: str) -> list[int]: ...
    def decode(self, token_ids: Sequence[int]) -> str: ...
    def identity(self) -> dict[str, Any]: ...


class HuggingFaceTextTokenizer:
    """Fast tokenizer with token-to-original-UTF-8-byte accounting."""

    def __init__(self, name: str = "gpt2", *, revision: str = "main") -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError("install the 'train' extra to use Hugging Face tokenization") from error
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, use_fast=True)
        if not tokenizer.is_fast or tokenizer.eos_token_id is None:
            raise ValueError("next-token training requires a fast tokenizer with an EOS token")
        self.tokenizer = tokenizer
        self.name, self.revision = name, revision
        self.vocabulary_size = len(tokenizer)
        self.eos_token_id = int(tokenizer.eos_token_id)

    def encode_document(self, text: str) -> TokenizedDocument:
        if not isinstance(text, str):
            raise TypeError("document text must be a string")
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_offsets_mapping=True,
            verbose=False,
        )
        token_ids = [int(value) for value in encoded["input_ids"]]
        offsets = encoded["offset_mapping"]
        if len(token_ids) != len(offsets):
            raise ValueError("tokenizer returned misaligned token offsets")
        byte_lengths = []
        for start, end in offsets:
            if not 0 <= start <= end <= len(text):
                raise ValueError("tokenizer returned an invalid source-text offset")
            byte_lengths.append(len(text[start:end].encode("utf-8")))
        token_ids.append(self.eos_token_id)
        byte_lengths.append(0)
        return TokenizedDocument(tuple(token_ids), tuple(byte_lengths))

    def encode_prompt(self, text: str) -> list[int]:
        result = self.tokenizer.encode(text, add_special_tokens=False)
        return [int(value) for value in result] or [self.eos_token_id]

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def identity(self) -> dict[str, Any]:
        return {
            "kind": "huggingface",
            "name": self.name,
            "revision": self.revision,
            "vocabulary_size": self.vocabulary_size,
            "eos_token_id": self.eos_token_id,
        }


class ByteTextTokenizer:
    """Dependency-free UTF-8 byte tokenizer used by deterministic smoke tests."""

    vocabulary_size = 257
    eos_token_id = 256

    def encode_document(self, text: str) -> TokenizedDocument:
        values = list(text.encode("utf-8"))
        return TokenizedDocument(tuple(values + [self.eos_token_id]), tuple([1] * len(values) + [0]))

    def encode_prompt(self, text: str) -> list[int]:
        return list(text.encode("utf-8")) or [self.eos_token_id]

    def decode(self, token_ids: Sequence[int]) -> str:
        return bytes(value for value in token_ids if 0 <= value < 256).decode("utf-8", errors="replace")

    def identity(self) -> dict[str, Any]:
        return {
            "kind": "utf8-bytes",
            "vocabulary_size": self.vocabulary_size,
            "eos_token_id": self.eos_token_id,
        }


@dataclass(frozen=True, slots=True)
class TextDocument:
    identifier: str
    text: str


class StatefulTextSource(Protocol):
    def __iter__(self) -> Iterator[TextDocument]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


def _is_evaluation_document(identifier: str, fraction_permyriad: int) -> bool:
    value = int.from_bytes(sha256(identifier.encode("utf-8")).digest()[:8], "big")
    return value % 10_000 < fraction_permyriad


def _heldout_role(identifier: str, fraction_permyriad: int) -> str:
    """Return train/progress/eval using one stable, pairwise-disjoint hash split."""

    digest = sha256(identifier.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") % 10_000
    if value >= fraction_permyriad:
        return "train"
    # Use independent digest bytes instead of parity of the inclusion bucket so
    # changing the retained fraction does not systematically bias either role.
    return "progress" if int.from_bytes(digest[8:16], "big") & 1 == 0 else "eval"


class FineWebTextSource:
    """Deterministic stateful stream over the original English FineWeb dataset."""

    def __init__(
        self,
        *,
        dataset_id: str = "HuggingFaceFW/fineweb",
        dataset_config: str = "sample-10BT",
        split: str = "train",
        revision: str = "main",
        partition: str = "train",
        evaluation_fraction_permyriad: int = 100,
        shuffle_seed: int = 20260721,
        shuffle_buffer: int = 10_000,
    ) -> None:
        if partition not in {"train", "progress", "eval"}:
            raise ValueError("FineWeb partition must be train, progress, or eval")
        if not 1 <= evaluation_fraction_permyriad < 10_000:
            raise ValueError("evaluation fraction must lie in 1..9999 permyriad")
        if shuffle_buffer <= 0:
            raise ValueError("shuffle buffer must be positive")
        self.dataset_id, self.dataset_config, self.split, self.revision = (
            dataset_id, dataset_config, split, revision
        )
        self.partition = partition
        self.evaluation_fraction_permyriad = evaluation_fraction_permyriad
        self.shuffle_seed, self.shuffle_buffer = shuffle_seed, shuffle_buffer
        self.raw_rows_scanned = 0
        self.documents_yielded = 0
        self._started = False

    def _dataset(self):
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError("install the 'train' extra to stream FineWeb") from error
        dataset = load_dataset(
            self.dataset_id,
            name=self.dataset_config,
            split=self.split,
            revision=self.revision,
            streaming=True,
        )
        if self.partition == "train":
            dataset = dataset.shuffle(seed=self.shuffle_seed, buffer_size=self.shuffle_buffer)
        if self.raw_rows_scanned:
            dataset = dataset.skip(self.raw_rows_scanned)
        return dataset

    def __iter__(self) -> Iterator[TextDocument]:
        if self._started:
            raise RuntimeError("a FineWebTextSource is a single active stateful stream")
        self._started = True
        for row in self._dataset():
            self.raw_rows_scanned += 1
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError("FineWeb row schema must contain string field 'text'")
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("FineWeb row schema must contain nonempty string field 'id'")
            role = _heldout_role(identifier, self.evaluation_fraction_permyriad)
            if self.partition != role or not row["text"]:
                continue
            self.documents_yielded += 1
            yield TextDocument(identifier, row["text"])

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "fineweb",
            "dataset_id": self.dataset_id,
            "dataset_config": self.dataset_config,
            "split": self.split,
            "revision": self.revision,
            "partition": self.partition,
            "evaluation_fraction_permyriad": self.evaluation_fraction_permyriad,
            "shuffle_seed": self.shuffle_seed,
            "shuffle_buffer": self.shuffle_buffer,
            "raw_rows_scanned": self.raw_rows_scanned,
            "documents_yielded": self.documents_yielded,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._started:
            raise RuntimeError("cannot restore a FineWeb stream after iteration starts")
        current = self.state_dict()
        for key in (
            "kind", "dataset_id", "dataset_config", "split", "revision", "partition",
            "evaluation_fraction_permyriad", "shuffle_seed", "shuffle_buffer",
        ):
            if state.get(key) != current[key]:
                raise ValueError(f"FineWeb checkpoint mismatch for {key}")
        raw_rows, documents = state.get("raw_rows_scanned"), state.get("documents_yielded")
        if not isinstance(raw_rows, int) or not isinstance(documents, int) or min(raw_rows, documents) < 0:
            raise ValueError("FineWeb checkpoint counters are invalid")
        self.raw_rows_scanned, self.documents_yielded = raw_rows, documents


class SequenceTextSource:
    """Restartable local source with the same checkpoint contract as FineWeb."""

    def __init__(self, documents: Sequence[str], *, repeat: bool = True) -> None:
        if not documents or any(not isinstance(value, str) or not value for value in documents):
            raise ValueError("local text source requires nonempty string documents")
        self.documents = tuple(documents)
        self.repeat = repeat
        self.raw_rows_scanned = 0
        self.documents_yielded = 0
        self._started = False

    def __iter__(self) -> Iterator[TextDocument]:
        if self._started:
            raise RuntimeError("a SequenceTextSource is a single active stateful stream")
        self._started = True
        while self.repeat or self.raw_rows_scanned < len(self.documents):
            index = self.raw_rows_scanned % len(self.documents)
            self.raw_rows_scanned += 1
            self.documents_yielded += 1
            yield TextDocument(f"local-{self.raw_rows_scanned - 1}", self.documents[index])

    def state_dict(self) -> dict[str, Any]:
        digest = sha256("\0".join(self.documents).encode("utf-8")).hexdigest()
        return {
            "kind": "sequence", "digest": digest, "repeat": self.repeat,
            "raw_rows_scanned": self.raw_rows_scanned,
            "documents_yielded": self.documents_yielded,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._started:
            raise RuntimeError("cannot restore a local stream after iteration starts")
        current = self.state_dict()
        if any(state.get(key) != current[key] for key in ("kind", "digest", "repeat")):
            raise ValueError("local text checkpoint does not match the source")
        self.raw_rows_scanned = int(state["raw_rows_scanned"])
        self.documents_yielded = int(state["documents_yielded"])


@dataclass(frozen=True, slots=True)
class PackedBatch:
    input_ids: Tensor
    labels: Tensor
    target_byte_lengths: Tensor
    segment_ids: Tensor | None = None
    target_segment_ids: Tensor | None = None
    source_uris_by_segment: tuple[tuple[tuple[int, str], ...], ...] | None = None
    continuity_keys: tuple[str | None, ...] | None = None

    def __post_init__(self) -> None:
        if self.input_ids.shape != self.labels.shape or self.labels.shape != self.target_byte_lengths.shape:
            raise ValueError("packed language batch tensors must have identical shapes")
        if self.input_ids.ndim != 2 or self.input_ids.dtype != torch.long or self.labels.dtype != torch.long:
            raise ValueError("packed token tensors must be int64 and shaped (batch,time)")
        if self.target_byte_lengths.dtype != torch.long or bool((self.target_byte_lengths < 0).any()):
            raise ValueError("target byte lengths must be nonnegative int64")
        if self.segment_ids is None:
            object.__setattr__(self, "segment_ids", torch.zeros_like(self.input_ids))
        if self.target_segment_ids is None:
            object.__setattr__(self, "target_segment_ids", torch.zeros_like(self.input_ids))
        for name in ("segment_ids", "target_segment_ids"):
            value = getattr(self, name)
            if value.shape != self.input_ids.shape or value.dtype != torch.long:
                raise ValueError(f"packed {name} must be int64 and match token shape")
        if self.source_uris_by_segment is None:
            object.__setattr__(self, "source_uris_by_segment", tuple(
                tuple((int(segment), f"packed://row/{row}/segment/{int(segment)}")
                      for segment in torch.unique(self.segment_ids[row]).tolist())
                for row in range(self.input_ids.shape[0])
            ))
        if len(self.source_uris_by_segment) != self.input_ids.shape[0]:
            raise ValueError("packed source declarations must have one row per batch item")
        for row, declarations in enumerate(self.source_uris_by_segment):
            mapping = dict(declarations)
            if len(mapping) != len(declarations) or any(
                not isinstance(segment, int) or segment < 0
                or not isinstance(uri, str) or not uri
                for segment, uri in declarations
            ):
                raise ValueError("packed source declarations must contain unique segment/URI pairs")
            if not set(torch.unique(self.segment_ids[row]).tolist()).issubset(mapping):
                raise ValueError("every packed input segment requires its original source URI")
        if self.continuity_keys is None:
            object.__setattr__(self, "continuity_keys", tuple(
                None for _ in range(self.input_ids.shape[0])
            ))
        if len(self.continuity_keys) != self.input_ids.shape[0] or any(
            key is not None and (not isinstance(key, str) or not key)
            for key in self.continuity_keys
        ):
            raise ValueError("packed continuity keys must be nonempty strings or None")

    @property
    def loss_mask(self) -> Tensor:
        """Exclude synthetic transitions between unrelated packed documents."""

        return self.segment_ids == self.target_segment_ids

    @property
    def boundary_classes(self) -> Tensor:
        boundary = torch.zeros_like(self.segment_ids)
        if boundary.shape[1]:
            boundary[:, 0] = int(BoundaryClass.HARD)
        if boundary.shape[1] > 1:
            changed = self.segment_ids[:, 1:] != self.segment_ids[:, :-1]
            boundary[:, 1:] = torch.where(
                changed,
                torch.full_like(boundary[:, 1:], int(BoundaryClass.SEGMENT)),
                boundary[:, 1:],
            )
        return boundary

    @property
    def token_count(self) -> int:
        return self.labels.numel()

    @property
    def byte_count(self) -> int:
        return int(self.target_byte_lengths.sum())

    def pin_memory(self) -> "PackedBatch":
        """Stage a CPU batch in page-locked memory for asynchronous CUDA copies."""

        return PackedBatch(
            self.input_ids.pin_memory(),
            self.labels.pin_memory(),
            self.target_byte_lengths.pin_memory(),
            self.segment_ids.pin_memory(),
            self.target_segment_ids.pin_memory(),
            self.source_uris_by_segment,
            self.continuity_keys,
        )

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "PackedBatch":
        return PackedBatch(
            self.input_ids.to(device, non_blocking=non_blocking),
            self.labels.to(device, non_blocking=non_blocking),
            self.target_byte_lengths.to(device, non_blocking=non_blocking),
            self.segment_ids.to(device, non_blocking=non_blocking),
            self.target_segment_ids.to(device, non_blocking=non_blocking),
            self.source_uris_by_segment,
            self.continuity_keys,
        )

    @property
    def external_source_uris(self) -> tuple[dict[int, str], ...]:
        return tuple(dict(row) for row in self.source_uris_by_segment)


class PackedTokenStream:
    """Lossless document packing with one-token overlap between sequence blocks."""

    def __init__(self, source: StatefulTextSource, tokenizer: TextTokenizer) -> None:
        self.source, self.tokenizer = source, tokenizer
        self.token_buffer: list[int] = []
        self.byte_buffer: list[int] = []
        self.segment_buffer: list[int] = []
        self.source_buffer: list[str] = []
        self.next_segment_id = 0
        self.documents_packed = 0
        self._iterator: Iterator[TextDocument] | None = None

    def _fill(self, required: int) -> None:
        if required <= 0:
            raise ValueError("packed stream fill size must be positive")
        if self._iterator is None:
            self._iterator = iter(self.source)
        while len(self.token_buffer) < required:
            document = next(self._iterator)
            encoded = self.tokenizer.encode_document(document.text)
            if max(encoded.token_ids) >= self.tokenizer.vocabulary_size:
                raise ValueError("tokenizer emitted an id outside its declared vocabulary")
            self.token_buffer.extend(encoded.token_ids)
            self.byte_buffer.extend(encoded.byte_lengths)
            self.segment_buffer.extend([self.next_segment_id] * len(encoded.token_ids))
            self.source_buffer.extend(
                [f"dataset://document/{document.identifier}"] * len(encoded.token_ids)
            )
            self.next_segment_id += 1
            self.documents_packed += 1

    def next_batch(self, batch_size: int, sequence_length: int) -> PackedBatch:
        if min(batch_size, sequence_length) <= 0:
            raise ValueError("batch size and sequence length must be positive")
        token_rows, byte_rows, segment_rows, source_rows = [], [], [], []
        for _ in range(batch_size):
            self._fill(sequence_length + 1)
            token_rows.append(self.token_buffer[: sequence_length + 1])
            byte_rows.append(self.byte_buffer[: sequence_length + 1])
            segment_rows.append(self.segment_buffer[: sequence_length + 1])
            source_rows.append(self.source_buffer[: sequence_length + 1])
            # Retain the final label as the next block's first input.  No
            # transition is silently dropped at a packing boundary.
            del self.token_buffer[:sequence_length]
            del self.byte_buffer[:sequence_length]
            del self.segment_buffer[:sequence_length]
            del self.source_buffer[:sequence_length]
        tokens = torch.tensor(token_rows, dtype=torch.long)
        byte_lengths = torch.tensor(byte_rows, dtype=torch.long)
        segments = torch.tensor(segment_rows, dtype=torch.long)
        declarations = tuple(
            tuple(dict(zip(segments[row, :-1].tolist(), source_rows[row][:-1])).items())
            for row in range(batch_size)
        )
        return PackedBatch(
            tokens[:, :-1], tokens[:, 1:], byte_lengths[:, 1:],
            segments[:, :-1], segments[:, 1:], declarations,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.state_dict(),
            "tokenizer": self.tokenizer.identity(),
            "token_buffer": list(self.token_buffer),
            "byte_buffer": list(self.byte_buffer),
            "segment_buffer": list(self.segment_buffer),
            "source_buffer": list(self.source_buffer),
            "next_segment_id": self.next_segment_id,
            "documents_packed": self.documents_packed,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._iterator is not None:
            raise RuntimeError("cannot restore a token packer after iteration starts")
        if state.get("tokenizer") != self.tokenizer.identity():
            raise ValueError("checkpoint tokenizer identity does not match")
        token_buffer, byte_buffer = state.get("token_buffer"), state.get("byte_buffer")
        segment_buffer = state.get("segment_buffer")
        source_buffer = state.get("source_buffer")
        if segment_buffer is None and isinstance(token_buffer, list):
            # Legacy sequence-only checkpoints had no boundary metadata.  They
            # can resume the legacy trainer, while new checkpoints are exact.
            segment_buffer = [0] * len(token_buffer)
        if source_buffer is None and isinstance(token_buffer, list):
            source_buffer = ["dataset://legacy-packed-stream"] * len(token_buffer)
        if (
            not isinstance(token_buffer, list) or not isinstance(byte_buffer, list)
            or not isinstance(segment_buffer, list)
            or not isinstance(source_buffer, list)
            or len(token_buffer) != len(byte_buffer) or len(token_buffer) != len(segment_buffer)
            or len(token_buffer) != len(source_buffer)
            or any(not isinstance(value, int) or value < 0 for value in token_buffer + byte_buffer + segment_buffer)
            or any(not isinstance(value, str) or not value for value in source_buffer)
        ):
            raise ValueError("checkpoint token packer buffer is invalid")
        self.source.load_state_dict(state["source"])
        self.token_buffer, self.byte_buffer = list(token_buffer), list(byte_buffer)
        self.segment_buffer = list(segment_buffer)
        self.source_buffer = list(source_buffer)
        self.next_segment_id = int(
            state.get("next_segment_id", max(segment_buffer, default=-1) + 1)
        )
        if self.next_segment_id < max(self.segment_buffer, default=-1) + 1:
            raise ValueError("checkpoint next segment ID is inconsistent with buffered segments")
        self.documents_packed = int(state["documents_packed"])


@dataclass(frozen=True, slots=True)
class NextTokenStatistics:
    cross_entropy: Tensor
    nll_sum: Tensor
    token_count: int
    byte_count: int
    correct_top1: int
    correct_top5: int

    @property
    def effective_cross_entropy(self) -> Tensor:
        return self.nll_sum / max(1, self.byte_count)


def next_token_statistics(
    logits: Tensor, labels: Tensor, target_byte_lengths: Tensor,
    mask: Tensor | None = None,
) -> NextTokenStatistics:
    """Compute CE and tokenizer-independent UTF-8-byte-normalized CE."""

    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("next-token logits and labels must be (batch,time,vocabulary)/(batch,time)")
    if target_byte_lengths.shape != labels.shape or target_byte_lengths.dtype != torch.long:
        raise ValueError("target byte lengths must be int64 and match labels")
    if mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)
    if mask.shape != labels.shape or mask.dtype != torch.bool or not bool(mask.any()):
        raise ValueError("next-token statistics require a nonempty boolean mask")
    flat_logits, flat_labels = logits.flatten(0, 1), labels.flatten()
    flat_mask = mask.flatten()
    nll = F.cross_entropy(flat_logits.float(), flat_labels, reduction="none")[flat_mask]
    selected_labels = flat_labels[flat_mask]
    selected_logits = flat_logits[flat_mask]
    prediction = selected_logits.argmax(-1)
    top_count = min(5, flat_logits.shape[-1])
    top5 = selected_logits.topk(top_count, -1).indices
    return NextTokenStatistics(
        nll.mean(), nll.sum(), int(flat_mask.sum()),
        int(target_byte_lengths[mask].sum()),
        int((prediction == selected_labels).sum()),
        int((top5 == selected_labels[:, None]).any(-1).sum()),
    )


def resonator_state_rms(output: LanguageModelOutput) -> tuple[Tensor, Tensor]:
    """Return differentiable mean and maximum RMS across final resonator states."""

    energy = torch.stack([
        state.value.float().square().mean()
        for block in output.mrrn.state.blocks for state in block.resonators
    ])
    rms = energy.clamp_min(0).sqrt()
    return rms.mean(), rms.max()


def resonator_state_regularization(output: LanguageModelOutput, *, target_rms: float) -> Tensor:
    """Penalize only state energy above a physically interpretable RMS budget."""

    if target_rms <= 0:
        raise ValueError("resonator state target RMS must be positive")
    energy = torch.stack([
        state.value.float().square().mean()
        for block in output.mrrn.state.blocks for state in block.resonators
    ])
    return (energy - target_rms**2).clamp_min(0).mean()


def stability_metrics(output: LanguageModelOutput) -> dict[str, float]:
    """Cheap per-update measurements used by the stability guard and dashboard."""

    state_mean, state_max = resonator_state_rms(output)
    decay = torch.stack([
        diagnostic.alpha.detach().float().mean().cpu()
        for block in output.mrrn.diagnostics for diagnostic in block.resonance
    ])
    branch = torch.stack([
        weight.detach().float().mean((0, 1)).cpu()
        for diagnostic in output.mrrn.diagnostics for weight in diagnostic.branch_weights
    ]).mean(0)
    return {
        "architecture/state_rms": float(state_mean.detach().cpu()),
        "architecture/state_rms_max": float(state_max.detach().cpu()),
        "architecture/mean_decay": float(decay.mean()),
        "architecture/branch_resonance": float(branch[0]),
        "architecture/branch_local": float(branch[1]),
        "architecture/branch_attention": float(branch[2]),
        "architecture/branch_identity": float(branch[3]),
    }


def architecture_metrics(output: LanguageModelOutput) -> dict[str, float]:
    """Reduce retained MRRN diagnostics to low-cardinality chart metrics."""

    result = stability_metrics(output)
    energies = torch.stack([
        (band.data.detach().float().square() * band.mask.unsqueeze(-1)).sum()
        / (band.mask.sum().clamp_min(1) * band.data.shape[-1])
        for band in output.mrrn.bands
    ])
    fractions = energies / energies.sum().clamp_min(1e-12)
    for index, value in enumerate(fractions):
        result[f"architecture/scale_{index}_energy_fraction"] = float(value.cpu())
    frequency, attention_entropy, spectral_fraction = [], [], []
    for block_state, diagnostic in zip(output.mrrn.state.blocks, output.mrrn.diagnostics, strict=True):
        for state, resonance in zip(block_state.resonators, diagnostic.resonance, strict=True):
            frequency.append(resonance.omega.detach().float().abs().mean().cpu())
        for weights in diagnostic.attention_weights:
            if weights is not None:
                probability = weights.detach().float().mean(-1)
                probability /= probability.sum(2, keepdim=True).clamp_min(1e-12)
                attention_entropy.append(
                    -(probability * probability.clamp_min(1e-12).log()).sum(2).mean().cpu()
                )
        for mixer in diagnostic.spectral_mixers:
            if mixer is not None:
                spectral_fraction.append(mixer.spectral_fraction.detach().float().mean().cpu())
    for name, values in (
        ("mean_absolute_frequency", frequency), ("attention_entropy", attention_entropy),
        ("spectral_fraction", spectral_fraction),
    ):
        if values:
            result[f"architecture/{name}"] = float(torch.stack(values).mean())
    return result


@dataclass(frozen=True, slots=True)
class LMTrainingConfig:
    output_dir: str = "outputs/fineweb-4p7m-stable"
    total_tokens: int = 20_000_000
    sequence_length: int = 2048
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_tokens: int = 800_000
    minimum_learning_rate_ratio: float = 0.1
    maximum_gradient_norm: float = 1.0
    spectral_regularization_weight: float = 1e-4
    state_regularization_weight: float = 1e-4
    state_target_rms: float = 8.0
    state_warning_rms: float = 16.0
    state_abort_rms: float = 32.0
    gradient_warning_norm: float = 100.0
    gradient_abort_norm: float = 1_000.0
    stability_patience: int = 3
    gradient_recovery: bool = True
    gradient_backoff_factor: float = 0.5
    gradient_recovery_limit: int = 4
    log_interval: int = 1
    architecture_log_interval: int = 10
    evaluation_interval: int = 100
    evaluation_batches: int = 8
    checkpoint_interval: int = 100
    keep_checkpoints: int = 3
    generation_tokens: int = 64
    generation_prompt: str = "The meaning of resonance is"
    seed: int = 20260721
    device: str = "auto"
    precision: str = "auto"
    trackio_project: str = "mrrn-fineweb"
    run_name: str = "mrrn-4p7m-fineweb-stable-20m"
    trackio_space_id: str | None = None
    trackio_remote_log_interval: int = 4
    # Logging remains authoritative by default.  The resource-intensive local
    # web observer is opt-in and can run in a separate process.
    show_dashboard: bool = False
    spectral_dashboard: bool = True
    spectral_snapshot_interval: int = 100
    spectral_snapshot_tokens: int = 32
    spectral_dashboard_prompt: str = (
        "Resonance lets a pattern persist across time while multiresolution pathways "
        "separate fast local detail from slow global structure."
    )
    spectral_baseline_metrics: str | None = None

    def __post_init__(self) -> None:
        positive = (
            self.total_tokens, self.sequence_length, self.micro_batch_size,
            self.gradient_accumulation_steps, self.warmup_tokens,
            self.maximum_gradient_norm, self.log_interval,
            self.architecture_log_interval, self.evaluation_interval,
            self.evaluation_batches, self.checkpoint_interval, self.keep_checkpoints,
            self.state_target_rms, self.state_warning_rms, self.state_abort_rms,
            self.gradient_warning_norm, self.gradient_abort_norm,
            self.stability_patience,
            self.spectral_snapshot_interval, self.spectral_snapshot_tokens,
            self.trackio_remote_log_interval,
        )
        if min(positive) <= 0:
            raise ValueError("token, batch, interval, and limit controls must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning rate must be positive and weight decay nonnegative")
        if (
            self.spectral_regularization_weight < 0
            or self.state_regularization_weight < 0
            or not 0 <= self.minimum_learning_rate_ratio <= 1
        ):
            raise ValueError("regularization and minimum learning-rate ratio are invalid")
        if not self.state_target_rms < self.state_warning_rms < self.state_abort_rms:
            raise ValueError("state RMS controls must satisfy target < warning < abort")
        if not self.maximum_gradient_norm < self.gradient_warning_norm < self.gradient_abort_norm:
            raise ValueError("gradient controls must satisfy clip < warning < abort")
        if not 0 < self.gradient_backoff_factor < 1 or self.gradient_recovery_limit < 0:
            raise ValueError("gradient recovery requires a factor in (0,1) and a nonnegative limit")
        if self.total_tokens < self.micro_batch_size * self.sequence_length:
            raise ValueError("total token budget must contain at least one microbatch")
        cuda_index = self.device.removeprefix("cuda:")
        valid_device = self.device in {"auto", "cpu", "mps", "cuda"} or (
            self.device.startswith("cuda:") and cuda_index.isdigit()
        )
        if (
            self.generation_tokens < 0
            or not valid_device
            or self.precision not in {"auto", "fp32", "bf16", "fp16"}
        ):
            raise ValueError("generation length, device, or precision selection is invalid")

    @property
    def tokens_per_microbatch(self) -> int:
        return self.micro_batch_size * self.sequence_length

    @property
    def tokens_per_update(self) -> int:
        return self.tokens_per_microbatch * self.gradient_accumulation_steps

    @property
    def total_steps(self) -> int:
        return ceil(self.total_tokens / self.tokens_per_update)

    @property
    def warmup_steps(self) -> int:
        return max(1, min(self.total_steps - 1, ceil(self.warmup_tokens / self.tokens_per_update)))


@dataclass(slots=True)
class LMTrainingState:
    step: int = 0
    tokens_seen: int = 0
    bytes_seen: int = 0
    best_effective_cross_entropy: float = float("inf")
    loss_ema: float | None = None
    elapsed_seconds: float = 0.0
    consecutive_state_warnings: int = 0
    consecutive_state_aborts: int = 0
    consecutive_gradient_warnings: int = 0
    consecutive_gradient_aborts: int = 0
    learning_rate_scale: float = 1.0
    gradient_recoveries: int = 0


@dataclass(slots=True)
class _MetricAccumulator:
    nll_sum: float = 0.0
    token_count: int = 0
    byte_count: int = 0
    correct_top1: int = 0
    correct_top5: int = 0
    total_loss_weighted: float = 0.0
    spectral_sum: float = 0.0
    state_regularization_sum: float = 0.0
    microbatches: int = 0

    def add(
        self,
        statistics: NextTokenStatistics,
        total_loss: Tensor,
        spectral: Tensor,
        state_regularization: Tensor,
    ) -> None:
        self.nll_sum += float(statistics.nll_sum.detach().cpu())
        self.token_count += statistics.token_count
        self.byte_count += statistics.byte_count
        self.correct_top1 += statistics.correct_top1
        self.correct_top5 += statistics.correct_top5
        self.total_loss_weighted += float(total_loss.detach().cpu()) * statistics.token_count
        self.spectral_sum += float(spectral.detach().cpu())
        self.state_regularization_sum += float(state_regularization.detach().cpu())
        self.microbatches += 1

    def metrics(self, prefix: str) -> dict[str, float]:
        if self.token_count <= 0 or self.byte_count <= 0 or self.microbatches <= 0:
            raise ValueError("metric accumulator has no valid language data")
        cross_entropy = self.nll_sum / self.token_count
        effective = self.nll_sum / self.byte_count
        return {
            f"{prefix}/total_loss": self.total_loss_weighted / self.token_count,
            f"{prefix}/cross_entropy_nats_per_token": cross_entropy,
            f"{prefix}/effective_cross_entropy_nats_per_byte": effective,
            f"{prefix}/bits_per_byte": effective / log(2),
            f"{prefix}/token_perplexity": exp(min(20.0, cross_entropy)),
            f"{prefix}/byte_perplexity": exp(min(20.0, effective)),
            f"{prefix}/top1_accuracy": self.correct_top1 / self.token_count,
            f"{prefix}/top5_accuracy": self.correct_top5 / self.token_count,
            f"{prefix}/spectral_regularization": self.spectral_sum / self.microbatches,
            f"{prefix}/state_energy_regularization": (
                self.state_regularization_sum / self.microbatches
            ),
            f"{prefix}/tokens": float(self.token_count),
            f"{prefix}/utf8_bytes": float(self.byte_count),
        }


class TrackioReporter:
    """Trackio dashboard plus an append-only JSONL metric/event mirror."""

    def __init__(
        self,
        config: LMTrainingConfig,
        run_config: dict[str, Any],
        *,
        resume: bool,
        initial_step: int = 0,
        initial_tokens: int = 0,
    ) -> None:
        if initial_step < 0 or initial_tokens < 0:
            raise ValueError("initial Trackio progress cannot be negative")
        try:
            import trackio
        except ImportError as error:
            raise RuntimeError("install the 'train' extra to enable required live Trackio charts") from error
        self.trackio, self.config = trackio, config
        output = Path(config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.metric_path = output / "metrics.jsonl"
        if not resume and self.metric_path.exists() and self.metric_path.stat().st_size:
            raise FileExistsError(
                "metrics.jsonl already exists; use resume or a fresh output directory"
            )
        self.frontend_dir = (
            prepare_runtime_frontend(output) if config.spectral_dashboard else None
        )
        previous_frontend = os.environ.get("TRACKIO_FRONTEND_DIR")
        if self.frontend_dir is not None:
            # Trackio's Space deployment path resolves this environment value
            # during init; the local show call also receives it explicitly.
            os.environ["TRACKIO_FRONTEND_DIR"] = str(self.frontend_dir)
        try:
            self.run = trackio.init(
                project=config.trackio_project,
                name=config.run_name,
                config=run_config,
                space_id=config.trackio_space_id,
                resume="allow" if resume else "never",
                embed=False,
                # The trainer logs portable process, host, and accelerator memory
                # itself at every step.  Avoid redundant background monitoring
                # threads so a completed local run exits promptly.
                auto_log_cpu=False,
                auto_log_gpu=False,
            )
        finally:
            if previous_frontend is None:
                os.environ.pop("TRACKIO_FRONTEND_DIR", None)
            else:
                os.environ["TRACKIO_FRONTEND_DIR"] = previous_frontend
        # Keep one bounded append stream instead of reopening the local
        # authority file on every optimizer step.  Metric rows are flushed at
        # a fixed cadence; alerts, artifacts, remote backpressure, and finish
        # force durability immediately.
        self._metric_handle = self.metric_path.open(
            "a",
            encoding="utf-8",
            buffering=64 * 1024,
        )
        self._metric_lock = Lock()
        self._metric_rows_since_flush = 0
        self._metric_flush_interval = 16
        self._remote_queue: Queue[
            tuple[int, dict[str, float]] | object
        ] = Queue(maxsize=64)
        self._remote_stop = object()
        self._remote_error: Exception | None = None
        self._remote_dropped = 0
        self._remote_drain_timeouts = 0
        self._remote_coalesced_metric_rows = 0
        self._pending_remote_metric: (
            tuple[int, dict[str, float]] | None
        ) = None
        self._remote_log_interval = config.trackio_remote_log_interval
        self._checkpoint_alerts_seen = 0
        self._checkpoint_alerts_remotely_coalesced = 0
        if config.trackio_space_id is None:
            self._register_local_run_before_dashboard(
                step=initial_step,
                tokens_seen=initial_tokens,
            )

        def remote_worker() -> None:
            while True:
                item = self._remote_queue.get()
                try:
                    if item is self._remote_stop:
                        return
                    step, metrics = item
                    # Trackio keeps its module-level current run in a
                    # context variable, which is not inherited by this bounded
                    # writer thread. Address the explicit run object so every
                    # queued point retains the initialized run identity.
                    self.run.log(metrics, step=step)
                except Exception as error:
                    # Trackio is observational. Preserve the first failure for
                    # a local receipt while training metrics continue to the
                    # authoritative JSONL mirror.
                    if self._remote_error is None:
                        self._remote_error = error
                finally:
                    self._remote_queue.task_done()

        self._remote_thread = Thread(
            target=remote_worker,
            name="mrrn-trackio-log-writer",
            daemon=True,
        )
        self._remote_thread.start()
        if config.show_dashboard and config.trackio_space_id is None:
            try:
                trackio.show(
                    project=config.trackio_project,
                    open_browser=True,
                    block_thread=False,
                    frontend_dir=(
                        None if self.frontend_dir is None else str(self.frontend_dir)
                    ),
                )
            except Exception as error:
                self._write({"kind": "dashboard_warning", "message": str(error)})

    def _flush_metric_mirror(self) -> None:
        with self._metric_lock:
            self._metric_handle.flush()
            self._metric_rows_since_flush = 0

    def _write(self, payload: dict[str, Any]) -> None:
        metric_row = payload.get("kind") == "metrics"
        with self._metric_lock:
            self._metric_handle.write(
                json.dumps(
                    payload,
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            self._metric_rows_since_flush += 1
            if (
                not metric_row
                or self._metric_rows_since_flush
                >= self._metric_flush_interval
            ):
                self._metric_handle.flush()
                self._metric_rows_since_flush = 0

    def _local_run_is_visible(self) -> bool | None:
        """Return whether the exact initialized run reached local persistence.

        Trackio 0.31 creates a run object and prints its creation notice before
        the first metric makes the run discoverable through ``Api().runs``.
        A missing API means a compatible test double or older Trackio version;
        in that case visibility is unknown rather than falsely certified.
        """

        api_factory = getattr(self.trackio, "Api", None)
        run_id = getattr(self.run, "id", None)
        if api_factory is None or run_id is None:
            return None
        try:
            records = api_factory().runs(self.config.trackio_project)
        except ValueError:
            # Trackio reports a not-yet-materialized local project as an
            # exception until its first asynchronous metric write commits.
            return False
        for record in records:
            record_id = (
                record.get("id")
                if isinstance(record, dict)
                else getattr(record, "id", None)
            )
            if record_id == run_id:
                return True
        return False

    def _register_local_run_before_dashboard(
        self,
        *,
        step: int,
        tokens_seen: int,
        timeout_seconds: float = 2.0,
    ) -> None:
        """Persist initial progress before any local observer can be opened."""

        error: str | None = None
        try:
            # This is an existing canonical metric, not a synthetic training
            # result. It records the exact checkpoint/start position and also
            # causes Trackio to durably register the run and its configuration.
            self.run.log(
                {"progress/tokens_seen": float(tokens_seen)},
                step=step,
            )
            deadline = monotonic() + timeout_seconds
            visible = self._local_run_is_visible()
            while visible is False and monotonic() < deadline:
                sleep(0.01)
                visible = self._local_run_is_visible()
        except Exception as registration_error:
            visible = False
            error = (
                f"{type(registration_error).__name__}: "
                f"{registration_error}"
            )
        self._write({
            "kind": "trackio_run_registration",
            "project": self.config.trackio_project,
            "run": self.config.run_name,
            "run_id": getattr(self.run, "id", None),
            "step": step,
            "tokens_seen": tokens_seen,
            "visible_before_dashboard": visible,
            "error": error,
        })
        if visible is False:
            print(
                "[TRACKIO WARN] The initialized run was not visible in the "
                "local Trackio store before dashboard launch; the dashboard "
                "will continue polling while training remains authoritative.",
                flush=True,
            )

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        cleaned = {name: float(value) for name, value in metrics.items()}
        if not all(isfinite(value) for value in cleaned.values()):
            raise FloatingPointError("refusing to log non-finite training metrics")
        self._write({"kind": "metrics", "step": step, "metrics": cleaned})
        if step % self._remote_log_interval:
            if self._pending_remote_metric is not None:
                self._remote_coalesced_metric_rows += 1
            self._pending_remote_metric = (step, cleaned)
            return
        if self._pending_remote_metric is not None:
            self._remote_coalesced_metric_rows += 1
            self._pending_remote_metric = None
        try:
            self._remote_queue.put_nowait((step, cleaned))
        except Full:
            # The complete local stream remains authoritative; bounded remote
            # backpressure may omit dashboard-only intermediate points.
            self._remote_dropped += 1
            self._flush_metric_mirror()

    def _enqueue_pending_remote_metric(
        self, *, deadline: float,
    ) -> bool:
        pending = self._pending_remote_metric
        if pending is None:
            return True
        while monotonic() < deadline:
            try:
                self._remote_queue.put_nowait(pending)
                self._pending_remote_metric = None
                return True
            except Full:
                sleep(0.001)
        self._remote_dropped += 1
        self._pending_remote_metric = None
        self._flush_metric_mirror()
        return False

    def _drain_remote_logs(self, *, timeout_seconds: float = 5.0) -> bool:
        """Wait a bounded time for observational delivery.

        ``Queue.join`` has no timeout and can deadlock training shutdown if a
        third-party logging call stops returning. Polling the queue's protected
        unfinished-task counter keeps the training authority bounded.
        """

        if timeout_seconds <= 0:
            raise ValueError("Trackio drain timeout must be positive")
        deadline = monotonic() + timeout_seconds
        if not self._enqueue_pending_remote_metric(deadline=deadline):
            self._remote_drain_timeouts += 1
            return False
        while monotonic() < deadline:
            with self._remote_queue.all_tasks_done:
                if self._remote_queue.unfinished_tasks == 0:
                    return True
            sleep(0.005)
        self._remote_drain_timeouts += 1
        return False

    def alert(self, title: str, text: str, *, level: str, step: int) -> None:
        if level not in {"info", "warn", "error"}:
            raise ValueError("unknown Trackio alert level")
        checkpoint_alert = title == "MRCRA checkpoint saved" and level == "info"
        if checkpoint_alert:
            self._checkpoint_alerts_seen += 1
        # Preserve every alert in the authoritative local stream. Repetitive
        # success notices are sent remotely on the first occurrence and every
        # tenth checkpoint; warnings, errors, and first-event alerts are never
        # coalesced.
        coalesced = (
            checkpoint_alert
            and self._checkpoint_alerts_seen != 1
            and self._checkpoint_alerts_seen % 10 != 0
        )
        if coalesced:
            self._checkpoint_alerts_remotely_coalesced += 1
            self._write({
                "kind": "alert",
                "step": step,
                "title": title,
                "text": text,
                "level": level,
                "remote_coalesced": True,
            })
            return
        drained = self._drain_remote_logs()
        alert_level = {
            "info": self.trackio.AlertLevel.INFO,
            "warn": self.trackio.AlertLevel.WARN,
            "error": self.trackio.AlertLevel.ERROR,
        }[level]
        if drained:
            try:
                self.trackio.alert(
                    title=title, text=text, level=alert_level
                )
            except Exception as error:
                if self._remote_error is None:
                    self._remote_error = error
        self._write({
            "kind": "alert",
            "step": step,
            "title": title,
            "text": text,
            "level": level,
            "remote_coalesced": False,
            "remote_delivery_attempted": drained,
        })

    def log_spectral_evidence(self, evidence: dict[str, Any], *, step: int) -> int:
        """Version a complete evidence snapshot as a Trackio run artifact."""

        if not self.config.spectral_dashboard:
            return 0
        if not self._drain_remote_logs():
            self._write({
                "kind": "spectral_snapshot_deferred",
                "step": step,
                "reason": "remote metric queue did not drain within its bound",
            })
            return 0
        self._flush_metric_mirror()
        baseline = self.config.spectral_baseline_metrics
        if baseline is None:
            candidate = Path(self.config.output_dir).parent / "fineweb-4p7m" / "metrics.jsonl"
            if candidate.is_file() and candidate.resolve() != self.metric_path.resolve():
                baseline = str(candidate)
        enriched = attach_training_evidence(
            evidence,
            current_metrics=self.metric_path,
            baseline_metrics=baseline,
        )
        snapshot = write_evidence_atomically(
            Path(self.config.output_dir) / "spectral" / SPECTRAL_DATA_FILENAME,
            enriched,
        )
        artifact = self.trackio.Artifact(
            SPECTRAL_ARTIFACT_NAME,
            type=SPECTRAL_ARTIFACT_TYPE,
            description="Checkpoint-grounded MRCRA cognitive and spectral dashboard evidence.",
            metadata={
                "schema_version": enriched["schema_version"],
                "step": step,
                "tokens_seen": enriched.get("checkpoint", {}).get("tokens_seen", 0),
            },
        )
        artifact.add_file(snapshot, name=SPECTRAL_DATA_FILENAME)
        logged = self.trackio.log_artifact(
            artifact,
            aliases=[f"step-{step:07d}"],
        )
        raw_version = logged.version or 0
        if isinstance(raw_version, str) and raw_version.startswith("v"):
            raw_version = raw_version[1:]
        version = int(raw_version)
        self._write(
            {
                "kind": "spectral_snapshot",
                "step": step,
                "artifact": SPECTRAL_ARTIFACT_NAME,
                "version": version,
            }
        )
        return version

    def log_phase_transition_trace(self, path: Path, *, step: int) -> int:
        """Persist the first hard-event receipt as a versioned Trackio artifact."""

        if not self._drain_remote_logs():
            self._write({
                "kind": "phase_transition_trace_deferred",
                "step": step,
                "path": str(path),
                "reason": "remote metric queue did not drain within its bound",
            })
            return 0
        artifact = self.trackio.Artifact(
            "mrcra-first-hard-event",
            type="mrcra-phase-transition",
            description=(
                "Exact first hard-event threshold, graph, workspace, and gradient receipt."
            ),
            metadata={"schema_version": 1, "step": step},
        )
        artifact.add_file(path, name="first-hard-event.json")
        logged = self.trackio.log_artifact(
            artifact, aliases=["first-hard-event", f"step-{step:07d}"],
        )
        raw_version = logged.version or 0
        if isinstance(raw_version, str) and raw_version.startswith("v"):
            raw_version = raw_version[1:]
        version = int(raw_version)
        self._write({
            "kind": "phase_transition_trace",
            "step": step,
            "artifact": "mrcra-first-hard-event",
            "version": version,
            "path": str(path),
        })
        return version

    def finish(self) -> None:
        drained = self._drain_remote_logs()
        if drained:
            try:
                self._remote_queue.put_nowait(self._remote_stop)
            except Full:  # pragma: no cover - drain guarantees room
                drained = False
        if drained:
            self._remote_thread.join(timeout=5.0)
        worker_alive = self._remote_thread.is_alive()
        if (
            self._remote_dropped
            or self._remote_error is not None
            or self._remote_drain_timeouts
            or self._remote_coalesced_metric_rows
            or self._checkpoint_alerts_remotely_coalesced
            or worker_alive
        ):
            self._write({
                "kind": "trackio_remote_summary",
                "dropped_remote_metric_rows": self._remote_dropped,
                "coalesced_remote_metric_rows": (
                    self._remote_coalesced_metric_rows
                ),
                "drain_timeouts": self._remote_drain_timeouts,
                "worker_alive_at_bounded_finish": worker_alive,
                "checkpoint_alerts_remotely_coalesced": (
                    self._checkpoint_alerts_remotely_coalesced
                ),
                "error": (
                    None
                    if self._remote_error is None
                    else f"{type(self._remote_error).__name__}: "
                    f"{self._remote_error}"
                ),
            })
        if not worker_alive:
            try:
                self.trackio.finish()
            except Exception as error:
                self._write({
                    "kind": "trackio_finish_warning",
                    "error": f"{type(error).__name__}: {error}",
                })
        self._flush_metric_mirror()
        self._metric_handle.close()


def _device_for(selection: str) -> torch.device:
    if selection == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(selection)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda" and device.index is not None:
        count = torch.cuda.device_count()
        if device.index >= count:
            raise RuntimeError(
                f"CUDA device {device.index} was requested, but only {count} device(s) are visible"
            )
    return device


def _precision_for(device: torch.device, selection: str) -> torch.dtype | None:
    """Resolve CUDA AMP precision; ``None`` means full FP32 execution."""

    if device.type != "cuda":
        if selection not in {"auto", "fp32"}:
            raise RuntimeError(f"{selection} mixed precision is currently supported only on CUDA")
        return None
    if selection == "fp32":
        return None
    if selection == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but this CUDA device does not support it")
        return torch.bfloat16
    if selection == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _configure_cuda(device: torch.device) -> None:
    """Enable safe high-throughput CUDA defaults without changing model weights."""

    if device.type != "cuda":
        return
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _runtime_details(device: torch.device, amp_dtype: torch.dtype | None) -> dict[str, Any]:
    details: dict[str, Any] = {
        "device": str(device),
        "precision": "fp32" if amp_dtype is None else str(amp_dtype).removeprefix("torch."),
        "torch_version": str(torch.__version__),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        details.update(
            {
                "cuda_runtime": torch.version.cuda,
                "gpu_name": properties.name,
                "gpu_compute_capability": f"{properties.major}.{properties.minor}",
                "gpu_memory_gib": properties.total_memory / 2**30,
                "cudnn_version": torch.backends.cudnn.version(),
                "fused_adamw": True,
                "pinned_transfers": True,
            }
        )
    return details


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_metrics(device: torch.device) -> dict[str, float]:
    result: dict[str, float] = {}
    try:
        import psutil

        result["system/process_rss_gib"] = psutil.Process().memory_info().rss / 2**30
        result["system/system_memory_percent"] = float(psutil.virtual_memory().percent)
    except ImportError:
        if resource is not None:
            maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            divisor = 2**30 if platform.system() == "Darwin" else 2**20
            result["system/process_peak_rss_gib"] = maximum / divisor
    if device.type == "mps":
        result.update(
            {
                "system/accelerator_current_gib": torch.mps.current_allocated_memory() / 2**30,
                "system/accelerator_driver_gib": torch.mps.driver_allocated_memory() / 2**30,
                "system/accelerator_recommended_gib": torch.mps.recommended_max_memory() / 2**30,
            }
        )
    elif device.type == "cuda":
        result.update(
            {
                "system/accelerator_current_gib": torch.cuda.memory_allocated(device) / 2**30,
                "system/accelerator_reserved_gib": torch.cuda.memory_reserved(device) / 2**30,
                "system/accelerator_peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            }
        )
    return result


class NextTokenTrainer:
    """Complete finite-token-budget trainer with fail-closed resume semantics."""

    def __init__(
        self,
        model: MRRNLanguageModel,
        tokenizer: TextTokenizer,
        train_stream: PackedTokenStream,
        evaluation_batches: Sequence[PackedBatch],
        config: LMTrainingConfig,
    ) -> None:
        if tokenizer.vocabulary_size != model.vocabulary_size:
            raise ValueError("tokenizer vocabulary does not match the language model")
        if len(evaluation_batches) != config.evaluation_batches:
            raise ValueError("the retained evaluation batch count does not match training configuration")
        if any(batch.input_ids.shape != (config.micro_batch_size, config.sequence_length)
               for batch in evaluation_batches):
            raise ValueError("evaluation batches must match the configured batch and sequence sizes")
        self.model, self.tokenizer, self.train_stream = model, tokenizer, train_stream
        self.config = config
        self.device = _device_for(config.device)
        _configure_cuda(self.device)
        self.amp_dtype = _precision_for(self.device, config.precision)
        self.runtime = _runtime_details(self.device, self.amp_dtype)
        self._non_blocking_transfers = self.device.type == "cuda"
        self.evaluation_batches = tuple(
            batch.pin_memory() if self._non_blocking_transfers else batch
            for batch in evaluation_batches
        )
        self.model.to(self.device)
        policy = OptimizerPolicy(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            total_steps=max(config.total_steps, config.warmup_steps + 1),
            minimum_learning_rate_ratio=config.minimum_learning_rate_ratio,
        )
        self.optimizer = build_adamw(model, policy, fused=self.device.type == "cuda")
        self.scheduler = build_scheduler(self.optimizer, policy)
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if self.amp_dtype == torch.float16
            else None
        )
        self.state = LMTrainingState()
        self._spectral_modules = tuple(
            module for module in model.modules() if isinstance(module, ResonantSpectralGLU)
        )
        if not self._spectral_modules:
            raise ValueError("the full language model requires spectral activation modules")
        self._resumed = False
        self._last_spectral_snapshot_step = -1
        self._safety_checkpoint_pending = False

    def _run_identity(self) -> dict[str, Any]:
        return {
            "model_config": asdict(self.model.config),
            "model_parameters": self.model.parameter_count,
            "tokenizer": self.tokenizer.identity(),
            "training": asdict(self.config),
            "runtime": self.runtime,
            "source": {
                key: value for key, value in self.train_stream.source.state_dict().items()
                if key not in {"raw_rows_scanned", "documents_yielded"}
            },
        }

    def _checkpoint_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "identity": self._run_identity(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "training_state": asdict(self.state),
            "train_stream": self.train_stream.state_dict(),
            "torch_rng_state": torch.random.get_rng_state(),
        }
        if self.scaler is not None:
            payload["gradient_scaler"] = self.scaler.state_dict()
        if self.device.type == "mps":
            payload["accelerator_rng_state"] = torch.mps.get_rng_state()
        elif self.device.type == "cuda":
            payload["accelerator_rng_state"] = torch.cuda.get_rng_state(self.device)
        return payload

    def save_checkpoint(self, *, best: bool = False) -> Path:
        output = Path(self.config.output_dir)
        checkpoint_dir = output / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        destination = checkpoint_dir / ("best.pt" if best else f"step-{self.state.step:07d}.pt")
        descriptor, temporary_name = tempfile.mkstemp(prefix="checkpoint-", suffix=".tmp", dir=checkpoint_dir)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(self._checkpoint_payload(), temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        if not best:
            checkpoints = sorted(checkpoint_dir.glob("step-*.pt"))
            for obsolete in checkpoints[: -self.config.keep_checkpoints]:
                obsolete.unlink()
            latest = checkpoint_dir / "latest.json"
            latest.write_text(
                json.dumps({"checkpoint": destination.name, "step": self.state.step}, indent=2) + "\n",
                encoding="utf-8",
            )
        return destination

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=True)
        if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("unsupported language-training checkpoint version")
        expected, actual = self._run_identity(), payload.get("identity")
        if not isinstance(actual, dict):
            raise ValueError("language-training checkpoint identity is missing")
        # The token budget and output directory may be extended/relocated on
        # resume; every semantic model/data/batching contract remains exact.
        expected_training = dict(expected["training"])
        actual_training = dict(actual.get("training", {}))
        for key in (
            "total_tokens", "output_dir", "show_dashboard", "spectral_dashboard",
            "spectral_snapshot_interval", "spectral_snapshot_tokens",
            "spectral_dashboard_prompt", "spectral_baseline_metrics",
            "device", "precision",
            "gradient_recovery", "gradient_backoff_factor", "gradient_recovery_limit",
        ):
            expected_training.pop(key, None)
            actual_training.pop(key, None)
        if (
            actual.get("model_config") != expected["model_config"]
            or actual.get("model_parameters") != expected["model_parameters"]
            or actual.get("tokenizer") != expected["tokenizer"]
            or actual.get("source") != expected["source"]
            or actual_training != expected_training
        ):
            raise ValueError("checkpoint model, tokenizer, source, or training contract does not match")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        if self.scaler is not None and payload.get("gradient_scaler") is not None:
            self.scaler.load_state_dict(payload["gradient_scaler"])
        state = payload["training_state"]
        self.state = LMTrainingState(**state)
        if self.state.tokens_seen >= self.config.total_tokens:
            raise ValueError("resumed checkpoint has already exhausted the configured token budget")
        self.train_stream.load_state_dict(payload["train_stream"])
        torch.random.set_rng_state(payload["torch_rng_state"].cpu())
        if "accelerator_rng_state" in payload:
            if self.device.type == "mps":
                torch.mps.set_rng_state(payload["accelerator_rng_state"].cpu())
            elif self.device.type == "cuda":
                torch.cuda.set_rng_state(payload["accelerator_rng_state"], self.device)
        self._resumed = True

    def _autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.amp.autocast("cuda", dtype=self.amp_dtype)

    def _backoff_learning_rate(self) -> None:
        factor = self.config.gradient_backoff_factor
        self.state.learning_rate_scale *= factor
        for group in self.optimizer.param_groups:
            group["lr"] *= factor

    def _step_scheduler(self) -> None:
        self.scheduler.step()
        if self.state.learning_rate_scale != 1.0:
            for group, scheduled_rate in zip(
                self.optimizer.param_groups, self.scheduler.get_last_lr(), strict=True
            ):
                group["lr"] = scheduled_rate * self.state.learning_rate_scale

    def _move_batch(self, batch: PackedBatch) -> PackedBatch:
        if self._non_blocking_transfers:
            batch = batch.pin_memory()
        return batch.to(self.device, non_blocking=self._non_blocking_transfers)

    def _publish_spectral_snapshot(self, reporter: TrackioReporter) -> None:
        """Publish interpretability evidence without entering training authority."""

        if (
            not self.config.spectral_dashboard
            or self._last_spectral_snapshot_step == self.state.step
        ):
            return
        was_training = self.model.training
        try:
            from .visualization import model_spectral_evidence

            evidence = model_spectral_evidence(
                self.model,
                self.tokenizer,
                prompt=self.config.spectral_dashboard_prompt,
                maximum_tokens=self.config.spectral_snapshot_tokens,
                step=self.state.step,
                tokens_seen=self.state.tokens_seen,
                source="live training model",
                format_version=CHECKPOINT_FORMAT_VERSION,
            )
            version = reporter.log_spectral_evidence(evidence, step=self.state.step)
            self._last_spectral_snapshot_step = self.state.step
            reporter.alert(
                "Spectral dashboard updated",
                f"Published {SPECTRAL_ARTIFACT_NAME}:v{version} at step {self.state.step}.",
                level="info",
                step=self.state.step,
            )
        except Exception as error:
            reporter.alert(
                "Spectral dashboard snapshot failed",
                f"{type(error).__name__}: {error}. Training continues because visualization is non-authoritative.",
                level="warn",
                step=self.state.step,
            )
        finally:
            self.model.train(was_training)

    def evaluate(self) -> tuple[dict[str, float], dict[str, float]]:
        self.model.eval()
        accumulator = _MetricAccumulator()
        architecture: dict[str, float] = {}
        started = perf_counter()
        with torch.no_grad():
            for retained in self.evaluation_batches:
                batch = retained.to(
                    self.device, non_blocking=self._non_blocking_transfers
                )
                with self._autocast():
                    output = self.model(batch.input_ids)
                    statistics = next_token_statistics(
                        output.logits, batch.labels, batch.target_byte_lengths,
                        batch.loss_mask,
                    )
                    spectral = spectral_activation_regularization(self._spectral_modules)
                    state_regularization = resonator_state_regularization(
                        output, target_rms=self.config.state_target_rms
                    )
                    total = (
                        statistics.cross_entropy
                        + self.config.spectral_regularization_weight * spectral
                        + self.config.state_regularization_weight * state_regularization
                    )
                accumulator.add(statistics, total, spectral, state_regularization)
                architecture = architecture_metrics(output)
        _synchronize(self.device)
        metrics = accumulator.metrics("eval")
        elapsed = perf_counter() - started
        metrics["eval/seconds"] = elapsed
        metrics["eval/tokens_per_second"] = accumulator.token_count / max(elapsed, 1e-9)
        self.model.train()
        return metrics, architecture

    def _write_generation_sample(self) -> tuple[str, int]:
        prompt_ids = self.tokenizer.encode_prompt(self.config.generation_prompt)
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        with self._autocast():
            generated = self.model.generate(
                prompt,
                maximum_new_tokens=self.config.generation_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
            )
        new_ids = generated[0, len(prompt_ids):].tolist()
        text = self.tokenizer.decode(new_ids)
        sample_dir = Path(self.config.output_dir) / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        path = sample_dir / f"step-{self.state.step:07d}.txt"
        path.write_text(
            self.config.generation_prompt + text + "\n",
            encoding="utf-8",
        )
        return text, len(new_ids)

    def _diagnostic_alerts(self, reporter: TrackioReporter, metrics: dict[str, float]) -> None:
        step = self.state.step
        loss = metrics["train/cross_entropy_nats_per_token"]
        previous = self.state.loss_ema
        self.state.loss_ema = loss if previous is None else 0.98 * previous + 0.02 * loss
        if previous is not None and step >= 20 and loss > max(previous * 1.75, previous + 1.0):
            reporter.alert(
                "Training loss spike",
                f"Cross entropy {loss:.4f} is far above EMA {previous:.4f} at step {step}.",
                level="warn", step=step,
            )
        driver = metrics.get("system/accelerator_driver_gib")
        recommended = metrics.get("system/accelerator_recommended_gib")
        if driver is not None and recommended is not None and driver / max(recommended, 1e-9) > 0.90:
            reporter.alert(
                "Accelerator memory pressure",
                f"Driver allocation is {driver:.2f} GiB of {recommended:.2f} GiB recommended.",
                level="warn", step=step,
            )

    def _write_stability_abort(
        self, *, candidate_step: int, gradient_norm: float, state_rms_max: float, reason: str
    ) -> Path:
        path = Path(self.config.output_dir) / f"stability-abort-step-{candidate_step:07d}.json"
        path.write_text(json.dumps({
            "reason": reason,
            "candidate_step_not_applied": candidate_step,
            "last_completed_step": self.state.step,
            "tokens_seen_before_candidate": self.state.tokens_seen,
            "gradient_norm_before_clip": gradient_norm,
            "state_rms_max": state_rms_max,
            "thresholds": {
                "gradient_warning_norm": self.config.gradient_warning_norm,
                "gradient_abort_norm": self.config.gradient_abort_norm,
                "state_warning_rms": self.config.state_warning_rms,
                "state_abort_rms": self.config.state_abort_rms,
                "patience": self.config.stability_patience,
            },
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def _write_gradient_recovery(
        self, *, candidate_step: int, gradient_norm: float, state_rms_max: float
    ) -> Path:
        path = Path(self.config.output_dir) / f"gradient-recovery-step-{candidate_step:07d}.json"
        path.write_text(json.dumps({
            "action": "apply_clipped_update_with_learning_rate_backoff",
            "candidate_step": candidate_step,
            "gradient_norm_before_clip": gradient_norm,
            "gradient_norm_after_clip": self.config.maximum_gradient_norm,
            "state_rms_max": state_rms_max,
            "backoff_factor": self.config.gradient_backoff_factor,
            "learning_rate_scale": self.state.learning_rate_scale,
            "recovery": self.state.gradient_recoveries,
            "recovery_limit": self.config.gradient_recovery_limit,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def _enforce_stability(
        self,
        reporter: TrackioReporter,
        gradient: GradientReport,
        stability: dict[str, float],
    ) -> None:
        """Warn persistently and abort before applying a repeatedly unsafe update."""

        step = self.state.step + 1
        gradient_norm = float(gradient.total_before_clip.detach().cpu())
        state_rms_max = stability["architecture/state_rms_max"]
        self.state.consecutive_gradient_warnings = (
            self.state.consecutive_gradient_warnings + 1
            if gradient_norm >= self.config.gradient_warning_norm else 0
        )
        self.state.consecutive_gradient_aborts = (
            self.state.consecutive_gradient_aborts + 1
            if gradient_norm >= self.config.gradient_abort_norm else 0
        )
        self.state.consecutive_state_warnings = (
            self.state.consecutive_state_warnings + 1
            if state_rms_max >= self.config.state_warning_rms else 0
        )
        self.state.consecutive_state_aborts = (
            self.state.consecutive_state_aborts + 1
            if state_rms_max >= self.config.state_abort_rms else 0
        )
        patience = self.config.stability_patience
        for counter, title, text in (
            (
                self.state.consecutive_gradient_warnings,
                "Persistent large pre-clip gradient",
                f"Gradient norm is {gradient_norm:.3f}; applied clip coefficient is "
                f"{float(gradient.clip_coefficient.detach().cpu()):.6f}.",
            ),
            (
                self.state.consecutive_state_warnings,
                "Persistent resonator-state growth",
                f"Maximum resonator state RMS is {state_rms_max:.3f}.",
            ),
        ):
            if counter == patience or (counter > patience and counter % 25 == 0):
                reporter.alert(title, text, level="warn", step=step)
        gradient_pressure = self.state.consecutive_gradient_aborts >= patience
        state_failure = self.state.consecutive_state_aborts >= patience
        if (
            gradient_pressure
            and not state_failure
            and self.config.gradient_recovery
            and self.state.gradient_recoveries < self.config.gradient_recovery_limit
        ):
            self.state.gradient_recoveries += 1
            self._backoff_learning_rate()
            self._safety_checkpoint_pending = True
            self.state.consecutive_gradient_aborts = 0
            self.state.consecutive_gradient_warnings = 0
            path = self._write_gradient_recovery(
                candidate_step=step,
                gradient_norm=gradient_norm,
                state_rms_max=state_rms_max,
            )
            reporter.alert(
                "Gradient-pressure recovery",
                f"Pre-clip norm {gradient_norm:.3f} remained above "
                f"{self.config.gradient_abort_norm:g}, but clipping produced a finite "
                f"norm-{self.config.maximum_gradient_norm:g} update. Reduced the learning-rate "
                f"scale to {self.state.learning_rate_scale:.5f}; diagnostics: {path.name}.",
                level="warn", step=step,
            )
            return
        reasons = []
        if gradient_pressure:
            recovery_status = (
                "gradient recovery was disabled"
                if not self.config.gradient_recovery
                else f"{self.state.gradient_recoveries} recoveries were exhausted"
            )
            reasons.append(
                f"gradient norm exceeded {self.config.gradient_abort_norm:g} for {patience} updates "
                f"and {recovery_status}"
            )
        if state_failure:
            reasons.append(
                f"state RMS exceeded {self.config.state_abort_rms:g} for {patience} updates"
            )
        if reasons:
            reason = "; ".join(reasons)
            path = self._write_stability_abort(
                candidate_step=step,
                gradient_norm=gradient_norm,
                state_rms_max=state_rms_max,
                reason=reason,
            )
            reporter.alert(
                "Stability guard aborted update",
                f"{reason}. The unsafe update was not applied; diagnostics: {path.name}.",
                level="error", step=step,
            )
            raise FloatingPointError(f"language-model stability guard: {reason}")

    def train(self) -> LMTrainingState:
        if not self._resumed:
            torch.manual_seed(self.config.seed)
        run_identity = self._run_identity()
        reporter = TrackioReporter(
            self.config,
            run_identity,
            resume=self._resumed,
            initial_step=self.state.step,
            initial_tokens=self.state.tokens_seen,
        )
        wall_started = perf_counter()
        reporter.alert(
            "Training started" if not self._resumed else "Training resumed",
            f"Starting at step {self.state.step}, token {self.state.tokens_seen}.",
            level="info", step=self.state.step,
        )
        if self._resumed:
            # A resumed run should populate the new tab immediately rather than
            # waiting until the next periodic snapshot boundary.
            self._publish_spectral_snapshot(reporter)
        try:
            while self.state.tokens_seen < self.config.total_tokens:
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                remaining = self.config.total_tokens - self.state.tokens_seen
                microsteps = min(
                    self.config.gradient_accumulation_steps,
                    ceil(remaining / self.config.tokens_per_microbatch),
                )
                accumulator = _MetricAccumulator()
                architecture: dict[str, float] = {}
                stability: dict[str, float] = {}
                data_seconds = 0.0
                _synchronize(self.device)
                step_started = perf_counter()
                for microstep in range(microsteps):
                    remaining = self.config.total_tokens - self.state.tokens_seen - accumulator.token_count
                    final_length = min(
                        self.config.sequence_length,
                        max(1, ceil(remaining / self.config.micro_batch_size)),
                    )
                    data_started = perf_counter()
                    batch = self._move_batch(
                        self.train_stream.next_batch(
                            self.config.micro_batch_size, final_length
                        )
                    )
                    data_seconds += perf_counter() - data_started
                    with self._autocast():
                        output = self.model(batch.input_ids)
                        statistics = next_token_statistics(
                            output.logits, batch.labels, batch.target_byte_lengths,
                            batch.loss_mask,
                        )
                        spectral = spectral_activation_regularization(self._spectral_modules)
                        state_regularization = resonator_state_regularization(
                            output, target_rms=self.config.state_target_rms
                        )
                        total = (
                            statistics.cross_entropy
                            + self.config.spectral_regularization_weight * spectral
                            + self.config.state_regularization_weight * state_regularization
                        )
                    if not bool(torch.isfinite(total)):
                        reporter.alert(
                            "Non-finite loss", "Training loss became non-finite; update aborted.",
                            level="error", step=self.state.step,
                        )
                        raise FloatingPointError("language-model loss became non-finite")
                    scaled_loss = total / microsteps
                    if self.scaler is None:
                        scaled_loss.backward()
                    else:
                        self.scaler.scale(scaled_loss).backward()
                    accumulator.add(statistics, total, spectral, state_regularization)
                    if microstep == microsteps - 1:
                        stability = stability_metrics(output)
                    if (
                        (self.state.step + 1) % self.config.architecture_log_interval == 0
                        and microstep == microsteps - 1
                    ):
                        architecture = architecture_metrics(output)
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                gradient = clip_and_report_gradients(
                    self.model, maximum_norm=self.config.maximum_gradient_norm
                )
                if not gradient.finite:
                    reporter.alert(
                        "Non-finite gradients", "Gradient tensors became non-finite; update aborted.",
                        level="error", step=self.state.step,
                    )
                    raise FloatingPointError("language-model gradients became non-finite")
                self._enforce_stability(reporter, gradient, stability)
                before_parameters = None
                if (self.state.step + 1) % self.config.architecture_log_interval == 0:
                    before_parameters = torch.stack([
                        parameter.detach().float().square().sum()
                        for parameter in self.model.parameters() if parameter.requires_grad
                    ]).sum().sqrt()
                if self.scaler is None:
                    self.optimizer.step()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                self._step_scheduler()
                _synchronize(self.device)
                step_seconds = perf_counter() - step_started
                self.state.step += 1
                self.state.tokens_seen += accumulator.token_count
                self.state.bytes_seen += accumulator.byte_count
                metrics = accumulator.metrics("train")
                metrics.update(
                    {
                        "progress/step": float(self.state.step),
                        "progress/tokens_seen": float(self.state.tokens_seen),
                        "progress/utf8_bytes_seen": float(self.state.bytes_seen),
                        "progress/fraction": min(1.0, self.state.tokens_seen / self.config.total_tokens),
                        "progress/documents_packed": float(self.train_stream.documents_packed),
                        "performance/step_seconds": step_seconds,
                        "performance/data_wait_seconds": data_seconds,
                        "performance/tokens_per_second": accumulator.token_count / max(step_seconds, 1e-9),
                        "performance/utf8_bytes_per_second": accumulator.byte_count / max(step_seconds, 1e-9),
                        "optimization/learning_rate": max(group["lr"] for group in self.optimizer.param_groups),
                        "optimization/learning_rate_scale": self.state.learning_rate_scale,
                        "optimization/gradient_recoveries": float(self.state.gradient_recoveries),
                        "optimization/gradient_norm_before_clip": float(gradient.total_before_clip.cpu()),
                        "optimization/gradient_norm_after_clip": float(gradient.total_after_clip.cpu()),
                        "optimization/gradient_clip_coefficient": float(gradient.clip_coefficient.cpu()),
                        "optimization/phase_gradient_norm": float(gradient.phase_norm.cpu()),
                        "optimization/amplitude_gradient_norm": float(gradient.amplitude_norm.cpu()),
                    }
                )
                if before_parameters is not None:
                    after_parameters = torch.stack([
                        parameter.detach().float().square().sum()
                        for parameter in self.model.parameters() if parameter.requires_grad
                    ]).sum().sqrt()
                    metrics["optimization/parameter_norm"] = float(after_parameters.cpu())
                    metrics["optimization/relative_parameter_norm_change"] = float(
                        ((after_parameters - before_parameters).abs() / before_parameters.clamp_min(1e-12)).cpu()
                    )
                metrics.update(_memory_metrics(self.device))
                if self.scaler is not None:
                    metrics["optimization/gradient_scale"] = self.scaler.get_scale()
                metrics.update(stability)
                metrics.update(architecture)
                elapsed = self.state.elapsed_seconds + perf_counter() - wall_started
                metrics["progress/elapsed_seconds"] = elapsed
                metrics["progress/estimated_remaining_seconds"] = (
                    elapsed / max(self.state.tokens_seen, 1)
                    * max(0, self.config.total_tokens - self.state.tokens_seen)
                )
                if self.state.step % self.config.log_interval == 0:
                    reporter.log(metrics, step=self.state.step)
                    print(
                        f"step={self.state.step}/{self.config.total_steps} "
                        f"tokens={self.state.tokens_seen}/{self.config.total_tokens} "
                        f"loss={metrics['train/total_loss']:.4f} "
                        f"ce={metrics['train/cross_entropy_nats_per_token']:.4f} "
                        f"effective_ce={metrics['train/effective_cross_entropy_nats_per_byte']:.4f} "
                        f"bpb={metrics['train/bits_per_byte']:.4f} "
                        f"tok/s={metrics['performance/tokens_per_second']:.1f}",
                        flush=True,
                    )
                self._diagnostic_alerts(reporter, metrics)
                abort_pressure = max(
                    self.state.consecutive_gradient_aborts,
                    self.state.consecutive_state_aborts,
                )
                needs_safety_anchor = self._safety_checkpoint_pending or (
                    abort_pressure > 0
                    and abort_pressure == self.config.stability_patience - 1
                )
                if (
                    needs_safety_anchor
                    and self.state.step % self.config.checkpoint_interval != 0
                ):
                    self.state.elapsed_seconds += perf_counter() - wall_started
                    wall_started = perf_counter()
                    path = self.save_checkpoint()
                    reporter.alert(
                        "Safety checkpoint saved",
                        f"Saved {path.name} before the next stability decision.",
                        level="info", step=self.state.step,
                    )
                self._safety_checkpoint_pending = False
                if (
                    self.state.step == 1
                    or self.state.step % self.config.spectral_snapshot_interval == 0
                ):
                    self._publish_spectral_snapshot(reporter)
                if self.state.step % self.config.evaluation_interval == 0:
                    eval_metrics, eval_architecture = self.evaluate()
                    sample, generated_count = self._write_generation_sample()
                    eval_metrics["generation/generated_tokens"] = float(generated_count)
                    eval_metrics.update(eval_architecture)
                    reporter.log(eval_metrics, step=self.state.step)
                    effective = eval_metrics["eval/effective_cross_entropy_nats_per_byte"]
                    improved = effective < self.state.best_effective_cross_entropy
                    if improved:
                        self.state.best_effective_cross_entropy = effective
                        path = self.save_checkpoint(best=True)
                        reporter.alert(
                            "New best evaluation",
                            f"Effective CE improved to {effective:.5f}; saved {path.name}. Sample: {sample[:120]!r}",
                            level="info", step=self.state.step,
                        )
                if self.state.step % self.config.checkpoint_interval == 0:
                    self.state.elapsed_seconds += perf_counter() - wall_started
                    wall_started = perf_counter()
                    path = self.save_checkpoint()
                    reporter.alert(
                        "Checkpoint saved", f"Saved resumable checkpoint {path.name}.",
                        level="info", step=self.state.step,
                    )
            self.state.elapsed_seconds += perf_counter() - wall_started
            final_eval, final_architecture = self.evaluate()
            final_eval.update(final_architecture)
            reporter.log(final_eval, step=self.state.step)
            self.save_checkpoint()
            self._publish_spectral_snapshot(reporter)
            reporter.alert(
                "Training complete",
                f"Processed {self.state.tokens_seen} tokens in {self.state.elapsed_seconds:.1f} seconds.",
                level="info", step=self.state.step,
            )
            return self.state
        except Exception as error:
            try:
                reporter.alert(
                    "Training aborted", f"{type(error).__name__}: {error}",
                    level="error", step=self.state.step,
                )
            except Exception:
                pass
            raise
        finally:
            # finish() is idempotent in Trackio; this also covers normal exit.
            reporter.finish()


def build_evaluation_batches(
    stream: PackedTokenStream,
    *,
    count: int,
    batch_size: int,
    sequence_length: int,
) -> tuple[PackedBatch, ...]:
    if min(count, batch_size, sequence_length) <= 0:
        raise ValueError("evaluation batch controls must be positive")
    return tuple(stream.next_batch(batch_size, sequence_length) for _ in range(count))
