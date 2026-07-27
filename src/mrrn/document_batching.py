"""Deterministic document-major static batching for integrated MRCRA training.

The ordinary language packer deliberately fills a fixed token context with
independent documents and marks synthetic cross-document transitions invalid.
The integrated cognitive path must additionally reset recurrent and cognitive
state at every such boundary.  Executing those documents one after another is
semantically simple but creates dozens of small carrier/autograd graphs.

This module converts one :class:`~mrrn.lm_training.PackedBatch` into stable
document cohorts.  Documents in a cohort retain their row for every TBPTT span,
so recurrent state, cognitive state, and provenance never need to be remapped
mid-document.  Every emitted tensor has a static bucket length and explicit
token, loss, and event masks.  A target-bijection receipt proves that batching
neither drops nor duplicates any authoritative next-token target.

The planner is intentionally free of model execution and randomness.  Given
the same packed batch and configuration it produces the same cohorts, tensor
contents, and receipts on every platform.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import gcd, isfinite

import torch
from torch import Tensor

from .document_cost_model import (
    DocumentExecutionCostModel,
    DocumentPlanCostReceipt,
)
from .lm_training import PackedBatch


def _least_common_multiple(left: int, right: int) -> int:
    if min(left, right) <= 0:
        raise ValueError("static batching alignment values must be positive")
    return left * right // gcd(left, right)


def _tensor_digest(values: tuple[Tensor, ...]) -> str:
    digest = sha256()
    for value in values:
        contiguous = value.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DocumentSpan:
    """One document-local TBPTT span before static padding."""

    sequence_id: int
    context_row: int
    segment_id: int
    source_uri: str
    document_order: int
    span_index: int
    context_start: int
    context_end: int
    reset_state: bool
    final_span: bool
    input_ids: Tensor
    labels: Tensor
    target_byte_lengths: Tensor
    target_segment_ids: Tensor
    loss_mask: Tensor
    boundary_classes: Tensor
    input_ordinals: Tensor
    target_ordinals: Tensor

    def __post_init__(self) -> None:
        length = self.context_end - self.context_start
        tensors = (
            self.input_ids,
            self.labels,
            self.target_byte_lengths,
            self.target_segment_ids,
            self.loss_mask,
            self.boundary_classes,
            self.input_ordinals,
            self.target_ordinals,
        )
        if (
            min(
                self.sequence_id,
                self.context_row,
                self.segment_id,
                self.document_order,
                self.span_index,
                self.context_start,
            )
            < 0
            or self.context_end <= self.context_start
            or not self.source_uri
        ):
            raise ValueError("document span identity and bounds must be valid")
        if any(value.ndim != 1 or value.shape[0] != length for value in tensors):
            raise ValueError("document span tensors must be aligned one-dimensional values")
        if (
            self.input_ids.dtype != torch.int64
            or self.labels.dtype != torch.int64
            or self.target_byte_lengths.dtype != torch.int64
            or self.target_segment_ids.dtype != torch.int64
            or self.boundary_classes.dtype != torch.int64
            or self.input_ordinals.dtype != torch.int64
            or self.target_ordinals.dtype != torch.int64
            or self.loss_mask.dtype != torch.bool
        ):
            raise ValueError("document span tensors have invalid authority dtypes")
        if bool((self.target_byte_lengths < 0).any()):
            raise ValueError("document target byte lengths cannot be negative")
        if bool((self.target_ordinals[self.loss_mask] < 0).any()):
            raise ValueError("valid targets require nonnegative source ordinals")
        if bool((self.target_ordinals[~self.loss_mask] != -1).any()):
            raise ValueError("invalid targets must use the -1 receipt sentinel")
        if self.reset_state != (self.span_index == 0):
            raise ValueError("only the first document span may reset state")

    @property
    def length(self) -> int:
        return self.context_end - self.context_start

    @property
    def valid_targets(self) -> int:
        return int(self.loss_mask.sum())


@dataclass(frozen=True, slots=True)
class DocumentSequence:
    """Every TBPTT span belonging to one contiguous packed document."""

    sequence_id: int
    context_row: int
    segment_id: int
    source_uri: str
    document_order: int
    spans: tuple[DocumentSpan, ...]

    def __post_init__(self) -> None:
        if not self.spans:
            raise ValueError("a document sequence requires at least one trainable span")
        if any(
            span.sequence_id != self.sequence_id
            or span.context_row != self.context_row
            or span.segment_id != self.segment_id
            or span.source_uri != self.source_uri
            or span.document_order != self.document_order
            or span.span_index != index
            or span.reset_state != (index == 0)
            or span.final_span != (index == len(self.spans) - 1)
            for index, span in enumerate(self.spans)
        ):
            raise ValueError("document sequence spans have inconsistent identity or order")
        if any(
            right.context_start != left.context_end
            for left, right in zip(self.spans, self.spans[1:], strict=False)
        ):
            raise ValueError("document sequence spans must be contiguous")

    @property
    def valid_targets(self) -> int:
        return sum(span.valid_targets for span in self.spans)


@dataclass(frozen=True, slots=True)
class StaticDocumentSpanBatch:
    """One static-shaped physical invocation for a stable document cohort."""

    span_index: int
    padded_length: int
    sequence_ids: Tensor
    document_orders: Tensor
    segment_ids: Tensor
    source_uris: tuple[str, ...]
    context_starts: Tensor
    valid_lengths: Tensor
    input_ids: Tensor
    labels: Tensor
    target_byte_lengths: Tensor
    target_segment_ids: Tensor
    token_mask: Tensor
    loss_mask: Tensor
    event_mask: Tensor
    boundary_classes: Tensor
    input_ordinals: Tensor
    target_ordinals: Tensor
    reset_state: bool
    final_rows: Tensor

    def __post_init__(self) -> None:
        batch = self.input_ids.shape[0]
        shape = (batch, self.padded_length)
        if self.span_index < 0 or self.padded_length <= 0 or batch <= 0:
            raise ValueError("static document span dimensions must be positive")
        if any(
            value.shape != (batch,)
            for value in (
                self.sequence_ids,
                self.document_orders,
                self.segment_ids,
                self.context_starts,
                self.valid_lengths,
                self.final_rows,
            )
        ):
            raise ValueError("static document metadata must have one value per row")
        if len(self.source_uris) != batch or any(not value for value in self.source_uris):
            raise ValueError("static document sources must have one URI per row")
        if any(
            value.shape != shape
            for value in (
                self.input_ids,
                self.labels,
                self.target_byte_lengths,
                self.target_segment_ids,
                self.token_mask,
                self.loss_mask,
                self.boundary_classes,
                self.input_ordinals,
                self.target_ordinals,
            )
        ):
            raise ValueError("static document token tensors have inconsistent shapes")
        if self.event_mask.ndim != 2 or self.event_mask.shape[0] != batch:
            raise ValueError("event mask must have batch/event shape")
        if any(
            value.dtype != torch.int64
            for value in (
                self.sequence_ids,
                self.document_orders,
                self.segment_ids,
                self.context_starts,
                self.valid_lengths,
                self.input_ids,
                self.labels,
                self.target_byte_lengths,
                self.target_segment_ids,
                self.boundary_classes,
                self.input_ordinals,
                self.target_ordinals,
            )
        ) or any(
            value.dtype != torch.bool
            for value in (
                self.token_mask,
                self.loss_mask,
                self.event_mask,
                self.final_rows,
            )
        ):
            raise ValueError("static document batch tensors have invalid authority dtypes")
        if bool((self.valid_lengths <= 0).any()) or bool(
            (self.valid_lengths > self.padded_length).any()
        ):
            raise ValueError("valid lengths must lie inside the static bucket")
        expected_tokens = (
            torch.arange(self.padded_length, device=self.valid_lengths.device)[None]
            < self.valid_lengths[:, None]
        )
        if not torch.equal(self.token_mask, expected_tokens):
            raise ValueError("token mask must name exactly every unpadded token")
        if bool((self.loss_mask & ~self.token_mask).any()):
            raise ValueError("loss mask cannot authorize padded tokens")
        if bool((self.target_ordinals[self.loss_mask] < 0).any()) or bool(
            (self.target_ordinals[~self.loss_mask] != -1).any()
        ):
            raise ValueError("static target receipts do not agree with the loss mask")
        if self.reset_state != (self.span_index == 0):
            raise ValueError("only the first static cohort span may reset state")

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]

    @property
    def valid_tokens(self) -> int:
        return int(self.token_mask.sum())

    @property
    def valid_targets(self) -> int:
        return int(self.loss_mask.sum())

    @property
    def padding_efficiency(self) -> float:
        return self.valid_tokens / self.token_mask.numel()

    @property
    def digest(self) -> str:
        return _tensor_digest(
            (
                self.sequence_ids,
                self.document_orders,
                self.segment_ids,
                self.context_starts,
                self.valid_lengths,
                self.input_ids,
                self.labels,
                self.target_byte_lengths,
                self.target_segment_ids,
                self.token_mask,
                self.loss_mask,
                self.event_mask,
                self.boundary_classes,
                self.input_ordinals,
                self.target_ordinals,
                self.final_rows,
            )
        )


@dataclass(frozen=True, slots=True)
class StaticDocumentCohort:
    """Stable rows with one static physical batch for each TBPTT span."""

    cohort_id: int
    sequences: tuple[DocumentSequence, ...]
    padded_lengths: tuple[int, ...]
    spans: tuple[StaticDocumentSpanBatch, ...]

    def __post_init__(self) -> None:
        if self.cohort_id < 0 or not self.sequences or not self.spans:
            raise ValueError("static document cohort cannot be empty")
        if len(self.padded_lengths) != len(self.spans):
            raise ValueError("cohort bucket signature must name every span")
        sequence_ids = tuple(sequence.sequence_id for sequence in self.sequences)
        if len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError("cohort rows require unique document sequences")
        for index, span in enumerate(self.spans):
            if (
                span.span_index != index
                or span.padded_length != self.padded_lengths[index]
                or span.batch_size != len(self.sequences)
                or tuple(span.sequence_ids.tolist()) != sequence_ids
            ):
                raise ValueError("cohort physical spans do not preserve stable rows")

    @property
    def valid_targets(self) -> int:
        return sum(span.valid_targets for span in self.spans)

    @property
    def physical_tokens(self) -> int:
        return sum(span.token_mask.numel() for span in self.spans)

    @property
    def valid_tokens(self) -> int:
        return sum(span.valid_tokens for span in self.spans)

    def target_authority(self) -> "DocumentTargetAuthority":
        """Materialize complete document-local targets for cross-span CSTM."""

        lengths = tuple(
            sum(span.length for span in sequence.spans)
            for sequence in self.sequences
        )
        # The carrier advances through every static position, including the
        # final right-padding of shorter rows.  CSTM source positions are
        # therefore expressed in this shared physical timeline.  Retain that
        # full timeline here and leave padded positions unauthorized rather
        # than making otherwise valid shared source indices fall outside a
        # shorter row's target tensor.
        maximum = sum(self.padded_lengths)
        device = self.sequences[0].spans[0].labels.device
        labels = torch.zeros(
            len(self.sequences), maximum, dtype=torch.int64, device=device
        )
        byte_lengths = torch.zeros_like(labels)
        segment_ids = torch.full_like(labels, -1)
        target_segment_ids = torch.full_like(labels, -1)
        loss_mask = torch.zeros_like(labels, dtype=torch.bool)
        for row, sequence in enumerate(self.sequences):
            cursor = 0
            for span in sequence.spans:
                end = cursor + span.length
                labels[row, cursor:end] = span.labels
                byte_lengths[row, cursor:end] = span.target_byte_lengths
                segment_ids[row, cursor:end] = sequence.segment_id
                target_segment_ids[row, cursor:end] = span.target_segment_ids
                loss_mask[row, cursor:end] = span.loss_mask
                cursor = end
            if cursor != lengths[row]:
                raise RuntimeError("document authority materialization lost a span")
        return DocumentTargetAuthority(
            labels,
            byte_lengths,
            segment_ids,
            target_segment_ids,
            loss_mask,
            torch.tensor(lengths, dtype=torch.int64, device=device),
        )


@dataclass(frozen=True, slots=True)
class DocumentTargetAuthority:
    """Complete row-local target tensors retained across TBPTT spans."""

    labels: Tensor
    target_byte_lengths: Tensor
    segment_ids: Tensor
    target_segment_ids: Tensor
    loss_mask: Tensor
    valid_lengths: Tensor

    def __post_init__(self) -> None:
        if (
            self.labels.ndim != 2
            or self.labels.shape[0] <= 0
            or any(
                value.shape != self.labels.shape
                for value in (
                    self.target_byte_lengths,
                    self.segment_ids,
                    self.target_segment_ids,
                    self.loss_mask,
                )
            )
            or self.valid_lengths.shape != self.labels.shape[:1]
        ):
            raise ValueError("document target authority tensors are misaligned")
        if any(
            value.dtype != torch.int64
            for value in (
                self.labels,
                self.target_byte_lengths,
                self.segment_ids,
                self.target_segment_ids,
                self.valid_lengths,
            )
        ) or self.loss_mask.dtype != torch.bool:
            raise ValueError("document target authority has invalid dtypes")
        if bool((self.valid_lengths <= 0).any()) or bool(
            (self.valid_lengths > self.labels.shape[1]).any()
        ):
            raise ValueError("document target authority lengths are invalid")


@dataclass(frozen=True, slots=True)
class TargetBijectionReceipt:
    """Exact proof that planned loss rows equal the packed target authority."""

    original_valid_ordinals: tuple[int, ...]
    planned_valid_ordinals: tuple[int, ...]
    missing_ordinals: tuple[int, ...]
    unexpected_ordinals: tuple[int, ...]
    duplicate_ordinals: tuple[int, ...]
    original_digest: str
    planned_digest: str

    @property
    def passed(self) -> bool:
        return not (
            self.missing_ordinals
            or self.unexpected_ordinals
            or self.duplicate_ordinals
        ) and self.original_valid_ordinals == self.planned_valid_ordinals


@dataclass(frozen=True, slots=True)
class DocumentBatchPlan:
    """Complete deterministic plan for one packed optimization context."""

    cohorts: tuple[StaticDocumentCohort, ...]
    sequences: tuple[DocumentSequence, ...]
    receipt: TargetBijectionReceipt
    bucket_lengths: tuple[int, ...]
    token_budget: int
    alignment: int
    cognitive_stride: int
    original_token_count: int
    original_valid_targets: int
    cost_receipt: DocumentPlanCostReceipt

    def __post_init__(self) -> None:
        if (
            not self.cohorts
            or not self.sequences
            or not self.receipt.passed
            or min(
                self.token_budget,
                self.alignment,
                self.cognitive_stride,
                self.original_token_count,
                self.original_valid_targets,
            )
            <= 0
            or self.cost_receipt.selected_invocations
            != self.physical_invocations
        ):
            raise ValueError("document batch plan must be nonempty and bijective")
        planned_sequence_ids = tuple(
            sequence.sequence_id
            for cohort in self.cohorts
            for sequence in cohort.sequences
        )
        if sorted(planned_sequence_ids) != sorted(
            sequence.sequence_id for sequence in self.sequences
        ):
            raise ValueError("document cohorts must cover every sequence exactly once")
        if self.planned_valid_targets != self.original_valid_targets:
            raise ValueError("document plan changed the authoritative target count")

    @property
    def planned_valid_targets(self) -> int:
        return sum(cohort.valid_targets for cohort in self.cohorts)

    @property
    def physical_tokens(self) -> int:
        return sum(cohort.physical_tokens for cohort in self.cohorts)

    @property
    def valid_document_tokens(self) -> int:
        return sum(cohort.valid_tokens for cohort in self.cohorts)

    @property
    def padding_efficiency(self) -> float:
        return self.valid_document_tokens / self.physical_tokens

    @property
    def physical_invocations(self) -> int:
        return sum(len(cohort.spans) for cohort in self.cohorts)


class DocumentMajorBatchPlanner:
    """Create stable, static-shaped document cohorts without changing targets."""

    def __init__(
        self,
        *,
        tbptt_length: int,
        bucket_lengths: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096),
        token_budget: int = 8192,
        alignment: int = 16,
        cognitive_stride: int = 128,
        padding_token_id: int = 0,
        grouping_policy: str = "cost_aware",
        activation_policy: str = "retain",
        cost_model: DocumentExecutionCostModel | None = None,
        plan_cache_capacity: int = 128,
        maximum_candidate_activation_bytes: int | None = None,
        activation_bytes_per_token: int = 0,
        device_torch_fingerprint: str = "portable",
        actor_configuration_digest: str = "portable",
        compiler_policy: str = "off",
        activation_policy_token_limits: dict[str, int] | None = None,
        activation_policy_timings: dict[str, float] | None = None,
    ) -> None:
        if min(tbptt_length, token_budget, alignment, cognitive_stride) <= 0:
            raise ValueError("document planner sizes must be positive")
        if padding_token_id < 0:
            raise ValueError("padding token ID cannot be negative")
        if grouping_policy not in {"cost_aware", "exact_signature"}:
            raise ValueError("document grouping policy is unknown")
        if activation_policy not in {"retain", "selective", "whole_span"}:
            raise ValueError("document activation policy is unknown")
        if plan_cache_capacity < 0:
            raise ValueError("document plan cache capacity cannot be negative")
        if (
            maximum_candidate_activation_bytes is not None
            and maximum_candidate_activation_bytes <= 0
        ) or activation_bytes_per_token < 0:
            raise ValueError("document activation-memory budget is invalid")
        if (
            maximum_candidate_activation_bytes is not None
            and activation_bytes_per_token <= 0
        ):
            raise ValueError(
                "an activation-memory budget requires a positive per-token estimate"
            )
        if (
            not device_torch_fingerprint
            or not actor_configuration_digest
            or compiler_policy not in {"off", "on", "auto"}
        ):
            raise ValueError("document plan cache identity is malformed")
        if (
            not bucket_lengths
            or tuple(sorted(set(bucket_lengths))) != bucket_lengths
            or any(length <= 0 for length in bucket_lengths)
        ):
            raise ValueError("document buckets must be unique increasing positive lengths")
        # Static carrier shapes must align with the multiresolution analysis
        # support. They need not be an integer number of cognitive strides:
        # the final partial cognitive interval is already represented by the
        # exact event mask. Requiring the stride here adds up to one full
        # cognition interval of useless right padding per document.
        if any(length % alignment for length in bucket_lengths):
            raise ValueError(
                "every static bucket must align with the carrier support"
            )
        policy_names = {"retain", "selective", "whole_span"}
        if (
            any(
                name not in policy_names
                or not isinstance(limit, int)
                or limit < 0
                for name, limit in (
                    activation_policy_token_limits or {}
                ).items()
            )
            or any(
                name not in policy_names
                or not isinstance(seconds, (int, float))
                or not isfinite(float(seconds))
                or seconds < 0
                for name, seconds in (
                    activation_policy_timings or {}
                ).items()
            )
        ):
            raise ValueError(
                "document activation-policy calibration is malformed"
            )
        if tbptt_length > bucket_lengths[-1]:
            raise ValueError("largest static bucket must cover the TBPTT length")
        if token_budget < bucket_lengths[0]:
            raise ValueError("document token budget must fit the smallest bucket")
        self.tbptt_length = tbptt_length
        self.bucket_lengths = bucket_lengths
        self.token_budget = token_budget
        self.alignment = alignment
        self.cognitive_stride = cognitive_stride
        self.padding_token_id = padding_token_id
        self.grouping_policy = grouping_policy
        self.activation_policy = activation_policy
        self.cost_model = cost_model or DocumentExecutionCostModel()
        self.plan_cache_capacity = plan_cache_capacity
        self.maximum_candidate_activation_bytes = (
            maximum_candidate_activation_bytes
        )
        self.activation_bytes_per_token = activation_bytes_per_token
        self.device_torch_fingerprint = device_torch_fingerprint
        self.actor_configuration_digest = actor_configuration_digest
        self.compiler_policy = compiler_policy
        self.activation_policy_token_limits = dict(
            activation_policy_token_limits or {}
        )
        self.activation_policy_timings = dict(
            activation_policy_timings or {}
        )
        self._last_rejected_memory_candidates = 0
        self._group_cache: dict[
            tuple[object, ...], tuple[tuple[int, ...], ...]
        ] = {}

    def _bucket(self, length: int) -> int:
        for candidate in self.bucket_lengths:
            if length <= candidate:
                return candidate
        raise ValueError(
            f"document span length {length} exceeds maximum bucket "
            f"{self.bucket_lengths[-1]}"
        )

    def _effective_tbptt_length(self) -> int:
        """Return the largest single-row span authorized by every hard limit.

        ``tbptt_length`` is a maximum truncation interval, not permission to
        violate the physical token or measured activation-memory budget.  A
        long document can always be divided at an earlier recurrent-state
        handoff without changing token, provenance, or state-continuation
        authority.  Deriving that handoff from an existing static bucket also
        guarantees that every extracted single-row span has a feasible
        physical representation before cohort optimization begins.
        """

        authorized_buckets = tuple(
            length
            for length in self.bucket_lengths
            if (
                length <= self.token_budget
                and (
                    self.maximum_candidate_activation_bytes is None
                    or (
                        length * self.activation_bytes_per_token
                        <= self.maximum_candidate_activation_bytes
                    )
                )
            )
        )
        if not authorized_buckets:
            smallest = self.bucket_lengths[0]
            estimated_bytes = (
                smallest * self.activation_bytes_per_token
                if self.maximum_candidate_activation_bytes is not None
                else 0
            )
            raise ValueError(
                "no single-row document span fits the configured hard "
                "constraints: "
                f"smallest_bucket={smallest}, "
                f"token_budget={self.token_budget}, "
                "estimated_activation_bytes="
                f"{estimated_bytes}, "
                "maximum_candidate_activation_bytes="
                f"{self.maximum_candidate_activation_bytes}"
            )
        return min(self.tbptt_length, authorized_buckets[-1])

    def _extract_sequences(self, batch: PackedBatch) -> tuple[DocumentSequence, ...]:
        context_length = batch.input_ids.shape[1]
        boundaries = batch.boundary_classes
        loss_mask = batch.loss_mask
        effective_tbptt_length = self._effective_tbptt_length()
        result: list[DocumentSequence] = []
        sequence_id = 0
        document_order = 0
        for row in range(batch.input_ids.shape[0]):
            segments = batch.segment_ids[row]
            declarations = batch.external_source_uris[row]
            seen: set[int] = set()
            start = 0
            while start < context_length:
                segment_id = int(segments[start])
                end = start + 1
                while end < context_length and int(segments[end]) == segment_id:
                    end += 1
                if segment_id in seen:
                    raise ValueError(
                        "packed input contains a noncontiguous repeated segment ID"
                    )
                seen.add(segment_id)
                if segment_id not in declarations:
                    raise ValueError("packed document is missing its source declaration")
                spans: list[DocumentSpan] = []
                local_start = start
                span_index = 0
                while local_start < end:
                    local_end = min(
                        end, local_start + effective_tbptt_length
                    )
                    local_loss = loss_mask[row, local_start:local_end].clone()
                    if bool(local_loss.any()):
                        ordinals = torch.arange(
                            row * context_length + local_start,
                            row * context_length + local_end,
                            dtype=torch.int64,
                            device=batch.input_ids.device,
                        )
                        spans.append(
                            DocumentSpan(
                                sequence_id=sequence_id,
                                context_row=row,
                                segment_id=segment_id,
                                source_uri=declarations[segment_id],
                                document_order=document_order,
                                span_index=span_index,
                                context_start=local_start,
                                context_end=local_end,
                                reset_state=span_index == 0,
                                final_span=local_end == end,
                                input_ids=batch.input_ids[
                                    row, local_start:local_end
                                ].clone(),
                                labels=batch.labels[
                                    row, local_start:local_end
                                ].clone(),
                                target_byte_lengths=batch.target_byte_lengths[
                                    row, local_start:local_end
                                ].clone(),
                                target_segment_ids=batch.target_segment_ids[
                                    row, local_start:local_end
                                ].clone(),
                                loss_mask=local_loss,
                                boundary_classes=boundaries[
                                    row, local_start:local_end
                                ].clone(),
                                input_ordinals=ordinals,
                                target_ordinals=torch.where(
                                    local_loss,
                                    ordinals,
                                    torch.full_like(ordinals, -1),
                                ),
                            )
                        )
                        span_index += 1
                    local_start = local_end
                if spans:
                    # A terminal span can be shorter than the actual segment
                    # only when a completely untrainable tail followed its last
                    # valid target. Preserve the span's own terminal authority.
                    if not spans[-1].final_span:
                        spans[-1] = DocumentSpan(
                            **{
                                field: getattr(spans[-1], field)
                                for field in DocumentSpan.__dataclass_fields__
                                if field != "final_span"
                            },
                            final_span=True,
                        )
                    result.append(
                        DocumentSequence(
                            sequence_id,
                            row,
                            segment_id,
                            declarations[segment_id],
                            document_order,
                            tuple(spans),
                        )
                    )
                    sequence_id += 1
                    document_order += 1
                start = end
        if not result:
            raise ValueError("packed context contains no trainable documents")
        return tuple(result)

    def _materialize_span(
        self,
        sequences: tuple[DocumentSequence, ...],
        *,
        span_index: int,
        padded_length: int,
    ) -> StaticDocumentSpanBatch:
        batch_size = len(sequences)
        device = sequences[0].spans[span_index].input_ids.device
        input_ids = torch.full(
            (batch_size, padded_length),
            self.padding_token_id,
            dtype=torch.int64,
            device=device,
        )
        labels = torch.zeros_like(input_ids)
        byte_lengths = torch.zeros_like(input_ids)
        target_segment_ids = torch.full_like(input_ids, -1)
        token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        loss_mask = torch.zeros_like(token_mask)
        boundary_classes = torch.zeros_like(input_ids)
        input_ordinals = torch.full_like(input_ids, -1)
        target_ordinals = torch.full_like(input_ids, -1)
        valid_lengths: list[int] = []
        context_starts: list[int] = []
        final_rows: list[bool] = []
        for row, sequence in enumerate(sequences):
            span = sequence.spans[span_index]
            length = span.length
            if length > padded_length:
                raise RuntimeError("document span exceeds its selected static bucket")
            input_ids[row, :length] = span.input_ids
            labels[row, :length] = span.labels
            byte_lengths[row, :length] = span.target_byte_lengths
            target_segment_ids[row, :length] = span.target_segment_ids
            token_mask[row, :length] = True
            loss_mask[row, :length] = span.loss_mask
            boundary_classes[row, :length] = span.boundary_classes
            input_ordinals[row, :length] = span.input_ordinals
            target_ordinals[row, :length] = span.target_ordinals
            valid_lengths.append(length)
            context_starts.append(span.context_start)
            final_rows.append(span.final_span)
        anchor_positions = torch.arange(
            0,
            padded_length,
            self.cognitive_stride,
            dtype=torch.int64,
            device=device,
        )
        valid_length_tensor = torch.tensor(
            valid_lengths, dtype=torch.int64, device=device
        )
        return StaticDocumentSpanBatch(
            span_index=span_index,
            padded_length=padded_length,
            sequence_ids=torch.tensor(
                [sequence.sequence_id for sequence in sequences],
                dtype=torch.int64,
                device=device,
            ),
            document_orders=torch.tensor(
                [sequence.document_order for sequence in sequences],
                dtype=torch.int64,
                device=device,
            ),
            segment_ids=torch.tensor(
                [sequence.segment_id for sequence in sequences],
                dtype=torch.int64,
                device=device,
            ),
            source_uris=tuple(sequence.source_uri for sequence in sequences),
            context_starts=torch.tensor(
                context_starts, dtype=torch.int64, device=device
            ),
            valid_lengths=valid_length_tensor,
            input_ids=input_ids,
            labels=labels,
            target_byte_lengths=byte_lengths,
            target_segment_ids=target_segment_ids,
            token_mask=token_mask,
            loss_mask=loss_mask,
            event_mask=anchor_positions[None] < valid_length_tensor[:, None],
            boundary_classes=boundary_classes,
            input_ordinals=input_ordinals,
            target_ordinals=target_ordinals,
            reset_state=span_index == 0,
            final_rows=torch.tensor(
                final_rows, dtype=torch.bool, device=device
            ),
        )

    def _candidate_signature(
        self, sequences: tuple[DocumentSequence, ...],
    ) -> tuple[int, ...]:
        span_count = len(sequences[0].spans)
        if any(len(sequence.spans) != span_count for sequence in sequences):
            raise ValueError("candidate rows must have the same TBPTT span count")
        return tuple(
            self._bucket(
                max(sequence.spans[index].length for sequence in sequences)
            )
            for index in range(span_count)
        )

    def _candidate_cost(
        self,
        sequences: tuple[DocumentSequence, ...],
        signature: tuple[int, ...],
        *,
        known_shapes: frozenset[tuple[int, int]] = frozenset(),
    ) -> float:
        valid_lengths = tuple(
            tuple(span.length for span in sequence.spans)
            for sequence in sequences
        )
        if not self.activation_policy_token_limits:
            return self.cost_model.estimate(
                padded_lengths=signature,
                valid_lengths_by_row=valid_lengths,
                cognitive_stride=self.cognitive_stride,
                activation_policy=self.activation_policy,
                known_shapes=known_shapes,
                compiler_enabled=self.compiler_policy == "on",
            )

        # Price each physical span with the same shape-conditional activation
        # policy that execution will use. Treating every span as the
        # maximum-shape fallback systematically overprices safe selective or
        # retained cohorts and can make the dynamic planner choose a slower
        # grouping despite an otherwise accurate measured cost model.
        total = 0.0
        observed_shapes = known_shapes
        for index, padded_length in enumerate(signature):
            physical_tokens = len(sequences) * padded_length
            activation_policy = self._activation_policy_for_physical_tokens(
                physical_tokens
            )
            total += self.cost_model.estimate(
                padded_lengths=(padded_length,),
                valid_lengths_by_row=tuple(
                    (row[index],) for row in valid_lengths
                ),
                cognitive_stride=self.cognitive_stride,
                activation_policy=activation_policy,
                known_shapes=observed_shapes,
                compiler_enabled=self.compiler_policy == "on",
            )
            observed_shapes = observed_shapes.union(
                ((len(sequences), padded_length),)
            )
        return total

    def _activation_policy_for_physical_tokens(
        self, physical_tokens: int,
    ) -> str:
        if physical_tokens <= 0:
            raise ValueError("document activation shape must be positive")
        feasible = tuple(
            name
            for name in ("retain", "selective", "whole_span")
            if (
                self.activation_policy_token_limits.get(name, 0)
                >= physical_tokens
            )
        )
        if not feasible:
            return self.activation_policy
        return min(
            feasible,
            key=lambda name: (
                self.activation_policy_timings.get(name, float("inf")),
                ("retain", "selective", "whole_span").index(name),
            ),
        )

    def _candidate_fits_memory(
        self,
        sequences: tuple[DocumentSequence, ...],
        signature: tuple[int, ...],
    ) -> bool:
        if self.maximum_candidate_activation_bytes is None:
            return True
        # TBPTT releases each physical span before the next. The live
        # activation authority is therefore the largest one-span cohort, not
        # the sum of all spans in the document.
        estimated = (
            len(sequences)
            * max(signature)
            * self.activation_bytes_per_token
        )
        return estimated <= self.maximum_candidate_activation_bytes

    def _validate_single_sequence_feasibility(
        self,
        sequences: tuple[DocumentSequence, ...],
    ) -> None:
        """Fail with the violated authority, not a misleading generic error."""

        for sequence in sequences:
            signature = self._candidate_signature((sequence,))
            excessive_lengths = tuple(
                length
                for length in signature
                if length > self.token_budget
            )
            if excessive_lengths:
                raise ValueError(
                    "single-row document span exceeds the physical token "
                    "budget after extraction: "
                    f"document_order={sequence.document_order}, "
                    f"signature={signature}, "
                    f"token_budget={self.token_budget}"
                )
            if not self._candidate_fits_memory((sequence,), signature):
                estimated = (
                    max(signature) * self.activation_bytes_per_token
                )
                raise ValueError(
                    "single-row document span exceeds the measured "
                    "activation-memory budget after extraction: "
                    f"document_order={sequence.document_order}, "
                    f"signature={signature}, "
                    f"estimated_activation_bytes={estimated}, "
                    "maximum_candidate_activation_bytes="
                    f"{self.maximum_candidate_activation_bytes}"
                )

    def _exact_groups(
        self, sequences: tuple[DocumentSequence, ...],
    ) -> tuple[tuple[DocumentSequence, ...], ...]:
        by_signature: dict[tuple[int, ...], list[DocumentSequence]] = {}
        for sequence in sequences:
            signature = tuple(self._bucket(span.length) for span in sequence.spans)
            by_signature.setdefault(signature, []).append(sequence)
        result: list[tuple[DocumentSequence, ...]] = []
        for signature in sorted(
            by_signature,
            key=lambda item: (
                item,
                by_signature[item][0].document_order,
            ),
        ):
            candidates = sorted(
                by_signature[signature], key=lambda item: item.document_order
            )
            capacity = max(1, self.token_budget // max(signature))
            if self.maximum_candidate_activation_bytes is not None:
                memory_capacity = (
                    self.maximum_candidate_activation_bytes
                    // (
                        max(signature)
                        * self.activation_bytes_per_token
                    )
                )
                if memory_capacity <= 0:
                    raise ValueError(
                        "one document span exceeds the activation-memory budget"
                    )
                capacity = min(capacity, memory_capacity)
            for start in range(0, len(candidates), capacity):
                result.append(tuple(candidates[start : start + capacity]))
        return tuple(result)

    def _cost_groups_for_span_count(
        self, sequences: tuple[DocumentSequence, ...],
    ) -> tuple[tuple[DocumentSequence, ...], ...]:
        groups, _ = self._cost_groups_for_span_count_with_shapes(
            sequences,
            initial_known_shapes=frozenset(),
        )
        return groups

    def _cost_groups_for_span_count_with_shapes(
        self,
        sequences: tuple[DocumentSequence, ...],
        *,
        initial_known_shapes: frozenset[tuple[int, int]],
    ) -> tuple[
        tuple[tuple[DocumentSequence, ...], ...],
        frozenset[tuple[int, int]],
    ]:
        ordered = tuple(sorted(
            sequences,
            key=lambda item: (
                tuple(span.length for span in item.spans),
                item.document_order,
            ),
        ))
        self._validate_single_sequence_feasibility(ordered)
        count = len(ordered)
        if self.compiler_policy == "on" and self.cost_model.shape_compile_cost:
            # Exact dynamic program over both the prefix boundary and the set
            # of already-compiled static shapes. No state pruning is allowed:
            # pruning could silently exchange the declared global compile-cost
            # objective for a heuristic local one.
            dynamic_shapes: list[
                dict[
                    frozenset[tuple[int, int]],
                    tuple[float, int, tuple[int, ...]],
                ]
            ] = [dict() for _ in range(count + 1)]
            dynamic_shapes[0][initial_known_shapes] = (0.0, 0, ())
            for end in range(1, count + 1):
                for start in range(end - 1, -1, -1):
                    selected = ordered[start:end]
                    signature = self._candidate_signature(selected)
                    if any(
                        len(selected) * length > self.token_budget
                        for length in signature
                    ):
                        break
                    if not self._candidate_fits_memory(selected, signature):
                        self._last_rejected_memory_candidates += 1
                        continue
                    candidate_shapes = frozenset(
                        (len(selected), length) for length in signature
                    )
                    for known, prior in dynamic_shapes[start].items():
                        resulting = known | candidate_shapes
                        candidate = (
                            prior[0] + self._candidate_cost(
                                selected,
                                signature,
                                known_shapes=known,
                            ),
                            prior[1] + 1,
                            prior[2] + (end,),
                        )
                        existing = dynamic_shapes[end].get(resulting)
                        candidate_key = (
                            round(candidate[0], 12),
                            candidate[1],
                            tuple(-value for value in candidate[2]),
                        )
                        existing_key = (
                            None
                            if existing is None
                            else (
                                round(existing[0], 12),
                                existing[1],
                                tuple(-value for value in existing[2]),
                            )
                        )
                        if existing is None or candidate_key < existing_key:
                            dynamic_shapes[end][resulting] = candidate
                if not dynamic_shapes[end]:
                    raise RuntimeError(
                        "document cost planner lost every feasible prefix "
                        "despite a valid single-row partition"
                    )
            resulting_shapes, final = min(
                dynamic_shapes[count].items(),
                key=lambda item: (
                    round(item[1][0], 12),
                    item[1][1],
                    tuple(-value for value in item[1][2]),
                    tuple(sorted(item[0])),
                ),
            )
            result: list[tuple[DocumentSequence, ...]] = []
            start = 0
            for end in final[2]:
                result.append(tuple(sorted(
                    ordered[start:end],
                    key=lambda item: item.document_order,
                )))
                start = end
            return tuple(result), resulting_shapes

        # (estimated cost, cohort count, deterministic right boundaries)
        dynamic: list[
            tuple[float, int, tuple[int, ...]] | None
        ] = [None] * (count + 1)
        dynamic[0] = (0.0, 0, ())
        for end in range(1, count + 1):
            best: tuple[float, int, tuple[int, ...]] | None = None
            for start in range(end - 1, -1, -1):
                selected = ordered[start:end]
                signature = self._candidate_signature(selected)
                if any(
                    len(selected) * length > self.token_budget
                    for length in signature
                ):
                    break
                if not self._candidate_fits_memory(selected, signature):
                    self._last_rejected_memory_candidates += 1
                    continue
                prior = dynamic[start]
                if prior is None:
                    continue
                candidate = (
                    prior[0] + self._candidate_cost(selected, signature),
                    prior[1] + 1,
                    prior[2] + (end,),
                )
                candidate_key = (
                    round(candidate[0], 12),
                    candidate[1],
                    tuple(-value for value in candidate[2]),
                )
                best_key = (
                    None if best is None else (
                        round(best[0], 12),
                        best[1],
                        tuple(-value for value in best[2]),
                    )
                )
                if best is None or candidate_key < best_key:
                    best = candidate
            if best is None:
                raise RuntimeError(
                    "document cost planner lost every feasible prefix despite "
                    "a valid single-row partition"
                )
            dynamic[end] = best
        final = dynamic[count]
        if final is None:
            raise RuntimeError("document cost planner has no terminal state")
        result: list[tuple[DocumentSequence, ...]] = []
        start = 0
        for end in final[2]:
            result.append(tuple(sorted(
                ordered[start:end],
                key=lambda item: item.document_order,
            )))
            start = end
        shapes = initial_known_shapes.union(
            (len(group), length)
            for group in result
            for length in self._candidate_signature(group)
        )
        return tuple(result), frozenset(shapes)

    def _selected_groups(
        self, sequences: tuple[DocumentSequence, ...],
    ) -> tuple[tuple[tuple[DocumentSequence, ...], ...], bool]:
        exact_length_multiset = tuple(sorted(
            (
                len(sequence.spans),
                tuple(span.length for span in sequence.spans),
            )
            for sequence in sequences
        ))
        key: tuple[object, ...] = (
            self.device_torch_fingerprint,
            self.actor_configuration_digest,
            self.grouping_policy,
            self.activation_policy,
            self.compiler_policy,
            self.cost_model.digest,
            self.token_budget,
            self.cognitive_stride,
            self.maximum_candidate_activation_bytes,
            self.activation_bytes_per_token,
            exact_length_multiset,
        )
        cached = self._group_cache.get(key)
        canonical = tuple(sorted(
            sequences,
            key=lambda sequence: (
                len(sequence.spans),
                tuple(span.length for span in sequence.spans),
                sequence.document_order,
            ),
        ))
        if cached is not None:
            flattened = tuple(
                rank for group in cached for rank in group
            )
            cache_valid = (
                bool(cached)
                and not any(not group for group in cached)
                and len(flattened) == len(set(flattened))
                and set(flattened) == set(range(len(canonical)))
            )
            if cache_valid:
                try:
                    restored = tuple(
                        tuple(
                            canonical[rank] for rank in group
                        )
                        for group in cached
                    )
                    cache_valid = all(
                        self._candidate_fits_memory(
                            group, self._candidate_signature(group)
                        )
                        and all(
                            len(group) * length <= self.token_budget
                            for length in self._candidate_signature(group)
                        )
                        for group in restored
                    )
                except (KeyError, ValueError):
                    cache_valid = False
            if cache_valid:
                return restored, True
            self._group_cache.pop(key, None)
        if self.grouping_policy == "exact_signature":
            result = self._exact_groups(sequences)
        else:
            by_span_count: dict[int, list[DocumentSequence]] = {}
            for sequence in sequences:
                by_span_count.setdefault(len(sequence.spans), []).append(sequence)
            if (
                self.compiler_policy == "on"
                and self.cost_model.shape_compile_cost
            ):
                groups: list[tuple[DocumentSequence, ...]] = []
                known_shapes: frozenset[tuple[int, int]] = frozenset()
                for span_count in sorted(by_span_count):
                    selected, known_shapes = (
                        self._cost_groups_for_span_count_with_shapes(
                            tuple(by_span_count[span_count]),
                            initial_known_shapes=known_shapes,
                        )
                    )
                    groups.extend(selected)
                result = tuple(groups)
            else:
                result = tuple(
                    group
                    for span_count in sorted(by_span_count)
                    for group in self._cost_groups_for_span_count(
                        tuple(by_span_count[span_count])
                    )
                )
        if self.plan_cache_capacity:
            if len(self._group_cache) >= self.plan_cache_capacity:
                self._group_cache.pop(next(iter(self._group_cache)))
            rank_by_id = {
                sequence.sequence_id: rank
                for rank, sequence in enumerate(canonical)
            }
            self._group_cache[key] = tuple(
                tuple(rank_by_id[sequence.sequence_id] for sequence in group)
                for group in result
            )
        return result, False

    def _cohorts(
        self, sequences: tuple[DocumentSequence, ...],
    ) -> tuple[
        tuple[StaticDocumentCohort, ...],
        DocumentPlanCostReceipt,
    ]:
        selected_groups, cache_hit = self._selected_groups(sequences)
        exact_groups = self._exact_groups(sequences)

        def estimated_cost(
            groups: tuple[tuple[DocumentSequence, ...], ...],
        ) -> float:
            total = 0.0
            known_shapes: frozenset[tuple[int, int]] = frozenset()
            for group in groups:
                signature = self._candidate_signature(group)
                total += self._candidate_cost(
                    group,
                    signature,
                    known_shapes=known_shapes,
                )
                known_shapes = known_shapes.union(
                    (len(group), length) for length in signature
                )
            return total

        cohorts: list[StaticDocumentCohort] = []
        for cohort_id, selected in enumerate(selected_groups):
            signature = self._candidate_signature(selected)
            spans = tuple(
                self._materialize_span(
                    selected,
                    span_index=index,
                    padded_length=padded_length,
                )
                for index, padded_length in enumerate(signature)
            )
            cohorts.append(
                StaticDocumentCohort(
                    cohort_id, selected, signature, spans,
                )
            )
        result = tuple(cohorts)
        selected_shapes = frozenset(
            (len(group), length)
            for group in selected_groups
            for length in self._candidate_signature(group)
        )
        selected_peak_memory = max(
            (
                self.cost_model.shape_memory_bytes(batch, length)
                for batch, length in selected_shapes
            ),
            default=0,
        )
        return result, DocumentPlanCostReceipt(
            schema_version=1,
            policy=self.grouping_policy,
            cost_model_digest=self.cost_model.digest,
            selected_estimated_cost=estimated_cost(selected_groups),
            exact_signature_estimated_cost=estimated_cost(exact_groups),
            selected_invocations=sum(
                len(group[0].spans) for group in selected_groups
            ),
            exact_signature_invocations=sum(
                len(group[0].spans) for group in exact_groups
            ),
            rejected_memory_candidates=(
                self._last_rejected_memory_candidates
            ),
            cache_hit=cache_hit,
            unique_static_shapes=len(selected_shapes),
            predicted_peak_memory_bytes=selected_peak_memory,
            shape_compile_cost=(
                self.cost_model.shape_compile_cost
                if self.compiler_policy == "on" else 0.0
            ),
        )

    @staticmethod
    def _target_receipt(
        batch: PackedBatch,
        cohorts: tuple[StaticDocumentCohort, ...],
    ) -> TargetBijectionReceipt:
        context_length = batch.input_ids.shape[1]
        original = tuple(
            sorted(
                row * context_length + position
                for row in range(batch.input_ids.shape[0])
                for position in torch.nonzero(
                    batch.loss_mask[row], as_tuple=False
                ).flatten().tolist()
            )
        )
        emitted = [
            int(value)
            for cohort in cohorts
            for span in cohort.spans
            for value in span.target_ordinals[span.loss_mask].tolist()
        ]
        planned = tuple(sorted(emitted))
        original_set, planned_set = set(original), set(planned)
        counts: dict[int, int] = {}
        for value in emitted:
            counts[value] = counts.get(value, 0) + 1
        duplicates = tuple(sorted(value for value, count in counts.items() if count > 1))
        original_tensor = torch.tensor(original, dtype=torch.int64)
        planned_tensor = torch.tensor(planned, dtype=torch.int64)
        return TargetBijectionReceipt(
            original,
            planned,
            tuple(sorted(original_set - planned_set)),
            tuple(sorted(planned_set - original_set)),
            duplicates,
            _tensor_digest((original_tensor,)),
            _tensor_digest((planned_tensor,)),
        )

    def plan(self, batch: PackedBatch) -> DocumentBatchPlan:
        """Build and validate a complete document-major plan."""

        self._last_rejected_memory_candidates = 0
        sequences = self._extract_sequences(batch)
        cohorts, cost_receipt = self._cohorts(sequences)
        receipt = self._target_receipt(batch, cohorts)
        if not receipt.passed:
            raise RuntimeError(
                "document-major planning changed target authority: "
                f"missing={receipt.missing_ordinals}, "
                f"unexpected={receipt.unexpected_ordinals}, "
                f"duplicates={receipt.duplicate_ordinals}"
            )
        return DocumentBatchPlan(
            cohorts,
            sequences,
            receipt,
            self.bucket_lengths,
            self.token_budget,
            self.alignment,
            self.cognitive_stride,
            batch.token_count,
            int(batch.loss_mask.sum()),
            cost_receipt,
        )
