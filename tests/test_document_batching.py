from __future__ import annotations

from dataclasses import replace

from hypothesis import given, settings
from hypothesis import strategies as st
import pytest
import torch

from mrrn.document_batching import (
    DocumentMajorBatchPlanner,
    StaticDocumentCohort,
)
from mrrn.lm_training import PackedBatch


def packed_documents(
    lengths: tuple[int, ...],
    *,
    rows: int = 1,
    final_target_valid: bool = True,
) -> PackedBatch:
    """Construct a loss-authority fixture with explicit document boundaries."""

    if rows != 1:
        raise ValueError("the local fixture currently constructs one packed row")
    segments = torch.cat(
        tuple(
            torch.full((length,), index, dtype=torch.int64)
            for index, length in enumerate(lengths)
        )
    )
    total = int(segments.numel())
    inputs = torch.arange(1, total + 1, dtype=torch.int64).unsqueeze(0)
    labels = (inputs + 17).clone()
    byte_lengths = (inputs.remainder(4) + 1).to(torch.int64)
    target_segments = segments.clone()
    if total > 1:
        target_segments[:-1] = segments[1:]
    if not final_target_valid:
        target_segments[-1] = int(segments[-1]) + 1
    declarations = (
        tuple(
            (index, f"dataset://fixture/document-{index}")
            for index in range(len(lengths))
        ),
    )
    return PackedBatch(
        inputs,
        labels,
        byte_lengths,
        segments.unsqueeze(0),
        target_segments.unsqueeze(0),
        declarations,
    )


def planner(
    *,
    tbptt_length: int = 4,
    token_budget: int = 8,
) -> DocumentMajorBatchPlanner:
    return DocumentMajorBatchPlanner(
        tbptt_length=tbptt_length,
        bucket_lengths=(2, 4, 8),
        token_budget=token_budget,
        alignment=2,
        cognitive_stride=2,
    )


def test_document_plan_is_target_bijective_and_preserves_authority_tensors():
    batch = packed_documents((3, 2, 4, 3))
    plan = planner().plan(batch)

    assert plan.receipt.passed
    assert plan.original_valid_targets == int(batch.loss_mask.sum())
    assert plan.planned_valid_targets == int(batch.loss_mask.sum())
    assert plan.receipt.original_digest == plan.receipt.planned_digest
    assert plan.receipt.missing_ordinals == ()
    assert plan.receipt.unexpected_ordinals == ()
    assert plan.receipt.duplicate_ordinals == ()

    for cohort in plan.cohorts:
        for physical in cohort.spans:
            for row in range(physical.batch_size):
                sequence = cohort.sequences[row]
                source = sequence.spans[physical.span_index]
                length = source.length
                torch.testing.assert_close(
                    physical.input_ids[row, :length], source.input_ids
                )
                torch.testing.assert_close(
                    physical.labels[row, :length], source.labels
                )
                torch.testing.assert_close(
                    physical.target_byte_lengths[row, :length],
                    source.target_byte_lengths,
                )
                torch.testing.assert_close(
                    physical.boundary_classes[row, :length],
                    source.boundary_classes,
                )
                torch.testing.assert_close(
                    physical.loss_mask[row, :length], source.loss_mask
                )
                assert not physical.token_mask[row, length:].any()
                assert not physical.loss_mask[row, length:].any()
                assert torch.all(physical.target_ordinals[row, length:] == -1)


def test_carrier_aligned_bucket_may_end_inside_cognitive_stride_without_leakage():
    local = DocumentMajorBatchPlanner(
        tbptt_length=4,
        bucket_lengths=(2, 4),
        token_budget=4,
        alignment=2,
        cognitive_stride=4,
    )
    plan = local.plan(packed_documents((2, 2)))
    assert plan.receipt.passed
    assert plan.original_valid_targets == plan.planned_valid_targets
    assert any(
        physical.padded_length == 2
        and physical.padded_length % local.cognitive_stride != 0
        for cohort in plan.cohorts
        for physical in cohort.spans
    )
    for cohort in plan.cohorts:
        for physical in cohort.spans:
            assert physical.event_mask.shape[1] == (
                physical.padded_length + local.cognitive_stride - 1
            ) // local.cognitive_stride
            sampled_token_mask = physical.token_mask[
                :, :: local.cognitive_stride
            ]
            assert not bool(
                (physical.event_mask & ~sampled_token_mask).any()
            )


def test_static_cohorts_preserve_rows_across_every_tbptt_span():
    batch = packed_documents((9, 9, 5))
    target_segments = batch.target_segment_ids.clone()
    # Authorize the terminal target in both long documents so their final
    # one-token TBPTT spans remain part of the execution authority.
    target_segments[0, 8] = batch.segment_ids[0, 8]
    target_segments[0, 17] = batch.segment_ids[0, 17]
    batch = replace(batch, target_segment_ids=target_segments)
    plan = planner(tbptt_length=4, token_budget=8).plan(batch)
    long = [
        cohort
        for cohort in plan.cohorts
        if cohort.padded_lengths == (4, 4, 2)
    ]
    assert len(long) == 1
    cohort = long[0]
    assert len(cohort.sequences) == 2
    expected = tuple(sequence.sequence_id for sequence in cohort.sequences)
    for index, physical in enumerate(cohort.spans):
        assert tuple(physical.sequence_ids.tolist()) == expected
        assert physical.reset_state == (index == 0)
        assert physical.final_rows.tolist() == [index == 2, index == 2]
        assert physical.context_starts.tolist() == [4 * index, 9 + 4 * index]


def test_event_mask_authorizes_only_anchors_before_each_document_length():
    plan = planner().plan(packed_documents((1, 3, 4)))
    sequence = next(item for item in plan.sequences if item.segment_id == 1)
    cohort = next(
        item for item in plan.cohorts if sequence.sequence_id in {
            value.sequence_id for value in item.sequences
        }
    )
    row = next(
        index
        for index, value in enumerate(cohort.sequences)
        if value.sequence_id == sequence.sequence_id
    )
    physical = cohort.spans[0]
    assert physical.valid_lengths[row] == 3
    assert physical.event_mask[row].tolist() == [True, True]
    assert physical.token_mask[row].tolist() == [True, True, True, False]


def test_token_budget_deterministically_splits_same_signature_documents():
    batch = packed_documents((4, 4, 4, 4, 4))
    first = planner(token_budget=8).plan(batch)
    second = planner(token_budget=8).plan(batch)
    matching = [
        cohort for cohort in first.cohorts if cohort.padded_lengths == (4,)
    ]
    assert [len(cohort.sequences) for cohort in matching] == [2, 2, 1]
    assert [
        tuple(sequence.document_order for sequence in cohort.sequences)
        for cohort in first.cohorts
    ] == [
        tuple(sequence.document_order for sequence in cohort.sequences)
        for cohort in second.cohorts
    ]
    assert [
        tuple(span.digest for span in cohort.spans)
        for cohort in first.cohorts
    ] == [
        tuple(span.digest for span in cohort.spans)
        for cohort in second.cohorts
    ]


def test_tbptt_upper_bound_subdivides_at_memory_safe_static_bucket():
    local = DocumentMajorBatchPlanner(
        tbptt_length=8,
        bucket_lengths=(2, 4, 6, 8),
        token_budget=8,
        alignment=2,
        cognitive_stride=2,
        maximum_candidate_activation_bytes=40,
        activation_bytes_per_token=10,
    )
    batch = packed_documents((10,))
    plan = local.plan(batch)

    assert plan.receipt.passed
    assert tuple(span.length for span in plan.sequences[0].spans) == (4, 4, 2)
    assert plan.cohorts[0].padded_lengths == (4, 4, 2)
    assert plan.cohorts[0].spans[0].reset_state
    assert not plan.cohorts[0].spans[1].reset_state
    assert not plan.cohorts[0].spans[2].reset_state
    assert plan.cohorts[0].spans[2].final_rows.tolist() == [True]
    assert plan.planned_valid_targets == int(batch.loss_mask.sum())
    assert all(
        physical.batch_size
        * physical.padded_length
        * local.activation_bytes_per_token
        <= local.maximum_candidate_activation_bytes
        for cohort in plan.cohorts
        for physical in cohort.spans
    )


def test_tbptt_upper_bound_subdivides_at_single_row_token_budget():
    local = DocumentMajorBatchPlanner(
        tbptt_length=8,
        bucket_lengths=(2, 4, 6, 8),
        token_budget=4,
        alignment=2,
        cognitive_stride=2,
    )
    plan = local.plan(packed_documents((10,)))
    assert tuple(span.length for span in plan.sequences[0].spans) == (4, 4, 2)
    assert plan.receipt.passed


def test_smallest_bucket_memory_infeasibility_names_the_hard_constraint():
    local = DocumentMajorBatchPlanner(
        tbptt_length=8,
        bucket_lengths=(2, 4, 6, 8),
        token_budget=8,
        alignment=2,
        cognitive_stride=2,
        maximum_candidate_activation_bytes=19,
        activation_bytes_per_token=10,
    )
    with pytest.raises(
        ValueError,
        match=(
            "no single-row document span fits.*"
            "smallest_bucket=2.*"
            "estimated_activation_bytes=20"
        ),
    ):
        local.plan(packed_documents((10,)))


def test_cost_aware_planner_merges_compatible_unequal_signatures_and_proves_savings():
    batch = packed_documents((2, 3, 2, 3))
    cost_aware = planner(token_budget=16).plan(batch)
    exact = DocumentMajorBatchPlanner(
        tbptt_length=4,
        bucket_lengths=(2, 4, 8),
        token_budget=16,
        alignment=2,
        cognitive_stride=2,
        grouping_policy="exact_signature",
    ).plan(batch)
    assert cost_aware.receipt == exact.receipt
    assert cost_aware.physical_invocations < exact.physical_invocations
    assert (
        cost_aware.cost_receipt.selected_estimated_cost
        < cost_aware.cost_receipt.exact_signature_estimated_cost
    )
    assert cost_aware.cost_receipt.estimated_savings_fraction > 0


def test_cost_plan_cache_is_bounded_deterministic_and_receipted():
    local = DocumentMajorBatchPlanner(
        tbptt_length=4,
        bucket_lengths=(2, 4, 8),
        token_budget=8,
        alignment=2,
        cognitive_stride=2,
        plan_cache_capacity=1,
    )
    first = local.plan(packed_documents((2, 3, 2)))
    second = local.plan(packed_documents((2, 3, 2)))
    assert not first.cost_receipt.cache_hit
    assert second.cost_receipt.cache_hit
    assert first.receipt == second.receipt
    assert [cohort.padded_lengths for cohort in first.cohorts] == [
        cohort.padded_lengths for cohort in second.cohorts
    ]


def test_context_rows_receive_global_unique_target_ordinals():
    first = packed_documents((3, 3))
    second = packed_documents((2, 4))
    batch = PackedBatch(
        torch.cat((first.input_ids, second.input_ids), 0),
        torch.cat((first.labels, second.labels), 0),
        torch.cat((first.target_byte_lengths, second.target_byte_lengths), 0),
        torch.cat((first.segment_ids, second.segment_ids + 2), 0),
        torch.cat((first.target_segment_ids, second.target_segment_ids + 2), 0),
        (
            first.source_uris_by_segment[0],
            tuple((segment + 2, uri) for segment, uri in second.source_uris_by_segment[0]),
        ),
    )
    plan = planner().plan(batch)
    assert plan.receipt.passed
    assert len(plan.receipt.planned_valid_ordinals) == len(
        set(plan.receipt.planned_valid_ordinals)
    )
    assert any(value >= batch.input_ids.shape[1] for value in plan.receipt.planned_valid_ordinals)


@settings(max_examples=500, deadline=None)
@given(
    lengths=st.lists(
        st.integers(min_value=1, max_value=10),
        min_size=1,
        max_size=12,
    ),
    final_target_valid=st.booleans(),
    token_budget=st.sampled_from((8, 12, 16, 32)),
)
def test_random_document_mixtures_are_bijective_deterministic_and_aligned(
    lengths,
    final_target_valid,
    token_budget,
):
    batch = packed_documents(
        tuple(lengths), final_target_valid=final_target_valid
    )
    if not bool(batch.loss_mask.any()):
        with pytest.raises(ValueError, match="no trainable documents"):
            planner(token_budget=token_budget).plan(batch)
        return
    first = planner(token_budget=token_budget).plan(batch)
    second = planner(token_budget=token_budget).plan(batch)
    assert first.receipt.passed and second.receipt.passed
    assert first.receipt == second.receipt
    assert first.planned_valid_targets == int(batch.loss_mask.sum())
    assert first.physical_invocations == second.physical_invocations
    for cohort in first.cohorts:
        assert isinstance(cohort, StaticDocumentCohort)
        for physical in cohort.spans:
            assert physical.padded_length % 2 == 0
            assert physical.token_mask.numel() <= max(
                token_budget,
                physical.padded_length,
            )
            assert 0 < physical.padding_efficiency <= 1
            assert not bool((physical.loss_mask & ~physical.token_mask).any())


@pytest.mark.parametrize(
    "options, message",
    (
        ({"bucket_lengths": ()}, "buckets"),
        ({"bucket_lengths": (4, 2)}, "buckets"),
        ({"bucket_lengths": (3, 4)}, "align"),
        ({"bucket_lengths": (2, 4), "tbptt_length": 8}, "TBPTT"),
        ({"token_budget": 1}, "smallest"),
        ({"padding_token_id": -1}, "padding"),
    ),
)
def test_document_planner_configuration_fails_closed(options, message):
    values = dict(
        tbptt_length=4,
        bucket_lengths=(2, 4, 8),
        token_budget=8,
        alignment=2,
        cognitive_stride=2,
    )
    values.update(options)
    with pytest.raises(ValueError, match=message):
        DocumentMajorBatchPlanner(**values)


def test_noncontiguous_repeated_segment_fails_closed():
    batch = packed_documents((2, 2, 2))
    broken_segments = batch.segment_ids.clone()
    broken_targets = batch.target_segment_ids.clone()
    broken_segments[0, 4:] = 0
    broken_targets[0, 4:] = 0
    broken = replace(
        batch,
        segment_ids=broken_segments,
        target_segment_ids=broken_targets,
    )
    with pytest.raises(ValueError, match="noncontiguous"):
        planner().plan(broken)
