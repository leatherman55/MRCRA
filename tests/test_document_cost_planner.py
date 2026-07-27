"""Production contracts for the cost and memory authority of document plans."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
import json
from time import perf_counter

import pytest
import torch

from mrrn.document_batching import DocumentMajorBatchPlanner
from mrrn.document_cost_model import (
    DocumentExecutionCostModel,
    measured_document_cost_model,
)
from mrrn.lm_training import PackedBatch


def packed_documents(lengths: tuple[int, ...]) -> PackedBatch:
    segments = torch.cat(
        tuple(
            torch.full((length,), index, dtype=torch.int64)
            for index, length in enumerate(lengths)
        )
    )
    total = int(segments.numel())
    inputs = torch.arange(1, total + 1, dtype=torch.int64).unsqueeze(0)
    labels = inputs + 31
    bytes_ = inputs.remainder(4) + 1
    target_segments = segments.clone()
    if total > 1:
        target_segments[:-1] = segments[1:]
    return PackedBatch(
        inputs,
        labels,
        bytes_,
        segments.unsqueeze(0),
        target_segments.unsqueeze(0),
        (
            tuple(
                (index, f"dataset://cost-fixture/{index}")
                for index in range(len(lengths))
            ),
        ),
    )


def make_planner(**changes) -> DocumentMajorBatchPlanner:
    values = {
        "tbptt_length": 4,
        "bucket_lengths": (2, 4, 8),
        "token_budget": 16,
        "alignment": 2,
        "cognitive_stride": 2,
        "plan_cache_capacity": 4,
        "cost_model": DocumentExecutionCostModel(
            launch_cost=11.0,
            token_forward_cost=1.0,
            padding_cost=0.5,
            cognitive_anchor_cost=2.0,
            backward_multiplier_retain=1.0,
            backward_multiplier_selective=1.0,
            backward_multiplier_whole_span=1.0,
            calibration_kind="test_exact",
        ),
    }
    values.update(changes)
    return DocumentMajorBatchPlanner(**values)


def test_dynamic_program_equals_exhaustive_contiguous_partition_optimum():
    planner = make_planner()
    sequences = planner._extract_sequences(
        packed_documents((2, 3, 4, 2, 4))
    )
    selected = planner._cost_groups_for_span_count(sequences)
    selected_cost = sum(
        planner._candidate_cost(
            group, planner._candidate_signature(group)
        )
        for group in selected
    )

    ordered = tuple(sorted(
        sequences,
        key=lambda item: (
            tuple(span.length for span in item.spans),
            item.document_order,
        ),
    ))
    count = len(ordered)
    exhaustive: list[float] = []
    for cuts_count in range(count):
        for cuts in combinations(range(1, count), cuts_count):
            boundaries = (0, *cuts, count)
            groups = tuple(
                ordered[left:right]
                for left, right in zip(
                    boundaries, boundaries[1:]
                )
            )
            signatures = tuple(
                planner._candidate_signature(group) for group in groups
            )
            if any(
                any(
                    len(group) * length > planner.token_budget
                    for length in signature
                )
                for group, signature in zip(
                    groups, signatures, strict=True
                )
            ):
                continue
            exhaustive.append(
                sum(
                    planner._candidate_cost(group, signature)
                    for group, signature in zip(
                        groups, signatures, strict=True
                    )
                )
            )
    assert exhaustive
    assert selected_cost == pytest.approx(min(exhaustive), abs=1e-12)


def test_measured_cost_fit_uses_real_affine_observations_and_length_bands():
    model = measured_document_cost_model(
        single_invocation_seconds=0.012,
        batched_invocation_seconds=0.020,
        single_physical_tokens=128,
        batched_physical_tokens=384,
        length_bands=(128, 256, 512),
        activation_policy="retain",
        hardware_fingerprint="a" * 64,
        activation_bytes_per_token=256,
        shape_compile_cost=0.25,
    )
    assert model.calibration_kind.startswith("measured_")
    assert model.launch_cost == pytest.approx(0.008)
    assert model.token_cost(128) == pytest.approx(0.00003125)
    assert model.token_cost(4096) == model.token_cost(512)
    assert model.shape_memory_bytes(3, 128) == 3 * 128 * 256
    without_shape = model.estimate(
        padded_lengths=(128,),
        valid_lengths_by_row=((120,),),
        cognitive_stride=16,
        activation_policy="retain",
        compiler_enabled=True,
    )
    with_shape = model.estimate(
        padded_lengths=(128,),
        valid_lengths_by_row=((120,),),
        cognitive_stride=16,
        activation_policy="retain",
        known_shapes=frozenset({(1, 128)}),
        compiler_enabled=True,
    )
    assert without_shape - with_shape == pytest.approx(0.25)


def test_measured_cost_fit_binds_per_length_time_and_shape_memory():
    model = measured_document_cost_model(
        single_invocation_seconds=0.010,
        batched_invocation_seconds=0.022,
        single_physical_tokens=128,
        batched_physical_tokens=512,
        length_bands=(128, 256, 512),
        activation_policy="selective",
        hardware_fingerprint="b" * 64,
        activation_bytes_per_token=128,
        length_band_observations=(
            (128, 0.010, 1, 20_000),
            (256, 0.013, 1, 35_000),
            (512, 0.025, 1, 68_000),
        ),
    )
    assert "per_length" in model.calibration_kind
    assert len({
        seconds for _, seconds in model.token_seconds_by_length_band
    }) == 3
    assert model.shape_memory_bytes(1, 128) == 20_000
    assert model.shape_memory_bytes(1, 256) == 35_000
    assert model.shape_memory_bytes(1, 512) == 68_000
    assert model.shape_memory_bytes(2, 128) == 2 * 128 * 128
    with pytest.raises(ValueError, match="malformed"):
        measured_document_cost_model(
            single_invocation_seconds=0.010,
            batched_invocation_seconds=0.022,
            single_physical_tokens=128,
            batched_physical_tokens=512,
            length_bands=(128, 256),
            activation_policy="retain",
            hardware_fingerprint="b" * 64,
            activation_bytes_per_token=128,
            length_band_observations=((384, 0.01, 1, 10),),
        )


def test_plan_receipt_binds_shape_memory_and_compiler_cost_authority():
    cost = replace(
        make_planner().cost_model,
        shape_compile_cost=0.125,
        activation_bytes_per_token=64,
    )
    planner = make_planner(cost_model=cost, compiler_policy="on")
    plan = planner.plan(packed_documents((2, 3, 4, 2)))
    assert plan.cost_receipt.unique_static_shapes > 0
    assert plan.cost_receipt.predicted_peak_memory_bytes > 0
    assert plan.cost_receipt.shape_compile_cost == pytest.approx(0.125)


def test_cost_planner_prices_the_shape_conditional_activation_policy():
    planner = make_planner(
        activation_policy="whole_span",
        activation_policy_token_limits={
            "retain": 4,
            "selective": 8,
            "whole_span": 16,
        },
        activation_policy_timings={
            "retain": 1.0,
            "selective": 1.5,
            "whole_span": 2.0,
        },
        cost_model=replace(
            make_planner().cost_model,
            backward_multiplier_retain=1.0,
            backward_multiplier_selective=2.0,
            backward_multiplier_whole_span=4.0,
        ),
    )
    assert planner._activation_policy_for_physical_tokens(4) == "retain"
    assert planner._activation_policy_for_physical_tokens(8) == "selective"
    assert planner._activation_policy_for_physical_tokens(16) == "whole_span"
    sequences = planner._extract_sequences(packed_documents((4, 4)))
    group = tuple(sequences)
    signature = planner._candidate_signature(group)
    conditional = planner._candidate_cost(group, signature)
    planner.activation_policy_token_limits = {}
    all_recomputed = planner._candidate_cost(group, signature)
    assert conditional < all_recomputed


def test_compile_aware_dynamic_program_matches_global_unique_shape_optimum():
    planner = make_planner(
        compiler_policy="on",
        cost_model=replace(
            make_planner().cost_model,
            shape_compile_cost=50.0,
        ),
    )
    sequences = planner._extract_sequences(
        packed_documents((2, 3, 4, 2, 4))
    )
    ordered = tuple(sorted(
        sequences,
        key=lambda item: (
            tuple(span.length for span in item.spans),
            item.document_order,
        ),
    ))
    selected = planner._cost_groups_for_span_count(sequences)

    def global_cost(groups):
        known = frozenset()
        total = 0.0
        for group in groups:
            signature = planner._candidate_signature(group)
            total += planner._candidate_cost(
                group, signature, known_shapes=known
            )
            known = known.union(
                (len(group), length) for length in signature
            )
        return total

    exhaustive = []
    count = len(ordered)
    for cuts_count in range(count):
        for cuts in combinations(range(1, count), cuts_count):
            boundaries = (0, *cuts, count)
            groups = tuple(
                ordered[left:right]
                for left, right in zip(boundaries, boundaries[1:])
            )
            signatures = tuple(
                planner._candidate_signature(group) for group in groups
            )
            if any(
                any(
                    len(group) * length > planner.token_budget
                    for length in signature
                )
                for group, signature in zip(
                    groups, signatures, strict=True
                )
            ):
                continue
            exhaustive.append(global_cost(groups))
    assert exhaustive
    assert global_cost(selected) == pytest.approx(
        min(exhaustive), abs=1e-12
    )


def test_activation_memory_infeasible_cohorts_are_rejected_and_receipted():
    planner = make_planner(
        maximum_candidate_activation_bytes=80,
        activation_bytes_per_token=10,
    )
    plan = planner.plan(packed_documents((4, 4, 4, 4)))
    assert plan.cost_receipt.rejected_memory_candidates > 0
    assert all(
        span.batch_size * span.padded_length * 10 <= 80
        for cohort in plan.cohorts
        for span in cohort.spans
    )
    assert plan.receipt.passed


def test_corrupt_shape_cache_is_discarded_then_revalidated_bijectively():
    planner = make_planner()
    batch = packed_documents((2, 3, 4, 2))
    first = planner.plan(batch)
    assert len(planner._group_cache) == 1
    key = next(iter(planner._group_cache))
    planner._group_cache[key] = ((999_999,),)
    repaired = planner.plan(batch)
    assert repaired.receipt == first.receipt
    assert not repaired.cost_receipt.cache_hit
    assert all(
        sequence.sequence_id != 999_999
        for cohort in repaired.cohorts
        for sequence in cohort.sequences
    )


def test_cache_identity_includes_activation_memory_and_policy_authorities():
    planner = make_planner()
    batch = packed_documents((2, 3, 4, 2))
    planner.plan(batch)
    assert planner.plan(batch).cost_receipt.cache_hit
    planner.activation_policy = "selective"
    assert not planner.plan(batch).cost_receipt.cache_hit
    planner.maximum_candidate_activation_bytes = 1_000
    planner.activation_bytes_per_token = 10
    assert not planner.plan(batch).cost_receipt.cache_hit
    planner.device_torch_fingerprint = "different-device"
    assert not planner.plan(batch).cost_receipt.cache_hit
    planner.actor_configuration_digest = "different-actor"
    assert not planner.plan(batch).cost_receipt.cache_hit
    planner.compiler_policy = "on"
    assert not planner.plan(batch).cost_receipt.cache_hit


def test_adversarial_document_mixture_has_bounded_planning_time():
    planner = make_planner(token_budget=32)
    batch = packed_documents(
        tuple(1 + (index * 7) % 8 for index in range(96))
    )
    started = perf_counter()
    plan = planner.plan(batch)
    elapsed = perf_counter() - started
    assert plan.receipt.passed
    # A generous unit gate catches accidental exponential search without
    # pretending to be a target-hardware throughput claim.
    assert elapsed < 2.0


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"maximum_candidate_activation_bytes": 0}, "activation-memory"),
        (
            {
                "maximum_candidate_activation_bytes": 100,
                "activation_bytes_per_token": 0,
            },
            "activation-memory",
        ),
        ({"activation_bytes_per_token": -1}, "activation-memory"),
        ({"device_torch_fingerprint": ""}, "cache identity"),
        ({"actor_configuration_digest": ""}, "cache identity"),
        ({"compiler_policy": "maybe"}, "cache identity"),
    ),
)
def test_activation_memory_contract_rejects_ambiguous_configuration(
    changes, message,
):
    with pytest.raises(ValueError, match=message):
        make_planner(**changes)


def test_cost_model_json_round_trip_restores_tuple_authority():
    model = DocumentExecutionCostModel(
        calibration_kind="measured_cpu",
        launch_cost=0.001,
        token_forward_cost=1e-6,
        token_seconds_by_length_band=((128, 2e-6), (512, 1e-6)),
        shape_compile_cost=0.02,
        activation_bytes_per_token=64,
        memory_cost_per_byte=1e-12,
        memory_bytes_by_shape=((2, 128, 16_384),),
    )
    serialized = json.loads(json.dumps(model.to_dict()))
    restored = DocumentExecutionCostModel.from_dict(serialized)
    assert restored == model
    assert restored.digest == model.digest
