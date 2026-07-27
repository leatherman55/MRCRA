"""Production contracts for Causal Spectral Target Multiplexing.

These tests intentionally validate the mathematical target, causal indexing,
packed-document isolation, stop-gradient behavior, spectral order sensitivity,
running normalization, cognitive gate, and trainability independently.  The
trainer integration and empirical acceptance tests exercise the complete path.
"""

from __future__ import annotations

from math import pi

import pytest
import torch

from mrrn.cstm import (
    CSTMArchitectureConfig,
    CausalSpectralTargetPredictor,
    build_causal_spectral_targets,
    causal_spectral_target_mask,
    deterministic_token_codes,
)


def _packed_fixture():
    # Input positions 0..5 belong to document 0 and 6..11 to document 1.
    # labels[p] is the token after input position p.  The final position of
    # each document is intentionally invalid as a next-token target.
    labels = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    segments = torch.tensor([[0] * 6 + [1] * 6], dtype=torch.int64)
    target_segments = segments.clone()
    mask = torch.ones_like(labels, dtype=torch.bool)
    mask[:, 5] = False
    mask[:, 11] = False
    target_segments[:, 5] = 1
    target_segments[:, 11] = 2
    codes = deterministic_token_codes(32, 16, seed=17)
    return labels, mask, segments, target_segments, codes


def _direct_dft(codes: torch.Tensor) -> torch.Tensor:
    support = codes.shape[0]
    phase = 2 * pi * torch.arange(support, dtype=codes.dtype) / support
    return torch.stack(
        (
            codes.sum(0),
            (codes * phase.cos()[:, None]).sum(0),
            (codes * -phase.sin()[:, None]).sum(0),
        )
    ) / support**0.5


def test_fixed_codebook_is_reproducible_normalized_and_does_not_mutate_global_rng():
    torch.manual_seed(101)
    state = torch.random.get_rng_state()
    first = deterministic_token_codes(257, 64, seed=20260725)
    after = torch.random.get_rng_state()
    second = deterministic_token_codes(257, 64, seed=20260725)
    different = deterministic_token_codes(257, 64, seed=20260726)

    assert torch.equal(state, after)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    torch.testing.assert_close(first.square().sum(-1), torch.ones(257))
    assert not any(
        torch.equal(first[left], first[right])
        for left in range(first.shape[0])
        for right in range(left + 1, first.shape[0])
    )


def test_target_builder_matches_direct_negative_exponent_dft_exactly():
    labels, mask, segments, target_segments, codes = _packed_fixture()
    source = torch.tensor([1, 7], dtype=torch.int64)
    targets = build_causal_spectral_targets(
        labels,
        mask,
        segments,
        target_segments,
        codes,
        source,
        support=4,
        horizons=(1,),
    )

    assert targets.mask.tolist() == [[[True], [True]]]
    expected_first = _direct_dft(codes[labels[0, 1:5]])
    expected_second = _direct_dft(codes[labels[0, 7:11]])
    torch.testing.assert_close(targets.values[0, 0, 0], expected_first)
    torch.testing.assert_close(targets.values[0, 1, 0], expected_second)
    assert targets.valid_rows == 2
    assert targets.token_participations == 8
    assert not targets.values.requires_grad


def test_target_builder_vectorizes_independent_document_rows_without_cross_row_leakage():
    labels, mask, segments, target_segments, codes = _packed_fixture()
    batched_labels = torch.cat((labels[:, :6], labels[:, 6:] + 4), 0)
    batched_mask = torch.cat((mask[:, :6], mask[:, 6:]), 0)
    batched_segments = torch.tensor(
        [[7] * 6, [11] * 6], dtype=torch.int64
    )
    batched_target_segments = batched_segments.clone()
    batched_target_segments[:, -1] += 1
    sources = torch.tensor([0, 1, 3], dtype=torch.int64)
    batched = build_causal_spectral_targets(
        batched_labels,
        batched_mask,
        batched_segments,
        batched_target_segments,
        codes,
        sources,
        support=2,
        horizons=(1, 2),
    )
    assert batched.values.shape == (2, 3, 2, 3, codes.shape[1])
    for row in range(2):
        isolated = build_causal_spectral_targets(
            batched_labels[row : row + 1],
            batched_mask[row : row + 1],
            batched_segments[row : row + 1],
            batched_target_segments[row : row + 1],
            codes,
            sources,
            support=2,
            horizons=(1, 2),
        )
        torch.testing.assert_close(
            batched.values[row : row + 1], isolated.values
        )
        assert torch.equal(
            batched.mask[row : row + 1], isolated.mask
        )
    changed = batched_labels.clone()
    changed[1, :4] = torch.tensor([30, 29, 28, 27])
    perturbed = build_causal_spectral_targets(
        changed,
        batched_mask,
        batched_segments,
        batched_target_segments,
        codes,
        sources,
        support=2,
        horizons=(1, 2),
    )
    torch.testing.assert_close(perturbed.values[0], batched.values[0])
    assert not torch.equal(perturbed.values[1], batched.values[1])


def test_source_position_targets_only_strictly_future_tokens():
    labels, mask, segments, target_segments, codes = _packed_fixture()
    source = torch.tensor([1], dtype=torch.int64)
    baseline = build_causal_spectral_targets(
        labels, mask, segments, target_segments, codes, source,
        support=2, horizons=(1, 2),
    )

    changed_past = labels.clone()
    changed_past[0, 0] = 31
    past = build_causal_spectral_targets(
        changed_past, mask, segments, target_segments, codes, source,
        support=2, horizons=(1, 2),
    )
    torch.testing.assert_close(past.values, baseline.values)

    changed_next_block = labels.clone()
    changed_next_block[0, 1] = 31
    future = build_causal_spectral_targets(
        changed_next_block, mask, segments, target_segments, codes, source,
        support=2, horizons=(1, 2),
    )
    assert not torch.equal(future.values[:, :, 0], baseline.values[:, :, 0])
    torch.testing.assert_close(future.values[:, :, 1], baseline.values[:, :, 1])


def test_incomplete_or_cross_document_blocks_fail_closed_instead_of_padding():
    labels, mask, segments, target_segments, codes = _packed_fixture()
    sources = torch.tensor([3, 4, 5, 9, 10, 11], dtype=torch.int64)
    targets = build_causal_spectral_targets(
        labels, mask, segments, target_segments, codes, sources,
        support=2, horizons=(1, 2),
    )

    # Source 3 can predict labels[3:5], but its second horizon reaches the
    # boundary. All later sources in each document lack complete support.
    assert targets.mask.tolist() == [[
        [True, False],
        [False, False],
        [False, False],
        [True, False],
        [False, False],
        [False, False],
    ]]
    assert torch.count_nonzero(targets.values[~targets.mask]) == 0


def test_validity_only_authority_exactly_matches_constructed_target_mask():
    labels, mask, segments, target_segments, codes = _packed_fixture()
    sources = torch.tensor([0, 1, 3, 4, 6, 7, 9, 10], dtype=torch.int64)
    expected = causal_spectral_target_mask(
        mask,
        segments,
        target_segments,
        sources,
        support=2,
        horizons=(1, 2, 4),
    )
    targets = build_causal_spectral_targets(
        labels,
        mask,
        segments,
        target_segments,
        codes,
        sources,
        support=2,
        horizons=(1, 2, 4),
    )
    assert torch.equal(expected, targets.mask)


def test_target_builder_rejects_negative_or_out_of_vocabulary_labels():
    labels, mask, segments, target_segments, codes = _packed_fixture()
    for invalid in (-1, codes.shape[0]):
        changed = labels.clone()
        changed[0, 0] = invalid
        with pytest.raises(ValueError, match="codebook"):
            build_causal_spectral_targets(
                changed,
                mask,
                segments,
                target_segments,
                codes,
                torch.tensor([0]),
                support=2,
                horizons=(1,),
            )


def test_dc_is_permutation_invariant_while_first_harmonic_detects_order():
    labels, mask, segments, target_segments, codes = _packed_fixture()
    source = torch.tensor([0], dtype=torch.int64)
    original = build_causal_spectral_targets(
        labels, mask, segments, target_segments, codes, source,
        support=4, horizons=(1,),
    )
    permuted_labels = labels.clone()
    permuted_labels[0, 0:4] = permuted_labels[0, torch.tensor([2, 0, 3, 1])]
    permuted = build_causal_spectral_targets(
        permuted_labels, mask, segments, target_segments, codes, source,
        support=4, horizons=(1,),
    )

    torch.testing.assert_close(original.values[..., 0, :], permuted.values[..., 0, :])
    assert not torch.allclose(
        original.values[..., 1:, :],
        permuted.values[..., 1:, :],
    )


def test_two_token_first_harmonic_is_real_nyquist_with_zero_imaginary_part():
    labels, mask, segments, target_segments, codes = _packed_fixture()
    targets = build_causal_spectral_targets(
        labels, mask, segments, target_segments, codes,
        torch.tensor([0]), support=2, horizons=(1,),
    )
    torch.testing.assert_close(
        targets.values[..., 2, :],
        torch.zeros_like(targets.values[..., 2, :]),
        atol=1e-6,
        rtol=0,
    )
    expected = (codes[labels[0, 0]] - codes[labels[0, 1]]) / 2**0.5
    torch.testing.assert_close(targets.values[0, 0, 0, 1], expected)


def test_predictor_cognitive_arm_is_exactly_zero_initialized_but_trainable():
    config = CSTMArchitectureConfig(
        code_dimension=16,
        predictor_rank=4,
        horizon_blocks=(1, 2),
    )
    predictor = CausalSpectralTargetPredictor(8, 3, 32, config)
    carrier = torch.randn(1, 5, 8, requires_grad=True)
    cognition_a = torch.randn(1, 5, 8, requires_grad=True)
    cognition_b = torch.randn(1, 5, 8, requires_grad=True)

    output_a = predictor(carrier, cognition_a, scale=1, horizons=(1, 2))
    output_b = predictor(carrier, cognition_b, scale=1, horizons=(1, 2))
    torch.testing.assert_close(output_a, output_b)
    output_a.square().mean().backward()
    assert predictor.cognitive_gate.grad is not None
    assert predictor.cognitive_gate.grad[1].abs() > 0

    with torch.no_grad():
        predictor.cognitive_gate[1] = 0.5
    assert not torch.allclose(
        predictor(carrier, cognition_a, scale=1, horizons=(1, 2)),
        predictor(carrier, cognition_b, scale=1, horizons=(1, 2)),
    )


def test_standardized_huber_updates_running_statistics_and_backpropagates_only_prediction():
    labels, mask, segments, target_segments, _ = _packed_fixture()
    config = CSTMArchitectureConfig(
        code_dimension=16,
        predictor_rank=4,
        horizon_blocks=(1, 2),
        target_rms_decay=0.5,
    )
    predictor = CausalSpectralTargetPredictor(8, 2, 32, config)
    targets = build_causal_spectral_targets(
        labels, mask, segments, target_segments, predictor.token_codes,
        torch.tensor([0, 1, 2]), support=2, horizons=(1, 2),
    )
    carrier = torch.randn(1, 3, 8, requires_grad=True)
    cognition = torch.randn(1, 3, 8, requires_grad=True)
    prediction = predictor(carrier, cognition, scale=0, horizons=(1, 2))
    report = predictor.loss(
        prediction,
        targets,
        scale=0,
        update_statistics=True,
    )
    report.loss.backward()

    assert report.loss.isfinite() and report.loss > 0
    assert report.valid_rows == int(targets.mask.sum())
    assert report.coefficient_targets == report.valid_rows * 3 * 16
    assert report.token_participations == report.valid_rows * 2
    assert predictor.target_rms_initialized[0].all()
    assert torch.isfinite(predictor.target_rms(
        0, device=carrier.device, dtype=carrier.dtype
    )).all()
    assert carrier.grad is not None and torch.isfinite(carrier.grad).all()
    assert targets.values.grad is None


def test_zero_valid_targets_produce_differentiable_exact_zero():
    labels, mask, segments, target_segments, _ = _packed_fixture()
    config = CSTMArchitectureConfig(
        code_dimension=16, predictor_rank=4, horizon_blocks=(1, 2)
    )
    predictor = CausalSpectralTargetPredictor(8, 2, 32, config)
    targets = build_causal_spectral_targets(
        labels, mask, segments, target_segments, predictor.token_codes,
        torch.tensor([5, 11]), support=4, horizons=(1, 2),
    )
    carrier = torch.randn(1, 2, 8, requires_grad=True)
    prediction = predictor(
        carrier, torch.zeros_like(carrier), scale=0, horizons=(1, 2)
    )
    report = predictor.loss(
        prediction, targets, scale=0, update_statistics=True
    )
    assert report.loss == 0
    assert report.valid_rows == report.coefficient_targets == 0
    report.loss.backward()
    torch.testing.assert_close(carrier.grad, torch.zeros_like(carrier))


def test_sampled_target_statistics_apply_declared_importance_weight():
    labels, mask, segments, target_segments, _ = _packed_fixture()
    config = CSTMArchitectureConfig(
        code_dimension=16,
        predictor_rank=4,
        horizon_blocks=(1, 2),
        target_rms_decay=0.0,
    )
    reference = CausalSpectralTargetPredictor(8, 2, 32, config)
    weighted = CausalSpectralTargetPredictor(8, 2, 32, config)
    targets = build_causal_spectral_targets(
        labels,
        mask,
        segments,
        target_segments,
        reference.token_codes,
        torch.tensor([3, 7]),
        support=4,
        horizons=(1, 2),
    )
    reference.update_target_statistics(0, targets)
    weighted.update_target_statistics(
        0, targets, importance_weight=2.5
    )
    torch.testing.assert_close(
        weighted.target_second_moment[0],
        reference.target_second_moment[0] * 2.5,
        atol=0,
        rtol=0,
    )
    with pytest.raises(ValueError, match="importance"):
        weighted.update_target_statistics(
            0, targets, importance_weight=float("nan")
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"code_dimension": 4},
        {"predictor_rank": 0},
        {"horizon_blocks": (2, 4)},
        {"horizon_blocks": (1, 2, 2)},
        {"target_rms_decay": 1.0},
        {"minimum_target_rms": 0.0},
    ),
)
def test_configuration_rejects_ambiguous_or_numerically_unsafe_contracts(kwargs):
    with pytest.raises(ValueError):
        CSTMArchitectureConfig(**kwargs)
