from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from math import ceil, floor, sqrt
from statistics import mean, stdev

import pytest
import torch

from mrrn.cstm_schedule import (
    CSTMCoverageState,
    CSTMObligation,
    deterministic_cstm_rows,
    deterministic_cstm_sample,
)


DIGEST = sha256(b"fixed-target-authority").hexdigest()


def obligations() -> tuple[CSTMObligation, ...]:
    return (
        CSTMObligation(1, 0, 2.0),
        CSTMObligation(1, 2, 3.0),
        CSTMObligation(3, 1, 5.0),
    )


def test_counter_sampling_is_restart_exact_and_global_rng_free():
    torch.manual_seed(1729)
    before = torch.random.get_rng_state().clone()
    first = deterministic_cstm_sample(
        obligations(),
        duty_probability=0.25,
        uniform_mixture=0.05,
        seed=17,
        optimizer_step=31,
        target_digest=DIGEST,
    )
    second = deterministic_cstm_sample(
        obligations(),
        duty_probability=0.25,
        uniform_mixture=0.05,
        seed=17,
        optimizer_step=31,
        target_digest=DIGEST,
    )
    assert first == second
    assert torch.equal(torch.random.get_rng_state(), before)


def test_sampling_emits_at_most_one_physical_invocation_scale_obligation():
    for step in range(100):
        decision = deterministic_cstm_sample(
            obligations(),
            duty_probability=0.7,
            uniform_mixture=0.1,
            seed=91,
            optimizer_step=step,
            target_digest=DIGEST,
        )
        assert decision.obligation_count == 3
        assert decision.obligation is None or (
            decision.obligation.invocation,
            decision.obligation.scale,
        ) in {(1, 0), (1, 2), (3, 1)}


@pytest.mark.parametrize("duty", (0.1, 0.25, 0.4, 0.7, 1.0))
def test_systematic_duty_has_floor_or_ceil_count_in_every_contiguous_window(
    duty,
):
    decisions = tuple(
        deterministic_cstm_sample(
            obligations(),
            duty_probability=duty,
            uniform_mixture=0.1,
            seed=91,
            optimizer_step=step,
            target_digest=DIGEST,
        ).active
        for step in range(257)
    )
    for start in range(len(decisions)):
        for width in (1, 2, 3, 4, 7, 16, 31, 64):
            if start + width > len(decisions):
                continue
            count = sum(decisions[start : start + width])
            assert floor(width * duty) <= count <= ceil(width * duty)


def test_systematic_duty_preserves_first_order_inclusion_across_seed_phases():
    duty = 0.25
    for step in (0, 1, 2, 7, 31):
        active = sum(
            deterministic_cstm_sample(
                obligations(),
                duty_probability=duty,
                seed=seed,
                optimizer_step=step,
                target_digest=DIGEST,
            ).active
            for seed in range(4096)
        )
        assert active / 4096 == pytest.approx(duty, abs=0.025)


def test_complete_categorical_supercycle_covers_every_valid_scale():
    items = obligations()
    total = sum(item.dense_weight for item in items)
    mixture = 0.05
    probabilities = tuple(
        (1 - mixture) * item.dense_weight / total
        + mixture / len(items)
        for item in items
    )
    selected = []
    left = 0.0
    for probability in probabilities:
        decision = deterministic_cstm_sample(
            items,
            duty_probability=1.0,
            uniform_mixture=mixture,
            seed=0,
            optimizer_step=0,
            target_digest=DIGEST,
            uniform_override=(0.5, left + probability / 2),
        )
        assert decision.active
        assert decision.conditional_probability == pytest.approx(
            probability, abs=2e-15
        )
        selected.append(decision.obligation)
        left += probability
    assert tuple(selected) == items
    assert {item.scale for item in selected} == {0, 1, 2}


def test_exhaustive_conditional_estimator_equals_dense_objective():
    """Integrating every categorical interval proves exact unbiasedness."""

    items = obligations()
    duty = 0.4
    mixture = 0.15
    dense_numerators = (7.0, 11.0, 19.0)
    dense_denominator = sum(item.dense_weight for item in items)
    probabilities = tuple(
        (1 - mixture) * item.dense_weight / dense_denominator
        + mixture / len(items)
        for item in items
    )
    expected = 0.0
    left = 0.0
    for index, probability in enumerate(probabilities):
        midpoint = left + probability / 2
        decision = deterministic_cstm_sample(
            items,
            duty_probability=duty,
            uniform_mixture=mixture,
            seed=1,
            optimizer_step=0,
            target_digest=DIGEST,
            uniform_override=(duty / 2, midpoint),
        )
        assert decision.active
        assert decision.obligation == items[index]
        estimator = (
            dense_numerators[index]
            / dense_denominator
            * decision.inverse_probability
        )
        expected += duty * probability * estimator
        left += probability
    assert expected == pytest.approx(
        sum(dense_numerators) / dense_denominator,
        abs=2e-15,
    )


def test_exhaustive_importance_weighted_gradient_equals_dense_gradient():
    items = obligations()
    duty = 0.35
    mixture = 0.2
    denominator = sum(item.dense_weight for item in items)
    probabilities = tuple(
        (1 - mixture) * item.dense_weight / denominator
        + mixture / len(items)
        for item in items
    )
    parameter = torch.tensor([0.3, -0.7], dtype=torch.float64, requires_grad=True)
    group_losses = (
        (parameter.square() * torch.tensor([2.0, 1.0])).sum(),
        ((parameter - 0.4).square() * torch.tensor([1.0, 3.0])).sum(),
        (parameter.sin() * torch.tensor([4.0, -2.0])).sum(),
    )
    dense = sum(group_losses) / denominator
    dense_gradient = torch.autograd.grad(
        dense, parameter, retain_graph=True,
    )[0]
    expected_gradient = torch.zeros_like(parameter)
    left = 0.0
    for index, probability in enumerate(probabilities):
        decision = deterministic_cstm_sample(
            items,
            duty_probability=duty,
            uniform_mixture=mixture,
            seed=2,
            optimizer_step=0,
            target_digest=DIGEST,
            uniform_override=(duty / 2, left + probability / 2),
        )
        estimator = (
            group_losses[index]
            / denominator
            * decision.inverse_probability
        )
        gradient = torch.autograd.grad(
            estimator, parameter, retain_graph=True,
        )[0]
        expected_gradient += duty * probability * gradient
        left += probability
    torch.testing.assert_close(
        expected_gradient, dense_gradient, atol=2e-15, rtol=2e-15,
    )


def test_monte_carlo_hierarchical_gradient_confidence_interval_contains_dense():
    items = obligations()
    duty = 0.25
    mixture = 0.05
    denominator = sum(item.dense_weight for item in items)
    # One scalar gradient contribution per obligation is sufficient to test
    # the exact hierarchy/inclusion algebra independently of autograd.
    gradients = (2.0, -1.0, 4.0)
    dense = sum(gradients) / denominator
    samples = []
    for step in range(20_000):
        decision = deterministic_cstm_sample(
            items,
            duty_probability=duty,
            uniform_mixture=mixture,
            seed=8128,
            optimizer_step=step,
            target_digest=DIGEST,
        )
        if not decision.active:
            samples.append(0.0)
            continue
        index = items.index(decision.obligation)
        samples.append(
            gradients[index]
            / denominator
            * decision.inverse_probability
        )
    estimate = mean(samples)
    standard_error = stdev(samples) / sqrt(len(samples))
    assert estimate - 4.0 * standard_error <= dense
    assert dense <= estimate + 4.0 * standard_error
    assert estimate == pytest.approx(dense, abs=0.025)


def test_duty_inactive_receipt_has_no_false_supervision_claim():
    decision = deterministic_cstm_sample(
        obligations(),
        duty_probability=0.2,
        seed=1,
        optimizer_step=0,
        target_digest=DIGEST,
        uniform_override=(0.9, 0.0),
    )
    assert not decision.active
    assert decision.obligation is None
    assert decision.inclusion_probability == 0
    assert decision.inverse_probability == 0


def test_corrupt_inclusion_probability_and_sampler_version_fail_closed():
    decision = deterministic_cstm_sample(
        obligations(),
        duty_probability=1.0,
        seed=3,
        optimizer_step=0,
        target_digest=DIGEST,
        uniform_override=(0.0, 0.0),
    )
    with pytest.raises(ValueError, match="inconsistent"):
        replace(
            decision,
            inclusion_probability=decision.inclusion_probability * 0.5,
        )
    with pytest.raises(ValueError, match="malformed"):
        replace(decision, schema_version=999)


def test_row_sampling_is_bounded_restart_exact_and_rng_free():
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    first = deterministic_cstm_rows(
        101, 17, counter_digest=DIGEST, stream=4
    )
    second = deterministic_cstm_rows(
        101, 17, counter_digest=DIGEST, stream=4
    )
    assert first == second
    assert torch.equal(torch.random.get_rng_state(), before)
    assert len(first.selected_indices) == 17
    assert first.inclusion_probability == pytest.approx(17 / 101)
    assert first.inverse_probability == pytest.approx(101 / 17)


def test_row_importance_estimator_is_exact_over_every_cyclic_start():
    population, budget = 11, 4
    values = torch.arange(1, population + 1, dtype=torch.float64)
    expected = values.sum()
    estimate = 0.0
    # For any coprime stride, averaging the k-position cyclic window over all
    # starts includes every row exactly k times.
    stride = 3
    for start in range(population):
        indices = {
            (start + offset * stride) % population
            for offset in range(budget)
        }
        estimate += float(values[list(indices)].sum()) * population / budget
    estimate /= population
    assert estimate == pytest.approx(float(expected), abs=1e-12)


@pytest.mark.parametrize(
    "population,budget,digest,stream",
    ((0, 1, DIGEST, 0), (3, 0, DIGEST, 0), (3, 1, "short", 0), (3, 1, DIGEST, -1)),
)
def test_row_sampling_contract_fails_closed(
    population, budget, digest, stream,
):
    with pytest.raises(ValueError, match="row sampling"):
        deterministic_cstm_rows(
            population,
            budget,
            counter_digest=digest,
            stream=stream,
        )


def test_coverage_state_round_trips_and_reports_declared_starvation_gap():
    state = CSTMCoverageState()
    state.declare_required(("scale:0", "scale:1", "horizon:1"))
    decision = deterministic_cstm_sample(
        obligations(),
        duty_probability=1.0,
        seed=1,
        optimizer_step=0,
        target_digest=DIGEST,
        uniform_override=(0.0, 0.01),
    )
    state.record_predictor(
        decision, optimizer_step=3, horizons=(1,)
    )
    state.record_substrate(decision)
    restored = CSTMCoverageState.from_state_dict(state.state_dict())
    assert restored.state_dict() == state.state_dict()
    assert restored.predictor_updates == 1
    assert restored.substrate_updates == 1
    assert restored.maximum_gap(optimizer_step=8) == 9


@pytest.mark.parametrize(
    "mutation",
    (
        lambda state: state.update(schema_version=999),
        lambda state: state.update(coverage_counts={"scale:0": -1}),
        lambda state: state.update(required_keys=[""]),
        lambda state: state.update(last_obligation_digest="short"),
    ),
)
def test_coverage_state_corruption_fails_closed(mutation):
    serialized = CSTMCoverageState().state_dict()
    mutation(serialized)
    with pytest.raises(ValueError, match="coverage"):
        CSTMCoverageState.from_state_dict(serialized)


@pytest.mark.parametrize(
    "items, kwargs, message",
    (
        (
            (CSTMObligation(1, 0, 1.0), CSTMObligation(1, 0, 2.0)),
            {},
            "unique",
        ),
        ((), {"duty_probability": 0.0}, "duty"),
        ((), {"uniform_mixture": 1.0}, "mixture"),
    ),
)
def test_sampling_contract_fails_closed(items, kwargs, message):
    options = dict(
        duty_probability=0.5,
        uniform_mixture=0.05,
        seed=1,
        optimizer_step=0,
        target_digest=DIGEST,
    )
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        deterministic_cstm_sample(items, **options)
