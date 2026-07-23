from dataclasses import replace

import pytest
import torch

from mrrn.cognitive_types import InternalAction, RelationFamily, SourceClass
from mrrn.controller import (
    AdaptiveController, GoalState, OperationalSchemas, SystemModelState,
)
from mrrn.cognitive_model import MultimodalRelationalContinuityResonanceNetwork
from mrrn.hypotheses import HypothesisBank, HypothesisState
from mrrn.metacognition import MetacognitivePrediction
from mrrn.uncertainty import (
    DistributionalPredictionHead, OnlineCalibration, UncertaintyEstimator,
    UncertaintyInputs, pinball_loss, selective_abstention,
)
from mrrn.world_model import ActionConditionedWorldModel, apply_intervention


def test_distributional_head_orders_quantiles_and_exposes_ensemble_disagreement():
    torch.manual_seed(71)
    head = DistributionalPredictionHead(8, 5, 2, ensemble_heads=4)
    features = torch.randn(3, 8, requires_grad=True)
    output = head(features)
    assert output.categorical_logits.shape == (3, 5)
    assert output.continuous_quantiles.shape == (3, 3, 2)
    assert bool((output.continuous_quantiles[:, 1:] >= output.continuous_quantiles[:, :-1]).all())
    assert bool((output.aleatoric >= 0).all() & (output.epistemic >= 0).all())
    loss = pinball_loss(output.continuous_quantiles, torch.randn(3, 2), output.quantile_levels)
    loss.backward()
    assert torch.isfinite(features.grad).all()


def test_uncertainty_channels_remain_decomposed_and_drive_abstention():
    estimator = UncertaintyEstimator()
    inputs = UncertaintyInputs(
        torch.ones(2, 3), torch.randn(2, 4, 3),
        torch.tensor([[0.5, 0.5], [0.99, 0.01]]),
        torch.tensor([[0.5, 0.5], [0.99, 0.01]]),
        torch.tensor([[1.0, 0.5], [2.0, 0.0]]),
        torch.ones(2, 2, dtype=torch.bool), torch.tensor([0.2, 0.0]),
        torch.tensor([0.8, 0.2]), torch.tensor([0.0, 1.0]), torch.tensor([0.1, 0.5]),
    )
    uncertainty = estimator(inputs)
    assert uncertainty.shape == (2, 8)
    assert uncertainty[0, 2] > uncertainty[1, 2]
    decision = selective_abstention(
        torch.tensor([1.0, -0.1]), uncertainty,
        torch.tensor([100.0, 100.0]),
    )
    assert decision.tolist() == [False, True]


def test_online_calibration_reports_ece_and_brier_by_group():
    calibration = OnlineCalibration(2, bins=5)
    state = calibration.initial_state()
    probabilities = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    state = calibration.update(
        state, probabilities, torch.tensor([0, 1, 1]), torch.tensor([0, 0, 1]),
        torch.tensor([True, True, True]),
    )
    report = calibration.report(state)
    assert report.counts.sum() == 3
    assert bool((report.expected_calibration_error >= 0).all())
    assert bool((report.brier_score >= 0).all())


def test_hypothesis_weights_update_in_log_space_and_prune_with_hysteresis():
    torch.manual_seed(73)
    bank = HypothesisBank(6, 4, 3, len(RelationFamily), 2, 8, prune_hysteresis=2)
    state = bank.initial_state(1)
    for _ in range(3):
        state = bank.create(state, torch.randn(1, 6), torch.tensor([True]))
    assert state.active.sum() == 3
    assert len(set(state.scenario_ids[state.active].tolist())) == 3
    state = bank.update_evidence(state, torch.tensor([[0.0, -8.0, -8.0, 0.0]]))
    once = bank.prune(state)
    assert once.active.sum() == 3
    state = bank.update_evidence(state, torch.tensor([[0.0, -8.0, -8.0, 0.0]]))
    pruned = bank.prune(state)
    assert pruned.active.sum() == 1
    torch.testing.assert_close(pruned.weights.sum(-1), torch.ones(1))
    assert pruned.effective_count.item() >= 1


def test_hypothesis_duplicate_merge_and_slot_matching_preserve_identity():
    bank = HypothesisBank(4, 3, 2, len(RelationFamily), 2, 8, merge_similarity=0.99)
    state = bank.initial_state(1)
    state = bank.create(state, torch.ones(1, 4), torch.tensor([True]))
    state = bank.create(state, torch.ones(1, 4), torch.tensor([True]))
    values = {name: getattr(state, name).clone() if isinstance(getattr(state, name), torch.Tensor) else getattr(state, name) for name in state.__dataclass_fields__}
    values["residuals"][0, 1] = values["residuals"][0, 0]
    state = HypothesisState(**values)
    merged = bank.merge_duplicates(state)
    assert merged.active.sum() == 1
    distinct = {name: getattr(state, name).clone() if isinstance(getattr(state, name), torch.Tensor) else getattr(state, name) for name in state.__dataclass_fields__}
    distinct["residuals"][0, 0] = torch.tensor([1., 0., 0., 0.])
    distinct["residuals"][0, 1] = torch.tensor([0., 1., 0., 0.])
    distinct = HypothesisState(**distinct)
    proposed = distinct.residuals[:, torch.tensor([1, 0, 2])]
    mask = distinct.active[:, torch.tensor([1, 0, 2])]
    assignment = bank.match_slots(distinct, proposed, mask)
    assert assignment[0, :2].tolist() == [1, 0]


def test_world_model_is_action_conditioned_distributional_and_differentiable():
    torch.manual_seed(79)
    model = ActionConditionedWorldModel(8, 3, len(RelationFamily), 5, horizons=(1, 2, 4))
    latent = torch.randn(2, 8, requires_grad=True)
    graph = torch.randn(2, 8)
    first = model(latent, graph, torch.zeros(2, 3))
    second = model(latent, graph, torch.ones(2, 3))
    assert first.latent_mean.shape == (2, 3, 8)
    assert first.relation_logits.shape == (2, 3, len(RelationFamily), 3)
    assert first.reward_quantiles.shape == (2, 3, 3)
    assert bool((first.reward_quantiles[..., 1:] >= first.reward_quantiles[..., :-1]).all())
    assert not torch.allclose(first.latent_mean, second.latent_mean)
    first.latent_mean.square().mean().backward()
    assert torch.isfinite(latent.grad).all()


def test_counterfactual_rollout_is_scenario_isolated_and_simulated():
    model = ActionConditionedWorldModel(6, 2, len(RelationFamily), 3, horizons=(1, 2))
    rollout = model.rollout(
        torch.zeros(1, 6), torch.zeros(1, 6), torch.randn(1, 2, 6),
        torch.tensor([[4, 5]]), torch.randn(1, 2, 3, 2),
        torch.tensor([[[True, True, True], [True, False, False]]]),
    )
    assert rollout.latent_states.shape == (1, 2, 3, 6)
    assert rollout.scenario_ids[0, :, 0].tolist() == [4, 5]
    assert bool((rollout.source_classes == int(SourceClass.SIMULATED)).all())
    torch.testing.assert_close(rollout.latent_states[0, 1, 1], rollout.latent_states[0, 1, 2])


def test_intervention_requires_explicit_parents_and_causal_authority():
    values = torch.zeros(2, 3, 4)
    edges = torch.zeros(2, 3, 3, dtype=torch.bool)
    edges[:, 0, 1] = True
    result = apply_intervention(
        values, edges, torch.tensor([1, 2]), torch.ones(2, 4),
        torch.tensor([True, True]),
    )
    assert result.causal_intervention.tolist() == [True, False]
    assert not result.incoming_causal_edges[0, :, 1].any()
    assert result.conditional_simulation.tolist() == [False, True]


def goals():
    return GoalState(
        torch.randn(1, 2, 4), torch.zeros(1, 2, 2), torch.tensor([[1.0, 0.5]]),
        torch.tensor([[4.0, 8.0]]), torch.ones(1, 2), torch.zeros(1, 2),
        torch.tensor([[True, True]]),
    )


def test_conflicting_goals_remain_explicit_and_are_not_vector_averaged():
    desired = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    conflicts = torch.tensor([[[False, True], [True, False]]])
    state = GoalState(
        desired, torch.zeros(1, 2, 1), torch.tensor([[2.0, 1.0]]),
        torch.ones(1, 2), torch.ones(1, 2), torch.zeros(1, 2),
        torch.ones(1, 2, dtype=torch.bool),
        provenance_ids=torch.tensor([[10, 11]]),
        conflict_mask=conflicts,
    )
    torch.testing.assert_close(state.summary(), torch.tensor([[1.0, 0.0]]))
    assert state.conflict_mask[0, 0, 1]


def system():
    return SystemModelState(
        torch.ones(1, 2), torch.ones(1, 3), torch.ones(1, 3), torch.zeros(1, 3),
        torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 3),
        torch.ones(1, 3), torch.ones(1, 3),
        torch.ones(1, 1), torch.ones(1, 1), torch.ones(1, 1), torch.ones(1, 1),
        torch.ones(1, 3, dtype=torch.bool), torch.zeros(1, 2),
    )


def test_controller_factorizes_arguments_obeys_masks_and_hard_compute_budget():
    sys = system()
    controller = AdaptiveController(
        8, 4, 8, sys.features().shape[-1], 5, maximum_steps=4,
    )
    controller.halt_head.bias.data.fill_(-20)
    allowed = torch.zeros(1, len(InternalAction), dtype=torch.bool)
    allowed[:, int(InternalAction.COMPARE)] = True
    allowed[:, int(InternalAction.HALT)] = True
    uncertainty = torch.randn(1, 8, requires_grad=True)
    rollout = controller(
        torch.randn(1, 8), goals().summary(), uncertainty, sys.features(),
        torch.randn(1, 5, 8), torch.ones(1, 5, dtype=torch.bool), action_mask=allowed,
    )
    assert len(rollout.decisions) == 4
    assert rollout.state.remaining_steps.item() == 0
    assert rollout.state.history_mask.sum() == 4
    for decision in rollout.decisions:
        assert decision.action.item() in (int(InternalAction.COMPARE), int(InternalAction.HALT))
        assert decision.node_pointer.shape == (1,)
        assert decision.relation_family.shape == (1,)
        assert decision.memory_tier.shape == (1,)
    rollout.ponder_cost.backward()
    assert uncertainty.grad is None


def test_hypothesis_router_is_bounded_preserves_unknown_and_reports_mass():
    bank = HypothesisBank(4, 4, 1, 2, 4, 2)
    state = bank.initial_state(1)
    for index in range(4):
        state = bank.create(state, torch.eye(4)[index:index + 1], torch.tensor([True]))
    values = {name: getattr(state, name).clone() if isinstance(getattr(state, name), torch.Tensor) else getattr(state, name)
              for name in state.__dataclass_fields__}
    values["log_weights"][0] = torch.tensor([0.55, 0.25, 0.15, 0.05]).log()
    values["unknown"][0].zero_(); values["unknown"][0, 3] = True
    state = HypothesisState(**values)
    routed = bank.route(state, 2, diversity_penalty=0.0)
    assert routed.mask.sum() == 2
    assert 3 in routed.indices[0].tolist()
    assert routed.posterior_mass.item() == pytest.approx(0.60)


def test_controller_adaptive_halting_and_operational_schema_entropy_floor():
    sys = system()
    controller = AdaptiveController(8, 4, 8, sys.features().shape[-1], 3, maximum_steps=4)
    controller.halt_head.bias.data.fill_(20)
    rollout = controller(
        torch.zeros(1, 8), goals().summary(), torch.zeros(1, 8), sys.features(),
        torch.zeros(1, 3, 8), torch.ones(1, 3, dtype=torch.bool),
    )
    assert len(rollout.decisions) == 1 and rollout.state.halted.item()
    schemas = OperationalSchemas(8, 5, entropy_floor=0.2)
    _, schema_state = schemas(torch.randn(2, 8))
    assert bool((schema_state.probabilities >= 0.2 / 5 - 1e-7).all())
    torch.testing.assert_close(schema_state.probabilities.sum(-1), torch.ones(2))


def test_metacognitive_value_bias_routes_operations_but_cannot_bypass_hard_masks():
    prediction = MetacognitivePrediction(
        torch.tensor([0.1]), torch.tensor([0.1]), torch.tensor([10.0]),
        torch.tensor([0.1]), torch.tensor([0.1]), torch.tensor([0.1]),
        torch.tensor([0.0]), torch.zeros(1, 12),
    )
    bias = MultimodalRelationalContinuityResonanceNetwork._metacognitive_action_bias(
        prediction
    )
    assert bias[0, int(InternalAction.RETRIEVE_EPISODIC)] > bias[0, int(InternalAction.COMPARE)]
    assert bias.abs().max() <= 1

    controller = AdaptiveController(4, 4, 2, system().features().shape[-1], 2, maximum_steps=1)
    controller.action_head.weight.data.zero_()
    controller.action_head.bias.data.zero_()
    controller.halt_head.bias.data.fill_(-20)
    allowed = torch.zeros(1, len(InternalAction), dtype=torch.bool)
    allowed[0, int(InternalAction.COMPARE)] = True
    decision, _ = controller.step(
        controller.initial_state(1), torch.zeros(1, 4), torch.zeros(1, 4),
        torch.zeros(1, 2), system().features(), torch.zeros(1, 2, 4),
        torch.ones(1, 2, dtype=torch.bool), action_mask=allowed, action_bias=bias,
    )
    assert decision.action.item() == int(InternalAction.COMPARE)


def test_controller_rejects_nonfinite_or_misshaped_metacognitive_bias():
    controller = AdaptiveController(4, 4, 2, system().features().shape[-1], 2, maximum_steps=1)
    arguments = (
        controller.initial_state(1), torch.zeros(1, 4), torch.zeros(1, 4),
        torch.zeros(1, 2), system().features(), torch.zeros(1, 2, 4),
        torch.ones(1, 2, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="action bias"):
        controller.step(*arguments, action_bias=torch.zeros(1, 2))
    invalid = torch.zeros(1, len(InternalAction))
    invalid[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        controller.step(*arguments, action_bias=invalid)


def test_reasoning_contracts_fail_closed():
    with pytest.raises(ValueError):
        DistributionalPredictionHead(4, 2, 1, quantile_levels=(0.9, 0.1))
    with pytest.raises(ValueError):
        ActionConditionedWorldModel(4, 2, 3, 2, horizons=(2, 1))
    with pytest.raises(ValueError):
        GoalState(
            torch.zeros(1, 1, 2), torch.zeros(1, 1, 1), torch.ones(1, 1),
            torch.zeros(1, 1), torch.ones(1, 1), torch.zeros(1, 1),
            torch.ones(1, 1, dtype=torch.bool),
        )
