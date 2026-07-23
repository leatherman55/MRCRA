from dataclasses import replace

import torch

from mrrn.action_candidates import (
    authorize_candidate_provenance, build_action_candidates,
    evaluate_candidate_rollout, select_action_candidate,
)
from mrrn.cognitive_model import MultimodalRelationalContinuityResonanceNetwork
from mrrn.cognitive_types import ModalityClass, SourceClass, SupportInterval, VerificationClass
from mrrn.config import MRCRAConfig
from mrrn.interaction import decision_from_candidates
from mrrn.hypotheses import HypothesisState
from mrrn.provenance import ProvenanceLedger
from mrrn.world_model import ActionConditionedWorldModel
from test_cognitive_actions import action_config
from test_cognitive_model import force_events, packet


def candidate_fixture():
    ledger = ProvenanceLedger()
    provenance = ledger.append(
        source_class=SourceClass.EXTERNAL,
        source_uri_or_episode="test://candidate/evidence",
        support=SupportInterval(0, 0, 0), modality=ModalityClass.TEXT,
        operator="test:evidence", scenario_id=0, model_authority="test",
        verification=VerificationClass.EXTERNALLY_CHECKED,
    )
    state = build_action_candidates(
        proposal_logits=torch.tensor([[10.0, 0.0]]),
        expected_reward=torch.zeros(1, 2), expected_cost=torch.zeros(1, 2),
        constraint_probability=torch.zeros(1, 2),
        expected_success=torch.ones(1, 2),
        available=torch.ones(1, 2, dtype=torch.bool),
        permission_mask=torch.ones(1, 2, dtype=torch.bool),
        supporting_provenance_ids=torch.tensor([[provenance]]),
        supporting_mask=torch.tensor([[True]]), capacity=2, argument_dim=3,
    )
    return ledger, state


def test_multi_hypothesis_consequences_reverse_the_pre_deliberation_favorite():
    ledger, candidates = candidate_fixture()
    # Hypothesis zero favors action 1 strongly, hypothesis one weakly favors 0.
    rewards = torch.tensor([[[[[-12.0, -10.0, -8.0]], [[4.0, 5.0, 6.0]]],
                             [[[1.0, 2.0, 3.0]], [[3.0, 4.0, 5.0]]]]])
    lattice = rewards.shape[:4]
    zeros = torch.zeros(lattice)
    candidates = evaluate_candidate_rollout(
        candidates, reward_quantiles=rewards, costs=zeros,
        constraint_probabilities=zeros, success_probabilities=torch.ones(lattice),
        uncertainty=torch.ones(lattice),
        hypothesis_weights=torch.tensor([[0.8, 0.2]]),
        rollout_mask=torch.ones(lattice, dtype=torch.bool),
    )
    candidates = authorize_candidate_provenance(candidates, ledger)
    values = {name: getattr(candidates, name) for name in candidates.__dataclass_fields__}
    values["viability_authorized"] = candidates.active.clone()
    candidates = type(candidates)(**values)
    candidates = select_action_candidate(candidates)
    decision = decision_from_candidates(
        candidates, action_count=2, active_mask=torch.tensor([True])
    )
    assert candidates.schema_ids[0, 0] == 0  # initial routed favorite
    assert decision.selected_action.tolist() == [1]  # consequence-aware result
    assert decision.authorized.tolist() == [True]


def test_candidate_world_model_is_bounded_over_hypotheses_actions_and_horizons():
    torch.manual_seed(443)
    model = ActionConditionedWorldModel(8, 3, 16, 8, horizons=(1, 2, 4))
    rollout = model.rollout_candidates(
        torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 3, 8),
        torch.tensor([[1, 2, 3], [4, 5, 0]]),
        torch.tensor([[True, True, True], [True, True, False]]),
        torch.randn(2, 4, 3),
        torch.tensor([[True, True, False, True], [True, False, True, True]]),
    )
    assert rollout.latent_states.shape == (2, 3, 4, 3, 8)
    assert rollout.reward_quantiles.shape == (2, 3, 4, 3, 3)
    assert rollout.valid_mask.sum() == (3 * 3 + 2 * 3) * 3
    assert bool((rollout.source_classes[rollout.valid_mask] == int(SourceClass.SIMULATED)).all())
    assert bool((rollout.scenario_ids[rollout.valid_mask] > 0).all())


def test_observation_likelihood_automatically_updates_all_active_hypotheses():
    model = MultimodalRelationalContinuityResonanceNetwork(action_config()).eval()
    state = model.hypothesis_bank.initial_state(1)
    state = model.hypothesis_bank.create(state, torch.ones(1, 8), torch.tensor([True]))
    state = model.hypothesis_bank.create(state, -torch.ones(1, 8), torch.tensor([True]))
    values = {
        name: getattr(state, name).clone() if isinstance(getattr(state, name), torch.Tensor)
        else getattr(state, name)
        for name in state.__dataclass_fields__
    }
    values["predicted_outcomes"][0, 0] = 0
    values["predicted_outcomes"][0, 1] = 4
    values["uncertainty"][0, :2] = 1
    state = HypothesisState(**values)
    updated = model._update_hypotheses_from_observation(
        state, torch.zeros(1, 8), torch.tensor([True])
    )
    assert updated.active.sum() == 2
    assert updated.weights[0, 0] > updated.weights[0, 1]
    assert updated.supporting_evidence[0, 0] > 0
    assert updated.contradicting_evidence[0, 1] > 0


def test_production_cycle_calls_external_selection_after_internal_deliberation(monkeypatch):
    base = action_config()
    cognitive = replace(
        base.cognitive, enable_post_deliberation_action_selection=True,
        enable_multi_hypothesis_planning=True,
    )
    model = MultimodalRelationalContinuityResonanceNetwork(MRCRAConfig(
        base.carrier, cognitive, actor_parameter_minimum=1,
        actor_parameter_maximum=10_000_000,
    )).eval()
    force_events(model)
    order = []
    original_internal = model._run_internal_actions
    original_policy = model.external_action_policy.forward

    def internal(*args, **kwargs):
        order.append("internal")
        return original_internal(*args, **kwargs)

    def policy(*args, **kwargs):
        order.append("external_policy")
        return original_policy(*args, **kwargs)

    monkeypatch.setattr(model, "_run_internal_actions", internal)
    monkeypatch.setattr(model.external_action_policy, "forward", policy)
    ledger = ProvenanceLedger()
    observed = packet(torch.randn(1, 2, 8), ledger)
    state = model.initial_state(1)
    goals = replace(
        state.goals,
        authority=torch.ones_like(state.goals.authority),
        priorities=torch.ones_like(state.goals.priorities),
        horizons=torch.ones_like(state.goals.horizons),
        mask=torch.ones_like(state.goals.mask),
        status=torch.ones_like(state.goals.status),
    )
    system = replace(
        state.system_model,
        action_availability=torch.ones_like(state.system_model.action_availability),
        permission_mask=torch.ones_like(state.system_model.permission_mask),
    )
    model(observed, ledger, state=state, goals=goals, system_model=system)
    assert "internal" in order and "external_policy" in order
    assert order.index("internal") < order.index("external_policy")


def test_post_deliberation_action_path_is_dormant_without_goal_and_host_authority(monkeypatch):
    base = action_config()
    cognitive = replace(
        base.cognitive, enable_post_deliberation_action_selection=True,
        enable_multi_hypothesis_planning=True,
    )
    model = MultimodalRelationalContinuityResonanceNetwork(MRCRAConfig(
        base.carrier, cognitive, actor_parameter_minimum=1,
        actor_parameter_maximum=10_000_000,
    )).eval()
    force_events(model)
    calls = []
    monkeypatch.setattr(
        model.external_action_policy, "forward",
        lambda *args, **kwargs: calls.append(True),
    )
    ledger = ProvenanceLedger()
    output = model(packet(torch.randn(1, 2, 8), ledger), ledger)
    assert calls == []
    assert not output.external_action.authorized.any()
