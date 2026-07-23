import torch

from mrrn.abstraction_control import (
    AbstractionApplicabilityHead, AbstractionLevelSelector,
    AbstractionValidityState, LocalizedDescentPlanner,
)


def validity_state() -> AbstractionValidityState:
    state = AbstractionValidityState.empty(1, 3)
    values = {
        name: getattr(state, name).clone()
        for name in state.__dataclass_fields__
    }
    values["applicability"][0] = torch.tensor([0.9, 0.9, 0.2])
    values["reconstruction_distortion"][0] = torch.tensor([0.1, 0.1, 0.1])
    values["relation_distortion"][0] = torch.tensor([0.1, 0.1, 0.1])
    values["task_distortion"][0] = torch.tensor([0.1, 0.1, 0.1])
    values["provenance_sufficiency"][0] = 1
    values["precision_sufficiency"][0] = 1
    values["calibrated_confidence"][0] = torch.tensor([0.9, 0.8, 0.99])
    values["abstraction_depths"][0] = torch.tensor([1, 3, 5])
    values["physical_scales"][0] = torch.tensor([0, 2, 1])
    values["abstraction_node_indices"][0] = torch.tensor([2, 4, 6])
    values["provenance_ids"][0] = torch.tensor([10, 11, 12])
    values["versions"][0] = 1
    values["active"][0] = True
    return AbstractionValidityState(**values)


def test_highest_valid_abstraction_is_selected_and_invalid_higher_level_is_rejected():
    selected = AbstractionLevelSelector()(validity_state(),
        task_tolerance=torch.tensor([0.2]),
        reconstruction_tolerance=torch.tensor([0.2]),
        relation_tolerance=torch.tensor([0.2]),
        required_precision=torch.tensor([0.5]),
    )
    assert selected.mask.tolist() == [True]
    assert selected.validity_indices.tolist() == [1]
    assert selected.abstraction_node_indices.tolist() == [4]
    assert selected.abstraction_depths.tolist() == [3]
    assert selected.physical_scales.tolist() == [2]
    assert not selected.needs_descent.any()


def test_unknown_or_insufficient_validity_fails_closed_and_plans_local_descent():
    state = validity_state()
    selected = AbstractionLevelSelector()(state,
        task_tolerance=torch.tensor([0.01]),
        reconstruction_tolerance=torch.tensor([0.01]),
        relation_tolerance=torch.tensor([0.01]),
        required_precision=torch.tensor([1.0]),
    )
    assert selected.needs_descent.tolist() == [True]
    support = torch.zeros(1, 8, 3)
    support[0, 4] = torch.tensor([3.0, 5.0, 5.0])
    plan = LocalizedDescentPlanner()(
        selected, support, fallback_node_indices=torch.tensor([4]),
        requested_precision=torch.tensor([0.01]),
        trigger_reasons=torch.tensor([7]),
    )
    assert plan.mask.tolist() == [True]
    assert plan.abstraction_node_indices.tolist() == [4]
    torch.testing.assert_close(plan.requested_support[0], support[0, 4])
    # The planner leaves physical scale independent of abstraction depth.
    assert plan.target_physical_scales.tolist() == [-1]


def test_applicability_head_is_evidence_and_goal_conditioned_with_gradients():
    torch.manual_seed(431)
    head = AbstractionApplicabilityHead(8)
    abstraction = torch.randn(2, 8, requires_grad=True)
    observed = torch.randn(2, 8)
    relation = torch.randn(2, 8)
    hypothesis = torch.randn(2, 8)
    goal = torch.randn(2, 8)
    baseline = head(abstraction, observed, relation, hypothesis, goal)
    changed = head(abstraction, observed + 2, relation, hypothesis, goal + 1)
    assert not torch.allclose(baseline.applicability, changed.applicability)
    (baseline.applicability.mean() + baseline.task_distortion.mean()).backward()
    assert abstraction.grad is not None
    assert torch.isfinite(abstraction.grad).all()
