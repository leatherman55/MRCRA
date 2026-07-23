import pytest
import torch

from mrrn.cognitive_objectives import (
    CognitiveObjectiveSchedule, MetacognitiveTargets, ObjectiveFamily, ObjectiveTerm, brier_rows,
    combine_cognitive_objectives, contrastive_binding_loss, focal_binary_loss,
    gradient_conflicts, hypothesis_diversity_loss, masked_categorical_nll,
    metacognitive_objectives, quantile_coverage_penalty,
    relation_residual_decorrelation,
)


def test_masked_modular_objective_normalizes_families_and_respects_curriculum():
    anchor = torch.tensor([[1.0, 3.0], [100.0, 100.0]], requires_grad=True)
    mask = torch.tensor([[True, True], [False, False]])
    terms = (
        ObjectiveTerm("task", ObjectiveFamily.PRIMARY_TASK, anchor, mask),
        ObjectiveTerm("events", ObjectiveFamily.EVENTS_RELATIONS, anchor * 2, mask),
        ObjectiveTerm("relations", ObjectiveFamily.EVENTS_RELATIONS, anchor * 4, mask),
        ObjectiveTerm("future", ObjectiveFamily.CONTROLLER_CONSEQUENCE, anchor * 100, torch.zeros_like(mask)),
    )
    stage2 = combine_cognitive_objectives(terms, CognitiveObjectiveSchedule.curriculum(2))
    assert stage2.terms["task"].item() == 2
    assert stage2.family_totals[ObjectiveFamily.EVENTS_RELATIONS].item() == 6
    assert stage2.family_totals[ObjectiveFamily.CONTROLLER_CONSEQUENCE].item() == 0
    assert stage2.total.item() == 8
    stage2.total.backward()
    assert torch.isfinite(anchor.grad).all()


def test_event_relation_binding_hypothesis_and_calibration_losses_are_functional():
    logits = torch.tensor([[[2., 0.], [0., 2.]]])
    targets = torch.tensor([[0, 1]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    assert bool((masked_categorical_nll(logits, targets, mask) < 0.2).all())
    focal = focal_binary_loss(
        torch.tensor([[5., -5.]]), torch.tensor([[1., 0.]]), mask,
    )
    assert focal.max() < 1e-4

    residuals = torch.tensor([[[1., 0.], [0., 1.], [1., 0.]]])
    type_ids = torch.tensor([[0, 1, 1]])
    assert relation_residual_decorrelation(residuals, type_ids, torch.ones_like(mask[:, :1]).expand(1, 3)) >= 0
    binding = contrastive_binding_loss(
        torch.eye(3).unsqueeze(0), torch.eye(3).unsqueeze(0),
        torch.tensor([[0, 1, 2]]), torch.ones(1, 3, dtype=torch.bool),
    )
    assert binding.mean() < 0.01
    diverse = hypothesis_diversity_loss(
        torch.eye(3).unsqueeze(0), torch.full((1, 3), 1 / 3),
        torch.ones(1, 3, dtype=torch.bool),
    )
    collapsed = hypothesis_diversity_loss(
        torch.ones(1, 3, 3), torch.full((1, 3), 1 / 3),
        torch.ones(1, 3, dtype=torch.bool),
    )
    assert collapsed > diverse
    assert brier_rows(logits, targets).max() < 0.04
    quantiles = torch.tensor([[[0.0], [1.0], [2.0]], [[0.0], [1.0], [2.0]]])
    coverage = quantile_coverage_penalty(
        quantiles, torch.tensor([[0.5], [1.5]]), torch.tensor([0.25, 0.5, 0.75]),
    )
    assert torch.isfinite(coverage)


def test_gradient_conflict_monitor_detects_opposed_objectives():
    parameter = torch.tensor([1.0, -1.0], requires_grad=True)
    report = gradient_conflicts(
        {"forward": parameter.sum(), "reverse": -parameter.sum()}, [parameter]
    )
    assert report.names == ("forward", "reverse")
    torch.testing.assert_close(report.cosine[0, 1], torch.tensor(-1.0))
    assert report.conflict_fraction == 1


def test_metacognitive_objectives_train_exact_live_columns_and_masks():
    prediction = torch.zeros(1, 2, 7, requires_grad=True)
    production_mask = torch.tensor([[True, False]])
    targets = MetacognitiveTargets(
        realized_error=torch.tensor([[1.0, 100.0]]),
        operation_values=torch.ones(1, 2, 5),
        calibration_error=torch.tensor([[0.5, 100.0]]),
        mask=torch.ones(1, 2, dtype=torch.bool),
    )
    terms = metacognitive_objectives(prediction, production_mask, targets)
    assert {term.name for term in terms} == {
        "metacognitive_error_prediction", "metacognitive_operation_value",
        "metacognitive_calibration",
    }
    assert all(torch.equal(term.mask, production_mask) for term in terms)
    combine_cognitive_objectives(
        terms, CognitiveObjectiveSchedule.curriculum(8)
    ).total.backward()
    assert prediction.grad is not None
    assert prediction.grad[0, 0].abs().sum() > 0
    assert prediction.grad[0, 1].abs().sum() == 0


def test_metacognitive_objectives_reject_nonfinite_predictions_and_partial_shapes():
    targets = MetacognitiveTargets(
        torch.zeros(1, 2), torch.zeros(1, 2, 5), torch.zeros(1, 2),
        torch.ones(1, 2, dtype=torch.bool),
    )
    invalid = torch.zeros(1, 2, 7)
    invalid[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite floating"):
        metacognitive_objectives(
            invalid, torch.ones(1, 2, dtype=torch.bool), targets,
        )
    with pytest.raises(ValueError, match="batch,time,7"):
        metacognitive_objectives(
            torch.zeros(1, 2, 6), torch.ones(1, 2, dtype=torch.bool), targets,
        )


def test_cognitive_objective_contracts_fail_closed():
    with pytest.raises(ValueError):
        CognitiveObjectiveSchedule.curriculum(10)
    value = torch.ones(1)
    term = ObjectiveTerm("same", ObjectiveFamily.PRIMARY_TASK, value, torch.ones(1, dtype=torch.bool))
    with pytest.raises(ValueError):
        combine_cognitive_objectives((term, term), CognitiveObjectiveSchedule.curriculum(1))
    with pytest.raises(ValueError):
        ObjectiveTerm("bad", ObjectiveFamily.PRIMARY_TASK, value, torch.ones(1))
