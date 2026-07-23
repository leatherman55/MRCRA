import torch
import pytest

from mrrn.cognitive_objectives import ObjectiveFamily
from mrrn.gradient_governance import (
    objective_gradient_report, project_conflicting_gradients,
)


def test_gradient_report_detects_objective_conflict_without_touching_live_gradients():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    losses = {
        ObjectiveFamily.PRIMARY_TASK: parameter.sum(),
        ObjectiveFamily.RECONSTRUCTION_FIDELITY: -parameter.sum(),
    }
    report = objective_gradient_report(losses, (parameter,))
    assert report.finite
    assert report.conflict_mask[0, 1]
    assert report.cosine_similarity[0, 1].item() == pytest.approx(-1)
    assert parameter.grad is None


def test_conflict_projection_removes_negative_pairwise_component():
    gradients = torch.tensor([[1.0, 0.0], [-1.0, 1.0]])
    projected = project_conflicting_gradients(gradients)
    assert torch.dot(projected[0], gradients[1]) >= -1e-7
    assert torch.isfinite(projected).all()
