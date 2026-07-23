import torch

from mrrn.cognitive_objectives import (
    CognitiveObjectiveSchedule, ObjectiveFamily, ReconstructionTargets,
    WorldModelTargets, combine_cognitive_objectives,
    reconstruction_objectives, world_model_objectives,
)
from mrrn.reconstruction import (
    ConditionalGraphReconstructor, ReconstructionEvidence, ReconstructionQuery,
)
from mrrn.world_model import ActionConditionedWorldModel


def test_actual_reconstructor_heads_receive_finite_nonzero_training_gradients():
    torch.manual_seed(463)
    model = ConditionalGraphReconstructor(8, 13, 16, 3, 2)
    query = ReconstructionQuery(
        torch.tensor([0]), torch.tensor([0]), torch.tensor([[0.0, 1.0, 1.0]]),
        torch.tensor([3]), torch.tensor([2]), torch.tensor([0]), torch.tensor([1]),
        torch.tensor([0.1]), torch.randn(1, 8), torch.tensor([True]),
    )
    evidence = ReconstructionEvidence(
        torch.randn(1, 8), torch.randn(1, 2, 8), torch.tensor([[True, True]]),
        torch.tensor([[1, 2]]), torch.randn(1, 8), torch.tensor([[3, 4]]),
        torch.randn(1, 2, 8), torch.randn(1, 8), torch.randn(1, 8),
    )
    proposal = model(query, evidence)
    targets = ReconstructionTargets(
        torch.randn_like(proposal.node_content),
        torch.randint(0, 13, proposal.node_mask.shape), proposal.node_mask,
        torch.randn_like(proposal.relation_content),
        torch.randint(0, 16, proposal.relation_mask.shape), proposal.relation_mask,
        torch.tensor([0.8]), torch.tensor([0.9]), torch.tensor([0.7]),
        torch.tensor([1.0]), torch.tensor([True]),
    )
    terms = reconstruction_objectives(proposal, targets)
    loss = combine_cognitive_objectives(
        terms, CognitiveObjectiveSchedule.curriculum(4)
    ).total
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert any(
        gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        for gradient in gradients
    )
    assert {term.family for term in terms} == {ObjectiveFamily.RECONSTRUCTION_FIDELITY}


def test_actual_multihorizon_world_heads_receive_each_declared_objective_signal():
    torch.manual_seed(467)
    model = ActionConditionedWorldModel(8, 3, 16, 8, horizons=(1, 2))
    prediction = model(torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 3))
    shape = prediction.costs.shape
    targets = WorldModelTargets(
        torch.randn_like(prediction.latent_mean), torch.randn(shape),
        torch.rand(shape), torch.randint(0, 2, shape).float(),
        torch.randint(0, 2, shape).float(), torch.ones(shape, dtype=torch.bool),
    )
    terms = world_model_objectives(prediction, targets)
    expected = {
        ObjectiveFamily.WORLD_MODEL_HYPOTHESIS_LIKELIHOOD,
        ObjectiveFamily.ACTION_CONSEQUENCE_INFORMATION_GAIN,
        ObjectiveFamily.VIABILITY_CONSTRAINT,
    }
    assert {term.family for term in terms} == expected
    combine_cognitive_objectives(
        terms, CognitiveObjectiveSchedule.curriculum(8)
    ).total.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    for head in (model.latent_mean, model.reward_base, model.cost, model.constraint, model.success):
        assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
