import pytest
import torch

from mrrn import complex_ops as c
from mrrn.objectives import (
    LossWeights,
    combine_losses,
    physical_constraint_loss,
    pole_coverage_loss,
    predictive_state_loss,
    retrieval_contrastive_loss,
    router_balance_loss,
    sobolev_spectral_loss,
    state_energy_loss,
    supervised_task_loss,
)


def test_supervised_task_losses_masks_and_contracts():
    logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0]]])
    labels = torch.tensor([[0, 1]])
    expected = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
    torch.testing.assert_close(supervised_task_loss(logits, labels, kind="cross_entropy"), expected)
    prediction = torch.tensor([[[1.0, 2.0], [100.0, 100.0]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False]])
    assert supervised_task_loss(prediction, target, mask=mask) == 2.5
    assert supervised_task_loss(prediction, target, kind="l1", mask=mask) == 1.5
    with pytest.raises(ValueError):
        supervised_task_loss(prediction, target, kind="unknown")
    with pytest.raises(ValueError):
        supervised_task_loss(prediction, target, mask=mask.float())


def test_predictive_state_loss_stops_target_gradient_and_obeys_scale_masks():
    prediction = torch.randn(2, 5, 3, requires_grad=True)
    target = torch.randn(2, 5, 3, requires_grad=True)
    mask = torch.tensor([[True] * 5, [True] * 3 + [False] * 2])
    loss = predictive_state_loss([prediction], [target], [mask])
    loss.backward()
    assert prediction.grad is not None and target.grad is None
    exact = predictive_state_loss([target.detach()], [target.detach()])
    expected = (target[:, :-1] - target[:, 1:]).square().mean()
    torch.testing.assert_close(exact, expected)
    with pytest.raises(ValueError):
        predictive_state_loss([], [])


def test_contrastive_pole_energy_and_router_losses_have_expected_ordering():
    good = retrieval_contrastive_loss(torch.tensor([4.0]), torch.tensor([[0.0, -1.0]]))
    bad = retrieval_contrastive_loss(torch.tensor([0.0]), torch.tensor([[4.0, -1.0]]))
    assert good < bad

    collapsed = pole_coverage_loss(torch.ones(2, 4), torch.zeros(2, 4))
    spread = pole_coverage_loss(
        torch.tensor([[0.1, 0.3, 1.0, 3.0]]), torch.tensor([[0.0, 0.8, 1.7, 3.0]])
    )
    assert spread < collapsed
    assert pole_coverage_loss(torch.ones(2, 1), torch.zeros(2, 1)) == 0

    state = c.pair(torch.ones(2, 3), torch.zeros(2, 3))
    assert state_energy_loss(state, maximum_rms=2) == 1
    balanced = router_balance_loss(torch.full((10, 4), 0.25))
    collapsed_route = router_balance_loss(torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(10, 1))
    assert balanced == 0 and collapsed_route > balanced


def test_physical_and_periodic_sobolev_losses_are_domain_supplied_and_differentiable():
    prediction = torch.randn(2, 8, 6, 1, requires_grad=True)
    target = torch.randn_like(prediction)
    loss = sobolev_spectral_loss(prediction, target, spacing=(0.1, 0.2), order=1.5, periodic=True)
    physical = physical_constraint_loss({"sobolev": loss, "boundary": lambda: prediction[:, 0].square().mean()})
    physical.backward()
    assert torch.isfinite(prediction.grad).all()
    with pytest.raises(ValueError):
        sobolev_spectral_loss(prediction, target, spacing=(0.1, 0.2), periodic=False)
    with pytest.raises(ValueError):
        physical_constraint_loss({})


def test_weighted_loss_breakdown_is_exact_and_missing_terms_are_zero():
    task = torch.tensor(2.0, requires_grad=True)
    weights = LossWeights(predictive=0.5, energy=2.0, spectral=0.25)
    result = combine_losses(
        task, weights, predictive=torch.tensor(4.0), energy=torch.tensor(3.0),
        spectral=torch.tensor(8.0),
    )
    assert result.total == 12 and result.retrieval == 0 and result.physical == 0
    assert result.spectral == 8
    result.total.backward()
    assert task.grad == 1
    with pytest.raises(ValueError):
        LossWeights(pole=-1)
    with pytest.raises(ValueError):
        combine_losses(torch.ones(2), weights)


def test_objective_contract_failures():
    with pytest.raises(ValueError):
        supervised_task_loss(torch.ones(2, 3), torch.ones(2, 1), kind="cross_entropy")
    with pytest.raises(ValueError):
        supervised_task_loss(torch.ones(2), torch.ones(3), kind="mse")
    with pytest.raises(ValueError):
        supervised_task_loss(torch.ones(2), torch.ones(3), kind="l1")
    state = torch.ones(1, 2, 3)
    with pytest.raises(ValueError):
        predictive_state_loss([state], [state], masks=[])
    with pytest.raises(ValueError):
        predictive_state_loss([state], [state], scale_weights=[-1])
    with pytest.raises(ValueError):
        predictive_state_loss([state], [torch.ones(1, 2, 4)])
    with pytest.raises(ValueError):
        predictive_state_loss([state], [state], masks=[torch.ones(1, 2)])
    with pytest.raises(ValueError):
        retrieval_contrastive_loss(torch.ones(2), torch.ones(3, 2))
    with pytest.raises(ValueError):
        retrieval_contrastive_loss(torch.ones(2), torch.ones(2, 2), temperature=0)
    with pytest.raises(ValueError):
        pole_coverage_loss(torch.tensor([-1.0]), torch.tensor([0.0]))
    with pytest.raises(ValueError):
        pole_coverage_loss(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError):
        state_energy_loss([], maximum_rms=1)
    with pytest.raises(ValueError):
        state_energy_loss(torch.ones(1, 2, 2), maximum_rms=0)
    with pytest.raises(ValueError):
        router_balance_loss(torch.tensor([[1.0, -1.0]]))
    with pytest.raises(ValueError):
        router_balance_loss(torch.ones(3))
    with pytest.raises(ValueError):
        physical_constraint_loss({"bad": torch.ones(2)})
    with pytest.raises(ValueError):
        sobolev_spectral_loss(torch.ones(1, 2, 1), torch.ones(1, 3, 1), spacing=(1,), periodic=True)
    with pytest.raises(ValueError):
        sobolev_spectral_loss(torch.ones(1, 2, 1), torch.ones(1, 2, 1), spacing=(-1,), periodic=True)
    with pytest.raises(ValueError):
        combine_losses(torch.tensor(1.0), LossWeights(), pole=torch.ones(2))
