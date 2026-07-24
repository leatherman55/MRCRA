import pytest
import torch
from torch import nn

from mrrn.config import MRRNConfig
from mrrn.model import MRRN
from mrrn.optimization import (
    OptimizerPolicy,
    build_adamw,
    build_scheduler,
    clip_and_report_gradients,
    learning_rate_multiplier,
    merge_auxiliary_gradients,
)


def model():
    return MRRN(MRRNConfig(
        input_dim=2, model_dim=4, layers=1, scales=2, heads=1, modes=2,
        mimo_rank=1, attention_window=2, retrieved_items=1, memory_capacity=2,
        mixer_expansion=1, width_growth_cap=1, mode_growth_cap=1, width_multiple=1,
    ))


def test_optimizer_assigns_every_parameter_once_with_slow_no_decay_poles_and_norms():
    network = model()
    policy = OptimizerPolicy(learning_rate=1e-3, weight_decay=0.2, pole_learning_rate_multiplier=0.1, warmup_steps=2, total_steps=10)
    optimizer = build_adamw(network, policy)
    assert optimizer.defaults["fused"] is False
    assigned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    expected = [parameter for parameter in network.parameters() if parameter.requires_grad]
    assert len(assigned) == len(expected) == len({id(parameter) for parameter in assigned})
    assert any(group["pole_group"] and group["lr"] == 1e-4 for group in optimizer.param_groups)
    assert any(group["pole_group"] and group["weight_decay"] == 0 for group in optimizer.param_groups)
    assert any(not group["pole_group"] and group["weight_decay"] == 0.2 for group in optimizer.param_groups)


@pytest.mark.parametrize("schedule", ["cosine", "inverse_sqrt"])
def test_warmup_and_decay_schedule_is_bounded_and_scheduler_uses_it(schedule):
    policy = OptimizerPolicy(warmup_steps=2, total_steps=12, minimum_learning_rate_ratio=0.1, schedule=schedule)
    values = [learning_rate_multiplier(step, policy) for step in range(15)]
    assert values[0] == 0.5 and values[1] == 1
    assert all(0.1 <= value <= 1 for value in values)
    optimizer = build_adamw(model(), policy)
    scheduler = build_scheduler(optimizer, policy)
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr()


def test_gradient_clipping_reports_phase_and_amplitude_separately_and_handles_empty():
    network = model()
    empty = clip_and_report_gradients(network, maximum_norm=1)
    assert empty.total_before_clip == 0 and empty.finite
    loss = network(torch.randn(2, 5, 2)).prediction.square().mean()
    loss.backward()
    report = clip_and_report_gradients(network, maximum_norm=0.05)
    assert report.total_before_clip > 0 and report.phase_norm >= 0 and report.amplitude_norm > 0
    assert report.total_after_clip <= 0.05001
    assert 0 < report.clip_coefficient < 1
    assert report.finite
    total_after = torch.stack([parameter.grad.square().sum() for parameter in network.parameters() if parameter.grad is not None]).sum().sqrt()
    assert total_after <= 0.05001


def test_gradient_clipping_reports_complete_disjoint_subsystem_participation():
    class Instrumented(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding = nn.Linear(2, 2, bias=False)
            self.cognitive = nn.Module()
            self.cognitive.event_extractor = nn.Linear(2, 1, bias=False)
            self.cognitive.output_context_adapter = nn.Linear(2, 2, bias=False)
            self.cognitive.controller = nn.Linear(2, 1, bias=False)
            self.cognitive.workspace_graph = nn.Linear(2, 1, bias=False)
            self.cognitive.world_model = nn.Linear(2, 1, bias=False)
            self.cognitive.memory = nn.Linear(2, 1, bias=False)
            self.cognitive.compare_projection = nn.Linear(2, 1, bias=False)

    network = Instrumented()
    for index, parameter in enumerate(network.parameters(), start=1):
        parameter.grad = torch.full_like(parameter, float(index))
    expected_before = torch.stack([
        parameter.grad.float().square().sum()
        for parameter in network.parameters()
    ]).sum().sqrt()
    report = clip_and_report_gradients(network, maximum_norm=1.0)
    assert set(report.subsystem_norms_before) == {
        "carrier", "event", "output_bridge", "controller",
        "workspace_router", "world_hypothesis", "memory", "other_cognition",
    }
    torch.testing.assert_close(report.total_before_clip, expected_before)
    reconstructed = torch.stack([
        value.square() for value in report.subsystem_norms_before.values()
    ]).sum().sqrt()
    torch.testing.assert_close(reconstructed, report.total_before_clip)
    assert sum(report.subsystem_tensor_counts.values()) == len(
        tuple(network.parameters())
    )
    for name, before in report.subsystem_norms_before.items():
        torch.testing.assert_close(
            report.subsystem_norms_after[name],
            before * report.clip_coefficient,
        )


def test_optimizer_contracts_fail_closed():
    with pytest.raises(ValueError):
        OptimizerPolicy(learning_rate=0)
    with pytest.raises(ValueError):
        OptimizerPolicy(total_steps=10, warmup_steps=10)
    with pytest.raises(ValueError):
        OptimizerPolicy(schedule="linear")
    with pytest.raises(ValueError):
        learning_rate_multiplier(-1, OptimizerPolicy())
    with pytest.raises(ValueError):
        clip_and_report_gradients(model(), maximum_norm=0)
    network = model()
    network.encoder.weight.requires_grad_(False)
    optimizer = build_adamw(network)
    assert all(network.encoder.weight is not parameter for group in optimizer.param_groups for parameter in group["params"])


def test_auxiliary_gradient_merge_projects_conflicts_and_enforces_subsystem_caps():
    class Instrumented(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding = nn.Linear(2, 1, bias=False)
            self.cognitive = nn.Module()
            self.cognitive.controller = nn.Linear(2, 1, bias=False)

    network = Instrumented()
    named = dict(network.named_parameters())
    named["token_embedding.weight"].grad = torch.tensor([[1.0, 0.0]])
    named["cognitive.controller.weight"].grad = torch.tensor([[1.0, 0.0]])
    auxiliary = {
        "token_embedding.weight": torch.tensor([[-10.0, 10.0]]),
        "cognitive.controller.weight": torch.tensor([[-10.0, 10.0]]),
    }
    report = merge_auxiliary_gradients(
        network,
        auxiliary,
        {"carrier": 0.0, "controller": 0.1},
    )
    assert report.applied
    assert set(report.conflicting_subsystems) == {"carrier", "controller"}
    torch.testing.assert_close(
        named["token_embedding.weight"].grad,
        torch.tensor([[1.0, 0.0]]),
    )
    controller_auxiliary = (
        named["cognitive.controller.weight"].grad - torch.tensor([[1.0, 0.0]])
    )
    assert controller_auxiliary.norm() <= 0.100001
    assert float(
        (
            controller_auxiliary
            * torch.tensor([[1.0, 0.0]])
        ).sum()
    ) >= -1e-7


def test_auxiliary_gradient_merge_rejects_auxiliary_only_and_unknown_paths():
    network = nn.Linear(2, 1, bias=False)
    report = merge_auxiliary_gradients(
        network, {"weight": torch.ones_like(network.weight)}, {"carrier": 1.0}
    )
    assert not report.applied
    assert network.weight.grad is None
    with pytest.raises(ValueError, match="unknown parameters"):
        merge_auxiliary_gradients(
            network, {"missing": torch.ones(1)}, {"carrier": 1.0}
        )
