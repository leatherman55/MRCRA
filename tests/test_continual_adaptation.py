import torch
from torch import nn

from mrrn.continual_adaptation import (
    ContinualReplayBuffer, IsolatedContinualAdapter, ReplayItem,
)


class AdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(3, 3)
        self.adapter = nn.Linear(3, 3, bias=False)

    def forward(self, value):
        return self.base(value).detach() + self.adapter(value)


def test_continual_adapter_updates_only_allowlisted_weights_and_rolls_back_exactly():
    torch.manual_seed(991)
    model = AdapterModel()
    original = {name: value.detach().clone() for name, value in model.named_parameters()}
    transaction = IsolatedContinualAdapter(
        model, ("adapter.weight",), learning_rate=0.1
    )
    value = torch.randn(4, 3)
    transaction.step(model(value).square().mean())
    assert not torch.equal(model.adapter.weight, original["adapter.weight"])
    assert torch.equal(model.base.weight, original["base.weight"])
    receipt = transaction.retention_gate(
        lambda _: 0.0, baseline_metric=1.0, maximum_allowed_regression=0.1
    )
    assert receipt.rolled_back and not receipt.committed
    assert torch.equal(model.adapter.weight, original["adapter.weight"])
    assert torch.equal(model.base.weight, original["base.weight"])


def test_replay_buffer_is_bounded_and_retains_authority_metadata():
    replay = ContinualReplayBuffer(2)
    for index in range(3):
        replay.append(ReplayItem(
            torch.ones(1, 2) * index, torch.tensor([index]), index,
            f"session:{index}",
        ))
    assert len(replay) == 2
    assert {item.provenance_id for item in replay.items()} == {1, 2}
