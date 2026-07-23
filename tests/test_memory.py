import pytest
import torch

from mrrn.memory import EideticMemory, MemoryHandle, MemoryItem, MemoryWritePolicy


def item(index, *, priority=None, timestamp=None):
    key = torch.tensor([1.0, float(index), -0.5])
    return MemoryItem(
        key=key,
        value=torch.tensor([float(index), float(index + 1)]),
        signature=torch.tensor([1.0, float(index)]),
        timestamp=index if timestamp is None else timestamp,
        scale=index % 2,
        priority=float(index if priority is None else priority),
    )


def test_capacity_eviction_versioning_and_stale_handles():
    memory = EideticMemory(2, 3, 2, 2, age_weight=0, redundancy_weight=0, use_weight=0)
    low = memory.write(item(0, priority=0))
    high = memory.write(item(1, priority=10))
    replacement = memory.write(item(2, priority=5))
    assert len(memory) == 2 and memory.evictions == 1 and memory.writes == 3
    with pytest.raises(KeyError, match="stale"):
        memory.get(low)
    assert memory.get(high).priority == 10
    assert replacement.slot == low.slot and replacement.version > low.version


def test_signature_retrieval_is_causal_deterministic_and_exact_reranking_counts_use():
    memory = EideticMemory(5, 3, 2, 2)
    handles = [memory.write(item(index)) for index in range(4)]
    routed = memory.retrieve(torch.tensor([1.0, 2.1]), 4, query_time=2)
    assert all(memory.get(handle).timestamp <= 2 for handle in routed)
    reranked = memory.rerank(torch.tensor([1.0, 1.0, -0.5]), routed, 2)
    assert reranked[0] == handles[1]
    assert memory.get(reranked[0]).use_count == 1
    oracle = [handles[1], handles[2]]
    assert memory.recall(reranked, oracle, 2) >= 0.5
    assert memory.recall([], [], 2) == 1.0


def test_memory_roundtrip_serialization_and_clear():
    memory = EideticMemory(3, 3, 2, 2)
    handles = [memory.write(item(index)) for index in range(2)]
    state = memory.state_dict()
    restored = EideticMemory(3, 3, 2, 2)
    restored.load_state_dict(state)
    for handle in handles:
        torch.testing.assert_close(restored.get(handle).value, memory.get(handle).value)
    restored.clear()
    assert len(restored) == 0 and restored.writes == restored.evictions == 0
    replacement = restored.write(item(2))
    assert replacement.version != handles[0].version
    with pytest.raises(KeyError):
        restored.get(handles[0])


def test_write_policy_scores_and_enforces_quota_threshold():
    policy = MemoryWritePolicy(initial_bias=0)
    features = torch.randn(2, 6, 5)
    scores = policy(features)
    assert scores.shape == (2, 6)
    selected = policy.select(torch.tensor([[0.9, 0.8, 0.7], [0.2, 0.6, 0.5]]), quota=2, threshold=0.55)
    assert selected.tolist() == [[True, True, False], [False, True, False]]
    assert not policy.select(scores[:, :0], quota=2).any()
    assert not policy.select(scores, quota=0).any()
    differentiable_scores = torch.tensor([[0.9, 0.8, 0.1]], requires_grad=True)
    straight_through = policy.straight_through_select(differentiable_scores, quota=1)
    assert straight_through.detach().tolist() == [[1.0, 0.0, 0.0]]
    straight_through.sum().backward()
    assert differentiable_scores.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        policy.straight_through_select(differentiable_scores, quota=1, temperature=0)


def test_eviction_rewards_use_and_penalizes_age_and_redundancy():
    memory = EideticMemory(3, 3, 2, 2, age_weight=1, redundancy_weight=1, use_weight=1)
    first = memory.write(item(0, priority=2, timestamp=0))
    second = memory.write(item(1, priority=2, timestamp=9))
    third = memory.write(item(2, priority=2, timestamp=9))
    memory.get(first).use_count = 100
    scores = [memory._eviction_score(slot, 10) for slot in range(3)]
    assert all(torch.isfinite(torch.tensor(scores)))
    memory.write(item(3, priority=3, timestamp=10))
    assert len(memory) == 3 and memory.get(first).use_count == 100
    remaining = memory.retrieve(torch.tensor([1.0, 1.0]), 3)
    assert second in remaining or third in remaining


def test_memory_invalid_contracts_fail_closed():
    with pytest.raises(ValueError):
        EideticMemory(0, 1, 1, 1)
    with pytest.raises(ValueError):
        EideticMemory(1, 1, 1, 1, age_weight=-1)
    memory = EideticMemory(2, 3, 2, 2)
    with pytest.raises(ValueError):
        memory.write(MemoryItem(torch.ones(2), torch.ones(2), torch.ones(2), 0, 0, 1))
    with pytest.raises(ValueError):
        memory.write(MemoryItem(torch.ones(3), torch.ones(2), torch.ones(2), -1, 0, 1))
    with pytest.raises(KeyError):
        memory.get(MemoryHandle(5, 1))
    with pytest.raises(ValueError):
        memory.retrieve(torch.ones(3), 1)
    with pytest.raises(ValueError):
        memory.retrieve(torch.ones(2), 0)
    with pytest.raises(ValueError):
        memory.rerank(torch.ones(2), [], 1)
    with pytest.raises(ValueError):
        memory.recall([], [], 0)
    bad = memory.state_dict()
    bad["capacity"] = 3
    with pytest.raises(ValueError):
        memory.load_state_dict(bad)
    with pytest.raises(ValueError):
        MemoryWritePolicy()(torch.randn(2, 4))
    with pytest.raises(ValueError):
        MemoryWritePolicy.select(torch.randn(2, 3), quota=-1)
