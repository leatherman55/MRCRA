import pytest
import torch

from mrrn.cognitive_types import ModalityClass, SourceClass, SupportInterval, VerificationClass
from mrrn.memory_v2 import (
    BatchedTensorMemory, MemoryQuery, MemoryTier, MemoryWriteBatch,
    MemoryWriteEvidence, MemoryWritePolicyV2, TensorMemoryState,
)
from mrrn.provenance import ProvenanceLedger


def empty(capacity=5):
    return TensorMemoryState.empty(
        1, capacity, 4, 6, 3, heads=2, modes=2, uncertainty_channels=3,
        consequence_dim=2, association_degree=2,
    )


def write_batch(count=4, *, source=SourceClass.INFERRED, provenance_start=10):
    torch.manual_seed(41)
    keys = torch.nn.functional.normalize(torch.randn(1, count, 4), dim=-1)
    signatures = torch.nn.functional.normalize(torch.randn(1, count, 3), dim=-1)
    spectral = torch.zeros(1, count, 2, 2, 2)
    spectral[..., 0] = 1
    support = torch.zeros(1, count, 3)
    support[0, :, :] = torch.arange(count)[:, None]
    return MemoryWriteBatch(
        keys, torch.randn(1, count, 6), signatures, spectral, support,
        torch.zeros(1, count, dtype=torch.int64),
        torch.arange(provenance_start, provenance_start + count).view(1, count),
        torch.full((1, count), int(source), dtype=torch.int64),
        torch.zeros(1, count, dtype=torch.int64),
        torch.zeros(1, count, 3), torch.zeros(1, count, 2),
        torch.linspace(0.1, 0.9, count).view(1, count),
        torch.ones(1, count, dtype=torch.bool),
    )


def query_from(batch, indices, timestamps):
    return MemoryQuery(
        batch.keys[:, indices], batch.signatures[:, indices], batch.spectral[:, indices],
        torch.tensor([timestamps], dtype=batch.keys.dtype),
        batch.type_ids[:, indices], batch.source_classes[:, indices],
        batch.scenario_ids[:, indices], torch.ones(1, len(indices), dtype=torch.bool),
    )


def test_write_policy_uses_all_evidence_and_penalizes_noise_and_redundancy():
    policy = MemoryWritePolicyV2(hidden=4)
    evidence = MemoryWriteEvidence(*(
        torch.ones(1, 3) for _ in range(10)
    ), torch.tensor([[True, True, False]]))
    scores = policy(evidence)
    assert scores.shape == (1, 3) and scores[0, 2] == -torch.inf
    scores[:, :2].sum().backward()
    assert all(parameter.grad is not None for parameter in policy.parameters())


def test_tensor_memory_write_quota_detaches_and_exact_retrieval_is_causal():
    memory = BatchedTensorMemory(4, 3, 2, 2, route_candidates=4, retrieved_items=2)
    batch = write_batch()
    state = memory.write(
        empty(), batch, torch.tensor([[1.0, 4.0, 3.0, 2.0]]),
        quota=2, tier=MemoryTier.EPISODIC,
    )
    assert state.active.sum() == 2 and state.clock == 1
    assert set(state.provenance_ids[state.active].tolist()) == {11, 12}
    assert not state.keys.requires_grad
    # Query the exact record stored at timestamp 2, first before and then after
    # its causal completion time.
    query = query_from(batch, [2], [1.0])
    early = memory.retrieve(state, query, compute_oracle=True)
    assert not (early.provenance_ids == 12).any()
    query = query_from(batch, [2], [3.0])
    retrieval = memory.retrieve(state, query, compute_oracle=True)
    assert retrieval.provenance_ids[0, 0, 0] == 12
    assert retrieval.router_recall.item() == 1
    assert retrieval.oracle_indices.item() >= 0
    accessed = memory.mark_accessed(state, retrieval)
    assert accessed.use_count.sum() == retrieval.mask.sum()


def test_duplicate_write_updates_version_without_consuming_capacity():
    memory = BatchedTensorMemory(4, 3, 2, 2, route_candidates=2, retrieved_items=1)
    batch = write_batch(count=1)
    state = memory.write(empty(2), batch, torch.ones(1, 1), quota=1, tier=MemoryTier.EPISODIC)
    state = memory.write(state, batch, torch.ones(1, 1), quota=1, tier=MemoryTier.EPISODIC)
    assert state.active.sum() == 1
    assert state.versions[state.active].item() == 2


def test_semantic_writes_require_verified_nonsimulated_provenance():
    memory = BatchedTensorMemory(4, 3, 2, 2)
    ledger = ProvenanceLedger()
    root = ledger.append(
        source_class=SourceClass.EXTERNAL, source_uri_or_episode="doc://1",
        support=SupportInterval(0, 0, 0), modality=ModalityClass.TEXT,
        operator="adapter", scenario_id=0, model_authority="m",
        verification=VerificationClass.EXTERNALLY_CHECKED,
    )
    abstracted = ledger.derive(
        [root], source_class=SourceClass.ABSTRACTED, operator="consolidate",
        support=SupportInterval(0, 0, 0), modality=ModalityClass.TEXT,
        model_authority="m",
    )
    ledger.set_verification(
        abstracted, VerificationClass.INTERNALLY_CONSISTENT,
        authority="consolidator", reason="held-out checks",
    )
    batch = write_batch(count=1, source=SourceClass.ABSTRACTED, provenance_start=abstracted)
    semantic = memory.write(
        empty(), batch, torch.ones(1, 1), quota=1,
        tier=MemoryTier.SEMANTIC, ledger=ledger,
    )
    assert semantic.active.sum() == 1

    simulated_id = ledger.derive(
        [root], source_class=SourceClass.SIMULATED, operator="world",
        support=SupportInterval(0, 1, 1), modality=ModalityClass.PREDICTION,
        scenario_id=2, model_authority="m",
    )
    simulated = write_batch(count=1, source=SourceClass.SIMULATED, provenance_start=simulated_id)
    with pytest.raises(ValueError):
        memory.write(
            empty(), simulated, torch.ones(1, 1), quota=1,
            tier=MemoryTier.SEMANTIC, ledger=ledger,
        )
    # It remains legal to preserve an explicitly tagged simulation episodically.
    assert memory.write(
        empty(), simulated, torch.ones(1, 1), quota=1, tier=MemoryTier.EPISODIC,
    ).active.sum() == 1


def test_controlled_associative_spread_obeys_visited_depth_and_budget():
    memory = BatchedTensorMemory(4, 3, 2, 2)
    batch = write_batch(count=4)
    state = memory.write(empty(4), batch, torch.ones(1, 4), quota=4, tier=MemoryTier.EPISODIC)
    state = memory.link_associations(
        state,
        torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 2, 3, 0]]),
        torch.zeros(1, 4, dtype=torch.int64), torch.ones(1, 4),
        torch.ones(1, 4, dtype=torch.bool),
    )
    expansion = memory.expand_associations(
        state, torch.tensor([[0]]), torch.tensor([[True]]), maximum_depth=2, budget=3,
    )
    assert expansion.indices[0, expansion.mask[0]].tolist() == [0, 1, 2]
    assert expansion.depths[0, expansion.mask[0]].tolist() == [0, 1, 2]
    assert expansion.mask.sum() == 3


def test_memory_contracts_fail_closed():
    with pytest.raises(ValueError):
        BatchedTensorMemory(4, 3, 2, 2, route_candidates=1, retrieved_items=2)
    memory = BatchedTensorMemory(4, 3, 2, 2)
    with pytest.raises(ValueError):
        memory.write(empty(), write_batch(), torch.ones(1, 3), quota=1, tier=MemoryTier.EPISODIC)
    state = empty(2)
    with pytest.raises(ValueError):
        memory.link_associations(
            state, torch.tensor([[0]]), torch.tensor([[1]]), torch.tensor([[0]]),
            torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool),
        )
