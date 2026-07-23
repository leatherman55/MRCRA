import pytest
import torch

from mrrn.cognitive_types import BoundaryClass, NodeSlots, NodeType, SourceClass
from mrrn.events import (
    CausalEventExtractor, EventCandidates, EventEvidence, EventProposalNetwork,
    PersistentEventAllocator,
)


def evidence(features, *, boundary=None):
    batch, length, width = features.shape
    if boundary is None:
        boundary = torch.zeros(batch, length, dtype=torch.int64)
    return EventEvidence(
        features,
        torch.zeros(batch, length, 1),
        torch.zeros(batch, length, 1),
        torch.zeros(batch, length, 1),
        torch.zeros(batch, length, 3),
        torch.zeros(batch, 2),
        boundary,
        torch.arange(length, dtype=features.dtype).expand(batch, -1),
        torch.zeros(batch, length, dtype=torch.int64),
        torch.arange(batch * length, dtype=torch.int64).reshape(batch, length),
        torch.zeros(batch, length, dtype=torch.int64),
        torch.zeros(batch, length, 2, 3, 2),
        torch.ones(batch, length, dtype=torch.bool),
    )


def always_emit_extractor(width=6, *, quota=3):
    proposal = EventProposalNetwork(width, 3, 2, len(NodeType))
    for parameter in proposal.parameters():
        parameter.data.zero_()
    proposal.proposal.bias.data.fill_(8)
    proposal.end.bias.data.fill_(8)
    return CausalEventExtractor(
        proposal, chunk_size=4, proposals_per_chunk=quota,
        proposal_threshold=0.5, end_threshold=0.5,
    )


def test_event_extraction_is_causal_and_never_exceeds_chunk_quota():
    torch.manual_seed(3)
    extractor = always_emit_extractor(quota=2)
    x = torch.randn(1, 8, 6)
    reference, state = extractor(evidence(x))
    changed = x.clone()
    changed[:, 5:] = torch.randn_like(changed[:, 5:]) * 100
    alternative, _ = extractor(evidence(changed))
    assert reference.active.sum() == 4
    assert state.emitted_in_chunk.item() == 2
    torch.testing.assert_close(reference.content[:, :2], alternative.content[:, :2])
    torch.testing.assert_close(reference.support[:, :2], alternative.support[:, :2])


def test_event_transition_receipts_exactly_partition_emissions_and_quota_rejections():
    extractor = always_emit_extractor(quota=2)
    events, state, proposals, receipts = extractor.extract_with_proposals(
        evidence(torch.ones(1, 8, 6))
    )
    assert proposals.proposal_logits.shape == (1, 8)
    assert receipts.opened.all()
    assert receipts.finalized.all()
    assert receipts.emitted.tolist() == [[True, True, False, False] * 2]
    assert receipts.quota_rejected.tolist() == [[False, False, True, True] * 2]
    assert not bool((receipts.emitted & receipts.quota_rejected).any())
    assert not receipts.open_after.any()
    assert events.active.sum() == receipts.emitted.sum() == 4
    assert not state.open.any()


def test_event_span_closes_at_authoritative_boundary_and_stream_state_continues():
    extractor = always_emit_extractor(quota=4)
    # Suppress learned end so only the segment boundary finalizes the open span.
    extractor.proposal_network.end.bias.data.fill_(-8)
    boundary = torch.tensor([[0, 0, BoundaryClass.SEGMENT, 0]], dtype=torch.int64)
    first, state = extractor(evidence(torch.ones(1, 4, 6), boundary=boundary))
    assert first.active.sum() == 1
    assert first.support[0, 0].tolist() == [0.0, 2.0, 2.0]
    second_evidence = evidence(torch.ones(1, 2, 6))
    second_evidence = EventEvidence(
        second_evidence.features, second_evidence.detail_energy, second_evidence.prediction_error,
        second_evidence.resonator_change, second_evidence.uncertainty,
        second_evidence.goal_context, second_evidence.boundary_classes,
        second_evidence.timestamps + 4, second_evidence.modality_ids,
        second_evidence.parent_provenance_ids, second_evidence.scenario_ids,
        second_evidence.spectral, second_evidence.valid_mask,
    )
    _, continued = extractor(second_evidence, state)
    assert continued.position == 6


def make_candidates(content, provenance, score=4.0):
    batch, count, width = content.shape
    active = torch.ones(batch, count, dtype=torch.bool)
    return EventCandidates(
        content,
        torch.nn.functional.normalize(content, dim=-1),
        torch.zeros(batch, count, 2, 3, 2),
        torch.nn.functional.one_hot(
            torch.full((batch, count), int(NodeType.EVENT)), len(NodeType)
        ).to(content.dtype),
        torch.tensor([[[0.0, 1.0, 1.0]] * count] * batch),
        torch.zeros(batch, count, dtype=torch.int64),
        torch.zeros(batch, count, dtype=torch.int64),
        provenance,
        torch.full((batch, count), int(SourceClass.INFERRED), dtype=torch.int64),
        torch.zeros(batch, count, dtype=torch.int64),
        torch.zeros(batch, count, 8),
        torch.full((batch, count), score),
        active,
    )


def empty_nodes(batch=1, capacity=2, width=6):
    return NodeSlots.empty(
        batch, capacity, width, heads=2, modes=3, node_types=len(NodeType),
        modalities=16, uncertainty_channels=8, provenance_features=4, hypotheses=4,
    )


def test_persistent_allocator_allocates_updates_identity_and_evicts_by_utility():
    torch.manual_seed(7)
    allocator = PersistentEventAllocator(6, identity_threshold=0.6)
    allocator.identity_projection.weight.data.copy_(torch.eye(6))
    content = torch.nn.functional.normalize(torch.randn(1, 2, 6), dim=-1)
    state = allocator(empty_nodes(), make_candidates(content, torch.tensor([[10, 11]])))
    assert state.active.all()
    assert state.provenance_ids.tolist() == [[10, 11]]
    assert state.type_logits.argmax(-1).tolist() == [[int(NodeType.EVENT)] * 2]

    same = content[:, :1] + 1e-4
    updated = allocator(state, make_candidates(same, torch.tensor([[12]])))
    assert updated.provenance_ids[0, 0] == 12
    assert updated.versions[0, 0] == 2
    assert updated.active.sum() == 2

    updated_importance = updated.importance.clone()
    updated_importance[0] = torch.tensor([0.9, 0.01])
    values = {name: getattr(updated, name) for name in updated.__dataclass_fields__}
    values["importance"] = updated_importance
    state = NodeSlots(**values)
    novel = torch.nn.functional.normalize(torch.tensor([[[0., 0., 0., 0., 0., 1.]]]), dim=-1)
    evicted = allocator(state, make_candidates(novel, torch.tensor([[13]])))
    assert 13 in evicted.provenance_ids[0].tolist()
    assert 12 in evicted.provenance_ids[0].tolist()


def test_events_fail_closed_without_derived_provenance():
    allocator = PersistentEventAllocator(6)
    candidates = make_candidates(torch.randn(1, 1, 6), torch.tensor([[-1]]))
    with pytest.raises(ValueError):
        allocator(empty_nodes(), candidates)
    with pytest.raises(ValueError):
        CausalEventExtractor(always_emit_extractor().proposal_network, chunk_size=2, proposals_per_chunk=3)
