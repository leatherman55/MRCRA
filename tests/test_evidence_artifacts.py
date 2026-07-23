import pytest
import torch

from mrrn.cognitive_types import ModalityClass, SourceClass, SupportInterval
from mrrn.evidence_requests import (
    EvidenceRequestState, EvidenceRequestStatus, create_evidence_request,
    transition_evidence_requests,
)
from mrrn.external_artifacts import (
    ExternalArtifactState, record_external_artifact, verify_external_artifact,
)
from mrrn.provenance import ProvenanceLedger


def test_evidence_request_has_bounded_identity_and_valid_lifecycle():
    state = EvidenceRequestState.empty(1, 1, 4, 2, 2)
    state, slots = create_evidence_request(
        state, proposition=torch.ones(1, 4),
        requested_modality=torch.tensor([int(ModalityClass.SENSOR)]),
        tool_schema_id=torch.tensor([3]),
        hypothesis_indices=torch.tensor([[0, 1]]),
        hypothesis_mask=torch.tensor([[True, True]]),
        expected_information_gain=torch.tensor([0.7]),
        maximum_cost=torch.tensor([0.2]), maximum_latency=torch.tensor([1.0]),
        required_precision=torch.tensor([0.05]),
        supporting_provenance_ids=torch.tensor([[8, -1]]),
        supporting_mask=torch.tensor([[True, False]]),
        create_mask=torch.tensor([True]),
    )
    assert slots.tolist() == [0]
    assert state.status.item() == int(EvidenceRequestStatus.PENDING)
    state = transition_evidence_requests(
        state, slots, torch.tensor([int(EvidenceRequestStatus.DISPATCHED)]),
        torch.tensor([True]),
    )
    state = transition_evidence_requests(
        state, slots, torch.tensor([int(EvidenceRequestStatus.RESOLVED)]),
        torch.tensor([True]),
    )
    assert not state.active.any()
    with pytest.raises(ValueError, match="inactive"):
        transition_evidence_requests(
            state, slots, torch.tensor([int(EvidenceRequestStatus.EXPIRED)]),
            torch.tensor([True]),
        )


def test_external_artifact_versions_and_digests_are_exact_and_provenance_backed():
    ledger = ProvenanceLedger()
    root = ledger.append(
        source_class=SourceClass.EXTERNAL, source_uri_or_episode="test://action",
        support=SupportInterval(0, 0, 0), modality=ModalityClass.ACTION,
        operator="test:source", scenario_id=0, model_authority="environment",
    )
    state = ExternalArtifactState.empty(1, 2, 4)
    digest = torch.tensor([[1, 2, 3, 4]], dtype=torch.uint8)
    state, slots = record_external_artifact(
        state, ledger, artifact_ids=torch.tensor([19]), content_digests=digest,
        versions=torch.tensor([1]), creator_action_ids=torch.tensor([7]),
        parent_provenance_ids=torch.tensor([[root]]),
        parent_mask=torch.tensor([[True]]),
        expected_persistence=torch.tensor([100.0]), estimated_cost=torch.tensor([0.1]),
        timestamp=torch.tensor([2.0]), readable=torch.tensor([True]),
        writable=torch.tensor([True]), create_mask=torch.tensor([True]),
        model_authority="test-model",
    )
    assert slots.tolist() == [0]
    record = ledger.get(int(state.provenance_ids[0, 0]))
    assert record.source_class == SourceClass.EXTERNAL_ARTIFACT
    state, matches = verify_external_artifact(
        state, artifact_ids=torch.tensor([19]), content_digests=digest,
        versions=torch.tensor([1]), timestamp=torch.tensor([3.0]),
    )
    assert matches.item() and state.last_verified_time[0, 0] == 3
    _, stale = verify_external_artifact(
        state, artifact_ids=torch.tensor([19]),
        content_digests=torch.tensor([[1, 2, 3, 5]], dtype=torch.uint8),
        versions=torch.tensor([1]), timestamp=torch.tensor([4.0]),
    )
    assert not stale.item()
    with pytest.raises(ValueError, match="versions must increase"):
        record_external_artifact(
            state, ledger, artifact_ids=torch.tensor([19]), content_digests=digest,
            versions=torch.tensor([1]), creator_action_ids=torch.tensor([8]),
            parent_provenance_ids=torch.tensor([[root]]),
            parent_mask=torch.tensor([[True]]), expected_persistence=torch.tensor([100.0]),
            estimated_cost=torch.tensor([0.1]), timestamp=torch.tensor([5.0]),
            readable=torch.tensor([True]), writable=torch.tensor([True]),
            create_mask=torch.tensor([True]), model_authority="test-model",
        )
