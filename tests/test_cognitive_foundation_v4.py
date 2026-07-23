import copy

import pytest
import torch

from mrrn.abstraction_control import AbstractionValidityState
from mrrn.action_candidates import ActionCandidateState
from mrrn.boundaries import BoundaryContextState, legacy_scope
from mrrn.cognitive_types import (
    BoundaryClass, BoundaryScope, InternalAction, SourceClass,
)
from mrrn.evidence_requests import EvidenceRequestState
from mrrn.external_artifacts import ExternalArtifactState
from mrrn.metacognition import MetacognitiveState
from mrrn.reconstruction import ReconstructionState
from mrrn.viability import ViabilityState


def test_extended_ontologies_append_without_reinterpreting_legacy_values():
    assert int(SourceClass.GOAL_DERIVED) == 7
    assert int(SourceClass.RECONSTRUCTED) == 8
    assert int(SourceClass.TOOL_OUTPUT) == 9
    assert int(SourceClass.COMMUNICATED) == 10
    assert int(SourceClass.EXTERNAL_ARTIFACT) == 11
    assert int(InternalAction.ABSTAIN_OR_REQUEST_EXTERNAL_EVIDENCE) == 20
    assert int(InternalAction.RECONSTRUCT_LOCAL) == 21
    assert int(InternalAction.QUERY_TOOL) == 30


def test_legacy_boundaries_map_to_scoped_continuity_without_identity_reset():
    classes = torch.tensor([
        int(BoundaryClass.NONE), int(BoundaryClass.SOFT),
        int(BoundaryClass.SEGMENT), int(BoundaryClass.HARD),
    ])
    scopes = legacy_scope(classes)
    assert scopes.tolist() == [
        int(BoundaryScope.NONE), int(BoundaryScope.EVENT),
        int(BoundaryScope.SEGMENT), int(BoundaryScope.DOCUMENT),
    ]
    assert int(BoundaryScope.IDENTITY_RESET) not in scopes.tolist()


def test_boundary_context_transitions_are_typed_monotone_and_dtype_stable():
    state = BoundaryContextState.empty(2)
    updated = state.transition(torch.tensor([
        int(BoundaryScope.DOCUMENT), int(BoundaryScope.NONE),
    ]), continuity_ids=torch.tensor([7, 8]))
    assert updated.scope.tolist() == [int(BoundaryScope.DOCUMENT), int(BoundaryScope.NONE)]
    assert updated.continuity_ids.tolist() == [7, -1]
    assert updated.sequence_numbers.tolist() == [1, 1]
    assert updated.reset_counts.tolist() == [1, 0]
    moved = updated.to(dtype=torch.float64)
    assert moved.scope.dtype == torch.int64
    assert moved.discontinuity.dtype == torch.bool
    with pytest.raises(ValueError):
        state.transition(torch.tensor([99, 0]))


def test_all_v4_foundation_states_are_empty_bounded_detachable_and_dtype_safe():
    states = (
        ReconstructionState.empty(2, 3, 8, 4),
        AbstractionValidityState.empty(2, 3),
        ActionCandidateState.empty(2, 4, 5, 3),
        ViabilityState.empty(2, 6),
        EvidenceRequestState.empty(2, 3, 8, 4, 2),
        ExternalArtifactState.empty(2, 3, 16),
        MetacognitiveState.empty(2, 3),
    )
    for state in states:
        assert not state.active.any()
        detached = state.detach()
        moved = state.to(dtype=torch.float64)
        assert type(detached) is type(state)
        assert type(moved) is type(state)
        for name in state.__dataclass_fields__:
            original = getattr(state, name)
            converted = getattr(moved, name)
            if isinstance(original, torch.Tensor) and not original.is_floating_point():
                assert converted.dtype == original.dtype


def test_reconstruction_state_forbids_inferred_or_unprovenanced_active_rows():
    state = ReconstructionState.empty(1, 2, 4, 2)
    bad = copy.copy(state)
    active = state.active.clone(); active[0, 0] = True
    object.__setattr__(bad, "active", active)
    with pytest.raises(ValueError, match="provenance"):
        bad.__post_init__()

    bad = copy.copy(state)
    provenance = state.provenance_ids.clone(); provenance[0, 0] = 4
    sources = state.source_classes.clone(); sources[0, 0] = int(SourceClass.INFERRED)
    object.__setattr__(bad, "active", active)
    object.__setattr__(bad, "provenance_ids", provenance)
    object.__setattr__(bad, "source_classes", sources)
    with pytest.raises(ValueError, match="reconstructed source"):
        bad.__post_init__()


def test_viability_authority_requires_provenance_and_consistent_envelopes():
    state = ViabilityState.empty(1, 2)
    bad = copy.copy(state)
    active = state.active.clone(); active[0, 0] = True
    authority = state.authority_mask.clone(); authority[0, 0] = True
    object.__setattr__(bad, "active", active)
    object.__setattr__(bad, "authority_mask", authority)
    with pytest.raises(ValueError, match="provenance"):
        bad.__post_init__()


def test_action_candidate_selection_is_single_and_only_from_active_rows():
    state = ActionCandidateState.empty(1, 2, 3, 2)
    bad = copy.copy(state)
    selected = state.selected.clone(); selected[0, 0] = True
    object.__setattr__(bad, "selected", selected)
    with pytest.raises(ValueError, match="active"):
        bad.__post_init__()
