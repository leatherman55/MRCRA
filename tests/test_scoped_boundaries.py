from dataclasses import replace

import pytest
import torch

from mrrn.cognitive_model import MultimodalRelationalContinuityResonanceNetwork
from mrrn.cognitive_types import BoundaryScope
from test_cognitive_actions import action_config


def _marked_state(model, batch=1):
    state = model.initial_state(batch)
    event = replace(
        state.event_extractor,
        open=torch.ones_like(state.event_extractor.open),
        open_start_times=torch.ones_like(state.event_extractor.open_start_times),
    )
    goals = replace(
        state.goals,
        desired_outcomes=torch.ones_like(state.goals.desired_outcomes),
    )
    system = replace(
        state.system_model,
        action_availability=torch.ones_like(state.system_model.action_availability),
        permission_mask=torch.ones_like(state.system_model.permission_mask),
    )
    calibration = replace(
        state.calibration, counts=torch.ones_like(state.calibration.counts)
    )
    return replace(
        state,
        event_extractor=event,
        nodes=replace(state.nodes, content=torch.ones_like(state.nodes.content)),
        episodic_memory=replace(
            state.episodic_memory,
            values=torch.ones_like(state.episodic_memory.values),
        ),
        semantic_memory=replace(
            state.semantic_memory,
            values=torch.ones_like(state.semantic_memory.values),
        ),
        goals=goals,
        system_model=system,
        calibration=calibration,
        evidence_requests=replace(
            state.evidence_requests,
            proposition=torch.ones_like(state.evidence_requests.proposition),
        ),
        external_artifacts=replace(
            state.external_artifacts,
            content_digests=torch.ones_like(state.external_artifacts.content_digests),
        ),
        previous_latent=torch.ones_like(state.previous_latent),
    )


@pytest.mark.parametrize(
    "scope,carrier,nodes,episodic,semantic,goals,system,evidence,artifacts,calibration",
    [
        (BoundaryScope.EVENT, False, False, False, False, False, False, False, False, False),
        (BoundaryScope.SEGMENT, True, True, False, False, False, False, False, False, False),
        (BoundaryScope.DOCUMENT, True, True, True, False, False, False, True, False, False),
        (BoundaryScope.ENVIRONMENT_EPISODE, True, True, True, False, False, False, True, False, False),
        (BoundaryScope.SESSION, True, True, True, False, True, False, True, True, False),
        (BoundaryScope.IDENTITY_RESET, True, True, True, True, True, True, True, True, True),
        (BoundaryScope.STREAM_DISCONTINUITY, True, False, False, False, False, False, False, False, False),
    ],
)
def test_each_boundary_scope_resets_only_its_authorized_state(
    scope, carrier, nodes, episodic, semantic, goals, system, evidence,
    artifacts, calibration,
):
    model = MultimodalRelationalContinuityResonanceNetwork(action_config()).eval()
    marked = _marked_state(model)
    result = model.apply_boundary_scopes(
        marked, torch.tensor([int(scope)]),
        continuity_ids=torch.tensor([11]), environment_ids=torch.tensor([12]),
        session_ids=torch.tensor([13]),
    )
    assert (not bool(result.previous_latent.any())) is carrier
    assert (not bool(result.nodes.content.any())) is nodes
    assert (not bool(result.episodic_memory.values.any())) is episodic
    assert (not bool(result.semantic_memory.values.any())) is semantic
    assert (not bool(result.goals.desired_outcomes.any())) is goals
    assert (not bool(result.system_model.action_availability.any())) is system
    assert (not bool(result.evidence_requests.proposition.any())) is evidence
    assert (not bool(result.external_artifacts.content_digests.any())) is artifacts
    assert (not bool(result.calibration.counts.any())) is calibration
    assert not result.event_extractor.open.any()
    assert result.boundary_context.scope.item() == int(scope)
    assert result.boundary_context.continuity_ids.item() == 11
    assert result.boundary_context.reset_counts.item() == 1


def test_partial_identity_reset_fails_closed_because_calibration_is_global():
    model = MultimodalRelationalContinuityResonanceNetwork(action_config()).eval()
    state = _marked_state(model, batch=2)
    with pytest.raises(ValueError, match="complete runtime batch"):
        model.apply_boundary_scopes(
            state,
            torch.tensor([int(BoundaryScope.IDENTITY_RESET), int(BoundaryScope.NONE)]),
        )
