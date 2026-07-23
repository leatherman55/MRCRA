import json

import pytest
import torch

from mrrn.cognitive_model import MultimodalRelationalContinuityResonanceNetwork
from mrrn.cognitive_diagnostics import cognitive_evidence, cognitive_metrics
from mrrn.cognitive_types import BoundaryClass, BoundaryScope, ModalityClass
from mrrn.config import CognitiveConfig, MRCRAConfig, MRRNConfig
from mrrn.observation import register_external_observations
from mrrn.provenance import ProvenanceLedger


def tiny_config():
    carrier = MRRNConfig(
        input_dim=8, model_dim=8, output_dim=5, layers=1, scales=3,
        heads=2, modes=3, mimo_rank=1, attention_window=3,
        retrieved_items=2, memory_capacity=8, mixer_expansion=1.5,
        width_growth_cap=1.5, mode_growth_cap=1.5, width_multiple=4,
        spectral_modes=2, spectral_basis_order=2, spectral_triads_per_mode=1,
        enable_global_head=False, relational_branch=True, relational_context_dim=8,
    )
    cognitive = CognitiveConfig(
        workspace_dim=8, provenance_features=4, uncertainty_channels=8,
        relation_heads=2, relation_modes=2, relation_adapter_rank=2,
        goal_slots=2, goal_constraint_dim=2, system_action_channels=2,
        calibration_regimes=2, active_event_capacity=8, pair_edge_capacity=16,
        hyperedge_capacity=4, maximum_hyperedge_arity=4, graph_neighbors=2,
        global_workspace_slots=2, hypothesis_slots=2, maximum_hypothesis_slots=4,
        maximum_cognitive_steps=2, event_chunk_size=2, event_proposals_per_chunk=2,
        recent_candidates=3, landmark_candidates=1, episodic_candidates=2,
        semantic_candidates=2, episodic_memory_capacity=8,
        semantic_memory_capacity=4, associative_depth=2, associative_budget=2,
        world_model_horizons=(1, 2),
    )
    return MRCRAConfig(carrier, cognitive, actor_parameter_minimum=1, actor_parameter_maximum=10_000_000)


def packet(values, ledger, *, boundaries=None, segments=None, uri="doc://x"):
    batch, length = values.shape[:2]
    valid = torch.ones(batch, length, dtype=torch.bool)
    boundaries = (
        torch.zeros(batch, length, dtype=torch.int64)
        if boundaries is None else boundaries
    )
    segments = (
        torch.zeros(batch, length, dtype=torch.int64)
        if segments is None else segments
    )
    timestamps = torch.arange(length, dtype=values.dtype).expand(batch, -1)
    return register_external_observations(
        values, valid, observed_mask=valid, timestamps=timestamps,
        coordinates=timestamps.unsqueeze(-1), sample_intervals=torch.ones(batch),
        boundary_classes=boundaries,
        modality_ids=torch.full((batch, length), int(ModalityClass.TEXT), dtype=torch.int64),
        uncertainty_seed=torch.zeros(batch, length, 8), segment_ids=segments,
        source_uris=tuple(f"{uri}/{index}" for index in range(batch)), ledger=ledger,
        model_authority="adapter-v1",
    )


def force_events(model):
    proposal = model.event_extractor.proposal_network
    for parameter in proposal.parameters():
        parameter.data.zero_()
    proposal.proposal.bias.data.fill_(8)
    proposal.end.bias.data.fill_(8)


def test_v4_foundation_state_is_live_empty_and_document_scoped_by_legacy_hard_boundary():
    model = MultimodalRelationalContinuityResonanceNetwork(tiny_config())
    state = model.initial_state(1)
    assert not state.reconstructions.active.any()
    assert not state.abstraction_validity.active.any()
    assert not state.action_candidates.active.any()
    assert not state.viability.active.any()
    assert not state.evidence_requests.active.any()
    assert not state.external_artifacts.active.any()
    assert not state.metacognition.active.any()
    reset = model._reset_boundaries(
        state, torch.tensor([int(BoundaryClass.HARD)]), torch.ones(1),
    )
    assert reset.boundary_context.scope.tolist() == [int(BoundaryScope.DOCUMENT)]
    assert reset.boundary_context.reset_counts.tolist() == [1]
    assert reset.boundary_context.sequence_numbers.tolist() == [1]


def test_end_to_end_cognitive_stream_builds_bounded_typed_state_and_backpropagates():
    torch.manual_seed(83)
    model = MultimodalRelationalContinuityResonanceNetwork(tiny_config())
    force_events(model)
    ledger = ProvenanceLedger()
    values = torch.randn(1, 5, 8, requires_grad=True)
    observed = packet(values, ledger)
    output = model(observed, ledger)
    assert output.prediction.shape == (1, 5, 5)
    assert output.latent.shape == (1, 5, 8)
    assert output.uncertainty.shape == (1, 5, 8)
    assert output.event_counts.sum() == 5
    assert output.nodes.active.sum() <= 8
    assert output.relations.active.sum() <= 16
    assert output.workspace.active.sum() <= 2
    assert output.state.clocks.external == values.shape[1]
    expected_microstep_rounds = int(output.action_receipts.mask.any(0).sum())
    assert output.state.clocks.cognitive == expected_microstep_rounds
    assert len(ledger) > 1
    assert output.provenance_digest == ledger.digest()
    output.prediction.square().mean().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()


def test_future_perturbation_does_not_change_earlier_cognitive_outputs_or_cycles():
    torch.manual_seed(89)
    model = MultimodalRelationalContinuityResonanceNetwork(tiny_config()).double()
    force_events(model)
    values = torch.randn(1, 6, 8, dtype=torch.float64)
    first_ledger = ProvenanceLedger()
    first = model(packet(values, first_ledger), first_ledger)
    changed = values.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 100
    second_ledger = ProvenanceLedger()
    second = model(packet(changed, second_ledger), second_ledger)
    torch.testing.assert_close(first.prediction[:, :4], second.prediction[:, :4], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(first.uncertainty[:, :4], second.uncertainty[:, :4], atol=1e-10, rtol=1e-10)
    assert torch.equal(first.cognitive_cycles[:, :4], second.cognitive_cycles[:, :4])
    assert torch.equal(first.event_counts[:, :4], second.event_counts[:, :4])


def test_asynchronous_segment_boundary_prevents_cross_document_state_leakage():
    torch.manual_seed(97)
    model = MultimodalRelationalContinuityResonanceNetwork(tiny_config()).double()
    force_events(model)
    values = torch.randn(2, 5, 8, dtype=torch.float64)
    boundaries = torch.zeros(2, 5, dtype=torch.int64)
    boundaries[0, 2] = int(BoundaryClass.SEGMENT)
    segments = torch.zeros(2, 5, dtype=torch.int64)
    segments[0, 2:] = 1
    ledger_a = ProvenanceLedger()
    output_a = model(packet(values, ledger_a, boundaries=boundaries, segments=segments), ledger_a)
    changed = values.clone()
    changed[0, :2] *= 100
    ledger_b = ProvenanceLedger()
    output_b = model(packet(changed, ledger_b, boundaries=boundaries, segments=segments), ledger_b)
    torch.testing.assert_close(output_a.prediction[0, 2:], output_b.prediction[0, 2:], atol=1e-10, rtol=1e-10)
    # The other batch row was not reset and therefore remains an ordinary stream.
    torch.testing.assert_close(output_a.prediction[1], output_b.prediction[1], atol=1e-10, rtol=1e-10)


def test_integrated_multirate_training_is_causal_and_pressurizes_controller():
    torch.manual_seed(101)
    model = MultimodalRelationalContinuityResonanceNetwork(tiny_config()).double()
    values = torch.randn(1, 8, 8, dtype=torch.float64)
    ledger_a = ProvenanceLedger()
    first = model.forward_integrated_training(
        packet(values, ledger_a), ledger_a,
        cognitive_stride=2, cognitive_tbptt_events=2,
    )
    changed = values.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 100
    ledger_b = ProvenanceLedger()
    second = model.forward_integrated_training(
        packet(changed, ledger_b), ledger_b,
        cognitive_stride=2, cognitive_tbptt_events=2,
    )
    torch.testing.assert_close(
        first.output_latent[:, :4], second.output_latent[:, :4],
        atol=1e-10, rtol=1e-10,
    )
    first.output_latent.square().mean().backward()
    assert model.output_context_adapter.weight.grad is not None
    assert torch.linalg.vector_norm(model.output_context_adapter.weight.grad) > 0
    assert model.controller.action_head.weight.grad is not None
    assert torch.linalg.vector_norm(model.controller.action_head.weight.grad) > 0


def test_integrated_training_preserves_complete_carrier_state_across_tbptt_spans():
    torch.manual_seed(102)
    model = MultimodalRelationalContinuityResonanceNetwork(tiny_config()).double()
    ledger = ProvenanceLedger()
    first = model.forward_integrated_training(
        packet(torch.randn(1, 8, 8, dtype=torch.float64), ledger), ledger,
        cognitive_stride=2, cognitive_tbptt_events=2,
    )
    second = model.forward_integrated_training(
        packet(
            torch.randn(1, 8, 8, dtype=torch.float64),
            ledger,
            uri="doc://continuation",
        ),
        ledger,
        state=first.state.detach(),
        cognitive_stride=2,
        cognitive_tbptt_events=2,
    )
    assert first.state.carrier.position == first.state.carrier.lifting.steps == 8
    assert second.state.carrier.position == second.state.carrier.lifting.steps == 16
    assert all(item is None for item in second.state.carrier.lifting.pending)
    assert all(
        block.scale_steps[0] > 0 for block in second.state.carrier.blocks
    )
    assert all(
        block.exchange.latest_coarse[0] is not None
        for block in second.state.carrier.blocks
    )
    assert all(
        block.recent_features[0] for block in second.state.carrier.blocks
    )


def test_inactive_hard_events_receive_differentiable_environmental_credit():
    torch.manual_seed(103)
    model = MultimodalRelationalContinuityResonanceNetwork(tiny_config()).double()
    proposal = model.event_extractor.proposal_network
    with torch.no_grad():
        proposal.proposal.weight.zero_()
        proposal.proposal.bias.fill_(-8)
        proposal.end.weight.zero_()
        proposal.end.bias.fill_(-8)
    ledger = ProvenanceLedger()
    result = model.forward_integrated_training(
        packet(torch.randn(1, 8, 8, dtype=torch.float64), ledger), ledger,
        cognitive_stride=2, cognitive_tbptt_events=2,
    )
    assert result.event_counts.sum() == 0
    assert 0 < result.event_activation_mean < 0.001
    result.output_latent.square().mean().backward()
    for layer in (
        proposal.proposal, proposal.end, proposal.content,
        proposal.identity, proposal.node_type,
    ):
        assert layer.weight.grad is not None
        assert torch.linalg.vector_norm(layer.weight.grad) > 0


def test_integrated_cognition_arms_isolate_hard_soft_and_off_paths_exactly():
    torch.manual_seed(104)
    model = MultimodalRelationalContinuityResonanceNetwork(tiny_config()).double()
    force_events(model)
    values = torch.randn(1, 8, 8, dtype=torch.float64)
    outputs = {}
    for mode in ("full", "soft_only", "off"):
        ledger = ProvenanceLedger()
        outputs[mode] = model.forward_integrated_training(
            packet(values, ledger), ledger,
            cognitive_stride=2, cognitive_tbptt_events=2,
            cognition_mode=mode,
        )
    full, soft, off = outputs["full"], outputs["soft_only"], outputs["off"]
    assert full.event_counts.sum() > 0
    assert full.event_emitted.sum() == full.event_counts.sum()
    assert full.first_hard_event is not None
    assert full.state.cognitive.nodes.active.any()
    assert not soft.state.cognitive.nodes.active.any()
    assert soft.event_emitted.sum() == full.event_emitted.sum()
    assert not off.cognitive_cycles.any()
    assert not off.event_counts.any()
    initial = model.initial_state(
        1, sample_intervals=torch.ones(1, dtype=torch.float64),
        dtype=torch.float64,
    )
    carrier = model.carrier.prefill(
        values, torch.ones(1, 8, dtype=torch.bool),
        relational_context=initial.relational_context,
        project_output=False,
    )
    torch.testing.assert_close(off.output_latent, carrier.latent)
    assert torch.linalg.vector_norm(full.output_latent - soft.output_latent) > 0
    assert full.event_proposal_logits.shape == full.event_end_logits.shape == (1, 4)


def test_fifth_relational_branch_is_zero_context_equivalent_and_context_sensitive():
    model = MultimodalRelationalContinuityResonanceNetwork(tiny_config()).double()
    carrier = model.carrier
    x = torch.randn(1, 5, 8, dtype=torch.float64)
    without = carrier(x).prediction
    zero = carrier(x, relational_context=torch.zeros(1, 5, 8, dtype=torch.float64)).prediction
    context = carrier(x, relational_context=torch.ones(1, 5, 8, dtype=torch.float64)).prediction
    torch.testing.assert_close(without, zero, atol=1e-12, rtol=1e-12)
    assert torch.linalg.vector_norm(without - context) > 1e-10


def test_mrcra_requires_a_relational_carrier_and_matching_dimensions():
    with pytest.raises(ValueError):
        MRCRAConfig(
            MRRNConfig(input_dim=8, model_dim=8),
            CognitiveConfig(workspace_dim=8),
        )
    model = MultimodalRelationalContinuityResonanceNetwork(tiny_config())
    assert model.validate_serious_parameter_count() == model.parameter_count


def test_cognitive_diagnostics_are_finite_bounded_and_provenance_linked():
    torch.manual_seed(251)
    model = MultimodalRelationalContinuityResonanceNetwork(tiny_config())
    force_events(model)
    ledger = ProvenanceLedger()
    values = torch.randn(1, 5, 8)
    output = model(packet(values, ledger), ledger)
    metrics = cognitive_metrics(output, ledger)
    assert metrics["cognition/provenance_records"] == len(ledger)
    assert (
        metrics["cognition/active_pair_relations"]
        + metrics["cognition/active_hyperrelations"]
        == metrics["cognition/active_relations"]
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    evidence = cognitive_evidence(output, ledger, maximum_records=4)
    assert evidence["schema_version"] == 4
    assert evidence["provenance"]["digest"] == output.provenance_digest
    assert len(evidence["graph"]["nodes"]) <= 4
    assert len(evidence["graph"]["relations"]) <= 4
    assert "metacognition" in evidence
    assert len(evidence["metacognition"]["steps"]) <= 4
    assert "cognition/metacognitive_routed_positions" in metrics
    json.dumps(evidence, allow_nan=False)
