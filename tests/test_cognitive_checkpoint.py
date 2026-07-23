import pytest
import torch

from mrrn.cognitive_checkpoint import (
    load_mrcra_checkpoint, runtime_state_dict, runtime_state_from_dict,
    save_mrcra_checkpoint,
)
from mrrn.cognitive_model import MultimodalRelationalContinuityResonanceNetwork
from mrrn.cognitive_types import ModalityClass
from mrrn.config import CognitiveConfig, MRCRAConfig, MRRNConfig
from mrrn.observation import register_external_observations
from mrrn.provenance import ProvenanceLedger


def config():
    carrier = MRRNConfig(
        input_dim=8, model_dim=8, output_dim=5, layers=1, scales=3,
        heads=2, modes=2, mimo_rank=1, attention_window=2,
        retrieved_items=1, memory_capacity=4, width_multiple=4,
        mixer_expansion=1.5, spectral_modes=2, spectral_basis_order=2,
        enable_global_head=False, relational_branch=True, relational_context_dim=8,
    )
    cognitive = CognitiveConfig(
        workspace_dim=8, provenance_features=4, uncertainty_channels=8,
        relation_heads=2, relation_modes=2, relation_adapter_rank=2,
        goal_slots=2, goal_constraint_dim=2, system_action_channels=2,
        calibration_regimes=2, active_event_capacity=5, pair_edge_capacity=6,
        hyperedge_capacity=2, graph_neighbors=2, global_workspace_slots=2,
        hypothesis_slots=2, maximum_hypothesis_slots=2, maximum_cognitive_steps=2,
        event_chunk_size=2, event_proposals_per_chunk=2, recent_candidates=2,
        landmark_candidates=1, episodic_candidates=1, semantic_candidates=1,
        episodic_memory_capacity=3, semantic_memory_capacity=2,
        associative_depth=2, associative_budget=2, world_model_horizons=(1, 2),
    )
    return MRCRAConfig(carrier, cognitive, actor_parameter_minimum=1, actor_parameter_maximum=10_000_000)


def packet(values, ledger, *, start):
    batch, length = values.shape[:2]
    mask = torch.ones(batch, length, dtype=torch.bool)
    timestamps = torch.arange(start, start + length, dtype=values.dtype).expand(batch, -1)
    return register_external_observations(
        values, mask, observed_mask=mask, timestamps=timestamps,
        coordinates=timestamps.unsqueeze(-1), sample_intervals=torch.ones(batch),
        boundary_classes=torch.zeros(batch, length, dtype=torch.int64),
        modality_ids=torch.full((batch, length), int(ModalityClass.TEXT), dtype=torch.int64),
        uncertainty_seed=torch.zeros(batch, length, 8),
        segment_ids=torch.zeros(batch, length, dtype=torch.int64),
        source_uris=(f"doc://{start}",), ledger=ledger, model_authority="adapter",
    )


def force_events(model):
    proposal = model.event_extractor.proposal_network
    for parameter in proposal.parameters():
        parameter.data.zero_()
    proposal.proposal.bias.data.fill_(8)
    proposal.end.bias.data.fill_(8)


def test_full_runtime_provenance_optimizer_and_rng_resume_exactly(tmp_path):
    torch.manual_seed(107)
    model = MultimodalRelationalContinuityResonanceNetwork(config(), model_authority="checkpoint-test").double()
    force_events(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ledger = ProvenanceLedger()
    first_values = torch.randn(1, 3, 8, dtype=torch.float64)
    first = model(packet(first_values, ledger, start=0), ledger)
    # Populate optimizer moments without changing the weights after checkpoint hashing.
    optimizer.zero_grad(set_to_none=True)
    first.prediction.square().mean().backward()
    optimizer.step()
    # Re-run state under the updated weights, then checkpoint all authorities.
    ledger = ProvenanceLedger()
    first = model(packet(first_values, ledger, start=0), ledger)
    path = tmp_path / "mrcra.pt"
    torch.manual_seed(999)
    save_mrcra_checkpoint(
        path, model, first.state, ledger, optimizer=optimizer,
        metadata={"tokenizer": "test-v1", "external_clock": 3},
    )
    expected_random = torch.rand(4)

    second_values = torch.randn(1, 3, 8, dtype=torch.float64)
    direct_packet = packet(second_values, ledger, start=3)
    direct = model(direct_packet, ledger, state=first.state)

    restored_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    restored_state, restored_ledger, metadata = load_mrcra_checkpoint(
        path, model, optimizer=restored_optimizer,
    )
    torch.testing.assert_close(torch.rand(4), expected_random)
    restored_packet = packet(second_values, restored_ledger, start=3)
    restored = model(restored_packet, restored_ledger, state=restored_state)
    torch.testing.assert_close(restored.prediction, direct.prediction, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(restored.nodes.content, direct.nodes.content, atol=1e-12, rtol=1e-12)
    assert torch.equal(restored.relations.participant_indices, direct.relations.participant_indices)
    assert restored_ledger.digest() == ledger.digest()
    assert metadata == {"tokenizer": "test-v1", "external_clock": 3}
    assert restored_optimizer.state_dict()["state"]


def test_checkpoint_rejects_weight_or_authority_mismatch(tmp_path):
    model = MultimodalRelationalContinuityResonanceNetwork(config(), model_authority="a")
    ledger = ProvenanceLedger()
    output = model(packet(torch.randn(1, 1, 8), ledger, start=0), ledger)
    path = tmp_path / "mrcra.pt"
    save_mrcra_checkpoint(path, model, output.state, ledger)
    other = MultimodalRelationalContinuityResonanceNetwork(config(), model_authority="a")
    other.load_state_dict(model.state_dict())
    next(other.parameters()).data.add_(1)
    try:
        load_mrcra_checkpoint(path, other)
        assert False, "weight mismatch should fail"
    except ValueError as error:
        assert "weights" in str(error)
    authority = MultimodalRelationalContinuityResonanceNetwork(config(), model_authority="b")
    authority.load_state_dict(model.state_dict())
    try:
        load_mrcra_checkpoint(path, authority)
        assert False, "authority mismatch should fail"
    except ValueError as error:
        assert "authority" in str(error)


def test_v3_checkpoint_migrates_to_conservative_empty_v4_foundation(tmp_path):
    torch.manual_seed(131)
    model = MultimodalRelationalContinuityResonanceNetwork(
        config(), model_authority="migration-test"
    )
    ledger = ProvenanceLedger()
    output = model(packet(torch.randn(1, 2, 8), ledger, start=0), ledger)
    current = tmp_path / "current.pt"
    legacy = tmp_path / "legacy.pt"
    save_mrcra_checkpoint(current, model, output.state, ledger, metadata={"name": "v3"})
    payload = torch.load(current, weights_only=True)
    payload["format_version"] = 3
    for name in (
        "reconstructions", "abstraction_validity", "action_candidates", "viability",
        "evidence_requests", "external_artifacts", "metacognition", "boundary_context",
    ):
        payload["runtime"].pop(name)
    torch.save(payload, legacy)

    state, restored_ledger, metadata = load_mrcra_checkpoint(legacy, model)
    assert not state.reconstructions.active.any()
    assert not state.abstraction_validity.active.any()
    assert not state.action_candidates.active.any()
    assert not state.viability.active.any()
    assert not state.evidence_requests.active.any()
    assert not state.external_artifacts.active.any()
    assert not state.metacognition.active.any()
    assert state.boundary_context.reset_counts.tolist() == [0]
    assert restored_ledger.digest() == ledger.digest()
    assert metadata == {
        "name": "v3", "mrcra_migrated_from_format": 3,
        "mrcra_migrated_to_format": 5,
    }


def test_partial_v4_runtime_state_fails_closed():
    model = MultimodalRelationalContinuityResonanceNetwork(config())
    value = runtime_state_dict(model.initial_state(1))
    value.pop("viability")
    with pytest.raises(ValueError, match="partial v4 foundation state"):
        runtime_state_from_dict(value, cognitive=model.config.cognitive)
