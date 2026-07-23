import copy

import pytest
import torch

from mrrn.cognitive_types import (
    BoundaryClass,
    CognitiveClocks,
    ModalityClass,
    NodeSlots,
    NodeType,
    RelationFamily,
    RelationSlots,
    SourceClass,
    SupportInterval,
    VerificationClass,
    RELATION_COMPATIBILITY,
)
from mrrn.config import CognitiveConfig, MRCRAConfig, MRRNConfig
from mrrn.language import MRCRALanguageModel
from mrrn.observation import ObservationPacket, register_external_observations
from mrrn.provenance import ProvenanceLedger


def test_cognitive_configuration_is_bounded_and_canonical():
    config = CognitiveConfig()
    assert config.active_event_capacity == 256
    assert config.pair_edge_capacity == 2048
    assert config.maximum_router_candidates == 80
    assert config.world_model_horizons == (1, 4, 16, 64)
    serious = MRCRAConfig.serious_120m(output_dim=50_257)
    assert serious.carrier.scale_configs()[0].width == 256
    assert [item.width for item in serious.carrier.scale_configs()[1:]] == [288] * 5
    assert [item.modes for item in serious.carrier.scale_configs()] == [20, 25, 25, 25, 25, 25]
    assert all(
        getattr(serious.cognitive, name)
        for name in (
            "enable_conditional_reconstruction",
            "enable_abstraction_validity_control",
            "enable_post_deliberation_action_selection",
            "enable_multi_hypothesis_planning",
            "enable_agent_session_loop",
            "enable_viability_gate",
            "enable_integrated_invariant_discovery",
            "enable_persistent_session_training",
            "enable_metacognitive_routing",
        )
    )
    with pytest.raises(ValueError):
        CognitiveConfig(hypothesis_slots=9)
    with pytest.raises(ValueError):
        CognitiveConfig(workspace_dim=255)
    with pytest.raises(ValueError):
        CognitiveConfig(world_model_horizons=(4, 1))
    with pytest.raises(ValueError):
        MRCRAConfig(MRRNConfig(input_dim=8, model_dim=8), CognitiveConfig(workspace_dim=16))
    with pytest.raises(ValueError):
        serious.require_actor_parameter_count(1)


def test_light_8p4m_profile_preserves_integrated_cognition_and_exact_budget():
    light = MRCRAConfig.light_8p4m(output_dim=50_257)
    assert light.carrier.layers == 6
    assert light.carrier.share_depth_parameters is True
    assert light.carrier.structured_mixer_rank == 8
    assert [item.width for item in light.carrier.scale_configs()] == [96, 112, 112, 112, 112]
    assert [item.modes for item in light.carrier.scale_configs()] == [12, 15, 15, 15, 15]
    assert light.cognitive.active_event_capacity == 128
    assert light.cognitive.pair_edge_capacity == 512
    assert light.cognitive.episodic_memory_capacity == 4096
    assert all(
        getattr(light.cognitive, name)
        for name in (
            "enable_conditional_reconstruction",
            "enable_abstraction_validity_control",
            "enable_post_deliberation_action_selection",
            "enable_multi_hypothesis_planning",
            "enable_agent_session_loop",
            "enable_viability_gate",
            "enable_integrated_invariant_discovery",
            "enable_persistent_session_training",
            "enable_metacognitive_routing",
        )
    )
    model = MRCRALanguageModel(light, model_authority="light-profile-test")
    assert model.parameter_count == 8_413_442
    light.require_actor_parameter_count(model.parameter_count)
    with pytest.raises(ValueError):
        light.require_actor_parameter_count(8_500_000)


def test_node_and_relation_slots_enforce_authoritative_masks():
    nodes = NodeSlots.empty(
        2, 5, 8, heads=2, modes=3, node_types=len(NodeType), modalities=16,
        uncertainty_channels=8, provenance_features=4, hypotheses=4,
    )
    assert nodes.content.shape == (2, 5, 8) and not nodes.active.any()
    assert nodes.detach().content.data_ptr() == nodes.content.data_ptr()
    bad = copy.copy(nodes)
    object.__setattr__(bad, "provenance_ids", torch.zeros(2, 5, dtype=torch.int64))
    with pytest.raises(ValueError):
        bad.__post_init__()

    relations = RelationSlots.empty(
        2, 7, 8, relation_families=len(RelationFamily), arity=4,
        uncertainty_channels=8, hypotheses=4,
    )
    assert relations.participant_indices.shape == (2, 7, 4)
    broken = copy.copy(relations)
    participant_mask = relations.participant_mask.clone()
    participant_mask[0, 0, 0] = True
    object.__setattr__(broken, "participant_mask", participant_mask)
    with pytest.raises(ValueError):
        broken.__post_init__()


def test_relation_ontology_blocks_definitionally_invalid_causal_and_role_edges():
    assert not RELATION_COMPATIBILITY[
        NodeType.SYMBOL, NodeType.SYMBOL, RelationFamily.CAUSAL_INFLUENCE
    ]
    assert RELATION_COMPATIBILITY[
        NodeType.ACTION, NodeType.EVENT, RelationFamily.CAUSAL_INFLUENCE
    ]
    assert RELATION_COMPATIBILITY[
        NodeType.ENTITY, NodeType.EVENT, RelationFamily.EVENT_PARTICIPATION
    ]
    assert not RELATION_COMPATIBILITY[
        NodeType.GOAL, NodeType.MEMORY, RelationFamily.EVENT_PARTICIPATION
    ]


def test_three_clocks_advance_independently():
    clocks = CognitiveClocks(3, 2, 7)
    assert clocks.observation_tick() == CognitiveClocks(4, 2, 7)
    assert clocks.cognitive_tick() == CognitiveClocks(3, 3, 7)
    assert clocks.optimizer_tick() == CognitiveClocks(3, 2, 8)
    assert clocks.observation_tick(4).cognitive_tick(3).optimizer_tick(2) == CognitiveClocks(7, 5, 9)
    with pytest.raises(ValueError):
        clocks.cognitive_tick(-1)


def test_provenance_is_append_only_acyclic_scenario_safe_and_revocation_propagates():
    ledger = ProvenanceLedger()
    first = ledger.append(
        source_class=SourceClass.EXTERNAL,
        source_uri_or_episode="dataset://example/1",
        support=SupportInterval(0, 1, 1),
        modality=ModalityClass.TEXT,
        operator="tokenizer:v1",
        scenario_id=0,
        model_authority="model-sha",
        verification=VerificationClass.EXTERNALLY_CHECKED,
    )
    inferred = ledger.derive(
        [first], source_class=SourceClass.INFERRED, operator="eventizer",
        support=SupportInterval(0, 1, 1), modality=ModalityClass.TEXT,
        model_authority="model-sha",
    )
    simulated = ledger.derive(
        [inferred], source_class=SourceClass.SIMULATED, operator="world-model",
        support=SupportInterval(1, 2, 2), modality=ModalityClass.PREDICTION,
        scenario_id=5, model_authority="model-sha",
    )
    assert ledger.lineage(simulated) == (first, inferred)
    assert ledger.independent_roots(simulated) == (first,)
    assert not ledger.can_consolidate(simulated)
    original = ledger.get(first)
    ledger.set_verification(first, VerificationClass.REVOKED, authority="dataset", reason="withdrawn")
    assert ledger.get(first) is original
    assert ledger.get(first).source_class == SourceClass.EXTERNAL
    assert ledger.effective_verification(simulated) == VerificationClass.REVOKED
    assert ledger.descendants(first) == (inferred, simulated)
    with pytest.raises(ValueError):
        ledger.derive(
            [simulated], source_class=SourceClass.INFERRED, operator="leak",
            support=SupportInterval(2, 3, 3), modality=ModalityClass.TEXT,
            scenario_id=0, model_authority="model-sha",
        )
    with pytest.raises(ValueError):
        ledger.append(
            source_class=SourceClass.EXTERNAL, source_uri_or_episode="bad",
            support=SupportInterval(0, 0, 0), modality=ModalityClass.TEXT,
            parents=[first], operator="rewrite", scenario_id=0, model_authority="model-sha",
        )


def test_provenance_checkpoint_roundtrip_is_exact_and_detects_invalid_order():
    ledger = ProvenanceLedger()
    root = ledger.append(
        source_class=SourceClass.EXTERNAL, source_uri_or_episode="sensor://1",
        support=SupportInterval(2, 2, 2), modality=ModalityClass.SENSOR,
        operator="adapter", scenario_id=0, model_authority="m",
    )
    child = ledger.derive(
        [root], source_class=SourceClass.ABSTRACTED, operator="compress",
        support=SupportInterval(2, 2, 2), modality=ModalityClass.SENSOR,
        model_authority="m",
    )
    ledger.set_verification(
        child, VerificationClass.INTERNALLY_CONSISTENT, authority="test", reason="roundtrip"
    )
    saved = ledger.state_dict()
    restored = ProvenanceLedger()
    restored.load_state_dict(saved)
    assert restored.digest() == ledger.digest()
    assert restored.records() == ledger.records()
    broken = copy.deepcopy(saved)
    broken["records"][1]["parents"] = (9,)
    with pytest.raises(ValueError):
        ProvenanceLedger().load_state_dict(broken)


def _packet_inputs():
    values = torch.randn(2, 4, 6)
    values[1, 3] = 0
    valid = torch.tensor([[True, True, True, True], [True, True, True, False]])
    observed = valid.clone()
    observed[0, 2] = False
    timestamps = torch.tensor([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 0.0]])
    coordinates = timestamps.unsqueeze(-1)
    intervals = torch.tensor([1.0, 1.0])
    boundaries = torch.tensor([
        [BoundaryClass.HARD, BoundaryClass.NONE, BoundaryClass.SOFT, BoundaryClass.NONE],
        [BoundaryClass.SEGMENT, BoundaryClass.NONE, BoundaryClass.NONE, BoundaryClass.NONE],
    ], dtype=torch.int64)
    modalities = torch.full((2, 4), int(ModalityClass.TEXT), dtype=torch.int64)
    uncertainty = torch.zeros(2, 4, 8)
    uncertainty[~observed] = 1
    segments = torch.tensor([[0, 0, 0, 0], [1, 1, 1, -1]], dtype=torch.int64)
    return (
        values, valid, observed, timestamps, coordinates, intervals,
        boundaries, modalities, uncertainty, segments,
    )


def test_observation_registration_preserves_masks_boundaries_segments_and_authority():
    ledger = ProvenanceLedger()
    packet = register_external_observations(
        *_packet_inputs()[:2], observed_mask=_packet_inputs()[2],
        timestamps=_packet_inputs()[3], coordinates=_packet_inputs()[4],
        sample_intervals=_packet_inputs()[5], boundary_classes=_packet_inputs()[6],
        modality_ids=_packet_inputs()[7], uncertainty_seed=_packet_inputs()[8],
        segment_ids=_packet_inputs()[9], source_uris=("doc://a", "doc://b"),
        ledger=ledger, model_authority="adapter-v1",
    )
    assert len(ledger) == 2
    assert packet.hard_reset_mask[0, 0]
    assert packet.segment_reset_mask[1, 0]
    assert packet.soft_boundary_mask[0, 2]
    assert not packet.observed_mask[0, 2] and packet.valid_mask[0, 2]
    assert packet.source_record_ids[1, 3] == -1
    packet.assert_ledger_consistent(ledger)


def test_observation_packets_fail_closed_before_ledger_mutation():
    args = list(_packet_inputs())
    ledger = ProvenanceLedger()
    args[3] = args[3].clone()
    args[3][0, 2] = -1
    with pytest.raises(ValueError):
        register_external_observations(
            args[0], args[1], observed_mask=args[2], timestamps=args[3],
            coordinates=args[4], sample_intervals=args[5], boundary_classes=args[6],
            modality_ids=args[7], uncertainty_seed=args[8], segment_ids=args[9],
            source_uris=("a", "b"), ledger=ledger, model_authority="m",
        )
    assert len(ledger) == 0
    with pytest.raises(ValueError):
        SupportInterval(2, 1, 3)
