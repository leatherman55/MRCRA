from dataclasses import replace

import pytest
import torch

from mrrn.cognitive_objectives import ObjectiveFamily, combine_cognitive_objectives, CognitiveObjectiveSchedule
from mrrn.cognitive_supervision import (
    CognitiveSupervisionTargets, EvidenceBackedCognitiveSupervisor,
    language_evidence_targets,
)
from mrrn.config import CognitiveConfig, MRCRAConfig, MRRNConfig
from mrrn.language import MRCRALanguageModel
from mrrn.lm_training import ByteTextTokenizer, PackedTokenStream, SequenceTextSource


def tiny_config():
    carrier = MRRNConfig(
        input_dim=8, model_dim=8, output_dim=257, layers=1, scales=2,
        heads=2, modes=2, mimo_rank=1, attention_window=2,
        attention_query_tile_size=2, retrieved_items=1, memory_capacity=4,
        mixer_expansion=1.5, width_growth_cap=1, mode_growth_cap=1,
        width_multiple=2, spectral_modes=2, spectral_basis_order=2,
        spectral_triads_per_mode=1, enable_global_head=False,
        relational_branch=True, relational_context_dim=8,
    )
    cognition = CognitiveConfig(
        workspace_dim=8, provenance_features=4, uncertainty_channels=8,
        relation_heads=2, relation_modes=2, relation_adapter_rank=2,
        goal_slots=1, goal_constraint_dim=2, system_action_channels=2,
        calibration_regimes=2, active_event_capacity=4, pair_edge_capacity=8,
        hyperedge_capacity=2, maximum_hyperedge_arity=3, graph_neighbors=1,
        global_workspace_slots=2, hypothesis_slots=1, maximum_hypothesis_slots=2,
        maximum_cognitive_steps=1, event_chunk_size=2,
        event_proposals_per_chunk=1, recent_candidates=2, landmark_candidates=1,
        episodic_candidates=1, semantic_candidates=1, episodic_memory_capacity=4,
        semantic_memory_capacity=2, associative_depth=1, associative_budget=1,
        world_model_horizons=(1,),
    )
    return MRCRAConfig(carrier, cognition, actor_parameter_minimum=1, actor_parameter_maximum=10_000_000)


def packed():
    tokenizer = ByteTextTokenizer()
    return PackedTokenStream(
        SequenceTextSource(("ab", "cd")), tokenizer
    ).next_batch(1, 4)


def test_default_evidence_supervision_reaches_event_world_and_provenance_heads():
    torch.manual_seed(433)
    model = MRCRALanguageModel(tiny_config())
    batch = packed()
    output = model(
        batch.input_ids, segment_ids=batch.segment_ids,
        boundary_classes=batch.boundary_classes,
        source_uris=batch.external_source_uris, project_output=False,
    )
    terms = EvidenceBackedCognitiveSupervisor()(output, batch, 0, 4)
    assert {term.family for term in terms} == {
        ObjectiveFamily.EVENTS_RELATIONS,
        ObjectiveFamily.WORLD_HYPOTHESES_UNCERTAINTY,
        ObjectiveFamily.PROVENANCE_CONSISTENCY,
    }
    loss = combine_cognitive_objectives(
        terms, CognitiveObjectiveSchedule.curriculum(9)
    ).total
    loss.backward()
    assert model.cognitive.event_extractor.proposal_network.proposal.weight.grad is not None
    assert model.cognitive.next_latent_predictor.weight.grad is not None
    assert model.cognitive.provenance_source_head.weight.grad is not None
    assert model.cognitive.provenance_verification_head.weight.grad is not None
    assert all(torch.isfinite(parameter.grad).all() for parameter in (
        model.cognitive.event_extractor.proposal_network.proposal.weight,
        model.cognitive.next_latent_predictor.weight,
        model.cognitive.provenance_source_head.weight,
        model.cognitive.provenance_verification_head.weight,
    ))


def test_annotated_binding_and_functional_surprise_train_their_live_heads():
    torch.manual_seed(439)
    model = MRCRALanguageModel(tiny_config())
    batch = packed()
    output = model(
        batch.input_ids, segment_ids=batch.segment_ids,
        boundary_classes=batch.boundary_classes,
        source_uris=batch.external_source_uris, project_output=False,
    )
    base = language_evidence_targets(batch)
    steps = output.cognitive.action_receipts.actions.shape[-1]
    target = output.cognitive.action_receipts.actions.detach().clamp_min(0)
    controller_mask = output.cognitive.action_receipts.mask.detach()
    annotations = replace(
        base,
        binding_positive_indices=torch.tensor([[1, 0, 3, 2]]),
        binding_mask=torch.ones(1, 4, dtype=torch.bool),
        controller_action=target,
        controller_advantage=torch.ones(1, 4, steps),
        controller_mask=controller_mask,
    )
    terms = EvidenceBackedCognitiveSupervisor(lambda _: annotations)(output, batch, 0, 4)
    families = {term.family for term in terms}
    assert ObjectiveFamily.MULTIMODAL_BINDING in families
    assert ObjectiveFamily.CONTROLLER_CONSEQUENCE in families
    selected = [term for term in terms if term.family in {
        ObjectiveFamily.MULTIMODAL_BINDING, ObjectiveFamily.CONTROLLER_CONSEQUENCE,
    }]
    combine_cognitive_objectives(
        selected, CognitiveObjectiveSchedule.curriculum(9)
    ).total.backward()
    assert model.cognitive.controller.action_head.weight.grad is not None
    assert torch.isfinite(model.cognitive.controller.action_head.weight.grad).all()


def test_measured_metacognitive_targets_train_the_live_routing_head():
    torch.manual_seed(443)
    base = tiny_config()
    config = MRCRAConfig(
        base.carrier,
        replace(base.cognitive, enable_metacognitive_routing=True),
        actor_parameter_minimum=1, actor_parameter_maximum=10_000_000,
    )
    model = MRCRALanguageModel(config)
    batch = packed()
    output = model(
        batch.input_ids, segment_ids=batch.segment_ids,
        boundary_classes=batch.boundary_classes,
        source_uris=batch.external_source_uris, project_output=False,
    )
    assert output.cognitive.metacognitive_values.shape == (1, 4, 7)
    assert output.cognitive.metacognitive_mask.any()
    base_targets = language_evidence_targets(batch)
    annotations = replace(
        base_targets,
        metacognitive_realized_error=torch.zeros(1, 4),
        metacognitive_operation_values=torch.ones(1, 4, 5),
        metacognitive_calibration_error=torch.zeros(1, 4),
        metacognitive_mask=output.cognitive.metacognitive_mask.detach().clone(),
    )
    terms = EvidenceBackedCognitiveSupervisor(lambda _: annotations)(
        output, batch, 0, 4
    )
    selected = [
        term for term in terms
        if term.family == ObjectiveFamily.CONTROLLER_METACOGNITIVE_UTILITY
    ]
    assert {term.name for term in selected} == {
        "metacognitive_error_prediction", "metacognitive_operation_value",
        "metacognitive_calibration",
    }
    combine_cognitive_objectives(
        selected, CognitiveObjectiveSchedule.curriculum(9)
    ).total.backward()
    gradients = [parameter.grad for parameter in model.cognitive.metacognitive_router.parameters()]
    assert any(
        gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        for gradient in gradients
    )


def test_partial_annotation_group_fails_closed():
    batch = packed()
    model = MRCRALanguageModel(tiny_config())
    output = model(batch.input_ids, project_output=False)
    partial = CognitiveSupervisionTargets(
        relation_type=torch.zeros_like(batch.input_ids)
    )
    with pytest.raises(ValueError, match="partial cognitive target group"):
        EvidenceBackedCognitiveSupervisor(lambda _: partial)(output, batch, 0, 4)


def test_partial_metacognitive_authority_group_fails_closed():
    batch = packed()
    model = MRCRALanguageModel(tiny_config())
    output = model(batch.input_ids, project_output=False)
    partial = CognitiveSupervisionTargets(
        metacognitive_realized_error=torch.zeros(1, 4),
    )
    with pytest.raises(ValueError, match="partial cognitive target group"):
        EvidenceBackedCognitiveSupervisor(lambda _: partial)(output, batch, 0, 4)


def test_relation_and_external_consequence_annotations_are_strictly_masked():
    batch = packed()
    model = MRCRALanguageModel(tiny_config())
    output = model(
        batch.input_ids, segment_ids=batch.segment_ids,
        boundary_classes=batch.boundary_classes, project_output=False,
    )
    base = language_evidence_targets(batch)
    annotations = replace(
        base,
        relation_type=torch.zeros_like(batch.input_ids),
        relation_mask=torch.zeros_like(batch.input_ids, dtype=torch.bool),
        external_reward=torch.zeros_like(batch.input_ids, dtype=torch.float32),
        external_cost=torch.zeros_like(batch.input_ids, dtype=torch.float32),
        external_constraint=torch.zeros_like(batch.input_ids, dtype=torch.float32),
        external_success=torch.ones_like(batch.input_ids, dtype=torch.float32),
        external_mask=torch.zeros_like(batch.input_ids, dtype=torch.bool),
    )
    terms = EvidenceBackedCognitiveSupervisor(lambda _: annotations)(output, batch, 0, 4)
    names = {term.name for term in terms}
    assert "relation_family_nll" in names
    assert "external_consequence_model" in names
    assert not next(term for term in terms if term.name == "external_consequence_model").mask.any()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"event_mask": torch.ones(1, 3, dtype=torch.bool)}, "full packed context"),
        ({"event_type": torch.zeros(1, 4)}, "event types"),
        ({"relation_mask": torch.ones(1, 4)}, "boolean mask"),
        ({"binding_positive_indices": torch.zeros(1, 4)}, "int64 positives"),
        ({"provenance_mask": torch.ones(1, 4)}, "boolean mask"),
    ],
)
def test_supervision_rejects_misaligned_or_untyped_authority(change, message):
    batch = packed()
    model = MRCRALanguageModel(tiny_config())
    output = model(batch.input_ids, project_output=False)
    base = language_evidence_targets(batch)
    if "relation_mask" in change:
        change["relation_type"] = torch.zeros_like(batch.input_ids)
    if "binding_positive_indices" in change:
        change["binding_mask"] = torch.ones(1, 4, dtype=torch.bool)
    annotations = replace(base, **change)
    with pytest.raises(ValueError, match=message):
        EvidenceBackedCognitiveSupervisor(lambda _: annotations)(output, batch, 0, 4)
