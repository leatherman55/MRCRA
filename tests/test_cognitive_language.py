import pytest
import torch

from mrrn.cognitive_types import BoundaryClass, SourceClass
from mrrn.config import CognitiveConfig, MRCRAConfig, MRRNConfig
from mrrn.language import MRCRALanguageModel


def language_config(vocabulary=17):
    carrier = MRRNConfig(
        input_dim=8, model_dim=8, output_dim=vocabulary, layers=1, scales=3,
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
        calibration_regimes=2, active_event_capacity=6, pair_edge_capacity=8,
        hyperedge_capacity=3, graph_neighbors=2, global_workspace_slots=2,
        hypothesis_slots=2, maximum_hypothesis_slots=2, maximum_cognitive_steps=2,
        event_chunk_size=2, event_proposals_per_chunk=2, recent_candidates=2,
        landmark_candidates=1, episodic_candidates=1, semantic_candidates=1,
        episodic_memory_capacity=4, semantic_memory_capacity=3,
        associative_depth=2, associative_budget=2, world_model_horizons=(1, 2),
    )
    return MRCRAConfig(carrier, cognitive, actor_parameter_minimum=1, actor_parameter_maximum=10_000_000)


def test_cognitive_language_ties_embeddings_and_prepares_packed_boundaries():
    model = MRCRALanguageModel(language_config())
    assert model.token_embedding.weight is model.cognitive.carrier.output_head.weight
    tokens = torch.tensor([[1, 2, 3, 4]])
    segments = torch.tensor([[0, 0, 1, 1]])
    packet, ledger = model.prepare_external_input(tokens, segment_ids=segments)
    assert packet.boundary_classes[0, 0] == int(BoundaryClass.HARD)
    assert packet.boundary_classes[0, 2] == int(BoundaryClass.SEGMENT)
    assert len(ledger) == 2
    output = model(tokens, segment_ids=segments)
    assert output.logits.shape == (1, 4, 17)
    assert output.cognitive.state.carrier[0].position == 2


def test_generated_tokens_are_predicted_not_observed_and_have_provenance():
    torch.manual_seed(101)
    model = MRCRALanguageModel(language_config())
    generated = model.generate(
        torch.tensor([[1, 2, 3]]), maximum_new_tokens=3,
        temperature=0, top_k=None,
    )
    assert generated.tokens.shape == (1, 6)
    assert len(generated.generated_provenance_ids) == 3
    for record_id in generated.generated_provenance_ids:
        assert generated.ledger.get(record_id).source_class == SourceClass.PREDICTED
    assert generated.state.clocks.external == 6


def test_language_padding_is_zeroed_and_cannot_reactivate():
    model = MRCRALanguageModel(language_config())
    tokens = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[True, True, False, False]])
    output = model(tokens, mask)
    assert not output.logits[:, 2:].any()
    with pytest.raises(ValueError):
        model(tokens, torch.tensor([[True, False, True, False]]))


def test_cognitive_language_contracts_fail_closed():
    bad = language_config()
    bad = MRCRAConfig(
        MRRNConfig(
            input_dim=8, model_dim=8, output_dim=17, layers=1, scales=3,
            heads=2, modes=2, mimo_rank=1, attention_window=2,
            retrieved_items=1, memory_capacity=2, enable_global_head=True,
            relational_branch=True, relational_context_dim=8,
        ),
        bad.cognitive, actor_parameter_minimum=1, actor_parameter_maximum=10_000_000,
    )
    with pytest.raises(ValueError):
        MRCRALanguageModel(bad)
    model = MRCRALanguageModel(language_config())
    with pytest.raises(ValueError):
        model(torch.tensor([[99]]))
