import pytest
import torch

from mrrn.cognitive_types import NodeSlots, NodeType, RelationFamily, SourceClass
from mrrn.relational_router import NodeCandidateBuilder, RelationalResonanceRouter


def active_nodes(*, width=8, count=5, heads=2, modes=3):
    base = NodeSlots.empty(
        1, count, width, heads=heads, modes=modes, node_types=len(NodeType),
        modalities=16, uncertainty_channels=3, provenance_features=4, hypotheses=4,
    )
    values = {name: getattr(base, name).clone() for name in base.__dataclass_fields__}
    values["content"].normal_()
    values["spectral"][..., 0] = 1
    values["type_logits"][..., int(NodeType.EVENT)] = 4
    values["support"][0, :, 0] = torch.arange(count)
    values["support"][0, :, 1] = torch.arange(count)
    values["support"][0, :, 2] = torch.arange(count)
    values["modality_presence"][..., 0] = 1
    values["provenance_ids"][0] = torch.arange(count)
    values["source_classes"].fill_(int(SourceClass.INFERRED))
    values["scenario_ids"].zero_()
    values["active"].fill_(True)
    return NodeSlots(**values)


def test_candidate_builder_is_bounded_causal_and_scenario_isolated():
    torch.manual_seed(13)
    nodes = active_nodes(count=6)
    builder = NodeCandidateBuilder(8, 3, router_dim=4)
    candidates = builder(nodes)
    assert candidates.content.shape == (1, 6, 3, 8)
    assert not candidates.mask[0, 0].any()
    for query in range(6):
        selected = candidates.node_indices[0, query, candidates.mask[0, query]]
        assert bool((selected < query).all())
        assert selected.numel() <= 3

    values = {name: getattr(nodes, name).clone() for name in nodes.__dataclass_fields__}
    values["scenario_ids"][0, 2] = 9
    isolated = builder(NodeSlots(**values))
    assert 2 not in isolated.node_indices[0, 5, isolated.mask[0, 5]].tolist()


def test_relational_router_normalizes_joint_distribution_and_keeps_degree_bound():
    torch.manual_seed(17)
    nodes = active_nodes(count=6)
    candidates = NodeCandidateBuilder(8, 4, router_dim=4)(nodes)
    router = RelationalResonanceRouter(
        8, 2, 3, len(RelationFamily), 3, 4, adapter_rank=3, retained_edges=2,
    )
    output = router(nodes, candidates)
    assert output.update.shape == nodes.content.shape
    assert output.relation_messages.shape == (1, 6, len(RelationFamily), 8)
    assert output.selected_node_indices.shape == (1, 6, 2)
    assert bool((output.selected_mask.sum(-1) <= 2).all())
    active_queries = candidates.mask.any(-1)
    torch.testing.assert_close(
        output.joint_posterior.sum((-2, -1))[active_queries],
        torch.ones_like(output.joint_posterior.sum((-2, -1))[active_queries]),
    )
    output.update.square().mean().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in router.parameters())


def test_relation_compatibility_hard_masks_symbol_to_symbol_causation():
    nodes = active_nodes(count=2)
    values = {name: getattr(nodes, name).clone() for name in nodes.__dataclass_fields__}
    values["type_logits"].zero_()
    values["type_logits"][..., int(NodeType.SYMBOL)] = 10
    nodes = NodeSlots(**values)
    candidates = NodeCandidateBuilder(8, 1)(nodes, include_self=True)
    router = RelationalResonanceRouter(8, 2, 3, len(RelationFamily), 3, 4)
    output = router(nodes, candidates)
    assert not bool((output.relation_posterior[..., int(RelationFamily.CAUSAL_INFLUENCE)] > 0).any())


def test_phase_coherence_aligns_delay_and_distinguishes_antiphase():
    router = RelationalResonanceRouter(8, 2, 3, len(RelationFamily), 3, 4)
    query = torch.zeros(1, 1, 2, 3, 2)
    query[..., 0] = 1
    candidate = query[:, :, None].clone()
    aligned = router._coherence(query, candidate, torch.zeros(1, 1, 1))
    antiphase = router._coherence(query, candidate, torch.full((1, 1, 1), 2.0))
    torch.testing.assert_close(aligned, torch.ones_like(aligned))
    torch.testing.assert_close(antiphase, -torch.ones_like(antiphase), atol=1e-5, rtol=1e-5)


def test_router_rejects_dimension_and_mask_contract_violations():
    nodes = active_nodes()
    builder = NodeCandidateBuilder(8, 2)
    with pytest.raises(ValueError):
        builder(nodes, query_mask=torch.ones(1, 5))
    candidates = builder(nodes)
    with pytest.raises(ValueError):
        RelationalResonanceRouter(7, 2, 3, len(RelationFamily), 3, 4)(nodes, candidates)
    with pytest.raises(ValueError):
        RelationalResonanceRouter(8, 2, 3, 15, 3, 4)
