from dataclasses import replace

import pytest
import torch

from mrrn.compression import AbstractionDAG, GraphCompressor, GraphFragment
from mrrn.invariants import (
    BoundedGraphMatcher, IntegratedInvariantDiscoverer, InvariantEvidence,
    InvariantLedger, StructuralNormalizer, SymbolActivator,
)


def fragment(*, permutation=None):
    content = torch.tensor([[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]])
    types = torch.tensor([[3, 2, 4]], dtype=torch.int64)
    support = torch.tensor([[[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]]])
    provenance = torch.tensor([[10, 11, 12]], dtype=torch.int64)
    participants = torch.tensor([[[0, 1], [1, 2]]], dtype=torch.int64)
    relation_content = torch.tensor([[[1., 0., 1., 0.], [0., 1., 0., 1.]]])
    families = torch.tensor([[1, 9]], dtype=torch.int64)
    if permutation is not None:
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(len(permutation))
        content = content[:, permutation]
        types = types[:, permutation]
        support = support[:, permutation]
        provenance = provenance[:, permutation]
        participants = inverse[participants]
    return GraphFragment(
        content, types, support, provenance, torch.ones(1, 3, dtype=torch.bool),
        relation_content, families, participants, torch.tensor([[20, 21]]),
        torch.ones(1, 2, dtype=torch.bool),
    )


def test_graph_compressor_learns_reconstruction_and_accepts_only_measured_candidate():
    torch.manual_seed(51)
    model = GraphCompressor(4, 4, 13, 16, 3, 2, precision_bits=16)
    graph = fragment()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    initial = model(graph).distortion.total.detach()
    for _ in range(120):
        proposal = model(graph)
        loss = proposal.distortion.total.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    proposal = model(graph)
    assert proposal.distortion.total.item() < initial.item() * 0.2
    assert torch.isfinite(proposal.code.compressed_bits).all()
    accepted = model.decide(
        proposal,
        minimum_gain=max(0, proposal.code.gain_bits.item() - 1),
        maximum_distortion=proposal.distortion.total.item() + 1e-4,
        held_out_loss_before=torch.tensor([1.0]),
        held_out_loss_after=torch.tensor([0.9]),
    )
    assert accepted.accepted.item()
    rejected = model.decide(
        proposal,
        minimum_gain=max(0, proposal.code.gain_bits.item() - 1),
        maximum_distortion=proposal.distortion.total.item() + 1e-4,
        held_out_loss_before=torch.tensor([1.0]),
        held_out_loss_after=torch.tensor([2.0]),
    )
    assert not rejected.accepted.item() and not rejected.held_out_pass.item()


def test_abstraction_depth_is_graph_derived_not_physical_scale():
    dag = AbstractionDAG()
    first = dag.append(
        child_node_ids=[1, 2], child_relation_ids=[3], child_abstraction_ids=[],
        physical_scales=[5], latent=torch.tensor([1., 2.]), code_gain_bits=3,
        distortion=0.1, provenance_id=9,
    )
    second = dag.append(
        child_node_ids=[4], child_relation_ids=[], child_abstraction_ids=[first],
        physical_scales=[0], latent=torch.tensor([3., 4.]), code_gain_bits=2,
        distortion=0.2, provenance_id=10,
    )
    assert dag.get(first).abstraction_depth == 1
    assert dag.get(second).abstraction_depth == 2
    assert dag.get(first).physical_scales == (5,)
    with pytest.raises(ValueError):
        dag.append(
            child_node_ids=[], child_relation_ids=[], child_abstraction_ids=[99],
            physical_scales=[0], latent=torch.tensor([1.]), code_gain_bits=1,
            distortion=0, provenance_id=1,
        )


def test_structural_matching_is_permutation_aware_and_relation_family_preserving():
    torch.manual_seed(61)
    normalizer = StructuralNormalizer(4, 13, 6, 16)
    left = normalizer(fragment())
    right = normalizer(fragment(permutation=torch.tensor([2, 0, 1])))
    match = BoundedGraphMatcher(4, sinkhorn_iterations=20, temperature=0.03)(left, right)
    assert match.assignment.shape == (1, 3, 3)
    assert match.total_cost.item() < 0.15
    changed_adjacency = right.adjacency.roll(1, dims=1)
    altered = replace(right, adjacency=changed_adjacency)
    altered_match = BoundedGraphMatcher(4, sinkhorn_iterations=20, temperature=0.03)(left, altered)
    assert altered_match.relation_cost > match.relation_cost


def test_integrated_discovery_is_identity_permutation_invariant_and_rejects_relation_near_match():
    torch.manual_seed(67)
    discoverer = IntegratedInvariantDiscoverer(
        4, 13, 6, 16, 4, maximum_match_cost=1.0,
        minimum_counterexample_margin=0.0,
    )
    original = discoverer(fragment(), fragment(permutation=torch.tensor([2, 0, 1])))
    assert original.match_cost.item() < original.counterexample_cost.item()
    assert original.code_gain_bits.item() > 0
    assert original.supporting_mask.sum() == 6
    changed = fragment(permutation=torch.tensor([2, 0, 1]))
    changed = replace(
        changed,
        relation_family_ids=(changed.relation_family_ids + 1) % 16,
    )
    near_match = discoverer(fragment(), changed)
    assert near_match.relation_distortion > original.relation_distortion
    (original.latent.square().mean() + original.match_cost.mean()).backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in discoverer.parameters()
    )


def invariant_evidence():
    return InvariantEvidence(
        pattern=(1.0, 0.0), applicability_conditions=("resource is conserved",),
        known_failures=("external injection",), procedure=(0.2,), residual_decoder=(0.1,),
        episode_ids=(1, 2), transformation_ids=("rotate", "translate"),
        supporting_provenance_ids=(10, 11), contradicting_provenance_ids=(),
        independent_source_roots=(1, 2), predictive_utility=0.2, action_utility=0.0,
        code_gain_bits=4.0, reconstruction_distortion=0.03, relation_distortion=0.02,
        calibrated_confidence=0.8, counterexample_search_completed=True,
    )


def test_conditional_invariant_promotion_and_counterexample_revision_are_append_only():
    ledger = InvariantLedger()
    first = ledger.promote(
        invariant_evidence(), provenance_id=30,
        maximum_reconstruction_distortion=0.1, maximum_relation_distortion=0.1,
    )
    revised = ledger.add_counterexample(
        first, failure_condition="open boundary", provenance_id=31,
        calibrated_confidence=0.6,
    )
    assert ledger.get(first).evidence.calibrated_confidence == 0.8
    assert ledger.get(revised).revision_of == first
    assert "open boundary" in ledger.get(revised).evidence.known_failures
    invalid = replace(invariant_evidence(), counterexample_search_completed=False)
    with pytest.raises(ValueError):
        ledger.promote(
            invalid, provenance_id=32,
            maximum_reconstruction_distortion=0.1, maximum_relation_distortion=0.1,
        )
    dependent_sources = replace(invariant_evidence(), independent_source_roots=(1,))
    with pytest.raises(ValueError):
        ledger.promote(
            dependent_sources, provenance_id=33,
            maximum_reconstruction_distortion=0.1, maximum_relation_distortion=0.1,
        )


def test_symbols_are_context_activated_invariants_not_token_embeddings():
    activator = SymbolActivator(8, 4, 3)
    ids = torch.tensor([[1, 2]])
    first, gate = activator(ids, torch.tensor([[1., 0., 0.]]), torch.tensor([[True, True]]))
    second, _ = activator(ids, torch.tensor([[0., 1., 0.]]), torch.tensor([[True, True]]))
    assert first.shape == (1, 2, 4)
    assert bool(((gate > 0) & (gate < 1)).all())
    assert not torch.allclose(first, second)
