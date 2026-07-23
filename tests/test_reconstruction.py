import copy

import pytest
import torch

from mrrn.reconstruction import (
    ConditionalGraphReconstructor, ReconstructionEvidence, ReconstructionQuery,
)


def query(batch=2, width=8):
    return ReconstructionQuery(
        torch.tensor([1, 2]), torch.tensor([3, 4]),
        torch.tensor([[0.0, 2.0, 2.0], [4.0, 6.0, 6.0]]),
        torch.tensor([3, 2]), torch.tensor([2, 1]),
        torch.tensor([0, 1]), torch.tensor([1, 2]),
        torch.tensor([0.1, 0.2]), torch.randn(batch, width),
        torch.tensor([True, True]),
    )


def evidence(batch=2, width=8):
    return ReconstructionEvidence(
        torch.randn(batch, width), torch.randn(batch, 4, width),
        torch.tensor([[True, True, False, False], [True, True, True, False]]),
        torch.tensor([[10, 11, -1, -1], [20, 21, 22, -1]]),
        torch.randn(batch, width), torch.tensor([[30, 31], [40, -1]]),
        torch.randn(batch, 3, width), torch.randn(batch, width),
        torch.randn(batch, width),
    )


def test_conditional_reconstructor_emits_requested_bounded_graph_and_gradients():
    torch.manual_seed(401)
    model = ConditionalGraphReconstructor(8, 13, 16, 4, 3)
    q, e = query(), evidence()
    e.abstraction_latent.requires_grad_()
    proposal = model(q, e)
    assert proposal.node_content.shape == (2, 4, 8)
    assert proposal.relation_content.shape == (2, 3, 8)
    assert proposal.node_mask.sum(-1).tolist() == [3, 2]
    assert proposal.relation_mask.sum(-1).tolist() == [2, 1]
    assert bool((proposal.participant_indices[proposal.relation_mask] >= 0).all())
    loss = (
        proposal.node_content.square().mean()
        + proposal.relation_content.square().mean()
        + proposal.historical_fidelity.mean()
    )
    loss.backward()
    assert e.abstraction_latent.grad is not None
    assert bool(torch.isfinite(e.abstraction_latent.grad).all())


def test_present_evidence_and_surviving_traces_causally_condition_reconstruction():
    torch.manual_seed(409)
    model = ConditionalGraphReconstructor(8, 13, 16, 4, 3).eval()
    q, e = query(), evidence()
    baseline = model(q, e)
    changed_evidence = copy.copy(e)
    object.__setattr__(changed_evidence, "observed_context", e.observed_context + 3)
    changed = model(q, changed_evidence)
    assert not torch.allclose(baseline.node_content, changed.node_content)
    changed_traces = copy.copy(e)
    object.__setattr__(changed_traces, "trace_content", e.trace_content + 2)
    traced = model(q, changed_traces)
    assert not torch.allclose(baseline.node_content, traced.node_content)


def test_neural_proposal_requires_authority_provenance_before_finalization():
    torch.manual_seed(419)
    proposal = ConditionalGraphReconstructor(8, 13, 16, 4, 3)(query(), evidence())
    node_ids = torch.full(proposal.node_mask.shape, -1, dtype=torch.int64)
    relation_ids = torch.full(proposal.relation_mask.shape, -1, dtype=torch.int64)
    with pytest.raises(ValueError, match="provenance"):
        proposal.finalize(node_ids, relation_ids)
    node_ids[proposal.node_mask] = torch.arange(int(proposal.node_mask.sum()))
    relation_ids[proposal.relation_mask] = 100 + torch.arange(int(proposal.relation_mask.sum()))
    result = proposal.finalize(node_ids, relation_ids)
    assert torch.equal(result.provenance_ids, node_ids)
    assert torch.equal(result.relation_provenance_ids, relation_ids)


def test_reconstruction_query_rejects_active_negative_abstraction_and_sizes():
    q = query()
    bad = copy.copy(q)
    indices = q.abstraction_indices.clone(); indices[0] = -1
    object.__setattr__(bad, "abstraction_indices", indices)
    with pytest.raises(ValueError, match="abstraction"):
        bad.__post_init__()
