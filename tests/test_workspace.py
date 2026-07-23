from copy import deepcopy

import pytest
import torch

from mrrn.cognitive_types import NodeSlots, NodeType, RelationFamily, RelationSlots, SourceClass
from mrrn.relational_router import NodeCandidateBuilder, RelationalResonanceRouter
from mrrn.workspace import (
    GlobalWorkspace, RelationProposals, RelationSlotWriter, WorkspaceBroadcast, WorkspaceGraph,
    invalidate_stale_relations,
)


def nodes(count=6, width=8):
    state = NodeSlots.empty(
        1, count, width, heads=2, modes=3, node_types=len(NodeType), modalities=16,
        uncertainty_channels=3, provenance_features=4, hypotheses=4,
    )
    values = {name: getattr(state, name).clone() for name in state.__dataclass_fields__}
    torch.manual_seed(31)
    values["content"].normal_()
    values["spectral"][..., 0] = 1
    values["type_logits"][..., int(NodeType.EVENT)] = 5
    values["support"][0, :, :] = torch.arange(count)[:, None]
    values["modality_presence"][..., 0] = 1
    values["provenance_ids"][0] = torch.arange(100, 100 + count)
    values["source_classes"].fill_(int(SourceClass.INFERRED))
    values["scenario_ids"].zero_()
    values["activity"].fill_(1)
    values["importance"][0] = torch.linspace(0.1, 1, count)
    values["active"].fill_(True)
    return NodeSlots(**values)


def graph():
    builder = NodeCandidateBuilder(8, 4, router_dim=4)
    router = RelationalResonanceRouter(
        8, 2, 3, len(RelationFamily), 3, 4, adapter_rank=3, retained_edges=2,
    )
    workspace = GlobalWorkspace(8, 3)
    broadcast = WorkspaceBroadcast(8, (8, 12), rank=4)
    return WorkspaceGraph(builder, router, workspace, broadcast)


def test_workspace_graph_is_bounded_competitive_and_low_gain():
    model = graph()
    output = model(nodes())
    assert output.relation_proposals.active.sum() <= 6 * 2
    assert output.workspace.state.slots.shape == (1, 3, 8)
    # Each node owns at most unit mass across global slots after competition.
    assert bool((output.workspace.assignment.sum(1) <= 1.0001).all())
    assert float(output.broadcast.residual_gain.detach()) < 0.001
    assert len(output.broadcast.scale_and_bias) == 2
    assert output.broadcast.scale_and_bias[1][0].shape == (1, 12)
    assert torch.linalg.vector_norm(output.nodes.content - nodes().content) > 0


def test_relation_proposals_preserve_ordered_participant_roles_and_materialize_exactly():
    state = nodes()
    output = graph()(state)
    proposals = output.relation_proposals
    derived = torch.full_like(proposals.provenance_ids, -1)
    derived[proposals.active] = torch.arange(
        int(proposals.active.sum()), dtype=torch.int64
    ) + 1000
    proposals = proposals.with_provenance(derived)
    slots = RelationSlots.empty(
        1, 5, 8, relation_families=len(RelationFamily), arity=4,
        uncertainty_channels=3, hypotheses=4,
    )
    written = RelationSlotWriter(len(RelationFamily), 3)(slots, proposals, state)
    assert written.active.sum() <= 5
    assert bool((written.participant_mask[written.active].sum(-1) == 2).all())
    first = torch.nonzero(written.active[0], as_tuple=False).flatten()[0]
    assert written.participant_roles[0, first, :2].tolist() == [0, 1]
    assert written.participant_indices[0, first, 0] != written.participant_indices[0, first, 1]
    assert written.provenance_ids[0, first] >= 1000


def test_relation_ring_never_exceeds_capacity_under_adversarial_rewrites():
    state = nodes(count=10)
    model = WorkspaceGraph(
        NodeCandidateBuilder(8, 8),
        RelationalResonanceRouter(
            8, 2, 3, len(RelationFamily), 3, 4, retained_edges=6
        ),
        GlobalWorkspace(8, 4), WorkspaceBroadcast(8, (8,)),
    )
    proposals = model(state).relation_proposals
    provenance = torch.full_like(proposals.provenance_ids, -1)
    provenance[proposals.active] = torch.arange(int(proposals.active.sum())) + 500
    proposals = proposals.with_provenance(provenance)
    slots = RelationSlots.empty(
        1, 3, 8, relation_families=len(RelationFamily), arity=4,
        uncertainty_channels=3, hypotheses=4,
    )
    writer = RelationSlotWriter(len(RelationFamily), 3)
    for _ in range(4):
        slots = writer(slots, proposals, state)
        assert slots.active.sum() <= 3


def test_relation_is_invalidated_when_a_reused_node_slot_changes_version():
    state = nodes(count=4)
    output = graph()(state)
    provenance = torch.full_like(output.relation_proposals.provenance_ids, -1)
    provenance[output.relation_proposals.active] = torch.arange(
        int(output.relation_proposals.active.sum())
    ) + 700
    slots = RelationSlots.empty(
        1, 8, 8, relation_families=len(RelationFamily), arity=4,
        uncertainty_channels=3, hypotheses=4,
    )
    slots = RelationSlotWriter(len(RelationFamily), 3)(
        slots, output.relation_proposals.with_provenance(provenance), state
    )
    relation_slot = int(torch.nonzero(slots.active[0], as_tuple=False)[0])
    participant_slot = int(slots.participant_indices[0, relation_slot, 0])
    changed = {name: getattr(state, name).clone() for name in state.__dataclass_fields__}
    changed["versions"][0, participant_slot] += 1
    invalidated = invalidate_stale_relations(slots, NodeSlots(**changed))
    assert not invalidated.active[0, relation_slot]
    assert invalidated.provenance_ids[0, relation_slot] == -1
    assert not invalidated.participant_mask[0, relation_slot].any()


def test_pair_and_hyperedge_quotas_are_physically_separate():
    state = nodes(count=5)
    slots = RelationSlots.empty(
        1, 3, 8, relation_families=len(RelationFamily), arity=4,
        uncertainty_channels=3, hypotheses=4,
    )
    writer = RelationSlotWriter(
        len(RelationFamily), 3, pair_capacity=2, hyperedge_capacity=1,
    )

    def proposal(participants, provenance):
        arity = len(participants)
        shape = (1, 1, 1)
        pointers = torch.full((*shape, arity), -1, dtype=torch.int64)
        pointers[0, 0, 0] = torch.tensor(participants)
        mask = pointers >= 0
        parents = pointers.clone()
        parents[mask] += 100
        return RelationProposals(
            torch.randn(*shape, 8),
            torch.full(shape, int(RelationFamily.SIMILARITY), dtype=torch.int64),
            pointers, torch.arange(arity).view(1, 1, 1, arity), mask,
            torch.tensor([[[[0.0, 1.0, 1.0]]]]), torch.ones(shape), parents,
            torch.full(shape, provenance, dtype=torch.int64),
            torch.zeros(shape, dtype=torch.int64), torch.ones(shape, dtype=torch.bool),
        )

    slots = writer(slots, proposal([0, 1], 900), state)
    pair_version = int(slots.versions[0, 0])
    slots = writer(slots, proposal([0, 1, 2, -1], 901), state)
    assert slots.active[0, :2].sum() == 1
    assert slots.active[0, 2]
    assert slots.participant_mask[0, 2].sum() == 3
    slots = writer(slots, proposal([1, 2, 3, -1], 902), state)
    assert int(slots.versions[0, 0]) == pair_version
    assert slots.provenance_ids[0, 0] == 900
    assert slots.provenance_ids[0, 2] == 902
    assert slots.participant_indices[0, 2, :3].tolist() == [1, 2, 3]


def test_empty_workspace_is_stable_and_broadcasts_zero_context():
    empty = NodeSlots.empty(
        2, 4, 8, heads=2, modes=3, node_types=len(NodeType), modalities=16,
        uncertainty_channels=3, provenance_features=4, hypotheses=4,
    )
    workspace = GlobalWorkspace(8, 3)
    output = workspace(empty)
    assert not output.state.active.any()
    assert not output.assignment.any()
    broadcast = WorkspaceBroadcast(8, (8,))(output.state)
    torch.testing.assert_close(broadcast.output_context, torch.zeros_like(broadcast.output_context))


@pytest.mark.parametrize("active_indices", [(), (0, 3, 7)])
def test_active_prefix_workspace_is_equivalent_to_dense_authority_ring(active_indices):
    state = nodes(count=10)
    values = {
        name: getattr(state, name).clone()
        for name in state.__dataclass_fields__
    }
    values["active"].zero_()
    values["provenance_ids"].fill_(-1)
    if active_indices:
        selected = torch.tensor(active_indices, dtype=torch.int64)
        values["active"][0, selected] = True
        values["provenance_ids"][0, selected] = 100 + selected
    state = NodeSlots(**values)

    compact_model = graph()
    dense_model = deepcopy(compact_model)
    dense_model.compact_active = False
    compact = compact_model(state)
    dense = dense_model(state)

    torch.testing.assert_close(compact.nodes.content, dense.nodes.content)
    torch.testing.assert_close(
        compact.workspace.state.slots, dense.workspace.state.slots
    )
    torch.testing.assert_close(
        compact.workspace.assignment, dense.workspace.assignment
    )
    torch.testing.assert_close(
        compact.broadcast.output_context, dense.broadcast.output_context
    )
    torch.testing.assert_close(compact.router.update, dense.router.update)
    compact_active = compact.relation_proposals.active
    dense_active = dense.relation_proposals.active
    assert compact_active.sum() == dense_active.sum()
    assert torch.equal(
        compact.relation_proposals.participant_indices[compact_active],
        dense.relation_proposals.participant_indices[dense_active],
    )
    assert torch.equal(
        compact.relation_proposals.family_ids[compact_active],
        dense.relation_proposals.family_ids[dense_active],
    )


def test_active_prefix_workspace_preserves_learned_parameter_gradients():
    state = nodes(count=10)
    values = {
        name: getattr(state, name).clone()
        for name in state.__dataclass_fields__
    }
    values["active"].zero_()
    values["provenance_ids"].fill_(-1)
    selected = torch.tensor((0, 3, 7), dtype=torch.int64)
    values["active"][0, selected] = True
    values["provenance_ids"][0, selected] = 100 + selected
    state = NodeSlots(**{
        name: value.to(torch.float64) if value.is_floating_point() else value
        for name, value in values.items()
    })

    compact_model = graph().double()
    dense_model = deepcopy(compact_model)
    dense_model.compact_active = False
    compact = compact_model(state)
    dense = dense_model(state)
    compact_loss = (
        compact.nodes.content.square().sum()
        + compact.workspace.state.slots.square().sum()
        + compact.broadcast.output_context.square().sum()
    )
    dense_loss = (
        dense.nodes.content.square().sum()
        + dense.workspace.state.slots.square().sum()
        + dense.broadcast.output_context.square().sum()
    )
    compact_loss.backward()
    dense_loss.backward()
    for (compact_name, compact_parameter), (dense_name, dense_parameter) in zip(
        compact_model.named_parameters(), dense_model.named_parameters(), strict=True,
    ):
        assert compact_name == dense_name
        if compact_parameter.grad is None or dense_parameter.grad is None:
            assert compact_parameter.grad is dense_parameter.grad is None
        else:
            torch.testing.assert_close(
                compact_parameter.grad, dense_parameter.grad,
                atol=1e-10, rtol=1e-10,
            )


def test_workspace_contracts_fail_closed():
    with pytest.raises(ValueError):
        GlobalWorkspace(0, 3)
    with pytest.raises(ValueError):
        WorkspaceBroadcast(8, (0,))
    output = graph()(nodes())
    slots = RelationSlots.empty(
        1, 4, 8, relation_families=len(RelationFamily), arity=4,
        uncertainty_channels=3, hypotheses=4,
    )
    with pytest.raises(ValueError):
        RelationSlotWriter(len(RelationFamily), 3)(slots, output.relation_proposals, nodes())


def test_node_state_precision_conversion_preserves_authority_dtypes():
    state = nodes()
    converted = state.to(dtype=torch.float64)
    assert converted.content.dtype == torch.float64
    assert converted.provenance_ids.dtype == torch.int64
    assert converted.active.dtype == torch.bool
