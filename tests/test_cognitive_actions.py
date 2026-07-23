from dataclasses import replace

import torch

from mrrn.cognitive_model import (
    ActionStatus, MultimodalRelationalContinuityResonanceNetwork,
)
from mrrn.cognitive_types import (
    InternalAction, ModalityClass, NodeSlots, NodeType, RelationFamily,
    RelationSlots, SourceClass, SupportInterval, VerificationClass,
)
from mrrn.config import CognitiveConfig, MRCRAConfig, MRRNConfig
from mrrn.memory_v2 import MemoryTier
from mrrn.provenance import ProvenanceLedger


def action_config() -> MRCRAConfig:
    carrier = MRRNConfig(
        input_dim=8, model_dim=8, output_dim=17, layers=1, scales=3,
        heads=2, modes=2, mimo_rank=1, attention_window=2,
        attention_query_tile_size=2, retrieved_items=2, memory_capacity=8,
        mixer_expansion=1.5, width_growth_cap=1, mode_growth_cap=1,
        width_multiple=2, spectral_modes=2, spectral_basis_order=2,
        spectral_triads_per_mode=1, enable_global_head=False,
        relational_branch=True, relational_context_dim=8,
    )
    cognitive = CognitiveConfig(
        workspace_dim=8, provenance_features=4, uncertainty_channels=8,
        relation_heads=2, relation_modes=2, relation_adapter_rank=2,
        goal_slots=1, goal_constraint_dim=2, system_action_channels=2,
        calibration_regimes=2, active_event_capacity=4, pair_edge_capacity=8,
        hyperedge_capacity=2, maximum_hyperedge_arity=3, graph_neighbors=1,
        global_workspace_slots=2, hypothesis_slots=2,
        maximum_hypothesis_slots=2, maximum_cognitive_steps=1,
        event_chunk_size=2, event_proposals_per_chunk=1,
        recent_candidates=2, landmark_candidates=1, episodic_candidates=2,
        semantic_candidates=2, episodic_memory_capacity=4,
        semantic_memory_capacity=4, associative_depth=2,
        associative_budget=2, world_model_horizons=(1,),
    )
    return MRCRAConfig(
        carrier, cognitive, actor_parameter_minimum=1,
        actor_parameter_maximum=10_000_000,
    )


def _rich_state(model: MultimodalRelationalContinuityResonanceNetwork):
    ledger = ProvenanceLedger()
    provenance = [
        ledger.append(
            source_class=SourceClass.EXTERNAL,
            source_uri_or_episode=f"test://action-node/{index}",
            support=SupportInterval(float(index), float(index), float(index)),
            modality=ModalityClass.TEXT, operator="test:observation",
            scenario_id=0, model_authority="test",
            verification=VerificationClass.EXTERNALLY_CHECKED,
        )
        for index in range(2)
    ]
    state = model.initial_state(1, sample_intervals=torch.ones(1))
    node_values = {
        name: getattr(state.nodes, name).clone()
        for name in state.nodes.__dataclass_fields__
    }
    node_values["content"][0, 0, 0] = 1
    node_values["content"][0, 1, 1] = 1
    node_values["spectral"][0, :2, :, :, 0] = 1
    node_values["type_logits"][0, 0, int(NodeType.ABSTRACTION)] = 1
    node_values["type_logits"][0, 1, int(NodeType.EVENT)] = 1
    node_values["support"][0, 0] = torch.tensor([0.0, 0.0, 0.0])
    node_values["support"][0, 1] = torch.tensor([1.0, 1.0, 1.0])
    node_values["modality_presence"][0, :2, int(ModalityClass.TEXT)] = 1
    node_values["provenance_ids"][0, :2] = torch.tensor(provenance)
    node_values["source_classes"][0, :2] = int(SourceClass.EXTERNAL)
    node_values["scenario_ids"][0, :2] = 0
    node_values["activity"][0, :2] = 1
    node_values["importance"][0, :2] = torch.tensor([1.0, 0.9])
    node_values["versions"][0, :2] = 1
    node_values["active"][0, :2] = True
    nodes = NodeSlots(**node_values)

    relation_provenance = ledger.derive(
        provenance, source_class=SourceClass.INFERRED,
        operator="test:relation", support=SupportInterval(0, 1, 1),
        modality=ModalityClass.TEXT, scenario_id=0, model_authority="test",
    )
    relation_values = {
        name: getattr(state.relations, name).clone()
        for name in state.relations.__dataclass_fields__
    }
    relation_values["content"][0, 0] = nodes.content[0, 0] + nodes.content[0, 1]
    relation_values["type_logits"][0, 0, int(RelationFamily.SIMILARITY)] = 1
    relation_values["participant_indices"][0, 0, :2] = torch.tensor([0, 1])
    relation_values["participant_roles"][0, 0, :2] = torch.tensor([0, 1])
    relation_values["participant_versions"][0, 0, :2] = 1
    relation_values["participant_weights"][0, 0, :2] = 1
    relation_values["participant_mask"][0, 0, :2] = True
    relation_values["support"][0, 0] = torch.tensor([0.0, 1.0, 1.0])
    relation_values["confidence"][0, 0] = 1
    relation_values["provenance_ids"][0, 0] = relation_provenance
    relation_values["scenario_ids"][0, 0] = 0
    relation_values["versions"][0, 0] = 1
    relation_values["active"][0, 0] = True
    relations = RelationSlots(**relation_values)

    workspace = model.workspace_graph.workspace(nodes).state
    summary = model._workspace_summary(workspace)
    hypotheses = model.hypothesis_bank.initial_state(1)
    hypotheses = model.hypothesis_bank.create(
        hypotheses, summary, torch.tensor([True])
    )
    hypotheses = model.hypothesis_bank.create(
        hypotheses, summary, torch.tensor([True])
    )
    pointers = torch.tensor([0])
    selected = torch.tensor([True])
    episodic, _ = model._write_memory_action(
        state.episodic_memory, nodes, pointers, selected,
        tier=MemoryTier.EPISODIC, ledger=ledger,
    )
    episodic, _ = model._write_memory_action(
        episodic, nodes, torch.tensor([1]), selected,
        tier=MemoryTier.EPISODIC, ledger=ledger,
    )
    semantic, _ = model._write_memory_action(
        state.semantic_memory, nodes, pointers, selected,
        tier=MemoryTier.SEMANTIC, ledger=ledger,
    )
    semantic, _ = model._write_memory_action(
        semantic, nodes, torch.tensor([1]), selected,
        tier=MemoryTier.SEMANTIC, ledger=ledger,
    )
    active_memory = torch.nonzero(episodic.active[0], as_tuple=False).flatten()
    episodic = model.memory.link_associations(
        episodic, active_memory[:1][None], active_memory[1:2][None],
        torch.tensor([[int(RelationFamily.SIMILARITY)]]),
        torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool),
    )
    return ledger, nodes, relations, workspace, hypotheses, episodic, semantic


def _force_action(model, action: InternalAction) -> None:
    with torch.no_grad():
        model.controller.action_head.weight.zero_()
        model.controller.action_head.bias.fill_(-100)
        model.controller.action_head.bias[int(action)] = 100
        model.controller.halt_head.weight.zero_()
        model.controller.halt_head.bias.fill_(-100)
        model.controller.relation_head.weight.zero_()
        model.controller.relation_head.bias.fill_(-100)
        model.controller.relation_head.bias[int(RelationFamily.SIMILARITY)] = 100
        model.controller.memory_tier_head.weight.zero_()
        model.controller.memory_tier_head.bias.fill_(-100)
        model.controller.memory_tier_head.bias[int(MemoryTier.EPISODIC)] = 100


def test_all_internal_actions_execute_under_satisfied_preconditions_and_emit_receipts():
    torch.manual_seed(263)
    model = MultimodalRelationalContinuityResonanceNetwork(action_config()).eval()
    expected_status = {
        InternalAction.HALT: ActionStatus.HALTED,
        InternalAction.COMPRESS: ActionStatus.VALIDATION_REQUIRED,
        InternalAction.VERIFY: ActionStatus.EXTERNAL_EVIDENCE_REQUIRED,
        InternalAction.PROPOSE_INVARIANT: ActionStatus.VALIDATION_REQUIRED,
        InternalAction.ABSTAIN_OR_REQUEST_EXTERNAL_EVIDENCE:
            ActionStatus.EXTERNAL_EVIDENCE_REQUIRED,
    }
    observed = set()
    for action in InternalAction:
        ledger, nodes, relations, workspace, hypotheses, episodic, semantic = _rich_state(model)
        availability = model._action_availability(
            nodes, relations, hypotheses, episodic, semantic, workspace,
            torch.tensor([1]),
        )
        if action.value >= InternalAction.RECONSTRUCT_LOCAL.value:
            # Phase-1 ontology values are intentionally unavailable until their
            # production loops and integrated gates are enabled.  Declaring an
            # action name is not evidence that its capability exists.
            assert not bool(availability[0, int(action)]), action.name
            continue
        assert bool(availability[0, int(action)]), action.name
        _force_action(model, action)
        result = model._run_internal_actions(
            nodes, relations, workspace, hypotheses, episodic, semantic,
            torch.zeros(1, 8), torch.tensor([1]),
            model.default_goals(1, device=nodes.content.device, dtype=nodes.content.dtype),
            model.default_system_model(1, device=nodes.content.device, dtype=nodes.content.dtype),
            ledger, timestamps=torch.tensor([2.0]), active_rows=torch.tensor([True]),
            scale_contexts=torch.randn(1, 3, 8),
            scale_context_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        receipts = result[-1]
        assert receipts.mask[0, 0], action.name
        assert receipts.actions[0, 0] == int(action), action.name
        assert receipts.statuses[0, 0] == int(
            expected_status.get(action, ActionStatus.SUCCESS)
        ), action.name
        if action == InternalAction.VERIFY:
            assert not receipts.success[0, 0]
        else:
            assert receipts.success[0, 0], action.name
        observed.add(InternalAction(int(receipts.actions[0, 0])))
    assert observed == {
        action for action in InternalAction
        if action.value < InternalAction.RECONSTRUCT_LOCAL.value
    }


def test_conditional_reconstruction_is_an_integrated_bounded_provenance_transaction():
    """The production path must materialize a local graph without rewriting evidence."""

    torch.manual_seed(271)
    base = action_config()
    cognitive = replace(
        base.cognitive, event_proposals_per_chunk=2,
        enable_conditional_reconstruction=True,
        enable_abstraction_validity_control=True,
    )
    model = MultimodalRelationalContinuityResonanceNetwork(MRCRAConfig(
        base.carrier, cognitive, actor_parameter_minimum=1,
        actor_parameter_maximum=10_000_000,
    )).eval()
    ledger, nodes, relations, workspace, hypotheses, episodic, semantic = _rich_state(model)
    observed_records = ledger.records()
    original_nodes = nodes
    original_relation_count = int(relations.active.sum())
    _force_action(model, InternalAction.RECONSTRUCT_LOCAL)
    # Equal pointer logits deterministically choose slot zero, the accepted
    # abstraction, and make this test independent of random controller weights.
    with torch.no_grad():
        model.controller.node_query.weight.zero_()
        model.controller.node_key.weight.zero_()
    result = model._run_internal_actions(
        nodes, relations, workspace, hypotheses, episodic, semantic,
        torch.zeros(1, 8), torch.tensor([1]),
        model.default_goals(1, device=nodes.content.device, dtype=nodes.content.dtype),
        model.default_system_model(1, device=nodes.content.device, dtype=nodes.content.dtype),
        ledger, timestamps=torch.tensor([2.0]), active_rows=torch.tensor([True]),
        scale_contexts=torch.randn(1, 3, 8),
        scale_context_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    updated_nodes, updated_relations = result[:2]
    reconstructions, validity, receipts = result[-3:]
    reconstructed = (
        updated_nodes.active
        & (updated_nodes.source_classes == int(SourceClass.RECONSTRUCTED))
    )
    assert reconstructed.sum() == 2
    assert int(updated_relations.active.sum()) > original_relation_count
    assert reconstructions.active.sum() == 1
    assert validity.active.sum() == 1
    assert receipts.actions[0, 0] == int(InternalAction.RECONSTRUCT_LOCAL)
    assert receipts.success[0, 0]
    # Original observations and their immutable records remain byte-for-byte
    # equal; reconstructed values occupy newly allocated local slots.
    torch.testing.assert_close(updated_nodes.content[0, :2], original_nodes.content[0, :2])
    assert ledger.records()[:len(observed_records)] == observed_records
    root_ids = set()
    for provenance_id in updated_nodes.provenance_ids[reconstructed].tolist():
        record = ledger.get(provenance_id)
        assert record.source_class == SourceClass.RECONSTRUCTED
        assert record.operator == "mrcra:conditional_reconstruction:v1"
        root_ids.update(ledger.independent_roots(provenance_id))
    assert root_ids == {0, 1}


def test_retrieve_recent_uses_episodic_completion_order_not_the_selected_node_pointer():
    model = MultimodalRelationalContinuityResonanceNetwork(action_config()).eval()
    ledger, nodes, relations, workspace, hypotheses, episodic, semantic = _rich_state(model)
    latest = episodic.support[..., 2].masked_fill(~episodic.active, -torch.inf).argmax(-1)
    expected = episodic.values[torch.arange(1), latest].clone()
    _force_action(model, InternalAction.RETRIEVE_RECENT)
    # Force a graph pointer to the *older* node; the result must still come
    # from the independently ordered recent-memory buffer.
    with torch.no_grad():
        model.controller.node_query.weight.zero_()
        model.controller.node_key.weight.zero_()
    result = model._run_internal_actions(
        nodes, relations, workspace, hypotheses, episodic, semantic,
        torch.zeros(1, 8), torch.tensor([1]),
        model.default_goals(1, device=nodes.content.device, dtype=nodes.content.dtype),
        model.default_system_model(1, device=nodes.content.device, dtype=nodes.content.dtype),
        ledger, timestamps=torch.tensor([2.0]), active_rows=torch.tensor([True]),
        scale_contexts=torch.randn(1, 3, 8),
        scale_context_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    torch.testing.assert_close(result[5], expected)
    assert result[3].use_count[0, latest] > episodic.use_count[0, latest]


def test_verify_materializes_a_typed_persistent_evidence_request():
    base = action_config()
    cognitive = replace(base.cognitive, enable_agent_session_loop=True)
    model = MultimodalRelationalContinuityResonanceNetwork(MRCRAConfig(
        base.carrier, cognitive, actor_parameter_minimum=1,
        actor_parameter_maximum=10_000_000,
    )).eval()
    ledger, nodes, relations, workspace, hypotheses, episodic, semantic = _rich_state(model)
    _force_action(model, InternalAction.VERIFY)
    result = model._run_internal_actions(
        nodes, relations, workspace, hypotheses, episodic, semantic,
        torch.zeros(1, 8), torch.tensor([1]),
        model.default_goals(1, device=nodes.content.device, dtype=nodes.content.dtype),
        model.default_system_model(1, device=nodes.content.device, dtype=nodes.content.dtype),
        ledger, timestamps=torch.tensor([2.0]), active_rows=torch.tensor([True]),
        scale_contexts=torch.randn(1, 3, 8),
        scale_context_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    requests, receipts = result[-4], result[-1]
    assert requests.active.sum() == 1
    slot = requests.active[0].nonzero().item()
    assert requests.supporting_mask[0, slot].any()
    assert requests.hypothesis_mask[0, slot].any()
    assert receipts.statuses[0, 0] == int(ActionStatus.EXTERNAL_EVIDENCE_REQUIRED)
    assert not receipts.success[0, 0]


def test_self_inspection_materializes_a_provenance_backed_system_state_node():
    base = action_config()
    cognitive = replace(base.cognitive, enable_agent_session_loop=True)
    model = MultimodalRelationalContinuityResonanceNetwork(MRCRAConfig(
        base.carrier, cognitive, actor_parameter_minimum=1,
        actor_parameter_maximum=10_000_000,
    )).eval()
    ledger, nodes, relations, workspace, hypotheses, episodic, semantic = _rich_state(model)
    _force_action(model, InternalAction.INSPECT_SELF_STATE)
    result = model._run_internal_actions(
        nodes, relations, workspace, hypotheses, episodic, semantic,
        torch.zeros(1, 8), torch.tensor([1]),
        model.default_goals(1, device=nodes.content.device, dtype=nodes.content.dtype),
        model.default_system_model(1, device=nodes.content.device, dtype=nodes.content.dtype),
        ledger, timestamps=torch.tensor([2.0]), active_rows=torch.tensor([True]),
        scale_contexts=torch.randn(1, 3, 8),
        scale_context_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    updated = result[0]
    types = updated.type_logits.argmax(-1)
    reflective = updated.active & (types == int(NodeType.SYSTEM_STATE))
    assert reflective.sum() == 1
    provenance = int(updated.provenance_ids[reflective].item())
    record = ledger.get(provenance)
    assert record.source_class == SourceClass.INFERRED
    assert record.operator == "mrcra:reflective_system_state:v1"
