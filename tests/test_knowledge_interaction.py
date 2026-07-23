from dataclasses import replace

import pytest
import torch

from mrrn.cognitive_model import MultimodalRelationalContinuityResonanceNetwork
from mrrn.cognitive_types import (
    ModalityClass, NodeType, SourceClass, SupportInterval, VerificationClass,
)
from mrrn.config import CognitiveConfig, MRCRAConfig, MRRNConfig
from mrrn.interaction import (
    ExternalActionDecision, ExternalActionFeedback, authorize_external_actions,
    execute_authorized_actions, update_system_model_from_feedback,
)
from mrrn.knowledge import (
    KnowledgeKind, KnowledgeProposalBank, KnowledgeProposalBatch,
    KnowledgeProposalState, KnowledgeStatus, KnowledgeValidationBatch,
)
from mrrn.provenance import ProvenanceLedger


def _config() -> MRCRAConfig:
    carrier = MRRNConfig(
        input_dim=8, model_dim=8, output_dim=7, layers=1, scales=3,
        heads=2, modes=2, mimo_rank=1, attention_window=2,
        retrieved_items=1, memory_capacity=4, width_multiple=4,
        mixer_expansion=1.5, spectral_modes=2, spectral_basis_order=2,
        enable_global_head=False, relational_branch=True,
        relational_context_dim=8,
    )
    cognitive = CognitiveConfig(
        workspace_dim=8, provenance_features=16, uncertainty_channels=8,
        relation_heads=2, relation_modes=2, relation_adapter_rank=2,
        goal_slots=2, goal_constraint_dim=2, system_action_channels=2,
        calibration_regimes=2, calibration_bins=4,
        operational_schema_count=3, knowledge_candidate_capacity=3,
        knowledge_support_capacity=2, active_event_capacity=6,
        pair_edge_capacity=6, hyperedge_capacity=2, graph_neighbors=2,
        global_workspace_slots=2, hypothesis_slots=2,
        maximum_hypothesis_slots=2, maximum_cognitive_steps=2,
        event_chunk_size=2, event_proposals_per_chunk=2,
        recent_candidates=2, landmark_candidates=1,
        episodic_candidates=1, semantic_candidates=1,
        episodic_memory_capacity=4, semantic_memory_capacity=3,
        associative_depth=2, associative_budget=2,
        world_model_horizons=(1, 2),
    )
    return MRCRAConfig(
        carrier, cognitive, actor_parameter_minimum=1,
        actor_parameter_maximum=10_000_000,
    )


def _roots(ledger: ProvenanceLedger) -> tuple[int, int]:
    return tuple(
        ledger.append(
            source_class=SourceClass.EXTERNAL,
            source_uri_or_episode=f"evidence://root/{index}",
            support=SupportInterval(index, index, index),
            modality=ModalityClass.TEXT, operator="test:observe", scenario_id=0,
            model_authority="test", verification=VerificationClass.EXTERNALLY_CHECKED,
        )
        for index in range(2)
    )


def _proposal(
    ledger: ProvenanceLedger, roots: tuple[int, ...], *,
    kind: KnowledgeKind = KnowledgeKind.INVARIANT,
) -> KnowledgeProposalBatch:
    provenance = ledger.derive(
        roots, source_class=SourceClass.ABSTRACTED,
        operator="test:knowledge-proposal", support=SupportInterval(0, 1, 1),
        modality=ModalityClass.MEMORY, model_authority="test",
    )
    supporters = torch.full((1, 2), -1, dtype=torch.int64)
    mask = torch.zeros(1, 2, dtype=torch.bool)
    supporters[0, : len(roots)] = torch.tensor(roots)
    mask[0, : len(roots)] = True
    return KnowledgeProposalBatch(
        torch.randn(1, 8), torch.tensor([int(kind)]), torch.tensor([32.0]),
        torch.tensor([0.01]), torch.tensor([0.01]), torch.zeros(1),
        torch.zeros(1), torch.tensor([0.9]), torch.tensor([provenance]),
        supporters, mask, torch.tensor([True]),
    )


def _validation(index: int = 0, *, counterexample: bool = True):
    return KnowledgeValidationBatch(
        torch.tensor([index]), torch.tensor([1.0]), torch.tensor([0.5]),
        torch.tensor([0.4]), torch.tensor([0.0]), torch.tensor([0.85]),
        torch.tensor([counterexample]), torch.tensor([True]),
    )


def test_invariant_authority_requires_independent_roots_and_counterexample_search():
    ledger = ProvenanceLedger()
    roots = _roots(ledger)
    state = KnowledgeProposalState.empty(1, 3, 8, 2)
    state, written = KnowledgeProposalBank.propose(state, _proposal(ledger, roots))
    assert written.item() == 0
    validated, result = KnowledgeProposalBank.validate(
        state, _validation(), ledger, minimum_code_gain_bits=1,
        maximum_reconstruction_distortion=0.1,
        maximum_relation_distortion=0.1,
    )
    assert result.accepted.item()
    assert validated.status[0, 0] == int(KnowledgeStatus.ACCEPTED)

    one_root = KnowledgeProposalState.empty(1, 3, 8, 2)
    one_root, _ = KnowledgeProposalBank.propose(
        one_root, _proposal(ledger, roots[:1])
    )
    rejected, result = KnowledgeProposalBank.validate(
        one_root, _validation(), ledger, minimum_code_gain_bits=1,
        maximum_reconstruction_distortion=0.1,
        maximum_relation_distortion=0.1,
    )
    assert not result.provenance_pass.item()
    assert rejected.status[0, 0] == int(KnowledgeStatus.REJECTED)


def test_validated_knowledge_is_atomically_promoted_to_node_and_semantic_memory():
    torch.manual_seed(401)
    model = MultimodalRelationalContinuityResonanceNetwork(_config()).eval()
    ledger = ProvenanceLedger()
    roots = _roots(ledger)
    state = model.initial_state(1, sample_intervals=torch.ones(1))
    knowledge, written = KnowledgeProposalBank.propose(
        state.knowledge, _proposal(ledger, roots)
    )
    state = replace(state, knowledge=knowledge)
    state, result = model.validate_and_promote_knowledge(
        state, _validation(int(written.item())), ledger,
        minimum_code_gain_bits=1,
        maximum_reconstruction_distortion=0.1,
        maximum_relation_distortion=0.1,
    )
    assert result.accepted.item()
    assert state.semantic_memory.active.sum() == 1
    assert state.semantic_memory.type_ids[state.semantic_memory.active].item() == int(NodeType.INVARIANT)
    assert (state.nodes.type_logits.argmax(-1)[state.nodes.active] == int(NodeType.INVARIANT)).any()
    provenance_id = int(state.knowledge.provenance_ids[0, written])
    assert ledger.effective_verification(provenance_id) == VerificationClass.INTERNALLY_CONSISTENT


class _Executor:
    def __init__(self):
        self.calls = []

    def execute(self, action_index: int, batch_index: int):
        self.calls.append((action_index, batch_index))
        return action_index + 10


def test_external_actions_require_permission_and_provenance_and_feedback_updates_self_model():
    model = MultimodalRelationalContinuityResonanceNetwork(_config())
    state = model.initial_state(1, sample_intervals=torch.ones(1))
    ledger = ProvenanceLedger()
    root = _roots(ledger)[0]
    decision = ExternalActionDecision(
        torch.tensor([[2.0, 0.0]]), torch.tensor([[0.9, 0.1]]),
        torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]),
        torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1, 2),
        torch.tensor([[True, False]]), torch.tensor([0]),
        torch.tensor([[root, -1]]), torch.tensor([[True, False]]),
        torch.tensor([True]), torch.tensor([False]), torch.tensor([False]),
    )
    authorized = authorize_external_actions(decision, ledger)
    assert authorized.authorized.item()
    executor = _Executor()
    assert execute_authorized_actions(authorized, executor) == (10,)
    assert executor.calls == [(0, 0)]

    feedback = ExternalActionFeedback(
        torch.tensor([0]), torch.tensor([1.0]), torch.tensor([0.25]),
        torch.tensor([2.0]), torch.tensor([0.1]), torch.tensor([0.0]),
        torch.tensor([root]), torch.tensor([True]),
    )
    updated = update_system_model_from_feedback(
        state.system_model, feedback, ledger, momentum=0.0
    )
    assert updated.action_success[0, 0] == 1
    assert updated.action_latency[0, 0] == 0.25

    ledger.set_verification(
        root, VerificationClass.REVOKED, authority="test", reason="invalid sensor"
    )
    denied = authorize_external_actions(decision, ledger)
    assert not denied.authorized.item()
    assert execute_authorized_actions(denied, executor) == (None,)


def test_controller_cycles_preserve_hidden_state_but_reset_local_budget_and_history():
    model = MultimodalRelationalContinuityResonanceNetwork(_config())
    previous = model.controller.initial_state(1)
    previous = replace(
        previous, hidden=torch.randn_like(previous.hidden),
        remaining_steps=torch.zeros_like(previous.remaining_steps),
        halted=torch.ones_like(previous.halted), step=2,
    )
    cycle = model.controller.begin_cycle(previous, torch.tensor([True]))
    torch.testing.assert_close(cycle.hidden, previous.hidden)
    assert cycle.remaining_steps.item() == model.controller.maximum_steps
    assert cycle.step == 0 and not cycle.halted.item()
    assert not cycle.history_mask.any()


def test_knowledge_rejection_revocation_and_invalid_transitions_are_auditable():
    ledger = ProvenanceLedger()
    roots = _roots(ledger)
    state = KnowledgeProposalState.empty(1, 2, 8, 2)
    state, written = KnowledgeProposalBank.propose(state, _proposal(ledger, roots))
    with pytest.raises(ValueError, match="thresholds"):
        KnowledgeProposalBank.validate(
            state, _validation(), ledger, minimum_code_gain_bits=-1,
            maximum_reconstruction_distortion=1,
            maximum_relation_distortion=1,
        )
    with pytest.raises(ValueError, match="inactive"):
        KnowledgeProposalBank.validate(
            state, _validation(1), ledger, minimum_code_gain_bits=1,
            maximum_reconstruction_distortion=1,
            maximum_relation_distortion=1,
        )
    accepted, _ = KnowledgeProposalBank.validate(
        state, _validation(int(written)), ledger, minimum_code_gain_bits=1,
        maximum_reconstruction_distortion=1,
        maximum_relation_distortion=1,
    )
    revoked = KnowledgeProposalBank.revoke(
        accepted, written, torch.tensor([True])
    )
    assert revoked.status[0, written] == int(KnowledgeStatus.REVOKED)
    assert revoked.versions[0, written] == accepted.versions[0, written] + 1
    with pytest.raises(ValueError, match="accepted"):
        KnowledgeProposalBank.revoke(revoked, written, torch.tensor([True]))
    with pytest.raises(ValueError, match="int64"):
        KnowledgeProposalBank.revoke(revoked, written.float(), torch.tensor([True]))
    with pytest.raises(ValueError, match="boolean"):
        KnowledgeProposalBank.revoke(revoked, written, torch.ones(1))


def test_external_feedback_and_execution_contracts_reject_untrusted_receipts():
    model = MultimodalRelationalContinuityResonanceNetwork(_config())
    state = model.initial_state(1, sample_intervals=torch.ones(1))
    ledger = ProvenanceLedger()
    root = _roots(ledger)[0]
    with pytest.raises(ValueError, match="momentum"):
        update_system_model_from_feedback(
            state.system_model,
            ExternalActionFeedback(
                torch.tensor([0]), torch.tensor([1.0]), torch.tensor([0.0]),
                torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]),
                torch.tensor([root]), torch.tensor([True]),
            ),
            ledger, momentum=1.0,
        )
    unknown = ExternalActionFeedback(
        torch.tensor([0]), torch.tensor([1.0]), torch.tensor([0.0]),
        torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]),
        torch.tensor([999]), torch.tensor([True]),
    )
    with pytest.raises(KeyError, match="unknown provenance"):
        update_system_model_from_feedback(state.system_model, unknown, ledger)
    with pytest.raises(ValueError, match="success"):
        ExternalActionFeedback(
            torch.tensor([0]), torch.tensor([2.0]), torch.tensor([0.0]),
            torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]),
            torch.tensor([root]), torch.tensor([True]),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("logits", torch.zeros(2), "logits"),
        ("probabilities", torch.zeros(1, 3), "probabilities"),
        ("available", torch.zeros(1, 2), "availability"),
        ("selected_action", torch.zeros(1), "selected action"),
        ("supporting_provenance_ids", torch.zeros(2, dtype=torch.int64), "supporters"),
        ("supporting_mask", torch.ones(1, 1), "supporting mask"),
        ("active", torch.ones(1), "active"),
        ("supporting_provenance_ids", torch.tensor([[-1]], dtype=torch.int64), "provenance"),
        ("selected_action", torch.tensor([2]), "in-range"),
        ("authorized", torch.tensor([True]), "authorized"),
    ],
)
def test_external_decision_tensor_contracts_fail_closed(field, value, message):
    base = ExternalActionDecision(
        torch.zeros(1, 2), torch.full((1, 2), 0.5), torch.zeros(1, 2),
        torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2),
        torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool), torch.tensor([0]),
        torch.tensor([[0]]), torch.tensor([[True]]), torch.tensor([True]),
        torch.tensor([False]), torch.tensor([False]),
    )
    with pytest.raises(ValueError, match=message):
        if field == "authorized":
            replace(base, authorized=value, active=torch.tensor([False]))
        else:
            replace(base, **{field: value})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selected_action", torch.zeros(1), "action"),
        ("success", torch.ones(1, dtype=torch.int64), "success"),
        ("provenance_ids", torch.zeros(1), "provenance"),
        ("mask", torch.ones(1), "mask"),
        ("success", torch.tensor([-0.1]), "success"),
        ("latency", torch.tensor([-0.1]), "cannot be negative"),
        ("provenance_ids", torch.tensor([-1]), "requires provenance"),
    ],
)
def test_external_feedback_tensor_contracts_fail_closed(field, value, message):
    base = ExternalActionFeedback(
        torch.tensor([0]), torch.tensor([1.0]), torch.tensor([0.0]),
        torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]),
        torch.tensor([0]), torch.tensor([True]),
    )
    with pytest.raises(ValueError, match=message):
        replace(base, **{field: value})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("latent", torch.zeros(1, 1, 8), "latent"),
        ("kind", torch.zeros(1), "kind"),
        ("code_gain_bits", torch.zeros(1, 1), "code_gain_bits"),
        ("supporting_provenance_ids", torch.zeros(2, dtype=torch.int64), "supporters"),
        ("supporting_mask", torch.ones(1, 2), "supporting mask"),
        ("mask", torch.ones(1), "proposal mask"),
        ("provenance_ids", torch.tensor([-1]), "derived provenance"),
        ("kind", torch.tensor([99]), "outside"),
        ("confidence", torch.tensor([2.0]), "confidence"),
        ("reconstruction_distortion", torch.tensor([-1.0]), "distortion"),
    ],
)
def test_knowledge_proposal_tensor_contracts_fail_closed(field, value, message):
    ledger = ProvenanceLedger()
    base = _proposal(ledger, _roots(ledger))
    with pytest.raises(ValueError, match=message):
        replace(base, **{field: value})
