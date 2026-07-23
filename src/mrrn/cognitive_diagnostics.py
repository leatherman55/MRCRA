"""Bounded, source-grounded diagnostics for MRCRA cognitive state."""

from __future__ import annotations

from collections import Counter
from math import isfinite
from typing import Any

import torch

from .cognitive_model import ActionStatus, MRCRAOutput
from .cognitive_types import InternalAction, NodeType, RelationFamily, SourceClass
from .knowledge import KnowledgeKind, KnowledgeStatus
from .provenance import ProvenanceLedger


def _number(value) -> float:
    result = float(value.detach().float().cpu()) if isinstance(value, torch.Tensor) else float(value)
    if not isfinite(result):
        return 0.0
    return result


def cognitive_metrics(output: MRCRAOutput, ledger: ProvenanceLedger) -> dict[str, float]:
    """Low-cardinality metrics suitable for ordinary Trackio charts."""

    nodes, relations, workspace = output.nodes, output.relations, output.workspace
    hypotheses, state = output.hypotheses, output.state
    receipts = output.action_receipts
    attempted = receipts.mask
    successful = receipts.success & attempted
    metacognitive_mask = output.metacognitive_mask
    metacognitive = output.metacognitive_values
    routed_rows = metacognitive[metacognitive_mask]
    relation_arity = relations.participant_mask.sum(-1)
    metrics = {
        "cognition/active_nodes": _number(nodes.active.sum()),
        "cognition/node_capacity_fraction": _number(nodes.active.float().mean()),
        "cognition/active_relations": _number(relations.active.sum()),
        "cognition/active_pair_relations": _number(
            (relations.active & (relation_arity == 2)).sum()
        ),
        "cognition/active_hyperrelations": _number(
            (relations.active & (relation_arity > 2)).sum()
        ),
        "cognition/relation_capacity_fraction": _number(relations.active.float().mean()),
        "cognition/workspace_active": _number(workspace.active.sum()),
        "cognition/workspace_capacity_fraction": _number(workspace.active.float().mean()),
        "cognition/episodic_memory_active": _number(state.episodic_memory.active.sum()),
        "cognition/semantic_memory_active": _number(state.semantic_memory.active.sum()),
        "cognition/knowledge_pending": _number(
            (state.knowledge.active & (state.knowledge.status == int(KnowledgeStatus.PENDING))).sum()
        ),
        "cognition/knowledge_accepted": _number(
            (state.knowledge.active & (state.knowledge.status == int(KnowledgeStatus.ACCEPTED))).sum()
        ),
        "cognition/knowledge_rejected": _number(
            (state.knowledge.active & (state.knowledge.status == int(KnowledgeStatus.REJECTED))).sum()
        ),
        "cognition/active_hypotheses": _number(hypotheses.active.sum()),
        "cognition/effective_hypotheses": _number(hypotheses.effective_count.mean()),
        "cognition/cognitive_cycle_fraction": _number(output.cognitive_cycles.float().mean()),
        "cognition/events_per_position": _number(output.event_counts.float().mean()),
        "cognition/mean_uncertainty": _number(output.uncertainty.mean()),
        "cognition/action_attempts": _number(attempted.sum()),
        "cognition/action_success_rate": _number(
            successful.sum() / attempted.sum().clamp_min(1)
        ),
        "cognition/abstention_fraction": _number(output.abstained.float().mean()),
        "cognition/external_action_active": _number(output.external_action.active.sum()),
        "cognition/external_action_authorized": _number(output.external_action.authorized.sum()),
        "cognition/external_action_abstained": _number(output.external_action.abstained.sum()),
        "cognition/schema_entropy": _number(
            -(output.schema_probabilities.clamp_min(1e-8)
              * output.schema_probabilities.clamp_min(1e-8).log()).sum(-1).mean()
        ),
        "cognition/symbol_activation": _number(output.symbol_gates.mean()),
        "cognition/calibration_ece": _number(
            output.calibration.expected_calibration_error.mean()
        ),
        "cognition/calibration_brier": _number(output.calibration.brier_score.mean()),
        "cognition/remaining_compute": _number(state.system_model.remaining_compute.mean()),
        "cognition/remaining_memory": _number(state.system_model.remaining_memory.mean()),
        "cognition/provenance_records": float(len(ledger)),
        "cognition/provenance_verification_events": float(ledger.verification_event_count),
        "cognition/external_clock": float(state.clocks.external),
        "cognition/cognitive_clock": float(state.clocks.cognitive),
        "cognition/optimizer_clock": float(state.clocks.optimizer),
        "cognition/mean_selected_physical_scale": _number(
            state.selected_physical_scale.float().mean()
        ),
        "cognition/reconstructions_active": _number(state.reconstructions.active.sum()),
        "cognition/reconstruction_historical_fidelity": _number(
            state.reconstructions.historical_fidelity[state.reconstructions.active].mean()
            if bool(state.reconstructions.active.any()) else 0
        ),
        "cognition/reconstruction_evidence_agreement": _number(
            state.reconstructions.evidence_agreement[state.reconstructions.active].mean()
            if bool(state.reconstructions.active.any()) else 0
        ),
        "cognition/abstractions_validated": _number(state.abstraction_validity.active.sum()),
        "cognition/abstraction_applicability": _number(
            state.abstraction_validity.applicability[state.abstraction_validity.active].mean()
            if bool(state.abstraction_validity.active.any()) else 0
        ),
        "cognition/action_candidates": _number(state.action_candidates.active.sum()),
        "cognition/action_candidates_authorized": _number((
            state.action_candidates.active & state.action_candidates.permitted
            & state.action_candidates.provenance_authorized
            & state.action_candidates.viability_authorized
        ).sum()),
        "cognition/evidence_requests_pending": _number(state.evidence_requests.active.sum()),
        "cognition/external_artifacts_active": _number(state.external_artifacts.active.sum()),
        "cognition/viability_channels_active": _number(state.viability.active.sum()),
        "cognition/viability_hard_violations": _number(state.viability.hard_violation.sum()),
        "cognition/metacognitive_records": _number(state.metacognition.active.sum()),
        "cognition/metacognitive_prediction_error": _number(
            (state.metacognition.predicted_error - state.metacognition.realized_error).abs()[
                state.metacognition.active
            ].mean() if bool(state.metacognition.active.any()) else 0
        ),
        "cognition/metacognitive_routed_positions": _number(metacognitive_mask.sum()),
        "cognition/metacognitive_predicted_error": _number(
            routed_rows[:, 0].mean() if routed_rows.numel() else 0
        ),
        "cognition/metacognitive_value_compute": _number(
            routed_rows[:, 1].mean() if routed_rows.numel() else 0
        ),
        "cognition/metacognitive_value_retrieval": _number(
            routed_rows[:, 2].mean() if routed_rows.numel() else 0
        ),
        "cognition/metacognitive_value_reconstruction": _number(
            routed_rows[:, 3].mean() if routed_rows.numel() else 0
        ),
        "cognition/metacognitive_value_simulation": _number(
            routed_rows[:, 4].mean() if routed_rows.numel() else 0
        ),
        "cognition/metacognitive_value_evidence": _number(
            routed_rows[:, 5].mean() if routed_rows.numel() else 0
        ),
        "cognition/metacognitive_calibration_prediction": _number(
            routed_rows[:, 6].mean() if routed_rows.numel() else 0
        ),
        "cognition/boundary_scope": _number(state.boundary_context.scope.float().mean()),
        "cognition/boundary_reset_count": _number(state.boundary_context.reset_counts.sum()),
        "cognition/unknown_hypotheses": _number(hypotheses.unknown.sum()),
        "cognition/measured_action_reward": _number(state.system_model.action_reward.mean()),
        "cognition/measured_action_cost": _number(state.system_model.action_cost.mean()),
        "cognition/measured_constraint_violation": _number(
            state.system_model.action_constraint_violation.mean()
        ),
    }
    for action in InternalAction:
        selected = attempted & (receipts.actions == int(action))
        if bool(selected.any()):
            metrics[f"actions/{action.name.lower()}/count"] = _number(selected.sum())
            metrics[f"actions/{action.name.lower()}/success_rate"] = _number(
                (successful & selected).sum() / selected.sum().clamp_min(1)
            )
    return metrics


def cognitive_evidence(
    output: MRCRAOutput, ledger: ProvenanceLedger, *, maximum_records: int = 256,
) -> dict[str, Any]:
    """Serialize the live typed graph, bounded stores, actions, and authority.

    Neural vectors are intentionally summarized rather than exported wholesale.
    Every decoded node and relation retains the exact slot/version/provenance
    identifiers needed to relate a visual mark back to runtime authority.
    """

    if maximum_records <= 0:
        raise ValueError("maximum diagnostic record count must be positive")
    nodes, relations, workspace = output.nodes, output.relations, output.workspace
    result_nodes = []
    for batch, slot in nodes.active.nonzero(as_tuple=False).tolist()[:maximum_records]:
        type_probability = torch.softmax(nodes.type_logits[batch, slot].float(), -1)
        type_id = int(type_probability.argmax())
        result_nodes.append({
            "batch": batch, "slot": slot, "version": int(nodes.versions[batch, slot]),
            "type": NodeType(type_id).name.lower(),
            "type_confidence": _number(type_probability[type_id]),
            "source": SourceClass(int(nodes.source_classes[batch, slot])).name.lower(),
            "scenario": int(nodes.scenario_ids[batch, slot]),
            "provenance": int(nodes.provenance_ids[batch, slot]),
            "support": [_number(value) for value in nodes.support[batch, slot]],
            "uncertainty": [_number(value) for value in nodes.uncertainty[batch, slot]],
            "activity": _number(nodes.activity[batch, slot]),
            "importance": _number(nodes.importance[batch, slot]),
            "age": _number(nodes.age[batch, slot]),
        })
    result_relations = []
    for batch, slot in relations.active.nonzero(as_tuple=False).tolist()[:maximum_records]:
        family_probability = torch.softmax(relations.type_logits[batch, slot].float(), -1)
        family_id = int(family_probability.argmax())
        participant_mask = relations.participant_mask[batch, slot]
        result_relations.append({
            "batch": batch, "slot": slot, "version": int(relations.versions[batch, slot]),
            "family": RelationFamily(family_id).name.lower(),
            "family_confidence": _number(family_probability[family_id]),
            "participants": relations.participant_indices[batch, slot, participant_mask].tolist(),
            "participant_versions": relations.participant_versions[
                batch, slot, participant_mask
            ].tolist(),
            "roles": relations.participant_roles[batch, slot, participant_mask].tolist(),
            "confidence": _number(relations.confidence[batch, slot].mean()),
            "provenance": int(relations.provenance_ids[batch, slot]),
            "scenario": int(relations.scenario_ids[batch, slot]),
            "support": [_number(value) for value in relations.support[batch, slot]],
        })
    workspace_rows = [
        {
            "batch": batch, "slot": slot,
            "node": int(workspace.node_pointers[batch, slot]),
            "score": _number(workspace.pointer_scores[batch, slot]),
            "age": int(workspace.ages[batch, slot]),
        }
        for batch, slot in workspace.active.nonzero(as_tuple=False).tolist()
    ]

    def memory_rows(memory, tier: str) -> list[dict[str, Any]]:
        rows = []
        for batch, slot in memory.active.nonzero(as_tuple=False).tolist()[:maximum_records]:
            rows.append({
                "tier": tier, "batch": batch, "slot": slot,
                "version": int(memory.versions[batch, slot]),
                "type_id": int(memory.type_ids[batch, slot]),
                "source": SourceClass(int(memory.source_classes[batch, slot])).name.lower(),
                "scenario": int(memory.scenario_ids[batch, slot]),
                "provenance": int(memory.provenance_ids[batch, slot]),
                "utility": _number(memory.utility[batch, slot]),
                "uses": int(memory.use_count[batch, slot]),
                "last_access": int(memory.last_access[batch, slot]),
                "associations": int(memory.association_mask[batch, slot].sum()),
                "uncertainty": _number(memory.uncertainty[batch, slot].mean()),
            })
        return rows

    hypothesis_rows = []
    weights = output.hypotheses.weights
    for batch, slot in output.hypotheses.active.nonzero(as_tuple=False).tolist():
        hypothesis_rows.append({
            "batch": batch, "slot": slot,
            "scenario": int(output.hypotheses.scenario_ids[batch, slot]),
            "version": int(output.hypotheses.versions[batch, slot]),
            "weight": _number(weights[batch, slot]),
            "support": _number(output.hypotheses.supporting_evidence[batch, slot]),
            "contradiction": _number(output.hypotheses.contradicting_evidence[batch, slot]),
            "latest_supporting_provenance": int(
                output.hypotheses.latest_supporting_provenance_ids[batch, slot]
            ),
            "latest_contradicting_provenance": int(
                output.hypotheses.latest_contradicting_provenance_ids[batch, slot]
            ),
            "unknown": bool(output.hypotheses.unknown[batch, slot]),
            "uncertainty": _number(output.hypotheses.uncertainty[batch, slot].mean()),
        })

    receipts = output.action_receipts
    action_rows = []
    for batch, time, internal in receipts.mask.nonzero(as_tuple=False).tolist()[:maximum_records]:
        action_id = int(receipts.actions[batch, time, internal])
        status_id = int(receipts.statuses[batch, time, internal])
        action_rows.append({
            "batch": batch, "time": time, "internal_step": internal,
            "action": InternalAction(action_id).name.lower(),
            "status": ActionStatus(status_id).name.lower(),
            "success": bool(receipts.success[batch, time, internal]),
            "node": int(receipts.node_pointers[batch, time, internal]),
            "relation": int(receipts.relation_pointers[batch, time, internal]),
            "knowledge": int(receipts.knowledge_pointers[batch, time, internal]),
        })

    knowledge_rows = []
    knowledge = output.knowledge
    for batch, slot in knowledge.active.nonzero(as_tuple=False).tolist()[:maximum_records]:
        knowledge_rows.append({
            "batch": batch, "slot": slot,
            "kind": KnowledgeKind(int(knowledge.kind[batch, slot])).name.lower(),
            "status": KnowledgeStatus(int(knowledge.status[batch, slot])).name.lower(),
            "version": int(knowledge.versions[batch, slot]),
            "provenance": int(knowledge.provenance_ids[batch, slot]),
            "code_gain_bits": _number(knowledge.code_gain_bits[batch, slot]),
            "reconstruction_distortion": _number(
                knowledge.reconstruction_distortion[batch, slot]
            ),
            "relation_distortion": _number(knowledge.relation_distortion[batch, slot]),
            "predictive_utility": _number(knowledge.predictive_utility[batch, slot]),
            "action_utility": _number(knowledge.action_utility[batch, slot]),
            "confidence": _number(knowledge.confidence[batch, slot]),
            "counterexample_search_completed": bool(
                knowledge.counterexample_search_completed[batch, slot]
            ),
        })

    external = output.external_action
    external_rows = []
    for batch in range(external.logits.shape[0]):
        if bool(external.active[batch] | external.abstained[batch]):
            external_rows.append({
                "batch": batch,
                "selected_action": int(external.selected_action[batch]),
                "active": bool(external.active[batch]),
                "abstained": bool(external.abstained[batch]),
                "authorized": bool(external.authorized[batch]),
                "available": external.available[batch].detach().cpu().tolist(),
                "probabilities": [
                    _number(value) for value in external.probabilities[batch]
                ],
                "utility": [_number(value) for value in external.utility[batch]],
            })

    source_counts = Counter(record.source_class.name.lower() for record in ledger.records())
    verification_counts = Counter(
        ledger.effective_verification(record.record_id).name.lower()
        for record in ledger.records()
    )
    uncertainty_channels = output.uncertainty.detach().float().mean((0, 1)).cpu().tolist()
    reconstructions = [
        {
            "batch": batch, "slot": slot,
            "abstraction": int(output.state.reconstructions.abstraction_indices[batch, slot]),
            "historical_fidelity": _number(output.state.reconstructions.historical_fidelity[batch, slot]),
            "structural_plausibility": _number(output.state.reconstructions.structural_plausibility[batch, slot]),
            "evidence_agreement": _number(output.state.reconstructions.evidence_agreement[batch, slot]),
            "provenance": int(output.state.reconstructions.provenance_ids[batch, slot]),
            "scale": int(output.state.reconstructions.physical_scales[batch, slot]),
            "depth": int(output.state.reconstructions.abstraction_depths[batch, slot]),
        }
        for batch, slot in output.state.reconstructions.active.nonzero(as_tuple=False).tolist()[:maximum_records]
    ]
    candidates = [
        {
            "batch": batch, "slot": slot,
            "schema": int(output.state.action_candidates.schema_ids[batch, slot]),
            "utility": _number(output.state.action_candidates.normalized_utility[batch, slot]),
            "information_gain": _number(output.state.action_candidates.information_gain[batch, slot]),
            "tail_risk": _number(output.state.action_candidates.tail_risk[batch, slot]),
            "permitted": bool(output.state.action_candidates.permitted[batch, slot]),
            "provenance_authorized": bool(output.state.action_candidates.provenance_authorized[batch, slot]),
            "viability_authorized": bool(output.state.action_candidates.viability_authorized[batch, slot]),
            "selected": bool(output.state.action_candidates.selected[batch, slot]),
        }
        for batch, slot in output.state.action_candidates.active.nonzero(as_tuple=False).tolist()[:maximum_records]
    ]
    evidence_requests = [
        {
            "batch": batch, "slot": slot,
            "status": int(output.state.evidence_requests.status[batch, slot]),
            "modality": int(output.state.evidence_requests.requested_modalities[batch, slot]),
            "tool_schema": int(output.state.evidence_requests.tool_schema_ids[batch, slot]),
            "information_gain": _number(output.state.evidence_requests.expected_information_gain[batch, slot]),
            "maximum_cost": _number(output.state.evidence_requests.maximum_cost[batch, slot]),
            "required_precision": _number(output.state.evidence_requests.required_precision[batch, slot]),
        }
        for batch, slot in output.state.evidence_requests.active.nonzero(as_tuple=False).tolist()[:maximum_records]
    ]
    operation_names = (
        "compute", "retrieval", "reconstruction", "simulation", "evidence",
    )
    metacognitive_steps = []
    for batch, time in output.metacognitive_mask.nonzero(as_tuple=False).tolist()[:maximum_records]:
        values = output.metacognitive_values[batch, time]
        operation_values = values[1:6]
        selected = int(operation_values.argmax())
        metacognitive_steps.append({
            "batch": batch,
            "time": time,
            "predicted_error": _number(values[0]),
            "operation_values": {
                name: _number(operation_values[index])
                for index, name in enumerate(operation_names)
            },
            "highest_value_operation": operation_names[selected],
            "calibration_error": _number(values[6]),
        })
    metacognitive_history = [
        {
            "batch": batch,
            "slot": slot,
            "predicted_error": _number(output.state.metacognition.predicted_error[batch, slot]),
            "realized_error": _number(output.state.metacognition.realized_error[batch, slot]),
            "value_compute": _number(output.state.metacognition.value_of_compute[batch, slot]),
            "value_retrieval": _number(output.state.metacognition.value_of_retrieval[batch, slot]),
            "value_reconstruction": _number(output.state.metacognition.value_of_reconstruction[batch, slot]),
            "value_simulation": _number(output.state.metacognition.value_of_simulation[batch, slot]),
            "value_evidence": _number(output.state.metacognition.value_of_evidence[batch, slot]),
            "calibration_error": _number(output.state.metacognition.calibration_error[batch, slot]),
            "decision_action": int(output.state.metacognition.decision_actions[batch, slot]),
            "provenance": int(output.state.metacognition.provenance_ids[batch, slot]),
        }
        for batch, slot in output.state.metacognition.active.nonzero(as_tuple=False).tolist()[:maximum_records]
    ]
    return {
        "schema_version": 4,
        "summary": cognitive_metrics(output, ledger),
        "graph": {"nodes": result_nodes, "relations": result_relations},
        "workspace": workspace_rows,
        "memory": (
            memory_rows(output.state.episodic_memory, "episodic")
            + memory_rows(output.state.semantic_memory, "semantic")
        ),
        "hypotheses": hypothesis_rows,
        "knowledge": knowledge_rows,
        "reconstructions": reconstructions,
        "action_candidates": candidates,
        "evidence_requests": evidence_requests,
        "metacognition": {
            "steps": metacognitive_steps,
            "history": metacognitive_history,
        },
        "viability": {
            "active": output.state.viability.active.detach().cpu().tolist(),
            "values": output.state.viability.values.detach().float().cpu().tolist(),
            "target_low": output.state.viability.target_low.detach().float().cpu().tolist(),
            "target_high": output.state.viability.target_high.detach().float().cpu().tolist(),
            "hard_violations": output.state.viability.hard_violation.detach().cpu().tolist(),
        },
        "boundary_context": {
            "scope": output.state.boundary_context.scope.detach().cpu().tolist(),
            "continuity_ids": output.state.boundary_context.continuity_ids.detach().cpu().tolist(),
            "environment_ids": output.state.boundary_context.environment_ids.detach().cpu().tolist(),
            "session_ids": output.state.boundary_context.session_ids.detach().cpu().tolist(),
            "sequence_numbers": output.state.boundary_context.sequence_numbers.detach().cpu().tolist(),
            "reset_counts": output.state.boundary_context.reset_counts.detach().cpu().tolist(),
        },
        "uncertainty_channels": [round(float(value), 7) for value in uncertainty_channels],
        "timeline": {
            "event_counts": output.event_counts.detach().cpu().tolist(),
            "cognitive_cycles": output.cognitive_cycles.detach().cpu().tolist(),
            "selected_physical_scale": output.state.selected_physical_scale.detach().cpu().tolist(),
            "metacognitive_mask": output.metacognitive_mask.detach().cpu().tolist(),
        },
        "actions": action_rows,
        "external_actions": external_rows,
        "provenance": {
            "digest": output.provenance_digest,
            "records": len(ledger),
            "verification_events": ledger.verification_event_count,
            "by_source": dict(sorted(source_counts.items())),
            "by_verification": dict(sorted(verification_counts.items())),
        },
        "truncated": {
            "nodes": int(nodes.active.sum()) > len(result_nodes),
            "relations": int(relations.active.sum()) > len(result_relations),
            "actions": int(receipts.mask.sum()) > len(action_rows),
        },
    }
