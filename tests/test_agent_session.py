from dataclasses import replace

import pytest
import torch

from mrrn.agent_session import CognitiveAgentSession, ExecutorResult
from mrrn.cognitive_model import MultimodalRelationalContinuityResonanceNetwork
from mrrn.cognitive_types import AgentMode, ModalityClass, SourceClass
from mrrn.config import MRCRAConfig
from mrrn.interaction import (
    ActionParameterSpec, ActionSchema, ActionSchemaRegistry,
)
from mrrn.provenance import ProvenanceLedger
from test_cognitive_actions import _force_action, action_config
from test_cognitive_model import force_events, packet
from mrrn.cognitive_types import InternalAction


def session_model():
    base = action_config()
    cognitive = replace(
        base.cognitive, enable_agent_session_loop=True,
        enable_post_deliberation_action_selection=True,
        enable_multi_hypothesis_planning=True,
    )
    model = MultimodalRelationalContinuityResonanceNetwork(MRCRAConfig(
        base.carrier, cognitive, actor_parameter_minimum=1,
        actor_parameter_maximum=10_000_000,
    ), model_authority="test-agent-v1").eval()
    force_events(model)
    _force_action(model, InternalAction.HALT)
    with torch.no_grad():
        model.external_action_policy.logits.weight.zero_()
        model.external_action_policy.logits.bias.copy_(torch.tensor([10.0, -10.0]))
        model.world_model.constraint.weight.zero_()
        model.world_model.constraint.bias.fill_(-10)
    return model


def explicit_goals(model):
    goals = model.default_goals(1, device=torch.device("cpu"), dtype=torch.float32)
    return replace(
        goals, desired_outcomes=torch.ones_like(goals.desired_outcomes),
        priorities=torch.ones_like(goals.priorities),
        authority=torch.ones_like(goals.authority),
        mask=torch.ones_like(goals.mask),
    )


def registry():
    return ActionSchemaRegistry((ActionSchema(
        0, "inspect", (ActionParameterSpec("focus", -1, 1, False),),
        required_capabilities=("sensor",), required_permissions=("inspect",),
        expected_modalities=(int(ModalityClass.SENSOR),),
        information_gathering=True,
    ),), capabilities=("sensor",), permissions=("inspect",))


class DeterministicExecutor:
    def execute(self, action, sequence_number):
        return ExecutorResult(
            f"receipt-{sequence_number}", sequence_number, action.schema_id,
            action.batch_index, 1.0, 0.02, 0.5, 0.1, 0.0,
            torch.full((8,), 0.25), ModalityClass.SENSOR,
            float(sequence_number + 2), "tool://sensor/reading",
            SourceClass.TOOL_OUTPUT, 1.0, 0.95,
        )


def test_registry_requires_explicit_capability_permission_and_bounded_arguments():
    schema = ActionSchema(
        0, "move", (ActionParameterSpec("distance", -1, 1),),
        required_capabilities=("actuator",), required_permissions=("move",),
    )
    denied = ActionSchemaRegistry((schema,), capabilities=("actuator",))
    assert not denied.availability_mask(1, 2)[0, 0]
    allowed = ActionSchemaRegistry(
        (schema,), capabilities=("actuator",), permissions=("move",)
    )
    assert allowed.availability_mask(1, 2)[0, 0]


def test_action_capable_session_fails_closed_without_explicit_authorized_goal():
    model = session_model()
    with pytest.raises(ValueError, match="explicit authorized goal"):
        CognitiveAgentSession(
            model, mode=AgentMode.TASK_AGENT, action_registry=registry()
        )


def test_full_observe_deliberate_execute_feedback_loop_is_idempotent_and_resumable(tmp_path):
    torch.manual_seed(457)
    model = session_model()
    ledger = ProvenanceLedger()
    session = CognitiveAgentSession(
        model, mode=AgentMode.TASK_AGENT, action_registry=registry(),
        goals=explicit_goals(model), ledger=ledger, environment_id=7, session_id=9,
    )
    first = session.observe(packet(torch.randn(1, 2, 8), ledger))
    assert first.output is not None
    deliberation = session.deliberate()
    assert len(deliberation.actions) == 1
    assert deliberation.actions[0].schema_name == "inspect"
    receipts = session.execute(DeterministicExecutor(), deliberation)
    assert len(receipts) == 1
    checkpoint = tmp_path / "after-execute.pt"
    session.checkpoint(checkpoint)
    resumed = CognitiveAgentSession.resume(checkpoint, model, registry())
    # The already executed receipt survives restart and is applied exactly once.
    ingested = resumed.ingest_result(receipts[0])
    assert ingested.output is not None
    assert resumed.state.system_model.action_success[0, 0] == 1
    assert resumed.state.system_model.action_latency[0, 0] == pytest.approx(0.02)
    assert resumed.state.system_model.action_reward[0, 0] == pytest.approx(0.5)
    assert resumed.state.system_model.action_cost[0, 0] == pytest.approx(0.1)
    assert resumed.state.system_model.action_constraint_violation[0, 0] == 0
    assert resumed.state.system_model.action_reversibility[0, 0] == 1
    assert resumed.state.system_model.executor_reliability[0, 0] == pytest.approx(0.95)
    assert resumed.state.goals.progress[0, 0] == pytest.approx(0.5)
    artifact_slot = resumed.record_artifact(
        artifact_id=44, content_digest=bytes(range(32)), version=1,
        creator_action_id=0,
        parent_provenance_ids=(receipts[0].action_provenance_id,),
        expected_persistence=100.0, estimated_cost=0.01, timestamp=4.0,
    )
    assert artifact_slot == 0
    assert resumed.state.external_artifacts.active[0, artifact_slot]
    with pytest.raises(ValueError, match="already ingested"):
        resumed.ingest_result(receipts[0])
    assert any(
        record.source_class == SourceClass.TOOL_OUTPUT
        for record in resumed.ledger.records()
    )
