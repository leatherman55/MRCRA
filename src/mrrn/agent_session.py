"""Application-owned perception-deliberation-execution-feedback orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor

from .cognitive_checkpoint import load_mrcra_checkpoint, save_mrcra_checkpoint
from .cognitive_model import (
    MRCRAOutput, MRCRARuntimeState,
    MultimodalRelationalContinuityResonanceNetwork,
)
from .cognitive_types import (
    AgentMode, BoundaryClass, ModalityClass, SourceClass, SupportInterval,
)
from .controller import GoalState, SystemModelState
from .interaction import (
    ActionSchemaRegistry, ExternalActionFeedback, StructuredExternalAction,
    update_system_model_from_feedback,
)
from .evidence_requests import (
    EvidenceRequestStatus, transition_evidence_requests,
)
from .external_artifacts import record_external_artifact
from .observation import ObservationPacket, register_internal_inputs
from .provenance import ProvenanceLedger
from .viability import ViabilityState, update_measured_viability


@dataclass(frozen=True, slots=True)
class DeliberationResult:
    actions: tuple[StructuredExternalAction, ...]
    abstained: Tensor
    candidate_count: Tensor
    provenance_digest: str


@dataclass(frozen=True, slots=True)
class ExecutorResult:
    receipt_id: str
    sequence_number: int
    schema_id: int
    batch_index: int
    success: float
    latency_seconds: float
    reward: float
    cost: float
    constraint_violation: float
    observation: Tensor | None
    modality: ModalityClass
    timestamp: float
    source_uri: str
    source_class: SourceClass = SourceClass.TOOL_OUTPUT
    reversibility_outcome: float = 1.0
    executor_reliability: float = 1.0
    resource_measurements: Tensor | None = None
    resource_mask: Tensor | None = None
    replenishment: Tensor | None = None

    def __post_init__(self) -> None:
        if not self.receipt_id or self.sequence_number < 0 or self.schema_id < 0 or self.batch_index < 0:
            raise ValueError("executor result identity is invalid")
        if not self.source_uri or self.timestamp < 0:
            raise ValueError("executor result source and timestamp are invalid")
        if not 0 <= self.success <= 1 or not 0 <= self.constraint_violation <= 1:
            raise ValueError("executor success and constraint fields must lie in [0,1]")
        if not 0 <= self.reversibility_outcome <= 1 or not 0 <= self.executor_reliability <= 1:
            raise ValueError("executor reliability fields must lie in [0,1]")
        if self.latency_seconds < 0 or self.cost < 0:
            raise ValueError("executor latency and cost cannot be negative")
        if self.source_class in (SourceClass.EXTERNAL, SourceClass.BODILY):
            raise ValueError("executor-derived results require a derived source class")
        if self.observation is not None and self.observation.ndim != 1:
            raise ValueError("executor observation must be one feature vector")
        resource_values = (
            self.resource_measurements, self.resource_mask, self.replenishment
        )
        if any(value is not None for value in resource_values):
            if any(value is None for value in resource_values):
                raise ValueError("executor resource feedback must be complete")
            if self.resource_measurements.ndim != 1:
                raise ValueError("executor resource measurements must be one channel row")
            if self.resource_mask.shape != self.resource_measurements.shape or self.resource_mask.dtype != torch.bool:
                raise ValueError("executor resource mask is invalid")
            if self.replenishment.shape != self.resource_measurements.shape or bool((self.replenishment < 0).any()):
                raise ValueError("executor replenishment is invalid")


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    result: ExecutorResult
    action: StructuredExternalAction
    action_provenance_id: int

    def __post_init__(self) -> None:
        if self.action_provenance_id < 0:
            raise ValueError("execution receipt requires action provenance")
        if self.result.schema_id != self.action.schema_id or self.result.batch_index != self.action.batch_index:
            raise ValueError("execution result does not match dispatched action")


@dataclass(frozen=True, slots=True)
class SessionStepResult:
    output: MRCRAOutput | None
    state: MRCRARuntimeState
    ledger_digest: str
    applied_sequence_number: int


class StructuredEnvironmentExecutor(Protocol):
    def execute(self, action: StructuredExternalAction, sequence_number: int) -> ExecutorResult: ...


class CognitiveAgentSession:
    """Own persistent neural state and audit state, never host permissions."""

    def __init__(
        self, model: MultimodalRelationalContinuityResonanceNetwork, *,
        mode: AgentMode, action_registry: ActionSchemaRegistry,
        goals: GoalState | None = None, ledger: ProvenanceLedger | None = None,
        state: MRCRARuntimeState | None = None,
        viability: ViabilityState | None = None,
        environment_id: int = 0, session_id: int = 0,
    ) -> None:
        self.model = model
        if not model.config.cognitive.enable_agent_session_loop:
            raise ValueError("agent session loop is disabled by the checkpoint configuration")
        self.mode = AgentMode(mode)
        self.action_registry = action_registry
        self.ledger = ProvenanceLedger() if ledger is None else ledger
        if environment_id < 0 or session_id < 0:
            raise ValueError("session environment and session IDs cannot be negative")
        self.environment_id, self.session_id = environment_id, session_id
        if state is None:
            state = model.initial_state(1)
        if state.batch != 1:
            raise ValueError("the application session currently requires batch one")
        if self.mode != AgentMode.OFFLINE_MODELING:
            if goals is None or not bool(goals.mask.any()):
                raise ValueError("action-capable agent modes require an explicit authorized goal")
            if bool((goals.mask & (goals.authority <= 0)).any()):
                raise ValueError("active session goals require positive caller authority")
            if model.config.cognitive.enable_viability_gate and (
                viability is None and not bool(state.viability.active.any())
            ):
                raise ValueError("viability-gated agent modes require application authority state")
            # The session boundary is the authority that receives explicit
            # goals.  Register missing goal roots rather than letting a tensor
            # silently masquerade as provenance-backed intent.
            provenance_ids = goals.provenance_ids.clone()
            status = goals.status.clone()
            for slot in torch.nonzero(goals.mask[0], as_tuple=False).flatten().tolist():
                if int(provenance_ids[0, slot]) < 0:
                    provenance_ids[0, slot] = self.ledger.append(
                        source_class=SourceClass.EXTERNAL,
                        source_uri_or_episode=(
                            f"agent-session://{session_id}/goal/{slot}"
                        ),
                        support=SupportInterval(0.0, 0.0, 0.0),
                        modality=ModalityClass.GOAL,
                        operator="application:authorized_goal:v1",
                        scenario_id=0, model_authority="application",
                    )
                else:
                    self.ledger.get(int(provenance_ids[0, slot]))
                if int(status[0, slot]) == 0:
                    status[0, slot] = 1
            goals = replace(
                goals, provenance_ids=provenance_ids, status=status
            )
        goals = state.goals if goals is None else goals
        action_count = model.config.cognitive.system_action_channels
        registered = action_registry.availability_mask(1, action_count, device=state.nodes.content.device)
        if self.mode == AgentMode.OFFLINE_MODELING:
            registered.zero_()
        system = SystemModelState(
            state.system_model.modality_availability,
            registered.to(state.nodes.content.dtype),
            state.system_model.action_success, state.system_model.action_latency,
            state.system_model.action_reward, state.system_model.action_cost,
            state.system_model.action_constraint_violation,
            state.system_model.action_reversibility,
            state.system_model.executor_reliability,
            state.system_model.memory_reliability, state.system_model.router_reliability,
            state.system_model.remaining_compute, state.system_model.remaining_memory,
            registered, state.system_model.calibration_regime,
        )
        boundary = replace(
            state.boundary_context,
            environment_ids=torch.full_like(state.boundary_context.environment_ids, environment_id),
            session_ids=torch.full_like(state.boundary_context.session_ids, session_id),
        )
        viability = state.viability if viability is None else viability
        self.state = replace(
            state, goals=goals, system_model=system, boundary_context=boundary,
            viability=viability,
        )
        self._last_output: MRCRAOutput | None = None
        self._next_sequence = 0
        self._executed_receipt_ids: set[str] = set()
        self._ingested_receipt_ids: set[str] = set()
        self._pending: dict[str, ExecutionReceipt] = {}

    def observe(self, packets: ObservationPacket) -> SessionStepResult:
        if packets.batch != self.state.batch:
            raise ValueError("session packet batch differs from persistent state")
        output = self.model(packets, self.ledger, state=self.state)
        self.state = output.state
        self._last_output = output
        return SessionStepResult(
            output, self.state, self.ledger.digest(), self._next_sequence - 1,
        )

    def deliberate(self) -> DeliberationResult:
        if self._last_output is None:
            raise RuntimeError("session must observe before deliberating")
        decision = self._last_output.external_action
        actions: list[StructuredExternalAction] = []
        for row in torch.nonzero(decision.authorized, as_tuple=False).flatten().tolist():
            actions.append(self.action_registry.materialize(decision, batch_index=row))
        candidate_count = self.state.action_candidates.active.sum(-1)
        return DeliberationResult(
            tuple(actions), decision.abstained | ~decision.authorized,
            candidate_count, self.ledger.digest(),
        )

    def execute(
        self, executor: StructuredEnvironmentExecutor,
        deliberation: DeliberationResult | None = None,
    ) -> tuple[ExecutionReceipt, ...]:
        deliberation = self.deliberate() if deliberation is None else deliberation
        receipts: list[ExecutionReceipt] = []
        for action in deliberation.actions:
            result = executor.execute(action, self._next_sequence)
            if result.sequence_number != self._next_sequence:
                raise ValueError("executor result is out of order")
            if result.receipt_id in self._executed_receipt_ids:
                raise ValueError("duplicate executor receipt")
            if result.schema_id != action.schema_id or result.batch_index != action.batch_index:
                raise ValueError("executor result does not match dispatched action")
            support = SupportInterval(result.timestamp, result.timestamp, result.timestamp)
            provenance = self.ledger.derive(
                action.supporter_provenance_ids,
                source_class=SourceClass.COMMUNICATED,
                operator="mrcra:agent_session:executor_dispatch:v1",
                support=support, modality=ModalityClass.ACTION,
                scenario_id=0, model_authority=self.model.model_authority,
            )
            receipt = ExecutionReceipt(result, action, provenance)
            self._executed_receipt_ids.add(result.receipt_id)
            self._pending[result.receipt_id] = receipt
            receipts.append(receipt)
            self._next_sequence += 1
        return tuple(receipts)

    def ingest_result(self, receipt: ExecutionReceipt) -> SessionStepResult:
        result = receipt.result
        pending = self._pending.get(result.receipt_id)
        if pending is None or (
            pending.result.sequence_number != result.sequence_number
            or pending.result.schema_id != result.schema_id
            or pending.result.batch_index != result.batch_index
            or pending.action_provenance_id != receipt.action_provenance_id
        ):
            if result.receipt_id in self._ingested_receipt_ids:
                raise ValueError("executor receipt was already ingested")
            raise ValueError("executor receipt was not dispatched by this session")
        expected = len(self._ingested_receipt_ids)
        if result.sequence_number != expected:
            raise ValueError("executor receipts must be ingested in order")
        feedback_provenance = self.ledger.derive(
            [receipt.action_provenance_id], source_class=result.source_class,
            operator="mrcra:agent_session:executor_feedback:v1",
            support=SupportInterval(result.timestamp, result.timestamp, result.timestamp),
            modality=result.modality, scenario_id=0,
            model_authority=self.model.model_authority,
            source_reliability=result.executor_reliability,
        )
        feedback = ExternalActionFeedback(
            torch.tensor([result.schema_id], dtype=torch.int64, device=self.state.nodes.content.device),
            self.state.nodes.content.new_tensor([result.success]),
            self.state.nodes.content.new_tensor([result.latency_seconds]),
            self.state.nodes.content.new_tensor([result.reward]),
            self.state.nodes.content.new_tensor([result.cost]),
            self.state.nodes.content.new_tensor([result.constraint_violation]),
            torch.tensor([feedback_provenance], dtype=torch.int64, device=self.state.nodes.content.device),
            torch.tensor([True], dtype=torch.bool, device=self.state.nodes.content.device),
        )
        system = update_system_model_from_feedback(
            self.state.system_model, feedback, self.ledger, momentum=0.0,
        )
        reversibility = system.action_reversibility.clone()
        reliability = system.executor_reliability.clone()
        reversibility[0, result.schema_id] = result.reversibility_outcome
        reliability[0, result.schema_id] = result.executor_reliability
        system = replace(
            system, action_reversibility=reversibility,
            executor_reliability=reliability,
        )
        goals = self.state.goals
        if bool(goals.mask.any()):
            progress = goals.progress.clone()
            status = goals.status.clone()
            eligible = goals.mask & (goals.authority > 0) & (status == 1)
            score = (
                goals.priorities.clamp_min(0) * goals.authority.clamp_min(0)
                / goals.horizons.clamp_min(1e-8)
            ).masked_fill(~eligible, -torch.inf)
            if bool(eligible[0].any()):
                slot = int(score[0].argmax())
                progress[0, slot] = (
                    progress[0, slot]
                    + max(0.0, result.reward) * result.success
                ).clamp(0, 1)
                threshold = goals.termination[0, slot]
                if bool(threshold > 0) and bool(progress[0, slot] >= threshold):
                    status[0, slot] = 2
            goals = replace(goals, progress=progress, status=status)
        self.state = replace(self.state, system_model=system, goals=goals)
        if result.resource_measurements is not None:
            channels = self.state.viability.values.shape[-1]
            if result.resource_measurements.shape != (channels,):
                raise ValueError("executor resource channel count does not match viability state")
            measurement = result.resource_measurements.to(
                device=self.state.nodes.content.device,
                dtype=self.state.nodes.content.dtype,
            )[None]
            measurement_mask = result.resource_mask.to(
                device=self.state.nodes.content.device
            )[None]
            replenishment = result.replenishment.to(
                device=self.state.nodes.content.device,
                dtype=self.state.nodes.content.dtype,
            )[None]
            provenance = torch.full_like(
                self.state.viability.provenance_ids, feedback_provenance
            )
            viability = update_measured_viability(
                self.state.viability, measurements=measurement,
                measurement_mask=measurement_mask,
                provenance_ids=provenance, replenishment=replenishment,
            )
            self.state = replace(self.state, viability=viability)
        self._ingested_receipt_ids.add(result.receipt_id)
        del self._pending[result.receipt_id]
        if result.observation is None:
            return SessionStepResult(
                None, self.state, self.ledger.digest(), result.sequence_number,
            )
        value = result.observation.to(
            device=self.state.nodes.content.device, dtype=self.state.nodes.content.dtype
        ).view(1, 1, -1)
        if value.shape[-1] != self.model.config.carrier.input_dim:
            raise ValueError("executor observation width does not match model input")
        packet = register_internal_inputs(
            value, torch.ones(1, 1, dtype=torch.bool, device=value.device),
            parent_record_ids=torch.tensor(
                [[[feedback_provenance]]], dtype=torch.int64, device=value.device
            ),
            timestamps=value.new_tensor([[result.timestamp]]),
            coordinates=value.new_tensor([[[result.timestamp]]]),
            sample_intervals=value.new_ones(1),
            boundary_classes=torch.full(
                (1, 1), int(BoundaryClass.NONE), dtype=torch.int64, device=value.device
            ),
            modality_ids=torch.full(
                (1, 1), int(result.modality), dtype=torch.int64, device=value.device
            ),
            uncertainty_seed=value.new_zeros(
                1, 1, self.model.config.cognitive.uncertainty_channels
            ),
            segment_ids=torch.zeros(1, 1, dtype=torch.int64, device=value.device),
            ledger=self.ledger, source_class=result.source_class,
            operator="mrcra:agent_session:feedback_observation:v1",
            scenario_ids=torch.zeros(1, 1, dtype=torch.int64, device=value.device),
            model_authority=self.model.model_authority,
        )
        return self.observe(packet)

    def transition_evidence_request(
        self, request_index: int, status: EvidenceRequestStatus,
    ) -> None:
        """Apply an application-owned request lifecycle transition."""

        device = self.state.nodes.content.device
        requests = transition_evidence_requests(
            self.state.evidence_requests,
            torch.tensor([request_index], dtype=torch.int64, device=device),
            torch.tensor([int(status)], dtype=torch.int64, device=device),
            torch.tensor([True], dtype=torch.bool, device=device),
        )
        self.state = replace(self.state, evidence_requests=requests)

    def record_artifact(
        self, *, artifact_id: int, content_digest: bytes, version: int,
        creator_action_id: int, parent_provenance_ids: tuple[int, ...],
        expected_persistence: float, estimated_cost: float, timestamp: float,
        readable: bool = True, writable: bool = True,
    ) -> int:
        """Record an artifact already created by the host; never create it."""

        width = self.state.external_artifacts.content_digests.shape[-1]
        if len(content_digest) != width or not parent_provenance_ids:
            raise ValueError("artifact digest width and action provenance are required")
        device, dtype = (
            self.state.nodes.content.device, self.state.nodes.content.dtype
        )
        parents = torch.tensor(
            [parent_provenance_ids], dtype=torch.int64, device=device
        )
        artifacts, written = record_external_artifact(
            self.state.external_artifacts, self.ledger,
            artifact_ids=torch.tensor([artifact_id], dtype=torch.int64, device=device),
            content_digests=torch.tensor(
                [list(content_digest)], dtype=torch.uint8, device=device
            ),
            versions=torch.tensor([version], dtype=torch.int64, device=device),
            creator_action_ids=torch.tensor(
                [creator_action_id], dtype=torch.int64, device=device
            ),
            parent_provenance_ids=parents,
            parent_mask=torch.ones_like(parents, dtype=torch.bool),
            expected_persistence=torch.tensor(
                [expected_persistence], dtype=dtype, device=device
            ),
            estimated_cost=torch.tensor([estimated_cost], dtype=dtype, device=device),
            timestamp=torch.tensor([timestamp], dtype=dtype, device=device),
            readable=torch.tensor([readable], dtype=torch.bool, device=device),
            writable=torch.tensor([writable], dtype=torch.bool, device=device),
            create_mask=torch.tensor([True], dtype=torch.bool, device=device),
            model_authority=self.model.model_authority,
        )
        if int(written[0]) < 0:
            raise RuntimeError("external artifact live-state capacity is exhausted")
        self.state = replace(self.state, external_artifacts=artifacts)
        return int(written[0])

    def checkpoint(self, path: str | Path) -> None:
        pending = {
            receipt_id: {
                "result": {
                    **asdict(receipt.result),
                    "observation": None if receipt.result.observation is None
                    else receipt.result.observation.detach().cpu().tolist(),
                    "modality": int(receipt.result.modality),
                    "source_class": int(receipt.result.source_class),
                    "resource_measurements": None
                    if receipt.result.resource_measurements is None
                    else receipt.result.resource_measurements.detach().cpu().tolist(),
                    "resource_mask": None
                    if receipt.result.resource_mask is None
                    else receipt.result.resource_mask.detach().cpu().tolist(),
                    "replenishment": None
                    if receipt.result.replenishment is None
                    else receipt.result.replenishment.detach().cpu().tolist(),
                },
                "action": asdict(receipt.action),
                "action_provenance_id": receipt.action_provenance_id,
            }
            for receipt_id, receipt in self._pending.items()
        }
        save_mrcra_checkpoint(
            path, self.model, self.state, self.ledger,
            metadata={
                "agent_session": {
                    "mode": int(self.mode), "environment_id": self.environment_id,
                    "session_id": self.session_id, "next_sequence": self._next_sequence,
                    "executed_receipt_ids": sorted(self._executed_receipt_ids),
                    "ingested_receipt_ids": sorted(self._ingested_receipt_ids),
                    "pending": pending,
                    "registered_schema_ids": list(self.action_registry.schema_ids),
                }
            },
        )

    @classmethod
    def resume(
        cls, path: str | Path,
        model: MultimodalRelationalContinuityResonanceNetwork,
        action_registry: ActionSchemaRegistry,
    ) -> "CognitiveAgentSession":
        state, ledger, metadata = load_mrcra_checkpoint(path, model)
        session_data = metadata.get("agent_session")
        if not isinstance(session_data, dict):
            raise ValueError("checkpoint does not contain agent-session metadata")
        if tuple(session_data.get("registered_schema_ids", ())) != action_registry.schema_ids:
            raise ValueError("checkpoint action registry differs from host registry")
        session = cls(
            model, mode=AgentMode(session_data["mode"]),
            action_registry=action_registry, goals=state.goals, ledger=ledger,
            state=state, environment_id=session_data["environment_id"],
            session_id=session_data["session_id"],
        )
        session._next_sequence = int(session_data["next_sequence"])
        session._executed_receipt_ids = set(session_data["executed_receipt_ids"])
        session._ingested_receipt_ids = set(session_data["ingested_receipt_ids"])
        for receipt_id, raw in session_data["pending"].items():
            raw_result = dict(raw["result"])
            raw_result["observation"] = (
                None if raw_result["observation"] is None
                else state.nodes.content.new_tensor(raw_result["observation"])
            )
            raw_result["modality"] = ModalityClass(raw_result["modality"])
            raw_result["source_class"] = SourceClass(raw_result["source_class"])
            if raw_result["resource_measurements"] is not None:
                raw_result["resource_measurements"] = state.nodes.content.new_tensor(
                    raw_result["resource_measurements"]
                )
                raw_result["resource_mask"] = torch.tensor(
                    raw_result["resource_mask"], dtype=torch.bool,
                    device=state.nodes.content.device,
                )
                raw_result["replenishment"] = state.nodes.content.new_tensor(
                    raw_result["replenishment"]
                )
            result = ExecutorResult(**raw_result)
            action = StructuredExternalAction(**raw["action"])
            session._pending[receipt_id] = ExecutionReceipt(
                result, action, int(raw["action_provenance_id"])
            )
        return session
