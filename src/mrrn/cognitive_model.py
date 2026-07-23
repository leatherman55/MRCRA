"""End-to-end Multimodal Relational-Continuity Resonance Architecture."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from enum import IntEnum

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cognitive_types import (
    BoundaryClass, BoundaryScope, CognitiveClocks, CognitiveTrigger, InternalAction, ModalityClass, NodeSlots,
    NodeType, RELATION_COMPATIBILITY, RelationFamily, RelationSlots, SourceClass,
    SupportInterval, VerificationClass,
)
from .abstraction_control import (
    AbstractionApplicabilityHead, AbstractionLevelSelector,
    AbstractionValidityState, LocalizedDescentPlanner,
)
from .action_candidates import (
    ActionCandidateState, authorize_candidate_provenance,
    build_action_candidates, evaluate_candidate_rollout,
    select_action_candidate,
)
from .boundaries import BoundaryContextState, legacy_scope
from .config import MRCRAConfig
from .runtime_validation import defer_runtime_validation, validate_dataclass_tree
from .compression import GraphCompressor, GraphFragment
from .controller import (
    AdaptiveController, ControllerState, GoalState, OperationalSchemaState,
    OperationalSchemas, SystemModelState,
)
from .events import (
    CausalEventExtractor, EventCandidates, EventEvidence, EventExtractorState,
    EventProposalNetwork, EventTransitionReceipts, PersistentEventAllocator,
)
from .evidence_requests import EvidenceRequestState, create_evidence_request
from .external_artifacts import ExternalArtifactState
from .hypotheses import HypothesisBank, HypothesisState
from .interaction import (
    ExternalActionDecision, ExternalActionFeedback, ExternalActionPolicy,
    authorize_external_actions, decision_from_candidates,
    update_system_model_from_feedback,
)
from .invariants import IntegratedInvariantDiscoverer, SymbolActivator
from .knowledge import (
    KnowledgeKind, KnowledgeProposalBank, KnowledgeProposalBatch,
    KnowledgeProposalState, KnowledgeStatus, KnowledgeValidationBatch,
    KnowledgeValidationResult,
)
from .memory_v2 import (
    BatchedTensorMemory, MemoryQuery, MemoryTier, MemoryWriteBatch,
    MemoryWriteEvidence, MemoryWritePolicyV2, TensorMemoryState,
)
from .metacognition import (
    MetacognitivePrediction, MetacognitiveRouter, MetacognitiveState,
    append_metacognitive_record,
)
from .model import MRRN, MRRNStreamState
from .observation import ObservationPacket, register_internal_inputs
from .provenance import ProvenanceLedger
from .reconstruction import (
    ConditionalGraphReconstructor, ReconstructionEvidence, ReconstructionQuery,
    ReconstructionState,
)
from .uncertainty import (
    CalibrationReport, CalibrationState, DistributionalOutput,
    DistributionalPredictionHead, OnlineCalibration, UncertaintyEstimator,
    UncertaintyInputs,
)
from .world_model import ActionConditionedWorldModel, WorldModelPrediction
from .viability import CandidateViabilityForecaster, ViabilityGate, ViabilityState
from .workspace import (
    GlobalWorkspace, GlobalWorkspaceState, RelationProposals, RelationSlotWriter,
    WorkspaceBroadcast, WorkspaceGraph, WorkspaceGraphOutput,
    invalidate_stale_relations,
)
from .relational_router import NodeCandidateBuilder, RelationalResonanceRouter


@dataclass(frozen=True, slots=True)
class MRCRARuntimeState:
    carrier: tuple[MRRNStreamState, ...]
    event_extractor: EventExtractorState
    nodes: NodeSlots
    relations: RelationSlots
    workspace: GlobalWorkspaceState
    hypotheses: HypothesisState
    episodic_memory: TensorMemoryState
    semantic_memory: TensorMemoryState
    controller: ControllerState
    goals: GoalState
    system_model: SystemModelState
    schemas: OperationalSchemaState
    calibration: CalibrationState
    knowledge: KnowledgeProposalState
    last_external_action: ExternalActionDecision
    clocks: CognitiveClocks
    previous_latent: Tensor
    predicted_next_latent: Tensor
    relational_context: Tensor
    selected_physical_scale: Tensor
    reconstructions: ReconstructionState
    abstraction_validity: AbstractionValidityState
    action_candidates: ActionCandidateState
    viability: ViabilityState
    evidence_requests: EvidenceRequestState
    external_artifacts: ExternalArtifactState
    metacognition: MetacognitiveState
    boundary_context: BoundaryContextState

    @property
    def batch(self) -> int:
        return self.nodes.batch

    def detach(self) -> "MRCRARuntimeState":
        return MRCRARuntimeState(
            tuple(item.detach() for item in self.carrier), self.event_extractor.detach(),
            self.nodes.detach(), self.relations.detach(), self.workspace.detach(),
            self.hypotheses.detach(), self.episodic_memory.detach(),
            self.semantic_memory.detach(), self.controller.detach(),
            self.goals.detach(), self.system_model.detach(), self.schemas.detach(),
            self.calibration.detach(), self.knowledge.detach(),
            self.last_external_action.detach(),
            self.clocks,
            self.previous_latent.detach(), self.predicted_next_latent.detach(),
            self.relational_context.detach(), self.selected_physical_scale.detach(),
            self.reconstructions.detach(), self.abstraction_validity.detach(),
            self.action_candidates.detach(), self.viability.detach(),
            self.evidence_requests.detach(), self.external_artifacts.detach(),
            self.metacognition.detach(), self.boundary_context.detach(),
        )


@dataclass(frozen=True, slots=True)
class MRCRAStepOutput:
    prediction: Tensor
    latent: Tensor
    output_latent: Tensor
    cognitive_features: Tensor
    workspace_features: Tensor
    relation_features: Tensor
    relation_type_probabilities: Tensor
    uncertainty: Tensor
    events: EventCandidates
    graph: WorkspaceGraphOutput | None
    state: MRCRARuntimeState
    cognitive_cycle_mask: Tensor
    action_receipts: CognitiveActionReceipts
    external_action: ExternalActionDecision
    world_prediction: WorldModelPrediction | None
    distributional_prediction: DistributionalOutput
    schema_probabilities: Tensor
    symbol_gates: Tensor
    event_proposal_logits: Tensor
    event_end_logits: Tensor
    event_type_logits: Tensor
    event_soft_content: Tensor
    event_transition_receipts: EventTransitionReceipts
    predicted_next_latent: Tensor
    provenance_source_logits: Tensor
    provenance_verification_logits: Tensor
    metacognitive_values: Tensor
    metacognitive_mask: Tensor


class ActionStatus(IntEnum):
    SUCCESS = 0
    HALTED = 1
    NO_TARGET = 2
    INCOMPATIBLE = 3
    EMPTY_MEMORY = 4
    EXTERNAL_EVIDENCE_REQUIRED = 5
    VALIDATION_REQUIRED = 6
    CAPACITY_BLOCKED = 7


@dataclass(frozen=True, slots=True)
class CognitiveActionReceipts:
    actions: Tensor
    statuses: Tensor
    success: Tensor
    node_pointers: Tensor
    relation_pointers: Tensor
    knowledge_pointers: Tensor
    mask: Tensor
    action_logits: Tensor
    relation_logits: Tensor
    halt_probability: Tensor
    secondary_node_pointers: Tensor
    secondary_relation_pointers: Tensor
    argument_schema_ids: Tensor
    arguments: Tensor
    argument_mask: Tensor
    trigger_classes: Tensor
    expected_operation_cost: Tensor


@dataclass(frozen=True, slots=True)
class MRCRAOutput:
    prediction: Tensor
    latent: Tensor
    output_latent: Tensor
    cognitive_features: Tensor
    workspace_features: Tensor
    relation_features: Tensor
    relation_type_probabilities: Tensor
    uncertainty: Tensor
    event_counts: Tensor
    cognitive_cycles: Tensor
    nodes: NodeSlots
    relations: RelationSlots
    workspace: GlobalWorkspaceState
    hypotheses: HypothesisState
    state: MRCRARuntimeState
    abstained: Tensor
    provenance_digest: str
    action_receipts: CognitiveActionReceipts
    external_action: ExternalActionDecision
    world_prediction: WorldModelPrediction | None
    distributional_prediction: DistributionalOutput | None
    schema_probabilities: Tensor
    symbol_gates: Tensor
    knowledge: KnowledgeProposalState
    calibration: CalibrationReport
    event_proposal_logits: Tensor
    event_end_logits: Tensor
    event_type_logits: Tensor
    predicted_next_latent: Tensor
    provenance_source_logits: Tensor
    provenance_verification_logits: Tensor
    metacognitive_values: Tensor
    metacognitive_mask: Tensor


@dataclass(frozen=True, slots=True)
class MRCRAIntegratedTrainingState:
    """Bounded dense-carrier and event-cognition state for language training."""

    carrier: MRRNStreamState
    cognitive: MRCRARuntimeState
    feedback: Tensor

    def detach(self) -> "MRCRAIntegratedTrainingState":
        return MRCRAIntegratedTrainingState(
            self.carrier.detach(), self.cognitive.detach(), self.feedback.detach()
        )


@dataclass(frozen=True, slots=True)
class MRCRAIntegratedTrainingOutput:
    """Minimal differentiable result of the multirate integrated training path."""

    output_latent: Tensor
    state: MRCRAIntegratedTrainingState
    cognitive_cycles: Tensor
    event_counts: Tensor
    feedback_rms: Tensor
    event_activation_mean: Tensor
    active_nodes_mean: Tensor
    active_nodes_max: Tensor
    event_proposal_logits: Tensor
    event_end_logits: Tensor
    event_opened: Tensor
    event_finalized: Tensor
    event_emitted: Tensor
    event_quota_rejected: Tensor
    event_open_after: Tensor
    first_hard_event: "HardEventTrace | None"


@dataclass(frozen=True, slots=True)
class HardEventTrace:
    """Small, exact receipt for the first emitted hard event in a span."""

    anchor_index: int
    timestamp: Tensor
    proposal_logit: Tensor
    end_logit: Tensor
    event_type: Tensor
    confidence: Tensor
    support: Tensor
    active_nodes_before: Tensor
    active_nodes_after: Tensor
    active_relations_before: Tensor
    active_relations_after: Tensor
    workspace_before: Tensor
    workspace_after: Tensor


def _replace_rows(current, fresh, reset: Tensor):
    values = {}
    for field in fields(current):
        value = getattr(current, field.name)
        replacement = getattr(fresh, field.name)
        if isinstance(value, Tensor) and value.ndim and value.shape[0] == reset.shape[0]:
            shape = (reset.shape[0],) + (1,) * (value.ndim - 1)
            values[field.name] = torch.where(reset.view(shape), replacement, value)
        else:
            values[field.name] = value
    return type(current)(**values)


def _mask_relation_proposals(proposals: RelationProposals, mask: Tensor) -> RelationProposals:
    if mask.shape != proposals.active.shape:
        raise ValueError("relation proposal mask must match proposal rows")
    active = proposals.active & mask
    return RelationProposals(
        proposals.content * active.unsqueeze(-1),
        proposals.family_ids.masked_fill(~active, -1),
        proposals.participant_indices.masked_fill(~active.unsqueeze(-1), -1),
        proposals.participant_roles,
        proposals.participant_mask & active.unsqueeze(-1),
        proposals.support * active.unsqueeze(-1),
        proposals.confidence * active,
        proposals.parent_provenance_ids.masked_fill(~active.unsqueeze(-1), -1),
        proposals.provenance_ids.masked_fill(~active, -1),
        proposals.scenario_ids.masked_fill(~active, -1),
        active,
    )


class MultimodalRelationalContinuityResonanceNetwork(nn.Module):
    """MRRN carrier plus bounded event-driven relational cognition.

    Batch elements hold independent carrier streams.  This is intentional:
    document/episode boundaries may be asynchronous, and a shared binary-lifting
    carry cannot represent different reset phases without cross-document leaks.
    The serious 32K training profile uses microbatch one, so the correctness
    choice has no throughput cost on its target hardware.
    """

    def __init__(self, config: MRCRAConfig, *, model_authority: str = "mrcra-untrained") -> None:
        super().__init__()
        if not model_authority:
            raise ValueError("MRCRA requires a nonempty model authority identifier")
        self.config = config
        self.model_authority = model_authority
        carrier_config, cognitive = config.carrier, config.cognitive
        self.carrier = MRRN(carrier_config)
        width = cognitive.workspace_dim
        self.uncertainty_head = nn.Linear(width, cognitive.uncertainty_channels)
        self.next_latent_predictor = nn.Linear(width, width)
        self.provenance_source_head = nn.Linear(width, len(SourceClass))
        self.provenance_verification_head = nn.Linear(width, len(VerificationClass))
        self.spectral_projection = nn.Linear(
            width, cognitive.relation_heads * cognitive.relation_modes * 2
        )
        proposal = EventProposalNetwork(
            width, cognitive.uncertainty_channels, width, cognitive.node_type_count,
        )
        self.event_extractor = CausalEventExtractor(
            proposal, chunk_size=cognitive.event_chunk_size,
            proposals_per_chunk=cognitive.event_proposals_per_chunk,
            maximum_span=cognitive.event_chunk_size,
        )
        self.event_allocator = PersistentEventAllocator(width)
        candidate_builder = NodeCandidateBuilder(
            width, cognitive.maximum_router_candidates, router_dim=min(64, width)
        )
        router = RelationalResonanceRouter(
            width, cognitive.relation_heads, cognitive.relation_modes,
            cognitive.relation_family_count, cognitive.uncertainty_channels,
            cognitive.provenance_features, adapter_rank=cognitive.relation_adapter_rank,
            retained_edges=cognitive.graph_neighbors,
        )
        global_workspace = GlobalWorkspace(
            width, cognitive.global_workspace_slots,
            update_maximum=cognitive.workspace_update_maximum,
        )
        broadcast = WorkspaceBroadcast(
            width, tuple(item.width for item in carrier_config.scale_configs()),
            maximum_gain=cognitive.broadcast_gain_maximum,
        )
        self.workspace_graph = WorkspaceGraph(
            candidate_builder, router, global_workspace, broadcast,
        )
        self.relation_writer = RelationSlotWriter(
            cognitive.relation_family_count, cognitive.uncertainty_channels,
            pair_capacity=cognitive.pair_edge_capacity,
            hyperedge_capacity=cognitive.hyperedge_capacity,
        )
        signature_dim = max(16, width // 4)
        self.memory_key = nn.Linear(width, width)
        self.memory_signature = nn.Linear(width, signature_dim)
        self.memory_value = nn.Linear(width, width)
        self.memory = BatchedTensorMemory(
            width, signature_dim, cognitive.relation_heads, cognitive.relation_modes,
            route_candidates=max(
                cognitive.episodic_candidates, cognitive.semantic_candidates, 4
            ) * 4,
            retrieved_items=max(cognitive.episodic_candidates, cognitive.semantic_candidates),
        )
        self.memory_write_policy = MemoryWritePolicyV2()
        self.hypothesis_bank = HypothesisBank(
            width, cognitive.hypothesis_slots, cognitive.hyperedge_capacity,
            cognitive.relation_family_count, width, cognitive.uncertainty_channels,
        )
        self.world_model = ActionConditionedWorldModel(
            width, cognitive.system_action_channels,
            cognitive.relation_family_count, width,
            horizons=cognitive.world_model_horizons,
        )
        self.reconstructor = ConditionalGraphReconstructor(
            width, cognitive.node_type_count, cognitive.relation_family_count,
            cognitive.event_proposals_per_chunk,
            cognitive.event_proposals_per_chunk, arity=2,
        )
        self.abstraction_applicability = AbstractionApplicabilityHead(width)
        self.abstraction_selector = AbstractionLevelSelector()
        self.localized_descent_planner = LocalizedDescentPlanner()
        self.viability_forecaster = CandidateViabilityForecaster(
            cognitive.viability_channels
        )
        self.viability_gate = ViabilityGate()
        system_feature_dim = (
            cognitive.modality_count + 8 * cognitive.system_action_channels
            + 4 + cognitive.calibration_regimes
        )
        self.controller = AdaptiveController(
            width, width, cognitive.uncertainty_channels, system_feature_dim,
            cognitive.active_event_capacity,
            maximum_steps=cognitive.maximum_cognitive_steps,
            horizon_count=len(cognitive.world_model_horizons),
            maximum_relations=(
                cognitive.pair_edge_capacity + cognitive.hyperedge_capacity
            ),
            action_argument_dim=cognitive.action_argument_dim,
        )
        self.metacognitive_router = MetacognitiveRouter(
            width, system_feature_dim, cognitive.uncertainty_channels,
        )
        self.self_model_projection = nn.Linear(system_feature_dim, width)
        self.operational_schemas = OperationalSchemas(
            width, cognitive.operational_schema_count,
        )
        self.symbol_activator = SymbolActivator(
            cognitive.semantic_memory_capacity, width, width,
        )
        self.invariant_discoverer = IntegratedInvariantDiscoverer(
            width, cognitive.node_type_count,
            role_count=max(4, cognitive.maximum_hyperedge_arity * 2),
            relation_families=cognitive.relation_family_count,
            maximum_nodes=cognitive.active_event_capacity,
        )
        self.distributional_head = DistributionalPredictionHead(
            width, cognitive.relation_family_count, 4, ensemble_heads=4,
        )
        self.uncertainty_estimator = UncertaintyEstimator(
            cognitive.uncertainty_channels
        )
        self.online_calibration = OnlineCalibration(
            cognitive.calibration_regimes, cognitive.calibration_bins,
        )
        self.external_action_policy = ExternalActionPolicy(
            width, width, cognitive.uncertainty_channels, system_feature_dim,
            cognitive.system_action_channels,
        )
        self.knowledge_bank = KnowledgeProposalBank()
        self.output_context_adapter = nn.Linear(width, carrier_config.model_dim, bias=False)
        self.cognitive_state_projection = nn.Linear(
            4 * width + cognitive.uncertainty_channels, width
        )
        # A small nonzero coupling lets environmental/task loss reach cognition
        # from the first update while keeping the initially untrained broadcast
        # several orders of magnitude below the carrier residual stream.
        nn.init.normal_(self.output_context_adapter.weight, std=1e-3)
        self.compare_projection = nn.Linear(4 * width, width)
        self.graph_compressor = GraphCompressor(
            width, width, cognitive.node_type_count,
            cognitive.relation_family_count, cognitive.active_event_capacity,
            cognitive.pair_edge_capacity + cognitive.hyperedge_capacity,
        )
        self.relation_write_threshold = 1e-3

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def validate_serious_parameter_count(self) -> int:
        count = self.parameter_count
        self.config.require_actor_parameter_count(count)
        return count

    def initial_state(
        self, batch: int, *, sample_intervals: Tensor | None = None,
        device=None, dtype=None,
    ) -> MRCRARuntimeState:
        if batch <= 0:
            raise ValueError("MRCRA batch must be positive")
        cognitive = self.config.cognitive
        if sample_intervals is None:
            sample_intervals = torch.ones(batch, device=device, dtype=dtype)
        if sample_intervals.shape != (batch,) or bool((sample_intervals <= 0).any()):
            raise ValueError("one positive sample interval is required per stream")
        carrier = tuple(
            self.carrier.initial_stream_state(
                1, sample_interval=float(sample_intervals[index].item()),
                device=device, dtype=dtype,
            )
            for index in range(batch)
        )
        nodes = NodeSlots.empty(
            batch, cognitive.active_event_capacity, cognitive.workspace_dim,
            heads=cognitive.relation_heads, modes=cognitive.relation_modes,
            node_types=cognitive.node_type_count, modalities=cognitive.modality_count,
            uncertainty_channels=cognitive.uncertainty_channels,
            provenance_features=cognitive.provenance_features,
            hypotheses=cognitive.hypothesis_slots, device=device, dtype=dtype,
        )
        relations = RelationSlots.empty(
            batch, cognitive.pair_edge_capacity + cognitive.hyperedge_capacity,
            cognitive.workspace_dim,
            relation_families=cognitive.relation_family_count,
            arity=cognitive.maximum_hyperedge_arity,
            uncertainty_channels=cognitive.uncertainty_channels,
            hypotheses=cognitive.hypothesis_slots, device=device, dtype=dtype,
        )
        memory_kwargs = dict(
            batch=batch, key_dim=cognitive.workspace_dim,
            value_dim=cognitive.workspace_dim,
            signature_dim=max(16, cognitive.workspace_dim // 4),
            heads=cognitive.relation_heads, modes=cognitive.relation_modes,
            uncertainty_channels=cognitive.uncertainty_channels,
            consequence_dim=4, association_degree=cognitive.associative_budget,
            device=device, dtype=dtype,
        )
        goals = self.default_goals(batch, device=device, dtype=dtype)
        system_model = self.default_system_model(batch, device=device, dtype=dtype)
        schemas = self.operational_schemas.initial_state(
            batch, device=device, dtype=dtype
        )
        calibration = self.online_calibration.initial_state(
            device=device, dtype=dtype
        )
        knowledge = KnowledgeProposalState.empty(
            batch, cognitive.knowledge_candidate_capacity, cognitive.workspace_dim,
            cognitive.knowledge_support_capacity, device=device, dtype=dtype,
        )
        last_external_action = ExternalActionDecision.empty(
            batch, cognitive.system_action_channels,
            cognitive.knowledge_support_capacity, device=device, dtype=dtype,
        )
        reconstructions = ReconstructionState.empty(
            batch, cognitive.reconstruction_capacity, cognitive.workspace_dim,
            cognitive.uncertainty_channels, device=device, dtype=dtype,
        )
        abstraction_validity = AbstractionValidityState.empty(
            batch, cognitive.knowledge_candidate_capacity, device=device, dtype=dtype,
        )
        action_candidates = ActionCandidateState.empty(
            batch, cognitive.action_candidate_capacity, cognitive.action_argument_dim,
            cognitive.knowledge_support_capacity, device=device, dtype=dtype,
        )
        viability = ViabilityState.empty(
            batch, cognitive.viability_channels, device=device, dtype=dtype,
        )
        evidence_requests = EvidenceRequestState.empty(
            batch, cognitive.evidence_request_capacity, cognitive.workspace_dim,
            cognitive.hypothesis_slots, cognitive.knowledge_support_capacity,
            device=device, dtype=dtype,
        )
        external_artifacts = ExternalArtifactState.empty(
            batch, cognitive.external_artifact_capacity,
            cognitive.external_artifact_digest_width, device=device, dtype=dtype,
        )
        metacognition = MetacognitiveState.empty(
            batch, cognitive.metacognitive_capacity, device=device, dtype=dtype,
        )
        boundary_context = BoundaryContextState.empty(batch, device=device)
        return MRCRARuntimeState(
            carrier,
            self.event_extractor.initial_state(batch, device=device, dtype=dtype),
            nodes, relations,
            self.workspace_graph.workspace.initial_state(batch, device=device, dtype=dtype),
            self.hypothesis_bank.initial_state(batch, device=device, dtype=dtype),
            TensorMemoryState.empty(capacity=cognitive.episodic_memory_capacity, **memory_kwargs),
            TensorMemoryState.empty(capacity=cognitive.semantic_memory_capacity, **memory_kwargs),
            self.controller.initial_state(batch, device=device, dtype=dtype),
            goals, system_model, schemas, calibration, knowledge,
            last_external_action,
            CognitiveClocks(0, 0, 0),
            torch.zeros(batch, cognitive.workspace_dim, device=device, dtype=dtype),
            torch.zeros(batch, cognitive.workspace_dim, device=device, dtype=dtype),
            torch.zeros(batch, cognitive.workspace_dim, device=device, dtype=dtype),
            torch.zeros(batch, dtype=torch.int64, device=device),
            reconstructions, abstraction_validity, action_candidates, viability,
            evidence_requests, external_artifacts, metacognition, boundary_context,
        )

    def default_goals(self, batch: int, *, device, dtype) -> GoalState:
        cognitive = self.config.cognitive
        base = (batch, cognitive.goal_slots)
        return GoalState(
            torch.zeros(*base, cognitive.workspace_dim, device=device, dtype=dtype),
            torch.zeros(*base, cognitive.goal_constraint_dim, device=device, dtype=dtype),
            torch.zeros(base, device=device, dtype=dtype),
            torch.ones(base, device=device, dtype=dtype),
            torch.zeros(base, device=device, dtype=dtype),
            torch.zeros(base, device=device, dtype=dtype),
            torch.zeros(base, dtype=torch.bool, device=device),
        )

    def default_system_model(self, batch: int, *, device, dtype) -> SystemModelState:
        cognitive = self.config.cognitive
        actions = cognitive.system_action_channels
        return SystemModelState(
            torch.ones(batch, cognitive.modality_count, device=device, dtype=dtype),
            # A raw neural runtime has no host capabilities or permissions.
            # CognitiveAgentSession replaces these rows from an explicit,
            # application-owned ActionSchemaRegistry.
            torch.zeros(batch, actions, device=device, dtype=dtype),
            torch.zeros(batch, actions, device=device, dtype=dtype),
            torch.zeros(batch, actions, device=device, dtype=dtype),
            torch.zeros(batch, actions, device=device, dtype=dtype),
            torch.zeros(batch, actions, device=device, dtype=dtype),
            torch.zeros(batch, actions, device=device, dtype=dtype),
            torch.zeros(batch, actions, device=device, dtype=dtype),
            torch.zeros(batch, actions, device=device, dtype=dtype),
            torch.ones(batch, 1, device=device, dtype=dtype),
            torch.ones(batch, 1, device=device, dtype=dtype),
            torch.ones(batch, 1, device=device, dtype=dtype),
            torch.ones(batch, 1, device=device, dtype=dtype),
            torch.zeros(batch, actions, dtype=torch.bool, device=device),
            torch.zeros(batch, cognitive.calibration_regimes, device=device, dtype=dtype),
        )

    def apply_external_feedback(
        self, state: MRCRARuntimeState, feedback: ExternalActionFeedback,
        ledger: ProvenanceLedger, *, momentum: float = 0.95,
    ) -> MRCRARuntimeState:
        """Incorporate measured environment outcomes into the persistent self-model."""

        updated = update_system_model_from_feedback(
            state.system_model, feedback, ledger, momentum=momentum,
        )
        return replace(state, system_model=updated)

    def update_calibration(
        self, state: MRCRARuntimeState, probabilities: Tensor, targets: Tensor,
        group_ids: Tensor, mask: Tensor,
    ) -> MRCRARuntimeState:
        """Apply externally scored predictions to the persistent calibration ledger."""

        calibration = self.online_calibration.update(
            state.calibration, probabilities, targets, group_ids, mask,
        )
        return replace(state, calibration=calibration)

    def validate_and_promote_knowledge(
        self, state: MRCRARuntimeState, evidence: KnowledgeValidationBatch,
        ledger: ProvenanceLedger, *, minimum_code_gain_bits: float | None = None,
        maximum_reconstruction_distortion: float | None = None,
        maximum_relation_distortion: float | None = None,
        held_out_tolerance: float = 0.0,
    ) -> tuple[MRCRARuntimeState, KnowledgeValidationResult]:
        """Validate pending knowledge and atomically promote accepted rows.

        Promotion writes both a typed active node and semantic-memory record.
        Failed candidates remain visible as rejected evidence and are never
        silently converted into an invariant or abstraction.
        """

        minimum_code_gain_bits = (
            self.config.cognitive.abstraction_minimum_gain
            if minimum_code_gain_bits is None else minimum_code_gain_bits
        )
        maximum_reconstruction_distortion = (
            self.config.cognitive.abstraction_maximum_distortion
            if maximum_reconstruction_distortion is None
            else maximum_reconstruction_distortion
        )
        maximum_relation_distortion = (
            self.config.cognitive.abstraction_maximum_distortion
            if maximum_relation_distortion is None else maximum_relation_distortion
        )
        knowledge, result = self.knowledge_bank.validate(
            state.knowledge, evidence, ledger,
            minimum_code_gain_bits=minimum_code_gain_bits,
            maximum_reconstruction_distortion=maximum_reconstruction_distortion,
            maximum_relation_distortion=maximum_relation_distortion,
            held_out_tolerance=held_out_tolerance,
        )
        batch = state.batch
        indices = evidence.proposal_indices.clamp(0, knowledge.capacity - 1)
        rows = torch.arange(batch, device=state.nodes.content.device)
        latent = knowledge.latent[rows, indices]
        kind = knowledge.kind[rows, indices]
        provenance = knowledge.provenance_ids[rows, indices]
        supporters = knowledge.supporting_provenance_ids[rows, indices]
        supporter_mask = knowledge.supporting_mask[rows, indices]
        promoted = result.accepted & evidence.mask
        for row in torch.nonzero(promoted, as_tuple=False).flatten().tolist():
            ledger.set_verification(
                int(provenance[row]), VerificationClass.INTERNALLY_CONSISTENT,
                authority=self.model_authority,
                reason="held-out knowledge validation and counterexample gate passed",
            )
        node_type = torch.where(
            kind == int(KnowledgeKind.INVARIANT),
            torch.full_like(kind, int(NodeType.INVARIANT)),
            torch.where(
                kind == int(KnowledgeKind.SYMBOL),
                torch.full_like(kind, int(NodeType.SYMBOL)),
                torch.full_like(kind, int(NodeType.ABSTRACTION)),
            ),
        )
        support = latent.new_zeros(batch, 1, 3)
        scenario = torch.zeros(batch, 1, dtype=torch.int64, device=latent.device)
        for row in torch.nonzero(promoted, as_tuple=False).flatten().tolist():
            record = ledger.get(int(provenance[row]))
            support[row, 0] = latent.new_tensor((
                record.support.start, record.support.end,
                record.support.completion_time,
            ))
            scenario[row, 0] = record.scenario_id
        spectral = self.spectral_projection(latent).reshape(
            batch, self.config.cognitive.relation_heads,
            self.config.cognitive.relation_modes, 2,
        )
        spectral = spectral / spectral.square().sum(
            (-1, -2, -3), keepdim=True
        ).sqrt().clamp_min(1e-6)
        uncertainty = latent.new_zeros(
            batch, 1, self.config.cognitive.uncertainty_channels
        )
        consequence = latent.new_zeros(batch, 1, 4)
        consequence[:, 0, 0] = evidence.predictive_utility
        consequence[:, 0, 1] = evidence.action_utility
        writes = MemoryWriteBatch(
            self.memory_key(latent)[:, None], self.memory_value(latent)[:, None],
            self.memory_signature(latent)[:, None], spectral[:, None], support,
            node_type[:, None], provenance[:, None],
            torch.full(
                (batch, 1), int(SourceClass.ABSTRACTED), dtype=torch.int64,
                device=latent.device,
            ),
            scenario, uncertainty, consequence,
            (evidence.predictive_utility + evidence.action_utility)[:, None],
            promoted[:, None],
        )
        semantic = state.semantic_memory
        if bool(promoted.any()):
            semantic = self.memory.write(
                semantic, writes, writes.utility, quota=1,
                tier=MemoryTier.SEMANTIC, ledger=ledger,
            )
        type_logits = latent.new_zeros(
            batch, 1, self.config.cognitive.node_type_count
        )
        type_logits.scatter_(2, node_type[:, None, None], 1)
        first_supporter = torch.full(
            (batch, 1), -1, dtype=torch.int64, device=latent.device
        )
        for row in torch.nonzero(promoted, as_tuple=False).flatten().tolist():
            available = supporters[row][supporter_mask[row]]
            if available.numel():
                first_supporter[row, 0] = available[0]
        promoted_events = EventCandidates(
            latent[:, None], F.normalize(latent, dim=-1)[:, None],
            spectral[:, None], type_logits, support,
            torch.full(
                (batch, 1), int(ModalityClass.MEMORY), dtype=torch.int64,
                device=latent.device,
            ),
            first_supporter, provenance[:, None],
            torch.full(
                (batch, 1), int(SourceClass.ABSTRACTED), dtype=torch.int64,
                device=latent.device,
            ),
            scenario, uncertainty,
            evidence.calibrated_confidence[:, None], promoted[:, None],
        )
        nodes = state.nodes
        relations = state.relations
        if bool(promoted.any()):
            nodes = self.event_allocator(nodes, promoted_events)
            nodes = self._refresh_node_provenance_features(nodes, ledger)
            relations = invalidate_stale_relations(relations, nodes)
        return replace(
            state, knowledge=knowledge, semantic_memory=semantic,
            nodes=nodes, relations=relations,
        ), result

    def _reset_boundaries(
        self, state: MRCRARuntimeState, boundary: Tensor, sample_intervals: Tensor,
    ) -> MRCRARuntimeState:
        return self.apply_boundary_scopes(
            state, legacy_scope(boundary), sample_intervals=sample_intervals
        )

    def apply_boundary_scopes(
        self, state: MRCRARuntimeState, scopes: Tensor, *,
        sample_intervals: Tensor | None = None,
        continuity_ids: Tensor | None = None,
        environment_ids: Tensor | None = None,
        session_ids: Tensor | None = None,
    ) -> MRCRARuntimeState:
        """Apply explicit, independently testable continuity resets.

        Scope authority is deliberately not inferred from signal magnitude.
        Audit clocks and the append-only provenance ledger are external to this
        operation and are never rolled back by a reset.
        """

        batch = state.batch
        if scopes.shape != (batch,) or scopes.dtype != torch.int64:
            raise ValueError("boundary scopes must be int64 per batch")
        if bool(((scopes < 0) | (scopes >= len(BoundaryScope))).any()):
            raise ValueError("boundary scope is outside the ontology")
        if sample_intervals is None:
            sample_intervals = torch.tensor(
                [item.sample_interval for item in state.carrier],
                device=state.previous_latent.device,
                dtype=state.previous_latent.dtype,
            )
        if sample_intervals.shape != (batch,) or bool((sample_intervals <= 0).any()):
            raise ValueError("boundary resets require one positive sample interval per row")

        def one_of(*members: BoundaryScope) -> Tensor:
            result = torch.zeros(batch, dtype=torch.bool, device=scopes.device)
            for member in members:
                result |= scopes == int(member)
            return result

        event_reset = scopes != int(BoundaryScope.NONE)
        carrier_reset = one_of(
            BoundaryScope.SEGMENT, BoundaryScope.DOCUMENT,
            BoundaryScope.ENVIRONMENT_EPISODE, BoundaryScope.SESSION,
            BoundaryScope.IDENTITY_RESET, BoundaryScope.STREAM_DISCONTINUITY,
        )
        fast_reset = one_of(
            BoundaryScope.SEGMENT, BoundaryScope.DOCUMENT,
            BoundaryScope.ENVIRONMENT_EPISODE, BoundaryScope.SESSION,
            BoundaryScope.IDENTITY_RESET,
        )
        document_reset = one_of(
            BoundaryScope.DOCUMENT, BoundaryScope.ENVIRONMENT_EPISODE,
            BoundaryScope.SESSION, BoundaryScope.IDENTITY_RESET,
        )
        session_reset = one_of(BoundaryScope.SESSION, BoundaryScope.IDENTITY_RESET)
        identity_reset = one_of(BoundaryScope.IDENTITY_RESET)
        environment_reset = one_of(
            BoundaryScope.ENVIRONMENT_EPISODE, BoundaryScope.SESSION,
            BoundaryScope.IDENTITY_RESET,
        )
        # Calibration is process-global rather than batch-row state.  A partial
        # identity reset would ambiguously erase evidence owned by other rows.
        if bool(identity_reset.any()) and not bool(identity_reset.all()):
            raise ValueError("identity reset must cover the complete runtime batch")

        boundary_context = state.boundary_context.transition(
            scopes, continuity_ids=continuity_ids,
            environment_ids=environment_ids, session_ids=session_ids,
        )
        if not bool(event_reset.any()):
            return replace(state, boundary_context=boundary_context)

        carrier = list(state.carrier)
        for batch_index in torch.nonzero(carrier_reset, as_tuple=False).flatten().tolist():
            carrier[batch_index] = self.carrier.initial_stream_state(
                1, sample_interval=float(sample_intervals[batch_index].item()),
                device=state.previous_latent.device, dtype=state.previous_latent.dtype,
            )
        fresh = self.initial_state(
            state.batch, sample_intervals=sample_intervals,
            device=state.previous_latent.device, dtype=state.previous_latent.dtype,
        )
        event = EventExtractorState(
            torch.where(event_reset, fresh.event_extractor.open_start_times, state.event_extractor.open_start_times),
            torch.where(event_reset, fresh.event_extractor.open, state.event_extractor.open),
            torch.where(event_reset, fresh.event_extractor.emitted_in_chunk, state.event_extractor.emitted_in_chunk),
            state.event_extractor.position,
        )
        episodic = _replace_rows(state.episodic_memory, fresh.episodic_memory, document_reset)
        semantic = _replace_rows(state.semantic_memory, fresh.semantic_memory, identity_reset)
        calibration = (
            fresh.calibration if bool(identity_reset.all()) else state.calibration
        )
        return MRCRARuntimeState(
            tuple(carrier), event, _replace_rows(state.nodes, fresh.nodes, fast_reset),
            _replace_rows(state.relations, fresh.relations, fast_reset),
            _replace_rows(state.workspace, fresh.workspace, fast_reset),
            _replace_rows(state.hypotheses, fresh.hypotheses, document_reset),
            episodic, semantic, _replace_rows(state.controller, fresh.controller, fast_reset),
            _replace_rows(state.goals, fresh.goals, session_reset),
            _replace_rows(state.system_model, fresh.system_model, identity_reset),
            _replace_rows(state.schemas, fresh.schemas, fast_reset),
            calibration,
            _replace_rows(state.knowledge, fresh.knowledge, fast_reset),
            _replace_rows(
                state.last_external_action, fresh.last_external_action, fast_reset
            ),
            state.clocks,
            torch.where(carrier_reset[:, None], fresh.previous_latent, state.previous_latent),
            torch.where(carrier_reset[:, None], fresh.predicted_next_latent, state.predicted_next_latent),
            torch.where(carrier_reset[:, None], fresh.relational_context, state.relational_context),
            torch.where(carrier_reset, fresh.selected_physical_scale, state.selected_physical_scale),
            _replace_rows(state.reconstructions, fresh.reconstructions, fast_reset),
            _replace_rows(state.abstraction_validity, fresh.abstraction_validity, document_reset),
            _replace_rows(state.action_candidates, fresh.action_candidates, fast_reset),
            _replace_rows(state.viability, fresh.viability, environment_reset),
            _replace_rows(state.evidence_requests, fresh.evidence_requests, document_reset),
            _replace_rows(state.external_artifacts, fresh.external_artifacts, session_reset),
            _replace_rows(state.metacognition, fresh.metacognition, fast_reset),
            boundary_context,
        )

    def _derive_event_provenance(
        self, events: EventCandidates, ledger: ProvenanceLedger,
    ) -> EventCandidates:
        derived = torch.full_like(events.provenance_ids, -1)
        for batch_index, event_index in torch.nonzero(events.active, as_tuple=False).tolist():
            parent = int(events.parent_provenance_ids[batch_index, event_index].item())
            support = events.support[batch_index, event_index]
            derived[batch_index, event_index] = ledger.derive(
                [parent], source_class=SourceClass.INFERRED, operator="mrcra:eventizer",
                support=SupportInterval(*(float(value) for value in support.tolist())),
                modality=ModalityClass(int(events.modality_ids[batch_index, event_index].item())),
                scenario_id=int(events.scenario_ids[batch_index, event_index].item()),
                model_authority=self.model_authority,
            )
        return events.with_provenance(derived)

    def _derive_relation_provenance(
        self, proposals: RelationProposals, relations: RelationSlots,
        nodes: NodeSlots, ledger: ProvenanceLedger,
    ) -> RelationProposals:
        derived = torch.full_like(proposals.provenance_ids, -1)
        for batch_index, query_index, edge_index in torch.nonzero(
            proposals.active, as_tuple=False
        ).tolist():
            participants = proposals.participant_indices[batch_index, query_index, edge_index]
            participant_mask = proposals.participant_mask[batch_index, query_index, edge_index]
            active_participants = participants[participant_mask]
            family = proposals.family_ids[batch_index, query_index, edge_index]
            proposal_arity = participants.shape[-1]
            padded_participants = torch.full(
                (relations.participant_indices.shape[-1],), -1,
                dtype=torch.int64, device=participants.device,
            )
            padded_mask = torch.zeros_like(padded_participants, dtype=torch.bool)
            padded_participants[:proposal_arity] = participants
            padded_mask[:proposal_arity] = participant_mask
            existing = (
                relations.active[batch_index]
                & (relations.type_logits[batch_index].argmax(-1) == family)
                & (relations.participant_mask[batch_index] == padded_mask).all(-1)
                & (relations.participant_indices[batch_index] == padded_participants).all(-1)
                & (
                    relations.participant_versions[batch_index, :, :proposal_arity]
                    == nodes.versions[batch_index, participants.clamp_min(0)]
                ).masked_fill(~participant_mask, True).all(-1)
            )
            match = torch.nonzero(existing, as_tuple=False).flatten()
            if match.numel():
                derived[batch_index, query_index, edge_index] = relations.provenance_ids[
                    batch_index, match[0]
                ]
                continue
            parent_ids = proposals.parent_provenance_ids[
                batch_index, query_index, edge_index
            ][participant_mask]
            support = proposals.support[batch_index, query_index, edge_index]
            modality = int(nodes.modality_presence[
                batch_index, active_participants[0]
            ].argmax().item())
            derived[batch_index, query_index, edge_index] = ledger.derive(
                parent_ids.tolist(), source_class=SourceClass.INFERRED,
                operator=f"mrcra:relation:{int(family.item())}",
                support=SupportInterval(*(float(value) for value in support.tolist())),
                modality=ModalityClass(modality),
                scenario_id=int(proposals.scenario_ids[batch_index, query_index, edge_index].item()),
                model_authority=self.model_authority,
            )
        return proposals.with_provenance(derived)

    @staticmethod
    def _refresh_node_provenance_features(
        nodes: NodeSlots, ledger: ProvenanceLedger,
    ) -> NodeSlots:
        """Synchronize the neural metadata view from immutable ledger records."""

        features = nodes.provenance_features.clone()
        features[~nodes.active] = 0
        for row, slot in torch.nonzero(nodes.active, as_tuple=False).tolist():
            record_id = int(nodes.provenance_ids[row, slot])
            features[row, slot] = ledger.feature_vector(
                record_id, features.shape[-1], device=features.device,
                dtype=features.dtype,
            )
        values = {
            field.name: getattr(nodes, field.name)
            for field in fields(nodes)
        }
        values["provenance_features"] = features
        return NodeSlots(**values)

    @staticmethod
    def _workspace_summary(state: GlobalWorkspaceState) -> Tensor:
        weight = state.pointer_scores * state.active.to(state.slots.dtype)
        return (state.slots * weight.unsqueeze(-1)).sum(1) / weight.sum(1, keepdim=True).clamp_min(1)

    @staticmethod
    def _relation_summary(relations: RelationSlots) -> tuple[Tensor, Tensor]:
        confidence = relations.confidence.mean(-1) * relations.active.to(relations.content.dtype)
        normalizer = confidence.sum(-1, keepdim=True).clamp_min(1e-8)
        content = (relations.content * confidence.unsqueeze(-1)).sum(1) / normalizer
        family = (
            torch.softmax(relations.type_logits, -1) * confidence.unsqueeze(-1)
        ).sum(1) / normalizer
        empty = ~relations.active.any(-1)
        return content.masked_fill(empty[:, None], 0), family.masked_fill(empty[:, None], 0)

    @staticmethod
    def _graph_fragment(nodes: NodeSlots, relations: RelationSlots) -> GraphFragment:
        return GraphFragment(
            nodes.content, nodes.type_logits.argmax(-1), nodes.support,
            nodes.provenance_ids, nodes.active, relations.content,
            relations.type_logits.argmax(-1), relations.participant_indices,
            relations.provenance_ids, relations.active,
        )

    @staticmethod
    def _split_temporal_fragments(
        fragment: GraphFragment,
    ) -> tuple[GraphFragment, GraphFragment]:
        """Route two bounded temporal episodes without changing slot identity."""

        left_mask = torch.zeros_like(fragment.node_mask)
        right_mask = torch.zeros_like(fragment.node_mask)
        for row in range(fragment.node_mask.shape[0]):
            active = torch.nonzero(fragment.node_mask[row], as_tuple=False).flatten()
            if active.numel() < 4:
                continue
            ordered = active[
                fragment.node_support[row, active, 2].argsort()
            ]
            midpoint = ordered.numel() // 2
            left_mask[row, ordered[:midpoint]] = True
            right_mask[row, ordered[midpoint:]] = True

        def routed(node_mask: Tensor) -> GraphFragment:
            participants = fragment.participant_indices
            participant_mask = participants >= 0
            safe = participants.clamp_min(0)
            rows = torch.arange(
                participants.shape[0], device=participants.device
            )[:, None, None]
            inside = (~participant_mask) | node_mask[rows, safe]
            relation_mask = (
                fragment.relation_mask & inside.all(-1)
                & (participant_mask.sum(-1) >= 2)
            )
            return GraphFragment(
                fragment.node_content, fragment.node_type_ids,
                fragment.node_support, fragment.node_provenance_ids, node_mask,
                fragment.relation_content, fragment.relation_family_ids,
                participants.masked_fill(~relation_mask[..., None], -1),
                fragment.relation_provenance_ids, relation_mask,
            )

        return routed(left_mask), routed(right_mask)

    def _empty_action_receipts(self, batch: int, *, device) -> CognitiveActionReceipts:
        shape = (batch, self.config.cognitive.maximum_cognitive_steps)
        return CognitiveActionReceipts(
            torch.full(shape, -1, dtype=torch.int64, device=device),
            torch.full(shape, int(ActionStatus.HALTED), dtype=torch.int64, device=device),
            torch.zeros(shape, dtype=torch.bool, device=device),
            torch.full(shape, -1, dtype=torch.int64, device=device),
            torch.full(shape, -1, dtype=torch.int64, device=device),
            torch.full(shape, -1, dtype=torch.int64, device=device),
            torch.zeros(shape, dtype=torch.bool, device=device),
            torch.zeros(*shape, len(InternalAction), device=device),
            torch.zeros(*shape, len(RelationFamily), device=device),
            torch.ones(shape, device=device),
            torch.full(shape, -1, dtype=torch.int64, device=device),
            torch.full(shape, -1, dtype=torch.int64, device=device),
            torch.full(shape, -1, dtype=torch.int64, device=device),
            torch.zeros(*shape, self.config.cognitive.action_argument_dim, device=device),
            torch.zeros(
                *shape, self.config.cognitive.action_argument_dim,
                dtype=torch.bool, device=device,
            ),
            torch.zeros(shape, dtype=torch.int64, device=device),
            torch.zeros(shape, device=device),
        )

    @staticmethod
    def _clear_relation_rows(relations: RelationSlots, selection: Tensor) -> RelationSlots:
        values = {name: getattr(relations, name).clone() for name in relations.__dataclass_fields__}
        values["active"][selection] = False
        values["provenance_ids"][selection] = -1
        values["scenario_ids"][selection] = -1
        values["participant_indices"][selection] = -1
        values["participant_roles"][selection] = 0
        values["participant_versions"][selection] = -1
        values["participant_weights"][selection] = 0
        values["participant_mask"][selection] = False
        for name in ("content", "type_logits", "support", "confidence", "hypothesis_membership"):
            values[name][selection] = 0
        return RelationSlots(**values)

    def _retrieve_memory_action(
        self, memory_state: TensorMemoryState, nodes: NodeSlots, pointers: Tensor,
        action_mask: Tensor, ledger: ProvenanceLedger, *, timestamps: Tensor,
    ) -> tuple[TensorMemoryState, NodeSlots, Tensor, Tensor]:
        batch = nodes.batch
        safe = pointers.clamp_min(0)
        batch_index = torch.arange(batch, device=nodes.content.device)
        query_mask = action_mask & (pointers >= 0) & nodes.active[batch_index, safe]
        query = MemoryQuery(
            self.memory_key(nodes.content[batch_index, safe])[:, None],
            self.memory_signature(nodes.content[batch_index, safe])[:, None],
            nodes.spectral[batch_index, safe][:, None],
            timestamps[:, None].to(nodes.content),
            nodes.type_logits[batch_index, safe].argmax(-1)[:, None],
            nodes.source_classes[batch_index, safe][:, None],
            nodes.scenario_ids[batch_index, safe][:, None],
            query_mask[:, None],
        )
        retrieval = self.memory.retrieve(memory_state, query)
        found = retrieval.mask[:, 0, 0] & query_mask
        context = nodes.content.new_zeros(batch, nodes.content.shape[-1])
        if bool(found.any()):
            context[found] = retrieval.values[found, 0, 0]
            memory_state = self.memory.mark_accessed(memory_state, retrieval)
            candidate_content = retrieval.values[:, 0, 0]
            type_logits = candidate_content.new_zeros(batch, 1, self.config.cognitive.node_type_count)
            type_logits[..., int(NodeType.MEMORY)] = 1
            support = candidate_content.new_zeros(batch, 1, 3)
            support[found] = timestamps[found, None].to(support)
            provenance = torch.full((batch, 1), -1, dtype=torch.int64, device=nodes.content.device)
            for row in torch.nonzero(found, as_tuple=False).flatten().tolist():
                parent = int(retrieval.provenance_ids[row, 0, 0].item())
                provenance[row, 0] = ledger.derive(
                    [parent], source_class=SourceClass.RETRIEVED,
                    operator="mrcra:memory_retrieval",
                    support=SupportInterval(
                        float(timestamps[row]), float(timestamps[row]), float(timestamps[row])
                    ),
                    modality=ModalityClass.MEMORY,
                    scenario_id=int(nodes.scenario_ids[row, safe[row]].item()),
                    model_authority=self.model_authority,
                )
            retrieved_events = EventCandidates(
                candidate_content[:, None], F.normalize(candidate_content, dim=-1)[:, None],
                nodes.spectral[batch_index, safe][:, None], type_logits, support,
                torch.full((batch, 1), int(ModalityClass.MEMORY), dtype=torch.int64, device=nodes.content.device),
                retrieval.provenance_ids[:, :, 0], provenance,
                torch.full((batch, 1), int(SourceClass.RETRIEVED), dtype=torch.int64, device=nodes.content.device),
                nodes.scenario_ids[batch_index, safe][:, None],
                retrieval.uncertainty[:, :, 0],
                retrieval.scores[:, :, 0], found[:, None],
            )
            nodes = self.event_allocator(nodes, retrieved_events)
            nodes = self._refresh_node_provenance_features(nodes, ledger)
        return memory_state, nodes, context, found

    @staticmethod
    def _retrieve_recent_buffer(
        state: TensorMemoryState, action_mask: Tensor,
    ) -> tuple[TensorMemoryState, Tensor, Tensor]:
        """Retrieve the most recently completed episodic item, not a graph node."""

        if action_mask.shape != (state.batch,) or action_mask.dtype != torch.bool:
            raise ValueError("recent-buffer action mask is invalid")
        score = state.support[..., 2].masked_fill(~state.active, -torch.inf)
        indices = score.argmax(-1)
        found = action_mask & state.active.any(-1)
        rows = torch.arange(state.batch, device=state.values.device)
        context = state.values[rows, indices].masked_fill(~found[:, None], 0)
        values = {
            name: getattr(state, name).clone()
            if isinstance(getattr(state, name), Tensor) else getattr(state, name)
            for name in state.__dataclass_fields__
        }
        for row in torch.nonzero(found, as_tuple=False).flatten().tolist():
            slot = int(indices[row])
            values["use_count"][row, slot] += 1
            values["last_access"][row, slot] = state.clock
            values["utility"][row, slot] += 0.01
        values["clock"] = state.clock + 1
        return TensorMemoryState(**values), context, found

    def _write_memory_action(
        self, state: TensorMemoryState, nodes: NodeSlots, pointers: Tensor,
        action_mask: Tensor, *, tier: MemoryTier, ledger: ProvenanceLedger,
    ) -> tuple[TensorMemoryState, Tensor]:
        batch = nodes.batch
        safe = pointers.clamp_min(0)
        batch_index = torch.arange(batch, device=nodes.content.device)
        valid = action_mask & (pointers >= 0) & nodes.active[batch_index, safe]
        content = nodes.content[batch_index, safe]
        similarity = F.cosine_similarity(
            content[:, None], state.values, dim=-1
        ).masked_fill(~state.active, -torch.inf)
        redundancy = similarity.amax(-1)
        redundancy = torch.where(
            torch.isfinite(redundancy), redundancy.clamp_min(0),
            torch.zeros_like(redundancy),
        )
        uncertainty_mean = nodes.uncertainty[batch_index, safe].mean(-1)
        write_evidence = MemoryWriteEvidence(
            nodes.activity[batch_index, safe][:, None],
            uncertainty_mean[:, None],
            (nodes.support[batch_index, safe, 1] - nodes.support[batch_index, safe, 0]).abs()[:, None],
            nodes.importance[batch_index, safe][:, None],
            nodes.hypothesis_membership[batch_index, safe].amax(-1)[:, None],
            content.norm(dim=-1)[:, None],
            nodes.uncertainty[batch_index, safe, 0][:, None],
            nodes.hypothesis_membership[batch_index, safe].amax(-1)[:, None],
            redundancy[:, None],
            nodes.uncertainty[batch_index, safe, -1][:, None],
            valid[:, None],
        )
        write_score = self.memory_write_policy(write_evidence)
        writes = MemoryWriteBatch(
            self.memory_key(content)[:, None], self.memory_value(content)[:, None],
            self.memory_signature(content)[:, None], nodes.spectral[batch_index, safe][:, None],
            nodes.support[batch_index, safe][:, None],
            nodes.type_logits[batch_index, safe].argmax(-1)[:, None],
            nodes.provenance_ids[batch_index, safe][:, None],
            nodes.source_classes[batch_index, safe][:, None],
            nodes.scenario_ids[batch_index, safe][:, None],
            nodes.uncertainty[batch_index, safe][:, None],
            content.new_zeros(batch, 1, 4), write_score.masked_fill(~valid[:, None], 0),
            valid[:, None],
        )
        if bool(valid.any()):
            state = self.memory.write(
                state, writes, write_score, quota=1,
                tier=tier, ledger=ledger if tier == MemoryTier.SEMANTIC else None,
            )
        return state, valid

    def _association_context(
        self, memory_state: TensorMemoryState, nodes: NodeSlots, pointers: Tensor,
        action_mask: Tensor, *, timestamps: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Retrieve a seed and perform bounded, visited associative spreading."""

        batch = nodes.batch
        safe = pointers.clamp_min(0)
        batch_index = torch.arange(batch, device=nodes.content.device)
        query_mask = action_mask & (pointers >= 0) & nodes.active[batch_index, safe]
        query = MemoryQuery(
            self.memory_key(nodes.content[batch_index, safe])[:, None],
            self.memory_signature(nodes.content[batch_index, safe])[:, None],
            nodes.spectral[batch_index, safe][:, None], timestamps[:, None].to(nodes.content),
            nodes.type_logits[batch_index, safe].argmax(-1)[:, None],
            nodes.source_classes[batch_index, safe][:, None],
            nodes.scenario_ids[batch_index, safe][:, None], query_mask[:, None],
        )
        retrieval = self.memory.retrieve(memory_state, query)
        seed_mask = retrieval.mask[:, 0, :1] & query_mask[:, None]
        expansion = self.memory.expand_associations(
            memory_state, retrieval.indices[:, 0, :1].clamp_min(0), seed_mask,
            maximum_depth=self.config.cognitive.associative_depth,
            budget=self.config.cognitive.associative_budget,
        )
        expanded = self.memory._gather(memory_state.values, expansion.indices[:, None])[:, 0]
        weight = expansion.scores * expansion.mask.to(expansion.scores.dtype)
        context = (expanded * weight.unsqueeze(-1)).sum(1) / weight.sum(
            1, keepdim=True
        ).clamp_min(1e-8)
        found = expansion.mask.any(-1) & action_mask
        return context, found

    def _decomposed_uncertainty(
        self, latent: Tensor, state: MRCRARuntimeState, packet: ObservationPacket,
        index: int, ledger: ProvenanceLedger,
    ) -> tuple[Tensor, DistributionalOutput]:
        """Compute all named uncertainty causes without collapsing their meaning."""

        distribution = self.distributional_head(latent)
        _, relation_posterior = self._relation_summary(state.relations)
        empty_relations = ~state.relations.active.any(-1)
        relation_posterior = torch.where(
            empty_relations[:, None],
            torch.full_like(
                relation_posterior, 1 / relation_posterior.shape[-1]
            ),
            relation_posterior / relation_posterior.sum(-1, keepdim=True).clamp_min(1e-8),
        )
        retrieval_rows, retrieval_masks = [], []
        for memory in (state.episodic_memory, state.semantic_memory):
            count = min(8, memory.capacity)
            scores = memory.utility.masked_fill(~memory.active, -torch.inf)
            top = scores.topk(count, -1)
            retrieval_rows.append(top.values)
            retrieval_masks.append(torch.isfinite(top.values))
        retrieval_scores = torch.cat(retrieval_rows, -1)
        retrieval_mask = torch.cat(retrieval_masks, -1)
        source_reliability = latent.new_zeros(state.batch)
        for row in range(state.batch):
            if bool(packet.valid_mask[row, index]):
                record_id = int(packet.source_record_ids[row, index])
                source_reliability[row] = ledger.get(record_id).source_reliability
        family = state.relations.type_logits.argmax(-1)
        contradiction = (
            state.relations.active
            & (family == int(RelationFamily.CONTRADICTION_ALTERNATIVE))
        )
        structural_conflict = (
            state.relations.confidence.mean(-1) * contradiction
        ).sum(-1) / contradiction.sum(-1).clamp_min(1)
        calibration_report = self.online_calibration.report(state.calibration)
        groups = packet.modality_ids[:, index].clamp_min(0) % self.config.cognitive.calibration_regimes
        calibration_error = calibration_report.expected_calibration_error[groups]
        named = self.uncertainty_estimator(UncertaintyInputs(
            distribution.aleatoric,
            distribution.ensemble_values,
            state.hypotheses.weights,
            relation_posterior,
            retrieval_scores,
            retrieval_mask,
            latent.new_zeros(state.batch),
            source_reliability,
            structural_conflict,
            calibration_error,
        ))
        return F.softplus(self.uncertainty_head(latent)) + named, distribution

    def _semantic_symbol_context(
        self, semantic: TensorMemoryState, context: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Activate accepted invariant/symbol memory rows conditionally."""

        count = min(
            self.config.cognitive.knowledge_support_capacity,
            semantic.capacity,
        )
        symbolic = semantic.active & (
            (semantic.type_ids == int(NodeType.INVARIANT))
            | (semantic.type_ids == int(NodeType.SYMBOL))
        )
        score = semantic.utility.masked_fill(~symbolic, -torch.inf)
        top = score.topk(count, -1)
        mask = torch.isfinite(top.values)
        values, gates = self.symbol_activator(
            top.indices.masked_fill(~mask, 0), context, mask,
        )
        normalizer = gates.sum(-1, keepdim=True).clamp_min(1e-8)
        modulation = (values * gates.unsqueeze(-1)).sum(1) / normalizer
        modulation = modulation.masked_fill(~mask.any(-1, keepdim=True), 0)
        padded = context.new_zeros(
            context.shape[0], self.config.cognitive.knowledge_support_capacity
        )
        padded[:, :count] = gates
        return modulation, padded

    def _workspace_supporters(
        self, workspace: GlobalWorkspaceState, nodes: NodeSlots,
    ) -> tuple[Tensor, Tensor]:
        capacity = self.config.cognitive.knowledge_support_capacity
        ids = torch.full(
            (nodes.batch, capacity), -1, dtype=torch.int64,
            device=nodes.content.device,
        )
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for row in range(nodes.batch):
            active_slots = torch.nonzero(
                workspace.active[row], as_tuple=False
            ).flatten()
            if active_slots.numel() == 0:
                continue
            ordered = active_slots[
                workspace.pointer_scores[row, active_slots].argsort(descending=True)
            ][:capacity]
            pointers = workspace.node_pointers[row, ordered]
            safe = pointers.clamp(0, nodes.capacity - 1)
            valid = (
                (pointers >= 0) & (pointers < nodes.capacity)
                & nodes.active[row, safe]
            )
            pointers = pointers[valid]
            count = pointers.numel()
            if count:
                ids[row, :count] = nodes.provenance_ids[row, pointers]
                mask[row, :count] = True
        return ids, mask

    def _knowledge_proposal(
        self, *, kind: KnowledgeKind, latent: Tensor, code_gain: Tensor,
        reconstruction_distortion: Tensor, relation_distortion: Tensor,
        confidence: Tensor, mask: Tensor, workspace: GlobalWorkspaceState,
        nodes: NodeSlots, ledger: ProvenanceLedger, operator: str,
        timestamp: Tensor,
    ) -> KnowledgeProposalBatch:
        supporters, supporting_mask = self._workspace_supporters(workspace, nodes)
        provenance = torch.full(
            (nodes.batch,), -1, dtype=torch.int64, device=nodes.content.device
        )
        actual_mask = mask & supporting_mask.any(-1)
        for row in torch.nonzero(actual_mask, as_tuple=False).flatten().tolist():
            parents = supporters[row][supporting_mask[row]].tolist()
            time = float(timestamp[row])
            provenance[row] = ledger.derive(
                parents, source_class=SourceClass.ABSTRACTED, operator=operator,
                support=SupportInterval(time, time, time),
                modality=ModalityClass.MEMORY, scenario_id=0,
                model_authority=self.model_authority,
            )
        return KnowledgeProposalBatch(
            latent,
            torch.full(
                (nodes.batch,), int(kind), dtype=torch.int64,
                device=nodes.content.device,
            ),
            code_gain, reconstruction_distortion, relation_distortion,
            latent.new_zeros(nodes.batch), latent.new_zeros(nodes.batch),
            confidence.clamp(0, 1), provenance, supporters, supporting_mask,
            actual_mask,
        )

    def _update_hypotheses_from_observation(
        self, hypotheses: HypothesisState, observation: Tensor,
        active_rows: Tensor, provenance_ids: Tensor | None = None,
    ) -> HypothesisState:
        """Apply production observation likelihoods to every active alternative."""

        active = hypotheses.active & active_rows[:, None]
        if not bool(active.any()):
            return hypotheses
        predicted = hypotheses.predicted_outcomes
        if predicted.shape[-1] != observation.shape[-1]:
            raise ValueError("hypothesis outcome and observation widths differ")
        variance = hypotheses.uncertainty.mean(-1).clamp_min(1e-4)
        squared_error = (predicted - observation[:, None]).square().mean(-1)
        log_likelihood = -0.5 * (
            squared_error / variance + variance.log()
        )
        log_likelihood = log_likelihood.masked_fill(~active, 0)
        centered = log_likelihood - (
            (log_likelihood * active).sum(-1, keepdim=True)
            / active.sum(-1, keepdim=True).clamp_min(1)
        )
        support = (centered > 0).to(observation.dtype)
        contradiction = (centered < 0).to(observation.dtype)
        return self.hypothesis_bank.update_evidence(
            hypotheses, log_likelihood, support, contradiction,
            provenance_ids=provenance_ids,
        )

    def _post_deliberation_action(
        self, *, context: Tensor, nodes: NodeSlots,
        workspace: GlobalWorkspaceState,
        relations: RelationSlots, hypotheses: HypothesisState,
        uncertainty: Tensor, goals: GoalState, system: SystemModelState,
        active_rows: Tensor, ledger: ProvenanceLedger,
        viability: ViabilityState | None = None,
    ) -> tuple[ActionCandidateState, ExternalActionDecision]:
        """Generate, simulate, hard-gate, and finally select external actions."""

        supporters, supporter_mask = self._workspace_supporters(workspace, nodes)
        proposal = self.external_action_policy(
            context, goals, uncertainty, system,
            supporting_provenance_ids=supporters,
            supporting_mask=supporter_mask, active_mask=active_rows,
        )
        candidates = build_action_candidates(
            proposal_logits=proposal.logits,
            expected_reward=proposal.expected_reward,
            expected_cost=proposal.expected_cost,
            constraint_probability=proposal.constraint_probability,
            expected_success=proposal.expected_success,
            available=proposal.available,
            permission_mask=system.permission_mask,
            supporting_provenance_ids=supporters,
            supporting_mask=supporter_mask,
            capacity=self.config.cognitive.action_candidate_capacity,
            argument_dim=self.config.cognitive.action_argument_dim,
        )
        routed_mass = context.new_zeros(context.shape[0])
        if bool(candidates.active.any()):
            routed = self.hypothesis_bank.route(
                hypotheses, min(
                    hypotheses.capacity,
                    self.config.cognitive.planning_hypothesis_top_k,
                )
            )
            safe_route = routed.indices.clamp_min(0)
            rows = torch.arange(context.shape[0], device=context.device)[:, None]
            routed_residuals = hypotheses.residuals[rows, safe_route]
            routed_scenarios = hypotheses.scenario_ids[rows, safe_route]
            hypothesis_mask = routed.mask
            routed_weights = hypotheses.weights[rows, safe_route] * routed.mask
            routed_weights = routed_weights / routed_weights.sum(-1, keepdim=True).clamp_min(1e-8)
            routed_mass = routed.posterior_mass
            # An action-capable deliberation without an explicit alternative is
            # not allowed to fabricate scenario authority.  It abstains below.
            if bool(hypothesis_mask.any()):
                schema = candidates.schema_ids.clamp_min(0)
                action_vectors = F.one_hot(
                    schema, self.config.cognitive.system_action_channels
                ).to(context.dtype)
                rollout = self.world_model.rollout_candidates(
                    self._workspace_summary(workspace), self._relation_summary(relations)[0],
                    routed_residuals, routed_scenarios,
                    hypothesis_mask, action_vectors, candidates.active,
                )
                candidates = evaluate_candidate_rollout(
                    candidates, reward_quantiles=rollout.reward_quantiles,
                    costs=rollout.costs,
                    constraint_probabilities=rollout.constraint_probabilities,
                    success_probabilities=rollout.action_success_probabilities,
                    uncertainty=rollout.uncertainty,
                    hypothesis_weights=routed_weights,
                    rollout_mask=rollout.valid_mask,
                )
        candidates = authorize_candidate_provenance(candidates, ledger)
        if self.config.cognitive.enable_viability_gate:
            if viability is None:
                viable = torch.zeros_like(candidates.active)
            else:
                forecast = self.viability_forecaster(
                    viability, expected_reward=candidates.expected_reward,
                    expected_cost=candidates.expected_cost,
                    constraint_probability=candidates.constraint_probability,
                    expected_success=candidates.expected_success,
                    expected_energy=candidates.expected_energy,
                    tail_risk=candidates.tail_risk,
                    candidate_mask=candidates.active,
                )
                viable = self.viability_gate(viability, forecast).authorized
        else:
            viable = candidates.active
        viable &= (
            routed_mass >= self.config.cognitive.minimum_routed_posterior_mass
        )[:, None]
        if viable.shape != candidates.active.shape:
            raise ValueError("viability authorization must match candidate rows")
        candidate_values = {
            name: getattr(candidates, name) for name in candidates.__dataclass_fields__
        }
        candidate_values["viability_authorized"] = viable & candidates.active
        candidates = ActionCandidateState(**candidate_values)
        candidates = select_action_candidate(candidates)
        return candidates, decision_from_candidates(
            candidates,
            action_count=self.config.cognitive.system_action_channels,
            active_mask=active_rows,
        )

    def _reconstruct_local_graph(
        self, nodes: NodeSlots, relations: RelationSlots,
        workspace: GlobalWorkspaceState, hypotheses: HypothesisState,
        episodic: TensorMemoryState, reconstruction_state: ReconstructionState,
        validity_state: AbstractionValidityState, abstraction_indices: Tensor,
        action_mask: Tensor, goal_context: Tensor, ledger: ProvenanceLedger,
        *, target_physical_scale: Tensor | None = None,
        target_abstraction_depth: Tensor | None = None,
        precision_tolerance: Tensor | None = None,
    ) -> tuple[
        NodeSlots, RelationSlots, ReconstructionState, AbstractionValidityState,
        Tensor, Tensor,
    ]:
        """Generate, provenance-mark, and materialize one localized graph per row."""

        batch, width = nodes.batch, nodes.content.shape[-1]
        safe = abstraction_indices.clamp(0, nodes.capacity - 1)
        rows = torch.arange(batch, device=nodes.content.device)
        node_types = nodes.type_logits.argmax(-1)
        valid_abstraction = (
            action_mask & (abstraction_indices >= 0) & nodes.active[rows, safe]
            & ((node_types[rows, safe] == int(NodeType.ABSTRACTION))
               | (node_types[rows, safe] == int(NodeType.INVARIANT)))
        )
        trace_count = min(4, episodic.capacity)
        trace_score = torch.einsum(
            "bd,bmd->bm", nodes.content[rows, safe], episodic.values
        ).masked_fill(~episodic.active, -torch.inf)
        routed = trace_score.topk(trace_count, -1).indices
        trace_mask = torch.gather(episodic.active, 1, routed)
        trace_content = episodic.values[rows[:, None], routed]
        trace_provenance = episodic.provenance_ids[rows[:, None], routed]
        relation_summary, _ = self._relation_summary(relations)
        workspace_summary = self._workspace_summary(workspace)
        hypothesis_context = (
            hypotheses.residuals * hypotheses.weights.unsqueeze(-1)
        ).sum(1)
        applicability = self.abstraction_applicability(
            nodes.content[rows, safe], workspace_summary, relation_summary,
            hypothesis_context, goal_context,
        )
        local_capacity = min(self.config.cognitive.event_proposals_per_chunk, 4)
        if target_physical_scale is None:
            target_physical_scale = torch.zeros(
                batch, dtype=torch.int64, device=nodes.content.device
            )
        if target_abstraction_depth is None:
            target_abstraction_depth = torch.ones(
                batch, dtype=torch.int64, device=nodes.content.device
            )
        if precision_tolerance is None:
            precision_tolerance = nodes.content.new_full(
                (batch,), self.config.cognitive.abstraction_maximum_distortion
            )
        query = ReconstructionQuery(
            safe, safe, nodes.support[rows, safe],
            torch.full((batch,), local_capacity, dtype=torch.int64, device=nodes.content.device),
            torch.full((batch,), local_capacity, dtype=torch.int64, device=nodes.content.device),
            target_physical_scale, target_abstraction_depth,
            precision_tolerance,
            goal_context, valid_abstraction,
        )
        supporters, supporter_mask = self._workspace_supporters(workspace, nodes)
        evidence = ReconstructionEvidence(
            nodes.content[rows, safe], trace_content, trace_mask, trace_provenance,
            workspace_summary, supporters, relation_summary[:, None],
            hypothesis_context, goal_context,
        )
        proposal = self.reconstructor(query, evidence)
        node_provenance = torch.full(
            proposal.node_mask.shape, -1, dtype=torch.int64, device=nodes.content.device
        )
        relation_provenance = torch.full(
            proposal.relation_mask.shape, -1, dtype=torch.int64, device=nodes.content.device
        )
        abstraction_provenance = nodes.provenance_ids[rows, safe]
        scenario_ids = nodes.scenario_ids[rows, safe].clamp_min(0)
        for row, item in torch.nonzero(proposal.node_mask, as_tuple=False).tolist():
            parents = [int(abstraction_provenance[row])]
            parents.extend(int(value) for value in trace_provenance[row][trace_mask[row]].tolist())
            parents.extend(int(value) for value in supporters[row][supporter_mask[row]].tolist())
            parents = list(dict.fromkeys(value for value in parents if value >= 0))
            support = query.requested_support[row]
            node_provenance[row, item] = ledger.derive(
                parents, source_class=SourceClass.RECONSTRUCTED,
                operator="mrcra:conditional_reconstruction:v1",
                support=SupportInterval(*(float(value) for value in support.tolist())),
                modality=ModalityClass.MEMORY, scenario_id=int(scenario_ids[row]),
                model_authority=self.model_authority,
            )
        for row, item in torch.nonzero(proposal.relation_mask, as_tuple=False).tolist():
            local = proposal.participant_indices[row, item, :2]
            parents = [int(node_provenance[row, int(index)]) for index in local.tolist()]
            support = query.requested_support[row]
            relation_provenance[row, item] = ledger.derive(
                parents, source_class=SourceClass.RECONSTRUCTED,
                operator="mrcra:conditional_relation_reconstruction:v1",
                support=SupportInterval(*(float(value) for value in support.tolist())),
                modality=ModalityClass.MEMORY, scenario_id=int(scenario_ids[row]),
                model_authority=self.model_authority,
            )
        result = proposal.finalize(node_provenance, relation_provenance)

        node_count = result.node_content.shape[1]
        spectral = self.spectral_projection(result.node_content).reshape(
            batch, node_count, self.config.cognitive.relation_heads,
            self.config.cognitive.relation_modes, 2,
        )
        spectral = spectral / spectral.square().sum(
            (-1, -2, -3), keepdim=True
        ).sqrt().clamp_min(1e-6)
        support = query.requested_support[:, None].expand(-1, node_count, -1)
        uncertainty = (
            result.epistemic_uncertainty + result.aleatoric_uncertainty
        )[:, None, None].expand(
            -1, node_count, self.config.cognitive.uncertainty_channels
        )
        reconstructed_events = EventCandidates(
            result.node_content, F.normalize(result.node_content, dim=-1), spectral,
            result.node_type_logits, support,
            torch.full(
                (batch, node_count), int(ModalityClass.MEMORY),
                dtype=torch.int64, device=nodes.content.device,
            ),
            abstraction_provenance[:, None].expand(-1, node_count),
            result.provenance_ids,
            torch.full(
                (batch, node_count), int(SourceClass.RECONSTRUCTED),
                dtype=torch.int64, device=nodes.content.device,
            ),
            scenario_ids[:, None].expand(-1, node_count), uncertainty,
            torch.logit(result.evidence_agreement.clamp(1e-4, 1 - 1e-4))[:, None].expand(-1, node_count),
            result.node_mask,
        )
        nodes, allocation = self.event_allocator.allocate_with_receipts(
            nodes, reconstructed_events
        )
        nodes = self._refresh_node_provenance_features(nodes, ledger)
        relations = invalidate_stale_relations(relations, nodes)

        relation_count = result.relation_content.shape[1]
        local_participants = result.participant_indices[..., :2].clamp_min(0)
        allocated = torch.gather(
            allocation[:, None, :].expand(-1, relation_count, -1),
            2, local_participants,
        )
        participant_mask = result.relation_mask.unsqueeze(-1).expand(-1, -1, 2)
        family = result.relation_type_logits.argmax(-1).masked_fill(~result.relation_mask, -1)
        relation_support = query.requested_support[:, None].expand(-1, relation_count, -1)
        parent_provenance = torch.gather(
            result.provenance_ids[:, None, :].expand(-1, relation_count, -1),
            2, local_participants,
        )
        relation_proposals = RelationProposals(
            result.relation_content[:, None], family[:, None], allocated[:, None],
            torch.tensor([0, 1], dtype=torch.int64, device=nodes.content.device).view(
                1, 1, 1, 2
            ).expand(batch, 1, relation_count, 2),
            participant_mask[:, None], relation_support[:, None],
            result.structural_plausibility[:, None, None].expand(-1, 1, relation_count),
            parent_provenance[:, None], result.relation_provenance_ids[:, None],
            scenario_ids[:, None, None].expand(-1, 1, relation_count),
            result.relation_mask[:, None],
        )
        if bool(result.relation_mask.any()):
            relations = self.relation_writer(relations, relation_proposals, nodes)

        reconstruction_values = {
            name: getattr(reconstruction_state, name).clone()
            for name in reconstruction_state.__dataclass_fields__
        }
        validity_values = {
            name: getattr(validity_state, name).clone()
            for name in validity_state.__dataclass_fields__
        }
        context = nodes.content.new_zeros(batch, width)
        for row in torch.nonzero(valid_abstraction, as_tuple=False).flatten().tolist():
            active_nodes = result.node_mask[row]
            context[row] = result.node_content[row, active_nodes].mean(0)
            free = torch.nonzero(~reconstruction_state.active[row], as_tuple=False).flatten()
            slot = int(free[0]) if free.numel() else int(reconstruction_state.versions[row].argmin())
            reconstruction_values["latent"][row, slot] = context[row]
            reconstruction_values["historical_fidelity"][row, slot] = result.historical_fidelity[row]
            reconstruction_values["structural_plausibility"][row, slot] = result.structural_plausibility[row]
            reconstruction_values["evidence_agreement"][row, slot] = result.evidence_agreement[row]
            reconstruction_values["uncertainty"][row, slot] = uncertainty[row, 0]
            reconstruction_values["abstraction_indices"][row, slot] = safe[row]
            reconstruction_values["provenance_ids"][row, slot] = result.provenance_ids[row, active_nodes][0]
            reconstruction_values["source_classes"][row, slot] = int(SourceClass.RECONSTRUCTED)
            reconstruction_values["scenario_ids"][row, slot] = scenario_ids[row]
            reconstruction_values["physical_scales"][row, slot] = query.target_scale[row]
            reconstruction_values["abstraction_depths"][row, slot] = query.target_abstraction_depth[row]
            reconstruction_values["support"][row, slot] = query.requested_support[row]
            reconstruction_values["versions"][row, slot] += 1
            reconstruction_values["active"][row, slot] = True

            free_validity = torch.nonzero(~validity_state.active[row], as_tuple=False).flatten()
            validity_slot = int(free_validity[0]) if free_validity.numel() else int(validity_state.versions[row].argmin())
            validity_values["applicability"][row, validity_slot] = (
                result.applicability_probability[row] * applicability.applicability[row]
            ).sqrt()
            validity_values["reconstruction_distortion"][row, validity_slot] = torch.maximum(
                1 - result.historical_fidelity[row], applicability.reconstruction_distortion[row]
            )
            validity_values["relation_distortion"][row, validity_slot] = torch.maximum(
                1 - result.structural_plausibility[row], applicability.relation_distortion[row]
            )
            validity_values["task_distortion"][row, validity_slot] = torch.maximum(
                1 - result.evidence_agreement[row], applicability.task_distortion[row]
            )
            validity_values["provenance_sufficiency"][row, validity_slot] = supporter_mask[row].any().to(nodes.content.dtype)
            validity_values["precision_sufficiency"][row, validity_slot] = torch.minimum(
                (1 - result.epistemic_uncertainty[row]).clamp(0, 1),
                applicability.supported_precision[row],
            )
            validity_values["calibrated_confidence"][row, validity_slot] = (
                result.historical_fidelity[row] * result.evidence_agreement[row]
                * applicability.calibrated_confidence[row]
            ).pow(1 / 3)
            validity_values["abstraction_depths"][row, validity_slot] = query.target_abstraction_depth[row]
            validity_values["physical_scales"][row, validity_slot] = query.target_scale[row]
            validity_values["abstraction_node_indices"][row, validity_slot] = safe[row]
            validity_values["provenance_ids"][row, validity_slot] = abstraction_provenance[row]
            validity_values["last_checked_observation"][row, validity_slot] = 0
            validity_values["versions"][row, validity_slot] += 1
            validity_values["active"][row, validity_slot] = True
        return (
            nodes, relations, ReconstructionState(**reconstruction_values),
            AbstractionValidityState(**validity_values), context, valid_abstraction,
        )

    def _action_availability(
        self, nodes: NodeSlots, relations: RelationSlots,
        hypotheses: HypothesisState, episodic: TensorMemoryState,
        semantic: TensorMemoryState, workspace: GlobalWorkspaceState,
        selected_scale: Tensor,
    ) -> Tensor:
        """Construct hard per-row action preconditions for the controller."""

        batch = nodes.batch
        allowed = torch.zeros(
            batch, len(InternalAction), dtype=torch.bool, device=nodes.content.device
        )
        node_count = nodes.active.sum(-1)
        relation_count = relations.active.sum(-1)
        hypothesis_count = hypotheses.active.sum(-1)
        node_types = nodes.type_logits.argmax(-1)
        has_abstraction = (
            nodes.active
            & ((node_types == int(NodeType.ABSTRACTION)) | (node_types == int(NodeType.INVARIANT)))
        ).any(-1)
        has_association = (
            (episodic.association_mask & episodic.active.unsqueeze(-1)).any((1, 2))
            | (semantic.association_mask & semantic.active.unsqueeze(-1)).any((1, 2))
        )
        allowed[:, int(InternalAction.HALT)] = True
        allowed[:, int(InternalAction.BIND)] = node_count >= 2
        allowed[:, int(InternalAction.UNBIND)] = relation_count > 0
        allowed[:, int(InternalAction.RETYPE_RELATION)] = relation_count > 0
        allowed[:, int(InternalAction.RETRIEVE_RECENT)] = episodic.active.any(-1)
        allowed[:, int(InternalAction.RETRIEVE_EPISODIC)] = (node_count > 0) & episodic.active.any(-1)
        allowed[:, int(InternalAction.RETRIEVE_SEMANTIC)] = (node_count > 0) & semantic.active.any(-1)
        allowed[:, int(InternalAction.EXPAND_ASSOCIATION)] = (node_count > 0) & has_association
        allowed[:, int(InternalAction.COMPARE)] = node_count >= 2
        allowed[:, int(InternalAction.COMPRESS)] = node_count > 0
        allowed[:, int(InternalAction.DECOMPRESS)] = (
            has_abstraction
            & ~torch.full_like(
                has_abstraction,
                self.config.cognitive.enable_conditional_reconstruction,
            )
        )
        allowed[:, int(InternalAction.CREATE_HYPOTHESIS)] = workspace.active.any(-1)
        allowed[:, int(InternalAction.MERGE_HYPOTHESES)] = hypothesis_count >= 2
        allowed[:, int(InternalAction.PRUNE_HYPOTHESIS)] = hypothesis_count > 0
        allowed[:, int(InternalAction.SIMULATE)] = workspace.active.any(-1)
        allowed[:, int(InternalAction.VERIFY)] = node_count > 0
        allowed[:, int(InternalAction.WRITE_EPISODE)] = node_count > 0
        allowed[:, int(InternalAction.PROPOSE_INVARIANT)] = episodic.active.sum(-1) >= 2
        allowed[:, int(InternalAction.DESCEND_SCALE)] = selected_scale > 0
        allowed[:, int(InternalAction.ASCEND_SCALE)] = selected_scale < self.config.carrier.scales - 1
        allowed[:, int(InternalAction.ABSTAIN_OR_REQUEST_EXTERNAL_EVIDENCE)] = True
        if self.config.cognitive.enable_conditional_reconstruction:
            allowed[:, int(InternalAction.RECONSTRUCT_LOCAL)] = has_abstraction
        if self.config.cognitive.enable_abstraction_validity_control:
            allowed[:, int(InternalAction.TEST_APPLICABILITY)] = has_abstraction
        if (
            self.config.cognitive.enable_agent_session_loop
            or self.config.cognitive.enable_viability_gate
        ):
            allowed[:, int(InternalAction.INSPECT_SELF_STATE)] = True
            # These operations create model-owned *requests*.  They never call
            # a tool or mutate an application-owned artifact directly.
            allowed[:, int(InternalAction.CREATE_EVIDENCE_REQUEST)] = node_count > 0
            allowed[:, int(InternalAction.QUERY_TOOL)] = node_count > 0
        if self.config.cognitive.enable_multi_hypothesis_planning:
            allowed[:, int(InternalAction.UPDATE_HYPOTHESES)] = hypothesis_count > 0
            allowed[:, int(InternalAction.GENERATE_ACTION_CANDIDATES)] = workspace.active.any(-1)
            allowed[:, int(InternalAction.EVALUATE_CANDIDATES)] = hypothesis_count > 0
        if self.config.cognitive.enable_abstraction_validity_control:
            allowed[:, int(InternalAction.REVISE_ABSTRACTION)] = has_abstraction
        return allowed

    @staticmethod
    def _metacognitive_action_bias(
        prediction: MetacognitivePrediction,
    ) -> Tensor:
        """Map predicted marginal operation value to a bounded advisory bias.

        Hard availability, permission, provenance, viability, and microstep
        masks remain authoritative and are applied separately by the
        controller.  Centering makes the signal comparative within each row;
        clipping prevents an uncalibrated self-model from dominating learned
        controller logits.
        """

        retrieval = {
            InternalAction.RETRIEVE_RECENT, InternalAction.RETRIEVE_EPISODIC,
            InternalAction.RETRIEVE_SEMANTIC, InternalAction.EXPAND_ASSOCIATION,
        }
        reconstruction = {
            InternalAction.RECONSTRUCT_LOCAL, InternalAction.TEST_APPLICABILITY,
            InternalAction.REVISE_ABSTRACTION, InternalAction.DECOMPRESS,
        }
        simulation = {
            InternalAction.SIMULATE, InternalAction.GENERATE_ACTION_CANDIDATES,
            InternalAction.EVALUATE_CANDIDATES,
        }
        evidence = {
            InternalAction.VERIFY, InternalAction.CREATE_EVIDENCE_REQUEST,
            InternalAction.QUERY_TOOL,
            InternalAction.ABSTAIN_OR_REQUEST_EXTERNAL_EVIDENCE,
        }
        values = []
        for action in InternalAction:
            if action == InternalAction.HALT:
                value = 1 / (1 + prediction.value_of_compute)
            elif action in retrieval:
                value = prediction.value_of_retrieval
            elif action in reconstruction:
                value = prediction.value_of_reconstruction
            elif action in simulation:
                value = prediction.value_of_simulation
            elif action in evidence:
                value = prediction.value_of_evidence
            else:
                value = prediction.value_of_compute
            values.append(value)
        raw = torch.stack(values, -1).clamp_min(0).log1p()
        centered = raw - raw.mean(-1, keepdim=True)
        normalized = centered / raw.std(-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        return (0.5 * normalized).clamp(-1.0, 1.0)

    def _run_internal_actions(
        self, nodes: NodeSlots, relations: RelationSlots,
        workspace: GlobalWorkspaceState, hypotheses: HypothesisState,
        episodic: TensorMemoryState, semantic: TensorMemoryState,
        relational_context: Tensor, selected_scale: Tensor,
        goal_state: GoalState, system_state: SystemModelState,
        ledger: ProvenanceLedger, *, knowledge: KnowledgeProposalState | None = None,
        reconstructions: ReconstructionState | None = None,
        abstraction_validity: AbstractionValidityState | None = None,
        evidence_requests: EvidenceRequestState | None = None,
        controller_state: ControllerState | None = None,
        external_action: ExternalActionDecision | None = None,
        metacognitive_prediction: MetacognitivePrediction | None = None,
        timestamps: Tensor, active_rows: Tensor,
        scale_contexts: Tensor, scale_context_mask: Tensor,
    ) -> tuple[
        NodeSlots, RelationSlots, HypothesisState, TensorMemoryState,
        TensorMemoryState, Tensor, Tensor, ControllerState,
        KnowledgeProposalState, EvidenceRequestState, ReconstructionState,
        AbstractionValidityState,
        CognitiveActionReceipts,
    ]:
        batch = nodes.batch
        if knowledge is None:
            knowledge = KnowledgeProposalState.empty(
                batch, self.config.cognitive.knowledge_candidate_capacity,
                self.config.cognitive.workspace_dim,
                self.config.cognitive.knowledge_support_capacity,
                device=nodes.content.device, dtype=nodes.content.dtype,
            )
        if reconstructions is None:
            reconstructions = ReconstructionState.empty(
                batch, self.config.cognitive.reconstruction_capacity,
                self.config.cognitive.workspace_dim,
                self.config.cognitive.uncertainty_channels,
                device=nodes.content.device, dtype=nodes.content.dtype,
            )
        if abstraction_validity is None:
            abstraction_validity = AbstractionValidityState.empty(
                batch, self.config.cognitive.reconstruction_capacity,
                device=nodes.content.device, dtype=nodes.content.dtype,
            )
        if evidence_requests is None:
            evidence_requests = EvidenceRequestState.empty(
                batch, self.config.cognitive.evidence_request_capacity,
                self.config.cognitive.workspace_dim,
                self.config.cognitive.hypothesis_slots,
                self.config.cognitive.knowledge_support_capacity,
                device=nodes.content.device, dtype=nodes.content.dtype,
            )
        if controller_state is None:
            controller_state = self.controller.initial_state(
                batch, device=nodes.content.device, dtype=nodes.content.dtype
            )
        if external_action is None:
            external_action = ExternalActionDecision.empty(
                batch, self.config.cognitive.system_action_channels,
                self.config.cognitive.knowledge_support_capacity,
                device=nodes.content.device, dtype=nodes.content.dtype,
            )
        relational_context = relational_context.clone()
        selected_scale = selected_scale.clone()
        controller_state = self.controller.begin_cycle(controller_state, active_rows)
        actions = torch.full(
            (batch, self.config.cognitive.maximum_cognitive_steps), -1,
            dtype=torch.int64, device=nodes.content.device,
        )
        statuses = torch.full_like(actions, int(ActionStatus.HALTED))
        success = torch.zeros_like(actions, dtype=torch.bool)
        node_receipts = torch.full_like(actions, -1)
        relation_receipts = torch.full_like(actions, -1)
        knowledge_receipts = torch.full_like(actions, -1)
        secondary_node_receipts = torch.full_like(actions, -1)
        secondary_relation_receipts = torch.full_like(actions, -1)
        argument_schema_receipts = torch.full_like(actions, -1)
        trigger_receipts = torch.zeros_like(actions)
        argument_receipts = nodes.content.new_zeros(
            batch, self.config.cognitive.maximum_cognitive_steps,
            self.config.cognitive.action_argument_dim,
        )
        argument_mask_receipts = torch.zeros_like(argument_receipts, dtype=torch.bool)
        operation_cost_receipts = nodes.content.new_zeros(
            batch, self.config.cognitive.maximum_cognitive_steps
        )
        receipt_mask = torch.zeros_like(actions, dtype=torch.bool)
        action_logits = nodes.content.new_zeros(
            batch, self.config.cognitive.maximum_cognitive_steps, len(InternalAction)
        )
        relation_logits = nodes.content.new_zeros(
            batch, self.config.cognitive.maximum_cognitive_steps, len(RelationFamily)
        )
        halt_probability = nodes.content.new_ones(
            batch, self.config.cognitive.maximum_cognitive_steps
        )
        workspace_summary = self._workspace_summary(workspace)
        metacognitive_bias = (
            self._metacognitive_action_bias(metacognitive_prediction)
            if metacognitive_prediction is not None else None
        )
        for step_index in range(self.config.cognitive.maximum_cognitive_steps):
            if not bool((~controller_state.halted).any()):
                break
            node_uncertainty = (
                (nodes.uncertainty * nodes.active.unsqueeze(-1)).sum(1)
                / nodes.active.sum(1, keepdim=True).clamp_min(1)
            )
            decision, controller_state = self.controller.step(
                controller_state, workspace_summary, goal_state.summary(), node_uncertainty,
                system_state.features(), nodes.content, nodes.active,
                relations=relations.content, relation_mask=relations.active,
                action_mask=self._action_availability(
                    nodes, relations, hypotheses, episodic, semantic, workspace,
                    selected_scale,
                ),
                action_bias=metacognitive_bias,
            )
            active = decision.active
            action_logits[:, step_index] = decision.action_logits
            relation_logits[:, step_index] = decision.relation_logits
            halt_probability[:, step_index] = decision.halt_probability
            actions[:, step_index] = decision.action
            node_receipts[:, step_index] = decision.node_pointer
            relation_receipts[:, step_index] = decision.relation_pointer
            secondary_node_receipts[:, step_index] = decision.secondary_node_pointer
            secondary_relation_receipts[:, step_index] = decision.secondary_relation_pointer
            argument_schema_receipts[:, step_index] = decision.argument_schema_id
            argument_receipts[:, step_index] = decision.arguments
            argument_mask_receipts[:, step_index] = decision.argument_mask
            trigger_receipts[:, step_index] = decision.trigger_class
            operation_cost_receipts[:, step_index] = decision.expected_operation_cost
            receipt_mask[:, step_index] = active
            statuses[:, step_index] = torch.where(
                active, torch.full_like(statuses[:, step_index], int(ActionStatus.SUCCESS)),
                statuses[:, step_index],
            )
            success[:, step_index] = active
            for row in torch.nonzero(active, as_tuple=False).flatten().tolist():
                action = InternalAction(int(decision.action[row].item()))
                node_pointer = int(decision.node_pointer[row].item())
                relation_pointer = int(decision.relation_pointer[row].item())
                if action == InternalAction.HALT:
                    statuses[row, step_index] = int(ActionStatus.HALTED)
                    continue
                if action == InternalAction.BIND:
                    if node_pointer < 0 or int(nodes.active[row].sum()) < 2:
                        success[row, step_index] = False
                        statuses[row, step_index] = int(ActionStatus.NO_TARGET)
                        continue
                    target = int(decision.secondary_node_pointer[row].item())
                    if target < 0 or target == node_pointer or not bool(nodes.active[row, target]):
                        candidates = torch.nonzero(nodes.active[row], as_tuple=False).flatten()
                        candidates = candidates[candidates != node_pointer]
                        similarity = F.cosine_similarity(
                            nodes.content[row, node_pointer][None], nodes.content[row, candidates], dim=-1
                        )
                        target = int(candidates[similarity.argmax()].item())
                    family = int(decision.relation_family[row].item())
                    source_type = int(nodes.type_logits[row, node_pointer].argmax())
                    target_type = int(nodes.type_logits[row, target].argmax())
                    if not bool(RELATION_COMPATIBILITY[source_type, target_type, family]):
                        success[row, step_index] = False
                        statuses[row, step_index] = int(ActionStatus.INCOMPATIBLE)
                        continue
                    proposal_content = self.compare_projection(torch.cat((
                        nodes.content[row, node_pointer], nodes.content[row, target],
                        nodes.content[row, target] - nodes.content[row, node_pointer],
                        nodes.content[row, target] * nodes.content[row, node_pointer],
                    )))
                    shape = (batch, 1, 1)
                    proposal_active = torch.zeros(shape, dtype=torch.bool, device=nodes.content.device)
                    participant = torch.full((*shape, 2), -1, dtype=torch.int64, device=nodes.content.device)
                    participant[row, 0, 0] = torch.tensor([node_pointer, target], device=nodes.content.device)
                    proposal = RelationProposals(
                        nodes.content.new_zeros(batch, 1, 1, nodes.content.shape[-1]),
                        torch.full(shape, -1, dtype=torch.int64, device=nodes.content.device),
                        participant,
                        torch.tensor([0, 1], device=nodes.content.device).view(1, 1, 1, 2).expand(*shape, 2),
                        proposal_active.unsqueeze(-1).expand(*shape, 2),
                        nodes.content.new_zeros(*shape, 3), nodes.content.new_zeros(shape),
                        torch.full((*shape, 2), -1, dtype=torch.int64, device=nodes.content.device),
                        torch.full(shape, -1, dtype=torch.int64, device=nodes.content.device),
                        torch.full(shape, -1, dtype=torch.int64, device=nodes.content.device),
                        proposal_active,
                    )
                    # Populate the single authoritative proposal without touching other rows.
                    content_value = proposal.content.clone(); content_value[row, 0, 0] = proposal_content
                    family_value = proposal.family_ids.clone(); family_value[row, 0, 0] = family
                    support_value = proposal.support.clone()
                    support_value[row, 0, 0] = torch.stack((
                        torch.minimum(nodes.support[row, node_pointer, 0], nodes.support[row, target, 0]),
                        torch.maximum(nodes.support[row, node_pointer, 1], nodes.support[row, target, 1]),
                        torch.maximum(nodes.support[row, node_pointer, 2], nodes.support[row, target, 2]),
                    ))
                    confidence_value = proposal.confidence.clone(); confidence_value[row, 0, 0] = 1
                    parents = proposal.parent_provenance_ids.clone()
                    parents[row, 0, 0] = nodes.provenance_ids[row, [node_pointer, target]]
                    scenarios = proposal.scenario_ids.clone(); scenarios[row, 0, 0] = nodes.scenario_ids[row, node_pointer]
                    participant_mask = proposal.participant_mask.clone()
                    participant_mask[row, 0, 0] = True
                    final_active = proposal_active.clone(); final_active[row, 0, 0] = True
                    proposal = RelationProposals(
                        content_value, family_value, participant, proposal.participant_roles,
                        participant_mask, support_value, confidence_value, parents,
                        proposal.provenance_ids, scenarios, final_active,
                    )
                    proposal = self._derive_relation_provenance(proposal, relations, nodes, ledger)
                    relations = self.relation_writer(relations, proposal, nodes)
                elif action == InternalAction.UNBIND:
                    if relation_pointer < 0 or not bool(relations.active[row, relation_pointer]):
                        success[row, step_index] = False; statuses[row, step_index] = int(ActionStatus.NO_TARGET)
                    else:
                        selection = torch.zeros_like(relations.active); selection[row, relation_pointer] = True
                        relations = self._clear_relation_rows(relations, selection)
                elif action == InternalAction.RETYPE_RELATION:
                    if relation_pointer < 0 or not bool(relations.active[row, relation_pointer]):
                        success[row, step_index] = False; statuses[row, step_index] = int(ActionStatus.NO_TARGET)
                    else:
                        family = int(decision.relation_family[row].item())
                        participants = relations.participant_indices[row, relation_pointer, :2]
                        types = nodes.type_logits[row, participants].argmax(-1)
                        if not bool(RELATION_COMPATIBILITY[types[0], types[1], family]):
                            success[row, step_index] = False; statuses[row, step_index] = int(ActionStatus.INCOMPATIBLE)
                        else:
                            old = int(relations.provenance_ids[row, relation_pointer])
                            support = relations.support[row, relation_pointer]
                            provenance = ledger.derive(
                                [old], source_class=SourceClass.INFERRED,
                                operator=f"mrcra:retype:{family}",
                                support=SupportInterval(*(float(value) for value in support.tolist())),
                                modality=ModalityClass.MEMORY,
                                scenario_id=int(relations.scenario_ids[row, relation_pointer]),
                                model_authority=self.model_authority,
                            )
                            values = {name: getattr(relations, name).clone() for name in relations.__dataclass_fields__}
                            values["type_logits"][row, relation_pointer].zero_(); values["type_logits"][row, relation_pointer, family] = 1
                            values["provenance_ids"][row, relation_pointer] = provenance
                            values["versions"][row, relation_pointer] += 1
                            relations = RelationSlots(**values)
                elif action == InternalAction.RETRIEVE_RECENT:
                    # Executed below against the temporal episodic buffer.
                    pass
                elif action == InternalAction.COMPARE:
                    active_nodes = torch.nonzero(nodes.active[row], as_tuple=False).flatten()
                    if node_pointer < 0 or active_nodes.numel() < 2:
                        success[row, step_index] = False; statuses[row, step_index] = int(ActionStatus.NO_TARGET)
                    else:
                        other = int(decision.secondary_node_pointer[row].item())
                        if other < 0 or other == node_pointer or not bool(nodes.active[row, other]):
                            other = int(active_nodes[active_nodes != node_pointer][0].item())
                        relational_context[row] = self.compare_projection(torch.cat((
                            nodes.content[row, node_pointer], nodes.content[row, other],
                            nodes.content[row, other] - nodes.content[row, node_pointer],
                            nodes.content[row, other] * nodes.content[row, node_pointer],
                        )))
                elif action == InternalAction.COMPRESS:
                    if not bool(nodes.active[row].any()):
                        success[row, step_index] = False; statuses[row, step_index] = int(ActionStatus.NO_TARGET)
                    else:
                        proposal = self.graph_compressor(self._graph_fragment(nodes, relations))
                        relational_context[row] = proposal.latent[row]
                        proposal_mask = torch.zeros(
                            batch, dtype=torch.bool, device=nodes.content.device
                        )
                        proposal_mask[row] = True
                        batch_proposal = self._knowledge_proposal(
                            kind=KnowledgeKind.ABSTRACTION,
                            latent=proposal.latent,
                            code_gain=proposal.code.gain_bits,
                            reconstruction_distortion=proposal.distortion.node,
                            relation_distortion=proposal.distortion.relation,
                            confidence=torch.exp(-proposal.distortion.total),
                            mask=proposal_mask, workspace=workspace, nodes=nodes,
                            ledger=ledger, operator="mrcra:compression_proposal",
                            timestamp=timestamps,
                        )
                        knowledge, written = self.knowledge_bank.propose(
                            knowledge, batch_proposal
                        )
                        knowledge_receipts[row, step_index] = written[row]
                        if int(written[row]) < 0:
                            success[row, step_index] = False
                            statuses[row, step_index] = int(ActionStatus.NO_TARGET)
                        else:
                            statuses[row, step_index] = int(ActionStatus.VALIDATION_REQUIRED)
                elif action == InternalAction.DECOMPRESS:
                    types = nodes.type_logits[row].argmax(-1)
                    candidate = nodes.active[row] & (
                        (types == int(NodeType.ABSTRACTION)) | (types == int(NodeType.INVARIANT))
                    )
                    if not bool(candidate.any()):
                        success[row, step_index] = False; statuses[row, step_index] = int(ActionStatus.NO_TARGET)
                    else:
                        latent_value = nodes.content[row, candidate][0][None]
                        reconstructed = self.graph_compressor.decode_latent(
                            latent_value, node_count=nodes.capacity,
                            relation_count=relations.capacity,
                        )[0]
                        relational_context[row] = reconstructed.mean(1)[0]
                elif action == InternalAction.RECONSTRUCT_LOCAL:
                    # Executed below as one masked batch so the learned
                    # reconstructor retains vectorized gradients and produces
                    # one atomic graph/ledger transaction per microstep.
                    pass
                elif action == InternalAction.TEST_APPLICABILITY:
                    pointer = int(decision.abstraction_pointer[row].item())
                    matching = (
                        abstraction_validity.active[row]
                        & (abstraction_validity.abstraction_node_indices[row] == pointer)
                    )
                    if pointer < 0 or not bool(matching.any()):
                        success[row, step_index] = False
                        statuses[row, step_index] = int(ActionStatus.NO_TARGET)
                    else:
                        latest = torch.nonzero(matching, as_tuple=False).flatten()
                        chosen = latest[
                            abstraction_validity.versions[row, latest].argmax()
                        ]
                        relational_context[row] = nodes.content[row, pointer] * (
                            abstraction_validity.applicability[row, chosen]
                            * abstraction_validity.calibrated_confidence[row, chosen]
                        )
                elif action == InternalAction.INSPECT_SELF_STATE:
                    relational_context[row] = self.self_model_projection(
                        system_state.features()[row]
                    )
                elif action == InternalAction.VERIFY:
                    success[row, step_index] = False
                    statuses[row, step_index] = int(ActionStatus.EXTERNAL_EVIDENCE_REQUIRED)
                elif action in {
                    InternalAction.CREATE_EVIDENCE_REQUEST,
                    InternalAction.QUERY_TOOL,
                }:
                    # The typed request is materialized below in one bounded
                    # batch operation.  QUERY_TOOL deliberately does not call
                    # an executor from inside the neural runtime.
                    statuses[row, step_index] = int(ActionStatus.EXTERNAL_EVIDENCE_REQUIRED)
                elif action == InternalAction.UPDATE_HYPOTHESES:
                    # Evidence likelihoods are updated once per observation
                    # before deliberation.  This receipt records that the
                    # current posterior was explicitly inspected/accepted.
                    if not bool(hypotheses.active[row].any()):
                        success[row, step_index] = False
                        statuses[row, step_index] = int(ActionStatus.NO_TARGET)
                elif action in {
                    InternalAction.GENERATE_ACTION_CANDIDATES,
                    InternalAction.EVALUATE_CANDIDATES,
                }:
                    # Candidate generation/evaluation is the post-deliberation
                    # production stage.  The receipt cannot bypass its hard
                    # permission, provenance, and viability gates.
                    if not bool(workspace.active[row].any()):
                        success[row, step_index] = False
                        statuses[row, step_index] = int(ActionStatus.NO_TARGET)
                elif action == InternalAction.REVISE_ABSTRACTION:
                    pointer = int(decision.abstraction_pointer[row].item())
                    matching = abstraction_validity.active[row] & (
                        abstraction_validity.abstraction_node_indices[row] == pointer
                    )
                    if pointer < 0 or not bool(matching.any()):
                        success[row, step_index] = False
                        statuses[row, step_index] = int(ActionStatus.NO_TARGET)
                    else:
                        # Revision is conservative: counterevidence can only
                        # reduce live authority.  A new abstraction must be
                        # proposed and validated separately.
                        selected = torch.nonzero(matching, as_tuple=False).flatten()
                        selected = selected[
                            abstraction_validity.versions[row, selected].argmax()
                        ]
                        values = {
                            name: getattr(abstraction_validity, name).clone()
                            for name in abstraction_validity.__dataclass_fields__
                        }
                        values["applicability"][row, selected] = 0
                        values["calibrated_confidence"][row, selected] = 0
                        values["versions"][row, selected] += 1
                        abstraction_validity = AbstractionValidityState(**values)
                elif action == InternalAction.PROPOSE_INVARIANT:
                    if int(episodic.active[row].sum()) < 2:
                        success[row, step_index] = False; statuses[row, step_index] = int(ActionStatus.NO_TARGET)
                    elif self.config.cognitive.enable_integrated_invariant_discovery:
                        left, right = self._split_temporal_fragments(
                            self._graph_fragment(nodes, relations)
                        )
                        discovered = self.invariant_discoverer(left, right)
                        roots: set[int] = set()
                        for provenance_id in discovered.supporting_provenance_ids[row][
                            discovered.supporting_mask[row]
                        ].tolist():
                            roots.update(ledger.independent_roots(int(provenance_id)))
                        valid = bool(discovered.mask[row]) and len(roots) >= 2
                        if not valid:
                            success[row, step_index] = False
                            statuses[row, step_index] = int(ActionStatus.VALIDATION_REQUIRED)
                            continue
                        relational_context[row] = discovered.latent[row]
                        proposal_mask = torch.zeros(
                            batch, dtype=torch.bool, device=nodes.content.device
                        )
                        proposal_mask[row] = True
                        invariant_proposal = self._knowledge_proposal(
                            kind=KnowledgeKind.INVARIANT,
                            latent=discovered.latent,
                            code_gain=discovered.code_gain_bits,
                            reconstruction_distortion=discovered.reconstruction_distortion,
                            relation_distortion=discovered.relation_distortion,
                            confidence=(
                                discovered.applicability_probability
                                * torch.exp(-discovered.match_cost)
                            ),
                            mask=proposal_mask, workspace=workspace, nodes=nodes,
                            ledger=ledger,
                            operator="mrcra:role_normalized_invariant_proposal:v1",
                            timestamp=timestamps,
                        )
                        knowledge, written = self.knowledge_bank.propose(
                            knowledge, invariant_proposal
                        )
                        knowledge_receipts[row, step_index] = written[row]
                        if int(written[row]) < 0:
                            success[row, step_index] = False
                            statuses[row, step_index] = int(ActionStatus.NO_TARGET)
                        else:
                            statuses[row, step_index] = int(ActionStatus.VALIDATION_REQUIRED)
                    else:
                        selected_memory = episodic.values[row, episodic.active[row]]
                        memory_content = selected_memory.mean(0)
                        relational_context[row] = self.compare_projection(torch.cat((
                            memory_content, workspace_summary[row],
                            memory_content - workspace_summary[row],
                            memory_content * workspace_summary[row],
                        )))
                        count = episodic.active.sum(-1).to(nodes.content.dtype)
                        reconstruction = nodes.content.new_zeros(batch)
                        reconstruction[row] = (
                            selected_memory - memory_content
                        ).square().mean()
                        code_gain = (
                            (count - 1).clamp_min(0)
                            * nodes.content.shape[-1] * 16
                        )
                        proposal_mask = torch.zeros(
                            batch, dtype=torch.bool, device=nodes.content.device
                        )
                        proposal_mask[row] = True
                        invariant_proposal = self._knowledge_proposal(
                            kind=KnowledgeKind.INVARIANT,
                            latent=relational_context,
                            code_gain=code_gain,
                            reconstruction_distortion=reconstruction,
                            relation_distortion=nodes.content.new_zeros(batch),
                            confidence=torch.exp(-reconstruction),
                            mask=proposal_mask, workspace=workspace, nodes=nodes,
                            ledger=ledger, operator="mrcra:invariant_proposal",
                            timestamp=timestamps,
                        )
                        knowledge, written = self.knowledge_bank.propose(
                            knowledge, invariant_proposal
                        )
                        knowledge_receipts[row, step_index] = written[row]
                        if int(written[row]) < 0:
                            success[row, step_index] = False
                            statuses[row, step_index] = int(ActionStatus.NO_TARGET)
                        else:
                            statuses[row, step_index] = int(ActionStatus.VALIDATION_REQUIRED)
                elif action == InternalAction.DESCEND_SCALE:
                    selected_scale[row] = max(0, int(selected_scale[row]) - 1)
                    scale = int(selected_scale[row])
                    if bool(scale_context_mask[row, scale]):
                        relational_context[row] = scale_contexts[row, scale]
                elif action == InternalAction.ASCEND_SCALE:
                    selected_scale[row] = min(
                        self.config.carrier.scales - 1, int(selected_scale[row]) + 1
                    )
                    scale = int(selected_scale[row])
                    if bool(scale_context_mask[row, scale]):
                        relational_context[row] = scale_contexts[row, scale]
                elif action == InternalAction.ABSTAIN_OR_REQUEST_EXTERNAL_EVIDENCE:
                    statuses[row, step_index] = int(ActionStatus.EXTERNAL_EVIDENCE_REQUIRED)
            # Batched actions whose kernels already enforce masks.
            episodic_retrieve = active & (decision.action == int(InternalAction.RETRIEVE_EPISODIC))
            recent_retrieve = active & (decision.action == int(InternalAction.RETRIEVE_RECENT))
            semantic_retrieve = active & (decision.action == int(InternalAction.RETRIEVE_SEMANTIC))
            expand = active & (decision.action == int(InternalAction.EXPAND_ASSOCIATION))
            reconstruct = active & (
                decision.action == int(InternalAction.RECONSTRUCT_LOCAL)
            )
            request_evidence = active & (
                (decision.action == int(InternalAction.VERIFY))
                | (decision.action == int(InternalAction.CREATE_EVIDENCE_REQUEST))
                | (decision.action == int(InternalAction.QUERY_TOOL))
                | (
                    decision.action
                    == int(InternalAction.ABSTAIN_OR_REQUEST_EXTERNAL_EVIDENCE)
                )
            )
            inspect_self = active & (
                decision.action == int(InternalAction.INSPECT_SELF_STATE)
            )
            if bool(inspect_self.any()):
                supporters, supporter_mask = self._workspace_supporters(
                    workspace, nodes
                )
                reflective_mask = inspect_self & supporter_mask.any(-1)
                content = self.self_model_projection(system_state.features())[:, None]
                spectral = self.spectral_projection(content[:, 0]).reshape(
                    batch, 1, self.config.cognitive.relation_heads,
                    self.config.cognitive.relation_modes, 2,
                )
                spectral = spectral / spectral.square().sum(
                    (-1, -2, -3), keepdim=True
                ).sqrt().clamp_min(1e-6)
                types = content.new_zeros(batch, 1, len(NodeType))
                types[:, 0, int(NodeType.SYSTEM_STATE)] = 1
                support = torch.stack((timestamps, timestamps, timestamps), -1)[:, None]
                ids = torch.full(
                    (batch, 1), -1, dtype=torch.int64,
                    device=nodes.content.device,
                )
                for row in torch.nonzero(reflective_mask, as_tuple=False).flatten().tolist():
                    parents = supporters[row][supporter_mask[row]].tolist()
                    ids[row, 0] = ledger.derive(
                        parents, source_class=SourceClass.INFERRED,
                        operator="mrcra:reflective_system_state:v1",
                        support=SupportInterval(
                            float(timestamps[row]), float(timestamps[row]),
                            float(timestamps[row]),
                        ),
                        modality=ModalityClass.MEMORY, scenario_id=0,
                        model_authority=self.model_authority,
                    )
                reflective = EventCandidates(
                    content, F.normalize(content, dim=-1), spectral, types,
                    support,
                    torch.full((batch, 1), int(ModalityClass.MEMORY), dtype=torch.int64, device=nodes.content.device),
                    supporters[:, :1], ids,
                    torch.full((batch, 1), int(SourceClass.INFERRED), dtype=torch.int64, device=nodes.content.device),
                    torch.zeros(batch, 1, dtype=torch.int64, device=nodes.content.device),
                    content.new_zeros(batch, 1, self.config.cognitive.uncertainty_channels),
                    content.new_ones(batch, 1), reflective_mask[:, None],
                )
                nodes = self.event_allocator(nodes, reflective)
                nodes = self._refresh_node_provenance_features(nodes, ledger)
                blocked = inspect_self & ~reflective_mask
                success[blocked, step_index] = False
                statuses[blocked, step_index] = int(ActionStatus.NO_TARGET)
            if bool(request_evidence.any()):
                supporters, supporter_mask = self._workspace_supporters(
                    workspace, nodes
                )
                proposition = workspace_summary.clone()
                safe_pointer = decision.node_pointer.clamp(0, nodes.capacity - 1)
                pointer_valid = (
                    (decision.node_pointer >= 0)
                    & nodes.active[
                        torch.arange(batch, device=nodes.content.device), safe_pointer
                    ]
                )
                proposition = torch.where(
                    pointer_valid[:, None],
                    nodes.content[
                        torch.arange(batch, device=nodes.content.device), safe_pointer
                    ],
                    proposition,
                )
                hypothesis_indices = torch.arange(
                    hypotheses.capacity, dtype=torch.int64,
                    device=nodes.content.device,
                )[None].expand(batch, -1)
                tool_request = (
                    decision.action == int(InternalAction.QUERY_TOOL)
                )
                tool_schema = torch.where(
                    tool_request,
                    decision.argument_schema_id,
                    torch.full_like(decision.argument_schema_id, -1),
                )
                evidence_requests, request_slots = create_evidence_request(
                    evidence_requests,
                    proposition=proposition,
                    requested_modality=torch.full(
                        (batch,), int(ModalityClass.SENSOR), dtype=torch.int64,
                        device=nodes.content.device,
                    ),
                    tool_schema_id=tool_schema,
                    hypothesis_indices=hypothesis_indices,
                    hypothesis_mask=hypotheses.active,
                    expected_information_gain=(
                        hypotheses.effective_count - 1
                    ).clamp_min(0),
                    maximum_cost=decision.expected_operation_cost.clamp_min(0),
                    maximum_latency=torch.ones(
                        batch, device=nodes.content.device,
                        dtype=nodes.content.dtype,
                    ),
                    required_precision=decision.precision_tolerance.clamp_min(0),
                    supporting_provenance_ids=supporters,
                    supporting_mask=supporter_mask,
                    create_mask=request_evidence & supporter_mask.any(-1),
                )
                blocked = request_evidence & (request_slots < 0)
                success[blocked, step_index] = False
                statuses[blocked, step_index] = int(ActionStatus.CAPACITY_BLOCKED)
            if bool(reconstruct.any()):
                abstraction_pointer = torch.where(
                    decision.abstraction_pointer >= 0,
                    decision.abstraction_pointer,
                    decision.node_pointer,
                )
                (
                    nodes, relations, reconstructions, abstraction_validity,
                    context, reconstructed,
                ) = self._reconstruct_local_graph(
                    nodes, relations, workspace, hypotheses, episodic,
                    reconstructions, abstraction_validity,
                    abstraction_pointer, reconstruct, goal_state.summary(), ledger,
                    target_physical_scale=decision.requested_physical_scale,
                    target_abstraction_depth=decision.requested_abstraction_depth,
                    precision_tolerance=decision.precision_tolerance,
                )
                relational_context = torch.where(
                    reconstructed[:, None], context, relational_context
                )
                missing = reconstruct & ~reconstructed
                success[missing, step_index] = False
                statuses[missing, step_index] = int(ActionStatus.NO_TARGET)
            if bool(recent_retrieve.any()):
                episodic, recent_context, found = self._retrieve_recent_buffer(
                    episodic, recent_retrieve
                )
                relational_context = torch.where(
                    found[:, None], recent_context, relational_context
                )
                missing = recent_retrieve & ~found
                success[missing, step_index] = False
                statuses[missing, step_index] = int(ActionStatus.EMPTY_MEMORY)
            if bool(episodic_retrieve.any()):
                episodic, nodes, context, found = self._retrieve_memory_action(
                    episodic, nodes, decision.node_pointer, episodic_retrieve,
                    ledger, timestamps=timestamps,
                )
                relational_context = torch.where(found[:, None], context, relational_context)
                missing = episodic_retrieve & ~found
                success[missing, step_index] = False; statuses[missing, step_index] = int(ActionStatus.EMPTY_MEMORY)
            if bool(semantic_retrieve.any()):
                semantic, nodes, context, found = self._retrieve_memory_action(
                    semantic, nodes, decision.node_pointer, semantic_retrieve,
                    ledger, timestamps=timestamps,
                )
                relational_context = torch.where(found[:, None], context, relational_context)
                missing = semantic_retrieve & ~found
                success[missing, step_index] = False; statuses[missing, step_index] = int(ActionStatus.EMPTY_MEMORY)
            if bool(expand.any()):
                use_semantic = expand & (decision.memory_tier == int(MemoryTier.SEMANTIC))
                use_episodic = expand & ~use_semantic
                context = torch.zeros_like(relational_context)
                found = torch.zeros(batch, dtype=torch.bool, device=nodes.content.device)
                if bool(use_episodic.any()):
                    episodic_context, episodic_found = self._association_context(
                        episodic, nodes, decision.node_pointer, use_episodic,
                        timestamps=timestamps,
                    )
                    context = torch.where(episodic_found[:, None], episodic_context, context)
                    found |= episodic_found
                if bool(use_semantic.any()):
                    semantic_context, semantic_found = self._association_context(
                        semantic, nodes, decision.node_pointer, use_semantic,
                        timestamps=timestamps,
                    )
                    context = torch.where(semantic_found[:, None], semantic_context, context)
                    found |= semantic_found
                relational_context = torch.where(found[:, None], context, relational_context)
                missing = expand & ~found
                success[missing, step_index] = False
                statuses[missing, step_index] = int(ActionStatus.EMPTY_MEMORY)
            write_episode = active & (decision.action == int(InternalAction.WRITE_EPISODE))
            if bool(write_episode.any()):
                episodic, written = self._write_memory_action(
                    episodic, nodes, decision.node_pointer, write_episode,
                    tier=MemoryTier.EPISODIC, ledger=ledger,
                )
                missing = write_episode & ~written
                success[missing, step_index] = False; statuses[missing, step_index] = int(ActionStatus.NO_TARGET)
            create = active & (decision.action == int(InternalAction.CREATE_HYPOTHESIS))
            if bool(create.any()):
                hypotheses = self.hypothesis_bank.create(hypotheses, workspace_summary, create)
            merge = active & (decision.action == int(InternalAction.MERGE_HYPOTHESES))
            if bool(merge.any()):
                merged = self.hypothesis_bank.merge_duplicates(hypotheses)
                hypotheses = _replace_rows(hypotheses, merged, merge)
            prune = active & (decision.action == int(InternalAction.PRUNE_HYPOTHESIS))
            if bool(prune.any()):
                pruned = self.hypothesis_bank.prune(hypotheses)
                hypotheses = _replace_rows(hypotheses, pruned, prune)
            simulate = active & (decision.action == int(InternalAction.SIMULATE))
            if bool(simulate.any()):
                needs = simulate & ~hypotheses.active.any(-1)
                if bool(needs.any()):
                    hypotheses = self.hypothesis_bank.create(hypotheses, workspace_summary, needs)
                action_vector = F.one_hot(
                    external_action.selected_action.clamp_min(0),
                    self.config.cognitive.system_action_channels,
                ).to(nodes.content.dtype)[:, None]
                if self.config.cognitive.enable_multi_hypothesis_planning:
                    routed = self.hypothesis_bank.route(
                        hypotheses, min(
                            hypotheses.capacity,
                            self.config.cognitive.planning_hypothesis_top_k,
                        ),
                    )
                    safe_route = routed.indices.clamp_min(0)
                    route_rows = torch.arange(
                        batch, device=nodes.content.device
                    )[:, None]
                    routed_residuals = hypotheses.residuals[
                        route_rows, safe_route
                    ]
                    routed_scenarios = hypotheses.scenario_ids[
                        route_rows, safe_route
                    ]
                    rollout = self.world_model.rollout_candidates(
                        workspace_summary, self._relation_summary(relations)[0],
                        routed_residuals, routed_scenarios,
                        routed.mask, action_vector, simulate[:, None],
                    )
                    posterior = (
                        hypotheses.weights[route_rows, safe_route] * routed.mask
                    )
                    posterior = posterior / posterior.sum(
                        -1, keepdim=True
                    ).clamp_min(1e-8)
                    posterior = posterior[:, :, None]
                    simulated_context = (
                        rollout.latent_states[:, :, 0, -1]
                        * posterior
                    ).sum(1)
                else:
                    first = hypotheses.weights.argmax(-1)
                    bindex = torch.arange(batch, device=nodes.content.device)
                    residual = hypotheses.residuals[bindex, first][:, None]
                    scenario = hypotheses.scenario_ids[bindex, first][:, None].clamp_min(1)
                    rollout = self.world_model.rollout(
                        workspace_summary, workspace_summary, residual, scenario,
                        action_vector[:, None], simulate[:, None, None],
                    )
                    simulated_context = rollout.latent_states[:, 0, 0]
                relational_context = torch.where(
                    simulate[:, None], simulated_context, relational_context
                )
        receipts = CognitiveActionReceipts(
            actions, statuses, success, node_receipts, relation_receipts,
            knowledge_receipts, receipt_mask, action_logits, relation_logits,
            halt_probability, secondary_node_receipts, secondary_relation_receipts,
            argument_schema_receipts, argument_receipts, argument_mask_receipts,
            trigger_receipts, operation_cost_receipts,
        )
        return (
            nodes, relations, hypotheses, episodic, semantic, relational_context,
            selected_scale, controller_state, knowledge, evidence_requests,
            reconstructions, abstraction_validity, receipts,
        )

    def step(
        self, packet: ObservationPacket, index: int, state: MRCRARuntimeState,
        ledger: ProvenanceLedger, *, goals: GoalState | None = None,
        system_model: SystemModelState | None = None,
        project_output: bool = True,
        force_cognitive_cycle: bool = False,
        allow_hard_event_allocation: bool = True,
    ) -> MRCRAStepOutput:
        if not 0 <= index < packet.length or packet.batch != state.batch:
            raise ValueError("MRCRA step index or batch is invalid")
        if index == 0:
            packet.assert_ledger_consistent(ledger, allow_internal=True)
        boundary = packet.boundary_classes[:, index]
        state = self._reset_boundaries(state, boundary, packet.sample_intervals)
        goal_state = state.goals if goals is None else goals
        system_state = state.system_model if system_model is None else system_model
        if goal_state.desired_outcomes.shape[0] != state.batch:
            raise ValueError("goal batch does not match MRCRA runtime state")
        if system_state.modality_availability.shape[0] != state.batch:
            raise ValueError("system-model batch does not match MRCRA runtime state")
        predictions, latent_rows, carrier_states = [], [], []
        soft = boundary == int(BoundaryClass.SOFT)
        for batch_index, carrier_state in enumerate(state.carrier):
            if not bool(packet.valid_mask[batch_index, index]):
                prediction_width = self.config.carrier.resolved_output_dim if project_output else 0
                predictions.append(packet.values.new_zeros(1, prediction_width))
                latent_rows.append(packet.values.new_zeros(1, self.config.cognitive.workspace_dim))
                carrier_states.append(carrier_state)
                continue
            result = self.carrier.step(
                packet.values[batch_index : batch_index + 1, index], carrier_state,
                packet.valid_mask[batch_index : batch_index + 1, index],
                soft_boundary=soft[batch_index : batch_index + 1],
                relational_context=state.relational_context[batch_index : batch_index + 1],
                project_output=project_output,
            )
            predictions.append(result.prediction)
            if result.latent is None:
                raise RuntimeError("causal carrier step did not expose its latent state")
            latent_rows.append(result.latent)
            carrier_states.append(result.state)
        latent = torch.cat(latent_rows, 0)
        prediction = torch.cat(predictions, 0)
        output_latent = latent
        scale_contexts = latent.new_zeros(
            state.batch, self.config.carrier.scales, self.config.cognitive.workspace_dim
        )
        scale_context_mask = torch.zeros(
            state.batch, self.config.carrier.scales, dtype=torch.bool, device=latent.device
        )
        for batch_index, carrier_state in enumerate(carrier_states):
            for scale, band in enumerate(carrier_state.latest_bands):
                if band is not None:
                    scale_contexts[batch_index, scale] = self.carrier.synthesis_adapters[scale](
                        band.data[:, 0]
                    )[0]
                    scale_context_mask[batch_index, scale] = True
        uncertainty, distributional_prediction = self._decomposed_uncertainty(
            latent, state, packet, index, ledger,
        )
        spectral = self.spectral_projection(latent).reshape(
            state.batch, 1, self.config.cognitive.relation_heads,
            self.config.cognitive.relation_modes, 2,
        )
        spectral = spectral / spectral.square().sum((-1, -2, -3), keepdim=True).sqrt().clamp_min(1e-6)
        prediction_error = (latent - state.predicted_next_latent).square().mean(-1, keepdim=True)
        evidence = EventEvidence(
            latent[:, None], latent.square().mean(-1, keepdim=True)[:, None],
            prediction_error[:, None],
            (latent - state.previous_latent).square().mean(-1, keepdim=True)[:, None],
            uncertainty[:, None],
            goal_state.summary(),
            boundary[:, None], packet.timestamps[:, index : index + 1],
            packet.modality_ids[:, index : index + 1],
            packet.source_record_ids[:, index : index + 1],
            torch.zeros_like(packet.source_record_ids[:, index : index + 1]),
            spectral, packet.valid_mask[:, index : index + 1],
        )
        (
            events, event_state, event_proposals, event_transition_receipts,
        ) = self.event_extractor.extract_with_proposals(
            evidence, state.event_extractor
        )
        nodes = state.nodes
        relations = state.relations
        if allow_hard_event_allocation and bool(events.active.any()):
            events = self._derive_event_provenance(events, ledger)
            nodes = self.event_allocator(nodes, events)
            nodes = self._refresh_node_provenance_features(nodes, ledger)
            relations = invalidate_stale_relations(relations, nodes)
        cycle_mask = packet.valid_mask[:, index] & (
            force_cognitive_cycle
            | (event_state.position % self.config.cognitive.event_chunk_size == 0)
            | (boundary != int(BoundaryClass.NONE))
            | events.active.any(-1)
        )
        graph_output = None
        workspace = state.workspace
        relational_context = state.relational_context
        controller_state = state.controller
        hypotheses = state.hypotheses
        episodic_memory = state.episodic_memory
        semantic_memory = state.semantic_memory
        knowledge = state.knowledge
        reconstructions = state.reconstructions
        abstraction_validity = state.abstraction_validity
        action_candidates = state.action_candidates
        evidence_requests = state.evidence_requests
        metacognition = state.metacognition
        schemas = state.schemas
        selected_scale = state.selected_physical_scale
        action_receipts = self._empty_action_receipts(state.batch, device=latent.device)
        external_action = ExternalActionDecision.empty(
            state.batch, self.config.cognitive.system_action_channels,
            self.config.cognitive.knowledge_support_capacity,
            device=latent.device, dtype=latent.dtype,
        )
        world_prediction = None
        schema_probabilities = schemas.probabilities
        symbol_gates = latent.new_zeros(
            state.batch, self.config.cognitive.knowledge_support_capacity
        )
        metacognitive_prediction: MetacognitivePrediction | None = None
        if bool(cycle_mask.any()):
            graph_output = self.workspace_graph(nodes, workspace, goal_context=goal_state.summary())
            nodes = _replace_rows(nodes, graph_output.nodes, cycle_mask)
            proposals = _mask_relation_proposals(
                graph_output.relation_proposals,
                cycle_mask[:, None, None]
                & (graph_output.relation_proposals.confidence >= self.relation_write_threshold),
            )
            if bool(proposals.active.any()):
                proposals = self._derive_relation_provenance(proposals, relations, nodes, ledger)
                relations = self.relation_writer(relations, proposals, nodes)
            workspace = _replace_rows(workspace, graph_output.workspace.state, cycle_mask)
            broadcast_context = graph_output.broadcast.controller_context + graph_output.broadcast.output_context
            workspace_summary = self._workspace_summary(workspace)
            schema_modulation, proposed_schemas = self.operational_schemas(
                workspace_summary, schemas,
            )
            schemas = _replace_rows(schemas, proposed_schemas, cycle_mask)
            schema_probabilities = schemas.probabilities
            symbol_modulation, symbol_gates = self._semantic_symbol_context(
                semantic_memory, workspace_summary,
            )
            integrated_context = (
                broadcast_context
                + self.config.cognitive.broadcast_gain_maximum
                * torch.tanh(schema_modulation + symbol_modulation)
            )
            relational_context = torch.where(
                cycle_mask[:, None], integrated_context, relational_context
            )
            if self.config.cognitive.enable_metacognitive_routing:
                metacognitive_prediction = self.metacognitive_router(
                    workspace_summary, relational_context, uncertainty,
                    system_state.features(),
                )
            if self.config.cognitive.enable_multi_hypothesis_planning:
                needs_unknown = cycle_mask & ~hypotheses.active.any(-1)
                if bool(needs_unknown.any()):
                    hypotheses = self.hypothesis_bank.create(
                        hypotheses, workspace_summary, needs_unknown
                    )
                hypotheses = self._update_hypotheses_from_observation(
                    hypotheses, latent, cycle_mask,
                    packet.source_record_ids[:, index],
                )
            if not self.config.cognitive.enable_post_deliberation_action_selection:
                supporters, supporter_mask = self._workspace_supporters(workspace, nodes)
                external_action = self.external_action_policy(
                    workspace_summary + schema_modulation + symbol_modulation,
                    goal_state, uncertainty, system_state,
                    supporting_provenance_ids=supporters,
                    supporting_mask=supporter_mask, active_mask=cycle_mask,
                    controller_abstained=controller_state.abstained,
                )
                external_action = authorize_external_actions(external_action, ledger)
            (
                nodes, relations, hypotheses, episodic_memory, semantic_memory,
                relational_context, selected_scale, controller_state, knowledge,
                evidence_requests, reconstructions, abstraction_validity,
                action_receipts,
            ) = self._run_internal_actions(
                nodes, relations, workspace, hypotheses, episodic_memory,
                semantic_memory, relational_context, selected_scale,
                goal_state, system_state, ledger,
                knowledge=knowledge, controller_state=controller_state,
                reconstructions=state.reconstructions,
                abstraction_validity=state.abstraction_validity,
                evidence_requests=state.evidence_requests,
                external_action=external_action,
                metacognitive_prediction=metacognitive_prediction,
                timestamps=packet.timestamps[:, index],
                active_rows=cycle_mask, scale_contexts=scale_contexts,
                scale_context_mask=scale_context_mask,
            )
            if self.config.cognitive.enable_post_deliberation_action_selection:
                authorized_goal = (
                    goal_state.mask & (goal_state.authority > 0)
                    & (goal_state.status == 1)
                ).any(-1)
                authorized_channel = (
                    system_state.permission_mask
                    & (system_state.action_availability > 0)
                ).any(-1)
                action_rows = cycle_mask & authorized_goal & authorized_channel
                # Proposal, rollout, and selection cannot yield an executable
                # action without both caller-owned goal authority and a
                # host-owned permitted capability.  Skip that entire branch
                # for ordinary perception cycles instead of paying for a
                # result that must be discarded.
                if bool(action_rows.any()):
                    action_candidates, external_action = self._post_deliberation_action(
                        context=relational_context, nodes=nodes, workspace=workspace,
                        relations=relations, hypotheses=hypotheses,
                        uncertainty=uncertainty, goals=goal_state,
                        system=system_state, active_rows=action_rows,
                        ledger=ledger, viability=state.viability,
                    )
            if bool((external_action.active & ~controller_state.abstained).any()):
                action_ids = external_action.selected_action.clamp_min(0)
                action_vector = F.one_hot(
                    action_ids, self.config.cognitive.system_action_channels
                ).to(latent.dtype)
                world_prediction = self.world_model(
                    workspace_summary, self._relation_summary(relations)[0],
                    action_vector,
                )
            if bool(controller_state.abstained.any()):
                values = {
                    field.name: getattr(external_action, field.name)
                    for field in fields(external_action)
                }
                values["abstained"] = external_action.abstained | controller_state.abstained
                values["active"] = external_action.active & ~controller_state.abstained
                values["authorized"] = external_action.authorized & ~controller_state.abstained
                values["selected_action"] = external_action.selected_action.masked_fill(
                    controller_state.abstained, -1
                )
                external_action = ExternalActionDecision(**values)
            if (
                self.config.cognitive.enable_agent_session_loop
                or self.config.cognitive.enable_viability_gate
            ):
                if metacognitive_prediction is None:
                    metacognitive_prediction = self.metacognitive_router(
                        self._workspace_summary(workspace), relational_context,
                        uncertainty, system_state.features(),
                    )
                meta_supporters, meta_support_mask = self._workspace_supporters(
                    workspace, nodes
                )
                meta_provenance = torch.full(
                    (state.batch,), -1, dtype=torch.int64, device=latent.device
                )
                meta_mask = cycle_mask & meta_support_mask.any(-1)
                for row in torch.nonzero(meta_mask, as_tuple=False).flatten().tolist():
                    parents = meta_supporters[row][meta_support_mask[row]].tolist()
                    timestamp = float(packet.timestamps[row, index])
                    meta_provenance[row] = ledger.derive(
                        parents, source_class=SourceClass.INFERRED,
                        operator="mrcra:metacognitive_receipt:v1",
                        support=SupportInterval(timestamp, timestamp, timestamp),
                        modality=ModalityClass.MEMORY, scenario_id=0,
                        model_authority=self.model_authority,
                    )
                trigger = torch.where(
                    prediction_error.squeeze(-1)
                    > self.config.cognitive.deliberation_prediction_error_threshold,
                    torch.full(
                        (state.batch,), int(CognitiveTrigger.PREDICTION_ERROR),
                        dtype=torch.int64, device=latent.device,
                    ),
                    torch.full(
                        (state.batch,), int(CognitiveTrigger.EVENT),
                        dtype=torch.int64, device=latent.device,
                    ),
                )
                metacognition = append_metacognitive_record(
                    metacognition, metacognitive_prediction,
                    realized_error=prediction_error.squeeze(-1),
                    decision_actions=external_action.selected_action,
                    trigger_classes=trigger, provenance_ids=meta_provenance,
                    mask=meta_mask,
                )
            action_fraction = action_receipts.mask.sum(-1).to(latent.dtype) / max(
                1, self.config.cognitive.maximum_cognitive_steps
            )
            free_memory = (
                (~episodic_memory.active).sum(-1) + (~semantic_memory.active).sum(-1)
            ).to(latent.dtype) / (
                episodic_memory.capacity + semantic_memory.capacity
            )
            remaining_compute = (
                system_state.remaining_compute
                if self.config.cognitive.enable_viability_gate
                else (system_state.remaining_compute - action_fraction[:, None]).clamp_min(0)
            )
            system_state = SystemModelState(
                system_state.modality_availability,
                system_state.action_availability,
                system_state.action_success,
                system_state.action_latency,
                system_state.action_reward,
                system_state.action_cost,
                system_state.action_constraint_violation,
                system_state.action_reversibility,
                system_state.executor_reliability,
                system_state.memory_reliability,
                system_state.router_reliability,
                remaining_compute,
                free_memory[:, None].expand_as(system_state.remaining_memory),
                system_state.permission_mask,
                system_state.calibration_regime,
            )
            output_context = self.output_context_adapter(integrated_context)
            output_latent = output_latent + output_context * cycle_mask.unsqueeze(-1)
            if project_output:
                cognitive_logits = F.linear(
                    output_context, self.carrier.output_head.weight, None,
                )
                prediction = prediction + cognitive_logits * cycle_mask.unsqueeze(-1)
        # Controller rows execute synchronously.  Count microstep rounds, not
        # batch-row actions, so the cognitive clock is invariant to batch size.
        cognitive_rounds = int(action_receipts.mask.any(0).sum())
        clocks = state.clocks.observation_tick().cognitive_tick(cognitive_rounds)
        valid = packet.valid_mask[:, index, None]
        predicted_next = self.next_latent_predictor(latent)
        next_state = MRCRARuntimeState(
            tuple(carrier_states), event_state, nodes, relations, workspace, hypotheses,
            episodic_memory, semantic_memory, controller_state,
            goal_state, system_state, schemas, state.calibration, knowledge,
            _replace_rows(state.last_external_action, external_action, cycle_mask),
            clocks,
            torch.where(valid, latent, state.previous_latent),
            torch.where(valid, predicted_next, state.predicted_next_latent),
            relational_context, selected_scale,
            reconstructions, abstraction_validity,
            action_candidates, state.viability, evidence_requests,
            state.external_artifacts, metacognition, state.boundary_context,
        )
        node_summary = (nodes.content * nodes.active.unsqueeze(-1)).sum(1) / nodes.active.sum(
            1, keepdim=True
        ).clamp_min(1)
        workspace_summary = self._workspace_summary(workspace)
        relation_summary, relation_types = self._relation_summary(relations)
        cognitive_features = self.cognitive_state_projection(torch.cat((
            latent, node_summary, workspace_summary,
            relational_context, uncertainty,
        ), -1))
        provenance_source_logits = self.provenance_source_head(cognitive_features)
        provenance_verification_logits = self.provenance_verification_head(cognitive_features)
        metacognitive_values = latent.new_zeros(state.batch, 7)
        metacognitive_mask = torch.zeros(
            state.batch, dtype=torch.bool, device=latent.device
        )
        if metacognitive_prediction is not None:
            metacognitive_values = torch.stack((
                metacognitive_prediction.predicted_error,
                metacognitive_prediction.value_of_compute,
                metacognitive_prediction.value_of_retrieval,
                metacognitive_prediction.value_of_reconstruction,
                metacognitive_prediction.value_of_simulation,
                metacognitive_prediction.value_of_evidence,
                metacognitive_prediction.calibration_error,
            ), -1)
            metacognitive_mask = cycle_mask
        return MRCRAStepOutput(
            prediction, latent, output_latent, cognitive_features, workspace_summary,
            relation_summary, relation_types, uncertainty, events,
            graph_output, next_state, cycle_mask, action_receipts,
            external_action, world_prediction, distributional_prediction,
            schema_probabilities, symbol_gates,
            event_proposals.proposal_logits[:, 0],
            event_proposals.end_logits[:, 0],
            event_proposals.node_type_logits[:, 0],
            0.5 * (
                event_proposals.content[:, 0]
                + event_proposals.identity_keys[:, 0]
            ),
            event_transition_receipts,
            predicted_next,
            provenance_source_logits, provenance_verification_logits,
            metacognitive_values, metacognitive_mask,
        )

    def forward_integrated_training(
        self, packet: ObservationPacket, ledger: ProvenanceLedger, *,
        state: MRCRAIntegratedTrainingState | None = None,
        cognitive_stride: int | None = None,
        cognitive_tbptt_events: int = 4,
        cognition_mode: str = "full",
    ) -> MRCRAIntegratedTrainingOutput:
        """Run dense carrier time and sparse causal cognition under one loss.

        The dense carrier is evaluated vectorially for work efficiency.  The
        complete cognitive runtime observes causal summaries at physical event
        cadence.  A summary ending at token ``t`` may modulate token ``t``'s
        next-token prediction and later predictions, never an earlier one.
        Consequently ordinary language loss is a legitimate environmental
        pressure on cognition rather than a fabricated cognitive label.
        """

        packet.assert_ledger_consistent(ledger, allow_internal=True)
        if packet.batch != 1 or not bool(packet.valid_mask.all()):
            raise ValueError(
                "integrated language training requires one fully valid document span"
            )
        stride = (
            self.config.cognitive.event_chunk_size
            if cognitive_stride is None else cognitive_stride
        )
        if stride <= 0 or cognitive_tbptt_events <= 0:
            raise ValueError("cognitive training cadence and TBPTT horizon must be positive")
        if cognition_mode not in {"full", "soft_only", "off"}:
            raise ValueError("cognition_mode must be full, soft_only, or off")
        if state is None:
            cognitive_state = self.initial_state(
                packet.batch,
                sample_intervals=packet.sample_intervals * stride,
                device=packet.values.device,
                dtype=packet.values.dtype,
            )
            carrier_state = None
            feedback = packet.values.new_zeros(
                packet.batch, self.config.cognitive.workspace_dim
            )
        else:
            cognitive_state = state.cognitive
            carrier_state = state.carrier
            feedback = state.feedback
        carrier_output = self.carrier.prefill(
            packet.values,
            packet.valid_mask,
            state=carrier_state,
            relational_context=cognitive_state.relational_context,
            project_output=False,
        )
        base_latent = carrier_output.latent
        anchors = tuple(range(0, packet.length, stride))
        if cognition_mode == "off":
            count = len(anchors)
            batch = packet.batch
            boolean = torch.zeros(
                batch, count, dtype=torch.bool, device=packet.values.device
            )
            zero_logits = packet.values.new_zeros(batch, count)
            zero = packet.values.new_zeros(())
            return MRCRAIntegratedTrainingOutput(
                base_latent,
                MRCRAIntegratedTrainingState(
                    carrier_output.state, cognitive_state,
                    torch.zeros_like(feedback),
                ),
                boolean,
                torch.zeros(
                    batch, count, dtype=torch.int64, device=packet.values.device
                ),
                zero, zero, zero, zero, zero_logits, zero_logits,
                boolean, boolean, boolean, boolean, boolean, None,
            )
        summaries, parent_ids, timestamps = [], [], []
        modalities, uncertainties, segments, boundaries = [], [], [], []
        previous_anchor = -1
        for anchor in anchors:
            causal_start = previous_anchor + 1
            causal_end = anchor + 1
            summaries.append(base_latent[:, causal_start:causal_end].mean(1))
            parent_ids.append(packet.source_record_ids[:, anchor])
            timestamps.append(packet.timestamps[:, anchor])
            modalities.append(packet.modality_ids[:, anchor])
            uncertainties.append(
                packet.uncertainty_seed[:, causal_start:causal_end].mean(1)
            )
            segments.append(packet.segment_ids[:, anchor])
            local_boundaries = packet.boundary_classes[:, causal_start:causal_end]
            boundary = torch.zeros(
                packet.batch, dtype=torch.int64, device=packet.values.device
            )
            for offset in range(local_boundaries.shape[1]):
                candidate = local_boundaries[:, offset]
                boundary = torch.where(
                    (boundary == int(BoundaryClass.NONE))
                    & (candidate != int(BoundaryClass.NONE)),
                    candidate, boundary,
                )
            boundaries.append(boundary)
            previous_anchor = anchor
        event_values = torch.stack(summaries, 1)
        event_mask = torch.ones(
            packet.batch, len(anchors), dtype=torch.bool, device=packet.values.device
        )
        event_timestamps = torch.stack(timestamps, 1)
        event_packet = register_internal_inputs(
            event_values,
            event_mask,
            parent_record_ids=torch.stack(parent_ids, 1).unsqueeze(-1),
            timestamps=event_timestamps,
            coordinates=event_timestamps.unsqueeze(-1),
            sample_intervals=packet.sample_intervals * stride,
            boundary_classes=torch.stack(boundaries, 1),
            modality_ids=torch.stack(modalities, 1),
            uncertainty_seed=torch.stack(uncertainties, 1),
            segment_ids=torch.stack(segments, 1),
            ledger=ledger,
            source_class=SourceClass.INFERRED,
            operator="mrcra:causal_event_summary:v1",
            scenario_ids=torch.zeros(
                packet.batch, len(anchors), dtype=torch.int64,
                device=packet.values.device,
            ),
            model_authority=self.model_authority,
        )
        control_basis_cache: dict[tuple[int, torch.device, torch.dtype], Tensor] = {}

        def soft_choice_context(logits: Tensor, mask: Tensor) -> Tensor:
            """Map a bounded soft policy to a fixed spectral control basis."""

            choices = logits.shape[-1]
            width = self.config.cognitive.workspace_dim
            key = (choices, logits.device, logits.dtype)
            basis = control_basis_cache.get(key)
            if basis is None:
                choice = torch.arange(
                    1, choices + 1, device=logits.device, dtype=logits.dtype
                )[:, None]
                feature = torch.arange(
                    1, width + 1, device=logits.device, dtype=logits.dtype
                )[None]
                phase = torch.pi * choice * feature / max(1, choices)
                basis = (
                    torch.sin(phase) + torch.cos(phase * 0.5)
                ) / width**0.5
                control_basis_cache[key] = basis
            probability = torch.softmax(logits, -1)
            context = torch.einsum("bsc,cw->bsw", probability, basis)
            weight = mask.to(context.dtype)
            return (context * weight[..., None]).sum(1) / weight.sum(
                1, keepdim=True
            ).clamp_min(1)

        modulated: list[Tensor] = []
        cycles, events = [], []
        event_activations, active_node_counts = [], []
        proposal_rows, end_rows = [], []
        opened_rows, finalized_rows, emitted_rows = [], [], []
        rejected_rows, open_after_rows = [], []
        first_hard_event: HardEventTrace | None = None
        with defer_runtime_validation():
            for event_index, anchor in enumerate(anchors):
                nodes_before = cognitive_state.nodes.active.sum(-1)
                relations_before = cognitive_state.relations.active.sum(-1)
                workspace_before = cognitive_state.workspace.active.sum(-1)
                cognitive_output = self.step(
                    event_packet, event_index, cognitive_state, ledger,
                    project_output=False, force_cognitive_cycle=True,
                    allow_hard_event_allocation=cognition_mode == "full",
                )
                cognitive_state = cognitive_output.state
                transition = cognitive_output.event_transition_receipts
                proposal_rows.append(cognitive_output.event_proposal_logits)
                end_rows.append(cognitive_output.event_end_logits)
                opened_rows.append(transition.opened[:, 0])
                finalized_rows.append(transition.finalized[:, 0])
                emitted_rows.append(transition.emitted[:, 0])
                rejected_rows.append(transition.quota_rejected[:, 0])
                open_after_rows.append(transition.open_after[:, 0])
                if (
                    first_hard_event is None
                    and bool(cognitive_output.events.active.any())
                ):
                    active_index = torch.nonzero(
                        cognitive_output.events.active, as_tuple=False
                    )[0]
                    batch_index = int(active_index[0])
                    event_slot = int(active_index[1])
                    event = cognitive_output.events
                    first_hard_event = HardEventTrace(
                        anchor_index=anchor,
                        timestamp=event_packet.timestamps[
                            batch_index, event_index
                        ].detach(),
                        proposal_logit=cognitive_output.event_proposal_logits[
                            batch_index
                        ].detach(),
                        end_logit=cognitive_output.event_end_logits[
                            batch_index
                        ].detach(),
                        event_type=event.type_logits[
                            batch_index, event_slot
                        ].argmax().detach(),
                        confidence=torch.sigmoid(
                            event.score[batch_index, event_slot]
                        ).detach(),
                        support=event.support[
                            batch_index, event_slot
                        ].detach(),
                        active_nodes_before=nodes_before[batch_index].detach(),
                        active_nodes_after=cognitive_state.nodes.active.sum(-1)[
                            batch_index
                        ].detach(),
                        active_relations_before=relations_before[
                            batch_index
                        ].detach(),
                        active_relations_after=cognitive_state.relations.active.sum(
                            -1
                        )[batch_index].detach(),
                        workspace_before=workspace_before[batch_index].detach(),
                        workspace_after=cognitive_state.workspace.active.sum(-1)[
                            batch_index
                        ].detach(),
                    )
                receipts = cognitive_output.action_receipts
                action_context = soft_choice_context(
                    receipts.action_logits, receipts.mask
                )
                relation_context = soft_choice_context(
                    receipts.relation_logits, receipts.mask
                )
                active_weight = receipts.mask.to(feedback.dtype)
                halt = (
                    receipts.halt_probability * active_weight
                ).sum(1) / active_weight.sum(1).clamp_min(1)
                policy_context = 0.5 * (action_context + relation_context)
                event_type_context = soft_choice_context(
                    cognitive_output.event_type_logits[:, None],
                    torch.ones(
                        cognitive_output.event_type_logits.shape[0], 1,
                        dtype=torch.bool,
                        device=cognitive_output.event_type_logits.device,
                    ),
                )
                proposal_probability = torch.sigmoid(
                    cognitive_output.event_proposal_logits
                )
                completion_probability = torch.sigmoid(
                    cognitive_output.event_end_logits
                )
                # Continuous expected-event feedback lets environmental CE train
                # every proposal head before a hard event crosses its allocation
                # threshold.  The hard bounded allocator remains authoritative
                # for persistent state; this path supplies differentiable credit.
                event_activation = proposal_probability * (
                    0.5 + 0.5 * completion_probability
                )
                event_context = 0.5 * (
                    cognitive_output.event_soft_content + event_type_context
                )
                feedback = 0.5 * (
                    cognitive_output.cognitive_features
                    + cognitive_state.relational_context
                ) + (
                    self.config.cognitive.broadcast_gain_maximum
                    * (1 - halt[:, None]) * policy_context
                ) + (
                    self.config.cognitive.broadcast_gain_maximum
                    * event_activation[:, None] * torch.tanh(event_context)
                )
                end = (
                    anchors[event_index + 1]
                    if event_index + 1 < len(anchors) else packet.length
                )
                context = self.output_context_adapter(torch.tanh(feedback))
                modulated.append(base_latent[:, anchor:end] + context[:, None])
                cycles.append(cognitive_output.cognitive_cycle_mask)
                events.append(cognitive_output.events.active.sum(-1))
                event_activations.append(event_activation)
                active_node_counts.append(
                    cognitive_state.nodes.active.sum(-1).to(feedback.dtype)
                )
                if (event_index + 1) % cognitive_tbptt_events == 0:
                    # Forward state persists exactly; only its adjoint history is
                    # truncated at an explicit cognitive-event horizon.
                    cognitive_state = cognitive_state.detach()
        validate_dataclass_tree(cognitive_state)
        output_latent = torch.cat(modulated, 1)
        feedback_rms = feedback.float().square().mean(-1).sqrt().max()
        event_activation_mean = torch.stack(event_activations, 1).mean()
        active_node_count = torch.stack(active_node_counts, 1)
        return MRCRAIntegratedTrainingOutput(
            output_latent,
            MRCRAIntegratedTrainingState(
                carrier_output.state, cognitive_state, feedback,
            ),
            torch.stack(cycles, 1),
            torch.stack(events, 1),
            feedback_rms,
            event_activation_mean,
            active_node_count.mean(),
            active_node_count.max(),
            torch.stack(proposal_rows, 1),
            torch.stack(end_rows, 1),
            torch.stack(opened_rows, 1),
            torch.stack(finalized_rows, 1),
            torch.stack(emitted_rows, 1),
            torch.stack(rejected_rows, 1),
            torch.stack(open_after_rows, 1),
            first_hard_event,
        )

    def forward(
        self, packet: ObservationPacket, ledger: ProvenanceLedger, *,
        state: MRCRARuntimeState | None = None, goals: GoalState | None = None,
        system_model: SystemModelState | None = None,
        project_output: bool = True,
    ) -> MRCRAOutput:
        packet.assert_ledger_consistent(ledger, allow_internal=True)
        if state is None:
            state = self.initial_state(
                packet.batch, sample_intervals=packet.sample_intervals,
                device=packet.values.device, dtype=packet.values.dtype,
            )
        predictions, latents, output_latents, cognitive_features = [], [], [], []
        workspace_features, relation_features, relation_types = [], [], []
        uncertainties, event_counts, cycles, receipt_rows = [], [], [], []
        schema_rows, symbol_rows = [], []
        event_proposal_rows, event_end_rows, event_type_rows = [], [], []
        predicted_next_rows, provenance_source_rows, provenance_verification_rows = [], [], []
        metacognitive_rows, metacognitive_mask_rows = [], []
        external_action = state.last_external_action
        world_prediction: WorldModelPrediction | None = None
        distributional_prediction: DistributionalOutput | None = None
        for index in range(packet.length):
            output = self.step(
                packet, index, state, ledger,
                goals=goals if index == 0 else None,
                system_model=system_model if index == 0 else None,
                project_output=project_output,
            )
            state = output.state
            predictions.append(output.prediction)
            latents.append(output.latent)
            output_latents.append(output.output_latent)
            cognitive_features.append(output.cognitive_features)
            workspace_features.append(output.workspace_features)
            relation_features.append(output.relation_features)
            relation_types.append(output.relation_type_probabilities)
            uncertainties.append(output.uncertainty)
            event_counts.append(output.events.active.sum(-1))
            cycles.append(output.cognitive_cycle_mask)
            receipt_rows.append(output.action_receipts)
            schema_rows.append(output.schema_probabilities)
            symbol_rows.append(output.symbol_gates)
            event_proposal_rows.append(output.event_proposal_logits)
            event_end_rows.append(output.event_end_logits)
            event_type_rows.append(output.event_type_logits)
            predicted_next_rows.append(output.predicted_next_latent)
            provenance_source_rows.append(output.provenance_source_logits)
            provenance_verification_rows.append(output.provenance_verification_logits)
            metacognitive_rows.append(output.metacognitive_values)
            metacognitive_mask_rows.append(output.metacognitive_mask)
            distributional_prediction = output.distributional_prediction
            if bool(output.external_action.active.any() | output.external_action.abstained.any()):
                external_action = output.external_action
            if output.world_prediction is not None:
                world_prediction = output.world_prediction
        if packet.length:
            prediction = torch.stack(predictions, 1)
            latent = torch.stack(latents, 1)
            output_latent = torch.stack(output_latents, 1)
            cognitive = torch.stack(cognitive_features, 1)
            workspace_sequence = torch.stack(workspace_features, 1)
            relation_sequence = torch.stack(relation_features, 1)
            relation_type_sequence = torch.stack(relation_types, 1)
            uncertainty = torch.stack(uncertainties, 1)
            event_count = torch.stack(event_counts, 1)
            cycle = torch.stack(cycles, 1)
            action_receipts = CognitiveActionReceipts(*(
                torch.stack([getattr(item, name) for item in receipt_rows], 1)
                for name in CognitiveActionReceipts.__dataclass_fields__
            ))
            schema_sequence = torch.stack(schema_rows, 1)
            symbol_sequence = torch.stack(symbol_rows, 1)
            event_proposal_sequence = torch.stack(event_proposal_rows, 1)
            event_end_sequence = torch.stack(event_end_rows, 1)
            event_type_sequence = torch.stack(event_type_rows, 1)
            predicted_next_sequence = torch.stack(predicted_next_rows, 1)
            provenance_source_sequence = torch.stack(provenance_source_rows, 1)
            provenance_verification_sequence = torch.stack(
                provenance_verification_rows, 1
            )
            metacognitive_sequence = torch.stack(metacognitive_rows, 1)
            metacognitive_mask_sequence = torch.stack(metacognitive_mask_rows, 1)
        else:
            prediction = packet.values.new_zeros(
                packet.batch, 0, self.config.carrier.resolved_output_dim
            )
            latent = packet.values.new_zeros(packet.batch, 0, self.config.cognitive.workspace_dim)
            output_latent = packet.values.new_zeros(
                packet.batch, 0, self.config.cognitive.workspace_dim
            )
            cognitive = packet.values.new_zeros(packet.batch, 0, self.config.cognitive.workspace_dim)
            workspace_sequence = packet.values.new_zeros(
                packet.batch, 0, self.config.cognitive.workspace_dim
            )
            relation_sequence = packet.values.new_zeros(
                packet.batch, 0, self.config.cognitive.workspace_dim
            )
            relation_type_sequence = packet.values.new_zeros(
                packet.batch, 0, self.config.cognitive.relation_family_count
            )
            uncertainty = packet.values.new_zeros(
                packet.batch, 0, self.config.cognitive.uncertainty_channels
            )
            event_count = torch.zeros(packet.batch, 0, dtype=torch.int64, device=packet.values.device)
            cycle = torch.zeros(packet.batch, 0, dtype=torch.bool, device=packet.values.device)
            schema_sequence = packet.values.new_zeros(
                packet.batch, 0, self.config.cognitive.operational_schema_count
            )
            symbol_sequence = packet.values.new_zeros(
                packet.batch, 0, self.config.cognitive.knowledge_support_capacity
            )
            steps = self.config.cognitive.maximum_cognitive_steps
            receipt_shape = (packet.batch, 0, steps)
            action_receipts = CognitiveActionReceipts(
                torch.full(receipt_shape, -1, dtype=torch.int64, device=packet.values.device),
                torch.full(receipt_shape, int(ActionStatus.HALTED), dtype=torch.int64, device=packet.values.device),
                torch.zeros(receipt_shape, dtype=torch.bool, device=packet.values.device),
                torch.full(receipt_shape, -1, dtype=torch.int64, device=packet.values.device),
                torch.full(receipt_shape, -1, dtype=torch.int64, device=packet.values.device),
                torch.full(receipt_shape, -1, dtype=torch.int64, device=packet.values.device),
                torch.zeros(receipt_shape, dtype=torch.bool, device=packet.values.device),
                packet.values.new_zeros(*receipt_shape, len(InternalAction)),
                packet.values.new_zeros(*receipt_shape, len(RelationFamily)),
                packet.values.new_ones(receipt_shape),
                torch.full(receipt_shape, -1, dtype=torch.int64, device=packet.values.device),
                torch.full(receipt_shape, -1, dtype=torch.int64, device=packet.values.device),
                torch.full(receipt_shape, -1, dtype=torch.int64, device=packet.values.device),
                packet.values.new_zeros(
                    *receipt_shape, self.config.cognitive.action_argument_dim
                ),
                torch.zeros(
                    *receipt_shape, self.config.cognitive.action_argument_dim,
                    dtype=torch.bool, device=packet.values.device,
                ),
                torch.zeros(receipt_shape, dtype=torch.int64, device=packet.values.device),
                packet.values.new_zeros(receipt_shape),
            )
            event_proposal_sequence = packet.values.new_zeros(packet.batch, 0)
            event_end_sequence = packet.values.new_zeros(packet.batch, 0)
            event_type_sequence = packet.values.new_zeros(
                packet.batch, 0, self.config.cognitive.node_type_count
            )
            predicted_next_sequence = packet.values.new_zeros(
                packet.batch, 0, self.config.cognitive.workspace_dim
            )
            provenance_source_sequence = packet.values.new_zeros(
                packet.batch, 0, len(SourceClass)
            )
            provenance_verification_sequence = packet.values.new_zeros(
                packet.batch, 0, len(VerificationClass)
            )
            metacognitive_sequence = packet.values.new_zeros(packet.batch, 0, 7)
            metacognitive_mask_sequence = torch.zeros(
                packet.batch, 0, dtype=torch.bool, device=packet.values.device
            )
        return MRCRAOutput(
            prediction, latent, output_latent, cognitive, workspace_sequence, relation_sequence,
            relation_type_sequence, uncertainty, event_count, cycle, state.nodes,
            state.relations, state.workspace, state.hypotheses, state,
            state.controller.abstained, ledger.digest(), action_receipts,
            external_action, world_prediction, distributional_prediction,
            schema_sequence, symbol_sequence, state.knowledge,
            self.online_calibration.report(state.calibration),
            event_proposal_sequence, event_end_sequence, event_type_sequence,
            predicted_next_sequence, provenance_source_sequence,
            provenance_verification_sequence, metacognitive_sequence,
            metacognitive_mask_sequence,
        )
