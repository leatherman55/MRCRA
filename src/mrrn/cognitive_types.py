"""Authoritative ontologies and bounded tensor-state contracts for MRCRA.

The neural state may estimate a type or source distribution, but authoritative
metadata lives in integer fields and the immutable provenance ledger.  Keeping
those two layers separate prevents a learned projection from rewriting history.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum
from typing import TypeVar

import torch
from torch import Tensor

from .runtime_validation import runtime_validation_enabled
from .tensor_state import map_tensor_fields


class SourceClass(IntEnum):
    EXTERNAL = 0
    BODILY = 1
    RETRIEVED = 2
    INFERRED = 3
    PREDICTED = 4
    SIMULATED = 5
    ABSTRACTED = 6
    GOAL_DERIVED = 7
    RECONSTRUCTED = 8
    TOOL_OUTPUT = 9
    COMMUNICATED = 10
    EXTERNAL_ARTIFACT = 11


class VerificationClass(IntEnum):
    UNVERIFIED = 0
    INTERNALLY_CONSISTENT = 1
    EXTERNALLY_CHECKED = 2
    CONTRADICTED = 3
    REVOKED = 4


class NodeType(IntEnum):
    OBSERVATION = 0
    FEATURE = 1
    ENTITY = 2
    EVENT = 3
    ACTION = 4
    GOAL = 5
    HYPOTHESIS = 6
    MEMORY = 7
    ABSTRACTION = 8
    INVARIANT = 9
    SYMBOL = 10
    SYSTEM_STATE = 11
    COUNTERFACTUAL = 12


class RelationFamily(IntEnum):
    IDENTITY_PERSISTENCE = 0
    TEMPORAL_ORDER = 1
    SPATIAL_TOPOLOGY = 2
    PART_WHOLE = 3
    TRANSFORMATION = 4
    COREFERENCE = 5
    EVENT_PARTICIPATION = 6
    CORRELATION = 7
    PREDICTIVE_SUPPORT = 8
    CAUSAL_INFLUENCE = 9
    SIMILARITY = 10
    STRUCTURAL_ANALOGY = 11
    INSTANCE_TYPE = 12
    GOAL_INSTRUMENTAL = 13
    DERIVATION_PROVENANCE = 14
    CONTRADICTION_ALTERNATIVE = 15


class InternalAction(IntEnum):
    HALT = 0
    BIND = 1
    UNBIND = 2
    RETYPE_RELATION = 3
    RETRIEVE_RECENT = 4
    RETRIEVE_EPISODIC = 5
    RETRIEVE_SEMANTIC = 6
    EXPAND_ASSOCIATION = 7
    COMPARE = 8
    COMPRESS = 9
    DECOMPRESS = 10
    CREATE_HYPOTHESIS = 11
    MERGE_HYPOTHESES = 12
    PRUNE_HYPOTHESIS = 13
    SIMULATE = 14
    VERIFY = 15
    WRITE_EPISODE = 16
    PROPOSE_INVARIANT = 17
    DESCEND_SCALE = 18
    ASCEND_SCALE = 19
    ABSTAIN_OR_REQUEST_EXTERNAL_EVIDENCE = 20
    RECONSTRUCT_LOCAL = 21
    TEST_APPLICABILITY = 22
    UPDATE_HYPOTHESES = 23
    GENERATE_ACTION_CANDIDATES = 24
    EVALUATE_CANDIDATES = 25
    INSPECT_SELF_STATE = 26
    CREATE_EVIDENCE_REQUEST = 27
    REVISE_ABSTRACTION = 28
    RECORD_EXTERNAL_ARTIFACT = 29
    QUERY_TOOL = 30


class BoundaryClass(IntEnum):
    NONE = 0
    SOFT = 1
    SEGMENT = 2
    HARD = 3


class BoundaryScope(IntEnum):
    """What continuity a boundary is authorized to reset.

    Values are separate from the legacy signal-strength ``BoundaryClass`` so a
    hard document boundary cannot be confused with an identity reset.
    """

    NONE = 0
    EVENT = 1
    SEGMENT = 2
    DOCUMENT = 3
    ENVIRONMENT_EPISODE = 4
    SESSION = 5
    IDENTITY_RESET = 6
    STREAM_DISCONTINUITY = 7


class CognitiveTrigger(IntEnum):
    PERIODIC = 0
    EVENT = 1
    BOUNDARY = 2
    PREDICTION_ERROR = 3
    CONTRADICTION = 4
    NOVELTY = 5
    CONSEQUENTIAL_UNCERTAINTY = 6
    PROVENANCE_GAP = 7
    RECONSTRUCTION_FAILURE = 8
    ACTION_PRECISION = 9
    VIABILITY_RISK = 10
    EXPLICIT_REQUEST = 11


class AgentMode(IntEnum):
    OFFLINE_MODELING = 0
    TASK_AGENT = 1
    PERSISTENT_AGENT = 2
    EVALUATION = 3


class ModalityClass(IntEnum):
    TEXT = 0
    SYMBOLIC = 1
    AUDIO = 2
    SENSOR = 3
    IMAGE = 4
    VIDEO = 5
    GRAPH = 6
    MESH = 7
    FIELD = 8
    SET = 9
    ACTION = 10
    REWARD = 11
    MEMORY = 12
    PREDICTION = 13
    SIMULATION = 14
    GOAL = 15


class RelationDirection(IntEnum):
    UNDIRECTED = 0
    FORWARD = 1
    INVERSE = 2


@dataclass(frozen=True, slots=True)
class SupportInterval:
    """Half-open source support with a causal completion time."""

    start: float
    end: float
    completion_time: float

    def __post_init__(self) -> None:
        if not self.start <= self.end <= self.completion_time:
            raise ValueError("support requires start <= end <= completion_time")


def _require_tensor(name: str, value: Tensor, shape: tuple[int | None, ...], dtype=None) -> None:
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} must use dtype {dtype}, got {value.dtype}")


T = TypeVar("T")


def _tensor_map(instance: T, method: str, *args, **kwargs) -> T:
    return map_tensor_fields(instance, method, *args, **kwargs)


@dataclass(frozen=True, slots=True)
class NodeSlots:
    """Dense bounded node ring; inactive rows are masked and semantically empty."""

    content: Tensor
    spectral: Tensor
    type_logits: Tensor
    support: Tensor
    modality_presence: Tensor
    uncertainty: Tensor
    provenance_features: Tensor
    provenance_ids: Tensor
    source_classes: Tensor
    scenario_ids: Tensor
    hypothesis_membership: Tensor
    activity: Tensor
    age: Tensor
    importance: Tensor
    versions: Tensor
    active: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.content.ndim != 3:
            raise ValueError("node content must have shape (batch,nodes,width)")
        batch, nodes, _ = self.content.shape
        base = (batch, nodes)
        if self.spectral.ndim != 5 or self.spectral.shape[:2] != base or self.spectral.shape[-1] != 2:
            raise ValueError("node spectral state must be (batch,nodes,heads,modes,2)")
        for name in (
            "type_logits", "modality_presence", "uncertainty", "provenance_features",
            "hypothesis_membership",
        ):
            value = getattr(self, name)
            if value.ndim != 3 or value.shape[:2] != base:
                raise ValueError(f"{name} must have shape (batch,nodes,features)")
        _require_tensor("support", self.support, (*base, 3))
        for name in ("provenance_ids", "source_classes", "scenario_ids", "versions"):
            _require_tensor(name, getattr(self, name), base, torch.int64)
        for name in ("activity", "age", "importance"):
            _require_tensor(name, getattr(self, name), base)
        _require_tensor("active", self.active, base, torch.bool)
        if not torch.equal(self.active, self.provenance_ids >= 0):
            raise ValueError("active node rows must have a nonnegative provenance ID and only those rows may")
        if bool(self.active.any()):
            selected = self.support[self.active]
            if not bool(((selected[:, 0] <= selected[:, 1]) & (selected[:, 1] <= selected[:, 2])).all()):
                raise ValueError("active node support must satisfy start <= end <= completion")
            sources = self.source_classes[self.active]
            if int(sources.min()) < 0 or int(sources.max()) >= len(SourceClass):
                raise ValueError("active node source class is outside the ontology")

    @classmethod
    def empty(
        cls, batch: int, capacity: int, width: int, *, heads: int, modes: int,
        node_types: int, modalities: int, uncertainty_channels: int,
        provenance_features: int, hypotheses: int, device=None, dtype=None,
    ) -> "NodeSlots":
        if min(
            batch, capacity, width, heads, modes, node_types, modalities,
            uncertainty_channels, provenance_features, hypotheses,
        ) <= 0:
            raise ValueError("all node-slot dimensions must be positive")
        options = dict(device=device, dtype=dtype)
        base = (batch, capacity)
        return cls(
            torch.zeros(*base, width, **options),
            torch.zeros(*base, heads, modes, 2, **options),
            torch.zeros(*base, node_types, **options),
            torch.zeros(*base, 3, **options),
            torch.zeros(*base, modalities, **options),
            torch.zeros(*base, uncertainty_channels, **options),
            torch.zeros(*base, provenance_features, **options),
            torch.full(base, -1, dtype=torch.int64, device=device),
            torch.full(base, -1, dtype=torch.int64, device=device),
            torch.full(base, -1, dtype=torch.int64, device=device),
            torch.zeros(*base, hypotheses, **options),
            torch.zeros(*base, **options),
            torch.zeros(*base, **options),
            torch.zeros(*base, **options),
            torch.zeros(base, dtype=torch.int64, device=device),
            torch.zeros(base, dtype=torch.bool, device=device),
        )

    @property
    def batch(self) -> int:
        return self.content.shape[0]

    @property
    def capacity(self) -> int:
        return self.content.shape[1]

    def detach(self) -> "NodeSlots":
        return _tensor_map(self, "detach")

    def to(self, *args, **kwargs) -> "NodeSlots":
        return _tensor_map(self, "to", *args, **kwargs)


@dataclass(frozen=True, slots=True)
class RelationSlots:
    """Bounded typed edge/hyperedge records with authoritative pointers."""

    content: Tensor
    type_logits: Tensor
    participant_indices: Tensor
    participant_roles: Tensor
    participant_versions: Tensor
    participant_weights: Tensor
    participant_mask: Tensor
    direction: Tensor
    support: Tensor
    confidence: Tensor
    provenance_ids: Tensor
    scenario_ids: Tensor
    hypothesis_membership: Tensor
    versions: Tensor
    active: Tensor

    def __post_init__(self) -> None:
        if not runtime_validation_enabled():
            return
        if self.content.ndim != 3:
            raise ValueError("relation content must have shape (batch,relations,width)")
        batch, count, _ = self.content.shape
        base = (batch, count)
        if self.type_logits.ndim != 3 or self.type_logits.shape[:2] != base:
            raise ValueError("relation type_logits must be (batch,relations,families)")
        if self.participant_indices.ndim != 3 or self.participant_indices.shape[:2] != base:
            raise ValueError("participant_indices must be (batch,relations,arity)")
        arity_shape = self.participant_indices.shape
        _require_tensor("participant_roles", self.participant_roles, arity_shape, torch.int64)
        _require_tensor("participant_versions", self.participant_versions, arity_shape, torch.int64)
        _require_tensor("participant_weights", self.participant_weights, arity_shape)
        _require_tensor("participant_mask", self.participant_mask, arity_shape, torch.bool)
        _require_tensor("direction", self.direction, base, torch.int64)
        _require_tensor("support", self.support, (*base, 3))
        _require_tensor("confidence", self.confidence, (*base, None))
        _require_tensor("provenance_ids", self.provenance_ids, base, torch.int64)
        _require_tensor("scenario_ids", self.scenario_ids, base, torch.int64)
        if self.hypothesis_membership.ndim != 3 or self.hypothesis_membership.shape[:2] != base:
            raise ValueError("relation hypothesis_membership has invalid shape")
        _require_tensor("versions", self.versions, base, torch.int64)
        _require_tensor("active", self.active, base, torch.bool)
        if not torch.equal(self.active, self.provenance_ids >= 0):
            raise ValueError("active relation rows must have a nonnegative provenance ID and only those rows may")
        if bool((self.participant_mask & (self.participant_indices < 0)).any()):
            raise ValueError("valid relation participants require nonnegative node indices")
        counts = self.participant_mask.sum(-1)
        if bool((self.active & (counts < 2)).any()):
            raise ValueError("active relations require at least two participants")
        if bool((~self.active.unsqueeze(-1) & self.participant_mask).any()):
            raise ValueError("inactive relations cannot retain participant pointers")

    @classmethod
    def empty(
        cls, batch: int, capacity: int, width: int, *, relation_families: int,
        arity: int, uncertainty_channels: int, hypotheses: int,
        device=None, dtype=None,
    ) -> "RelationSlots":
        if min(
            batch, capacity, width, relation_families, arity,
            uncertainty_channels, hypotheses,
        ) <= 0:
            raise ValueError("all relation-slot dimensions must be positive")
        options = dict(device=device, dtype=dtype)
        base = (batch, capacity)
        arity_shape = (*base, arity)
        return cls(
            torch.zeros(*base, width, **options),
            torch.zeros(*base, relation_families, **options),
            torch.full(arity_shape, -1, dtype=torch.int64, device=device),
            torch.zeros(arity_shape, dtype=torch.int64, device=device),
            torch.full(arity_shape, -1, dtype=torch.int64, device=device),
            torch.zeros(*arity_shape, **options),
            torch.zeros(arity_shape, dtype=torch.bool, device=device),
            torch.zeros(base, dtype=torch.int64, device=device),
            torch.zeros(*base, 3, **options),
            torch.zeros(*base, uncertainty_channels, **options),
            torch.full(base, -1, dtype=torch.int64, device=device),
            torch.full(base, -1, dtype=torch.int64, device=device),
            torch.zeros(*base, hypotheses, **options),
            torch.zeros(base, dtype=torch.int64, device=device),
            torch.zeros(base, dtype=torch.bool, device=device),
        )

    @property
    def batch(self) -> int:
        return self.content.shape[0]

    @property
    def capacity(self) -> int:
        return self.content.shape[1]

    def detach(self) -> "RelationSlots":
        return _tensor_map(self, "detach")

    def to(self, *args, **kwargs) -> "RelationSlots":
        return _tensor_map(self, "to", *args, **kwargs)


@dataclass(frozen=True, slots=True)
class CognitiveClocks:
    """External observation, internal microstep, and optimizer clocks."""

    external: int
    cognitive: int
    optimizer: int

    def __post_init__(self) -> None:
        if min(self.external, self.cognitive, self.optimizer) < 0:
            raise ValueError("cognitive clocks cannot be negative")

    def observation_tick(self, count: int = 1) -> "CognitiveClocks":
        """Advance evidence time without erasing either independent clock."""

        if count < 0:
            raise ValueError("clock increments cannot be negative")
        return CognitiveClocks(self.external + count, self.cognitive, self.optimizer)

    def cognitive_tick(self, count: int = 1) -> "CognitiveClocks":
        """Advance completed internal microstep rounds at fixed evidence time."""

        if count < 0:
            raise ValueError("clock increments cannot be negative")
        return CognitiveClocks(self.external, self.cognitive + count, self.optimizer)

    def optimizer_tick(self, count: int = 1) -> "CognitiveClocks":
        """Advance parameter-update time without changing inference time."""

        if count < 0:
            raise ValueError("clock increments cannot be negative")
        return CognitiveClocks(self.external, self.cognitive, self.optimizer + count)


def relation_compatibility() -> Tensor:
    """Auditable hard ontology mask ``[source type, target type, relation]``.

    The mask is permissive where semantics are context dependent, but blocks
    combinations that are definitionally invalid (for example causation by a
    bare type symbol with no event/action realization).
    """

    result = torch.ones(len(NodeType), len(NodeType), len(RelationFamily), dtype=torch.bool)
    event_like = (NodeType.EVENT, NodeType.ACTION, NodeType.HYPOTHESIS, NodeType.COUNTERFACTUAL)
    allowed_causal = torch.zeros(len(NodeType), dtype=torch.bool)
    allowed_causal[list(map(int, event_like))] = True
    result[:, :, RelationFamily.CAUSAL_INFLUENCE] = (
        allowed_causal[:, None] & allowed_causal[None, :]
    )
    result[:, :, RelationFamily.EVENT_PARTICIPATION] = False
    participants = (NodeType.ENTITY, NodeType.ACTION, NodeType.OBSERVATION, NodeType.FEATURE)
    for participant in participants:
        result[int(participant), int(NodeType.EVENT), RelationFamily.EVENT_PARTICIPATION] = True
        result[int(NodeType.EVENT), int(participant), RelationFamily.EVENT_PARTICIPATION] = True
    result[:, :, RelationFamily.INSTANCE_TYPE] = False
    for source in NodeType:
        result[int(source), int(NodeType.SYMBOL), RelationFamily.INSTANCE_TYPE] = True
        result[int(source), int(NodeType.INVARIANT), RelationFamily.INSTANCE_TYPE] = True
    return result


RELATION_COMPATIBILITY = relation_compatibility()
