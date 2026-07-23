"""Atomic complete checkpoints for MRCRA inference and training state."""

from __future__ import annotations

from dataclasses import asdict, fields
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import torch
from torch import Tensor

from .checkpoint import stream_state_dict, stream_state_from_dict
from .abstraction_control import AbstractionValidityState
from .action_candidates import ActionCandidateState
from .boundaries import BoundaryContextState
from .cognitive_model import MRCRARuntimeState, MultimodalRelationalContinuityResonanceNetwork
from .cognitive_types import CognitiveClocks, NodeSlots, RelationSlots
from .controller import (
    ControllerState, GoalState, OperationalSchemaState, SystemModelState,
)
from .events import EventExtractorState
from .evidence_requests import EvidenceRequestState
from .external_artifacts import ExternalArtifactState
from .hypotheses import HypothesisState
from .interaction import ExternalActionDecision
from .knowledge import KnowledgeProposalState
from .memory_v2 import TensorMemoryState
from .metacognition import MetacognitiveState
from .provenance import ProvenanceLedger
from .reconstruction import ReconstructionState
from .uncertainty import CalibrationState
from .viability import ViabilityState
from .workspace import GlobalWorkspaceState


# Version 5 adds complete consequence measurements, goal authority metadata,
# hypothesis evidence provenance, and explicit scoped-reset semantics.  Both
# earlier production formats migrate conservatively.
FORMAT_VERSION = 5
LEGACY_FORMAT_VERSIONS = {3, 4}


def configuration_hash(model: MultimodalRelationalContinuityResonanceNetwork) -> str:
    payload = json.dumps(asdict(model.config), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


def model_hash(model: MultimodalRelationalContinuityResonanceNetwork) -> str:
    digest = sha256(configuration_hash(model).encode())
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _tensor_fields(value) -> dict:
    return {
        field.name: getattr(value, field.name)
        for field in fields(value)
    }


def runtime_state_dict(state: MRCRARuntimeState) -> dict:
    detached = state.detach()
    return {
        "carrier": [stream_state_dict(item) for item in detached.carrier],
        "event_extractor": _tensor_fields(detached.event_extractor),
        "nodes": _tensor_fields(detached.nodes),
        "relations": _tensor_fields(detached.relations),
        "workspace": _tensor_fields(detached.workspace),
        "hypotheses": _tensor_fields(detached.hypotheses),
        "episodic_memory": _tensor_fields(detached.episodic_memory),
        "semantic_memory": _tensor_fields(detached.semantic_memory),
        "controller": _tensor_fields(detached.controller),
        "goals": _tensor_fields(detached.goals),
        "system_model": _tensor_fields(detached.system_model),
        "schemas": _tensor_fields(detached.schemas),
        "calibration": _tensor_fields(detached.calibration),
        "knowledge": _tensor_fields(detached.knowledge),
        "last_external_action": _tensor_fields(detached.last_external_action),
        "clocks": asdict(detached.clocks),
        "previous_latent": detached.previous_latent,
        "predicted_next_latent": detached.predicted_next_latent,
        "relational_context": detached.relational_context,
        "selected_physical_scale": detached.selected_physical_scale,
        "reconstructions": _tensor_fields(detached.reconstructions),
        "abstraction_validity": _tensor_fields(detached.abstraction_validity),
        "action_candidates": _tensor_fields(detached.action_candidates),
        "viability": _tensor_fields(detached.viability),
        "evidence_requests": _tensor_fields(detached.evidence_requests),
        "external_artifacts": _tensor_fields(detached.external_artifacts),
        "metacognition": _tensor_fields(detached.metacognition),
        "boundary_context": _tensor_fields(detached.boundary_context),
    }


def _legacy_foundation_state(value: Mapping, cognitive=None) -> dict:
    """Build an explicit conservative v3 -> v4 state migration."""

    nodes = value["nodes"]
    knowledge = value["knowledge"]
    system = value["system_model"]
    batch, _, width = nodes["content"].shape
    device, dtype = nodes["content"].device, nodes["content"].dtype
    uncertainty_channels = nodes["uncertainty"].shape[-1]
    hypothesis_capacity = value["hypotheses"]["active"].shape[-1]
    supporter_capacity = knowledge["supporting_provenance_ids"].shape[-1]
    get = lambda name, fallback: getattr(cognitive, name, fallback) if cognitive is not None else fallback
    return {
        "reconstructions": ReconstructionState.empty(
            batch, get("reconstruction_capacity", knowledge["active"].shape[-1]),
            width, uncertainty_channels, device=device, dtype=dtype,
        ),
        "abstraction_validity": AbstractionValidityState.empty(
            batch, knowledge["active"].shape[-1], device=device, dtype=dtype,
        ),
        "action_candidates": ActionCandidateState.empty(
            batch, get("action_candidate_capacity", system["action_availability"].shape[-1]),
            get("action_argument_dim", 8), supporter_capacity, device=device, dtype=dtype,
        ),
        "viability": ViabilityState.empty(
            batch, get("viability_channels", 8), device=device, dtype=dtype,
        ),
        "evidence_requests": EvidenceRequestState.empty(
            batch, get("evidence_request_capacity", 4), width,
            hypothesis_capacity, supporter_capacity, device=device, dtype=dtype,
        ),
        "external_artifacts": ExternalArtifactState.empty(
            batch, get("external_artifact_capacity", 16),
            get("external_artifact_digest_width", 32), device=device, dtype=dtype,
        ),
        "metacognition": MetacognitiveState.empty(
            batch, get("metacognitive_capacity", 16), device=device, dtype=dtype,
        ),
        "boundary_context": BoundaryContextState.empty(batch, device=device),
    }


def runtime_state_from_dict(value: Mapping, *, cognitive=None) -> MRCRARuntimeState:
    required = {
        "carrier", "event_extractor", "nodes", "relations", "workspace",
        "hypotheses", "episodic_memory", "semantic_memory", "controller",
        "goals", "system_model", "schemas", "calibration", "knowledge",
        "last_external_action",
        "clocks", "previous_latent", "predicted_next_latent", "relational_context",
        "selected_physical_scale",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"MRCRA runtime checkpoint is missing {sorted(missing)}")
    foundation_names = {
        "reconstructions", "abstraction_validity", "action_candidates", "viability",
        "evidence_requests", "external_artifacts", "metacognition", "boundary_context",
    }
    present = foundation_names & value.keys()
    if present and present != foundation_names:
        raise ValueError(
            "MRCRA runtime checkpoint contains a partial v4 foundation state; "
            f"missing {sorted(foundation_names - present)}"
        )
    legacy = _legacy_foundation_state(value, cognitive) if not present else None

    def state(name: str, constructor):
        return legacy[name] if legacy is not None else constructor(**value[name])

    # Early v4 checkpoints predate the complete executor-consequence state.
    # Initialize missing measurements conservatively without granting either
    # capabilities or permissions.
    system_value = dict(value["system_model"])
    action_template = system_value["action_success"]
    system_value.setdefault("action_reward", torch.zeros_like(action_template))
    system_value.setdefault("action_cost", torch.zeros_like(action_template))
    system_value.setdefault(
        "action_constraint_violation", torch.zeros_like(action_template)
    )
    system_value.setdefault("action_reversibility", torch.ones_like(action_template))
    system_value.setdefault("executor_reliability", torch.zeros_like(action_template))
    hypothesis_value = dict(value["hypotheses"])
    hypothesis_active = hypothesis_value["active"]
    hypothesis_value.setdefault(
        "latest_supporting_provenance_ids",
        torch.full_like(hypothesis_value["scenario_ids"], -1),
    )
    hypothesis_value.setdefault(
        "latest_contradicting_provenance_ids",
        torch.full_like(hypothesis_value["scenario_ids"], -1),
    )
    if "unknown" not in hypothesis_value:
        unknown = torch.zeros_like(hypothesis_active)
        for row in range(hypothesis_active.shape[0]):
            active_slots = torch.nonzero(
                hypothesis_active[row], as_tuple=False
            ).flatten()
            if active_slots.numel():
                unknown[row, active_slots[0]] = True
        hypothesis_value["unknown"] = unknown

    return MRCRARuntimeState(
        tuple(stream_state_from_dict(item) for item in value["carrier"]),
        EventExtractorState(**value["event_extractor"]),
        NodeSlots(**value["nodes"]), RelationSlots(**value["relations"]),
        GlobalWorkspaceState(**value["workspace"]),
        HypothesisState(**hypothesis_value),
        TensorMemoryState(**value["episodic_memory"]),
        TensorMemoryState(**value["semantic_memory"]),
        ControllerState(**value["controller"]),
        GoalState(**value["goals"]),
        SystemModelState(**system_value),
        OperationalSchemaState(**value["schemas"]),
        CalibrationState(**value["calibration"]),
        KnowledgeProposalState(**value["knowledge"]),
        ExternalActionDecision(**value["last_external_action"]),
        CognitiveClocks(**value["clocks"]),
        value["previous_latent"], value["predicted_next_latent"],
        value["relational_context"],
        value["selected_physical_scale"],
        state("reconstructions", ReconstructionState),
        state("abstraction_validity", AbstractionValidityState),
        state("action_candidates", ActionCandidateState),
        state("viability", ViabilityState),
        state("evidence_requests", EvidenceRequestState),
        state("external_artifacts", ExternalArtifactState),
        state("metacognition", MetacognitiveState),
        state("boundary_context", BoundaryContextState),
    )


def save_mrcra_checkpoint(
    path: str | Path,
    model: MultimodalRelationalContinuityResonanceNetwork,
    state: MRCRARuntimeState,
    ledger: ProvenanceLedger,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    metadata: Mapping | None = None,
) -> None:
    """Write one crash-safe checkpoint containing every runtime authority."""

    if state.batch != len(state.carrier):
        raise ValueError("MRCRA carrier-state count does not match cognitive batch")
    metadata = {} if metadata is None else dict(metadata)
    try:
        json.dumps(metadata, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint metadata must be JSON serializable") from error
    payload = {
        "format_version": FORMAT_VERSION,
        "configuration_hash": configuration_hash(model),
        "model_hash": model_hash(model),
        "model_authority": model.model_authority,
        "runtime": runtime_state_dict(state),
        "provenance": ledger.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "metadata": metadata,
        "torch_rng": torch.random.get_rng_state(),
        "mps_rng": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_mrcra_checkpoint(
    path: str | Path,
    model: MultimodalRelationalContinuityResonanceNetwork,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
    restore_rng: bool = True,
) -> tuple[MRCRARuntimeState, ProvenanceLedger, dict]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    saved_format = payload.get("format_version")
    if saved_format not in {FORMAT_VERSION, *LEGACY_FORMAT_VERSIONS}:
        raise ValueError("unsupported MRCRA checkpoint format")
    if payload.get("configuration_hash") != configuration_hash(model):
        raise ValueError("MRCRA checkpoint configuration hash does not match")
    if payload.get("model_hash") != model_hash(model):
        raise ValueError("MRCRA checkpoint model weights do not match")
    if payload.get("model_authority") != model.model_authority:
        raise ValueError("MRCRA checkpoint model authority does not match")
    state = runtime_state_from_dict(
        payload["runtime"], cognitive=model.config.cognitive,
    )
    ledger = ProvenanceLedger()
    ledger.load_state_dict(payload["provenance"])
    saved_optimizer = payload.get("optimizer")
    if optimizer is not None:
        if saved_optimizer is None:
            raise ValueError("checkpoint does not contain optimizer state")
        optimizer.load_state_dict(saved_optimizer)
    if restore_rng:
        torch.random.set_rng_state(payload["torch_rng"].cpu())
        if payload.get("mps_rng") is not None and torch.backends.mps.is_available():
            torch.mps.set_rng_state(payload["mps_rng"].cpu())
        if payload.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    metadata = dict(payload.get("metadata", {}))
    if saved_format != FORMAT_VERSION:
        metadata["mrcra_migrated_from_format"] = int(saved_format)
        metadata["mrcra_migrated_to_format"] = FORMAT_VERSION
    return state, ledger, metadata
