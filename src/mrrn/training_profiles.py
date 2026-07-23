"""Fail-closed named MRCRA training and persistence contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

from .cognitive_objectives import ObjectiveFamily


class TrainerMode(str, Enum):
    INDEPENDENT_PACKED_DOCUMENTS = "independent_packed_documents"
    CONTINUOUS_WITHIN_DOCUMENT = "continuous_within_document"
    ENVIRONMENT_TRAJECTORY = "environment_trajectory"
    PERSISTENT_AGENT = "persistent_agent"
    EVALUATION_FROZEN_MEMORY = "evaluation_frozen_memory"
    CONTINUAL_ISOLATED_ADAPTER = "continual_isolated_adapter"

    @property
    def permits_persistence(self) -> bool:
        return self != TrainerMode.INDEPENDENT_PACKED_DOCUMENTS


@dataclass(frozen=True, slots=True)
class TrainingProfile:
    name: str
    curriculum_stage: int
    required_families: tuple[ObjectiveFamily, ...]
    required_data_authorities: tuple[str, ...]
    maximum_claim_maturity: str
    permits_persistence: bool = False

    def __post_init__(self) -> None:
        if not self.name or not 1 <= self.curriculum_stage <= 9:
            raise ValueError("training profile identity or stage is invalid")
        if len(self.required_families) != len(set(self.required_families)):
            raise ValueError("training profile families must be unique")
        if not self.required_data_authorities or not self.maximum_claim_maturity:
            raise ValueError("training profile authority and claim boundary are required")

    @property
    def required_auxiliary_families(self) -> tuple[ObjectiveFamily, ...]:
        return tuple(
            family for family in self.required_families
            if family not in (ObjectiveFamily.PRIMARY_TASK, ObjectiveFamily.SPECTRAL_SUBSTRATE)
        )


PROFILES: dict[str, TrainingProfile] = {
    profile.name: profile for profile in (
        TrainingProfile(
            "substrate_language_pretraining", 1,
            (ObjectiveFamily.PRIMARY_TASK, ObjectiveFamily.SPECTRAL_SUBSTRATE),
            ("full_vocabulary_language_targets", "document_boundaries"),
            "mechanism",
        ),
        TrainingProfile(
            "relational_event_pretraining", 2,
            (ObjectiveFamily.PRIMARY_TASK, ObjectiveFamily.SPECTRAL_SUBSTRATE,
             ObjectiveFamily.EVENTS_RELATIONS, ObjectiveFamily.PROVENANCE_CONSISTENCY),
            ("event_annotations", "typed_relation_annotations", "source_records"),
            "mechanism",
        ),
        TrainingProfile(
            "multimodal_grounding", 3,
            (ObjectiveFamily.MULTIMODAL_BINDING, ObjectiveFamily.EVENTS_RELATIONS,
             ObjectiveFamily.PROVENANCE_CONSISTENCY),
            ("asynchronous_modality_observations", "shared_event_annotations"),
            "integrated_loop",
        ),
        TrainingProfile(
            "reconstructive_hierarchy", 4,
            (ObjectiveFamily.COMPRESSION_ABSTRACTION_VALIDITY,
             ObjectiveFamily.RECONSTRUCTION_FIDELITY,
             ObjectiveFamily.PROVENANCE_CONSISTENCY),
            ("structured_masking", "partial_traces", "applicability_negatives"),
            "integrated_loop",
        ),
        TrainingProfile(
            "world_model_trajectory", 5,
            (ObjectiveFamily.WORLD_MODEL_HYPOTHESIS_LIKELIHOOD,
             ObjectiveFamily.ACTION_CONSEQUENCE_INFORMATION_GAIN,
             ObjectiveFamily.PROVENANCE_CONSISTENCY),
            ("action_trajectories", "reward_cost_constraint_receipts", "hypothesis_labels"),
            "integrated_loop",
        ),
        TrainingProfile(
            "active_agent_control", 6,
            (ObjectiveFamily.ACTION_CONSEQUENCE_INFORMATION_GAIN,
             ObjectiveFamily.CONTROLLER_METACOGNITIVE_UTILITY,
             ObjectiveFamily.VIABILITY_CONSTRAINT),
            ("authorized_executor_receipts", "information_gathering_actions", "permissions"),
            "integrated_loop", True,
        ),
        TrainingProfile(
            "invariant_transfer", 7,
            (ObjectiveFamily.INVARIANT_TRANSFER,
             ObjectiveFamily.COMPRESSION_ABSTRACTION_VALIDITY),
            ("held_out_transformations", "near_match_counterexamples", "domain_shift_splits"),
            "transfer",
        ),
        TrainingProfile(
            "continual_validation", 9,
            (ObjectiveFamily.CONTINUAL_ADAPTATION_SAFETY,
             ObjectiveFamily.MEMORY_RETRIEVAL_UTILITY,
             ObjectiveFamily.PROVENANCE_CONSISTENCY),
            ("authorized_continuity_keys", "replay_receipts", "retention_splits"),
            "integrated_loop", True,
        ),
        TrainingProfile(
            "integrated_serious_checkpoint", 9,
            (
                ObjectiveFamily.PRIMARY_TASK, ObjectiveFamily.SPECTRAL_SUBSTRATE,
                ObjectiveFamily.EVENTS_RELATIONS, ObjectiveFamily.MULTIMODAL_BINDING,
                ObjectiveFamily.PROVENANCE_CONSISTENCY,
                ObjectiveFamily.MEMORY_RETRIEVAL_UTILITY,
                ObjectiveFamily.COMPRESSION_ABSTRACTION_VALIDITY,
                ObjectiveFamily.RECONSTRUCTION_FIDELITY,
                ObjectiveFamily.WORLD_MODEL_HYPOTHESIS_LIKELIHOOD,
                ObjectiveFamily.ACTION_CONSEQUENCE_INFORMATION_GAIN,
                ObjectiveFamily.VIABILITY_CONSTRAINT,
                ObjectiveFamily.CONTROLLER_METACOGNITIVE_UTILITY,
                ObjectiveFamily.INVARIANT_TRANSFER,
                ObjectiveFamily.CONTINUAL_ADAPTATION_SAFETY,
            ),
            (
                "language", "relational_events", "multimodal_observations",
                "reconstruction", "trajectories", "executor_receipts",
                "viability_measurements", "invariant_transfer", "continual_replay",
            ),
            "serious_checkpoint", True,
        ),
    )
}


def get_training_profile(name: str) -> TrainingProfile:
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ValueError(f"unknown MRCRA training profile {name!r}") from error


@dataclass(frozen=True, slots=True)
class TrainingAuthorityReport:
    profile: str
    active_families: tuple[str, ...]
    missing_families: tuple[str, ...]
    target_sources: tuple[tuple[str, str], ...]
    target_coverage: tuple[tuple[str, float], ...]
    valid: bool


def validate_training_authority(
    profile: TrainingProfile, active_families: Iterable[ObjectiveFamily], *,
    target_sources: Mapping[ObjectiveFamily, str],
    target_coverage: Mapping[ObjectiveFamily, float],
) -> TrainingAuthorityReport:
    active = frozenset(active_families)
    missing = tuple(sorted(
        (family for family in profile.required_families if family not in active),
        key=int,
    ))
    for family in active:
        source = target_sources.get(family, "")
        coverage = target_coverage.get(family, 0.0)
        if not source or not 0 <= coverage <= 1:
            raise ValueError(f"objective family {family.name} lacks valid target authority")
    report = TrainingAuthorityReport(
        profile.name, tuple(sorted((family.name for family in active))),
        tuple(family.name for family in missing),
        tuple(sorted((family.name, target_sources[family]) for family in active)),
        tuple(sorted((family.name, float(target_coverage[family])) for family in active)),
        not missing,
    )
    if missing:
        raise ValueError(
            f"training profile {profile.name!r} omitted mandatory families: "
            f"{list(report.missing_families)}"
        )
    return report
