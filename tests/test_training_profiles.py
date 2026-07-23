import pytest

from mrrn.cognitive_objectives import ObjectiveFamily
from mrrn.cognitive_training import MRCRATrainingConfig
from mrrn.training_profiles import (
    get_training_profile, validate_training_authority,
)


def test_integrated_profile_fails_closed_when_any_mandatory_family_is_missing():
    profile = get_training_profile("integrated_serious_checkpoint")
    active = set(ObjectiveFamily) - {ObjectiveFamily.RECONSTRUCTION_FIDELITY}
    sources = {family: f"test://{family.name.lower()}" for family in active}
    coverage = {family: 1.0 for family in active}
    with pytest.raises(ValueError, match="RECONSTRUCTION_FIDELITY"):
        validate_training_authority(
            profile, active, target_sources=sources, target_coverage=coverage
        )


def test_training_authority_manifest_records_source_and_coverage_per_family():
    profile = get_training_profile("substrate_language_pretraining")
    active = set(profile.required_families)
    report = validate_training_authority(
        profile, active,
        target_sources={
            ObjectiveFamily.PRIMARY_TASK: "fineweb:revision",
            ObjectiveFamily.SPECTRAL_SUBSTRATE: "self_supervised:phase",
        },
        target_coverage={
            ObjectiveFamily.PRIMARY_TASK: 1.0,
            ObjectiveFamily.SPECTRAL_SUBSTRATE: 1.0,
        },
    )
    assert report.valid
    assert report.missing_families == ()
    assert dict(report.target_coverage)["PRIMARY_TASK"] == 1.0


def test_profile_stage_and_required_auxiliary_families_are_not_silently_relaxed(tmp_path):
    common = dict(
        output_dir=str(tmp_path), total_tokens=16, context_length=8,
        execution_chunk_size=2, tbptt_length=4, vocabulary_tile_size=8,
        warmup_tokens=8,
    )
    with pytest.raises(ValueError, match="stage"):
        MRCRATrainingConfig(
            **common, training_profile="reconstructive_hierarchy",
            curriculum_stage=1,
        )
    profile = get_training_profile("reconstructive_hierarchy")
    with pytest.raises(ValueError, match="explicit auxiliary"):
        MRCRATrainingConfig(
            **common, training_profile=profile.name,
            curriculum_stage=profile.curriculum_stage,
        )
