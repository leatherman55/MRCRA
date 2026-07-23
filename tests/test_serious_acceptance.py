from __future__ import annotations

import json
from pathlib import Path

import torch

from mrrn.cognitive_training import MRCRANextTokenTrainer
from mrrn.config import MRCRAConfig
from mrrn.language import MRCRALanguageModel
from mrrn.lm_training import (
    ByteTextTokenizer, PackedTokenStream, SequenceTextSource,
    build_evaluation_batches,
)
from mrrn.serious_acceptance import (
    REQUIRED_HELD_OUT_TASKS, SeriousCriterion, SeriousPerformanceEvidence,
    SeriousTaskEvidence, _wilson, audit_serious_checkpoint,
    build_serious_evaluation_artifact, file_sha256,
)
from test_cognitive_training import tiny_config, training_config
from dataclasses import replace


def fixture(tmp_path: Path):
    tokenizer = ByteTextTokenizer()
    retained = build_evaluation_batches(
        PackedTokenStream(SequenceTextSource(("held out",)), tokenizer),
        count=1, batch_size=1, sequence_length=8,
    )
    config = replace(
        training_config(tmp_path / "training"),
        evaluation_interval=1, evaluation_batches=1, require_evaluation=True,
    )
    base = tiny_config()
    integrated = MRCRAConfig(
        base.carrier,
        replace(
            base.cognitive,
            enable_conditional_reconstruction=True,
            enable_abstraction_validity_control=True,
            enable_post_deliberation_action_selection=True,
            enable_multi_hypothesis_planning=True,
            enable_agent_session_loop=True,
            enable_viability_gate=True,
            enable_integrated_invariant_discovery=True,
            enable_persistent_session_training=True,
            enable_metacognitive_routing=True,
        ),
        base.actor_parameter_minimum, base.actor_parameter_maximum,
    )
    trainer = MRCRANextTokenTrainer(
        MRCRALanguageModel(integrated), tokenizer,
        PackedTokenStream(SequenceTextSource(("training",)), tokenizer),
        config, retained,
    )
    trainer.train(maximum_steps=1)
    checkpoint = trainer.save_checkpoint()
    source_common = {
        "kind": "fineweb", "dataset_id": "fixture", "dataset_config": "fixture",
        "split": "train", "revision": "fixture",
        "evaluation_fraction_permyriad": 100,
    }
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps({
        "model_parameters": trainer.model.parameter_count,
        "evaluation_identity": trainer.evaluation_identity,
        "training_source": {**source_common, "partition": "train"},
        "evaluation_source": {**source_common, "partition": "eval"},
    }), encoding="utf-8")
    evaluation = tmp_path / "evaluation.json"
    confidence_low, confidence_high = _wilson(1, 1)
    evaluation.write_text(json.dumps({
        "schema_version": 1,
        "suite": "mrcra-serious-held-out-evaluation-v1",
        "checkpoint_sha256": file_sha256(checkpoint),
        "evaluation_identity": trainer.evaluation_identity,
        "tasks": [
            {
                "name": name, "passed": True, "examples": 1, "seeds": [101],
                "trials": 1, "successes": 1,
                "confidence_low": confidence_low,
                "confidence_high": confidence_high,
                "minimum_success_rate": 0.2,
                "effect_mean": 1.0,
                "checkpoint_involved": True,
                "matched_ablation": f"fixture control for {name}",
                "split_sha256": "a" * 64,
                "data_revision": "contract-fixture-v1",
                "production_trainable_parameters": trainer.model.parameter_count,
                "ablation_trainable_parameters": trainer.model.parameter_count,
                "metrics": {"effect": 1.0},
                "criteria": [{
                    "metric": "effect", "threshold": 0.5,
                    "direction": "at_least",
                }],
            }
            for name in REQUIRED_HELD_OUT_TASKS
        ],
        "performance": {
            "passed": True, "context_length": 32_768,
            "tokens_per_second": 1.0, "peak_memory_gib": 1.0,
            "minimum_tokens_per_second": 0.5,
            "maximum_peak_memory_gib": 2.0,
            "hardware": "contract fixture", "dtype": "float32",
        },
    }), encoding="utf-8")
    return checkpoint, manifest, evaluation, trainer


def audit_fixture(checkpoint, manifest, evaluation):
    return audit_serious_checkpoint(
        checkpoint, manifest, evaluation,
        minimum_parameters=1, maximum_parameters=10_000_000,
        minimum_training_tokens=8, minimum_examples_per_task=1,
        minimum_seeds_per_task=1, contract_fixture=True,
    )


def test_complete_contract_fixture_passes_without_claiming_serious_scale(tmp_path):
    checkpoint, manifest, evaluation, trainer = fixture(tmp_path)
    report = audit_fixture(checkpoint, manifest, evaluation)
    assert report.passed
    assert not report.serious_scale and report.maturity == "contract"
    assert report.parameter_count == trainer.model.parameter_count
    assert report.tokens_seen == 8
    assert all(gate.passed for gate in report.gates)
    assert "does not establish serious capability" in report.claim_boundary
    json.dumps(report.to_dict(), allow_nan=False)


def test_checkpoint_and_evaluation_tampering_fail_identity_and_schema(tmp_path):
    checkpoint, manifest, evaluation, _ = fixture(tmp_path)
    artifact = json.loads(evaluation.read_text())
    artifact["checkpoint_sha256"] = "0" * 64
    evaluation.write_text(json.dumps(artifact), encoding="utf-8")
    report = audit_fixture(checkpoint, manifest, evaluation)
    assert not report.passed
    assert any("evaluation_artifact_identity" in failure for failure in report.failures)

    payload = torch.load(checkpoint, weights_only=True)
    payload["model"].pop(next(iter(payload["model"])))
    malformed = tmp_path / "malformed.pt"
    torch.save(payload, malformed)
    artifact["checkpoint_sha256"] = file_sha256(malformed)
    evaluation.write_text(json.dumps(artifact), encoding="utf-8")
    report = audit_fixture(malformed, manifest, evaluation)
    assert not report.passed
    assert any("model_tensor_schema" in failure for failure in report.failures)


def test_production_gate_rejects_unpinned_fixture_and_missing_artifacts(tmp_path):
    checkpoint, manifest, evaluation, _ = fixture(tmp_path)
    report = audit_serious_checkpoint(
        checkpoint, manifest, evaluation,
        minimum_parameters=1, maximum_parameters=10_000_000,
        minimum_training_tokens=8, minimum_examples_per_task=1,
        minimum_seeds_per_task=1,
    )
    assert not report.passed
    assert any("pinned_disjoint_data_identity" in failure for failure in report.failures)
    missing = audit_serious_checkpoint(
        tmp_path / "missing.pt", manifest, evaluation,
    )
    assert not missing.passed
    assert missing.gates[-1].name == "artifact_integrity"


def _typed_task(name: str, parameter_count: int) -> SeriousTaskEvidence:
    return SeriousTaskEvidence(
        name=name, examples=32, seeds=tuple(range(100, 132)), successes=32,
        minimum_success_rate=0.8, effect_mean=1.0,
        matched_ablation=f"matched control for {name}", split_sha256="b" * 64,
        data_revision="fixture-v1", production_trainable_parameters=parameter_count,
        ablation_trainable_parameters=parameter_count,
        metrics={"effect": 1.0},
        criteria=(SeriousCriterion("effect", 0.5, "at_least"),),
    )


def test_canonical_artifact_builder_requires_complete_passing_recomputable_evidence(tmp_path):
    checkpoint, _, _, trainer = fixture(tmp_path)
    tasks = [_typed_task(name, trainer.model.parameter_count) for name in REQUIRED_HELD_OUT_TASKS]
    performance = SeriousPerformanceEvidence(
        context_length=32_768, tokens_per_second=2.0, peak_memory_gib=1.0,
        minimum_tokens_per_second=1.0, maximum_peak_memory_gib=2.0,
        hardware="contract fixture", dtype="float32",
    )
    artifact = build_serious_evaluation_artifact(
        checkpoint, trainer.evaluation_identity, tasks, performance,
    )
    assert artifact["checkpoint_sha256"] == file_sha256(checkpoint)
    assert artifact["performance"]["passed"]
    assert len(artifact["tasks"]) == len(REQUIRED_HELD_OUT_TASKS)
    json.dumps(artifact, allow_nan=False)

    try:
        build_serious_evaluation_artifact(
            checkpoint, trainer.evaluation_identity, tasks[:-1], performance,
        )
    except ValueError as error:
        assert "exactly once" in str(error)
    else:
        raise AssertionError("partial task evidence was accepted")

    failed_performance = SeriousPerformanceEvidence(
        context_length=32_768, tokens_per_second=0.5, peak_memory_gib=3.0,
        minimum_tokens_per_second=1.0, maximum_peak_memory_gib=2.0,
        hardware="contract fixture", dtype="float32",
    )
    try:
        build_serious_evaluation_artifact(
            checkpoint, trainer.evaluation_identity, tasks, failed_performance,
        )
    except ValueError as error:
        assert "failed evidence" in str(error)
    else:
        raise AssertionError("failed hardware evidence was accepted")


def test_hardware_pass_flag_is_recomputed_by_consumer(tmp_path):
    checkpoint, manifest, evaluation, _ = fixture(tmp_path)
    artifact = json.loads(evaluation.read_text())
    artifact["performance"]["tokens_per_second"] = 0.1
    artifact["performance"]["passed"] = True
    evaluation.write_text(json.dumps(artifact), encoding="utf-8")
    report = audit_fixture(checkpoint, manifest, evaluation)
    assert not report.passed
    assert any("target_hardware_efficiency" in failure for failure in report.failures)
