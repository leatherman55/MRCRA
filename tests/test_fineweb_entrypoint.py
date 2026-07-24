from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import torch

from mrrn.cognitive_surprise import CognitiveResonantAdjointSurpriseLearner
from mrrn.cognitive_training import (
    MRCRATrainingConfig, MRCRA_TRAINING_FORMAT_VERSION,
    progress_conditioned_rasl_configuration,
)
from mrrn.language import MRCRALanguageModel
from scripts.train_mrcra_fineweb import parser, production_configuration


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "scripts" / "train_fineweb.py"
INTEGRATED_FLAGS = (
    "enable_conditional_reconstruction",
    "enable_abstraction_validity_control",
    "enable_post_deliberation_action_selection",
    "enable_multi_hypothesis_planning",
    "enable_agent_session_loop",
    "enable_viability_gate",
    "enable_integrated_invariant_discovery",
    "enable_persistent_session_training",
    "enable_metacognitive_routing",
)


def run_entrypoint(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ENTRYPOINT), *arguments], cwd=ROOT,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False,
    )


def test_lightmodel_flag_selects_the_strict_8p4m_profile():
    light = production_configuration(50_257, lightmodel=True)
    serious = production_configuration(50_257, lightmodel=False)
    assert light.carrier.model_dim == light.cognitive.workspace_dim == 96
    assert light.actor_parameter_minimum == 8_350_000
    assert light.actor_parameter_maximum == 8_450_000
    assert serious.carrier.model_dim == serious.cognitive.workspace_dim == 256
    assert serious.actor_parameter_minimum == 110_000_000


def test_lightmodel_pc_rasl_is_a_compact_nonduplicating_production_learner():
    actor = MRCRALanguageModel(
        production_configuration(50_257, lightmodel=True)
    )
    training = MRCRATrainingConfig(
        integrated_cognitive_path=True,
        progress_conditioned_rasl=True,
        progress_probe_batches=2,
    )
    learner = CognitiveResonantAdjointSurpriseLearner(
        actor,
        progress_conditioned_rasl_configuration(actor, training),
    )
    critic_parameters = sum(
        parameter.numel() for parameter in learner.critic.parameters()
    )

    assert actor.parameter_count == 8_413_442
    assert critic_parameters == 139_537
    assert critic_parameters / actor.parameter_count < 0.02
    assert learner.target_actor is None
    assert learner.target_critic is not learner.critic
    assert learner.config.core.task_weight == 0.0
    assert learner.config.core.require_external_reward is True


def test_measured_apple_optimization_policy_is_the_no_flag_default():
    arguments = parser().parse_args([])
    configuration = MRCRATrainingConfig()

    assert (arguments.cpu_threads, arguments.cpu_interop_threads) == (4, 1)
    assert arguments.apple_mps_loss_offload is False
    assert arguments.compile_tensor_cores is None
    assert arguments.phase_transition_telemetry is True
    assert arguments.phase_transition_ablation is True
    assert arguments.phase_ablation_batches == 1
    assert arguments.progress_conditioned_rasl is True
    assert arguments.progress_probe_interval == 5
    assert arguments.progress_probe_batches == 2
    assert arguments.progress_probe_length == 4_096
    assert arguments.pc_rasl_candidates == 48
    assert (configuration.cpu_threads, configuration.cpu_interop_threads) == (4, 1)
    assert configuration.compile_tensor_cores is None
    assert configuration.apple_mps_loss_offload is False
    assert configuration.maximum_fused_loss_bytes == 512 << 20
    assert configuration.vocabulary_tile_size == 2_048
    assert configuration.phase_transition_telemetry is True
    assert configuration.phase_transition_ablation is True
    assert configuration.phase_transition_ablation_batches == 1
    assert configuration.low_clip_coefficient_threshold == 0.05
    assert configuration.low_clip_coefficient_patience == 10


def test_familiar_fineweb_entrypoint_help_is_mrcra_not_legacy_mrrn():
    completed = run_entrypoint("--help")
    assert completed.returncode == 0, completed.stdout
    assert "serious MRCRA actor" in completed.stdout
    assert "--context-length" in completed.stdout
    assert "--training-profile" in completed.stdout
    assert "--lightmodel" in completed.stdout
    assert "--progress-interval-tokens" in completed.stdout
    assert "--cognitive-stride" in completed.stdout
    assert "--compile-tensor-cores" in completed.stdout
    assert "--phase-transition-telemetry" in completed.stdout
    assert "--phase-transition-ablation" in completed.stdout
    assert "--progress-conditioned-rasl" in completed.stdout
    assert "--progress-probe-interval" in completed.stdout
    assert "--pc-rasl-trajectory-length" in completed.stdout
    assert "--pc-rasl-controller-gradient-cap" in completed.stdout
    assert "--sequence-length" not in completed.stdout

    legacy = run_entrypoint("--legacy-mrrn", "--help")
    assert legacy.returncode == 0, legacy.stdout
    assert "4.695M MRRN" in legacy.stdout
    assert "--sequence-length" in legacy.stdout


def test_default_fineweb_smoke_runs_integrated_mrcra_pc_rasl_and_format10_checkpoint(tmp_path):
    output = tmp_path / "canonical-fineweb-smoke"
    completed = run_entrypoint(
        "--smoke-test", "--output-dir", str(output), "--no-dashboard",
        "--no-trackio",
    )
    assert completed.returncode == 0, completed.stdout
    assert "MRCRA actor:" in completed.stdout
    assert "MRRN language model:" not in completed.stdout

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["architecture"] == "integrated_mrcra"
    assert manifest["model_profile"] == "mrcra_tiny_smoke"
    cognition = manifest["model_config"]["cognitive"]
    assert all(cognition[name] is True for name in INTEGRATED_FLAGS)
    training = manifest["training_config"]
    assert training["require_evaluation"] is True
    assert training["trackio_enabled"] is False
    assert training["trackio_project"] == "mrcra-fineweb"
    assert training["run_name"] == "mrcra-integrated-default-smoke"
    assert training["evaluation_interval"] == training["evaluation_batches"] == 1
    assert training["integrated_cognitive_path"] is True
    assert training["progress_conditioned_rasl"] is True
    assert training["progress_probe_batches"] == 1
    assert manifest["functional_surprise_enabled"] is True
    assert manifest["functional_surprise_mode"] == "progress_conditioned_rasl"
    assert manifest["evaluation_source"]["kind"] == "sequence"
    assert manifest["evaluation_identity"]["batch_count"] == 1
    assert len(manifest["evaluation_identity"]["sha256"]) == 64
    assert manifest["progress_probe_source"]["kind"] == "sequence"
    assert manifest["progress_probe_identity"]["batch_count"] == 1
    assert len(manifest["progress_probe_identity"]["sha256"]) == 64
    metric_rows = [
        json.loads(line)
        for line in (output / "evaluation_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        "eval/cross_entropy_nats_per_token" in row.get("metrics", {})
        for row in metric_rows
    )
    progress_rows = [
        json.loads(line)
        for line in (
            output / "progress_metrics.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert len(progress_rows) == 2
    assert all(
        row["progress_probe_identity"] == manifest["progress_probe_identity"]
        and row["guard_evaluation_identity"] == manifest["evaluation_identity"]
        for row in progress_rows
    )
    assert all(
        "pc_rasl/progress_pressure" in row["metrics"]
        for row in progress_rows
    )
    assert manifest["completed"] is True
    assert manifest["final_training_state"]["last_evaluation_step"] == 2
    assert manifest["final_training_state"]["last_evaluation_metrics"]

    pointer = json.loads(
        (output / "checkpoints" / "latest.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        output / "checkpoints" / pointer["checkpoint"], weights_only=True,
    )
    assert checkpoint["format_version"] == MRCRA_TRAINING_FORMAT_VERSION == 10
    assert checkpoint["identity"]["evaluation"] == manifest["evaluation_identity"]
    assert checkpoint["identity"]["progress_probe"] == (
        manifest["progress_probe_identity"]
    )
    assert checkpoint["identity"]["training"]["require_evaluation"] is True
    assert checkpoint["learning_progress"] is not None
    assert checkpoint["pc_rasl"] is not None
    assert checkpoint["last_runtime"] is not None
    assert checkpoint["last_provenance"] is not None


def test_smoke_can_explicitly_disable_progress_conditioned_rasl(tmp_path):
    output = tmp_path / "pc-rasl-disabled-smoke"
    completed = run_entrypoint(
        "--smoke-test",
        "--output-dir",
        str(output),
        "--no-progress-conditioned-rasl",
        "--no-dashboard",
        "--no-trackio",
    )
    assert completed.returncode == 0, completed.stdout
    manifest = json.loads(
        (output / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["functional_surprise_enabled"] is False
    assert manifest["functional_surprise_mode"] == "disabled"
    assert manifest["progress_probe_source"] is None
    assert manifest["progress_probe_identity"] is None
    assert not (output / "progress_metrics.jsonl").exists()
    pointer = json.loads(
        (output / "checkpoints" / "latest.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        output / "checkpoints" / pointer["checkpoint"], weights_only=True,
    )
    assert checkpoint["learning_progress"] is None
    assert checkpoint["pc_rasl"] is None
