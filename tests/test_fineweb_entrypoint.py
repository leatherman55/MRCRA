from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import torch

from mrrn.cognitive_training import MRCRATrainingConfig, MRCRA_TRAINING_FORMAT_VERSION
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


def test_measured_apple_optimization_policy_is_the_no_flag_default():
    arguments = parser().parse_args([])
    configuration = MRCRATrainingConfig()

    assert (arguments.cpu_threads, arguments.cpu_interop_threads) == (4, 1)
    assert arguments.apple_mps_loss_offload is False
    assert arguments.compile_tensor_cores is None
    assert arguments.phase_transition_telemetry is True
    assert arguments.phase_transition_ablation is True
    assert arguments.phase_ablation_batches == 1
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
    assert "--sequence-length" not in completed.stdout

    legacy = run_entrypoint("--legacy-mrrn", "--help")
    assert legacy.returncode == 0, legacy.stdout
    assert "4.695M MRRN" in legacy.stdout
    assert "--sequence-length" in legacy.stdout


def test_default_fineweb_smoke_runs_integrated_mrcra_evaluation_and_format8_checkpoint(tmp_path):
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
    assert manifest["evaluation_source"]["kind"] == "sequence"
    assert manifest["evaluation_identity"]["batch_count"] == 1
    assert len(manifest["evaluation_identity"]["sha256"]) == 64
    metric_rows = [
        json.loads(line)
        for line in (output / "evaluation_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        "eval/cross_entropy_nats_per_token" in row.get("metrics", {})
        for row in metric_rows
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
    assert checkpoint["format_version"] == MRCRA_TRAINING_FORMAT_VERSION == 8
    assert checkpoint["identity"]["evaluation"] == manifest["evaluation_identity"]
    assert checkpoint["identity"]["training"]["require_evaluation"] is True
    assert checkpoint["last_runtime"] is not None
    assert checkpoint["last_provenance"] is not None
