from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import mrrn.language as language_module
from mrrn.cognitive_surprise import CognitiveResonantAdjointSurpriseLearner
from mrrn.cognitive_training import (
    MRCRATrainingConfig, MRCRA_TRAINING_FORMAT_VERSION,
    progress_conditioned_rasl_configuration,
)
from mrrn.language import MRCRALanguageModel
from mrrn.vocabulary_router import VocabularyRouterConfig
from scripts.train_mrcra_fineweb import (
    parser, production_cognitive_stride, production_configuration,
    production_profile, resolve_activation_checkpointing,
)


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "scripts" / "train_fineweb.py"
PARAMETER_REPORT = ROOT / "scripts" / "report_mrcra_parameters.py"
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
    light = production_configuration(
        50_257, lightmodel=True, ultralightmodel=False
    )
    serious = production_configuration(
        50_257, lightmodel=False, ultralightmodel=False
    )
    assert light.carrier.model_dim == light.cognitive.workspace_dim == 96
    assert light.actor_parameter_minimum == 8_350_000
    assert light.actor_parameter_maximum == 8_450_000
    assert serious.carrier.model_dim == serious.cognitive.workspace_dim == 256
    assert serious.actor_parameter_minimum == 110_000_000


def test_ultralightmodel_selects_exact_complete_2p7m_profile_and_names():
    ultralight = production_configuration(
        50_257, lightmodel=False, ultralightmodel=True
    )
    model = MRCRALanguageModel(ultralight)
    selected = production_profile(
        lightmodel=False, ultralightmodel=True, total_tokens=20_000_000
    )

    assert model.parameter_count == 2_699_463
    assert ultralight.actor_parameter_minimum == 2_690_000
    assert ultralight.actor_parameter_maximum == 2_710_000
    assert ultralight.carrier.model_dim == ultralight.cognitive.workspace_dim == 36
    assert ultralight.carrier.scales == 6
    assert selected.name == "mrcra_2p7m_ultralight"
    assert selected.model_authority == "mrcra-ultralight-2p7m-fineweb-stage1"
    assert selected.output_directory.endswith(
        "mrcra-2p7m-fineweb-20000000-tokens"
    )
    assert selected.run_name == (
        "mrcra-2p7m-ultralight-integrated-fineweb-"
        "20000000-tokens-32k"
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        production_configuration(
            50_257, lightmodel=True, ultralightmodel=True
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        production_profile(
            lightmodel=True, ultralightmodel=True, total_tokens=20_000_000
        )


def test_ultralightmodel_is_eligible_for_the_default_certified_router(monkeypatch):
    configuration = production_configuration(
        50_257, lightmodel=False, ultralightmodel=True
    )
    model = MRCRALanguageModel(configuration)
    router_config = VocabularyRouterConfig()
    constructed: list[tuple[torch.Tensor, torch.Tensor, VocabularyRouterConfig]] = []
    sentinel = object()

    def capture_router(weight, bias, config):
        constructed.append((weight, bias, config))
        return sentinel

    monkeypatch.setattr(
        language_module, "CertifiedBalancedVocabularyRouter", capture_router
    )
    assert configuration.carrier.model_dim == 36
    assert configuration.carrier.model_dim >= router_config.minimum_model_dimension
    assert model.vocabulary_size >= router_config.minimum_vocabulary_size
    assert model.build_vocabulary_router() is sentinel
    assert len(constructed) == 1
    assert constructed[0][0] is model.token_embedding.weight
    assert constructed[0][1] is model.cognitive.carrier.output_head.bias
    assert constructed[0][2] == router_config


def test_ultralight_parameter_report_is_reproducible(tmp_path):
    output = tmp_path / "ultralight-parameters.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(PARAMETER_REPORT),
            "--ultralightmodel",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["model_profile"] == "mrcra_2p7m_ultralight"
    assert report["parameter_count"] == report["trainable_parameter_count"] == 2_699_463
    assert report["declared_range"]["passed"] is True
    assert report["tied_token_and_output_weights"] is True
    assert report["configuration"]["carrier"]["scales"] == 6
    assert report["parameter_count_by_subsystem"]["cognitive.carrier"] > 0
    assert report["parameter_count_by_subsystem"]["cognitive.workspace_graph"] > 0
    assert report["parameter_count_by_subsystem"]["cognitive.controller"] > 0
    assert report["parameter_count_by_subsystem"]["cognitive.world_model"] > 0
    assert report["parameter_count_by_subsystem"]["cstm_predictor"] == 2_414


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

    assert actor.parameter_count == 8_416_803
    assert critic_parameters == 139_537
    assert critic_parameters / actor.parameter_count < 0.02
    assert learner.target_actor is None
    assert learner.target_critic is not learner.critic
    assert learner.config.core.task_weight == 0.0
    assert learner.config.core.require_external_reward is True


def test_ultralight_pc_rasl_remains_bounded_and_does_not_duplicate_actor():
    actor = MRCRALanguageModel(
        production_configuration(
            50_257, lightmodel=False, ultralightmodel=True
        )
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

    assert critic_parameters == 104_077
    assert critic_parameters / actor.parameter_count < 0.04
    assert learner.target_actor is None
    assert learner.target_critic is not learner.critic


def test_measured_apple_optimization_policy_is_the_no_flag_default():
    arguments = parser().parse_args([])
    configuration = MRCRATrainingConfig()

    # The CLI requests bounded startup calibration with zero; direct API users
    # retain the conservative fixed four-thread default unless they opt in.
    assert (arguments.cpu_threads, arguments.cpu_interop_threads) == (0, 1)
    assert arguments.apple_mps_loss_offload is False
    assert arguments.compile_tensor_cores is None
    assert arguments.phase_transition_telemetry is True
    assert arguments.phase_transition_ablation is True
    assert arguments.phase_ablation_batches == 1
    assert arguments.progress_conditioned_rasl is False
    assert arguments.progress_probe_interval == 5
    assert arguments.progress_probe_batches == 2
    assert arguments.progress_probe_length == 4_096
    assert arguments.pc_rasl_candidates == 48
    assert arguments.pc_rasl_captures_per_observation == 1
    assert arguments.pc_rasl_updates_per_observation == 1
    assert arguments.activation_checkpointing is None
    assert arguments.allow_unsafe_activation_policy is False
    assert arguments.dashboard is False
    assert arguments.trackio_remote_log_interval == 4
    assert arguments.maximum_compiled_cce_mib == 512
    assert arguments.document_static_batching is True
    assert arguments.document_planner == "auto"
    assert arguments.document_cost_calibration is True
    assert tuple(arguments.document_bucket_lengths) == tuple(
        range(64, 4_096 + 1, 64)
    )
    assert arguments.document_batch_token_budget == 8_192
    assert arguments.cstm_execution == "sampled"
    assert arguments.cstm_sampling_duty_cycle == 0.25
    assert arguments.compile_carrier == "auto"
    assert arguments.performance_calibration is True
    assert arguments.cstm_max_substrate_vjps == 1
    assert arguments.cstm_target_participation_budget == 8_192
    assert arguments.cstm_predictor_update_interval == 1
    assert arguments.cstm_maximum_coverage_gap == 4_096
    assert (configuration.cpu_threads, configuration.cpu_interop_threads) == (4, 1)
    assert configuration.compile_tensor_cores is None
    assert configuration.performance_calibration is True
    assert configuration.document_static_batching is True
    assert configuration.document_cost_calibration is True
    assert configuration.document_batch_token_budget == 8_192
    assert configuration.cstm_execution == "sampled"
    assert configuration.cstm_sampling_duty_cycle == 0.25
    assert configuration.cstm_max_substrate_vjps == 1
    assert configuration.cstm_target_participation_budget == 8_192
    assert configuration.cstm_predictor_update_interval == 1
    assert configuration.cstm_maximum_coverage_gap == 4_096
    assert configuration.apple_mps_loss_offload is False
    assert configuration.maximum_fused_loss_bytes == 512 << 20
    assert configuration.vocabulary_tile_size == 4_096
    assert configuration.phase_transition_telemetry is True
    assert configuration.phase_transition_ablation is True
    assert configuration.phase_transition_ablation_batches == 1
    assert configuration.low_clip_coefficient_threshold == 0.05
    assert configuration.low_clip_coefficient_patience == 10
    assert configuration.pc_rasl_captures_per_observation == 1
    assert configuration.pc_rasl_updates_per_observation == 1
    assert configuration.trackio_enabled is True
    assert configuration.trackio_remote_log_interval == 4
    assert configuration.show_dashboard is False


def test_canonical_execution_control_aliases_are_explicit_and_parseable():
    arguments = parser().parse_args(
        [
            "--document-planner", "fixed",
            "--no-document-cost-calibration",
            "--cstm-substrate-duty-probability", "0.125",
            "--upgrade-cstm-execution-policy",
            "--compile-carrier", "off",
            "--no-performance-calibration",
        ]
    )
    assert arguments.document_planner == "fixed"
    assert arguments.document_cost_calibration is False
    assert arguments.cstm_sampling_duty_cycle == 0.125
    assert arguments.allow_cstm_execution_upgrade is True
    assert arguments.compile_carrier == "off"
    assert arguments.performance_calibration is False


def test_ultralight_uses_measured_stride_and_memory_aware_checkpoint_policy(
    monkeypatch,
):
    ultralight = production_configuration(
        50_257, lightmodel=False, ultralightmodel=True
    )
    monkeypatch.setattr(
        "scripts.train_mrcra_fineweb._host_memory_capacity_bytes",
        lambda: 16 << 30,
    )
    resolved, policy = resolve_activation_checkpointing(
        ultralight,
        tbptt_length=4_096,
        device="cpu",
        precision="fp32",
        override=None,
    )

    assert production_cognitive_stride(
        ultralight, ultralightmodel=True, override=None
    ) == 128
    assert production_cognitive_stride(
        ultralight, ultralightmodel=True, override=256
    ) == 256
    assert resolved.carrier.activation_checkpointing is True
    assert policy["carrier_activation_checkpointing_policy"] == (
        "automatic_recompute_over_budget"
    )
    assert (
        policy["estimated_uncheckpointed_carrier_activation_bytes"]
        > policy["carrier_activation_memory_budget_bytes"]
    )


def test_activation_checkpoint_policy_preserves_explicit_override(monkeypatch):
    ultralight = production_configuration(
        50_257, lightmodel=False, ultralightmodel=True
    )
    monkeypatch.setattr(
        "scripts.train_mrcra_fineweb._host_memory_capacity_bytes",
        lambda: 16 << 30,
    )
    resolved, policy = resolve_activation_checkpointing(
        ultralight,
        tbptt_length=4_096,
        device="cpu",
        precision="fp32",
        override=True,
    )
    assert resolved.carrier.activation_checkpointing is True
    assert policy["carrier_activation_checkpointing_policy"] == (
        "explicit_recompute"
    )


def test_familiar_fineweb_entrypoint_help_is_mrcra_not_legacy_mrrn():
    completed = run_entrypoint("--help")
    assert completed.returncode == 0, completed.stdout
    assert "complete MRCRA actor" in completed.stdout
    assert "--context-length" in completed.stdout
    assert "--training-profile" in completed.stdout
    assert "--lightmodel" in completed.stdout
    assert "--ultralightmodel" in completed.stdout
    assert "--progress-interval-tokens" in completed.stdout
    assert "--cognitive-stride" in completed.stdout
    assert "--compile-tensor-cores" in completed.stdout
    assert "--maximum-compiled-cce-mib" in completed.stdout
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

    conflicting = run_entrypoint("--lightmodel", "--ultralightmodel")
    assert conflicting.returncode == 2
    assert "not allowed with argument --lightmodel" in conflicting.stdout


def test_default_fineweb_smoke_runs_integrated_mrcra_with_cstm_and_format16_checkpoint(tmp_path):
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
    assert training["progress_conditioned_rasl"] is False
    assert training["cstm_enabled"] is True
    assert manifest["cstm_enabled"] is True
    assert manifest["cstm_effective"] is True
    assert manifest["cstm_architecture"]["code_dimension"] == 64
    assert training["progress_probe_batches"] == 0
    assert manifest["functional_surprise_enabled"] is False
    assert manifest["functional_surprise_mode"] == "disabled"
    assert manifest["evaluation_source"]["kind"] == "sequence"
    assert manifest["evaluation_identity"]["batch_count"] == 1
    assert len(manifest["evaluation_identity"]["sha256"]) == 64
    assert manifest["progress_probe_source"] is None
    assert manifest["progress_probe_identity"] is None
    metric_rows = [
        json.loads(line)
        for line in (output / "evaluation_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        "eval/cross_entropy_nats_per_token" in row.get("metrics", {})
        for row in metric_rows
    )
    assert not (output / "progress_metrics.jsonl").exists()
    assert manifest["completed"] is True
    assert manifest["final_training_state"]["last_evaluation_step"] == 2
    assert manifest["final_training_state"]["last_evaluation_metrics"]

    pointer = json.loads(
        (output / "checkpoints" / "latest.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        output / "checkpoints" / pointer["checkpoint"], weights_only=True,
    )
    assert checkpoint["format_version"] == MRCRA_TRAINING_FORMAT_VERSION == 16
    assert checkpoint["identity"]["semantic"]["evaluation"] == manifest["evaluation_identity"]
    assert checkpoint["identity"]["semantic"]["progress_probe"] == (
        manifest["progress_probe_identity"]
    )
    assert checkpoint["identity"]["observation"]["training"][
        "require_evaluation"
    ] is True
    assert checkpoint["execution_policy_history"]
    assert checkpoint["learning_progress"] is None
    assert checkpoint["pc_rasl"] is None
    assert checkpoint["last_runtime"] is not None
    assert checkpoint["last_provenance"] is not None


def test_canonical_smoke_forwards_compiled_cce_workspace_policy(tmp_path):
    output = tmp_path / "bounded-cce-smoke"
    completed = run_entrypoint(
        "--smoke-test",
        "--output-dir",
        str(output),
        "--maximum-compiled-cce-mib",
        "0",
        "--no-dashboard",
        "--no-trackio",
    )
    assert completed.returncode == 0, completed.stdout
    manifest = json.loads(
        (output / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["training_config"]["maximum_fused_loss_bytes"] == 0
    assert manifest["runtime"]["compiled_cce_fits_workspace"] is False
    assert manifest["runtime"]["exact_loss_backend"] == "tiled"


def test_ultralight_smoke_runs_the_real_2p7m_actor_offline(tmp_path):
    output = tmp_path / "ultralight-fineweb-smoke"
    completed = run_entrypoint(
        "--smoke-test",
        "--ultralightmodel",
        "--output-dir",
        str(output),
        "--no-phase-transition-ablation",
        "--no-dashboard",
        "--no-trackio",
    )
    assert completed.returncode == 0, completed.stdout
    assert "MRCRA actor: 2,699,463 parameters" in completed.stdout

    manifest = json.loads(
        (output / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model_profile"] == "mrcra_2p7m_ultralight"
    assert manifest["model_parameters"] == 2_699_463
    assert manifest["tokenizer"] == {
        "kind": "utf8-bytes-production-width-smoke",
        "vocabulary_size": 50_257,
        "eos_token_id": 50_256,
        "semantic_tokenizer": False,
    }
    carrier = manifest["model_config"]["carrier"]
    cognition = manifest["model_config"]["cognitive"]
    assert carrier["model_dim"] == cognition["workspace_dim"] == 36
    assert carrier["scales"] == 6
    assert carrier["share_depth_parameters"] is True
    assert all(cognition[name] is True for name in INTEGRATED_FLAGS)
    training = manifest["training_config"]
    assert training["run_name"] == "mrcra-2p7m-ultralight-smoke"
    assert training["integrated_cognitive_path"] is True
    assert training["cstm_enabled"] is True
    assert manifest["cstm_effective"] is True
    assert training["progress_conditioned_rasl"] is False
    assert manifest["functional_surprise_enabled"] is False
    assert manifest["completed"] is True


def test_smoke_can_explicitly_enable_progress_conditioned_rasl(tmp_path):
    output = tmp_path / "pc-rasl-enabled-smoke"
    completed = run_entrypoint(
        "--smoke-test",
        "--output-dir",
        str(output),
        "--progress-conditioned-rasl",
        "--no-dashboard",
        "--no-trackio",
    )
    assert completed.returncode == 0, completed.stdout
    manifest = json.loads(
        (output / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["functional_surprise_enabled"] is True
    assert manifest["functional_surprise_mode"] == "progress_conditioned_rasl"
    assert manifest["progress_probe_source"]["kind"] == "sequence"
    assert manifest["progress_probe_identity"]["batch_count"] == 1
    progress_rows = [
        json.loads(line)
        for line in (
            output / "progress_metrics.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert len(progress_rows) == 2
    assert all(
        "pc_rasl/progress_pressure" in row["metrics"]
        for row in progress_rows
    )
    pointer = json.loads(
        (output / "checkpoints" / "latest.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        output / "checkpoints" / pointer["checkpoint"], weights_only=True,
    )
    assert checkpoint["learning_progress"] is not None
    assert checkpoint["pc_rasl"] is not None
