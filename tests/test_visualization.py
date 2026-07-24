import json
from dataclasses import asdict

import pytest
import torch

from mrrn.language import MRRNLanguageModel, tiny_language_config
from mrrn.visualization import (
    build_visualization_dataset,
    checkpoint_spectral_evidence,
    load_training_series,
    model_spectral_evidence,
    write_visualization_dataset,
)


def test_training_series_selects_optimizer_rows(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rows = [
        {"kind": "alert", "step": 1, "text": "ignored"},
        {
            "kind": "metrics",
            "step": 1,
            "metrics": {
                "optimization/gradient_norm_before_clip": 3.5,
                "optimization/gradient_norm_after_clip": 1.0,
                "optimization/gradient_clip_coefficient": 2 / 7,
                "architecture/state_rms": 0.25,
                "architecture/branch_resonance": 0.1,
                "architecture/event_proposal_probability_max": 0.45,
                "architecture/event_phase_distance_to_threshold": 0.2,
                "optimization/gradient/event_before_clip": 0.75,
                "train/cross_entropy_nats_per_token": 2.0,
            },
        },
        {
            "kind": "metrics",
            "step": 1,
            "metrics": {
                "eval/phase_ablation/full_ce_nats_per_token": 2.0,
                "eval/phase_ablation/soft_only_ce_nats_per_token": 2.1,
                "eval/phase_ablation/cognition_off_ce_nats_per_token": 2.3,
                "eval/phase_ablation/hard_structure_ce_gain": 0.1,
                "eval/phase_ablation/soft_bridge_ce_gain": 0.2,
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    result = load_training_series(path, label="test")
    assert result["label"] == "test"
    assert len(result["samples"]) == 1
    sample = result["samples"][0]
    assert sample["step"] == 1
    assert sample["gradient_pre"] == 3.5
    assert sample["proposal_max"] == 0.45
    assert sample["phase_distance"] == 0.2
    assert sample["gradient_event"] == 0.75
    assert sample["ablation_full_ce"] == 2.0
    assert sample["ablation_soft_ce"] == 2.1
    assert sample["ablation_off_ce"] == 2.3
    assert sample["hard_ce_gain"] == 0.1
    assert sample["soft_ce_gain"] == 0.2


def test_visualization_dataset_is_compact_json(tmp_path):
    path = tmp_path / "evidence.json"
    write_visualization_dataset(path, {"schema_version": 1, "values": [1.0, 2.0]})
    assert path.read_text(encoding="utf-8") == '{"schema_version":1,"values":[1.0,2.0]}'


def test_training_series_uses_latest_monotonic_run(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rows = []
    for step, norm in ((1, 9.0), (2, 8.0)):
        rows.append(
            {
                "kind": "metrics",
                "step": step,
                "metrics": {"optimization/gradient_norm_before_clip": norm},
            }
        )
    rows.append({
        "kind": "metrics", "step": 2,
        "metrics": {
            "eval/phase_ablation/full_ce_nats_per_token": 9.0,
            "eval/phase_ablation/soft_only_ce_nats_per_token": 9.1,
        },
    })
    for step, norm in ((1, 3.0), (2, 2.0), (3, 1.0)):
        rows.append({
            "kind": "metrics", "step": step,
            "metrics": {"optimization/gradient_norm_before_clip": norm},
        })
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    samples = load_training_series(path, label="latest")["samples"]
    assert [sample["step"] for sample in samples] == [1, 2, 3]
    assert [sample["gradient_pre"] for sample in samples] == [3.0, 2.0, 1.0]
    assert all(sample["ablation_full_ce"] is None for sample in samples)


def test_training_series_exports_progress_rasl_and_guard_without_phase_telemetry(
    tmp_path,
):
    path = tmp_path / "metrics.jsonl"
    rows = [
        {
            "kind": "metrics",
            "step": 4,
            "metrics": {
                "optimization/gradient_norm_before_clip": 1.0,
                "progress/tokens_seen": 4096,
                "pc_rasl/progress_pressure": -0.25,
                "pc_rasl/raw_progress_pressure": -0.30,
                "pc_rasl/progress_confidence": 0.8,
                "pc_rasl/probe_ce_nats_per_token": 3.5,
                "pc_rasl/expected_ce_nats_per_token": 3.4,
                "pc_rasl/observed_ce_slope_per_million_tokens": -1.0,
                "pc_rasl/expected_ce_slope_per_million_tokens": -2.0,
                "pc_rasl/progress_debt_nats_per_token": 0.1,
                "pc_rasl/critic_loss": 0.7,
                "pc_rasl/internal_policy_loss": 0.2,
                "pc_rasl/replay_transitions": 32,
                "pc_rasl/replay_storage_bytes": 4096,
                "pc_rasl/behavior_evidence_bound": 1,
            },
        },
        {
            "kind": "metrics",
            "step": 4,
            "metrics": {
                "pc_rasl/guard_ce_nats_per_token": 3.6,
                "pc_rasl/guard_best_ce_nats_per_token": 3.55,
                "pc_rasl/guard_allows_positive_pressure": 0,
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    sample = load_training_series(path, label="pc-rasl")["samples"][0]
    assert sample["pc_pressure"] == -0.25
    assert sample["pc_probe_ce"] == 3.5
    assert sample["pc_expected_ce"] == 3.4
    assert sample["pc_observed_slope"] == -1.0
    assert sample["pc_expected_slope"] == -2.0
    assert sample["pc_debt"] == 0.1
    assert sample["pc_critic_loss"] == 0.7
    assert sample["pc_internal_policy_loss"] == 0.2
    assert sample["pc_replay_transitions"] == 32
    assert sample["pc_replay_storage_bytes"] == 4096
    assert sample["pc_behavior_evidence_bound"] == 1
    assert sample["pc_guard_ce"] == 3.6
    assert sample["pc_guard_best_ce"] == 3.55
    assert sample["pc_guard_allows_positive"] == 0
    assert sample["phase_distance"] is None


class _TokenizerStub:
    def __init__(self, name, *, revision):
        self.name, self.revision = name, revision

    def encode_prompt(self, text):
        return [1, 2, 3, 4]

    def decode(self, token_ids):
        return {1: "A", 2: "\n", 3: "\t", 4: ""}[token_ids[0]]


def _checkpoint(tmp_path):
    torch.manual_seed(7)
    config = tiny_language_config(vocabulary_size=17)
    model = MRRNLanguageModel(config)
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format_version": 2,
            "identity": {
                "model_config": asdict(config),
                "model_parameters": model.parameter_count,
                "tokenizer": {
                    "kind": "huggingface",
                    "name": "stub",
                    "revision": "fixed",
                },
            },
            "model": model.state_dict(),
            "training_state": {"step": 9, "tokens_seen": 99},
        },
        path,
    )
    return path, config, model.parameter_count


def _metrics(path, norm):
    path.write_text(
        json.dumps(
            {
                "kind": "metrics",
                "step": 1,
                "metrics": {"optimization/gradient_norm_before_clip": norm},
            }
        ),
        encoding="utf-8",
    )


def test_checkpoint_spectral_evidence_is_complete(tmp_path, monkeypatch):
    path, config, parameter_count = _checkpoint(tmp_path)
    monkeypatch.setattr("mrrn.lm_training.HuggingFaceTextTokenizer", _TokenizerStub)
    result = checkpoint_spectral_evidence(path, prompt="ignored", maximum_tokens=3)
    assert result["checkpoint"]["step"] == 9
    assert result["checkpoint"]["tokens_seen"] == 99
    assert result["checkpoint"]["parameter_count"] == parameter_count
    assert [token["text"] for token in result["tokens"]] == ["A", "↵", "⇥"]
    assert len(result["traces"]) == config.layers
    assert all(len(block) == config.scales for block in result["traces"])
    assert all(len(scale["amplitude"]) == 3 for block in result["traces"] for scale in block)
    expected_poles = config.layers * sum(item.modes for item in config.scale_configs())
    assert len(result["poles"]) == expected_poles
    assert len(result["branch_mix"]) == config.layers * config.scales
    assert len(result["triads"]) == config.layers * config.scales * config.spectral_modes
    assert {item["operation"] for item in result["triads"]} <= {"sum", "difference"}


def test_build_visualization_dataset_links_training_and_checkpoint(tmp_path, monkeypatch):
    checkpoint, _, _ = _checkpoint(tmp_path)
    stable, baseline = tmp_path / "stable.jsonl", tmp_path / "baseline.jsonl"
    _metrics(stable, 1.0)
    _metrics(baseline, 4.0)
    monkeypatch.setattr("mrrn.lm_training.HuggingFaceTextTokenizer", _TokenizerStub)
    result = build_visualization_dataset(
        checkpoint=checkpoint,
        stable_metrics=stable,
        baseline_metrics=baseline,
        prompt="prompt",
        maximum_tokens=1,
    )
    assert result["schema_version"] == 1
    assert [item["label"] for item in result["training"]] == [
        "legacy drive",
        "decay-normalized drive",
    ]


def test_checkpoint_spectral_evidence_rejects_invalid_inputs(tmp_path):
    with pytest.raises(ValueError, match="maximum_tokens"):
        checkpoint_spectral_evidence(tmp_path / "absent.pt", prompt="x", maximum_tokens=0)
    missing = tmp_path / "missing.pt"
    torch.save({"identity": {}}, missing)
    with pytest.raises(ValueError, match="identity"):
        checkpoint_spectral_evidence(missing, prompt="x")
    invalid = tmp_path / "invalid.pt"
    torch.save(
        {
            "identity": {
                "model_config": asdict(tiny_language_config(17)),
                "tokenizer": {"kind": "byte"},
            }
        },
        invalid,
    )
    with pytest.raises(ValueError, match="Hugging Face"):
        checkpoint_spectral_evidence(invalid, prompt="x")


def test_live_model_spectral_evidence_supports_byte_tokenizer_and_restores_mode():
    from mrrn.lm_training import ByteTextTokenizer

    tokenizer = ByteTextTokenizer()
    model = MRRNLanguageModel(tiny_language_config(tokenizer.vocabulary_size))
    model.train()
    result = model_spectral_evidence(
        model,
        tokenizer,
        prompt="abc",
        maximum_tokens=2,
        step=5,
        tokens_seen=40,
    )
    assert model.training
    assert result["checkpoint"]["step"] == 5
    assert result["checkpoint"]["tokens_seen"] == 40
    assert [token["text"] for token in result["tokens"]] == ["a", "b"]
