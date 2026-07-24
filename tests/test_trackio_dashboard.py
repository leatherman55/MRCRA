import json
from types import SimpleNamespace

import mrrn.trackio_dashboard as dashboard
from mrrn.trackio_dashboard import (
    attach_training_evidence,
    discover_completed_run,
    ensure_trackio_project,
    packaged_frontend_dir,
    prepare_runtime_frontend,
    write_evidence_atomically,
)


def _metrics(path, step, norm):
    path.write_text(
        json.dumps(
            {
                "kind": "metrics",
                "step": step,
                "metrics": {"optimization/gradient_norm_before_clip": norm},
            }
        ),
        encoding="utf-8",
    )


def test_packaged_frontend_has_spectral_and_mrcra_cognitive_views(tmp_path):
    source = packaged_frontend_dir()
    assert (source / "index.html").is_file()
    assert (source / "mrrn-spectral-view.html").is_file()
    spectral_view = (source / "mrrn-spectral-view.html").read_text(encoding="utf-8")
    assert "Cognitive Atlas" in spectral_view
    assert "Learning Progress" in spectral_view
    assert "Phase Transition" in spectral_view
    assert "hyperedge-hub" in spectral_view
    assert "cognition/active_hyperrelations" in spectral_view
    assert "event_phase_distance_to_threshold" not in spectral_view
    assert "phase_distance" in spectral_view
    assert "hard_structure_ce_gain" not in spectral_view
    assert "hard_ce_gain" in spectral_view
    assert "pc_pressure" in spectral_view
    assert "pc_observed_slope" in spectral_view
    assert "phase-transition telemetry is not an input" in spectral_view
    javascript = "".join(path.read_text(encoding="utf-8") for path in (source / "assets").glob("*.js"))
    assert "Spectral Network" in javascript
    assert "MRCRA Cognition" in javascript
    assert "Reconstructive Descent" in javascript
    assert "Deliberation Lattice" in javascript
    assert "Viability Envelope" in javascript
    assert "Invariant Transfer" in javascript
    assert "Cognitive Causal Timeline" in javascript
    runtime = prepare_runtime_frontend(tmp_path)
    assert (runtime / "mrrn-spectral-view.html").is_file()
    stale = runtime / "assets" / "legacy-project-selector.js"
    stale.write_text("obsolete", encoding="utf-8")
    runtime = prepare_runtime_frontend(tmp_path)
    assert not stale.exists()


def test_attach_training_evidence_and_atomic_write(tmp_path):
    baseline, current = tmp_path / "baseline.jsonl", tmp_path / "current.jsonl"
    _metrics(baseline, 1, 4.0)
    _metrics(current, 2, 1.0)
    original = {"checkpoint": {"step": 2}}
    result = attach_training_evidence(
        original,
        current_metrics=current,
        baseline_metrics=baseline,
    )
    assert original == {"checkpoint": {"step": 2}}
    assert [item["label"] for item in result["training"]] == ["baseline", "current run"]
    destination = write_evidence_atomically(tmp_path / "data.json", result)
    assert json.loads(destination.read_text(encoding="utf-8"))["schema_version"] == 1


def _completed_run(path, *, tokens=32_768, completed=True):
    path.mkdir(parents=True)
    manifest = {
        "completed": completed,
        "model_profile": "mrcra_8p4m_light",
        "model_parameters": 8_413_442,
        "final_training_state": {
            "step": 1, "tokens_seen": tokens, "valid_targets_seen": tokens - 48,
            "bytes_seen": 147_183, "elapsed_seconds": 104.0,
        },
        "runtime": {
            "cpu_threads": 4, "cpu_interop_threads": 1,
            "apple_hybrid_loss_offload": False,
        },
        "training_config": {
            "run_name": "canonical-optimized-32k",
            "integrated_cognitive_path": True,
        },
    }
    (path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    checkpoints = path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "step-0000001.pt").write_bytes(b"checkpoint")
    (checkpoints / "latest.json").write_text(
        json.dumps({"checkpoint": "step-0000001.pt"}), encoding="utf-8",
    )
    return path


def test_discover_completed_run_prefers_largest_completed_run(tmp_path):
    _completed_run(tmp_path / "work" / "small", tokens=16)
    expected = _completed_run(tmp_path / "outputs" / "large", tokens=32)
    _completed_run(tmp_path / "work" / "unfinished", tokens=64, completed=False)
    assert discover_completed_run(tmp_path) == expected


def test_missing_trackio_project_recovers_canonical_metrics_and_evidence(tmp_path, monkeypatch):
    output = _completed_run(tmp_path / "work" / "canonical")
    calls = []

    class Api:
        def runs(self, project):
            raise ValueError(project)

    class Artifact:
        def __init__(self, name, **kwargs):
            self.name, self.kwargs, self.files = name, kwargs, []

        def add_file(self, path, *, name):
            self.files.append((path, name))

    fake = SimpleNamespace(
        Api=Api,
        Artifact=Artifact,
        init=lambda **kwargs: calls.append(("init", kwargs)),
        log=lambda metrics, step: calls.append(("log", metrics, step)),
        log_artifact=lambda artifact, aliases: calls.append(
            ("artifact", artifact, aliases)
        ),
        finish=lambda: calls.append(("finish",)),
    )
    monkeypatch.setattr(
        dashboard, "_checkpoint_dashboard_evidence",
        lambda checkpoint, **kwargs: {
            "checkpoint": {"step": 1, "tokens_seen": 32_768},
            "cognitive": {"schema_version": 4},
        },
    )

    assert ensure_trackio_project(
        fake, project="mrcra-fineweb", output_dir=output,
    ) is True
    assert [item[0] for item in calls] == ["init", "log", "artifact", "finish"]
    assert calls[0][1]["name"] == "canonical-optimized-32k"
    assert calls[1][1]["runtime/cpu_threads"] == 4
    assert calls[1][1]["training/integrated_cognitive_path"] == 1
    artifact = calls[2][1]
    assert artifact.name == "mrcra-cognitive-spectral-evidence"
    assert artifact.files[0][1] == "mrcra-cognitive-spectral-data.json"
    metric = json.loads((output / "metrics.jsonl").read_text(encoding="utf-8"))
    assert metric["metrics"]["progress/tokens_seen"] == 32_768


def test_existing_trackio_project_is_never_backfilled(tmp_path):
    class Api:
        def runs(self, project):
            return [SimpleNamespace(name="live-training")]

    fake = SimpleNamespace(Api=Api)
    assert ensure_trackio_project(
        fake, project="mrcra-fineweb", output_dir=tmp_path / "missing",
    ) is False
