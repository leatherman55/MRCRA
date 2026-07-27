from __future__ import annotations

import json
from pathlib import Path

import scripts.show_trackio_dashboard as launcher


def _completed(path: Path, *, tokens: int, completed: bool = True) -> Path:
    path.mkdir(parents=True)
    (path / "run_manifest.json").write_text(
        json.dumps({
            "completed": completed,
            "final_training_state": {"tokens_seen": tokens},
        }),
        encoding="utf-8",
    )
    return path


def test_lightweight_launcher_discovers_only_largest_completed_run(tmp_path):
    _completed(tmp_path / "work" / "small", tokens=10)
    expected = _completed(tmp_path / "outputs" / "large", tokens=20)
    _completed(tmp_path / "outputs" / "unfinished", tokens=30, completed=False)
    assert launcher._discover_completed_run(tmp_path) == expected


def test_lightweight_launcher_replaces_stale_runtime_bundle(tmp_path, monkeypatch):
    packaged = tmp_path / "packaged"
    packaged.mkdir()
    (packaged / "index.html").write_text("bounded", encoding="utf-8")
    monkeypatch.setattr(launcher, "PACKAGED_FRONTEND", packaged)

    runtime = launcher._prepare_runtime_frontend(tmp_path / "run")
    stale = runtime / "stale.js"
    stale.write_text("old", encoding="utf-8")
    runtime = launcher._prepare_runtime_frontend(tmp_path / "run")

    assert (runtime / "index.html").read_text(encoding="utf-8") == "bounded"
    assert not stale.exists()


def test_launcher_checks_existing_project_without_model_recovery(monkeypatch):
    class Api:
        def runs(self, project):
            assert project == "mrcra-fineweb"
            return [object()]

    monkeypatch.setattr(launcher.trackio, "Api", Api)
    assert launcher._project_has_runs("mrcra-fineweb") is True
