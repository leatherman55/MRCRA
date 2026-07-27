"""Packaging and evidence helpers for the custom MRRN Trackio frontend."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .visualization import load_training_series


SPECTRAL_ARTIFACT_NAME = "mrcra-cognitive-spectral-evidence"
SPECTRAL_ARTIFACT_TYPE = "mrcra-cognitive-spectral"
SPECTRAL_DATA_FILENAME = "mrcra-cognitive-spectral-data.json"
CANONICAL_TRACKIO_RUN_SOURCE = "retained-completed-run"


def packaged_frontend_dir() -> Path:
    """Return the wheel-bundled custom frontend and validate its build output."""

    path = Path(__file__).with_name("trackio_frontend")
    if not (path / "index.html").is_file() or not (path / "mrrn-spectral-view.html").is_file():
        raise FileNotFoundError("the packaged MRRN Trackio frontend is incomplete")
    return path


def prepare_runtime_frontend(output_dir: str | Path) -> Path:
    """Materialize the immutable frontend build inside a training run directory."""

    destination = Path(output_dir) / ".trackio-mrrn-frontend"
    # This directory is a rebuildable cache. Replacing it exactly prevents old
    # hashed frontend bundles from surviving an upgrade and being mistaken for
    # active dashboard code.
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(packaged_frontend_dir(), destination)
    return destination


def discover_completed_run(search_root: str | Path = ".") -> Path | None:
    """Return the largest completed local run available for dashboard recovery."""

    root = Path(search_root)
    candidates: list[tuple[int, int, Path]] = []
    for parent_name in ("work", "outputs"):
        parent = root / parent_name
        if not parent.is_dir():
            continue
        for manifest_path in parent.rglob("run_manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                final = manifest["final_training_state"]
                completed = manifest["completed"] is True
                tokens = int(final["tokens_seen"])
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if completed:
                candidates.append(
                    (tokens, manifest_path.stat().st_mtime_ns, manifest_path.parent)
                )
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _canonical_checkpoint(output_dir: Path) -> Path:
    pointer = output_dir / "checkpoints" / "latest.json"
    if not pointer.is_file():
        raise FileNotFoundError(f"retained run has no checkpoint pointer: {pointer}")
    value = json.loads(pointer.read_text(encoding="utf-8"))
    checkpoint = pointer.parent / value["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"retained run checkpoint is missing: {checkpoint}")
    return checkpoint


def _write_canonical_metric_mirror(
    output_dir: Path, *, step: int, metrics: dict[str, float],
) -> Path:
    destination = output_dir / "metrics.jsonl"
    payload = json.dumps(
        {"kind": "metrics", "step": step, "metrics": metrics},
        sort_keys=True, allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="canonical-trackio-metrics-", suffix=".tmp", dir=output_dir,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return destination


def _checkpoint_dashboard_evidence(
    checkpoint: Path, *, prompt: str, maximum_tokens: int,
) -> dict[str, Any]:
    """Reconstruct non-authoritative dashboard evidence from a retained MRCRA run."""

    import torch

    from .cognitive_checkpoint import runtime_state_from_dict
    from .cognitive_diagnostics import cognitive_evidence
    from .config import CognitiveConfig, MRCRAConfig, MRRNConfig
    from .language import MRCRALanguageModel
    from .lm_training import HuggingFaceTextTokenizer
    from .provenance import ProvenanceLedger
    from .visualization import model_spectral_evidence

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    identity = payload.get("identity", {})
    raw = identity.get("model_config")
    tokenizer_identity = identity.get("tokenizer")
    if not isinstance(raw, dict) or not isinstance(tokenizer_identity, dict):
        raise ValueError("retained checkpoint lacks model/tokenizer identity")
    config = MRCRAConfig(
        MRRNConfig(**raw["carrier"]), CognitiveConfig(**raw["cognitive"]),
        int(raw["actor_parameter_minimum"]), int(raw["actor_parameter_maximum"]),
    )
    model = MRCRALanguageModel(config)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    tokenizer = HuggingFaceTextTokenizer(
        tokenizer_identity["name"], revision=tokenizer_identity["revision"],
    )
    training_state = payload.get("training_state", {})
    evidence = model_spectral_evidence(
        model, tokenizer, prompt=prompt, maximum_tokens=maximum_tokens,
        step=int(training_state.get("step", 0)),
        tokens_seen=int(training_state.get("tokens_seen", 0)),
        source=str(checkpoint), format_version=payload.get("format_version"),
    )
    ids = tokenizer.encode_prompt(prompt)[:maximum_tokens]
    if not ids:
        raise ValueError("dashboard prompt produced no tokens")
    ledger = ProvenanceLedger()
    runtime = None
    if payload.get("last_runtime") is not None and payload.get("last_provenance") is not None:
        runtime = runtime_state_from_dict(
            payload["last_runtime"], cognitive=config.cognitive,
        )
        ledger.load_state_dict(payload["last_provenance"])
    with torch.no_grad():
        output = model(
            torch.tensor([ids], dtype=torch.int64),
            source_uris=("diagnostic://retained-checkpoint",),
            state=runtime, ledger=ledger,
        )
    evidence["checkpoint"]["mrcra_configuration"] = asdict(config)
    evidence["cognitive"] = cognitive_evidence(output.cognitive, output.ledger)
    return evidence


def ensure_trackio_project(
    trackio_module: Any,
    *,
    project: str,
    output_dir: str | Path,
    prompt: str = "Relational continuity binds events across multiple temporal scales.",
    maximum_tokens: int = 32,
) -> bool:
    """Recover a missing/empty local project from one completed retained run.

    Returns true only when a canonical run was created. Existing projects with
    at least one run are never modified.
    """

    try:
        existing = list(trackio_module.Api().runs(project))
    except ValueError:
        existing = []
    if existing:
        return False
    output = Path(output_dir)
    manifest_path = output / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dashboard recovery requires {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("completed") is not True:
        raise ValueError("dashboard recovery requires a completed retained run")
    final = manifest["final_training_state"]
    runtime = manifest["runtime"]
    training = manifest["training_config"]
    step = int(final["step"])
    tokens = int(final["tokens_seen"])
    elapsed = float(final["elapsed_seconds"])
    metrics = {
        "progress/step": float(step),
        "progress/tokens_seen": float(tokens),
        "progress/valid_targets_seen": float(final["valid_targets_seen"]),
        "progress/fraction": 1.0,
        "train/utf8_bytes": float(final["bytes_seen"]),
        "performance/end_to_end_seconds": elapsed,
        "performance/end_to_end_tokens_per_second": tokens / max(elapsed, 1e-9),
        "architecture/model_parameters": float(manifest["model_parameters"]),
        "runtime/cpu_threads": float(runtime.get("cpu_threads", 0)),
        "runtime/cpu_interop_threads": float(runtime.get("cpu_interop_threads", 0)),
        "runtime/apple_hybrid_loss_offload": float(
            bool(runtime.get("apple_hybrid_loss_offload", False))
        ),
        "training/integrated_cognitive_path": float(
            bool(training.get("integrated_cognitive_path", False))
        ),
    }
    metric_path = _write_canonical_metric_mirror(
        output, step=step, metrics=metrics,
    )
    run_name = str(training.get("run_name") or output.name)
    trackio_module.init(
        project=project, name=run_name,
        config={
            "source": CANONICAL_TRACKIO_RUN_SOURCE,
            "run_manifest": str(manifest_path.resolve()),
            "model_profile": manifest.get("model_profile"),
            "model_parameters": manifest.get("model_parameters"),
            "runtime": runtime,
            "training": training,
        },
        resume="allow", embed=False, auto_log_cpu=False, auto_log_gpu=False,
    )
    try:
        trackio_module.log(metrics, step=step)
        evidence = _checkpoint_dashboard_evidence(
            _canonical_checkpoint(output), prompt=prompt,
            maximum_tokens=maximum_tokens,
        )
        evidence = attach_training_evidence(
            evidence, current_metrics=metric_path,
        )
        snapshot = write_evidence_atomically(
            output / "spectral" / SPECTRAL_DATA_FILENAME, evidence,
        )
        artifact = trackio_module.Artifact(
            SPECTRAL_ARTIFACT_NAME, type=SPECTRAL_ARTIFACT_TYPE,
            description=(
                "Checkpoint-grounded MRCRA cognitive and spectral dashboard evidence."
            ),
            metadata={
                "schema_version": evidence["schema_version"],
                "step": step, "tokens_seen": tokens,
            },
        )
        artifact.add_file(snapshot, name=SPECTRAL_DATA_FILENAME)
        trackio_module.log_artifact(artifact, aliases=[f"step-{step:07d}"])
    finally:
        trackio_module.finish()
    return True


def _try_series(path: Path, label: str) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return load_training_series(path, label=label)
    except (OSError, ValueError, json.JSONDecodeError):
        # The last JSONL record can be observed while its append is in flight;
        # the next snapshot will retry it. Spectral instrumentation must never
        # become part of optimization authority.
        return None


def attach_training_evidence(
    evidence: dict[str, Any],
    *,
    current_metrics: str | Path,
    baseline_metrics: str | Path | None = None,
) -> dict[str, Any]:
    """Attach available latest-run telemetry without mutating caller data."""

    # Only top-level observer fields are added.  Avoid duplicating the complete
    # cognitive/spectral tensor projection in memory before JSON serialization.
    result = dict(evidence)
    training = []
    if baseline_metrics is not None:
        baseline = _try_series(Path(baseline_metrics), "baseline")
        if baseline is not None:
            training.append(baseline)
    current = _try_series(Path(current_metrics), "current run")
    if current is not None:
        training.append(current)
    result["training"] = training
    result["schema_version"] = 1
    return result


def write_evidence_atomically(path: str | Path, evidence: dict[str, Any]) -> Path:
    """Publish complete JSON so Trackio never artifacts a partial snapshot."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="spectral-evidence-", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(evidence, handle, separators=(",", ":"), allow_nan=False)
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return destination
