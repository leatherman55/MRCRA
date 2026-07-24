"""Evidence exporters for the MRRN spectral visualization suite.

The exporter deliberately emits plain JSON-compatible values.  The visual
instruments therefore never need to import PyTorch, execute a model, or infer
missing quantities in the browser: every plotted mark is derived here from a
training log or a checkpoint tensor.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from math import log
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import MRRNConfig
from .language import MRRNLanguageModel
from .mixer import HybridSpectralMixer

if TYPE_CHECKING:
    from .lm_training import TextTokenizer


def _finite(value: Any) -> float | None:
    """Return a finite float or ``None`` for absent/non-finite telemetry."""

    if value is None:
        return None
    result = float(value)
    return result if torch.isfinite(torch.tensor(result)) else None


def _observed_mean(
    value: Tensor, *, digits: int, label: str,
) -> float | None:
    """Reduce observed diagnostic samples without inventing inactive values.

    Empty coarse-scale tensors are a valid consequence of a prompt shorter
    than that scale's support.  They are absent observations, not numerical
    zeroes.  Non-empty non-finite tensors remain an error so instrumentation
    cannot conceal a genuine model or diagnostic instability.
    """

    if value.numel() == 0:
        return None
    result = float(value.detach().float().mean())
    if not torch.isfinite(torch.tensor(result)):
        raise FloatingPointError(f"non-finite {label} diagnostic")
    return round(result, digits)


def _observed_rms(
    value: Tensor, *, digits: int, label: str,
) -> float | None:
    """Return an RMS for observed samples, or absence for an empty scale."""

    if value.numel() == 0:
        return None
    result = float(value.detach().float().square().sum(-1).mean().sqrt())
    if not torch.isfinite(torch.tensor(result)):
        raise FloatingPointError(f"non-finite {label} diagnostic")
    return round(result, digits)


def load_training_series(path: str | Path, *, label: str) -> dict[str, Any]:
    """Read the metric records that contain an optimization step.

    Trackio alert/evaluation records may share a step with the training row.
    Keeping only rows with a pre-clip gradient norm gives one unambiguous sample
    per optimizer update, including legacy logs that predate the explicit
    ``gradient_norm_after_clip`` field.
    """

    samples: list[dict[str, float | int | None]] = []
    evaluation_rows: list[tuple[int, int, dict[str, float | None]]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            metrics = record.get("metrics")
            if record.get("kind") != "metrics" or not isinstance(metrics, dict):
                continue
            step = int(record["step"])
            if (
                any(name.startswith("eval/phase_ablation/") for name in metrics)
                or "pc_rasl/guard_ce_nats_per_token" in metrics
            ):
                evaluation_rows.append((line_index, step, {
                    "ablation_full_ce": _finite(metrics.get("eval/phase_ablation/full_ce_nats_per_token")),
                    "ablation_soft_ce": _finite(metrics.get("eval/phase_ablation/soft_only_ce_nats_per_token")),
                    "ablation_off_ce": _finite(metrics.get("eval/phase_ablation/cognition_off_ce_nats_per_token")),
                    "hard_ce_gain": _finite(metrics.get("eval/phase_ablation/hard_structure_ce_gain")),
                    "soft_ce_gain": _finite(metrics.get("eval/phase_ablation/soft_bridge_ce_gain")),
                    "pc_guard_ce": _finite(metrics.get("pc_rasl/guard_ce_nats_per_token")),
                    "pc_guard_best_ce": _finite(metrics.get("pc_rasl/guard_best_ce_nats_per_token")),
                    "pc_guard_allows_positive": _finite(metrics.get("pc_rasl/guard_allows_positive_pressure")),
                }))
            pre = metrics.get("optimization/gradient_norm_before_clip")
            if pre is None:
                continue
            samples.append(
                {
                    "step": step,
                    "_line": line_index,
                    "tokens": int(metrics.get("progress/tokens_seen", 0)),
                    "gradient_pre": _finite(pre),
                    "gradient_post": _finite(metrics.get("optimization/gradient_norm_after_clip")),
                    "clip_coefficient": _finite(metrics.get("optimization/gradient_clip_coefficient")),
                    "state_rms": _finite(metrics.get("architecture/state_rms")),
                    "state_rms_max": _finite(metrics.get("architecture/state_rms_max")),
                    "mean_decay": _finite(metrics.get("architecture/mean_decay")),
                    "branch_resonance": _finite(metrics.get("architecture/branch_resonance")),
                    "branch_local": _finite(metrics.get("architecture/branch_local")),
                    "branch_attention": _finite(metrics.get("architecture/branch_attention")),
                    "branch_identity": _finite(metrics.get("architecture/branch_identity")),
                    "ce": _finite(metrics.get("train/cross_entropy_nats_per_token")),
                    "ece": _finite(metrics.get("train/effective_cross_entropy_nats_per_byte")),
                    "tokens_per_second": _finite(metrics.get("performance/tokens_per_second")),
                    "step_seconds": _finite(metrics.get("performance/step_seconds")),
                    "pc_update_seconds": _finite(metrics.get("performance/pc_rasl_seconds")),
                    "pc_capture_seconds": _finite(metrics.get("performance/pc_rasl_capture_seconds")),
                    "proposal_mean": _finite(metrics.get("architecture/event_proposal_probability_mean")),
                    "proposal_median": _finite(metrics.get("architecture/event_proposal_probability_median")),
                    "proposal_p90": _finite(metrics.get("architecture/event_proposal_probability_p90")),
                    "proposal_p99": _finite(metrics.get("architecture/event_proposal_probability_p99")),
                    "proposal_max": _finite(metrics.get("architecture/event_proposal_probability_max")),
                    "phase_distance": _finite(metrics.get("architecture/event_phase_distance_to_threshold")),
                    "proposal_slope": _finite(metrics.get("architecture/event_proposal_logit_slope_ema")),
                    "proposal_ge_025": _finite(metrics.get("architecture/event_proposal_fraction_ge_0p25")),
                    "proposal_ge_035": _finite(metrics.get("architecture/event_proposal_fraction_ge_0p35")),
                    "proposal_ge_045": _finite(metrics.get("architecture/event_proposal_fraction_ge_0p45")),
                    "proposal_ge_050": _finite(metrics.get("architecture/event_proposal_fraction_ge_0p50")),
                    "event_opened": _finite(metrics.get("architecture/event_opened")),
                    "event_finalized": _finite(metrics.get("architecture/event_finalized")),
                    "event_emitted": _finite(metrics.get("architecture/event_emitted")),
                    "event_quota_rejected": _finite(metrics.get("architecture/event_quota_rejected")),
                    "gradient_carrier": _finite(metrics.get("optimization/gradient/carrier_before_clip")),
                    "gradient_event": _finite(metrics.get("optimization/gradient/event_before_clip")),
                    "gradient_output_bridge": _finite(metrics.get("optimization/gradient/output_bridge_before_clip")),
                    "gradient_controller": _finite(metrics.get("optimization/gradient/controller_before_clip")),
                    "gradient_workspace_router": _finite(metrics.get("optimization/gradient/workspace_router_before_clip")),
                    "gradient_world_hypothesis": _finite(metrics.get("optimization/gradient/world_hypothesis_before_clip")),
                    "gradient_memory": _finite(metrics.get("optimization/gradient/memory_before_clip")),
                    "effective_lr": _finite(metrics.get("optimization/effective_learning_rate")),
                    "ablation_full_ce": None,
                    "ablation_soft_ce": None,
                    "ablation_off_ce": None,
                    "hard_ce_gain": None,
                    "soft_ce_gain": None,
                    "pc_pressure": _finite(metrics.get("pc_rasl/progress_pressure")),
                    "pc_raw_pressure": _finite(metrics.get("pc_rasl/raw_progress_pressure")),
                    "pc_confidence": _finite(metrics.get("pc_rasl/progress_confidence")),
                    "pc_probe_ce": _finite(metrics.get("pc_rasl/probe_ce_nats_per_token")),
                    "pc_probe_seconds": _finite(metrics.get("pc_rasl/probe_seconds")),
                    "pc_expected_ce": _finite(metrics.get("pc_rasl/expected_ce_nats_per_token")),
                    "pc_observed_slope": _finite(metrics.get("pc_rasl/observed_ce_slope_per_million_tokens")),
                    "pc_expected_slope": _finite(metrics.get("pc_rasl/expected_ce_slope_per_million_tokens")),
                    "pc_slope_advantage_z": _finite(metrics.get("pc_rasl/slope_advantage_z")),
                    "pc_debt": _finite(metrics.get("pc_rasl/progress_debt_nats_per_token")),
                    "pc_debt_z": _finite(metrics.get("pc_rasl/debt_z")),
                    "pc_baseline_ready": _finite(metrics.get("pc_rasl/baseline_ready")),
                    "pc_warmup_complete": _finite(metrics.get("pc_rasl/warmup_complete")),
                    "pc_guard_allows_positive": _finite(metrics.get("pc_rasl/guard_allows_positive_pressure")),
                    "pc_guard_ce": None,
                    "pc_guard_best_ce": None,
                    "pc_critic_loss": _finite(metrics.get("pc_rasl/critic_loss")),
                    "pc_fsce": _finite(metrics.get("pc_rasl/functional_cross_entropy")),
                    "pc_internal_policy_loss": _finite(metrics.get("pc_rasl/internal_policy_loss")),
                    "pc_progress_return_loss": _finite(metrics.get("pc_rasl/progress_return_loss")),
                    "pc_internal_value_loss": _finite(metrics.get("pc_rasl/internal_action_value_loss")),
                    "pc_aux_gradient_before": _finite(metrics.get("pc_rasl/actor_auxiliary_gradient_norm_before")),
                    "pc_aux_gradient_after": _finite(metrics.get("pc_rasl/actor_auxiliary_gradient_norm_after")),
                    "pc_actor_ready": _finite(metrics.get("pc_rasl/actor_auxiliary_ready")),
                    "pc_performance_guard_allows": _finite(metrics.get("pc_rasl/performance_guard_allows_actor")),
                    "pc_performance_guard_rejections": _finite(metrics.get("pc_rasl/performance_guard_rejections")),
                    "pc_actor_warmup_ready": _finite(metrics.get("pc_rasl/actor_warmup_ready")),
                    "pc_replay_transitions": _finite(metrics.get("pc_rasl/replay_transitions")),
                    "pc_replay_storage_bytes": _finite(metrics.get("pc_rasl/replay_storage_bytes")),
                    "pc_behavior_evidence_bound": _finite(metrics.get("pc_rasl/behavior_evidence_bound")),
                }
            )
    if not samples:
        raise ValueError(f"{path} contains no optimizer-step telemetry")
    # Training logs are intentionally append-only.  A fresh run in the same
    # output directory is visible as a step reset, so visualize the latest
    # monotonic run rather than drawing a false line backward through time.
    latest_start = 0
    for index in range(1, len(samples)):
        if samples[index]["step"] <= samples[index - 1]["step"]:
            latest_start = index
    samples = samples[latest_start:]
    latest_line = int(samples[0]["_line"])
    evaluation_by_step: dict[int, dict[str, float | None]] = {}
    for line_index, step, values in evaluation_rows:
        if line_index < latest_line:
            continue
        retained = {name: value for name, value in values.items() if value is not None}
        evaluation_by_step.setdefault(step, {}).update(retained)
    for sample in samples:
        sample.update(evaluation_by_step.get(int(sample["step"]), {}))
        sample.pop("_line")
    return {"label": label, "source": str(Path(path)), "samples": samples}


def _round_list(values: Tensor, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values.detach().cpu().reshape(-1)]


def _modal_snapshot(value: Tensor) -> tuple[list[float], list[float]]:
    """Reduce batch/head/rank while preserving every resonant mode."""

    real, imag = value[..., 0], value[..., 1]
    amplitude = (real.square() + imag.square()).mean(dim=(0, 1, 3)).sqrt()
    resultant_real = real.sum(dim=(0, 1, 3))
    resultant_imag = imag.sum(dim=(0, 1, 3))
    phase = torch.atan2(resultant_imag, resultant_real)
    return _round_list(amplitude), _round_list(phase)


def _base_resonator_parameters(module) -> tuple[Tensor, Tensor, Tensor]:
    alpha = module.alpha_min + F.softplus(module.raw_alpha.detach().float())
    omega = module.omega_max * torch.tanh(module.raw_omega.detach().float())
    alpha = alpha.mean(0)
    omega = omega.mean(0)
    half_life = log(2.0) / alpha
    return alpha, omega, half_life


def _visible_token(tokenizer: "TextTokenizer", token_id: int) -> str:
    value = tokenizer.decode([token_id])
    return value.replace("\n", "↵").replace("\t", "⇥") or "∅"


@torch.no_grad()
def model_spectral_evidence(
    model: MRRNLanguageModel,
    tokenizer: "TextTokenizer",
    *,
    prompt: str,
    maximum_tokens: int = 32,
    step: int = 0,
    tokens_seen: int = 0,
    source: str = "live model",
    format_version: int | None = None,
) -> dict[str, Any]:
    """Extract all four linked spectral views from an existing language model."""

    if maximum_tokens <= 0:
        raise ValueError("maximum_tokens must be positive")
    # Both the sequence-only language model and MRCRA expose the same tied
    # embedding plus carrier contract.  Resolve the carrier without importing
    # the cognitive adapter here (which would create an import cycle).
    if hasattr(model, "actor"):
        actor = model.actor
        config = model.config
    elif hasattr(model, "cognitive") and hasattr(model.config, "carrier"):
        actor = model.cognitive.carrier
        config = model.config.carrier
    else:
        raise TypeError("spectral evidence requires an MRRN or MRCRA language actor")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    ids = tokenizer.encode_prompt(prompt)[:maximum_tokens]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    labels = [_visible_token(tokenizer, value) for value in ids]

    scale_configs = config.scale_configs()
    traces: list[list[dict[str, Any]]] = []
    for block_index, block in enumerate(actor.blocks):
        block_scales = []
        for scale_index, resonator in enumerate(block.resonators):
            alpha, omega, half_life = _base_resonator_parameters(resonator)
            block_scales.append(
                {
                    "block": block_index,
                    "scale": scale_index,
                    "support": 2 ** (scale_index + 1) if scale_index < config.scales - 1 else 2**scale_index,
                    "frequencies": _round_list(omega),
                    "decays": _round_list(alpha),
                    "half_lives": _round_list(half_life, 3),
                    "amplitude": [],
                    "phase": [],
                    "active": [],
                }
            )
        traces.append(block_scales)

    state = actor.initial_stream_state(1, device=device, dtype=model.token_embedding.weight.dtype)
    for token_id in ids:
        stream_step = actor.step(
            model.token_embedding(torch.tensor([token_id], device=device)), state
        )
        state = stream_step.state
        for block_index, block_state in enumerate(state.blocks):
            for scale_index, resonator_state in enumerate(block_state.resonators):
                amplitude, phase = _modal_snapshot(resonator_state.value)
                traces[block_index][scale_index]["amplitude"].append(amplitude)
                traces[block_index][scale_index]["phase"].append(phase)
                traces[block_index][scale_index]["active"].append(
                    stream_step.active_bands[scale_index] is not None
                )

    batch_output = actor(model.token_embedding(input_ids), output_mode="sequence")
    branch_mix: list[dict[str, Any]] = []
    triads: list[dict[str, Any]] = []
    for block_index, (block, diagnostic) in enumerate(
        zip(actor.blocks, batch_output.diagnostics, strict=True)
    ):
        for scale_index, (mixer, branch, mixer_diagnostic) in enumerate(
            zip(block.mixers, diagnostic.branch_weights, diagnostic.spectral_mixers, strict=True)
        ):
            sample_count = int(branch[..., 0].numel())
            branch_values = [
                _observed_mean(
                    branch[..., index], digits=6,
                    label=f"branch {name} at block {block_index} scale {scale_index}",
                )
                for index, name in enumerate(
                    ("resonance", "local", "attention", "identity")
                )
            ]
            branch_mix.append(
                {
                    "block": block_index,
                    "scale": scale_index,
                    "active": sample_count > 0,
                    "sample_count": sample_count,
                    "resonance": branch_values[0],
                    "local": branch_values[1],
                    "attention": branch_values[2],
                    "identity": branch_values[3],
                    "spectral_fraction": None,
                }
            )
            if not isinstance(mixer, HybridSpectralMixer) or mixer_diagnostic is None:
                continue
            spectral = mixer.spectral
            branch_mix[-1]["spectral_fraction"] = _observed_mean(
                mixer_diagnostic.spectral_fraction,
                digits=6,
                label=(
                    f"spectral fraction at block {block_index} "
                    f"scale {scale_index}"
                ),
            )
            actual_weight = spectral.maximum_triad_gain * torch.tanh(
                spectral.raw_triad_weight.detach().float()
            )
            strength = actual_weight.abs().mean(dim=(0, 2))
            signed = actual_weight.mean(dim=(0, 2))
            triad_values = mixer_diagnostic.spectral.triad.detach().float()
            gate_values = mixer_diagnostic.spectral.amplitude_gate.detach().float()
            phase_values = mixer_diagnostic.spectral.phase_rotation.detach().float()
            frequencies = spectral.frequencies.detach().float()
            for edge in range(spectral.triad_target.numel()):
                target = int(spectral.triad_target[edge])
                left = int(spectral.triad_left[edge])
                right = int(spectral.triad_right[edge])
                difference = bool(spectral.triad_conjugate[edge])
                triads.append(
                    {
                        "block": block_index,
                        "scale": scale_index,
                        "target": target,
                        "left": left,
                        "right": right,
                        "operation": "difference" if difference else "sum",
                        "target_frequency": round(float(frequencies[target]), 6),
                        "left_frequency": round(float(frequencies[left]), 6),
                        "right_frequency": round(float(frequencies[right]), 6),
                        "strength": round(float(strength[edge]), 8),
                        "signed_weight": round(float(signed[edge]), 8),
                        "active": sample_count > 0,
                        "sample_count": sample_count,
                        "activity": _observed_rms(
                            triad_values[..., target, :, :],
                            digits=8,
                            label=(
                                f"triad activity at block {block_index} "
                                f"scale {scale_index} mode {target}"
                            ),
                        ),
                        "mean_gate": _observed_mean(
                            gate_values[..., target, :],
                            digits=6,
                            label=(
                                f"triad gate at block {block_index} "
                                f"scale {scale_index} mode {target}"
                            ),
                        ),
                        "mean_phase": _observed_mean(
                            phase_values[..., target, :],
                            digits=6,
                            label=(
                                f"triad phase at block {block_index} "
                                f"scale {scale_index} mode {target}"
                            ),
                        ),
                    }
                )

    poles: list[dict[str, Any]] = []
    final_index = len(ids) - 1
    for block_scales in traces:
        for scale in block_scales:
            for mode, (decay, frequency, half_life) in enumerate(
                zip(scale["decays"], scale["frequencies"], scale["half_lives"], strict=True)
            ):
                poles.append(
                    {
                        "block": scale["block"],
                        "scale": scale["scale"],
                        "mode": mode,
                        "decay": decay,
                        "frequency": frequency,
                        "half_life": half_life,
                        "amplitude": scale["amplitude"][final_index][mode],
                    }
                )

    evidence = {
        "checkpoint": {
            "path": source,
            "format_version": format_version,
            "step": int(step),
            "tokens_seen": int(tokens_seen),
            "parameter_count": int(model.parameter_count),
            "configuration": asdict(config),
        },
        "prompt": prompt,
        "tokens": [{"index": index, "id": token_id, "text": text} for index, (token_id, text) in enumerate(zip(ids, labels, strict=True))],
        "traces": traces,
        "branch_mix": branch_mix,
        "poles": poles,
        "triads": triads,
    }
    model.train(was_training)
    return evidence


@torch.no_grad()
def checkpoint_spectral_evidence(
    checkpoint: str | Path,
    *,
    prompt: str,
    maximum_tokens: int = 32,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a language-model checkpoint and extract four linked spectral views."""

    if maximum_tokens <= 0:
        raise ValueError("maximum_tokens must be positive")
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    identity = payload.get("identity", {})
    configuration = identity.get("model_config")
    tokenizer_identity = identity.get("tokenizer")
    if not isinstance(configuration, dict) or not isinstance(tokenizer_identity, dict):
        raise ValueError("checkpoint lacks model/tokenizer identity")
    if tokenizer_identity.get("kind") != "huggingface":
        raise ValueError("the standalone exporter currently requires a Hugging Face tokenizer")

    # Lazy import prevents the training module from forming an import cycle when
    # it publishes live spectral snapshots to Trackio.
    from .lm_training import HuggingFaceTextTokenizer

    config = MRRNConfig(**configuration)
    model = MRRNLanguageModel(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    tokenizer = HuggingFaceTextTokenizer(
        tokenizer_identity["name"], revision=tokenizer_identity["revision"]
    )
    training_state = payload.get("training_state", {})
    return model_spectral_evidence(
        model,
        tokenizer,
        prompt=prompt,
        maximum_tokens=maximum_tokens,
        step=int(training_state.get("step", 0)),
        tokens_seen=int(training_state.get("tokens_seen", 0)),
        source=str(checkpoint),
        format_version=payload.get("format_version"),
    )


def build_visualization_dataset(
    *,
    checkpoint: str | Path,
    stable_metrics: str | Path,
    baseline_metrics: str | Path,
    prompt: str,
    maximum_tokens: int = 32,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Build the single evidence bundle consumed by all four instruments."""

    evidence = checkpoint_spectral_evidence(
        checkpoint, prompt=prompt, maximum_tokens=maximum_tokens, device=device
    )
    evidence["training"] = [
        load_training_series(baseline_metrics, label="legacy drive"),
        load_training_series(stable_metrics, label="decay-normalized drive"),
    ]
    evidence["schema_version"] = 1
    return evidence


def write_visualization_dataset(path: str | Path, evidence: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(evidence, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
