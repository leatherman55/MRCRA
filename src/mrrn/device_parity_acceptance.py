"""Integrated, fail-closed device parity acceptance for MRCRA training.

The carrier micro-kernels and the full cognitive training path are both
exercised.  An available backend is accepted only after a finite 1K
forward/backward, optimized-composite parity, retain/checkpoint parity, sampled
CSTM reachability, checkpoint resume, and padding/cross-row isolation all
pass.  Unavailable devices remain explicitly untested.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import Tensor

from .carrier_execution import fused_simplex_residual
from .config import CognitiveConfig, MRCRAConfig, MRRNConfig
from .cognitive_training import MRCRANextTokenTrainer, MRCRATrainingConfig
from .language import MRCRALanguageModel
from .lm_training import ByteTextTokenizer
from .model import MRRN
from .resonance import associative_affine_scan
from .training_execution_fixture import (
    RepeatingPackedFixtureStream,
    build_execution_fixture,
)


DEVICE_PARITY_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class DeviceParityCriterion:
    name: str
    measurement: float
    threshold: float
    direction: str
    passed: bool


@dataclass(frozen=True, slots=True)
class DeviceParityResult:
    device: str
    status: str
    torch_device_name: str
    criteria: tuple[DeviceParityCriterion, ...]
    telemetry: dict[str, Any]
    passed: bool | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DeviceParityAcceptanceReport:
    format_version: int
    suite: str
    torch_version: str
    results: tuple[DeviceParityResult, ...]
    passed: bool
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _maximum(
    name: str, measurement: float, threshold: float
) -> DeviceParityCriterion:
    return DeviceParityCriterion(
        name, measurement, threshold, "maximum", measurement <= threshold
    )


def _minimum(
    name: str, measurement: float, threshold: float
) -> DeviceParityCriterion:
    return DeviceParityCriterion(
        name, measurement, threshold, "minimum", measurement >= threshold
    )


def _available(device_type: str) -> bool:
    if device_type == "cpu":
        return True
    if device_type == "mps":
        return bool(torch.backends.mps.is_available())
    if device_type == "cuda":
        return bool(torch.cuda.is_available())
    raise ValueError("unknown parity device")


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Metal Performance Shaders"
    return torch.backends.cpu.get_cpu_capability()


def _model_config(vocabulary_size: int) -> MRCRAConfig:
    carrier = MRRNConfig(
        input_dim=8,
        model_dim=8,
        output_dim=vocabulary_size,
        layers=1,
        scales=2,
        heads=2,
        modes=2,
        mimo_rank=1,
        attention_window=4,
        attention_query_tile_size=4,
        retrieved_items=1,
        memory_capacity=4,
        mixer_expansion=1.5,
        width_growth_cap=1,
        mode_growth_cap=1,
        width_multiple=2,
        spectral_modes=2,
        spectral_basis_order=2,
        spectral_triads_per_mode=1,
        enable_global_head=False,
        relational_branch=True,
        relational_context_dim=8,
        activation_checkpointing=False,
    )
    cognition = CognitiveConfig(
        workspace_dim=8,
        provenance_features=4,
        uncertainty_channels=8,
        relation_heads=2,
        relation_modes=2,
        relation_adapter_rank=2,
        goal_slots=1,
        goal_constraint_dim=2,
        system_action_channels=2,
        calibration_regimes=2,
        active_event_capacity=4,
        pair_edge_capacity=8,
        hyperedge_capacity=2,
        maximum_hyperedge_arity=3,
        graph_neighbors=1,
        global_workspace_slots=2,
        hypothesis_slots=1,
        maximum_hypothesis_slots=2,
        maximum_cognitive_steps=1,
        event_chunk_size=16,
        event_proposals_per_chunk=1,
        recent_candidates=2,
        landmark_candidates=1,
        episodic_candidates=1,
        semantic_candidates=1,
        episodic_memory_capacity=4,
        semantic_memory_capacity=2,
        associative_depth=1,
        associative_budget=1,
        world_model_horizons=(1,),
    )
    return MRCRAConfig(
        carrier,
        cognition,
        actor_parameter_minimum=1,
        actor_parameter_maximum=10_000_000,
    )


def _training_config(
    output_dir: str,
    *,
    device: str,
    activation_policy: str,
    total_tokens: int = 3_072,
) -> MRCRATrainingConfig:
    return MRCRATrainingConfig(
        output_dir=output_dir,
        total_tokens=total_tokens,
        context_length=1_024,
        execution_chunk_size=64,
        tbptt_length=256,
        vocabulary_tile_size=128,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_tokens=1_024,
        integrated_cognitive_path=True,
        document_static_batching=True,
        document_bucket_lengths=(128, 256),
        document_batch_token_budget=1_024,
        document_grouping_policy="cost_aware",
        cognitive_stride=128,
        cstm_enabled=True,
        cstm_execution="sampled",
        cstm_weight=0.04,
        cstm_warmup_tokens=0,
        cstm_ramp_tokens=1,
        cstm_sampling_duty_cycle=1.0,
        cstm_target_participation_budget=1_024,
        activation_policy=activation_policy,
        activation_calibration=False,
        allow_unsafe_activation_policy=True,
        exact_loss_backend="tiled",
        device=device,
        precision="fp32",
        cpu_threads=2,
        cpu_interop_threads=1,
        compile_tensor_cores=False,
        data_prefetch=False,
        trackio_enabled=False,
        show_dashboard=False,
        spectral_dashboard=False,
        phase_transition_telemetry=False,
        phase_transition_ablation=False,
        checkpoint_interval=100,
        evaluation_interval=0,
        evaluation_batches=0,
    )


def _state_digest(model: torch.nn.Module) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        local = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(local.dtype).encode("ascii"))
        digest.update(repr(tuple(local.shape)).encode("ascii"))
        digest.update(local.numpy().tobytes())
    return digest.hexdigest()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _integrated_adjoint(
    *,
    device_type: str,
    activation_policy: str,
    initial_state: dict[str, Tensor],
    output_dir: str,
) -> tuple[dict[str, float], dict[str, Tensor], dict[str, Any]]:
    tokenizer = ByteTextTokenizer()
    fixture = build_execution_fixture(
        "unit_1k", vocabulary_size=tokenizer.vocabulary_size
    )
    model = MRCRALanguageModel(_model_config(tokenizer.vocabulary_size))
    model.load_state_dict(initial_state)
    trainer = MRCRANextTokenTrainer(
        model,
        tokenizer,
        RepeatingPackedFixtureStream(fixture),
        _training_config(
            output_dir,
            device=device_type,
            activation_policy=activation_policy,
        ),
    )
    trainer.optimizer.zero_grad(set_to_none=True)
    metrics = trainer._run_context(fixture.batch, gradient_divisor=1)
    auxiliary_names = tuple(trainer._cstm_auxiliary_gradients)
    merge = trainer._merge_cstm_gradients()
    _synchronize(trainer.device)
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in trainer.model.named_parameters()
        if parameter.grad is not None
    }
    evidence = {
        "runtime": {
            "activation": trainer.activation_execution_policy.resolved,
            "carrier_backend": trainer.runtime["carrier_execution_backend"],
            "checkpoint_granularity": trainer.runtime[
                "carrier_checkpoint_granularity"
            ],
        },
        "auxiliary_names": auxiliary_names,
        "merge": merge,
        "model_state_digest": _state_digest(trainer.model),
    }
    return metrics, gradients, evidence


def _gradient_error(
    left: dict[str, Tensor], right: dict[str, Tensor]
) -> tuple[float, float]:
    if set(left) != set(right):
        raise RuntimeError("parity variants produced different gradient support")
    absolute = 0.0
    relative = 0.0
    for name in left:
        difference = (left[name].float() - right[name].float()).abs()
        absolute = max(absolute, float(difference.max()))
        denominator = right[name].float().abs().max().clamp_min(1e-7)
        relative = max(
            relative, float(difference.max() / denominator)
        )
    return absolute, relative


def _composite_parity(
    device: torch.device,
) -> tuple[float, float]:
    dtype = torch.float32
    generator = torch.Generator(device="cpu").manual_seed(20260726)
    transition = (
        torch.randn(2, 31, 2, 3, 2, generator=generator) * 0.04
    )
    transition[..., 0] += 0.86
    drive = torch.randn(
        2, 31, 2, 3, 2, generator=generator
    ) * 0.03
    initial = torch.randn(2, 2, 3, 2, generator=generator) * 0.05
    cotangent = torch.randn(
        2, 31, 2, 3, 2, generator=generator
    )

    def leaves(values):
        return tuple(
            value.to(device=device, dtype=dtype)
            .detach()
            .requires_grad_(True)
            for value in values
        )

    eager_inputs = leaves((transition, drive, initial))
    optimized_inputs = leaves((transition, drive, initial))
    eager = associative_affine_scan(*eager_inputs, implementation="composite")
    optimized = associative_affine_scan(
        *optimized_inputs, implementation="custom_adjoint"
    )
    eager_grad = torch.autograd.grad(eager, eager_inputs, cotangent.to(device))
    optimized_grad = torch.autograd.grad(
        optimized, optimized_inputs, cotangent.to(device)
    )
    errors = [
        float((optimized - eager).abs().max().detach().cpu()),
        *(
            float((actual - expected).abs().max().detach().cpu())
            for actual, expected in zip(
                optimized_grad, eager_grad, strict=True
            )
        ),
    ]

    band = torch.randn(2, 17, 8, generator=generator)
    logits = torch.randn(2, 17, 3, generator=generator)
    branches = tuple(
        torch.randn(2, 17, 8, generator=generator) for _ in range(3)
    )
    mask = torch.tensor(
        [[True] * 17, [True] * 13 + [False] * 4]
    )
    eager_values = leaves((band, logits, *branches))
    optimized_values = leaves((band, logits, *branches))
    eager_band, eager_logits, *eager_branches = eager_values
    opt_band, opt_logits, *opt_branches = optimized_values
    eager_weights = torch.softmax(eager_logits, -1)
    eager_simplex = (
        eager_band
        + 0.1
        * sum(
            eager_weights[..., index : index + 1] * branch
            for index, branch in enumerate(eager_branches)
        )
    ) * mask.to(device).unsqueeze(-1)
    optimized_simplex = fused_simplex_residual(
        opt_band,
        torch.softmax(opt_logits, -1),
        opt_band.new_tensor(0.1),
        mask.to(device),
        *opt_branches,
    )
    simplex_cotangent = torch.randn(
        2, 17, 8, generator=generator
    ).to(device)
    eager_simplex_grad = torch.autograd.grad(
        eager_simplex, eager_values, simplex_cotangent
    )
    optimized_simplex_grad = torch.autograd.grad(
        optimized_simplex, optimized_values, simplex_cotangent
    )
    simplex_errors = [
        float(
            (optimized_simplex - eager_simplex).abs().max().detach().cpu()
        ),
        *(
            float((actual - expected).abs().max().detach().cpu())
            for actual, expected in zip(
                optimized_simplex_grad,
                eager_simplex_grad,
                strict=True,
            )
        ),
    ]
    return max(errors), max(simplex_errors)


def _row_isolation(device: torch.device) -> tuple[float, float]:
    config = _model_config(257).carrier
    torch.manual_seed(20260727)
    model = MRRN(config).to(device).train()
    full = torch.randn(2, 12, config.input_dim, device=device)
    mask = torch.tensor(
        [[True] * 12, [True] * 7 + [False] * 5],
        device=device,
    )
    joint_input = full.detach().clone().requires_grad_(True)
    joint = model.prefill(joint_input, mask, project_output=False).latent
    joint_row_loss = joint[0].float().square().sum()
    joint_gradient = torch.autograd.grad(joint_row_loss, joint_input)[0]

    isolated_input = full[:1].detach().clone().requires_grad_(True)
    isolated = model.prefill(
        isolated_input,
        mask[:1],
        project_output=False,
    ).latent
    isolated_gradient = torch.autograd.grad(
        isolated.float().square().sum(), isolated_input
    )[0]
    row_error = max(
        float((joint[0] - isolated[0]).abs().max().detach().cpu()),
        float(
            (joint_gradient[0] - isolated_gradient[0])
            .abs()
            .max()
            .detach()
            .cpu()
        ),
    )
    padded_gradient = float(
        joint_gradient[1, 7:].abs().max().detach().cpu()
    )
    return row_error, padded_gradient


def _checkpoint_resume(
    *,
    device_type: str,
    initial_state: dict[str, Tensor],
    root: Path,
) -> tuple[float, bool, bool]:
    tokenizer = ByteTextTokenizer()
    fixture = build_execution_fixture(
        "unit_1k", vocabulary_size=tokenizer.vocabulary_size
    )

    def trainer(path: Path) -> MRCRANextTokenTrainer:
        model = MRCRALanguageModel(_model_config(tokenizer.vocabulary_size))
        model.load_state_dict(initial_state)
        return MRCRANextTokenTrainer(
            model,
            tokenizer,
            RepeatingPackedFixtureStream(fixture),
            _training_config(
                str(path),
                device=device_type,
                activation_policy="retain",
            ),
        )

    continuous = trainer(root / "continuous")
    continuous.train(maximum_steps=1)
    continuous.train(maximum_steps=1)

    interrupted = trainer(root / "interrupted")
    interrupted.train(maximum_steps=1)
    checkpoint = interrupted.save_checkpoint()
    resumed = trainer(root / "interrupted")
    resumed.load_checkpoint(checkpoint)
    resumed.train(maximum_steps=1)

    error = max(
        float(
            (
                resumed.model.state_dict()[name].detach().cpu().float()
                - expected.detach().cpu().float()
            )
            .abs()
            .max()
        )
        for name, expected in continuous.model.state_dict().items()
    )
    counters_exact = (
        resumed.state.step == continuous.state.step == 2
        and resumed.state.tokens_seen == continuous.state.tokens_seen == 2_048
        and resumed.state.valid_targets_seen
        == continuous.state.valid_targets_seen
        and resumed.state.bytes_seen == continuous.state.bytes_seen
    )
    schedule_exact = all(
        resumed.last_step_metrics[key]
        == continuous.last_step_metrics[key]
        for key in (
            "cstm/substrate_update",
            "cstm/predictor_update",
            "cstm/selected_invocation",
            "cstm/selected_scale",
            "cstm/sampling_inclusion_probability",
            "cstm/substrate_vjp_count",
        )
    )
    return error, counters_exact, schedule_exact


def _run_device(device_type: str) -> DeviceParityResult:
    if not _available(device_type):
        return DeviceParityResult(
            device_type,
            "untested_unavailable",
            "unavailable",
            (),
            {},
            None,
            None,
        )
    device = torch.device(device_type)
    tolerance = 2e-5 if device_type == "cpu" else 2e-3
    try:
        tokenizer = ByteTextTokenizer()
        torch.manual_seed(20260726)
        seed_model = MRCRALanguageModel(
            _model_config(tokenizer.vocabulary_size)
        )
        initial_state = {
            name: tensor.detach().clone()
            for name, tensor in seed_model.state_dict().items()
        }
        with tempfile.TemporaryDirectory(
            prefix=f"mrcra-device-parity-{device_type}-"
        ) as temporary:
            retain_metrics, retain_gradients, retain_evidence = (
                _integrated_adjoint(
                    device_type=device_type,
                    activation_policy="retain",
                    initial_state=initial_state,
                    output_dir=str(Path(temporary) / "retain"),
                )
            )
            checkpoint_metrics, checkpoint_gradients, checkpoint_evidence = (
                _integrated_adjoint(
                    device_type=device_type,
                    activation_policy="whole_span",
                    initial_state=initial_state,
                    output_dir=str(Path(temporary) / "checkpoint"),
                )
            )
            gradient_absolute, gradient_relative = _gradient_error(
                checkpoint_gradients, retain_gradients
            )
            checkpoint_error, counters_exact, schedule_exact = (
                _checkpoint_resume(
                    device_type=device_type,
                    initial_state=initial_state,
                    root=Path(temporary) / "resume",
                )
            )
        scan_error, simplex_error = _composite_parity(device)
        row_error, padded_gradient = _row_isolation(device)
        auxiliary_names = retain_evidence["auxiliary_names"]
        finite = all(
            torch.isfinite(value).all()
            for value in retain_gradients.values()
        ) and all(
            torch.isfinite(value).all()
            for value in checkpoint_gradients.values()
        )
        cstm_carrier = any(
            name.startswith("cognitive.carrier.")
            for name in auxiliary_names
        )
        cstm_cognition = any(
            name.startswith("cognitive.")
            and not name.startswith("cognitive.carrier.")
            for name in auxiliary_names
        )
        criteria = (
            _minimum(
                "finite_1k_integrated_forward_backward",
                float(finite),
                1.0,
            ),
            _maximum(
                "optimized_affine_scan_parity",
                scan_error,
                tolerance,
            ),
            _maximum(
                "optimized_simplex_parity",
                simplex_error,
                tolerance,
            ),
            _maximum(
                "retain_checkpoint_ce_error",
                abs(
                    checkpoint_metrics[
                        "train/cross_entropy_nats_per_token"
                    ]
                    - retain_metrics[
                        "train/cross_entropy_nats_per_token"
                    ]
                ),
                tolerance,
            ),
            _maximum(
                "retain_checkpoint_gradient_absolute_error",
                gradient_absolute,
                tolerance,
            ),
            _minimum(
                "sampled_cstm_carrier_reachability",
                float(cstm_carrier),
                1.0,
            ),
            _minimum(
                "sampled_cstm_cognition_reachability",
                float(cstm_cognition),
                1.0,
            ),
            _maximum(
                "checkpoint_resume_parameter_error",
                checkpoint_error,
                tolerance,
            ),
            _minimum(
                "checkpoint_resume_counter_exactness",
                float(counters_exact),
                1.0,
            ),
            _minimum(
                "checkpoint_resume_schedule_exactness",
                float(schedule_exact),
                1.0,
            ),
            _maximum(
                "cross_row_forward_gradient_leakage",
                row_error,
                tolerance,
            ),
            _maximum(
                "padding_gradient_leakage",
                padded_gradient,
                tolerance,
            ),
        )
        telemetry = {
            "retain_cross_entropy_nats_per_token": retain_metrics[
                "train/cross_entropy_nats_per_token"
            ],
            "checkpoint_cross_entropy_nats_per_token": checkpoint_metrics[
                "train/cross_entropy_nats_per_token"
            ],
            "gradient_relative_error": gradient_relative,
            "retain_runtime": retain_evidence["runtime"],
            "checkpoint_runtime": checkpoint_evidence["runtime"],
            "retain_model_state_digest": retain_evidence[
                "model_state_digest"
            ],
            "checkpoint_model_state_digest": checkpoint_evidence[
                "model_state_digest"
            ],
            "cstm_substrate_vjps": retain_metrics[
                "cstm/substrate_vjp_count"
            ],
            "cstm_auxiliary_parameter_count": len(auxiliary_names),
        }
        return DeviceParityResult(
            device_type,
            "tested",
            _device_name(device),
            criteria,
            telemetry,
            all(item.passed for item in criteria),
            None,
        )
    except Exception as error:
        return DeviceParityResult(
            device_type,
            "failed",
            _device_name(device),
            (),
            {},
            False,
            f"{type(error).__name__}: {error}",
        )


def run_device_parity_acceptance() -> DeviceParityAcceptanceReport:
    results = tuple(_run_device(name) for name in ("cpu", "mps", "cuda"))
    tested = tuple(item for item in results if item.status != "untested_unavailable")
    return DeviceParityAcceptanceReport(
        DEVICE_PARITY_FORMAT_VERSION,
        "mrcra_integrated_available_device_parity",
        torch.__version__,
        results,
        bool(tested) and all(item.passed is True for item in tested),
        (
            "Pass proves the specified tiny 1K integrated training, custom "
            "composite, activation recomputation, CSTM reachability, exact "
            "resume, and row-isolation contracts on every locally available "
            "backend. Unavailable backends are untested, never passed."
        ),
    )
