"""Empirical acceptance authority for Causal Spectral Target Multiplexing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import redirect_stdout
from io import StringIO
from math import isfinite, pi
from typing import Any

import torch
from torch import Tensor

from .config import CognitiveConfig, MRCRAConfig, MRRNConfig
from .cstm import (
    CSTMArchitectureConfig,
    CausalSpectralTargetPredictor,
    build_causal_spectral_targets,
    deterministic_token_codes,
)
from .language import MRCRALanguageModel
from .cognitive_training import MRCRANextTokenTrainer, MRCRATrainingConfig
from .lm_training import (
    ByteTextTokenizer, PackedTokenStream, SequenceTextSource,
)
from .optimization import gradient_subsystem


@dataclass(frozen=True, slots=True)
class CSTMCriterion:
    name: str
    threshold: float
    comparison: str
    observed: float

    def __post_init__(self) -> None:
        if (
            self.comparison not in {"at_most", "at_least", "equal"}
            or not isfinite(self.threshold)
            or not isfinite(self.observed)
        ):
            raise ValueError("CSTM criterion must be finite and well formed")

    @property
    def passed(self) -> bool:
        if self.comparison == "at_most":
            return self.observed <= self.threshold
        if self.comparison == "at_least":
            return self.observed >= self.threshold
        if self.comparison == "equal":
            return self.observed == self.threshold
        raise ValueError(f"unsupported CSTM comparison {self.comparison!r}")


@dataclass(frozen=True, slots=True)
class CSTMExperiment:
    name: str
    metrics: dict[str, float]
    description: str

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.description
            or not self.metrics
            or any(not isfinite(value) for value in self.metrics.values())
        ):
            raise ValueError(
                "CSTM experiment metrics must be nonempty and finite"
            )


@dataclass(frozen=True, slots=True)
class CSTMAcceptanceReport:
    schema_version: int
    passed: bool
    criteria: tuple[CSTMCriterion, ...]
    experiments: tuple[CSTMExperiment, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["criteria"] = [
            {**asdict(item), "passed": item.passed}
            for item in self.criteria
        ]
        return result


def _direct_dft(codes: Tensor) -> Tensor:
    support = codes.shape[0]
    phase = 2 * pi * torch.arange(support, dtype=codes.dtype) / support
    return torch.stack(
        (
            codes.sum(0),
            (codes * phase.cos()[:, None]).sum(0),
            (codes * -phase.sin()[:, None]).sum(0),
        )
    ) / support**0.5


def benchmark_target_authority() -> CSTMExperiment:
    """Measure direct-DFT agreement, order sensitivity, and boundary rejection."""

    labels = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    segments = torch.tensor([[0] * 6 + [1] * 6], dtype=torch.int64)
    target_segments = segments.clone()
    mask = torch.ones_like(labels, dtype=torch.bool)
    mask[:, 5] = False
    mask[:, 11] = False
    target_segments[:, 5] = 1
    target_segments[:, 11] = 2
    codes = deterministic_token_codes(32, 64, seed=20260725)
    targets = build_causal_spectral_targets(
        labels,
        mask,
        segments,
        target_segments,
        codes,
        torch.tensor([0, 3, 6, 9]),
        support=4,
        horizons=(1,),
    )
    direct = _direct_dft(codes[labels[0, :4]])
    dft_error = float(
        (targets.values[0, 0, 0] - direct).abs().max()
    )
    permuted_labels = labels.clone()
    permuted_labels[0, :4] = permuted_labels[0, torch.tensor([2, 0, 3, 1])]
    permuted = build_causal_spectral_targets(
        permuted_labels,
        mask,
        segments,
        target_segments,
        codes,
        torch.tensor([0]),
        support=4,
        horizons=(1,),
    )
    dc_change = float(
        (
            targets.values[0, 0, 0, 0]
            - permuted.values[0, 0, 0, 0]
        ).abs().max()
    )
    harmonic_change = float(
        (
            targets.values[0, 0, 0, 1:]
            - permuted.values[0, 0, 0, 1:]
        )
        .square()
        .mean()
        .sqrt()
    )
    boundary_leak_rows = float(
        targets.mask[0, 1].sum() + targets.mask[0, 3].sum()
    )
    return CSTMExperiment(
        "fixed_target_authority",
        {
            "direct_dft_max_abs_error": dft_error,
            "permutation_dc_max_abs_change": dc_change,
            "permutation_harmonic_rms_change": harmonic_change,
            "cross_boundary_valid_rows": boundary_leak_rows,
        },
        (
            "Compares production target construction with a direct negative-"
            "exponent DFT, permutes block order, and probes blocks that would "
            "cross packed-document boundaries."
        ),
    )


def benchmark_predictor_learning() -> CSTMExperiment:
    """Fit the real shared low-rank head to fixed multihorizon spectral targets."""

    torch.manual_seed(20260725)
    configuration = CSTMArchitectureConfig(
        code_dimension=32,
        predictor_rank=8,
        horizon_blocks=(1, 2),
        target_rms_decay=0.9,
    )
    predictor = CausalSpectralTargetPredictor(16, 3, 64, configuration)
    labels = torch.randint(0, 64, (1, 96), dtype=torch.int64)
    mask = torch.ones_like(labels, dtype=torch.bool)
    segments = torch.zeros_like(labels)
    sources = torch.arange(3, 83, 4, dtype=torch.int64)
    targets = build_causal_spectral_targets(
        labels,
        mask,
        segments,
        segments,
        predictor.token_codes,
        sources,
        support=4,
        horizons=(1, 2),
    )
    carrier = torch.randn(1, sources.numel(), 16)
    cognition = torch.randn_like(carrier)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=2e-2)

    def loss(update_statistics: bool) -> Tensor:
        prediction = predictor(
            carrier, cognition, scale=1, horizons=(1, 2)
        )
        return predictor.loss(
            prediction,
            targets,
            scale=1,
            update_statistics=update_statistics,
        ).loss

    initial = float(loss(True).detach())
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        current = loss(False)
        current.backward()
        optimizer.step()
    final = float(loss(False).detach())
    return CSTMExperiment(
        "predictor_learning",
        {
            "initial_standardized_huber": initial,
            "final_standardized_huber": final,
            "final_to_initial_ratio": final / max(initial, 1e-12),
            "valid_target_rows": float(targets.valid_rows),
            "token_participations": float(targets.token_participations),
        },
        (
            "Optimizes the production shared rank-eight, scale/horizon-"
            "conditioned, paired-real predictor against fixed CSTM targets."
        ),
    )


def _tiny_integrated_config(vocabulary_size: int) -> MRCRAConfig:
    carrier = MRRNConfig(
        input_dim=8,
        model_dim=8,
        output_dim=vocabulary_size,
        layers=1,
        scales=3,
        heads=2,
        modes=2,
        mimo_rank=1,
        attention_window=2,
        attention_query_tile_size=4,
        retrieved_items=1,
        memory_capacity=8,
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
        event_chunk_size=2,
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


def benchmark_integrated_causality() -> CSTMExperiment:
    """Perturb future tokens and measure earlier CSTM prediction invariance."""

    torch.manual_seed(20260726)
    model = MRCRALanguageModel(_tiny_integrated_config(64)).double().eval()
    tokens = torch.randint(0, 64, (1, 20), dtype=torch.int64)
    changed = tokens.clone()
    changed[:, 12:] = (changed[:, 12:] + 17) % 64

    def predictions(values: Tensor):
        packet, ledger = model.prepare_external_input(values)
        output = model.cognitive.forward_integrated_training(
            packet,
            ledger,
            cognitive_stride=2,
            cognition_mode="full",
        )
        return model.predict_causal_spectral_targets(
            output,
            extra_horizon_offset=0,
        )

    with torch.no_grad():
        baseline = predictions(tokens)
        perturbed = predictions(changed)
    past_change = 0.0
    future_change = 0.0
    compared_past = 0
    for left, right in zip(baseline, perturbed, strict=True):
        torch.testing.assert_close(left.source_positions, right.source_positions)
        past = left.source_positions < 12
        future = left.source_positions >= 12
        if bool(past.any()):
            past_change = max(
                past_change,
                float(
                    (left.values[:, past] - right.values[:, past]).abs().max()
                ),
            )
            compared_past += int(past.sum())
        if bool(future.any()):
            future_change = max(
                future_change,
                float(
                    (left.values[:, future] - right.values[:, future]).abs().max()
                ),
            )
    return CSTMExperiment(
        "integrated_causality",
        {
            "past_prediction_max_abs_change": past_change,
            "future_prediction_max_abs_change": future_change,
            "past_rows_compared": float(compared_past),
        },
        (
            "Runs the real causal carrier, event-rate cognitive residual, band "
            "history, synthesis adapters, and CSTM head twice while changing "
            "only tokens at and after position twelve."
        ),
    )


def benchmark_efficiency_and_parameter_contract() -> CSTMExperiment:
    """Measure the geometric target-row bound and real 2.7M head allocation."""

    context = 32_768
    supports = (2, 4, 8, 16, 32, 32)
    horizons_per_scale = 2
    rows = sum(context // support for support in supports) * horizons_per_scale
    row_ratio = rows / context
    configuration = MRCRAConfig.ultralight_2p7m(output_dim=50_257)
    model = MRCRALanguageModel(configuration)
    head_parameters = sum(
        parameter.numel()
        for parameter in model.cstm_predictor.parameters()
    )
    return CSTMExperiment(
        "efficiency_and_parameter_contract",
        {
            "context_tokens": float(context),
            "two_horizon_target_rows": float(rows),
            "target_rows_per_physical_token": row_ratio,
            "cstm_predictor_parameters": float(head_parameters),
            "ultralight_actor_parameters": float(model.parameter_count),
            "ultralight_parameter_maximum": float(
                configuration.actor_parameter_maximum
            ),
        },
        (
            "Uses the production six-scale supports and real GPT-2-vocabulary "
            "ultralight actor to measure derived work and parameter overhead."
        ),
    )


def benchmark_gradient_governance_and_accounting() -> CSTMExperiment:
    """Run one real integrated context and inspect capped auxiliary contributions."""

    tokenizer = ByteTextTokenizer()
    model = MRCRALanguageModel(_tiny_integrated_config(tokenizer.vocabulary_size))
    configuration = MRCRATrainingConfig(
        output_dir="outputs/.cstm-acceptance-scratch",
        total_tokens=32,
        context_length=32,
        execution_chunk_size=4,
        tbptt_length=8,
        vocabulary_tile_size=32,
        warmup_tokens=8,
        integrated_cognitive_path=True,
        cognitive_stride=2,
        cognitive_tbptt_events=2,
        progress_interval_tokens=32,
        cstm_enabled=True,
        cstm_weight=0.04,
        cstm_warmup_tokens=0,
        cstm_ramp_tokens=1,
        cstm_sampling_duty_cycle=1.0,
        trackio_enabled=False,
        show_dashboard=False,
        spectral_dashboard=False,
        phase_transition_ablation=False,
        data_prefetch=False,
    )
    trainer = MRCRANextTokenTrainer(
        model,
        tokenizer,
        PackedTokenStream(
            SequenceTextSource((
                "abcdefghijklmnopqrstuvwxyz0123456789 causal spectral evidence",
            )),
            tokenizer,
        ),
        configuration,
    )
    batch = trainer.train_stream.next_batch(1, 32)
    before_tokens = trainer.state.tokens_seen
    trainer.optimizer.zero_grad(set_to_none=True)
    with redirect_stdout(StringIO()):
        local = trainer._run_context(batch)
    named = dict(model.named_parameters())
    task = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in named.items()
    }
    auxiliary = {
        name: None if value is None else value.detach().clone()
        for name, value in trainer._cstm_auxiliary_gradients.items()
    }
    merge = trainer._merge_cstm_gradients()

    subsystem_caps = {
        "carrier": configuration.cstm_carrier_gradient_cap,
        "event": configuration.cstm_cognitive_gradient_cap,
        "output_bridge": configuration.cstm_cognitive_gradient_cap,
        "controller": configuration.cstm_cognitive_gradient_cap,
        "workspace_router": configuration.cstm_cognitive_gradient_cap,
        "world_hypothesis": configuration.cstm_cognitive_gradient_cap,
        "memory": configuration.cstm_cognitive_gradient_cap,
        "other_cognition": configuration.cstm_cognitive_gradient_cap,
    }
    maximum_cap_ratio = 0.0
    alignments: list[float] = []
    governed_overlap_subsystems = 0
    for subsystem, cap in subsystem_caps.items():
        rows = [
            (name, named[name])
            for name, value in auxiliary.items()
            if value is not None
            and task.get(name) is not None
            and gradient_subsystem(name) == subsystem
        ]
        if not rows:
            continue
        governed_overlap_subsystems += 1
        task_norm = torch.stack([
            task[name].float().square().sum() for name, _ in rows
        ]).sum().sqrt()
        contribution = [
            parameter.grad.detach() - task[name]
            for name, parameter in rows
        ]
        contribution_norm = torch.stack([
            value.float().square().sum() for value in contribution
        ]).sum().sqrt()
        maximum_cap_ratio = max(
            maximum_cap_ratio,
            float(contribution_norm / (cap * task_norm).clamp_min(1e-12)),
        )
        alignment = sum(
            (
                task[name].float().mul(value.float()).sum()
                for (name, _), value in zip(rows, contribution, strict=True)
            ),
            start=torch.tensor(0.0),
        )
        alignments.append(float(alignment))
    minimum_alignment = min(alignments, default=0.0)
    head_gradient_norm = torch.stack([
        parameter.grad.detach().float().square().sum()
        for name, parameter in named.items()
        if name.startswith("cstm_predictor.") and parameter.grad is not None
    ]).sum().sqrt()
    return CSTMExperiment(
        "gradient_governance_and_accounting",
        {
            "auxiliary_applied": merge["cstm/auxiliary_applied"],
            "maximum_subsystem_cap_ratio": maximum_cap_ratio,
            "minimum_task_auxiliary_alignment": minimum_alignment,
            "governed_overlap_subsystems": float(
                governed_overlap_subsystems
            ),
            "cstm_head_gradient_norm": float(head_gradient_norm),
            "spectral_target_views": local["cstm/spectral_target_views"],
            "physical_token_counter_delta": float(
                trainer.state.tokens_seen - before_tokens
            ),
            "packed_physical_tokens": float(batch.token_count),
        },
        (
            "Runs the real integrated trainer through CSTM target construction "
            "and a separate auxiliary backward, then inspects conflict "
            "projection, subsystem-relative caps, auxiliary-head gradients, and "
            "the unchanged physical-token counter."
        ),
    )


def run_cstm_acceptance() -> CSTMAcceptanceReport:
    experiments = (
        benchmark_target_authority(),
        benchmark_predictor_learning(),
        benchmark_integrated_causality(),
        benchmark_efficiency_and_parameter_contract(),
        benchmark_gradient_governance_and_accounting(),
    )
    metrics = {
        key: value
        for experiment in experiments
        for key, value in experiment.metrics.items()
    }
    criteria = (
        CSTMCriterion(
            "direct_dft_max_abs_error",
            1e-6,
            "at_most",
            metrics["direct_dft_max_abs_error"],
        ),
        CSTMCriterion(
            "permutation_dc_max_abs_change",
            1e-6,
            "at_most",
            metrics["permutation_dc_max_abs_change"],
        ),
        CSTMCriterion(
            "permutation_harmonic_rms_change",
            1e-3,
            "at_least",
            metrics["permutation_harmonic_rms_change"],
        ),
        CSTMCriterion(
            "cross_boundary_valid_rows",
            0,
            "equal",
            metrics["cross_boundary_valid_rows"],
        ),
        CSTMCriterion(
            "predictor_final_to_initial_ratio",
            0.35,
            "at_most",
            metrics["final_to_initial_ratio"],
        ),
        CSTMCriterion(
            "integrated_past_prediction_change",
            1e-9,
            "at_most",
            metrics["past_prediction_max_abs_change"],
        ),
        CSTMCriterion(
            "integrated_future_prediction_change",
            1e-5,
            "at_least",
            metrics["future_prediction_max_abs_change"],
        ),
        CSTMCriterion(
            "past_rows_compared",
            1,
            "at_least",
            metrics["past_rows_compared"],
        ),
        CSTMCriterion(
            "geometric_two_horizon_row_ratio",
            2.0,
            "at_most",
            metrics["target_rows_per_physical_token"],
        ),
        CSTMCriterion(
            "ultralight_cstm_predictor_parameters",
            5_000,
            "at_most",
            metrics["cstm_predictor_parameters"],
        ),
        CSTMCriterion(
            "ultralight_actor_parameter_ceiling",
            metrics["ultralight_parameter_maximum"],
            "at_most",
            metrics["ultralight_actor_parameters"],
        ),
        CSTMCriterion(
            "gradient_auxiliary_applied",
            1,
            "equal",
            metrics["auxiliary_applied"],
        ),
        CSTMCriterion(
            "gradient_maximum_subsystem_cap_ratio",
            1.00001,
            "at_most",
            metrics["maximum_subsystem_cap_ratio"],
        ),
        CSTMCriterion(
            "gradient_minimum_task_auxiliary_alignment",
            -1e-7,
            "at_least",
            metrics["minimum_task_auxiliary_alignment"],
        ),
        CSTMCriterion(
            "gradient_governed_overlap_subsystems",
            1,
            "at_least",
            metrics["governed_overlap_subsystems"],
        ),
        CSTMCriterion(
            "cstm_head_gradient_norm",
            1e-8,
            "at_least",
            metrics["cstm_head_gradient_norm"],
        ),
        CSTMCriterion(
            "spectral_target_views",
            1,
            "at_least",
            metrics["spectral_target_views"],
        ),
        CSTMCriterion(
            "physical_token_counter_delta",
            0,
            "equal",
            metrics["physical_token_counter_delta"],
        ),
    )
    return CSTMAcceptanceReport(
        1,
        all(item.passed for item in criteria),
        criteria,
        experiments,
        (
            "This acceptance proves target mathematics, boundary isolation, "
            "causal integrated prediction behavior, predictor trainability, "
            "geometric work, and parameter bounds on deterministic fixtures. "
            "It does not claim downstream language-quality improvement; that "
            "requires matched corpus-scale ablation."
        ),
    )
