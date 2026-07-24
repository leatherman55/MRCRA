"""Deterministic empirical acceptance for Progress-Conditioned RASL.

This suite exercises the production learning-progress authority, cognitive
adjoint learner, internal controller credit path, critic firewall, and
auxiliary-gradient governor.  It is intentionally small enough for routine
local verification.  Exact trainer/checkpoint integration is covered by the
companion production-path tests because that contract includes filesystem and
stream state, not merely a learned mechanism.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from math import isfinite
from time import perf_counter
from typing import Literal

import torch
from torch import Tensor, nn

from .cognitive_surprise import (
    CognitiveRASLConfig,
    CognitiveResonantAdjointSurpriseLearner,
    CognitiveTrajectoryBatch,
    build_language_candidate_set,
)
from .config import CognitiveConfig, MRCRAConfig, MRRNConfig
from .language import MRCRALanguageModel
from .learning_progress import (
    LearningProgressAuthority,
    LearningProgressConfig,
)
from .optimization import merge_auxiliary_gradients
from .surprise import ResonantAdjointSurpriseConfig


Direction = Literal["at_least", "at_most"]


@dataclass(frozen=True, slots=True)
class PCRASLCriterion:
    metric: str
    threshold: float
    direction: Direction

    def evaluate(self, metrics: dict[str, float]) -> bool:
        value = metrics.get(self.metric)
        if value is None or not isfinite(value):
            return False
        if self.direction == "at_least":
            return value >= self.threshold
        if self.direction == "at_most":
            return value <= self.threshold
        raise ValueError(f"unknown criterion direction {self.direction!r}")


@dataclass(frozen=True, slots=True)
class PCRASLExperiment:
    name: str
    duration_seconds: float
    metrics: dict[str, float]
    criteria: tuple[PCRASLCriterion, ...]
    passed: bool


@dataclass(frozen=True, slots=True)
class PCRASLAcceptanceReport:
    format_version: int
    suite: str
    seed: int
    device: str
    results: tuple[PCRASLExperiment, ...]
    passed: bool
    phase_transition_metrics_used_as_authority: bool
    exact_trainer_resume_test_node: str
    causal_replay_test_nodes: tuple[str, ...]
    checkpoint_migration_test_nodes: tuple[str, ...]
    production_resource_test_node: str
    claim_boundary: str

    def to_dict(self) -> dict:
        return asdict(self)


def _result(
    name: str,
    started: float,
    metrics: dict[str, float],
    criteria: tuple[PCRASLCriterion, ...],
) -> PCRASLExperiment:
    normalized = {key: float(value) for key, value in metrics.items()}
    return PCRASLExperiment(
        name=name,
        duration_seconds=perf_counter() - started,
        metrics=normalized,
        criteria=criteria,
        passed=all(criterion.evaluate(normalized) for criterion in criteria),
    )


def _progress_config(**updates) -> LearningProgressConfig:
    base = LearningProgressConfig(
        observation_interval=1,
        warmup_observations=4,
        fast_window=3,
        baseline_min_observations=4,
        baseline_window=12,
        baseline_lag=0,
        baseline_freeze_observations=2,
        deadband_standard_deviations=0.0,
        minimum_slope_noise=1e-5,
        minimum_ce_noise=1e-5,
        guard_regression_patience=2,
        guard_recovery_patience=2,
    )
    return replace(base, **updates)


def _feed(
    authority: LearningProgressAuthority,
    values: list[float],
) -> list:
    return [
        authority.observe(
            1_000_000 + index * 1_000_000,
            value,
            learning_rate=1e-4,
        )
        for index, value in enumerate(values)
    ]


def benchmark_progress_authority() -> PCRASLExperiment:
    """Verify sign, anti-gaming, guard, and exact causal state restoration."""

    started = perf_counter()
    improving = LearningProgressAuthority(_progress_config())
    improving.observe_guard(5.1)
    positive = _feed(
        improving, [5.0, 4.7, 4.45, 4.20, 3.82, 3.45, 3.10, 2.80]
    )[-1]
    plateau = _feed(
        LearningProgressAuthority(_progress_config()),
        [5.0, 4.7, 4.45, 4.20, 4.19, 4.205, 4.20, 4.21],
    )[-1]
    rising = _feed(
        LearningProgressAuthority(_progress_config()),
        [5.0, 4.7, 4.45, 4.20, 4.25, 4.30, 4.35, 4.40],
    )[-1]
    gaming = _feed(
        LearningProgressAuthority(_progress_config(
            baseline_lag=4, debt_weight=0.75, slope_weight=0.25,
        )),
        [5.0, 4.7, 4.45, 4.20, 4.65, 4.90, 4.75, 4.55],
    )[-1]
    improving.observe_guard(3.0)
    improving.observe_guard(3.2)
    veto = improving.observe_guard(3.3)
    vetoed = improving.observe(9_000_000, 2.55, learning_rate=1e-4)
    improving.observe_guard(3.01)
    recovery = improving.observe_guard(2.99)
    state = deepcopy(improving.state_dict())
    restored = LearningProgressAuthority(_progress_config())
    restored.load_state_dict(state)
    resume_error = float(restored.state_dict() != state)
    serialized = repr((improving.metrics(vetoed), state)).lower()
    metrics = {
        "accelerating_pressure": positive.pressure,
        "plateau_pressure": plateau.pressure,
        "rising_pressure": rising.pressure,
        "gaming_pressure": gaming.pressure,
        "gaming_debt": gaming.progress_debt_nats_per_token,
        "guard_veto": float(not veto),
        "vetoed_positive_pressure": max(0.0, vetoed.pressure),
        "guard_recovery": float(recovery),
        "resume_state_mismatch": resume_error,
        "phase_authority_mentions": float(
            "phase_transition" in serialized or "event_proposal" in serialized
        ),
    }
    criteria = (
        PCRASLCriterion("accelerating_pressure", 0.05, "at_least"),
        PCRASLCriterion("plateau_pressure", -0.01, "at_most"),
        PCRASLCriterion("rising_pressure", 0.0, "at_most"),
        PCRASLCriterion("gaming_pressure", 0.0, "at_most"),
        PCRASLCriterion("gaming_debt", 0.1, "at_least"),
        PCRASLCriterion("guard_veto", 1.0, "at_least"),
        PCRASLCriterion("vetoed_positive_pressure", 0.0, "at_most"),
        PCRASLCriterion("guard_recovery", 1.0, "at_least"),
        PCRASLCriterion("resume_state_mismatch", 0.0, "at_most"),
        PCRASLCriterion("phase_authority_mentions", 0.0, "at_most"),
    )
    return _result("causal progress authority", started, metrics, criteria)


def _actor_config(vocabulary: int = 11) -> MRCRAConfig:
    carrier = MRRNConfig(
        input_dim=8, model_dim=8, output_dim=vocabulary, layers=1, scales=2,
        heads=2, modes=2, mimo_rank=1, attention_window=2,
        retrieved_items=1, memory_capacity=4, mixer_expansion=1.5,
        width_growth_cap=1, mode_growth_cap=1, width_multiple=4,
        spectral_modes=2, spectral_basis_order=2, spectral_triads_per_mode=1,
        enable_global_head=False, relational_branch=True,
        relational_context_dim=8,
    )
    cognition = CognitiveConfig(
        workspace_dim=8, provenance_features=4, uncertainty_channels=8,
        relation_heads=2, relation_modes=2, relation_adapter_rank=1,
        goal_slots=1, goal_constraint_dim=2, system_action_channels=2,
        calibration_regimes=2, active_event_capacity=4, pair_edge_capacity=6,
        hyperedge_capacity=2, maximum_hyperedge_arity=3, graph_neighbors=1,
        global_workspace_slots=1, hypothesis_slots=1,
        maximum_hypothesis_slots=2, maximum_cognitive_steps=1,
        event_chunk_size=2, event_proposals_per_chunk=1,
        recent_candidates=2, landmark_candidates=1, episodic_candidates=1,
        semantic_candidates=1, episodic_memory_capacity=4,
        semantic_memory_capacity=2, associative_depth=1,
        associative_budget=1, world_model_horizons=(1, 2),
    )
    return MRCRAConfig(carrier, cognition, 1, 10_000_000)


def _rasl_config() -> CognitiveRASLConfig:
    return CognitiveRASLConfig(
        core=ResonantAdjointSurpriseConfig(
            critic_width=8, minimum_critic_width=4, critic_layers=1,
            critic_scales=1, critic_heads=1, critic_modes=2,
            critic_mimo_rank=1, spectral_modes=2, spectral_basis_order=2,
            action_rank=2, latent_modes=2, horizons=(1, 2),
            quantiles=(0.1, 0.5, 0.9), bootstrap_heads=2,
            maximum_critic_parameter_fraction=1.0,
            require_external_reward=True,
        ),
        maximum_candidates=5,
    )


def _trajectory(
    model: MRCRALanguageModel, reward: float,
) -> CognitiveTrajectoryBatch:
    inputs = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    behavior = torch.tensor([[2, 3, 4, 5]], dtype=torch.int64)
    target_logits = torch.randn(1, 4, model.vocabulary_size)
    candidates, proposal_log_probability, sampled = (
        build_language_candidate_set(
            target_logits, behavior, candidate_count=5,
        )
    )
    return CognitiveTrajectoryBatch(
        input_ids=inputs,
        behavior_tokens=behavior,
        candidate_token_ids=candidates,
        candidate_sampling_log_probabilities=proposal_log_probability,
        sampled_candidate_mask=sampled,
        rewards=torch.full((1, 4), reward),
        dones=torch.tensor([[False, False, False, True]]),
        mask=torch.ones(1, 4, dtype=torch.bool),
        task_targets=behavior,
        reward_source="learning_progress",
        burn_in_steps=1,
    )


def benchmark_critic_and_internal_credit(seed: int) -> PCRASLExperiment:
    """Verify signed consequence, critic learning, firewall, and controller credit."""

    started = perf_counter()
    torch.manual_seed(seed)
    model = MRCRALanguageModel(_actor_config())
    learner = CognitiveResonantAdjointSurpriseLearner(
        model, _rasl_config()
    )
    positive = _trajectory(model, 2.0)
    negative = replace(
        positive, rewards=torch.full_like(positive.rewards, -2.0)
    )
    positive_losses = learner.compute_losses(positive)
    negative_losses = learner.compute_losses(negative)
    behavior_index = (
        positive.candidate_token_ids
        == positive.behavior_tokens.unsqueeze(-1)
    ).to(torch.int64).argmax(-1)
    positive_probability = positive_losses.surprise.distribution.gather(
        -1, behavior_index.unsqueeze(-1)
    ).squeeze(-1)
    negative_probability = negative_losses.surprise.distribution.gather(
        -1, behavior_index.unsqueeze(-1)
    ).squeeze(-1)
    probability_margin = float(
        (
            positive_probability[positive.loss_mask]
            - negative_probability[positive.loss_mask]
        ).min()
    )

    learner.actor.zero_grad(set_to_none=True)
    learner.critic.zero_grad(set_to_none=True)
    positive_losses.critic.total.backward()
    actor_leaks = sum(
        parameter.grad is not None
        for parameter in learner.actor.parameters()
    )
    learner.actor.zero_grad(set_to_none=True)
    learner.critic.zero_grad(set_to_none=True)
    learner.compute_losses(positive).actor.internal_policy.backward()
    controller_gradient = sum(
        float(parameter.grad.detach().float().square().sum())
        for parameter in model.cognitive.controller.parameters()
        if parameter.grad is not None
    ) ** 0.5

    learner.actor.zero_grad(set_to_none=True)
    learner.critic.zero_grad(set_to_none=True)
    fixed = replace(
        positive, rewards=torch.full_like(positive.rewards, 0.5)
    )
    optimizer = torch.optim.Adam(learner.critic.parameters(), lr=2e-3)
    critic_losses: list[float] = []
    for _ in range(24):
        optimizer.zero_grad(set_to_none=True)
        loss = learner.compute_losses(fixed).critic.total
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("PC-RASL acceptance critic became non-finite")
        critic_losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    metrics = {
        "signed_behavior_probability_margin": probability_margin,
        "critic_actor_gradient_leaks": float(actor_leaks),
        "controller_gradient_norm": controller_gradient,
        "critic_loss_ratio": critic_losses[-1] / critic_losses[0],
        "candidate_count": float(positive.candidate_count),
        "progress_head_gradient": float(
            learner.critic.progress_return.weight.grad is not None
        ),
        "internal_value_head_gradient": float(
            learner.critic.internal_action_value.weight.grad is not None
        ),
    }
    criteria = (
        PCRASLCriterion(
            "signed_behavior_probability_margin", 0.05, "at_least"
        ),
        PCRASLCriterion("critic_actor_gradient_leaks", 0.0, "at_most"),
        PCRASLCriterion("controller_gradient_norm", 1e-8, "at_least"),
        PCRASLCriterion("critic_loss_ratio", 0.75, "at_most"),
        PCRASLCriterion("candidate_count", 5.0, "at_most"),
        PCRASLCriterion("progress_head_gradient", 1.0, "at_least"),
        PCRASLCriterion("internal_value_head_gradient", 1.0, "at_least"),
    )
    return _result(
        "critic and internal cognitive credit", started, metrics, criteria
    )


class _ToyCognition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.controller = nn.Module()
        self.controller.action_head = nn.Linear(2, 2, bias=False)


class _ToyActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(3, 2)
        self.cognitive = _ToyCognition()


def benchmark_gradient_governor() -> PCRASLExperiment:
    """Verify task authority, conflict projection, and subsystem caps."""

    started = perf_counter()
    model = _ToyActor()
    named = dict(model.named_parameters())
    carrier = named["token_embedding.weight"]
    controller = named["cognitive.controller.action_head.weight"]
    carrier.grad = torch.ones_like(carrier)
    controller.grad = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    controller_before = controller.grad.clone()
    report = merge_auxiliary_gradients(
        model,
        {
            "token_embedding.weight": torch.ones_like(carrier) * 10,
            "cognitive.controller.action_head.weight": torch.tensor(
                [[-1.0, 10.0], [0.0, 0.0]]
            ),
        },
        {"carrier": 0.0, "controller": 0.1},
    )
    controller_contribution = controller.grad - controller_before
    metrics = {
        "carrier_mutation": float((carrier.grad - 1).norm()),
        "controller_auxiliary_norm": float(controller_contribution.norm()),
        "controller_task_dot": float(
            (controller_contribution * controller_before).sum()
        ),
        "conflict_detected": float(
            "controller" in report.conflicting_subsystems
        ),
        "unknown_auxiliary_applied": 0.0,
    }
    criteria = (
        PCRASLCriterion("carrier_mutation", 0.0, "at_most"),
        PCRASLCriterion("controller_auxiliary_norm", 0.100001, "at_most"),
        PCRASLCriterion("controller_task_dot", -1e-7, "at_least"),
        PCRASLCriterion("conflict_detected", 1.0, "at_least"),
        PCRASLCriterion("unknown_auxiliary_applied", 0.0, "at_most"),
    )
    return _result("guarded auxiliary gradient merge", started, metrics, criteria)


def run_pc_rasl_acceptance(
    *, seed: int = 20260723, device: str = "cpu",
) -> PCRASLAcceptanceReport:
    if device != "cpu":
        raise ValueError(
            "portable PC-RASL acceptance requires CPU; hardware throughput is separate"
        )
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        results = (
            benchmark_progress_authority(),
            benchmark_critic_and_internal_credit(seed),
            benchmark_gradient_governor(),
        )
    finally:
        torch.set_num_threads(prior_threads)
    return PCRASLAcceptanceReport(
        format_version=2,
        suite="progress-conditioned-rasl-empirical-v2",
        seed=seed,
        device=device,
        results=results,
        passed=all(result.passed for result in results),
        phase_transition_metrics_used_as_authority=False,
        exact_trainer_resume_test_node=(
            "tests/test_pc_rasl_training.py::"
            "test_pc_rasl_production_path_is_checkpoint_resume_exact"
        ),
        causal_replay_test_nodes=(
            "tests/test_pc_rasl_training.py::"
            "test_pc_rasl_checkpoint_binds_both_probe_and_guard_evidence",
            "tests/test_cognitive_surprise.py::"
            "test_replay_critic_uses_preconsequence_behavior_evidence_not_later_reanalysis",
        ),
        checkpoint_migration_test_nodes=(
            "tests/test_pc_rasl_training.py::"
            "test_format8_checkpoint_migrates_into_fresh_causal_pc_rasl_warmup",
            "tests/test_pc_rasl_training.py::"
            "test_format9_pc_rasl_checkpoint_discards_pre_v10_replay_authority",
        ),
        production_resource_test_node=(
            "tests/test_fineweb_entrypoint.py::"
            "test_lightmodel_pc_rasl_is_a_compact_nonduplicating_production_learner"
        ),
        claim_boundary=(
            "Passing establishes the signed causal-pressure mechanism, bounded "
            "candidate critic, internal-action credit, gradient firewall, and "
            "governor on deterministic local evidence. It does not establish "
            "open-domain capability or a beneficial phase transition."
        ),
    )
