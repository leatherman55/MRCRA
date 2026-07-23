#!/usr/bin/env python3
"""Generate retained empirical evidence for Resonant Adjoint Surprise Learning."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import platform
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from mrrn import MRRN, MRRNConfig
from mrrn.optimization import OptimizerPolicy
from mrrn.surprise import (
    AdjointCriticOutput,
    FunctionalSurpriseCalibrator,
    PerformanceGuard,
    ResonantAdjointSurpriseConfig,
    ResonantAdjointSurpriseLearner,
    TrajectoryBatch,
    functional_surprise_target,
)


ROOT = Path(__file__).resolve().parents[1]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def reference_actor_config() -> MRRNConfig:
    return MRRNConfig(
        input_dim=8, model_dim=32, output_dim=8, layers=2, scales=4, heads=4,
        modes=8, mimo_rank=2, attention_window=16, retrieved_items=4,
        memory_capacity=64, mixer_expansion=2, width_growth_cap=1.5,
        mode_growth_cap=1.5, width_multiple=4,
    )


def tiny_actor_config() -> MRRNConfig:
    return MRRNConfig(
        input_dim=2, model_dim=12, output_dim=2, layers=1, scales=2, heads=1,
        modes=2, mimo_rank=1, attention_window=3, retrieved_items=1,
        memory_capacity=4, mixer_expansion=1, width_growth_cap=1,
        mode_growth_cap=1, width_multiple=1, spectral_modes=2,
        spectral_basis_order=2, spectral_triads_per_mode=0,
    )


def tiny_surprise_config() -> ResonantAdjointSurpriseConfig:
    return ResonantAdjointSurpriseConfig(
        critic_width=6, minimum_critic_width=4, critic_scales=2, critic_heads=1,
        critic_modes=2, critic_mimo_rank=1, spectral_modes=2,
        spectral_basis_order=2, spectral_triads_per_mode=0, action_rank=2,
        latent_modes=2, horizons=(1, 6), quantiles=(0.2, 0.5, 0.8),
        bootstrap_heads=2, maximum_critic_parameter_fraction=2,
        ema_decay=0.9, calibration_decay=0.9,
        surprise_cross_entropy_weight=1, trust_region_weight=0.01,
        spectral_regularization_weight=0, critic_latent_weight=0.05,
        maximum_gradient_norm=2, performance_tolerance=100,
    )


def delayed_batch(size: int, *, device: torch.device) -> TrajectoryBatch:
    cue = torch.arange(size, device=device) % 2
    inputs = torch.zeros(size, 6, 2, device=device)
    inputs[:, 0] = F.one_hot(cue, 2).float()
    actions = torch.randint(0, 2, (size, 6), device=device)
    actions[:, 0] = torch.arange(size, device=device).div(2, rounding_mode="floor") % 2
    rewards = torch.zeros(size, 6, device=device)
    rewards[:, -1] = torch.where(actions[:, 0] == cue, 1.0, -1.0)
    dones = torch.zeros(size, 6, dtype=torch.bool, device=device)
    dones[:, -1] = True
    return TrajectoryBatch(inputs, actions, rewards, dones)


def delayed_policy_score(learner: ResonantAdjointSurpriseLearner, device: torch.device) -> float:
    inputs = torch.zeros(2, 6, 2, device=device)
    inputs[:, 0] = torch.eye(2, device=device)
    with torch.no_grad():
        policy = learner.actor(inputs).prediction[:, 0].softmax(-1)
    return float(policy[torch.arange(2, device=device), torch.arange(2, device=device)].mean().cpu())


def delayed_credit_experiment(device: torch.device, steps: int) -> dict:
    torch.manual_seed(123)
    learner = ResonantAdjointSurpriseLearner(
        MRRN(tiny_actor_config()).to(device), tiny_surprise_config()
    ).to(device)
    actor_optimizer, critic_optimizer = learner.make_optimizers(
        actor_policy=OptimizerPolicy(learning_rate=3e-3, warmup_steps=0, total_steps=max(100, steps + 1)),
        critic_policy=OptimizerPolicy(learning_rate=3e-3, warmup_steps=0, total_steps=max(100, steps + 1)),
    )
    before = delayed_policy_score(learner, device)
    critic_start = None
    critic_end = None
    started = perf_counter()
    for index in range(steps):
        report = learner.train_step(
            delayed_batch(64, device=device), actor_optimizer, critic_optimizer,
            add_to_replay=False,
        )
        if index == 0:
            critic_start = report.critic_loss
        critic_end = report.critic_loss
    synchronize(device)
    elapsed = perf_counter() - started
    after = delayed_policy_score(learner, device)
    return {
        "decision_time": 0, "reward_time": 5, "delay_transitions": 5,
        "steps": steps, "policy_correct_probability_before": before,
        "policy_correct_probability_after": after,
        "policy_probability_gain": after - before,
        "critic_loss_first": critic_start, "critic_loss_last": critic_end,
        "elapsed_seconds": elapsed,
        "passed": after > before + 0.1,
    }


def mock_critic() -> AdjointCriticOutput:
    batch, length, actions, horizons, scales = 1, 3, 2, 2, 2
    mask = torch.ones(batch, length, dtype=torch.bool)
    quantiles = torch.zeros(batch, length, 2, horizons, 3)
    action_values = torch.zeros(batch, length, 2, horizons, actions)
    reward = torch.zeros(batch, length, actions)
    latent = tuple(torch.zeros(batch, length, horizons, 2, 2) for _ in range(scales))
    targets = tuple(torch.ones(batch, length, 2, 2) for _ in range(scales))
    features = torch.zeros(batch, length, 4)
    return AdjointCriticOutput(
        quantiles, action_values, reward, reward, latent, targets,
        torch.zeros(batch, length, actions), features, features,
        torch.zeros(batch, length, horizons, actions),
        torch.ones(batch, length, horizons, 1), mask,
    )


def safety_experiments() -> dict:
    config = replace(
        tiny_surprise_config(), horizons=(1, 6), advantage_weight=0,
        adjoint_credit_weight=0, exploration_weight=0,
        return_surprise_weight=1, surprise_temperature=1,
    )
    critic = mock_critic()
    logits = torch.zeros(1, 3, 2)
    actions = torch.zeros(1, 3, dtype=torch.long)
    phase = torch.zeros(1, 3, 2)
    calibrator = FunctionalSurpriseCalibrator(2, 2)
    negative = functional_surprise_target(
        logits, logits, critic, actions, -torch.ones(1, 3, 2), phase,
        calibrator, config,
    )
    positive = functional_surprise_target(
        logits, logits, critic, actions, torch.ones(1, 3, 2), phase,
        calibrator, config,
    )
    exploratory_config = replace(
        config, return_surprise_weight=0, exploration_weight=1,
    )
    action_values = critic.action_values.clone()
    action_values[..., 0] = 2
    epistemic = torch.ones_like(critic.epistemic_uncertainty) * 4
    learnable = replace(
        critic, action_values=action_values, epistemic_uncertainty=epistemic,
        aleatoric_uncertainty=torch.ones_like(critic.aleatoric_uncertainty) * 0.1,
    )
    calibrator.updates.fill_(2)
    calibrator.previous_scale_error.fill_(2)
    calibrator.scale_error.fill_(1)
    learnable_target = functional_surprise_target(
        logits, logits, learnable, actions, torch.zeros(1, 3, 2),
        torch.ones(1, 3, 2), calibrator, exploratory_config,
    )
    noisy = replace(
        learnable,
        aleatoric_uncertainty=torch.ones_like(critic.aleatoric_uncertainty) * 100,
    )
    noisy_target = functional_surprise_target(
        logits, logits, noisy, actions, torch.zeros(1, 3, 2),
        torch.ones(1, 3, 2), calibrator, exploratory_config,
    )
    guard = PerformanceGuard(0)
    guard.allows(1.0, 2.0)
    rejected = not guard.allows(0.5, 1.0)
    max_normalization_error = float((negative.distribution.sum(-1) - 1).abs().max())
    learnable_bonus = float(learnable_target.exploration_bonus.max())
    noisy_bonus = float(noisy_target.exploration_bonus.max())
    return {
        "negative_outcome_action_probability": float(negative.distribution[..., 0].mean()),
        "positive_outcome_action_probability": float(positive.distribution[..., 0].mean()),
        "maximum_target_normalization_error": max_normalization_error,
        "maximum_absolute_score": float(negative.score.abs().max()),
        "learnable_epistemic_bonus": learnable_bonus,
        "aleatoric_noise_bonus": noisy_bonus,
        "noise_bonus_ratio": noisy_bonus / max(learnable_bonus, 1e-12),
        "performance_guard_rejected_proxy_only_improvement": rejected,
        "passed": bool(
            (negative.distribution[..., 0] < 0.5).all()
            and (positive.distribution[..., 0] > 0.5).all()
            and max_normalization_error < 1e-6
            and noisy_bonus < learnable_bonus / 10
            and rejected
        ),
    }


def architecture_and_firewall_experiments(device: torch.device) -> tuple[dict, ResonantAdjointSurpriseLearner]:
    torch.manual_seed(20260721)
    learner = ResonantAdjointSurpriseLearner(MRRN(reference_actor_config()).to(device)).to(device)
    batch, length = 2, 32
    inputs = torch.randn(batch, length, 8, device=device)
    actions = torch.randint(0, 8, (batch, length), device=device)
    rewards = torch.randn(batch, length, device=device)
    dones = torch.zeros(batch, length, dtype=torch.bool, device=device)
    trajectories = TrajectoryBatch(inputs, actions, rewards, dones)
    bootstrap = torch.ones(batch, length, 4, dtype=torch.bool, device=device)
    losses = learner.compute_losses(trajectories, bootstrap_mask=bootstrap)
    losses.critic.total.backward()
    critic_to_actor_leaks = sum(parameter.grad is not None for parameter in learner.actor.parameters())
    critic_gradients = sum(parameter.grad is not None for parameter in learner.critic.parameters())
    learner.zero_grad(set_to_none=True)
    losses.actor.total.backward()
    actor_to_critic_leaks = sum(parameter.grad is not None for parameter in learner.critic.parameters())
    actor_gradients = sum(parameter.grad is not None for parameter in learner.actor.parameters())
    learner.zero_grad(set_to_none=True)
    with torch.no_grad():
        actor_output = learner.actor(inputs)
        first = learner.critic(
            actor_output.bands, actor_output.prediction, actions=actions,
            rewards=torch.zeros_like(rewards), dones=dones,
        )
        changed_rewards = torch.zeros_like(rewards)
        changed_rewards[:, -1] = 100
        second = learner.critic(
            actor_output.bands, actor_output.prediction, actions=(actions + 1) % 8,
            rewards=changed_rewards, dones=dones,
        )
    q_leak = float((first.value_quantiles - second.value_quantiles).abs().max().cpu())
    adjoint_sensitivity = float(
        (first.adjoint_credit[:, 0] - second.adjoint_credit[:, 0]).abs().max().cpu()
    )
    parameters = asdict(learner.parameter_report())
    report = {
        "parameters": parameters,
        "critic_to_actor_gradient_leaks": critic_to_actor_leaks,
        "actor_to_critic_gradient_leaks": actor_to_critic_leaks,
        "critic_parameters_with_gradient": critic_gradients,
        "actor_parameters_with_gradient": actor_gradients,
        "maximum_forward_value_change_from_outcomes": q_leak,
        "early_adjoint_change_from_terminal_outcome": adjoint_sensitivity,
        "passed": bool(
            parameters["critic_fraction"] <= 0.20
            and critic_to_actor_leaks == 0 and actor_to_critic_leaks == 0
            and critic_gradients > 0 and actor_gradients > 0
            and q_leak == 0 and adjoint_sensitivity > 0
        ),
    }
    return report, learner


def timing_experiment(device: torch.device, repeats: int) -> dict:
    torch.manual_seed(991)
    config = reference_actor_config()
    actor = MRRN(config).to(device)
    learner = ResonantAdjointSurpriseLearner(MRRN(config).to(device)).to(device)
    inputs = torch.randn(2, 64, 8, device=device)
    batch = TrajectoryBatch(
        inputs, torch.randint(0, 8, (2, 64), device=device),
        torch.randn(2, 64, device=device),
        torch.zeros(2, 64, dtype=torch.bool, device=device),
    )

    def baseline():
        actor.zero_grad(set_to_none=True)
        actor(inputs).prediction.square().mean().backward()

    def rasl():
        learner.zero_grad(set_to_none=True)
        losses = learner.compute_losses(batch, update_calibration=False)
        losses.critic.total.backward()
        losses.actor.total.backward()

    baseline()
    rasl()
    synchronize(device)

    def measure(operation):
        samples = []
        for _ in range(repeats):
            synchronize(device)
            start = perf_counter()
            operation()
            synchronize(device)
            samples.append((perf_counter() - start) * 1000)
        return {
            "median_ms": sorted(samples)[len(samples) // 2],
            "minimum_ms": min(samples), "samples_ms": samples,
        }

    baseline_timing = measure(baseline)
    rasl_timing = measure(rasl)
    return {
        "device": str(device), "batch": 2, "length": 64, "repeats": repeats,
        "actor_forward_backward": baseline_timing,
        "complete_rasl_two_loss_backward": rasl_timing,
        "total_training_time_ratio": rasl_timing["median_ms"] / baseline_timing["median_ms"],
        "boundary": (
            "This compares one supervised actor forward/backward with complete RASL actor and "
            "critic forwards, EMA actor/critic readout targets, and both backwards. Detached "
            "multiscale representations are reused, so no target backbone is rerun. It is a "
            "systems-cost measurement, not a quality-normalized speed claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--delayed-steps", type=int, default=40)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs" / "surprise_verification_report.json",
    )
    args = parser.parse_args()
    if min(args.delayed_steps, args.timing_repeats) <= 0:
        raise SystemExit("verification repeat and training counts must be positive")
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cpu":
        torch.set_num_threads(1)
    architecture, _ = architecture_and_firewall_experiments(device)
    safety = safety_experiments()
    delayed = delayed_credit_experiment(device, args.delayed_steps)
    timing = timing_experiment(device, args.timing_repeats)
    gates = {
        "architecture_and_firewalls": architecture["passed"],
        "bounded_functional_safety": safety["passed"],
        "delayed_credit_learning": delayed["passed"],
    }
    payload = {
        "format_version": 1, "seed": args.seed,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "platform": platform.platform(), "processor": platform.processor(),
            "device": str(device), "mps_available": torch.backends.mps.is_available(),
        },
        "architecture_and_firewalls": architecture,
        "functional_safety": safety,
        "delayed_credit": delayed,
        "timing": timing,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "claim_boundary": (
            "These tests establish construction, gradient isolation, bounded targets, causal "
            "value separation, reverse consequence sensitivity, safety-gate behavior, and one "
            "seeded delayed-credit learning result. They do not prove universal convergence or "
            "task-independent superiority."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "all_gates_passed": payload["all_gates_passed"]}))
    if not payload["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
