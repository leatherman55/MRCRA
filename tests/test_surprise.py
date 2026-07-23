from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from mrrn import MRRN, MRRNConfig
from mrrn.lifting import ScaleTensor
from mrrn.optimization import OptimizerPolicy
from mrrn.surprise import (
    AdjointCriticOutput,
    FunctionalSurpriseCalibrator,
    PerformanceGuard,
    PrioritizedTrajectoryReplay,
    ResonantAdjointCritic,
    ResonantAdjointSurpriseConfig,
    ResonantAdjointSurpriseLearner,
    TrajectoryBatch,
    _AdjointResonanceLayer,
    _FrozenSpectralProjection,
    _select_scale_indices,
    _support_expand,
    actor_losses,
    critic_losses,
    differentiable_quantile_calibration_loss,
    functional_surprise_target,
    load_rasl_checkpoint,
    multihorizon_returns,
    phase_aware_latent_error,
    quantile_huber_loss,
    save_rasl_checkpoint,
)


def actor_config(**overrides):
    values = dict(
        input_dim=3, model_dim=16, output_dim=3, layers=1, scales=3, heads=1,
        modes=2, mimo_rank=1, attention_window=4, retrieved_items=1,
        memory_capacity=8, mixer_expansion=1, width_growth_cap=1,
        mode_growth_cap=1, width_multiple=1, spectral_modes=2,
        spectral_basis_order=2, spectral_triads_per_mode=0,
    )
    values.update(overrides)
    return MRRNConfig(**values)


def surprise_config(**overrides):
    values = dict(
        critic_width=8, minimum_critic_width=4, critic_scales=3, critic_heads=1,
        critic_modes=2, critic_mimo_rank=1, spectral_modes=2,
        spectral_basis_order=2, spectral_triads_per_mode=0, action_rank=2,
        latent_modes=2, horizons=(1, 3), quantiles=(0.25, 0.5, 0.75),
        bootstrap_heads=2, maximum_critic_parameter_fraction=2.0,
        replay_capacity=32,
    )
    values.update(overrides)
    return ResonantAdjointSurpriseConfig(**values)


def trajectory(batch=2, length=8, actions=3):
    mask = torch.ones(batch, length, dtype=torch.bool)
    if batch > 1:
        mask[1, -2:] = False
    return TrajectoryBatch(
        torch.randn(batch, length, 3), torch.randint(0, actions, (batch, length)),
        torch.randn(batch, length), torch.zeros(batch, length, dtype=torch.bool), mask,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"critic_width": 0}, {"minimum_critic_width": 9, "critic_width": 8},
        {"spectral_triads_per_mode": -1}, {"horizons": ()},
        {"horizons": (4, 1)}, {"quantiles": (0.0, 0.5)},
        {"quantiles": (0.7, 0.3)}, {"discount": 0}, {"ema_decay": 1},
        {"calibration_decay": -1}, {"task_weight": -1},
        {"surprise_temperature": 0}, {"replay_priority_fraction": 2},
    ],
)
def test_surprise_configuration_fails_closed(kwargs):
    with pytest.raises(ValueError):
        surprise_config(**kwargs)


def test_trajectory_contract_and_detached_cpu_boundary():
    batch = trajectory().validated(input_dim=3, action_dim=3)
    assert batch.mask.dtype == torch.bool
    copied = batch.detached_cpu()
    assert copied.inputs.device.type == "cpu" and copied.inputs.data_ptr() != batch.inputs.data_ptr()
    implicit = replace(batch, mask=None).validated(input_dim=3, action_dim=3)
    assert implicit.mask.all()
    failures = [
        replace(batch, inputs=torch.ones(2, 8)),
        replace(batch, actions=batch.actions.float()),
        replace(batch, actions=torch.full_like(batch.actions, 9)),
        replace(batch, rewards=batch.rewards.long()),
        replace(batch, dones=batch.dones.float()),
        replace(batch, mask=batch.mask.float()),
        replace(batch, task_targets=torch.ones(2, 2)),
        replace(batch, behavior_logits=torch.ones(2, 8, 2)),
        replace(batch, behavior_logits=torch.full((2, 8, 3), float("nan"))),
        replace(batch, importance_weights=torch.ones(2, dtype=torch.long)),
        replace(batch, importance_weights=torch.tensor([1.0, 0.0])),
        replace(batch, reward_source="unknown"),
        replace(batch, inputs=batch.inputs.clone().fill_(float("nan"))),
    ]
    reactivated = batch.mask.clone()
    reactivated[0, 2] = False
    reactivated[0, 3] = True
    failures.append(replace(batch, mask=reactivated))
    for invalid in failures:
        with pytest.raises(ValueError):
            invalid.validated(input_dim=3, action_dim=3)


def test_scale_selection_and_support_expansion_cover_endpoints_and_empty_inputs():
    assert _select_scale_indices(5, 3) == (0, 2, 4)
    assert _select_scale_indices(5, 1) == (4,)
    assert _select_scale_indices(2, 8) == (0, 1)
    with pytest.raises(ValueError):
        _select_scale_indices(0, 1)
    values = torch.tensor([[[1.0], [2.0], [3.0]]])
    assert _support_expand(values, 7, 2).flatten().tolist() == [1, 1, 2, 2, 3, 3, 3]
    assert _support_expand(values, 0, 2).shape[1] == 0
    assert _support_expand(values[:, :0], 3, 2).eq(0).all()
    with pytest.raises(ValueError):
        _support_expand(values, 2, 0)


def test_critic_reuses_detached_actor_bands_has_distributional_shapes_and_no_forward_leakage():
    torch.manual_seed(10)
    actor = MRRN(actor_config()).double()
    critic = ResonantAdjointCritic(actor.config, surprise_config(), width=8).double()
    x = torch.randn(2, 9, 3, dtype=torch.float64, requires_grad=True)
    actor_output = actor(x)
    actions = torch.randint(0, 3, (2, 9))
    rewards = torch.randn(2, 9, dtype=torch.float64)
    dones = torch.zeros(2, 9, dtype=torch.bool)
    baseline = critic(
        actor_output.bands, actor_output.prediction, actions=actions, rewards=rewards,
        dones=dones,
    )
    changed = critic(
        actor_output.bands, actor_output.prediction,
        actions=(actions + 1) % 3, rewards=rewards.flip(1) + 50, dones=dones,
    )
    assert baseline.value_quantiles.shape == (2, 9, 2, 2, 3)
    assert baseline.action_values.shape == (2, 9, 2, 2, 3)
    assert baseline.epistemic_uncertainty.shape == (2, 9, 2, 3)
    assert baseline.aleatoric_uncertainty.shape == (2, 9, 2, 1)
    assert baseline.quantiles_for(actions).shape == (2, 9, 2, 2, 3)
    assert (baseline.value_quantiles.diff(dim=-1) >= 0).all()
    torch.testing.assert_close(baseline.forward_features, changed.forward_features)
    torch.testing.assert_close(baseline.action_values, changed.action_values)
    assert not torch.allclose(baseline.adjoint_features, changed.adjoint_features)
    baseline.quantiles_for(actions).sum().backward()
    assert x.grad is None and all(parameter.grad is None for parameter in actor.parameters())
    assert any(parameter.grad is not None for parameter in critic.parameters())


def test_reverse_adjoint_propagates_a_terminal_consequence_to_earlier_time_without_value_leakage():
    torch.manual_seed(21)
    actor = MRRN(actor_config()).eval()
    critic = ResonantAdjointCritic(actor.config, surprise_config(), width=8).eval()
    x = torch.randn(1, 12, 3)
    output = actor(x)
    actions = torch.zeros(1, 12, dtype=torch.long)
    dones = torch.zeros(1, 12, dtype=torch.bool)
    zero = torch.zeros(1, 12)
    terminal = zero.clone()
    terminal[:, -1] = 100
    without = critic(output.bands, output.prediction, actions=actions, rewards=zero, dones=dones)
    with_terminal = critic(output.bands, output.prediction, actions=actions, rewards=terminal, dones=dones)
    torch.testing.assert_close(without.value_quantiles, with_terminal.value_quantiles)
    difference = (without.adjoint_credit[:, 0] - with_terminal.adjoint_credit[:, 0]).abs().max()
    assert difference > 1e-9
    forward_only = critic(output.bands, output.prediction, mask=torch.ones(1, 12, dtype=torch.bool), include_adjoint=False)
    assert forward_only.adjoint_features.eq(0).all()
    with pytest.raises(ValueError):
        critic(output.bands, output.prediction, include_adjoint=True)
    with pytest.raises(ValueError):
        critic(output.bands, output.prediction, actions=actions, include_adjoint=False)


def test_multihorizon_returns_obey_terminal_mask_discount_and_bootstrap():
    rewards = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    dones = torch.tensor([[False, True, False, False]])
    mask = torch.ones_like(dones)
    bootstrap = torch.full((1, 4, 2), 10.0)
    result, valid = multihorizon_returns(
        rewards, dones, mask, (1, 3), discount=0.5, bootstrap=bootstrap
    )
    assert result[0, 0].tolist() == pytest.approx([6.0, 2.0])
    assert result[0, 1].tolist() == pytest.approx([2.0, 2.0])
    assert valid.all()
    unbootstrapped, _ = multihorizon_returns(rewards, dones, mask, (3,), discount=0.5)
    assert unbootstrapped[0, 0, 0] == 2
    with pytest.raises(ValueError):
        multihorizon_returns(rewards, dones.float(), mask, (1,), discount=1)
    with pytest.raises(ValueError):
        multihorizon_returns(rewards, dones, mask, (), discount=1)
    with pytest.raises(ValueError):
        multihorizon_returns(rewards, dones, mask, (1,), discount=1, bootstrap=torch.ones(1))


def test_phase_aware_error_respects_amplitude_and_circular_phase():
    target = torch.tensor([[[1.0, 0.0]]])
    exact = phase_aware_latent_error(target, target)
    opposite = phase_aware_latent_error(-target, target)
    amplitude = phase_aware_latent_error(2 * target, target)
    assert exact.item() < 1e-10 and opposite.item() > amplitude.item() > exact.item()
    zero = torch.zeros_like(target)
    assert phase_aware_latent_error(zero, zero).item() == 0
    with pytest.raises(ValueError):
        phase_aware_latent_error(target, target[..., :1])
    with pytest.raises(ValueError):
        phase_aware_latent_error(target, target, phase_weight=-1)


def test_quantile_regression_and_calibration_prefer_correct_distributions():
    target = torch.ones(2, 4, 2)
    mask = torch.ones_like(target, dtype=torch.bool)
    correct = target[:, :, None, :, None].expand(2, 4, 2, 2, 3).clone()
    wrong = correct + 5
    bootstrap = torch.ones(2, 4, 2, dtype=torch.bool)
    assert quantile_huber_loss(correct, target, (0.2, 0.5, 0.8), mask, bootstrap_mask=bootstrap) == 0
    assert quantile_huber_loss(wrong, target, (0.2, 0.5, 0.8), mask) > 0
    weighted_prediction = correct.clone()
    weighted_prediction[1] += 10
    unweighted = quantile_huber_loss(weighted_prediction, target, (0.2, 0.5, 0.8), mask)
    weighted = quantile_huber_loss(
        weighted_prediction, target, (0.2, 0.5, 0.8), mask,
        sample_weights=torch.tensor([1.0, 0.01]),
    )
    assert weighted < unweighted
    calibration = differentiable_quantile_calibration_loss(
        correct, target, (0.2, 0.5, 0.8), mask
    )
    assert torch.isfinite(calibration)
    with pytest.raises(ValueError):
        quantile_huber_loss(correct[..., 0], target, (0.5,), mask)
    with pytest.raises(ValueError):
        quantile_huber_loss(correct, target, (0.5,), mask)
    with pytest.raises(ValueError):
        quantile_huber_loss(correct, target, (0.2, 0.5, 0.8), mask, bootstrap_mask=torch.ones(1))
    with pytest.raises(ValueError):
        differentiable_quantile_calibration_loss(correct, target, (0.2, 0.5, 0.8), mask, smoothing=0)


def test_calibrator_standardization_reliability_weights_and_learning_progress():
    calibrator = FunctionalSurpriseCalibrator(2, 3, decay=0)
    mask = torch.ones(2, 4, dtype=torch.bool)
    residual = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    first = torch.full((2, 4, 3), 2.0)
    calibrator.update(residual, first, mask)
    assert int(calibrator.updates) == 1 and calibrator.learning_progress().eq(0).all()
    second = torch.tensor([1.0, 1.5, 2.0]).expand(2, 4, 3)
    calibrator.update(residual, second, mask)
    assert calibrator.learning_progress()[0] > calibrator.learning_progress()[1] > calibrator.learning_progress()[2] - 1e-7
    assert torch.isfinite(calibrator.standardize_returns(residual)).all()
    torch.testing.assert_close(calibrator.scale_weights().sum(), torch.tensor(1.0))
    before = calibrator.updates.clone()
    calibrator.update(residual, second, torch.zeros_like(mask))
    assert calibrator.updates == before
    with pytest.raises(ValueError):
        FunctionalSurpriseCalibrator(0, 1)
    with pytest.raises(ValueError):
        calibrator.update(residual[..., :1], second, mask)
    with pytest.raises(ValueError):
        calibrator.update(residual, second[..., :2], mask)
    with pytest.raises(ValueError):
        calibrator.update(residual, second, mask.float())


def mock_critic(batch=1, length=3, actions=2, horizons=2, scales=2):
    mask = torch.ones(batch, length, dtype=torch.bool)
    quantiles = torch.zeros(batch, length, 2, horizons, 3)
    action_values = torch.zeros(batch, length, 2, horizons, actions)
    reward = torch.zeros(batch, length, actions)
    done = torch.zeros_like(reward)
    latent = tuple(torch.zeros(batch, length, horizons, 2, 2) for _ in range(scales))
    targets = tuple(torch.ones(batch, length, 2, 2) for _ in range(scales))
    credit = torch.zeros(batch, length, actions)
    features = torch.zeros(batch, length, 4)
    epistemic = torch.zeros(batch, length, horizons, actions)
    aleatoric = torch.ones(batch, length, horizons, 1)
    return AdjointCriticOutput(
        quantiles, action_values, reward, done, latent, targets, credit,
        features, features, epistemic, aleatoric, mask,
    )


def test_functional_surprise_is_bounded_normalized_and_negative_outcome_reduces_action_probability():
    config = surprise_config(
        horizons=(1, 3), advantage_weight=0, adjoint_credit_weight=0,
        exploration_weight=0, return_surprise_weight=1, surprise_temperature=1,
    )
    critic = mock_critic()
    logits = torch.zeros(1, 3, 2, requires_grad=True)
    actions = torch.zeros(1, 3, dtype=torch.long)
    returns = -torch.ones(1, 3, 2)
    phase = torch.zeros(1, 3, 2)
    calibrator = FunctionalSurpriseCalibrator(2, 2)
    target = functional_surprise_target(
        logits, logits.detach(), critic, actions, returns, phase, calibrator, config
    )
    assert (target.distribution[..., 0] < 0.5).all()
    torch.testing.assert_close(target.distribution.sum(-1), torch.ones(1, 3))
    assert target.score.abs().max() <= config.maximum_surprise
    assert not target.distribution.requires_grad
    positive = functional_surprise_target(
        logits, logits.detach(), critic, actions, -returns, phase, calibrator, config,
        update_calibration=True,
    )
    assert (positive.distribution[..., 0] > 0.5).all() and int(calibrator.updates) == 1


def test_exploration_requires_epistemic_uncertainty_learning_progress_and_controllability():
    config = surprise_config(horizons=(1, 3), return_surprise_weight=0, advantage_weight=0, adjoint_credit_weight=0)
    critic = mock_critic()
    critic = replace(
        critic,
        action_values=critic.action_values.clone(),
        epistemic_uncertainty=torch.ones_like(critic.epistemic_uncertainty) * 4,
        aleatoric_uncertainty=torch.ones_like(critic.aleatoric_uncertainty) * 0.1,
    )
    critic.action_values[..., 0] = 2
    calibrator = FunctionalSurpriseCalibrator(2, 2)
    calibrator.updates.fill_(2)
    calibrator.previous_scale_error.fill_(2)
    calibrator.scale_error.fill_(1)
    logits = torch.zeros(1, 3, 2)
    result = functional_surprise_target(
        logits, logits, critic, torch.zeros(1, 3, dtype=torch.long),
        torch.zeros(1, 3, 2), torch.ones(1, 3, 2), calibrator, config,
    )
    assert result.exploration_bonus.max() > 0
    noisy = replace(critic, aleatoric_uncertainty=torch.ones_like(critic.aleatoric_uncertainty) * 100)
    noisy_result = functional_surprise_target(
        logits, logits, noisy, torch.zeros(1, 3, dtype=torch.long),
        torch.zeros(1, 3, 2), torch.ones(1, 3, 2), calibrator, config,
    )
    assert noisy_result.exploration_bonus.max() < result.exploration_bonus.max() / 10
    calibrator.updates.zero_()
    no_progress = functional_surprise_target(
        logits, logits, critic, torch.zeros(1, 3, dtype=torch.long),
        torch.zeros(1, 3, 2), torch.ones(1, 3, 2), calibrator, config,
    )
    assert no_progress.exploration_bonus.eq(0).all()


def test_actor_loss_applies_fsce_task_trust_and_spectral_terms_without_target_gradient():
    actor = MRRN(actor_config(output_dim=2))
    logits = torch.randn(1, 4, 2, requires_grad=True)
    target_logits = torch.randn(1, 4, 2, requires_grad=True)
    critic = mock_critic(length=4)
    config = surprise_config(horizons=(1, 3))
    calibrator = FunctionalSurpriseCalibrator(2, 2)
    surprise = functional_surprise_target(
        logits, target_logits, critic, torch.zeros(1, 4, dtype=torch.long),
        torch.zeros(1, 4, 2), torch.zeros(1, 4, 2), calibrator, config,
    )
    losses = actor_losses(
        actor, logits, target_logits, surprise, torch.ones(1, 4, dtype=torch.bool),
        config, task_targets=torch.zeros(1, 4, dtype=torch.long),
    )
    losses.total.backward()
    assert logits.grad is not None and target_logits.grad is None
    assert all(torch.isfinite(value) for value in (
        losses.task, losses.functional_cross_entropy, losses.trust_region,
        losses.spectral_regularization,
    ))
    with pytest.raises(ValueError):
        actor_losses(actor, logits, target_logits, surprise, torch.ones(1, 4), config)
    with pytest.raises(ValueError):
        actor_losses(actor, logits, target_logits, surprise, torch.ones(1, 4, dtype=torch.bool), config, task_targets=torch.zeros(1, 4, dtype=torch.long), task_loss=torch.tensor(1.0))
    explicit = actor_losses(
        actor, logits.detach(), target_logits.detach(), surprise,
        torch.ones(1, 4, dtype=torch.bool), config, task_loss=torch.tensor(2.0),
    )
    assert explicit.task == 2
    with pytest.raises(ValueError):
        actor_losses(actor, logits, target_logits, surprise, torch.ones(1, 4, dtype=torch.bool), config, task_loss=torch.ones(2))
    with pytest.raises(ValueError):
        actor_losses(actor, logits, target_logits, surprise, torch.ones(1, 4, dtype=torch.bool), config, task_targets=torch.zeros(1, 4))


def test_replay_priority_is_bounded_product_stratified_padded_and_detached():
    replay = PrioritizedTrajectoryReplay(10, priority_cap=2, prioritized_fraction=0.5)
    batch = trajectory(batch=2, length=6).validated(input_dim=3, action_dim=3)
    values = torch.ones(2, 6)
    indices = replay.add(batch, 10 * values, values, 0.5 * values)
    assert len(indices) == 2 and max(replay.priorities) == 2
    assert replay.transition_count <= 10
    sample = replay.sample(2, generator=torch.Generator().manual_seed(4))
    assert sample.batch.inputs.shape[0] == 2
    assert sample.importance_weights.max() == 1
    torch.testing.assert_close(sample.batch.importance_weights, sample.importance_weights)
    assert not sample.batch.inputs.requires_grad
    replay.update_priorities(sample.indices, torch.tensor([0.2, 0.4]))
    with pytest.raises(ValueError):
        replay.add(batch, values[:, :-1], values, values)
    with pytest.raises(ValueError):
        replay.sample(0)
    with pytest.raises(ValueError):
        replay.update_priorities((0,), torch.ones(2))


def test_performance_guard_blocks_proxy_improvement_when_real_reward_regresses():
    guard = PerformanceGuard(0)
    assert guard.allows(1.0, 2.0)
    assert not guard.allows(0.5, 1.0)
    assert guard.rejections == 1
    assert guard.allows(1.1, 1.5)
    with pytest.raises(ValueError):
        guard.allows(float("nan"), 1)
    with pytest.raises(ValueError):
        PerformanceGuard(-1)


def test_full_learner_parameter_budget_gradient_firewalls_ema_and_train_step():
    torch.manual_seed(44)
    learner = ResonantAdjointSurpriseLearner(MRRN(actor_config()), surprise_config())
    report = learner.parameter_report()
    assert report.critic_fraction <= learner.config.maximum_critic_parameter_fraction
    assert report.target_actor == report.actor and report.target_critic == report.critic
    assert all(not parameter.requires_grad for parameter in learner.target_actor.parameters())
    batch = trajectory()
    padded_actions = batch.actions.clone()
    padded_actions[~batch.mask] = -1
    padded_targets = padded_actions.clone()
    batch = replace(batch, actions=padded_actions, task_targets=padded_targets)
    bootstrap = torch.ones(2, 8, 2, dtype=torch.bool)
    losses = learner.compute_losses(batch, bootstrap_mask=bootstrap)
    losses.critic.total.backward()
    assert all(parameter.grad is None for parameter in learner.actor.parameters())
    assert any(parameter.grad is not None for parameter in learner.critic.parameters())
    learner.zero_grad(set_to_none=True)
    losses.actor.total.backward()
    assert any(parameter.grad is not None for parameter in learner.actor.parameters())
    assert all(parameter.grad is None for parameter in learner.critic.parameters())
    learner.zero_grad(set_to_none=True)
    actor_optimizer, critic_optimizer = learner.make_optimizers(
        actor_policy=OptimizerPolicy(learning_rate=1e-3, warmup_steps=0, total_steps=10),
        critic_policy=OptimizerPolicy(learning_rate=1e-3, warmup_steps=0, total_steps=10),
    )
    target_before = next(learner.target_critic.parameters()).clone()
    step = learner.train_step(batch, actor_optimizer, critic_optimizer)
    assert step.actor_update_applied and step.replay_size > 0
    assert step.actor_gradient_norm > 0 and step.critic_gradient_norm > 0
    assert not torch.equal(target_before, next(learner.target_critic.parameters()))
    assert not learner.target_actor.training and not learner.target_critic.training
    replay_step = learner.train_replay_step(
        2, actor_optimizer, critic_optimizer,
        generator=torch.Generator().manual_seed(2), performance=step.mean_reward,
    )
    assert replay_step.replay_size == step.replay_size
    with pytest.raises(ValueError):
        learner.train_step(
            batch, actor_optimizer, critic_optimizer,
            add_to_replay=True, replay_indices=(0, 1),
        )


def test_learner_rejects_task_loss_as_reward_and_impossible_parameter_budget():
    actor = MRRN(actor_config())
    learner = ResonantAdjointSurpriseLearner(actor, surprise_config())
    with pytest.raises(ValueError, match="external consequences"):
        learner.compute_losses(replace(trajectory(), reward_source="task_loss"))
    with pytest.raises(ValueError, match="parameter budget"):
        ResonantAdjointSurpriseLearner(
            MRRN(actor_config()), surprise_config(maximum_critic_parameter_fraction=1e-5)
        )
    with pytest.raises(ValueError):
        ResonantAdjointSurpriseLearner(MRRN(actor_config(causal=False)), surprise_config())


def test_rasl_checkpoint_restores_actor_critic_targets_calibration_replay_guard_optimizers_and_rng(tmp_path):
    torch.manual_seed(81)
    learner = ResonantAdjointSurpriseLearner(MRRN(actor_config()), surprise_config())
    actor_optimizer, critic_optimizer = learner.make_optimizers(
        actor_policy=OptimizerPolicy(learning_rate=1e-3, warmup_steps=0, total_steps=10),
        critic_policy=OptimizerPolicy(learning_rate=1e-3, warmup_steps=0, total_steps=10),
    )
    learner.train_step(trajectory(), actor_optimizer, critic_optimizer)
    path = tmp_path / "rasl.pt"
    save_rasl_checkpoint(
        path, learner, actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer, step=7,
    )
    expected_parameter = next(learner.critic.parameters()).clone()
    expected_priorities = learner.replay.priorities
    expected_guard = learner.performance_guard.state_dict()
    expected_random = torch.rand(4)
    restored = ResonantAdjointSurpriseLearner(MRRN(actor_config()), surprise_config())
    restored_actor_optimizer, restored_critic_optimizer = restored.make_optimizers(
        actor_policy=OptimizerPolicy(learning_rate=1e-3, warmup_steps=0, total_steps=10),
        critic_policy=OptimizerPolicy(learning_rate=1e-3, warmup_steps=0, total_steps=10),
    )
    assert load_rasl_checkpoint(
        path, restored, actor_optimizer=restored_actor_optimizer,
        critic_optimizer=restored_critic_optimizer,
    ) == 7
    torch.testing.assert_close(next(restored.critic.parameters()), expected_parameter)
    assert restored.replay.priorities == expected_priorities
    assert restored.performance_guard.state_dict() == expected_guard
    torch.testing.assert_close(torch.rand(4), expected_random)
    assert restored_actor_optimizer.state_dict()["state"]
    with pytest.raises(ValueError, match="presence"):
        load_rasl_checkpoint(path, restored)
    payload = torch.load(path, weights_only=True)
    payload["format_version"] = 99
    torch.save(payload, path)
    with pytest.raises(ValueError, match="version"):
        load_rasl_checkpoint(
            path, restored, actor_optimizer=restored_actor_optimizer,
            critic_optimizer=restored_critic_optimizer,
        )


def test_reference_system_default_critic_is_under_five_percent_of_actor_parameters():
    config = MRRNConfig(
        input_dim=8, model_dim=32, output_dim=8, layers=2, scales=4, heads=4,
        modes=8, mimo_rank=2, attention_window=16, retrieved_items=4,
        memory_capacity=64, mixer_expansion=2, width_growth_cap=1.5,
        mode_growth_cap=1.5, width_multiple=4,
    )
    learner = ResonantAdjointSurpriseLearner(MRRN(config))
    report = learner.parameter_report()
    assert report.selected_width == 16 and report.selected_scales == (0, 2, 3)
    assert report.critic_fraction < 0.05


def test_critic_and_distribution_contract_failures_are_explicit():
    actor = MRRN(actor_config())
    config = surprise_config()
    critic = ResonantAdjointCritic(actor.config, config, width=8)
    output = actor(torch.randn(1, 8, 3))
    actions = torch.zeros(1, 8, dtype=torch.long)
    rewards = torch.zeros(1, 8)
    dones = torch.zeros(1, 8, dtype=torch.bool)
    result = critic(output.bands, output.prediction, actions=actions, rewards=rewards, dones=dones)
    assert critic.spectral_modules
    with pytest.raises(ValueError):
        result.quantiles_for(actions.float())
    with pytest.raises(ValueError):
        ResonantAdjointCritic(actor_config(output_dim=1), config, width=8)
    with pytest.raises(ValueError):
        ResonantAdjointCritic(actor.config, config, width=3)
    with pytest.raises(ValueError):
        critic.value_distribution(torch.ones(1, 2, 7))
    with pytest.raises(ValueError):
        critic(output.bands, torch.ones(1, 8, 2), actions=actions, rewards=rewards, dones=dones)
    with pytest.raises(ValueError):
        critic(output.bands, output.prediction, mask=torch.ones(1, 8), include_adjoint=False)
    with pytest.raises(ValueError):
        critic(output.bands, output.prediction, actions=actions.float(), rewards=rewards, dones=dones)
    with pytest.raises(ValueError):
        critic(output.bands, output.prediction, actions=actions, rewards=rewards[:, :-1], dones=dones)
    with pytest.raises(ValueError):
        critic(output.bands, output.prediction, actions=actions, rewards=rewards, dones=dones.float())
    with pytest.raises(ValueError):
        critic._prepare_bands(output.bands[:1])
    projection = _FrozenSpectralProjection(4, 2)
    with pytest.raises(ValueError):
        projection(torch.ones(1, 3))
    empty = ScaleTensor(torch.zeros(1, 0, 8), torch.zeros(1, 0, dtype=torch.bool), 0, 1, 2, "detail")
    assert critic._align_outcome(torch.zeros(1, 8, 5), empty).shape[1] == 0

    prepared = critic._prepare_bands(output.bands)
    layer = critic.layers[0]
    with pytest.raises(ValueError):
        layer(prepared, (torch.zeros(1, 1, 5),))
    bad_outcomes = tuple(torch.zeros(1, band.data.shape[1] + 1, 5) for band in prepared)
    with pytest.raises(ValueError):
        layer(prepared, bad_outcomes)


def test_loss_and_target_contract_failures_cover_every_public_shape_boundary():
    target = torch.ones(1, 2, 2)
    prediction = torch.ones(1, 2, 2, 2, 3)
    mask = torch.ones(1, 2, 2, dtype=torch.bool)
    with pytest.raises(ValueError):
        quantile_huber_loss(prediction, target, (0.2, 0.5, 0.8), mask.float())
    with pytest.raises(ValueError):
        quantile_huber_loss(prediction, target, (0.0, 0.5, 0.8), mask)
    with pytest.raises(ValueError):
        quantile_huber_loss(prediction, target, (0.2, 0.5, 0.8), mask, kappa=0)
    with pytest.raises(ValueError):
        differentiable_quantile_calibration_loss(prediction[..., 0], target, (0.5,), mask)
    with pytest.raises(ValueError):
        differentiable_quantile_calibration_loss(prediction, target, (0.2, 0.5, 0.8), mask.float())
    with pytest.raises(ValueError):
        differentiable_quantile_calibration_loss(prediction, target, (0.2,), mask)
    rewards = torch.ones(1, 2)
    dones = torch.zeros(1, 2, dtype=torch.bool)
    valid = torch.ones_like(dones)
    long, _ = multihorizon_returns(rewards, dones, valid, (8,), discount=1)
    assert long[0, 0, 0] == 2
    with pytest.raises(ValueError):
        multihorizon_returns(rewards.unsqueeze(-1), dones, valid, (1,), discount=1)

    critic = mock_critic(length=3)
    config = surprise_config(horizons=(1, 3))
    calibrator = FunctionalSurpriseCalibrator(2, 2)
    logits = torch.zeros(1, 3, 2)
    actions = torch.zeros(1, 3, dtype=torch.long)
    returns = torch.zeros(1, 3, 2)
    phase = torch.zeros(1, 3, 2)
    with pytest.raises(ValueError):
        functional_surprise_target(logits, logits[..., :1], critic, actions, returns, phase, calibrator, config)
    with pytest.raises(ValueError):
        functional_surprise_target(logits, logits, critic, actions[:, :-1], returns, phase, calibrator, config)
    with pytest.raises(ValueError):
        functional_surprise_target(logits, logits, critic, actions, returns, phase[..., :1], calibrator, config)
    with pytest.raises(FloatingPointError):
        functional_surprise_target(logits, logits.fill_(float("nan")), critic, actions, returns, phase, calibrator, config)


def test_replay_guard_and_checkpoint_corruption_fail_closed(tmp_path):
    with pytest.raises(ValueError):
        PrioritizedTrajectoryReplay(0)
    with pytest.raises(ValueError):
        PrioritizedTrajectoryReplay(1, prioritized_fraction=2)
    replay = PrioritizedTrajectoryReplay(20, prioritized_fraction=0)
    raw = replace(trajectory(batch=1, length=4), mask=None)
    with pytest.raises(ValueError):
        replay.add(raw, torch.ones(1, 4), torch.ones(1, 4), torch.ones(1, 4))
    batch = raw.validated(input_dim=3, action_dim=3)
    bad = torch.ones(1, 4)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        replay.add(batch, bad, torch.ones(1, 4), torch.ones(1, 4))
    empty_mask = torch.zeros(1, 4, dtype=torch.bool)
    assert replay.add(replace(batch, mask=empty_mask), torch.ones(1, 4), torch.ones(1, 4), torch.ones(1, 4)) == ()
    replay.add(batch, torch.ones(1, 4), torch.ones(1, 4), torch.ones(1, 4))
    sample = replay.sample(1, device="cpu", generator=torch.Generator().manual_seed(1))
    assert sample.batch.inputs.device.type == "cpu"
    with pytest.raises(ValueError):
        replay.update_priorities((0,), torch.tensor([float("nan")]))
    with pytest.raises(ValueError):
        replay.update_priorities((9,), torch.tensor([1.0]))
    state = replay.state_dict()
    wrong = dict(state, capacity=99)
    with pytest.raises(ValueError):
        replay.load_state_dict(wrong)
    wrong = dict(state, transitions=99)
    with pytest.raises(ValueError):
        replay.load_state_dict(wrong)
    wrong = dict(state)
    wrong["items"] = [dict(state["items"][0], priority=-1)]
    with pytest.raises(ValueError):
        replay.load_state_dict(wrong)
    tiny_replay = PrioritizedTrajectoryReplay(5)
    tiny_replay.add(batch, torch.ones(1, 4), torch.ones(1, 4), torch.ones(1, 4))
    tiny_replay.add(batch, torch.ones(1, 4), torch.ones(1, 4), torch.ones(1, 4))
    assert len(tiny_replay) == 1 and tiny_replay.transition_count == 4
    invalid_item = replay.state_dict()
    invalid_item["items"][0]["trajectory"]["inputs"] = torch.zeros(2, 0, 3)
    invalid_item["transitions"] = 0
    with pytest.raises(ValueError):
        replay.load_state_dict(invalid_item)

    guard = PerformanceGuard(0.1)
    with pytest.raises(ValueError):
        guard.load_state_dict({"tolerance": 0.2})
    with pytest.raises(ValueError):
        guard.load_state_dict({"tolerance": 0.1, "reference_performance": 1, "best_surprise_loss": None, "rejections": 0})
    with pytest.raises(ValueError):
        guard.load_state_dict({"tolerance": 0.1, "reference_performance": float("nan"), "best_surprise_loss": 1, "rejections": 0})
    with pytest.raises(ValueError):
        guard.load_state_dict({"tolerance": 0.1, "reference_performance": None, "best_surprise_loss": None, "rejections": -1})

    learner = ResonantAdjointSurpriseLearner(MRRN(actor_config()), surprise_config())
    path = tmp_path / "minimal.pt"
    with pytest.raises(ValueError):
        save_rasl_checkpoint(path, learner, step=-1)
    save_rasl_checkpoint(path, learner, step=1)
    with pytest.raises(ValueError, match="configuration"):
        load_rasl_checkpoint(path, ResonantAdjointSurpriseLearner(MRRN(actor_config()), surprise_config(ema_decay=0.8)))
    with pytest.raises(ValueError, match="actor configuration"):
        load_rasl_checkpoint(path, ResonantAdjointSurpriseLearner(MRRN(actor_config(model_dim=17)), surprise_config()))
    payload = torch.load(path, weights_only=True)
    payload["step"] = -1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="step"):
        load_rasl_checkpoint(path, learner)


def test_rollout_behavior_logits_fast_target_path_and_update_veto_are_exercised():
    learner = ResonantAdjointSurpriseLearner(MRRN(actor_config()), surprise_config(performance_tolerance=0))
    batch = trajectory()
    behavior = learner.rollout_policy(batch.inputs, batch.mask).prediction.detach()
    losses = learner.compute_losses(replace(batch, behavior_logits=behavior))
    torch.testing.assert_close(losses.target_logits, behavior)
    actor_optimizer, critic_optimizer = learner.make_optimizers(
        actor_policy=OptimizerPolicy(learning_rate=1e-3, warmup_steps=0, total_steps=10),
        critic_policy=OptimizerPolicy(learning_rate=1e-3, warmup_steps=0, total_steps=10),
    )
    learner.performance_guard.reference_performance = 100
    learner.performance_guard.best_surprise_loss = 1e9
    target_before = next(learner.target_actor.parameters()).clone()
    report = learner.train_step(
        replace(batch, behavior_logits=behavior), actor_optimizer, critic_optimizer,
        performance=0, add_to_replay=False,
    )
    assert not report.actor_update_applied and report.replay_size == 0
    torch.testing.assert_close(next(learner.target_actor.parameters()), target_before)
    with pytest.raises(ValueError):
        ResonantAdjointSurpriseLearner(MRRN(actor_config(output_dim=1)), surprise_config())


def test_end_to_end_delayed_consequence_training_improves_the_earliest_action_policy():
    """The only reward arrives five transitions after the decision under test."""

    torch.manual_seed(123)
    torch.set_num_threads(1)
    actor = MRRN(actor_config(
        input_dim=2, model_dim=12, output_dim=2, scales=2, attention_window=3,
    ))
    config = surprise_config(
        critic_width=6, critic_scales=2, action_rank=2, horizons=(1, 6),
        ema_decay=0.9, calibration_decay=0.9,
        surprise_cross_entropy_weight=1, trust_region_weight=0.01,
        spectral_regularization_weight=0, critic_latent_weight=0.05,
        maximum_gradient_norm=2, performance_tolerance=100,
    )
    learner = ResonantAdjointSurpriseLearner(actor, config)
    actor_optimizer, critic_optimizer = learner.make_optimizers(
        actor_policy=OptimizerPolicy(learning_rate=3e-3, warmup_steps=0, total_steps=100),
        critic_policy=OptimizerPolicy(learning_rate=3e-3, warmup_steps=0, total_steps=100),
    )

    def policy_accuracy():
        inputs = torch.zeros(2, 6, 2)
        inputs[:, 0] = torch.eye(2)
        with torch.no_grad():
            probabilities = learner.actor(inputs).prediction[:, 0].softmax(-1)
        return probabilities[torch.arange(2), torch.arange(2)].mean()

    def delayed_batch(size=64):
        cue = torch.arange(size) % 2
        inputs = torch.zeros(size, 6, 2)
        inputs[:, 0] = F.one_hot(cue, 2).float()
        actions = torch.randint(0, 2, (size, 6))
        actions[:, 0] = torch.arange(size).div(2, rounding_mode="floor") % 2
        rewards = torch.zeros(size, 6)
        rewards[:, -1] = torch.where(actions[:, 0] == cue, 1.0, -1.0)
        dones = torch.zeros(size, 6, dtype=torch.bool)
        dones[:, -1] = True
        return TrajectoryBatch(inputs, actions, rewards, dones)

    before = policy_accuracy()
    for _ in range(40):
        learner.train_step(
            delayed_batch(), actor_optimizer, critic_optimizer, add_to_replay=False
        )
    after = policy_accuracy()
    assert after > before + 0.1
