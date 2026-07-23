from dataclasses import replace

import pytest
import torch

from mrrn.cognitive_surprise import (
    CognitiveRASLConfig, CognitiveResonantAdjointSurpriseLearner,
    CognitiveTrajectoryBatch, CognitiveTrajectoryReplay, build_language_candidate_set,
    load_cognitive_rasl_checkpoint, save_cognitive_rasl_checkpoint,
)
from mrrn.config import CognitiveConfig, MRCRAConfig, MRRNConfig
from mrrn.language import MRCRALanguageModel
from mrrn.optimization import OptimizerPolicy
from mrrn.surprise import ResonantAdjointSurpriseConfig


def configuration(vocabulary: int = 11):
    carrier = MRRNConfig(
        input_dim=8, model_dim=8, output_dim=vocabulary, layers=1, scales=2,
        heads=2, modes=2, mimo_rank=1, attention_window=2,
        retrieved_items=1, memory_capacity=4, mixer_expansion=1.5,
        width_growth_cap=1.0, mode_growth_cap=1.0, width_multiple=4,
        spectral_modes=2, spectral_basis_order=2, spectral_triads_per_mode=1,
        enable_global_head=False, relational_branch=True, relational_context_dim=8,
    )
    cognitive = CognitiveConfig(
        workspace_dim=8, provenance_features=4, uncertainty_channels=8,
        relation_heads=2, relation_modes=2, relation_adapter_rank=1,
        goal_slots=1, goal_constraint_dim=2, system_action_channels=2,
        calibration_regimes=2, active_event_capacity=4, pair_edge_capacity=6,
        hyperedge_capacity=2, maximum_hyperedge_arity=3, graph_neighbors=1,
        global_workspace_slots=1, hypothesis_slots=1, maximum_hypothesis_slots=2,
        maximum_cognitive_steps=1, event_chunk_size=2, event_proposals_per_chunk=1,
        recent_candidates=2, landmark_candidates=1, episodic_candidates=1,
        semantic_candidates=1, episodic_memory_capacity=4,
        semantic_memory_capacity=2, associative_depth=1, associative_budget=1,
        world_model_horizons=(1, 2),
    )
    return MRCRAConfig(
        carrier, cognitive, actor_parameter_minimum=1,
        actor_parameter_maximum=10_000_000,
    )


def rasl_configuration():
    core = ResonantAdjointSurpriseConfig(
        critic_width=8, minimum_critic_width=4, critic_layers=1,
        critic_scales=1, critic_heads=1, critic_modes=2,
        critic_mimo_rank=1, spectral_modes=2, spectral_basis_order=2,
        action_rank=2, latent_modes=2, horizons=(1, 2),
        quantiles=(0.1, 0.5, 0.9), bootstrap_heads=2,
        maximum_critic_parameter_fraction=1.0, require_external_reward=True,
    )
    return CognitiveRASLConfig(core=core, maximum_candidates=5)


def trajectory(model, *, reward=1.0, reward_source="environment"):
    torch.manual_seed(301)
    inputs = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    behavior = torch.tensor([[2, 3, 4, 5]], dtype=torch.int64)
    target_logits = torch.randn(1, 4, model.vocabulary_size)
    candidates, log_probability, sampled = build_language_candidate_set(
        target_logits, behavior, candidate_count=5,
    )
    return CognitiveTrajectoryBatch(
        inputs, behavior, candidates, log_probability, sampled,
        torch.full((1, 4), reward), torch.tensor([[False, False, False, True]]),
        torch.ones(1, 4, dtype=torch.bool), task_targets=behavior,
        reward_source=reward_source, burn_in_steps=1,
    )


def test_candidate_builder_is_unique_bounded_and_keeps_behavior_action():
    logits = torch.randn(2, 3, 17)
    behavior = torch.randint(0, 17, (2, 3))
    candidate, log_probability, sampled = build_language_candidate_set(
        logits, behavior, candidate_count=7,
    )
    assert candidate.shape == (2, 3, 7)
    assert bool(((candidate == behavior.unsqueeze(-1)).sum(-1) == 1).all())
    assert bool((candidate.sort(-1).values[..., 1:] != candidate.sort(-1).values[..., :-1]).all())
    assert bool((log_probability[sampled] < 0).all())
    assert bool((log_probability[~sampled] == 0).all())


def test_cognitive_rasl_uses_bounded_candidates_and_firewalls_gradients():
    model = MRCRALanguageModel(configuration())
    learner = CognitiveResonantAdjointSurpriseLearner(model, rasl_configuration())
    batch = trajectory(model)
    losses = learner.compute_losses(batch)
    assert losses.candidate_logits.shape == (1, 4, 5)
    assert losses.surprise.distribution.shape == (1, 4, 5)
    assert losses.critic.returns.shape == (1, 4, 2)
    losses.critic.total.backward(retain_graph=True)
    assert any(parameter.grad is not None for parameter in learner.critic.parameters())
    assert all(parameter.grad is None for parameter in learner.actor.parameters())
    learner.critic.zero_grad(set_to_none=True)
    learner.actor.zero_grad(set_to_none=True)
    losses.actor.total.backward()
    assert any(parameter.grad is not None for parameter in learner.actor.parameters())
    assert all(parameter.grad is None for parameter in learner.critic.parameters())


def test_negative_external_consequence_reduces_behavior_target_probability():
    model = MRCRALanguageModel(configuration())
    learner = CognitiveResonantAdjointSurpriseLearner(model, rasl_configuration())
    positive_batch = trajectory(model, reward=2.0)
    negative_batch = replace(positive_batch, rewards=torch.full((1, 4), -2.0))
    positive = learner.compute_losses(positive_batch).surprise.distribution
    negative = learner.compute_losses(negative_batch).surprise.distribution
    local = (
        positive_batch.candidate_token_ids == positive_batch.behavior_tokens.unsqueeze(-1)
    ).to(torch.int64).argmax(-1)
    positive_chosen = positive.gather(-1, local.unsqueeze(-1)).squeeze(-1)
    negative_chosen = negative.gather(-1, local.unsqueeze(-1)).squeeze(-1)
    assert bool((positive_chosen[positive_batch.loss_mask] > negative_chosen[positive_batch.loss_mask]).all())


def test_cognitive_rasl_rejects_task_loss_as_reward_and_validates_candidates():
    model = MRCRALanguageModel(configuration())
    learner = CognitiveResonantAdjointSurpriseLearner(model, rasl_configuration())
    with pytest.raises(ValueError, match="external downstream consequence"):
        learner.compute_losses(trajectory(model, reward_source="task_loss"))
    batch = trajectory(model)
    duplicate = batch.candidate_token_ids.clone()
    duplicate[..., 1] = duplicate[..., 0]
    with pytest.raises(ValueError, match="unique"):
        learner.compute_losses(replace(batch, candidate_token_ids=duplicate))


def test_cognitive_rasl_checkpoint_restores_targets_calibration_guard_and_optimizers(tmp_path):
    model = MRCRALanguageModel(configuration())
    learner = CognitiveResonantAdjointSurpriseLearner(model, rasl_configuration())
    policy = OptimizerPolicy(warmup_steps=0, total_steps=2)
    actor_optimizer, critic_optimizer = learner.make_optimizers(
        actor_policy=policy, critic_policy=policy,
    )
    learner.train_step(trajectory(model), actor_optimizer, critic_optimizer)
    destination = tmp_path / "cognitive-rasl.pt"
    save_cognitive_rasl_checkpoint(
        destination, learner, actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer, step=7,
    )
    restored = CognitiveResonantAdjointSurpriseLearner(
        MRCRALanguageModel(configuration()), rasl_configuration()
    )
    restored_actor_optimizer, restored_critic_optimizer = restored.make_optimizers(
        actor_policy=policy, critic_policy=policy,
    )
    assert load_cognitive_rasl_checkpoint(
        destination, restored, actor_optimizer=restored_actor_optimizer,
        critic_optimizer=restored_critic_optimizer,
    ) == 7
    for left, right in zip(learner.state_dict().values(), restored.state_dict().values(), strict=True):
        torch.testing.assert_close(left, right)
    assert restored.performance_guard.state_dict() == learner.performance_guard.state_dict()
    assert restored.replay.state_dict()["transitions"] == learner.replay.state_dict()["transitions"]


def test_cognitive_replay_requires_burn_in_samples_and_restores_exactly():
    model = MRCRALanguageModel(configuration())
    batch = trajectory(model)
    replay = CognitiveTrajectoryReplay(32, burn_in_steps=1)
    signal = torch.ones_like(batch.rewards)
    indices = replay.add(batch, signal, signal, signal)
    assert indices == (0,) and replay.transition_count == 4
    sample = replay.sample(1)
    assert sample.batch.burn_in_steps == 1
    assert not bool(sample.batch.loss_mask[:, 0].any())
    sample.batch.validated(
        vocabulary_size=model.vocabulary_size,
        width=model.config.cognitive.workspace_dim,
        maximum_candidates=5,
    )
    restored = CognitiveTrajectoryReplay(32, burn_in_steps=1)
    restored.load_state_dict(replay.state_dict())
    assert restored.priorities == replay.priorities
    with pytest.raises(ValueError, match="burn-in"):
        replay.add(replace(batch, burn_in_steps=0), signal, signal, signal)
