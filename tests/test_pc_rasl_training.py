from dataclasses import replace
import json

import torch

from mrrn.cognitive_training import (
    MRCRANextTokenTrainer, MRCRATrainingConfig,
)
from mrrn.language import MRCRALanguageModel
from mrrn.learning_progress import LearningProgressConfig
from mrrn.lm_training import (
    ByteTextTokenizer, PackedTokenStream, SequenceTextSource,
    build_evaluation_batches,
)
from mrrn.optimization import gradient_subsystem
from tests.test_cognitive_training import tiny_config


TRAINING_DOCUMENTS = (
    "alpha beta gamma delta",
    "epsilon zeta eta theta",
    "iota kappa lambda mu",
)
PROGRESS_DOCUMENTS = (
    "progress authority has no phase metric input",
    "learning rate is measured on a disjoint stream",
)
GUARD_DOCUMENTS = (
    "the guard remains outside optimization",
    "retained evidence detects progress probe overfitting",
)


def retained(tokenizer, documents):
    return build_evaluation_batches(
        PackedTokenStream(SequenceTextSource(documents), tokenizer),
        count=1,
        batch_size=1,
        sequence_length=8,
    )


def pc_config(path):
    return MRCRATrainingConfig(
        output_dir=str(path),
        total_tokens=48,
        context_length=8,
        execution_chunk_size=2,
        tbptt_length=4,
        vocabulary_tile_size=32,
        integrated_cognitive_path=True,
        cognitive_stride=2,
        cognitive_tbptt_events=2,
        progress_interval_tokens=8,
        warmup_tokens=8,
        checkpoint_interval=100,
        evaluation_interval=1,
        evaluation_batches=1,
        require_evaluation=True,
        progress_conditioned_rasl=True,
        progress_probe_batches=1,
        progress_probe_length=8,
        pc_rasl_trajectory_length=8,
        pc_rasl_candidate_count=8,
        pc_rasl_max_interval_trajectories=1,
        pc_rasl_critic_warmup_observations=1,
        learning_progress=LearningProgressConfig(
            observation_interval=1,
            warmup_observations=4,
            fast_window=3,
            baseline_min_observations=4,
            baseline_window=8,
            baseline_lag=0,
            baseline_freeze_observations=1,
            deadband_standard_deviations=0,
        ),
        phase_transition_ablation=False,
        trackio_enabled=False,
        show_dashboard=False,
        spectral_dashboard=False,
        data_prefetch=True,
        device="cpu",
        precision="fp32",
        seed=20260723,
    )


def trainer(path):
    tokenizer = ByteTextTokenizer()
    return MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        PackedTokenStream(SequenceTextSource(TRAINING_DOCUMENTS), tokenizer),
        pc_config(path),
        retained(tokenizer, GUARD_DOCUMENTS),
        progress_probe_batches=retained(tokenizer, PROGRESS_DOCUMENTS),
    )


def assert_tree_exact(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            assert_tree_exact(left_item, right_item)
    else:
        assert left == right


def test_pc_rasl_production_path_is_checkpoint_resume_exact(tmp_path):
    torch.manual_seed(20260723)
    reference = trainer(tmp_path / "reference")
    reference.train(maximum_steps=6)

    torch.manual_seed(20260723)
    split = trainer(tmp_path / "split")
    split.train(maximum_steps=3)
    checkpoint = split.save_checkpoint()

    restored = trainer(tmp_path / "split")
    restored.load_checkpoint(checkpoint)
    restored.train(maximum_steps=3)

    for name in (
        "step", "tokens_seen", "valid_targets_seen", "bytes_seen",
        "last_evaluation_step", "last_event_proposal_logit_max",
        "event_proposal_logit_slope_ema", "event_proposal_observations",
        "last_progress_observation_step", "last_progress_pressure",
        "progress_observations", "pc_rasl_updates_due",
        "pc_rasl_trajectories_captured", "pc_rasl_replay_updates",
    ):
        assert getattr(restored.state, name) == getattr(reference.state, name)
    deterministic_evaluation = {
        key: value
        for key, value in restored.state.last_evaluation_metrics.items()
        if not key.endswith("seconds") and "tokens_per_second" not in key
    }
    assert deterministic_evaluation == {
        key: value
        for key, value in reference.state.last_evaluation_metrics.items()
        if not key.endswith("seconds") and "tokens_per_second" not in key
    }
    assert restored.learning_progress.state_dict() == (
        reference.learning_progress.state_dict()
    )
    assert_tree_exact(
        restored.pc_rasl.replay.state_dict(),
        reference.pc_rasl.replay.state_dict(),
    )
    assert restored.pc_rasl.performance_guard.state_dict() == (
        reference.pc_rasl.performance_guard.state_dict()
    )
    for left, right in zip(
        restored.model.state_dict().values(),
        reference.model.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    for left, right in zip(
        restored.pc_rasl.critic.state_dict().values(),
        reference.pc_rasl.critic.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    for left, right in zip(
        restored.pc_rasl.target_critic.state_dict().values(),
        reference.pc_rasl.target_critic.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert_tree_exact(
        restored.pc_rasl.calibrator.state_dict(),
        reference.pc_rasl.calibrator.state_dict(),
    )
    assert_tree_exact(
        restored.pc_rasl_critic_optimizer.state_dict(),
        reference.pc_rasl_critic_optimizer.state_dict(),
    )
    assert_tree_exact(
        [
            restored._trajectory_state(batch)
            for batch in restored._pc_rasl_pending_batches
        ],
        [
            reference._trajectory_state(batch)
            for batch in reference._pc_rasl_pending_batches
        ],
    )
    assert_tree_exact(
        [
            restored._trajectory_state(batch)
            for batch in restored._pc_rasl_finalized_batches
        ],
        [
            reference._trajectory_state(batch)
            for batch in reference._pc_rasl_finalized_batches
        ],
    )


def test_pc_rasl_checkpoint_binds_both_probe_and_guard_evidence(tmp_path):
    torch.manual_seed(20260723)
    source = trainer(tmp_path / "bound")
    source.train(maximum_steps=1)
    assert len(source._pc_rasl_finalized_batches) == 1
    retained_trajectory = source._pc_rasl_finalized_batches[0]
    assert retained_trajectory.behavior_candidate_logits is not None
    assert retained_trajectory.behavior_cognitive_features is not None
    assert retained_trajectory.behavior_workspace_features is not None
    assert retained_trajectory.behavior_relation_features is not None
    assert retained_trajectory.behavior_relation_type_probabilities is not None
    assert retained_trajectory.behavior_internal_actions is not None
    assert retained_trajectory.behavior_internal_statuses is not None
    assert retained_trajectory.behavior_internal_mask is not None
    checkpoint = source.save_checkpoint()
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["identity"]["progress_probe"] == source.progress_probe_identity
    assert payload["identity"]["evaluation"] == source.evaluation_identity
    assert payload["pc_rasl"] is not None
    assert payload["learning_progress"] is not None
    finalized = payload["pc_rasl"]["finalized_batches"][0]
    assert finalized["behavior_candidate_logits"] is not None
    assert finalized["behavior_cognitive_features"] is not None
    assert finalized["behavior_internal_actions"] is not None

    tokenizer = ByteTextTokenizer()
    mismatched_progress = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        PackedTokenStream(SequenceTextSource(TRAINING_DOCUMENTS), tokenizer),
        pc_config(tmp_path / "bound"),
        retained(tokenizer, GUARD_DOCUMENTS),
        progress_probe_batches=retained(
            tokenizer, ("different progress evidence",)
        ),
    )
    try:
        mismatched_progress.load_checkpoint(checkpoint)
    except ValueError as error:
        assert "contract differs" in str(error)
    else:
        raise AssertionError("checkpoint accepted a different progress probe")


def test_pc_rasl_metrics_and_gradient_routing_are_observable(tmp_path):
    torch.manual_seed(20260723)
    run = trainer(tmp_path / "observable")
    run.train(maximum_steps=6)
    rows = [
        json.loads(line)
        for line in (
            tmp_path / "observable" / "progress_metrics.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 6
    assert all(
        row["progress_probe_identity"] == run.progress_probe_identity
        for row in rows
    )
    assert run.state.progress_observations == 6
    assert run.pc_rasl.replay.transition_count > 0
    assert run._pc_rasl_step_metrics["pc_rasl/progress_return_loss"] >= 0
    assert run._pc_rasl_step_metrics["pc_rasl/internal_action_value_loss"] >= 0
    assert run._pc_rasl_step_metrics["pc_rasl/behavior_evidence_bound"] == 1
    assert run._pc_rasl_step_metrics["pc_rasl/replay_storage_bytes"] > 0
    # The output/token carrier receives the strictest auxiliary cap; controller
    # and cognitive subsystems receive the stronger adaptation budget.
    assert gradient_subsystem("token_embedding.weight") == "carrier"
    assert gradient_subsystem(
        "cognitive.controller.action_head.weight"
    ) == "controller"


def test_pc_rasl_work_is_issued_only_by_new_progress_consequences(tmp_path):
    tokenizer = ByteTextTokenizer()
    base = pc_config(tmp_path / "consequence-cadence")
    configuration = replace(
        base,
        learning_progress=replace(
            base.learning_progress,
            observation_interval=3,
        ),
    )
    run = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        PackedTokenStream(SequenceTextSource(TRAINING_DOCUMENTS), tokenizer),
        configuration,
        retained(tokenizer, GUARD_DOCUMENTS),
        progress_probe_batches=retained(tokenizer, PROGRESS_DOCUMENTS),
    )

    run.train(maximum_steps=6)

    assert run.state.progress_observations == 2
    assert run.state.pc_rasl_trajectories_captured == 2
    assert run.state.pc_rasl_replay_updates == 1
    assert run.state.pc_rasl_updates_due == 1
    assert len(run.pc_rasl.replay) == 1


def test_format8_checkpoint_migrates_into_fresh_causal_pc_rasl_warmup(tmp_path):
    torch.manual_seed(20260723)
    tokenizer = ByteTextTokenizer()
    path = tmp_path / "format8-migration"
    old_configuration = replace(
        pc_config(path),
        progress_conditioned_rasl=False,
        progress_probe_batches=0,
    )
    old = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        PackedTokenStream(
            SequenceTextSource(TRAINING_DOCUMENTS), tokenizer
        ),
        old_configuration,
        retained(tokenizer, GUARD_DOCUMENTS),
    )
    old.train(maximum_steps=1)
    current = old.save_checkpoint()
    payload = torch.load(current, weights_only=True)
    payload["format_version"] = 8
    payload["identity"].pop("progress_probe", None)
    for name in (
        "progress_conditioned_rasl",
        "progress_probe_batches",
        "progress_probe_length",
        "learning_progress",
        "pc_rasl_trajectory_length",
        "pc_rasl_candidate_count",
        "pc_rasl_replay_batch_size",
        "pc_rasl_max_interval_trajectories",
        "pc_rasl_critic_warmup_observations",
        "pc_rasl_consequence_weight",
        "pc_rasl_critic_learning_rate",
        "pc_rasl_carrier_gradient_cap",
        "pc_rasl_cognitive_gradient_cap",
        "pc_rasl_controller_gradient_cap",
    ):
        payload["identity"]["training"].pop(name, None)
    payload.pop("learning_progress", None)
    payload.pop("pc_rasl", None)
    for name in (
        "last_progress_observation_step",
        "last_progress_pressure",
        "progress_observations",
    ):
        payload["training_state"].pop(name, None)
    legacy = path / "format8.pt"
    torch.save(payload, legacy)

    restored = trainer(path)
    restored.load_checkpoint(legacy)
    assert restored.state.step == 1
    assert restored.state.progress_observations == 0
    assert restored.learning_progress.observations == []
    assert restored.pc_rasl.replay.transition_count == 0
    assert not restored._pc_rasl_pending_batches
    assert not restored._pc_rasl_finalized_batches
    restored.train(maximum_steps=1)
    assert restored.state.progress_observations == 1


def test_format9_pc_rasl_checkpoint_discards_pre_v10_replay_authority(tmp_path):
    """Pre-v10 replay cannot be represented as exact behavior evidence."""

    torch.manual_seed(20260723)
    path = tmp_path / "format9-migration"
    old = trainer(path)
    old.train(maximum_steps=2)
    current = old.save_checkpoint()
    payload = torch.load(current, weights_only=True)
    assert payload["learning_progress"] is not None
    assert payload["pc_rasl"] is not None
    assert payload["pc_rasl"]["finalized_batches"]
    payload["format_version"] = 9
    legacy = path / "format9.pt"
    torch.save(payload, legacy)

    restored = trainer(path)
    restored.load_checkpoint(legacy)
    assert restored.state.step == 2
    assert restored.state.progress_observations == 0
    assert restored.learning_progress.observations == []
    assert restored.pc_rasl.replay.transition_count == 0
    assert not restored._pc_rasl_pending_batches
    assert not restored._pc_rasl_finalized_batches
    restored.train(maximum_steps=1)
    assert restored.state.progress_observations == 1


def test_format10_checkpoint_migrates_outstanding_consequence_once(tmp_path):
    torch.manual_seed(20260723)
    path = tmp_path / "format10-migration"
    source = trainer(path)
    source.train(maximum_steps=1)
    current = source.save_checkpoint()
    payload = torch.load(current, weights_only=True)
    assert payload["pc_rasl"]["finalized_batches"]
    payload["format_version"] = 10
    for name in (
        "pc_rasl_captures_per_observation",
        "pc_rasl_updates_per_observation",
    ):
        payload["identity"]["training"].pop(name)
    for name in (
        "pc_rasl_updates_due",
        "pc_rasl_trajectories_captured",
        "pc_rasl_replay_updates",
    ):
        payload["training_state"].pop(name)
    legacy = path / "format10.pt"
    torch.save(payload, legacy)

    restored = trainer(path)
    restored.load_checkpoint(legacy)

    assert restored.state.pc_rasl_updates_due == 1
    assert len(restored._pc_rasl_finalized_batches) == 1
    restored.train(maximum_steps=1)
    assert restored.state.pc_rasl_replay_updates == 1
    assert restored.state.pc_rasl_updates_due == 1


def test_format11_pc_rasl_checkpoint_can_resume_with_subsystem_retired(tmp_path):
    torch.manual_seed(20260723)
    path = tmp_path / "format11-retirement"
    source = trainer(path)
    source.train(maximum_steps=2)
    current = source.save_checkpoint()
    payload = torch.load(current, weights_only=True)
    assert payload["learning_progress"] is not None
    assert payload["pc_rasl"] is not None
    payload["format_version"] = 11
    legacy = path / "format11.pt"
    torch.save(payload, legacy)

    tokenizer = ByteTextTokenizer()
    disabled_config = replace(
        pc_config(path),
        progress_conditioned_rasl=False,
        progress_probe_batches=0,
    )
    restored = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        PackedTokenStream(SequenceTextSource(TRAINING_DOCUMENTS), tokenizer),
        disabled_config,
        retained(tokenizer, GUARD_DOCUMENTS),
        progress_probe_batches=(),
    )
    restored.load_checkpoint(legacy)

    assert restored.state.step == source.state.step
    assert restored.state.tokens_seen == source.state.tokens_seen
    assert restored.learning_progress is None
    assert restored.pc_rasl is None
    assert restored.state.last_progress_observation_step == 0
    assert restored.state.progress_observations == 0
    assert restored.state.pc_rasl_updates_due == 0
    assert restored.state.pc_rasl_trajectories_captured == 0
    assert restored.state.pc_rasl_replay_updates == 0

    def forbidden_pc_rasl_path(*_args, **_kwargs):
        raise AssertionError("disabled PC-RASL path was executed")

    restored._prepare_pc_rasl_gradients = forbidden_pc_rasl_path
    restored._capture_pc_rasl_trajectory = forbidden_pc_rasl_path
    restored._merge_pc_rasl_gradients = forbidden_pc_rasl_path
    restored.train(maximum_steps=1)
    assert restored.state.step == source.state.step + 1
