"""End-to-end authority tests for sampled CSTM execution.

These tests intentionally inspect the live auxiliary-gradient ledger rather
than inferring traversal from loss values.  A sampled estimator is correct only
if its execution boundary, gradient destinations, counters, and reported
physical work all agree.
"""

from __future__ import annotations

from dataclasses import replace
from math import isfinite

import pytest
import torch

import mrrn.cognitive_training as cognitive_training
from mrrn.config import CognitiveConfig, MRCRAConfig, MRRNConfig
from mrrn.cognitive_training import MRCRANextTokenTrainer, MRCRATrainingConfig
from mrrn.cstm_schedule import deterministic_cstm_sample
from mrrn.language import MRCRALanguageModel
from mrrn.lm_training import ByteTextTokenizer, PackedTokenStream, SequenceTextSource
from mrrn.optimization import gradient_subsystem, merge_auxiliary_gradients


def _model_config() -> MRCRAConfig:
    carrier = MRRNConfig(
        input_dim=8,
        model_dim=8,
        output_dim=257,
        layers=1,
        scales=2,
        heads=2,
        modes=2,
        mimo_rank=1,
        attention_window=2,
        attention_query_tile_size=2,
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


def _training_config(path, *, duty: float = 1.0) -> MRCRATrainingConfig:
    return MRCRATrainingConfig(
        output_dir=str(path),
        total_tokens=32,
        context_length=32,
        execution_chunk_size=4,
        tbptt_length=8,
        vocabulary_tile_size=32,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_tokens=8,
        integrated_cognitive_path=True,
        document_static_batching=True,
        cognitive_stride=2,
        cstm_enabled=True,
        cstm_execution="sampled",
        cstm_weight=0.04,
        cstm_warmup_tokens=0,
        cstm_ramp_tokens=1,
        cstm_sampling_duty_cycle=duty,
        cstm_target_participation_budget=4,
        trackio_enabled=False,
        show_dashboard=False,
        spectral_dashboard=False,
        phase_transition_telemetry=False,
        activation_calibration=False,
        checkpoint_interval=100,
    )


def _trainer(path, *, duty: float = 1.0) -> MRCRANextTokenTrainer:
    tokenizer = ByteTextTokenizer()
    return MRCRANextTokenTrainer(
        MRCRALanguageModel(_model_config()),
        tokenizer,
        PackedTokenStream(
            SequenceTextSource(
                (
                    "abcdefghijklmnopqrstuvwxyz0123456789 "
                    "causal spectral execution authority",
                )
            ),
            tokenizer,
        ),
        _training_config(path, duty=duty),
    )


def test_predictor_and_substrate_authorities_have_exact_gradient_destinations(
    tmp_path,
):
    trainer = _trainer(tmp_path / "active")
    batch = trainer.train_stream.next_batch(1, 32)
    tokens_before = trainer.state.tokens_seen
    trainer.optimizer.zero_grad(set_to_none=True)
    metrics = trainer._run_context(batch, gradient_divisor=1)

    names = tuple(trainer._cstm_auxiliary_gradients)
    assert any(name.startswith("cstm_predictor.") for name in names)
    assert any(name.startswith("cognitive.carrier.") for name in names)
    assert any(
        name.startswith("cognitive.")
        and not name.startswith("cognitive.carrier.")
        for name in names
    )
    discovered = set(
        trainer._cstm_reachable_parameter_names["predictor"]
        + trainer._cstm_reachable_parameter_names["substrate"]
    )
    assert set(names) == discovered
    assert "token_embedding.weight" in discovered
    assert "output_bias" not in discovered
    assert trainer.runtime["cstm_gradient_registry_substrate_count"] > 0
    assert len(trainer.runtime["cstm_gradient_registry_digest"]) == 64
    assert metrics["cstm/substrate_vjp_count"] == 1
    assert metrics["cstm/substrate_vjp_count"] <= metrics[
        "cstm/max_substrate_vjps"
    ]
    assert metrics["cstm/predictor_update"] == 1
    assert metrics["cstm/substrate_update"] == 1
    assert trainer.state.tokens_seen == tokens_before
    assert metrics["train/valid_targets"] <= batch.token_count
    assert metrics["cstm/actual_target_views"] > 0
    assert metrics["cstm/estimated_dense_target_views"] >= metrics[
        "cstm/actual_target_views"
    ]
    assert metrics["cstm/actual_token_participations"] <= metrics[
        "cstm/target_participation_budget"
    ]


def test_off_duty_execution_has_predictor_only_adjoint_and_no_substrate_vjp(
    tmp_path, monkeypatch,
):
    original = cognitive_training.deterministic_cstm_sample

    def force_off_duty(*args, duty_probability, **kwargs):
        return original(
            *args,
            duty_probability=duty_probability,
            uniform_override=(0.999, 0.0),
            **kwargs,
        )

    monkeypatch.setattr(
        cognitive_training, "deterministic_cstm_sample", force_off_duty
    )
    trainer = _trainer(tmp_path / "off-duty", duty=0.25)
    metrics = trainer._run_context(
        trainer.train_stream.next_batch(1, 32),
        gradient_divisor=1,
    )

    assert metrics["cstm/predictor_update"] == 1
    assert metrics["cstm/substrate_update"] == 0
    assert metrics["cstm/substrate_vjp_count"] == 0
    assert metrics["cstm/substrate_backward_seconds"] == 0
    assert trainer._cstm_auxiliary_gradients
    assert all(
        name.startswith("cstm_predictor.")
        for name in trainer._cstm_auxiliary_gradients
    )


def test_conflict_projection_and_caps_are_enforced_on_live_parameter_grads():
    model = MRCRALanguageModel(_model_config())
    name, parameter = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if gradient_subsystem(name) == "carrier"
    )
    parameter.grad = torch.ones_like(parameter)
    auxiliary = -2.0 * torch.ones_like(parameter)
    task_before = parameter.grad.detach().clone()
    report = merge_auxiliary_gradients(
        model,
        {name: auxiliary},
        {"carrier": 0.10},
    )

    contribution = parameter.grad - task_before
    assert report.conflicting_subsystems == ("carrier",)
    assert float((task_before.float() * contribution.float()).sum()) >= -1e-7
    assert report.subsystem_scales["carrier"] <= 1.00001
    allowed = 0.10 * task_before.float().norm()
    assert contribution.float().norm() <= allowed + 1e-6


def test_zero_obligation_schedule_is_finite_and_performs_zero_auxiliary_work():
    decision = deterministic_cstm_sample(
        (),
        duty_probability=0.25,
        seed=23,
        optimizer_step=7,
        target_digest="0" * 64,
    )
    assert not decision.active
    assert decision.obligation_count == 0
    assert decision.dense_weight == 0
    assert decision.inclusion_probability == 0
    assert decision.inverse_probability == 0
    assert all(
        isfinite(value)
        for value in (
            decision.duty_probability,
            decision.conditional_probability,
            decision.inclusion_probability,
            decision.inverse_probability,
            decision.dense_weight,
        )
    )
