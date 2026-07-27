"""Checkpoint, migration, and fail-closed tests for sampled CSTM authority."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import io

import pytest
import torch

from mrrn.config import CognitiveConfig, MRCRAConfig, MRRNConfig
from mrrn.cognitive_training import MRCRANextTokenTrainer, MRCRATrainingConfig
from mrrn.language import MRCRALanguageModel
from mrrn.lm_training import ByteTextTokenizer, PackedTokenStream, SequenceTextSource


DOCUMENTS = (
    "abcdefghijklmnopqrstuvwxyz0123456789 restart exact spectral schedule",
    "a second independent causal document for checkpoint continuity",
)


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


def _config(path, **changes) -> MRCRATrainingConfig:
    base = MRCRATrainingConfig(
        output_dir=str(path),
        total_tokens=160,
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
        cstm_sampling_duty_cycle=0.5,
        cstm_target_participation_budget=4,
        trackio_enabled=False,
        show_dashboard=False,
        spectral_dashboard=False,
        phase_transition_telemetry=False,
        activation_calibration=False,
        device="cpu",
        checkpoint_interval=100,
    )
    return replace(base, **changes)


def _trainer(path, *, initial=None, **changes) -> MRCRANextTokenTrainer:
    tokenizer = ByteTextTokenizer()
    model = MRCRALanguageModel(_model_config())
    if initial is not None:
        model.load_state_dict(initial)
    return MRCRANextTokenTrainer(
        model,
        tokenizer,
        PackedTokenStream(SequenceTextSource(DOCUMENTS), tokenizer),
        _config(path, **changes),
    )


def _semantic_digest(value) -> str:
    """Hash a checkpoint tree independent of zip timestamps and file names."""

    digest = sha256()

    def visit(item) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for nested in item:
                visit(nested)
        elif isinstance(item, bytes):
            digest.update(b"bytes")
            digest.update(item)
        else:
            digest.update(type(item).__name__.encode())
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def _checkpoint_payload(trainer: MRCRANextTokenTrainer):
    return torch.load(trainer.save_checkpoint(), weights_only=True)


def test_resume_on_both_sides_of_duty_step_is_schedule_and_gradient_exact(
    tmp_path,
):
    torch.manual_seed(771)
    seed_model = MRCRALanguageModel(_model_config())
    initial = {
        name: value.detach().clone()
        for name, value in seed_model.state_dict().items()
    }

    reference = _trainer(tmp_path / "reference", initial=initial)
    reference_metrics = []
    reference_payloads = []
    for _ in range(5):
        reference.train(maximum_steps=1)
        reference_metrics.append(deepcopy(reference.last_step_metrics))
        reference_payloads.append(_checkpoint_payload(reference))
    active = [
        index
        for index, metrics in enumerate(reference_metrics)
        if metrics["cstm/substrate_update"] == 1
    ]
    assert active, "fixture must include at least one sampled substrate duty step"
    duty_index = active[0]

    for interruption_index in sorted(
        {max(0, duty_index - 1), duty_index}
    ):
        interrupted = _trainer(
            tmp_path / f"interrupt-{interruption_index}",
            initial=initial,
        )
        for _ in range(interruption_index + 1):
            interrupted.train(maximum_steps=1)
        checkpoint = interrupted.save_checkpoint()
        resumed = _trainer(tmp_path / f"interrupt-{interruption_index}")
        resumed.load_checkpoint(checkpoint)
        resumed.train(maximum_steps=1)

        expected_index = interruption_index + 1
        expected = reference_metrics[expected_index]
        actual = resumed.last_step_metrics
        for key in (
            "cstm/substrate_update",
            "cstm/predictor_update",
            "cstm/selected_invocation",
            "cstm/selected_scale",
            "cstm/sampling_inclusion_probability",
            "cstm/substrate_vjp_count",
        ):
            assert actual[key] == expected[key]
        for name, parameter in resumed.model.state_dict().items():
            torch.testing.assert_close(
                parameter,
                reference_payloads[expected_index]["model"][name],
                atol=1e-6,
                rtol=1e-6,
            )
        assert resumed.cstm_coverage.state_dict() == reference_payloads[
            expected_index
        ]["cstm_sampling"]
        resumed_payload = _checkpoint_payload(resumed)
        # Wall-clock accumulation is observational and necessarily differs
        # across an interrupted process.  Every optimization-authoritative
        # training-state field must remain bit exact.
        resumed_training_state = dict(resumed_payload["training_state"])
        reference_training_state = dict(
            reference_payloads[expected_index]["training_state"]
        )
        resumed_training_state.pop("elapsed_seconds")
        reference_training_state.pop("elapsed_seconds")
        assert _semantic_digest(resumed_training_state) == _semantic_digest(
            reference_training_state
        )
        for key in (
            "model",
            "optimizer",
            "scheduler",
            "train_stream",
            "cstm_sampling",
            "torch_rng",
        ):
            assert _semantic_digest(resumed_payload[key]) == _semantic_digest(
                reference_payloads[expected_index][key]
            ), key


def test_format15_defaults_to_legacy_dense_and_sampled_upgrade_is_explicit(
    tmp_path,
):
    source = _trainer(
        tmp_path / "format15-source",
        cstm_execution="legacy_dense",
        cstm_sampling_duty_cycle=1.0,
    )
    source.train(maximum_steps=1)
    payload = _checkpoint_payload(source)
    payload["format_version"] = 15
    payload["identity"] = source._legacy_identity()
    legacy = tmp_path / "legacy-format15.pt"
    torch.save(payload, legacy)

    default_resume = _trainer(
        tmp_path / "format15-default",
        cstm_execution="legacy_dense",
        cstm_sampling_duty_cycle=1.0,
    )
    default_resume.load_checkpoint(legacy)
    assert default_resume.config.cstm_execution == "legacy_dense"
    assert len(default_resume.execution_policy_history) == 1
    assert "migrated legacy format-15" in default_resume.execution_policy_history[
        0
    ]["reason"]

    forbidden = _trainer(
        tmp_path / "format15-forbidden",
        cstm_execution="sampled",
        allow_cstm_execution_upgrade=False,
    )
    with pytest.raises(ValueError, match="explicit"):
        forbidden.load_checkpoint(legacy)

    upgraded = _trainer(
        tmp_path / "format15-upgraded",
        cstm_execution="sampled",
        allow_cstm_execution_upgrade=True,
    )
    upgraded.load_checkpoint(legacy)
    assert upgraded.config.cstm_execution == "sampled"
    assert len(upgraded.execution_policy_history) == 2
    transition = upgraded.execution_policy_history[-1]
    assert "explicit format-15 CSTM estimator upgrade" in transition["reason"]
    assert transition["cstm_execution_transition"] == {
        "from": "legacy_dense",
        "to": "sampled",
    }
    assert upgraded.cstm_coverage.state_dict()["predictor_updates"] == 0
    assert upgraded.state.step == source.state.step
    assert _semantic_digest(upgraded.model.state_dict()) == _semantic_digest(
        source.model.state_dict()
    )
    assert _semantic_digest(upgraded.optimizer.state_dict()) == _semantic_digest(
        source.optimizer.state_dict()
    )
    assert _semantic_digest(upgraded.scheduler.state_dict()) == _semantic_digest(
        source.scheduler.state_dict()
    )
    assert upgraded.train_stream.state_dict() == source.train_stream.state_dict()


@pytest.mark.parametrize(
    "mutation, message",
    (
        (
            lambda payload: payload["cstm_sampling"].update(
                schema_version=999
            ),
            "coverage",
        ),
        (
            lambda payload: payload.update(cstm_sampling=None),
            "coverage",
        ),
        (
            lambda payload: payload["cstm_sampling"].update(
                last_obligation_digest="not-a-valid-digest"
            ),
            "coverage",
        ),
        (
            lambda payload: payload["cstm_gradient_registry"].update(
                digest="0" * 64
            ),
            "gradient registry",
        ),
        (
            lambda payload: payload["cstm_gradient_registry"][
                "substrate"
            ].append("output_bias"),
            "gradient registry",
        ),
    ),
)
def test_corrupt_sampled_schedule_state_fails_closed(
    tmp_path, mutation, message,
):
    source = _trainer(tmp_path / "corruption-source")
    source.train(maximum_steps=1)
    payload = _checkpoint_payload(source)
    mutation(payload)
    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    corrupt = tmp_path / f"corrupt-{sha256(stream.getvalue()).hexdigest()}.pt"
    torch.save(payload, corrupt)

    restored = _trainer(tmp_path / "corruption-restored")
    with pytest.raises(ValueError, match=message):
        restored.load_checkpoint(corrupt)
