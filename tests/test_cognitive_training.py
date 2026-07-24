from dataclasses import replace
import json

import pytest
import torch
from torch.nn import functional as F

import mrrn.cognitive_training as cognitive_training
from mrrn.cognitive_training import (
    MRCRANextTokenTrainer, MRCRATrainingConfig,
    _concatenate_event_phase_logits, _diagnostic_snapshot_due,
    event_phase_metrics, exact_fused_cross_entropy, exact_tiled_cross_entropy,
)
from mrrn.controller import OperationalSchemaState
from mrrn.config import CognitiveConfig, MRCRAConfig, MRRNConfig
from mrrn.language import MRCRALanguageModel
from mrrn.lm_training import (
    ByteTextTokenizer, PackedTokenStream, SequenceTextSource,
    build_evaluation_batches,
)
from mrrn.runtime_validation import defer_runtime_validation, validate_dataclass_tree


def test_trackio_diagnostic_snapshot_publishes_at_step_one_then_uses_interval():
    assert _diagnostic_snapshot_due(1, 25)
    assert _diagnostic_snapshot_due(9, 25, -1)
    assert not _diagnostic_snapshot_due(2, 25, 1)
    assert not _diagnostic_snapshot_due(9, 25, 1)
    assert _diagnostic_snapshot_due(25, 25, 1)
    assert _diagnostic_snapshot_due(50, 25, 25)
    assert not _diagnostic_snapshot_due(25, 25, 25)


def test_failed_diagnostic_snapshot_records_attempt_and_does_not_retry_same_step(
    tmp_path, monkeypatch,
):
    tokenizer = ByteTextTokenizer()
    trainer = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        PackedTokenStream(SequenceTextSource(("diagnostic cadence",)), tokenizer),
        replace(
            training_config(tmp_path / "snapshot-cadence"),
            spectral_dashboard=True,
            spectral_snapshot_interval=25,
        ),
    )
    trainer.state.step = 1

    def fail_snapshot(*args, **kwargs):
        raise ValueError("diagnostic fixture failure")

    monkeypatch.setattr(
        "mrrn.visualization.model_spectral_evidence", fail_snapshot,
    )

    class Reporter:
        def __init__(self):
            self.alerts = []

        def alert(self, title, text, *, level, step):
            self.alerts.append((title, text, level, step))

    reporter = Reporter()
    trainer._publish_cognitive_snapshot(reporter)
    trainer._publish_cognitive_snapshot(reporter)

    assert trainer._last_snapshot_step == -1
    assert trainer._last_snapshot_attempt_step == 1
    assert len(reporter.alerts) == 1
    assert reporter.alerts[0][0] == "MRCRA diagnostic snapshot failed"
    assert not _diagnostic_snapshot_due(
        2, trainer.config.spectral_snapshot_interval,
        trainer._last_snapshot_attempt_step,
    )


def test_event_phase_metrics_match_known_probabilities_and_threshold_distance():
    probability = torch.tensor([0.10, 0.30, 0.50, 0.90], dtype=torch.float64)
    logits = torch.logit(probability)
    metrics = event_phase_metrics(logits, logits - 0.25)
    assert metrics["architecture/event_proposal_probability_mean"] == pytest.approx(
        probability.mean().item(), abs=1e-7
    )
    assert metrics["architecture/event_proposal_probability_max"] == pytest.approx(
        0.9, abs=1e-7
    )
    assert metrics["architecture/event_proposal_fraction_ge_0p25"] == 0.75
    assert metrics["architecture/event_proposal_fraction_ge_0p35"] == 0.5
    assert metrics["architecture/event_proposal_fraction_ge_0p45"] == 0.5
    assert metrics["architecture/event_proposal_fraction_ge_0p50"] == 0.5
    assert metrics["architecture/event_phase_distance_to_threshold"] == pytest.approx(
        -torch.logit(torch.tensor(0.9)).item(), abs=1e-6
    )
    with pytest.raises(FloatingPointError):
        event_phase_metrics(torch.tensor([float("nan")]), torch.zeros(1))


def test_event_phase_telemetry_concatenates_variable_length_document_spans():
    proposals, endings = _concatenate_event_phase_logits(
        (
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            torch.tensor([[5.0, 6.0, 7.0]]),
        ),
        (
            torch.tensor([[-1.0, -2.0, -3.0, -4.0]]),
            torch.tensor([[-5.0, -6.0, -7.0]]),
        ),
    )
    torch.testing.assert_close(
        proposals, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    )
    torch.testing.assert_close(
        endings, torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0])
    )
    metrics = event_phase_metrics(proposals, endings)
    assert metrics["architecture/event_proposal_logit_max"] == 7.0
    with pytest.raises(ValueError, match="aligned"):
        _concatenate_event_phase_logits(
            (torch.zeros(1, 4),), (torch.zeros(1, 3),)
        )


def tiny_config(vocabulary_size=257):
    carrier = MRRNConfig(
        input_dim=8, model_dim=8, output_dim=vocabulary_size, layers=1, scales=2,
        heads=2, modes=2, mimo_rank=1, attention_window=2,
        attention_query_tile_size=2, retrieved_items=1, memory_capacity=4,
        mixer_expansion=1.5, width_growth_cap=1, mode_growth_cap=1,
        width_multiple=2, spectral_modes=2, spectral_basis_order=2,
        spectral_triads_per_mode=1, enable_global_head=False,
        relational_branch=True, relational_context_dim=8,
        activation_checkpointing=True,
    )
    cognition = CognitiveConfig(
        workspace_dim=8, provenance_features=4, uncertainty_channels=8,
        relation_heads=2, relation_modes=2, relation_adapter_rank=2,
        goal_slots=1, goal_constraint_dim=2, system_action_channels=2,
        calibration_regimes=2, active_event_capacity=4, pair_edge_capacity=8,
        hyperedge_capacity=2, maximum_hyperedge_arity=3, graph_neighbors=1,
        global_workspace_slots=2, hypothesis_slots=1, maximum_hypothesis_slots=2,
        maximum_cognitive_steps=1, event_chunk_size=2,
        event_proposals_per_chunk=1, recent_candidates=2,
        landmark_candidates=1, episodic_candidates=1, semantic_candidates=1,
        episodic_memory_capacity=4, semantic_memory_capacity=2,
        associative_depth=1, associative_budget=1, world_model_horizons=(1,),
    )
    return MRCRAConfig(
        carrier, cognition, actor_parameter_minimum=1,
        actor_parameter_maximum=10_000_000,
    )


def training_config(path):
    return MRCRATrainingConfig(
        output_dir=str(path), total_tokens=16, context_length=8,
        execution_chunk_size=2, tbptt_length=4, vocabulary_tile_size=32,
        micro_batch_size=1, gradient_accumulation_steps=1,
        warmup_tokens=8, trackio_enabled=False, show_dashboard=False,
        spectral_dashboard=False, checkpoint_interval=10,
    )


def test_exact_tiled_cross_entropy_matches_dense_loss_and_gradients():
    torch.manual_seed(229)
    hidden_a = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    weight_a = torch.randn(11, 5, dtype=torch.float64, requires_grad=True)
    bias_a = torch.randn(11, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 11, (2, 4))
    byte_lengths = torch.randint(0, 3, (2, 4), dtype=torch.int64)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    dense = F.cross_entropy(F.linear(hidden_a, weight_a, bias_a)[mask], labels[mask])
    dense.backward()

    hidden_b = hidden_a.detach().clone().requires_grad_(True)
    weight_b = weight_a.detach().clone().requires_grad_(True)
    bias_b = bias_a.detach().clone().requires_grad_(True)
    tiled = exact_tiled_cross_entropy(
        hidden_b, labels, byte_lengths, mask, weight_b, bias_b,
        vocabulary_tile_size=3, checkpoint_tiles=True,
    )
    tiled.loss.backward()
    torch.testing.assert_close(tiled.loss, dense, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(hidden_b.grad, hidden_a.grad, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(weight_b.grad, weight_a.grad, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(bias_b.grad, bias_a.grad, atol=1e-11, rtol=1e-11)


def test_exact_fused_cross_entropy_matches_dense_loss_and_gradients():
    torch.manual_seed(230)
    hidden_a = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    weight_a = torch.randn(11, 5, dtype=torch.float64, requires_grad=True)
    bias_a = torch.randn(11, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 11, (2, 4))
    byte_lengths = torch.randint(0, 3, (2, 4), dtype=torch.int64)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    dense = F.cross_entropy(F.linear(hidden_a, weight_a, bias_a)[mask], labels[mask])
    dense.backward()

    hidden_b = hidden_a.detach().clone().requires_grad_(True)
    weight_b = weight_a.detach().clone().requires_grad_(True)
    bias_b = bias_a.detach().clone().requires_grad_(True)
    fused = exact_fused_cross_entropy(
        hidden_b, labels, byte_lengths, mask, weight_b, bias_b,
    )
    fused.loss.backward()
    torch.testing.assert_close(fused.loss, dense, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(hidden_b.grad, hidden_a.grad, atol=1e-15, rtol=1e-15)
    torch.testing.assert_close(weight_b.grad, weight_a.grad, atol=1e-15, rtol=1e-15)
    torch.testing.assert_close(bias_b.grad, bias_a.grad, atol=1e-15, rtol=1e-15)


def test_deferred_runtime_validation_checks_completed_state_at_boundary():
    with torch.no_grad(), defer_runtime_validation():
        state = OperationalSchemaState(
            torch.tensor([[-0.25, 1.25]]),
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_dataclass_tree(state)


def test_projection_free_cognitive_forward_reconstructs_exact_logits():
    torch.manual_seed(233)
    model = MRCRALanguageModel(tiny_config()).double().eval()
    tokens = torch.randint(0, model.vocabulary_size, (1, 5))
    dense = model(tokens)
    latent = model(tokens, project_output=False)
    assert latent.logits.shape == (1, 5, 0)
    torch.testing.assert_close(
        model.project_output(latent.cognitive.output_latent), dense.logits,
        atol=1e-10, rtol=1e-10,
    )


def test_stateful_chunked_trainer_updates_and_checkpoint_resume_is_exact(tmp_path):
    documents = ("alpha beta gamma", "delta epsilon", "zeta eta theta")
    tokenizer = ByteTextTokenizer()
    torch.manual_seed(239)
    reference_model = MRCRALanguageModel(tiny_config())
    initial = {name: value.detach().clone() for name, value in reference_model.state_dict().items()}
    reference = MRCRANextTokenTrainer(
        reference_model, tokenizer,
        PackedTokenStream(SequenceTextSource(documents), tokenizer),
        training_config(tmp_path / "reference"),
    )
    reference.train(maximum_steps=2)
    assert reference._last_runtime is not None
    assert reference._last_runtime.clocks.optimizer == 2

    split_model = MRCRALanguageModel(tiny_config())
    split_model.load_state_dict(initial)
    split = MRCRANextTokenTrainer(
        split_model, tokenizer,
        PackedTokenStream(SequenceTextSource(documents), tokenizer),
        training_config(tmp_path / "split"),
    )
    split.train(maximum_steps=1)
    checkpoint_path = split.save_checkpoint()

    restored_model = MRCRALanguageModel(tiny_config())
    restored = MRCRANextTokenTrainer(
        restored_model, tokenizer,
        PackedTokenStream(SequenceTextSource(documents), tokenizer),
        training_config(tmp_path / "split"),
    )
    restored.load_checkpoint(checkpoint_path)
    restored.train(maximum_steps=1)
    assert restored.state.step == reference.state.step == 2
    assert restored.state.tokens_seen == reference.state.tokens_seen == 16
    assert restored._last_runtime is not None
    assert restored._last_runtime.clocks.optimizer == 2
    for expected, actual in zip(
        reference.model.state_dict().values(), restored.model.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_bounded_training_calls_preserve_prefetch_and_rng_continuity(tmp_path):
    documents = ("alpha beta gamma", "delta epsilon", "zeta eta theta")
    tokenizer = ByteTextTokenizer()
    torch.manual_seed(241)
    reference_model = MRCRALanguageModel(tiny_config())
    initial = {
        name: value.detach().clone()
        for name, value in reference_model.state_dict().items()
    }
    reference = MRCRANextTokenTrainer(
        reference_model, tokenizer,
        PackedTokenStream(SequenceTextSource(documents), tokenizer),
        training_config(tmp_path / "continuous"),
    )
    reference.train(maximum_steps=2)

    resumed_model = MRCRALanguageModel(tiny_config())
    resumed_model.load_state_dict(initial)
    resumed = MRCRANextTokenTrainer(
        resumed_model, tokenizer,
        PackedTokenStream(SequenceTextSource(documents), tokenizer),
        training_config(tmp_path / "bounded"),
    )
    resumed.train(maximum_steps=1)
    assert resumed._prefetch_executor is not None
    resumed.train(maximum_steps=1)
    assert resumed._prefetch_executor is None
    assert resumed.state.step == reference.state.step
    assert resumed.state.tokens_seen == reference.state.tokens_seen
    assert resumed.state.valid_targets_seen == reference.state.valid_targets_seen
    assert resumed.state.bytes_seen == reference.state.bytes_seen
    for expected, actual in zip(
        reference.model.state_dict().values(),
        resumed.model.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_integrated_phase_slope_state_is_checkpoint_resume_exact(tmp_path):
    documents = ("alpha beta gamma", "delta epsilon", "zeta eta theta")
    tokenizer = ByteTextTokenizer()
    config_reference = replace(
        training_config(tmp_path / "phase-reference"),
        integrated_cognitive_path=True,
        cognitive_stride=2,
        progress_interval_tokens=8,
    )
    torch.manual_seed(243)
    initial_model = MRCRALanguageModel(tiny_config())
    initial = {
        name: value.detach().clone()
        for name, value in initial_model.state_dict().items()
    }
    reference_model = MRCRALanguageModel(tiny_config())
    reference_model.load_state_dict(initial)
    reference = MRCRANextTokenTrainer(
        reference_model, tokenizer,
        PackedTokenStream(SequenceTextSource(documents), tokenizer),
        config_reference,
    )
    reference.train(maximum_steps=2)

    config_split = replace(
        config_reference, output_dir=str(tmp_path / "phase-split")
    )
    split_model = MRCRALanguageModel(tiny_config())
    split_model.load_state_dict(initial)
    split = MRCRANextTokenTrainer(
        split_model, tokenizer,
        PackedTokenStream(SequenceTextSource(documents), tokenizer),
        config_split,
    )
    split.train(maximum_steps=1)
    checkpoint = split.save_checkpoint()
    restored = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()), tokenizer,
        PackedTokenStream(SequenceTextSource(documents), tokenizer),
        config_split,
    )
    restored.load_checkpoint(checkpoint)
    restored.train(maximum_steps=1)
    assert restored.state.event_proposal_observations == 2
    assert (
        restored.state.last_event_proposal_logit_max
        == reference.state.last_event_proposal_logit_max
    )
    assert (
        restored.state.event_proposal_logit_slope_ema
        == reference.state.event_proposal_logit_slope_ema
    )


def test_v3_training_checkpoint_migrates_append_only_action_rows_and_foundation_state(tmp_path):
    tokenizer = ByteTextTokenizer()
    source = SequenceTextSource(("alpha beta", "gamma delta"))
    model = MRCRALanguageModel(tiny_config())
    trainer = MRCRANextTokenTrainer(
        model, tokenizer, PackedTokenStream(source, tokenizer),
        training_config(tmp_path / "legacy"),
    )
    trainer.train(maximum_steps=1)
    current_path = trainer.save_checkpoint()
    payload = torch.load(current_path, weights_only=True)
    payload["format_version"] = 3
    cognitive_identity = payload["identity"]["model_config"]["cognitive"]
    for name in (
        "reconstruction_capacity", "action_candidate_capacity", "action_argument_dim",
        "evidence_request_capacity", "external_artifact_capacity",
        "external_artifact_digest_width", "viability_channels", "metacognitive_capacity",
        "enable_conditional_reconstruction", "enable_abstraction_validity_control",
        "enable_post_deliberation_action_selection", "enable_multi_hypothesis_planning",
        "enable_agent_session_loop", "enable_viability_gate",
        "enable_integrated_invariant_discovery", "enable_persistent_session_training",
    ):
        cognitive_identity.pop(name)
    for name in tuple(payload["model"]):
        if name.endswith("cognitive.controller.action_head.weight"):
            payload["model"][name] = payload["model"][name][:21]
        elif name.endswith("cognitive.controller.action_head.bias"):
            payload["model"][name] = payload["model"][name][:21]
    current_optimizer = trainer.optimizer.state_dict()
    names = {id(parameter): name for name, parameter in trainer.model.named_parameters()}
    for live_group, serialized_group in zip(
        trainer.optimizer.param_groups, current_optimizer["param_groups"], strict=True,
    ):
        for parameter, parameter_id in zip(
            live_group["params"], serialized_group["params"], strict=True,
        ):
            if "cognitive.controller.action_head" not in names[id(parameter)]:
                continue
            for state_name, state_value in tuple(payload["optimizer"]["state"].get(parameter_id, {}).items()):
                if isinstance(state_value, torch.Tensor) and state_value.ndim and state_value.shape[0] == 31:
                    payload["optimizer"]["state"][parameter_id][state_name] = state_value[:21]
    for name in (
        "reconstructions", "abstraction_validity", "action_candidates", "viability",
        "evidence_requests", "external_artifacts", "metacognition", "boundary_context",
    ):
        payload["last_runtime"].pop(name)
    legacy_path = tmp_path / "legacy-v3.pt"
    torch.save(payload, legacy_path)

    restored = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()), tokenizer,
        PackedTokenStream(SequenceTextSource(("alpha beta", "gamma delta")), tokenizer),
        training_config(tmp_path / "legacy"),
    )
    restored.load_checkpoint(legacy_path)
    assert restored._last_runtime is not None
    assert not restored._last_runtime.reconstructions.active.any()
    assert restored.model.cognitive.controller.action_head.weight.shape[0] == 31
    assert torch.count_nonzero(
        restored.model.cognitive.controller.action_head.weight[21:]
    ) == 0


def test_packer_preserves_original_document_uri_and_boundary_metadata():
    tokenizer = ByteTextTokenizer()
    stream = PackedTokenStream(SequenceTextSource(("a", "bc")), tokenizer)
    batch = stream.next_batch(1, 4)
    declarations = batch.external_source_uris[0]
    assert declarations[0].endswith("local-0")
    assert declarations[1].endswith("local-1")
    assert not bool(batch.loss_mask[0, 1])
    assert int(batch.boundary_classes[0, 2]) != 0


def _retained(tokenizer, text="retained evaluation document"):
    return build_evaluation_batches(
        PackedTokenStream(SequenceTextSource((text,)), tokenizer),
        count=1, batch_size=1, sequence_length=8,
    )


def test_retained_evaluation_is_exact_finite_and_side_effect_free(tmp_path):
    tokenizer = ByteTextTokenizer()
    config = replace(
        training_config(tmp_path / "evaluation"),
        evaluation_interval=1, evaluation_batches=1, require_evaluation=True,
    )
    model = MRCRALanguageModel(tiny_config())
    trainer = MRCRANextTokenTrainer(
        model, tokenizer,
        PackedTokenStream(SequenceTextSource(("training document",)), tokenizer),
        config, _retained(tokenizer),
    )
    parameters = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    rng = torch.random.get_rng_state().clone()
    metrics = trainer.evaluate()
    assert model.training
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert trainer._last_runtime is None and trainer._last_ledger is None
    assert metrics["eval/valid_targets"] > 0
    assert metrics["eval/utf8_bytes"] > 0
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, parameters[name], atol=0, rtol=0)


def test_integrated_stage1_path_trains_cognition_and_uses_matching_evaluation(tmp_path):
    tokenizer = ByteTextTokenizer()
    config = replace(
        training_config(tmp_path / "integrated-multirate"),
        integrated_cognitive_path=True,
        cognitive_stride=2,
        progress_interval_tokens=8,
        evaluation_interval=1,
        evaluation_batches=1,
        require_evaluation=True,
    )
    model = MRCRALanguageModel(tiny_config())
    carrier_before = model.token_embedding.weight.detach().clone()
    cognitive_names = (
        "cognitive.output_context_adapter.weight",
        "cognitive.controller.action_head.weight",
        "cognitive.operational_schemas.logits.weight",
    )
    cognitive_before = {
        name: dict(model.named_parameters())[name].detach().clone()
        for name in cognitive_names
    }
    trainer = MRCRANextTokenTrainer(
        model, tokenizer,
        PackedTokenStream(SequenceTextSource(("training document",)), tokenizer),
        config, _retained(tokenizer),
    )
    trainer.train(maximum_steps=1)
    assert not torch.equal(model.token_embedding.weight.detach().cpu(), carrier_before)
    for name in cognitive_names:
        assert not torch.equal(
            dict(model.named_parameters())[name].detach().cpu(), cognitive_before[name]
        ), name
    assert trainer._last_runtime is not None and trainer._last_ledger is not None
    assert trainer.state.last_evaluation_metrics["eval/integrated_cognitive_path"] == 1.0
    ablation = trainer.state.last_evaluation_metrics
    assert ablation["eval/phase_ablation/hard_structure_ce_gain"] == pytest.approx(
        ablation["eval/phase_ablation/soft_only_ce_nats_per_token"]
        - ablation["eval/phase_ablation/full_ce_nats_per_token"]
    )
    assert ablation["eval/phase_ablation/soft_bridge_ce_gain"] == pytest.approx(
        ablation["eval/phase_ablation/cognition_off_ce_nats_per_token"]
        - ablation["eval/phase_ablation/soft_only_ce_nats_per_token"]
    )
    assert ablation["eval/phase_ablation/valid_targets"] > 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "integrated-multirate" / "evaluation_metrics.jsonl")
        .read_text(encoding="utf-8").splitlines()
    ]
    assert rows[-1]["metrics"]["eval/integrated_cognitive_path"] == 1.0


def test_first_hard_event_is_alerted_artifacted_and_immediately_checkpointed(
    tmp_path, monkeypatch,
):
    class FakeReporter:
        instances = []

        def __init__(self, config, run_config, *, resume):
            self.logs, self.alerts, self.traces = [], [], []
            self.resume = resume
            self.__class__.instances.append(self)

        def log(self, metrics, *, step):
            self.logs.append((step, dict(metrics)))

        def alert(self, title, text, *, level, step):
            self.alerts.append((title, text, level, step))

        def log_phase_transition_trace(self, path, *, step):
            self.traces.append((step, path))
            return 1

        def finish(self):
            return None

    monkeypatch.setattr(cognitive_training, "TrackioReporter", FakeReporter)
    tokenizer = ByteTextTokenizer()
    config = replace(
        training_config(tmp_path / "first-event"),
        integrated_cognitive_path=True,
        cognitive_stride=2,
        progress_interval_tokens=8,
        trackio_enabled=True,
        spectral_dashboard=False,
        low_clip_coefficient_threshold=0.99,
        low_clip_coefficient_patience=1,
    )
    model = MRCRALanguageModel(tiny_config())
    proposal = model.cognitive.event_extractor.proposal_network
    with torch.no_grad():
        for parameter in proposal.parameters():
            parameter.zero_()
        proposal.proposal.bias.fill_(8)
        proposal.end.bias.fill_(8)
    trainer = MRCRANextTokenTrainer(
        model, tokenizer,
        PackedTokenStream(
            SequenceTextSource(("first event training document",)), tokenizer
        ),
        config,
    )
    trainer.train(maximum_steps=1)
    trace_path = tmp_path / "first-event" / "diagnostics" / "first-hard-event.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    checkpoint = (
        tmp_path / "first-event" / "checkpoints"
        / "phase-transition-first-event-step-0000001.pt"
    )
    assert checkpoint.is_file()
    assert trace["trigger"]["proposal_probability"] > 0.99
    assert trace["state_transition"]["active_nodes"]["after"] > 0
    assert trace["gradient_transition"]["subsystems"]["event"]["before_clip"] > 0
    assert trace["phase"]["architecture/event_emitted"] > 0
    first_reporter = FakeReporter.instances[-1]
    assert first_reporter.traces == [(1, trace_path)]
    assert [alert[0] for alert in first_reporter.alerts].count(
        "First MRCRA hard event"
    ) == 1
    assert [alert[0] for alert in first_reporter.alerts].count(
        "Sustained strong gradient clipping"
    ) == 1
    assert trainer.state.first_hard_event_step == 1
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["training_state"]["first_hard_event_step"] == 1
    trainer.train(maximum_steps=1)
    assert trainer.state.first_hard_event_step == 1
    all_alerts = [
        alert for reporter in FakeReporter.instances for alert in reporter.alerts
    ]
    assert [alert[0] for alert in all_alerts].count("First MRCRA hard event") == 1


def test_integrated_multirate_path_is_restricted_to_authorized_stage1_profile(tmp_path):
    try:
        replace(
            training_config(tmp_path / "invalid-fast"),
            integrated_cognitive_path=True,
            cognitive_stride=2,
            curriculum_stage=2,
            training_profile="relational_event_pretraining",
        )
    except ValueError as error:
        assert "restricted to independent stage-1" in str(error)
    else:
        raise AssertionError("integrated multirate path accepted another training stage")


def test_evaluation_authority_and_checkpoint_identity_fail_closed(tmp_path):
    tokenizer = ByteTextTokenizer()
    with torch.no_grad():
        with_zeros = training_config(tmp_path / "invalid")
    try:
        replace(with_zeros, evaluation_interval=1)
    except ValueError as error:
        assert "enabled together" in str(error)
    else:
        raise AssertionError("partial evaluation configuration was accepted")
    try:
        replace(with_zeros, require_evaluation=True)
    except ValueError as error:
        assert "requires retained" in str(error)
    else:
        raise AssertionError("required evaluation was accepted without batches")

    config = replace(
        training_config(tmp_path / "bound"),
        evaluation_interval=1, evaluation_batches=1, require_evaluation=True,
    )
    with torch.no_grad():
        retained_a = _retained(tokenizer, "AAAAAAAA retained")
        retained_b = _retained(tokenizer, "BBBBBBBB retained")
    trainer = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()), tokenizer,
        PackedTokenStream(SequenceTextSource(("train",)), tokenizer),
        config, retained_a,
    )
    trainer.train(maximum_steps=1)
    checkpoint = trainer.save_checkpoint()
    assert trainer.evaluation_identity["sha256"] != MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()), tokenizer,
        PackedTokenStream(SequenceTextSource(("train",)), tokenizer),
        config, retained_b,
    ).evaluation_identity["sha256"]
    mismatched = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()), tokenizer,
        PackedTokenStream(SequenceTextSource(("train",)), tokenizer),
        config, retained_b,
    )
    try:
        mismatched.load_checkpoint(checkpoint)
    except ValueError as error:
        assert "contract differs" in str(error)
    else:
        raise AssertionError("checkpoint accepted a different retained split")


def test_v5_checkpoint_migrates_to_digest_bound_retained_evaluation(tmp_path):
    tokenizer = ByteTextTokenizer()
    config = replace(
        training_config(tmp_path / "v5"),
        evaluation_interval=1, evaluation_batches=1, require_evaluation=True,
    )
    retained = _retained(tokenizer)
    trainer = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()), tokenizer,
        PackedTokenStream(SequenceTextSource(("train",)), tokenizer),
        config, retained,
    )
    trainer.train(maximum_steps=1)
    payload = torch.load(trainer.save_checkpoint(), weights_only=True)
    payload["format_version"] = 5
    payload["identity"].pop("evaluation")
    for name in ("evaluation_interval", "evaluation_batches", "require_evaluation"):
        payload["identity"]["training"].pop(name)
    legacy = tmp_path / "v5.pt"
    torch.save(payload, legacy)
    restored = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()), tokenizer,
        PackedTokenStream(SequenceTextSource(("train",)), tokenizer),
        config, retained,
    )
    restored.load_checkpoint(legacy)
    assert restored.evaluation_identity["batch_count"] == 1


def test_v7_checkpoint_migrates_missing_phase_transition_contract_fields(tmp_path):
    tokenizer = ByteTextTokenizer()
    config = training_config(tmp_path / "v7")
    trainer = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()), tokenizer,
        PackedTokenStream(SequenceTextSource(("training document",)), tokenizer),
        config,
    )
    trainer.train(maximum_steps=1)
    payload = torch.load(trainer.save_checkpoint(), weights_only=True)
    payload["format_version"] = 7
    phase_fields = (
        "phase_transition_telemetry", "phase_transition_ablation",
        "phase_transition_ablation_batches", "proposal_slope_ema_decay",
        "low_clip_coefficient_threshold", "low_clip_coefficient_patience",
    )
    for name in phase_fields:
        payload["identity"]["training"].pop(name)
    for name in (
        "last_event_proposal_logit_max", "event_proposal_logit_slope_ema",
        "event_proposal_observations", "low_clip_coefficient_steps",
        "first_hard_event_step", "first_hard_event_tokens",
        "first_hard_event_checkpoint",
    ):
        payload["training_state"].pop(name)
    legacy = tmp_path / "v7.pt"
    torch.save(payload, legacy)
    restored = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()), tokenizer,
        PackedTokenStream(SequenceTextSource(("training document",)), tokenizer),
        config,
    )
    restored.load_checkpoint(legacy)
    assert restored.state.first_hard_event_step == 0
    assert restored.state.event_proposal_observations == 0
    assert restored.config.phase_transition_telemetry is True
