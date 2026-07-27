from dataclasses import replace
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

import mrrn.cognitive_training as cognitive_training
from mrrn.cognitive_training import (
    MRCRANextTokenTrainer, MRCRATrainingConfig,
    _concatenate_event_phase_logits, _diagnostic_snapshot_due,
    event_phase_metrics, exact_cut_cross_entropy,
    exact_fused_cross_entropy, exact_tiled_cross_entropy,
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
        activation_calibration=False,
    )


def test_sampled_cstm_fails_closed_without_document_major_authority(tmp_path):
    with pytest.raises(ValueError, match="sampled CSTM requires"):
        replace(
            training_config(tmp_path),
            integrated_cognitive_path=True,
            document_static_batching=False,
            cstm_enabled=True,
            cstm_execution="sampled",
            cognitive_stride=2,
        )
    reference = replace(
        training_config(tmp_path),
        integrated_cognitive_path=True,
        document_static_batching=False,
        cstm_enabled=True,
        cstm_execution="legacy_dense",
        cognitive_stride=2,
    )
    assert reference.cstm_execution == "legacy_dense"


def test_pre_optimizer_oom_replays_cached_batch_once_under_safer_policy(
    tmp_path, monkeypatch,
):
    tokenizer = ByteTextTokenizer()
    config = replace(
        training_config(tmp_path / "oom-recovery"),
        total_tokens=8,
        activation_policy="retain",
        activation_memory_reserve_bytes=1,
        cstm_enabled=False,
    )
    stream = PackedTokenStream(
        SequenceTextSource(("recoverable allocator pressure",)),
        tokenizer,
    )
    trainer = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        stream,
        config,
    )
    original = trainer._run_context
    calls = 0

    def fail_first(batch, *, gradient_divisor):
        nonlocal calls
        calls += 1
        if calls == 1:
            trainer.model.cstm_predictor.target_second_moment.fill_(999)
            trainer.cstm_coverage.predictor_updates = 7
            raise torch.OutOfMemoryError("synthetic out of memory")
        return original(batch, gradient_divisor=gradient_divisor)

    monkeypatch.setattr(trainer, "_run_context", fail_first)
    state = trainer.train(maximum_steps=1)

    assert calls == 2
    assert state.step == 1
    assert state.tokens_seen == 8
    assert trainer.activation_execution_policy.resolved == "whole_span"
    assert trainer.runtime["activation_oom_retries"] == 1
    assert trainer.last_step_metrics["execution/activation_oom_retries"] == 1
    assert len(trainer.execution_policy_history) == 2
    assert "OOM" in trainer.execution_policy_history[-1]["reason"]
    assert not bool(
        trainer.model.cstm_predictor.target_second_moment.ne(0).any()
    )
    assert trainer.cstm_coverage.predictor_updates == 0


def test_cpu_thread_calibration_uses_actual_carrier_and_preserves_state_and_rng():
    torch.manual_seed(227)
    model = MRCRALanguageModel(tiny_config())
    before_rng = torch.random.get_rng_state().clone()
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    selected, timings = cognitive_training._calibrate_cpu_thread_count(
        model, maximum_length=4,
    )
    assert selected in timings
    assert timings[selected] == min(timings.values())
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_activation_calibration_preserves_model_optimizer_rng_and_stream(
    tmp_path,
):
    tokenizer = ByteTextTokenizer()
    documents = ("activation calibration is execution-only evidence",)
    torch.manual_seed(311)
    seed_model = MRCRALanguageModel(tiny_config())
    initial = {
        name: value.detach().clone()
        for name, value in seed_model.state_dict().items()
    }

    def construct(path, *, calibrate):
        model = MRCRALanguageModel(tiny_config())
        model.load_state_dict(initial)
        stream = PackedTokenStream(
            SequenceTextSource(documents), tokenizer
        )
        before_stream = stream.state_dict()
        torch.manual_seed(971)
        expected_rng = torch.random.get_rng_state().clone()
        trainer = MRCRANextTokenTrainer(
            model,
            tokenizer,
            stream,
            replace(
                training_config(path),
                device="cpu",
                cpu_threads=1,
                cpu_interop_threads=1,
                cstm_enabled=False,
                activation_policy="retain",
                activation_calibration=calibrate,
                    activation_memory_reserve_bytes=1,
                exact_loss_backend="tiled",
            ),
        )
        assert torch.equal(torch.random.get_rng_state(), expected_rng)
        assert stream.state_dict() == before_stream
        assert all(
            parameter.grad is None
            for parameter in trainer.model.parameters()
        )
        return trainer

    uncalibrated = construct(
        tmp_path / "activation-uncalibrated", calibrate=False
    )
    calibrated = construct(
        tmp_path / "activation-calibrated", calibrate=True
    )
    for name, expected in uncalibrated.model.state_dict().items():
        torch.testing.assert_close(
            calibrated.model.state_dict()[name],
            expected,
            atol=0,
            rtol=0,
            msg=name,
        )
    assert calibrated.optimizer.state_dict() == (
        uncalibrated.optimizer.state_dict()
    )
    assert calibrated.scheduler.state_dict() == (
        uncalibrated.scheduler.state_dict()
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


@pytest.mark.parametrize(
    "implementation", ["cce_kahan_full_c", "cce_exact", "torch_compile"]
)
def test_cut_cross_entropy_adapter_preserves_exact_loss_bias_mask_and_gradients(
    monkeypatch, implementation
):
    calls = []

    def linear_cross_entropy(
        hidden, weight, labels, *, bias, reduction, impl
    ):
        calls.append((reduction, impl))
        return F.cross_entropy(
            F.linear(hidden, weight, bias),
            labels,
            reduction=reduction,
        )

    monkeypatch.setitem(
        sys.modules,
        "cut_cross_entropy",
        SimpleNamespace(linear_cross_entropy=linear_cross_entropy),
    )
    torch.manual_seed(231)
    hidden_a = torch.randn(2, 5, 7, requires_grad=True)
    weight_a = torch.randn(19, 7, requires_grad=True)
    bias_a = torch.randn(19, requires_grad=True)
    labels = torch.randint(0, 19, (2, 5))
    lengths = torch.randint(0, 4, (2, 5), dtype=torch.int64)
    mask = torch.tensor([
        [True, True, False, True, False],
        [True, False, True, True, True],
    ])
    dense = F.cross_entropy(
        F.linear(hidden_a, weight_a, bias_a)[mask], labels[mask]
    )
    dense.backward()

    hidden_b = hidden_a.detach().clone().requires_grad_(True)
    weight_b = weight_a.detach().clone().requires_grad_(True)
    bias_b = bias_a.detach().clone().requires_grad_(True)
    cce = exact_cut_cross_entropy(
        hidden_b,
        labels,
        lengths,
        mask,
        weight_b,
        bias_b,
        implementation=implementation,
    )
    cce.loss.backward()
    assert calls == [("none", implementation)]
    assert cce.token_count == int(mask.sum())
    assert cce.byte_count == int(lengths[mask].sum())
    torch.testing.assert_close(cce.loss, dense)
    torch.testing.assert_close(hidden_b.grad, hidden_a.grad)
    torch.testing.assert_close(weight_b.grad, weight_a.grad)
    torch.testing.assert_close(bias_b.grad, bias_a.grad)


def test_cut_cross_entropy_adapter_rejects_unsafe_filtered_pretraining_policy():
    with pytest.raises(ValueError, match="unsafe"):
        exact_cut_cross_entropy(
            torch.randn(1, 2, 3),
            torch.tensor([[1, 2]]),
            torch.ones(1, 2, dtype=torch.int64),
            torch.ones(1, 2, dtype=torch.bool),
            torch.randn(4, 3),
            torch.randn(4),
            implementation="cce",
        )


def test_exact_loss_backend_configuration_rejects_unknown_or_unbounded_fused():
    with pytest.raises(ValueError, match="unknown"):
        replace(training_config("unused"), exact_loss_backend="approximate")
    with pytest.raises(ValueError, match="workspace"):
        replace(
            training_config("unused"),
            exact_loss_backend="fused",
            maximum_fused_loss_bytes=0,
        )


@pytest.mark.parametrize(
    ("requested", "cce_available", "resolved"),
    [
        ("auto", True, "torch_compile"),
        ("cce_kahan_full_c", True, "torch_compile"),
        ("cce_exact", True, "torch_compile"),
        ("auto", False, "tiled"),
        ("cce_kahan_full_c", False, "tiled"),
        ("cce_exact", False, "tiled"),
    ],
)
def test_cpu_and_macos_exact_cce_authority_never_depends_on_cuda(
    tmp_path, monkeypatch, requested, cce_available, resolved
):
    monkeypatch.setattr(
        cognitive_training,
        "find_spec",
        lambda name: object()
        if name == "cut_cross_entropy" and cce_available
        else None,
    )
    tokenizer = ByteTextTokenizer()
    trainer = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        PackedTokenStream(SequenceTextSource(("portable exact authority",)), tokenizer),
        replace(
            training_config(tmp_path / f"{requested}-{cce_available}"),
            device="cpu",
            exact_loss_backend=requested,
        ),
    )
    assert trainer._exact_loss_backend == resolved
    assert trainer.runtime["requested_exact_loss_backend"] == requested
    assert trainer.runtime["exact_loss_backend"] == resolved
    assert trainer.runtime["loss_projection"] == f"{resolved}_exact_full_softmax"


def test_explicit_torch_compile_requires_the_external_cce_package(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(cognitive_training, "find_spec", lambda name: None)
    tokenizer = ByteTextTokenizer()
    with pytest.raises(RuntimeError, match="unavailable"):
        MRCRANextTokenTrainer(
            MRCRALanguageModel(tiny_config()),
            tokenizer,
            PackedTokenStream(SequenceTextSource(("missing cce",)), tokenizer),
            replace(
                training_config(tmp_path / "missing-cce"),
                device="cpu",
                exact_loss_backend="torch_compile",
            ),
        )


@pytest.mark.parametrize(
    "requested", ["auto", "cce_kahan_full_c", "cce_exact"]
)
def test_portable_compiled_cce_fails_closed_to_tiled_when_workspace_is_too_small(
    tmp_path, monkeypatch, requested
):
    monkeypatch.setattr(
        cognitive_training,
        "find_spec",
        lambda name: object() if name == "cut_cross_entropy" else None,
    )
    tokenizer = ByteTextTokenizer()
    trainer = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        PackedTokenStream(SequenceTextSource(("bounded cce",)), tokenizer),
        replace(
            training_config(tmp_path / requested),
            device="cpu",
            exact_loss_backend=requested,
            maximum_fused_loss_bytes=1,
        ),
    )
    assert trainer._exact_loss_backend == "tiled"
    assert trainer.runtime["compiled_cce_fits_workspace"] is False


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


def test_reporter_exception_cannot_cancel_valid_optimizer_step(
    tmp_path, monkeypatch,
):
    class FailingReporter:
        def __init__(self, *args, **kwargs):
            self.finished = False

        def log(self, metrics, *, step):
            raise OSError(f"intentional observer failure at {step}")

        def alert(self, *args, **kwargs):
            raise OSError("intentional alert failure")

        def finish(self):
            self.finished = True

    monkeypatch.setattr(
        cognitive_training, "TrackioReporter", FailingReporter
    )
    tokenizer = ByteTextTokenizer()
    model = MRCRALanguageModel(tiny_config())
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    config = replace(
        training_config(tmp_path / "reporter-failure"),
        total_tokens=8,
        trackio_enabled=True,
    )
    trainer = MRCRANextTokenTrainer(
        model,
        tokenizer,
        PackedTokenStream(
            SequenceTextSource(("observer failures are not gradients",)),
            tokenizer,
        ),
        config,
    )
    trainer.train(maximum_steps=1)
    assert trainer.state.step == 1
    assert trainer.state.tokens_seen == 8
    assert trainer.last_step_metrics["observation/failure_count"] == 1.0
    assert any(
        not torch.equal(before[name], value.detach().cpu())
        for name, value in trainer.model.state_dict().items()
        if value.dtype.is_floating_point
    )
    failures = [
        json.loads(line)
        for line in (
            tmp_path
            / "reporter-failure"
            / "observation_failures.jsonl"
        ).read_text().splitlines()
    ]
    assert failures == [{
        "error_type": "OSError",
        "kind": "observation_failure",
        "message": "intentional observer failure at 1",
        "operation": "log",
        "sequence": 1,
    }]


def test_trackio_and_null_observers_preserve_identical_optimization_authority(
    tmp_path, monkeypatch,
):
    class BoundedReporter:
        def __init__(self, *args, **kwargs):
            self.rows = []

        def log(self, metrics, *, step):
            self.rows.append((step, tuple(sorted(metrics))))

        def alert(self, *args, **kwargs):
            return None

        def log_phase_transition_trace(self, *args, **kwargs):
            return 0

        def finish(self):
            return None

    monkeypatch.setattr(
        cognitive_training, "TrackioReporter", BoundedReporter
    )
    tokenizer = ByteTextTokenizer()
    documents = ("observer authority cannot mutate optimization",)
    torch.manual_seed(20260726)
    initial_model = MRCRALanguageModel(tiny_config())
    initial = {
        name: value.detach().clone()
        for name, value in initial_model.state_dict().items()
    }

    def execute(path, *, trackio):
        model = MRCRALanguageModel(tiny_config())
        model.load_state_dict(initial)
        trainer = MRCRANextTokenTrainer(
            model,
            tokenizer,
            PackedTokenStream(SequenceTextSource(documents), tokenizer),
            replace(
                training_config(path),
                total_tokens=8,
                device="cpu",
                trackio_enabled=trackio,
                exact_loss_backend="tiled",
            ),
        )
        torch.manual_seed(911)
        trainer.train(maximum_steps=1)
        payload = torch.load(trainer.save_checkpoint(), weights_only=True)
        payload["training_state"] = dict(payload["training_state"])
        payload["training_state"].pop("elapsed_seconds")
        return payload

    null_payload = execute(tmp_path / "null", trackio=False)
    observed_payload = execute(tmp_path / "observed", trackio=True)

    def assert_tree_equal(left, right):
        assert type(left) is type(right)
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, dict):
            assert set(left) == set(right)
            for key in left:
                assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for first, second in zip(left, right, strict=True):
                assert_tree_equal(first, second)
        else:
            assert left == right

    for key in (
        "model",
        "optimizer",
        "scheduler",
        "training_state",
        "train_stream",
        "cstm_sampling",
        "torch_rng",
    ):
        assert_tree_equal(null_payload[key], observed_payload[key])


def test_format16_execution_and_observation_changes_resume_with_append_only_receipt(
    tmp_path,
):
    tokenizer = ByteTextTokenizer()
    documents = ("execution identity remains semantically exact",)
    config = replace(
        training_config(tmp_path / "format16-execution"),
        activation_policy="retain",
        log_interval=1,
        cstm_enabled=False,
    )
    trainer = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        PackedTokenStream(SequenceTextSource(documents), tokenizer),
        config,
    )
    trainer.train(maximum_steps=1)
    checkpoint = trainer.save_checkpoint()
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["format_version"] == 16
    assert len(saved["execution_policy_history"]) == 1

    resumed = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        PackedTokenStream(SequenceTextSource(documents), tokenizer),
        replace(config, activation_policy="whole_span", log_interval=2),
    )
    resumed.load_checkpoint(checkpoint)
    assert resumed.state.step == trainer.state.step
    assert len(resumed.execution_policy_history) == 2
    assert (
        resumed.execution_policy_history[-1]["execution_digest"]
        == resumed._identity_digest(resumed._identity()["execution"])
    )
    transition = resumed.execution_policy_history[-1]
    assert transition["old_policy_digest"] == (
        resumed.execution_policy_history[-2]["execution_digest"]
    )
    assert transition["new_policy_digest"] == transition["execution_digest"]
    assert transition["equivalence_receipt_digest"] == (
        resumed._equivalence_receipt_digest(
            transition["old_policy_digest"],
            transition["new_policy_digest"],
            transition["execution"],
        )
    )
    assert "resume-time execution-policy change" in (
        resumed.execution_policy_history[-1]["reason"]
    )


def test_format16_optimization_change_still_fails_closed(tmp_path):
    tokenizer = ByteTextTokenizer()
    documents = ("optimization identity cannot drift",)
    config = replace(
        training_config(tmp_path / "format16-optimization"),
        cstm_enabled=False,
    )
    trainer = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        PackedTokenStream(SequenceTextSource(documents), tokenizer),
        config,
    )
    trainer.train(maximum_steps=1)
    checkpoint = trainer.save_checkpoint()
    changed = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        PackedTokenStream(SequenceTextSource(documents), tokenizer),
        replace(config, learning_rate=config.learning_rate * 2),
    )
    with pytest.raises(ValueError, match="training contract differs"):
        changed.load_checkpoint(checkpoint)


def test_checkpoint_save_is_atomic_and_cleans_temporary_files_on_failure(
    tmp_path, monkeypatch,
):
    tokenizer = ByteTextTokenizer()
    trainer = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        PackedTokenStream(SequenceTextSource(("atomic checkpoint",)), tokenizer),
        replace(
            training_config(tmp_path / "atomic-checkpoint"),
            cstm_enabled=False,
        ),
    )

    def fail_save(*_args, **_kwargs):
        raise OSError("injected serialization failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="injected"):
        trainer.save_checkpoint()
    directory = tmp_path / "atomic-checkpoint" / "checkpoints"
    assert not list(directory.glob("*.tmp"))
    assert not list(directory.glob("step-*.pt"))
    assert not (directory / "latest.json").exists()


def test_format15_migrates_activation_checkpointing_out_of_semantic_identity(
    tmp_path,
):
    tokenizer = ByteTextTokenizer()
    documents = ("legacy activation policy is execution only",)
    config = replace(
        training_config(tmp_path / "format15-activation"),
        activation_policy="retain",
        cstm_enabled=False,
    )
    trainer = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        PackedTokenStream(SequenceTextSource(documents), tokenizer),
        config,
    )
    trainer.train(maximum_steps=1)
    payload = torch.load(trainer.save_checkpoint(), weights_only=True)
    payload["format_version"] = 15
    payload["identity"] = trainer._legacy_identity()
    for field in (
        "cstm_max_substrate_vjps",
        "cstm_target_participation_budget",
        "cstm_predictor_update_interval",
        "trackio_remote_log_interval",
    ):
        payload["identity"]["training"].pop(field)
    carrier = payload["identity"]["model_config"]["carrier"]
    carrier["activation_checkpointing"] = not carrier[
        "activation_checkpointing"
    ]
    legacy = tmp_path / "format15-activation.pt"
    torch.save(payload, legacy)

    resumed = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        PackedTokenStream(SequenceTextSource(documents), tokenizer),
        replace(config, activation_policy="whole_span"),
    )
    resumed.load_checkpoint(legacy)
    assert resumed.state.step == trainer.state.step
    assert resumed.activation_execution_policy.resolved == "whole_span"


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
    payload["identity"] = trainer._legacy_identity()
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
    timing = trainer.last_step_metrics
    assert timing["performance/evaluation_seconds"] > 0
    assert timing["performance/wall_clock_step_seconds"] >= timing[
        "performance/step_seconds"
    ]
    assert timing["performance/wall_clock_tokens_per_second"] <= timing[
        "performance/training_tokens_per_second"
    ]
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


def test_integrated_cstm_produces_honest_multiscale_receipts_and_governed_gradients(
    tmp_path,
):
    tokenizer = ByteTextTokenizer()
    config = replace(
        training_config(tmp_path / "integrated-cstm"),
        total_tokens=32,
        context_length=32,
        execution_chunk_size=4,
        tbptt_length=8,
        warmup_tokens=8,
        integrated_cognitive_path=True,
        cognitive_stride=2,
        progress_interval_tokens=32,
        cstm_enabled=True,
        cstm_weight=0.04,
        cstm_warmup_tokens=0,
        cstm_ramp_tokens=1,
        cstm_sampling_duty_cycle=1.0,
        cstm_target_participation_budget=4,
    )
    model = MRCRALanguageModel(tiny_config())
    head_before = {
        name: parameter.detach().clone()
        for name, parameter in model.cstm_predictor.named_parameters()
    }
    trainer = MRCRANextTokenTrainer(
        model,
        tokenizer,
        PackedTokenStream(
            SequenceTextSource((
                "abcdefghijklmnopqrstuvwxyz0123456789 repeated causal structure",
            )),
            tokenizer,
        ),
        config,
    )
    batch = trainer.train_stream.next_batch(1, 32)
    trainer.optimizer.zero_grad(set_to_none=True)
    local = trainer._run_context(batch, gradient_divisor=1)

    assert local["cstm/enabled"] == 1
    assert local["cstm/objective_weight"] == pytest.approx(0.04)
    assert local["cstm/standardized_huber"] > 0
    assert local["cstm/standardized_huber_sum"] > 0
    assert local["cstm/spectral_target_views"] > 0
    assert (
        local["cstm/context_valid_weight"]
        >= local["cstm/weighted_prediction_rows"]
    )
    assert local["cstm/sampling_active"] == 1
    assert local["cstm/sampling_obligations"] >= 1
    assert 0 < local["cstm/sampling_inclusion_probability"] <= 1
    assert local["cstm/substrate_vjp_count"] == 1
    assert 0 < local["cstm/row_inclusion_probability_min"] < 1
    assert local["cstm/row_inclusion_weight_max"] > 1
    assert local["cstm/actual_token_participations"] <= 4
    assert (
        local["cstm/estimated_dense_token_participations"]
        > local["cstm/actual_token_participations"]
    )
    assert local["cstm/estimated_dense_standardized_huber"] > 0
    assert local["cstm/coefficient_targets"] > local["cstm/spectral_target_views"]
    assert local["cstm/raw_token_view_equivalents"] > local["cstm/spectral_target_views"]
    assert local["cstm/supervision_relations_per_primary_target"] > 0
    assert local["softmax/training/exact_full_vocabulary"] == 1
    assert local["softmax/training/backend_id"] in {0, 1, 2, 3, 4}
    assert local["softmax/training/external_cce_available"] in {0, 1}
    assert local["softmax/training/compiled_cce_fits_workspace"] in {0, 1}
    assert local["softmax/training/estimated_full_logits_mib"] > 0
    # CSTM is retained separately until exact-CE gradients are authoritative.
    assert trainer._cstm_auxiliary_gradients
    assert any(
        name.startswith("cstm_predictor.")
        for name in trainer._cstm_auxiliary_gradients
    )
    assert any(
        name.startswith("cognitive.carrier.")
        for name in trainer._cstm_auxiliary_gradients
    )
    assert any(
        name.startswith("cognitive.")
        and not name.startswith("cognitive.carrier.")
        for name in trainer._cstm_auxiliary_gradients
    )
    assert all(
        name.startswith("cstm/") or not name.startswith("progress/")
        for name in local
    )

    merge = trainer._merge_cstm_gradients()
    assert merge["cstm/auxiliary_applied"] == 1
    assert merge["cstm/auxiliary_gradient_norm_before"] > 0
    assert (
        merge["cstm/auxiliary_gradient_norm_after"]
        <= merge["cstm/auxiliary_gradient_norm_before"] + 1e-7
    )
    assert merge["cstm/auxiliary_gradient_norm_after/carrier"] > 0
    assert not trainer._cstm_auxiliary_gradients
    trainer.optimizer.step()
    assert any(
        not torch.equal(
            parameter.detach(),
            head_before[name],
        )
        for name, parameter in model.cstm_predictor.named_parameters()
    )
    # The cognitive gate is deliberately zero-initialized: the first update
    # establishes a predictor-supported coupling before CSTM is allowed to
    # press on cognition. Its nonzero update is the causal precondition; the
    # multi-step learning acceptance test proves the subsequent live
    # cognitive-substrate adjoint on an eligible sampled obligation.
    assert bool(model.cstm_predictor.cognitive_gate.detach().ne(0).any())
    # The corpus accounting remains the packer's real token/target count.
    assert batch.token_count == 32
    assert int(local["train/valid_targets"]) <= batch.token_count
    assert "progress/tokens_seen" not in local


def test_sampled_cstm_budget_must_fit_one_complete_coarsest_row(tmp_path):
    tokenizer = ByteTextTokenizer()
    config = replace(
        training_config(tmp_path / "undersized-cstm-row-budget"),
        integrated_cognitive_path=True,
        cognitive_stride=2,
        cstm_enabled=True,
        cstm_execution="sampled",
        cstm_target_participation_budget=3,
    )
    with pytest.raises(
        ValueError,
        match="cannot fit one complete row at the coarsest carrier scale",
    ):
        MRCRANextTokenTrainer(
            MRCRALanguageModel(tiny_config()),
            tokenizer,
            PackedTokenStream(
                SequenceTextSource(("strict CSTM budget",)), tokenizer
            ),
            config,
        )


def test_off_duty_sampled_cstm_traverses_only_detached_predictor_path(
    tmp_path, monkeypatch,
):
    original = cognitive_training.deterministic_cstm_sample

    def controlled_sample(*args, duty_probability, **kwargs):
        return original(
            *args,
            duty_probability=duty_probability,
            uniform_override=(0.9, 0.01),
            **kwargs,
        )

    monkeypatch.setattr(
        cognitive_training,
        "deterministic_cstm_sample",
        controlled_sample,
    )
    tokenizer = ByteTextTokenizer()
    config = replace(
        training_config(tmp_path / "off-duty-cstm"),
        total_tokens=32,
        context_length=32,
        execution_chunk_size=4,
        tbptt_length=8,
        warmup_tokens=8,
        integrated_cognitive_path=True,
        cognitive_stride=2,
        cstm_enabled=True,
        cstm_warmup_tokens=0,
        cstm_ramp_tokens=1,
        cstm_sampling_duty_cycle=0.25,
    )
    trainer = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        PackedTokenStream(
            SequenceTextSource((
                "abcdefghijklmnopqrstuvwxyz0123456789 detached predictor",
            )),
            tokenizer,
        ),
        config,
    )
    local = trainer._run_context(
        trainer.train_stream.next_batch(1, 32),
        gradient_divisor=1,
    )
    assert local["cstm/predictor_update"] == 1
    assert local["cstm/substrate_update"] == 0
    assert local["cstm/substrate_vjp_count"] == 0
    assert trainer._cstm_auxiliary_gradients
    assert all(
        name.startswith("cstm_predictor.")
        for name in trainer._cstm_auxiliary_gradients
    )


def test_document_major_static_execution_preserves_exact_objective_and_gradients(
    tmp_path,
):
    """Regrouping may reduce invocations but may not change learning pressure."""

    tokenizer = ByteTextTokenizer()
    source_documents = (
        "alpha",
        "bravo",
        "cider",
        "delta",
        "ember",
        "fable",
        "gamut",
        "helix",
    )
    batch_stream = PackedTokenStream(
        SequenceTextSource(source_documents), tokenizer
    )
    batch = batch_stream.next_batch(1, 32)
    base_training = replace(
        training_config(tmp_path / "document-major"),
        total_tokens=32,
        context_length=32,
        execution_chunk_size=4,
        tbptt_length=8,
        warmup_tokens=8,
        integrated_cognitive_path=True,
        cognitive_stride=2,
        cstm_enabled=False,
        phase_transition_telemetry=False,
        spectral_regularization_weight=0.0,
        exact_loss_backend="fused",
        maximum_fused_loss_bytes=64 << 20,
    )
    model_config = replace(
        tiny_config(),
        carrier=replace(
            tiny_config().carrier, activation_checkpointing=False
        ),
    )
    torch.manual_seed(1291)
    initial_model = MRCRALanguageModel(model_config)
    initial_state = {
        name: value.detach().clone()
        for name, value in initial_model.state_dict().items()
    }

    serial_model = MRCRALanguageModel(model_config)
    serial_model.load_state_dict(initial_state)
    serial = MRCRANextTokenTrainer(
        serial_model,
        tokenizer,
        PackedTokenStream(SequenceTextSource(source_documents), tokenizer),
        replace(base_training, document_static_batching=False),
    )
    document_model = MRCRALanguageModel(model_config)
    document_model.load_state_dict(initial_state)
    document = MRCRANextTokenTrainer(
        document_model,
        tokenizer,
        PackedTokenStream(SequenceTextSource(source_documents), tokenizer),
        replace(base_training, document_static_batching=True),
    )
    exact_model = MRCRALanguageModel(model_config)
    exact_model.load_state_dict(initial_state)
    exact = MRCRANextTokenTrainer(
        exact_model,
        tokenizer,
        PackedTokenStream(SequenceTextSource(source_documents), tokenizer),
        replace(
            base_training,
            document_static_batching=True,
            document_grouping_policy="exact_signature",
        ),
    )
    serial.optimizer.zero_grad(set_to_none=True)
    document.optimizer.zero_grad(set_to_none=True)
    exact.optimizer.zero_grad(set_to_none=True)
    serial_metrics = serial._run_context(batch, gradient_divisor=1)
    document_metrics = document._run_context(batch, gradient_divisor=1)
    exact_metrics = exact._run_context(batch, gradient_divisor=1)

    assert document_metrics["document_batching/target_bijection"] == 1
    assert document_metrics["train/valid_targets"] == serial_metrics[
        "train/valid_targets"
    ]
    assert document_metrics["train/utf8_bytes"] == serial_metrics[
        "train/utf8_bytes"
    ]
    assert document_metrics["train/nll_sum"] == pytest.approx(
        serial_metrics["train/nll_sum"], rel=2e-6, abs=2e-5
    )
    assert document_metrics[
        "train/cross_entropy_nats_per_token"
    ] == pytest.approx(
        serial_metrics["train/cross_entropy_nats_per_token"],
        rel=2e-6,
        abs=2e-6,
    )
    for metrics in (document_metrics, exact_metrics):
        assert metrics["train/valid_targets"] == serial_metrics[
            "train/valid_targets"
        ]
        assert metrics["train/utf8_bytes"] == serial_metrics[
            "train/utf8_bytes"
        ]
        assert metrics["train/nll_sum"] == pytest.approx(
            serial_metrics["train/nll_sum"], rel=2e-6, abs=2e-5
        )
    logical_spans = sum(
        len(sequence.spans)
        for sequence in document.document_batch_planner.plan(batch).sequences
    )
    assert (
        document_metrics["document_batching/physical_invocations"]
        < logical_spans
    )
    serial_parameters = dict(serial_model.named_parameters())
    for candidate_model in (document_model, exact_model):
        checked = 0
        for name, parameter in candidate_model.named_parameters():
            reference = serial_parameters[name]
            if reference.grad is None:
                assert parameter.grad is None, name
            else:
                torch.testing.assert_close(
                    parameter.grad,
                    reference.grad,
                    atol=3e-5,
                    rtol=3e-4,
                    msg=name,
                )
                checked += 1
        assert checked > 20


def test_cstm_warmup_is_exact_pure_ce_and_allocates_no_auxiliary_gradients(tmp_path):
    tokenizer = ByteTextTokenizer()
    config = replace(
        training_config(tmp_path / "cstm-warmup"),
        integrated_cognitive_path=True,
        cognitive_stride=2,
        cstm_enabled=True,
        cstm_warmup_tokens=10_000,
        cstm_ramp_tokens=1,
    )
    trainer = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        PackedTokenStream(SequenceTextSource(("warmup must remain exact",)), tokenizer),
        config,
    )
    local = trainer._run_context(trainer.train_stream.next_batch(1, 8))
    assert local["cstm/enabled"] == 1
    assert local["cstm/objective_weight"] == 0
    assert local["cstm/spectral_target_views"] == 0
    assert not trainer._cstm_auxiliary_gradients


def test_active_cstm_checkpoint_resume_preserves_parameters_rms_and_horizon_schedule(
    tmp_path,
):
    tokenizer = ByteTextTokenizer()
    documents = (
        "abcdefghijklmnopqrstuvwxyz0123456789 repeating spectral sequence",
        "another sufficiently long independent causal training document",
    )
    config = replace(
        training_config(tmp_path / "cstm-reference"),
        total_tokens=64,
        context_length=32,
        execution_chunk_size=4,
        tbptt_length=8,
        warmup_tokens=8,
        integrated_cognitive_path=True,
        cognitive_stride=2,
        progress_interval_tokens=32,
        cstm_enabled=True,
        cstm_weight=0.04,
        cstm_warmup_tokens=0,
        cstm_ramp_tokens=1,
        cstm_sampling_duty_cycle=1.0,
    )
    torch.manual_seed(743)
    initial_model = MRCRALanguageModel(tiny_config())
    initial = {
        name: value.detach().clone()
        for name, value in initial_model.state_dict().items()
    }

    reference_model = MRCRALanguageModel(tiny_config())
    reference_model.load_state_dict(initial)
    reference = MRCRANextTokenTrainer(
        reference_model,
        tokenizer,
        PackedTokenStream(SequenceTextSource(documents), tokenizer),
        config,
    )
    reference.train(maximum_steps=2)

    interrupted_model = MRCRALanguageModel(tiny_config())
    interrupted_model.load_state_dict(initial)
    interrupted = MRCRANextTokenTrainer(
        interrupted_model,
        tokenizer,
        PackedTokenStream(SequenceTextSource(documents), tokenizer),
        replace(config, output_dir=str(tmp_path / "cstm-interrupted")),
    )
    interrupted.train(maximum_steps=1)
    checkpoint = interrupted.save_checkpoint()
    saved_rms = interrupted.model.cstm_predictor.target_second_moment.clone()
    saved_coverage = interrupted.cstm_coverage.state_dict()
    assert interrupted.model.cstm_predictor.target_rms_initialized.any()

    resumed_model = MRCRALanguageModel(tiny_config())
    resumed = MRCRANextTokenTrainer(
        resumed_model,
        tokenizer,
        PackedTokenStream(SequenceTextSource(documents), tokenizer),
        replace(config, output_dir=str(tmp_path / "cstm-interrupted")),
    )
    resumed.load_checkpoint(checkpoint)
    torch.testing.assert_close(
        resumed.model.cstm_predictor.target_second_moment,
        saved_rms,
        atol=0,
        rtol=0,
    )
    assert resumed.cstm_coverage.state_dict() == saved_coverage
    resumed.train(maximum_steps=1)

    assert resumed.state.step == reference.state.step == 2
    assert resumed.state.tokens_seen == reference.state.tokens_seen == 64
    for name, expected in reference.model.state_dict().items():
        torch.testing.assert_close(
            resumed.model.state_dict()[name],
            expected,
            atol=1e-6,
            rtol=1e-6,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def test_format16_sampled_cstm_missing_or_corrupt_coverage_fails_closed(
    tmp_path,
):
    tokenizer = ByteTextTokenizer()
    documents = (
        "abcdefghijklmnopqrstuvwxyz0123456789 sampled checkpoint authority",
    )
    config = replace(
        training_config(tmp_path / "cstm-corrupt-state"),
        total_tokens=32,
        context_length=32,
        execution_chunk_size=4,
        tbptt_length=8,
        warmup_tokens=8,
        integrated_cognitive_path=True,
        cognitive_stride=2,
        cstm_enabled=True,
        cstm_warmup_tokens=0,
        cstm_ramp_tokens=1,
        cstm_sampling_duty_cycle=1.0,
    )
    trainer = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        PackedTokenStream(SequenceTextSource(documents), tokenizer),
        config,
    )
    trainer.train(maximum_steps=1)
    payload = torch.load(trainer.save_checkpoint(), weights_only=True)
    for name, state in (
        ("missing", None),
        (
            "corrupt",
            {
                **payload["cstm_sampling"],
                "schema_version": 999,
            },
        ),
    ):
        changed = dict(payload, cstm_sampling=state)
        path = tmp_path / f"{name}-coverage.pt"
        torch.save(changed, path)
        restored = MRCRANextTokenTrainer(
            MRCRALanguageModel(tiny_config()),
            tokenizer,
            PackedTokenStream(SequenceTextSource(documents), tokenizer),
            config,
        )
        with pytest.raises(ValueError, match="coverage"):
            restored.load_checkpoint(path)


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
    payload["identity"] = trainer._legacy_identity()
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
    payload["identity"] = trainer._legacy_identity()
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


def test_v12_checkpoint_migrates_to_deterministic_cstm_head_and_format16_contract(
    tmp_path,
):
    """The immediate pre-CSTM format must continue without invented history."""

    tokenizer = ByteTextTokenizer()
    config = replace(
        training_config(tmp_path / "v12"),
        integrated_cognitive_path=True,
        cognitive_stride=2,
        cstm_enabled=True,
        cstm_warmup_tokens=10_000,
    )
    trainer = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        PackedTokenStream(
            SequenceTextSource(("format twelve migration document",)),
            tokenizer,
        ),
        config,
    )
    trainer.train(maximum_steps=1)
    payload = torch.load(trainer.save_checkpoint(), weights_only=True)
    payload["format_version"] = 12
    payload["identity"] = trainer._legacy_identity()
    payload["identity"].pop("cstm_architecture")
    for name in (
        "cstm_enabled",
        "cstm_weight",
        "cstm_warmup_tokens",
        "cstm_ramp_tokens",
        "cstm_carrier_gradient_cap",
        "cstm_cognitive_gradient_cap",
        "cstm_head_gradient_cap",
    ):
        payload["identity"]["training"].pop(name)
    cstm_model_names = {
        name for name in payload["model"]
        if name.startswith("cstm_predictor.")
    }
    for name in cstm_model_names:
        payload["model"].pop(name)

    # Format 12 had no CSTM parameters in its optimizer groups. Remove those
    # exact serialized IDs while preserving every pre-existing actor state.
    serialized_optimizer = payload["optimizer"]
    live_names = {
        id(parameter): name
        for name, parameter in trainer.model.named_parameters()
    }
    cstm_parameter_ids = set()
    for live_group, saved_group in zip(
        trainer.optimizer.param_groups,
        serialized_optimizer["param_groups"],
        strict=True,
    ):
        retained_ids = []
        for parameter, parameter_id in zip(
            live_group["params"], saved_group["params"], strict=True
        ):
            if live_names[id(parameter)].startswith("cstm_predictor."):
                cstm_parameter_ids.add(parameter_id)
            else:
                retained_ids.append(parameter_id)
        saved_group["params"] = retained_ids
    for parameter_id in cstm_parameter_ids:
        serialized_optimizer["state"].pop(parameter_id, None)

    legacy = tmp_path / "v12.pt"
    torch.save(payload, legacy)
    torch.manual_seed(991)
    restored_model = MRCRALanguageModel(tiny_config())
    expected_cstm = {
        name: value.detach().clone()
        for name, value in restored_model.state_dict().items()
        if name.startswith("cstm_predictor.")
    }
    restored = MRCRANextTokenTrainer(
        restored_model,
        tokenizer,
        PackedTokenStream(
            SequenceTextSource(("format twelve migration document",)),
            tokenizer,
        ),
        config,
    )
    restored.load_checkpoint(legacy)

    for name, expected in expected_cstm.items():
        torch.testing.assert_close(
            restored.model.state_dict()[name],
            expected,
            atol=0,
            rtol=0,
        )
    assert not restored.model.cstm_predictor.target_rms_initialized.any()
    assert restored.cstm_enabled


def test_v14_checkpoint_migrates_to_default_document_major_execution_contract(
    tmp_path,
):
    tokenizer = ByteTextTokenizer()
    config = replace(
        training_config(tmp_path / "v14-document-migration"),
        total_tokens=32,
        integrated_cognitive_path=True,
        cognitive_stride=2,
        cstm_enabled=False,
        document_static_batching=True,
    )
    documents = ("document batching migration authority",)
    trainer = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        PackedTokenStream(SequenceTextSource(documents), tokenizer),
        config,
    )
    trainer.train(maximum_steps=1)
    payload = torch.load(trainer.save_checkpoint(), weights_only=True)
    payload["format_version"] = 14
    payload["identity"] = trainer._legacy_identity()
    for name in (
        "document_static_batching",
        "document_bucket_lengths",
        "document_batch_token_budget",
    ):
        payload["identity"]["training"].pop(name)
    legacy = tmp_path / "format-14.pt"
    torch.save(payload, legacy)

    restored = MRCRANextTokenTrainer(
        MRCRALanguageModel(tiny_config()),
        tokenizer,
        PackedTokenStream(SequenceTextSource(documents), tokenizer),
        config,
    )
    restored.load_checkpoint(legacy)
    assert restored.state.step == trainer.state.step
    assert restored.document_batch_planner is not None
    assert restored.runtime["document_static_batching"] is True
