from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from threading import Event, get_ident
from types import SimpleNamespace

import pytest
import torch

from mrrn.language import MRRNLanguageModel, tiny_language_config
import mrrn.lm_training as lm
from mrrn.lm_training import (
    ByteTextTokenizer,
    FineWebTextSource,
    HuggingFaceTextTokenizer,
    LMTrainingConfig,
    NextTokenTrainer,
    PackedBatch,
    PackedTokenStream,
    SequenceTextSource,
    TextDocument,
    TokenizedDocument,
    TrackioReporter,
    _configure_cuda,
    _MetricAccumulator,
    _device_for,
    _heldout_role,
    _is_evaluation_document,
    _memory_metrics,
    _precision_for,
    _runtime_details,
    build_evaluation_batches,
    next_token_statistics,
    resonator_state_regularization,
    resonator_state_rms,
    stability_metrics,
)


class _FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def shuffle(self, *, seed, buffer_size):
        assert seed == 9 and buffer_size == 4
        return self

    def skip(self, count):
        return _FakeDataset(self.rows[count:])

    def __iter__(self):
        return iter(self.rows)


def test_fineweb_stream_validates_schema_partitions_and_resume(monkeypatch):
    eval_id = next(f"id-{i}" for i in range(20_000) if _is_evaluation_document(f"id-{i}", 100))
    train_id = next(f"id-{i}" for i in range(20_000) if not _is_evaluation_document(f"id-{i}", 100))
    rows = [
        {"id": eval_id, "text": "evaluation"},
        {"id": train_id, "text": "training"},
        {"id": "empty", "text": ""},
    ]
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: _FakeDataset(rows)),
    )
    source = FineWebTextSource(
        revision="fixed", partition="train", evaluation_fraction_permyriad=100,
        shuffle_seed=9, shuffle_buffer=4,
    )
    assert list(source) == [TextDocument(train_id, "training")]
    with pytest.raises(RuntimeError, match="single active"):
        list(source)
    state = source.state_dict()
    restored = FineWebTextSource(
        revision="fixed", partition="train", evaluation_fraction_permyriad=100,
        shuffle_seed=9, shuffle_buffer=4,
    )
    restored.load_state_dict(state)
    assert list(restored) == []
    mismatch = FineWebTextSource(
        revision="other", partition="train", evaluation_fraction_permyriad=100,
        shuffle_seed=9, shuffle_buffer=4,
    )
    with pytest.raises(ValueError, match="revision"):
        mismatch.load_state_dict(state)
    with pytest.raises(RuntimeError, match="after iteration"):
        restored.load_state_dict(state)
    bad_counters = dict(state, raw_rows_scanned=-1)
    with pytest.raises(ValueError, match="counters"):
        FineWebTextSource(
            revision="fixed", partition="train", evaluation_fraction_permyriad=100,
            shuffle_seed=9, shuffle_buffer=4,
        ).load_state_dict(bad_counters)


def test_fineweb_progress_and_guard_partitions_are_pairwise_disjoint(monkeypatch):
    progress_id = next(
        f"progress-{index}" for index in range(100_000)
        if _heldout_role(f"progress-{index}", 100) == "progress"
    )
    guard_id = next(
        f"guard-{index}" for index in range(100_000)
        if _heldout_role(f"guard-{index}", 100) == "eval"
    )
    train_id = next(
        f"train-{index}" for index in range(100_000)
        if _heldout_role(f"train-{index}", 100) == "train"
    )
    rows = [
        {"id": progress_id, "text": "progress"},
        {"id": guard_id, "text": "guard"},
        {"id": train_id, "text": "train"},
    ]
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: _FakeDataset(rows)),
    )
    partitions = {}
    for partition in ("train", "progress", "eval"):
        source = FineWebTextSource(
            partition=partition,
            evaluation_fraction_permyriad=100,
            shuffle_seed=9,
            shuffle_buffer=4,
        )
        partitions[partition] = {row.identifier for row in source}
    assert partitions == {
        "train": {train_id},
        "progress": {progress_id},
        "eval": {guard_id},
    }
    assert not (
        partitions["train"] & partitions["progress"]
        or partitions["train"] & partitions["eval"]
        or partitions["progress"] & partitions["eval"]
    )


def test_fineweb_stream_fails_closed_on_schema_drift_and_invalid_configuration(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: _FakeDataset([{"id": "x"}])),
    )
    source = FineWebTextSource(partition="eval", shuffle_seed=9, shuffle_buffer=4)
    with pytest.raises(ValueError, match="text"):
        list(source)
    for kwargs in (
        {"partition": "holdout"},
        {"evaluation_fraction_permyriad": 0},
        {"shuffle_buffer": 0},
    ):
        with pytest.raises(ValueError):
            FineWebTextSource(**kwargs)


class _FakeTokenizer:
    is_fast = True
    eos_token_id = 9

    def __len__(self):
        return 10

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2], "offset_mapping": [(0, 1), (1, len(text))]}

    def encode(self, text, **kwargs):
        return [3] if text else []

    def decode(self, token_ids, **kwargs):
        return "decoded:" + ",".join(map(str, token_ids))


def test_huggingface_tokenizer_tracks_original_utf8_bytes_and_identity(monkeypatch):
    fake = _FakeTokenizer()
    monkeypatch.setitem(
        sys.modules, "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: fake)),
    )
    tokenizer = HuggingFaceTextTokenizer("fake", revision="sha")
    encoded = tokenizer.encode_document("aé")
    assert encoded == TokenizedDocument((1, 2, 9), (1, 2, 0))
    assert tokenizer.encode_prompt("") == [9]
    assert tokenizer.decode([1, 2]) == "decoded:1,2"
    assert tokenizer.identity()["revision"] == "sha"
    with pytest.raises(TypeError):
        tokenizer.encode_document(3)
    fake.__call__ = lambda *args, **kwargs: {"input_ids": [1], "offset_mapping": []}


def test_huggingface_overlapping_byte_tokens_do_not_double_count_unicode(
    monkeypatch,
):
    class OverlapTokenizer(_FakeTokenizer):
        def __call__(self, text, **kwargs):
            assert text == "🙂e\u0301"
            return {
                "input_ids": [1, 2, 3, 4],
                "offset_mapping": [(0, 1), (0, 1), (1, 2), (2, 3)],
            }

    fake = OverlapTokenizer()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: fake
            )
        ),
    )
    tokenizer = HuggingFaceTextTokenizer("fake", revision="sha")
    encoded = tokenizer.encode_document("🙂e\u0301")

    assert encoded.byte_lengths == (4, 0, 1, 2, 0)
    assert sum(encoded.byte_lengths) == len("🙂e\u0301".encode("utf-8"))


def test_token_source_packer_and_batch_contract_failures():
    with pytest.raises(ValueError):
        TokenizedDocument((), ())
    with pytest.raises(ValueError):
        TokenizedDocument((1,), (-1,))
    with pytest.raises(ValueError):
        SequenceTextSource(())
    source = SequenceTextSource(("abc",), repeat=False)
    assert next(iter(source)).text == "abc"
    with pytest.raises(RuntimeError, match="single active"):
        list(source)
    state = SequenceTextSource(("abc",)).state_dict()
    with pytest.raises(ValueError, match="does not match"):
        SequenceTextSource(("different",)).load_state_dict(state)

    tokenizer = ByteTextTokenizer()
    stream = PackedTokenStream(SequenceTextSource(("abcdef",)), tokenizer)
    with pytest.raises(ValueError):
        stream.next_batch(0, 1)
    with pytest.raises(ValueError):
        stream._fill(0)
    batch = stream.next_batch(1, 2)
    assert batch.token_count == 2 and batch.byte_count == 2
    assert batch.to("cpu").input_ids.device.type == "cpu"
    with pytest.raises(RuntimeError, match="after iteration"):
        stream.load_state_dict(stream.state_dict())
    fresh = PackedTokenStream(SequenceTextSource(("abcdef",)), tokenizer)
    checkpoint = fresh.state_dict()
    with pytest.raises(ValueError, match="tokenizer"):
        fresh.load_state_dict(dict(checkpoint, tokenizer={}))
    with pytest.raises(ValueError, match="buffer"):
        fresh.load_state_dict(dict(checkpoint, token_buffer=[1], byte_buffer=[]))

    with pytest.raises(ValueError):
        PackedBatch(torch.ones(1, 2), torch.ones(1, 2), torch.ones(1, 2, dtype=torch.long))
    with pytest.raises(ValueError):
        PackedBatch(
            torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.long),
            torch.tensor([[1, -1]], dtype=torch.long),
        )


def test_statistics_accumulator_and_training_configuration_contracts():
    config = LMTrainingConfig(
        total_tokens=20_000_000, sequence_length=2048, micro_batch_size=1,
        gradient_accumulation_steps=4,
    )
    assert config.tokens_per_update == 8192
    assert config.total_steps == 2442
    assert config.warmup_steps == 98
    assert config.output_dir.endswith("fineweb-4p7m-stable")
    assert config.run_name.startswith("mrrn-4p7m-")
    assert LMTrainingConfig(device="cuda:1", precision="bf16").device == "cuda:1"
    invalid = (
        {"total_tokens": 4, "sequence_length": 8},
        {"learning_rate": 0},
        {"weight_decay": -1},
        {"spectral_regularization_weight": -1},
        {"minimum_learning_rate_ratio": 2},
        {"generation_tokens": -1},
        {"device": "tpu"},
        {"precision": "tf32"},
        {"state_target_rms": 20, "state_warning_rms": 10},
        {"gradient_warning_norm": 0.5},
        {"gradient_backoff_factor": 1},
        {"gradient_recovery_limit": -1},
    )
    for kwargs in invalid:
        with pytest.raises(ValueError):
            LMTrainingConfig(**kwargs)

    with pytest.raises(ValueError):
        next_token_statistics(torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 3))
    with pytest.raises(ValueError):
        next_token_statistics(
            torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2)
        )
    accumulator = _MetricAccumulator()
    with pytest.raises(ValueError, match="no valid"):
        accumulator.metrics("x")
    stats = next_token_statistics(
        torch.zeros(1, 2, 3), torch.tensor([[0, 1]]), torch.tensor([[1, 1]])
    )
    accumulator.add(stats, stats.cross_entropy, torch.tensor(0.25), torch.tensor(0.5))
    metrics = accumulator.metrics("x")
    assert metrics["x/cross_entropy_nats_per_token"] == pytest.approx(math.log(3))
    assert metrics["x/bits_per_byte"] == pytest.approx(math.log(3) / math.log(2))
    assert metrics["x/state_energy_regularization"] == 0.5


class _TrackioModule:
    AlertLevel = SimpleNamespace(INFO="info", WARN="warn", ERROR="error")

    def __init__(self, *, show_error=False):
        self.events = []
        self.show_error = show_error
        self.current_run = None
        self.persisted_run_ids = set()
        class Artifact:
            def __init__(self, name, type, description=None, metadata=None):
                self.name, self.type = name, type
                self.description, self.metadata = description, metadata
                self.files = []
                self.version = None

            def add_file(self, path, name=None):
                self.files.append((Path(path), name))

        self.Artifact = Artifact

    def init(self, **kwargs):
        self.events.append(("init", kwargs))
        owner = self

        class Run:
            id = "test-trackio-run-id"
            name = kwargs["name"]

            def log(self, metrics, step=None):
                owner._record_log(metrics, step=step)

        self.current_run = Run()
        return self.current_run

    def Api(self):
        owner = self

        class Api:
            def runs(self, project):
                if (
                    owner.current_run is None
                    or owner.current_run.id not in owner.persisted_run_ids
                ):
                    return []
                return [
                    SimpleNamespace(
                        id=owner.current_run.id,
                        name=owner.current_run.name,
                        project=project,
                    )
                ]

        return Api()

    def show(self, **kwargs):
        self.events.append(("show", kwargs))
        if self.show_error:
            raise RuntimeError("dashboard unavailable")

    def _record_log(self, metrics, *, step):
        self.events.append(("log", step, metrics))
        if self.current_run is not None:
            self.persisted_run_ids.add(self.current_run.id)

    def log(self, metrics, *, step):
        self._record_log(metrics, step=step)

    def alert(self, **kwargs):
        self.events.append(("alert", kwargs))

    def log_artifact(self, artifact, aliases=None):
        artifact.version = 1
        self.events.append(("artifact", artifact, aliases))
        return artifact

    def finish(self):
        self.events.append(("finish",))


def test_trackio_reporter_logs_jsonl_dashboard_warning_and_rejects_nonfinite(tmp_path, monkeypatch):
    trackio = _TrackioModule(show_error=True)
    monkeypatch.setitem(sys.modules, "trackio", trackio)
    config = LMTrainingConfig(
        output_dir=str(tmp_path), total_tokens=4, sequence_length=4,
        evaluation_batches=1, show_dashboard=True,
    )
    reporter = TrackioReporter(config, {"model": "tiny"}, resume=True)
    reporter.log({"loss": 1}, step=2)
    reporter.alert("notice", "text", level="warn", step=2)
    version = reporter.log_spectral_evidence(
        {"checkpoint": {"tokens_seen": 4}, "tokens": [], "traces": [], "poles": [], "triads": []},
        step=2,
    )
    assert version == 1
    transition = tmp_path / "first-hard-event.json"
    transition.write_text('{"schema_version":1}', encoding="utf-8")
    assert reporter.log_phase_transition_trace(transition, step=2) == 1
    with pytest.raises(FloatingPointError):
        reporter.log({"bad": float("nan")}, step=3)
    reporter.finish()
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert records[0]["kind"] == "trackio_run_registration"
    assert records[0]["visible_before_dashboard"] is True
    assert records[1]["kind"] == "dashboard_warning"
    assert any(record["kind"] == "metrics" for record in records)
    assert any(record["kind"] == "spectral_snapshot" for record in records)
    assert any(record["kind"] == "phase_transition_trace" for record in records)
    phase_artifact = next(
        event[1] for event in trackio.events
        if event[0] == "artifact" and event[1].name == "mrcra-first-hard-event"
    )
    assert phase_artifact.files[0][1] == "first-hard-event.json"
    assert trackio.events[0][1]["auto_log_cpu"] is False
    show = next(event for event in trackio.events if event[0] == "show")
    assert Path(show[1]["frontend_dir"], "mrrn-spectral-view.html").is_file()
    registration = next(event for event in trackio.events if event[0] == "log")
    assert registration == ("log", 0, {"progress/tokens_seen": 0.0})
    assert trackio.events.index(registration) < trackio.events.index(show)
    with pytest.raises(FileExistsError, match="already exists"):
        TrackioReporter(config, {}, resume=False)


def test_trackio_remote_delivery_is_bounded_and_local_stream_is_complete(
    tmp_path, monkeypatch,
):
    gate = Event()

    class SlowTrackio(_TrackioModule):
        def _record_log(self, metrics, *, step):
            if "progress/tokens_seen" not in metrics:
                gate.wait(timeout=2.0)
            super()._record_log(metrics, step=step)

    trackio = SlowTrackio()
    monkeypatch.setitem(sys.modules, "trackio", trackio)
    config = LMTrainingConfig(
        output_dir=str(tmp_path),
        total_tokens=4,
        sequence_length=4,
        evaluation_batches=1,
        show_dashboard=False,
        spectral_dashboard=False,
        trackio_remote_log_interval=1,
    )
    reporter = TrackioReporter(config, {"model": "tiny"}, resume=True)
    for step in range(100):
        reporter.log({"loss": float(step)}, step=step)
    assert not reporter._drain_remote_logs(timeout_seconds=0.02)
    assert reporter._remote_dropped > 0
    records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert sum(record["kind"] == "metrics" for record in records) == 100
    gate.set()
    reporter.finish()
    records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    summary = next(
        record
        for record in records
        if record["kind"] == "trackio_remote_summary"
    )
    assert summary["dropped_remote_metric_rows"] > 0
    assert summary["drain_timeouts"] == 1
    assert summary["worker_alive_at_bounded_finish"] is False


def test_trackio_coalesces_remote_scalars_but_retains_every_local_row(
    tmp_path, monkeypatch,
):
    trackio = _TrackioModule()
    monkeypatch.setitem(sys.modules, "trackio", trackio)
    config = LMTrainingConfig(
        output_dir=str(tmp_path),
        total_tokens=4,
        sequence_length=4,
        evaluation_batches=1,
        show_dashboard=False,
        spectral_dashboard=False,
        trackio_remote_log_interval=4,
    )
    reporter = TrackioReporter(config, {}, resume=True)
    for step in range(1, 11):
        reporter.log({"loss": float(step)}, step=step)
    reporter.finish()

    remote_steps = [
        event[1] for event in trackio.events if event[0] == "log"
    ]
    assert remote_steps == [0, 4, 8, 10]
    records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert sum(record["kind"] == "metrics" for record in records) == 10
    summary = next(
        record
        for record in records
        if record["kind"] == "trackio_remote_summary"
    )
    assert summary["coalesced_remote_metric_rows"] == 7
    assert summary["dropped_remote_metric_rows"] == 0


def test_trackio_writer_uses_explicit_run_outside_initializing_thread(
    tmp_path, monkeypatch,
):
    initializing_thread = get_ident()

    class ContextBoundTrackio(_TrackioModule):
        def log(self, metrics, *, step):
            if get_ident() != initializing_thread:
                raise RuntimeError("Call trackio.init() before trackio.log().")
            super().log(metrics, step=step)

    trackio = ContextBoundTrackio()
    monkeypatch.setitem(sys.modules, "trackio", trackio)
    config = LMTrainingConfig(
        output_dir=str(tmp_path),
        total_tokens=4,
        sequence_length=4,
        evaluation_batches=1,
        show_dashboard=False,
        spectral_dashboard=False,
        trackio_remote_log_interval=1,
    )
    reporter = TrackioReporter(config, {}, resume=True)
    reporter.log({"loss": 1.0}, step=1)
    reporter.finish()

    assert ("log", 1, {"loss": 1.0}) in trackio.events
    records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    summaries = [
        record
        for record in records
        if record["kind"] == "trackio_remote_summary"
    ]
    assert not summaries or summaries[-1]["error"] is None


def test_trackio_coalesces_only_repetitive_checkpoint_info_alerts(
    tmp_path, monkeypatch,
):
    trackio = _TrackioModule()
    monkeypatch.setitem(sys.modules, "trackio", trackio)
    config = LMTrainingConfig(
        output_dir=str(tmp_path),
        total_tokens=4,
        sequence_length=4,
        evaluation_batches=1,
        show_dashboard=False,
        spectral_dashboard=False,
    )
    reporter = TrackioReporter(config, {}, resume=True)
    for step in range(1, 12):
        reporter.alert(
            "MRCRA checkpoint saved",
            f"step-{step}.pt",
            level="info",
            step=step,
        )
    reporter.alert("failure", "important", level="error", step=12)
    reporter.alert("First MRCRA hard event", "important", level="info", step=12)
    reporter.finish()
    remote_alerts = [
        event for event in trackio.events if event[0] == "alert"
    ]
    # Checkpoints 1 and 10, plus both nonrepetitive alerts.
    assert len(remote_alerts) == 4
    records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert sum(record["kind"] == "alert" for record in records) == 13
    assert sum(
        record.get("remote_coalesced") is True for record in records
    ) == 9


class _Reporter:
    instances = []

    def __init__(
        self,
        config,
        run_config,
        *,
        resume,
        initial_step=0,
        initial_tokens=0,
    ):
        self.events = []
        self.resume = resume
        self.initial_step = initial_step
        self.initial_tokens = initial_tokens
        _Reporter.instances.append(self)

    def log(self, metrics, *, step):
        assert all(math.isfinite(value) for value in metrics.values())
        self.events.append(("log", step, metrics))

    def alert(self, title, text, *, level, step):
        self.events.append(("alert", step, title, level))

    def log_spectral_evidence(self, evidence, *, step):
        assert evidence["checkpoint"]["step"] == step
        self.events.append(("spectral", step, len(evidence["tokens"])))
        return 1

    def finish(self):
        self.events.append(("finish",))


def _tiny_trainer(tmp_path, monkeypatch, *, total_tokens=8, source_text="training data"):
    monkeypatch.setattr(lm, "TrackioReporter", _Reporter)
    tokenizer = ByteTextTokenizer()
    torch.manual_seed(11)
    model = MRRNLanguageModel(tiny_language_config(tokenizer.vocabulary_size))
    train = PackedTokenStream(SequenceTextSource((source_text,)), tokenizer)
    evaluation = PackedTokenStream(SequenceTextSource(("evaluation",)), tokenizer)
    config = LMTrainingConfig(
        output_dir=str(tmp_path), total_tokens=total_tokens, sequence_length=4,
        micro_batch_size=1, gradient_accumulation_steps=2, warmup_tokens=4,
        log_interval=1, architecture_log_interval=1, evaluation_interval=1,
        evaluation_batches=1, checkpoint_interval=1, keep_checkpoints=1,
        generation_tokens=1, generation_prompt="x", device="cpu", show_dashboard=False,
    )
    batches = build_evaluation_batches(evaluation, count=1, batch_size=1, sequence_length=4)
    return NextTokenTrainer(model, tokenizer, train, batches, config)


def test_complete_tiny_training_metrics_generation_checkpoint_and_resume(tmp_path, monkeypatch):
    _Reporter.instances.clear()
    trainer = _tiny_trainer(tmp_path / "first", monkeypatch)
    state = trainer.train()
    assert state.step == 1 and state.tokens_seen == 8
    latest = tmp_path / "first" / "checkpoints" / "step-0000001.pt"
    assert latest.is_file() and (tmp_path / "first" / "checkpoints" / "best.pt").is_file()
    assert (tmp_path / "first" / "samples" / "step-0000001.txt").is_file()
    logged = [event[2] for event in _Reporter.instances[-1].events if event[0] == "log"]
    merged = {key for metrics in logged for key in metrics}
    assert "train/effective_cross_entropy_nats_per_byte" in merged
    assert "optimization/relative_parameter_norm_change" in merged
    assert "optimization/gradient_norm_after_clip" in merged
    assert "optimization/gradient_clip_coefficient" in merged
    assert "architecture/state_rms_max" in merged
    assert "train/state_energy_regularization" in merged
    assert "architecture/branch_resonance" in merged
    assert _Reporter.instances[-1].events[-1] == ("finish",)

    resumed = _tiny_trainer(tmp_path / "second", monkeypatch, total_tokens=12)
    resumed.load_checkpoint(latest)
    assert resumed.state.step == 1 and resumed.state.tokens_seen == 8
    final = resumed.train()
    assert final.tokens_seen == 12 and _Reporter.instances[-1].resume is True


def test_trainer_checkpoint_and_constructor_fail_closed(tmp_path, monkeypatch):
    trainer = _tiny_trainer(tmp_path / "run", monkeypatch)
    checkpoint = trainer.save_checkpoint()
    payload = torch.load(checkpoint, weights_only=True)

    legacy = deepcopy(payload)
    for key in (
        "precision", "gradient_recovery", "gradient_backoff_factor",
        "gradient_recovery_limit",
    ):
        legacy["identity"]["training"].pop(key, None)
    legacy["training_state"].pop("learning_rate_scale", None)
    legacy["training_state"].pop("gradient_recoveries", None)
    legacy_path = tmp_path / "legacy-stability.pt"
    torch.save(legacy, legacy_path)
    restored_legacy = _tiny_trainer(tmp_path / "legacy", monkeypatch)
    restored_legacy.load_checkpoint(legacy_path)
    assert restored_legacy.state.learning_rate_scale == 1
    assert restored_legacy.state.gradient_recoveries == 0

    wrong_version = tmp_path / "wrong-version.pt"
    torch.save(dict(payload, format_version=99), wrong_version)
    with pytest.raises(ValueError, match="version"):
        _tiny_trainer(tmp_path / "v", monkeypatch).load_checkpoint(wrong_version)

    missing_identity = tmp_path / "missing-identity.pt"
    torch.save(dict(payload, identity=None), missing_identity)
    with pytest.raises(ValueError, match="identity"):
        _tiny_trainer(tmp_path / "i", monkeypatch).load_checkpoint(missing_identity)

    mismatch = dict(payload)
    mismatch["identity"] = dict(payload["identity"], model_parameters=0)
    mismatch_path = tmp_path / "mismatch.pt"
    torch.save(mismatch, mismatch_path)
    with pytest.raises(ValueError, match="does not match"):
        _tiny_trainer(tmp_path / "m", monkeypatch).load_checkpoint(mismatch_path)

    exhausted = dict(payload)
    exhausted["training_state"] = dict(payload["training_state"], tokens_seen=8)
    exhausted_path = tmp_path / "exhausted.pt"
    torch.save(exhausted, exhausted_path)
    with pytest.raises(ValueError, match="exhausted"):
        _tiny_trainer(tmp_path / "e", monkeypatch).load_checkpoint(exhausted_path)

    tokenizer = ByteTextTokenizer()
    model = MRRNLanguageModel(tiny_language_config(tokenizer.vocabulary_size))
    stream = PackedTokenStream(SequenceTextSource(("text",)), tokenizer)
    batch = build_evaluation_batches(
        PackedTokenStream(SequenceTextSource(("eval",)), tokenizer),
        count=1, batch_size=1, sequence_length=4,
    )
    config = LMTrainingConfig(
        output_dir=str(tmp_path), total_tokens=4, sequence_length=4, evaluation_batches=1,
    )
    with pytest.raises(ValueError, match="vocabulary"):
        NextTokenTrainer(model, SimpleNamespace(vocabulary_size=2), stream, batch, config)
    with pytest.raises(ValueError, match="batch count"):
        NextTokenTrainer(model, tokenizer, stream, (), config)
    with pytest.raises(ValueError, match="batch and sequence"):
        NextTokenTrainer(model, tokenizer, stream, (
            PackedBatch(torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.long),
                        torch.ones(1, 2, dtype=torch.long)),
        ), config)


def test_device_and_memory_helpers(monkeypatch):
    assert _device_for("cpu").type == "cpu"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert _device_for("auto").type == "cuda"
    assert _device_for("cuda:1") == torch.device("cuda:1")
    with pytest.raises(RuntimeError, match="only 2"):
        _device_for("cuda:2")
    assert _precision_for(torch.device("cuda"), "auto") == torch.bfloat16
    assert _precision_for(torch.device("cuda"), "fp16") == torch.float16
    assert _precision_for(torch.device("cuda"), "fp32") is None
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert _precision_for(torch.device("cuda"), "auto") == torch.float16
    with pytest.raises(RuntimeError, match="does not support"):
        _precision_for(torch.device("cuda"), "bf16")
    with pytest.raises(RuntimeError, match="only on CUDA"):
        _precision_for(torch.device("cpu"), "fp16")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _device_for("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="MPS"):
        _device_for("mps")
    with pytest.raises(RuntimeError, match="CUDA"):
        _device_for("cuda")
    metrics = _memory_metrics(torch.device("cpu"))
    assert "system/process_rss_gib" in metrics and "system/system_memory_percent" in metrics
    with pytest.raises(ValueError):
        build_evaluation_batches(
            PackedTokenStream(SequenceTextSource(("x",)), ByteTextTokenizer()),
            count=0, batch_size=1, sequence_length=1,
        )


def test_cuda_runtime_configuration_and_manifest_details(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: calls.append(device))
    monkeypatch.setattr(
        torch, "set_float32_matmul_precision", lambda value: calls.append(value)
    )
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            name="Test CUDA GPU", major=8, minor=6, total_memory=20 * 2**30
        ),
    )
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 9000)

    _configure_cuda(torch.device("cpu"))
    assert calls == []
    device = torch.device("cuda:0")
    _configure_cuda(device)
    assert calls == [device, "high"]
    assert torch.backends.cuda.matmul.allow_tf32
    assert torch.backends.cudnn.allow_tf32 and torch.backends.cudnn.benchmark
    details = _runtime_details(device, torch.bfloat16)
    assert details["gpu_name"] == "Test CUDA GPU"
    assert details["gpu_compute_capability"] == "8.6"
    assert details["gpu_memory_gib"] == 20
    assert details["precision"] == "bfloat16"
    assert details["fused_adamw"] and details["pinned_transfers"]


def test_state_regularizer_and_stability_metrics_are_finite_and_bounded():
    tokenizer = ByteTextTokenizer()
    model = MRRNLanguageModel(tiny_language_config(tokenizer.vocabulary_size))
    output = model(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
    mean_rms, maximum_rms = resonator_state_rms(output)
    assert 0 <= mean_rms <= maximum_rms
    assert resonator_state_regularization(output, target_rms=100) == 0
    assert resonator_state_regularization(output, target_rms=1e-6) > 0
    metrics = stability_metrics(output)
    assert metrics["architecture/state_rms_max"] == pytest.approx(float(maximum_rms.detach()))
    assert 0 <= metrics["architecture/branch_resonance"] <= 1
    with pytest.raises(ValueError):
        resonator_state_regularization(output, target_rms=0)


def test_persistent_stability_guard_aborts_before_unsafe_update_and_writes_evidence(
    tmp_path, monkeypatch
):
    trainer = _tiny_trainer(tmp_path, monkeypatch)
    trainer.config = LMTrainingConfig(
        output_dir=str(tmp_path), total_tokens=8, sequence_length=4,
        micro_batch_size=1, gradient_accumulation_steps=2, warmup_tokens=4,
        evaluation_batches=1, device="cpu", show_dashboard=False,
        state_target_rms=1, state_warning_rms=2, state_abort_rms=3,
        gradient_warning_norm=2, gradient_abort_norm=3, stability_patience=2,
    )
    reporter = _Reporter(trainer.config, {}, resume=False)
    gradient = SimpleNamespace(
        total_before_clip=torch.tensor(4.0), clip_coefficient=torch.tensor(0.25)
    )
    stability = {"architecture/state_rms_max": 4.0}
    trainer._enforce_stability(reporter, gradient, stability)
    with pytest.raises(FloatingPointError, match="stability guard"):
        trainer._enforce_stability(reporter, gradient, stability)
    evidence = tmp_path / "stability-abort-step-0000001.json"
    assert evidence.is_file()
    record = json.loads(evidence.read_text())
    assert record["candidate_step_not_applied"] == 1
    assert any(event[2] == "Stability guard aborted update" for event in reporter.events)


def test_finite_clipped_gradient_pressure_backs_off_before_abort(tmp_path, monkeypatch):
    trainer = _tiny_trainer(tmp_path, monkeypatch)
    trainer.config = LMTrainingConfig(
        output_dir=str(tmp_path), total_tokens=8, sequence_length=4,
        micro_batch_size=1, gradient_accumulation_steps=2, warmup_tokens=4,
        evaluation_batches=1, device="cpu", show_dashboard=False,
        state_target_rms=1, state_warning_rms=2, state_abort_rms=3,
        gradient_warning_norm=2, gradient_abort_norm=3, stability_patience=2,
        gradient_backoff_factor=0.5, gradient_recovery_limit=1,
    )
    reporter = _Reporter(trainer.config, {}, resume=False)
    gradient = SimpleNamespace(
        total_before_clip=torch.tensor(4.0), clip_coefficient=torch.tensor(0.25)
    )
    stability = {"architecture/state_rms_max": 1.0}
    rates = [group["lr"] for group in trainer.optimizer.param_groups]

    trainer._enforce_stability(reporter, gradient, stability)
    trainer._enforce_stability(reporter, gradient, stability)
    assert trainer.state.learning_rate_scale == 0.5
    assert trainer.state.gradient_recoveries == 1
    assert trainer._safety_checkpoint_pending
    assert [group["lr"] for group in trainer.optimizer.param_groups] == pytest.approx(
        [rate * 0.5 for rate in rates]
    )
    evidence = tmp_path / "gradient-recovery-step-0000001.json"
    assert json.loads(evidence.read_text())["action"] == (
        "apply_clipped_update_with_learning_rate_backoff"
    )
    assert any(event[2] == "Gradient-pressure recovery" for event in reporter.events)

    trainer.optimizer.step()
    trainer._step_scheduler()
    assert [group["lr"] for group in trainer.optimizer.param_groups] == pytest.approx(
        [rate * trainer.state.learning_rate_scale for rate in trainer.scheduler.get_last_lr()]
    )

    trainer._enforce_stability(reporter, gradient, stability)
    with pytest.raises(FloatingPointError, match="recoveries were exhausted"):
        trainer._enforce_stability(reporter, gradient, stability)


def test_training_logs_recovery_metrics_and_unscheduled_safety_checkpoint(
    tmp_path, monkeypatch
):
    _Reporter.instances.clear()
    trainer = _tiny_trainer(tmp_path, monkeypatch)
    trainer.config = replace(
        trainer.config,
        maximum_gradient_norm=0.001,
        gradient_warning_norm=0.002,
        gradient_abort_norm=0.003,
        stability_patience=1,
        gradient_recovery_limit=2,
        evaluation_interval=100,
        checkpoint_interval=100,
    )
    state = trainer.train()
    assert state.gradient_recoveries == 1
    assert state.learning_rate_scale == 0.5
    events = _Reporter.instances[-1].events
    assert any(event[0] == "alert" and event[2] == "Safety checkpoint saved" for event in events)
    train_metrics = next(
        event[2] for event in events
        if event[0] == "log" and "optimization/learning_rate_scale" in event[2]
    )
    assert train_metrics["optimization/learning_rate_scale"] == 0.5
    assert train_metrics["optimization/gradient_recoveries"] == 1
    assert (tmp_path / "checkpoints" / "step-0000001.pt").is_file()
