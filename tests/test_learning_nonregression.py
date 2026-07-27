from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from mrrn.learning_nonregression import (
    LEARNING_VARIANTS,
    LearningObservation,
    LearningRun,
    build_learning_nonregression_report,
    learning_study_controls_digest,
    learning_study_journal_payload,
    restore_learning_study_journal,
)


def observation(step: int, *, carrier: float = 0.0, cognition: float = 0.0):
    return LearningObservation(
        step=step,
        physical_tokens=step * 1_024,
        wall_clock_seconds=float(step),
        train_ce_nats_per_token=5.0 - step * 0.1,
        cstm_standardized_huber=0.2,
        carrier_auxiliary_norm_after=carrier,
        cognition_auxiliary_norm_after=cognition,
        gradient_clip_coefficient=0.8,
        state_rms_max=0.3,
        feedback_rms_max=0.1,
        cognitive_cycles=4,
        events=1,
    )


def run(variant: str, seed: int, ce: float) -> LearningRun:
    sampled = variant == "sampled"
    return LearningRun(
        variant=variant,
        seed=seed,
        physical_tokens=2_048,
        eval_ce_nats_per_token=ce,
        eval_ece_nats_per_byte=ce / 2,
        cstm_standardized_huber=0.2 if variant != "ce_only" else 0.0,
        carrier_auxiliary_participation=sampled,
        cognition_auxiliary_participation=sampled,
        gradient_clip_frequency=0.5,
        state_rms_max=0.3,
        feedback_rms_max=0.1,
        cognitive_cycles=8,
        events=2,
        training_seconds=2.0,
        observations=(
            observation(1, carrier=float(sampled), cognition=float(sampled)),
            observation(2, carrier=float(sampled), cognition=float(sampled)),
        ),
        finite=True,
        checkpoint_resumable=True,
    )


def matrix(sampled_delta: float = 0.01):
    rows = []
    for seed in (17, 29, 43):
        for variant in LEARNING_VARIANTS:
            ce = 4.0 + seed * 1e-4
            if variant == "sampled":
                ce += sampled_delta
            elif variant == "ce_only":
                ce += 0.03
            rows.append(run(variant, seed, ce))
    return tuple(rows)


def test_learning_nonregression_builds_paired_three_seed_confidence_report():
    report = build_learning_nonregression_report(matrix())
    assert report.passed
    assert report.mean_difference == pytest.approx(0.01)
    assert len(report.paired_sampled_minus_legacy) == 3
    assert report.confidence_interval_95[0] == pytest.approx(0.01)
    assert report.confidence_interval_95[1] == pytest.approx(0.01)
    assert report.to_dict()["runs"][0]["observations"][0]["physical_tokens"] == 1_024


def test_learning_nonregression_rejects_regression_and_missing_participation():
    regressed = build_learning_nonregression_report(matrix(0.03))
    assert not regressed.passed
    rows = list(matrix())
    target = next(
        index
        for index, item in enumerate(rows)
        if item.variant == "sampled"
    )
    value = rows[target]
    rows[target] = replace(
        value,
        carrier_auxiliary_participation=False,
    )
    assert not build_learning_nonregression_report(tuple(rows)).passed


@pytest.mark.parametrize(
    "mutator",
    (
        lambda rows: rows[:-1],
        lambda rows: rows + (rows[0],),
        lambda rows: rows[:-1] + (
            replace(
                rows[-1],
                physical_tokens=3_072,
                observations=rows[-1].observations + (observation(3),),
            ),
        ),
    ),
)
def test_learning_nonregression_fails_closed_on_unmatched_matrix(mutator):
    with pytest.raises(ValueError):
        build_learning_nonregression_report(mutator(matrix()))


def test_learning_observation_rejects_nonmonotonic_curve():
    value = run("sampled", 1, 4.0)
    with pytest.raises(ValueError):
        replace(
            value,
            observations=(
                value.observations[1],
                value.observations[0],
            ),
        )


def test_learning_journal_exactly_restores_completed_fresh_process_arms():
    controls = learning_study_controls_digest({
        "profile": "quick",
        "steps": 2,
        "seeds": [17, 29, 43],
    })
    authority = "a" * 64
    rows = matrix()[:4]
    payload = learning_study_journal_payload(
        profile="quick",
        controls_digest=controls,
        authority_digest=authority,
        runs=rows,
        complete=False,
    )
    restored = restore_learning_study_journal(
        payload,
        profile="quick",
        controls_digest=controls,
        authority_digest=authority,
    )
    assert restored == rows
    assert restore_learning_study_journal(
        payload,
        profile="quick",
        controls_digest="b" * 64,
        authority_digest=authority,
    ) == ()


def test_learning_journal_rejects_duplicate_and_corrupt_matching_evidence():
    controls = "c" * 64
    authority = "d" * 64
    with pytest.raises(ValueError, match="authority"):
        learning_study_journal_payload(
            profile="quick",
            controls_digest=controls,
            authority_digest=authority,
            runs=(matrix()[0], matrix()[0]),
            complete=False,
        )
    malformed = {
        "schema_version": 1,
        "profile": "quick",
        "controls_digest": controls,
        "authority_digest": authority,
        "complete": False,
        "runs": [{"variant": "sampled"}],
    }
    with pytest.raises(ValueError, match="journal"):
        restore_learning_study_journal(
            malformed,
            profile="quick",
            controls_digest=controls,
            authority_digest=authority,
        )


@pytest.mark.parametrize(
    ("failure", "return_code", "expected_stream"),
    (
        (False, 0, "durable-result"),
        (True, 1, "deliberate-worker-failure"),
    ),
)
def test_fineweb_worker_bypasses_only_arrow_finalizers_and_preserves_status(
    failure, return_code, expected_stream,
):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/run_mrcra_learning_nonregression.py"
    replacement = (
        "lambda: (_ for _ in ()).throw("
        "RuntimeError('deliberate-worker-failure'))"
        if failure
        else "lambda: print('durable-result', flush=True)"
    )
    code = f"""
import importlib.util
import sys
import types
spec = importlib.util.spec_from_file_location("learning_runner", {str(script)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.main = {replacement}
module.sys.platform = "darwin"
module.sys.modules["pyarrow"] = types.ModuleType("pyarrow")
module._run_cli_without_arrow_finalizer_deadlock()
raise RuntimeError("os._exit boundary did not execute")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == return_code
    combined = completed.stdout + completed.stderr
    assert expected_stream in combined
    assert "boundary did not execute" not in combined
