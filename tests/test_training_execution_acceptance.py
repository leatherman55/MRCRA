from __future__ import annotations

from dataclasses import replace

import pytest

from mrrn.training_execution_acceptance import (
    COMPILED_VARIANT,
    PRODUCTION_REQUIRED_VARIANTS,
    PRODUCTION_VARIANTS,
    CompilerCandidateReceipt,
    TrainingExecutionSample,
    build_acceptance_report,
)


def sample(variant: str, *, rate: float) -> TrainingExecutionSample:
    phase_metrics = {
        "performance/primary_forward_seconds": 0.1,
        "performance/loss_forward_seconds": 0.1,
        "performance/primary_backward_seconds": 0.2,
        "cstm/predictor_backward_seconds": 0.01,
        "cstm/substrate_backward_seconds": 0.02,
        "cstm/gradient_merge_seconds": 0.01,
        "performance/gradient_reduction_seconds": 0.01,
        "performance/optimizer_seconds": 0.01,
        "performance/unattributed_step_seconds": 0.01,
    }
    metrics = {
        "document_batching/target_bijection": 1.0,
        "cstm/substrate_vjp_count": 1.0,
        "train/cross_entropy_nats_per_token": 4.0,
        **phase_metrics,
    }
    return TrainingExecutionSample(
        variant=variant,
        profile="quick",
        parameter_count=100,
        context_length=128,
        steps=1,
        initialization_seconds=0.1,
        training_seconds=128 / rate,
        tokens_per_second=rate,
        peak_rss_bytes=1,
        metrics=metrics,
        runtime={
            "activation_execution_policy_digest": "a" * 64,
            "compiled_tensor_cores": variant == COMPILED_VARIANT,
            "carrier_compiler_backend": (
                "aot_eager"
                if variant == COMPILED_VARIANT else "none"
            ),
        },
        source_run_id="1" * 64,
        model_state_digest="2" * 64,
        optimizer_state_digest="6" * 64,
        tokenizer_identity_digest="3" * 64,
        fixture_digest="4" * 64,
        hardware_fingerprint="5" * 64,
        torch_version="test",
        metric_keys=tuple(sorted(metrics)),
        resolved_variant=variant,
    )


def compiler_receipt(
    *,
    outcome: str = "executed",
) -> CompilerCandidateReceipt:
    return CompilerCandidateReceipt(
        requested_variant=COMPILED_VARIANT,
        profile="quick",
        requested_backend="aot_eager",
        outcome=outcome,
        resolved_variant=(
            COMPILED_VARIANT
            if outcome == "executed"
            else "static_cost_model_auto_repaired_cstm"
        ),
        wall_clock_seconds=10.0,
        timeout_seconds=10.0,
        stdout_sha256="7" * 64,
        stderr_sha256="8" * 64,
    )


def test_acceptance_requires_speed_correctness_and_bounded_substrate_vjp():
    report = build_acceptance_report((
        sample("legacy_reference", rate=100),
        sample("repaired", rate=120),
    ))
    assert report.passed
    assert report.format_version == 3
    assert all(item.passed for item in report.criteria)
    assert all(item.unit for item in report.criteria)


def test_acceptance_fails_when_target_authority_or_speed_regresses():
    repaired = sample("repaired", rate=90)
    repaired = replace(
        repaired,
        metrics={
            **repaired.metrics,
            "document_batching/target_bijection": 0.0,
        },
    )
    report = build_acceptance_report((
        sample("legacy_reference", rate=100),
        repaired,
    ))
    assert not report.passed
    assert {
        item.name for item in report.criteria if not item.passed
    } == {"repaired_speedup", "target_bijection"}


def test_acceptance_rejects_unmatched_or_missing_variants():
    with pytest.raises(ValueError, match="requires"):
        build_acceptance_report((sample("repaired", rate=100),))
    with pytest.raises(ValueError, match="matched"):
        build_acceptance_report((
            sample("legacy_reference", rate=100),
            replace(sample("repaired", rate=120), context_length=256),
        ))


def test_complete_named_variant_matrix_uses_profile_appropriate_gates():
    rates = {
        "legacy_serial_checkpoint_dense_cstm": 100,
        "static_coarse_checkpoint_ce": 140,
        "static_coarse_checkpoint_dense_cstm": 120,
        "static_auto_ce": 175,
        "static_auto_repaired_cstm": 160,
        "static_cost_model_auto_repaired_cstm": 150,
        "compiled_cost_model_auto_repaired_cstm": 180,
    }
    samples = tuple(
        replace(
            sample(name, rate=rates[name]),
            raw_step_seconds=(128 / rates[name],),
            median_step_seconds=128 / rates[name],
            minimum_step_seconds=128 / rates[name],
            maximum_step_seconds=128 / rates[name],
            median_absolute_deviation_seconds=0.0,
            step_metrics=(
                {
                    **sample(name, rate=rates[name]).metrics,
                    "document_batching/padding_efficiency": 0.9,
                    "document_batching/estimated_savings_fraction": 0.1,
                },
            ),
            metrics={
                **sample(name, rate=rates[name]).metrics,
                "document_batching/padding_efficiency": 0.9,
                "document_batching/estimated_savings_fraction": 0.1,
            },
        )
        for name in PRODUCTION_VARIANTS
    )
    report = build_acceptance_report(
        samples, compiler_candidate=compiler_receipt()
    )
    assert report.passed
    assert {sample.variant for sample in report.samples} == set(
        PRODUCTION_VARIANTS
    )
    assert all(item.unit for item in report.criteria)


def test_cost_aware_throughput_can_authorize_subthreshold_padding():
    rates = {
        "legacy_serial_checkpoint_dense_cstm": 100,
        "static_coarse_checkpoint_ce": 140,
        "static_coarse_checkpoint_dense_cstm": 120,
        "static_auto_ce": 150,
        "static_auto_repaired_cstm": 145,
        "static_cost_model_auto_repaired_cstm": 155,
        "compiled_cost_model_auto_repaired_cstm": 150,
    }
    samples = []
    for name in PRODUCTION_VARIANTS:
        base = sample(name, rate=rates[name])
        metrics = {
            **base.metrics,
            "document_batching/padding_efficiency": 0.65,
            "document_batching/estimated_savings_fraction": 0.0,
        }
        samples.append(replace(
            base,
            raw_step_seconds=(128 / rates[name],),
            median_step_seconds=128 / rates[name],
            minimum_step_seconds=128 / rates[name],
            maximum_step_seconds=128 / rates[name],
            median_absolute_deviation_seconds=0.0,
            metrics=metrics,
            step_metrics=(metrics,),
            metric_keys=tuple(sorted(metrics)),
        ))
    report = build_acceptance_report(
        tuple(samples), compiler_candidate=compiler_receipt()
    )
    criterion = next(
        item for item in report.criteria
        if item.name == "padding_or_measured_cost_advantage"
    )
    assert criterion.passed
    assert criterion.measurement == pytest.approx(155 / 145)


def test_timeout_receipt_truthfully_rejects_compiler_and_accepts_eager_matrix():
    rates = {
        "legacy_serial_checkpoint_dense_cstm": 100,
        "static_coarse_checkpoint_ce": 140,
        "static_coarse_checkpoint_dense_cstm": 120,
        "static_auto_ce": 150,
        "static_auto_repaired_cstm": 145,
        "static_cost_model_auto_repaired_cstm": 155,
    }
    samples = []
    for name in PRODUCTION_REQUIRED_VARIANTS:
        base = sample(name, rate=rates[name])
        metrics = {
            **base.metrics,
            "document_batching/padding_efficiency": 0.9,
            "document_batching/estimated_savings_fraction": 0.1,
        }
        samples.append(replace(
            base,
            raw_step_seconds=(128 / rates[name],),
            median_step_seconds=128 / rates[name],
            minimum_step_seconds=128 / rates[name],
            maximum_step_seconds=128 / rates[name],
            median_absolute_deviation_seconds=0.0,
            metrics=metrics,
            step_metrics=(metrics,),
            metric_keys=tuple(sorted(metrics)),
        ))
    report = build_acceptance_report(
        tuple(samples),
        compiler_candidate=compiler_receipt(outcome="timeout"),
    )
    assert report.passed
    assert report.compiler_candidate is not None
    assert report.compiler_candidate.outcome == "timeout"
    assert next(
        criterion
        for criterion in report.criteria
        if criterion.name
        == "compiler_candidate_bounded_and_truthfully_resolved"
    ).passed


def test_production_matrix_requires_valid_compiler_candidate_receipt():
    samples = tuple(
        sample(name, rate=100)
        for name in PRODUCTION_REQUIRED_VARIANTS
    )
    with pytest.raises(ValueError, match="compiler candidate receipt"):
        build_acceptance_report(samples)
    with pytest.raises(ValueError, match="malformed"):
        CompilerCandidateReceipt(
            requested_variant=COMPILED_VARIANT,
            profile="quick",
            requested_backend="aot_eager",
            outcome="timeout",
            resolved_variant=COMPILED_VARIANT,
            wall_clock_seconds=1.0,
            timeout_seconds=10.0,
            stdout_sha256="7" * 64,
            stderr_sha256="8" * 64,
        )
