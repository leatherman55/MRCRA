"""Acceptance-level proof that CSTM's production claims are actually measured."""

import json
from math import isfinite

from mrrn.cstm_acceptance import (
    benchmark_efficiency_and_parameter_contract,
    benchmark_gradient_governance_and_accounting,
    benchmark_integrated_causality,
    benchmark_predictor_learning,
    benchmark_target_authority,
    run_cstm_acceptance,
)


def test_target_authority_experiment_proves_math_order_and_boundary_contracts():
    metrics = benchmark_target_authority().metrics
    assert metrics["direct_dft_max_abs_error"] <= 1e-6
    assert metrics["permutation_dc_max_abs_change"] <= 1e-6
    assert metrics["permutation_harmonic_rms_change"] >= 1e-3
    assert metrics["cross_boundary_valid_rows"] == 0


def test_real_low_rank_predictor_learns_fixed_multihorizon_targets():
    metrics = benchmark_predictor_learning().metrics
    assert metrics["valid_target_rows"] > 0
    assert metrics["token_participations"] > metrics["valid_target_rows"]
    assert metrics["final_standardized_huber"] < metrics["initial_standardized_huber"]
    assert metrics["final_to_initial_ratio"] <= 0.35


def test_integrated_carrier_cognition_and_head_are_strictly_causal():
    metrics = benchmark_integrated_causality().metrics
    assert metrics["past_rows_compared"] > 0
    assert metrics["past_prediction_max_abs_change"] <= 1e-9
    assert metrics["future_prediction_max_abs_change"] >= 1e-5


def test_production_geometric_work_and_ultralight_parameter_bounds_hold():
    metrics = benchmark_efficiency_and_parameter_contract().metrics
    assert metrics["target_rows_per_physical_token"] <= 2
    assert metrics["cstm_predictor_parameters"] <= 5_000
    assert (
        metrics["ultralight_actor_parameters"]
        <= metrics["ultralight_parameter_maximum"]
    )


def test_real_trainer_projects_conflicts_caps_subsystems_and_preserves_token_count():
    metrics = benchmark_gradient_governance_and_accounting().metrics
    assert metrics["auxiliary_applied"] == 1
    assert metrics["maximum_subsystem_cap_ratio"] <= 1.00001
    assert metrics["minimum_task_auxiliary_alignment"] >= -1e-7
    assert metrics["governed_overlap_subsystems"] >= 1
    assert metrics["cstm_head_gradient_norm"] > 0
    assert metrics["spectral_target_views"] > 0
    assert metrics["physical_token_counter_delta"] == 0
    assert metrics["packed_physical_tokens"] == 32


def test_complete_cstm_acceptance_report_passes_every_named_gate():
    report = run_cstm_acceptance()
    assert report.passed
    assert len(report.experiments) == 5
    assert len(report.criteria) >= 16
    assert all(criterion.passed for criterion in report.criteria)
    payload = report.to_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["schema_version"] == 1
    assert payload["passed"]
    assert all(item["passed"] for item in payload["criteria"])
    assert all(
        isfinite(value)
        for experiment in payload["experiments"]
        for value in experiment["metrics"].values()
    )
    assert "does not claim downstream language-quality improvement" in (
        payload["claim_boundary"]
    )
