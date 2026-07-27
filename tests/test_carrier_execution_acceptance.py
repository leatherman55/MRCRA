from mrrn.carrier_execution_acceptance import (
    run_carrier_execution_acceptance,
)


def test_carrier_execution_acceptance_is_complete_finite_and_fail_closed():
    report = run_carrier_execution_acceptance()
    names = {item.name for item in report.criteria}
    assert {
        "affine_scan_forward_error",
        "affine_scan_gradient_error",
        "affine_scan_saved_tensor_byte_ratio",
        "simplex_residual_forward_error",
        "simplex_residual_gradient_error",
        "simplex_residual_autograd_node_ratio",
        "coarse_checkpoint_forward_error",
        "coarse_checkpoint_input_gradient_error",
        "coarse_checkpoint_continuation_state_error",
        "document_target_bijection",
            "document_physical_invocation_ratio",
            "document_padding_efficiency",
            "document_cost_ratio_vs_exact_signature",
        }.issubset(names)
    assert all(item.passed for item in report.criteria)
    assert report.passed
    assert (
        report.telemetry["scan_optimized_saved_bytes"]
        < report.telemetry["scan_reference_saved_bytes"]
    )
    assert (
        report.telemetry["document_physical_invocations"]
        < report.telemetry["document_logical_spans"]
    )
    assert report.telemetry["checkpoint_granularity"] == "whole_carrier_span"
