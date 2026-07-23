from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from mrrn.integrated_acceptance import run_integrated_acceptance


@pytest.fixture(scope="module")
def report():
    return run_integrated_acceptance()


def test_all_production_path_matched_ablations_pass(report):
    assert report.format_version == 2
    assert report.passed and report.maturity == "integrated_loop"
    assert not report.serious_scale_capability_tested
    assert not report.open_domain_transfer_tested
    assert report.checkpoint_digest is None
    assert "remains unresolved" in report.checkpoint_status
    assert report.unresolved_external_gates
    assert not report.failures
    assert len(report.results) == 15
    assert all(result.passed for result in report.results)
    assert all(result.confidence_low >= result.minimum_success_rate for result in report.results)
    assert all(result.successes == result.trials == 16 for result in report.results)
    assert all(result.examples_per_arm >= result.trials for result in report.results)
    assert all(len(result.split_sha256) == 64 for result in report.results)
    assert all(result.paired_compute_seconds == result.duration_seconds for result in report.results)
    assert {result.name for result in report.results} == {
        "spectral_phase_delay_information", "reconstruction_trace_conditioning",
        "evidence_conditioned_reconstruction", "explicit_reconstructed_source_class",
        "adaptive_abstraction_selection", "posterior_multi_hypothesis_deliberation",
        "information_gain_deliberation", "post_deliberation_action_selection",
        "hard_viability_authorization", "role_normalized_invariant_transfer",
        "metacognitive_operation_routing", "authorized_cross_context_persistence",
        "learned_evidential_memory_write", "functional_surprise_consequence_learning",
        "provenance_feature_ablation",
    }
    assert "does not prove" in report.claim_boundary


def test_integrated_report_is_json_serializable(report):
    restored = json.loads(json.dumps(report.to_dict()))
    assert restored["passed"] is True
    assert len(restored["source_digest"]) == 64
    assert restored["source_sha256"]
    assert all(len(value) == 64 for value in restored["source_sha256"].values())
    assert restored["exact_test_node_ids"] == list(report.exact_test_node_ids)
    assert restored["hardware"]["logical_cpu_count"] >= 1
    consequence = next(
        result for result in restored["results"]
        if result["name"] == "functional_surprise_consequence_learning"
    )
    assert consequence["interactions_per_arm"] == 16 * 45 * 128
    assert consequence["optimization_steps_per_arm"] == 16 * 45


def test_integrated_acceptance_rejects_weak_or_nonreproducible_controls():
    with pytest.raises(ValueError, match="eight unique"):
        run_integrated_acceptance(seeds=(1, 2))
    with pytest.raises(ValueError, match="CPU-authoritative"):
        run_integrated_acceptance(device="mps")


def test_integrated_acceptance_cli_writes_artifact(tmp_path: Path):
    output = tmp_path / "integrated.json"
    completed = subprocess.run(
        [sys.executable, "scripts/run_mrcra_integrated_acceptance.py", "--output", str(output)],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    artifact = json.loads(output.read_text())
    assert artifact["passed"] and len(artifact["results"]) == 15
