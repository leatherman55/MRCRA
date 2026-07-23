from __future__ import annotations
import json
from pathlib import Path
import subprocess, sys
import pytest

from mrrn.performance_acceptance import run_performance_acceptance


@pytest.fixture(scope="module")
def report():
    return run_performance_acceptance()


def test_relative_and_structural_performance_budgets_pass(report):
    assert report.passed
    assert not report.absolute_target_hardware_throughput_tested
    assert len(report.results) == 6 and all(result.passed for result in report.results)
    assert report.telemetry["checkpoint_embeds_model_weights"] is False
    assert report.telemetry["runtime_declared_tensor_bytes"] > 0
    assert report.torch_threads == 1
    assert report.telemetry["latency_clock"] == "process_time"
    assert report.telemetry["latency_arm_observations_per_repeat"] == 2
    assert report.telemetry["dormant_paired_overhead_mad_percent"] >= 0
    assert report.telemetry["cycle_paired_overhead_mad_percent"] >= 0
    assert report.telemetry["complete_state_prefill_median_ms"] > 0
    assert report.telemetry["tokenwise_transition_median_ms"] > 0
    assert "unclaimed" in report.claim_boundary


def test_performance_report_is_json_serializable(report):
    restored = json.loads(json.dumps(report.to_dict()))
    assert restored["passed"] and restored["suite"].endswith("v3")


def test_performance_acceptance_rejects_invalid_controls():
    with pytest.raises(ValueError, match="odd repeat"):
        run_performance_acceptance(repeats=20)
    with pytest.raises(ValueError, match="requires CPU"):
        run_performance_acceptance(device="mps")


def test_performance_cli_writes_artifact(tmp_path: Path):
    output = tmp_path / "performance.json"
    completed = subprocess.run(
        [sys.executable, "scripts/run_mrcra_performance_acceptance.py", "--output", str(output)],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(output.read_text())["passed"]
