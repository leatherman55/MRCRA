from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from mrrn.pc_rasl_acceptance import (
    PCRASLCriterion, run_pc_rasl_acceptance,
)


@pytest.fixture(scope="module")
def report():
    return run_pc_rasl_acceptance()


def test_pc_rasl_empirical_acceptance_passes_all_preregistered_gates(report):
    assert report.passed
    assert report.format_version == 2
    assert report.phase_transition_metrics_used_as_authority is False
    assert len(report.results) == 3
    assert all(result.passed for result in report.results)
    assert all(
        criterion.evaluate(result.metrics)
        for result in report.results
        for criterion in result.criteria
    )
    assert report.exact_trainer_resume_test_node.endswith(
        "test_pc_rasl_production_path_is_checkpoint_resume_exact"
    )
    assert len(report.causal_replay_test_nodes) == 2
    assert report.causal_replay_test_nodes[1].endswith(
        "test_replay_critic_uses_preconsequence_behavior_evidence_not_later_reanalysis"
    )
    assert len(report.checkpoint_migration_test_nodes) == 2
    assert report.production_resource_test_node.endswith(
        "test_lightmodel_pc_rasl_is_a_compact_nonduplicating_production_learner"
    )


def test_pc_rasl_acceptance_is_json_serializable(report):
    restored = json.loads(json.dumps(report.to_dict()))
    assert restored["passed"] is True
    assert restored["suite"] == "progress-conditioned-rasl-empirical-v2"
    assert "phase transition" in restored["claim_boundary"]


def test_pc_rasl_criterion_fails_closed():
    minimum = PCRASLCriterion("value", 1.0, "at_least")
    maximum = PCRASLCriterion("value", 1.0, "at_most")
    assert minimum.evaluate({"value": 1.0})
    assert maximum.evaluate({"value": 1.0})
    assert not minimum.evaluate({})
    assert not maximum.evaluate({"value": float("nan")})
    invalid = PCRASLCriterion("value", 0.0, "sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown criterion"):
        invalid.evaluate({"value": 1.0})


def test_pc_rasl_acceptance_rejects_nonportable_controls():
    with pytest.raises(ValueError, match="requires CPU"):
        run_pc_rasl_acceptance(device="mps")
    with pytest.raises(ValueError, match="nonnegative"):
        run_pc_rasl_acceptance(seed=-1)


def test_pc_rasl_acceptance_cli_writes_authoritative_artifact(tmp_path):
    output = tmp_path / "pc-rasl.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_pc_rasl_acceptance.py",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "overall=PASS" in completed.stdout
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["passed"] is True
    assert len(artifact["results"]) == 3
