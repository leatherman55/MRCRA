from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from mrrn.empirical_acceptance import (
    BENCHMARKS, EmpiricalCriterion, run_empirical_acceptance,
)


@pytest.fixture(scope="module")
def empirical_report():
    return run_empirical_acceptance(seed=17)


def test_complete_empirical_suite_passes_preregistered_gates(empirical_report) -> None:
    assert empirical_report.passed
    assert empirical_report.format_version == 2
    assert empirical_report.maturity == "mechanism"
    assert empirical_report.serious_scale_capability_tested is False
    assert empirical_report.physical_cuda_tested is False
    assert len(empirical_report.results) == len(BENCHMARKS) == 8
    assert {result.gate for result in empirical_report.results} == {
        "G", "H", "I", "J", "K", "L", "Stage 4", "Stage 9",
    }
    for result in empirical_report.results:
        assert result.passed
        assert result.duration_seconds >= 0
        assert result.trainable_parameters > 0
        assert result.scope
        assert result.maturity == "mechanism"
        assert all(criterion.evaluate(result.metrics) for criterion in result.criteria)


def test_empirical_report_is_json_serializable(empirical_report) -> None:
    restored = json.loads(json.dumps(empirical_report.to_dict()))
    assert restored["passed"] is True
    assert restored["results"][0]["gate"] == "G"
    assert "does not prove" in restored["claim_boundary"]


def test_empirical_criterion_is_fail_closed() -> None:
    minimum = EmpiricalCriterion("score", 0.5, "at_least")
    maximum = EmpiricalCriterion("loss", 0.5, "at_most")
    assert minimum.evaluate({"score": 0.5})
    assert maximum.evaluate({"loss": 0.5})
    assert not minimum.evaluate({"score": float("nan")})
    assert not maximum.evaluate({"loss": float("inf")})
    with pytest.raises(KeyError, match="missing metric"):
        minimum.evaluate({})
    invalid = EmpiricalCriterion("score", 0.5, "sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown criterion"):
        invalid.evaluate({"score": 0.5})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param({"steps_scale": 0.0}, "step scale", id="nonpositive-step-scale"),
        pytest.param({"device": "mps"}, "requires CPU", id="noncpu-device"),
    ],
)
def test_empirical_suite_rejects_non_authoritative_controls(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        run_empirical_acceptance(**kwargs)


def test_empirical_cli_writes_authoritative_artifact(tmp_path: Path) -> None:
    output = tmp_path / "empirical.json"
    completed = subprocess.run(
        [
            sys.executable, "scripts/run_mrcra_empirical_acceptance.py",
            "--output", str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "overall=PASS" in completed.stdout
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["passed"] is True
    assert len(artifact["results"]) == 8
