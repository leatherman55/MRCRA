from __future__ import annotations

import json
from pathlib import Path

from mrrn.retained_training_acceptance import (
    BENCHMARK_AUTHORITY_PATHS,
    LEARNING_AUTHORITY_PATHS,
    SOAK_AUTHORITY_PATHS,
    source_authority_digest,
    validate_retained_training_acceptance,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n", encoding="utf-8"
    )


def complete_fixture(root: Path) -> None:
    for relative in sorted(set(
        BENCHMARK_AUTHORITY_PATHS
        + LEARNING_AUTHORITY_PATHS
        + SOAK_AUTHORITY_PATHS
    )):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"authority:{relative}\n", encoding="utf-8")
    benchmark_digest = source_authority_digest(
        root, BENCHMARK_AUTHORITY_PATHS
    )
    learning_digest = source_authority_digest(
        root, LEARNING_AUTHORITY_PATHS
    )
    soak_digest = source_authority_digest(root, SOAK_AUTHORITY_PATHS)
    samples = [{"variant": "static_cost_model_auto_repaired_cstm"}]
    compiler = {
        "outcome": "timeout",
        "resolved_variant": "static_cost_model_auto_repaired_cstm",
    }
    write_json(
        root / "outputs/mrcra_training_execution_baseline.json",
        {
            "schema_version": 3,
            "complete": True,
            "profile": "production_8p4m_32k",
            "steps": 3,
            "authority_digest": benchmark_digest,
            "samples": samples,
            "compiler_candidate": compiler,
        },
    )
    write_json(
        root / "outputs/mrcra_training_execution_acceptance.json",
        {
            "format_version": 3,
            "passed": True,
            "samples": samples,
            "compiler_candidate": compiler,
        },
    )
    seeds = [17, 29, 43]
    runs = [
        {"variant": variant, "seed": seed}
        for seed in seeds
        for variant in ("legacy_dense", "sampled", "ce_only")
    ]
    controls = {
        "dataset_id": "HuggingFaceFW/fineweb",
        "dataset_config": "sample-10BT",
        "dataset_revision": "a" * 40,
        "tokenizer_revision": "b" * 40,
    }
    write_json(
        root / "outputs/mrcra_learning_nonregression_procedure.json",
        {
            "profile": "fineweb_8p4m_32k",
            "passed": True,
            "physical_token_budget": 1_048_576,
            "seeds": seeds,
            "runs": runs,
            "controls": controls,
            "source_authority_digest": learning_digest,
        },
    )
    write_json(
        root
        / "outputs/mrcra_learning_nonregression_procedure_runs.json",
        {
            "complete": True,
            "runs": runs,
            "authority_digest": learning_digest,
        },
    )
    write_json(
        root / "outputs/mrcra_resource_soak_acceptance.json",
        {
            "passed": True,
            "sample": {
                "profile": "production_8p4m_32k",
                "steps": 100,
            },
            "source_authority_digest": soak_digest,
        },
    )
    write_json(
        root / "outputs/mrcra_trackio_overhead_acceptance.json",
        {
            "passed": True,
            "criteria": [{"name": "overhead", "passed": True}],
        },
    )
    write_json(
        root / "outputs/mrcra_device_parity_acceptance.json",
        {
            "passed": True,
            "results": [
                {"device": "cpu", "status": "tested", "passed": True},
                {"device": "mps", "status": "tested", "passed": True},
                {
                    "device": "cuda",
                    "status": "unavailable",
                    "passed": False,
                },
            ],
        },
    )


def test_complete_retained_authority_requires_every_current_full_scale_gate(
    tmp_path: Path,
):
    complete_fixture(tmp_path)
    report = validate_retained_training_acceptance(tmp_path)
    assert report.passed
    assert all(criterion.passed for criterion in report.criteria)


def test_stale_or_quick_artifact_fails_closed_without_hiding_other_gates(
    tmp_path: Path,
):
    complete_fixture(tmp_path)
    path = (
        tmp_path
        / "outputs/mrcra_learning_nonregression_procedure.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    value["profile"] = "quick"
    write_json(path, value)
    report = validate_retained_training_acceptance(tmp_path)
    assert not report.passed
    failures = {
        criterion.name
        for criterion in report.criteria
        if not criterion.passed
    }
    assert failures == {"fineweb_learning_complete_and_passed"}
