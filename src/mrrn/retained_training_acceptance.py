"""Cross-artifact authority audit for the complete training-execution repair."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Iterable


BENCHMARK_AUTHORITY_PATHS = (
    "scripts/benchmark_mrcra_training_execution.py",
    "src/mrrn/cognitive_training.py",
    "src/mrrn/activation_execution.py",
    "src/mrrn/document_batching.py",
    "src/mrrn/document_cost_model.py",
    "src/mrrn/carrier_execution.py",
    "src/mrrn/cstm_schedule.py",
)
LEARNING_AUTHORITY_PATHS = (
    "scripts/run_mrcra_learning_nonregression.py",
    "src/mrrn/learning_nonregression.py",
    "src/mrrn/cognitive_training.py",
    "src/mrrn/cstm.py",
    "src/mrrn/cstm_sampling.py",
    "src/mrrn/cstm_schedule.py",
    "src/mrrn/document_batching.py",
)
SOAK_AUTHORITY_PATHS = (
    "scripts/run_mrcra_resource_soak.py",
    "src/mrrn/resource_soak_acceptance.py",
    "src/mrrn/cognitive_training.py",
    "src/mrrn/cstm.py",
    "src/mrrn/cstm_sampling.py",
    "src/mrrn/cstm_schedule.py",
    "src/mrrn/document_batching.py",
)


@dataclass(frozen=True, slots=True)
class RetainedAcceptanceCriterion:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class RetainedTrainingAcceptanceReport:
    schema_version: int
    criteria: tuple[RetainedAcceptanceCriterion, ...]
    passed: bool
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def source_authority_digest(
    root: Path, relative_paths: Iterable[str],
) -> str:
    authority = sha256()
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"authority source is missing: {relative}")
        authority.update(relative.encode("utf-8"))
        authority.update(path.read_bytes())
    return authority.hexdigest()


def _load_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"retained acceptance artifact is unreadable: {path.name}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(
            f"retained acceptance artifact is not an object: {path.name}"
        )
    return value


def _criterion(
    name: str, passed: object, detail: str,
) -> RetainedAcceptanceCriterion:
    return RetainedAcceptanceCriterion(name, bool(passed), detail)


def validate_retained_training_acceptance(
    root: Path,
) -> RetainedTrainingAcceptanceReport:
    """Fail closed unless every full-scale and local support receipt is live."""

    outputs = root / "outputs"
    baseline = _load_object(
        outputs / "mrcra_training_execution_baseline.json"
    )
    execution = _load_object(
        outputs / "mrcra_training_execution_acceptance.json"
    )
    learning = _load_object(
        outputs / "mrcra_learning_nonregression_procedure.json"
    )
    learning_journal = _load_object(
        outputs / "mrcra_learning_nonregression_procedure_runs.json"
    )
    soak = _load_object(
        outputs / "mrcra_resource_soak_acceptance.json"
    )
    trackio = _load_object(
        outputs / "mrcra_trackio_overhead_acceptance.json"
    )
    parity = _load_object(
        outputs / "mrcra_device_parity_acceptance.json"
    )

    benchmark_digest = source_authority_digest(
        root, BENCHMARK_AUTHORITY_PATHS
    )
    learning_digest = source_authority_digest(
        root, LEARNING_AUTHORITY_PATHS
    )
    soak_digest = source_authority_digest(
        root, SOAK_AUTHORITY_PATHS
    )
    execution_samples = execution.get("samples")
    baseline_samples = baseline.get("samples")
    compiler = execution.get("compiler_candidate")
    baseline_compiler = baseline.get("compiler_candidate")
    controls = learning.get("controls")
    learning_runs = learning.get("runs")
    journal_runs = learning_journal.get("runs")
    soak_sample = soak.get("sample")
    parity_results = parity.get("results")

    revision_pattern = re.compile(r"^[0-9a-f]{40,64}$")
    immutable_revisions = (
        isinstance(controls, dict)
        and isinstance(controls.get("dataset_revision"), str)
        and isinstance(controls.get("tokenizer_revision"), str)
        and revision_pattern.fullmatch(controls["dataset_revision"])
        and revision_pattern.fullmatch(controls["tokenizer_revision"])
    )
    learning_keys = (
        {
            (run.get("variant"), run.get("seed"))
            for run in learning_runs
            if isinstance(run, dict)
        }
        if isinstance(learning_runs, list)
        else set()
    )
    seeds = (
        set(learning.get("seeds", ()))
        if isinstance(learning.get("seeds"), list)
        else set()
    )
    expected_learning_keys = {
        (variant, seed)
        for variant in ("legacy_dense", "sampled", "ce_only")
        for seed in seeds
    }
    tested_devices = (
        {
            row.get("device")
            for row in parity_results
            if isinstance(row, dict)
            and row.get("status") == "tested"
            and row.get("passed") is True
        }
        if isinstance(parity_results, list)
        else set()
    )

    criteria = (
        _criterion(
            "production_execution_source_current",
            baseline.get("authority_digest") == benchmark_digest,
            "baseline authority digest equals current benchmark sources",
        ),
        _criterion(
            "production_execution_complete_and_passed",
            baseline.get("schema_version") == 3
            and baseline.get("complete") is True
            and baseline.get("profile") == "production_8p4m_32k"
            and int(baseline.get("steps", 0)) >= 3
            and execution.get("format_version") == 3
            and execution.get("passed") is True,
            "8.4M/32K matrix is complete and every criterion passes",
        ),
        _criterion(
            "production_execution_raw_summary_identity",
            execution_samples == baseline_samples
            and compiler == baseline_compiler,
            "accepted samples and compiler receipt equal the raw journal",
        ),
        _criterion(
            "compiler_candidate_truthful",
            isinstance(compiler, dict)
            and compiler.get("outcome") in {"executed", "timeout"}
            and (
                compiler.get("outcome") != "timeout"
                or compiler.get("resolved_variant")
                == "static_cost_model_auto_repaired_cstm"
            ),
            "optional compilation is executed or explicitly rejected",
        ),
        _criterion(
            "fineweb_learning_source_current",
            learning.get("source_authority_digest") == learning_digest
            and learning_journal.get("authority_digest") == learning_digest,
            "learning report and journal match current source authority",
        ),
        _criterion(
            "fineweb_learning_complete_and_passed",
            learning.get("profile") == "fineweb_8p4m_32k"
            and learning.get("passed") is True
            and int(learning.get("physical_token_budget", 0))
            >= 1_048_576
            and len(seeds) >= 3
            and learning_keys == expected_learning_keys
            and learning_journal.get("complete") is True
            and learning_runs == journal_runs,
            "three paired seeds, three variants, exact journal identity",
        ),
        _criterion(
            "fineweb_learning_immutable_data_identity",
            immutable_revisions
            and controls.get("dataset_id") == "HuggingFaceFW/fineweb"
            and controls.get("dataset_config") == "sample-10BT",
            "FineWeb English sample and tokenizer are commit-pinned",
        ),
        _criterion(
            "production_soak_source_current",
            soak.get("source_authority_digest") == soak_digest,
            "100-step soak matches current source authority",
        ),
        _criterion(
            "production_soak_complete_and_passed",
            soak.get("passed") is True
            and isinstance(soak_sample, dict)
            and soak_sample.get("profile") == "production_8p4m_32k"
            and int(soak_sample.get("steps", 0)) >= 100,
            "full 8.4M/32K resource/resume soak passes",
        ),
        _criterion(
            "trackio_resource_budget_passed",
            trackio.get("passed") is True
            and all(
                item.get("passed") is True
                for item in trackio.get("criteria", ())
                if isinstance(item, dict)
            ),
            "Trackio timing and memory budgets pass",
        ),
        _criterion(
            "available_device_parity_passed",
            parity.get("passed") is True
            and {"cpu", "mps"}.issubset(tested_devices),
            "CPU and locally available MPS execute the parity suite",
        ),
    )
    return RetainedTrainingAcceptanceReport(
        1,
        criteria,
        all(item.passed for item in criteria),
        (
            "This cross-artifact receipt authorizes the complete local "
            "training-execution repair only when its production performance, "
            "FineWeb learning, 100-step soak, Trackio, and available-device "
            "evidence are simultaneously current. CUDA remains untested when "
            "no CUDA device is present."
        ),
    )
