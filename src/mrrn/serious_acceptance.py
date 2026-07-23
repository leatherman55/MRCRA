"""Fail-closed authority for trained serious-checkpoint acceptance.

This module does not infer capability from architecture, parameter count, or
bounded fixture tests.  It binds a completed training checkpoint to its pinned
training/evaluation split and to preregistered held-out task and hardware
artifacts.  Missing, mismatched, or weak evidence produces a failed gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite, sqrt
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch

from .config import CognitiveConfig, MRCRAConfig, MRRNConfig
from .cognitive_training import MRCRA_TRAINING_FORMAT_VERSION
from .language import MRCRALanguageModel


SERIOUS_PARAMETER_MINIMUM = 110_000_000
SERIOUS_PARAMETER_MAXIMUM = 125_000_000
SERIOUS_TRAINING_TOKEN_MINIMUM = 1_000_000_000
REQUIRED_HELD_OUT_TASKS = (
    "long_context_language_retention_generation",
    "asynchronous_multimodal_binding",
    "reconstructive_fidelity_partial_traces",
    "active_information_acquisition",
    "multihorizon_consequence_calibration",
    "persistent_memory_utility",
    "invariant_cross_domain_transfer",
    "contradictory_unreliable_source_robustness",
    "long_stream_stability",
)
_INTEGRATED_FLAGS = (
    "enable_conditional_reconstruction",
    "enable_abstraction_validity_control",
    "enable_post_deliberation_action_selection",
    "enable_multi_hypothesis_planning",
    "enable_agent_session_loop",
    "enable_viability_gate",
    "enable_integrated_invariant_discovery",
    "enable_persistent_session_training",
    "enable_metacognitive_routing",
)
_PINNED_REVISION = re.compile(r"^[0-9a-f]{40,64}$")


@dataclass(frozen=True, slots=True)
class SeriousGate:
    name: str
    passed: bool
    evidence: str
    failure: str | None = None


@dataclass(frozen=True, slots=True)
class SeriousCheckpointAcceptanceReport:
    schema_version: int
    suite: str
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    checkpoint_format: int | None
    parameter_count: int | None
    tokens_seen: int | None
    gates: tuple[SeriousGate, ...]
    passed: bool
    serious_scale: bool
    maturity: str
    failures: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SeriousCriterion:
    """One machine-recomputable task decision boundary."""

    metric: str
    threshold: float
    direction: str

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("criterion metric must be nonempty")
        if self.direction not in {"at_least", "at_most"}:
            raise ValueError("criterion direction must be at_least or at_most")
        if not isfinite(self.threshold):
            raise ValueError("criterion threshold must be finite")


@dataclass(frozen=True, slots=True)
class SeriousTaskEvidence:
    """Canonical evidence record for one preregistered held-out task."""

    name: str
    examples: int
    seeds: tuple[int, ...]
    successes: int
    minimum_success_rate: float
    effect_mean: float
    matched_ablation: str
    split_sha256: str
    data_revision: str
    production_trainable_parameters: int
    ablation_trainable_parameters: int
    metrics: Mapping[str, float]
    criteria: tuple[SeriousCriterion, ...]
    checkpoint_involved: bool = True

    def __post_init__(self) -> None:
        if self.name not in REQUIRED_HELD_OUT_TASKS:
            raise ValueError(f"unknown serious held-out task: {self.name}")
        if self.examples <= 0 or not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("task examples and unique seeds must be nonempty")
        if not 0 <= self.successes <= len(self.seeds):
            raise ValueError("task success count is outside the trial range")
        if not 0 <= self.minimum_success_rate <= 1:
            raise ValueError("minimum success rate must be in [0, 1]")
        if not isfinite(self.effect_mean):
            raise ValueError("task effect must be finite")
        if not self.checkpoint_involved or not self.matched_ablation:
            raise ValueError("checkpoint involvement and a matched ablation are required")
        if not re.fullmatch(r"[0-9a-f]{64}", self.split_sha256):
            raise ValueError("task split SHA-256 must be a lowercase hexadecimal digest")
        if not self.data_revision:
            raise ValueError("task data revision must be nonempty")
        if self.production_trainable_parameters <= 0 or self.ablation_trainable_parameters < 0:
            raise ValueError("task parameter counts are invalid")
        if not self.metrics or not self.criteria:
            raise ValueError("task metrics and criteria must be nonempty")
        normalized = {name: float(value) for name, value in self.metrics.items()}
        if any(not name or not isfinite(value) for name, value in normalized.items()):
            raise ValueError("task metrics must be named and finite")
        missing = [criterion.metric for criterion in self.criteria if criterion.metric not in normalized]
        if missing:
            raise ValueError(f"criteria reference missing metrics: {missing}")

    def to_dict(self) -> dict[str, Any]:
        trials = len(self.seeds)
        low, high = _wilson(self.successes, trials)
        normalized_metrics = {name: float(value) for name, value in self.metrics.items()}
        criteria = tuple(asdict(criterion) for criterion in self.criteria)
        criteria_pass = all(
            normalized_metrics[criterion.metric] >= criterion.threshold
            if criterion.direction == "at_least"
            else normalized_metrics[criterion.metric] <= criterion.threshold
            for criterion in self.criteria
        )
        return {
            "name": self.name,
            "passed": bool(criteria_pass and low >= self.minimum_success_rate),
            "examples": self.examples,
            "seeds": list(self.seeds),
            "trials": trials,
            "successes": self.successes,
            "confidence_low": low,
            "confidence_high": high,
            "minimum_success_rate": self.minimum_success_rate,
            "effect_mean": self.effect_mean,
            "checkpoint_involved": self.checkpoint_involved,
            "matched_ablation": self.matched_ablation,
            "split_sha256": self.split_sha256,
            "data_revision": self.data_revision,
            "production_trainable_parameters": self.production_trainable_parameters,
            "ablation_trainable_parameters": self.ablation_trainable_parameters,
            "metrics": normalized_metrics,
            "criteria": list(criteria),
        }


@dataclass(frozen=True, slots=True)
class SeriousPerformanceEvidence:
    """Measured 32K target-hardware result with recomputable budgets."""

    context_length: int
    tokens_per_second: float
    peak_memory_gib: float
    minimum_tokens_per_second: float
    maximum_peak_memory_gib: float
    hardware: str
    dtype: str

    def __post_init__(self) -> None:
        values = (
            self.tokens_per_second, self.peak_memory_gib,
            self.minimum_tokens_per_second, self.maximum_peak_memory_gib,
        )
        if self.context_length <= 0 or not all(isfinite(value) and value > 0 for value in values):
            raise ValueError("performance dimensions and measurements must be finite and positive")
        if not self.hardware or not self.dtype:
            raise ValueError("performance hardware and dtype must be nonempty")

    def to_dict(self) -> dict[str, Any]:
        passed = (
            self.context_length >= 32_768
            and self.tokens_per_second >= self.minimum_tokens_per_second
            and self.peak_memory_gib <= self.maximum_peak_memory_gib
        )
        return {**asdict(self), "passed": passed}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_serious_evaluation_artifact(
    checkpoint: str | Path,
    evaluation_identity: Mapping[str, Any],
    tasks: Sequence[SeriousTaskEvidence],
    performance: SeriousPerformanceEvidence,
) -> dict[str, Any]:
    """Build the accepted evidence shape from typed, recomputable inputs.

    Partial and failed research measurements may be retained elsewhere, but
    they cannot be serialized as an acceptance artifact.
    """

    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise ValueError("checkpoint file does not exist")
    if (
        int(evaluation_identity.get("batch_count", 0)) <= 0
        or not re.fullmatch(r"[0-9a-f]{64}", str(evaluation_identity.get("sha256", "")))
    ):
        raise ValueError("evaluation identity must contain a retained batch count and SHA-256")
    names = [task.name for task in tasks]
    if len(names) != len(set(names)) or set(names) != set(REQUIRED_HELD_OUT_TASKS):
        raise ValueError("evaluation artifact requires each preregistered task exactly once")
    task_dicts = [task.to_dict() for task in tasks]
    failed = [item["name"] for item in task_dicts if not item["passed"]]
    performance_dict = performance.to_dict()
    if failed or not performance_dict["passed"]:
        raise ValueError(
            f"acceptance artifact cannot contain failed evidence: tasks={failed}; "
            f"performance={performance_dict['passed']}"
        )
    return {
        "schema_version": 1,
        "suite": "mrcra-serious-held-out-evaluation-v1",
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "evaluation_identity": dict(evaluation_identity),
        "tasks": task_dicts,
        "performance": performance_dict,
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _configuration(identity: dict[str, Any]) -> MRCRAConfig:
    raw = identity.get("model_config")
    if not isinstance(raw, dict):
        raise ValueError("checkpoint identity has no model configuration")
    try:
        return MRCRAConfig(
            MRRNConfig(**raw["carrier"]), CognitiveConfig(**raw["cognitive"]),
            int(raw["actor_parameter_minimum"]),
            int(raw["actor_parameter_maximum"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"checkpoint model configuration is invalid: {error}") from error


def _gate(name: str, condition: bool, evidence: str, failure: str) -> SeriousGate:
    return SeriousGate(name, bool(condition), evidence, None if condition else failure)


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("task success counts are invalid")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = z / denominator * sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _task_evidence_valid(
    item: dict[str, Any] | None, *, minimum_examples: int, minimum_seeds: int,
    expected_parameters: int,
) -> bool:
    if not isinstance(item, dict) or item.get("checkpoint_involved") is not True:
        return False
    seeds = item.get("seeds")
    if not isinstance(seeds, list) or any(not isinstance(seed, int) for seed in seeds):
        return False
    unique_seeds = set(seeds)
    trials, successes = item.get("trials"), item.get("successes")
    if not isinstance(trials, int) or not isinstance(successes, int):
        return False
    if trials != len(seeds) or len(unique_seeds) < minimum_seeds:
        return False
    try:
        low, high = _wilson(successes, trials)
        recorded_low = float(item.get("confidence_low"))
        recorded_high = float(item.get("confidence_high"))
        minimum_rate = float(item.get("minimum_success_rate"))
        effect = float(item.get("effect_mean"))
    except (TypeError, ValueError):
        return False
    if not all(isfinite(value) for value in (recorded_low, recorded_high, minimum_rate, effect)):
        return False
    if abs(recorded_low - low) > 1e-9 or abs(recorded_high - high) > 1e-9:
        return False
    metrics = item.get("metrics")
    criteria = item.get("criteria")
    if not isinstance(metrics, dict) or not metrics or not isinstance(criteria, list) or not criteria:
        return False
    normalized_metrics: dict[str, float] = {}
    try:
        for name, value in metrics.items():
            if not isinstance(name, str):
                return False
            normalized_metrics[name] = float(value)
        if not all(isfinite(value) for value in normalized_metrics.values()):
            return False
    except (TypeError, ValueError):
        return False
    criteria_pass = True
    for criterion in criteria:
        if not isinstance(criterion, dict):
            return False
        metric = criterion.get("metric")
        direction = criterion.get("direction")
        try:
            threshold = float(criterion.get("threshold"))
        except (TypeError, ValueError):
            return False
        if metric not in normalized_metrics or direction not in {"at_least", "at_most"} or not isfinite(threshold):
            return False
        value = normalized_metrics[metric]
        criteria_pass &= value >= threshold if direction == "at_least" else value <= threshold
    recomputed_pass = criteria_pass and recorded_low >= minimum_rate
    return (
        item.get("passed") is recomputed_pass
        and recomputed_pass
        and isinstance(item.get("examples"), int)
        and item["examples"] >= minimum_examples
        and isinstance(item.get("matched_ablation"), str)
        and bool(item["matched_ablation"])
        and isinstance(item.get("split_sha256"), str)
        and bool(re.fullmatch(r"[0-9a-f]{64}", item["split_sha256"]))
        and isinstance(item.get("data_revision"), str)
        and bool(item["data_revision"])
        and isinstance(item.get("production_trainable_parameters"), int)
        and item["production_trainable_parameters"] == expected_parameters
        and isinstance(item.get("ablation_trainable_parameters"), int)
        and 0 <= item["ablation_trainable_parameters"] <= expected_parameters
    )


def audit_serious_checkpoint(
    checkpoint: str | Path,
    run_manifest: str | Path,
    evaluation_artifact: str | Path,
    *,
    minimum_parameters: int = SERIOUS_PARAMETER_MINIMUM,
    maximum_parameters: int = SERIOUS_PARAMETER_MAXIMUM,
    minimum_training_tokens: int = SERIOUS_TRAINING_TOKEN_MINIMUM,
    minimum_examples_per_task: int = 32,
    minimum_seeds_per_task: int = 8,
    contract_fixture: bool = False,
) -> SeriousCheckpointAcceptanceReport:
    """Audit exact checkpoint/data/evaluation identity and decision thresholds."""

    if min(minimum_parameters, minimum_training_tokens, minimum_examples_per_task, minimum_seeds_per_task) <= 0:
        raise ValueError("serious acceptance thresholds must be positive")
    if maximum_parameters < minimum_parameters:
        raise ValueError("serious parameter band is inverted")
    checkpoint_path = Path(checkpoint).resolve()
    manifest_path = Path(run_manifest).resolve()
    evaluation_path = Path(evaluation_artifact).resolve()
    gates: list[SeriousGate] = []
    digest = None
    checkpoint_format = parameter_count = tokens_seen = None
    try:
        if not checkpoint_path.is_file():
            raise ValueError("checkpoint file does not exist")
        digest = file_sha256(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="meta", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("checkpoint payload is not a mapping")
        checkpoint_format = payload.get("format_version")
        gates.append(_gate(
            "checkpoint_format",
            checkpoint_format == MRCRA_TRAINING_FORMAT_VERSION,
            f"format={checkpoint_format}",
            f"format {MRCRA_TRAINING_FORMAT_VERSION} is required",
        ))
        identity = payload.get("identity")
        if not isinstance(identity, dict):
            raise ValueError("checkpoint identity is missing")
        config = _configuration(identity)
        parameter_count = int(identity.get("parameter_count", -1))
        # Some authority-state constructors validate scalar tensor contents and
        # therefore cannot be instantiated on the meta device.  Acceptance is
        # an offline operation; materializing one CPU architecture image is
        # preferable to weakening those live-state validations.
        expected_model = MRCRALanguageModel(config)
        gates.append(_gate(
            "structural_parameter_identity",
            expected_model.parameter_count == parameter_count
            and minimum_parameters <= parameter_count <= maximum_parameters,
            f"declared={parameter_count}; reconstructed={expected_model.parameter_count}; "
            f"required={minimum_parameters}..{maximum_parameters}",
            "checkpoint parameter identity or declared band is invalid",
        ))
        model_state = payload.get("model")
        expected_state = expected_model.state_dict()
        state_valid = isinstance(model_state, dict) and set(model_state) == set(expected_state)
        if state_valid:
            state_valid = all(
                tuple(model_state[name].shape) == tuple(expected.shape)
                for name, expected in expected_state.items()
            )
        gates.append(_gate(
            "model_tensor_schema", state_valid,
            f"expected_tensors={len(expected_state)}",
            "checkpoint model tensors do not match the reconstructed architecture",
        ))
        flags = config.cognitive
        enabled = all(bool(getattr(flags, name)) for name in _INTEGRATED_FLAGS)
        gates.append(_gate(
            "integrated_runtime_configuration", enabled,
            ",".join(name for name in _INTEGRATED_FLAGS if getattr(flags, name)),
            "one or more consequential runtime mechanisms are disabled",
        ))
        state = payload.get("training_state")
        if not isinstance(state, dict):
            raise ValueError("checkpoint training state is missing")
        tokens_seen = int(state.get("tokens_seen", -1))
        gates.append(_gate(
            "minimum_training_tokens", tokens_seen >= minimum_training_tokens,
            f"tokens_seen={tokens_seen}; required={minimum_training_tokens}",
            "checkpoint has insufficient training exposure for the declared maturity",
        ))
        evaluation_identity = identity.get("evaluation")
        evaluation_bound = (
            isinstance(evaluation_identity, dict)
            and int(evaluation_identity.get("batch_count", 0)) > 0
            and isinstance(evaluation_identity.get("sha256"), str)
            and len(evaluation_identity["sha256"]) == 64
            and bool(identity.get("training", {}).get("require_evaluation"))
        )
        gates.append(_gate(
            "checkpoint_bound_retained_evaluation", evaluation_bound,
            repr(evaluation_identity),
            "checkpoint is not bound to a required retained evaluation split",
        ))

        manifest = _load_json(manifest_path, "run manifest")
        training_source = manifest.get("training_source")
        eval_source = manifest.get("evaluation_source")
        same_revision = (
            isinstance(training_source, dict) and isinstance(eval_source, dict)
            and training_source.get("revision") == eval_source.get("revision")
            and training_source.get("partition") == "train"
            and eval_source.get("partition") == "eval"
            and training_source.get("evaluation_fraction_permyriad")
            == eval_source.get("evaluation_fraction_permyriad")
        )
        pinned = contract_fixture or (
            same_revision
            and isinstance(training_source.get("revision"), str)
            and bool(_PINNED_REVISION.fullmatch(training_source["revision"]))
        )
        checkpoint_source = identity.get("source")
        production_identity = contract_fixture or (
            manifest.get("model_config") == identity.get("model_config")
            and manifest.get("tokenizer") == identity.get("tokenizer")
            and isinstance(checkpoint_source, dict)
            and all(
                training_source.get(name) == checkpoint_source.get(name)
                for name in (
                    "kind", "dataset_id", "dataset_config", "split", "revision",
                    "partition", "evaluation_fraction_permyriad", "shuffle_seed",
                    "shuffle_buffer",
                )
            )
        )
        declared_total = manifest.get("training_config", {}).get("total_tokens")
        completed_budget = contract_fixture or (
            isinstance(declared_total, int)
            and declared_total >= minimum_training_tokens
            and tokens_seen >= declared_total
        )
        manifest_bound = (
            manifest.get("model_parameters") == parameter_count
            and manifest.get("evaluation_identity") == evaluation_identity
            and same_revision and pinned and production_identity and completed_budget
        )
        gates.append(_gate(
            "pinned_disjoint_data_identity", manifest_bound,
            f"training={training_source}; evaluation={eval_source}",
            "run manifest is not bound to pinned disjoint train/evaluation data",
        ))

        artifact = _load_json(evaluation_path, "evaluation artifact")
        artifact_schema = (
            artifact.get("schema_version") == 1
            and artifact.get("suite") == "mrcra-serious-held-out-evaluation-v1"
        )
        gates.append(_gate(
            "evaluation_artifact_schema", artifact_schema,
            f"schema={artifact.get('schema_version')}; suite={artifact.get('suite')}",
            "evaluation artifact schema or suite identity is unsupported",
        ))
        artifact_bound = (
            artifact.get("checkpoint_sha256") == digest
            and artifact.get("evaluation_identity") == evaluation_identity
        )
        gates.append(_gate(
            "evaluation_artifact_identity", artifact_bound,
            f"checkpoint_sha256={artifact.get('checkpoint_sha256')}",
            "evaluation artifact does not identify this checkpoint and retained split",
        ))
        tasks = artifact.get("tasks")
        task_map = {
            item.get("name"): item for item in tasks
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        } if isinstance(tasks, list) else {}
        unique_task_names = isinstance(tasks, list) and len(task_map) == len(tasks)
        exact_task_set = set(task_map) == set(REQUIRED_HELD_OUT_TASKS)
        task_failures = []
        for name in REQUIRED_HELD_OUT_TASKS:
            item = task_map.get(name)
            valid = _task_evidence_valid(
                item, minimum_examples=minimum_examples_per_task,
                minimum_seeds=minimum_seeds_per_task,
                expected_parameters=parameter_count,
            )
            if not valid:
                task_failures.append(name)
        gates.append(_gate(
            "preregistered_held_out_tasks",
            unique_task_names and exact_task_set and not task_failures,
            f"required={len(REQUIRED_HELD_OUT_TASKS)}; present={len(task_map)}",
            f"duplicate, missing, or insufficient task evidence: {task_failures}",
        ))
        performance = artifact.get("performance")
        try:
            throughput = float(performance.get("tokens_per_second", 0))
            peak_memory = float(performance.get("peak_memory_gib", 0))
            minimum_throughput = float(performance.get("minimum_tokens_per_second", 0))
            maximum_memory = float(performance.get("maximum_peak_memory_gib", 0))
        except (AttributeError, TypeError, ValueError):
            throughput = peak_memory = minimum_throughput = maximum_memory = float("nan")
        recomputed_performance_pass = (
            isfinite(throughput) and throughput >= minimum_throughput > 0
            and isfinite(peak_memory) and 0 < peak_memory <= maximum_memory
            and isfinite(maximum_memory)
        )
        performance_valid = (
            isinstance(performance, dict)
            and performance.get("passed") is recomputed_performance_pass
            and recomputed_performance_pass
            and int(performance.get("context_length", 0)) >= 32_768
            and isinstance(performance.get("hardware"), str)
            and bool(performance["hardware"])
            and isinstance(performance.get("dtype"), str)
            and bool(performance["dtype"])
        )
        gates.append(_gate(
            "target_hardware_efficiency", performance_valid,
            repr(performance),
            "32K target-hardware efficiency evidence is absent or failed",
        ))
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        gates.append(SeriousGate(
            "artifact_integrity", False, "audit aborted", f"{type(error).__name__}: {error}"
        ))
    failures = tuple(
        f"{gate.name}: {gate.failure}" for gate in gates if not gate.passed
    )
    serious_scale = (
        minimum_parameters >= SERIOUS_PARAMETER_MINIMUM
        and minimum_training_tokens >= SERIOUS_TRAINING_TOKEN_MINIMUM
        and not contract_fixture
    )
    passed = bool(gates) and not failures
    return SeriousCheckpointAcceptanceReport(
        1, "mrcra-serious-checkpoint-acceptance-v1", str(checkpoint_path),
        digest, checkpoint_format, parameter_count, tokens_seen, tuple(gates),
        passed, serious_scale, "serious_checkpoint" if passed and serious_scale else "contract",
        failures,
        "Passing requires one exact trained checkpoint, a pinned disjoint retained split, "
        "all preregistered held-out matched tasks, and measured 32K target-hardware budgets. "
        "Contract-fixture passage does not establish serious capability.",
    )
