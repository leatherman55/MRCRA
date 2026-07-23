"""Measured and structural performance-budget acceptance for MRCRA.

Relative CPU timings are local regression sentinels, not target-hardware
throughput claims.  Shape, routing-bound, and checkpoint-layout gates are exact
authority checks and are therefore portable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from statistics import median
from time import process_time

import torch
from torch import Tensor

from .cognitive_checkpoint import runtime_state_dict
from .cognitive_model import MultimodalRelationalContinuityResonanceNetwork
from .cognitive_types import ModalityClass
from .config import CognitiveConfig, MRCRAConfig, MRRNConfig
from .model import MRRN
from .observation import register_external_observations
from .provenance import ProvenanceLedger
from .reconstruction import ConditionalGraphReconstructor, ReconstructionEvidence, ReconstructionQuery


@dataclass(frozen=True, slots=True)
class PerformanceGateResult:
    name: str
    metric: float
    threshold: float
    direction: str
    unit: str
    repeats: int
    measurement: str
    passed: bool


@dataclass(frozen=True, slots=True)
class PerformanceAcceptanceReport:
    format_version: int
    suite: str
    device: str
    dtype: str
    torch_threads: int
    results: tuple[PerformanceGateResult, ...]
    telemetry: dict[str, float | int | str | bool]
    passed: bool
    absolute_target_hardware_throughput_tested: bool
    claim_boundary: str

    def to_dict(self) -> dict:
        return asdict(self)


def _config(**flags: bool) -> MRCRAConfig:
    carrier = MRRNConfig(
        input_dim=8, model_dim=8, output_dim=17, layers=1, scales=3,
        heads=2, modes=2, mimo_rank=1, attention_window=2,
        attention_query_tile_size=2, retrieved_items=2, memory_capacity=8,
        mixer_expansion=1.5, width_growth_cap=1, mode_growth_cap=1,
        width_multiple=2, spectral_modes=2, spectral_basis_order=2,
        spectral_triads_per_mode=1, enable_global_head=False,
        relational_branch=True, relational_context_dim=8,
    )
    cognitive = CognitiveConfig(
        workspace_dim=8, provenance_features=4, uncertainty_channels=8,
        relation_heads=2, relation_modes=2, relation_adapter_rank=2,
        goal_slots=1, goal_constraint_dim=2, system_action_channels=2,
        calibration_regimes=2, active_event_capacity=4, pair_edge_capacity=8,
        hyperedge_capacity=2, maximum_hyperedge_arity=3, graph_neighbors=1,
        global_workspace_slots=2, hypothesis_slots=2, maximum_hypothesis_slots=2,
        planning_hypothesis_top_k=2, maximum_cognitive_steps=1,
        event_chunk_size=32, event_proposals_per_chunk=1,
        recent_candidates=2, landmark_candidates=1, episodic_candidates=2,
        semantic_candidates=2, episodic_memory_capacity=4,
        semantic_memory_capacity=4, associative_depth=2, associative_budget=2,
        world_model_horizons=(1,), **flags,
    )
    return MRCRAConfig(carrier, cognitive, 1, 10_000_000)


_FLAGS = (
    "enable_conditional_reconstruction", "enable_abstraction_validity_control",
    "enable_post_deliberation_action_selection", "enable_multi_hypothesis_planning",
    "enable_agent_session_loop", "enable_viability_gate",
    "enable_integrated_invariant_discovery", "enable_persistent_session_training",
    "enable_metacognitive_routing",
)


def _packet(ledger: ProvenanceLedger):
    values = torch.randn(1, 1, 8)
    valid = torch.ones(1, 1, dtype=torch.bool)
    return register_external_observations(
        values, valid, observed_mask=valid, timestamps=torch.zeros(1, 1),
        coordinates=torch.zeros(1, 1, 1), sample_intervals=torch.ones(1),
        boundary_classes=torch.zeros(1, 1, dtype=torch.int64),
        modality_ids=torch.full((1, 1), int(ModalityClass.TEXT), dtype=torch.int64),
        uncertainty_seed=torch.zeros(1, 1, 8), segment_ids=torch.zeros(1, 1, dtype=torch.int64),
        source_uris=("performance://local",), ledger=ledger, model_authority="performance-suite",
    )


def _force_event_behavior(model, active: bool) -> None:
    proposal = model.event_extractor.proposal_network
    with torch.no_grad():
        for parameter in proposal.parameters():
            parameter.zero_()
        proposal.proposal.bias.fill_(8 if active else -8)
        proposal.end.bias.fill_(8 if active else -8)


def _paired_latency(
    *, event: bool, repeats: int, calls_per_repeat: int = 16,
) -> tuple[float, float, float, float]:
    baseline_config = _config(**{name: False for name in _FLAGS})
    integrated_config = _config(**{name: True for name in _FLAGS})
    torch.manual_seed(701)
    # One weight/storage image is used for both arms; only the immutable
    # runtime feature configuration changes.  This isolates added path cost
    # from allocator placement and cross-model cache artifacts.
    model = MultimodalRelationalContinuityResonanceNetwork(integrated_config).eval()
    _force_event_behavior(model, event)
    base_ledger, integrated_ledger = ProvenanceLedger(), ProvenanceLedger()
    base_packet, integrated_packet = _packet(base_ledger), _packet(integrated_ledger)
    base_state, integrated_state = model.initial_state(1), model.initial_state(1)
    base, current, paired_overhead = [], [], []

    def measure(config, packet, state, ledger) -> float:
        model.config = config
        started = process_time()
        for _ in range(calls_per_repeat):
            model.step(packet, 0, state, ledger)
        return (process_time() - started) / calls_per_repeat

    with torch.inference_mode():
        for _ in range(8):
            measure(baseline_config, base_packet, base_state, base_ledger)
            measure(integrated_config, integrated_packet, integrated_state, integrated_ledger)
        # Each repeat uses an ABBA/BAAB block. Averaging both observations per
        # arm cancels first-order thermal and clock drift within the pair while
        # process CPU time excludes scheduler descheduling. The decision is the
        # median of within-block ratios, not a ratio of separated medians.
        for index in range(repeats):
            if index % 2:
                current_first = measure(integrated_config, integrated_packet, integrated_state, integrated_ledger)
                base_first = measure(baseline_config, base_packet, base_state, base_ledger)
                base_second = measure(baseline_config, base_packet, base_state, base_ledger)
                current_second = measure(integrated_config, integrated_packet, integrated_state, integrated_ledger)
            else:
                base_first = measure(baseline_config, base_packet, base_state, base_ledger)
                current_first = measure(integrated_config, integrated_packet, integrated_state, integrated_ledger)
                current_second = measure(integrated_config, integrated_packet, integrated_state, integrated_ledger)
                base_second = measure(baseline_config, base_packet, base_state, base_ledger)
            base_time = (base_first + base_second) / 2
            current_time = (current_first + current_second) / 2
            base.append(base_time); current.append(current_time)
            paired_overhead.append(100 * (current_time / base_time - 1))
    overhead_median = median(paired_overhead)
    median_absolute_deviation = median(
        abs(value - overhead_median) for value in paired_overhead
    )
    return median(base), median(current), overhead_median, median_absolute_deviation


def _tensor_bytes(value) -> int:
    if isinstance(value, Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _model_tensor_references(value) -> int:
    if isinstance(value, dict):
        return sum(
            (1 if key in {"model", "model_state", "state_dict", "model_weights"} else 0)
            + _model_tensor_references(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return sum(_model_tensor_references(item) for item in value)
    return 0


def _reconstruction_locality() -> tuple[bool, dict[str, int]]:
    torch.manual_seed(709)
    model = ConditionalGraphReconstructor(8, 13, 16, 4, 3).eval()
    evidence = ReconstructionEvidence(
        torch.randn(1, 8), torch.randn(1, 4, 8), torch.tensor([[True, True, False, False]]),
        torch.tensor([[1, 2, -1, -1]]), torch.randn(1, 8), torch.tensor([[3]]),
        torch.randn(1, 2, 8), torch.randn(1, 8), torch.randn(1, 8),
    )
    def query(nodes: int, relations: int, context_extent: float):
        return ReconstructionQuery(
            torch.tensor([0]), torch.tensor([0]), torch.tensor([[0., context_extent, context_extent]]),
            torch.tensor([nodes]), torch.tensor([relations]), torch.tensor([0]), torch.tensor([1]),
            torch.tensor([0.1]), torch.randn(1, 8), torch.tensor([True]),
        )
    with torch.inference_mode():
        small = model(query(1, 0, 32), evidence)
        large_context = model(query(1, 0, 32768), evidence)
        local_maximum = model(query(4, 3, 32768), evidence)
    telemetry = {
        "short_context_allocated_node_slots": small.node_content.shape[1],
        "long_context_allocated_node_slots": large_context.node_content.shape[1],
        "long_context_active_nodes": int(large_context.node_mask.sum()),
        "maximum_local_active_nodes": int(local_maximum.node_mask.sum()),
        "maximum_local_active_relations": int(local_maximum.relation_mask.sum()),
    }
    passed = (
        small.node_content.shape == large_context.node_content.shape
        and int(large_context.node_mask.sum()) == 1
        and int(local_maximum.node_mask.sum()) == model.maximum_nodes
        and int(local_maximum.relation_mask.sum()) == model.maximum_relations
    )
    return passed, telemetry


def _carrier_prefill_speedup(*, repeats: int = 7, length: int = 64) -> tuple[float, float, float]:
    """Compare complete-state vectorized prefill with identical token transitions."""

    torch.manual_seed(719)
    model = MRRN(_config().carrier).eval()
    values = torch.randn(1, length, model.config.input_dim)

    def vectorized() -> None:
        state = model.initial_stream_state(1)
        model.prefill(values, state=state, project_output=False)

    def tokenwise() -> None:
        state = model.initial_stream_state(1)
        for position in range(length):
            state = model.step(
                values[:, position], state, project_output=False
            ).state

    vectorized_times, token_times = [], []
    with torch.inference_mode():
        vectorized()
        tokenwise()
        for index in range(repeats):
            first, second = (
                (vectorized, tokenwise) if index % 2 == 0
                else (tokenwise, vectorized)
            )
            started = process_time()
            first()
            first_time = process_time() - started
            started = process_time()
            second()
            second_time = process_time() - started
            if index % 2 == 0:
                vectorized_times.append(first_time)
                token_times.append(second_time)
            else:
                token_times.append(first_time)
                vectorized_times.append(second_time)
    vectorized_median, token_median = (
        median(vectorized_times), median(token_times)
    )
    return vectorized_median, token_median, token_median / vectorized_median


def run_performance_acceptance(*, repeats: int = 41, device: str = "cpu") -> PerformanceAcceptanceReport:
    if device != "cpu":
        raise ValueError("portable performance acceptance currently requires CPU")
    if repeats < 21 or repeats % 2 == 0:
        raise ValueError("performance acceptance requires an odd repeat count of at least 21")
    previous_threads = torch.get_num_threads()
    measurement_threads = 1
    torch.set_num_threads(measurement_threads)
    try:
        dormant_base, dormant_full, dormant_overhead, dormant_mad = _paired_latency(
            event=False, repeats=repeats
        )
        cycle_base, cycle_full, cycle_overhead, cycle_mad = _paired_latency(
            event=True, repeats=repeats
        )
        prefill_time, token_time, prefill_speedup = _carrier_prefill_speedup()
        locality_passed, locality = _reconstruction_locality()
        serious = MRCRAConfig.serious_120m()
        planning_cells = (
            serious.cognitive.planning_hypothesis_top_k
            * serious.cognitive.action_candidate_capacity
            * len(serious.cognitive.world_model_horizons)
        )
        maximum_cells = (
            serious.cognitive.maximum_hypothesis_slots
            * serious.cognitive.action_candidate_capacity
            * len(serious.cognitive.world_model_horizons)
        )
        checkpoint_model = MultimodalRelationalContinuityResonanceNetwork(
            _config(**{name: True for name in _FLAGS})
        )
        runtime = runtime_state_dict(checkpoint_model.initial_state(1))
        runtime_bytes = _tensor_bytes(runtime)
        duplicate_model_references = _model_tensor_references(runtime)
    finally:
        torch.set_num_threads(previous_threads)
    results = (
        PerformanceGateResult("dormant_cognitive_overhead", dormant_overhead, 5.0, "at_most", "percent", repeats, "median ABBA-paired single-thread CPU-time overhead with event path dormant", dormant_overhead <= 5.0),
        PerformanceGateResult("ordinary_event_cycle_overhead", cycle_overhead, 25.0, "at_most", "percent", repeats, "median ABBA-paired single-thread CPU-time overhead at matched capacities", cycle_overhead <= 25.0),
        PerformanceGateResult("reconstruction_locality", 1.0 if locality_passed else 0.0, 1.0, "at_least", "boolean", 1, "requested support extent 32 versus 32768 with bounded local decoder", locality_passed),
        PerformanceGateResult("bounded_planning_lattice", float(planning_cells), float(maximum_cells), "at_most", "hypothesis-action-horizon cells", 1, "exact configured K_h*K_a*H bound", planning_cells <= maximum_cells),
        PerformanceGateResult("checkpoint_duplicate_model_references", float(duplicate_model_references), 0.0, "at_most", "references", 1, "recursive runtime-state schema inspection", duplicate_model_references == 0),
        PerformanceGateResult("complete_state_prefill_speedup", prefill_speedup, 2.0, "at_least", "times", 7, "median alternating single-thread CPU time for vectorized prefill versus identical token transitions", prefill_speedup >= 2.0),
    )
    telemetry: dict[str, float | int | str | bool] = {
        "dormant_baseline_median_ms": dormant_base * 1000,
        "dormant_integrated_median_ms": dormant_full * 1000,
        "cycle_baseline_median_ms": cycle_base * 1000,
        "cycle_integrated_median_ms": cycle_full * 1000,
        "dormant_paired_overhead_mad_percent": dormant_mad,
        "cycle_paired_overhead_mad_percent": cycle_mad,
        "planning_hypothesis_top_k": serious.cognitive.planning_hypothesis_top_k,
        "planning_action_capacity": serious.cognitive.action_candidate_capacity,
        "planning_horizon_count": len(serious.cognitive.world_model_horizons),
        "planning_cells": planning_cells,
        "runtime_declared_tensor_bytes": runtime_bytes,
        "checkpoint_embeds_model_weights": bool(duplicate_model_references),
        "complete_state_prefill_median_ms": prefill_time * 1000,
        "tokenwise_transition_median_ms": token_time * 1000,
        "latency_calls_per_arm_observation": 16,
        "latency_arm_observations_per_repeat": 2,
        "latency_clock": "process_time",
        **locality,
    }
    return PerformanceAcceptanceReport(
        3, "mrcra-relative-performance-budgets-v3", device,
        str(torch.get_default_dtype()).removeprefix("torch."), measurement_threads,
        results, telemetry, all(result.passed for result in results), False,
        "Relative timings are CPU regression sentinels for this source revision. "
        "Absolute RTX A4500 or other target-hardware throughput remains intentionally unclaimed.",
    )
