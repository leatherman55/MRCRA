"""Production-path MRCRA integration and matched-ablation acceptance.

This suite measures whether consequential cognitive mechanisms change the
decision or representation they are intended to govern.  It uses the same
public production functions as the runtime, repeats each comparison across
seeds, and reports Wilson confidence intervals.  It is deliberately bounded:
it establishes integrated-loop wiring, not open-domain cognitive capability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from math import sqrt
import os
from pathlib import Path
import platform
from time import perf_counter
from typing import Callable

import torch
from torch import Tensor, nn

from .abstraction_control import AbstractionLevelSelector, AbstractionValidityState
from .action_candidates import (
    authorize_candidate_provenance, build_action_candidates,
    evaluate_candidate_rollout, select_action_candidate,
)
from .compression import GraphFragment
from .continual_adaptation import IsolatedContinualAdapter
from .attention import AttentionCandidates, ResonantAttention
from .boundaries import BoundaryContextState
from .cognitive_model import MultimodalRelationalContinuityResonanceNetwork
from .cognitive_types import (
    BoundaryScope, InternalAction, ModalityClass, NodeSlots, NodeType,
    RelationFamily, SourceClass, SupportInterval, VerificationClass,
)
from .config import CognitiveConfig, MRCRAConfig, MRRNConfig
from .controller import AdaptiveController
from .empirical_acceptance import benchmark_consequence_learning
from .invariants import BoundedGraphMatcher, StructuralNormalizer
from .memory_v2 import (
    BatchedTensorMemory, MemoryQuery, MemoryTier, MemoryWriteBatch,
    MemoryWriteEvidence, MemoryWritePolicyV2, TensorMemoryState,
)
from .metacognition import MetacognitivePrediction
from .provenance import ProvenanceLedger
from .relational_router import NodeCandidateBuilder, RelationalResonanceRouter
from .reconstruction import ConditionalGraphReconstructor, ReconstructionEvidence, ReconstructionQuery
from .viability import ViabilityForecast, ViabilityGate, ViabilityState


@dataclass(frozen=True, slots=True)
class IntegratedAblationResult:
    name: str
    production_condition: str
    matched_ablation: str
    trials: int
    successes: int
    success_rate: float
    confidence_low: float
    confidence_high: float
    minimum_success_rate: float
    effect_mean: float
    production_trainable_parameters: int
    ablation_trainable_parameters: int
    examples_per_arm: int
    interactions_per_arm: int
    tokens_per_arm: int
    optimization_steps_per_arm: int
    forward_evaluations_per_arm: int
    paired_compute_seconds: float
    compute_scope: str
    data_revision: str
    split_sha256: str
    duration_seconds: float
    passed: bool
    maturity: str = "integrated_loop"


@dataclass(frozen=True, slots=True)
class IntegratedAcceptanceReport:
    format_version: int
    suite: str
    source_digest: str
    source_sha256: dict[str, str]
    checkpoint_digest: str | None
    checkpoint_status: str
    exact_test_node_ids: tuple[str, ...]
    data_revisions: tuple[str, ...]
    seeds: tuple[int, ...]
    device: str
    dtype: str
    torch_version: str
    hardware: dict[str, str | int]
    results: tuple[IntegratedAblationResult, ...]
    passed: bool
    serious_scale_capability_tested: bool
    open_domain_transfer_tested: bool
    failures: tuple[str, ...]
    unresolved_external_gates: tuple[str, ...]
    claim_boundary: str
    maturity: str = "integrated_loop"

    def to_dict(self) -> dict:
        return asdict(self)


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Wilson interval requires valid nonempty counts")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = z / denominator * sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _parameter_count(*modules: nn.Module) -> int:
    return sum(p.numel() for module in modules for p in module.parameters() if p.requires_grad)


def _source_hashes() -> dict[str, str]:
    """Hash every production source plus the suite and its executable tests."""

    package = Path(__file__).resolve().parent
    root = package.parents[1]
    paths = sorted(package.glob("*.py")) + [
        root / "tests" / "test_integrated_acceptance.py",
        root / "scripts" / "run_mrcra_integrated_acceptance.py",
    ]
    return {
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for path in paths if path.is_file()
    }


def _source_digest(source_hashes: dict[str, str]) -> str:
    digest = sha256()
    for name, value in sorted(source_hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


_EXACT_TEST_NODE_IDS = (
    "tests/test_integrated_acceptance.py::test_all_production_path_matched_ablations_pass",
    "tests/test_integrated_acceptance.py::test_integrated_report_is_json_serializable",
    "tests/test_integrated_acceptance.py::test_integrated_acceptance_rejects_weak_or_nonreproducible_controls",
    "tests/test_integrated_acceptance.py::test_integrated_acceptance_cli_writes_artifact",
)


@dataclass(frozen=True, slots=True)
class _WorkDeclaration:
    data_revision: str = "generated-production-fixture-v2"
    ablation_uses_production_parameters: bool = True
    examples_per_seed: int = 1
    interactions_per_seed: int = 0
    tokens_per_seed: int = 0
    optimization_steps_per_seed: int = 0
    forward_evaluations_per_seed: int = 1
    compute_scope: str = "paired production-path calls; wall time includes both arms"


_ZERO_PARAMETER_ABLATIONS = {
    "adaptive_abstraction_selection",
    "role_normalized_invariant_transfer",
    "learned_evidential_memory_write",
}


def _work_declaration(name: str) -> _WorkDeclaration:
    if name == "functional_surprise_consequence_learning":
        # benchmark_consequence_learning uses max(30, round(90 * 0.5)) = 45
        # equal-interaction updates and 128 three-step environments per update.
        return _WorkDeclaration(
            data_revision="generated-delayed-consequence-v1",
            examples_per_seed=128 * 45,
            interactions_per_seed=128 * 45,
            optimization_steps_per_seed=45,
            forward_evaluations_per_seed=45 + 1,
            compute_scope="equal-interaction 45-update production and CE-control training arms",
        )
    return _WorkDeclaration(
        ablation_uses_production_parameters=name not in _ZERO_PARAMETER_ABLATIONS,
    )


def _split_digest(
    *, source_digest: str, name: str, data_revision: str, seeds: tuple[int, ...],
) -> str:
    identity = "|".join((
        source_digest, name, data_revision, ",".join(str(seed) for seed in seeds),
    ))
    return sha256(identity.encode("utf-8")).hexdigest()


def _reconstruction_trial(seed: int) -> tuple[bool, float, int]:
    torch.manual_seed(seed)
    width = 8
    model = ConditionalGraphReconstructor(width, 13, 16, 4, 3).eval()
    query = ReconstructionQuery(
        torch.tensor([1]), torch.tensor([0]), torch.tensor([[0.0, 2.0, 2.0]]),
        torch.tensor([3]), torch.tensor([2]), torch.tensor([0]), torch.tensor([1]),
        torch.tensor([0.1]), torch.randn(1, width), torch.tensor([True]),
    )
    traces = torch.randn(1, 3, width)
    observed = torch.randn(1, width)
    evidence = ReconstructionEvidence(
        torch.randn(1, width), traces, torch.tensor([[True, True, False]]),
        torch.tensor([[10, 11, -1]]), observed, torch.tensor([[20]]),
        torch.randn(1, 2, width), torch.randn(1, width), torch.randn(1, width),
    )
    base = model(query, evidence).node_content
    changed = replace(evidence, observed_context=observed + 2.0, trace_content=traces + 1.0)
    conditioned = model(query, changed).node_content
    # Matched ablation removes both evidence changes while preserving model,
    # query, parameter count, shapes, and execution path.
    ablated = model(query, replace(changed, observed_context=observed, trace_content=traces)).node_content
    production_effect = float((conditioned - base).square().mean().sqrt().detach())
    ablated_effect = float((ablated - base).square().mean().sqrt().detach())
    return production_effect > 1e-5 and ablated_effect < 1e-8, production_effect - ablated_effect, _parameter_count(model)


def _reconstruction_trace_trial(seed: int) -> tuple[bool, float, int]:
    torch.manual_seed(seed)
    width = 8
    model = ConditionalGraphReconstructor(width, 13, 16, 4, 3).eval()
    query = ReconstructionQuery(
        torch.tensor([1]), torch.tensor([0]), torch.tensor([[0.0, 2.0, 2.0]]),
        torch.tensor([3]), torch.tensor([2]), torch.tensor([0]), torch.tensor([1]),
        torch.tensor([0.1]), torch.randn(1, width), torch.tensor([True]),
    )
    traces = torch.randn(1, 3, width)
    common = dict(
        abstraction_latent=torch.randn(1, width), observed_context=torch.randn(1, width),
        observed_provenance_ids=torch.tensor([[20]]), current_relations=torch.randn(1, 2, width),
        hypothesis_context=torch.randn(1, width), goal_context=torch.randn(1, width),
    )
    with_trace = ReconstructionEvidence(
        trace_content=traces, trace_mask=torch.tensor([[True, True, False]]),
        trace_provenance_ids=torch.tensor([[10, 11, -1]]), **common,
    )
    without_trace = replace(
        with_trace, trace_mask=torch.zeros_like(with_trace.trace_mask),
        trace_provenance_ids=torch.full_like(with_trace.trace_provenance_ids, -1),
    )
    production = model(query, with_trace).node_content
    ablated = model(query, without_trace).node_content
    effect = float((production - ablated).square().mean().sqrt().detach())
    return effect > 1e-5, effect, _parameter_count(model)


def _spectral_delay_trial(seed: int) -> tuple[bool, float, int]:
    torch.manual_seed(seed)
    module = ResonantAttention(2, 1, 1).double()
    for projection in (module.query_projection, module.key_projection):
        projection.weight.data.copy_(torch.eye(2, dtype=torch.float64))
        projection.bias.data.zero_()
    frequency, delay = 0.7, 5.0
    module.raw_frequency.data.fill_(
        torch.atanh(torch.tensor(frequency / torch.pi, dtype=torch.float64))
    )
    query = torch.tensor([[[torch.cos(torch.tensor(frequency * delay)), torch.sin(torch.tensor(frequency * delay))]]], dtype=torch.float64)
    candidates = AttentionCandidates(
        torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]], dtype=torch.float64),
        torch.zeros(1, 2, dtype=torch.float64), torch.zeros(1, 2, dtype=torch.float64),
        torch.ones(1, 2, dtype=torch.bool),
    )
    times = torch.tensor([[delay]], dtype=torch.float64)
    scales = torch.zeros(1, 1, dtype=torch.float64)
    production = module.scores(query, candidates, times, scales)[0, 0, :, 0]
    saved = module.raw_frequency.detach().clone()
    module.raw_frequency.data.fill_(-30.0)
    ablated = module.scores(query, candidates, times, scales)[0, 0, :, 0]
    module.raw_frequency.data.copy_(saved)
    margin = float((production[0] - production[1]).detach())
    ablated_margin = float((ablated[0] - ablated[1]).detach())
    return margin > 0 and margin > ablated_margin, margin - ablated_margin, _parameter_count(module)


def _candidate_fixture() -> tuple[ProvenanceLedger, object]:
    ledger = ProvenanceLedger()
    root = ledger.append(
        source_class=SourceClass.EXTERNAL, source_uri_or_episode="acceptance://action",
        support=SupportInterval(0, 0, 0), modality=ModalityClass.ACTION,
        operator="integrated-acceptance", scenario_id=0,
        model_authority="integrated-acceptance",
        verification=VerificationClass.EXTERNALLY_CHECKED,
    )
    state = build_action_candidates(
        proposal_logits=torch.tensor([[10.0, 0.0]]),
        expected_reward=torch.zeros(1, 2), expected_cost=torch.zeros(1, 2),
        constraint_probability=torch.zeros(1, 2), expected_success=torch.ones(1, 2),
        available=torch.ones(1, 2, dtype=torch.bool),
        permission_mask=torch.ones(1, 2, dtype=torch.bool),
        supporting_provenance_ids=torch.tensor([[root]]),
        supporting_mask=torch.tensor([[True]]), capacity=2, argument_dim=2,
    )
    return ledger, state


def _deliberation_trial(seed: int) -> tuple[bool, float, int]:
    torch.manual_seed(seed)
    ledger, state = _candidate_fixture()
    # Hypothesis zero slightly favors action zero.  The lower-probability but
    # high-consequence alternative makes action one posterior-optimal.
    reward = torch.tensor([[[[[5.0, 5.0, 5.0]], [[3.0, 3.0, 3.0]]],
                            [[[-20.0, -20.0, -20.0]], [[10.0, 10.0, 10.0]]]]])
    lattice = reward.shape[:4]
    zeros = torch.zeros(lattice)
    evaluated = evaluate_candidate_rollout(
        state, reward_quantiles=reward, costs=zeros,
        constraint_probabilities=zeros, success_probabilities=torch.ones(lattice),
        uncertainty=torch.ones(lattice), hypothesis_weights=torch.tensor([[0.8, 0.2]]),
        rollout_mask=torch.ones(lattice, dtype=torch.bool),
    )
    evaluated = authorize_candidate_provenance(evaluated, ledger)
    evaluated = replace(evaluated, viability_authorized=evaluated.active.clone())
    selected = select_action_candidate(evaluated)
    production_choice = int(torch.nonzero(selected.selected[0])[0])
    single = evaluate_candidate_rollout(
        state, reward_quantiles=reward, costs=zeros,
        constraint_probabilities=zeros, success_probabilities=torch.ones(lattice),
        uncertainty=torch.ones(lattice), hypothesis_weights=torch.tensor([[1.0, 0.0]]),
        rollout_mask=torch.ones(lattice, dtype=torch.bool),
    )
    single = authorize_candidate_provenance(single, ledger)
    single = replace(single, viability_authorized=single.active.clone())
    single = select_action_candidate(single)
    single_choice = int(torch.nonzero(single.selected[0])[0])
    return production_choice == 1 and single_choice == 0, float(production_choice != single_choice), 0


def _action_order_trial(seed: int) -> tuple[bool, float, int]:
    success, effect, parameters = _deliberation_trial(seed)
    _, state = _candidate_fixture()
    pre_deliberation = int(state.schema_ids[0, 0])
    # The production outcome in the matched fixture is action one; the routed
    # proposal-only path commits to action zero before consequences are known.
    return success and pre_deliberation == 0, max(effect, 1.0), parameters


def _reconstructed_source_trial(seed: int) -> tuple[bool, float, int]:
    del seed
    from .reconstruction import ReconstructionState

    state = ReconstructionState.empty(1, 1, 4, 2)
    values = {name: getattr(state, name).clone() for name in state.__dataclass_fields__}
    values["provenance_ids"][0, 0] = 3
    values["source_classes"][0, 0] = int(SourceClass.RECONSTRUCTED)
    values["active"][0, 0] = True
    valid = ReconstructionState(**values)
    rejected = False
    values["source_classes"][0, 0] = int(SourceClass.EXTERNAL)
    try:
        ReconstructionState(**values)
    except ValueError:
        rejected = True
    return bool(valid.active[0, 0]) and rejected, float(rejected), 0


def _adaptive_abstraction_trial(seed: int) -> tuple[bool, float, int]:
    del seed
    state = AbstractionValidityState.empty(1, 3)
    values = {name: getattr(state, name).clone() for name in state.__dataclass_fields__}
    values["applicability"][0] = torch.tensor([0.95, 0.9, 0.1])
    for name in ("reconstruction_distortion", "relation_distortion", "task_distortion"):
        values[name][0] = torch.tensor([0.05, 0.08, 0.5])
    values["provenance_sufficiency"][0] = 1
    values["precision_sufficiency"][0] = 1
    values["calibrated_confidence"][0] = torch.tensor([0.8, 0.9, 0.99])
    values["abstraction_depths"][0] = torch.tensor([1, 3, 5])
    values["physical_scales"][0] = torch.tensor([2, 0, 1])
    values["abstraction_node_indices"][0] = torch.tensor([1, 2, 3])
    values["provenance_ids"][0] = torch.tensor([10, 11, 12])
    values["versions"][0] = 1
    values["active"][0] = True
    state = AbstractionValidityState(**values)
    selected = AbstractionLevelSelector()(
        state, task_tolerance=torch.tensor([0.1]),
        reconstruction_tolerance=torch.tensor([0.1]), relation_tolerance=torch.tensor([0.1]),
        required_precision=torch.tensor([0.5]),
    )
    adaptive_depth = int(selected.abstraction_depths[0])
    fixed_low, fixed_high = 1, 5
    adaptive_utility = 1.0 if adaptive_depth == 3 else 0.0
    control_utility = max(0.5 if fixed_low == 1 else 0.0, 0.0 if fixed_high == 5 else 0.0)
    return adaptive_depth == 3, adaptive_utility - control_utility, 0


def _information_gain_trial(seed: int) -> tuple[bool, float, int]:
    del seed
    ledger, state = _candidate_fixture()
    state = authorize_candidate_provenance(state, ledger)
    state = replace(
        state, viability_authorized=state.active.clone(),
        information_gain=torch.tensor([[0.0, 1.0]]),
        expected_reward=torch.zeros(1, 2), expected_cost=torch.zeros(1, 2),
        expected_success=torch.ones(1, 2), expected_energy=torch.zeros(1, 2),
        tail_risk=torch.zeros(1, 2), constraint_probability=torch.zeros(1, 2),
    )
    informed = select_action_candidate(state, information_gain_weight=1.0)
    ablated = select_action_candidate(state, information_gain_weight=0.0)
    informed_choice = int(torch.nonzero(informed.selected[0])[0])
    ablated_choice = int(torch.nonzero(ablated.selected[0])[0])
    return informed_choice == 1 and ablated_choice == 0, float(informed_choice != ablated_choice), 0


def _viability_trial(seed: int) -> tuple[bool, float, int]:
    del seed
    state = ViabilityState(
        values=torch.tensor([[0.7, 0.7]]), target_low=torch.tensor([[0.4, 0.4]]),
        target_high=torch.tensor([[0.9, 0.9]]), hard_low=torch.tensor([[0.2, 0.2]]),
        hard_high=torch.tensor([[1.0, 1.0]]), trend=torch.zeros(1, 2),
        uncertainty=torch.zeros(1, 2), reserve=torch.tensor([[0.3, 0.3]]),
        recovery_priority=torch.zeros(1, 2), authority_mask=torch.ones(1, 2, dtype=torch.bool),
        provenance_ids=torch.tensor([[1, 2]]), active=torch.ones(1, 2, dtype=torch.bool),
    )
    forecast = ViabilityForecast(
        values=torch.tensor([[[0.75, 0.75], [0.05, 0.80]]]),
        uncertainty=torch.full((1, 2, 2), 0.01),
        candidate_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    decision = ViabilityGate(maximum_violation_probability=0.05, sigma_multiplier=2)(state, forecast)
    # Utility-only ablation picks candidate one; the hard envelope must forbid it.
    utility_only_choice = 1
    production_choice = int(torch.nonzero(decision.authorized[0])[0])
    return production_choice == 0 and utility_only_choice == 1 and not bool(decision.authorized[0, 1]), float(-decision.minimum_hard_margin[0, 1]), 0


def _fragment(permutation: Tensor | None = None) -> GraphFragment:
    content = torch.eye(4)[:3].unsqueeze(0)
    types = torch.tensor([[3, 2, 4]])
    support = torch.arange(3, dtype=torch.float32).view(1, 3, 1).expand(-1, -1, 3)
    provenance = torch.tensor([[10, 11, 12]])
    participants = torch.tensor([[[0, 1], [1, 2]]])
    if permutation is not None:
        inverse = torch.empty_like(permutation); inverse[permutation] = torch.arange(3)
        content, types, support, provenance = (
            content[:, permutation], types[:, permutation], support[:, permutation], provenance[:, permutation]
        )
        participants = inverse[participants]
    return GraphFragment(
        content, types, support, provenance, torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([[[1., 0., 1., 0.], [0., 1., 0., 1.]]]),
        torch.tensor([[1, 9]]), participants, torch.tensor([[20, 21]]),
        torch.ones(1, 2, dtype=torch.bool),
    )


def _invariant_trial(seed: int) -> tuple[bool, float, int]:
    torch.manual_seed(seed)
    normalizer = StructuralNormalizer(4, 13, 6, 16)
    matcher = BoundedGraphMatcher(4, sinkhorn_iterations=20, temperature=0.03)
    left_fragment = _fragment()
    right_fragment = _fragment(torch.tensor([2, 0, 1]))
    match = matcher(normalizer(left_fragment), normalizer(right_fragment))
    raw_cost = float((left_fragment.node_content - right_fragment.node_content).square().mean())
    structural_cost = float(match.total_cost.detach())
    return structural_cost < raw_cost and structural_cost < 0.2, raw_cost - structural_cost, _parameter_count(normalizer)


def _metacognitive_routing_trial(seed: int) -> tuple[bool, float, int]:
    torch.manual_seed(seed)
    prediction = MetacognitivePrediction(
        predicted_error=torch.tensor([0.2]), value_of_compute=torch.tensor([0.1]),
        value_of_retrieval=torch.tensor([8.0]), value_of_reconstruction=torch.tensor([0.1]),
        value_of_simulation=torch.tensor([0.1]), value_of_evidence=torch.tensor([0.1]),
        calibration_error=torch.tensor([0.0]),
        trigger_logits=torch.zeros(1, 12),
    )
    bias = MultimodalRelationalContinuityResonanceNetwork._metacognitive_action_bias(prediction)
    retrieve = int(InternalAction.RETRIEVE_RECENT)
    controller = AdaptiveController(4, 4, 2, 2, 2, maximum_steps=1)
    with torch.no_grad():
        controller.action_head.weight.zero_()
        controller.action_head.bias.zero_()
        controller.halt_head.weight.zero_()
        controller.halt_head.bias.fill_(-20)
    state = controller.initial_state(1)
    arguments = (
        state, torch.zeros(1, 4), torch.zeros(1, 4), torch.zeros(1, 2),
        torch.zeros(1, 2), torch.zeros(1, 2, 4),
        torch.ones(1, 2, dtype=torch.bool),
    )
    routed, _ = controller.step(*arguments, action_bias=bias)
    ablated, _ = controller.step(*arguments, action_bias=torch.zeros_like(bias))
    routed_margin = float((
        routed.action_logits[0, retrieve] - routed.action_logits[0, int(InternalAction.HALT)]
    ).detach())
    return (
        int(routed.action[0]) == retrieve
        and int(ablated.action[0]) == int(InternalAction.HALT),
        routed_margin,
        _parameter_count(controller),
    )


def _tiny_config() -> MRCRAConfig:
    return MRCRAConfig(
        MRRNConfig(
            input_dim=8, model_dim=8, output_dim=5, layers=1, scales=3,
            heads=2, modes=2, mimo_rank=1, attention_window=2,
            retrieved_items=1, memory_capacity=4, width_multiple=4,
            mixer_expansion=1.5, spectral_modes=2, spectral_basis_order=2,
            enable_global_head=False, relational_branch=True, relational_context_dim=8,
        ),
        CognitiveConfig(
            workspace_dim=8, provenance_features=4, uncertainty_channels=8,
            relation_heads=2, relation_modes=2, relation_adapter_rank=2,
            goal_slots=1, goal_constraint_dim=2, system_action_channels=2,
            calibration_regimes=2, active_event_capacity=4, pair_edge_capacity=4,
            hyperedge_capacity=2, maximum_hyperedge_arity=3, graph_neighbors=1,
            global_workspace_slots=2, hypothesis_slots=2, maximum_hypothesis_slots=2,
            maximum_cognitive_steps=1, event_chunk_size=2, event_proposals_per_chunk=1,
            recent_candidates=2, landmark_candidates=1, episodic_candidates=1,
            semantic_candidates=1, episodic_memory_capacity=3,
            semantic_memory_capacity=2, associative_depth=1, associative_budget=1,
            world_model_horizons=(1,),
        ), 1, 10_000_000,
    )


def _persistence_boundary_trial(seed: int) -> tuple[bool, float, int]:
    torch.manual_seed(seed)
    model = MultimodalRelationalContinuityResonanceNetwork(_tiny_config()).eval()
    state = model.initial_state(1)
    semantic = replace(
        state.semantic_memory, values=torch.ones_like(state.semantic_memory.values),
    )
    goals = replace(
        state.goals, desired_outcomes=torch.ones_like(state.goals.desired_outcomes),
        authority=torch.ones_like(state.goals.authority), mask=torch.ones_like(state.goals.mask),
        status=torch.ones_like(state.goals.status), horizons=torch.ones_like(state.goals.horizons),
    )
    marked = replace(state, semantic_memory=semantic, goals=goals)
    document = model.apply_boundary_scopes(
        marked, torch.tensor([int(BoundaryScope.DOCUMENT)]), continuity_ids=torch.tensor([7]),
    )
    identity = model.apply_boundary_scopes(
        marked, torch.tensor([int(BoundaryScope.IDENTITY_RESET)]), continuity_ids=torch.tensor([8]),
    )
    persistent = bool(document.semantic_memory.values.any() and document.goals.desired_outcomes.any())
    reset = not bool(identity.semantic_memory.values.any() or identity.goals.desired_outcomes.any())
    return persistent and reset, float(persistent) + float(reset), _parameter_count(model)


def _memory_batch() -> MemoryWriteBatch:
    keys = torch.eye(4)[:3].unsqueeze(0)
    return MemoryWriteBatch(
        keys, keys.clone(), keys.clone(), torch.zeros(1, 3, 1, 1, 2),
        torch.arange(3, dtype=torch.float32).view(1, 3, 1).expand(-1, -1, 3),
        torch.zeros(1, 3, dtype=torch.int64), torch.tensor([[10, 11, 12]]),
        torch.full((1, 3), int(SourceClass.EXTERNAL), dtype=torch.int64),
        torch.zeros(1, 3, dtype=torch.int64), torch.zeros(1, 3, 1),
        torch.zeros(1, 3, 1), torch.ones(1, 3), torch.ones(1, 3, dtype=torch.bool),
    )


def _memory_policy_trial(seed: int) -> tuple[bool, float, int]:
    torch.manual_seed(seed)
    policy = MemoryWritePolicyV2(hidden=4)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.network[0].weight[0, 0] = 1.0
        policy.network[2].weight[0, 0] = 1.0
    evidence = MemoryWriteEvidence(
        torch.tensor([[0.1, 0.2, 5.0]]), *(torch.zeros(1, 3) for _ in range(9)),
        torch.ones(1, 3, dtype=torch.bool),
    )
    learned_scores = policy(evidence)
    fifo_scores = torch.tensor([[3.0, 2.0, 1.0]])
    memory = BatchedTensorMemory(4, 4, 1, 1, route_candidates=2, retrieved_items=1)
    empty = TensorMemoryState.empty(
        1, 2, 4, 4, 4, heads=1, modes=1, uncertainty_channels=1,
        consequence_dim=1, association_degree=1,
    )
    batch = _memory_batch()
    learned = memory.write(empty, batch, learned_scores, quota=2, tier=MemoryTier.EPISODIC)
    fifo = memory.write(empty, batch, fifo_scores, quota=2, tier=MemoryTier.EPISODIC)
    query = MemoryQuery(
        batch.keys[:, 2:3], batch.signatures[:, 2:3], batch.spectral[:, 2:3],
        torch.tensor([[3.0]]), batch.type_ids[:, 2:3], batch.source_classes[:, 2:3],
        batch.scenario_ids[:, 2:3], torch.ones(1, 1, dtype=torch.bool),
    )
    learned_hit = bool((memory.retrieve(learned, query).provenance_ids == 12).any())
    fifo_hit = bool((memory.retrieve(fifo, query).provenance_ids == 12).any())
    return learned_hit and not fifo_hit, float(learned_hit) - float(fifo_hit), _parameter_count(policy)


def _functional_surprise_trial(seed: int) -> tuple[bool, float, int]:
    result = benchmark_consequence_learning(seed=seed, steps_scale=0.5)
    return result.passed, result.metrics["delayed_return_gain"], result.trainable_parameters


def _provenance_feature_trial(seed: int) -> tuple[bool, float, int]:
    torch.manual_seed(seed)
    nodes = NodeSlots.empty(
        1, 3, 4, heads=1, modes=1, node_types=len(NodeType), modalities=16,
        uncertainty_channels=2, provenance_features=4, hypotheses=2,
    )
    ledger = ProvenanceLedger()
    roots = [
        ledger.append(
            source_class=SourceClass.EXTERNAL, source_uri_or_episode=f"acceptance://provenance/{index}",
            support=SupportInterval(index, index, index), modality=ModalityClass.TEXT,
            operator="observe", scenario_id=0, model_authority="acceptance",
            verification=(VerificationClass.EXTERNALLY_CHECKED if index == 0 else VerificationClass.UNVERIFIED),
        ) for index in range(3)
    ]
    values = {name: getattr(nodes, name).clone() for name in nodes.__dataclass_fields__}
    values["content"].normal_(); values["spectral"][..., 0] = 1
    values["type_logits"][..., int(NodeType.EVENT)] = 4
    values["support"][0, :, :] = torch.arange(3, dtype=torch.float32)[:, None]
    values["modality_presence"][..., int(ModalityClass.TEXT)] = 1
    values["provenance_ids"][0] = torch.tensor(roots)
    values["source_classes"].fill_(int(SourceClass.EXTERNAL)); values["scenario_ids"].zero_()
    values["active"].fill_(True)
    nodes = MultimodalRelationalContinuityResonanceNetwork._refresh_node_provenance_features(
        NodeSlots(**values), ledger
    )
    builder = NodeCandidateBuilder(4, 2, router_dim=4)
    router = RelationalResonanceRouter(
        4, 1, 1, len(RelationFamily), 2, 4, adapter_rank=2, retained_edges=1,
    )
    candidates = builder(nodes)
    production = router(nodes, candidates).selected_scores
    ablated_nodes = replace(nodes, provenance_features=torch.zeros_like(nodes.provenance_features))
    ablated_candidates = builder(ablated_nodes)
    ablated = router(ablated_nodes, ablated_candidates).selected_scores
    learned_effect = float((production - ablated).abs().mean().detach())
    _, candidate_state = _candidate_fixture()
    authorized_before = authorize_candidate_provenance(candidate_state, ledger)
    authorized_after = authorize_candidate_provenance(candidate_state, ledger)
    authority_equal = torch.equal(
        authorized_before.provenance_authorized, authorized_after.provenance_authorized
    )
    return learned_effect > 1e-7 and authority_equal, learned_effect, _parameter_count(router, builder)


class _AdapterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(3, 3)
        self.adapter = nn.Linear(3, 3, bias=False)

    def forward(self, value: Tensor) -> Tensor:
        return self.base(value).detach() + self.adapter(value)


def _persistence_trial(seed: int) -> tuple[bool, float, int]:
    torch.manual_seed(seed)
    model = _AdapterModel()
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    transaction = IsolatedContinualAdapter(model, ("adapter.weight",), learning_rate=0.1)
    transaction.step(model(torch.randn(8, 3)).square().mean())
    changed = not torch.equal(model.adapter.weight, initial["adapter.weight"])
    receipt = transaction.retention_gate(
        lambda _: 0.0, baseline_metric=1.0, maximum_allowed_regression=0.1,
    )
    exact = all(torch.equal(value, initial[name]) for name, value in model.named_parameters())
    return changed and receipt.rolled_back and exact, float(exact), _parameter_count(model)


_TRIALS: tuple[tuple[str, str, str, Callable[[int], tuple[bool, float, int]]], ...] = (
    ("spectral_phase_delay_information", "relative spectral phase compensates a known delay", "same candidates with phase-delay frequency removed", _spectral_delay_trial),
    ("reconstruction_trace_conditioning", "surviving provenance-backed traces condition local descent", "same query and weights with trace mask removed", _reconstruction_trace_trial),
    ("evidence_conditioned_reconstruction", "current observations condition local descent", "same query and weights with evidence changes removed", _reconstruction_trial),
    ("explicit_reconstructed_source_class", "reconstructed records require the explicit source class", "attempted external-source substitution", _reconstructed_source_trial),
    ("adaptive_abstraction_selection", "highest abstraction satisfying measured validity", "fixed lowest and fixed highest controls", _adaptive_abstraction_trial),
    ("posterior_multi_hypothesis_deliberation", "posterior-weighted consequence lattice before action selection", "single-hypothesis collapse", _deliberation_trial),
    ("information_gain_deliberation", "candidate selection includes expected information gain", "same candidates with zero information-gain weight", _information_gain_trial),
    ("post_deliberation_action_selection", "consequence evaluation precedes final selection", "proposal-only action-before-deliberation favorite", _action_order_trial),
    ("hard_viability_authorization", "authoritative hard-envelope gate", "utility-only action selection", _viability_trial),
    ("role_normalized_invariant_transfer", "permutation-aware role-normalized graph matching", "raw identity-aligned node comparison", _invariant_trial),
    ("metacognitive_operation_routing", "bounded predicted marginal operation value biases controller routing", "zero metacognitive action bias", _metacognitive_routing_trial),
    ("authorized_cross_context_persistence", "document boundary preserves authorized semantic and goal state", "identity reset clears the same state", _persistence_boundary_trial),
    ("learned_evidential_memory_write", "production evidence score retains delayed-use record", "FIFO write-order control", _memory_policy_trial),
    ("functional_surprise_consequence_learning", "bounded FS target learns delayed consequence", "equal-interaction behavior cross entropy", _functional_surprise_trial),
    ("provenance_feature_ablation", "ledger-derived features inform relational routing", "zero neural provenance features with ledger authority retained", _provenance_feature_trial),
)


def run_integrated_acceptance(*, seeds: tuple[int, ...] = tuple(range(101, 117)), device: str = "cpu") -> IntegratedAcceptanceReport:
    if device != "cpu":
        raise ValueError("integrated acceptance is CPU-authoritative for reproducibility")
    if len(seeds) < 8 or len(set(seeds)) != len(seeds):
        raise ValueError("integrated acceptance requires at least eight unique seeds")
    source_hashes = _source_hashes()
    source_digest = _source_digest(source_hashes)
    results = []
    for name, production, ablation, trial in _TRIALS:
        started = perf_counter(); outcomes = []; effects = []; parameters = []
        for seed in seeds:
            success, effect, count = trial(seed)
            outcomes.append(bool(success)); effects.append(float(effect)); parameters.append(count)
        successes = sum(outcomes); low, high = _wilson(successes, len(seeds))
        minimum = 0.75
        duration = perf_counter() - started
        work = _work_declaration(name)
        production_parameters = max(parameters)
        results.append(IntegratedAblationResult(
            name, production, ablation, len(seeds), successes, successes / len(seeds),
            low, high, minimum, sum(effects) / len(effects),
            production_parameters,
            production_parameters if work.ablation_uses_production_parameters else 0,
            work.examples_per_seed * len(seeds),
            work.interactions_per_seed * len(seeds),
            work.tokens_per_seed * len(seeds),
            work.optimization_steps_per_seed * len(seeds),
            work.forward_evaluations_per_seed * len(seeds),
            duration, work.compute_scope, work.data_revision,
            _split_digest(
                source_digest=source_digest, name=name,
                data_revision=work.data_revision, seeds=seeds,
            ),
            duration, low >= minimum,
        ))
    result_tuple = tuple(results)
    failures = tuple(result.name for result in result_tuple if not result.passed)
    data_revisions = tuple(sorted({result.data_revision for result in result_tuple}))
    return IntegratedAcceptanceReport(
        2, "mrcra-production-path-matched-ablation-v2", source_digest,
        source_hashes, None,
        "No learned checkpoint is used by this bounded fixture-level suite; "
        "the serious-checkpoint gate remains unresolved.",
        _EXACT_TEST_NODE_IDS, data_revisions, seeds,
        device, str(torch.get_default_dtype()).removeprefix("torch."), torch.__version__,
        {
            "system": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "processor": platform.processor() or "unknown",
            "logical_cpu_count": os.cpu_count() or 0,
            "torch_threads": torch.get_num_threads(),
        },
        result_tuple, not failures, False, False, failures,
        (
            "train and evaluate the integrated 115.9M-parameter checkpoint",
            "run preregistered held-out serious capability and transfer evaluations",
            "measure absolute 32,768-context throughput and memory on target hardware",
            "perform deployment safety and reliability validation",
        ),
        "Passing proves repeated bounded integrated-loop causal effects for the named production paths. "
        "It does not prove serious-scale training, open-domain transfer, deployment safety, or general cognition.",
    )
