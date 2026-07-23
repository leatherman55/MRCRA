"""Bounded learned-behavior acceptance experiments for MRCRA.

These experiments occupy the layer between unit tests and expensive model
training.  They use the production components on deterministic held-out
synthetic tasks, compare against explicit ablations, and emit raw metrics plus
machine-checkable criteria.  Passing this suite establishes that the local
mechanisms can learn their intended role; it deliberately does not claim that
an untrained or small model has acquired open-domain cognitive capability.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isfinite
from time import perf_counter
from typing import Callable, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .compression import GraphCompressor, GraphFragment
from .controller import AdaptiveController
from .cognitive_types import (
    InternalAction, ModalityClass, SourceClass, SupportInterval,
    VerificationClass,
)
from .hypotheses import HypothesisBank
from .knowledge import (
    KnowledgeKind, KnowledgeProposalBank, KnowledgeProposalBatch,
    KnowledgeProposalState, KnowledgeStatus, KnowledgeValidationBatch,
)
from .memory_v2 import (
    BatchedTensorMemory, MemoryQuery, MemoryTier, MemoryWriteBatch,
    TensorMemoryState,
)
from .modalities import ContinuousSignalEncoder, TokenEncoder
from .provenance import ProvenanceLedger
from .surprise import (
    AdjointCriticOutput, FunctionalSurpriseCalibrator, PerformanceGuard,
    ResonantAdjointSurpriseConfig, functional_surprise_target,
    multihorizon_returns,
)
from .uncertainty import DistributionalPredictionHead, OnlineCalibration
from .world_model import ActionConditionedWorldModel


CriterionDirection = Literal["at_least", "at_most"]


@dataclass(frozen=True, slots=True)
class EmpiricalCriterion:
    """One preregistered scalar decision rule."""

    metric: str
    threshold: float
    direction: CriterionDirection

    def evaluate(self, metrics: dict[str, float]) -> bool:
        if self.metric not in metrics:
            raise KeyError(f"criterion references missing metric {self.metric!r}")
        value = metrics[self.metric]
        if not isfinite(value):
            return False
        if self.direction == "at_least":
            return value >= self.threshold
        if self.direction == "at_most":
            return value <= self.threshold
        raise ValueError(f"unknown criterion direction {self.direction!r}")


@dataclass(frozen=True, slots=True)
class EmpiricalBenchmarkResult:
    name: str
    gate: str
    seed: int
    duration_seconds: float
    trainable_parameters: int
    metrics: dict[str, float]
    criteria: tuple[EmpiricalCriterion, ...]
    passed: bool
    scope: str
    maturity: str


@dataclass(frozen=True, slots=True)
class EmpiricalAcceptanceReport:
    format_version: int
    suite: str
    torch_version: str
    device: str
    results: tuple[EmpiricalBenchmarkResult, ...]
    passed: bool
    serious_scale_capability_tested: bool
    physical_cuda_tested: bool
    claim_boundary: str
    maturity: str

    def to_dict(self) -> dict:
        return asdict(self)


def _parameters(*modules: nn.Module) -> int:
    return sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _optimize(
    loss_fn: Callable[[], Tensor], parameters, *, steps: int, learning_rate: float,
) -> None:
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        if loss.numel() != 1 or not bool(torch.isfinite(loss)):
            raise FloatingPointError("empirical benchmark loss became non-finite")
        loss.backward()
        optimizer.step()


def _result(
    name: str, gate: str, seed: int, started: float, parameters: int,
    metrics: dict[str, float], criteria: tuple[EmpiricalCriterion, ...], scope: str,
) -> EmpiricalBenchmarkResult:
    normalized = {key: float(value) for key, value in metrics.items()}
    passed = all(criterion.evaluate(normalized) for criterion in criteria)
    return EmpiricalBenchmarkResult(
        name, gate, seed, perf_counter() - started, parameters, normalized,
        criteria, passed, scope, "mechanism",
    )


def _write_batch(
    keys: Tensor, signatures: Tensor, values: Tensor, *, provenance_start: int,
    utility: float, completion_start: float,
) -> MemoryWriteBatch:
    batch, count, _ = keys.shape
    support = keys.new_zeros(batch, count, 3)
    support[..., 0] = torch.arange(count, device=keys.device) + completion_start
    support[..., 1] = support[..., 0]
    support[..., 2] = support[..., 0]
    return MemoryWriteBatch(
        keys, values, signatures,
        keys.new_zeros(batch, count, 1, 1, 2), support,
        torch.arange(count, device=keys.device).remainder(values.shape[-1])[None].expand(batch, -1),
        torch.arange(provenance_start, provenance_start + count, device=keys.device)[None].expand(batch, -1),
        torch.full((batch, count), int(SourceClass.EXTERNAL), dtype=torch.int64, device=keys.device),
        torch.zeros(batch, count, dtype=torch.int64, device=keys.device),
        keys.new_zeros(batch, count, 2), keys.new_zeros(batch, count, 1),
        keys.new_full((batch, count), utility),
        torch.ones(batch, count, dtype=torch.bool, device=keys.device),
    )


def benchmark_retrieval(seed: int = 17, *, steps_scale: float = 1.0) -> EmpiricalBenchmarkResult:
    """Gate G: routed retrieval must help a delayed downstream decision."""

    del steps_scale
    started = perf_counter()
    torch.manual_seed(seed)
    width, capacity, classes = 16, 64, 4
    memory = BatchedTensorMemory(
        width, width, 1, 1, route_candidates=8, retrieved_items=1,
    )
    state = TensorMemoryState.empty(
        1, capacity, width, classes, width, heads=1, modes=1,
        uncertainty_channels=2, consequence_dim=1, association_degree=2,
    )
    keys = F.normalize(torch.randn(1, 48, width), dim=-1)
    signatures = F.normalize(keys + 0.025 * torch.randn_like(keys), dim=-1)
    labels = torch.arange(48).remainder(classes)
    values = F.one_hot(labels, classes).float()[None]
    state = memory.write(
        state, _write_batch(
            keys, signatures, values, provenance_start=100, utility=5.0,
            completion_start=0,
        ), torch.ones(1, 48), quota=48, tier=MemoryTier.EPISODIC,
    )
    target = torch.arange(24)
    query_keys = F.normalize(keys[:, target] + 0.035 * torch.randn(1, 24, width), dim=-1)
    query_signatures = F.normalize(
        signatures[:, target] + 0.035 * torch.randn(1, 24, width), dim=-1
    )
    query = MemoryQuery(
        query_keys, query_signatures, query_keys.new_zeros(1, 24, 1, 1, 2),
        query_keys.new_full((1, 24), 100.0), labels[target][None],
        torch.full((1, 24), int(SourceClass.EXTERNAL), dtype=torch.int64),
        torch.zeros(1, 24, dtype=torch.int64), torch.ones(1, 24, dtype=torch.bool),
    )
    retrieved = memory.retrieve(state, query, compute_oracle=True)
    prediction = retrieved.values[:, :, 0].argmax(-1)
    downstream_accuracy = (prediction == labels[target][None]).float().mean()
    top1_recall = (retrieved.indices[:, :, 0] == target[None]).float().mean()
    router_recall = retrieved.router_recall.mean()
    recent_indices = torch.arange(40, 48)
    recent_score = torch.einsum("bqd,bmd->bqm", query_keys, keys[:, recent_indices])
    recent_choice = recent_indices[recent_score.argmax(-1)]
    recent_prediction = labels[recent_choice]
    recent_accuracy = (recent_prediction == labels[target][None]).float().mean()

    # Fill the ring with low-utility distractors, then force eviction.  The
    # high-utility delayed records must survive the capacity pressure.
    for offset in (0, 16):
        extra_keys = F.normalize(torch.randn(1, 16, width), dim=-1)
        extra_values = F.one_hot(torch.arange(16).remainder(classes), classes).float()[None]
        state = memory.write(
            state, _write_batch(
                extra_keys, extra_keys, extra_values,
                provenance_start=1000 + offset, utility=0.01,
                completion_start=200 + offset,
            ), torch.ones(1, 16), quota=16, tier=MemoryTier.EPISODIC,
        )
    survived = memory.retrieve(state, query, compute_oracle=True)
    survival_accuracy = (
        survived.values[:, :, 0].argmax(-1) == labels[target][None]
    ).float().mean()
    metrics = {
        "write_recall": float((state.provenance_ids >= 100).any(-1).float().mean()),
        "router_recall": float(router_recall),
        "reranker_top1_recall": float(top1_recall),
        "downstream_accuracy": float(downstream_accuracy),
        "recent_only_accuracy": float(recent_accuracy),
        "downstream_gain": float(downstream_accuracy - recent_accuracy),
        "eviction_survival_accuracy": float(survival_accuracy),
        "oracle_gap": float(1 - router_recall),
    }
    criteria = (
        EmpiricalCriterion("router_recall", 0.95, "at_least"),
        EmpiricalCriterion("reranker_top1_recall", 0.90, "at_least"),
        EmpiricalCriterion("downstream_gain", 0.50, "at_least"),
        EmpiricalCriterion("eviction_survival_accuracy", 0.85, "at_least"),
        EmpiricalCriterion("oracle_gap", 0.05, "at_most"),
    )
    return _result(
        "delayed_retrieval_utility", "G", seed, started, _parameters(memory),
        metrics, criteria, "bounded exact tensor-memory task with distractors and eviction",
    )


def benchmark_multimodal_binding(
    seed: int = 17, *, steps_scale: float = 1.0,
) -> EmpiricalBenchmarkResult:
    """Stage 4: independently encoded modalities must bind by shared event."""

    started = perf_counter()
    torch.manual_seed(seed)
    classes, width, sensor_dim = 16, 12, 6
    prototypes = F.normalize(torch.randn(classes, sensor_dim), dim=-1)

    def models():
        return TokenEncoder(classes, width), ContinuousSignalEncoder(
            sensor_dim, width, anti_alias=False
        )

    aligned_token, aligned_sensor = models()
    shuffled_token, shuffled_sensor = deepcopy(aligned_token), deepcopy(aligned_sensor)

    def train_pair(token: TokenEncoder, sensor: ContinuousSignalEncoder, shuffle: bool) -> None:
        params = list(token.parameters()) + list(sensor.parameters())
        permutation = torch.roll(torch.arange(classes), 5)

        def loss_fn() -> Tensor:
            labels = torch.arange(classes)
            token_features = F.normalize(token(labels[:, None]).values[:, 0], dim=-1)
            signal = prototypes + 0.04 * torch.randn_like(prototypes)
            sensor_features = F.normalize(
                sensor(
                    signal[:, None], torch.ones(classes, 1, dtype=torch.bool),
                    sample_interval=1.0,
                ).values[:, 0], dim=-1,
            )
            target = permutation if shuffle else labels
            logits = token_features @ sensor_features.T / 0.08
            return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2

        _optimize(
            loss_fn, params, steps=max(20, round(60 * steps_scale)), learning_rate=0.03,
        )

    train_pair(aligned_token, aligned_sensor, False)
    train_pair(shuffled_token, shuffled_sensor, True)

    @torch.no_grad()
    def accuracy(token: TokenEncoder, sensor: ContinuousSignalEncoder) -> float:
        labels = torch.arange(classes).repeat_interleave(8)
        token_features = F.normalize(token(torch.arange(classes)[:, None]).values[:, 0], dim=-1)
        signal = prototypes[labels] + 0.08 * torch.randn(labels.numel(), sensor_dim)
        sensor_features = F.normalize(
            sensor(
                signal[:, None], torch.ones(labels.numel(), 1, dtype=torch.bool),
                sample_interval=1.0,
            ).values[:, 0], dim=-1,
        )
        return float(((sensor_features @ token_features.T).argmax(-1) == labels).float().mean())

    aligned = accuracy(aligned_token, aligned_sensor)
    shuffled = accuracy(shuffled_token, shuffled_sensor)
    metrics = {
        "held_out_cross_modal_recall_at_1": aligned,
        "shuffled_pair_recall_at_1": shuffled,
        "binding_gain": aligned - shuffled,
        "chance_recall": 1 / classes,
        "parameter_ratio": _parameters(aligned_token, aligned_sensor)
        / _parameters(shuffled_token, shuffled_sensor),
    }
    criteria = (
        EmpiricalCriterion("held_out_cross_modal_recall_at_1", 0.90, "at_least"),
        EmpiricalCriterion("binding_gain", 0.65, "at_least"),
        EmpiricalCriterion("parameter_ratio", 1.0, "at_least"),
        EmpiricalCriterion("parameter_ratio", 1.0, "at_most"),
    )
    return _result(
        "cross_modal_event_binding", "Stage 4", seed, started,
        _parameters(aligned_token, aligned_sensor), metrics, criteria,
        "paired token/sensor event retrieval with an equal-parameter shuffled-pair control",
    )


def _graph_data(count: int, width: int, *, generator: torch.Generator) -> tuple[GraphFragment, Tensor]:
    labels = torch.arange(count).remainder(2)
    node_base = torch.stack((torch.linspace(-1, 1, width), torch.linspace(1, -1, width)))
    relation_base = torch.stack((torch.ones(width) * -0.6, torch.ones(width) * 0.6))
    nodes = node_base[labels, None].expand(-1, 3, -1).clone()
    nodes += 0.04 * torch.randn(count, 3, width, generator=generator)
    relations = relation_base[labels, None].expand(-1, 2, -1).clone()
    relations += 0.04 * torch.randn(count, 2, width, generator=generator)
    node_types = labels[:, None].expand(-1, 3).clone()
    relation_types = (labels + 2)[:, None].expand(-1, 2).clone()
    support = torch.zeros(count, 3, 3)
    support[..., 0] = torch.arange(3).float()
    support[..., 1] = support[..., 0] + 0.25
    support[..., 2] = support[..., 1]
    participants = torch.tensor([[[0, 1], [1, 2]]]).expand(count, -1, -1).clone()
    return GraphFragment(
        nodes, node_types, support,
        torch.arange(1, count * 3 + 1).reshape(count, 3),
        torch.ones(count, 3, dtype=torch.bool), relations, relation_types,
        participants, torch.arange(10_000, 10_000 + count * 2).reshape(count, 2),
        torch.ones(count, 2, dtype=torch.bool),
    ), labels


def benchmark_hierarchy(seed: int = 17, *, steps_scale: float = 1.0) -> EmpiricalBenchmarkResult:
    """Gate H: learned compression must save code while preserving held-out use."""

    started = perf_counter()
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    width = 6
    train, train_labels = _graph_data(64, width, generator=generator)
    valid, valid_labels = _graph_data(32, width, generator=generator)
    compressor = GraphCompressor(width, 8, 4, 6, 3, 2)
    latent_head = nn.Linear(8, 2)
    raw_head = nn.Linear(2 * width, 2)

    def raw_features(fragment: GraphFragment) -> Tensor:
        return torch.cat((fragment.node_content.mean(1), fragment.relation_content.mean(1)), -1)

    def loss_fn() -> Tensor:
        proposal = compressor(train)
        reconstruction = proposal.distortion.total.mean()
        latent_loss = F.cross_entropy(latent_head(proposal.latent), train_labels)
        raw_loss = F.cross_entropy(raw_head(raw_features(train)), train_labels)
        return reconstruction + 0.5 * latent_loss + 0.5 * raw_loss

    _optimize(
        loss_fn, list(compressor.parameters()) + list(latent_head.parameters())
        + list(raw_head.parameters()),
        steps=max(30, round(100 * steps_scale)), learning_rate=0.02,
    )
    with torch.no_grad():
        proposal = compressor(valid)
        latent_logits = latent_head(proposal.latent)
        raw_logits = raw_head(raw_features(valid))
        latent_accuracy = float((latent_logits.argmax(-1) == valid_labels).float().mean())
        raw_accuracy = float((raw_logits.argmax(-1) == valid_labels).float().mean())
        latent_loss = F.cross_entropy(latent_logits, valid_labels, reduction="none")
        raw_loss = F.cross_entropy(raw_logits, valid_labels, reduction="none")
        decision = compressor.decide(
            proposal, minimum_gain=1.0, maximum_distortion=0.75,
            held_out_loss_before=raw_loss, held_out_loss_after=latent_loss,
            prediction_tolerance=0.15,
        )
    metrics = {
        "mean_code_gain_bits": float(proposal.code.gain_bits.mean()),
        "held_out_distortion": float(proposal.distortion.total.mean()),
        "raw_predictive_accuracy": raw_accuracy,
        "compressed_predictive_accuracy": latent_accuracy,
        "predictive_accuracy_delta": latent_accuracy - raw_accuracy,
        "promotion_acceptance_rate": float(decision.accepted.float().mean()),
        "relation_distortion": float(proposal.distortion.relation.mean()),
    }
    criteria = (
        EmpiricalCriterion("mean_code_gain_bits", 1.0, "at_least"),
        EmpiricalCriterion("held_out_distortion", 0.75, "at_most"),
        EmpiricalCriterion("compressed_predictive_accuracy", 0.95, "at_least"),
        EmpiricalCriterion("predictive_accuracy_delta", -0.05, "at_least"),
        EmpiricalCriterion("promotion_acceptance_rate", 0.90, "at_least"),
    )
    return _result(
        "validated_hierarchical_compression", "H", seed, started,
        _parameters(compressor, latent_head, raw_head), metrics, criteria,
        "held-out recurring graph motifs with raw and compressed predictive heads",
    )


def benchmark_world_model(seed: int = 17, *, steps_scale: float = 1.0) -> EmpiricalBenchmarkResult:
    """Gate J: interventions must change learned predictions through action."""

    started = perf_counter()
    torch.manual_seed(seed)
    width, actions = 8, 3
    model = ActionConditionedWorldModel(
        width, actions, relation_families=4, observation_dim=3, horizons=(1,),
    )
    action_map = torch.randn(actions, width) * 0.8
    train_latent = torch.randn(256, width)
    train_graph = torch.randn(256, width) * 0.3
    train_ids = torch.arange(256).remainder(actions)
    train_actions = F.one_hot(train_ids, actions).float()
    target = 0.55 * train_latent + action_map[train_ids] + 0.15 * train_graph

    def loss_fn() -> Tensor:
        prediction = model(train_latent, train_graph, train_actions)
        return F.mse_loss(prediction.latent_mean[:, 0], target)

    _optimize(
        loss_fn, model.parameters(), steps=max(30, round(120 * steps_scale)),
        learning_rate=0.02,
    )
    with torch.no_grad():
        latent = torch.randn(128, width)
        graph = torch.randn(128, width) * 0.3
        ids = torch.arange(128).remainder(actions)
        action = F.one_hot(ids, actions).float()
        expected = 0.55 * latent + action_map[ids] + 0.15 * graph
        prediction = model(latent, graph, action).latent_mean[:, 0]
        ablated = model(latent, graph, torch.zeros_like(action)).latent_mean[:, 0]
        full_mse = F.mse_loss(prediction, expected)
        ablated_mse = F.mse_loss(ablated, expected)
        fixed = latent[:32]
        fixed_graph = graph[:32]
        action0 = F.one_hot(torch.zeros(32, dtype=torch.int64), actions).float()
        action1 = F.one_hot(torch.ones(32, dtype=torch.int64), actions).float()
        predicted_separation = (
            model(fixed, fixed_graph, action0).latent_mean[:, 0]
            - model(fixed, fixed_graph, action1).latent_mean[:, 0]
        ).norm(dim=-1).mean()
        true_separation = (action_map[0] - action_map[1]).norm()
    metrics = {
        "interventional_mse": float(full_mse),
        "action_ablated_mse": float(ablated_mse),
        "ablation_error_ratio": float(ablated_mse / full_mse.clamp_min(1e-8)),
        "counterfactual_separation_ratio": float(predicted_separation / true_separation),
    }
    criteria = (
        EmpiricalCriterion("interventional_mse", 0.03, "at_most"),
        EmpiricalCriterion("ablation_error_ratio", 5.0, "at_least"),
        EmpiricalCriterion("counterfactual_separation_ratio", 0.80, "at_least"),
    )
    return _result(
        "action_conditioned_intervention", "J", seed, started, _parameters(model),
        metrics, criteria,
        "held-out controlled transitions with the action channel ablated at evaluation",
    )


def benchmark_uncertainty(seed: int = 17, *, steps_scale: float = 1.0) -> EmpiricalBenchmarkResult:
    """Gate I: distributional coverage and alternative retention must be useful."""

    started = perf_counter()
    torch.manual_seed(seed)
    width = 8
    feature_map = nn.Sequential(nn.Linear(1, width), nn.Tanh())
    head = DistributionalPredictionHead(width, 2, 1, ensemble_heads=4)
    generator = torch.Generator().manual_seed(seed + 9)
    train_x = torch.rand(384, 1, generator=generator) * 2 - 1
    scale = 0.08 + 0.30 * train_x.abs()
    train_y = 1.4 * train_x + scale * torch.randn(384, 1, generator=generator)
    train_class = (train_y[:, 0] > 0).long()
    bootstrap = torch.rand(384, 4, generator=generator) > 0.25

    def loss_fn() -> Tensor:
        output = head(feature_map(train_x))
        error = train_y[:, None] - output.continuous_quantiles
        levels = output.quantile_levels[None, :, None]
        pinball = torch.maximum(levels * error, (levels - 1) * error).mean()
        ensemble_error = (output.ensemble_values - train_y[:, None]).square().squeeze(-1)
        ensemble = (ensemble_error * bootstrap).sum() / bootstrap.sum().clamp_min(1)
        categorical = F.cross_entropy(output.categorical_logits, train_class)
        return pinball + 0.4 * ensemble + 0.3 * categorical

    _optimize(
        loss_fn, list(feature_map.parameters()) + list(head.parameters()),
        steps=max(30, round(120 * steps_scale)), learning_rate=0.02,
    )
    with torch.no_grad():
        x = torch.linspace(-1, 1, 512)[:, None]
        eval_scale = 0.08 + 0.30 * x.abs()
        y = 1.4 * x + eval_scale * torch.randn(512, 1, generator=generator)
        output = head(feature_map(x))
        lower, upper = output.continuous_quantiles[:, 0, 0], output.continuous_quantiles[:, -1, 0]
        coverage = ((y[:, 0] >= lower) & (y[:, 0] <= upper)).float().mean()
        probabilities = output.categorical_logits.softmax(-1)
        calibration = OnlineCalibration(1, bins=10)
        state = calibration.update(
            calibration.initial_state(), probabilities, (y[:, 0] > 0).long(),
            torch.zeros(512, dtype=torch.int64), torch.ones(512, dtype=torch.bool),
        )
        report = calibration.report(state)
        id_epistemic = output.epistemic.mean()
        ood = torch.tensor([[-3.0], [3.0]])
        ood_epistemic = head(feature_map(ood)).epistemic.mean()

        hypotheses = HypothesisBank(4, 2, 1, 2, 1, 2)
        hstate = hypotheses.initial_state(1)
        context = torch.ones(1, 4)
        hstate = hypotheses.create(hstate, context, torch.tensor([True]))
        hstate = hypotheses.create(hstate, -context, torch.tensor([True]))
        ambiguous_count = hstate.effective_count.item()
        hstate = hypotheses.update_evidence(hstate, torch.tensor([[0.0, -6.0]]))
        resolved_count = hstate.effective_count.item()
    metrics = {
        "central_80_coverage": float(coverage),
        "classification_ece": float(report.expected_calibration_error[0]),
        "ood_to_id_epistemic_ratio": float(ood_epistemic / id_epistemic.clamp_min(1e-8)),
        "ambiguous_effective_hypotheses": ambiguous_count,
        "resolved_effective_hypotheses": resolved_count,
    }
    criteria = (
        EmpiricalCriterion("central_80_coverage", 0.70, "at_least"),
        EmpiricalCriterion("central_80_coverage", 0.90, "at_most"),
        EmpiricalCriterion("classification_ece", 0.10, "at_most"),
        EmpiricalCriterion("ood_to_id_epistemic_ratio", 1.05, "at_least"),
        EmpiricalCriterion("ambiguous_effective_hypotheses", 1.90, "at_least"),
        EmpiricalCriterion("resolved_effective_hypotheses", 1.10, "at_most"),
    )
    return _result(
        "calibrated_uncertainty_and_hypotheses", "I", seed, started,
        _parameters(feature_map, head, hypotheses), metrics, criteria,
        "heteroscedastic held-out regression/classification plus two-alternative evidence update",
    )


def benchmark_controller(seed: int = 17, *, steps_scale: float = 1.0) -> EmpiricalBenchmarkResult:
    """Gate K: adaptive microsteps must spend compute only on hard rows."""

    started = perf_counter()
    torch.manual_seed(seed)
    width, batch = 8, 128
    controller = AdaptiveController(
        width, goal_dim=2, uncertainty_channels=2, system_feature_dim=2,
        maximum_nodes=2, maximum_steps=3,
    )
    difficulty = torch.arange(batch).remainder(2).bool()
    workspace = torch.zeros(batch, width)
    workspace[:, 0] = torch.where(difficulty, 1.0, -1.0)
    workspace[:, 1:] = 0.04 * torch.randn(batch, width - 1)
    goals = torch.zeros(batch, 2)
    uncertainty = torch.zeros(batch, 2)
    system = torch.ones(batch, 2)
    hard_sequence = torch.tensor([
        int(InternalAction.RETRIEVE_EPISODIC), int(InternalAction.COMPARE),
        int(InternalAction.HALT),
    ])

    def loss_fn() -> Tensor:
        hidden = torch.zeros(batch, width)
        total = workspace.new_zeros(())
        features = torch.cat((workspace, goals, uncertainty, system), -1)
        for step in range(3):
            hidden = controller.recurrence(F.silu(controller.input(features)), hidden)
            target = torch.where(
                difficulty, hard_sequence[step], torch.tensor(int(InternalAction.HALT))
            )
            active = difficulty | (step == 0)
            action_loss = F.cross_entropy(controller.action_head(hidden)[active], target[active])
            halt_target = ((~difficulty) | (step == 2)).float()
            halt_loss = F.binary_cross_entropy_with_logits(
                controller.halt_head(hidden).squeeze(-1)[active], halt_target[active]
            )
            total = total + action_loss + halt_loss
        return total

    _optimize(
        loss_fn, controller.parameters(), steps=max(30, round(100 * steps_scale)),
        learning_rate=0.02,
    )
    with torch.no_grad():
        nodes = torch.randn(batch, 2, width)
        rollout = controller(
            workspace, goals, uncertainty, system, nodes,
            torch.ones(batch, 2, dtype=torch.bool), ponder_weight=0.05,
        )
        history = rollout.state.action_history
        easy_success = (history[~difficulty, 0] == int(InternalAction.HALT)).float().mean()
        hard_success = (
            (history[difficulty, 0] == hard_sequence[0])
            & (history[difficulty, 1] == hard_sequence[1])
            & (history[difficulty, 2] == hard_sequence[2])
        ).float().mean()
        steps = rollout.state.history_mask.sum(-1).float()
        easy_steps, hard_steps = steps[~difficulty].mean(), steps[difficulty].mean()
        adaptive_utility = (torch.cat((
            (history[~difficulty, 0] == int(InternalAction.HALT)).float(),
            ((history[difficulty, 0] == hard_sequence[0])
             & (history[difficulty, 1] == hard_sequence[1])
             & (history[difficulty, 2] == hard_sequence[2])).float(),
        )).mean() - 0.05 * steps.mean())
        fixed_one_utility = 0.5 - 0.05
        # The adaptive policy uses one step on half the rows and three on the
        # other half, hence exactly two steps on average.  A fixed-two policy
        # is the compute-matched baseline and cannot complete the three-action
        # hard trace.  Fixed-three is retained as the maximum-compute control.
        fixed_two_utility = 0.5 - 0.10
        fixed_three_utility = 1.0 - 0.15
    metrics = {
        "easy_success": float(easy_success),
        "hard_success": float(hard_success),
        "easy_mean_steps": float(easy_steps),
        "hard_mean_steps": float(hard_steps),
        "adaptive_net_utility": float(adaptive_utility),
        "fixed_one_net_utility": fixed_one_utility,
        "matched_fixed_two_net_utility": fixed_two_utility,
        "fixed_three_net_utility": fixed_three_utility,
        "matched_baseline_gain": float(adaptive_utility - fixed_two_utility),
    }
    criteria = (
        EmpiricalCriterion("easy_success", 0.95, "at_least"),
        EmpiricalCriterion("hard_success", 0.95, "at_least"),
        EmpiricalCriterion("easy_mean_steps", 1.1, "at_most"),
        EmpiricalCriterion("hard_mean_steps", 2.9, "at_least"),
        EmpiricalCriterion("matched_baseline_gain", 0.02, "at_least"),
    )
    return _result(
        "adaptive_internal_compute", "K", seed, started, _parameters(controller),
        metrics, criteria,
        "easy/hard action traces against fixed-one, compute-matched fixed-two, and fixed-three policies",
    )


def _oracle_critic(context: Tensor, mask: Tensor, *, horizons: int) -> AdjointCriticOutput:
    batch, time = mask.shape
    actions, bootstrap, quantiles = 2, 1, 3
    correct = (context[:, 0, 0] > 0).long()
    values = context.new_full((batch, time, bootstrap, horizons, actions), -1.0)
    for row in range(batch):
        values[row, :, :, :, correct[row]] = 1.0
    return AdjointCriticOutput(
        context.new_zeros(batch, time, bootstrap, horizons, quantiles), values,
        context.new_zeros(batch, time, actions), context.new_zeros(batch, time, actions),
        (), (), context.new_zeros(batch, time, actions), context, context,
        context.new_full((batch, time, bootstrap, horizons), 0.05),
        context.new_full((batch, time, bootstrap, horizons), 0.10), mask,
    )


def benchmark_consequence_learning(
    seed: int = 17, *, steps_scale: float = 1.0,
) -> EmpiricalBenchmarkResult:
    """Gate L: FSCE must use delayed consequence that behavior CE cannot see."""

    started = perf_counter()
    torch.manual_seed(seed)
    batch, time = 128, 3
    contexts = torch.where(
        torch.arange(batch).remainder(2)[:, None, None].bool(),
        torch.ones(batch, time, 1), -torch.ones(batch, time, 1),
    )
    correct = (contexts[:, 0, 0] > 0).long()
    policy = nn.Linear(1, 2)
    ce_control = deepcopy(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.04)
    control_optimizer = torch.optim.Adam(ce_control.parameters(), lr=0.04)
    config = ResonantAdjointSurpriseConfig(
        horizons=(1, 3), quantiles=(0.1, 0.5, 0.9), bootstrap_heads=1,
        critic_scales=1, critic_heads=1, critic_modes=1, spectral_modes=1,
        latent_modes=1,
    )
    calibrator = FunctionalSurpriseCalibrator(2, 1)
    interaction_steps = max(30, round(90 * steps_scale))
    for _ in range(interaction_steps):
        with torch.no_grad():
            behavior = torch.randint(0, 2, (batch, time))
            rewards = torch.zeros(batch, time)
            rewards[:, -1] = torch.where(behavior[:, 0] == correct, 1.0, -1.0)
            dones = torch.zeros(batch, time, dtype=torch.bool)
            dones[:, -1] = True
            full_mask = torch.ones(batch, time, dtype=torch.bool)
            returns, _ = multihorizon_returns(
                rewards, dones, full_mask, config.horizons, discount=1.0,
            )
        actor_logits = policy(contexts)
        target_logits = actor_logits.detach()
        critic_mask = torch.zeros(batch, time, dtype=torch.bool)
        critic_mask[:, 0] = True
        critic = _oracle_critic(contexts, critic_mask, horizons=2)
        surprise = functional_surprise_target(
            actor_logits, target_logits, critic, behavior, returns,
            torch.zeros(batch, time, 1), calibrator, config,
            update_calibration=True,
        )
        optimizer.zero_grad(set_to_none=True)
        fsce = -(surprise.distribution[:, 0] * F.log_softmax(actor_logits[:, 0], -1)).sum(-1).mean()
        fsce.backward()
        optimizer.step()

        control_optimizer.zero_grad(set_to_none=True)
        control_logits = ce_control(contexts[:, 0])
        control_loss = F.cross_entropy(control_logits, behavior[:, 0])
        control_loss.backward()
        control_optimizer.step()

    with torch.no_grad():
        fsce_probability = policy(contexts[:, 0]).softmax(-1).gather(1, correct[:, None]).mean()
        ce_probability = ce_control(contexts[:, 0]).softmax(-1).gather(1, correct[:, None]).mean()
        fsce_return = 2 * fsce_probability - 1
        ce_return = 2 * ce_probability - 1
    guard = PerformanceGuard(tolerance=0.01)
    guard.allows(float(fsce_return), 1.0)
    vetoed = not guard.allows(float(fsce_return - 0.2), 0.5)
    metrics = {
        "fsce_correct_action_probability": float(fsce_probability),
        "ce_only_correct_action_probability": float(ce_probability),
        "delayed_return_gain": float(fsce_return - ce_return),
        "equal_environment_interactions": 1.0,
        "performance_veto_triggered": float(vetoed),
        "calibrator_updates": float(calibrator.updates),
    }
    criteria = (
        EmpiricalCriterion("fsce_correct_action_probability", 0.85, "at_least"),
        EmpiricalCriterion("delayed_return_gain", 0.60, "at_least"),
        EmpiricalCriterion("equal_environment_interactions", 1.0, "at_least"),
        EmpiricalCriterion("performance_veto_triggered", 1.0, "at_least"),
    )
    return _result(
        "delayed_functional_surprise", "L", seed, started,
        _parameters(policy), metrics, criteria,
        "three-step delayed contextual consequence with equal-interaction random-behavior CE control",
    )


def _binary_accuracy(model: nn.Module, x: Tensor, y: Tensor) -> float:
    with torch.no_grad():
        return float((model(x).argmax(-1) == y).float().mean())


def benchmark_continual_adaptation(
    seed: int = 17, *, steps_scale: float = 1.0,
) -> EmpiricalBenchmarkResult:
    """Stage 9: replay, isolated adaptation, validation, revocation, rollback."""

    started = perf_counter()
    torch.manual_seed(seed)
    x = torch.randn(512, 2)
    task_a = (x[:, 0] > 0).long()
    task_b = (x[:, 1] > 0).long()
    base = nn.Linear(2, 2)
    _optimize(
        lambda: F.cross_entropy(base(x), task_a), base.parameters(),
        steps=max(20, round(60 * steps_scale)), learning_rate=0.05,
    )
    before_a = _binary_accuracy(base, x, task_a)
    rollback_state = deepcopy(base.state_dict())
    naive = deepcopy(base)
    replay = deepcopy(base)
    adapter = nn.Linear(2, 2)
    _optimize(
        lambda: F.cross_entropy(naive(x), task_b), naive.parameters(),
        steps=max(20, round(60 * steps_scale)), learning_rate=0.05,
    )
    _optimize(
        lambda: F.cross_entropy(replay(x), task_b) + F.cross_entropy(replay(x), task_a),
        replay.parameters(), steps=max(20, round(60 * steps_scale)), learning_rate=0.05,
    )
    _optimize(
        lambda: F.cross_entropy(adapter(x), task_b), adapter.parameters(),
        steps=max(20, round(60 * steps_scale)), learning_rate=0.05,
    )
    naive_a = _binary_accuracy(naive, x, task_a)
    replay_a = _binary_accuracy(replay, x, task_a)
    replay_b = _binary_accuracy(replay, x, task_b)
    adapter_b = _binary_accuracy(adapter, x, task_b)
    base.load_state_dict(rollback_state)
    rollback_exact = all(
        torch.equal(value, rollback_state[name]) for name, value in base.state_dict().items()
    )

    ledger = ProvenanceLedger()
    root = ledger.append(
        source_class=SourceClass.EXTERNAL, source_uri_or_episode="empirical://task-b",
        support=SupportInterval(0, 1, 1), modality=ModalityClass.SYMBOLIC,
        operator="observe", scenario_id=0, model_authority="empirical-suite",
        verification=VerificationClass.EXTERNALLY_CHECKED,
    )
    derived = ledger.derive(
        [root], source_class=SourceClass.ABSTRACTED, operator="adapter-proposal",
        support=SupportInterval(0, 1, 1), modality=ModalityClass.SYMBOLIC,
        model_authority="empirical-suite",
    )
    state = KnowledgeProposalState.empty(1, 2, 4, 2)
    proposals = KnowledgeProposalBatch(
        torch.randn(1, 4), torch.tensor([int(KnowledgeKind.ABSTRACTION)]),
        torch.tensor([12.0]), torch.tensor([0.05]), torch.tensor([0.04]),
        torch.tensor([adapter_b - 0.5]), torch.tensor([0.0]), torch.tensor([adapter_b]),
        torch.tensor([derived]), torch.tensor([[root, -1]]),
        torch.tensor([[True, False]]), torch.tensor([True]),
    )
    state, indices = KnowledgeProposalBank.propose(state, proposals)
    evidence = KnowledgeValidationBatch(
        indices, torch.tensor([1.0]), torch.tensor([1.0 - adapter_b]),
        torch.tensor([adapter_b - 0.5]), torch.tensor([0.0]), torch.tensor([adapter_b]),
        torch.tensor([True]), torch.tensor([True]),
    )
    state, validation = KnowledgeProposalBank.validate(
        state, evidence, ledger, minimum_code_gain_bits=1.0,
        maximum_reconstruction_distortion=0.2, maximum_relation_distortion=0.2,
    )
    state = KnowledgeProposalBank.revoke(state, indices, torch.tensor([True]))
    revoked = state.status[0, indices[0]] == int(KnowledgeStatus.REVOKED)
    metrics = {
        "base_task_a_accuracy": before_a,
        "naive_task_a_accuracy": naive_a,
        "replay_task_a_accuracy": replay_a,
        "replay_task_b_accuracy": replay_b,
        "isolated_adapter_task_b_accuracy": adapter_b,
        "replay_retention_gain": replay_a - naive_a,
        "accepted_after_validation": float(validation.accepted[0]),
        "revocation_recorded": float(revoked),
        "rollback_exact": float(rollback_exact),
    }
    criteria = (
        EmpiricalCriterion("base_task_a_accuracy", 0.95, "at_least"),
        EmpiricalCriterion("replay_task_a_accuracy", 0.70, "at_least"),
        EmpiricalCriterion("replay_task_b_accuracy", 0.70, "at_least"),
        EmpiricalCriterion("isolated_adapter_task_b_accuracy", 0.95, "at_least"),
        EmpiricalCriterion("replay_retention_gain", 0.15, "at_least"),
        EmpiricalCriterion("accepted_after_validation", 1.0, "at_least"),
        EmpiricalCriterion("revocation_recorded", 1.0, "at_least"),
        EmpiricalCriterion("rollback_exact", 1.0, "at_least"),
    )
    return _result(
        "consolidation_replay_and_rollback", "Stage 9", seed, started,
        _parameters(base, adapter), metrics, criteria,
        "two-task continual adaptation with naive, replay, isolated-adapter, authority, and rollback paths",
    )


BENCHMARKS: tuple[Callable[..., EmpiricalBenchmarkResult], ...] = (
    benchmark_retrieval,
    benchmark_multimodal_binding,
    benchmark_hierarchy,
    benchmark_uncertainty,
    benchmark_world_model,
    benchmark_controller,
    benchmark_consequence_learning,
    benchmark_continual_adaptation,
)


def run_empirical_acceptance(
    *, seed: int = 17, steps_scale: float = 1.0, device: str = "cpu",
) -> EmpiricalAcceptanceReport:
    """Run every bounded learned-behavior gate on a deterministic local device."""

    if steps_scale <= 0:
        raise ValueError("empirical training step scale must be positive")
    if device != "cpu":
        raise ValueError(
            "the reproducible acceptance authority currently requires CPU; "
            "hardware throughput belongs to a separate benchmark"
        )
    results = tuple(
        benchmark(seed=seed + index * 1009, steps_scale=steps_scale)
        for index, benchmark in enumerate(BENCHMARKS)
    )
    return EmpiricalAcceptanceReport(
        2, "mrcra-bounded-learned-behavior-v2", torch.__version__, device,
        results, all(result.passed for result in results), False, False,
        "Passing proves bounded mechanism learnability and matched local effects; "
        "it does not prove serious-scale, open-domain, or deployment capability.",
        "mechanism",
    )
