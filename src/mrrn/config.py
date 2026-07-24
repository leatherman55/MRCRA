"""Validated architecture configuration and scale-allocation rules."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log, log2, pi, sqrt


def _positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True, slots=True)
class ScaleConfig:
    """Compute allocation for one resolution scale."""

    width: int
    heads: int
    modes: int
    mimo_rank: int
    attention_window: int

    def __post_init__(self) -> None:
        for name in ("width", "heads", "modes", "mimo_rank", "attention_window"):
            _positive(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class MRRNConfig:
    """Complete baseline configuration with deterministic scale derivation."""

    input_dim: int
    model_dim: int = 256
    output_dim: int | None = None
    layers: int = 12
    scales: int = 5
    heads: int = 4
    modes: int = 16
    mimo_rank: int = 2
    attention_window: int = 128
    attention_query_tile_size: int = 256
    retrieved_items: int = 16
    memory_capacity: int = 2048
    lifting_kernel: int = 3
    mixer_expansion: float = 3.0
    width_growth_cap: float = 2.0
    mode_growth_cap: float = 2.0
    width_multiple: int = 8
    alpha_min: float = 1e-4
    delta_min: float = 1e-4
    omega_max: float = 3.141592653589793
    residual_scale: float = 1e-2
    continuous_signal: bool = False
    causal: bool = True
    share_depth_parameters: bool = False
    structured_mixer_rank: int | None = None
    spectral_activation: bool = True
    spectral_modes: int = 8
    spectral_basis_order: int = 6
    spectral_maximum_gain: float = 2.0
    spectral_maximum_phase: float = pi / 8
    spectral_triads_per_mode: int = 2
    spectral_maximum_triad_gain: float = 0.25
    decay_normalized_resonance: bool = True
    enable_global_head: bool = True
    relational_branch: bool = False
    relational_context_dim: int | None = None
    activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        integer_fields = (
            "input_dim",
            "model_dim",
            "layers",
            "scales",
            "heads",
            "modes",
            "mimo_rank",
            "attention_window",
            "attention_query_tile_size",
            "retrieved_items",
            "memory_capacity",
            "lifting_kernel",
            "width_multiple",
            "spectral_modes",
            "spectral_basis_order",
        )
        for name in integer_fields:
            _positive(name, getattr(self, name))
        for name in (
            "mixer_expansion",
            "width_growth_cap",
            "mode_growth_cap",
            "alpha_min",
            "delta_min",
            "omega_max",
            "residual_scale",
        ):
            _positive(name, getattr(self, name))
        if self.output_dim is not None:
            _positive("output_dim", self.output_dim)
        if self.structured_mixer_rank is not None:
            _positive("structured_mixer_rank", self.structured_mixer_rank)
        if self.relational_context_dim is not None:
            _positive("relational_context_dim", self.relational_context_dim)
        if self.relational_branch and self.relational_context_dim is None:
            raise ValueError("relational_branch requires relational_context_dim")
        if not self.relational_branch and self.relational_context_dim is not None:
            raise ValueError("relational_context_dim requires relational_branch")
        if self.spectral_maximum_gain <= 1:
            raise ValueError("spectral_maximum_gain must exceed one")
        if not 0 <= self.spectral_maximum_phase <= pi:
            raise ValueError("spectral_maximum_phase must lie in [0,pi]")
        if self.spectral_triads_per_mode < 0 or self.spectral_maximum_triad_gain < 0:
            raise ValueError("spectral triad controls cannot be negative")
        if self.lifting_kernel % 2 == 0:
            raise ValueError("lifting_kernel must be odd")
        if self.retrieved_items > self.memory_capacity:
            raise ValueError("retrieved_items cannot exceed memory_capacity")
        if self.omega_max > 3.141592653589793:
            raise ValueError("omega_max cannot exceed the normalized Nyquist limit pi")

    @staticmethod
    def _round_width(value: float, multiple: int) -> int:
        return max(multiple, int(round(value / multiple)) * multiple)

    def scale_configs(self) -> tuple[ScaleConfig, ...]:
        """Allocate more coarse capacity while keeping geometric total work bounded."""

        result = []
        for scale in range(self.scales):
            growth = sqrt(2.0**scale)
            width = self._round_width(
                self.model_dim * min(growth, self.width_growth_cap), self.width_multiple
            )
            modes = max(1, round(self.modes * min(growth, self.mode_growth_cap)))
            result.append(
                ScaleConfig(
                    width=width,
                    heads=self.heads,
                    modes=modes,
                    mimo_rank=self.mimo_rank,
                    attention_window=self.attention_window,
                )
            )
        return tuple(result)

    @property
    def resolved_output_dim(self) -> int:
        return self.input_dim if self.output_dim is None else self.output_dim

    @classmethod
    def research(cls, input_dim: int, output_dim: int | None = None, **overrides) -> "MRRNConfig":
        defaults = dict(
            model_dim=256, output_dim=output_dim, layers=12, scales=5, heads=4,
            modes=16, mimo_rank=2, attention_window=128, retrieved_items=16,
            memory_capacity=2048,
        )
        defaults.update(overrides)
        return cls(input_dim=input_dim, **defaults)

    @classmethod
    def capability_first(cls, input_dim: int, output_dim: int | None = None, **overrides) -> "MRRNConfig":
        defaults = dict(
            model_dim=512, output_dim=output_dim, layers=24, scales=7, heads=8,
            modes=32, mimo_rank=4, attention_window=128, retrieved_items=32,
            memory_capacity=8192, mixer_expansion=4, spectral_modes=16,
            spectral_basis_order=8, spectral_triads_per_mode=4,
        )
        defaults.update(overrides)
        return cls(input_dim=input_dim, **defaults)

    @classmethod
    def efficiency_first(cls, input_dim: int, output_dim: int | None = None, **overrides) -> "MRRNConfig":
        defaults = dict(
            model_dim=256, output_dim=output_dim, layers=12, scales=5, heads=4,
            modes=16, mimo_rank=2, attention_window=64, retrieved_items=8,
            memory_capacity=2048, mixer_expansion=2.5, share_depth_parameters=True,
            structured_mixer_rank=32, spectral_modes=6, spectral_basis_order=4,
            spectral_triads_per_mode=1,
        )
        defaults.update(overrides)
        return cls(input_dim=input_dim, **defaults)

    @classmethod
    def mrcra_120m_carrier(
        cls, input_dim: int = 256, output_dim: int | None = None, **overrides,
    ) -> "MRRNConfig":
        """Canonical six-scale carrier for the serious MRCRA actor.

        The final parameter count is enforced by :class:`MRCRAConfig` after the
        cognitive modules have been constructed.  This method only fixes the
        carrier allocation described by the architecture specification.
        """

        defaults = dict(
            model_dim=256,
            output_dim=output_dim,
            layers=6,
            scales=6,
            heads=8,
            modes=20,
            mimo_rank=2,
            attention_window=32,
            retrieved_items=8,
            memory_capacity=8192,
            lifting_kernel=3,
            mixer_expansion=2.0,
            width_growth_cap=1.125,
            mode_growth_cap=1.25,
            width_multiple=32,
            spectral_modes=8,
            spectral_basis_order=6,
            spectral_triads_per_mode=1,
            enable_global_head=False,
            activation_checkpointing=True,
            relational_branch=True,
            relational_context_dim=256,
        )
        defaults.update(overrides)
        return cls(input_dim=input_dim, **defaults)

    @classmethod
    def mrcra_light_8p4m_carrier(
        cls, input_dim: int = 96, output_dim: int | None = None, **overrides,
    ) -> "MRRNConfig":
        """Parameter-efficient five-scale carrier for the integrated light actor.

        Six recurrent refinement passes share one learned block.  This preserves
        iterative depth and independent recurrent state at each pass while
        spending parameters on a 96-wide representation, five physical scales,
        and the complete relational branch.  Coarser scales receive a modest
        width increase to 112 rather than duplicating full-width depth weights.
        """

        defaults = dict(
            model_dim=96,
            output_dim=output_dim,
            layers=6,
            scales=5,
            heads=4,
            modes=12,
            mimo_rank=2,
            attention_window=32,
            attention_query_tile_size=256,
            retrieved_items=8,
            memory_capacity=4096,
            lifting_kernel=3,
            mixer_expansion=2.0,
            width_growth_cap=1.125,
            mode_growth_cap=1.25,
            width_multiple=8,
            share_depth_parameters=True,
            structured_mixer_rank=8,
            spectral_modes=5,
            spectral_basis_order=5,
            spectral_triads_per_mode=1,
            enable_global_head=False,
            activation_checkpointing=True,
            relational_branch=True,
            relational_context_dim=96,
        )
        defaults.update(overrides)
        return cls(input_dim=input_dim, **defaults)

    @classmethod
    def mrcra_ultralight_1p3m_carrier(
        cls, input_dim: int = 20, output_dim: int | None = None, **overrides,
    ) -> "MRRNConfig":
        """Six-scale spectral carrier for the complete 1.3M MRCRA actor.

        With a 50,257-token vocabulary, tied token/output weights consume most
        of a 1.3M budget.  A 20-wide representation is the largest practical
        width that still leaves enough capacity for every cognitive subsystem.
        Learned depth is shared, but all six physical resolution scales retain
        independent recurrent state.  The profile preserves explicit
        resonance modes, spectral activation, structured mixing, local
        attention, memory retrieval, and the relational feedback branch.
        """

        defaults = dict(
            model_dim=20,
            output_dim=output_dim,
            layers=6,
            scales=6,
            heads=2,
            modes=8,
            mimo_rank=1,
            attention_window=32,
            attention_query_tile_size=256,
            retrieved_items=4,
            memory_capacity=1024,
            lifting_kernel=3,
            mixer_expansion=2.5,
            width_growth_cap=1.25,
            mode_growth_cap=1.33,
            width_multiple=4,
            share_depth_parameters=True,
            structured_mixer_rank=8,
            spectral_modes=5,
            spectral_basis_order=5,
            spectral_triads_per_mode=1,
            enable_global_head=False,
            activation_checkpointing=True,
            relational_branch=True,
            relational_context_dim=20,
        )
        defaults.update(overrides)
        return cls(input_dim=input_dim, **defaults)


@dataclass(frozen=True, slots=True)
class CognitiveConfig:
    """Bounded capacities and dimensions for the relational-continuity layer."""

    workspace_dim: int = 256
    provenance_features: int = 16
    uncertainty_channels: int = 8
    modality_count: int = 16
    node_type_count: int = 13
    relation_family_count: int = 16
    relation_heads: int = 8
    relation_modes: int = 8
    relation_adapter_rank: int = 8
    goal_slots: int = 8
    goal_constraint_dim: int = 8
    system_action_channels: int = 8
    calibration_regimes: int = 8
    calibration_bins: int = 15
    operational_schema_count: int = 8
    knowledge_candidate_capacity: int = 16
    knowledge_support_capacity: int = 16
    reconstruction_capacity: int = 16
    action_candidate_capacity: int = 8
    action_argument_dim: int = 8
    evidence_request_capacity: int = 4
    external_artifact_capacity: int = 16
    external_artifact_digest_width: int = 32
    viability_channels: int = 8
    metacognitive_capacity: int = 16
    active_event_capacity: int = 256
    pair_edge_capacity: int = 2048
    hyperedge_capacity: int = 128
    maximum_hyperedge_arity: int = 4
    graph_neighbors: int = 8
    global_workspace_slots: int = 16
    hypothesis_slots: int = 4
    maximum_hypothesis_slots: int = 8
    planning_hypothesis_top_k: int = 4
    maximum_cognitive_steps: int = 4
    event_chunk_size: int = 256
    event_proposals_per_chunk: int = 8
    recent_candidates: int = 32
    landmark_candidates: int = 8
    episodic_candidates: int = 8
    semantic_candidates: int = 8
    episodic_memory_capacity: int = 8192
    semantic_memory_capacity: int = 2048
    associative_depth: int = 3
    associative_budget: int = 16
    world_model_horizons: tuple[int, ...] = (1, 4, 16, 64)
    broadcast_gain_maximum: float = 0.1
    workspace_update_maximum: float = 0.5
    abstraction_minimum_gain: float = 0.0
    abstraction_maximum_distortion: float = 0.1
    deliberation_prediction_error_threshold: float = 1.0
    minimum_routed_posterior_mass: float = 0.8
    enable_conditional_reconstruction: bool = False
    enable_abstraction_validity_control: bool = False
    enable_post_deliberation_action_selection: bool = False
    enable_multi_hypothesis_planning: bool = False
    enable_agent_session_loop: bool = False
    enable_viability_gate: bool = False
    enable_integrated_invariant_discovery: bool = False
    enable_persistent_session_training: bool = False
    enable_metacognitive_routing: bool = False

    def __post_init__(self) -> None:
        integer_fields = (
            "workspace_dim", "provenance_features", "uncertainty_channels",
            "modality_count", "node_type_count", "relation_family_count",
            "relation_heads", "relation_modes", "relation_adapter_rank",
            "goal_slots", "goal_constraint_dim", "system_action_channels",
            "calibration_regimes", "calibration_bins", "operational_schema_count",
            "knowledge_candidate_capacity", "knowledge_support_capacity",
            "reconstruction_capacity", "action_candidate_capacity",
            "action_argument_dim", "evidence_request_capacity",
            "external_artifact_capacity", "external_artifact_digest_width",
            "viability_channels", "metacognitive_capacity",
            "active_event_capacity", "pair_edge_capacity", "hyperedge_capacity",
            "maximum_hyperedge_arity", "graph_neighbors", "global_workspace_slots",
            "hypothesis_slots", "maximum_hypothesis_slots", "planning_hypothesis_top_k",
            "maximum_cognitive_steps",
            "event_chunk_size", "event_proposals_per_chunk", "recent_candidates",
            "landmark_candidates", "episodic_candidates", "semantic_candidates",
            "episodic_memory_capacity", "semantic_memory_capacity",
            "associative_depth", "associative_budget",
        )
        for name in integer_fields:
            _positive(name, getattr(self, name))
        if not self.world_model_horizons or any(value <= 0 for value in self.world_model_horizons):
            raise ValueError("world_model_horizons must contain positive steps")
        if tuple(sorted(set(self.world_model_horizons))) != self.world_model_horizons:
            raise ValueError("world_model_horizons must be unique and increasing")
        if self.workspace_dim % self.relation_heads:
            raise ValueError("workspace_dim must be divisible by relation_heads")
        if self.hypothesis_slots > self.maximum_hypothesis_slots:
            raise ValueError("hypothesis_slots cannot exceed maximum_hypothesis_slots")
        if self.graph_neighbors > self.active_event_capacity:
            raise ValueError("graph_neighbors cannot exceed active_event_capacity")
        if self.event_proposals_per_chunk > self.active_event_capacity:
            raise ValueError("event proposal quota cannot exceed event capacity")
        if self.maximum_hyperedge_arity > self.active_event_capacity:
            raise ValueError("hyperedge arity cannot exceed event capacity")
        if not 0 < self.broadcast_gain_maximum <= 1:
            raise ValueError("broadcast_gain_maximum must lie in (0,1]")
        if not 0 < self.workspace_update_maximum < 1:
            raise ValueError("workspace_update_maximum must lie in (0,1)")
        if self.abstraction_minimum_gain < 0 or self.abstraction_maximum_distortion < 0:
            raise ValueError("compression thresholds cannot be negative")
        if self.deliberation_prediction_error_threshold < 0:
            raise ValueError("deliberation prediction-error threshold cannot be negative")
        if not 0 < self.minimum_routed_posterior_mass <= 1:
            raise ValueError("minimum routed posterior mass must lie in (0,1]")
        for name in (
            "enable_conditional_reconstruction", "enable_abstraction_validity_control",
            "enable_post_deliberation_action_selection", "enable_multi_hypothesis_planning",
            "enable_agent_session_loop", "enable_viability_gate",
            "enable_integrated_invariant_discovery", "enable_persistent_session_training",
            "enable_metacognitive_routing",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")

    @property
    def maximum_router_candidates(self) -> int:
        """Worst-case candidates scored exactly by one relational query."""

        return (
            self.recent_candidates + self.landmark_candidates
            + self.global_workspace_slots + self.graph_neighbors
            + self.episodic_candidates + self.semantic_candidates
        )


@dataclass(frozen=True, slots=True)
class MRCRAConfig:
    """Complete carrier-plus-cognition configuration."""

    carrier: MRRNConfig
    cognitive: CognitiveConfig = CognitiveConfig()
    actor_parameter_minimum: int = 110_000_000
    actor_parameter_maximum: int = 125_000_000

    def __post_init__(self) -> None:
        if not self.carrier.causal:
            raise ValueError("MRCRA online cognition requires a causal carrier")
        if self.carrier.model_dim != self.cognitive.workspace_dim:
            raise ValueError("carrier model_dim must equal cognitive workspace_dim")
        if not self.carrier.relational_branch:
            raise ValueError("MRCRA requires the carrier's fifth relational branch")
        if self.carrier.relational_context_dim != self.cognitive.workspace_dim:
            raise ValueError("carrier relational_context_dim must equal cognitive workspace_dim")
        if self.actor_parameter_minimum <= 0:
            raise ValueError("actor_parameter_minimum must be positive")
        if self.actor_parameter_maximum < self.actor_parameter_minimum:
            raise ValueError("actor parameter bounds are invalid")

    @classmethod
    def serious_120m(
        cls, *, input_dim: int = 256, output_dim: int | None = None,
        carrier_overrides: dict | None = None, cognitive_overrides: dict | None = None,
    ) -> "MRCRAConfig":
        carrier = MRRNConfig.mrcra_120m_carrier(
            input_dim, output_dim, **(carrier_overrides or {})
        )
        # The serious profile is the integrated MRCRA, not the compatibility
        # substrate.  External effects still fail closed because raw system
        # permissions and viability authority are empty until an application
        # session supplies them.
        mature = {
            "enable_conditional_reconstruction": True,
            "enable_abstraction_validity_control": True,
            "enable_post_deliberation_action_selection": True,
            "enable_multi_hypothesis_planning": True,
            "enable_agent_session_loop": True,
            "enable_viability_gate": True,
            "enable_integrated_invariant_discovery": True,
            "enable_persistent_session_training": True,
            "enable_metacognitive_routing": True,
        }
        mature.update(cognitive_overrides or {})
        cognitive = CognitiveConfig(**mature)
        return cls(carrier, cognitive)

    @classmethod
    def light_8p4m(
        cls, *, input_dim: int = 96, output_dim: int | None = None,
        carrier_overrides: dict | None = None, cognitive_overrides: dict | None = None,
    ) -> "MRCRAConfig":
        """Complete 8.4M-class MRCRA profile for the GPT-2 vocabulary.

        The profile reduces bounded runtime capacities and uses shared-depth
        carrier refinement, but retains every integrated cognitive mechanism.
        Its declared parameter band deliberately rejects accidental changes to
        the tokenizer width or architecture that would make the name untrue.
        """

        carrier = MRRNConfig.mrcra_light_8p4m_carrier(
            input_dim, output_dim, **(carrier_overrides or {})
        )
        mature = {
            "workspace_dim": 96,
            "relation_heads": 4,
            "relation_modes": 8,
            "relation_adapter_rank": 8,
            "active_event_capacity": 128,
            "pair_edge_capacity": 512,
            "hyperedge_capacity": 64,
            "graph_neighbors": 8,
            "global_workspace_slots": 12,
            "event_chunk_size": 128,
            "event_proposals_per_chunk": 8,
            "recent_candidates": 24,
            "landmark_candidates": 8,
            "episodic_candidates": 8,
            "semantic_candidates": 8,
            "episodic_memory_capacity": 4096,
            "semantic_memory_capacity": 1024,
            "enable_conditional_reconstruction": True,
            "enable_abstraction_validity_control": True,
            "enable_post_deliberation_action_selection": True,
            "enable_multi_hypothesis_planning": True,
            "enable_agent_session_loop": True,
            "enable_viability_gate": True,
            "enable_integrated_invariant_discovery": True,
            "enable_persistent_session_training": True,
            "enable_metacognitive_routing": True,
        }
        mature.update(cognitive_overrides or {})
        cognitive = CognitiveConfig(**mature)
        return cls(
            carrier, cognitive,
            actor_parameter_minimum=8_350_000,
            actor_parameter_maximum=8_450_000,
        )

    @classmethod
    def ultralight_1p3m(
        cls, *, input_dim: int = 20, output_dim: int | None = None,
        carrier_overrides: dict | None = None, cognitive_overrides: dict | None = None,
    ) -> "MRCRAConfig":
        """Complete 1.3M-class MRCRA profile for the GPT-2 vocabulary.

        This profile compresses dimensions, ranks, and bounded runtime
        capacities without deleting mechanisms.  It retains the six-scale
        MRRN carrier and every integrated cognitive pathway used by the light
        and serious actors.  Its narrow declared band fails closed if tokenizer
        width or architecture drift makes the ``1.3M`` name inaccurate.
        """

        carrier = MRRNConfig.mrcra_ultralight_1p3m_carrier(
            input_dim, output_dim, **(carrier_overrides or {})
        )
        mature = {
            "workspace_dim": 20,
            "provenance_features": 8,
            "uncertainty_channels": 8,
            "relation_heads": 2,
            "relation_modes": 8,
            "relation_adapter_rank": 6,
            "goal_slots": 4,
            "goal_constraint_dim": 4,
            "system_action_channels": 4,
            "calibration_regimes": 4,
            "calibration_bins": 15,
            "operational_schema_count": 4,
            "knowledge_candidate_capacity": 8,
            "knowledge_support_capacity": 8,
            "reconstruction_capacity": 8,
            "action_candidate_capacity": 4,
            "action_argument_dim": 4,
            "evidence_request_capacity": 2,
            "external_artifact_capacity": 8,
            "external_artifact_digest_width": 16,
            "viability_channels": 8,
            "metacognitive_capacity": 8,
            "active_event_capacity": 64,
            "pair_edge_capacity": 256,
            "hyperedge_capacity": 32,
            "maximum_hyperedge_arity": 4,
            "graph_neighbors": 4,
            "global_workspace_slots": 6,
            "hypothesis_slots": 2,
            "maximum_hypothesis_slots": 4,
            "planning_hypothesis_top_k": 2,
            "maximum_cognitive_steps": 4,
            "event_chunk_size": 64,
            "event_proposals_per_chunk": 4,
            "recent_candidates": 12,
            "landmark_candidates": 4,
            "episodic_candidates": 4,
            "semantic_candidates": 4,
            "episodic_memory_capacity": 1024,
            "semantic_memory_capacity": 256,
            "associative_depth": 2,
            "associative_budget": 8,
            "world_model_horizons": (1, 4, 16, 64),
            "enable_conditional_reconstruction": True,
            "enable_abstraction_validity_control": True,
            "enable_post_deliberation_action_selection": True,
            "enable_multi_hypothesis_planning": True,
            "enable_agent_session_loop": True,
            "enable_viability_gate": True,
            "enable_integrated_invariant_discovery": True,
            "enable_persistent_session_training": True,
            "enable_metacognitive_routing": True,
        }
        mature.update(cognitive_overrides or {})
        cognitive = CognitiveConfig(**mature)
        return cls(
            carrier, cognitive,
            actor_parameter_minimum=1_290_000,
            actor_parameter_maximum=1_310_000,
        )

    def require_actor_parameter_count(self, count: int) -> None:
        """Fail closed if a serious actor falls outside its declared budget."""

        if not self.actor_parameter_minimum <= count <= self.actor_parameter_maximum:
            raise ValueError(
                f"actor has {count:,} parameters; expected "
                f"{self.actor_parameter_minimum:,}..{self.actor_parameter_maximum:,}"
            )


def choose_scale_count(
    finest_event_width: int, direct_local_span: int, maximum_context: int, *, maximum_scales: int,
) -> int:
    if min(finest_event_width, direct_local_span, maximum_context, maximum_scales) <= 0:
        raise ValueError("scale decision spans must be positive")
    if direct_local_span < finest_event_width or maximum_context < direct_local_span:
        raise ValueError("event width <= local span <= maximum context is required")
    return min(maximum_scales, max(1, ceil(log2(maximum_context / direct_local_span)) + 1))


def decay_limits(
    minimum_half_life_steps: float, maximum_half_life_steps: float, sample_interval: float,
) -> tuple[float, float]:
    if min(minimum_half_life_steps, maximum_half_life_steps, sample_interval) <= 0 or maximum_half_life_steps < minimum_half_life_steps:
        raise ValueError("half-life range and sample interval are invalid")
    return (
        log(2) / (maximum_half_life_steps * sample_interval),
        log(2) / (minimum_half_life_steps * sample_interval),
    )


def usable_frequency_limit(sample_interval: float, transition_fraction: float = 0.1) -> float:
    if sample_interval <= 0 or not 0 <= transition_fraction < 1:
        raise ValueError("frequency sampling controls are invalid")
    return (1 - transition_fraction) * pi / sample_interval


def memory_capacity(write_rate: float, retention_horizon: int, *, burst_factor: float = 1.25) -> int:
    if write_rate < 0 or retention_horizon <= 0 or burst_factor < 1:
        raise ValueError("memory sizing controls are invalid")
    return max(1, ceil(write_rate * retention_horizon * burst_factor))


def multiresolution_cost_factor(growth_exponent: float) -> float:
    if not 0 <= growth_exponent < 1:
        raise ValueError("growth exponent must lie in [0,1)")
    return 1 / (1 - 2 ** (-(1 - growth_exponent)))
