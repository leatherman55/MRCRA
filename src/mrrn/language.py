"""Language-model adaptation for causal MRRN actors.

The adapter deliberately keeps tokenization outside the neural module.  This
makes checkpoints explicit about tokenizer identity while keeping the model
portable and lets the same MRRN consume any validated integer vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import MRRNConfig
from .model import MRRN, MRRNOutput, MRRNStreamState
from .cognitive_model import (
    MRCRAIntegratedTrainingOutput, MRCRAOutput, MRCRARuntimeState,
    MultimodalRelationalContinuityResonanceNetwork,
)
from .cstm import (
    CSTMArchitectureConfig, CSTMPredictionBatch,
    CausalSpectralTargetPredictor,
)
from .cstm_schedule import deterministic_cstm_rows
from .cognitive_types import BoundaryClass, ModalityClass, SourceClass
from .controller import GoalState, SystemModelState
from .config import MRCRAConfig
from .observation import register_external_observations, register_internal_inputs
from .provenance import ProvenanceLedger
from .vocabulary_router import (
    CertifiedBalancedVocabularyRouter,
    VocabularyRouterConfig,
    VocabularyRouterIndex,
    VocabularyRoutingMetrics,
)


def _apply_repetition_penalty(
    logits: Tensor, seen_token_mask: Tensor, repetition_penalty: float
) -> Tensor:
    if repetition_penalty == 1:
        return logits
    selected = logits[:, seen_token_mask]
    logits[:, seen_token_mask] = torch.where(
        selected < 0, selected * repetition_penalty, selected / repetition_penalty
    )
    return logits


@dataclass(frozen=True, slots=True)
class _GenerationCandidates:
    """Compact exact sampling authority with inactive padding."""

    token_ids: Tensor
    logits: Tensor
    mask: Tensor
    routing: VocabularyRoutingMetrics | None


def _compact_threshold_candidates(
    logits: Tensor, top_k: int | None
) -> tuple[Tensor, Tensor, Tensor]:
    """Compress dense logits to all values eligible under the top-k threshold."""

    if top_k is None:
        token_ids = torch.arange(logits.shape[-1], device=logits.device)
        token_ids = token_ids.unsqueeze(0).expand(logits.shape[0], -1)
        return token_ids, logits, torch.ones_like(token_ids, dtype=torch.bool)
    threshold = logits.topk(min(top_k, logits.shape[-1]), -1).values[:, -1:]
    eligible = logits >= threshold
    counts = eligible.sum(-1)
    width = int(counts.max())
    token_ids = torch.full(
        (logits.shape[0], width), -1, dtype=torch.int64, device=logits.device
    )
    values = logits.new_full((logits.shape[0], width), -torch.inf)
    mask = torch.zeros_like(token_ids, dtype=torch.bool)
    vocabulary_ids = torch.arange(logits.shape[-1], device=logits.device)
    for row in range(logits.shape[0]):
        row_ids = vocabulary_ids[eligible[row]]
        count = row_ids.numel()
        token_ids[row, :count] = row_ids
        values[row, :count] = logits[row, row_ids]
        mask[row, :count] = True
    return token_ids, values, mask


def _generation_candidates(
    latent: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    vocabulary_size: int,
    router: CertifiedBalancedVocabularyRouter | None,
    seen_token_mask: Tensor,
    forbidden_token_mask: Tensor,
    repetition_penalty: float,
    top_k: int | None,
) -> _GenerationCandidates:
    """Return the compact exact sampling authority without dense re-expansion."""

    if (
        latent.ndim != 2
        or seen_token_mask.shape != (vocabulary_size,)
        or forbidden_token_mask.shape != (vocabulary_size,)
    ):
        raise ValueError("generation latent and seen-token mask are incompatible")
    forbidden_count = int(forbidden_token_mask.sum())
    allowed_count = vocabulary_size - forbidden_count
    if allowed_count <= 0:
        raise ValueError("generation forbids the complete vocabulary")
    top_k = None if top_k is None else min(top_k, allowed_count)
    if router is not None and top_k is not None:
        routed = router.exact_top_k(
            latent,
            min(vocabulary_size, top_k + forbidden_count),
            seen_token_mask=seen_token_mask,
            repetition_penalty=repetition_penalty,
            return_on_input_device=False,
        )
        # Canonical token order makes the random stream independent of the
        # router's cluster traversal while preserving every threshold tie.
        allowed = routed.mask & ~forbidden_token_mask.to(
            routed.token_ids.device
        )[routed.token_ids.clamp_min(0)]
        allowed_values = routed.logits.masked_fill(~allowed, -torch.inf)
        threshold = allowed_values.topk(top_k, -1).values[:, -1:]
        allowed &= routed.logits >= threshold
        ordering = routed.token_ids.masked_fill(~allowed, vocabulary_size).argsort(-1)
        return _GenerationCandidates(
            routed.token_ids.gather(-1, ordering),
            routed.logits.gather(-1, ordering),
            allowed.gather(-1, ordering),
            routed.metrics,
        )
    logits = F.linear(latent.float(), weight.float(), bias.float())
    logits.masked_fill_(
        forbidden_token_mask.to(logits.device).unsqueeze(0), -torch.inf,
    )
    logits = _apply_repetition_penalty(
        logits, seen_token_mask, repetition_penalty
    )
    token_ids, values, mask = _compact_threshold_candidates(logits, top_k)
    return _GenerationCandidates(token_ids, values, mask, None)


def _sample_generation_candidates(
    candidates: _GenerationCandidates,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None,
) -> Tensor:
    """Sample directly from exact candidates using one inverse-CDF draw."""

    adjusted = candidates.logits.masked_fill(~candidates.mask, -torch.inf)
    if temperature == 0:
        position = adjusted.argmax(-1, keepdim=True)
        return candidates.token_ids.gather(-1, position)
    adjusted = adjusted / temperature
    token_ids = candidates.token_ids
    if top_p < 1:
        ordering = torch.argsort(
            adjusted, dim=-1, descending=True, stable=True
        )
        adjusted = adjusted.gather(-1, ordering)
        token_ids = token_ids.gather(-1, ordering)
        probability = torch.softmax(adjusted, -1)
        remove = probability.cumsum(-1) - probability > top_p
        adjusted = adjusted.masked_fill(remove, -torch.inf)
    probability = torch.softmax(adjusted, -1)
    random_device = (
        probability.device if generator is None else torch.device(generator.device)
    )
    uniform = torch.rand(
        (probability.shape[0], 1),
        device=random_device,
        generator=generator,
        dtype=torch.float32,
    ).to(probability.device)
    position = (probability.cumsum(-1) < uniform).sum(-1, keepdim=True)
    position.clamp_(max=probability.shape[-1] - 1)
    return token_ids.gather(-1, position)


@dataclass(frozen=True, slots=True)
class LanguageModelOutput:
    logits: Tensor
    mrrn: MRRNOutput


def fineweb_27m_config(vocabulary_size: int = 50_257) -> MRRNConfig:
    """Legacy sequence-only configuration for 26.4M checkpoints."""

    if vocabulary_size < 2:
        raise ValueError("language modeling requires a vocabulary of at least two tokens")
    return MRRNConfig(
        input_dim=120,
        model_dim=120,
        output_dim=vocabulary_size,
        layers=5,
        scales=5,
        heads=4,
        modes=16,
        mimo_rank=2,
        attention_window=16,
        retrieved_items=8,
        memory_capacity=2048,
        mixer_expansion=2.5,
        width_growth_cap=1.5,
        mode_growth_cap=1.5,
        width_multiple=8,
        spectral_modes=8,
        spectral_basis_order=4,
        spectral_triads_per_mode=1,
        enable_global_head=False,
    )


def fineweb_4p7m_config(vocabulary_size: int = 50_257) -> MRRNConfig:
    """Balanced five-scale configuration with 4,695,023 GPT-2-vocabulary parameters."""

    if vocabulary_size < 2:
        raise ValueError("language modeling requires a vocabulary of at least two tokens")
    return MRRNConfig(
        input_dim=48,
        model_dim=48,
        output_dim=vocabulary_size,
        layers=3,
        scales=5,
        heads=4,
        modes=10,
        mimo_rank=2,
        attention_window=16,
        retrieved_items=8,
        memory_capacity=2048,
        mixer_expansion=2.0,
        width_growth_cap=1.25,
        mode_growth_cap=1.25,
        width_multiple=8,
        spectral_modes=8,
        spectral_basis_order=4,
        spectral_triads_per_mode=1,
        enable_global_head=False,
    )


def tiny_language_config(vocabulary_size: int = 257) -> MRRNConfig:
    """Small but structurally complete configuration for smoke tests."""

    return MRRNConfig(
        input_dim=16,
        model_dim=16,
        output_dim=vocabulary_size,
        layers=2,
        scales=3,
        heads=2,
        modes=4,
        mimo_rank=1,
        attention_window=4,
        retrieved_items=2,
        memory_capacity=16,
        mixer_expansion=1.5,
        width_growth_cap=1.5,
        mode_growth_cap=1.5,
        width_multiple=4,
        spectral_modes=3,
        spectral_basis_order=3,
        spectral_triads_per_mode=1,
        enable_global_head=False,
    )


class MRRNLanguageModel(nn.Module):
    """Tied-embedding autoregressive language model backed by a causal MRRN."""

    def __init__(
        self,
        config: MRRNConfig,
        *,
        vocabulary_size: int | None = None,
        vocabulary_router_config: VocabularyRouterConfig = VocabularyRouterConfig(),
    ) -> None:
        super().__init__()
        vocabulary_size = config.resolved_output_dim if vocabulary_size is None else vocabulary_size
        if not config.causal:
            raise ValueError("next-token prediction requires a causal MRRN")
        if config.enable_global_head:
            raise ValueError("language models must disable the unused global output head")
        if config.input_dim != config.model_dim:
            raise ValueError("tied language embeddings require input_dim == model_dim")
        if config.resolved_output_dim != vocabulary_size or vocabulary_size < 2:
            raise ValueError("MRRN output width must exactly equal the tokenizer vocabulary")
        self.config = config
        self.vocabulary_size = vocabulary_size
        self.token_embedding = nn.Embedding(vocabulary_size, config.input_dim)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        self.actor = MRRN(config)
        # This is one Parameter object registered under two paths.  PyTorch
        # optimizers deduplicate it, so model-size reports count it once.
        self.actor.output_head.weight = self.token_embedding.weight
        nn.init.zeros_(self.actor.output_head.bias)
        self.vocabulary_router_config = vocabulary_router_config
        self._vocabulary_router: CertifiedBalancedVocabularyRouter | None = None

    def build_vocabulary_router(
        self, *, force: bool = False
    ) -> CertifiedBalancedVocabularyRouter | None:
        """Build or return the checkpoint-bound inference index lazily."""

        config = self.vocabulary_router_config
        if (
            not config.enabled
            or self.vocabulary_size < config.minimum_vocabulary_size
            or self.config.model_dim < config.minimum_model_dimension
        ):
            return None
        if self._vocabulary_router is None or force:
            self._vocabulary_router = CertifiedBalancedVocabularyRouter(
                self.token_embedding.weight,
                self.actor.output_head.bias,
                config,
            )
        return self._vocabulary_router

    def save_vocabulary_router_index(self, destination: str) -> None:
        router = self.build_vocabulary_router(force=True)
        if router is None:
            raise RuntimeError("vocabulary routing is disabled for this model")
        router.index.save(destination)

    def load_vocabulary_router_index(self, source: str) -> None:
        index = VocabularyRouterIndex.load(
            source, self.token_embedding.weight, self.actor.output_head.bias
        )
        self.vocabulary_router_config = index.config
        self._vocabulary_router = CertifiedBalancedVocabularyRouter(
            self.token_embedding.weight,
            self.actor.output_head.bias,
            index.config,
            index=index,
        )

    def vocabulary_routing_metrics(self) -> dict[str, float]:
        return (
            {}
            if self._vocabulary_router is None
            else self._vocabulary_router.cumulative_metrics()
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> LanguageModelOutput:
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be an integer tensor shaped (batch,time)")
        valid = input_ids.numel() == 0 or (
            int(input_ids.min()) >= 0 and int(input_ids.max()) < self.vocabulary_size
        )
        if not valid:
            raise ValueError("input_ids contain tokens outside the vocabulary")
        if attention_mask is not None and (
            attention_mask.shape != input_ids.shape or attention_mask.dtype != torch.bool
        ):
            raise ValueError("attention_mask must be boolean and match input_ids")
        embedded = self.token_embedding(input_ids)
        output = self.actor(embedded, attention_mask, output_mode="sequence")
        return LanguageModelOutput(output.prediction, output)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def initial_stream_state(self, batch: int, *, device=None, dtype=None) -> MRRNStreamState:
        return self.actor.initial_stream_state(batch, device=device, dtype=dtype)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        maximum_new_tokens: int,
        eos_token_id: int | None = None,
        forbidden_token_ids: Sequence[int] = (),
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate recurrently with bounded neural/cache state."""

        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("generation requires a nonempty (1,time) prompt")
        if maximum_new_tokens < 0 or not isfinite(temperature) or temperature < 0:
            raise ValueError("generation length and temperature are invalid")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when supplied")
        if not 0 < top_p <= 1 or repetition_penalty < 1:
            raise ValueError("top_p and repetition_penalty are invalid")
        if eos_token_id is not None and not 0 <= eos_token_id < self.vocabulary_size:
            raise ValueError("eos_token_id lies outside the vocabulary")
        if any(
            not isinstance(value, int) or not 0 <= value < self.vocabulary_size
            for value in forbidden_token_ids
        ):
            raise ValueError("forbidden generation token lies outside the vocabulary")
        device = next(self.parameters()).device
        tokens = input_ids.to(device=device, dtype=torch.long)
        router = self.build_vocabulary_router()
        seen_token_mask = torch.zeros(
            self.vocabulary_size,
            dtype=torch.bool,
            device=device if router is None else router.execution_device,
        )
        seen_token_mask[tokens.reshape(-1).to(seen_token_mask.device)] = True
        forbidden_token_mask = torch.zeros_like(seen_token_mask)
        if forbidden_token_ids:
            forbidden_token_mask[
                torch.tensor(
                    tuple(dict.fromkeys(forbidden_token_ids)),
                    dtype=torch.int64,
                    device=forbidden_token_mask.device,
                )
            ] = True
        state = self.initial_stream_state(1, device=device, dtype=self.token_embedding.weight.dtype)
        latent = None
        for position in range(tokens.shape[1]):
            step = self.actor.step(
                self.token_embedding(tokens[:, position]),
                state,
                project_output=False,
            )
            state, latent = step.state, step.latent
        if latent is None:
            raise RuntimeError("language prefill did not produce an output latent")
        for _ in range(maximum_new_tokens):
            retrieval_top_k = 1 if temperature == 0 else top_k
            candidates = _generation_candidates(
                latent,
                self.actor.output_head.weight,
                self.actor.output_head.bias,
                vocabulary_size=self.vocabulary_size,
                router=router,
                seen_token_mask=seen_token_mask,
                forbidden_token_mask=forbidden_token_mask,
                repetition_penalty=repetition_penalty,
                top_k=retrieval_top_k,
            )
            next_token = _sample_generation_candidates(
                candidates,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            seen_token_mask[
                next_token.reshape(-1).to(seen_token_mask.device)
            ] = True
            next_token = next_token.to(device)
            tokens = torch.cat((tokens, next_token), 1)
            if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                break
            step = self.actor.step(
                self.token_embedding(next_token[:, 0]),
                state,
                project_output=False,
            )
            state, latent = step.state, step.latent
            if latent is None:
                raise RuntimeError("language decode step did not expose its output latent")
        return tokens


@dataclass(frozen=True, slots=True)
class MRCRALanguageOutput:
    logits: Tensor
    cognitive: MRCRAOutput
    ledger: ProvenanceLedger


@dataclass(frozen=True, slots=True)
class MRCRAGenerationOutput:
    tokens: Tensor
    state: MRCRARuntimeState
    ledger: ProvenanceLedger
    generated_provenance_ids: tuple[int, ...]
    routing_receipts: tuple[VocabularyRoutingMetrics, ...]


class MRCRALanguageModel(nn.Module):
    """Tied-embedding language interface with source-safe cognitive streaming."""

    def __init__(
        self, config: MRCRAConfig, *, vocabulary_size: int | None = None,
        model_authority: str = "mrcra-language-untrained",
        cstm_config: CSTMArchitectureConfig = CSTMArchitectureConfig(),
        vocabulary_router_config: VocabularyRouterConfig = VocabularyRouterConfig(),
    ) -> None:
        super().__init__()
        carrier = config.carrier
        vocabulary_size = carrier.resolved_output_dim if vocabulary_size is None else vocabulary_size
        if carrier.input_dim != carrier.model_dim:
            raise ValueError("tied MRCRA language embeddings require input_dim == model_dim")
        if carrier.resolved_output_dim != vocabulary_size or vocabulary_size < 2:
            raise ValueError("carrier output width must exactly equal the tokenizer vocabulary")
        if carrier.enable_global_head:
            raise ValueError("MRCRA language uses the shared sequence head, not a dense global vocabulary head")
        self.config = config
        self.vocabulary_size = vocabulary_size
        self.model_authority = model_authority
        self.token_embedding = nn.Embedding(vocabulary_size, carrier.input_dim)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        self.cognitive = MultimodalRelationalContinuityResonanceNetwork(
            config, model_authority=model_authority,
        )
        self.cognitive.carrier.output_head.weight = self.token_embedding.weight
        nn.init.zeros_(self.cognitive.carrier.output_head.bias)
        self.cstm_predictor = CausalSpectralTargetPredictor(
            carrier.model_dim,
            carrier.scales,
            vocabulary_size,
            cstm_config,
        )
        self.vocabulary_router_config = vocabulary_router_config
        self._vocabulary_router: CertifiedBalancedVocabularyRouter | None = None

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def build_vocabulary_router(
        self, *, force: bool = False
    ) -> CertifiedBalancedVocabularyRouter | None:
        """Build the inference-only vocabulary index without registering parameters."""

        config = self.vocabulary_router_config
        if (
            not config.enabled
            or self.vocabulary_size < config.minimum_vocabulary_size
            or self.config.carrier.model_dim < config.minimum_model_dimension
        ):
            return None
        if self._vocabulary_router is None or force:
            self._vocabulary_router = CertifiedBalancedVocabularyRouter(
                self.token_embedding.weight,
                self.cognitive.carrier.output_head.bias,
                config,
            )
        return self._vocabulary_router

    def save_vocabulary_router_index(self, destination: str) -> None:
        router = self.build_vocabulary_router(force=True)
        if router is None:
            raise RuntimeError("vocabulary routing is disabled for this model")
        router.index.save(destination)

    def load_vocabulary_router_index(self, source: str) -> None:
        index = VocabularyRouterIndex.load(
            source,
            self.token_embedding.weight,
            self.cognitive.carrier.output_head.bias,
        )
        self.vocabulary_router_config = index.config
        self._vocabulary_router = CertifiedBalancedVocabularyRouter(
            self.token_embedding.weight,
            self.cognitive.carrier.output_head.bias,
            index.config,
            index=index,
        )

    def vocabulary_routing_metrics(self) -> dict[str, float]:
        return (
            {}
            if self._vocabulary_router is None
            else self._vocabulary_router.cumulative_metrics()
        )

    def _validate_tokens(self, input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be an integer tensor shaped (batch,time)")
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.vocabulary_size
        ):
            raise ValueError("input_ids contain tokens outside the vocabulary")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask.shape != input_ids.shape or attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must be boolean and match input_ids")
        if input_ids.shape[1] > 1 and bool((attention_mask[:, 1:] & ~attention_mask[:, :-1]).any()):
            raise ValueError("language padding mask cannot reactivate")
        return attention_mask

    def prepare_external_input(
        self, input_ids: Tensor, *, attention_mask: Tensor | None = None,
        segment_ids: Tensor | None = None, boundary_classes: Tensor | None = None,
        source_uris: Sequence[str | Mapping[int, str]] | None = None,
        ledger: ProvenanceLedger | None = None, continuing: bool = False,
        timestamp_offset: int | Tensor = 0,
    ):
        mask = self._validate_tokens(input_ids, attention_mask)
        batch, length = input_ids.shape
        device = input_ids.device
        offset = torch.as_tensor(
            timestamp_offset, dtype=self.token_embedding.weight.dtype, device=device
        )
        if offset.ndim == 0:
            offset = offset.expand(batch)
        if offset.shape != (batch,) or bool((offset < 0).any()):
            raise ValueError(
                "timestamp_offset must be nonnegative and scalar or shaped (batch,)"
            )
        segment_ids = (
            torch.zeros_like(input_ids, dtype=torch.int64)
            if segment_ids is None else segment_ids
        )
        if segment_ids.shape != input_ids.shape or segment_ids.dtype != torch.int64:
            raise ValueError("segment_ids must be int64 and match input_ids")
        if boundary_classes is None:
            boundary_classes = torch.zeros_like(input_ids, dtype=torch.int64)
            if length and not continuing:
                boundary_classes[:, 0] = int(BoundaryClass.HARD)
            if length > 1:
                changed = mask[:, 1:] & (segment_ids[:, 1:] != segment_ids[:, :-1])
                boundary_classes[:, 1:] = torch.where(
                    changed,
                    torch.full_like(boundary_classes[:, 1:], int(BoundaryClass.SEGMENT)),
                    boundary_classes[:, 1:],
                )
        if boundary_classes.shape != input_ids.shape or boundary_classes.dtype != torch.int64:
            raise ValueError("boundary_classes must be int64 and match input_ids")
        source_uris = source_uris or tuple(f"language://batch/{index}" for index in range(batch))
        ledger = ProvenanceLedger() if ledger is None else ledger
        values = self.token_embedding(input_ids) * mask.unsqueeze(-1)
        timestamps = (
            torch.arange(length, device=device, dtype=values.dtype)[None]
            + offset.to(values.dtype)[:, None]
        )
        packet = register_external_observations(
            values, mask, observed_mask=mask, timestamps=timestamps,
            coordinates=timestamps.unsqueeze(-1),
            sample_intervals=torch.ones(batch, device=device, dtype=values.dtype),
            boundary_classes=boundary_classes,
            modality_ids=torch.full(
                (batch, length), int(ModalityClass.TEXT), dtype=torch.int64, device=device
            ),
            uncertainty_seed=torch.zeros(
                batch, length, self.config.cognitive.uncertainty_channels,
                device=device, dtype=values.dtype,
            ),
            segment_ids=segment_ids.masked_fill(~mask, -1),
            source_uris=source_uris, ledger=ledger,
            model_authority=f"tokenizer+{self.model_authority}",
        )
        return packet, ledger

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None, *,
        segment_ids: Tensor | None = None, boundary_classes: Tensor | None = None,
        source_uris: Sequence[str | Mapping[int, str]] | None = None,
        ledger: ProvenanceLedger | None = None,
        state: MRCRARuntimeState | None = None,
        goals: GoalState | None = None,
        system_model: SystemModelState | None = None,
        project_output: bool = True,
    ) -> MRCRALanguageOutput:
        packet, ledger = self.prepare_external_input(
            input_ids, attention_mask=attention_mask, segment_ids=segment_ids,
            boundary_classes=boundary_classes, source_uris=source_uris,
            ledger=ledger, continuing=state is not None,
            timestamp_offset=0 if state is None else state.clocks.external,
        )
        cognitive = self.cognitive(
            packet, ledger, state=state, goals=goals,
            system_model=system_model, project_output=project_output
        )
        return MRCRALanguageOutput(cognitive.prediction, cognitive, ledger)

    def project_output(self, output_latent: Tensor) -> Tensor:
        """Apply the tied vocabulary head to an exposed output latent.

        Long-context training uses the same weights through a tiled exact
        projection, while ordinary inference may call this dense convenience
        method.  The bias is owned by the carrier head and is not tied.
        """

        if output_latent.ndim not in (2, 3) or output_latent.shape[-1] != self.config.carrier.model_dim:
            raise ValueError("output latents must end in the carrier model dimension")
        return self.cognitive.carrier.output_head(output_latent)

    def predict_causal_spectral_targets(
        self,
        output: MRCRAIntegratedTrainingOutput,
        *,
        extra_horizon_offset: int,
        selected_scales: Sequence[int] | None = None,
        detach_substrate: bool = False,
        target_participation_budget: int | None = None,
        row_sampling_digest: str | None = None,
        row_sampling_stream: int = 0,
    ) -> tuple[CSTMPredictionBatch, ...]:
        """Predict future block spectra from every coefficient emitted in a span.

        The primary next-block horizon is always active.  Each scale also uses
        one deterministic extra horizon selected by the optimizer step and
        scale index.  Source positions are document-relative end timestamps;
        the trainer maps them into the packed context before constructing fixed
        targets.
        """

        configured = self.cstm_predictor.config.horizon_blocks
        extras = configured[1:]
        if extra_horizon_offset < 0:
            raise ValueError("CSTM horizon offset cannot be negative")
        if (
            target_participation_budget is not None
            and (
                target_participation_budget <= 0
                or row_sampling_digest is None
                or len(row_sampling_digest) != 64
                or row_sampling_stream < 0
            )
        ):
            raise ValueError("CSTM bounded row-sampling request is invalid")
        selected = (
            None if selected_scales is None
            else frozenset(int(value) for value in selected_scales)
        )
        if selected is not None and (
            len(selected) != len(tuple(selected_scales))
            or any(value < 0 or value >= len(output.band_histories) for value in selected)
        ):
            raise ValueError("CSTM selected scales are invalid")
        span_start = output.state.carrier.position - output.output_latent.shape[1]
        if span_start < 0:
            raise RuntimeError("CSTM carrier position precedes the integrated span")
        predictions: list[CSTMPredictionBatch] = []
        for scale, history in enumerate(output.band_histories):
            if history is None or (selected is not None and scale not in selected):
                continue
            horizons = (
                (1,)
                if not extras
                else (1, extras[(extra_horizon_offset + scale) % len(extras)])
            )
            local_indices = history.end_positions - span_start
            if (
                local_indices.numel()
                and (
                    int(local_indices.min()) < 0
                    or int(local_indices.max()) >= output.cognitive_residual.shape[1]
                )
            ):
                raise RuntimeError(
                    "CSTM band emissions do not lie inside their integrated span"
                )
            carrier_features = self.cognitive.carrier.synthesis_adapters[scale](
                history.band.data
            )
            if detach_substrate:
                # Predictor-only updates must not traverse either the emitted
                # coefficient or its trainable synthesis adapter.
                carrier_features = carrier_features.detach()
            cognitive = output.cognitive_residual[:, local_indices]
            if detach_substrate:
                cognitive = cognitive.detach()
            row_probability = 1.0
            row_digest = None
            source_positions = history.end_positions
            if (
                target_participation_budget is not None
                and source_positions.numel()
            ):
                row_budget = max(
                    1,
                    target_participation_budget
                    // max(
                        1,
                        (
                            history.band.support
                            * carrier_features.shape[0]
                            * len(horizons)
                        ),
                    ),
                )
                row_decision = deterministic_cstm_rows(
                    int(source_positions.numel()),
                    row_budget,
                    counter_digest=row_sampling_digest,
                    stream=row_sampling_stream + scale,
                )
                indices = torch.tensor(
                    row_decision.selected_indices,
                    dtype=torch.int64,
                    device=source_positions.device,
                )
                carrier_features = carrier_features.index_select(1, indices)
                cognitive = cognitive.index_select(1, indices)
                source_positions = source_positions.index_select(0, indices)
                row_probability = row_decision.inclusion_probability
                row_digest = row_decision.counter_digest
            values = self.cstm_predictor(
                carrier_features,
                cognitive,
                scale=scale,
                horizons=horizons,
            )
            predictions.append(CSTMPredictionBatch(
                values,
                source_positions,
                horizons,
                history.band.support,
                scale,
                history.band.kind,
                row_probability,
                row_digest,
            ))
        return tuple(predictions)

    @staticmethod
    def _sample(
        logits: Tensor, *, temperature: float, top_k: int | None, top_p: float,
        generator: torch.Generator | None,
    ) -> Tensor:
        if temperature == 0:
            return logits.argmax(-1, keepdim=True)
        adjusted = logits / temperature
        if top_k is not None:
            count = min(top_k, adjusted.shape[-1])
            threshold = adjusted.topk(count, -1).values[:, -1:]
            adjusted = adjusted.masked_fill(adjusted < threshold, -torch.inf)
        if top_p < 1:
            sorted_logits, sorted_indices = adjusted.sort(-1, descending=True)
            probability = torch.softmax(sorted_logits, -1)
            remove = probability.cumsum(-1) - probability > top_p
            sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
            adjusted = torch.full_like(adjusted, -torch.inf).scatter(
                -1, sorted_indices, sorted_logits
            )
        return torch.multinomial(torch.softmax(adjusted, -1), 1, generator=generator)

    @torch.no_grad()
    def generate(
        self, input_ids: Tensor, *, maximum_new_tokens: int,
        eos_token_id: int | None = None,
        forbidden_token_ids: Sequence[int] = (),
        temperature: float = 0.8,
        top_k: int | None = 50, top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        source_uri: str = "language://prompt",
        generator: torch.Generator | None = None,
    ) -> MRCRAGenerationOutput:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("generation requires a nonempty (1,time) prompt")
        if maximum_new_tokens < 0 or not isfinite(temperature) or temperature < 0:
            raise ValueError("generation length and temperature are invalid")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when supplied")
        if not 0 < top_p <= 1 or repetition_penalty < 1:
            raise ValueError("top_p and repetition_penalty are invalid")
        if eos_token_id is not None and not 0 <= eos_token_id < self.vocabulary_size:
            raise ValueError("eos_token_id lies outside the vocabulary")
        if any(
            not isinstance(value, int) or not 0 <= value < self.vocabulary_size
            for value in forbidden_token_ids
        ):
            raise ValueError("forbidden generation token lies outside the vocabulary")
        device = next(self.parameters()).device
        tokens = input_ids.to(device=device, dtype=torch.long)
        router = self.build_vocabulary_router()
        seen_token_mask = torch.zeros(
            self.vocabulary_size,
            dtype=torch.bool,
            device=device if router is None else router.execution_device,
        )
        seen_token_mask[tokens.reshape(-1).to(seen_token_mask.device)] = True
        forbidden_token_mask = torch.zeros_like(seen_token_mask)
        if forbidden_token_ids:
            forbidden_token_mask[
                torch.tensor(
                    tuple(dict.fromkeys(forbidden_token_ids)),
                    dtype=torch.int64,
                    device=forbidden_token_mask.device,
                )
            ] = True
        prompt = self(
            tokens,
            source_uris=(source_uri,),
            project_output=False,
        )
        state, ledger = prompt.cognitive.state, prompt.ledger
        latent = prompt.cognitive.output_latent[:, -1]
        parent_record_id = int(
            prompt.cognitive.nodes.provenance_ids[prompt.cognitive.nodes.active][-1]
            if bool(prompt.cognitive.nodes.active.any())
            else max(record.record_id for record in ledger.records())
        )
        generated_ids: list[int] = []
        routing_receipts: list[VocabularyRoutingMetrics] = []
        for step in range(maximum_new_tokens):
            retrieval_top_k = 1 if temperature == 0 else top_k
            candidates = _generation_candidates(
                latent,
                self.cognitive.carrier.output_head.weight,
                self.cognitive.carrier.output_head.bias,
                vocabulary_size=self.vocabulary_size,
                router=router,
                seen_token_mask=seen_token_mask,
                forbidden_token_mask=forbidden_token_mask,
                repetition_penalty=repetition_penalty,
                top_k=retrieval_top_k,
            )
            if candidates.routing is not None:
                routing_receipts.append(candidates.routing)
            next_token = _sample_generation_candidates(
                candidates,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            seen_token_mask[
                next_token.reshape(-1).to(seen_token_mask.device)
            ] = True
            next_token = next_token.to(device)
            tokens = torch.cat((tokens, next_token), 1)
            value = self.token_embedding(next_token)
            timestamp = value.new_tensor([[float(state.clocks.external)]])
            internal = register_internal_inputs(
                value, torch.ones(1, 1, dtype=torch.bool, device=device),
                parent_record_ids=torch.tensor(
                    [[[parent_record_id]]], dtype=torch.int64, device=device
                ),
                timestamps=timestamp, coordinates=timestamp.unsqueeze(-1),
                sample_intervals=value.new_ones(1),
                boundary_classes=torch.zeros(1, 1, dtype=torch.int64, device=device),
                modality_ids=torch.full(
                    (1, 1), int(ModalityClass.TEXT), dtype=torch.int64, device=device
                ),
                uncertainty_seed=value.new_zeros(
                    1, 1, self.config.cognitive.uncertainty_channels
                ),
                segment_ids=torch.zeros(1, 1, dtype=torch.int64, device=device),
                ledger=ledger, source_class=SourceClass.PREDICTED,
                operator="mrcra:language_sample", scenario_ids=torch.zeros(
                    1, 1, dtype=torch.int64, device=device
                ), model_authority=self.model_authority,
            )
            parent_record_id = int(internal.source_record_ids.item())
            generated_ids.append(parent_record_id)
            result = self.cognitive(
                internal,
                ledger,
                state=state,
                project_output=False,
            )
            state, latent = result.state, result.output_latent[:, -1]
            if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                break
        return MRCRAGenerationOutput(
            tokens,
            state,
            ledger,
            tuple(generated_ids),
            tuple(routing_receipts),
        )
