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

from .config import MRRNConfig
from .model import MRRN, MRRNOutput, MRRNStreamState
from .cognitive_model import (
    MRCRAOutput, MRCRARuntimeState,
    MultimodalRelationalContinuityResonanceNetwork,
)
from .cognitive_types import BoundaryClass, ModalityClass, SourceClass
from .controller import GoalState, SystemModelState
from .config import MRCRAConfig
from .observation import register_external_observations, register_internal_inputs
from .provenance import ProvenanceLedger


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

    def __init__(self, config: MRRNConfig, *, vocabulary_size: int | None = None) -> None:
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
        device = next(self.parameters()).device
        tokens = input_ids.to(device=device, dtype=torch.long)
        state = self.initial_stream_state(1, device=device, dtype=self.token_embedding.weight.dtype)
        logits = None
        for position in range(tokens.shape[1]):
            step = self.actor.step(self.token_embedding(tokens[:, position]), state)
            state, logits = step.state, step.prediction
        for _ in range(maximum_new_tokens):
            adjusted = logits.float().clone()
            if repetition_penalty > 1:
                seen = tokens.unique()
                selected = adjusted[:, seen]
                adjusted[:, seen] = torch.where(
                    selected < 0, selected * repetition_penalty, selected / repetition_penalty
                )
            if temperature == 0:
                next_token = adjusted.argmax(-1, keepdim=True)
            else:
                adjusted /= temperature
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
                probability = torch.softmax(adjusted, -1)
                next_token = torch.multinomial(probability, 1, generator=generator)
            tokens = torch.cat((tokens, next_token), 1)
            if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                break
            step = self.actor.step(self.token_embedding(next_token[:, 0]), state)
            state, logits = step.state, step.prediction
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


class MRCRALanguageModel(nn.Module):
    """Tied-embedding language interface with source-safe cognitive streaming."""

    def __init__(
        self, config: MRCRAConfig, *, vocabulary_size: int | None = None,
        model_authority: str = "mrcra-language-untrained",
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

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

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
        timestamp_offset: int = 0,
    ):
        mask = self._validate_tokens(input_ids, attention_mask)
        batch, length = input_ids.shape
        if timestamp_offset < 0:
            raise ValueError("timestamp_offset cannot be negative")
        device = input_ids.device
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
            torch.arange(length, device=device, dtype=values.dtype) + timestamp_offset
        ).expand(batch, -1)
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
        eos_token_id: int | None = None, temperature: float = 0.8,
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
        device = next(self.parameters()).device
        tokens = input_ids.to(device=device, dtype=torch.long)
        prompt = self(tokens, source_uris=(source_uri,))
        state, ledger = prompt.cognitive.state, prompt.ledger
        logits = prompt.logits[:, -1]
        parent_record_id = int(
            prompt.cognitive.nodes.provenance_ids[prompt.cognitive.nodes.active][-1]
            if bool(prompt.cognitive.nodes.active.any())
            else max(record.record_id for record in ledger.records())
        )
        generated_ids: list[int] = []
        for step in range(maximum_new_tokens):
            adjusted = logits.float().clone()
            if repetition_penalty > 1:
                seen = tokens.unique()
                selected = adjusted[:, seen]
                adjusted[:, seen] = torch.where(
                    selected < 0, selected * repetition_penalty, selected / repetition_penalty
                )
            next_token = self._sample(
                adjusted, temperature=temperature, top_k=top_k, top_p=top_p,
                generator=generator,
            )
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
            result = self.cognitive(internal, ledger, state=state)
            state, logits = result.state, result.prediction[:, -1]
            if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                break
        return MRCRAGenerationOutput(tokens, state, ledger, tuple(generated_ids))
