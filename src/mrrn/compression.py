"""Dynamic abstraction proposals with explicit code and distortion authority."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class GraphFragment:
    node_content: Tensor
    node_type_ids: Tensor
    node_support: Tensor
    node_provenance_ids: Tensor
    node_mask: Tensor
    relation_content: Tensor
    relation_family_ids: Tensor
    participant_indices: Tensor
    relation_provenance_ids: Tensor
    relation_mask: Tensor

    def __post_init__(self) -> None:
        if self.node_content.ndim != 3:
            raise ValueError("graph fragment nodes must be (batch,nodes,width)")
        node_base = self.node_content.shape[:2]
        if self.node_type_ids.shape != node_base or self.node_type_ids.dtype != torch.int64:
            raise ValueError("node type IDs must be int64 with node shape")
        if self.node_support.shape != (*node_base, 3):
            raise ValueError("node support must have start, end, completion")
        if self.node_provenance_ids.shape != node_base or self.node_provenance_ids.dtype != torch.int64:
            raise ValueError("node provenance IDs must be int64 with node shape")
        if self.node_mask.shape != node_base or self.node_mask.dtype != torch.bool:
            raise ValueError("node mask must be boolean with node shape")
        if self.relation_content.ndim != 3 or self.relation_content.shape[0] != node_base[0]:
            raise ValueError("relation content must be (batch,relations,width)")
        relation_base = self.relation_content.shape[:2]
        if self.relation_family_ids.shape != relation_base or self.relation_family_ids.dtype != torch.int64:
            raise ValueError("relation family IDs must be int64 with relation shape")
        if self.participant_indices.ndim != 3 or self.participant_indices.shape[:2] != relation_base:
            raise ValueError("participant pointers must be (batch,relations,arity)")
        if self.relation_provenance_ids.shape != relation_base or self.relation_provenance_ids.dtype != torch.int64:
            raise ValueError("relation provenance IDs must be int64 with relation shape")
        if self.relation_mask.shape != relation_base or self.relation_mask.dtype != torch.bool:
            raise ValueError("relation mask must be boolean with relation shape")
        if bool((self.node_mask & (self.node_provenance_ids < 0)).any()):
            raise ValueError("active fragment nodes require provenance")
        if bool((self.relation_mask & (self.relation_provenance_ids < 0)).any()):
            raise ValueError("active fragment relations require provenance")
        participant_mask = self.participant_indices >= 0
        active_participants = self.relation_mask.unsqueeze(-1) & participant_mask
        if bool((active_participants & (self.participant_indices >= node_base[1])).any()):
            raise ValueError("active relation participant pointers lie outside the fragment")
        if bool((self.relation_mask & (participant_mask.sum(-1) < 2)).any()):
            raise ValueError("active fragment relations require at least two participants")
        if bool((~self.relation_mask.unsqueeze(-1) & participant_mask).any()):
            raise ValueError("inactive fragment relations cannot retain participant pointers")
        safe = self.participant_indices.clamp_min(0)
        batch = torch.arange(node_base[0], device=safe.device)[:, None, None]
        if bool((active_participants & ~self.node_mask[batch, safe]).any()):
            raise ValueError("active relation participants must point to active fragment nodes")


@dataclass(frozen=True, slots=True)
class DescriptionLengthReport:
    baseline_bits: Tensor
    type_bits: Tensor
    participant_bits: Tensor
    parameter_bits: Tensor
    residual_bits: Tensor
    compressed_bits: Tensor
    gain_bits: Tensor


@dataclass(frozen=True, slots=True)
class DistortionReport:
    node: Tensor
    relation: Tensor
    temporal: Tensor
    causal: Tensor
    total: Tensor


@dataclass(frozen=True, slots=True)
class CompressionProposal:
    latent: Tensor
    reconstructed_nodes: Tensor
    reconstructed_relations: Tensor
    reconstructed_support: Tensor
    node_type_logits: Tensor
    relation_type_logits: Tensor
    code: DescriptionLengthReport
    distortion: DistortionReport


@dataclass(frozen=True, slots=True)
class CompressionDecision:
    accepted: Tensor
    code_gain_pass: Tensor
    distortion_pass: Tensor
    held_out_pass: Tensor
    proposal: CompressionProposal


class GraphCompressor(nn.Module):
    """Bounded graph autoencoder whose output is advisory until hard checks pass."""

    def __init__(
        self, width: int, latent_dim: int, node_types: int, relation_families: int,
        maximum_nodes: int, maximum_relations: int, *, precision_bits: int = 16,
        relation_weight: float = 1.0, temporal_weight: float = 0.25,
        causal_weight: float = 2.0,
    ) -> None:
        super().__init__()
        if min(
            width, latent_dim, node_types, relation_families, maximum_nodes,
            maximum_relations, precision_bits,
        ) <= 0:
            raise ValueError("graph compressor dimensions must be positive")
        if min(relation_weight, temporal_weight, causal_weight) < 0:
            raise ValueError("distortion weights cannot be negative")
        self.width = width
        self.latent_dim = latent_dim
        self.node_types = node_types
        self.relation_families = relation_families
        self.maximum_nodes = maximum_nodes
        self.maximum_relations = maximum_relations
        self.precision_bits = precision_bits
        self.relation_weight = relation_weight
        self.temporal_weight = temporal_weight
        self.causal_weight = causal_weight
        self.node_type_embedding = nn.Embedding(node_types, width)
        self.relation_type_embedding = nn.Embedding(relation_families, width)
        self.node_encoder = nn.Linear(2 * width + 3, width)
        self.relation_encoder = nn.Linear(2 * width, width)
        self.encoder = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, latent_dim))
        self.node_positions = nn.Parameter(torch.empty(maximum_nodes, width))
        self.relation_positions = nn.Parameter(torch.empty(maximum_relations, width))
        self.node_decoder = nn.Sequential(nn.Linear(latent_dim + width, width), nn.SiLU(), nn.Linear(width, width))
        self.relation_decoder = nn.Sequential(nn.Linear(latent_dim + width, width), nn.SiLU(), nn.Linear(width, width))
        self.node_type_head = nn.Linear(width, node_types)
        self.relation_type_head = nn.Linear(width, relation_families)
        self.node_support_head = nn.Linear(width, 3)
        self.residual_log_scale = nn.Parameter(torch.tensor(-1.0))
        nn.init.normal_(self.node_positions, std=width ** -0.5)
        nn.init.normal_(self.relation_positions, std=width ** -0.5)

    @staticmethod
    def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
        return (value * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)

    def decode_latent(
        self, latent: Tensor, *, node_count: int | None = None,
        relation_count: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Decode an accepted abstraction without requiring its source fragment.

        The returned tensors are node content, relation content, temporal support,
        node-type logits, and relation-family logits.  Keeping this operation on
        the compressor's learned decoder makes ``DECOMPRESS`` the actual inverse
        proposal path rather than an unrelated projection.
        """

        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError("compression latent must be (batch,latent_dim)")
        nodes = self.maximum_nodes if node_count is None else node_count
        relations = self.maximum_relations if relation_count is None else relation_count
        if not 0 < nodes <= self.maximum_nodes or not 0 < relations <= self.maximum_relations:
            raise ValueError("requested decode size lies outside compressor bounds")
        batch = latent.shape[0]
        node_inputs = torch.cat((
            latent[:, None].expand(-1, nodes, -1),
            self.node_positions[:nodes][None].expand(batch, -1, -1),
        ), -1)
        relation_inputs = torch.cat((
            latent[:, None].expand(-1, relations, -1),
            self.relation_positions[:relations][None].expand(batch, -1, -1),
        ), -1)
        reconstructed_nodes = self.node_decoder(node_inputs)
        reconstructed_relations = self.relation_decoder(relation_inputs)
        raw_support = self.node_support_head(reconstructed_nodes)
        support_start = raw_support[..., 0]
        support_end = support_start + F.softplus(raw_support[..., 1])
        support = torch.stack((
            support_start, support_end, support_end + F.softplus(raw_support[..., 2]),
        ), -1)
        return (
            reconstructed_nodes, reconstructed_relations, support,
            self.node_type_head(reconstructed_nodes),
            self.relation_type_head(reconstructed_relations),
        )

    def forward(self, fragment: GraphFragment) -> CompressionProposal:
        batch, nodes, width = fragment.node_content.shape
        relations = fragment.relation_content.shape[1]
        if width != self.width or fragment.relation_content.shape[-1] != width:
            raise ValueError("graph fragment width does not match compressor")
        if nodes > self.maximum_nodes or relations > self.maximum_relations:
            raise ValueError("graph fragment exceeds compressor bounds")
        node_encoded = F.silu(self.node_encoder(torch.cat((
            fragment.node_content,
            self.node_type_embedding(fragment.node_type_ids.clamp(0, self.node_types - 1)),
            fragment.node_support,
        ), -1)))
        relation_encoded = F.silu(self.relation_encoder(torch.cat((
            fragment.relation_content,
            self.relation_type_embedding(
                fragment.relation_family_ids.clamp(0, self.relation_families - 1)
            ),
        ), -1)))
        pooled_nodes = self._masked_mean(node_encoded, fragment.node_mask)
        pooled_relations = self._masked_mean(relation_encoded, fragment.relation_mask)
        latent = self.encoder(torch.cat((pooled_nodes, pooled_relations), -1))
        (
            reconstructed_nodes, reconstructed_relations, reconstructed_support,
            node_type_logits, relation_type_logits,
        ) = self.decode_latent(latent, node_count=nodes, relation_count=relations)
        node_error = (
            (reconstructed_nodes - fragment.node_content).square().mean(-1)
            * fragment.node_mask
        ).sum(-1) / fragment.node_mask.sum(-1).clamp_min(1)
        relation_content_error = (
            (reconstructed_relations - fragment.relation_content).square().mean(-1)
            * fragment.relation_mask
        ).sum(-1) / fragment.relation_mask.sum(-1).clamp_min(1)
        node_type_error = F.cross_entropy(
            node_type_logits.transpose(1, 2), fragment.node_type_ids.clamp_min(0), reduction="none"
        )
        node_type_error = (node_type_error * fragment.node_mask).sum(-1) / fragment.node_mask.sum(-1).clamp_min(1)
        relation_type_error = F.cross_entropy(
            relation_type_logits.transpose(1, 2), fragment.relation_family_ids.clamp_min(0), reduction="none"
        )
        relation_type_error = (
            relation_type_error * fragment.relation_mask
        ).sum(-1) / fragment.relation_mask.sum(-1).clamp_min(1)
        relation_error = relation_content_error + relation_type_error
        temporal_error = (
            (reconstructed_support - fragment.node_support).square().mean(-1)
            * fragment.node_mask
        ).sum(-1) / fragment.node_mask.sum(-1).clamp_min(1)
        causal_family = 9
        causal_mask = fragment.relation_mask & (fragment.relation_family_ids == causal_family)
        causal_error = (
            (relation_type_logits.argmax(-1) != causal_family).to(fragment.node_content.dtype)
            * causal_mask
        ).sum(-1) / causal_mask.sum(-1).clamp_min(1)
        total_distortion = (
            node_error + node_type_error + self.relation_weight * relation_error
            + self.temporal_weight * temporal_error + self.causal_weight * causal_error
        )
        distortion = DistortionReport(
            node_error + node_type_error, relation_error, temporal_error, causal_error,
            total_distortion,
        )
        node_count = fragment.node_mask.sum(-1).to(fragment.node_content.dtype)
        relation_count = fragment.relation_mask.sum(-1).to(fragment.node_content.dtype)
        participant_count = (
            (fragment.participant_indices >= 0)
            & fragment.relation_mask.unsqueeze(-1)
        ).sum((-1, -2)).to(fragment.node_content.dtype)
        pointer_bits = max(1.0, log2(max(2, nodes)))
        baseline_bits = (
            node_count * (width * self.precision_bits + log2(self.node_types))
            + relation_count * (
                width * self.precision_bits + log2(self.relation_families)
            )
            + participant_count * pointer_bits
        )
        type_bits = fragment.node_content.new_full((batch,), log2(self.node_types))
        participant_bits = (
            node_count * pointer_bits + participant_count * pointer_bits
        )
        parameter_bits = fragment.node_content.new_full((batch,), self.latent_dim * self.precision_bits)
        residual_variance = torch.exp(2 * self.residual_log_scale).clamp_min(1e-6)
        residual_elements = node_count * width + relation_count * width
        residual_bits = residual_elements * (
            0.5 * torch.log2(2 * torch.pi * residual_variance)
            + total_distortion / (2 * residual_variance * torch.log(fragment.node_content.new_tensor(2.0)))
        ).clamp_min(0)
        compressed = type_bits + participant_bits + parameter_bits + residual_bits
        code = DescriptionLengthReport(
            baseline_bits, type_bits, participant_bits, parameter_bits,
            residual_bits, compressed, baseline_bits - compressed,
        )
        return CompressionProposal(
            latent, reconstructed_nodes, reconstructed_relations, reconstructed_support,
            node_type_logits, relation_type_logits, code, distortion,
        )

    @staticmethod
    def decide(
        proposal: CompressionProposal, *, minimum_gain: float,
        maximum_distortion: float, held_out_loss_before: Tensor,
        held_out_loss_after: Tensor, prediction_tolerance: float = 0.0,
    ) -> CompressionDecision:
        if min(minimum_gain, maximum_distortion, prediction_tolerance) < 0:
            raise ValueError("compression acceptance thresholds cannot be negative")
        batch = proposal.latent.shape[0]
        if held_out_loss_before.shape != (batch,) or held_out_loss_after.shape != (batch,):
            raise ValueError("held-out losses must have one scalar per batch item")
        code_pass = proposal.code.gain_bits > minimum_gain
        distortion_pass = proposal.distortion.total <= maximum_distortion
        held_out_pass = held_out_loss_after <= held_out_loss_before + prediction_tolerance
        return CompressionDecision(
            code_pass & distortion_pass & held_out_pass,
            code_pass, distortion_pass, held_out_pass, proposal,
        )


@dataclass(frozen=True, slots=True)
class AbstractionRecord:
    record_id: int
    child_node_ids: tuple[int, ...]
    child_relation_ids: tuple[int, ...]
    child_abstraction_ids: tuple[int, ...]
    physical_scales: tuple[int, ...]
    abstraction_depth: int
    latent: tuple[float, ...]
    code_gain_bits: float
    distortion: float
    provenance_id: int


class AbstractionDAG:
    """Append-only hierarchy where depth is derived from graph ancestry."""

    def __init__(self) -> None:
        self._records: list[AbstractionRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    def get(self, record_id: int) -> AbstractionRecord:
        if not 0 <= record_id < len(self._records):
            raise KeyError(f"unknown abstraction {record_id}")
        return self._records[record_id]

    def append(
        self, *, child_node_ids: Iterable[int], child_relation_ids: Iterable[int],
        child_abstraction_ids: Iterable[int], physical_scales: Iterable[int],
        latent: Tensor, code_gain_bits: float, distortion: float, provenance_id: int,
    ) -> int:
        child_nodes = tuple(dict.fromkeys(map(int, child_node_ids)))
        child_relations = tuple(dict.fromkeys(map(int, child_relation_ids)))
        child_abstractions = tuple(dict.fromkeys(map(int, child_abstraction_ids)))
        scales = tuple(dict.fromkeys(map(int, physical_scales)))
        if not child_nodes and not child_relations and not child_abstractions:
            raise ValueError("an abstraction requires children")
        if any(value < 0 for value in (*child_nodes, *child_relations, *scales)):
            raise ValueError("abstraction pointers and physical scales cannot be negative")
        if any(value < 0 or value >= len(self._records) for value in child_abstractions):
            raise ValueError("child abstractions must already exist, preserving the DAG")
        if latent.ndim != 1 or latent.numel() == 0 or latent.requires_grad:
            raise ValueError("stored abstraction latent must be a detached nonempty vector")
        if code_gain_bits <= 0 or distortion < 0 or provenance_id < 0:
            raise ValueError("only beneficial, measured abstractions with provenance may be stored")
        depth = 1 + max(
            (self.get(child).abstraction_depth for child in child_abstractions), default=0
        )
        record_id = len(self._records)
        self._records.append(AbstractionRecord(
            record_id, child_nodes, child_relations, child_abstractions, scales,
            depth, tuple(float(value) for value in latent.cpu().tolist()),
            float(code_gain_bits), float(distortion), provenance_id,
        ))
        return record_id
