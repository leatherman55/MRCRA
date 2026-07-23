"""Domain-faithful encoders and exact spatial/temporal multiresolution transforms."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .lifting import LiftingAnalysisBank, LiftingLevel, ReconstructionContext, ScaleTensor
from .mixer import AntiAliasActivation
from .resonance import ComplexResonator


@dataclass(frozen=True, slots=True)
class DomainSpec:
    kind: str
    sample_interval: float = 1.0
    boundary: str = "causal"
    ordered: bool = True
    amplitude_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in {"sequence", "audio", "sensor", "image", "video", "field", "graph", "set"}:
            raise ValueError("unsupported domain kind")
        if self.sample_interval <= 0 or self.amplitude_scale <= 0:
            raise ValueError("sample interval and amplitude scale must be positive")
        if self.boundary not in {"causal", "reflect", "physical", "graph", "none"}:
            raise ValueError("unsupported boundary rule")
        if self.kind == "set" and self.ordered:
            raise ValueError("an unordered set cannot claim an intrinsic order")


@dataclass(frozen=True, slots=True)
class EncodedDomain:
    values: Tensor
    mask: Tensor
    domain: DomainSpec


class TokenEncoder(nn.Module):
    def __init__(self, vocabulary: int, width: int, padding_index: int | None = None) -> None:
        super().__init__()
        if min(vocabulary, width) <= 0:
            raise ValueError("vocabulary and width must be positive")
        self.embedding = nn.Embedding(vocabulary, width, padding_idx=padding_index)
        self.padding_index = padding_index

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> EncodedDomain:
        if tokens.ndim != 2 or tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("tokens must be an integer tensor shaped (batch,time)")
        if mask is None:
            mask = torch.ones_like(tokens, dtype=torch.bool)
            if self.padding_index is not None:
                mask = tokens != self.padding_index
        if mask.shape != tokens.shape or mask.dtype != torch.bool:
            raise ValueError("token mask must be boolean and match tokens")
        return EncodedDomain(
            self.embedding(tokens) * mask.unsqueeze(-1), mask,
            DomainSpec("sequence", boundary="causal"),
        )


class ContinuousSignalEncoder(nn.Module):
    def __init__(self, channels: int, width: int, *, anti_alias: bool = True) -> None:
        super().__init__()
        if min(channels, width) <= 0:
            raise ValueError("channels and width must be positive")
        self.channels = channels
        self.projection = nn.Linear(channels, width)
        self.anti_alias = AntiAliasActivation(width) if anti_alias else nn.Identity()

    def forward(
        self, signal: Tensor, mask: Tensor, *, sample_interval: float,
        amplitude_scale: float = 1.0, kind: str = "sensor",
    ) -> EncodedDomain:
        if signal.ndim != 3 or signal.shape[-1] != self.channels:
            raise ValueError("signal must have shape (batch,time,channels)")
        if mask.shape != signal.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("signal mask must be boolean with shape (batch,time)")
        domain = DomainSpec(kind, sample_interval, "causal", True, amplitude_scale)
        calibrated = signal / amplitude_scale
        values = self.anti_alias(self.projection(calibrated)) * mask.unsqueeze(-1)
        return EncodedDomain(values, mask, domain)


@dataclass(frozen=True, slots=True)
class ImageBand:
    data: Tensor
    mask: Tensor
    level: int
    orientation: str

    def __post_init__(self) -> None:
        if self.data.ndim != 4 or self.mask.shape != self.data.shape[:3] or self.mask.dtype != torch.bool:
            raise ValueError("image bands require (batch,height,width,channels) and a matching mask")
        if self.level < 0 or self.orientation not in {"LL", "LH", "HL", "HH"}:
            raise ValueError("invalid image level or orientation")


@dataclass(frozen=True, slots=True)
class ImageLevelContext:
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class ImageReconstructionContext:
    levels: tuple[ImageLevelContext, ...]


class SeparableLifting2D(nn.Module):
    """Learned separable 2-D lifting with LL/LH/HL/HH bands and exact inverse."""

    def __init__(self, channels: int, levels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if min(channels, levels) <= 0:
            raise ValueError("channels and levels must be positive")
        self.channels, self.level_count = channels, levels
        self.horizontal = nn.ModuleList(LiftingLevel(channels, kernel_size) for _ in range(levels))
        self.vertical = nn.ModuleList(LiftingLevel(channels, kernel_size) for _ in range(levels))

    @staticmethod
    def _horizontal(level: LiftingLevel, x: Tensor, mask: Tensor):
        batch, height, width, channels = x.shape
        values = x.reshape(batch * height, width, channels)
        valid = mask.reshape(batch * height, width)
        detail, detail_mask, approximation, approximation_mask = level.analysis(values, valid, boundary="reflect")
        return (
            detail.reshape(batch, height, detail.shape[1], channels),
            detail_mask.reshape(batch, height, detail.shape[1]),
            approximation.reshape(batch, height, approximation.shape[1], channels),
            approximation_mask.reshape(batch, height, approximation.shape[1]),
        )

    @staticmethod
    def _vertical(level: LiftingLevel, x: Tensor, mask: Tensor):
        batch, height, width, channels = x.shape
        values = x.permute(0, 2, 1, 3).reshape(batch * width, height, channels)
        valid = mask.permute(0, 2, 1).reshape(batch * width, height)
        detail, detail_mask, approximation, approximation_mask = level.analysis(values, valid, boundary="reflect")
        def restore(value: Tensor) -> Tensor:
            return value.reshape(batch, width, value.shape[1], *value.shape[2:]).transpose(1, 2)
        return restore(detail), restore(detail_mask), restore(approximation), restore(approximation_mask)

    @staticmethod
    def _vertical_inverse(level: LiftingLevel, detail: Tensor, approximation: Tensor, height: int) -> Tensor:
        batch, _, width, channels = approximation.shape
        d = detail.permute(0, 2, 1, 3).reshape(batch * width, detail.shape[1], channels)
        a = approximation.permute(0, 2, 1, 3).reshape(batch * width, approximation.shape[1], channels)
        return level.synthesis(d, a, height, boundary="reflect").reshape(batch, width, height, channels).transpose(1, 2)

    @staticmethod
    def _horizontal_inverse(level: LiftingLevel, detail: Tensor, approximation: Tensor, width: int) -> Tensor:
        batch, height, _, channels = approximation.shape
        d = detail.reshape(batch * height, detail.shape[2], channels)
        a = approximation.reshape(batch * height, approximation.shape[2], channels)
        return level.synthesis(d, a, width, boundary="reflect").reshape(batch, height, width, channels)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[tuple[ImageBand, ...], ImageReconstructionContext]:
        if x.ndim != 4 or x.shape[-1] != self.channels:
            raise ValueError("image input must have shape (batch,height,width,channels)")
        if mask is None:
            mask = torch.ones(x.shape[:3], dtype=torch.bool, device=x.device)
        if mask.shape != x.shape[:3] or mask.dtype != torch.bool:
            raise ValueError("image mask must be boolean with shape (batch,height,width)")
        current, current_mask = x, mask
        bands, contexts = [], []
        for index, (horizontal, vertical) in enumerate(zip(self.horizontal, self.vertical, strict=True)):
            contexts.append(ImageLevelContext(current.shape[1], current.shape[2]))
            h_detail, h_detail_mask, h_approx, h_approx_mask = self._horizontal(horizontal, current, current_mask)
            lh, lh_mask, ll, ll_mask = self._vertical(vertical, h_approx, h_approx_mask)
            hh, hh_mask, hl, hl_mask = self._vertical(vertical, h_detail, h_detail_mask)
            bands.extend((ImageBand(lh, lh_mask, index, "LH"), ImageBand(hl, hl_mask, index, "HL"), ImageBand(hh, hh_mask, index, "HH")))
            current, current_mask = ll, ll_mask
        bands.append(ImageBand(current, current_mask, self.level_count, "LL"))
        return tuple(bands), ImageReconstructionContext(tuple(contexts))

    def inverse(self, bands: tuple[ImageBand, ...] | list[ImageBand], context: ImageReconstructionContext) -> Tensor:
        if len(bands) != 3 * self.level_count + 1 or len(context.levels) != self.level_count:
            raise ValueError("image bands/context do not match this transform")
        current = bands[-1].data
        for index in range(self.level_count - 1, -1, -1):
            lh, hl, hh = bands[3 * index : 3 * index + 3]
            if (lh.orientation, hl.orientation, hh.orientation) != ("LH", "HL", "HH"):
                raise ValueError("image band orientations are out of order")
            shape = context.levels[index]
            h_approx = self._vertical_inverse(self.vertical[index], lh.data, current, shape.height)
            h_detail = self._vertical_inverse(self.vertical[index], hh.data, hl.data, shape.height)
            current = self._horizontal_inverse(self.horizontal[index], h_detail, h_approx, shape.width)
        return current


@dataclass(frozen=True, slots=True)
class VideoBand:
    data: Tensor
    mask: Tensor
    spatial_level: int
    orientation: str
    temporal_scale: int
    temporal_kind: str


@dataclass(frozen=True, slots=True)
class VideoReconstructionContext:
    spatial: ImageReconstructionContext
    temporal: tuple[ReconstructionContext, ...]
    spatial_masks: tuple[Tensor, ...]
    frames: int


class FactorizedVideoLifting(nn.Module):
    """Exact spatial-per-frame lifting followed by exact temporal lifting per spatial band."""

    def __init__(self, channels: int, spatial_levels: int, temporal_levels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.spatial = SeparableLifting2D(channels, spatial_levels, kernel_size)
        self.temporal = LiftingAnalysisBank(channels, temporal_levels, kernel_size)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[tuple[VideoBand, ...], VideoReconstructionContext]:
        if x.ndim != 5 or x.shape[-1] != self.channels:
            raise ValueError("video must have shape (batch,time,height,width,channels)")
        batch, frames, height, width, channels = x.shape
        if mask is None:
            mask = torch.ones(x.shape[:4], dtype=torch.bool, device=x.device)
        if mask.shape != x.shape[:4] or mask.dtype != torch.bool:
            raise ValueError("video mask must match batch,time,height,width")
        spatial, spatial_context = self.spatial(
            x.reshape(batch * frames, height, width, channels), mask.reshape(batch * frames, height, width)
        )
        result, temporal_contexts, spatial_masks = [], [], []
        for item in spatial:
            h, w = item.data.shape[1:3]
            values = item.data.reshape(batch, frames, h, w, channels).permute(0, 2, 3, 1, 4).reshape(batch * h * w, frames, channels)
            valid = item.mask.reshape(batch, frames, h, w).permute(0, 2, 3, 1).reshape(batch * h * w, frames)
            temporal, temporal_context = self.temporal(values, valid)
            temporal_contexts.append(temporal_context)
            spatial_masks.append(item.mask.reshape(batch, frames, h, w))
            for band in temporal:
                data = band.data.reshape(batch, h, w, band.data.shape[1], channels).permute(0, 3, 1, 2, 4)
                band_mask = band.mask.reshape(batch, h, w, band.mask.shape[1]).permute(0, 3, 1, 2)
                result.append(VideoBand(data, band_mask, item.level, item.orientation, band.scale, band.kind))
        return tuple(result), VideoReconstructionContext(spatial_context, tuple(temporal_contexts), tuple(spatial_masks), frames)

    def inverse(self, bands: tuple[VideoBand, ...] | list[VideoBand], context: VideoReconstructionContext) -> Tensor:
        temporal_count = self.temporal.level_count + 1
        spatial_count = 3 * self.spatial.level_count + 1
        if len(bands) != temporal_count * spatial_count or len(context.temporal) != spatial_count:
            raise ValueError("video bands/context do not match this transform")
        spatial_reconstructed = []
        for spatial_index in range(spatial_count):
            group = bands[spatial_index * temporal_count : (spatial_index + 1) * temporal_count]
            batch, _, height, width, channels = group[0].data.shape
            temporal_bands = []
            for item in group:
                data = item.data.permute(0, 2, 3, 1, 4).reshape(batch * height * width, item.data.shape[1], channels)
                mask = item.mask.permute(0, 2, 3, 1).reshape(batch * height * width, item.mask.shape[1])
                temporal_bands.append(ScaleTensor(data, mask, item.temporal_scale, 2**item.temporal_scale, 2 ** (item.temporal_scale + (item.temporal_kind == "detail")), item.temporal_kind))
            restored = self.temporal.inverse(temporal_bands, context.temporal[spatial_index])
            restored = restored.reshape(batch, height, width, context.frames, channels).permute(0, 3, 1, 2, 4)
            template = group[0]
            spatial_reconstructed.append((restored, context.spatial_masks[spatial_index], template.spatial_level, template.orientation))
        frames = []
        for time in range(context.frames):
            image_bands = tuple(ImageBand(data[:, time], mask[:, time], level, orientation) for data, mask, level, orientation in spatial_reconstructed)
            frames.append(self.spatial.inverse(image_bands, context.spatial).unsqueeze(1))
        return torch.cat(frames, 1)


class PhysicalFieldEncoder(nn.Module):
    def __init__(self, value_dim: int, coordinate_dim: int, coefficient_dim: int, forcing_dim: int, width: int) -> None:
        super().__init__()
        if min(value_dim, coordinate_dim, width) <= 0 or min(coefficient_dim, forcing_dim) < 0:
            raise ValueError("field dimensions must be nonnegative with positive values/coordinates/width")
        self.dimensions = (value_dim, coordinate_dim, coefficient_dim, forcing_dim)
        self.projection = nn.Linear(sum(self.dimensions) + 2, width)

    def forward(
        self, values: Tensor, coordinates: Tensor, coefficients: Tensor, forcing: Tensor,
        boundary_mask: Tensor, valid_mask: Tensor, grid_spacing: Tensor,
    ) -> EncodedDomain:
        spatial = values.shape[:-1]
        tensors = (coordinates, coefficients, forcing)
        expected_dims = self.dimensions[1:]
        if values.shape[-1] != self.dimensions[0] or any(t.shape[:-1] != spatial or t.shape[-1] != d for t, d in zip(tensors, expected_dims, strict=True)):
            raise ValueError("field feature shapes do not match the configured dimensions")
        if boundary_mask.shape != spatial or valid_mask.shape != spatial or boundary_mask.dtype != torch.bool or valid_mask.dtype != torch.bool:
            raise ValueError("field boundary and validity masks must match the spatial grid")
        if grid_spacing.ndim != 1 or grid_spacing.numel() != coordinates.shape[-1] or not bool((grid_spacing > 0).all()):
            raise ValueError("grid spacing must be a positive vector matching coordinates")
        spacing = grid_spacing.log().mean().expand(*spatial, 1)
        inputs = torch.cat((*((values, coordinates, coefficients, forcing)), boundary_mask.unsqueeze(-1), spacing), -1)
        encoded = self.projection(inputs) * valid_mask.unsqueeze(-1)
        return EncodedDomain(encoded, valid_mask, DomainSpec("field", float(grid_spacing.mean()), "physical"))


class ConservationProjector(nn.Module):
    """Project scalar updates to zero integral while enforcing supplied boundary values."""

    def forward(self, update: Tensor, valid_mask: Tensor, boundary_mask: Tensor | None = None) -> Tensor:
        if update.shape[:-1] != valid_mask.shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid mask must match the field update")
        weight = valid_mask.unsqueeze(-1)
        mean = (update * weight).sum(tuple(range(1, update.ndim - 1)), keepdim=True) / weight.sum(tuple(range(1, weight.ndim - 1)), keepdim=True).clamp_min(1)
        result = (update - mean) * weight
        if boundary_mask is not None:
            if boundary_mask.shape != valid_mask.shape or boundary_mask.dtype != torch.bool:
                raise ValueError("boundary mask must match the field")
            result = result.masked_fill(boundary_mask.unsqueeze(-1), 0)
        return result


class GraphPolynomialEncoder(nn.Module):
    """Permutation-equivariant localized Chebyshev filters of the normalized graph Laplacian."""

    def __init__(self, input_dim: int, width: int, order: int = 3) -> None:
        super().__init__()
        if min(input_dim, width, order) <= 0:
            raise ValueError("graph dimensions and polynomial order must be positive")
        self.input_dim, self.order = input_dim, order
        self.output = nn.Linear((order + 1) * input_dim, width)

    def forward(self, nodes: Tensor, adjacency: Tensor, mask: Tensor) -> EncodedDomain:
        if nodes.ndim != 3 or nodes.shape[-1] != self.input_dim:
            raise ValueError("nodes must have shape (batch,nodes,input_dim)")
        batch, count = nodes.shape[:2]
        if adjacency.ndim == 2:
            adjacency = adjacency.expand(batch, -1, -1)
        if adjacency.shape != (batch, count, count) or mask.shape != (batch, count) or mask.dtype != torch.bool:
            raise ValueError("adjacency and node mask shapes are invalid")
        valid_edges = mask.unsqueeze(1) & mask.unsqueeze(2)
        graph = adjacency * valid_edges
        degree = graph.sum(-1).clamp_min(1e-12)
        normalized = graph * degree.rsqrt().unsqueeze(-1) * degree.rsqrt().unsqueeze(-2)
        identity = torch.eye(count, device=nodes.device, dtype=nodes.dtype).expand(batch, -1, -1)
        laplacian = identity - normalized
        terms = [nodes]
        if self.order:
            terms.append(torch.bmm(laplacian, nodes))
        for _ in range(2, self.order + 1):
            terms.append(2 * torch.bmm(laplacian, terms[-1]) - terms[-2])
        values = self.output(torch.cat(terms, -1)) * mask.unsqueeze(-1)
        return EncodedDomain(values, mask, DomainSpec("graph", boundary="graph", ordered=False))


class InducingPointSetEncoder(nn.Module):
    """Permutation-invariant inducing-point representation for domains with no meaningful order."""

    def __init__(self, input_dim: int, width: int, inducing_points: int) -> None:
        super().__init__()
        if min(input_dim, width, inducing_points) <= 0:
            raise ValueError("set dimensions and inducing point count must be positive")
        self.input_dim, self.width = input_dim, width
        self.input = nn.Linear(input_dim, width)
        self.inducing = nn.Parameter(torch.randn(inducing_points, width) / width**0.5)
        self.key, self.value = nn.Linear(width, width), nn.Linear(width, width)

    def forward(self, items: Tensor, mask: Tensor) -> EncodedDomain:
        if items.ndim != 3 or items.shape[-1] != self.input_dim or mask.shape != items.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("set items/mask have invalid shapes")
        encoded = self.input(items)
        scores = torch.einsum("md,bnd->bmn", self.inducing, self.key(encoded)) / self.width**0.5
        scores = scores.masked_fill(~mask.unsqueeze(1), -torch.inf)
        valid = mask.any(1, keepdim=True).unsqueeze(-1)
        weights = torch.softmax(torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores)), -1)
        weights = weights * mask.unsqueeze(1)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        values = torch.einsum("bmn,bnd->bmd", weights, self.value(encoded)) * valid
        inducing_mask = mask.any(1, keepdim=True).expand(-1, self.inducing.shape[0])
        return EncodedDomain(values, inducing_mask, DomainSpec("set", boundary="none", ordered=False))


def require_meaningful_order(domain: DomainSpec) -> None:
    if not domain.ordered or domain.kind in {"graph", "set"}:
        raise ValueError("one-dimensional spectral lifting requires a meaningful domain order")


class DirectionalResonator2D(nn.Module):
    """Four noncausal directional sweeps that retain native two-dimensional geometry."""

    def __init__(self, width: int, heads: int, modes: int, mimo_rank: int) -> None:
        super().__init__()
        if min(width, heads, modes, mimo_rank) <= 0:
            raise ValueError("directional resonator dimensions must be positive")
        self.width = width
        self.sweeps = nn.ModuleList(
            ComplexResonator(width, heads, modes, mimo_rank) for _ in range(4)
        )
        self.gate = nn.Linear(width, 4)

    @staticmethod
    def _restore(value: Tensor, batch: int, height: int, width: int, axis: int) -> Tensor:
        if axis == 2:
            return value.reshape(batch, height, width, -1)
        return value.reshape(batch, width, height, -1).transpose(1, 2)

    def forward(
        self, field: Tensor, mask: Tensor, *, spacing: tuple[float, float] = (1.0, 1.0)
    ) -> Tensor:
        if field.ndim != 4 or field.shape[-1] != self.width or mask.shape != field.shape[:3] or mask.dtype != torch.bool:
            raise ValueError("directional field/mask shapes are invalid")
        if len(spacing) != 2 or min(spacing) <= 0:
            raise ValueError("two positive spatial spacings are required")
        batch, height, width, channels = field.shape
        row = field.reshape(batch * height, width, channels)
        row_mask = mask.reshape(batch * height, width)
        column = field.transpose(1, 2).reshape(batch * width, height, channels)
        column_mask = mask.transpose(1, 2).reshape(batch * width, height)
        outputs = []
        for index, (values, valid, axis, step) in enumerate((
            (row, row_mask, 2, spacing[1]), (row.flip(1), row_mask.flip(1), 2, spacing[1]),
            (column, column_mask, 1, spacing[0]), (column.flip(1), column_mask.flip(1), 1, spacing[0]),
        )):
            output, _, _ = self.sweeps[index](values, mask=valid, sample_interval=step)
            if index % 2:
                output = output.flip(1)
            outputs.append(self._restore(output, batch, height, width, axis))
        weights = torch.softmax(self.gate(field), -1)
        mixed = sum(weights[..., index : index + 1] * output for index, output in enumerate(outputs))
        return mixed * mask.unsqueeze(-1)


class SpatialResonanceNetwork(nn.Module):
    """Image/field MRRN form: 2-D lifting, directional band processing, exact synthesis."""

    def __init__(
        self, input_dim: int, model_dim: int, output_dim: int, *, levels: int = 3,
        layers: int = 4, heads: int = 4, modes: int = 8, mimo_rank: int = 2,
    ) -> None:
        super().__init__()
        if min(input_dim, model_dim, output_dim, levels, layers, heads, modes, mimo_rank) <= 0:
            raise ValueError("spatial network dimensions must be positive")
        self.input_dim = input_dim
        self.encoder, self.decoder = nn.Linear(input_dim, model_dim), nn.Linear(model_dim, output_dim)
        self.analysis = SeparableLifting2D(model_dim, levels)
        band_count = 3 * levels + 1
        self.norms = nn.ModuleList(nn.ModuleList(nn.RMSNorm(model_dim) for _ in range(band_count)) for _ in range(layers))
        self.blocks = nn.ModuleList(
            nn.ModuleList(
                DirectionalResonator2D(model_dim, heads, modes, mimo_rank) for _ in range(band_count)
            )
            for _ in range(layers)
        )
        self.layer_scale = nn.Parameter(torch.full((layers, band_count), 1e-2))

    def forward(
        self, field: Tensor, mask: Tensor | None = None, *, spacing: tuple[float, float] = (1.0, 1.0)
    ) -> Tensor:
        if field.ndim != 4 or field.shape[-1] != self.input_dim:
            raise ValueError("spatial network input must be (batch,height,width,input_dim)")
        if mask is None:
            mask = torch.ones(field.shape[:3], dtype=torch.bool, device=field.device)
        if mask.shape != field.shape[:3] or mask.dtype != torch.bool:
            raise ValueError("spatial network mask must match its field")
        encoded = self.encoder(field) * mask.unsqueeze(-1)
        bands, context = self.analysis(encoded, mask)
        bands = list(bands)
        for layer, (norms, blocks) in enumerate(zip(self.norms, self.blocks, strict=True)):
            for index, (norm, block) in enumerate(zip(norms, blocks, strict=True)):
                band = bands[index]
                normalized = norm(band.data) * band.mask.unsqueeze(-1)
                updated = band.data + self.layer_scale[layer, index] * block(
                    normalized, band.mask, spacing=spacing
                )
                bands[index] = ImageBand(updated * band.mask.unsqueeze(-1), band.mask, band.level, band.orientation)
        return self.decoder(self.analysis.inverse(bands, context)) * mask.unsqueeze(-1)
