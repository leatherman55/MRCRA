"""Versioned deterministic streaming checkpoints, resets, and quantization diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from . import complex_ops as c
from .lifting import LiftingStreamState, ScaleTensor
from .memory import EideticMemory
from .mixer import AntiAliasState
from .model import BlockStreamState, MRRN, MRRNStreamState
from .resonance import ResonatorState
from .scale_exchange import ScaleExchangeStreamState


FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class QuantizedTensor:
    values: Tensor
    scale: Tensor
    original_dtype: str


@dataclass(frozen=True, slots=True)
class QuantizationReport:
    rms_error: Tensor
    amplitude_bias: Tensor
    phase_error: Tensor | None


def configuration_hash(model: MRRN) -> str:
    payload = json.dumps(asdict(model.config), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


def model_hash(model: MRRN) -> str:
    digest = sha256(configuration_hash(model).encode())
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _band_to_dict(band: ScaleTensor | None) -> dict | None:
    if band is None:
        return None
    return {
        "data": band.data, "mask": band.mask, "scale": band.scale,
        "sample_interval": band.sample_interval, "support": band.support, "kind": band.kind,
    }


def _band_from_dict(value: dict | None) -> ScaleTensor | None:
    return None if value is None else ScaleTensor(**value)


def stream_state_dict(state: MRRNStreamState) -> dict:
    lifting = {
        "pending": [None if item is None else {"data": item[0], "mask": item[1]} for item in state.lifting.pending],
        "even_history": state.lifting.even_history,
        "detail_history": state.lifting.detail_history,
        "emitted": state.lifting.emitted,
        "steps": state.lifting.steps,
        "sample_interval": state.lifting.sample_interval,
    }
    blocks = []
    for block in state.blocks:
        blocks.append({
            "resonators": [
                {"value": item.value, "previous_drive": item.previous_drive, "steps": item.steps}
                for item in block.resonators
            ],
            "exchange": {
                "fine_values": block.exchange.fine_values,
                "fine_masks": block.exchange.fine_masks,
                "latest_coarse": block.exchange.latest_coarse,
            },
            "recent_features": block.recent_features,
            "recent_masks": block.recent_masks,
            "recent_times": block.recent_times,
            "scale_steps": block.scale_steps,
            "anti_alias": [
                None if state is None else {
                    "pre_history": state.pre_history, "post_history": state.post_history
                }
                for state in block.anti_alias
            ],
        })
    return {
        "lifting": lifting, "blocks": blocks,
        "latest_bands": [_band_to_dict(item) for item in state.latest_bands],
        "position": state.position, "batch": state.batch, "sample_interval": state.sample_interval,
    }


def stream_state_from_dict(value: dict) -> MRRNStreamState:
    lifting_value = value["lifting"]
    lifting = LiftingStreamState(
        [None if item is None else (item["data"], item["mask"]) for item in lifting_value["pending"]],
        list(lifting_value["even_history"]), list(lifting_value["detail_history"]),
        list(lifting_value["emitted"]), int(lifting_value["steps"]), float(lifting_value["sample_interval"]),
    )
    blocks = []
    for item in value["blocks"]:
        exchange = item["exchange"]
        blocks.append(BlockStreamState(
            [ResonatorState(entry["value"], entry["previous_drive"], int(entry["steps"])) for entry in item["resonators"]],
            ScaleExchangeStreamState(
                [list(group) for group in exchange["fine_values"]],
                [list(group) for group in exchange["fine_masks"]],
                list(exchange["latest_coarse"]),
            ),
            [list(group) for group in item["recent_features"]],
            [list(group) for group in item["recent_masks"]],
            [list(group) for group in item["recent_times"]],
            list(item["scale_steps"]),
            [
                None if state is None else AntiAliasState(state["pre_history"], state["post_history"])
                for state in item["anti_alias"]
            ],
        ))
    return MRRNStreamState(
        lifting, blocks, [_band_from_dict(item) for item in value["latest_bands"]],
        int(value["position"]), int(value["batch"]), float(value["sample_interval"]),
    )


def save_stream_checkpoint(
    path: str | Path,
    model: MRRN,
    state: MRRNStreamState,
    memories: Sequence[EideticMemory] | None = None,
) -> None:
    if state.batch <= 0 or (memories is not None and len(memories) != state.batch):
        raise ValueError("checkpoint state and memory batch are inconsistent")
    dtype = str(state.lifting.even_history[0].dtype)
    payload = {
        "format_version": FORMAT_VERSION,
        "configuration_hash": configuration_hash(model),
        "model_hash": model_hash(model),
        "dtype": dtype,
        "stream": stream_state_dict(state.detach()),
        "memories": None if memories is None else [memory.state_dict() for memory in memories],
    }
    torch.save(payload, Path(path))


def load_stream_checkpoint(
    path: str | Path, model: MRRN, *, map_location: str | torch.device | None = None
) -> tuple[MRRNStreamState, list[EideticMemory] | None]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported stream checkpoint version")
    if payload.get("configuration_hash") != configuration_hash(model):
        raise ValueError("checkpoint configuration hash does not match the model")
    if payload.get("model_hash") != model_hash(model):
        raise ValueError("checkpoint model weights do not match")
    state = stream_state_from_dict(payload["stream"])
    memory_states = payload.get("memories")
    memories = None
    if memory_states is not None:
        memories = []
        for saved in memory_states:
            memory = EideticMemory(saved["capacity"], saved["key_dim"], saved["value_dim"], saved["signature_dim"])
            memory.load_state_dict(saved)
            memories.append(memory)
    return state, memories


def hard_reset(
    model: MRRN, state: MRRNStreamState, memories: Sequence[EideticMemory] | None = None
) -> MRRNStreamState:
    if memories is not None:
        if len(memories) != state.batch:
            raise ValueError("hard-reset memory batch does not match stream state")
        for memory in memories:
            memory.clear()
    tensor = state.lifting.even_history[0]
    return model.initial_stream_state(
        state.batch, sample_interval=state.sample_interval, device=tensor.device, dtype=tensor.dtype
    )


def segment_reset(model: MRRN, state: MRRNStreamState) -> MRRNStreamState:
    reset = hard_reset(model, state)
    reset.position = state.position
    return reset


def quantize_symmetric(tensor: Tensor, *, bits: int = 8) -> QuantizedTensor:
    if not tensor.is_floating_point() or not 2 <= bits <= 8:
        raise ValueError("symmetric quantization requires a float tensor and 2..8 bits")
    maximum = 2 ** (bits - 1) - 1
    scale = tensor.abs().max().clamp_min(torch.finfo(tensor.dtype).tiny) / maximum
    values = (tensor / scale).round().clamp(-maximum, maximum).to(torch.int8)
    return QuantizedTensor(values, scale, str(tensor.dtype))


def dequantize_symmetric(value: QuantizedTensor, *, dtype: torch.dtype | None = None) -> Tensor:
    if value.values.dtype != torch.int8 or value.scale.numel() != 1:
        raise ValueError("invalid quantized tensor")
    return value.values.to(dtype or value.scale.dtype) * value.scale.to(dtype or value.scale.dtype)


def quantization_diagnostics(reference: Tensor, restored: Tensor, *, complex_pairs: bool = False) -> QuantizationReport:
    if reference.shape != restored.shape:
        raise ValueError("quantization reference and restored shapes must match")
    error = (restored - reference).square().mean().sqrt()
    phase_error = None
    if complex_pairs:
        c.validate(reference)
        c.validate(restored)
        amplitude_bias = (c.magnitude(restored) - c.magnitude(reference)).mean()
        delta = torch.atan2(c.imag(restored), c.real(restored)) - torch.atan2(c.imag(reference), c.real(reference))
        phase_error = torch.atan2(torch.sin(delta), torch.cos(delta)).abs().mean()
    else:
        amplitude_bias = restored.abs().mean() - reference.abs().mean()
    return QuantizationReport(error, amplitude_bias, phase_error)
