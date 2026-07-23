from copy import deepcopy

import pytest
import torch

from mrrn import complex_ops as c
from mrrn.checkpoint import (
    QuantizedTensor,
    dequantize_symmetric,
    hard_reset,
    load_stream_checkpoint,
    quantization_diagnostics,
    quantize_symmetric,
    save_stream_checkpoint,
    segment_reset,
    stream_state_dict,
    stream_state_from_dict,
)
from mrrn.config import MRRNConfig
from mrrn.memory import MemoryItem
from mrrn.model import MRRN


def config():
    return MRRNConfig(
        input_dim=2, model_dim=4, output_dim=3, layers=1, scales=2, heads=1,
        modes=2, mimo_rank=1, attention_window=3, retrieved_items=1,
        memory_capacity=3, mixer_expansion=1, width_growth_cap=1,
        mode_growth_cap=1, width_multiple=1,
    )


def test_checkpoint_restores_every_stream_carry_cache_state_position_and_memory(tmp_path):
    torch.manual_seed(180)
    model = MRRN(config()).double().eval()
    x = torch.randn(1, 10, 2, dtype=torch.float64)
    memory = model.create_memories(1)
    vector = torch.randn(4, dtype=torch.float64)
    memory[0].write(MemoryItem(vector, vector, vector, 0, 0, 1.0))
    state = model.initial_stream_state(1, sample_interval=0.25, dtype=torch.float64)
    with torch.no_grad():
        for index in range(5):
            state = model.step(x[:, index], state, memories=memory).state
    path = tmp_path / "stream.pt"
    save_stream_checkpoint(path, model, state, memory)
    with torch.no_grad():
        expected = torch.stack([
            model.step(x[:, index], state, memories=memory).prediction
            for index in range(5, 10)
        ], 1)
    restored, restored_memory = load_stream_checkpoint(path, model)
    assert restored_memory is not None and len(restored_memory[0]) == 1
    with torch.no_grad():
        actual = torch.stack([
            model.step(x[:, index], restored, memories=restored_memory).prediction
            for index in range(5, 10)
        ], 1)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert restored.position == 10


def test_stream_state_dictionary_roundtrip_and_reset_semantics():
    model = MRRN(config())
    state = model.initial_stream_state(2)
    state = model.step(torch.randn(2, 2), state).state
    restored = stream_state_from_dict(stream_state_dict(state))
    assert restored.position == 1 and restored.batch == 2
    segment = segment_reset(model, restored)
    assert segment.position == 1 and segment.lifting.steps == 0
    memories = model.create_memories(2)
    vector = torch.randn(4)
    memories[0].write(MemoryItem(vector, vector, vector, 0, 0, 1))
    hard = hard_reset(model, restored, memories)
    assert hard.position == 0 and hard.lifting.steps == 0
    assert not len(memories[0])
    with pytest.raises(ValueError):
        hard_reset(model, restored, model.create_memories(1))


def test_soft_boundary_injects_feature_and_prevents_attention_cache_crossing_per_batch():
    torch.manual_seed(191)
    model = MRRN(config()).eval()
    model.boundary_embedding.data.fill_(1)
    state = model.initial_stream_state(2)
    # Complete two fine-scale coefficients so a cache exists.
    for _ in range(4):
        state = model.step(torch.randn(2, 2), state).state
    old_masks = [item.clone() for item in state.blocks[0].recent_masks[0]]
    assert old_masks
    model.step(
        torch.zeros(2, 2), state, soft_boundary=torch.tensor([True, False])
    )
    for old, current in zip(old_masks, state.blocks[0].recent_masks[0][:-1], strict=False):
        assert not current[0].any()
        assert torch.equal(current[1], old[1])


def test_quantization_reports_amplitude_and_phase_error_not_only_tensor_error():
    state = c.pair(torch.randn(4, 5), torch.randn(4, 5))
    quantized = quantize_symmetric(state)
    restored = dequantize_symmetric(quantized, dtype=state.dtype)
    report = quantization_diagnostics(state, restored, complex_pairs=True)
    assert report.rms_error >= 0 and report.phase_error is not None
    assert report.phase_error < 0.05 and report.amplitude_bias.abs() < 0.05
    with pytest.raises(ValueError):
        quantize_symmetric(torch.ones(2, dtype=torch.int64))
    with pytest.raises(ValueError):
        quantization_diagnostics(torch.ones(2), torch.ones(3))


def test_checkpoint_rejects_wrong_configuration_weights_and_version(tmp_path):
    model = MRRN(config())
    state = model.initial_stream_state(1)
    path = tmp_path / "checkpoint.pt"
    save_stream_checkpoint(path, model, state)
    changed = deepcopy(model)
    changed.encoder.weight.data.add_(1)
    with pytest.raises(ValueError, match="weights"):
        load_stream_checkpoint(path, changed)
    payload = torch.load(path, weights_only=True)
    payload["format_version"] = 99
    wrong = tmp_path / "wrong.pt"
    torch.save(payload, wrong)
    with pytest.raises(ValueError, match="version"):
        load_stream_checkpoint(wrong, model)
    different = MRRN(MRRNConfig(
        input_dim=2, model_dim=6, output_dim=3, layers=1, scales=2, heads=1,
        modes=2, mimo_rank=1, attention_window=3, retrieved_items=1,
        memory_capacity=3, mixer_expansion=1, width_growth_cap=1,
        mode_growth_cap=1, width_multiple=1,
    ))
    with pytest.raises(ValueError, match="configuration"):
        load_stream_checkpoint(path, different)
    with pytest.raises(ValueError):
        save_stream_checkpoint(tmp_path / "bad.pt", model, state, model.create_memories(2))
    with pytest.raises(ValueError):
        dequantize_symmetric(QuantizedTensor(torch.ones(2), torch.ones(()), "torch.float32"))
    real = torch.randn(3)
    real_report = quantization_diagnostics(real, dequantize_symmetric(quantize_symmetric(real)))
    assert real_report.phase_error is None
