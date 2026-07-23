import pytest
import torch
from dataclasses import replace

from mrrn.config import MRRNConfig
from mrrn.memory import MemoryItem
from mrrn.attention import AttentionCandidates
from mrrn.memory import EideticMemory
from mrrn.model import (
    BlockState, MRRN, MRRNState, _causal_expand, _join_candidates,
    _landmark_candidates, _local_candidates, _memory_candidates,
)


def tiny_config(*, causal=True):
    return MRRNConfig(
        input_dim=3,
        model_dim=8,
        output_dim=5,
        layers=2,
        scales=3,
        heads=2,
        modes=3,
        mimo_rank=1,
        attention_window=4,
        memory_capacity=8,
        retrieved_items=2,
        mixer_expansion=1.5,
        width_growth_cap=1,
        mode_growth_cap=1,
        width_multiple=2,
        causal=causal,
    )


@pytest.mark.parametrize("length", [7, 8])
def test_complete_model_sequence_global_and_operator_shapes_with_gradients(length):
    torch.manual_seed(3)
    model = MRRN(tiny_config()).double()
    x = torch.randn(2, length, 3, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(2, length, dtype=torch.bool)
    mask[0, -1] = False
    sequence = model(x, mask, output_mode="sequence")
    global_output = model(x, mask, output_mode="global")
    operator = model(x, mask, output_mode="operator")
    assert sequence.prediction.shape == operator.prediction.shape == (2, length, 5)
    assert global_output.prediction.shape == (2, 5)
    assert (sequence.prediction[0, -1] == 0).all()
    assert len(sequence.bands) == 3 and len(sequence.state.blocks) == 2
    assert not torch.allclose(sequence.prediction, operator.prediction)
    sequence.prediction.square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_causal_model_future_perturbation_cannot_change_past_output():
    torch.manual_seed(7)
    model = MRRN(tiny_config()).double().eval()
    x = torch.randn(1, 12, 3, dtype=torch.float64)
    changed = x.clone()
    changed[:, 8:] += 100
    with torch.no_grad():
        baseline = model(x).prediction
        actual = model(changed).prediction
    torch.testing.assert_close(actual[:, :8], baseline[:, :8], atol=1e-10, rtol=1e-10)


def test_noncausal_model_uses_bidirectional_and_exact_offline_synthesis_path():
    model = MRRN(tiny_config(causal=False))
    x = torch.randn(1, 9, 3)
    output = model(x)
    assert output.prediction.shape == (1, 9, 5)
    assert model.blocks[0].reverse_resonators is not None


def test_sequence_only_model_omits_dead_global_parameters_and_rejects_global_output():
    config = replace(tiny_config(), enable_global_head=False)
    model = MRRN(config)
    assert model.global_head is None
    assert model.global_memory_adapter is None
    assert model.global_memory_gate is None
    assert not any(name.startswith("global_") for name, _ in model.named_parameters())
    with pytest.raises(ValueError, match="global output is disabled"):
        model(torch.randn(1, 4, 3), output_mode="global")


def test_local_candidate_windows_are_bounded_and_causal():
    model = MRRN(tiny_config())
    x = torch.randn(1, 9, 3)
    encoded = model.encoder(x)
    bands, _ = model.analysis(encoded)
    adapted = model._adapt_analysis(bands)
    query, candidates, times, scales = _local_candidates(adapted[0], 3)
    assert query.shape[1] == 1 and candidates.features.shape[1] == 3
    assert (candidates.times <= times).all()
    assert scales.unique().item() == adapted[0].scale


def test_offline_local_candidate_window_contains_past_and_future_without_crossing_bounds():
    model = MRRN(tiny_config(causal=False))
    raw, _ = model.analysis(model.encoder(torch.randn(1, 10, 3)), boundary="reflect")
    band = model._adapt_analysis(raw)[0]
    _, candidates, times, _ = _local_candidates(band, 3, causal=False)
    assert candidates.times[2].tolist() == [3.0, 5.0, 7.0]
    assert times[2].item() == 5.0
    assert not candidates.mask[0, 0] and not candidates.mask[-1, -1]


def test_causal_expand_releases_only_completed_coefficients():
    data = torch.tensor([[[1.0], [2.0], [3.0]]])
    expanded = _causal_expand(data, 8, support=2)
    assert expanded.flatten().tolist() == [0, 1, 1, 2, 2, 3, 3, 3]
    assert _causal_expand(data, 0, 2).shape == (1, 0, 1)
    assert _causal_expand(data[:, :0], 3, 2).shape == (1, 3, 1)


def test_coarse_landmark_candidates_are_bounded_typed_and_completion_time_causal():
    model = MRRN(tiny_config())
    raw, _ = model.analysis(model.encoder(torch.randn(1, 12, 3)))
    bands = model._adapt_analysis(raw)
    candidates = _landmark_candidates(
        bands[0], bands[1], model.blocks[0].landmark_values[0], 2, causal=True
    )
    assert candidates.features.shape == (6, 2, 8)
    assert (candidates.kinds == 1).all()
    query_completion = (torch.arange(6) + 1) * bands[0].support - 1
    valid_times = candidates.times.reshape(1, 6, 2)[0]
    valid_mask = candidates.mask.reshape(1, 6, 2)[0]
    assert (valid_times[valid_mask] <= query_completion[:, None].expand_as(valid_times)[valid_mask]).all()


def test_model_state_detach_and_contract_failures():
    model = MRRN(tiny_config())
    state = model.initial_state(2)
    detached = state.detach()
    assert len(detached.blocks) == 2
    with pytest.raises(ValueError):
        MRRN(MRRNConfig(input_dim=2, scales=1))
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 2))
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 3), torch.ones(2, 3))
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 3), output_mode="bad")
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 3), state=MRRNState(tuple()))
    bad_block = BlockState(tuple())
    with pytest.raises(ValueError):
        model.blocks[0](model(torch.randn(2, 3, 3)).bands, bad_block)


@pytest.mark.parametrize("length", [7, 8, 15])
def test_constant_state_streaming_matches_batch_causal_output_and_completed_bands(length):
    torch.manual_seed(71 + length)
    model = MRRN(tiny_config()).double().eval()
    x = torch.randn(2, length, 3, dtype=torch.float64)
    mask = torch.ones(2, length, dtype=torch.bool)
    mask[0, -2:] = False
    with torch.no_grad():
        batch = model(x, mask, sample_interval=0.125)
        state = model.initial_stream_state(2, sample_interval=0.125, dtype=torch.float64)
        predictions, collected = [], [[] for _ in batch.bands]
        for position in range(length):
            result = model.step(x[:, position], state, mask[:, position])
            state = result.state
            predictions.append(result.prediction.unsqueeze(1))
            for scale, band in enumerate(result.active_bands):
                if band is not None:
                    collected[scale].append(band.data)
    torch.testing.assert_close(torch.cat(predictions, 1), batch.prediction, atol=2e-10, rtol=2e-10)
    for scale, reference in enumerate(batch.bands):
        count = length // reference.support
        actual = torch.cat(collected[scale], 1) if collected[scale] else reference.data[:, :0]
        torch.testing.assert_close(actual, reference.data[:, :count], atol=2e-10, rtol=2e-10)
    assert state.position == length
    assert all(len(cache) <= window for block in state.blocks for cache, window in zip(block.recent_features, model.blocks[0].windows, strict=True))
    detached = state.detach()
    assert all(not resonator.value.requires_grad for block in detached.blocks for resonator in block.resonators)


@pytest.mark.parametrize("continuous_signal", [False, True])
def test_vectorized_prefill_matches_every_stream_output_and_continuation_state(
    continuous_signal,
):
    torch.manual_seed(607 + int(continuous_signal))
    model = MRRN(replace(
        tiny_config(), continuous_signal=continuous_signal
    )).double().eval()
    values = torch.randn(2, 39, 3, dtype=torch.float64)
    mask = torch.rand(2, 39) > 0.1
    with torch.no_grad():
        prefill = model.prefill(values, mask)
        stream_state = model.initial_stream_state(2, dtype=torch.float64)
        stream_predictions, stream_latents = [], []
        for position in range(values.shape[1]):
            result = model.step(
                values[:, position], stream_state, mask[:, position],
                relational_context=None,
            )
            stream_state = result.state
            stream_predictions.append(result.prediction.unsqueeze(1))
            stream_latents.append(result.latent.unsqueeze(1))
    torch.testing.assert_close(
        prefill.prediction, torch.cat(stream_predictions, 1),
        atol=2e-6, rtol=2e-6,
    )
    torch.testing.assert_close(
        prefill.latent, torch.cat(stream_latents, 1),
        atol=2e-6, rtol=2e-6,
    )
    assert prefill.state.position == stream_state.position == values.shape[1]
    for chunk_block, stream_block in zip(
        prefill.state.blocks, stream_state.blocks, strict=True,
    ):
        for chunk_resonator, stream_resonator in zip(
            chunk_block.resonators, stream_block.resonators, strict=True,
        ):
            torch.testing.assert_close(
                chunk_resonator.value, stream_resonator.value,
                atol=2e-6, rtol=2e-6,
            )
            torch.testing.assert_close(
                chunk_resonator.previous_drive, stream_resonator.previous_drive,
                atol=2e-6, rtol=2e-6,
            )
        assert chunk_block.scale_steps == stream_block.scale_steps
    continuation = torch.randn(2, 3, dtype=torch.float64)
    with torch.no_grad():
        chunk_next = model.step(continuation, prefill.state).prediction
        stream_next = model.step(continuation, stream_state).prediction
    torch.testing.assert_close(chunk_next, stream_next, atol=2e-6, rtol=2e-6)


def test_prefill_preserves_chunk_state_across_repeated_calls_and_arbitrary_tail():
    torch.manual_seed(613)
    model = MRRN(tiny_config()).double().eval()
    values = torch.randn(1, 23, 3, dtype=torch.float64)
    with torch.no_grad():
        complete = model.prefill(values)
        state = model.initial_stream_state(1, dtype=torch.float64)
        first = model.prefill(values[:, :7], state=state)
        second = model.prefill(values[:, 7:18], state=first.state)
        third = model.prefill(values[:, 18:], state=second.state)
    torch.testing.assert_close(
        torch.cat((first.prediction, second.prediction, third.prediction), 1),
        complete.prediction,
        atol=2e-6,
        rtol=2e-6,
    )
    assert third.state.position == complete.state.position == 23


def test_compiled_tensor_core_boundary_preserves_outputs_and_ordered_state(
    monkeypatch,
):
    torch.manual_seed(617)
    eager = MRRN(tiny_config()).double().eval()
    compiled = MRRN(tiny_config()).double().eval()
    compiled.load_state_dict(eager.state_dict())
    compile_calls = []

    def identity_compile(function, **options):
        compile_calls.append(options)
        return function

    monkeypatch.setattr(torch, "compile", identity_compile)
    compiled.enable_compiled_tensor_cores()
    values = torch.randn(1, 12, 3, dtype=torch.float64)
    with torch.no_grad():
        expected = eager.prefill(values)
        actual = compiled.prefill(values)
    torch.testing.assert_close(actual.prediction, expected.prediction)
    assert actual.state.position == expected.state.position == 12
    assert len(compile_calls) == sum(
        len(block.widths) for block in compiled.blocks
    )
    for actual_block, expected_block in zip(
        actual.state.blocks, expected.state.blocks, strict=True,
    ):
        for actual_state, expected_state in zip(
            actual_block.resonators, expected_block.resonators, strict=True,
        ):
            torch.testing.assert_close(actual_state.value, expected_state.value)
    compiled.disable_compiled_tensor_cores()
    assert all(not block._compiled_chunk_cores for block in compiled.blocks)


def test_stream_activation_checkpointing_preserves_outputs_and_gradients():
    torch.manual_seed(223)
    ordinary = MRRN(replace(tiny_config(), activation_checkpointing=False)).double().train()
    checkpointed = MRRN(replace(tiny_config(), activation_checkpointing=True)).double().train()
    checkpointed.load_state_dict(ordinary.state_dict())
    ordinary_input = torch.randn(1, 9, 3, dtype=torch.float64, requires_grad=True)
    checkpointed_input = ordinary_input.detach().clone().requires_grad_(True)

    def run(model, values):
        state = model.initial_stream_state(1, dtype=torch.float64)
        predictions = []
        for position in range(values.shape[1]):
            result = model.step(values[:, position], state)
            state = result.state
            predictions.append(result.prediction)
        output = torch.stack(predictions, 1)
        output.square().mean().backward()
        return output

    expected = run(ordinary, ordinary_input)
    actual = run(checkpointed, checkpointed_input)
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(checkpointed_input.grad, ordinary_input.grad, atol=2e-10, rtol=2e-10)
    expected_gradients = dict(ordinary.named_parameters())
    for name, parameter in checkpointed.named_parameters():
        reference = expected_gradients[name]
        if reference.grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, reference.grad, atol=2e-9, rtol=2e-9)


def test_stream_model_contracts_and_noncausal_rejection():
    noncausal = MRRN(tiny_config(causal=False))
    with pytest.raises(ValueError):
        noncausal.initial_stream_state(1)
    model = MRRN(tiny_config())
    state = model.initial_stream_state(2)
    with pytest.raises(ValueError):
        model.step(torch.randn(2, 2), state)
    with pytest.raises(ValueError):
        model.step(torch.randn(2, 3), state, torch.ones(2))
    with pytest.raises(ValueError):
        model.blocks[0].step((None,), state.blocks[0])


def test_efficiency_configuration_shares_depth_parameters_without_sharing_recurrent_state():
    configuration = replace(tiny_config(), layers=3, share_depth_parameters=True, structured_mixer_rank=2)
    model = MRRN(configuration)
    assert model.blocks[0] is model.blocks[1] is model.blocks[2]
    state = model.initial_stream_state(1)
    assert state.blocks[0] is not state.blocks[1]
    assert model.attention_schedules[0] != model.attention_schedules[1]
    output = model(torch.randn(1, 9, 3))
    assert output.prediction.shape == (1, 9, 5)


def test_scale_specific_frequency_limits_respect_each_decimated_band_nyquist_rate():
    model = MRRN(tiny_config())
    limits = [resonator.omega_max for resonator in model.blocks[0].resonators]
    assert limits == pytest.approx([torch.pi / 2, torch.pi / 4, torch.pi / 4])
    attention_limits = [attention.frequency_max for attention in model.blocks[0].attentions]
    assert attention_limits == pytest.approx(limits)


def test_continuous_signal_alias_control_is_active_and_stream_exact():
    torch.manual_seed(88)
    model = MRRN(replace(tiny_config(), continuous_signal=True)).double().eval()
    assert model.blocks[0].anti_aliases is not None
    x = torch.randn(1, 11, 3, dtype=torch.float64)
    with torch.no_grad():
        expected = model(x).prediction
        state = model.initial_stream_state(1, dtype=torch.float64)
        actual = []
        for position in range(x.shape[1]):
            result = model.step(x[:, position], state)
            state = result.state
            actual.append(result.prediction.unsqueeze(1))
    torch.testing.assert_close(torch.cat(actual, 1), expected, atol=2e-10, rtol=2e-10)


def test_bounded_eidetic_memory_is_integrated_into_scheduled_attention_and_stream_writes():
    torch.manual_seed(91)
    model = MRRN(tiny_config()).double().eval()
    memories = model.create_memories(1)
    for timestamp in (0, 2, 100):
        vector = torch.randn(8, dtype=torch.float64)
        memories[0].write(MemoryItem(vector, vector, vector, timestamp, 0, 1.0))
    x = torch.randn(1, 8, 3, dtype=torch.float64)
    with torch.no_grad():
        output = model(x, memories=memories)
        global_output = model(x, memories=memories, output_mode="global")
    assert global_output.prediction.shape == (1, 5)
    weights = output.diagnostics[0].attention_weights[0]
    assert weights is not None and weights.shape[2] == 4 + model.blocks[0].landmark_count + model.config.retrieved_items
    assert sum(item.use_count for item in memories[0].items()) > 0
    # The item from the future is excluded by the causal router at every earlier query.
    assert memories[0].get(memories[0].retrieve(torch.randn(8), 8)[-1]).timestamp in {0, 2, 100}

    model.memory_write_policy.linear.bias.data.fill_(20)
    state = model.initial_stream_state(1, dtype=torch.float64)
    before = len(memories[0])
    result = model.step(
        x[:, 0], state, memories=memories,
        write_features=torch.zeros(1, 5, dtype=torch.float64),
    )
    assert len(memories[0]) == before + 1 and result.state.position == 1


def test_memory_integration_contracts_fail_closed():
    model = MRRN(tiny_config())
    with pytest.raises(ValueError):
        model.create_memories(0)
    state = model.initial_stream_state(1)
    with pytest.raises(ValueError):
        model.step(torch.randn(1, 3), state, write_features=torch.zeros(1, 5))
    memories = model.create_memories(1)
    with pytest.raises(ValueError):
        model.write_memory_step(torch.randn(1, 7), torch.randn(1, 5), memories, timestamp=0)
    with pytest.raises(ValueError):
        model.write_memory_step(torch.randn(1, 8), torch.randn(1, 4), memories, timestamp=0)
    with pytest.raises(ValueError):
        model.write_memory_step(torch.randn(1, 8), torch.randn(1, 5), [], timestamp=-1)


def test_candidate_helper_and_attention_schedule_contracts_fail_closed():
    model = MRRN(tiny_config())
    raw, _ = model.analysis(model.encoder(torch.randn(1, 8, 3)))
    bands = model._adapt_analysis(raw)
    block = model.blocks[0]
    with pytest.raises(ValueError):
        _memory_candidates(
            bands[0], [], block.memory_keys[0], block.memory_signatures[0],
            block.memory_values[0], 1,
        )
    wrong_memory = EideticMemory(2, 3, 3, 3)
    with pytest.raises(ValueError):
        _memory_candidates(
            bands[0], [wrong_memory], block.memory_keys[0], block.memory_signatures[0],
            block.memory_values[0], 1,
        )
    with pytest.raises(ValueError):
        _memory_candidates(
            bands[0], model.create_memories(1), block.memory_keys[0], block.memory_signatures[0],
            block.memory_values[0], 1, absolute_positions=torch.ones(1, 2, dtype=torch.long),
        )
    candidate = AttentionCandidates(
        torch.randn(1, 2, 8), torch.zeros(1, 2), torch.zeros(1, 2),
        torch.ones(1, 2, dtype=torch.bool),
    )
    with pytest.raises(ValueError):
        _join_candidates(candidate, AttentionCandidates(
            torch.randn(2, 1, 8), torch.zeros(2, 1), torch.zeros(2, 1),
            torch.ones(2, 1, dtype=torch.bool),
        ))
    with pytest.raises(ValueError):
        _landmark_candidates(bands[0], bands[0], block.landmark_values[0], 0, causal=True)
    state = block.initial_stream_state(1)
    with pytest.raises(ValueError):
        block.step((None, None, None), state, attention_enabled=(True,))
    with pytest.raises(ValueError):
        block(tuple(bands), attention_enabled=(True,))


def test_stream_singleton_axis_soft_boundary_validation_and_empty_global_memory_branch():
    model = MRRN(tiny_config())
    state = model.initial_stream_state(1)
    assert model.step(torch.randn(1, 1, 3), state).prediction.shape == (1, 5)
    with pytest.raises(ValueError):
        model.step(torch.randn(1, 3), state, soft_boundary=torch.ones(1))
    empty = model.create_memories(1)
    output = model(torch.randn(1, 5, 3), memories=empty, output_mode="global")
    assert output.prediction.shape == (1, 5)
