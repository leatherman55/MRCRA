import pytest
import torch

from mrrn import complex_ops as c
from mrrn.attention import AttentionCandidates, DotProductCandidateAttention, ResonantAttention, linear_cross_correlation


def setup_attention(dtype=torch.float64):
    torch.manual_seed(11)
    module = ResonantAttention(6, 2, 3).to(dtype=dtype)
    query = torch.randn(2, 4, 6, dtype=dtype)
    features = torch.randn(2, 7, 6, dtype=dtype)
    times = torch.arange(7, dtype=dtype).repeat(2, 1)
    candidates = AttentionCandidates(
        features, times, torch.zeros(2, 7, dtype=dtype), torch.ones(2, 7, dtype=torch.bool)
    )
    query_times = torch.tensor([[3, 4, 5, 6], [3, 4, 5, 6]], dtype=dtype)
    query_scales = torch.zeros(2, 4, dtype=dtype)
    return module, query, candidates, query_times, query_scales


def test_scores_match_direct_native_complex_formula():
    module, query, candidates, query_times, query_scales = setup_attention()
    scores = module.scores(query, candidates, query_times, query_scales)
    q = c.to_native(module._project(module.query_projection, query))
    k = c.to_native(module._project(module.key_projection, candidates.features))
    delta = query_times.unsqueeze(2) - candidates.times.unsqueeze(1)
    frequency = torch.pi * torch.tanh(module.raw_frequency)
    cross = q.unsqueeze(2) * k.unsqueeze(1).conj()
    cross /= (
        (q.abs().square().sum(-1) + module.eps).sqrt().unsqueeze(2)
        * (k.abs().square().sum(-1) + module.eps).sqrt().unsqueeze(1)
    ).unsqueeze(-1)
    aligned = cross * torch.exp(-1j * delta[..., None, None] * frequency)
    coherence = (aligned.real * module.band_logits.softmax(-1)).sum(-1) / 3**0.5
    amplitude = (q.abs().unsqueeze(2) * k.abs().unsqueeze(1)).sum(-1)
    expected = coherence + torch.nn.functional.softplus(module.raw_amplitude_weight) * torch.log(
        module.eps + amplitude
    )
    expected -= torch.nn.functional.softplus(module.raw_distance_decay) * torch.log1p(delta.abs()).unsqueeze(-1)
    expected -= torch.nn.functional.softplus(module.raw_scale_decay) * 0
    expected = expected.masked_fill((delta < 0).unsqueeze(-1), -torch.inf)
    torch.testing.assert_close(scores, expected)


def test_full_tiled_and_bandwise_attention_contracts():
    module, query, candidates, times, scales = setup_attention()
    full, weights = module.attend(query, candidates, times, scales)
    tiled = module.tiled_attend(query, candidates, times, scales, tile_size=3)
    bandwise, band_weights = module.attend(query, candidates, times, scales, bandwise=True)
    torch.testing.assert_close(tiled, full, atol=2e-12, rtol=2e-12)
    assert full.shape == bandwise.shape == query.shape
    assert weights.shape == (2, 4, 7, 2)
    assert band_weights.shape == (2, 4, 7, 2, 3)
    causal_valid = (times.unsqueeze(2) - candidates.times.unsqueeze(1)) >= 0
    torch.testing.assert_close((weights * causal_valid.unsqueeze(-1)).sum(2), torch.ones(2, 4, 2, dtype=query.dtype))


def test_query_tiled_sliding_attention_matches_individual_queries_with_extra_candidates():
    """Query tiling must be an execution detail even with retrieved context."""

    torch.manual_seed(13)
    module = ResonantAttention(6, 2, 3).double()
    features = torch.randn(1, 7, 6, dtype=torch.float64)
    mask = torch.tensor([[True, True, True, False, True, True, True]])
    extra_count = 2
    extra = AttentionCandidates(
        torch.randn(7, extra_count, 6, dtype=torch.float64),
        torch.tensor([[0.0, 1.0]] * 7, dtype=torch.float64),
        torch.ones(7, extra_count, dtype=torch.float64),
        torch.tensor([[True, False], [True, True], [False, True], [True, True],
                      [True, True], [True, False], [True, True]]),
        torch.ones(7, extra_count, dtype=torch.int64),
    )
    tiled, _ = module.sliding_window_attend(
        features, mask, window=3, query_tile_size=2, scale=0,
        sample_interval=1.0, coefficient_interval=1.0,
        additional_candidates=extra,
    )
    expected = []
    for position in range(features.shape[1]):
        local_indices = torch.arange(position - 2, position + 1)
        valid_index = (local_indices >= 0) & (local_indices < features.shape[1])
        safe = local_indices.clamp(0, features.shape[1] - 1)
        candidates = AttentionCandidates(
            torch.cat((features[:, safe], extra.features[position : position + 1]), 1),
            torch.cat((safe.to(features.dtype)[None], extra.times[position : position + 1]), 1),
            torch.cat((torch.zeros(1, 3, dtype=features.dtype), extra.scales[position : position + 1]), 1),
            torch.cat(((mask[:, safe] & valid_index), extra.mask[position : position + 1]), 1),
        )
        value, _ = module.attend(
            features[:, position : position + 1], candidates,
            torch.tensor([[float(position)]], dtype=features.dtype),
            torch.zeros(1, 1, dtype=features.dtype),
        )
        expected.append(value)
    torch.testing.assert_close(tiled, torch.cat(expected, 1), atol=2e-12, rtol=2e-12)


def test_sliding_attention_history_exactly_continues_a_causal_chunk():
    torch.manual_seed(319)
    module = ResonantAttention(8, 2, 3).double().eval()
    for parameter in module.parameters():
        parameter.data.normal_(std=0.08)
    features = torch.randn(2, 19, 8, dtype=torch.float64)
    mask = torch.rand(2, 19) > 0.1
    window, split = 6, 8
    complete, _ = module.sliding_window_attend(
        features, mask, window=window, query_tile_size=4, scale=1,
        sample_interval=0.5, coefficient_interval=1.0, causal=True,
        return_weights=False,
    )
    first, _ = module.sliding_window_attend(
        features[:, :split], mask[:, :split], window=window,
        query_tile_size=3, scale=1, sample_interval=0.5,
        coefficient_interval=1.0, causal=True, return_weights=False,
    )
    history_start = max(0, split - window)
    history_indices = torch.arange(
        history_start, split, dtype=features.dtype
    )
    history_times = (
        (history_indices + 1) * 1.0 - 0.5
    ).expand(features.shape[0], -1)
    history = AttentionCandidates(
        features[:, history_start:split],
        history_times,
        torch.full_like(history_times, 1.0),
        mask[:, history_start:split],
    )
    second, _ = module.sliding_window_attend(
        features[:, split:], mask[:, split:], window=window,
        query_tile_size=5, scale=1, sample_interval=0.5,
        coefficient_interval=1.0, causal=True, history=history,
        time_offset=float(split), return_weights=False,
    )
    torch.testing.assert_close(
        torch.cat((first, second), 1), complete, atol=2e-12, rtol=2e-12,
    )


def test_invalid_and_all_masked_candidates_receive_exactly_zero_weight():
    module, query, candidates, times, scales = setup_attention()
    masked = AttentionCandidates(
        candidates.features, candidates.times, candidates.scales, torch.zeros_like(candidates.mask)
    )
    output, weights = module.attend(query, masked, times, scales)
    assert (weights == 0).all()
    torch.testing.assert_close(output, module.output_projection.bias.expand_as(output))


def test_linear_cross_correlation_matches_direct_all_lags_and_finds_delay():
    query = torch.tensor([[0.0, 1.0, 2.0, 0.0, -1.0]])
    key = torch.tensor([[1.0, 2.0, 0.0]])
    actual, lags = linear_cross_correlation(query, key)
    expected = []
    for lag in lags.tolist():
        total = 0.0
        for index in range(query.shape[-1]):
            key_index = index - lag
            if 0 <= key_index < key.shape[-1]:
                total += query[0, index].item() * key[0, key_index].item()
        expected.append(total)
    torch.testing.assert_close(actual[0], torch.tensor(expected), atol=1e-6, rtol=1e-6)
    assert lags[actual[0].argmax()] == 1


def test_relative_phase_compensation_selects_a_known_shift_over_spectral_collision():
    module = ResonantAttention(2, 1, 1).double()
    for projection in (module.query_projection, module.key_projection):
        projection.weight.data.copy_(torch.eye(2, dtype=torch.float64))
        projection.bias.data.zero_()
    frequency, delay = 0.7, 5.0
    module.raw_frequency.data.fill_(torch.atanh(torch.tensor(frequency / torch.pi, dtype=torch.float64)))
    query = torch.tensor([[[torch.cos(torch.tensor(frequency * delay)), torch.sin(torch.tensor(frequency * delay))]]], dtype=torch.float64)
    candidates = AttentionCandidates(
        torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]], dtype=torch.float64),
        torch.zeros(1, 2, dtype=torch.float64), torch.zeros(1, 2, dtype=torch.float64),
        torch.ones(1, 2, dtype=torch.bool),
    )
    scores = module.scores(query, candidates, torch.tensor([[delay]], dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64))
    assert scores[0, 0, :, 0].argmax() == 0


def test_dot_product_ablation_uses_identical_candidates_masks_and_causal_contract():
    _, query, candidates, times, scales = setup_attention(dtype=torch.float32)
    module = DotProductCandidateAttention(6, 2)
    output, weights = module.attend(query, candidates, times, scales)
    assert output.shape == query.shape and weights.shape == (2, 4, 7, 2)
    invalid = (times.unsqueeze(2) - candidates.times.unsqueeze(1)) < 0
    assert (weights.masked_select(invalid.unsqueeze(-1)) == 0).all()
    with pytest.raises(ValueError):
        DotProductCandidateAttention(5, 2)


def test_attention_gradients_are_finite():
    module, query, candidates, times, scales = setup_attention()
    query.requires_grad_()
    output, _ = module.attend(query, candidates, times, scales, causal=False)
    output.square().mean().backward()
    assert torch.isfinite(query.grad).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in module.parameters())


def test_attention_contract_validation_and_empty_correlation():
    module, query, candidates, times, scales = setup_attention()
    bad = AttentionCandidates(candidates.features[..., :5], candidates.times, candidates.scales, candidates.mask)
    with pytest.raises(ValueError):
        module.attend(query, bad, times, scales)
    with pytest.raises(ValueError):
        module.attend(query[..., :5], candidates, times, scales)
    with pytest.raises(ValueError):
        module.attend(query, candidates, times[:, :2], scales)
    with pytest.raises(ValueError):
        module.tiled_attend(query, candidates, times, scales, tile_size=0)
    with pytest.raises(ValueError):
        linear_cross_correlation(torch.empty(0), torch.ones(2))
    with pytest.raises(ValueError):
        linear_cross_correlation(torch.ones(2, 3), torch.ones(3, 3))
    with pytest.raises(ValueError):
        ResonantAttention(0, 1, 1)
    with pytest.raises(ValueError):
        ResonantAttention(1, 1, 1, frequency_max=torch.pi + 0.1)


def test_candidate_metadata_validation_branches():
    features = torch.randn(1, 2, 3)
    base = dict(times=torch.zeros(1, 2), scales=torch.zeros(1, 2), mask=torch.ones(1, 2, dtype=torch.bool))
    AttentionCandidates(features, **base).validate(3)
    with pytest.raises(ValueError):
        AttentionCandidates(features, **base).validate(4)
    with pytest.raises(ValueError):
        AttentionCandidates(features, base["times"][:, :1], base["scales"], base["mask"]).validate(3)
    with pytest.raises(ValueError):
        AttentionCandidates(features, base["times"], base["scales"], base["mask"].float()).validate(3)
    with pytest.raises(ValueError):
        AttentionCandidates(features, **base, kinds=torch.zeros(1, 1)).validate(3)
