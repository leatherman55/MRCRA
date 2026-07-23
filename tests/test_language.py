import math

import pytest
import torch

from mrrn.language import (
    MRRNLanguageModel,
    fineweb_4p7m_config,
    fineweb_27m_config,
    tiny_language_config,
)
from mrrn.lm_training import (
    ByteTextTokenizer,
    PackedTokenStream,
    SequenceTextSource,
    architecture_metrics,
    next_token_statistics,
)


def test_fineweb_model_is_balanced_approximately_4p7m_sequence_only_and_tied():
    model = MRRNLanguageModel(fineweb_4p7m_config())
    assert 4_600_000 <= model.parameter_count <= 4_800_000
    assert model.parameter_count == model.trainable_parameter_count == 4_695_023
    assert model.token_embedding.weight is model.actor.output_head.weight
    assert model.actor.global_head is None
    assert model.config.decay_normalized_resonance
    assert tuple(scale.width for scale in model.config.scale_configs()) == (48, 64, 64, 64, 64)
    assert tuple(scale.modes for scale in model.config.scale_configs()) == (10, 12, 12, 12, 12)
    assert model.config.spectral_modes == 8


def test_legacy_fineweb_27m_configuration_remains_checkpoint_addressable():
    model = MRRNLanguageModel(fineweb_27m_config())
    assert model.parameter_count == 26_439_515


def test_language_forward_generation_and_contracts():
    torch.manual_seed(4)
    tokenizer = ByteTextTokenizer()
    model = MRRNLanguageModel(tiny_language_config(tokenizer.vocabulary_size)).eval()
    tokens = torch.tensor([[65, 66, 67]], dtype=torch.long)
    output = model(tokens)
    assert output.logits.shape == (1, 3, tokenizer.vocabulary_size)
    metrics = architecture_metrics(output)
    assert sum(metrics[f"architecture/scale_{i}_energy_fraction"] for i in range(3)) == pytest.approx(1)
    generated = model.generate(tokens, maximum_new_tokens=2, temperature=0, eos_token_id=None)
    assert generated.shape == (1, 5)
    with pytest.raises(ValueError, match="outside"):
        model(torch.tensor([[tokenizer.vocabulary_size]]))
    with pytest.raises(ValueError, match="nonempty"):
        model.generate(torch.empty(1, 0, dtype=torch.long), maximum_new_tokens=1)


def test_byte_tokenizer_packing_preserves_every_transition_and_resumes_exactly():
    tokenizer = ByteTextTokenizer()
    source = SequenceTextSource(("abcd", "EF"))
    stream = PackedTokenStream(source, tokenizer)
    first = stream.next_batch(1, 4)
    assert first.input_ids.tolist() == [[97, 98, 99, 100]]
    assert first.labels.tolist() == [[98, 99, 100, tokenizer.eos_token_id]]
    state = stream.state_dict()
    expected = stream.next_batch(1, 4)
    restored = PackedTokenStream(SequenceTextSource(("abcd", "EF")), tokenizer)
    restored.load_state_dict(state)
    actual = restored.next_batch(1, 4)
    torch.testing.assert_close(actual.input_ids, expected.input_ids)
    torch.testing.assert_close(actual.labels, expected.labels)
    torch.testing.assert_close(actual.target_byte_lengths, expected.target_byte_lengths)


def test_effective_cross_entropy_is_nll_per_original_utf8_byte():
    vocabulary = 7
    logits = torch.zeros(1, 3, vocabulary)
    labels = torch.tensor([[1, 2, 3]])
    bytes_per_target = torch.tensor([[1, 2, 0]])
    statistics = next_token_statistics(logits, labels, bytes_per_target)
    assert float(statistics.cross_entropy) == pytest.approx(math.log(vocabulary))
    assert float(statistics.effective_cross_entropy) == pytest.approx(math.log(vocabulary))
    assert statistics.token_count == 3 and statistics.byte_count == 3
    assert statistics.correct_top1 == 0


def test_language_configuration_rejects_noncausal_global_or_untied_shapes():
    tokenizer = ByteTextTokenizer()
    base = tiny_language_config(tokenizer.vocabulary_size)
    from dataclasses import replace

    with pytest.raises(ValueError, match="causal"):
        MRRNLanguageModel(replace(base, causal=False))
    with pytest.raises(ValueError, match="global"):
        MRRNLanguageModel(replace(base, enable_global_head=True))
    with pytest.raises(ValueError, match="input_dim"):
        MRRNLanguageModel(replace(base, input_dim=8))
