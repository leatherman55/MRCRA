from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from mrrn.lm_training import (
    SentencePieceTextTokenizer,
    TextDocument,
    materialize_tokenizer_artifacts,
    tokenizer_from_identity,
)
from mrrn.tokenizer_training import (
    UnigramTokenizerTrainingConfig,
    _utf8_chunks,
    train_unigram_tokenizer,
)


@pytest.fixture(scope="module")
def trained_tokenizer(tmp_path_factory):
    root = tmp_path_factory.mktemp("sentencepiece")
    destination = root / "test-unigram.model"
    stem = (
        "Spectral resonance composes local evidence into global context while "
        "preserving provenance, uncertainty, exact reconstruction, and causality. "
    )
    documents = [
        TextDocument(
            f"document-{index}",
            (
                f"{stem} document={index} decimal={index:05d} hex={index:04x}; "
                f"variant-{index % 97} café 漢字 العربية 🙂\n"
            ),
        )
        for index in range(2_000)
    ]
    configuration = UnigramTokenizerTrainingConfig(
        vocabulary_size=320,
        maximum_documents=len(documents),
        maximum_utf8_bytes=16 << 20,
        number_of_threads=2,
    )
    manifest = train_unigram_tokenizer(
        documents, destination, configuration,
    )
    return destination, manifest


def test_unigram_training_produces_exact_vocabulary_and_provenance(
    trained_tokenizer,
):
    destination, manifest = trained_tokenizer
    tokenizer = SentencePieceTextTokenizer(destination)

    assert tokenizer.vocabulary_size == 320
    assert manifest["vocabulary_size"] == 320
    assert manifest["training_corpus"]["documents"] == 2_000
    assert len(manifest["training_corpus"]["sha256"]) == 64
    assert tokenizer.identity()["training_corpus"] == manifest["training_corpus"]
    assert tokenizer.pad_token_id == tokenizer.eos_token_id == 1
    assert tokenizer.forbidden_generation_token_ids == (0,)


@pytest.mark.parametrize(
    "text",
    (
        "",
        "ordinary English",
        "  repeated   whitespace  ",
        "tabs\tCRLF\r\nnewline\n",
        "café cafe\u0301 漢字 العربية",
        "🙂 🧬 👩🏽‍💻 ❤️",
        "\x00\u0001 controls",
        "code(x <<= 1); C:\\Users\\name /tmp/a b",
    ),
)
def test_unigram_encoding_is_deterministic_lossless_and_byte_exact(
    trained_tokenizer, text,
):
    destination, _ = trained_tokenizer
    tokenizer = SentencePieceTextTokenizer(destination)

    first = tokenizer.encode_document(text)
    second = tokenizer.encode_document(text)

    assert first == second
    assert tokenizer.unk_token_id not in first.token_ids[:-1]
    assert first.token_ids[-1] == tokenizer.eos_token_id
    assert first.byte_lengths[-1] == 0
    assert sum(first.byte_lengths) == len(text.encode("utf-8"))
    assert tokenizer.decode(first.token_ids[:-1]).encode("utf-8") == text.encode(
        "utf-8"
    )


def test_unigram_native_batch_encoding_is_scalar_exact(trained_tokenizer):
    destination, _ = trained_tokenizer
    tokenizer = SentencePieceTextTokenizer(destination)
    texts = (
        "short document",
        "Unicode 🙂 漢字 e\u0301",
        "spaces  tabs\tand\nnewlines",
        "\x00 byte fallback control",
    )
    expected = tuple(tokenizer.encode_document(text) for text in texts)
    actual = tokenizer.encode_documents(texts)
    assert actual == expected
    assert tuple(
        sum(document.byte_lengths) for document in actual
    ) == tuple(len(text.encode("utf-8")) for text in texts)


def test_unigram_manifest_and_checkpoint_identity_fail_closed(
    trained_tokenizer, tmp_path,
):
    destination, _ = trained_tokenizer
    manifest_path = destination.with_suffix(".json")
    identity = SentencePieceTextTokenizer(destination).identity()

    restored = tokenizer_from_identity(identity, search_roots=(destination.parent,))
    assert restored.identity() == identity

    corrupt_manifest = tmp_path / manifest_path.name
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["model_sha256"] = "0" * 64
    corrupt_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        SentencePieceTextTokenizer(destination, manifest_path=corrupt_manifest)

    changed_identity = deepcopy(identity)
    changed_identity["model_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differs"):
        tokenizer_from_identity(
            changed_identity, search_roots=(destination.parent,),
        )


def test_run_artifact_materialization_is_self_contained_and_collision_safe(
    trained_tokenizer, tmp_path,
):
    destination, _ = trained_tokenizer
    tokenizer = SentencePieceTextTokenizer(destination)
    copied = materialize_tokenizer_artifacts(tokenizer, tmp_path)
    assert {path.name for path in copied} == {
        destination.name,
        destination.with_suffix(".json").name,
        destination.with_suffix(".vocabulary.json").name,
    }
    restored = tokenizer_from_identity(
        tokenizer.identity(), search_roots=(tmp_path,),
    )
    assert restored.identity() == tokenizer.identity()
    copied[0].write_bytes(b"different")
    with pytest.raises(ValueError, match="different tokenizer"):
        materialize_tokenizer_artifacts(tokenizer, tmp_path)


def test_utf8_chunking_preserves_scalar_boundaries_and_all_bytes():
    text = "ab🙂cd漢字ef"
    chunks = tuple(_utf8_chunks(text, 5))
    assert "".join(chunks) == text
    assert all(len(chunk.encode("utf-8")) <= 5 for chunk in chunks)


@pytest.mark.parametrize(
    "overrides",
    (
        {"vocabulary_size": 257},
        {"maximum_documents": 0},
        {"character_coverage": 0.0},
        {"shrinking_factor": 1.0},
        {"partition": "eval"},
    ),
)
def test_unigram_training_configuration_rejects_unsafe_contracts(overrides):
    with pytest.raises(ValueError):
        UnigramTokenizerTrainingConfig(**overrides)
