"""Reproducible training authority for the canonical MRCRA tokenizer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Iterator

from .lm_training import TextDocument


@dataclass(frozen=True, slots=True)
class UnigramTokenizerTrainingConfig:
    """Immutable corpus and SentencePiece construction contract."""

    vocabulary_size: int = 24_576
    maximum_documents: int = 250_000
    maximum_utf8_bytes: int = 512 << 20
    maximum_sentence_bytes: int = 16_384
    character_coverage: float = 0.9995
    seed_sentencepiece_size: int = 1_000_000
    shrinking_factor: float = 0.75
    number_of_threads: int = 8
    dataset_id: str = "HuggingFaceFW/fineweb"
    dataset_config: str = "sample-10BT"
    dataset_split: str = "train"
    dataset_revision: str = "main"
    partition: str = "train"
    evaluation_fraction_permyriad: int = 100
    shuffle_seed: int = 20260721
    shuffle_buffer: int = 10_000

    def __post_init__(self) -> None:
        if self.vocabulary_size < 258:
            raise ValueError("Unigram vocabulary must leave room for byte fallback")
        if min(
            self.maximum_documents,
            self.maximum_utf8_bytes,
            self.maximum_sentence_bytes,
            self.seed_sentencepiece_size,
            self.number_of_threads,
            self.shuffle_buffer,
        ) <= 0:
            raise ValueError("tokenizer training limits must be positive")
        if not 0 < self.character_coverage <= 1:
            raise ValueError("character coverage must lie in (0,1]")
        if not 0 < self.shrinking_factor < 1:
            raise ValueError("shrinking factor must lie in (0,1)")
        if self.partition != "train":
            raise ValueError("the tokenizer may only learn from the training partition")


def _utf8_chunks(text: str, maximum_bytes: int) -> Iterator[str]:
    """Split without normalization while preserving every Unicode scalar."""

    if not text:
        return
    start = 0
    current_bytes = 0
    for index, character in enumerate(text):
        width = len(character.encode("utf-8"))
        if current_bytes and current_bytes + width > maximum_bytes:
            yield text[start:index]
            start = index
            current_bytes = 0
        current_bytes += width
    if start < len(text):
        yield text[start:]


def _framed_digest_update(digest, identifier: str, raw: bytes) -> None:
    identifier_bytes = identifier.encode("utf-8")
    digest.update(len(identifier_bytes).to_bytes(8, "big"))
    digest.update(identifier_bytes)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)


def _lossless_validation_cases() -> tuple[str, ...]:
    return (
        "",
        "ordinary English and punctuation: isn't, won't, em-dash—done.",
        "  leading  internal   and trailing  ",
        "tabs\tCRLF\r\nnewlines\n",
        "café vs cafe\u0301; Ελληνικά; العربية; 漢字; 한글",
        "emoji 🙂 🧬 👩🏽‍💻 and variation selectors ❤️",
        "code: for (int i=0; i<10; ++i) { x <<= 1; }\n",
        "paths C:\\Users\\name and /tmp/a b; JSON {\"x\": null}",
        "\x00 embedded NUL and \u0001 controls",
    )


def train_unigram_tokenizer(
    documents: Iterable[TextDocument],
    destination_model: str | Path,
    config: UnigramTokenizerTrainingConfig,
) -> dict:
    """Train, validate, and atomically retain one lossless Unigram artifact."""

    try:
        import sentencepiece as sentencepiece
    except ImportError as error:
        raise RuntimeError("SentencePiece is required to train the tokenizer") from error

    destination_model = Path(destination_model).resolve()
    if destination_model.suffix != ".model":
        raise ValueError("tokenizer destination must end in .model")
    destination_model.parent.mkdir(parents=True, exist_ok=True)
    destination_vocab = destination_model.with_name(
        destination_model.stem + ".vocabulary.json"
    )
    destination_manifest = destination_model.with_suffix(".json")

    corpus_digest = sha256()
    counters = {"documents": 0, "utf8_bytes": 0, "sentences": 0}
    retained_validation: list[str] = []

    def sentences() -> Iterator[str]:
        for document in documents:
            if counters["documents"] >= config.maximum_documents:
                break
            raw = document.text.encode("utf-8")
            if (
                counters["documents"] > 0
                and counters["utf8_bytes"] + len(raw) > config.maximum_utf8_bytes
            ):
                break
            _framed_digest_update(corpus_digest, document.identifier, raw)
            counters["documents"] += 1
            counters["utf8_bytes"] += len(raw)
            if len(retained_validation) < 128:
                retained_validation.append(document.text)
            for chunk in _utf8_chunks(document.text, config.maximum_sentence_bytes):
                counters["sentences"] += 1
                yield chunk

    with tempfile.TemporaryDirectory(
        prefix="mrcra-unigram-training-", dir=destination_model.parent,
    ) as temporary_directory:
        prefix = Path(temporary_directory) / destination_model.stem
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=sentences(),
            model_prefix=str(prefix),
            model_type="unigram",
            vocab_size=config.vocabulary_size,
            hard_vocab_limit=True,
            byte_fallback=True,
            normalization_rule_name="identity",
            add_dummy_prefix=False,
            remove_extra_whitespaces=False,
            escape_whitespaces=True,
            character_coverage=config.character_coverage,
            seed_sentencepiece_size=config.seed_sentencepiece_size,
            shrinking_factor=config.shrinking_factor,
            shuffle_input_sentence=False,
            num_threads=config.number_of_threads,
            max_sentence_length=config.maximum_sentence_bytes,
            split_by_number=True,
            split_digits=True,
            unk_id=0,
            unk_piece="<|unknown|>",
            eos_id=1,
            eos_piece="<|document_end|>",
            bos_id=-1,
            pad_id=-1,
            minloglevel=1,
        )
        temporary_model = prefix.with_suffix(".model")
        if counters["documents"] <= 0 or counters["utf8_bytes"] <= 0:
            raise ValueError("tokenizer training corpus was empty")

        model_bytes = temporary_model.read_bytes()
        processor = sentencepiece.SentencePieceProcessor(model_proto=model_bytes)
        if processor.vocab_size() != config.vocabulary_size:
            raise ValueError("SentencePiece did not produce the exact vocabulary")
        cases = _lossless_validation_cases() + tuple(retained_validation)
        for text in cases:
            identifiers = processor.encode(text, out_type=int)
            if processor.unk_id() in identifiers:
                raise ValueError("byte fallback failed to eliminate unknown tokens")
            if processor.decode(identifiers).encode("utf-8") != text.encode("utf-8"):
                raise ValueError("trained tokenizer is not byte-exact and lossless")

        vocabulary = [
            {
                "id": identifier,
                "piece": processor.id_to_piece(identifier),
                "score": processor.get_score(identifier),
                "is_unknown": identifier == processor.unk_id(),
                "is_control": processor.is_control(identifier),
                "is_byte": processor.is_byte(identifier),
            }
            for identifier in range(processor.vocab_size())
        ]
        vocab_bytes = (
            json.dumps(
                vocabulary, ensure_ascii=True, separators=(",", ":"),
                allow_nan=False,
            ) + "\n"
        ).encode("utf-8")
        manifest = {
            "schema_version": 1,
            "artifact": "MRCRA deterministic lossless SentencePiece Unigram tokenizer",
            "vocabulary_size": int(processor.vocab_size()),
            "model_sha256": sha256(model_bytes).hexdigest(),
            "vocabulary_artifact": destination_vocab.name,
            "vocabulary_sha256": sha256(vocab_bytes).hexdigest(),
            "special_token_ids": {
                "unknown": int(processor.unk_id()),
                "document_end": int(processor.eos_id()),
            },
            "training_corpus": {
                "sha256": corpus_digest.hexdigest(),
                "documents": counters["documents"],
                "utf8_bytes": counters["utf8_bytes"],
                "sentences": counters["sentences"],
                "dataset_id": config.dataset_id,
                "dataset_config": config.dataset_config,
                "dataset_split": config.dataset_split,
                "dataset_revision": config.dataset_revision,
                "partition": config.partition,
                "evaluation_fraction_permyriad": (
                    config.evaluation_fraction_permyriad
                ),
                "shuffle_seed": config.shuffle_seed,
                "shuffle_buffer": config.shuffle_buffer,
            },
            "trainer": {
                **asdict(config),
                "model_type": "unigram",
                "byte_fallback": True,
                "normalization_rule_name": "identity",
                "add_dummy_prefix": False,
                "remove_extra_whitespaces": False,
                "split_digits": True,
                "bos_id": -1,
                "eos_id": 1,
                "pad_id": -1,
                "unk_id": 0,
            },
            "validation": {
                "byte_exact_round_trip_cases": len(cases),
                "unknown_tokens_emitted": 0,
            },
        }
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        for source, destination in ((temporary_model, destination_model),):
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(source.read_bytes())
            os.replace(temporary, destination)
        temporary_vocab_json = destination_vocab.with_suffix(".json.tmp")
        temporary_vocab_json.write_bytes(vocab_bytes)
        os.replace(temporary_vocab_json, destination_vocab)
        temporary_manifest = destination_manifest.with_suffix(".json.tmp")
        temporary_manifest.write_bytes(manifest_bytes)
        os.replace(temporary_manifest, destination_manifest)
    return manifest
