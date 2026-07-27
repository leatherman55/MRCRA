"""Source-byte-normalized tokenizer comparison and clock calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from itertools import islice
from typing import Iterable, Mapping

from .lm_training import TextDocument, TextTokenizer


@dataclass(frozen=True, slots=True)
class TokenizerCorpusMetrics:
    documents: int
    utf8_bytes: int
    content_tokens: int
    document_end_tokens: int
    bytes_per_content_token: float
    content_tokens_per_kibibyte: float
    encoding_seconds: float
    utf8_bytes_per_second: float
    content_tokens_per_second: float


def evaluate_tokenizers(
    documents: Iterable[TextDocument],
    tokenizers: Mapping[str, TextTokenizer],
    *,
    maximum_documents: int,
) -> dict:
    """Measure every tokenizer on exactly the same immutable source documents."""

    if not tokenizers or maximum_documents <= 0:
        raise ValueError("tokenizer evaluation requires candidates and documents")
    counters = {
        name: {
            "documents": 0,
            "utf8_bytes": 0,
            "content_tokens": 0,
            "document_end_tokens": 0,
            "encoding_seconds": 0.0,
        }
        for name in tokenizers
    }
    retained_documents = tuple(islice(documents, maximum_documents))
    for name, tokenizer in tokenizers.items():
        batch_encoder = getattr(tokenizer, "encode_documents", None)
        started = perf_counter()
        encoded_documents = (
            tuple(
                tokenizer.encode_document(document.text)
                for document in retained_documents
            )
            if batch_encoder is None
            else tuple(
                batch_encoder(
                    tuple(
                        document.text
                        for document in retained_documents
                    )
                )
            )
        )
        encoding_seconds = perf_counter() - started
        if len(encoded_documents) != len(retained_documents):
            raise ValueError(
                f"{name} changed tokenizer-evaluation document cardinality"
            )
        counters[name]["encoding_seconds"] = encoding_seconds
        for document, encoded in zip(
            retained_documents, encoded_documents, strict=True,
        ):
            raw_bytes = len(document.text.encode("utf-8"))
            if (
                encoded.token_ids[-1] != tokenizer.eos_token_id
                or encoded.byte_lengths[-1] != 0
                or sum(encoded.byte_lengths) != raw_bytes
            ):
                raise ValueError(
                    f"{name} violated document-boundary or byte accounting"
                )
            row = counters[name]
            row["documents"] += 1
            row["utf8_bytes"] += raw_bytes
            row["content_tokens"] += len(encoded.token_ids) - 1
            row["document_end_tokens"] += 1
    if not counters or min(row["documents"] for row in counters.values()) <= 0:
        raise ValueError("tokenizer evaluation source was empty")

    metrics = {}
    for name, row in counters.items():
        content_tokens = row["content_tokens"]
        if content_tokens <= 0:
            raise ValueError("tokenizer evaluation produced no content tokens")
        value = TokenizerCorpusMetrics(
            **row,
            bytes_per_content_token=row["utf8_bytes"] / content_tokens,
            content_tokens_per_kibibyte=content_tokens * 1024 / row["utf8_bytes"],
            utf8_bytes_per_second=(
                row["utf8_bytes"]
                / max(float(row["encoding_seconds"]), 1e-12)
            ),
            content_tokens_per_second=(
                content_tokens
                / max(float(row["encoding_seconds"]), 1e-12)
            ),
        )
        metrics[name] = asdict(value)
    return metrics


def source_equivalent_clock_report(
    metrics: Mapping[str, Mapping[str, float | int]],
    *,
    baseline: str,
    candidate: str,
    clocks: Mapping[str, int],
    alignment: int = 1,
) -> dict:
    """Report—not silently apply—token clocks matching baseline byte support."""

    if baseline not in metrics or candidate not in metrics or alignment <= 0:
        raise ValueError("clock calibration names or alignment are invalid")
    baseline_rate = float(metrics[baseline]["content_tokens_per_kibibyte"])
    candidate_rate = float(metrics[candidate]["content_tokens_per_kibibyte"])
    ratio = candidate_rate / baseline_rate
    resolved = {}
    for name, value in clocks.items():
        if value <= 0:
            raise ValueError("clock values must be positive")
        raw = value * ratio
        resolved[name] = int((raw + alignment - 1) // alignment * alignment)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "candidate_tokens_per_baseline_token": ratio,
        "alignment": alignment,
        "source_equivalent_candidate_clocks": resolved,
        "authority": (
            "measurement report only; training clocks change only through an "
            "explicit checkpointed configuration"
        ),
    }
