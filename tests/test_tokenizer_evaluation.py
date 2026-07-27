from __future__ import annotations

import pytest

from mrrn.lm_training import ByteTextTokenizer, TextDocument
from mrrn.tokenizer_evaluation import (
    evaluate_tokenizers,
    source_equivalent_clock_report,
)


def test_tokenizer_evaluation_uses_identical_source_bytes_and_excludes_eos():
    documents = (
        TextDocument("a", "abc"),
        TextDocument("b", "é🙂"),
    )
    metrics = evaluate_tokenizers(
        documents, {"bytes": ByteTextTokenizer()}, maximum_documents=2,
    )
    row = metrics["bytes"]
    assert row["documents"] == 2
    assert row["utf8_bytes"] == 9
    assert row["content_tokens"] == 9
    assert row["document_end_tokens"] == 2
    assert row["bytes_per_content_token"] == 1


def test_clock_report_is_explicit_aligned_and_non_authoritative():
    metrics = {
        "baseline": {"content_tokens_per_kibibyte": 256.0},
        "candidate": {"content_tokens_per_kibibyte": 320.0},
    }
    report = source_equivalent_clock_report(
        metrics,
        baseline="baseline",
        candidate="candidate",
        clocks={"context": 100, "window": 32},
        alignment=16,
    )
    assert report["candidate_tokens_per_baseline_token"] == 1.25
    assert report["source_equivalent_candidate_clocks"] == {
        "context": 128,
        "window": 48,
    }
    assert "report only" in report["authority"]


@pytest.mark.parametrize(
    "call",
    (
        lambda: evaluate_tokenizers((), {}, maximum_documents=1),
        lambda: evaluate_tokenizers(
            (), {"bytes": ByteTextTokenizer()}, maximum_documents=0,
        ),
        lambda: source_equivalent_clock_report(
            {}, baseline="x", candidate="y", clocks={"context": 1},
        ),
        lambda: source_equivalent_clock_report(
            {
                "x": {"content_tokens_per_kibibyte": 1.0},
                "y": {"content_tokens_per_kibibyte": 1.0},
            },
            baseline="x",
            candidate="y",
            clocks={"context": 0},
        ),
    ),
)
def test_tokenizer_comparison_rejects_invalid_contracts(call):
    with pytest.raises(ValueError):
        call()
