#!/usr/bin/env python3
"""Compare canonical Unigram-24K and GPT-2 on identical held-out FineWeb bytes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from huggingface_hub import HfApi

from mrrn.lm_training import FineWebTextSource, load_text_tokenizer
from mrrn.tokenizer_evaluation import (
    evaluate_tokenizers,
    source_equivalent_clock_report,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--dataset-id", default="HuggingFaceFW/fineweb")
    result.add_argument("--dataset-config", default="sample-10BT")
    result.add_argument("--dataset-revision", default="main")
    result.add_argument("--maximum-documents", type=int, default=2_000)
    result.add_argument("--output", type=Path, default=Path(
        "outputs/tokenizer_comparison.json"
    ))
    return result


def main() -> None:
    args = parser().parse_args()
    information = HfApi().dataset_info(
        args.dataset_id, revision=args.dataset_revision,
    )
    if not information.sha:
        raise RuntimeError("Hugging Face did not return an immutable dataset SHA")
    source = FineWebTextSource(
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        revision=information.sha,
        partition="eval",
    )
    tokenizers = {
        "mrcra_unigram_24k": load_text_tokenizer(),
        "gpt2_control": load_text_tokenizer("gpt2"),
    }
    metrics = evaluate_tokenizers(
        source, tokenizers, maximum_documents=args.maximum_documents,
    )
    clock_report = source_equivalent_clock_report(
        metrics,
        baseline="gpt2_control",
        candidate="mrcra_unigram_24k",
        clocks={
            "optimization_context": 32_768,
            "carrier_tbptt": 4_096,
            "carrier_execution_chunk": 256,
            "local_attention_window": 32,
            "ultralight_event_chunk": 64,
            "ultralight_cognitive_stride": 128,
            "light_event_chunk": 128,
            "serious_event_chunk": 256,
        },
        alignment=32,
    )
    payload = {
        "artifact": "MRCRA tokenizer source-byte comparison",
        "dataset": {
            "id": args.dataset_id,
            "config": args.dataset_config,
            "revision": information.sha,
            "partition": "eval",
            "maximum_documents": args.maximum_documents,
        },
        "tokenizers": {
            name: tokenizer.identity() for name, tokenizer in tokenizers.items()
        },
        "metrics": metrics,
        "clock_report": clock_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if "pyarrow" not in sys.modules:
            raise
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    if "pyarrow" in sys.modules:
        # PyArrow 25 can deadlock in its global thread-pool destructor on
        # macOS after a streaming dataset has been consumed.  The JSON
        # evidence is already durably closed before this process-level exit.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
