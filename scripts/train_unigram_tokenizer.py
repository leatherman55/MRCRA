#!/usr/bin/env python3
"""Train the canonical lossless 24,576-entry MRCRA tokenizer on FineWeb."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from huggingface_hub import HfApi

from mrrn.lm_training import FineWebTextSource, canonical_sentencepiece_paths
from mrrn.tokenizer_training import (
    UnigramTokenizerTrainingConfig,
    train_unigram_tokenizer,
)


def _dataset_revision(dataset_id: str, revision: str) -> str:
    information = HfApi().dataset_info(dataset_id, revision=revision)
    if not information.sha:
        raise RuntimeError("Hugging Face did not return an immutable dataset SHA")
    return information.sha


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Build the deterministic lossless SentencePiece Unigram-24K "
            "artifact used by train_fineweb.py."
        )
    )
    canonical_model, _ = canonical_sentencepiece_paths()
    result.add_argument("--output", type=Path, default=canonical_model)
    result.add_argument("--vocabulary-size", type=int, default=24_576)
    result.add_argument("--maximum-documents", type=int, default=250_000)
    result.add_argument("--maximum-utf8-bytes", type=int, default=512 << 20)
    result.add_argument("--dataset-id", default="HuggingFaceFW/fineweb")
    result.add_argument("--dataset-config", default="sample-10BT")
    result.add_argument("--dataset-revision", default="main")
    result.add_argument("--eval-fraction-permyriad", type=int, default=100)
    result.add_argument("--shuffle-seed", type=int, default=20260721)
    result.add_argument("--shuffle-buffer", type=int, default=10_000)
    result.add_argument("--threads", type=int, default=8)
    result.add_argument(
        "--pin-revision", action=argparse.BooleanOptionalAction, default=True,
    )
    return result


def main() -> None:
    args = parser().parse_args()
    revision = (
        _dataset_revision(args.dataset_id, args.dataset_revision)
        if args.pin_revision else args.dataset_revision
    )
    configuration = replace(
        UnigramTokenizerTrainingConfig(),
        vocabulary_size=args.vocabulary_size,
        maximum_documents=args.maximum_documents,
        maximum_utf8_bytes=args.maximum_utf8_bytes,
        number_of_threads=args.threads,
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        dataset_revision=revision,
        evaluation_fraction_permyriad=args.eval_fraction_permyriad,
        shuffle_seed=args.shuffle_seed,
        shuffle_buffer=args.shuffle_buffer,
    )
    source = FineWebTextSource(
        dataset_id=configuration.dataset_id,
        dataset_config=configuration.dataset_config,
        split=configuration.dataset_split,
        revision=configuration.dataset_revision,
        partition="train",
        evaluation_fraction_permyriad=(
            configuration.evaluation_fraction_permyriad
        ),
        shuffle_seed=configuration.shuffle_seed,
        shuffle_buffer=configuration.shuffle_buffer,
    )
    manifest = train_unigram_tokenizer(source, args.output, configuration)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
