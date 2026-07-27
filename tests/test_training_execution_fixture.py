from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from mrrn.training_execution_fixture import (
    RepeatingPackedFixtureStream,
    build_execution_fixture,
)


@pytest.mark.parametrize(
    ("profile", "length"),
    (
        ("unit_1k", 1_024),
        ("frequent_8k", 8_192),
        ("production_32k", 32_768),
    ),
)
def test_source_text_free_fixture_is_deterministic_bijective_authority(
    profile, length,
):
    before = torch.random.get_rng_state().clone()
    first = build_execution_fixture(profile, vocabulary_size=257)
    second = build_execution_fixture(profile, vocabulary_size=257)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert first.target_digest == second.target_digest
    assert first.metadata() == second.metadata()
    assert first.batch.input_ids.shape == (1, length)
    assert first.document_count > 1
    assert first.boundary_count == first.document_count - 1
    assert int(first.batch.input_ids.max()) < 257
    assert first.batch.external_source_uris[0]
    assert all(
        uri.startswith("fixture://") and " " not in uri
        for uri in first.batch.external_source_uris[0].values()
    )
    metadata_text = repr(first.metadata()).lower()
    assert "causal resonance" not in metadata_text
    assert "fineweb" not in metadata_text


def test_fixture_stream_resume_is_exact_and_fails_closed_on_identity_drift():
    fixture = build_execution_fixture("unit_1k", vocabulary_size=257)
    stream = RepeatingPackedFixtureStream(fixture)
    assert stream.next_batch(1, 1_024) is fixture.batch
    state = stream.state_dict()
    restored = RepeatingPackedFixtureStream(fixture)
    restored.load_state_dict(state)
    assert restored.batches_emitted == 1
    with pytest.raises(ValueError, match="immutable shape"):
        restored.next_batch(1, 8)
    corrupt = deepcopy(state)
    corrupt["fixture"]["target_digest"] = "0" * 64
    with pytest.raises(ValueError, match="stream"):
        restored.load_state_dict(corrupt)
