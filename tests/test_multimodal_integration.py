import pytest
import torch

from mrrn.cognitive_model import MultimodalRelationalContinuityResonanceNetwork
from mrrn.cognitive_types import BoundaryClass, ModalityClass
from mrrn.config import CognitiveConfig, MRCRAConfig, MRRNConfig
from mrrn.modalities import DomainSpec, EncodedDomain
from mrrn.multimodal_io import MultimodalPacketAssembler
from mrrn.provenance import ProvenanceLedger


def _config() -> MRCRAConfig:
    return MRCRAConfig(
        MRRNConfig(
            input_dim=8, model_dim=8, output_dim=6, layers=1, scales=3,
            heads=2, modes=2, mimo_rank=1, attention_window=2,
            retrieved_items=1, memory_capacity=4, width_multiple=4,
            mixer_expansion=1.5, spectral_modes=2, spectral_basis_order=2,
            enable_global_head=False, relational_branch=True,
            relational_context_dim=8,
        ),
        CognitiveConfig(
            workspace_dim=8, provenance_features=16, uncertainty_channels=8,
            relation_heads=2, relation_modes=2, relation_adapter_rank=2,
            goal_slots=2, goal_constraint_dim=2, system_action_channels=2,
            calibration_regimes=2, active_event_capacity=8,
            pair_edge_capacity=8, hyperedge_capacity=2, graph_neighbors=2,
            global_workspace_slots=2, hypothesis_slots=2,
            maximum_hypothesis_slots=2, maximum_cognitive_steps=2,
            event_chunk_size=2, event_proposals_per_chunk=2,
            recent_candidates=2, landmark_candidates=1,
            episodic_candidates=1, semantic_candidates=1,
            episodic_memory_capacity=4, semantic_memory_capacity=3,
            associative_depth=2, associative_budget=2,
            world_model_horizons=(1, 2),
        ),
        actor_parameter_minimum=1, actor_parameter_maximum=10_000_000,
    )


def test_spatial_and_temporal_modalities_fuse_causally_and_run_end_to_end():
    torch.manual_seed(409)
    assembler = MultimodalPacketAssembler(
        {ModalityClass.SENSOR: 3, ModalityClass.IMAGE: 4},
        width=8, uncertainty_channels=8,
    )
    sensor = EncodedDomain(
        torch.randn(1, 3, 3), torch.ones(1, 3, dtype=torch.bool),
        DomainSpec("sensor", sample_interval=0.5, boundary="causal"),
    )
    image = EncodedDomain(
        torch.randn(1, 2, 2, 4), torch.ones(1, 2, 2, dtype=torch.bool),
        DomainSpec("image", sample_interval=1.0, boundary="reflect"),
    )
    ledger = ProvenanceLedger()
    sensor_packet = assembler.prepare_external(
        sensor, ModalityClass.SENSOR, source_uris=("sensor://imu",),
        ledger=ledger, model_authority="assembler-test",
    )
    image_packet = assembler.prepare_external(
        image, ModalityClass.IMAGE, source_uris=("camera://front",),
        ledger=ledger, model_authority="assembler-test",
        timestamps=torch.zeros(1, 2, 2),
    )
    packet = assembler.fuse((sensor_packet, image_packet), ledger)
    assert packet.values.shape == (1, 7, 8)
    assert torch.equal(
        packet.modality_ids.unique().sort().values,
        torch.tensor([int(ModalityClass.SENSOR), int(ModalityClass.IMAGE)]),
    )
    assert torch.all(packet.timestamps[:, 1:] >= packet.timestamps[:, :-1])
    assert packet.boundary_classes[0, 0] == int(BoundaryClass.HARD)
    # The second modality begins at the same physical time and is a soft
    # synchronized boundary, not a destructive second hard reset.
    same_time = torch.nonzero(packet.timestamps[0] == packet.timestamps[0, 0]).flatten()
    assert (packet.boundary_classes[0, same_time[1:]] != int(BoundaryClass.HARD)).all()
    packet.assert_ledger_consistent(ledger)

    model = MultimodalRelationalContinuityResonanceNetwork(_config()).eval()
    output = model(packet, ledger)
    assert output.prediction.shape == (1, 7, 6)
    assert torch.isfinite(output.prediction).all()
    assert output.schema_probabilities.shape[:2] == (1, 7)
    assert output.symbol_gates.shape[:2] == (1, 7)
    assert output.state.goals.desired_outcomes.shape[0] == 1
    assert output.state.system_model.action_availability.shape == (1, 2)


def test_multimodal_assembler_rejects_false_domain_claims_and_width_mismatch():
    assembler = MultimodalPacketAssembler(
        {ModalityClass.AUDIO: 2}, width=4, uncertainty_channels=8
    )
    ledger = ProvenanceLedger()
    image = EncodedDomain(
        torch.randn(1, 2, 2, 2), torch.ones(1, 2, 2, dtype=torch.bool),
        DomainSpec("image", boundary="reflect"),
    )
    try:
        assembler.prepare_external(
            image, ModalityClass.AUDIO, source_uris=("bad://claim",),
            ledger=ledger, model_authority="test",
        )
    except ValueError as error:
        assert "cannot claim" in str(error)
    else:
        raise AssertionError("false modality/domain claim was accepted")


def test_multimodal_preparation_and_fusion_contracts_fail_closed():
    with pytest.raises(ValueError, match="dimensions"):
        MultimodalPacketAssembler({}, width=4, uncertainty_channels=8)
    with pytest.raises(ValueError, match="positive"):
        MultimodalPacketAssembler({ModalityClass.AUDIO: 0}, width=4, uncertainty_channels=8)
    audio = EncodedDomain(
        torch.randn(1, 3, 2), torch.ones(1, 3, dtype=torch.bool),
        DomainSpec("audio", sample_interval=0.25, boundary="causal"),
    )
    ledger = ProvenanceLedger()
    unconfigured = MultimodalPacketAssembler(
        {ModalityClass.SENSOR: 2}, width=4, uncertainty_channels=8
    )
    with pytest.raises(ValueError, match="no configured projection"):
        unconfigured.prepare_external(
            audio, ModalityClass.AUDIO, source_uris=("audio://x",),
            ledger=ledger, model_authority="test",
        )
    wrong_width = MultimodalPacketAssembler(
        {ModalityClass.AUDIO: 3}, width=4, uncertainty_channels=8
    )
    with pytest.raises(ValueError, match="feature width"):
        wrong_width.prepare_external(
            audio, ModalityClass.AUDIO, source_uris=("audio://x",),
            ledger=ledger, model_authority="test",
        )
    assembler = MultimodalPacketAssembler(
        {ModalityClass.AUDIO: 2}, width=4, uncertainty_channels=8
    )
    with pytest.raises(ValueError, match="source declaration"):
        assembler.prepare_external(
            audio, ModalityClass.AUDIO, source_uris=(),
            ledger=ledger, model_authority="test",
        )
    with pytest.raises(ValueError, match="coordinates"):
        assembler.prepare_external(
            audio, ModalityClass.AUDIO, source_uris=("audio://x",),
            coordinates=torch.zeros(1, 2, 1), ledger=ledger,
            model_authority="test",
        )
    with pytest.raises(ValueError, match="timestamps"):
        assembler.prepare_external(
            audio, ModalityClass.AUDIO, source_uris=("audio://x",),
            timestamps=torch.zeros(1, 2), ledger=ledger,
            model_authority="test",
        )
    with pytest.raises(ValueError, match="at least one"):
        assembler.fuse((), ledger)


def test_multimodal_fusion_rejects_incommensurate_packets():
    audio = EncodedDomain(
        torch.randn(1, 2, 2), torch.ones(1, 2, dtype=torch.bool),
        DomainSpec("audio", boundary="causal"),
    )
    ledger = ProvenanceLedger()
    first = MultimodalPacketAssembler(
        {ModalityClass.AUDIO: 2}, 4, 8
    ).prepare_external(
        audio, ModalityClass.AUDIO, source_uris=("audio://a",), ledger=ledger,
        model_authority="test", clock_units="seconds",
    )
    different_width = MultimodalPacketAssembler(
        {ModalityClass.AUDIO: 2}, 5, 8
    ).prepare_external(
        audio, ModalityClass.AUDIO, source_uris=("audio://b",), ledger=ledger,
        model_authority="test", clock_units="seconds",
    )
    with pytest.raises(ValueError, match="share batch, width"):
        MultimodalPacketAssembler.fuse((first, different_width), ledger)
    different_clock = MultimodalPacketAssembler(
        {ModalityClass.AUDIO: 2}, 4, 8
    ).prepare_external(
        audio, ModalityClass.AUDIO, source_uris=("audio://c",), ledger=ledger,
        model_authority="test", clock_units="samples",
    )
    with pytest.raises(ValueError, match="clock unit"):
        MultimodalPacketAssembler.fuse((first, different_clock), ledger)
    different_uncertainty = MultimodalPacketAssembler(
        {ModalityClass.AUDIO: 2}, 4, 7
    ).prepare_external(
        audio, ModalityClass.AUDIO, source_uris=("audio://d",), ledger=ledger,
        model_authority="test", clock_units="seconds",
    )
    with pytest.raises(ValueError, match="uncertainty width"):
        MultimodalPacketAssembler.fuse((first, different_uncertainty), ledger)
