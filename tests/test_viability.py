import torch

from mrrn.viability import (
    ViabilityForecast, ViabilityGate, ViabilityState, update_measured_viability,
)


def authoritative_state() -> ViabilityState:
    state = ViabilityState.empty(1, 2)
    values = {name: getattr(state, name).clone() for name in state.__dataclass_fields__}
    values["values"][0] = torch.tensor([0.8, 0.7])
    values["target_low"][0] = torch.tensor([0.4, 0.4])
    values["target_high"][0] = torch.tensor([0.9, 0.9])
    values["hard_low"][0] = torch.tensor([0.2, 0.2])
    values["hard_high"][0] = torch.tensor([1.0, 1.0])
    values["reserve"][0] = torch.tensor([0.6, 0.5])
    values["authority_mask"][0] = True
    values["provenance_ids"][0] = torch.tensor([10, 11])
    values["active"][0] = True
    return ViabilityState(**values)


def test_hard_viability_envelope_masks_unsafe_high_utility_candidate_before_selection():
    state = authoritative_state()
    forecast = ViabilityForecast(
        torch.tensor([[[0.1, 0.9], [0.6, 0.6]]]),
        torch.tensor([[[0.01, 0.01], [0.01, 0.01]]]),
        torch.tensor([[True, True]]),
    )
    decision = ViabilityGate(maximum_violation_probability=0.05)(state, forecast)
    assert decision.authorized.tolist() == [[False, True]]
    assert decision.minimum_hard_margin[0, 0] < 0


def test_increasing_hard_risk_cannot_make_an_action_newly_eligible():
    state = authoritative_state()
    safe = ViabilityForecast(
        torch.tensor([[[0.6, 0.6]]]), torch.tensor([[[0.01, 0.01]]]),
        torch.tensor([[True]]),
    )
    uncertain = ViabilityForecast(
        safe.values, torch.tensor([[[0.4, 0.4]]]), safe.candidate_mask,
    )
    gate = ViabilityGate()
    assert gate(state, safe).authorized.item()
    assert not gate(state, uncertain).authorized.item()


def test_measured_depletion_and_replenishment_update_authority_exactly():
    state = authoritative_state()
    depleted = update_measured_viability(
        state, measurements=torch.tensor([[0.25, 0.3]]),
        measurement_mask=torch.tensor([[True, True]]),
        provenance_ids=torch.tensor([[20, 20]]),
    )
    torch.testing.assert_close(depleted.values, torch.tensor([[0.25, 0.3]]))
    replenished = update_measured_viability(
        depleted, measurements=depleted.values,
        measurement_mask=torch.tensor([[True, False]]),
        provenance_ids=torch.tensor([[21, -1]]),
        replenishment=torch.tensor([[0.5, 0.0]]),
    )
    torch.testing.assert_close(replenished.values, torch.tensor([[0.75, 0.3]]))
    assert replenished.provenance_ids.tolist() == [[21, 20]]
