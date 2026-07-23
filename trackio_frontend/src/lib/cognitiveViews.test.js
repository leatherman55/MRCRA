import { describe, expect, it } from "vitest";
import {
  causalTimeline,
  deliberationLattice,
  invariantRows,
  reconstructionRows,
  viabilityRows,
} from "./cognitiveViews.js";

const evidence = {
  reconstructions: [{ batch: 0, slot: 1, historical_fidelity: 0.8, structural_plausibility: 0.9, evidence_agreement: 0.7 }],
  hypotheses: [{ batch: 0, scenario: 4, weight: 0.6, unknown: true }],
  action_candidates: [{ batch: 0, schema: 2, utility: 1.2, information_gain: 0.3, tail_risk: 0.1, permitted: true, provenance_authorized: true, viability_authorized: true, selected: true }],
  viability: { values: [[0.5]], target_low: [[0.2]], target_high: [[0.8]], active: [[true]], hard_violations: [[false]] },
  knowledge: [{ kind: "invariant", code_gain_bits: 10, reconstruction_distortion: 0.1, relation_distortion: 0.2, predictive_utility: 0.4, action_utility: 0.3 }],
  timeline: { event_counts: [[1]] },
  metacognition: { steps: [{ time: 0, highest_value_operation: "retrieval", predicted_error: 0.2 }] },
  actions: [{ time: 0, internal_step: 0, action: "simulate", status: "success", success: true }],
  external_actions: [{ selected_action: 2, authorized: true }],
};

describe("MRCRA cognitive views", () => {
  it("builds all five views from immutable evidence", () => {
    expect(reconstructionRows(evidence)).toHaveLength(1);
    expect(deliberationLattice(evidence)[0]).toMatchObject({ hypothesis: 4, action: 2, authorized: true });
    expect(viabilityRows(evidence)[0]).toMatchObject({ value: 0.5, hard_violation: false });
    expect(invariantRows(evidence)[0].distortion).toBeCloseTo(0.3);
    expect(invariantRows(evidence)[0].transfer_utility).toBeCloseTo(0.7);
    expect(causalTimeline(evidence).map((item) => item.stage)).toEqual([
      "observation", "metacognitive routing", "internal", "authorization",
    ]);
  });

  it("fails soft for absent optional diagnostic state", () => {
    expect(reconstructionRows()).toEqual([]);
    expect(deliberationLattice()).toEqual([]);
    expect(viabilityRows()).toEqual([]);
    expect(invariantRows()).toEqual([]);
    expect(causalTimeline()).toEqual([]);
  });
});
