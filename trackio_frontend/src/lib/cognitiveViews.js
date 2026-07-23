function finite(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

export function reconstructionRows(cognitive = {}) {
  return (cognitive.reconstructions || []).map((item) => ({
    ...item,
    historical_fidelity: finite(item.historical_fidelity),
    structural_plausibility: finite(item.structural_plausibility),
    evidence_agreement: finite(item.evidence_agreement),
  }));
}

export function deliberationLattice(cognitive = {}) {
  const hypotheses = cognitive.hypotheses || [];
  const actions = cognitive.action_candidates || [];
  return hypotheses.flatMap((hypothesis) => actions
    .filter((action) => action.batch === hypothesis.batch)
    .map((action) => ({
      hypothesis: hypothesis.scenario,
      unknown: Boolean(hypothesis.unknown),
      posterior: finite(hypothesis.weight),
      action: action.schema,
      utility: finite(action.utility),
      information_gain: finite(action.information_gain),
      tail_risk: finite(action.tail_risk),
      authorized: Boolean(
        action.permitted && action.provenance_authorized && action.viability_authorized,
      ),
      selected: Boolean(action.selected),
    })));
}

export function viabilityRows(cognitive = {}) {
  const state = cognitive.viability || {};
  const values = state.values || [];
  return values.flatMap((row, batch) => row.map((value, channel) => ({
    batch,
    channel,
    value: finite(value),
    target_low: finite(state.target_low?.[batch]?.[channel]),
    target_high: finite(state.target_high?.[batch]?.[channel]),
    hard_violation: Boolean(state.hard_violations?.[batch]?.[channel]),
    active: Boolean(state.active?.[batch]?.[channel]),
  }))).filter((item) => item.active);
}

export function invariantRows(cognitive = {}) {
  return (cognitive.knowledge || [])
    .filter((item) => item.kind === "invariant")
    .map((item) => ({
      ...item,
      code_gain_bits: finite(item.code_gain_bits),
      distortion: finite(item.reconstruction_distortion) + finite(item.relation_distortion),
      transfer_utility: finite(item.predictive_utility) + finite(item.action_utility),
    }));
}

export function causalTimeline(cognitive = {}) {
  const timeline = [];
  (cognitive.metacognition?.steps || []).forEach((item) => timeline.push({
    order: item.time * 100 - 0.5,
    stage: "metacognitive routing",
    label: item.highest_value_operation || "route operations",
    detail: `predicted error ${finite(item.predicted_error).toFixed(3)}`,
    success: true,
  }));
  (cognitive.actions || []).forEach((item) => timeline.push({
    order: item.time * 100 + item.internal_step,
    stage: "internal",
    label: item.action,
    detail: item.status,
    success: Boolean(item.success),
  }));
  (cognitive.external_actions || []).forEach((item, index) => timeline.push({
    order: 10000 + index,
    stage: item.authorized ? "authorization" : "abstention",
    label: item.authorized ? `action ${item.selected_action}` : "abstain",
    detail: item.authorized ? "authorized" : "not authorized",
    success: Boolean(item.authorized),
  }));
  const eventCounts = cognitive.timeline?.event_counts || [];
  eventCounts.forEach((row, batch) => row.forEach((count, time) => {
    if (count > 0) timeline.push({
      order: time * 100 - 1,
      stage: "observation",
      label: `${count} event${count === 1 ? "" : "s"}`,
      detail: `batch ${batch}`,
      success: true,
    });
  }));
  return timeline.sort((left, right) => left.order - right.order);
}
