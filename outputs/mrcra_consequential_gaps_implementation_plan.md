# MRCRA Consequential-Gap Closure Implementation Plan

**Status:** implementation-ready plan  
**Target:** the production Multimodal Relational-Continuity Resonance Architecture (MRCRA), not a prototype branch  
**Scope:** close every consequential gap identified between the current implementation and the supplied working definition of cognition  
**Out of scope:** replacing the dense multiresolution resonance carrier, increasing model size before capability closure, claiming consciousness, or authorizing unattended base-weight self-modification

---

## 1. Required outcome

The completed system shall be a bounded, stateful, evidence-aware cognitive agent substrate in which:

1. low-cost multiresolution resonant processing remains the continuous dense path;
2. completed or salient events are promoted into a bounded typed relational graph;
3. abstractions are used only while their applicability and precision remain valid;
4. descent reconstructs a localized lower-level relational state from an abstraction, surviving traces, present evidence, current context, and goals;
5. reconstructed, retrieved, inferred, predicted, simulated, and observed content remain epistemically distinct;
6. multiple hypotheses are updated from real evidence and used to evaluate multiple possible actions;
7. external action selection occurs after internal simulation and verification, not before it;
8. the application-owned executor closes the perception-action-perception loop without granting environmental authority to a neural tensor;
9. viability and hard constraints precede learned utility;
10. invariant discovery performs actual role-normalized cross-context structural comparison and counterexample search;
11. goals, system limitations, decision history, and calibration failures can themselves become objects of relational reasoning;
12. training, tests, and evidence labels distinguish a working contract, a learned local mechanism, an integrated loop, a serious checkpoint, and deployment evidence.

The design is complete only when all these properties are simultaneously active in the same runtime, persist correctly across declared boundaries, survive checkpoint/resume, receive admissible training signals, and pass integrated acceptance tests.

---

## 2. Non-negotiable architectural invariants

These constraints apply to every workstream.

### 2.1 Authority separation

- The append-only provenance ledger remains authoritative for source history, verification, revocation, and external-action justification.
- Learned source, confidence, applicability, or verification heads are estimates; they cannot rewrite the ledger.
- External effects remain application-owned. The model may propose and justify an action; only an executor can perform it.
- Permissions, hard constraints, and viability limits are masks or hard gates before probability normalization.
- Observed content is immutable. A reconstruction, inference, abstraction, or later correction creates a derived record instead of altering the original record.

### 2.2 Boundedness

- Every live tensor store has an explicit capacity, eviction policy, and telemetry.
- Hypothesis-action simulation is routed top-\(K\), not an unbounded Cartesian product.
- Reconstructive descent is localized to a requested subgraph, support interval, or relational query.
- Expensive cognition remains event-driven. Per-token dense processing must not silently acquire graph-scale loops.
- External archives and ledger stores may grow, but their storage and index costs must be reported separately from bounded live state.

### 2.3 Epistemic distinctions

- Plausibility is not historical fidelity.
- Model confidence is not provenance confidence.
- Correlation and temporal prediction are not automatically causal.
- A successful reconstruction is not an observation.
- A simulated success is not an executed success.
- A bounded mechanism test is not evidence of open-domain capability.

### 2.4 Stability and compatibility

- Existing MRRN causality, spectral stability, phase conventions, boundary isolation, and token-language authority remain passing throughout the migration.
- New checkpoint formats are versioned and explicitly migrated; ambiguous old state fails closed.
- New auxiliary objectives cannot silently dominate or erase the primary task.
- No phase may proceed while its prerequisite integration gate is failing.

### 2.5 Current gap-to-closure map

| Current production behavior | Consequence | Required closure |
|---|---|---|
| `DECOMPRESS` unconditionally decodes the first abstraction and averages its nodes into context | no localized detail, evidence conditioning, fidelity estimate, or reconstructed graph | Workstreams A and B |
| external action is selected before `_run_internal_actions` | simulation and verification cannot revise the same-cycle action | Workstreams C and D |
| `SIMULATE` uses the first active hypothesis and already-selected action | alternatives do not participate in control | Workstream D |
| `HypothesisBank.update_evidence` has no production caller | hypothesis probabilities do not learn from ongoing observation | Workstream D |
| `VERIFY` returns only `EXTERNAL_EVIDENCE_REQUIRED` | no actionable verification or active-sensing request | Workstream C |
| feedback updates success, latency, and availability but not the complete consequence state | reward, cost, constraints, and resultant evidence do not close the loop | Workstreams C and E |
| system compute is reduced by action fraction with no measured replenishment/forecast | resource fields are budgets, not viability regulation | Workstream E |
| `PROPOSE_INVARIANT` averages episodic values while structural invariant modules remain separate | no integrated role-normalized invariant discovery | Workstream F |
| default goals are inactive while default external permissions are permissive | action-capable behavior can lack explicit purpose and capability registration | Workstreams C and G |
| `COMPARE`, `BIND`, and retrieval rely on implicit or single operands | deliberate relational operations cannot select all required arguments | global state changes and Workstream H |
| every packed training context creates a fresh cognitive state and ledger | safe document isolation exists, but persistent cognition is not trained | Workstream I |
| stage-1 FineWeb training requires no cognitive auxiliary families | the cognitive runtime can remain largely unsupervised while the run is valid | Workstream J |
| bounded proxy tests and integrated capabilities share the word `verified` | passing tests can be read as stronger evidence than they contain | Workstream K |

---

## 3. Target runtime cycle

The production cycle shall use the following causal order:

```text
observation packet
  -> boundary and clock transition
  -> dense multiresolution resonant update
  -> event extraction and provenance derivation
  -> typed graph/workspace update
  -> memory and surviving-trace retrieval
  -> hypothesis likelihood update from the new evidence
  -> uncertainty, applicability, and viability assessment
  -> highest-valid-abstraction selection
  -> candidate internal operation and external action generation
  -> routed reconstruction / simulation / verification
  -> candidate consequence, information-gain, risk, and cost evaluation
  -> hard permission, provenance, and viability authorization
  -> one external action proposal or explicit abstention/evidence request
  -> application executor
  -> executor result translated into a new observation plus measured feedback
  -> posterior, calibration, action competence, goal, and viability update
  -> eligible memory writes and knowledge proposals
  -> output decoding with uncertainty and source metadata
```

External action selection must not occur before the deliberation portion of this cycle.

### 3.1 Two-speed execution

The runtime must distinguish:

- **fast path:** carrier update, prediction residual, lightweight uncertainty, event trigger, and already-validated abstraction execution;
- **deliberative path:** graph update, retrieval, reconstruction, multi-hypothesis simulation, verification, invariant search, or reflective inspection.

The deliberative path runs only on event, boundary, material uncertainty, contradiction, goal risk, reconstruction failure, requested precision, or explicit controller demand.

### 3.2 Candidate-deliberation contract

The controller shall first produce a bounded candidate set, not a final action. For each candidate \(a\) and routed hypothesis \(h\), the world model produces:

\[
p_\theta(o_{t+1:t+H}, r, c, v, d \mid s_t,a,h),
\]

where \(r\) is reward or goal progress, \(c\) cost, \(v\) constraint/viability outcome, and \(d\) termination. The selector computes a normalized, goal-conditioned score such as

\[
U(a)=\mathbb E[R\mid a]+\beta I(H;O\mid a)
-\lambda_c\widetilde C(a)-\lambda_e\widetilde E(a)
-\lambda_r\operatorname{CVaR}_\alpha(L\mid a),
\]

subject to hard availability, permission, provenance, precision, and viability constraints. Tilde quantities are calibrated dimensionless normalizations; incompatible physical units must not be added directly.

---

## 4. Global state and ontology changes

### 4.1 Extend source and verification semantics

Modify `src/mrrn/cognitive_types.py`.

Add source classes:

- `RECONSTRUCTED`: generated lower-level content derived from an abstraction and traces;
- `TOOL_OUTPUT`: content returned by an authorized tool or executor;
- `COMMUNICATED`: information asserted by another agent/source and awaiting its own authority assessment;
- `EXTERNAL_ARTIFACT`: content read from a deliberately created external memory aid.

Do not merge `RECONSTRUCTED` with `INFERRED` or `SIMULATED`; their fidelity and authority questions differ.

Add or formalize verification states for reconstruction agreement only if these cannot be represented as ledger events. `INTERNALLY_CONSISTENT` must never be interpreted as externally true.

### 4.2 Add targeted cognitive actions

Extend `InternalAction` with actions whose effects are semantically distinct:

- `RECONSTRUCT_LOCAL`;
- `TEST_APPLICABILITY`;
- `UPDATE_HYPOTHESES`;
- `GENERATE_ACTION_CANDIDATES`;
- `EVALUATE_CANDIDATES`;
- `INSPECT_SELF_STATE`;
- `CREATE_EVIDENCE_REQUEST`;
- `REVISE_ABSTRACTION`;
- `RECORD_EXTERNAL_ARTIFACT`;
- `QUERY_TOOL`.

Retain old enum numbers for checkpoint compatibility. Append new values rather than reordering existing values.

### 4.3 Replace single-pointer decisions with structured operands

Extend `ControllerDecision` and action receipts with:

- primary and secondary node pointers;
- primary and secondary relation pointers;
- abstraction or invariant pointer;
- hypothesis pointer set and mask;
- candidate-action pointer set and mask;
- requested physical scale;
- requested abstraction depth;
- requested support interval or subgraph seed;
- precision/tolerance request;
- typed argument vector and argument schema ID;
- expected operation cost;
- reason/trigger class.

This removes implicit behaviors such as selecting the first “other” comparison node or the first active hypothesis.

### 4.4 Expand runtime state

Add the following immutable dataclass fields to `MRCRARuntimeState`:

- `reconstructions: ReconstructionState`;
- `abstraction_validity: AbstractionValidityState`;
- `action_candidates: ActionCandidateState`;
- `viability: ViabilityState`;
- `evidence_requests: EvidenceRequestState`;
- `external_artifacts: ExternalArtifactState`;
- `metacognition: MetacognitiveState`;
- `boundary_context: BoundaryContextState`.

Every state must implement shape validation, `detach`, `to`, empty initialization, row replacement, checkpoint serialization, and migration tests.

### 4.5 Checkpoint format

Increment the cognitive checkpoint format version. Implement explicit migration from the current format by initializing new state conservatively:

- no active reconstructions;
- no active candidates or evidence requests;
- unknown abstraction validity;
- application-supplied permissions required;
- neutral viability only in offline/modeling mode;
- no claim that old hypotheses have updated likelihoods;
- no conversion of old inferred records into reconstructed records.

Resume equivalence must include ledger offset/digest, clocks, boundary context, viability, candidate state, hypothesis posteriors, calibration, executor-feedback sequence numbers, and RNG.

---

## 5. Workstream A: evidence-conditioned generative reconstruction

### 5.1 New modules

Create:

- `src/mrrn/reconstruction.py`;
- `tests/test_reconstruction.py`;
- `tests/test_reconstructive_descent_integration.py`.

Implement these core contracts:

```python
@dataclass(frozen=True, slots=True)
class ReconstructionQuery:
    abstraction_indices: Tensor
    seed_node_indices: Tensor
    requested_support: Tensor
    requested_node_count: Tensor
    requested_relation_count: Tensor
    target_scale: Tensor
    target_abstraction_depth: Tensor
    precision_tolerance: Tensor
    goal_context: Tensor
    mask: Tensor

@dataclass(frozen=True, slots=True)
class ReconstructionEvidence:
    abstraction_latent: Tensor
    trace_content: Tensor
    trace_mask: Tensor
    trace_provenance_ids: Tensor
    observed_context: Tensor
    observed_provenance_ids: Tensor
    current_relations: Tensor
    hypothesis_context: Tensor
    goal_context: Tensor

@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    node_content: Tensor
    node_type_logits: Tensor
    node_mask: Tensor
    relation_content: Tensor
    relation_type_logits: Tensor
    participant_indices: Tensor
    relation_mask: Tensor
    historical_fidelity: Tensor
    structural_plausibility: Tensor
    evidence_agreement: Tensor
    epistemic_uncertainty: Tensor
    aleatoric_uncertainty: Tensor
    applicability_probability: Tensor
    provenance_ids: Tensor
```

### 5.2 Reconstruction model

Implement a conditional bounded graph decoder:

\[
q_\phi(G^{\ell-1}\mid z^\ell,\tau,c,x,g,h,r).
\]

Recommended construction:

1. encode the abstraction latent and its explicit applicability record;
2. retrieve top-\(K\) surviving traces by support, relation, provenance, and phase compatibility;
3. cross-condition bounded node queries on traces, present evidence, goals, and selected hypothesis;
4. generate node existence, types, content, support, and uncertainty;
5. generate routed relation candidates, types, roles, and participants;
6. preserve residual modality-specific information separately from normalized relational roles;
7. use a deterministic provenance builder to derive one record per active reconstructed node/relation;
8. insert results into dedicated reconstruction state first, never directly into observation slots;
9. promote reconstructed elements into the working graph only after consistency and capacity checks.

The decoder may reuse `GraphCompressor.decode_latent` internally, but the existing unconditional decoder cannot be the complete operator.

### 5.3 Fidelity and provenance

For every reconstructed element, the ledger record must include:

- parent abstraction/invariant;
- surviving trace parents;
- current observed-evidence parents;
- selected hypothesis/scenario;
- reconstruction operator version;
- requested region and scale;
- model authority;
- support interval;
- calibration regime.

Historical fidelity and present plausibility must be separately exposed. Neither may be copied into verification authority.

### 5.4 Training objectives

Add an objective family `RECONSTRUCTION_APPLICABILITY` or split it into two families if gradient telemetry shows conflict.

Train with:

- graph node/relation reconstruction under structured masking;
- abstraction-to-detail reconstruction;
- partial-trace corruption and recovery;
- evidence-conditioned correction where new observation contradicts a plausible completion;
- negative examples in which the parent abstraction does not apply;
- historical-fidelity calibration;
- present-evidence consistency;
- cycle consistency without requiring an exact latent inverse;
- provenance parent/source reconstruction;
- abstention when requested fidelity cannot be supported.

The primary reconstruction loss is:

\[
\mathcal L_{\rm reconstruct}=
\mathcal L_{\rm node}+
\mathcal L_{\rm relation}+
\lambda_s\mathcal L_{\rm support}+
\lambda_e\mathcal L_{\rm evidence}+
\lambda_p\mathcal L_{\rm provenance}+
\lambda_c\mathcal L_{\rm calibration}+
\lambda_a\mathcal L_{\rm applicability}.
\]

### 5.5 Acceptance gate A-R

The workstream is complete only if:

- reconstructed nodes/relations have `RECONSTRUCTED` ledger records;
- observed records remain byte-for-byte unchanged;
- conditioning on correct new evidence improves held-out fidelity over unconditional decode;
- contradictory evidence changes reconstruction in the correct local region;
- unsupported exact-history queries cause calibrated abstention;
- historical-fidelity ECE and coverage meet declared bounds;
- localized descent is measurably cheaper than full-detail recomputation;
- checkpoint/resume produces identical reconstruction state and ledger digest;
- removing traces, evidence, or applicability conditioning causes a significant matched degradation.

---

## 6. Workstream B: abstraction validity and localized bidirectional hierarchy

### 6.1 New state and module

Create `src/mrrn/abstraction_control.py` with:

- `AbstractionValidityRecord`;
- `AbstractionValidityState`;
- `AbstractionApplicabilityHead`;
- `AbstractionLevelSelector`;
- `LocalizedDescentPlanner`;
- `AbstractionRevisionProposal`.

Each abstraction or invariant must carry:

- abstraction depth independent of physical carrier scale;
- applicability embedding and interpretable condition IDs where available;
- known failure/counterexample references;
- expected node/relation reconstruction distortion;
- expected task distortion by goal class;
- supported action precision;
- calibration regime;
- provenance sufficiency;
- residual decoder reference;
- validation version and expiry/recheck policy.

### 6.2 Highest-valid-abstraction policy

For goal \(g\), evidence \(E\), and abstraction depth \(\ell\), define validity as:

\[
V_\ell =
\Pr(D_{\rm task}\le\epsilon_g\mid E,g,\ell)
\cdot P_{\rm applicable}
\cdot P_{\rm provenance}
\cdot P_{\rm precision}.
\]

Select the greatest \(\ell\) satisfying all hard thresholds. Unknown applicability is not valid by default.

Descent triggers include:

- calibrated prediction residual above threshold;
- contradiction with independent evidence;
- applicability probability below threshold;
- novelty/OOD evidence;
- requested action precision beyond the abstraction contract;
- provenance insufficiency;
- unstable reconstruction;
- consequential uncertainty or viability risk.

### 6.3 Replace global scale switching

`DESCEND_SCALE` and `ASCEND_SCALE` shall no longer merely substitute one global carrier vector. They must either:

- select a physical scale for a declared support region; or
- change abstraction depth for a declared relational object/subgraph.

The two operations remain distinct in telemetry and state. If compatibility requires retaining the old action names, their receipts must specify which dimension changed.

### 6.4 Acceptance gate B-H

- Easy/known cases execute at a higher abstraction with less compute than forced-detail controls.
- Novel or contradictory local regions trigger localized descent.
- Unaffected regions remain compressed.
- Forced-high abstraction fails more often than adaptive abstraction on boundary cases.
- Forced-low detail costs more without a compensating accuracy gain on easy cases.
- The adaptive policy improves task utility per measured compute over both controls.
- Physical scale and abstraction depth interventions yield distinguishable behavior.

---

## 7. Workstream C: closed perception-action-perception orchestration

### 7.1 New application-layer session

Create `src/mrrn/agent_session.py` and keep it separate from the neural `nn.Module` graph.

```python
class CognitiveAgentSession:
    def observe(self, packets: ObservationPacket) -> SessionStepResult: ...
    def deliberate(self) -> DeliberationResult: ...
    def execute(self, executor: EnvironmentExecutor) -> ExecutionResult: ...
    def ingest_result(self, result: ExecutionResult) -> SessionStepResult: ...
    def checkpoint(self, path: Path) -> None: ...
```

`CognitiveAgentSession` owns:

- runtime state;
- provenance ledger;
- environment/session IDs;
- action schema registry;
- executor receipts and sequence numbers;
- feedback-to-observation conversion;
- calibration updates;
- boundary policy;
- failure recovery.

It does not own application permissions; the host registers those explicitly.

### 7.2 Structured external actions

Replace bare integer action channels as the complete external ontology with an `ActionSchemaRegistry`:

- stable action schema ID;
- parameter types, bounds, and masks;
- required capabilities and permissions;
- expected observation modalities;
- cost/resource units;
- reversibility;
- timeout;
- safety class;
- provenance requirements;
- optional information-gathering semantics.

The neural policy may emit a schema ID and bounded parameters. The authority layer validates both.

### 7.3 Evidence requests and tool operations

Replace the current status-only `VERIFY` behavior with a typed `EvidenceRequest` containing:

- proposition/hypothesis being tested;
- requested modality or tool;
- discriminated hypotheses;
- expected information gain;
- maximum cost/latency;
- required precision;
- supporting and contradicting provenance;
- fallback if unavailable.

An evidence request may become a candidate action, an application prompt, or an abstention result.

### 7.4 Feedback closure

Feedback updates must include:

- observed success;
- latency;
- reward/goal progress;
- physical or abstract cost;
- constraint violation;
- resultant observation availability;
- action-model prediction error;
- reversibility outcome;
- executor/tool reliability;
- provenance record.

Update the system model, world model supervision stream, hypothesis posterior, calibration, goal progress, and viability state from the same immutable receipt.

### 7.5 Acceptance gate C-A

- A deterministic environment test demonstrates observe -> deliberate -> authorize -> execute -> observe -> posterior update.
- Unauthorized and unjustified actions never reach the executor.
- Internal simulation changes final action selection in a task where the pre-deliberation favorite is wrong.
- Information-gathering actions are selected when their expected value exceeds immediate action.
- Feedback reward/cost/constraint fields alter future behavior in the correct direction.
- Duplicate, out-of-order, or replayed executor receipts are rejected or handled idempotently.
- Session checkpoint/resume does not repeat an already executed action.

---

## 8. Workstream D: multi-hypothesis inference and multi-action world modeling

### 8.1 Wire evidence updates into production

At every eligible observation/event:

1. compute each active hypothesis's observation likelihood;
2. add support and contradiction evidence counts with provenance references;
3. normalize posterior log weights;
4. update weak-step hysteresis;
5. merge only genuinely redundant hypotheses;
6. prune only after hysteresis or logical impossibility;
7. retain at least one explicit “other/unknown” hypothesis when evidence is incomplete.

No production path may select `argmax(active)` as a proxy for hypothesis quality.

### 8.2 Hypothesis-conditioned rollout

Extend the world model to accept:

- current workspace and relation summary;
- explicit selected hypothesis residual and relation overrides;
- structured candidate action and parameters;
- viability state;
- horizon mask;
- scenario ID.

Produce distributions for:

- future latent state by configured scale;
- relation changes;
- observations and event types;
- reward/goal progress;
- cost and resource use;
- constraint/viability violation;
- action success;
- termination;
- epistemic and aleatoric uncertainty.

### 8.3 Routed planning

Use:

- top-\(K_h\) hypotheses by posterior mass plus diversity;
- top-\(K_a\) actions by proposal score plus exploration;
- adaptive horizons chosen by decision consequence and model reliability;
- early termination when candidates are dominated;
- common-prefix sharing across scenario rollouts.

Report routed posterior mass and candidate recall. If routed mass is insufficient, abstain or widen within budget.

### 8.4 Causal discipline

World-model prediction does not create a causal relation. Causal promotion requires intervention evidence, randomized variation, supplied causal annotation, or a validated causal executor model.

### 8.5 Acceptance gate D-W

- Real observations update hypothesis posterior automatically.
- Ambiguous evidence preserves multiple alternatives; diagnostic evidence resolves them.
- Top-\(K\) planning matches exhaustive planning on bounded oracle tasks within tolerance.
- Action ablation damages interventional predictions but not unrelated passive predictions.
- Multihorizon rollout error and calibration remain within declared bounds.
- The selected action changes when hypothesis probabilities materially change.
- Simulated state remains scenario-isolated and never becomes observed authority.

---

## 9. Workstream E: viability, homeostasis, and resource authority

### 9.1 Viability state

Create `src/mrrn/viability.py` with:

```python
@dataclass(frozen=True, slots=True)
class ViabilityState:
    values: Tensor
    target_low: Tensor
    target_high: Tensor
    hard_low: Tensor
    hard_high: Tensor
    trend: Tensor
    uncertainty: Tensor
    reserve: Tensor
    recovery_priority: Tensor
    authority_mask: Tensor
    provenance_ids: Tensor
    active: Tensor
```

The generic tensor contract supports biological, robotic, server, or abstract operational variables without pretending they are identical.

Initial practical variables should include:

- measured compute reserve;
- memory reserve;
- deadline/latency reserve;
- sensor/actuator or tool availability;
- action error/failure rate;
- unresolved high-authority commitments;
- calibration degradation;
- capability-retention health;
- application-supplied safety variables.

### 9.2 Hard constraints before utility

Implement a `ViabilityGate` that masks candidates predicted to violate hard envelopes beyond allowed probability. Use risk-sensitive forecasts and conservative bounds. Learned policy logits cannot override the mask.

When all actions are unsafe or unsupported, emit abstention, recovery, or evidence request—not an arbitrary argmax.

### 9.3 Real resource accounting

Replace synthetic action-fraction depletion as the only compute model. Feed back:

- wall-clock latency;
- accelerator/CPU time where available;
- live tensor capacity and external-store pressure;
- executor costs;
- deadline slack;
- recovery/replenishment events.

Keep normalized differentiable estimates for planning, but retain measured authority separately.

### 9.4 Stability-plasticity regulation

Define plasticity eligibility from verified novelty, learnability, contradiction quality, replay coverage, and viability reserve. High aleatoric noise alone must not increase plasticity.

Parameter adaptation remains isolated and reversible until validation. Semantic memory promotion follows the same authority discipline.

### 9.5 Acceptance gate E-V

- Unsafe high-reward actions are masked before selection.
- Resource exhaustion causes adaptive reduction of deliberation and/or recovery behavior.
- Replenishment restores eligible compute rather than leaving a permanently drained scalar.
- Measured action costs improve future cost prediction.
- Viability-constrained policy outperforms unconstrained policy on survival/continuity metrics without hiding task loss.
- No learned gradient can alter hard thresholds or permissions unless the application explicitly authorizes that parameter class.

---

## 10. Workstream F: integrated invariant discovery and transfer

### 10.1 Use the existing strong components

Wire `StructuralNormalizer`, `BoundedGraphMatcher`, and `InvariantLedger` into the production cognitive model. `PROPOSE_INVARIANT` must no longer be a mean of active episodic vectors.

### 10.2 Discovery pipeline

1. Route candidate episodes/subgraphs by relation signatures, outcomes, goals, and phase-aligned temporal structure.
2. Select multiple episodes or declared transformations with independent provenance roots where possible.
3. Normalize participant roles while preserving original identities and residual attributes.
4. Perform bounded permutation-aware structural matching.
5. Induce a preserved pattern plus applicability conditions.
6. Estimate predictive and action utility on held-out contexts.
7. Measure code gain, node reconstruction distortion, and relation distortion.
8. Search declared counterexample families and adversarial structural near-matches.
9. Record known failures and contradicting provenance.
10. Promote only through the invariant authority gate.
11. Activate conditionally in new contexts.
12. Revise or revoke when a counterexample invalidates the prior scope.

### 10.3 Transfer evaluation

Every promoted invariant must be evaluated on:

- held-out surface identities;
- held-out episodes;
- held-out transformations;
- at least one domain-shifted instantiation for cross-domain claims;
- structural near-matches that should not activate;
- intervention or action utility where the invariant claims causal/procedural value.

### 10.4 Acceptance gate F-I

- Role-normalized matching improves transfer over raw-content matching.
- Surface-identity shuffling does not destroy invariant recognition.
- Structural near-matches with a critical relation changed are rejected.
- Counterexamples reduce applicability/confidence or create a revision.
- Promoted invariants improve held-out prediction or action while meeting reconstruction/distortion bounds.
- Cross-domain claims require cross-domain evidence; same-domain motif tests receive only mechanism-level status.

---

## 11. Workstream G: goals, self-modeling, and reflective cognition

### 11.1 Goal authority

Do not activate arbitrary default goals. Support explicit modes:

- `offline_modeling`: no external goals or actions;
- `task_agent`: caller must provide at least one authorized goal;
- `persistent_agent`: caller provides maintenance constraints plus task goals;
- `evaluation`: evaluator supplies controlled goal state.

Each goal records desired outcome, constraints, priority, horizon, authority, termination, provenance, status, achieved progress, and conflicts.

### 11.2 Technical self-model

Expand `SystemModelState` or add a complementary `SelfModelState` containing:

- modality/tool/action availability;
- empirical action success and latency distributions;
- world-model reliability by regime;
- reconstruction fidelity by regime;
- retrieval/router recall estimates;
- calibration state;
- current memory and compute limits;
- predicted effect of internal actions on uncertainty and cost;
- recent decision errors;
- known capability boundaries;
- update-transition predictions.

System estimates remain measurable and provenance-linked.

### 11.3 Reflective representation

Emit bounded `SYSTEM_STATE`, `GOAL`, `HYPOTHESIS`, and decision nodes representing:

- current interpretation;
- selected hypothesis distribution;
- selected action and alternatives;
- decisive evidence;
- unresolved contradiction;
- predicted confidence and realized error;
- controller trigger and compute expenditure;
- goal conflict;
- abstraction/reconstruction failure.

The ordinary graph and controller may inspect these nodes. Reflection must not create a separate unbounded chain of “thought about thought”; enforce a fixed reflection budget and cycle detection.

### 11.4 Metacognitive learning

Train predictions of:

- whether more compute will improve the decision;
- whether retrieval, reconstruction, simulation, or evidence request is likely to reduce consequential uncertainty;
- whether the current abstraction is outside its validity region;
- whether the system is calibrated in the current regime;
- whether abstention is preferable.

### 11.5 Acceptance gate G-S

- Caller omission of goals in an action-capable mode fails closed.
- Conflicting goals remain explicit and are not averaged into invisibility.
- The system predicts its own likely failure better than a base-rate control.
- Metacognitive routing improves utility per compute over fixed deliberation.
- Reflective nodes preserve provenance and cannot rewrite the underlying observation or receipt.
- Reflection depth and cost remain bounded.

---

## 12. Workstream H: memory policy, tools, and external artifacts

### 12.1 Correct memory operations

- Integrate `MemoryWritePolicyV2` into episodic write selection.
- Require consolidation authority for semantic writes.
- Implement a true recent buffer query for `RETRIEVE_RECENT`.
- Add selected secondary operands for comparison/binding.
- Store reconstruction traces and uncertainty without treating them as observations.
- Make memory usefulness measurable through downstream counterfactual ablation.

### 12.2 External artifacts

Represent notes, diagrams, files, tool outputs, or rearranged environment state as external artifact records with:

- artifact ID and URI/handle;
- creator action receipt;
- content digest/version;
- read/write permissions;
- expected persistence;
- cost;
- provenance roots;
- last verification;
- relation to goals and hypotheses.

Creating an artifact and later observing it are separate events.

### 12.3 Tool reliability

Track tool capability, argument validity, latency, failure modes, and result reliability by regime. Tool output enters as `TOOL_OUTPUT`, not automatically as externally verified truth.

### 12.4 Acceptance gate H-M

- Learned write policy beats FIFO/recent-only controls on downstream delayed utility at matched capacity.
- Semantic memory rejects simulated, predicted, reconstructed-only, or unverified writes without sufficient consolidation authority.
- A created external artifact reduces later memory load on an environment task.
- Stale or modified artifacts are detected by version/digest mismatch.
- Tool failures update reliability and future action choice.

---

## 13. Workstream I: boundary taxonomy and persistent session training

### 13.1 Boundary taxonomy

Replace the overloaded hard-boundary interpretation with explicit boundary scope:

- `EVENT`: eventizer boundary only;
- `SEGMENT`: reset fast local cognition as configured;
- `DOCUMENT`: prevent unrelated document leakage while preserving only explicitly allowed global training state;
- `ENVIRONMENT_EPISODE`: reset scenario-local state, preserve eligible semantic/system state;
- `SESSION`: reset goals and session artifacts according to policy;
- `IDENTITY_RESET`: clear all agent-specific state except immutable audit records;
- `STREAM_DISCONTINUITY`: fail closed or reinitialize time/phase carries where continuity cannot be established.

Represent boundary class, scope, continuity key, environment ID, session ID, and reset policy explicitly.

### 13.2 Persistent trainer modes

Add trainer modes:

- independent packed documents;
- continuous within-document stream;
- multi-episode environment trajectory;
- persistent-agent curriculum;
- evaluation with frozen memory;
- continual adaptation with isolated adapter.

Reuse `_last_runtime` only when continuity keys and boundary policy authorize it. Never infer continuity merely from dataloader adjacency.

### 13.3 Replay units

Replay records must include enough state to reconstruct the decision context:

- observation/event span;
- boundary/session identifiers;
- goal and viability context;
- hypothesis posterior;
- selected and alternative actions;
- executor feedback;
- provenance offsets/digest;
- optional recurrent burn-in or state snapshot.

### 13.4 Acceptance gate I-B

- Unrelated documents exert zero measurable state influence.
- Same-session episodes preserve only declared memory/system state.
- Identity reset clears every configured agent-specific tensor row.
- Persistent training learns a delayed cross-episode task that a per-context reset control cannot solve.
- Checkpoint/resume at every boundary type is bitwise or tolerance-equivalent as declared.

---

## 14. Workstream J: supervision, curricula, and serious training authority

### 14.1 Expand objective families

The schedule should separately expose:

- primary task/language;
- spectral substrate;
- events and typed relations;
- multimodal binding;
- memory write/retrieval utility;
- compression and abstraction validity;
- reconstruction and fidelity calibration;
- world model and hypothesis likelihood;
- candidate action consequence and information gain;
- viability and constraint prediction;
- controller/metacognitive utility;
- provenance consistency;
- invariant discovery and transfer;
- continual adaptation safety.

No family may be silently substituted by an easier proxy under the same name.

### 14.2 Fail-closed training profiles

Replace a single permissive default with named profiles containing required objective families and data authorities:

- `substrate_language_pretraining`;
- `relational_event_pretraining`;
- `multimodal_grounding`;
- `reconstructive_hierarchy`;
- `world_model_trajectory`;
- `active_agent_control`;
- `invariant_transfer`;
- `continual_validation`;
- `integrated_serious_checkpoint`.

The manifest records profile, active families, missing-family count, target source, target coverage, and gradient contribution. An integrated profile fails if any mandatory family is absent.

### 14.3 Data curriculum

#### Stage 1: substrate and language

- FineWeb English or equivalent language data;
- full-vocabulary CE remains authoritative;
- spectral regularization and causal boundary tests;
- no claim of cognitive competence.

#### Stage 2: explicit relational events

- event boundaries/types beyond document boundaries;
- entity continuity, temporal order, part/whole, coreference, transformation, contradiction;
- provenance and source-type annotations;
- synthetics with exact authority plus curated real corpora.

#### Stage 3: multimodal continuity

- asynchronously sampled paired modalities;
- missing modalities and unequal latency;
- shared-event positives and typed hard negatives;
- modality-specific residual reconstruction;
- bodily/interoceptive channels where the target environment supports them.

#### Stage 4: reconstructive hierarchy

- repeated relational motifs;
- structured masking and partial traces;
- context/evidence-conditioned reconstruction;
- non-applicable abstraction negatives;
- historical-fidelity and plausibility labels.

#### Stage 5: trajectories and hypotheses

- action traces, observations, rewards, costs, constraints, terminations;
- alternative hypotheses with diagnostic evidence;
- short horizons before longer horizons;
- passive versus interventional distinctions.

#### Stage 6: active sensing and tools

- environments where information-gathering actions have measurable value;
- tool calls and failures;
- external memory artifacts;
- permission and provenance constraints.

#### Stage 7: invariant transfer

- structurally equivalent problems across identities, modalities, and domains;
- near-match counterexamples;
- held-out transformations;
- intervention-based procedural utility.

#### Stage 8: metacognition and viability

- variable compute budgets;
- tasks where more deliberation sometimes helps and sometimes wastes resources;
- explicit constraint/viability outcomes;
- calibration shift and OOD regimes.

#### Stage 9: persistent and continual adaptation

- multi-episode tasks;
- replay and isolated adapters;
- validation, revocation, and rollback;
- retention and negative-transfer tests.

### 14.4 Production-head supervision

- Train the actual action-conditioned multihorizon world model, not only a separate one-step latent predictor.
- Train actual controller decisions through integrated consequences, not private-head teacher forcing alone.
- Train hypothesis likelihood updates from observations.
- Train reconstruction source/fidelity on actual reconstructed records.
- Train invariant proposals through the integrated normalizer/matcher/ledger path.
- Train viability predictions against measured receipts.

### 14.5 Gradient governance

For every objective family, log:

- gradient norm;
- cosine with primary task;
- cosine with other cognitive families;
- active target count and coverage;
- effective loss scale;
- affected parameter groups;
- clipping and projection events.

Use slower or staged activation, PCGrad-like projection, or isolated adapters only after a measured conflict. Hard provenance and safety gates are not learned loss weights.

### 14.6 Acceptance gate J-T

- Every integrated profile fails closed on missing target families.
- Target coverage and source authority are present in the run manifest.
- Production heads receive nonzero, finite gradients from their declared objectives.
- Primary-language degradation remains within an explicit tolerance at every stage.
- Cognitive gains survive matched parameter, data, and compute controls.
- Serious-checkpoint claims require held-out integrated tasks, not just synthetics.

---

## 15. Workstream K: evidence maturity and acceptance redesign

### 15.1 Evidence levels

Extend traceability status beyond `verified/documented/external` with a separate maturity field:

1. `contract`: shapes, masks, authority, invariants, and failure behavior tested;
2. `mechanism`: a local learned component beats a matched control;
3. `integrated_loop`: production modules close the named causal loop;
4. `transfer`: held-out shift and counterexamples pass;
5. `serious_checkpoint`: target-scale trained checkpoint passes preregistered thresholds;
6. `deployment`: target hardware/environment behavior is measured.

`verified` may still mean the cited test ran and passed; it must not imply maturity level 6.

### 15.2 Replace or supplement proxy gates

- Controller: train and evaluate through integrated environment consequence, while retaining private-head unit tests only at contract level.
- Functional surprise: use a learned, calibrated critic and compare against equal-interaction controls; retain oracle critic only as a correctness oracle.
- Continual adaptation: adapt actual MRCRA adapters/memory and test real checkpoint rollback.
- Multimodal binding: exercise observation preparation, event extraction, shared graph, provenance, and missing/asynchronous modalities.
- Uncertainty/hypotheses: generate likelihoods from the production observation model.
- Hierarchy: use integrated compression, conditional reconstruction, applicability, and counterexample records.

### 15.3 Ablation matrix

At minimum run matched ablations for:

- no spectral phase/delay information;
- no reconstruction traces;
- unconditional versus evidence-conditioned reconstruction;
- no explicit reconstructed source class;
- fixed highest versus fixed lowest versus adaptive abstraction;
- one hypothesis versus multi-hypothesis;
- no information gain;
- action-before-deliberation control;
- no viability gate;
- raw content versus role-normalized invariants;
- no metacognitive routing;
- per-context reset versus authorized persistence;
- FIFO/recent memory versus learned writes;
- behavior CE versus bounded functional-surprise consequence learning;
- no provenance features while retaining ledger authority.

### 15.4 Acceptance artifact

The acceptance artifact records:

- source revision hashes;
- checkpoint digest;
- exact test node IDs;
- data revisions and split hashes;
- random seeds and confidence intervals;
- trainable parameter count per arm;
- examples/interactions/tokens and compute per arm;
- hardware and dtype;
- maturity level;
- declared claim boundary;
- failures and unresolved external gates.

---

## 16. Workstream L: performance and computational efficiency

### 16.1 Hot-path restructuring

Profile before rewriting. Expected priorities:

- batch provenance feature gathering while keeping ledger mutation authoritative;
- replace row-wise Python controller/action dispatch with grouped batched kernels;
- vectorize relation and reconstruction allocation;
- cache workspace/relation summaries until their versions change;
- share rollout prefixes across candidate actions/hypotheses;
- perform top-\(K\) routing before expensive decoders;
- run external ledger persistence asynchronously after an in-memory atomic append;
- retain exact tensor retrieval until an ANN index wins end-to-end latency at required recall.

### 16.2 Compute budgets

Track separately:

- dense carrier FLOPs/token;
- event extraction cost/token;
- cognitive cycles/event;
- microsteps/cycle;
- routed nodes, relations, hypotheses, and actions;
- reconstruction nodes/relations generated;
- rollout horizon and branches;
- provenance append/query time;
- memory retrieval candidate recall;
- executor latency;
- peak live tensor memory;
- persistent external-store growth.

### 16.3 Performance gates

Initial relative budgets, to be frozen after baseline measurement:

- dormant cognitive additions add no more than 5% to dense language-token latency;
- an ordinary event cycle adds no more than 25% over the existing cognitive cycle at matched capacities;
- reconstruction cost scales with requested local output, not full context length;
- top-\(K\) planning cost scales as \(O(K_hK_aH)\) with strict configured bounds;
- checkpoint size growth equals declared tensor state and excludes accidental duplicate model copies;
- every optimization must preserve exact or tolerance-declared authority behavior.

Absolute throughput targets should be set only after target hardware measurement. Do not optimize merely for FLOP count if wall-clock latency worsens.

---

## 17. Observability and Trackio integration

Extend cognitive telemetry with:

- reconstruction count, region size, trace count, evidence agreement, fidelity, plausibility, uncertainty, and abstention;
- selected abstraction depth and physical scale separately;
- abstraction applicability/failure reason;
- hypothesis posterior entropy, effective count, posterior shift, and routed mass;
- candidate action count, predicted reward/cost/risk/information gain, and authorization result;
- viability margin, forecast violations, reserve, and recovery actions;
- evidence-request type and resolution;
- invariant match cost, coverage, transfer result, and counterexample count;
- self-predicted versus realized error/cost/latency;
- memory write scores and downstream utility;
- boundary scope and persistence transitions;
- evidence maturity and active training-family coverage.

Add dedicated Trackio views:

1. **Reconstructive Descent:** abstraction -> traces/evidence -> reconstructed subgraph with source/fidelity overlays.
2. **Deliberation Lattice:** hypotheses × actions × horizons with posterior, value, risk, and information gain.
3. **Viability Envelope:** regulated variables, hard limits, forecasts, and intervention points.
4. **Invariant Transfer:** normalized roles, graph assignment, applicability, failures, and domain transfer.
5. **Cognitive Causal Timeline:** observation, internal operations, authorization, execution, feedback, and posterior update.

Telemetry is diagnostic, not authoritative. Dashboard state must be generated from immutable receipts and current tensor state rather than becoming a control input.

---

## 18. File-level change map

### 18.1 Existing modules requiring material changes

| File | Required change |
|---|---|
| `src/mrrn/cognitive_types.py` | source classes, actions, scoped boundaries, trigger/reason enums |
| `src/mrrn/cognitive_model.py` | reorder cycle; integrate reconstruction, validity, hypotheses, candidates, viability, invariant pipeline, structured operands |
| `src/mrrn/controller.py` | candidate generation, multiple operands, metacognitive routing, goal conflicts, cost prediction |
| `src/mrrn/interaction.py` | structured action schemas, normalized utility inputs, full feedback state transition |
| `src/mrrn/world_model.py` | hypothesis-conditioned multiaction/multihorizon predictions and likelihood interface |
| `src/mrrn/hypotheses.py` | provenance-linked evidence updates, routed selection, unknown hypothesis, posterior diagnostics |
| `src/mrrn/compression.py` | conditional residual decoder hooks and applicability outputs |
| `src/mrrn/invariants.py` | tensor/runtime integration, batched ledger bridge, transfer receipts |
| `src/mrrn/memory_v2.py` | write-policy integration, reconstruction traces, true recent tier semantics |
| `src/mrrn/provenance.py` | reconstructed/tool/artifact derivation helpers and executor receipts |
| `src/mrrn/observation.py` | executor/tool/artifact observations, boundary context, continuous-time uncertainty |
| `src/mrrn/multimodal_io.py` | asynchronous alignment metadata and bodily/viability channels |
| `src/mrrn/knowledge.py` | applicability/failure records, reconstruction validator, invariant revisions |
| `src/mrrn/uncertainty.py` | reconstruction, model-regime, and self-prediction calibration |
| `src/mrrn/cognitive_training.py` | profiles, authorized persistence, new objective families, stateful trajectories |
| `src/mrrn/cognitive_supervision.py` | production-head targets and admissible evidence providers |
| `src/mrrn/cognitive_objectives.py` | expanded families, schedules, conflict telemetry |
| `src/mrrn/cognitive_checkpoint.py` | versioned new states and migration |
| `src/mrrn/cognitive_diagnostics.py` | new loop and maturity metrics |
| `src/mrrn/empirical_acceptance.py` | integrated gates and maturity labeling |
| `src/mrrn/traceability.py` | evidence maturity separate from test verification |
| `scripts/train_fineweb.py`, `scripts/train_mrcra_fineweb.py` | canonical integrated-MRCRA default, evidence-admissible Stage-1 supervision, retained evaluation, and explicit claim boundary; sequence-only MRRN requires `--legacy-mrrn` |
| `README.md` | modes, runtime loop, training profiles, evidence levels, migration |

### 18.2 New modules

- `src/mrrn/reconstruction.py`
- `src/mrrn/abstraction_control.py`
- `src/mrrn/action_candidates.py`
- `src/mrrn/agent_session.py`
- `src/mrrn/viability.py`
- `src/mrrn/evidence_requests.py`
- `src/mrrn/external_artifacts.py`
- `src/mrrn/metacognition.py`
- `src/mrrn/boundaries.py`
- `src/mrrn/training_profiles.py`

Do not create each module merely to satisfy the file list. If two contracts remain cohesive and smaller together, combine them, but retain the specified public responsibilities and test boundaries.

---

## 19. Dependency-ordered implementation phases

### 19.1 Migration flags and removal policy

Use temporary, manifest-recorded feature flags only to make migrations reversible:

- `enable_conditional_reconstruction`;
- `enable_abstraction_validity_control`;
- `enable_post_deliberation_action_selection`;
- `enable_multi_hypothesis_planning`;
- `enable_agent_session_loop`;
- `enable_viability_gate`;
- `enable_integrated_invariant_discovery`;
- `enable_persistent_session_training`.

Each flag defaults off only until its prerequisite contract/checkpoint gate passes. A flag may default on after its integrated gate passes and must be removed after the next checkpoint-format boundary. Do not maintain old and new action-order semantics indefinitely. Every run manifest records flag values, and checkpoints refuse to resume under a behaviorally incompatible flag set unless an explicit migration exists.

Emergency disable controls may stop external execution or plasticity, but must not reinterpret already-recorded state.

### 19.2 Change-unit completion rule

Every merged change unit must include, where applicable:

- public tensor/API contract and validation;
- unit, property, integration, and negative/fail-closed tests;
- checkpoint/state migration;
- provenance behavior;
- telemetry and performance measurement;
- traceability/evidence maturity update;
- user-facing or developer documentation;
- compatibility result with all previously passing authority tests;
- a declared claim boundary.

A new class or forward method without its training authority and integrated gate is recorded as `contract`, never as a completed capability.

### Phase 0: baseline freeze and claim discipline

**Deliverables**

- Snapshot current test, empirical, parameter, checkpoint, and throughput results.
- Add evidence maturity without changing existing pass/fail facts.
- Mark FineWeb stage 1 explicitly as substrate/language training.
- Add failing/xfail integration tests describing the future causal order.

**Exit gate**

- Current tests remain passing.
- Existing artifacts can be reproduced.
- No broad capability claim depends solely on a mechanism-level test.

### Phase 1: ontology, state, checkpoint, and boundary foundations

**Deliverables**

- New source classes, structured operands, runtime states, boundary context, and checkpoint version.
- Conservative migration and round-trip tests.
- Explicit agent modes and permission registration.

**Exit gate**

- Old checkpoints migrate or fail with an exact actionable error.
- New empty states preserve current behavior when features are disabled.
- Scoped reset tests pass for every state field.

### Phase 2: reconstructive descent and abstraction validity

**Deliverables**

- Conditional graph reconstruction.
- Reconstruction provenance/fidelity.
- Applicability model and localized descent planner.
- Training tasks and gate A-R/B-H.

**Exit gate**

- Evidence-conditioned local reconstruction beats unconditional decode.
- Highest-valid-abstraction controls pass matched compute tests.

### Phase 3: hypotheses, candidate actions, and world-model reordering

**Deliverables**

- Automatic posterior updates.
- Candidate-action state and structured actions.
- Hypothesis-conditioned multihorizon rollout.
- Final selection after simulation.

**Exit gate**

- An integrated test proves deliberation reverses an initially attractive wrong action.
- Hypothesis changes alter action choice appropriately.

### Phase 4: agent session and active environment closure

**Deliverables**

- `CognitiveAgentSession`.
- Executor receipts, idempotency, evidence requests, calibration/feedback closure.
- Tool and artifact contracts.

**Exit gate**

- Full deterministic perception-action-perception loop passes across checkpoint/resume.
- Unauthorized action execution remains impossible.

### Phase 5: viability and metacognitive control

**Deliverables**

- Viability state/gate.
- Measured resource feedback.
- Self-model and reflective nodes.
- Value-of-compute and value-of-information routing.

**Exit gate**

- Hard constraints dominate utility.
- Adaptive deliberation improves utility per measured compute.

### Phase 6: integrated invariant discovery

**Deliverables**

- Normalizer/matcher/ledger production wiring.
- Counterexample search, applicability revisions, cross-domain transfer suite.

**Exit gate**

- Cross-domain held-out transfer passes with near-match rejection.

### Phase 7: persistent curricula and continual adaptation

**Deliverables**

- Authorized persistent trainer modes.
- New profiles/objective families.
- Real MRCRA adapter/memory replay, validation, revocation, rollback.

**Exit gate**

- Cross-episode learning succeeds without document leakage.
- Retention and rollback gates pass on the actual model.

### Phase 8: serious integrated checkpoint and optimization

**Deliverables**

- Train a serious checkpoint through the completed curriculum.
- Run full ablations, confidence intervals, target-hardware benchmarks, and Trackio views.
- Optimize only measured bottlenecks.

**Exit gate**

- All serious-checkpoint gates and declared performance budgets pass.
- Remaining failures are labeled external/deployment rather than hidden.

---

## 20. Test strategy

### 20.1 Contract tests

For every new dataclass/module:

- shape and dtype validation;
- masks and capacity bounds;
- deterministic empty state;
- `detach` and device/dtype transfer;
- row replacement;
- invalid pointer/ID handling;
- checkpoint round trip;
- serialization version and migration;
- provenance consistency;
- unauthorized paths fail closed.

### 20.2 Property and metamorphic tests

- Reordering unrelated inactive slots does not change active outputs.
- Permuting surface identities preserves a correct role-normalized invariant.
- Adding irrelevant evidence does not raise historical-fidelity confidence.
- Removing evidence cannot increase authoritative verification.
- Simulation scenario changes do not mutate observed state.
- Increasing hard risk cannot make an action newly eligible.
- A higher abstraction with worse validity cannot be selected over a valid lower one.
- Duplicate executor receipts do not double-apply feedback.
- Boundary resets are idempotent.
- Checkpoint/resume commutes with the next deterministic step.

### 20.3 Integrated environments

Build small deterministic environments that isolate:

- viewpoint/action disambiguation;
- hidden-state hypothesis testing;
- delayed reward and constraint tradeoffs;
- external note creation and later retrieval;
- abstraction failure at one local boundary;
- viability recovery;
- tool reliability changes;
- cross-episode dependency;
- cross-domain structural transfer.

Each environment must include matched controls and an exact authority model.

### 20.4 Serious evaluations

- long-context language retention and generation;
- asynchronous multimodal binding;
- reconstructive fidelity under partial traces;
- active information acquisition;
- multihorizon consequence calibration;
- persistent memory utility;
- invariant transfer;
- robustness to contradictory and unreliable sources;
- stability under long streams;
- latency/memory on target hardware.

---

## 21. Completion criteria

The consequential gaps are closed only when all of the following are true:

1. `DECOMPRESS` has been replaced or superseded by evidence-conditioned localized relational reconstruction.
2. Reconstructed content has explicit source, parents, fidelity, plausibility, and uncertainty.
3. Abstraction depth and physical scale are independently selected and reported.
4. The runtime selects the highest abstraction meeting a tested validity contract.
5. New evidence automatically updates all active relevant hypotheses.
6. Multiple candidate actions are simulated under multiple hypotheses before final selection.
7. External action selection occurs after internal deliberation.
8. An application-owned session closes execution and feedback without authority leakage.
9. Reward, cost, constraints, latency, success, and resultant evidence all update persistent models.
10. Viability hard gates precede utility and use measured feedback.
11. The production invariant action uses role normalization, structural matching, held-out utility, applicability, and counterexamples.
12. Goals are explicit and required in action-capable modes.
13. Self-model and reflective records can be inspected by ordinary bounded cognitive operations.
14. Memory writes use the production learned/evidential policy and semantic authority rules.
15. Boundary scope permits both document isolation and authorized persistent cognition.
16. Integrated training profiles fail closed if required supervision is absent.
17. Production heads—not proxy heads—receive and demonstrate their declared training signal.
18. Evidence artifacts declare maturity and cannot confuse local learnability with serious capability.
19. Full checkpoint/resume, provenance, causality, and existing MRRN tests remain passing.
20. The completed integrated checkpoint passes matched ablations and declared efficiency budgets.

---

## 22. Principal risks and mitigations

| Risk | Mitigation |
|---|---|
| Cognitive modules become a large per-token tax | preserve event-driven two-speed path; profile dormant overhead |
| Reconstruction produces confident confabulation | separate fidelity/plausibility, evidence conditioning, calibration, explicit source, abstention |
| Planner cost explodes | routed top-\(K\), shared prefixes, dominance pruning, hard horizon/branch budgets |
| Hypotheses collapse prematurely | diversity routing, posterior hysteresis, unknown hypothesis, diagnostic-evidence tests |
| Utility overrides safety | hard viability/permission/provenance masks before scoring |
| Persistent training leaks unrelated documents | explicit continuity keys and scoped boundaries; fail closed on ambiguity |
| Auxiliary objectives damage language | staged profiles, gradient telemetry, matched retention gates, isolated adapters where justified |
| Invariants become surface clusters | role normalization, identity shuffles, structural near-match negatives, cross-domain tests |
| Reflection loops recursively consume compute | fixed budget, cycle detection, value-of-compute gate |
| Ledger becomes a hot-path bottleneck | compact tensor features plus authoritative asynchronous persistence; never replace authority with prediction |
| Old checkpoints silently acquire false semantics | explicit format migration with unknown/empty conservative initialization |
| Tests overstate capability | evidence maturity, integrated production paths, matched controls, serious checkpoint gates |

---

## 23. Recommended first implementation increment

The first code increment should be Phase 0 plus the non-behavior-changing part of Phase 1:

1. add evidence maturity;
2. add new source and boundary ontologies without reordering old enum values;
3. define empty reconstruction, validity, candidate, viability, evidence-request, artifact, and metacognitive states;
4. add checkpoint migration and scoped reset tables;
5. add structured controller operands while adapting old single-pointer behavior through a compatibility layer;
6. add failing integration specifications for action reordering and evidence-conditioned reconstruction;
7. prove existing outputs remain unchanged when all new feature flags are disabled.

Only after that compatibility gate should the live reconstruction and reordered deliberation paths be introduced. This provides a narrow, reversible foundation for the rest of the plan without training around temporary or contradictory semantics.
