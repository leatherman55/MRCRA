# MRCRA Consequential-Gap Closure Report

**Implementation version:** 4.0.0  
**Runtime checkpoint format:** 5  
**Training checkpoint format:** 7  
**Status:** production implementation complete; serious-checkpoint and deployment evidence remain external  
**Canonical plan:** `outputs/mrcra_consequential_gaps_implementation_plan.md`

## Outcome

The production MRCRA now implements the complete bounded cognitive substrate
described by the closure plan. The neural runtime can propose, reconstruct,
simulate, and justify; immutable ledger state, host permissions, hard viability
envelopes, and the application executor retain authority. Compatibility defaults
remain available, while `MRCRAConfig.serious_120m()` enables the integrated
mechanisms and still begins with no external capability or permission.

The canonical serious actor contains 115,925,944 trainable parameters, within the
declared 110M–125M range. The count is structural—no padding parameters—and is
recorded in `outputs/mrcra_120m_parameter_report.json`.

## Implemented closure matrix

| Plan area | Production closure | Principal evidence |
|---|---|---|
| Reconstruction | Evidence-conditioned, localized typed graph reconstruction with requested scale, depth, support and precision; immutable `RECONSTRUCTED` provenance | `reconstruction.py`, `test_reconstruction.py`, `test_cognitive_actions.py` |
| Abstraction control | Applicability, distortion, provenance and precision validity; highest-valid selection, localized descent and conservative revision | `abstraction_control.py`, `test_abstraction_control.py` |
| Deliberation | Observation likelihood updates; explicit unknown hypothesis; posterior/diversity top-K routing; multi-action/multi-hypothesis rollouts; final selection after internal operations | `hypotheses.py`, `action_candidates.py`, `world_model.py`, `test_action_deliberation.py` |
| Agent loop | Observe → deliberate → execute → ingest; structured schemas; explicit capabilities/permissions/goals; idempotent receipts; checkpoint-safe pending execution | `agent_session.py`, `test_agent_session.py` |
| Feedback closure | Persistent success, latency, reward, cost, constraint violation, reversibility, executor reliability, resultant evidence and goal progress | `interaction.py`, `agent_session.py`, `test_agent_session.py` |
| Evidence/tools/artifacts | Bounded evidence-request lifecycle; tool queries are requests, never neural execution; host-only artifact recording with exact version/digest verification | `evidence_requests.py`, `external_artifacts.py`, `test_evidence_artifacts.py` |
| Viability | Measured state, forecast uncertainty and hard envelope authorization before utility; unknown authority abstains | `viability.py`, `test_viability.py` |
| Invariants | Role normalization, permutation-aware structural matching, applicability, code gain, independent provenance roots and near-match counterexamples | `invariants.py`, `test_compression_invariants.py` |
| Reflection | Complete measured system model, bounded metacognitive history, provenance-backed `SYSTEM_STATE` graph nodes, and bounded marginal-value routing into the live controller without bypassing hard masks | `metacognition.py`, `cognitive_model.py`, `controller.py`, `test_cognitive_reasoning.py` |
| Memory | Learned/evidence-backed episodic write policy, completion-time recent retrieval, semantic consolidation authority and reconstructed-source exclusions | `memory_v2.py`, `cognitive_model.py`, `test_cognitive_actions.py` |
| Boundaries | Explicit EVENT/SEGMENT/DOCUMENT/ENVIRONMENT_EPISODE/SESSION/IDENTITY_RESET/STREAM_DISCONTINUITY persistence matrix; partial identity reset fails closed | `boundaries.py`, `test_scoped_boundaries.py` |
| Training authority | Named profiles, required data authorities and target coverage, explicit continuity keys, and demonstrated production reconstruction/world/metacognitive-head gradients from measured targets | `training_profiles.py`, `cognitive_objectives.py`, `cognitive_supervision.py`, `test_training_profiles.py`, `test_production_objectives.py`, `test_cognitive_supervision.py` |
| Default training entrypoint | `train_fineweb.py` selects the 115.9M integrated MRCRA, exact retained evaluation, Cognitive Atlas diagnostics and format-7 checkpoint path by default; sequence-only MRRN is an explicit compatibility mode | `train_fineweb.py`, `train_mrcra_fineweb.py`, `test_fineweb_entrypoint.py` |
| Continual adaptation | Exact adapter allowlist, bounded provenance replay, base-drift detection, retention commit and bit-exact rollback | `continual_adaptation.py`, `test_continual_adaptation.py` |
| Gradient governance | Per-family gradient norms/cosines and deterministic conflict projection without mutating live gradients | `gradient_governance.py`, `test_gradient_governance.py` |
| Checkpoints | Format-5 full runtime persistence and conservative v3/v4 migration; separate format-7 training persistence with v3–v6 migration, retained-evaluation digest binding and deterministic prefetch persistence | `cognitive_checkpoint.py`, `cognitive_training.py`, checkpoint tests |
| Serious evidence authority | Typed exact-task evidence construction; checkpoint/data/evaluation SHA binding; independently recomputed Wilson, criterion and 32K hardware-budget decisions; partial or malformed evidence fails closed | `serious_acceptance.py`, `run_mrcra_serious_checkpoint_acceptance.py`, `test_serious_acceptance.py` |
| Diagnostics | Schema-v4 cognitive evidence for reconstructions, hypotheses, candidates, requests, viability, live metacognitive route values, reflective history, boundaries and consequence measurements | `cognitive_diagnostics.py`, Trackio snapshot path |
| Integrated ablations | All 15 plan-required repeated-seed matched production-path comparisons, with Wilson confidence intervals, per-arm work declarations, source/data identity and unresolved-gate reporting | `integrated_acceptance.py`, `test_integrated_acceptance.py` |
| Performance budgets | Single-thread process-CPU-time ABBA/BAAB paired-median dormant/event-cycle latency with dispersion telemetry, exact local reconstruction bounds, exact planning-lattice bounds and duplicate-free runtime checkpoint structure | `performance_acceptance.py`, `test_performance_acceptance.py` |
| Cognitive visualization | Dedicated Trackio MRCRA Cognition tab with reconstructive descent, deliberation lattice, viability envelope, invariant transfer and causal timeline views | `CognitiveArchitecture.svelte`, `cognitiveViews.js` |

## Acceptance evidence

The authoritative manifest is `outputs/mrcra_acceptance_manifest.json`.

- Python acceptance: 517 tests pass in the normal hash-verifying run.
- Hash-building manifest run: 516 passes and one intentional skip: the test that
  cannot verify a manifest while that manifest is being replaced.
- Trackio frontend: 51 tests pass; lint and production build pass.
- Bounded learned-behavior suite: all eight preregistered tasks pass, including
  explicit ablations for retrieval, multimodal binding, hierarchical
  compression, uncertainty/hypotheses, intervention, adaptive compute,
  functional surprise and continual replay/rollback.
- Integrated production-path suite: all 15 matched ablations pass across 16
  unique seeds with preregistered Wilson lower-bound criteria.
- Relative performance suite: all five initial budgets pass on the local CPU;
  target-hardware absolute throughput remains outside this artifact's claim.
  The hash-bound v2 artifact uses symmetric within-repeat pairing and process
  CPU time, and retains medians, median absolute deviations, exact thresholds,
  and independently recomputed decisions instead of promoting one noisy local
  timing sample into a durable architectural claim.
- MRCRA traceability: all 174 specification headings are mapped with no missing
  or invalid entries. 170 are executable verified contracts and four are
  deliberately documented rather than overstated.
- Base MRRN traceability: all 189 headings are executable verified.

## Claim boundary

This evidence establishes a complete production implementation, contract
correctness, local mechanism learnability, integrated deterministic loops,
checkpoint equivalence, and fail-closed authority behavior. It does **not** turn
an untrained actor into a serious cognitive checkpoint.

Completion criterion 20 of the plan remains an external empirical gate: train
the 115.9M actor through the declared multi-stage data curriculum, then run
matched end-task ablations, confidence intervals, long-context evaluations and
target-hardware efficiency measurements through the typed evidence producer and
independent serious-checkpoint auditor. CUDA characterization was explicitly
excluded from local implementation acceptance by project direction, so this
external measurement cannot be fabricated locally. Until a real checkpoint and
those artifacts exist, the maximum justified maturity is
`integrated_loop`/bounded `transfer`, not `serious_checkpoint` or `deployment`.
