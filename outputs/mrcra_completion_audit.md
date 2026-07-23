# MRCRA Consequential-Gap Completion Audit

**Audit basis:** `outputs/mrcra_consequential_gaps_implementation_plan.md`  
**Implementation:** MRCRA 4.0.0  
**Runtime checkpoint format:** 5  
**Training checkpoint format:** 6  
**Serious actor:** 115,925,944 trainable parameters  
**Audit conclusion:** criteria 1–19 are implemented and locally evidenced; criterion 20 remains a deliberately unresolved serious-checkpoint/deployment gate.

## Completion-criterion ledger

| # | Required outcome | Status | Production implementation and proof |
|---:|---|---|---|
| 1 | Evidence-conditioned localized relational reconstruction supersedes decompression | Verified locally | `reconstruction.py`, `cognitive_model.py`; reconstruction contract/action tests and `reconstruction_trace_conditioning` plus `evidence_conditioned_reconstruction` ablations |
| 2 | Explicit source, parents, fidelity, plausibility, uncertainty | Verified locally | `ReconstructionState` and ledger-backed reconstruction records; fail-closed source-class and reconstruction tests |
| 3 | Abstraction depth and physical scale independently selected/reported | Verified locally | `abstraction_control.py`, runtime state, schema-v4 diagnostics |
| 4 | Highest abstraction satisfying measured validity is selected | Verified locally | `AbstractionLevelSelector`; `adaptive_abstraction_selection` matched controls |
| 5 | Evidence updates all active relevant hypotheses | Verified locally | observation likelihood update and hypothesis bank integration tests |
| 6 | Multiple actions are simulated under multiple hypotheses before selection | Verified locally | bounded hypothesis-action-horizon lattice; `posterior_multi_hypothesis_deliberation` |
| 7 | External selection occurs after internal deliberation | Verified locally | post-deliberation runtime branch and `post_deliberation_action_selection` |
| 8 | Application session closes execution/feedback without authority leakage | Verified locally | `CognitiveAgentSession`, structured executor, idempotent receipt tests |
| 9 | Reward/cost/constraint/latency/success/evidence update persistent models | Verified locally | system-model feedback integration and agent-session tests |
| 10 | Measured hard viability gates precede utility | Verified locally | `ViabilityGate`; `hard_viability_authorization` |
| 11 | Production invariants use role normalization, matching, utility, applicability, counterexamples | Verified locally | `invariants.py`, knowledge validation, `role_normalized_invariant_transfer` |
| 12 | Action-capable modes require authorized explicit goals | Verified locally | agent-mode/goal authority contracts and session fail-closed tests |
| 13 | Self-model/reflective records are inspectable by bounded cognitive operations | Verified locally | `INSPECT_SELF_STATE`, provenance-backed system-state nodes, metacognitive records |
| 14 | Learned/evidential memory writes and semantic authority rules are live | Verified locally | `MemoryWritePolicyV2`, consolidation rules, `learned_evidential_memory_write` |
| 15 | Document isolation and authorized persistence coexist | Verified locally | boundary scope matrix, idempotent resets, `authorized_cross_context_persistence` |
| 16 | Integrated profiles fail closed without required supervision | Verified locally | training profiles, target-coverage authority, partial-group rejection tests |
| 17 | Production heads receive demonstrated declared signals | Verified locally | measured reconstruction/world/controller/metacognitive targets; nonzero finite gradients on the live heads |
| — | Canonical FineWeb entrypoint selects the integrated architecture | Verified locally | `scripts/train_fineweb.py` delegates to the MRCRA trainer by default; CLI smoke proves all integrated flags, retained evaluation, live runtime/provenance and format-7 checkpoint binding; legacy MRRN requires `--legacy-mrrn` |
| 18 | Evidence maturity cannot imply serious capability from local tests | Verified locally | maturity taxonomy, checkpoint-null acceptance schema, claim boundary and unresolved-gate fields |
| 19 | Checkpoint/resume, provenance, causality and prior MRRN tests pass | Verified locally | format-5 runtime and format-7 training round trips/migrations, retained-evaluation digest enforcement, deterministic prefetch persistence, side-effect-free exact held-out evaluation, hash-bound normal suite and exact traceability audit |
| 20 | A completed trained checkpoint passes end-task ablations and efficiency budgets | External gate | A fail-closed typed producer/auditor now requires one exact format-7 serious checkpoint, pinned disjoint data, all nine held-out matched tasks with recomputed confidence/criteria, and recomputed 32K hardware budgets. No qualifying checkpoint or target-hardware artifact exists yet. |

## Required matched-ablation matrix

The production-path artifact runs 16 unique seeds per comparison. Every arm has
16/16 successes and a 95% Wilson lower bound of 0.806, exceeding the declared
0.75 threshold.

| Plan control | Artifact result |
|---|---|
| no spectral phase/delay | `spectral_phase_delay_information` — pass |
| no reconstruction traces | `reconstruction_trace_conditioning` — pass |
| unconditional reconstruction | `evidence_conditioned_reconstruction` — pass |
| no explicit reconstructed source | `explicit_reconstructed_source_class` — pass |
| fixed high/low abstraction | `adaptive_abstraction_selection` — pass |
| one hypothesis | `posterior_multi_hypothesis_deliberation` — pass |
| no information gain | `information_gain_deliberation` — pass |
| action before deliberation | `post_deliberation_action_selection` — pass |
| no viability authority | `hard_viability_authorization` — pass |
| raw identity-aligned invariant | `role_normalized_invariant_transfer` — pass |
| no metacognitive routing | `metacognitive_operation_routing` — pass |
| context reset instead of authorized persistence | `authorized_cross_context_persistence` — pass |
| FIFO/recent memory writes | `learned_evidential_memory_write` — pass |
| behavior CE instead of bounded functional surprise | `functional_surprise_consequence_learning` — pass |
| zero provenance features with ledger authority retained | `provenance_feature_ablation` — pass |

## Acceptance state

- Normal Python suite: **491 passed, 8 expected MLX skips**.
- Hash-building Python suite: **490 passed, 9 expected skips**; the extra skip is
  the self-referential manifest-verification test during atomic replacement.
- Trackio frontend: **51 passed**, lint passed, production build passed.
- Learned mechanism suite: **8/8 passed**.
- Integrated matched-ablation suite: **15/15 passed**.
- Relative/structural performance gates: **5/5 passed**; the current hash-bound
  artifact uses single-thread process CPU time and within-repeat ABBA/BAAB
  pairing, retains median absolute deviation and exact thresholds, while local
  reconstruction, planning-lattice size, and runtime checkpoint structure are
  bounded exactly.
- MRCRA traceability: **174/174 headings mapped**, 170 executable verified and
  four explicitly documented; the normal executable audit passes.

## Non-negotiable claim boundary

The current maximum justified maturity is `integrated_loop` (with bounded local
`mechanism` evidence). The implementation is ready for serious training, but it
is not itself a trained cognitive checkpoint. Criterion 20 can change status
only when a real checkpoint digest, immutable data/split revisions, training
compute, held-out end-task results, matched checkpoint ablations, long-context
results, and target-hardware measurements pass the typed serious-evidence
producer and independent auditor. No synthetic fixture, hand-set `passed` flag,
partial task suite, or untrained-module test may satisfy that gate.
