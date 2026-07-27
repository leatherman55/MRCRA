# MRCRA 8.4M Training-Execution Repair: Comprehensive Implementation and Empirical Acceptance Plan

## Document status

This is the implementation authority for correcting the measured 8.4M MRCRA
training regression without weakening the architecture, changing exact
next-token authority, silently disabling cognition, or relabeling auxiliary
work as corpus tokens.

This document is a plan, not a claim that the repair is already implemented.
No workstream may be called complete until its named correctness, continuity,
resource, and throughput gates have passed on the applicable execution
devices. Mechanism-level microbenchmarks are necessary but are not sufficient
evidence of end-to-end improvement.

## 1. Executive objective

The repaired default path must retain all of the following:

1. the complete 8.4M MRCRA actor, including the five-scale MRRN carrier and the
   integrated cognitive system;
2. exact full-vocabulary next-token cross entropy as the primary environmental
   pressure;
3. target-bijective, document-local causal authority;
4. Causal Spectral Target Multiplexing (CSTM) as a real learning pressure rather
   than a dashboard-only observer;
5. conflict projection and subsystem-relative caps for CSTM pressure applied
   to the carrier and cognition;
6. deterministic, checkpoint-resumable training;
7. portable behavior on CPU, Apple Silicon, CUDA, and MPS when those devices
   are available;
8. honest accounting that distinguishes corpus tokens, valid next-token
   targets, spectral target views, coefficient targets, and constituent-token
   participations.

The repaired path must remove the accidental costs that currently dominate
training:

- repeated full-model CSTM vector-Jacobian products inside every physical
  document cohort;
- repeated carrier recomputation caused by combining those CSTM products with
  whole-carrier activation checkpointing;
- an activation policy that assumes a fixed 2 GiB host budget even on a
  16 GiB unified-memory machine;
- 27-34% right-padding waste from coarse static buckets;
- exact-signature cohort fragmentation;
- Python and tensor-tree work inside hot carrier execution;
- performance claims based only on isolated custom-operation tests.

## 2. Measured baseline and the causal diagnosis

The following measurements are the starting evidence and must be captured in a
machine-readable baseline artifact before changing production behavior:

| Path | 32K training time | Valid tok/s | Important condition |
|---|---:|---:|---|
| Pre-optimization tail | about 76.2 s | about 440 | old integrated path |
| New carrier, CSTM not active | about 65.4 s | about 501 | whole-span checkpointing |
| New carrier, CSTM active | about 109.5 s | about 299 | second backward active |
| Diagnostic repaired CE path | 39.5 s | 828.5 | retained activations, dense buckets |

Additional measured facts:

- carrier forward time improved from roughly 27 seconds to roughly 14 seconds;
- backward increased from roughly 48 seconds to 93 seconds when CSTM became
  active;
- the active CSTM step constructed 12,234,624 coefficient targets and 310,422
  constituent-token participations;
- the default planner processed 45,056-49,280 physical positions for 32,768
  real document tokens;
- a tighter aligned bucket family reduced one matched context to 36,992
  physical positions;
- retained activations improved a controlled 8K path from 527 to 670 tok/s;
- four CPU intra-op threads were at least as good as eight on the tested M1;
- exact vocabulary loss occupied about three seconds and was not the dominant
  bottleneck;
- Trackio was disabled in the measured regression and was not its cause.

The baseline artifact must record raw per-step values, not only these rounded
summaries. It must include the source run IDs, model and tokenizer digests,
hardware fingerprint, Torch version, resolved execution policy, and the exact
metric keys used.

## 3. Non-negotiable semantic invariants

Every workstream must be tested against these invariants.

### 3.1 Language authority

- Every valid packed next-token target occurs exactly once in the training
  objective.
- No cross-document transition becomes valid because documents were regrouped.
- Padding contributes zero language loss and zero state/event authority.
- The exact vocabulary distribution is unchanged by execution optimization.
- CSTM never increments `tokens_seen`, `valid_targets_seen`, or any corpus-token
  counter.

### 3.2 Causality and continuity

- No state, gradient, spectral target, event, provenance record, or cognitive
  residual may move between independent documents.
- A document retains a stable physical row for all of its TBPTT spans.
- Future-token changes cannot affect prior carrier coefficients, cognitive
  states, CSTM predictions, or language logits.
- Resume must restore the exact next packed batch, schedule cursor, sampler
  state, optimizer state, and execution-policy provenance.

### 3.3 Cognition

- The integrated cognition path remains active by default.
- CSTM pressure must reach both carrier and cognitive parameters on scheduled
  substrate-pressure updates.
- CSTM predictor-only updates must not falsely count as substrate pressure.
- No phase-transition metric may directly select, gate, or reward CSTM
  pressure.

### 3.4 Numerical and optimization authority

- The retained, selectively checkpointed, and whole-span checkpointed paths
  must have matched forward values, continuation state, and first-order
  gradients within declared dtype-specific tolerances.
- CSTM sampling must have a mathematically explicit relation to the dense CSTM
  objective.
- Gradient conflict projection and cap enforcement remain fail-closed.
- Non-finite values abort the authoritative update; telemetry failure does not.

### 3.5 Portability

- The eager portable implementation remains the semantic reference.
- A compiled or custom backend may become authoritative only after matched
  forward, gradient, state, padding, and resume tests pass.
- No training feature may exist only on CUDA. CUDA-specific acceleration may
  be optional, but CPU and Apple Silicon must retain the same feature and
  objective semantics.

## 4. Target production architecture

The repaired trainer will have four separately testable authorities:

1. **semantic authority**: model, tokenizer, data partition, exact language
   objective, cognition, and CSTM objective definition;
2. **execution policy**: activation retention/checkpointing, document cohort
   plan, compiler/backend, thread count, and loss implementation;
3. **auxiliary sampling authority**: the checkpointed CSTM schedule and its
   inclusion weights;
4. **observation authority**: metrics, snapshots, evaluation, and Trackio,
   which can observe but cannot silently alter the first three.

These authorities must not be collapsed into a single configuration digest.
Execution-only changes must be resumable when proven equivalent; genuine
objective/sampling changes must be explicit and identity-bound.

---

## 5. Workstream A: End-to-end performance evidence before optimization

### A1. Add a reproducible packed-context benchmark fixture

Create a deterministic, text-free packed-token fixture representing the
observed FineWeb document-length distribution. Store:

- input IDs;
- labels;
- UTF-8 byte lengths;
- input and target segment IDs;
- source-URI declarations using synthetic fixture URIs;
- the original valid-target digest;
- document-length histogram;
- expected document count and boundary count.

The fixture must contain no recoverable source text. It should have:

- a small 1K version for unit/integration tests;
- an 8K version for frequent performance tests;
- a 32K production benchmark version.

### A2. Build an isolated benchmark runner

Add `scripts/benchmark_mrcra_training_execution.py` and
`src/mrrn/training_execution_acceptance.py`.

Each variant must execute in a fresh subprocess so allocator state, compiled
graph caches, memory peaks, and thread pools do not leak across variants.
Variants must:

1. construct the same 8.4M actor from the same seed;
2. load the same initial state dictionary;
3. consume the same packed batch;
4. disable evaluation, checkpoint saving, dashboards, and network access unless
   those systems are the feature under test;
5. perform one unmeasured warmup when the backend requires it;
6. measure at least three production steps or three isolated repeats;
7. report median, minimum, maximum, median absolute deviation, and raw samples;
8. synchronize the selected device at every timing boundary;
9. measure forward, loss, primary backward, CSTM head backward, CSTM substrate
   backward, gradient governance, optimizer, and unattributed time separately;
10. record peak allocated/resident memory and swap delta.

### A3. Establish named baselines

The runner must expose immutable variant names:

- `legacy_serial_checkpoint_dense_cstm`;
- `static_coarse_checkpoint_ce`;
- `static_coarse_checkpoint_dense_cstm`;
- `static_auto_ce` (measured retain/selective/whole-span policy);
- `static_auto_repaired_cstm`;
- `static_cost_model_auto_repaired_cstm`;
- `compiled_cost_model_auto_repaired_cstm`.

The names must describe actual resolved behavior. A requested policy that falls
back must be reported as a different resolved variant.

The compiler candidate is an optional execution authority, not a mandatory
successful sample. It runs in its own process under a hard wall-clock budget.
If it does not finish the matched warmup and measurements within that budget,
the process is terminated and reaped, and a schema-validated timeout receipt
records the requested backend, elapsed time, output digests, and the retained
eager resolution. The runner must never manufacture an eager sample under the
compiled name. Production acceptance then contains the six completed eager
samples plus either the genuine compiled sample or this explicit rejection
receipt.

### A4. Evidence output

Write:

- `outputs/mrcra_training_execution_baseline.json`;
- `outputs/mrcra_training_execution_acceptance.json`;
- `outputs/mrcra_training_execution_acceptance.md`;
- optional raw JSONL under the selected run output directory.

The JSON schema must have an explicit version and include all acceptance
criteria with `measurement`, `threshold`, `direction`, `unit`, and `passed`.

---

## 6. Workstream B: Activation-memory policy and checkpoint repair

### B1. Separate model semantics from activation execution

Introduce an execution-policy structure, tentatively:

```text
CarrierActivationExecutionPolicy
  requested: auto | retain | selective | whole_span
  resolved: retain | selective | whole_span
  calibration_kind
  available_memory_bytes
  required_reserve_bytes
  measured_or_estimated_peak_bytes
  selective_partitions
  hardware_fingerprint
  torch_version
  policy_schema_version
```

Do not use the current boolean `carrier.activation_checkpointing` as a semantic
model identity field. Retain it only as a format-15 compatibility input until
all call sites migrate.

### B2. Replace the fixed 2 GiB host rule

Automatic policy selection must use:

- current available memory, not only installed capacity;
- a required OS/application reserve;
- the maximum physical cohort shape selected by the planner;
- parameter, gradient, optimizer, exact-loss, CSTM, and retained-state storage;
- actual device peak APIs where available;
- a conservative fallback estimate when peak APIs are unavailable.

Device-specific observation:

- CPU/macOS: subprocess RSS sampler plus `psutil.virtual_memory().available`
  when installed; otherwise documented `sysctl`/`resource` fallbacks;
- CUDA: reset and read peak allocated/reserved memory;
- MPS: use current/driver allocation APIs where supported and host available
  memory because memory is unified;
- unsupported devices: conservative estimate and whole-span fallback.

Calibration must never update weights, optimizer state, CSTM running
statistics, data-stream position, or RNG state. The safest implementation is a
fresh subprocess or a cloned calibration model.

### B3. Candidate policies

Measure at least:

1. `retain`: no carrier recomputation;
2. `selective`: checkpoint only the activation-dominant carrier partitions;
3. `whole_span`: current safe fallback.

Selective partitions must be selected from an explicit saved-tensor census,
not guessed from parameter count. Start by measuring each scale/layer
partition's retained storage and recomputation time. The policy optimizer
chooses the lowest predicted step time satisfying the memory reserve.

### B4. Safe fallback

- Calibration rejection or insufficient reserve selects `whole_span`.
- An explicit user request that violates the measured safety reserve must fail
  before the real optimizer step unless an equally explicit unsafe override is
  provided.
- Runtime OOM recovery is not the normal policy mechanism. If a recoverable OOM
  occurs before any optimizer mutation, clear transient state and retry once
  with the next safer policy. Otherwise abort with the batch and checkpoint
  intact.

### B5. Resume semantics

Bump the training checkpoint format from 15 to 16 and split:

```text
identity.semantic
identity.optimization
execution_policy
execution_policy_history
```

Activation policy and compiler backend belong to `execution_policy`, not
semantic identity. A resume may change them after equivalence validation. The
checkpoint must append:

```text
step
old_policy_digest
new_policy_digest
reason
equivalence_receipt_digest
```

Format-15 migration must remove `activation_checkpointing` from the semantic
comparison while preserving every learned tensor and optimizer value.

### B6. Tests

Add or extend:

- `tests/test_activation_execution_policy.py`
- `tests/test_carrier_execution.py`
- `tests/test_cognitive_training.py`
- `tests/test_fineweb_entrypoint.py`

Required tests:

1. `test_auto_policy_selects_retain_when_measured_peak_fits_reserve`
2. `test_auto_policy_selects_selective_before_whole_span`
3. `test_auto_policy_fails_safe_when_memory_observation_is_unavailable`
4. `test_calibration_does_not_mutate_model_optimizer_rng_or_stream`
5. `test_retain_selective_and_whole_span_match_float64_forward_state_and_gradients`
6. `test_checkpoint_policy_override_resumes_identical_next_batch`
7. `test_format15_checkpoint_migrates_activation_policy_out_of_semantic_identity`
8. `test_execution_policy_history_is_append_only_and_digest_bound`
9. `test_unsafe_explicit_retain_request_fails_before_optimizer_mutation`
10. `test_peak_memory_measurement_is_finite_nonnegative_and_device_named`

Numerical gates:

- float64 forward/state maximum absolute error: `<= 2e-10`;
- float64 parameter/input gradient maximum absolute error: `<= 5e-9`;
- float32 forward/state: `atol <= 3e-5`, `rtol <= 3e-4`;
- float32 gradient cosine similarity: `>= 0.99999`;
- exact equality for masks, positions, digests, and counters.

---

## 7. Workstream C: Cost-aware document-major static batching

### C1. Preserve the current intermediate representation

Keep `DocumentSpan`, `DocumentSequence`, `StaticDocumentSpanBatch`,
`StaticDocumentCohort`, `DocumentTargetAuthority`, and
`TargetBijectionReceipt`. Extend them with a plan-cost receipt rather than
discarding the already tested target-bijection layer.

### C2. Replace exact-signature grouping

The current planner groups only identical full bucket signatures. Replace this
with deterministic bounded cohort optimization:

1. group sequences by span count, because rows with no valid token in a span
   cannot currently enter the integrated carrier;
2. sort signature vectors lexicographically with document order as the stable
   tiebreaker;
3. consider bounded contiguous cohort candidates;
4. use the elementwise maximum aligned span length as the cohort's padded
   signature;
5. reject candidates exceeding the measured token or activation-memory budget;
6. use dynamic programming to choose the minimum-cost partition;
7. retain document order within every selected cohort.

This permits nearby lengths to share a cohort without requiring them to land in
the same coarse power-of-two bucket.

### C3. Device-calibrated cost model

Introduce:

```text
DocumentExecutionCostModel
  fixed_invocation_seconds
  token_seconds_by_length_band
  backward_multiplier_by_activation_policy
  shape_compile_cost
  memory_bytes_by_shape
  calibration_digest
```

Candidate cohort cost is:

\[
C(G)=\sum_s K(B_G,L_{G,s},P)
     + \lambda_{\mathrm{compile}}N_{\mathrm{new\ shapes}}
     + \lambda_{\mathrm{memory}}M(G),
\]

where \(B_G\) is cohort batch size, \(L_{G,s}\) the aligned padded span length,
and \(P\) the activation policy.

The production objective is the sum of cohort costs, subject to target
bijection, stable rows, and memory constraints. The planner must not optimize
padding efficiency alone; invocation overhead and shape proliferation are real
costs.

### C4. Candidate alignment

Use carrier-aligned 64-token static tail refinement while preserving the
128-token cognition cadence through exact final-partial-stride event masks.
This avoids requiring every document tail to pad through a complete cognition
interval. The candidate family is:

```text
64, 128, 192, ..., 3968, 4032, 4096
```

The cost model may coarsen this family. The list is not itself the final
authority.

### C5. Plan cache

Cache only shape/cost decisions, never batches or target tensors. Cache keys:

- device and Torch fingerprint;
- actor configuration digest;
- activation policy digest;
- span-count and exact length multiset;
- cognition stride and TBPTT length;
- compiler policy.

All cached plans must be revalidated through a fresh target-bijection receipt.

### C6. Telemetry

Log:

- valid and physical tokens;
- padding efficiency;
- logical spans;
- physical invocations;
- unique static shapes;
- predicted and actual execution seconds;
- prediction error;
- selected cohort costs;
- rejected candidates by memory;
- planner and cache time;
- target receipt digest.

### C7. Tests

Extend `tests/test_document_batching.py` and add
`tests/test_document_cost_planner.py`.

Required tests:

1. randomized target bijection for at least 500 generated document mixtures;
2. exact preservation of labels, byte lengths, segments, ordinals, boundaries,
   and source declarations;
3. stable rows across every multi-span cohort;
4. no padded event, loss, provenance, or state authority;
5. deterministic identical plan/digest across repeated construction;
6. cost-model partition equals brute-force optimum on small fixtures;
7. memory-infeasible candidates are never selected;
8. corrupted cache entries fail validation and are ignored;
9. cost calibration cannot change training state;
10. dense/coarse/adaptive plans produce matched CE and gradients;
11. batched execution matches every document executed alone;
12. adversarial length distributions do not produce unbounded plan time;
13. 32K FineWeb-profile fixture reaches at least 0.85 padding efficiency unless
    a measured invocation-cost receipt proves a faster lower-efficiency plan.

Performance tests must assert actual step time, not only padding or invocation
ratios.

---

## 8. Workstream D: CSTM pressure-preserving execution redesign

### D1. Do not solve the regression by disabling CSTM

CSTM remains enabled in the canonical architecture. The repair separates:

1. **predictor learning**: teaches the small CSTM head to decode future spectral
   obligations from stopped carrier/cognitive features;
2. **substrate pressure**: propagates selected CSTM gradients into the carrier
   and cognition under the existing conflict and cap authority.

Predictor learning may be frequent and inexpensive. Substrate pressure is
scheduled and importance-weighted so the trainer does not perform a full
auxiliary VJP in every physical cohort.

### D2. Define the dense objective exactly

Let \(\mathcal O(x)\) be all valid CSTM obligations in context \(x\). An
obligation identifies:

- cohort;
- physical TBPTT span/invocation within that cohort;
- scale;
- horizon;
- coefficient row;
- component/code entries;
- valid mass \(w_o\);
- differentiable loss sum \(S_o(\theta)\).

The current dense normalized objective is:

\[
L_{\mathrm{CSTM}}(x,\theta)
=
\frac{\sum_{o\in\mathcal O(x)}S_o(\theta)}
     {\sum_{o\in\mathcal O(x)}w_o}.
\]

This definition, including horizon weights and target-RMS standardization, must
remain the mathematical reference.

### D3. Importance-weighted sampled substrate objective

Construct a checkpoint-stable `CSTMSamplingPlan` before executing the context.
For sampled obligation group \(J\) with probability \(p_J>0\):

\[
\widehat L
=
\frac{S_J}
     {W\,p_J},
\qquad
W=\sum_o w_o.
\]

Then:

\[
\mathbb E[\widehat L]=L_{\mathrm{CSTM}},\qquad
\mathbb E[\nabla\widehat L]=\nabla L_{\mathrm{CSTM}}
\]

before nonlinear conflict projection and cap enforcement.

If substrate pressure is active with probability \(q\), use:

\[
\widetilde L = \frac{D}{q}\widehat L,\qquad D\sim\mathrm{Bernoulli}(q).
\]

The counter-based sampler must be a pure function of:

- training seed;
- optimizer step;
- packed-context target digest;
- schedule schema version.

It must not consume the global Torch RNG. Resume therefore reconstructs the
same decision without saving a large RNG object.

The documentation and telemetry must state the important limitation:
importance weighting makes the **pre-governance** gradient estimator unbiased.
Conflict projection and norm caps are nonlinear safety authorities; the
post-governance update is intentionally bounded and is not claimed to be an
unbiased dense-gradient estimator.

### D4. Hierarchical sampling

Sample hierarchically:

1. whether this step has substrate pressure;
2. one eligible physical invocation (cohort plus TBPTT span), with probability
   proportional to its valid CSTM mass;
3. a bounded set of scales/horizons;
4. a bounded set of coefficient rows if the selected group still exceeds the
   target budget.

Use exact inclusion probabilities. For sampling without replacement, use
Horvitz-Thompson inclusion weights, not a naïve mean of selected rows.

All valid groups must have nonzero long-run inclusion probability. The sampler
must publish coverage counters and fail if any configured scale/horizon is
starved beyond its declared maximum gap.

### D5. Predictor-only path

For non-substrate obligations:

- detach carrier features and cognitive residuals;
- train only `cstm_predictor` parameters;
- apply the CSTM-head auxiliary-only cap;
- do not retain or traverse the carrier/cognitive graph;
- truthfully report `substrate_gradient_applied=0`.

Predictor target construction and running-statistics updates should use the
same bounded sampling plan. Running second moments must be updated with
inclusion-aware estimates or with a separately justified uniform sample.

### D6. One substrate VJP maximum per context

The production default must perform no more than one CSTM substrate VJP in an
optimizer context. It must not call `torch.autograd.grad` once per physical
cohort.

Selecting one physical invocation rather than retaining losses across an entire
multi-span cohort is intentional: it preserves the existing TBPTT graph-release
boundary and prevents the sampling repair from exchanging compute savings for
unbounded graph retention.

The initial candidate duty cycle is one substrate update per four optimizer
steps. The final resolved duty cycle and target budget are selected during
startup performance calibration and then frozen for the run. They become part
of optimization identity and checkpoint state.

Automatic mid-run timing feedback must not continually change objective
sampling. A safety watchdog may reduce or disable auxiliary execution after a
resource violation, but it must:

- emit an alert;
- append an optimization-policy transition;
- checkpoint immediately;
- never silently increase claimed language progress.

### D7. Gradient governance

Maintain separate task and auxiliary gradients for scheduled substrate
updates. Restrict `autograd.grad` to parameters that can actually receive CSTM
pressure, using an explicit named-parameter registry verified by a dependency
test.

The registry must include:

- carrier parameters reached through band histories;
- cognitive parameters reached through the cognitive residual;
- CSTM predictor parameters.

It must exclude unrelated parameters and fail if a newly reachable trainable
parameter is omitted.

After the VJP:

1. detach the auxiliary gradient vector;
2. run existing conflict projection;
3. apply subsystem-relative caps;
4. apply the predictor auxiliary-only cap;
5. merge into the exact-CE task gradients;
6. run global clipping once.

### D8. CSTM state and checkpoint migration

Add to training state or the checkpointed optimization policy:

- sampler schema version;
- duty probability/interval;
- target and row budgets;
- maximum substrate VJPs per context;
- coverage counts by scale/horizon;
- last selected obligation digest;
- predictor-only update count;
- substrate update count;
- safety-policy transitions.

Format-15 checkpoints without these fields map to `legacy_dense` behavior.
They must not silently change to sampled behavior on resume. Provide an
explicit one-way upgrade flag that:

1. validates the dense and sampled objective definitions;
2. initializes schedule counters from the current optimizer step;
3. records the transition in the new checkpoint;
4. preserves model, optimizer, scheduler, data, and RNG state.

New runs use the repaired sampled policy by default after acceptance passes.

### D9. CSTM telemetry

Log separately:

- configured/effective objective weight;
- duty probability and duty decision;
- predictor-only and substrate-pressure update flags;
- selected cohort/span/physical-invocation identity and
  scale/horizon/row counts;
- inclusion probability and inverse-probability weight bounds;
- sampled loss sum and estimated dense normalized loss;
- actual sampled target views, coefficient targets, and token participations;
- estimated dense equivalents, clearly labeled as estimates;
- CSTM head, carrier, and cognition gradient norms;
- pre/post-governance auxiliary norms;
- primary backward, predictor backward, substrate VJP, and merge seconds;
- cumulative coverage and maximum starvation gap;
- auxiliary-time fraction.

### D10. Tests

Add:

- `tests/test_cstm_sampling.py`
- `tests/test_cstm_execution_policy.py`
- `tests/test_cstm_checkpoint_resume.py`
- extensions to `tests/test_cstm_acceptance.py`
- extensions to `tests/test_cognitive_training.py`

Required mathematical tests:

1. enumerate every obligation in a tiny fixture and prove sampled inclusion
   probabilities sum correctly;
2. exhaustively average the ungoverned sampled loss/gradient over a complete
   categorical schedule and match the dense loss/gradient;
3. Monte Carlo test larger hierarchical samples with confidence intervals that
   contain the dense gradient norm and selected components;
4. prove future mutation cannot alter prior selected predictions or targets;
5. prove cross-document obligations have zero inclusion probability because
   they do not exist in the valid set;
6. prove every valid scale/horizon receives coverage within the declared
   supercycle;
7. prove the sampler does not mutate global RNG.

Required execution tests:

1. predictor-only backward gives gradients only to CSTM-head parameters;
2. substrate backward reaches both carrier and cognition;
3. no more than one substrate VJP occurs per context;
4. off-duty updates perform no carrier/cognition auxiliary traversal;
5. conflict projection removes negative subsystem alignment within tolerance;
6. every cap ratio is `<= 1.00001`;
7. CSTM cannot alter physical/valid token counters;
8. sampled target metrics and estimated dense equivalents are not conflated;
9. zero-valid-obligation contexts produce finite zero auxiliary work;
10. resume immediately before and after a duty step reproduces the exact
    schedule, gradients, and next checkpoint digest;
11. explicit format-15 upgrade is recorded and format-15 default resume remains
    legacy-dense;
12. corrupt inclusion probabilities, missing coverage state, or sampler-version
    mismatch fail closed.

Acceptance tolerances for tiny float64 fixtures:

- dense vs exhaustive sampled loss: `atol <= 2e-11`;
- dense vs exhaustive sampled gradient: `atol <= 2e-9`;
- gradient cosine: `>= 0.999999`;
- exact equality for schedule decisions, masks, target counts, and digests.

---

## 9. Workstream E: Larger carrier composites and static-shape execution

This work begins only after Workstreams B-D establish a faster correct eager
path. Otherwise compilation can hide or amplify the wrong execution policy.

### E1. Tensor-native internal carrier state

Introduce an internal flat/typed tensor state used throughout hot carrier
execution. Convert to rich public stream-state objects only at:

- API boundaries;
- checkpoint serialization;
- diagnostic snapshots;
- explicit test inspection.

Do not reconstruct dictionaries and dataclasses inside every checkpoint
recomputation.

### E2. Composite boundary

Build one portable composite per static `(batch, padded_length, policy)` shape
covering, where mathematically appropriate:

- mask preparation;
- spectral lifting;
- resonant affine scan;
- branch mixing;
- simplex/residual update;
- scale transition;
- state update;
- bounded history emission.

The transparent eager implementation remains the reference. Existing custom
paired-real and simplex adjoints remain reusable components.

### E3. Compilation cache

Cache compiled graphs by:

- model semantic digest;
- device/backend;
- dtype;
- batch and padded length;
- activation policy/partition;
- Torch/compiler version.

Compilation must occur outside measured steady-state steps. Cache misses,
compile seconds, graph breaks, and fallbacks must be visible.

### E4. Cross-platform backend sequence

1. portable eager composite on CPU/MPS/CUDA;
2. `torch.compile` where it produces a stable full graph;
3. Apple-optimized implementation using supported PyTorch/Metal or MLX
   primitives while retaining exact matched semantics;
4. CUDA compiled/custom kernels as an optional acceleration of the same
   operation, never as the only implementation.

No compiled backend becomes the default merely because it runs. It must beat
the eager reference after amortizing compilation over the declared training
horizon. A timeout or slower completed sample is a passing *governance*
outcome only when compilation remains rejected and the eager composite remains
the resolved production authority; it is not reported as a compiled
performance success.

### E5. Tests

Add:

- graph-break tests;
- compilation-cache identity/corruption tests;
- eager/compiled forward, state, history, and gradient comparisons;
- continuation across multiple TBPTT spans;
- padding and cross-row gradient isolation;
- compile fallback tests;
- thread-safety and multiprocessing-spawn tests;
- repeated-forward memory-growth test;
- parameter-sharing/alias preservation tests.

Performance gates:

- no increase in physical tokens or CSTM VJPs;
- median compiled carrier time lower than eager after amortization;
- no unbounded compiled-graph cache growth;
- no more than one compiled variant per actual static shape/policy;
- peak memory within the selected activation-policy reserve.

---

## 10. Workstream F: CPU, process, Trackio, evaluation, and checkpoint overhead

### F1. CPU calibration

Benchmark intra-op thread counts from a bounded set, initially `{2, 4, 6, 8}`,
in fresh subprocesses. Keep inter-op threads at one unless measured otherwise.
Freeze the selected thread count in execution policy. Do not assume every core
improves small-operator workloads.

Data loading remains one bounded lookahead worker. Add:

- queue-depth telemetry;
- time waiting for data;
- worker RSS;
- clean shutdown tests;
- no duplicated FineWeb rows after resume.

### F2. Trackio

Trackio remains observational. The default training process must not host the
dashboard. Logging must:

- use bounded metric batches;
- avoid storing large tensors or figures in memory;
- publish heavy snapshots from checkpoint-grounded subprocesses;
- coalesce repetitive checkpoint alerts;
- close database/file handles deterministically.

Measure Trackio enabled versus a null reporter on the same packed fixture.
Time the synchronous reporter insertion itself, subtract matched null-insertion
latency, and normalize the remainder to an explicit 10 ms optimization phase.
Do not put an operating-system sleep inside the measured interval: wake-up
jitter is not logger work. Keep the actual asynchronous delivery worker active
so its local contention remains part of the measurement.
Production gate:

- median steady-state step-time overhead `<= 3%`;
- additional trainer RSS `<= 256 MiB`;
- no monotonic per-step RSS growth over 100 lightweight logging steps.

### F3. Evaluation and checkpoint amortization

Training tok/s and wall-clock tok/s must both be reported. Add:

- evaluation seconds;
- checkpoint serialization/fsync seconds;
- snapshot seconds;
- amortized wall-clock tok/s since the prior optimizer step.

Evaluation and checkpoints remain required, but the 8.4M default interval
should be selected from an explicit overhead budget. Candidate production
defaults:

- evaluation every 100 updates with two or four retained batches;
- checkpoint every 100 updates, plus exceptional phase/safety checkpoints;
- lightweight scalar logs every step;
- heavy spectral snapshots at a lower bounded cadence.

Quality/safety requirements may override the overhead budget, but the resulting
cost must be visible.

### F4. Tests

- Trackio and null reporters produce identical model/optimizer/checkpoint
  tensors.
- A reporter exception cannot cancel a valid optimizer step.
- Evaluation cannot mutate model buffers, optimizer, RNG, sampler cursor, or
  training stream.
- Checkpoint save is atomic and leaves no temporary files after success/failure.
- Wall-clock accounting includes evaluation/checkpoint time exactly once.
- Alert coalescing cannot suppress errors or phase-transition first events.

---

## 11. Configuration and CLI design

The canonical `train_fineweb.py --lightmodel` path must expose explicit control
while retaining rational defaults.

Proposed controls:

```text
--activation-policy auto|retain|selective|whole-span
--activation-memory-reserve-mib
--activation-calibration | --no-activation-calibration

--document-planner auto|fixed|serial
--document-cost-calibration | --no-document-cost-calibration
--document-bucket-lengths ...
--document-batch-token-budget ...

--cstm | --no-cstm
--cstm-execution sampled|legacy-dense
--cstm-substrate-duty-probability
--cstm-max-substrate-vjps
--cstm-target-participation-budget
--cstm-predictor-update-interval
--upgrade-cstm-execution-policy

--compile-carrier auto|on|off
--performance-calibration | --no-performance-calibration
```

Rules:

- `--lightmodel` defaults to auto activation policy, auto cost planner, repaired
  sampled CSTM, cognition active, and Trackio logging without an in-process
  dashboard;
- `--ultralightmodel` performs independent calibration rather than inheriting
  8.4M decisions;
- explicit incompatible controls fail with a concrete explanation;
- resolved settings are printed before training and persisted in the manifest;
- a resumed run prints semantic, optimization, and execution differences
  separately;
- `--no-cstm` remains an explicit ablation, not the optimized default.

## 12. Telemetry schema

Add structured metric families.

### 12.1 Execution

```text
execution/policy_schema_version
execution/activation_requested
execution/activation_resolved
execution/activation_peak_bytes
execution/activation_available_bytes
execution/activation_reserve_bytes
execution/compiler_requested
execution/compiler_resolved
execution/compiled_shape_count
execution/compile_seconds
execution/fallback_count
```

String values belong in manifests/artifacts; numeric encodings used by Trackio
must have a documented enumeration.

### 12.2 Document planner

```text
document_batching/valid_tokens
document_batching/physical_tokens
document_batching/padding_efficiency
document_batching/logical_spans
document_batching/physical_invocations
document_batching/unique_shapes
document_batching/predicted_seconds
document_batching/actual_seconds
document_batching/cost_prediction_error
document_batching/target_bijection
```

### 12.3 CSTM

```text
cstm/predictor_update
cstm/substrate_update
cstm/substrate_duty_probability
cstm/substrate_vjp_count
cstm/inclusion_probability_min
cstm/inclusion_weight_max
cstm/actual_target_views
cstm/estimated_dense_target_views
cstm/actual_token_participations
cstm/estimated_dense_token_participations
cstm/coverage_gap_max
cstm/predictor_backward_seconds
cstm/substrate_backward_seconds
cstm/gradient_merge_seconds
cstm/auxiliary_time_fraction
```

### 12.4 Wall clock

```text
performance/primary_forward_seconds
performance/loss_forward_seconds
performance/primary_backward_seconds
performance/evaluation_seconds
performance/checkpoint_seconds
performance/snapshot_seconds
performance/training_tokens_per_second
performance/wall_clock_tokens_per_second
performance/unattributed_seconds
```

All derived metrics must be reconstructable from logged primitives.

---

## 13. Complete test and acceptance hierarchy

## 13.1 Fast presubmit tests

Run on every change:

- configuration validation;
- sampler probability and exhaustive-gradient fixtures;
- target-bijection property tests;
- eager/checkpoint forward-gradient parity;
- checkpoint migration/resume tests;
- null/Trackio reporter invariance;
- tiny CPU forward/backward.

Target runtime: under five minutes on the development M1.

## 13.2 Extended CPU suite

Run:

```bash
python3.11 -m pytest -q \
  tests/test_activation_execution_policy.py \
  tests/test_document_batching.py \
  tests/test_document_cost_planner.py \
  tests/test_cstm.py \
  tests/test_cstm_sampling.py \
  tests/test_cstm_execution_policy.py \
  tests/test_cstm_checkpoint_resume.py \
  tests/test_carrier_execution.py \
  tests/test_cognitive_training.py \
  tests/test_fineweb_entrypoint.py
```

Then run existing CSTM, carrier, vocabulary-router, and complete MRCRA
acceptance suites.

## 13.3 Available-device parity

For every locally available device:

- finite 1K integrated forward/backward;
- eager/optimized parity;
- retained/checkpointed parity where supported;
- sampled CSTM carrier/cognition reachability;
- checkpoint save/load/resume;
- no padding or cross-row gradient leakage.

Absence of CUDA on a Mac is reported as untested, not passed. CUDA evidence must
come from an actual CUDA runner.

## 13.4 Production 8.4M 32K acceptance

Protocol:

1. fixed packed 32K fixture;
2. same initial actor and optimizer state;
3. one warmup plus at least three measured contexts per variant;
4. fresh subprocess per variant;
5. no network, evaluation, Trackio, or checkpoint saving in the kernel-speed
   comparison;
6. separate variants adding each periodic subsystem;
7. median and dispersion reported;
8. raw artifacts retained.

Required gates on the reference M1:

| Criterion | Gate |
|---|---:|
| CE repaired path vs checkpoint/coarse CE baseline | at least 1.00x tok/s |
| repaired CSTM vs repaired CE path | at least 0.85x tok/s |
| repaired default vs current dense-CSTM regression baseline | at least 2.50x tok/s |
| median padding efficiency | at least 0.85, unless faster cost receipt proves otherwise |
| substrate VJPs per context | no more than 1 |
| mean substrate VJPs at duty 0.25 | no more than 0.25 plus finite-sample tolerance |
| Trackio overhead | no more than 3% |
| additional Trackio RSS | no more than 256 MiB |
| target bijection failures | exactly 0 |
| non-finite values | exactly 0 |

The production identity is the current integrated light actor
(`8,416,803` parameters, including the exact-authority vocabulary-router
upgrade), not the superseded pre-router `8,413,442` count.

The activation-only ratio is a strict matched non-regression gate. The original
planning estimate of `1.40x` divided the retained diagnostic by the historical
approximately `501 tok/s` coarse path. The executable named coarse baseline now
also contains the subsequently completed portable custom-adjoint and fused
carrier repairs and measures approximately `779 tok/s`; retaining `1.40x`
against that repaired denominator would count the same carrier gains twice.
Exact-signature fragmentation remains deliberately present in this diagnostic
arm and is removed only by the later cost-aware arm. To prevent this correction
from weakening end-to-end acceptance, the complete
repaired-default gate is raised from `1.75x` to `2.50x` against the immutable
fragmented, dense-CSTM reference. Both ratios remain reported.

The measured 828.5 tok/s CE diagnostic is a reference, not a hard-coded
universal pass value. Hardware-normalized ratios are authoritative; absolute
tok/s is always reported.

The 2,000 tok/s aspiration corresponds to a 16.384-second 32K update. It remains
a stretch target for native composite work, not an initial correctness gate.

## 13.5 Learning-quality non-regression

Execution speed alone cannot validate the sampled CSTM policy. Run a retained
FineWeb experiment with at least three seeds and matched:

- initial weights;
- physical corpus tokens;
- optimizer schedule;
- evaluation batches;
- exact language objective.

Compare:

1. legacy dense CSTM;
2. repaired sampled CSTM;
3. CE-only ablation.

Report:

- evaluation CE and ECE;
- CSTM standardized Huber loss;
- carrier/cognition auxiliary gradient participation;
- gradient clipping frequency;
- state and feedback RMS;
- event/cognitive metrics;
- wall-clock and token-normalized learning curves.

Initial statistical gate:

- repaired sampled CSTM's mean held-out CE must not regress from legacy dense
  CSTM by more than 0.02 nats/token at the matched token budget unless the 95%
  confidence interval demonstrates equivalence at a separately justified
  margin;
- all three seeds must remain finite and checkpoint-resumable;
- the repaired policy must demonstrate nonzero carrier and cognition CSTM
  participation;
- claims of improved learning require confidence intervals and are not inferred
  from one run.

## 13.6 Long-duration resource soak

Run at least 100 optimizer steps with lightweight logging and periodic
evaluation/checkpoint behavior enabled.

Gate:

- no monotonic unbounded RSS growth;
- no stale worker or dashboard processes;
- no checkpoint corruption;
- no schedule-coverage starvation;
- no data duplication/skipping after a mid-run resume;
- wall-clock accounting closes within 1% of measured elapsed time;
- all temporary files cleaned or explicitly retained as artifacts.

---

## 14. File-by-file implementation map

### New modules

- `src/mrrn/activation_execution.py`
  - policy schema, calibration, device memory observation, resolution receipt.
- `src/mrrn/document_cost_model.py`
  - calibrated kernel cost model and deterministic cohort partitioner.
- `src/mrrn/cstm_schedule.py`
  - obligation identities, inclusion probabilities, sampler, coverage state.
- `src/mrrn/training_execution_acceptance.py`
  - isolated benchmarks, acceptance criteria, report schema.
- `scripts/benchmark_mrcra_training_execution.py`
  - CLI runner producing raw and summarized evidence.

### Existing modules

- `src/mrrn/document_batching.py`
  - cost-aware cohort candidates, plan cost receipt, cache validation.
- `src/mrrn/cognitive_training.py`
  - separate primary/predictor/substrate phases, timing, checkpoint format 16,
    identity separation, migration, telemetry.
- `src/mrrn/model.py`
  - resolved activation policy, selective boundaries, tensor-native state path.
- `src/mrrn/carrier_execution.py`
  - expanded composite policy and compiler cache identity.
- `src/mrrn/cstm.py`
  - obligation-level loss sums and sampling-compatible accounting.
- `src/mrrn/language.py`
  - selected scale/horizon/row prediction interface and detached predictor path.
- `src/mrrn/optimization.py`
  - explicit reachable-parameter registry and governance receipts.
- `scripts/train_mrcra_fineweb.py`
  - new CLI, calibration, defaults, resolved-policy manifest.
- `scripts/train_fineweb.py`
  - canonical forwarding and help text.
- `src/mrrn/trackio_dashboard.py` and visualization code
  - new bounded performance metrics without authority.
- `scripts/run_mrcra_acceptance.py`
  - include the new acceptance artifact and fail if it is missing or stale.

### Tests

- `tests/test_activation_execution_policy.py`
- `tests/test_document_cost_planner.py`
- `tests/test_cstm_sampling.py`
- `tests/test_cstm_execution_policy.py`
- `tests/test_cstm_checkpoint_resume.py`
- `tests/test_training_execution_acceptance.py`
- focused extensions to existing document, CSTM, carrier, cognitive trainer,
  FineWeb entrypoint, Trackio, and acceptance tests.

### Documentation and evidence

- update README performance and execution-policy sections;
- update architecture report without claiming parameter count equals compute;
- update CSTM report with sampled-pressure mathematics and claim boundary;
- update carrier report with actual end-to-end acceptance results;
- regenerate parameter, acceptance, and evidence manifests;
- include exact reproduction commands and hardware fingerprints.

---

## 15. Implementation sequence and merge gates

### Phase 0: Freeze evidence

1. create deterministic packed fixtures;
2. implement subprocess benchmark harness;
3. capture current baseline;
4. add regression test proving CSTM activation increases backward traversals in
   the current path.

**Gate:** the harness reproduces the causal ranking of current variants and
produces complete raw evidence.

### Phase 1: Activation and identity separation

1. add format-16 identity split;
2. migrate format-15 checkpoints;
3. implement memory observation/calibration;
4. implement retain and whole-span resolution;
5. add selective policy only after saved-tensor census.

**Gate:** parity/resume tests pass and the 8.4M M1 auto policy selects the
fastest policy satisfying the reserve.

### Phase 2: Cost-aware document planning

1. add dense candidates;
2. relax exact signature grouping safely;
3. implement calibrated cost model and DP partition;
4. add plan cache/receipts;
5. integrate planner telemetry.

**Gate:** randomized bijection tests pass and actual 32K CE throughput beats
the coarse planner.

### Phase 3: CSTM redesign

1. factor obligation-level loss/accounting;
2. implement checkpoint-stable sampling plan;
3. implement detached predictor updates;
4. restrict reachable gradient parameters;
5. enforce one substrate VJP maximum;
6. persist schedule/coverage state;
7. add explicit legacy-dense upgrade path.

**Gate:** exhaustive-gradient, cap, causality, resume, and 32K auxiliary-overhead
tests pass.

### Phase 4: Default integration

1. enable auto activation policy;
2. enable auto cost planner;
3. enable repaired sampled CSTM;
4. keep PC-RASL disabled;
5. retain four CPU threads unless calibration proves otherwise;
6. keep dashboard out of the trainer process.

**Gate:** canonical `train_fineweb.py --lightmodel` resolves exactly these
systems and the manifest proves it.

### Phase 5: Composite/compiled execution

1. tensor-native internal state;
2. larger portable composite;
3. compiled shape cache;
4. Apple and CUDA acceleration behind parity gates.

**Gate:** matched semantics, bounded cache/memory, and positive amortized
throughput improvement.

### Phase 6: Learning and soak acceptance

1. three-seed matched learning study;
2. 100-step resource soak;
3. resume interruption matrix;
4. regenerate reports/manifests;
5. update README claims to measured evidence only.

**Gate:** all correctness, performance, learning, resource, and documentation
criteria pass.

No later phase may waive a failed earlier gate.

## 16. Failure handling and rollback

- Every new production policy has an explicit legacy/reference mode.
- The checkpoint always stores enough information to reconstruct the prior
  policy.
- An execution fallback may change checkpointing/compiler behavior but not
  semantic or CSTM sampling identity.
- A CSTM safety fallback is an optimization-policy transition and requires an
  immediate checkpoint and alert.
- Corrupt calibration/cache artifacts are discarded; corrupt semantic,
  optimizer, sampler, or checkpoint identity aborts resume.
- Failed performance gates revert the default while retaining the correct
  implementation behind an explicit experimental flag.
- Failed learning-quality gates prevent sampled CSTM from becoming default even
  if it is faster.

## 17. Definition of done

The repair is complete only when all of the following are true:

1. `train_fineweb.py --lightmodel` uses the complete integrated cognitive model;
2. CSTM remains enabled and produces measured carrier and cognition pressure;
3. at most one CSTM substrate VJP occurs per context;
4. the CSTM sampled pre-governance estimator matches the dense reference in
   exhaustive tests;
5. activation policy is selected from measured memory/cost and is resumable;
6. document planning remains target-bijective and reaches the performance gate;
7. exact CE, ECE, token counts, and byte counts remain correct;
8. format-15 migration and format-16 interruption/resume pass;
9. all available-device parity tests pass;
10. full 8.4M 32K performance acceptance passes;
11. Trackio and periodic operations remain within their resource budgets;
12. the multi-seed learning non-regression study passes;
13. the 100-step resource soak passes;
14. generated acceptance JSON and reports are current, reproducible, and
    included in the evidence manifest;
15. documentation describes only what the tests and artifacts prove.

Until then, the work must be described by its actual completed phase rather
than as a fully optimized production path.
