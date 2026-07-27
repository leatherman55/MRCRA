# Carrier Execution Optimization: Implemented Production Contract

## Scope

This report describes the implemented optimization of MRCRA’s integrated
carrier-training path. The change is an execution redesign, not a new
mathematical model and not a relaxation of next-token authority.

The implementation addresses five coupled costs:

1. independent documents were executed serially even when their carrier shapes
   were compatible;
2. the paired-real affine recurrence exposed a large recursive prefix graph to
   autograd;
3. branch simplex mixing created many feature-wide multiply/add nodes;
4. activation checkpointing occurred independently inside every layer and
   scale, fragmenting backward and causing repeated recomputation boundaries;
5. stream state used rich mutable Python structures at the checkpoint boundary,
   preventing one safe coarse tensor composite.

The production path now consists of deterministic document-major static
cohorts, portable custom adjoints, a whole-carrier-span checkpoint boundary,
and exact receipts proving that regrouping did not change target authority.

## Non-negotiable invariants

The implementation fails closed unless all of the following hold:

- every valid packed next-token target appears in the execution plan exactly
  once;
- no invalid cross-document transition becomes a target;
- a document keeps the same physical batch row for every one of its TBPTT
  spans;
- recurrent carrier state, cognitive state, feedback, and provenance never
  move between documents;
- padding is a prefix-invalid suffix and cannot reactivate;
- padded token values, event values, provenance references, and segment
  identifiers claim no authority;
- the carrier’s causal coefficient positions remain on the shared static
  physical timeline, while CSTM target validity remains document-local;
- regularization is normalized over logical document spans, not over the
  reduced number of physical invocations;
- exact full-vocabulary CE remains the primary environmental pressure;
- custom operations match the transparent composite forward and every
  first-order gradient within the declared numerical tolerance;
- checkpoint recomputation produces the same static tensor tree as the original
  forward.

## 1. Deterministic document-major static batching

`mrrn.document_batching.DocumentMajorBatchPlanner` converts a packed context
into a model-independent intermediate representation:

- `DocumentSpan` is one contiguous document-local TBPTT span;
- `DocumentSequence` is the ordered set of spans for one document;
- `StaticDocumentSpanBatch` is one padded physical invocation with explicit
  token, loss, event, input-ordinal, and target-ordinal masks;
- `StaticDocumentCohort` is a stable set of document rows with one static bucket
  signature across all spans;
- `DocumentTargetAuthority` retains the complete document-local CSTM target
  tensors across span boundaries;
- `TargetBijectionReceipt` compares the original and emitted target ordinals,
  detects omissions, additions, and duplication, and binds both sides to
  digests;
- `DocumentBatchPlan` records logical tokens, physical tokens, padding
  efficiency, cohort count, invocation count, and the passed bijection.

The default bucket set is:

```text
128, 256, 512, 1024, 2048, 4096
```

Smaller carrier/cognition-aligned powers of two are inserted automatically for
small models and tests. Cohort capacity is:

```text
max(1, document_batch_token_budget / largest_bucket_in_signature)
```

The default physical token budget is 8,192. Ordering is deterministic:
signatures, original document order, stable row assignment, and span order are
all fixed. Noncontiguous reuse of a segment identifier is rejected.

The canonical FineWeb CLI enables this path by default. Audit controls are:

```bash
--document-static-batching
--no-document-static-batching
--document-bucket-lengths 128 256 512 1024 2048 4096
--document-batch-token-budget 8192
```

The serial integrated path remains available only for matched semantic and
performance audits.

## 2. Batched mask-safe cognition

The integrated cognitive forward now accepts a document batch rather than
requiring batch size one and an all-valid mask.

The contract requires:

- at least one valid token in every row;
- validity is prefix-contiguous;
- continuation carrier, cognitive, and feedback state has the same batch;
- event validity is sampled from token validity at the exact cognitive anchors.

All causal summaries and uncertainty summaries use masked reductions. Cognitive
residuals are applied only to valid token intervals. Invalid event rows:

- have zero event activation;
- cannot update feedback;
- cannot open, finalize, emit, or reject an event;
- have no phase-logit telemetry authority.

Padding token latents and internal event values are exactly zero. Padding
provenance and segment identifiers use the `-1` sentinel. Tests compare a
batched execution against every row executed alone, verify zero padded outputs,
and prove zero cross-row and padded-tail gradients.

## 3. Batched Causal Spectral Target Multiplexing

CSTM predictions and targets now use:

```text
values: [batch, coefficient_rows, horizons, 3, code_dimension]
mask:   [batch, coefficient_rows, horizons]
```

Coefficient source positions are shared because stable cohorts execute one
static carrier timeline. Target labels, target segment identifiers, and masks
remain row-local. The full target authority is padded to the complete physical
cohort timeline, but padding remains unauthorized. This permits a shared source
index without ever generating a target beyond a real document.

The loss, running RMS statistics, per-scale accounting, per-horizon accounting,
coefficient target count, and constituent token participation count are all
batch-aware. A dedicated test changes one document row and proves that no
target in another row changes.

## 4. Custom paired-real affine scan adjoint

For each complex diagonal state lane:

```text
y_t = a_t y_(t-1) + b_t
```

The forward remains the work-efficient `O(T)`-composition,
`O(log T)`-dependency-depth associative prefix tree.

The previous transparent autograd path retained the intermediate tensors from
every recursive pair, slice, stack, concatenation, multiplication, and
addition. The custom operation retains only:

- transition coefficients;
- initial state;
- output states.

The mask-aware form additionally retains the boolean mask. Identity
transitions and zero drives for padding are created inside the custom forward
and are not retained by autograd.

For output cotangent `g_t`, the state cotangent obeys:

```text
q_t = g_t + conjugate(a_(t+1)) q_(t+1)
```

Reverse time turns this into the same affine-prefix primitive. Gradients are:

```text
dL/db_t = q_t
dL/da_t = q_t conjugate(y_(t-1))
dL/dy_initial = conjugate(a_0) q_0
```

At an invalid masked position:

```text
dL/da_t = 0
dL/db_t = 0
q_(t-1) = q_t
```

The custom first-order adjoint is used automatically in eager training on CPU,
MPS, and CUDA. The pure composite remains available for compiler tracing,
inference, numerical reference, and explicit audit.

## 5. Custom simplex/residual adjoint

Each MRRN band combines resonance, local mixing, attention, identity, and,
where configured, relational context:

```text
delta = sum_k p_k branch_k
updated = mask * (band + layer_scale * delta)
```

Softmax remains PyTorch’s standard tested operation. The feature-wide weighted
sum, residual scale, and final padding mask are one portable custom autograd
node.

Its direct adjoint computes:

```text
g_active = g_output * mask
g_delta = layer_scale * g_active
g_branch_k = p_k * g_delta
g_p_k = sum_features(g_delta * branch_k)
g_layer_scale = sum(g_active * delta)
g_band = g_active
```

This preserves the ordinary softmax Jacobian through `g_p` while removing the
many multiply/add nodes across model width.

## 6. Whole-carrier-span checkpoint recomputation

When carrier activation checkpointing is enabled on the integrated path,
checkpointing now wraps one complete carrier span.

`flatten_tensor_tree` converts the public stream state into:

- a flat tuple containing every tensor exactly once;
- an immutable template containing every mapping key, list/tuple kind, scalar,
  string, boolean, and `None` leaf.

The template is reversible and has a SHA-256 digest. Unsupported leaf types,
missing tensors, extra tensors, duplicate tensor indices, and unconsumed
tensors are rejected.

On both the original forward and recomputation:

1. the incoming state is reconstructed into a fresh stream-state object;
2. nested per-scale checkpointing is disabled;
3. the complete carrier prefill executes;
4. outgoing state and band histories are flattened;
5. the output static template must equal the first execution’s template.

The returned tensors are reconstructed into the ordinary `MRRNStreamState` and
`CausalBandHistory` public types. `CarrierCompositeReceipt` records input and
output template digests, state tensor count, history tensor count, and the
`whole_carrier_span` recomputation granularity.

This preserves all continuation authority while replacing a layer-by-scale
checkpoint forest with one coarse boundary.

## 7. Reduced history materialization

Aligned static execution normally stores one bounded history tensor per scale.
The carrier now returns that tensor directly instead of calling `torch.cat` on
a one-element list. Concatenation remains only when multiple step-mode fragments
actually exist.

The custom scan also keeps padding identity/zero tensors outside the retained
autograd graph, and the custom residual prevents intermediate branch-weighted
feature tensors from becoming separate autograd nodes.

## 8. Backend and fallback policy

The authoritative portable backend is:

```text
portable_custom_composites
```

It is available on CPU, MPS, and CUDA. Optional pure tensor-core compilation is
automatically selected on CUDA or may be explicitly enabled/disabled. It
augments, rather than replaces, the portable custom adjoints.

An automatic compiler setup failure falls back to portable custom composites
and records the reason. An explicitly requested compiler failure raises,
because silently ignoring an explicit execution contract would make the run
identity misleading.

Runtime telemetry records:

- carrier execution backend;
- affine scan implementation;
- simplex residual implementation;
- scan saved-tensor contract;
- state boundary contract;
- checkpoint granularity;
- nested checkpoint status;
- compiler policy and fallback reason;
- document bucket set and physical token budget.

Per-update metrics record document count, cohort count, physical invocation
count, physical/valid tokens, padding efficiency, target bijection, custom
adjoint activation, and whole-span checkpoint activation.

## 9. Checkpoint migration

MRCRA training checkpoint format is now version 15. Version 14 is accepted as a
legacy format. Its missing document execution fields migrate to the explicitly
constructed current contract:

- `document_static_batching`;
- `document_bucket_lengths`;
- `document_batch_token_budget`.

Model weights, optimizer state, scheduler state, RNG, stream state, retained
evaluation identity, and CSTM state remain governed by the existing exact
resume checks. A production test constructs a version-14 payload with the new
fields removed and verifies migration into the default document-major path.

## 10. Verification

The focused production suites cover:

- deterministic planner construction and randomized property tests;
- exact target bijection and stable row assignment;
- segment/boundary/source preservation;
- event mask construction;
- invalid planner configuration and noncontiguous segments;
- batched cognition versus isolated rows;
- zero padding outputs and zero forbidden cross-gradients;
- batched CSTM targets and row isolation;
- custom scan forward equivalence at odd, even, power-of-two, and non-power-of-two
  lengths;
- custom scan full gradient equivalence;
- float64 finite-difference gradcheck;
- exact masked recurrence and zero invalid transition/drive gradients;
- measured saved-tensor count and bytes;
- custom simplex forward and every input gradient for four and five branches;
- measured autograd node reduction;
- whole-span checkpoint output, state, history, parameter-gradient, and
  input-gradient equivalence;
- exact two-span continuation;
- document-major versus serial exact CE and parameter gradients;
- CSTM trainer receipts and governed auxiliary gradients;
- version-14 checkpoint migration;
- FineWeb default smoke execution.

Run the focused suite:

```bash
python3.11 -m pytest \
  tests/test_carrier_execution.py \
  tests/test_carrier_execution_acceptance.py \
  tests/test_document_batching.py \
  tests/test_resonance.py \
  tests/test_model.py \
  tests/test_cstm.py \
  tests/test_cognitive_model.py \
  tests/test_cognitive_training.py \
  tests/test_fineweb_entrypoint.py -q
```

Run the standalone empirical acceptance:

```bash
python3.11 scripts/run_carrier_execution_acceptance.py
```

The current local artifact contains the float64 CPU equivalence audit plus
finite float32 forward/backward smoke tests on every available local execution
device (CPU and MPS on the producing machine). It is stored in
`outputs/carrier_execution_empirical_acceptance.json`.

## Claim boundary

The acceptance artifact establishes mechanism-level semantic equivalence,
gradient equivalence, state continuity, reduced saved-tensor materialization,
reduced autograd node count, and fewer physical document invocations in its
controlled fixture.

It does not establish a universal throughput multiplier. Absolute training
throughput remains dependent on model profile, document-length distribution,
static padding efficiency, exact vocabulary-loss backend, CSTM activity,
cognitive event behavior, device, compiler, and memory bandwidth. A 32K
target-hardware benchmark is still required for any absolute tokens/second
claim.
