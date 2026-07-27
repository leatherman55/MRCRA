# MRCRA Training-Execution Repair Implementation Report

## Status and claim boundary

The repair mechanisms described by the companion
[implementation plan](mrcra_training_execution_repair_implementation_plan.md)
are implemented in the canonical MRCRA trainer. The six-eager-variant
8.4M/32K production execution matrix and bounded compiler decision were
measured before the final memory-safe long-document repair. The
current-source quick execution matrix, source-free 100-step resource soak,
100-step Trackio overhead probe, carrier/CSTM mechanism suites, checkpoint
migrations, and source-free three-seed learning procedure pass on the measured
local system.

This is a production-profile local execution claim, not a completed
corpus-scale learning or long-duration resource claim. The full disjoint
FineWeb study was deliberately stopped after one complete matched seed; its
three completed arms remain in the atomic raw journal and are explicitly
non-authoritative until all three seeds finish. The 100-step 8.4M/32K soak and
CUDA qualification also remain open. The correct description is therefore
**implemented, default-enabled, and production-profile performance-tested
locally**, with a source-current production rerun and the named learning/soak
qualification boundary still visible.

Measured environment for the retained local evidence:

- macOS 26.3, arm64;
- Python 3.11.15;
- PyTorch 2.13.0;
- CPU and Apple MPS available;
- CUDA unavailable and therefore reported as untested, never as passed.

## 1. Activation-memory authority

Activation retention is no longer inferred from model parameter count or a
fixed host-memory heuristic. `src/mrrn/activation_execution.py` defines a
versioned policy receipt containing:

- the requested and resolved policy;
- live device/host memory observations;
- the required reserve;
- conservative retained-activation estimates;
- measured candidate timing and incremental peak memory;
- output-equivalence digests;
- the hardware and PyTorch fingerprints;
- the decision reason.

The supported execution ladder is:

1. `retain`: retain the full carrier span when the measured reserve permits it;
2. `selective`: checkpoint only boundaries selected from the saved-tensor
   census;
3. `whole_span`: recompute the complete carrier span as the safest bounded
   fallback.

Automatic selection runs an isolated, non-authoritative calibration. It
restores RNG and mutable model buffers and cannot update weights, optimizer
state, CSTM statistics, or the training stream. An explicit unsafe policy
fails closed unless the caller also supplies the named
`--allow-unsafe-activation-policy` override.

### OOM recovery

The trainer permits exactly one pre-update OOM recovery:

- the packed batches for the optimizer update are fetched once and retained;
- mutable CSTM buffers, coverage state, RNG, gradients, and transient runtime
  references are snapshotted;
- recovery is allowed only before the main optimizer mutation and while
  PC-RASL is disabled;
- the next safer activation rung is installed;
- the identical packed batches are replayed without advancing the data stream
  or token counters twice;
- the execution-policy history records the transition and its effective step.

A second OOM, an OOM after optimizer/PC-RASL mutation, or an OOM at
`whole_span` aborts rather than pretending recovery.

## 2. Format-16 checkpoint identity

Checkpoint format 16 separates five authorities:

- `semantic`: model, tokenizer, source, retained evaluation/progress evidence,
  CSTM architecture, and parameter identity;
- `optimization`: objective, schedule, gradient, and optimizer-affecting
  training controls;
- `execution`: batching, recomputation, exact-loss backend, device, compiler,
  and resolved activation receipts;
- `observation`: evaluation cadence, checkpoints, Trackio, dashboards, and
  diagnostic cadence;
- an explicit equivalence contract naming which sections may vary without
  changing learned semantics.

Execution-policy history is digest-bound and monotonic in effective step.
Formats 3–15 migrate through the historical monolithic identity, then through
the same partitioner used by new checkpoints. Legacy dense CSTM checkpoints
cannot silently become sampled CSTM: they require legacy execution or the
explicit upgrade authority. Legacy PC-RASL retirement/migration remains
supported. Serious-checkpoint auditing now consumes the partitioned identity
without collapsing observation controls back into optimization authority.

## 3. Document-major static execution

`src/mrrn/document_batching.py` converts each packed context into immutable
document sequences and constructs physical static batches while preserving an
exact logical-target bijection.

The production candidate family is:

```text
64, 128, 192, ..., 3968, 4032, 4096
```

The planner:

- respects carrier and cognition alignment;
- never crosses source-document boundaries;
- preserves stable target order and byte weights;
- emits target and source receipts;
- searches exact-signature and relaxed cost-aware cohorts;
- rejects candidates violating token/memory limits;
- scores padding, physical invocations, event work, carrier work, and
  vocabulary-loss work through a calibrated device cost model;
- caches only bounded group decisions and revalidates every cached result
  through a fresh target-bijection receipt.

TBPTT is treated as a maximum interval. The planner derives a memory-safe
single-row ceiling from the token budget, measured activation bytes/token, and
available activation bytes. Documents longer than that ceiling are divided
into additional contiguous state-preserving spans before cohort optimization.
The pinned FineWeb regression that exposed this edge contained 4,096-token
spans on a host authorizing 3,904 tokens; after repair, the context planned at
90.14% padding efficiency with exact target bijection. If the smallest static
bucket cannot fit, the planner names the violated byte authority and aborts
before model execution.

Property-based tests exercise 500 randomized mixtures, including empty
fragments, variable document lengths, masking, EOS boundaries, bucket edges,
and cache hits. The quick fixture’s padding efficiency is 0.727 because the
128-token cognition alignment is an irreducible floor; production retains the
stricter 0.85 gate.

## 4. Carrier composites and custom adjoints

The carrier now has a tensor-tree codec for moving complete state through a
single coarse checkpoint/composite boundary without dropping static metadata
or tensor leaves.

Implemented portable composites include:

- an associative affine scan with an exact custom first-order adjoint;
- a fused simplex-residual operation;
- whole-carrier-span checkpoint recomputation;
- tensor-native carrier state at checkpoint/compiler boundaries;
- a bounded, digest-bound compilation-specialization registry;
- fresh lock construction for copied or unpickled registries.

Compilation is an optional measured backend, not an assumed optimization. The
registry records static shape keys, first-execution compilation time, cache
size, and fallbacks. CPU and MPS keep the portable custom-composite path.
CUDA may select compilation only behind parity and amortized-throughput gates.

The retained carrier acceptance proves:

- affine-scan forward error: `0`;
- affine-scan gradient maximum error: `1.78e-15`;
- saved-tensor byte ratio: `0.218`;
- simplex forward and gradient errors: `0`;
- simplex autograd-node ratio: `0.370`;
- coarse-checkpoint forward, input-gradient, and continuation-state errors:
  `0`;
- document cost ratio versus exact-signature grouping: `0.675`;
- finite forward/backward execution on both locally available devices, CPU
  and MPS.

## 5. Bounded Causal Spectral Target Multiplexing

CSTM remains enabled by default. It does not fabricate corpus tokens. It adds
strictly causal future spectral obligations to already-computed carrier
coefficients.

### Hierarchical sampling

The production sampled path uses two deterministic authorities:

1. a counter-based scale/horizon/invocation sampler with explicit inclusion
   probability and coverage state;
2. a content-bound cyclic-permutation row sampler without replacement.

Both are pure functions of checkpointed identity/counter state and consume
neither the global Torch RNG nor Python RNG. Every eligible obligation has
nonzero long-run inclusion probability. Coverage counters, maximum gaps, and
receipts survive resume.

The row sampler executes before predictor-head materialization. Its configured
participation budget counts batch × rows × horizons × support. Trainer
construction fails if the budget cannot hold one complete row at the coarsest
scale. Loss sums, normalization rows, target-RMS statistics, and telemetry use
the exact inverse inclusion probability.

### Predictor and substrate phases

Predictor-only work receives detached carrier/cognitive features. Scheduled
substrate work permits at most one CSTM substrate VJP per optimizer context.
The predictor update interval is explicit and checkpoint-bound. This removes
the earlier per-invocation/per-scale many-small-backward path.

### Gradient governance

Auxiliary gradients are retained separately until exact-CE gradients are
authoritative. The merge:

- identifies reachable parameters by named subsystem;
- projects conflicting task-aligned components;
- caps carrier, cognitive, controller, workspace, world, memory, and head
  contributions independently;
- permits bounded auxiliary-only pressure for cognitive parameters dormant
  under exact CE;
- requires a live exact-CE path for carrier pressure;
- publishes pre/post norms by subsystem.

The zero-initialized cognitive gate first learns a predictor-supported
coupling; subsequent eligible steps deliver nonzero cognitive-substrate
pressure. The multi-step acceptance procedure verifies that transition.

The retained CSTM acceptance proves direct-DFT equivalence, order sensitivity,
 boundary isolation, strict integrated causality, predictor learning
(`final/initial = 0.193`), six governed overlapping subsystems, cap ratio
`0.99999994`, finite JSON evidence, unchanged physical-token accounting, and a
1.3M-profile CSTM head of 2,158 parameters.

## 6. Exact vocabulary loss and tensor materialization

The language objective remains exact next-token cross entropy over the full
vocabulary. Exact dense, tiled, fused, official Cut Cross-Entropy, and compiled
CCE authorities retain matched loss/gradient tests and explicit workspace
gates. The trainer does not claim routing or CSTM target views as additional
corpus tokens.

The repaired static path reduces materialization by:

- forming document-major physical batches once;
- using one output-latent/statistics pass per physical invocation;
- selecting CSTM rows before predictor projection;
- accumulating scalar loss sums and bounded receipts instead of retaining
  dense per-obligation graphs;
- retaining one grouped substrate auxiliary graph at most.

## 7. CPU/process execution

CPU thread count accepts `0` as auto-calibration. Candidate calibration covers
2, 4, 6, and 8 intra-op threads with one inter-op thread, and records the
chosen result. Data prefetch remains bounded and can be disabled.

The long resource-soak resume is process-isolated. The first model/optimizer
process saves and exits before the replacement process is created, matching
real `--resume` behavior and preventing a double-live allocator high-water
artifact.

The 100-step quick soak passes:

- 100 optimizer steps;
- zero positive RSS growth slope after process isolation;
- 18.0 MiB total RSS range versus a 256 MiB limit;
- exact midpoint resume;
- four retained checkpoints;
- no temporary files;
- no stale MRCRA/Trackio threads;
- wall-clock accounting error `1.48e-5`, below 1%;
- no non-finite metrics.

## 8. Bounded Trackio observation

Trackio is observational and does not host the dashboard in the trainer
process by default. Every finite scalar row enters the authoritative local
mirror, while the remote stream is deterministically coalesced to one row per
four optimizer steps before entering a bounded queue. Remote backpressure may
drop dashboard-only intermediate delivery but never the local JSONL mirror.

The local mirror now uses one 64 KiB buffered append handle rather than
reopening the file every step. It flushes:

- every 16 metric records;
- immediately for alerts and non-metric authority rows;
- immediately when the remote queue is full;
- before evidence artifacts read the mirror;
- at bounded shutdown.

Non-finite metrics are rejected. Remote logging uses a bounded daemon worker,
bounded drain, failure receipt, and checkpoint-alert coalescing.

The 100-step null-versus-Trackio probe times synchronous insertion directly,
subtracts matched null insertion, and normalizes the difference to a 10 ms
optimization phase. This avoids misclassifying operating-system sleep wake-up
jitter as logger work while the real background delivery worker remains active.
Five consecutive repair-validation trials passed between `0.238%` and `0.258%`;
the retained acceptance artifact records the final trial:

- median steady-state step overhead: `0.258%` versus a `3%` limit;
- additional peak RSS: 30.2 MiB versus 256 MiB;
- steady-state RSS range: 16 KiB versus 64 MiB.

The dashboard polling layer independently enforces a ten-second minimum
interval, one in-flight request, bounded points/run, opt-in smoothing, newest
run default selection, and lazy chart construction.

## 9. Measured production execution matrix

Every variant ran in a fresh subprocess with one unmeasured warmup and three
measured steps:

| Variant | tok/s | Median step |
| --- | ---: | ---: |
| legacy serial + whole-span + dense CSTM | 235.87 | 138.922 s |
| static coarse + whole-span + CE | 812.42 | 40.334 s |
| static coarse + whole-span + dense CSTM | 495.80 | 66.091 s |
| static auto + CE | 815.22 | 40.195 s |
| static auto + sampled CSTM | 860.25 | 38.091 s |
| static cost model + auto + sampled CSTM | 802.94 | 40.810 s |

The repaired default is 3.40× the serial dense reference and retains 98.49% of
the matched repaired CE-only rate. Exact target bijection passes. Per-step sampled
substrate VJPs are `[0, 1, 0]`, so the maximum is one and the measured mean is
0.333.

The CPU AOT compiler candidate exceeded its 300-second hard budget and was
terminated and reaped. The retained receipt truthfully resolves the eager
cost-model variant; no eager fallback is mislabeled as compiled.

## 10. Three-seed learning procedure

The quick procedure launches fresh subprocesses for three seeds and three
matched variants: legacy dense CSTM, repaired sampled CSTM, and CE-only. Each
arm saves at its midpoint, reconstructs the trainer, resumes exactly, and
evaluates on a retained source-free split.

It passes:

- all nine arms finite;
- all nine arms checkpoint-resumable;
- carrier auxiliary participation in every sampled arm;
- cognitive auxiliary participation in every sampled arm;
- sampled-minus-dense mean CE difference
  `+7.17e-7 nats/token`;
- paired 95% confidence interval
  `[-5.78e-7, +2.01e-6]`.

This validates the statistical procedure and pressure/resume mechanisms only.
The production learning claim requires the real disjoint FineWeb profile.

One full pinned FineWeb seed was also completed before the multi-hour study was
stopped:

| Seed 17 arm | Wall time | Retained eval CE |
| --- | ---: | ---: |
| legacy dense CSTM | 2,391.8 s | 10.802069 |
| repaired sampled CSTM | 1,578.0 s | 10.802087 |
| CE-only | 1,521.3 s | 10.801638 |

For this seed, sampled execution is about 1.52× faster than legacy dense and
3.7% slower than CE-only. Sampled-minus-legacy CE is only
`+0.0000174 nats/token`. All three arms are finite and checkpoint-resumable;
the sampled arm reaches both carrier and cognition. These are informative
partial results, not a three-seed acceptance result. The raw journal is
explicitly marked `complete: false`.

## 11. Canonical defaults and controls

`scripts/train_fineweb.py --lightmodel` resolves to:

- the complete integrated cognitive path;
- document-major static batching;
- cost-aware planning;
- measured automatic activation policy;
- exact full-vocabulary loss;
- sampled CSTM enabled;
- 25% substrate duty cycle;
- one maximum substrate VJP;
- 8,192 target-participation budget;
- predictor updates every optimizer step;
- PC-RASL disabled;
- bounded Trackio logging with no in-process dashboard;
- four-step remote scalar coalescing with a complete local metric mirror;
- CPU thread calibration only when `--cpu-threads 0` is selected.

Explicit reference/rollback controls remain available for dense CSTM,
exact-signature planning, whole-span recomputation, compiler disablement,
Trackio disablement, and CSTM upgrade authorization.

## 12. Verification inventory

Retained empirical artifacts:

- `mrcra_training_execution_baseline.json`;
- `mrcra_training_execution_acceptance.json`;
- `mrcra_training_execution_acceptance.md`;
- `carrier_execution_empirical_acceptance.json`;
- `cstm_empirical_acceptance.json`;
- `mrcra_trackio_overhead_acceptance.json`;
- `mrcra_resource_soak_acceptance.json`;
- `mrcra_learning_nonregression_procedure.json`.

The final acceptance run passed 854 Python cases; the traceability self-check
was skipped only while its own hash-bound manifest was being replaced. After
the evidence ledger was rebuilt, all 10 traceability/manifest/artifact tests
passed. Frontend acceptance passed 58 tests across nine files, ESLint passed,
and the Vite production build passed. Every bounded command passed. The sole
nonzero manifest command is the deliberately fail-closed full-scale completion
validator, whose open criteria are listed below.

## 13. Remaining production gates

These are intentionally not represented as passing:

1. rerun the `production_8p4m_32k` execution matrix against the final
   memory-safe planner source digest;
2. real disjoint FineWeb three-seed learning non-regression (one seed is
   journaled; two remain);
3. the 100-step `production_8p4m_32k` resource/resume soak;
4. CUDA forward/gradient/compiler parity and memory/throughput qualification;
5. a seriously trained checkpoint with preregistered downstream evaluation.

Reproduction commands:

```bash
python3.11 scripts/benchmark_mrcra_training_execution.py \
  --profile production_8p4m_32k \
  --steps 5

python3.11 scripts/run_mrcra_learning_nonregression.py \
  --profile fineweb_8p4m_32k \
  --steps 32 \
  --total-tokens 1048576
```

Failure of either command blocks promotion. It does not authorize relaxing
semantic, numerical, learning, or resource thresholds.
