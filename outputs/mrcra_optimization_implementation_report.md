# MRCRA Optimization Implementation Report

**Date:** 2026-07-22  
**Scope:** the default integrated FineWeb path, exact 32K likelihood, sparse
cognitive execution, evaluation/generation projections, observability, and
checkpoint-resume correctness.

## Outcome

The default architecture now scales expensive workspace computation with live
cognition rather than configured authority capacity, gives inactive hard event
proposals differentiable environmental credit, eliminates redundant attention
projection and gradient-reduction work, automatically chooses the faster exact
loss-memory policy within a declared bound, overlaps data preparation with model
work, and caches unchanged packed projections during gradient-disabled
execution. None of these paths weakens provenance, permission, viability,
checkpoint, full-softmax, or persistent-slot authority.

The carrier also now preserves its complete causal operator state across
vectorized TBPTT spans. The same continuation state drives optimized training
prefill, arbitrary tails, and single-token generation; the former dense path's
implicit lifting, exchange, attention, and synthesis resets are removed.

The normal repository suite passes **527/527** tests. The rebuilt hash-bound
acceptance run passes 526 Python tests with one intentional build-time
traceability skip, 51 frontend tests, frontend lint/build, all eight empirical
tasks, all 15 integrated ablations, and all six performance gates. The
regenerated 174-heading evidence ledger passes its exact source-hash audit.

## Implemented changes

| Area | Implemented behavior | Correctness boundary |
|---|---|---|
| Resonator scan | One work-efficient recursive associative prefix scan in training and inference | Same recurrence; no no-grad O(T log T) special case |
| Local attention | Project each scale's K/V once, then gather projected windows; optional diagnostics no longer materialize unused weights | Exact candidate set and causal mask retained |
| Exact CE | `auto` retains tile activations only below the declared 1 GiB estimate and otherwise recomputes | Every vocabulary item remains in the partition |
| Gradient reduction | One device reduction, one finite synchronization, one foreach scaling pass | Same global norm and clipping coefficient |
| Sparse event credit | Continuous proposal/end/type/content feedback reaches CE before hard allocation | Hard threshold and bounded allocator remain persistent authority |
| Active graph | Stable live-node compaction, learned routing on the compact prefix, authoritative pointer remap to original ring | Dense output and parameter-gradient equivalence tested |
| Empty graph | Skip impossible relation scoring, top-k, and transport; retain broadcast/controller computation and clock advancement | Empty authority state and causal behavior unchanged |
| Data path | One deterministic CPU lookahead worker | Post-prefetch stream state and materialized batch are checkpointed together |
| Stateful carrier chunks | Complete-support vectorized lifting, exchange, block, attention, and synthesis transitions | Exact stream continuation; arbitrary unaligned tokens use the same one-token transition |
| Scale reduction | Masked prefix-sum fine-to-coarse aggregation | Same physical support intervals, masks, and empty-interval semantics without a per-coefficient Python loop |
| Prefill | Largest aligned prefix runs vectorially; boundary prefix/final tail streams exactly | Returned state can immediately continue with another prefill or decode token |
| CUDA compilation | Pure resonator/mixer tensor cores compile automatically on CUDA | Persistent authority commits remain ordered outside compiled regions; CPU/MPS default to eager |
| Packed inference | Cache fused identity/gate, K/V, SwiGLU, and spectral weights under no-grad | Invalidates on version, storage, device, or dtype change; training uses live graph |
| Precision moves | Floating state changes precision; integer IDs and boolean masks only change device | Learned precision can no longer corrupt authority dtypes |
| Telemetry | Phase timing, loss policy, active-node occupancy/utilization, event activation, gradient and memory fields | Trackio consumes the same ordinary metric stream |

## Measured evidence

Measurements are component-specific and should not be promoted into universal
end-to-end throughput claims:

- The real 8.4M profile's empty 128-slot workspace graph measured 26.30 ms
  dense versus 0.43 ms optimized, a **61.4x component speedup**.
- The retained work-efficient recurrence probe measured 4.11 ms for the prior
  O(T log T) path and 2.07 ms for the recursive O(T) path.
- The retained projected-window attention probe measured 161 ms before and
  140 ms after K/V reuse with bit-identical standalone output.
- At the serious profile's width, cached no-grad hybrid spectral projection
  measured about **1.30x** faster than forced repacking. At light width it was
  near break-even; its primary value is serious evaluation and recurrent decode.
- On the real 8.4M carrier with four CPU threads, a 256-token complete-state
  prefill measured 0.108 seconds versus 2.175 seconds for token-by-token
  execution, a **20.1x prefill speedup**. A paired comparison put the complete
  prefill within roughly 5% of the prior dense span that could not preserve all
  local operator state.
- Exact loss probes produced matching losses while retained tiles measured
  about 0.62 s versus 1.52 s with recomputation, trading roughly 1.29 GiB versus
  0.54 GiB isolated RSS. The default auto-policy selects this trade from the
  explicit memory ceiling.
- The rebuilt performance authority reports all six gates passing, including
  6.96% ordinary event-cycle overhead against its 25% maximum and a 6.57x
  complete-state prefill speedup against its 2x minimum sentinel.

The host was concurrently compiling an unrelated Rust/Bevy project during some
exploratory end-to-end probes. Those noisy samples were rejected; only paired
component probes and the repository's process-CPU-time acceptance authority are
reported above.

## Default runtime contract

`scripts/train_fineweb.py` continues to select the integrated MRCRA path. The
loss-memory policy is `auto`, data prefetch is enabled, CPU execution uses the
measured four intra-op/one inter-op policy, active-prefix/empty-workspace
execution is intrinsic to the model, and all new metrics enter the existing
Trackio run. Complete-state vectorized carrier prefill is the integrated
training default. CUDA automatically compiles only pure carrier tensor cores;
CPU and MPS remain eager. Training checkpoint format 7 binds the memory policy
and deterministic prefetched batch while conservatively migrating formats 3-6.
Older format-7 identities without the execution-only compiler policy inherit
the current explicit policy without changing weights or mathematical state.

## Claim boundary

These changes establish implementation correctness and strong local efficiency
evidence. They do not claim a trained serious checkpoint, target-hardware 32K
throughput, or improved language quality without a matched training run. The
next production run's phase telemetry and retained evaluation must decide
whether the realized end-to-end speed and CE/ECE tradeoff justify further
backend specialization.
