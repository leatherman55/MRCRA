# Causal Spectral Target Multiplexing Implementation Report

## Status and claim boundary

Causal Spectral Target Multiplexing (CSTM) is implemented as the default
auxiliary objective of the integrated MRCRA FineWeb trainer. The implementation
is checkpoint-bound, can be disabled for a matched ablation with `--no-cstm`,
and leaves exact next-token cross entropy as the primary optimization
authority.

CSTM does **not** create new corpus data, replay tokens, or justify increasing
the physical `tokens_seen` counter. It extracts additional, correlated causal
prediction constraints from carrier states already computed by the ordinary
forward pass. The retained empirical acceptance suite proves the target
mathematics, boundary isolation, causal alignment, local trainability,
gradient-governance contracts, parameter bounds, and accounting contracts on
deterministic fixtures. A matched corpus-scale CSTM-on/CSTM-off training study
is still required to establish downstream language-quality or
sample-efficiency benefit.

## Design objective

Ordinary next-token CE asks the finest output state to identify one future
token through a full vocabulary projection. MRRN already computes causal
coefficients whose receptive supports are aligned to multiple physical scales.
CSTM uses those existing coefficients as prediction sites:

1. each emitted band coefficient names the exact input position at which its
   causal support becomes complete;
2. that state predicts one or more complete future blocks at its own support
   size;
3. future token identities are represented by a fixed compact code;
4. the target retains block composition and one order-sensitive spectral
   component;
5. one small predictor is shared over all scales and horizons;
6. its gradients are calculated separately and admitted only through explicit
   conflict and magnitude governance.

This converts the multiresolution carrier topology into multiresolution
supervision without a second carrier pass and without another
vocabulary-sized output head.

## Fixed token representation

For vocabulary size \(V\) and code width \(d_c=64\), the implementation creates
a fixed matrix

$$
\Phi\in\{-d_c^{-1/2},+d_c^{-1/2}\}^{V\times d_c}.
$$

The matrix is produced by a private CPU random generator with seed `20260725`.
Construction does not mutate PyTorch's global random state. Every row has unit
Euclidean norm. The matrix is registered as a nonpersistent buffer:

- it follows the model to CPU, MPS, or CUDA;
- it receives no gradient;
- it does not enlarge checkpoints;
- it is reconstructed exactly from the checkpoint-bound architecture seed.

Tests verify deterministic reproduction, row normalization, distinct rows for
the tested vocabulary fixture, seed sensitivity, and noninterference with the
global RNG.

## Causal target mathematics

Let the carrier coefficient at scale \(s\) have support \(q_s\) and end after
input position \(p\). `labels[p]` is the token strictly after that input
position. For horizon \(h\), measured in blocks, the target starts at

$$
b(p,s,h)=p+(h-1)q_s.
$$

For token labels \(y_b,\ldots,y_{b+q_s-1}\), CSTM computes

$$
C_{s,p,h,m}
=
\frac{1}{\sqrt{q_s}}
\sum_{r=0}^{q_s-1}
\Phi_{y_{b+r}}
\exp\left(-2\pi i m r/q_s\right).
$$

Only \(m=0\) and \(m=1\) are retained. They are stored as three real channels:

1. DC;
2. real part of the first harmonic;
3. imaginary part of the first harmonic.

DC is invariant to a permutation of the tokens within the block and therefore
represents coded block composition. The first harmonic changes with order. For
support two it is the real Nyquist component, and its imaginary channel is
numerically zero. Direct-DFT tests verify the sign convention and normalization
exactly.

Targets are detached before loss construction. There is no learned target
encoder and no route by which the prediction head can collapse the target
space.

## Boundary and validity authority

A scale/horizon target row is valid only if all of the following hold:

- the complete \(q_s\)-token future block exists;
- every constituent position is a valid ordinary language target;
- every constituent target segment equals the source segment;
- the source position itself lies inside the packed context.

Partial blocks are not shortened, padded, wrapped, or assigned reduced weight.
Blocks that reach a packed-document boundary fail closed.

`causal_spectral_target_mask` is the single validity authority. It performs no
code lookup and no Fourier work. Both target construction and the trainer's
context-wide objective denominator use this same function, preventing
mathematical validity from drifting away from accounting validity.

## Exact band histories

The streaming MRRN carrier now retains `CausalBandHistory` for each physical
scale during prefill. A history contains:

- every coefficient emitted in the current integrated span;
- its `ScaleTensor` metadata: scale, sample interval, support, kind, and mask;
- a strictly increasing int64 end-position vector.

For detail scale \(s\), support is \(2^{s+1}\). The terminal approximation has
support \(2^{S-1}\), equal to the coarsest detail support. End positions name
the last input consumed by each coefficient, never a future target position.

The history collector covers three execution regimes without changing the
contract:

- initial scalar steps required to reach an aligned carrier position;
- the vectorized aligned chunk;
- trailing scalar steps after the aligned chunk.

Tests compare exact completion positions across chunking regimes and perturb
future inputs to verify that earlier retained coefficients do not change.

## Integrated carrier and cognition features

For every retained coefficient, the existing carrier synthesis adapter maps
that scale's width into the base model dimension. No new per-scale dense adapter
is introduced.

The integrated cognitive model also returns the event-rate cognitive residual
at every input position. CSTM gathers that residual at each coefficient's exact
end position. The predictor therefore receives aligned tensors

$$
X_{s}\in\mathbb R^{1\times N_s\times d},
\qquad
R_{s}\in\mathbb R^{1\times N_s\times d}.
$$

Changing tokens after a source position leaves that source's carrier features,
cognitive residual, and CSTM prediction unchanged. The empirical acceptance
suite runs the real integrated carrier and cognitive path twice, modifies only
the future suffix, observes zero change in past predictions, and verifies that
later predictions do change.

## Shared predictor

The predictor uses one rank-eight bottleneck across all scales and horizons:

$$
z_{s,p,h}
=
\tanh\left(
W_xX_{s,p}
+\tanh(g_s)W_rR_{s,p}
+e_s+e_h
\right).
$$

`W_x`, `W_r`, and the final projection are shared. \(e_s\) and \(e_h\) are
learned scale and horizon embeddings. The output projection produces
\(3d_c\) real values.

The learned cognitive gate \(g_s\) is initialized to exactly zero. At
initialization, changing the cognitive residual cannot change the prediction,
but the gate itself receives gradient and can open if cognition provides useful
future-block evidence. This prevents the new objective from imposing an
arbitrary cognitive route at model construction while leaving that route fully
learnable.

A bounded phase parameter rotates the predicted real/imaginary harmonic pair:

$$
\theta_{s,h}=\pi\tanh(\rho_{s,h}).
$$

This preserves harmonic magnitude and gives each scale/horizon a learned phase
alignment without an unconstrained angular parameter.

## Horizon policy

The architecture binds horizons `(1, 2, 4, 8)`. Every scale always predicts
horizon one. It predicts one additional horizon selected from `(2, 4, 8)` by

$$
j=(\mathrm{optimizer\ step}+s)\bmod 3.
$$

This keeps per-update work bounded while exposing every scale to all longer
horizons over three updates. The optimizer step is checkpointed, so horizon
rotation resumes exactly.

Longer-horizon rows receive weight 0.5; horizon one receives weight 1.0.

## Normalization and robust loss

The predictor stores a checkpointed second moment for every
scale/component/code coordinate. The exponential decay is 0.99, and the RMS
floor is \(10^{-4}\). These statistics prevent a naturally larger spectral
coordinate from dominating solely because of target scale.

For standardized error

$$
E=\frac{\widehat C-\mathrm{stopgrad}(C)}{r_s+\epsilon},
$$

the row loss is the mean Huber loss over all three components and all code
coordinates. The Huber delta is 1.0. The context objective is the exact sum of
valid weighted row losses divided by the exact valid horizon weight for the
whole packed context.

The denominator is counted before TBPTT graph release with the shared validity
authority. Consequently, changing `tbptt_length` changes gradient truncation
but cannot silently rescale the mathematical auxiliary objective by changing
the number of graph-release groups.

## Optimization authority and gradient firewall

Exact next-token CE is backpropagated normally. CSTM uses `torch.autograd.grad`
to retain a separate auxiliary gradient map. After AMP task gradients are
unscaled and before global gradient clipping:

1. actor parameters with no task gradient reject auxiliary-only updates;
2. within each actor subsystem, a negative task/auxiliary dot product is
   projected out;
3. the projected auxiliary norm is capped relative to that subsystem's task
   norm;
4. the CSTM-only head is the sole explicit auxiliary-only exception and is
   capped relative to the complete task-gradient norm;
5. the admitted result is added to the task gradient;
6. ordinary global finite checking and clipping run on the combined gradient.

Default caps are:

| Subsystem | Maximum CSTM norm relative to task authority |
| --- | ---: |
| Carrier | 0.10 |
| Cognitive subsystems | 0.05 |
| CSTM-only head | 0.10 of global task norm |

The merge fails on unknown parameters, shape/device mismatch, nonfinite
auxiliary gradients, nonfinite task gradients, or invalid caps. Tests prove
conflict projection, local caps, the explicit auxiliary-only exception, and
rejection of unrelated auxiliary-only parameters.

## Schedule

The default schedule is:

- pure CE for the first 100,000 physical tokens;
- linear CSTM ramp over the next 400,000 physical tokens;
- maximum objective coefficient 0.04.

During pure-CE warmup, target predictions are not constructed and no CSTM
auxiliary gradients are allocated. The schedule is driven only by physical
`tokens_seen`; derived target counts cannot accelerate their own schedule.

## Honest accounting

The trainer records:

| Metric | Meaning |
| --- | --- |
| `progress/tokens_seen` | Physical packed corpus token presentations only |
| `cstm/spectral_target_views` | Valid scale/horizon target rows |
| `cstm/weighted_prediction_rows` | Sum of valid horizon weights |
| `cstm/coefficient_targets` | Valid rows times 3 times code width |
| `cstm/raw_token_view_equivalents` | Constituent-token participation across all valid blocks |
| `cstm/supervision_relations_per_primary_target` | Token participation divided by ordinary valid next-token targets |
| `cstm/standardized_huber_sum` | Additive robust spectral loss numerator |
| `cstm/standardized_huber` | Numerator divided by exact weighted rows |

Counters remain additive across gradient-accumulation contexts. Likelihood and
CSTM ratios are recomputed from summed numerators and denominators, including
the final partial context.

For a 32,768-token context with production supports
`(2, 4, 8, 16, 32, 32)` and two active horizons, the geometric upper bound is
two derived rows per physical token. Boundary rejection lowers the realized
count. `raw_token_view_equivalents` can be much larger because a coarse row
contains many tokens; it is not a count of independent examples or corpus
presentations.

## Checkpoint and resume contract

Training checkpoint format 13 binds:

- CSTM enablement and schedule;
- all three gradient caps;
- code width, code seed, rank, horizons, RMS decay/floor, and Huber delta;
- predictor parameters;
- running target second moments and initialization masks;
- physical optimizer step, which determines horizon rotation.

Format-12 and older supported checkpoints migrate by initializing the absent
CSTM head from the current deterministic model construction and binding the new
training contract. A format-13 checkpoint cannot be silently resumed under a
different CSTM identity.

Exact-resume tests interrupt an active two-step run, serialize the model,
optimizer, scheduler, stream, random state, CSTM statistics, and step, resume,
and compare every resulting model tensor with an uninterrupted reference.

## Parameter and compute cost

The shared CSTM head contains:

| Profile | CSTM parameters | Total actor parameters |
| --- | ---: | ---: |
| Ultralight | 2,158 | 1,301,827 |
| Light | 3,361 | 8,416,803 |
| Serious | 5,934 | 115,931,878 |

The codebook is fixed and nonpersistent, so it is not included in trainable
parameter counts or checkpoint size.

For each valid row, target construction performs compact-code lookup and two
real first-harmonic reductions. Prediction uses the shared rank-eight head.
There is no additional MRRN forward, MRCRA cycle, or vocabulary projection.
Validity-only pre-counting is linear in derived rows and avoids all code lookup
and DFT work.

## Test and empirical evidence map

`tests/test_cstm.py` verifies fixed-code identity, DFT equivalence, strict future
indexing, validity-authority equivalence, boundary rejection, order
sensitivity, the support-two Nyquist case, cognitive-gate initialization,
running normalization, differentiable zero loss, malformed-label rejection,
and fail-closed configuration.

`tests/test_model.py` verifies exact multiscale completion histories, chunk
invariance, and future-perturbation causality.

`tests/test_cognitive_training.py` verifies live integrated target generation,
context-wide denominator agreement, governed gradients, pure-CE warmup, honest
token accounting, and exact active-checkpoint continuation.

`tests/test_optimization.py` verifies conflict projection, subsystem caps,
finite enforcement, and the narrowly authorized CSTM-head exception.

`tests/test_cstm_acceptance.py` and `scripts/run_cstm_acceptance.py` run
deterministic end-to-end mechanism experiments. The retained JSON report is
`outputs/cstm_empirical_acceptance.json`.

Run the focused authority with:

```bash
python scripts/run_cstm_acceptance.py
```

Run the repository-wide authority, which includes the focused CSTM authority,
with:

```bash
python scripts/run_mrcra_acceptance.py
```
