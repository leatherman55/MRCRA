# MRCRA 2.7M Ultralight Profile

## Purpose

The ultralight profile is a structurally complete MRCRA actor for fast
end-to-end experiments under tight parameter and compute constraints. It is
not a carrier-only ablation, it does not disable cognition, and it is not
presented as a serious language-capability scale.

The previous 1.3M allocation was dominated by its 50,257-token vocabulary and
forced the whole intelligence substrate to width 20. The replacement uses the
additional budget to remove that specific architectural bottleneck. It reaches
width 36 while retaining six physical scales, complete cognition, tied
input/output storage, shared learned depth, and small runtime capacities.

## Selection result

The chosen allocation is not a single-layer enlargement. Candidate widths and
rank allocations were constructed and counted as complete actors. Width 36 is
the strongest balanced point because it:

- is divisible across four carrier and relational heads;
- exceeds the default width-32 eligibility threshold for certified exact
  vocabulary routing;
- leaves 890,211 parameters outside the tied vocabulary embedding;
- supports rank-two cross-channel MIMO coupling;
- preserves six independently stateful physical resolutions;
- remains inside a narrow 2.69M–2.71M construction-time envelope; and
- avoids increasing runtime record capacities merely to consume parameters.

Width 32 left substantial representational budget unused. Width 40 exceeded
3.0M before any useful increase in bounded cognitive capacity. Width 36
therefore provides the most useful carrier/cognition balance at the requested
scale.

## Exact budget

| Allocation | Parameters | Share |
| --- | ---: | ---: |
| Tied token/input-output embedding | 1,809,252 | 67.02% |
| Six-scale MRRN carrier excluding tied embedding | 653,074 | 24.19% |
| Complete cognitive architecture | 234,723 | 8.70% |
| Shared CSTM predictor | 2,414 | 0.09% |
| **Actor total** | **2,699,463** | **100%** |

The tied weight is counted once and all actor parameters are trainable. The
CSTM predictor is included because it is a checkpointed part of the actor.
PC-RASL is an experimental, training-only auxiliary system, is disabled by
default, and is not included in the inference actor count.

## Carrier design

| Property | Value | Rationale |
| --- | ---: | --- |
| Base width | 36 | Strongest complete allocation within the declared budget |
| Physical scales | 6 | Preserves the full multiresolution topology |
| Learned refinement passes | 6 | Preserves iterative recurrent depth |
| Learned depth sharing | Enabled | Retains depth and independent state without duplicating parameters |
| Scale widths | 36, 44, 44, 44, 44, 44 | Adds coarse capacity while bounding geometric growth |
| Attention heads | 4 | Nine channels per finest-scale head and four-way relational factorization |
| Base/coarse resonance modes | 9 / 12 | Expands frequency capacity coherently with representation width |
| MIMO rank | 2 | Doubles learned cross-channel spectral coupling |
| Structured mixer rank | 8 | Retains efficient channel interaction |
| Mixer expansion | 2.5 | Preserves nonlinear spectral processing capacity |
| Spectral activation modes/order | 6 / 6 | Enlarges the learned spectral activation basis |
| Spectral triads per mode | 1 | Preserves bounded nonlinear frequency interaction |
| Local attention window | 32 | Matches the integrated profiles |
| Retrieved items | 4 | Keeps retrieval cost appropriate to an ultralight actor |
| Carrier memory capacity | 1,024 | Preserves recurrent retrieval while bounding state cost |
| Relational carrier branch | Enabled | Preserves the required cognition-to-carrier feedback path |

All physical scales maintain independent recurrent state. Sharing the learned
refinement block reduces storage; it does not collapse the temporal or
multiresolution state topology.

## Cognitive design

The cognitive workspace grows from width 20 to width 36. Relational heads grow
from two to four and typed-relation adapter rank grows from six to eight. Every
integrated mechanism remains present:

- event extraction and hard event allocation;
- typed nodes, pair relations, and explicit hyperrelations;
- relational routing and bounded global workspace;
- episodic and semantic memory with learned keys, values, signatures, and
  write policy;
- conditional reconstruction and localized reconstructive descent;
- abstraction applicability and integrated invariant discovery;
- multiple hypotheses and four world-model horizons;
- uncertainty and provenance prediction;
- the internal controller, action receipts, and metacognitive routing;
- self-model projection and viability forecasting;
- the external-action policy and permission boundary; and
- persistent-session and agent-session interfaces.

Runtime capacities deliberately remain bounded at the earlier ultralight
values:

| Capacity | Value |
| --- | ---: |
| Active events | 64 |
| Pair edges | 256 |
| Hyperrelations | 32 |
| Maximum hyperedge arity | 4 |
| Workspace slots | 6 |
| Default / maximum hypotheses | 2 / 4 |
| Maximum internal cognitive steps | 4 |
| Event chunk / proposal quota | 64 / 4 |
| Recent / landmark candidates | 12 / 4 |
| Episodic / semantic retrieval candidates | 4 / 4 |
| Episodic / semantic memory | 1,024 / 256 |
| Associative depth / budget | 2 / 8 |
| World-model horizons | 1, 4, 16, 64 |

This is intentional. These values determine per-cycle state, memory, and
routing work much more than parameter count. Increasing them would make the
model slower without strengthening the learned representation proportionally.
The new budget is therefore spent on the learned carrier/cognitive substrate,
not on larger empty buffers.

## CSTM and exact-authority vocabulary routing

The profile retains the complete Causal Spectral Target Multiplexing contract:
fixed normalized token codes, order-sensitive Fourier targets, strictly future
complete blocks, document-boundary rejection, multiscale horizons, a shared
conditioned predictor, a zero-initialized cognitive-residual gate, checkpointed
statistics, and governed auxiliary gradients relative to exact next-token CE.
Its complete predictor uses 2,414 parameters.

Unlike the retired width-20 profile, width 36 is eligible for the
`CertifiedBalancedVocabularyRouter`. Generation may therefore use certified
cluster bounds to recover exact top-k results without evaluating every output
row. Certification failure still falls back to the exact dense projection;
routing never changes model authority and contributes no trainable parameters.
Training cross entropy remains exact over the full vocabulary.

## PC-RASL boundary

Progress-Conditioned RASL remains experimental and is disabled in the canonical
FineWeb path. When explicitly enabled, its critic uses 104,077 parameters
(3.86% of the actor), retains a separate target critic, and does not duplicate
the actor. Its auxiliary gradient remains governed and cannot replace exact
next-token CE.

## Training identity and checkpoint isolation

Select the profile with:

```bash
python scripts/train_fineweb.py \
  --ultralightmodel \
  --total-tokens 20000000
```

The flag is mutually exclusive with `--lightmodel`. It now selects:

- model profile `mrcra_2p7m_ultralight`;
- output directory `outputs/mrcra-2p7m-fineweb-<tokens>-tokens`;
- Trackio run name
  `mrcra-2p7m-ultralight-integrated-fineweb-<tokens>-tokens-32k`;
- model authority `mrcra-ultralight-2p7m-fineweb-stage1`; and
- a 128-token production cognitive stride unless explicitly overridden.

The new identity prevents an old 1.3M checkpoint directory or Trackio run from
being silently resumed as the incompatible 2.7M architecture. The full model
configuration remains bound into checkpoint identity.

An offline smoke test constructs and trains the real actor without downloading
FineWeb or a tokenizer:

```bash
python scripts/train_fineweb.py \
  --smoke-test \
  --ultralightmodel \
  --no-trackio \
  --no-dashboard \
  --output-dir work/mrcra-ultralight-smoke
```

## Verification contract

Production tests assert:

- the exact 2,699,463 actor count and declared 2.69M–2.71M envelope;
- tied token/output storage;
- six physical scales with widths 36/44 and modes 9/12;
- four heads, rank-two MIMO coupling, six-mode/order spectral activation, and
  a rank-eight cognitive relation adapter;
- unchanged bounded runtime capacities;
- presence of every major cognitive parameter group;
- all integrated cognition switches enabled;
- the exact 2,414-parameter CSTM predictor;
- bounded opt-in PC-RASL construction without a target actor;
- stable profile, output-directory, authority, smoke, and Trackio identities;
- CLI mutual exclusion and reproducible parameter reporting; and
- successful offline training, evaluation, CSTM, checkpoint, and manifest
  completion using the real ultralight actor.

The exact subsystem allocation is retained in
`outputs/mrcra_2p7m_parameter_report.json`.

## Claim boundary

Structural completeness means every mechanism and authority boundary is present
and executable. It does not imply high-quality language modeling or general
cognitive capability at this scale. The 2.7M profile is for mechanism
development, causal ablations, training-system tests, and small-scale empirical
comparisons. The 8.4M light and 120M serious profiles remain the appropriate
next steps for representational and capability evaluation.
