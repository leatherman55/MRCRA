# MRCRA 1.3M Ultralight Profile

## Purpose

The ultralight profile is a structurally complete MRCRA actor for fast
end-to-end experiments under severe parameter and compute constraints. It is
not a reduced carrier with cognition disabled, and it is not presented as a
serious language-capability scale.

With the GPT-2 vocabulary fixed at 50,257 entries, vocabulary representation
dominates this budget. The selected design uses a tied 20-wide token and output
embedding, leaving enough capacity to retain every architectural subsystem
without exceeding a narrow 1.29M–1.31M construction-time envelope.

## Exact budget

| Allocation | Parameters | Share |
| --- | ---: | ---: |
| Tied token/input-output embedding | 1,005,140 | 77.21% |
| Six-scale MRRN carrier excluding tied embedding | 202,548 | 15.56% |
| Complete cognitive architecture | 91,981 | 7.07% |
| Shared CSTM predictor | 2,158 | 0.17% |
| **Actor total** | **1,301,827** | **100%** |

The tied weight is counted once. All actor parameters are trainable. CSTM's
shared predictor is included because its predictions and learned statistics are
part of the checkpointed actor. PC-RASL is a separate, experimental
training-only auxiliary system and is not included in the inference actor
count.

## Carrier design

| Property | Value | Rationale |
| --- | ---: | --- |
| Base width | 20 | Largest practical width that preserves a meaningful non-embedding budget |
| Physical scales | 6 | Retains the serious profile's full multiresolution topology |
| Learned refinement passes | 6 | Preserves iterative recurrent depth |
| Learned depth sharing | Enabled | Avoids duplicating narrow block weights while retaining independent per-pass state |
| Scale widths | 20, 24, 24, 24, 24, 24 | Modest coarse-scale capacity without geometric parameter growth |
| Attention heads | 2 | Ten channels per head at the finest scale |
| Base resonance modes | 8 | Preserves a nontrivial pole/frequency bank at width 20 |
| Coarse resonance modes | 11 | Modest coarse-scale mode expansion |
| MIMO rank | 1 | Appropriate low-rank channel coupling at this width |
| Structured mixer rank | 8 | Retains substantial channel interaction without a dense wide mixer |
| Mixer expansion | 2.5 | Spends remaining carrier budget on nonlinear spectral processing |
| Spectral activation modes/order | 5 / 5 | Preserves learned spectralized activation and triadic interactions |
| Local attention window | 32 | Matches larger MRCRA profiles |
| Retrieved items | 4 | Bounded content retrieval appropriate to narrow state |
| Carrier memory capacity | 1,024 | Preserves recurrent retrieval while bounding state cost |
| Relational carrier branch | Enabled | Required MRCRA cognition-to-carrier feedback path |

All six physical scales maintain independent recurrent state even though the
learned block parameters are shared across refinement passes. Parameter sharing
therefore reduces storage without collapsing the multiresolution state
topology.

## Cognitive design

The cognitive workspace remains 20-wide and retains the complete typed
architecture:

- event extraction and hard event allocation;
- typed nodes, pair relations, and explicit hyperrelations;
- relational routing and bounded global workspace;
- episodic and semantic memory with learned keys, values, signatures, and
  write policy;
- conditional reconstruction and localized reconstructive descent;
- abstraction applicability and integrated invariant discovery;
- multiple hypotheses and four world-model horizons;
- epistemic/aleatoric uncertainty channels;
- provenance source and verification prediction;
- internal cognitive controller and action receipts;
- metacognitive routing and self-model projection;
- viability forecasting;
- external-action policy and permission boundary;
- persistent-session and agent-session interfaces.

Runtime capacities are reduced coherently:

| Capacity | Ultralight value |
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

Ontology sizes, uncertainty-channel count, maximum relation arity, cognitive
step depth, and long world-model horizons are intentionally preserved rather
than minimized away. Capacity reductions affect how many simultaneous records
can be represented, not which kinds of reasoning operation exist.

## Causal Spectral Target Multiplexing

The ultralight profile retains the same default CSTM contract as the larger
profiles:

- fixed, non-trainable normalized token codes;
- DC plus the first order-sensitive Fourier harmonic;
- complete strictly future blocks at all six physical scales;
- exact packed-document boundary rejection;
- one always-on next-block horizon plus one rotating longer horizon;
- a shared rank-eight predictor with scale and horizon conditioning;
- a zero-initialized cognitive-residual gate;
- checkpointed per-scale target RMS;
- a separately computed auxiliary gradient governed relative to exact
  next-token CE.

The 2,158-parameter head preserves this complete mechanism while consuming only
0.17% of the actor budget. It does not duplicate the vocabulary projection and
does not cause additional carrier or cognitive forward passes.

## Progress-Conditioned RASL

The ultralight actor uses the same causal Progress-Conditioned RASL authority,
delayed pre-consequence replay, critic firewall, guards, and gradient governor
as larger profiles.

Its production critic has 94,621 parameters, 7.28% of the actor. It retains a
target critic but no duplicate target actor. The actor's exact next-token task
gradient remains authoritative, and auxiliary subsystem caps are unchanged.

## Training integration

Select the profile with:

```bash
python scripts/train_fineweb.py \
  --ultralightmodel \
  --total-tokens 20000000
```

The flag is mutually exclusive with `--lightmodel`. It selects:

- model profile `mrcra_1p3m_ultralight`;
- output directory `outputs/mrcra-1p3m-fineweb-<tokens>-tokens`;
- Trackio run name
  `mrcra-1p3m-ultralight-integrated-fineweb-<tokens>-tokens-32k`;
- model authority `mrcra-ultralight-1p3m-fineweb-stage1`;
- cognitive stride 64 unless explicitly overridden.

An offline smoke test constructs and trains the real 1,301,827-parameter actor
without downloading FineWeb or a tokenizer:

```bash
python scripts/train_fineweb.py \
  --smoke-test \
  --ultralightmodel \
  --no-trackio \
  --no-dashboard \
  --output-dir work/mrcra-ultralight-smoke
```

The smoke tokenizer has the production 50,257-entry tensor width but uses byte
IDs for dependency-free deterministic input. Its manifest explicitly records
that it does not provide GPT-2 token semantics.

## Verification contract

Production tests assert:

- exact actor parameter count and declared range;
- exact CSTM predictor allocation, causal target construction, and governed
  auxiliary gradient route;
- tied token/output storage;
- six physical scales and independent recurrent topology;
- expected width and resonance-mode allocation at every scale;
- presence of every major cognitive parameter group;
- all integrated cognition switches enabled;
- exact experimental PC-RASL critic size and absence of a target actor;
- stable profile, output-directory, authority, and Trackio names;
- CLI mutual exclusion;
- reproducible machine-readable parameter audit;
- successful offline training, evaluation, CSTM, checkpoint, and manifest
  completion using the real ultralight actor; PC-RASL is tested only when
  explicitly enabled.

The exact subsystem allocation is retained in
`outputs/mrcra_1p3m_parameter_report.json`.

## Claim boundary

Structural completeness means every mechanism and authority boundary is
present and executable. It does not imply sufficient width for high-quality
language modeling or general cognitive capability. Because 77.21% of the actor
is necessarily committed to the tied GPT-2 vocabulary embedding, the profile
should be used for mechanism development, causal ablations, training-system
tests, and small-scale empirical comparisons. The 8.4M light and 120M serious
profiles remain the appropriate next steps for representational and capability
evaluation.
