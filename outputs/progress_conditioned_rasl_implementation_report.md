# Progress-Conditioned Cognitive Resonant Adjoint Surprise Learner

## Completion status

The canonical FineWeb training path now enables Progress-Conditioned RASL
(PC-RASL) by default. The implementation introduces phase-transition pressure
without reading, optimizing, or checkpointing any phase-transition metric. Its
only meta-learning authority is causal held-out cross entropy in nats per
valid target token, optimizer progress, and an independent held-out regression
guard.

This report describes the implemented production contract. It does not claim
that a large trained model has undergone a beneficial phase transition.

## Causal authority

Three stable document-ID hash partitions have mutually exclusive roles:

1. the training stream supplies the ordinary exact next-token objective;
2. the progress-probe stream supplies the CE trajectory used to measure
   learning progress;
3. the independent guard stream can veto positive pressure and actor-side
   auxiliary updates.

The progress authority observes monotonically increasing valid-target-token
counts, exact probe CE, and the current learning rate. It estimates:

- a Huber-robust recent CE slope;
- a lagged, periodically frozen shifted power-law learning-curve baseline;
- the baseline's expected local slope;
- slope advantage relative to robust slope noise;
- CE level debt relative to the expected curve;
- an evidence confidence determined by causal sample maturity.

The bounded pressure combines slope advantage and progress debt after a
deadband. Faster-than-expected improvement can produce positive pressure.
Plateau, regression, or accumulated debt produces negative pressure. A
nonnegative observed CE slope categorically prevents positive pressure.
Positive pressure is also impossible until an independent guard observation
exists, and persistent guard regression disables it until sustained recovery.

The authority API has no phase-event probabilities, threshold distance, hard
event counts, or phase-transition telemetry. These remain diagnostic observers.

## Delayed consequence and exact behavior binding

Each ordinary training interval captures a bounded, valid, single-document
behavior trajectory before its outcome is known. The capture contains:

- input and behavior tokens;
- explicit candidate token IDs;
- behavior-policy candidate logits;
- exact candidate proposal log probabilities and sampled-candidate masks;
- cognitive, workspace, and relational features;
- relation-family probability evidence;
- internal cognitive action and status receipts;
- causal segment and boundary classes;
- valid-position, termination, and recurrent burn-in masks.

When the next progress-probe observation arrives, its signed bounded pressure is
assigned to trajectories from the preceding interval. This prevents outcome
leakage. The critic evaluates the stored historical state and behavior policy
that actually earned the delayed consequence. The current actor is evaluated
separately only to obtain a live auxiliary gradient.

Candidate construction always includes the observed behavior token and uses at
most the configured bounded count. Random negative candidates use exact
rejection sampling from the remaining vocabulary, avoiding a vocabulary-sized
mask and multinomial operation for every draw while preserving the correct
conditional proposal probability.

## Critic and internal self-learning path

The detached cognitive adjoint critic learns multihorizon distributional
returns, a dedicated progress-return head, immediate consequence, termination,
reverse credit, cognitive transition, uncertainty, memory utility, and values
for bounded internal cognitive actions.

The actor-side functional-surprise objective uses the critic's bounded
candidate consequence distribution. A separate internal-policy term supplies
credit to the cognitive controller's recorded internal actions. Critic
features and action receipts are detached, so critic backpropagation cannot
mutate the actor through the critic path.

The target critic is an exponential moving average of the online critic.
PC-RASL retains no full target-actor copy. It also has zero task-loss weight:
the main trainer remains the sole authority for exact next-token CE.

## Actor-gradient governance

PC-RASL critic learning can begin when finalized replay exists. Actor-side
auxiliary learning remains disabled until:

- the progress authority has a mature baseline;
- the additional critic warmup observation count has elapsed;
- the independent progress guard permits positive pressure where relevant;
- the separate RASL performance guard accepts the current auxiliary proposal.

The ordinary task gradient is calculated first. Auxiliary gradients:

- cannot create a gradient on a parameter with no live task-gradient path;
- are grouped by architectural subsystem;
- are projected away from an aggregate subsystem conflict with the task
  gradient;
- are capped relative to the task-gradient norm;
- preserve the task gradient as the update's primary authority.

Production caps are 2% for the carrier, 10% for general cognitive subsystems,
and 15% for the controller.

## Bounded production resources

The default trajectory length is 256 valid positions with 48 candidates and a
maximum of five retained trajectories per progress interval. One finalized
trajectory is admitted to replay per optimizer update to smooth computational
cost.

For the production 8.4M light profile:

- actor: 8,413,442 parameters;
- PC-RASL critic: 139,537 parameters;
- critic-to-actor ratio: approximately 1.66%;
- target actor: none;
- target critic: present.

Replay tensor storage is computed exactly and reported during training.
Capture, replay/critic update, and progress-probe time are logged separately.

## Exact continuation and migration

Checkpoint format 10 binds:

- the complete progress observations and fitted causal baseline;
- independent guard state and latest guard CE;
- pending and finalized delayed trajectories;
- complete pre-consequence behavior evidence;
- replay contents, priorities, schema, and sequence;
- critic and target-critic parameters;
- functional-surprise calibrator;
- performance guard;
- critic optimizer;
- actor, task optimizer, scheduler, stream, prefetch, RNG, retained runtime,
  provenance, and evidence identities.

A same-format resume is bit-exact under the deterministic production-path test.
Evidence or configuration drift fails closed.

Formats 3 through 9 preserve compatible actor/training continuation but restart
the complete progress authority, critic, replay, and warmup. Pre-v10
checkpoints did not retain every pre-consequence behavior component, so
claiming exact delayed-credit continuation would be causally false.

## Observability

Every progress observation is durably appended to
`progress_metrics.jsonl`. Trackio's **Spectral Network → Learning Progress**
instrument displays:

- probe and guard CE;
- expected CE and observed/expected slopes;
- slope advantage, debt, confidence, and bounded pressure;
- baseline and warmup readiness;
- both guard states and rejection counts;
- critic, progress-return, internal-value, functional-surprise, and
  internal-policy losses;
- replay transitions and exact tensor payload;
- pre-consequence behavior-evidence binding;
- auxiliary-gradient magnitudes and subsystem governance;
- capture, replay-update, probe, and total step time.

The instrument explicitly states that phase-transition telemetry is not an
input to this authority.

## Verification evidence

The hash-bound acceptance workflow records:

- 567 passing Python tests and one intentional self-referential ledger skip
  while rebuilding the manifest;
- a passing post-build executable traceability audit;
- 51 passing frontend tests, lint, and production build;
- passing PC-RASL causal-pressure, anti-gaming, guard-recovery, signed-credit,
  critic-learning, internal-controller, gradient-firewall, and
  gradient-governor gates;
- exact same-format trainer/checkpoint continuation;
- explicit format-8 and format-9 fail-closed migration tests;
- production resource-contract and default-entrypoint tests.

Machine-readable results are retained in
`outputs/pc_rasl_empirical_acceptance.json` and
`outputs/mrcra_acceptance_manifest.json`.

## Claim boundary

The accepted evidence establishes that the intended mechanism exists, is wired
into the default training authority, is causally delayed, cannot read
phase-transition metrics, has bounded resource and gradient authority, learns
signed local consequences in deterministic experiments, and resumes or
migrates honestly.

It does not yet establish that PC-RASL improves a seriously trained checkpoint,
causes a phase transition, or improves open-domain capability. Those are
empirical training claims requiring matched long-run ablations on retained
data.
