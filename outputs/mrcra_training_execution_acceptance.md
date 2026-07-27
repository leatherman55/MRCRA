# MRCRA training-execution acceptance

Overall result: **PASS**.

| Variant | Median step (s) | MAD (s) | tok/s | Peak RSS (MiB) |
|---|---:|---:|---:|---:|
| `legacy_serial_checkpoint_dense_cstm` | 138.922216 | 0.179248 | 235.87 | 3338.4 |
| `static_coarse_checkpoint_ce` | 40.333674 | 0.134203 | 812.42 | 5815.3 |
| `static_coarse_checkpoint_dense_cstm` | 66.091437 | 0.031732 | 495.80 | 5782.7 |
| `static_auto_ce` | 40.195104 | 1.146474 | 815.22 | 7866.9 |
| `static_auto_repaired_cstm` | 38.091242 | 1.146101 | 860.25 | 7728.1 |
| `static_cost_model_auto_repaired_cstm` | 40.809826 | 0.509183 | 802.94 | 7936.3 |

Compiler candidate: `timeout` after 300.669s (budget 300.000s, backend `aot_eager`, resolved `static_cost_model_auto_repaired_cstm`).

| Criterion | Measurement | Gate | Result |
|---|---:|---:|---:|
| `ce_repaired_vs_coarse_speedup` | 1.00345 ratio | >= 1 | PASS |
| `repaired_cstm_vs_repaired_ce_throughput` | 0.984937 ratio | >= 0.85 | PASS |
| `repaired_default_vs_legacy_speedup` | 3.40414 ratio | >= 2.5 | PASS |
| `padding_or_measured_cost_advantage` | 1.08532 normalized gate | >= 1 | PASS |
| `target_bijection` | 1 boolean | >= 1 | PASS |
| `sampled_cstm_substrate_vjp_count` | 1 VJPs/context | <= 1 | PASS |
| `sampled_cstm_mean_substrate_vjps` | 0.333333 VJPs/context | <= 0.583333 | PASS |
| `finite_cross_entropy` | 1 boolean | >= 1 | PASS |
| `timing_distributions_complete` | 1 boolean | >= 1 | PASS |
| `phase_timing_contract_complete` | 1 boolean | >= 1 | PASS |
| `evidence_identity_complete` | 1 boolean | >= 1 | PASS |
| `matched_initial_model_optimizer_fixture` | 1 boolean | >= 1 | PASS |
| `compiler_candidate_bounded_and_truthfully_resolved` | 1 boolean | >= 1 | PASS |
| `resolved_variant_names_match` | 1 boolean | >= 1 | PASS |
| `production_profile_contract` | 1 boolean | >= 1 | PASS |

This report proves matched local execution behavior, complete timing distributions, and measured throughput for the named hardware/profile. Learning-quality and long-duration resource acceptance remain separate evidence authorities.
