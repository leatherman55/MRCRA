# MRCRA training-execution acceptance

Overall result: **PASS**.

| Variant | Median step (s) | MAD (s) | tok/s | Peak RSS (MiB) |
|---|---:|---:|---:|---:|
| `legacy_serial_checkpoint_dense_cstm` | 5.655144 | 0.045389 | 181.07 | 503.4 |
| `static_coarse_checkpoint_ce` | 1.910570 | 0.005312 | 535.97 | 503.7 |
| `static_coarse_checkpoint_dense_cstm` | 3.165909 | 0.009749 | 323.45 | 528.7 |
| `static_auto_ce` | 1.112763 | 0.014127 | 920.23 | 1156.8 |
| `static_auto_repaired_cstm` | 1.127921 | 0.063923 | 907.87 | 1165.1 |
| `static_cost_model_auto_repaired_cstm` | 0.996469 | 0.069098 | 1027.63 | 1166.9 |
| `compiled_cost_model_auto_repaired_cstm` | 1.351678 | 0.283397 | 757.58 | 1250.4 |

Compiler candidate: `executed` after 14.669s (budget 120.000s, backend `aot_eager`, resolved `compiled_cost_model_auto_repaired_cstm`).

| Criterion | Measurement | Gate | Result |
|---|---:|---:|---:|
| `ce_repaired_vs_coarse_speedup` | 1.71696 ratio | >= 0.9 | PASS |
| `repaired_cstm_vs_repaired_ce_throughput` | 1.11671 ratio | >= 0.6 | PASS |
| `repaired_default_vs_legacy_speedup` | 5.67519 ratio | >= 1.05 | PASS |
| `padding_or_measured_cost_advantage` | 1.14129 normalized gate | >= 1 | PASS |
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
