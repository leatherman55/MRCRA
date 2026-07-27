# MRCRA training-execution acceptance

Overall result: **PASS**.

| Variant | Median step (s) | MAD (s) | tok/s | Peak RSS (MiB) |
|---|---:|---:|---:|---:|
| `legacy_serial_checkpoint_dense_cstm` | 7.174707 | 0.019203 | 142.72 | 709.7 |
| `static_coarse_checkpoint_ce` | 2.877144 | 0.038834 | 355.91 | 895.7 |
| `static_coarse_checkpoint_dense_cstm` | 4.744200 | 0.017836 | 215.84 | 899.7 |
| `static_auto_ce` | 1.758198 | 0.009376 | 582.41 | 2770.2 |
| `static_auto_repaired_cstm` | 1.782547 | 0.026281 | 574.46 | 2795.6 |
| `static_cost_model_auto_repaired_cstm` | 1.667186 | 0.069659 | 614.21 | 2775.0 |
| `compiled_cost_model_auto_repaired_cstm` | 2.241714 | 0.478272 | 456.79 | 2867.1 |

Compiler candidate: `executed` after 19.241s (budget 120.000s, backend `aot_eager`, resolved `compiled_cost_model_auto_repaired_cstm`).

| Criterion | Measurement | Gate | Result |
|---|---:|---:|---:|
| `ce_repaired_vs_coarse_speedup` | 1.63642 ratio | >= 0.9 | PASS |
| `repaired_cstm_vs_repaired_ce_throughput` | 1.05459 ratio | >= 0.6 | PASS |
| `repaired_default_vs_legacy_speedup` | 4.30348 ratio | >= 1.05 | PASS |
| `padding_or_measured_cost_advantage` | 1.0692 normalized gate | >= 1 | PASS |
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
