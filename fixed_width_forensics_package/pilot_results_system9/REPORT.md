# Fixed-width reuse-plan structural forensics — exploratory pilot

No new routing and no new plan generation were performed. A comparison here is **descriptive and hypothesis-generating**, not proof that a feature causes physical cost.

## system9.qasm
- Matching reuse plans: 13; active width: 7.
- Best by 30-seed routed depth: **P11** (283.900); worst: **P1** (304.433).
- Worst − best: **20.533** depth, **5.600** SWAP. SWAP-best/worst may be different plan IDs.
- Matched routing-seed depth gap (approx. 95% t interval, descriptive): [16.908, 24.159].
- Differences from feature data and newly derived pre-routing features are in `feature_differences.csv` and `per_plan_features.csv`.

| Pre-routing feature | Best | Worst | Worst − Best |
|---|---:|---:|---:|
| max_weighted_degree_rebuilt | 65 | 71 | +6 |
| weighted_degree_cv_rebuilt | 0.3839 | 0.4854 | +0.1015 |
| partner_switch_rate | 0.9215 | 0.9078 | -0.01365 |
| critical_reuse_boundaries | 4 | 4 | +0 |
| pre_routing_asap_gate_depth | 160 | 168 | +8 |
| mean_window_2q_chain_fraction | 0.277 | 0.2829 | +0.005881 |

### Exploratory across-plan signals
The feature-depth correlations below are within this circuit, across currently available plans only. They are *not independently validated*, and multiple features were inspected.

| Feature | Spearman with routed depth |
|---|---:|
| quotient_unique_edges_rebuilt | +0.384 |
| mean_unique_partners | +0.384 |
| mean_window_2q_chain_fraction | +0.341 |
| max_pair_interaction_weight | +0.183 |
| mean_pair_run_length | +0.075 |
| chain_length_cv | +0.071 |
| max_chain_length | +0.003 |
| max_weighted_degree_rebuilt | -0.006 |
| temporal_bin_2q_cv | -0.019 |
| partner_switches_total | -0.075 |
| partner_switch_rate | -0.075 |
| pre_routing_asap_gate_depth | -0.089 |
| weighted_degree_cv_rebuilt | -0.110 |
| n_multi_logical_chains | -0.148 |
| max_chain_partner_switches | -0.165 |
| critical_2q_count | -0.182 |
| critical_2q_fraction | -0.182 |
| max_window_2q_chain_fraction | -0.253 |
| critical_reuse_boundaries | -0.296 |
| critical_reuse_fraction | -0.296 |
| max_2q_events_in_plan_bin | -0.463 |

## Interpretation rules

1. Existing metrics (hotspot, weighted degree CV) are used as sanity checks, not claimed to be novel.
2. New candidates include partner switching/interaction run fragmentation, critical reuse boundaries, and temporal bin pressure. A strong association would be a *hypothesis*, not causality.
3. Selecting best and worst on routing outcomes creates selection bias; require confirmation on other circuits or held-out plans/seeds.
4. Plan-dependent DAG layering assumes one gate-time unit per operation and ignores physical layout/routing; it is a deliberately cheap *pre-routing proxy*, not a simulation of real execution time.
5. The QASM2 parser supports static gate-only single-qreg circuits. It stops rather than silently interpreting reset/measurement/classical conditions. Plan semantics and reset correctness must be tested elsewhere.
6. Absence of other matching original QASMs is a real data limit; do not claim cross-circuit structural validation when only system9 is matched.

## Existing-feature five-circuit comparison (not deep QASM reconstruction)

The best/worst comparisons for **all five** original benchmarks use existing Experiment 2 features, while novel gate/chain reconstruction is limited to matched QASM files.

| Circuit | Best / Worst | Routed depth best → worst | MaxWD best → worst |
|---|---|---:|---:|
| alu-bdd_288.qasm | P2 / P5 | 78.33 → 82.63 | 20 → 25 |
| mini_alu_305.qasm | P5 / P0 | 150.97 → 165.23 | 30 → 41 |
| rd73_140.qasm | P4 / P0 | 212.87 → 226.63 | 50 → 56 |
| sym9_146.qasm | P1 / P5 | 306.27 → 321.33 | 71 → 82 |
| system9.qasm | P11 / P1 | 283.90 → 304.43 | 65 → 71 |

## Matched normalized-logical-depth pairs in reconstructed circuits

- system9.qasm: P3 vs P1, identical logical depth 176; depth gap 7.400; same hotspot? False.
- system9.qasm: P5 vs P4, identical logical depth 175; depth gap 4.867; same hotspot? True.
- system9.qasm: P2 vs P9, identical logical depth 178; depth gap 4.100; same hotspot? False.
- system9.qasm: P7 vs P8, identical logical depth 177; depth gap 2.233; same hotspot? False.
- system9.qasm: P2 vs P12, identical logical depth 178; depth gap 2.233; same hotspot? False.
- system9.qasm: P12 vs P9, identical logical depth 178; depth gap 1.867; same hotspot? False.
- system9.qasm: P10 vs P6, identical logical depth 174; depth gap 1.467; same hotspot? True.

**Important:** Five-circuit best/worst trends may be driven by the selection of extremes. They do not establish within-circuit predictive reliability for all plans.
