# Fixed-width reuse-plan structural forensics — exploratory pilot

No new routing and no new plan generation were performed. A comparison here is **descriptive and hypothesis-generating**, not proof that a feature causes physical cost.

## alu-bdd_288.qasm
- Matching reuse plans: 6; active width: 5.
- Best by 30-seed routed depth: **P2** (78.333); worst: **P5** (82.633).
- Worst − best: **4.300** depth, **1.700** SWAP. SWAP-best/worst may be different plan IDs.
- Matched routing-seed depth gap (approx. 95% t interval, descriptive): [3.648, 4.952].
- Differences from feature data and newly derived pre-routing features are in `feature_differences.csv` and `per_plan_features.csv`.

| Pre-routing feature | Best | Worst | Worst − Best |
|---|---:|---:|---:|
| max_weighted_degree_rebuilt | 20 | 25 | +5 |
| weighted_degree_cv_rebuilt | 0.326 | 0.3391 | +0.01302 |
| partner_switch_rate | 0.9577 | 0.9577 | +0 |
| critical_reuse_boundaries | 0 | 0 | +0 |
| pre_routing_asap_gate_depth | 48 | 48 | +0 |
| mean_window_2q_chain_fraction | 0.2854 | 0.2697 | -0.01569 |

### Exploratory across-plan signals
The feature-depth correlations below are within this circuit, across currently available plans only. They are *not independently validated*, and multiple features were inspected.

| Feature | Spearman with routed depth |
|---|---:|
| max_weighted_degree_rebuilt | +0.621 |
| max_chain_partner_switches | +0.621 |
| quotient_unique_edges_rebuilt | +0.414 |
| mean_unique_partners | +0.414 |
| weighted_degree_cv_rebuilt | +0.145 |
| n_multi_logical_chains | +0.131 |
| max_pair_interaction_weight | -0.093 |
| max_chain_length | -0.131 |
| chain_length_cv | -0.131 |
| mean_window_2q_chain_fraction | -0.200 |
| max_window_2q_chain_fraction | -0.494 |

## mini_alu_305.qasm
- Matching reuse plans: 16; active width: 6.
- Best by 30-seed routed depth: **P5** (150.967); worst: **P0** (165.233).
- Worst − best: **14.267** depth, **1.200** SWAP. SWAP-best/worst may be different plan IDs.
- Matched routing-seed depth gap (approx. 95% t interval, descriptive): [11.462, 17.071].
- Differences from feature data and newly derived pre-routing features are in `feature_differences.csv` and `per_plan_features.csv`.

| Pre-routing feature | Best | Worst | Worst − Best |
|---|---:|---:|---:|
| max_weighted_degree_rebuilt | 30 | 41 | +11 |
| weighted_degree_cv_rebuilt | 0.1305 | 0.3497 | +0.2192 |
| partner_switch_rate | 0.9662 | 0.9459 | -0.02027 |
| critical_reuse_boundaries | 3 | 2 | -1 |
| pre_routing_asap_gate_depth | 89 | 97 | +8 |
| mean_window_2q_chain_fraction | 0.2442 | 0.2919 | +0.04766 |

### Exploratory across-plan signals
The feature-depth correlations below are within this circuit, across currently available plans only. They are *not independently validated*, and multiple features were inspected.

| Feature | Spearman with routed depth |
|---|---:|
| critical_2q_fraction | +0.820 |
| critical_2q_count | +0.820 |
| max_chain_partner_switches | +0.723 |
| max_weighted_degree_rebuilt | +0.632 |
| weighted_degree_cv_rebuilt | +0.588 |
| max_window_2q_chain_fraction | +0.511 |
| mean_window_2q_chain_fraction | +0.471 |
| mean_pair_run_length | +0.457 |
| pre_routing_asap_gate_depth | +0.341 |
| n_multi_logical_chains | +0.252 |
| max_pair_interaction_weight | +0.241 |
| quotient_unique_edges_rebuilt | -0.250 |
| mean_unique_partners | -0.250 |
| max_chain_length | -0.252 |
| chain_length_cv | -0.281 |
| partner_switches_total | -0.457 |
| partner_switch_rate | -0.457 |
| max_2q_events_in_plan_bin | -0.653 |
| temporal_bin_2q_cv | -0.724 |
| critical_reuse_boundaries | -0.751 |
| critical_reuse_fraction | -0.751 |

## rd73_140.qasm
- Matching reuse plans: 6; active width: 5.
- Best by 30-seed routed depth: **P4** (212.867); worst: **P0** (226.633).
- Worst − best: **13.767** depth, **7.733** SWAP. SWAP-best/worst may be different plan IDs.
- Matched routing-seed depth gap (approx. 95% t interval, descriptive): [12.864, 14.669].
- Differences from feature data and newly derived pre-routing features are in `feature_differences.csv` and `per_plan_features.csv`.

| Pre-routing feature | Best | Worst | Worst − Best |
|---|---:|---:|---:|
| max_weighted_degree_rebuilt | 50 | 56 | +6 |
| weighted_degree_cv_rebuilt | 0.2049 | 0.2843 | +0.07939 |
| partner_switch_rate | 0.8916 | 0.8818 | -0.009852 |
| critical_reuse_boundaries | 4 | 4 | +0 |
| pre_routing_asap_gate_depth | 131 | 131 | +0 |
| mean_window_2q_chain_fraction | 0.2169 | 0.2101 | -0.006883 |

### Exploratory across-plan signals
The feature-depth correlations below are within this circuit, across currently available plans only. They are *not independently validated*, and multiple features were inspected.

| Feature | Spearman with routed depth |
|---|---:|
| max_weighted_degree_rebuilt | +0.828 |
| max_pair_interaction_weight | +0.794 |
| weighted_degree_cv_rebuilt | +0.714 |
| mean_pair_run_length | +0.679 |
| max_chain_partner_switches | +0.555 |
| critical_2q_count | +0.441 |
| critical_2q_fraction | +0.441 |
| chain_length_cv | +0.213 |
| max_2q_events_in_plan_bin | +0.000 |
| pre_routing_asap_gate_depth | +0.000 |
| critical_reuse_fraction | +0.000 |
| critical_reuse_boundaries | +0.000 |
| max_chain_length | +0.000 |
| temporal_bin_2q_cv | +0.000 |
| n_multi_logical_chains | -0.338 |
| mean_unique_partners | -0.414 |
| quotient_unique_edges_rebuilt | -0.414 |
| mean_window_2q_chain_fraction | -0.600 |
| partner_switches_total | -0.679 |
| partner_switch_rate | -0.679 |
| max_window_2q_chain_fraction | -0.741 |

## sym9_146.qasm
- Matching reuse plans: 9; active width: 5.
- Best by 30-seed routed depth: **P1** (306.267); worst: **P5** (321.333).
- Worst − best: **15.067** depth, **11.433** SWAP. SWAP-best/worst may be different plan IDs.
- Matched routing-seed depth gap (approx. 95% t interval, descriptive): [12.208, 17.926].
- Differences from feature data and newly derived pre-routing features are in `feature_differences.csv` and `per_plan_features.csv`.

| Pre-routing feature | Best | Worst | Worst − Best |
|---|---:|---:|---:|
| max_weighted_degree_rebuilt | 71 | 82 | +11 |
| weighted_degree_cv_rebuilt | 0.1988 | 0.2758 | +0.07695 |
| partner_switch_rate | 0.9003 | 0.8935 | -0.006873 |
| critical_reuse_boundaries | 6 | 6 | +0 |
| pre_routing_asap_gate_depth | 187 | 187 | +0 |
| mean_window_2q_chain_fraction | 0.1963 | 0.1851 | -0.01116 |

### Exploratory across-plan signals
The feature-depth correlations below are within this circuit, across currently available plans only. They are *not independently validated*, and multiple features were inspected.

| Feature | Spearman with routed depth |
|---|---:|
| mean_pair_run_length | +0.552 |
| critical_2q_fraction | +0.455 |
| critical_2q_count | +0.455 |
| max_chain_partner_switches | +0.402 |
| max_weighted_degree_rebuilt | +0.368 |
| weighted_degree_cv_rebuilt | +0.300 |
| pre_routing_asap_gate_depth | +0.274 |
| critical_reuse_fraction | +0.274 |
| critical_reuse_boundaries | +0.274 |
| max_pair_interaction_weight | +0.222 |
| chain_length_cv | +0.184 |
| max_chain_length | +0.183 |
| n_multi_logical_chains | -0.149 |
| mean_unique_partners | -0.274 |
| quotient_unique_edges_rebuilt | -0.274 |
| mean_window_2q_chain_fraction | -0.433 |
| max_window_2q_chain_fraction | -0.495 |
| partner_switches_total | -0.552 |
| partner_switch_rate | -0.552 |
| max_2q_events_in_plan_bin | -0.606 |
| temporal_bin_2q_cv | -0.606 |

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

- rd73_140.qasm: P4 vs P0, identical logical depth 139; depth gap 13.767; same hotspot? False.
- rd73_140.qasm: P3 vs P2, identical logical depth 142; depth gap 9.133; same hotspot? False.
- sym9_146.qasm: P0 vs P5, identical logical depth 197; depth gap 9.067; same hotspot? False.
- rd73_140.qasm: P3 vs P1, identical logical depth 142; depth gap 8.567; same hotspot? False.
- system9.qasm: P3 vs P1, identical logical depth 176; depth gap 7.400; same hotspot? False.
- rd73_140.qasm: P5 vs P2, identical logical depth 142; depth gap 6.833; same hotspot? True.
- rd73_140.qasm: P5 vs P1, identical logical depth 142; depth gap 6.267; same hotspot? True.
- sym9_146.qasm: P8 vs P4, identical logical depth 200; depth gap 5.167; same hotspot? True.

**Important:** Five-circuit best/worst trends may be driven by the selection of extremes. They do not establish within-circuit predictive reliability for all plans.
