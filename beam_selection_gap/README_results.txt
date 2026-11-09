BEAM SEARCH DIAGNOSTICS — SEARCH COVERAGE VS SELECTION
======================================================

All gains are percent reductions relative to each ORIGINAL plan;
selection_gap_pp is measured in PERCENTAGE POINTS.

best_seen_gain_pct: hindsight oracle = max(0, actual heldout gain of all
  candidates visited by that search). It is NOT an online algorithm result.
selected_gain_pct: measured heldout gain of the method's actual pick.
selection_gap_pp: best_seen_gain_pct - selected_gain_pct.

If the gap is large: candidate selection/scoring left gains on the table
  for the candidate pool already found. Limited-routing selection may help.
If best_seen gain is small: this PARTICULAR search did not encounter
  much improvement. This alone does not prove the operators cannot reach
  better plans; inspect full-space oracle on comparable circuits/starts.
If label coverage < 100%: best_seen is only a LOWER BOUND on the best
  of all discovered candidates, and the measured gap can be understated.

No routing, fitting, feature extraction, or candidate generation was run.
All actual costs are saved heldout routing means from the prior run.
The data was reused during algorithm development: not untouched final test.

RESULTS — EQUAL CIRCUIT WEIGHTS
 objective         policy  n_circuits  n_searches  selected_gain_pct  best_seen_gain_pct  selection_gap_pp  fraction_gap_gt_1pp  fraction_selected_best  fraction_selected_harmful  fraction_positive_found  label_coverage  avg_n_candidates
swap_count  adaptive_beam          15         179              2.762               6.281             3.520                0.621                   0.316                      0.318                    0.888           1.000            49.054
swap_count    random_beam          15         179              2.935               6.221             3.286                0.663                   0.313                      0.311                    0.888           1.000            50.313
swap_count ridge_one_step          15         179              1.862               4.269             2.407                0.547                   0.354                      0.259                    0.875           1.000             6.815

Searches analyzed: 537
Searches with any missing candidate labels: 0

OUTPUT FILES
per_source_oracle_gap.csv: one row per (objective,policy,circuit,source)
per_circuit_summary.csv: one row per (objective,policy,circuit)
macro_summary.csv: circuit-equal summary for each policy
candidates_with_ranks.csv: all scored candidates, marked selected/oracle

IMPORTANT: Never report best_seen as method-achieved gain in the paper.
