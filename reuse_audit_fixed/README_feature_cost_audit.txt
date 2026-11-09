Pre-routing Feature Cost Audit for fixed-width qubit-reuse repair
=============================================================

PURPOSE
  Measures the COMPUTATION TIME and distribution of the 25 features returned by
  exp4d_features.py:plan_features for already valid original reuse plans and a
  small, capped sample of valid one-step neighbors. Uses only static features;
  does not route, fit a model, label gains, or select features by prediction quality.

REQUIRED FILES IN THE WORKING DIRECTORY
  feature_cost_audit.py
  exp4d_features.py (your supplied Experiment 4D file, rename from exp4d_features(1).py)
  reuse_validity.py
  reuse_adversarial.py

REQUIRED INPUT FOLDERS
  .\fixed_full\original_plans.csv  and circuit_audit.csv
  .\wave2\original_plans.csv       and circuit_audit.csv
  Original QASM files must still be at the original Windows paths recorded in circuit_audit.csv.

In PowerShell, activate your existing caqr-qiskit045-env environment, then:
  cd "C:\Users\ASUS\Downloads\reuse_audit_fixed"

  python .\feature_cost_audit.py --audit-dirs .\fixed_full .\wave2 `
    --feature-code .\exp4d_features.py --out-dir .\feature_audit_smoke `
    --only-circuits cnt3-5_179.qasm,sym6_316.qasm

  python .\feature_cost_audit.py --audit-dirs .\fixed_full .\wave2 `
    --feature-code .\exp4d_features.py --out-dir .\feature_audit_full

OUTPUT
  circuit_summary.csv: parse/validity errors, feature profiling coverage/timing
  feature_values.csv: raw features and wall time of each feature family
  feature_summary.csv: missing fractions and within-circuit feature variation
  known_redundancies.csv: checks three mathematically redundant feature pairs
  high_correlations_exploratory.csv: high rank correlations in sampled deltas
  audit_readme.json: summary and limitations

CAVEATS
  - No numerical relevance to SWAP/depth can be inferred yet: routing labels needed.
  - Only 25 features in exp4d_features.py are timed. The extra depth columns
    imported from exp4c1 features.csv (logical_depth, norm_depth) have not been
    benchmarked; they may have additional compilation costs.
  - Per-candidate timing measures the Python implementation, not optimized code.
  - Sample uses at most 2 GidNET originals per circuit, and 6 accepted neighbors
    per operator; time and correlations are only exploratory.
  - Screening may have used a compacted circuit; here the original QASM is
    rechecked to reject unsupported mid-circuit measurement/classical control.
  - All read operations; no original QASM, model, or research output overwritten.
