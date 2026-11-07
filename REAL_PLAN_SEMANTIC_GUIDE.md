# Real-plan semantic audit — after 22 regression tests pass

This package extends the previously supplied `semantic_correctness.py` and
`run_semantic_correctness.py` **without changing them**. It selects 2 original
reuse plans and 3 **statically valid mutated plans** per benchmark by default
(25 cases on your five benchmark circuits). Uses your locally fixed
`gidnet/qubitreuse.py`.

## What's improved?

Some original QASM benchmarks contain **no measurement instructions**. The old
runner could only check the compiled wire traces (`STRUCTURE_ONLY`) in those
cases. This improved runner has a clearly labeled **terminal-readout probe**:

- makes a copy of the unmeasured original circuit;
- appends measurements of each original logical qubit *at the end*;
- compiles **that same instrumented circuit** using a saved original/mutated
  fixed-width plan;
- compares the *joint classical output probability distribution* of the
  instrumented original and dynamic compilation (when the simulation limits
  permit).

This is useful semantic evidence for the tested classical readout. It is **not**
a proof of full quantum-channel equivalence or correctness for arbitrary
input states. Readout instruments are only created **in memory**; no existing
QASM, notebook, or result file is modified.

## Files

Place `run_real_plan_semantic_audit.py` and `test_real_plan_semantic_probe.py`
in your repository root. The earlier suite's `semantic_correctness.py`,
`run_semantic_correctness.py`, `reuse_validity.py`, and the reset-fix you've
already applied must still be there.

## Run from PowerShell with `.venv_gidnet` active

First check the additional regression tests:

```powershell
python -m pytest -q test_semantic_correctness.py test_reset_boundary_regression.py test_real_plan_semantic_probe.py
```

Then run the real-plan smoke audit:

```powershell
python run_real_plan_semantic_audit.py `
  --bench-dir "C:\Users\ASUS\Documents\Quantum computing\Qiskit\CaQR-main\CaQR-main\benchmarks" `
  --original-csv "experiment2_results\experiment2_plan_features.csv" `
  --mutation-csv "validity_exhaustive_results\adversarial_mutation_cases.csv" `
  --out-dir "semantic_realplan_results"
```

Correct `--bench-dir` if your benchmark QASM directory differs. Do **not**
use the earlier sampled-mutations folder in place of `validity_exhaustive_results`.

If the first 25 cases pass, you can then expand:

```powershell
python run_real_plan_semantic_audit.py `
  --bench-dir "C:\Users\ASUS\Documents\Quantum computing\Qiskit\CaQR-main\CaQR-main\benchmarks" `
  --original-per-circuit 100 `
  --mutated-per-circuit 20 `
  --out-dir "semantic_realplan_extended"
```

This larger stage may take time. To raise resource limits selectively, use
`--max-sim-qubits 9` or `--max-operations 450` **only if feasible**; test a
small subset first. There is no routing or GidNET plan search.

## Interpret results carefully

- `SEMANTIC_PASS`: exact classical-output distribution matches under the
  stated finite simulation assumptions (and structural checks passed).
- `STRUCTURE_ONLY`: compiled width and operations/reset placement passed,
  but **no semantic comparison ran** (typically size/branch limit).
- `COMPILE_OR_STRUCTURE_ERROR` or `SEMANTIC_MISMATCH`: STOP and investigate;
  the runner exits with a nonzero status.

Look at `readout_probe_added` and `verification_target` to distinguish true
source-observable comparisons from added-readout experiments.

## Return these files

- `semantic_realplan_results/semantic_audit_report.json`
- `semantic_realplan_results/semantic_audit_cases.csv`

After examining failures/skips, we can decide whether to add more tests or
move to **reachable physical improvement**: comparing actual SWAP/routed-depth
opportunities in the expanded valid neighborhoods *before* designing beam search.
