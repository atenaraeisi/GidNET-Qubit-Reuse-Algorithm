# End-to-end semantic checks for fixed-width qubit-reuse Repair

## What this checks

1. **Static admission:** the existing generator-independent `ReuseValidityChecker` must accept a proposed active reuse plan at the fixed target width.
2. **Fixed-plan compilation:** the existing GidNET *compilation* code is used with that plan forced in; the GidNET search itself is **not** rerun or altered.
3. **Structural audit:** original operations (including their mapped qubits and classical bits) must all be preserved; each physical reuse chain must have exactly one **single-qubit reset at each boundary in the correct position**, and operations per wire must remain in order. This catches missing/misplaced reset insertion and incorrect gate remapping, even if operation counts alone match.
4. **Exact ideal output comparison on tiny circuits:** a deliberately small, independent branch-by-branch statevector interpreter explicitly handles terminal measurements, native resets, and classical conditions created by reset lowering. It compares the **full classical output distributions** of original and compiled circuits for the standard `|0...0>` input. There are negative controls: removing a necessary reset or a measurement must cause a difference.
5. **Optional Aer shot comparison:** an additional independent sampling implementation, if `qiskit_aer` is installed.

**What this does not prove:** equivalence as quantum channels on arbitrary input states, arbitrary feedback/control-flow circuits, error/fidelity or physical routing correctness. The static checker deliberately rejects unsupported sources. Large benchmark circuits may be **STRUCTURE_ONLY**, not `SEMANTIC_PASS`.

## Files to put next to `reuse_validity.py` at repository root

- `semantic_correctness.py` — compiler adapter + structural audit + exact ideal interpreter
- `test_semantic_correctness.py` — small circuit tests + negative controls + multi-qubit terminal reset regression
- `run_semantic_correctness.py` — limited audit of saved original **and modified** reuse plans from existing CSVs

No modifications to `gidnet/qubitreuse.py` or the existing experiments are required.

## Windows PowerShell instructions

From your repository root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv_gidnet\Scripts\Activate.ps1
python -c "import qiskit; print(qiskit.__version__)"
python -m pytest -v test_semantic_correctness.py
```

You already have Qiskit 0.46.0 in `.venv_gidnet`. If you have not installed pytest in that environment, run `python -m pip install pytest` only once.

> **Do not interpret a failing test as a failure of the research hypothesis.** Some tests are specifically designed to reveal compiler bugs. In particular `test_regression_reset_after_two_qubit_tail_only_resets_reused_wire` checks whether a reset following a two-qubit last operation resets *only the outgoing reused qubit*. This is a plausible implementation bug in the current compiler, so the test is **not marked expected-to-fail**; we want to see the failure and repair the compiler if needed.

## Test saved real plans

**Prerequisites:** preserve the original `experiment2_results/experiment2_plan_features.csv`, the output `validity_exhaustive_results/adversarial_mutation_cases.csv`, and your benchmark `.qasm` files. The exact paths of the benchmark QASM files are not bundled in this package.

```powershell
python run_semantic_correctness.py `
  --bench-dir "C:\Users\ASUS\Documents\Quantum computing\Qiskit\CaQR-main\CaQR-main\benchmarks" `
  --original-csv "experiment2_results\experiment2_plan_features.csv" `
  --mutation-csv "validity_exhaustive_results\adversarial_mutation_cases.csv" `
  --out-dir "semantic_correctness_results"
```

**IMPORTANT:** Replace `--bench-dir` with your *actual* existing benchmark directory if it differs. This runner defaults to **two original plans plus three valid mutations per circuit** (25 cases across five circuits), selecting diverse operators. You can expand to `--original-per-circuit 10 --mutated-per-circuit 10` after the first smoke run.

The runner **does not** run routing, random GidNET generation or noisy hardware simulations. It compiles the fixed plan and audits structure; it only simulates exact distributions if small enough, with limits `--max-sim-qubits 7 --max-branches 4096 --max-operations 250`. It does **not** silently claim that a skipped circuit is semantically verified.

Outputs:

- `semantic_correctness_results/semantic_audit_cases.csv`
- `semantic_correctness_results/semantic_audit_report.json`

Rows with `STRUCTURE_ONLY` need a separate semantics strategy if required; rows with `SEMANTIC_MISMATCH` or `COMPILE_OR_STRUCTURE_ERROR` must be investigated, not ignored. The process exits with code 1 if any failure is found.

## Recommended interpretation

- Small tests all pass and mutations pass structure → move to *Reachable Physical Improvement* with a known tested compilation subset.
- Small tests fail on a compiler transformation → first fix that compiler behavior, then rerun the semantic tests (keep the failure-inducing example as a permanent regression).
- Existing 50 plans pass structure but most benchmark circuits skip simulation → **do not** call all 50 semantically proved; report exactly how many small-circuit semantic checks were actually run.

## What to send back to ChatGPT

1. The complete pytest output, especially any traceback (even if there are failures).
2. `semantic_correctness_results/semantic_audit_report.json`
3. `semantic_correctness_results/semantic_audit_cases.csv`

These will determine whether it is safe to start physical optimization, or whether the GidNET plan compiler requires a targeted reset fix first.
