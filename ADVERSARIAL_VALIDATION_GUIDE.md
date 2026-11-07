# Fixed-width reuse — Adversarial & property-based static validity audit

## Scope

This is a **validation audit**, not Experiment 5's physical optimization search.
It generates mutations of the 50 **existing** reuse plans and tests the
`ReuseValidityChecker` against a separately implemented *static* reference
scheduler. It does **not** rerun GidNET, CaQR, SABRE, or quantum simulation.

The reference scheduler does not call GidNET, NetworkX, `circuit_to_dag`, or
`ReuseValidityChecker`: it constructs event-level constraints directly from
`QuantumCircuit.data` and checks whether a topological schedule exists. Both
approaches share the *same static model*. Agreement cannot prove compiled
reset/measurement **semantic equivalence**.

## Files to place in the project root

- `reuse_adversarial.py` — mutations and separate event-level feasibility oracle
- `test_reuse_validity_adversarial.py` — exhaustive tiny example, invariants and deliberate failures
- `run_validity_adversarial.py` — run audit on all saved Experiment 2 plans
- `validate_adversarial_plans.ipynb` — notebook frontend for the audit and tables
- Keep your existing `reuse_validity.py` and `test_reuse_validity.py` unchanged.

## Start in PowerShell

```powershell
cd "C:\Users\ASUS\Documents\Quantum computing\Qiskit\GidNET-Qubit-Reuse-Algorithm"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv_gidnet\Scripts\Activate.ps1
python -m pytest -q test_reuse_validity.py test_reuse_validity_adversarial.py
```

Then open `validate_adversarial_plans.ipynb`, check `BENCH_DIR` points to the
**same QASM directory** as in `validate_existing_plans.ipynb`, and run all cells.

Alternatively run from PowerShell (edit the path to your QASM folder):

```powershell
python run_validity_adversarial.py --bench-dir "C:\Users\ASUS\Documents\Quantum computing\Qiskit\CaQR-main\CaQR-main\benchmarks" --per-operator 12
```

The runner requires:

- `experiment2_results/experiment2_plan_features.csv`
- The original `.qasm` circuits in `BENCH_DIR`
- Optionally (recommended), `validity_checker_results/validation_of_experiment2_saved_plans.csv`

If the previous validation CSV is present, the runner explicitly compares
`saved_width`, `len(chains)`, `checker_width`, and
`previous_actual_width` for **every saved plan** and fails on differences.

## Mutation families (valid partition and width by construction)

1. `suffix_swap` — exchange two suffixes while keeping prefixes nonempty
2. `chain_transfer` — move one qubit from one nonsingleton chain to another
3. `node_swap_reconnect` — swap two qubits in different chains (reconnect neighboring reuse relations)
4. `segment_reverse` — reverse a portion of a chain as a deliberate adversarial case

Also tests malformed plans with duplicate, missing, out-of-range, empty,
non-integer qubits and target-width mismatch. `--per-operator` controls how
many **unique** candidate mutations of each type are evaluated per saved plan.
The potential unique pool is enumerated first and a deterministic random
sample is evaluated. Pool size is recorded; it is *not* the number of globally
reachable valid plans.

## Output folder: validity_adversarial_results/

| File | Purpose |
| --- | --- |
| `audit_report.json` | PASS/FAIL, totals and known scope limitations |
| `width_and_original_plan_audit.csv` | Per-plan width, original validity, chain-order invariance and reference agreement |
| `adversarial_mutation_cases.csv` | Every selected mutation, checker verdict, reason and independent verdict |
| `malformed_input_tests.csv` | Inputs that must be rejected, with expected and observed reasons |
| `neighborhood_diagnostic_summary.csv` | Static acceptance rates and unique-neighbor pool statistics |
| `rejection_reasons.csv` | Breakdown of checker rejection reasons |
| `oracle_disagreements.csv` | **Must be empty** before trusting the checker |

### How to interpret

- **PASS**: the two implementations agree for all tested cases and structural invariants hold. It does not guarantee they are correct on every possible plan.
- **FAIL / any oracle disagreement**: inspect original circuit, reuse chain and graph-level counterexample. Do not alter the expected answer simply to make the test pass.
- **Valid mutation rate** measures static feasibility, *not* routed depth, SWAP, or physical benefit.
- **Future next step**: end-to-end tests of compilation, resets/measurements, and semantic output distribution; only then move to physical neighbor cost/search experiments.

Tested in this execution environment: Python syntax, six-plus no-Qiskit tests, and a mocked event-level oracle sanity check. Qiskit was not installed here, so **the real Qiskit integration and 50-plan batch must run in your `.venv_gidnet` environment (Qiskit 0.46.0)**.
