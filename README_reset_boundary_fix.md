# GidNET reset-boundary regression fix

Your Qiskit 0.46 tests found one compiler bug: a logical qubit can end its
lifetime at a multi-qubit operation, but the existing compiler passes **all**
operands of that operation to one `Reset` instruction. A Qiskit `reset` has
only one quantum operand.

## Included

- `apply_gidnet_reset_fix.py`: applies a narrowly scoped change to the local
  `gidnet/qubitreuse.py`. It verifies 5 exact source patterns, checks Python
  syntax, and creates a `.bak` copy before modifying the file. It does not
  modify test results or GitHub.
- `test_reset_boundary_regression.py`: independently tests the case in which
  one two-qubit gate is the final operation of **both** reused qubits. The
  existing `test_semantic_correctness.py` already tests one retired operand.

## Run in PowerShell, from the repository root

Activate your existing `.venv_gidnet` environment, then:

```powershell
python apply_gidnet_reset_fix.py
python -m pytest -v test_semantic_correctness.py test_reset_boundary_regression.py
```

If both pass, continue with:

```powershell
python -m pytest -q test_reuse_validity.py test_reuse_validity_adversarial.py test_semantic_correctness.py test_reset_boundary_regression.py
```

**Do not run the real-plan audit yet** if any test fails. Share the complete
failure output for review. Re-running the old expensive routing experiments
is not needed for this patch.

## Technical change

When `add_qubit_reuse_edges()` finds the terminal `DAGOpNode` of a reused
logical qubit, it now records **which logical qubit** expires there, not just
which operation expires. When the compiler visits that operation, it emits
one single-qubit reset per expired logical qubit, on its mapped physical wire.

This solves the observed multi-qubit reset arity bug. It does **not** prove
complete compiler correctness. In particular, zero-operation qubit lifetimes,
unsupported dynamic classical control, and semantic equivalence outside the
tested circuits remain separate issues.

To revert the patch if needed, restore the automatically saved
`gidnet/qubitreuse.py.before_reset_wire_fix.bak` file.
