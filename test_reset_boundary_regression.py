"""Extra regression test: one 2-qubit gate ends two logical lifetimes.

Run with the existing semantic suite:
    python -m pytest -v test_semantic_correctness.py test_reset_boundary_regression.py
"""
import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit
from semantic_correctness import (
    compile_from_reuse_plan, audit_compilation_structure, compare_exact_semantics,
)


def test_two_retired_qubits_share_last_cx_and_get_separate_resets():
    qc = QuantumCircuit(4, 2)
    qc.h(0)
    qc.cx(0, 1)          # last operation on q0 AND q1
    qc.x(2)
    qc.h(3)
    qc.measure(2, 0)
    qc.measure(3, 1)

    fixed = compile_from_reuse_plan(qc, [[0, 2], [1, 3]], target_width=2)
    report = audit_compilation_structure(fixed)
    resets = [inst for inst in fixed.circuit.data if inst.operation.name == "reset"]
    assert len(resets) == 2
    assert all(len(inst.qubits) == 1 for inst in resets)
    assert {fixed.circuit.find_bit(inst.qubits[0]).index for inst in resets} == {0, 1}
    assert report["status"] == "structurally_verified"
    assert compare_exact_semantics(qc, fixed.circuit)["equal"]
