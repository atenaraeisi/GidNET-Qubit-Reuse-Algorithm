"""Run from the repository root: python -m pytest -v test_semantic_correctness.py.

These tests REQUIRE Qiskit 0.46 and the GidNET package in the same repo.
Aer-only cross-checks are skipped if qiskit-aer is unavailable.
"""
import copy
import pytest

qiskit = pytest.importorskip('qiskit')

from qiskit import QuantumCircuit
from semantic_correctness import (
    SemanticAuditError, audit_compilation_structure, compare_exact_semantics,
    compare_aer_counts, compile_from_reuse_plan, exact_classical_distribution,
)


def compile_and_audit(qc, chains):
    result = compile_from_reuse_plan(qc, chains, target_width=len(chains))
    structural = audit_compilation_structure(result)
    semantic = compare_exact_semantics(qc, result.circuit)
    assert structural['status'] == 'structurally_verified'
    assert semantic['equal'], semantic
    return result


def test_deterministic_reuse_exact_distribution():
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.x(1)
    qc.measure([0, 1], [0, 1])
    result = compile_and_audit(qc, [[0, 1]])
    assert result.circuit.num_qubits == 1
    assert result.reuse_boundaries == 1
    assert exact_classical_distribution(qc) == {'11': 1.0}


def test_entangled_distribution_with_reuse():
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(2)
    qc.measure([0, 1, 2], [0, 1, 2])
    result = compile_and_audit(qc, [[0, 2], [1]])
    assert result.circuit.num_qubits == 2
    # GHZ-like correlations of 0/1 and independent q2.
    assert set(exact_classical_distribution(qc)) == {'000', '011', '100', '111'}


def test_chain_of_three_has_two_resets():
    qc = QuantumCircuit(3, 3)
    qc.x(0)
    qc.h(1)
    qc.x(2)
    qc.measure([0, 1, 2], [0, 1, 2])
    result = compile_and_audit(qc, [[0, 1, 2]])
    assert result.reuse_boundaries == 2
    assert result.circuit.count_ops().get('reset', 0) == 2


def test_two_entangled_blocks_with_multiple_reuse_boundaries():
    qc = QuantumCircuit(4, 4)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(2)
    qc.cx(2, 3)
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    result = compile_and_audit(qc, [[0, 2], [1, 3]])
    assert result.reuse_boundaries == 2
    assert result.circuit.num_qubits == 2


def test_declared_inactive_qubits_are_singletons():
    qc = QuantumCircuit(4, 4)
    qc.h(0)
    qc.x(1)
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    result = compile_and_audit(qc, [[0, 1]])
    assert result.expected_width == 3
    assert result.full_chains == ((0, 1), (2,), (3,))


@pytest.mark.parametrize('seed', list(range(8)))
def test_varied_random_gates_and_phases(seed):
    import random
    rng = random.Random(seed)
    qc = QuantumCircuit(4, 4)
    choices = ['x', 'h', 'z', 't', 's']
    for i in range(2):
        for q in (0, 1):
            getattr(qc, rng.choice(choices))(q)
    qc.cx(0, 1)
    for i in range(2):
        for q in (2, 3):
            getattr(qc, rng.choice(choices))(q)
    qc.cx(2, 3)
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    compile_and_audit(qc, [[0, 2], [1, 3]])


def _remove_first_op(qc, opname):
    """Deliberately corrupt a compiled circuit, preserving registers."""
    result = QuantumCircuit(*qc.qregs, *qc.cregs)
    removed = False
    for inst in qc.data:
        if inst.operation.name == opname and not removed:
            removed = True
            continue
        result.append(inst.operation, inst.qubits, inst.clbits)
    assert removed, f'No {opname} found'
    return result


def test_negative_control_missing_reset_is_detected():
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.x(1)
    qc.measure([0, 1], [0, 1])
    result = compile_and_audit(qc, [[0, 1]])
    broken = _remove_first_op(result.circuit, 'reset')
    wrong = compare_exact_semantics(qc, broken)
    assert not wrong['equal']
    with pytest.raises(SemanticAuditError, match='wire_trace_mismatch|reset_count_mismatch'):
        audit_compilation_structure(
            type(result)(broken, result.original, result.active_chains,
                         result.full_chains, result.original_to_wire,
                         result.expected_width, result.reuse_boundaries)
        )


def test_negative_control_missing_measurement_is_detected():
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.z(1)
    qc.measure([0, 1], [0, 1])
    result = compile_and_audit(qc, [[0, 1]])
    broken = _remove_first_op(result.circuit, 'measure')
    assert not compare_exact_semantics(qc, broken)['equal']


def test_invalid_plan_and_width_are_rejected_before_compilation():
    qc = QuantumCircuit(2, 2)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    with pytest.raises(SemanticAuditError, match='Not a static-valid reuse plan'):
        compile_from_reuse_plan(qc, [[0, 1]], target_width=1)
    with pytest.raises(SemanticAuditError, match='target_width'):
        compile_from_reuse_plan(qc, [[0], [1]])


def test_unsupported_mid_circuit_operation_is_rejected():
    qc = QuantumCircuit(2, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.x(0)
    qc.h(1)
    with pytest.raises(SemanticAuditError, match='mid_circuit_measurement'):
        compile_from_reuse_plan(qc, [[0], [1]], target_width=2)


def test_lowered_measurement_based_active_reset():
    from gidnet.reset_lowering import lower_reset_to_measure_x
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.h(1)
    qc.measure(1, 1)
    result = compile_and_audit(qc, [[0, 1]])
    lowered = lower_reset_to_measure_x(result.circuit)
    assert lowered.count_ops().get('reset', 0) == 0
    # lowerer introduces a private auxiliary classical measurement.
    assert lowered.num_clbits >= result.circuit.num_clbits
    compared = compare_exact_semantics(qc, lowered)
    assert compared['equal'], compared


def test_exact_simulator_handles_standalone_reset_and_control():
    from qiskit.circuit import ClassicalRegister, QuantumRegister
    qr = QuantumRegister(1, 'q')
    cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(qr, cr)
    qc.x(0)
    qc.reset(0)
    qc.measure(0, 0)
    assert exact_classical_distribution(qc) == {'0': 1.0}
    from gidnet.reset_lowering import lower_reset_to_measure_x
    lowered = lower_reset_to_measure_x(qc)
    assert compare_exact_semantics(qc, lowered)['equal']


def test_aer_native_reset_cross_check():
    pytest.importorskip('qiskit_aer')
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(2)
    qc.measure([0, 1, 2], [0, 1, 2])
    result = compile_and_audit(qc, [[0, 2], [1]])
    check = compare_aer_counts(qc, result.circuit, shots=16384, seed=17)
    assert check['equal_within_sampling_tolerance'], check


def test_regression_reset_after_two_qubit_tail_only_resets_reused_wire():
    """High-value compiler regression.

    q0's last *unitary* operation is CX(q0,q1), and q0 has no terminal
    measurement. Reusing q0 must reset only q0, never both q0 and q1.
    The existing GidNET implementation may fail here; a failure is a genuine
    finding to investigate before trusting arbitrary repaired plans.
    """
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.x(2)
    qc.measure(1, 1)
    qc.measure(2, 2)
    result = compile_from_reuse_plan(qc, [[0, 2], [1]], target_width=2)
    audit_compilation_structure(result)
    compared = compare_exact_semantics(qc, result.circuit)
    assert compared['equal'], compared
