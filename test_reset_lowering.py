import random

from qiskit import QuantumCircuit

from gidnet.qubitreuse import GidNET
from gidnet.reset_lowering import (
    lower_reset_to_measure_x,
    count_reset_lowering_changes,
)


def print_ops(label, circuit):
    print(f"\n--- {label} ---")
    print(circuit)
    print("num_qubits:", circuit.num_qubits)
    print("num_clbits:", circuit.num_clbits)
    print("count_ops:", dict(circuit.count_ops()))


def conditional_x_count(circuit):
    count = 0

    for instruction in circuit.data:
        op = instruction.operation

        if op.name == "x" and getattr(op, "condition", None) is not None:
            count += 1

    return count


# ============================================================
# TEST 1
# Existing measurement immediately before reset
#
# Expected:
#
# measure q -> c
# reset q
#
# becomes:
#
# measure q -> c
# conditional-X q
#
# No extra measurement should be added.
# ============================================================

print("\n============================================================")
print("TEST 1: measure -> reset")
print("============================================================")

qc1 = QuantumCircuit(1, 1)

qc1.h(0)
qc1.measure(0, 0)
qc1.reset(0)
qc1.h(0)

lowered1 = lower_reset_to_measure_x(qc1)

print_ops("Original", qc1)
print_ops("Lowered", lowered1)

assert qc1.count_ops().get("reset", 0) == 1
assert lowered1.count_ops().get("reset", 0) == 0

assert qc1.count_ops().get("measure", 0) == 1
assert lowered1.count_ops().get("measure", 0) == 1, (
    "Duplicate measurement was added."
)

assert conditional_x_count(lowered1) == 1

assert lowered1.num_qubits == qc1.num_qubits

print("PASS: existing measurement was reused.")


# ============================================================
# TEST 2
# Standalone reset
#
# Expected:
#
# reset q
#
# becomes:
#
# measure q -> temporary classical bit
# conditional-X q
# ============================================================

print("\n============================================================")
print("TEST 2: standalone reset")
print("============================================================")

qc2 = QuantumCircuit(1)

qc2.x(0)
qc2.reset(0)
qc2.h(0)

lowered2 = lower_reset_to_measure_x(qc2)

print_ops("Original", qc2)
print_ops("Lowered", lowered2)

assert qc2.count_ops().get("reset", 0) == 1
assert lowered2.count_ops().get("reset", 0) == 0

assert lowered2.count_ops().get("measure", 0) == 1
assert conditional_x_count(lowered2) == 1

assert lowered2.num_qubits == qc2.num_qubits

assert lowered2.num_clbits == 1, (
    "Standalone reset should create exactly one auxiliary classical bit."
)

print("PASS: standalone reset was lowered correctly.")


# ============================================================
# TEST 3
# Measurement on q0, unrelated operation on q1, then reset q0.
#
# The measurement must still be reusable because q0 has not
# been touched between measurement and reset.
# ============================================================

print("\n============================================================")
print("TEST 3: unrelated operation between measure and reset")
print("============================================================")

qc3 = QuantumCircuit(2, 1)

qc3.h(0)
qc3.measure(0, 0)

qc3.x(1)

qc3.reset(0)

lowered3 = lower_reset_to_measure_x(qc3)

print_ops("Original", qc3)
print_ops("Lowered", lowered3)

assert lowered3.count_ops().get("reset", 0) == 0

assert lowered3.count_ops().get("measure", 0) == 1, (
    "Measurement should have been reused because only q1 "
    "was touched between measurement(q0) and reset(q0)."
)

assert conditional_x_count(lowered3) == 1

print("PASS: unrelated operations do not invalidate measurement.")


# ============================================================
# TEST 4
# Measurement followed by another operation on SAME qubit.
#
# Existing measurement must NOT be reused.
#
# measure q
# X q
# reset q
#
# should cause a new measurement before active reset.
# ============================================================

print("\n============================================================")
print("TEST 4: same-qubit operation invalidates measurement")
print("============================================================")

qc4 = QuantumCircuit(1, 1)

qc4.measure(0, 0)
qc4.x(0)
qc4.reset(0)

lowered4 = lower_reset_to_measure_x(qc4)

print_ops("Original", qc4)
print_ops("Lowered", lowered4)

assert lowered4.count_ops().get("reset", 0) == 0

assert lowered4.count_ops().get("measure", 0) == 2, (
    "A new measurement should have been inserted because q0 "
    "was modified after the old measurement."
)

assert conditional_x_count(lowered4) == 1

print("PASS: stale measurement was not reused.")


# ============================================================
# TEST 5
# Integration test with actual GidNET.
#
# Independent logical qubits should have obvious reuse
# opportunities.
#
# Since every original qubit is measured, reuse boundaries
# should look like:
#
# measure
# reset
#
# therefore lowering should NOT need additional measurements
# at those boundaries.
# ============================================================

print("\n============================================================")
print("TEST 5: GidNET integration")
print("============================================================")

random.seed(12345)

qc5 = QuantumCircuit(4, 4)

qc5.x(0)
qc5.h(1)
qc5.z(2)
qc5.s(3)

qc5.measure(0, 0)
qc5.measure(1, 1)
qc5.measure(2, 2)
qc5.measure(3, 3)

print_ops("Original static circuit", qc5)

gidnet = GidNET(qc5)

native5 = gidnet.compile_to_dynamic_circuit(
    iterations=20,
    draw=False,
)

print_ops("GidNET native-reset circuit", native5)

native_reset_count = native5.count_ops().get("reset", 0)

assert native5.num_qubits <= qc5.num_qubits

if native5.num_qubits < qc5.num_qubits:
    expected_boundaries = qc5.num_qubits - native5.num_qubits

    assert native_reset_count == expected_boundaries, (
        f"Expected {expected_boundaries} reset/reuse boundaries, "
        f"but found {native_reset_count} resets."
    )
else:
    print(
        "WARNING: GidNET did not reduce the width in this run. "
        "Integration lowering will still be checked."
    )

lowered5 = lower_reset_to_measure_x(native5)

print_ops("GidNET lowered circuit", lowered5)

summary = count_reset_lowering_changes(native5, lowered5)

print("\nLowering summary:")
for key, value in summary.items():
    print(f"  {key}: {value}")

assert lowered5.count_ops().get("reset", 0) == 0

assert lowered5.num_qubits == native5.num_qubits

assert conditional_x_count(lowered5) == native_reset_count, (
    "Every native reset should produce exactly one conditional X."
)

# In this test circuit every logical qubit already ends in a measurement.
# Therefore reuse-boundary reset lowering should normally be able to reuse
# those measurements rather than adding new ones.
if native_reset_count > 0:
    assert (
        lowered5.count_ops().get("measure", 0)
        == native5.count_ops().get("measure", 0)
    ), (
        "Unexpected extra measurement was inserted at a GidNET "
        "reuse boundary."
    )

print("PASS: GidNET integration looks correct.")


# ============================================================
# TEST 6
# Make sure input circuits are NOT mutated.
# ============================================================

print("\n============================================================")
print("TEST 6: lowering does not mutate input")
print("============================================================")

qc6 = QuantumCircuit(1)

qc6.x(0)
qc6.reset(0)

before_ops = dict(qc6.count_ops())
before_qubits = qc6.num_qubits
before_clbits = qc6.num_clbits

_ = lower_reset_to_measure_x(qc6)

assert dict(qc6.count_ops()) == before_ops
assert qc6.num_qubits == before_qubits
assert qc6.num_clbits == before_clbits

print("PASS: original circuit was not modified.")


print("\n")
print("============================================================")
print("ALL STRUCTURAL TESTS PASSED")
print("============================================================")
