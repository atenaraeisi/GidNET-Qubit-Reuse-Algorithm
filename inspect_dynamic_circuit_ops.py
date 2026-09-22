"""Inspect reset/measure operations emitted by GidNET on a tiny circuit."""

import random

from qiskit import QuantumCircuit
from gidnet.qubitreuse import GidNET


def bit_indices(circuit, bits):
    """Return stable bit indices across supported Qiskit versions."""
    return [circuit.find_bit(bit).index for bit in bits]


def print_circuit_data(circuit):
    print("\nCircuit data operations:")
    for index, instruction in enumerate(circuit.data):
        op = instruction.operation
        qargs = bit_indices(circuit, instruction.qubits)
        cargs = bit_indices(circuit, instruction.clbits)
        print(f"{index:02d}: {op.name:8s} qargs={qargs} cargs={cargs}")


def print_dag_ops(circuit, dag):
    print("\nCompiler DAG operation nodes:")
    for index, node in enumerate(dag.topological_op_nodes()):
        qargs = bit_indices(circuit, node.qargs)
        cargs = bit_indices(circuit, node.cargs)
        print(f"{index:02d}: {node.op.name:8s} qargs={qargs} cargs={cargs}")


def has_measure_immediately_before_reset(circuit):
    names = [instruction.operation.name for instruction in circuit.data]
    return any(
        previous_name == "measure" and current_name == "reset"
        for previous_name, current_name in zip(names, names[1:])
    )


def main():
    # This input has no measurements, so any measure in the compiled circuit
    # would have been inserted by compile_to_dynamic_circuit().
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.x(1)

    random.seed(0)
    compiler = GidNET(circuit)
    dynamic_circuit = compiler.compile_to_dynamic_circuit(iterations=1)

    print("Reuse sequences:", compiler.qubit_reuse_sequences)
    print("Dynamic circuit qubits:", dynamic_circuit.num_qubits)
    print("dynamic_circuit.count_ops():", dynamic_circuit.count_ops())

    print_circuit_data(dynamic_circuit)
    print_dag_ops(dynamic_circuit, compiler.dynamic_circuit_dag)

    op_names = [instruction.operation.name for instruction in dynamic_circuit.data]
    has_measure = "measure" in op_names
    has_reset = "reset" in op_names
    measure_before_reset = has_measure_immediately_before_reset(dynamic_circuit)

    print("\nInspection verdict:")
    print(f"Contains measure instructions: {has_measure}")
    print(f"Contains reset instructions:   {has_reset}")
    print(f"Measure immediately before reset: {measure_before_reset}")

    if not has_reset:
        print("Conclusion: no reset gates were emitted for this example.")
    elif measure_before_reset:
        print("Conclusion: at least one measure instruction appears before a reset.")
    else:
        print("Conclusion: reset gates appear without explicit preceding measures.")


if __name__ == "__main__":
    main()
