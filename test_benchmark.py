from gidnet.qubitreuse import GidNET
from qiskit.qasm2 import load

path = r"C:\Users\ASUS\Documents\Quantum computing\Qiskit\CaQR-main\CaQR-main\benchmarks\bv_n10.qasm"

bv_circ = load(path)
print("Original circuit:")
print("  qubits:", bv_circ.num_qubits)
print("  depth :", bv_circ.depth())
print("  gates :", sum(bv_circ.count_ops().values()))

g = GidNET(bv_circ)
g.compile_to_dynamic_circuit(iterations=10, draw=False)

print("\nAfter GidNET reuse:")
print("  Dynamic width:", g.dynamic_circuit_width)
print("  Reduction    :", bv_circ.num_qubits - g.dynamic_circuit_width, "qubits saved")