from gidnet.qubitreuse import GidNET
from qiskit import QuantumCircuit

circ = QuantumCircuit(5)
circ.cx(1, 2); circ.cx(0, 3); circ.cx(1, 4); circ.cx(2, 4); circ.cx(3, 4)
circ.measure_all()

for it in [5, 10, 20]:
    g = GidNET(circ)
    g.compile_to_dynamic_circuit(iterations=it, draw=False)
    print(f"iterations={it} -> width={g.dynamic_circuit_width}")