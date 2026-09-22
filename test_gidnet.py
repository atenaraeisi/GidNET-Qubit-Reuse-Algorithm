from gidnet.qubitreuse import GidNET
from qiskit import QuantumCircuit

circ = QuantumCircuit(5)
circ.cx(1, 2)
circ.cx(0, 3)
circ.cx(1, 4)
circ.cx(2, 4)
circ.cx(3, 4)

gidnet = GidNET(circ)
dynamic_circ = gidnet.compile_to_dynamic_circuit(iterations=20, draw=False)
print("Dynamic Circuit Width:", gidnet.dynamic_circuit_width)
print(dynamic_circ)