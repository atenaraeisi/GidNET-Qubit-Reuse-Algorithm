from gidnet.qubitreuse import GidNET
from qiskit import QuantumCircuit
import time

path = r"C:\Users\ASUS\Documents\Quantum computing\Qiskit\CaQR-main\CaQR-main\benchmarks\cc_n10.qasm"
circ = QuantumCircuit.from_qasm_file(path)

for it in [5, 10, 20, 50]:
    widths = []
    t0 = time.time()
    for run in range(5):  # هر تعداد iteration را ۵ بار تکرار کن
        g = GidNET(circ)
        g.compile_to_dynamic_circuit(iterations=it, draw=False)
        widths.append(g.dynamic_circuit_width)
    elapsed = time.time() - t0
    print(f"iterations={it}: widths={widths}, avg_time={elapsed/5:.3f}s")