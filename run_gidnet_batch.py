from gidnet.qubitreuse import GidNET
from qiskit import QuantumCircuit
import csv, os, time

BENCH_DIR = r"C:\Users\ASUS\Documents\Quantum computing\Qiskit\CaQR-main\CaQR-main\benchmarks"

# یک زیرمجموعه‌ی کوچک برای شروع — بعداً می‌تونی گسترشش بدی
circuits = [
    "bv_n10.qasm", "cc_n10.qasm", "alu-v0_27.qasm", "alu-v2_30.qasm",
    "qft_10.qasm", "ising_model_10.qasm", "test3.qasm"
]

ITERATIONS = 20
results = []

for fname in circuits:
    path = os.path.join(BENCH_DIR, fname)
    if not os.path.exists(path):
        print(f"SKIP (not found): {fname}")
        continue
    try:
        circ = QuantumCircuit.from_qasm_file(path)
        t0 = time.time()
        g = GidNET(circ)
        g.compile_to_dynamic_circuit(iterations=ITERATIONS, draw=False)
        elapsed = time.time() - t0
        results.append({
            "circuit": fname,
            "orig_qubits": circ.num_qubits,
            "orig_depth": circ.depth(),
            "gidnet_width": g.dynamic_circuit_width,
            "runtime_s": round(elapsed, 4),
        })
        print(f"OK: {fname} -> width={g.dynamic_circuit_width}, time={elapsed:.3f}s")
    except Exception as e:
        print(f"ERROR on {fname}: {e}")

with open("gidnet_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nSaved to gidnet_results.csv")