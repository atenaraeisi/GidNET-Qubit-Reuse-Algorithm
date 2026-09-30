"""Run fixed GidNET on all 95 benchmarks, saving QASM outputs to output_v2/gidnet/."""
from pathlib import Path
import pandas as pd
import random
from qiskit import QuantumCircuit
from gidnet.qubitreuse import GidNET

# ---- Paths ----
CAQR_DIR   = Path(r"C:\Users\ASUS\Documents\Quantum computing\Qiskit\CaQR-main\CaQR-main")
BENCH_DIR  = CAQR_DIR / "benchmarks"
OUT_DIR    = CAQR_DIR / "output_v2" / "gidnet"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# extended_comparison.csv رو از چند جا ممکنه پیدا کنه
CSV_CANDIDATES = [
    Path(r"C:\Users\ASUS\Documents\Quantum computing\Qiskit\GidNET-Qubit-Reuse-Algorithm\extended_comparison.csv"),
    CAQR_DIR / "extended_comparison.csv",
    Path("extended_comparison.csv"),
]
CSV = next((p for p in CSV_CANDIDATES if p.exists()), None)
if CSV is None:
    raise FileNotFoundError("extended_comparison.csv not found in any candidate path")
print(f"Reading benchmark list from: {CSV}")

df = pd.read_csv(CSV)
benchmarks = df["Benchmark"].tolist()
print(f"Total benchmarks: {len(benchmarks)}")
print(f"Output folder: {OUT_DIR}\n")

# ---- Main loop ----
ok, fail, skip = 0, 0, 0
failures = []

RANDOM_SEED = 42
results = []

for i, fname in enumerate(benchmarks, 1):
    src = BENCH_DIR / fname
    if not src.exists():
        print(f"[{i:3d}/{len(benchmarks)}] SKIP  {fname} (missing in benchmarks/)")
        skip += 1
        continue

    try:
        qc = QuantumCircuit.from_qasm_file(str(src))
        random.seed(RANDOM_SEED)
        g = GidNET(qc)
        dyn = g.compile_to_dynamic_circuit(iterations=20, draw=False)

        out = OUT_DIR / f"{Path(fname).stem}_reuse.qasm"
        dyn.qasm(filename=str(out))

        ops = dyn.count_ops()

        results.append({
            "Benchmark": fname,
            "original_width": qc.num_qubits,
            "gidnet_width": dyn.num_qubits,
            "depth": dyn.depth(),
            "measure": ops.get("measure", 0),
            "reset": ops.get("reset", 0),
            "cx": ops.get("cx", 0),
        })

        print(f"[{i:3d}/{len(benchmarks)}] OK    {fname:38s} "
            f"width={g.dynamic_circuit_width:2d} "
            f"M={ops.get('measure', 0):3d} R={ops.get('reset', 0):3d} "
            f"depth={dyn.depth():4d}")
        ok += 1
    except Exception as e:
        print(f"[{i:3d}/{len(benchmarks)}] FAIL  {fname}: {type(e).__name__}: {e}")
        failures.append((fname, str(e)))
        fail += 1

# ---- Summary ----
print("\n" + "=" * 60)
print(f"OK:   {ok}")
print(f"FAIL: {fail}")
print(f"SKIP: {skip}")
if failures:
    print("\nFailures:")
    for fname, err in failures[:20]:
        print(f"  {fname}: {err}")
print("=" * 60)

results_df = pd.DataFrame(results)
results_csv = OUT_DIR / "gidnet_native_fixed_metrics.csv"
results_df.to_csv(results_csv, index=False)

print(f"Metrics saved to: {results_csv}")