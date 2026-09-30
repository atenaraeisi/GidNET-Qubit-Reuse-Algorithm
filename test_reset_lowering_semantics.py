"""Semantic equivalence: native reset vs lowered measure+conditional-X (ideal Aer)."""
import qiskit
import qiskit_aer
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from gidnet.qubitreuse import GidNET
from gidnet.reset_lowering import lower_reset_to_measure_x

print("Qiskit version:", qiskit.__version__)
print("qiskit-aer version:", qiskit_aer.__version__)

SIM = AerSimulator(seed_simulator=42)


def print_info(label, circ):
    print(f"\n--- {label} ---")
    print(circ)
    print("num_qubits:", circ.num_qubits, "num_clbits:", circ.num_clbits)
    print("count_ops:", dict(circ.count_ops()))
    print("cregs:", [(r.name, r.size) for r in circ.cregs])


def run_hex(circ, shots):
    res = SIM.run(circ, shots=shots).result()
    space_counts = res.get_counts(circ)
    hex_counts = res.data(0)["counts"]
    return space_counts, hex_counts


def marginal_probs(circ, hex_counts, target_idx):
    total = float(sum(hex_counts.values()))
    acc = {}
    for hk, cnt in hex_counts.items():
        v = int(hk, 16)
        key = "".join(str((v >> i) & 1) for i in reversed(target_idx))
        acc[key] = acc.get(key, 0) + cnt
    return {k: c / total for k, c in acc.items()}


def max_abs_diff(pa, pb):
    keys = set(pa) | set(pb)
    return max(abs(pa.get(k, 0.0) - pb.get(k, 0.0)) for k in keys)


def show_counts(label, space_counts, probs):
    print(f"{label} space-counts:", space_counts)
    print(f"{label} probs (marginal):", {k: round(v, 5) for k, v in sorted(probs.items())})


results = {}

# ================= TEST A =================
print("\n============================================================")
print("TEST A: X -> reset -> measure (deterministic |1> reset)")
print("============================================================")
try:
    na = QuantumCircuit(1, 1)
    na.x(0)
    na.reset(0)
    na.measure(0, 0)
    la = lower_reset_to_measure_x(na)
    print_info("Native A", na)
    print_info("Lowered A", la)
    sa, ha = run_hex(na, 10000)
    sl, hl = run_hex(la, 10000)
    # final logical bit is original c[0] -> index 0 in both circuits
    pa = marginal_probs(na, ha, [0])
    pl = marginal_probs(la, hl, [0])
    show_counts("Native A", sa, pa)
    show_counts("Lowered A", sl, pl)
    d = max_abs_diff(pa, pl)
    print("max abs prob diff (final bit):", round(d, 6))
    ok = pa.get("0", 0) >= 0.99 and pl.get("0", 0) >= 0.99 and d < 0.02
    print("PASS" if ok else "FAIL", ": TEST A")
    results["A"] = ("PASS" if ok else "FAIL", d)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Aer error TEST A:", repr(e))
    results["A"] = ("FAIL (sim error)", None)

# ================= TEST B =================
print("\n============================================================")
print("TEST B: H -> reset -> measure")
print("============================================================")
try:
    nb = QuantumCircuit(1, 1)
    nb.h(0)
    nb.reset(0)
    nb.measure(0, 0)
    lb = lower_reset_to_measure_x(nb)
    print_info("Native B", nb)
    print_info("Lowered B", lb)
    sa, ha = run_hex(nb, 10000)
    sl, hl = run_hex(lb, 10000)
    pa = marginal_probs(nb, ha, [0])
    pl = marginal_probs(lb, hl, [0])
    show_counts("Native B", sa, pa)
    show_counts("Lowered B", sl, pl)
    d = max_abs_diff(pa, pl)
    print("max abs prob diff (final bit):", round(d, 6))
    ok = pa.get("0", 0) >= 0.99 and pl.get("0", 0) >= 0.99 and d < 0.02
    print("PASS" if ok else "FAIL", ": TEST B")
    results["B"] = ("PASS" if ok else "FAIL", d)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Aer error TEST B:", repr(e))
    results["B"] = ("FAIL (sim error)", None)

# ================= TEST C =================
print("\n============================================================")
print("TEST C: X -> reset -> H -> measure (reuse qubit)")
print("============================================================")
try:
    nc = QuantumCircuit(1, 1)
    nc.x(0)
    nc.reset(0)
    nc.h(0)
    nc.measure(0, 0)
    lc = lower_reset_to_measure_x(nc)
    print_info("Native C", nc)
    print_info("Lowered C", lc)
    sa, ha = run_hex(nc, 10000)
    sl, hl = run_hex(lc, 10000)
    pa = marginal_probs(nc, ha, [0])
    pl = marginal_probs(lc, hl, [0])
    show_counts("Native C", sa, pa)
    show_counts("Lowered C", sl, pl)
    d = max_abs_diff(pa, pl)
    print("max abs prob diff (final bit):", round(d, 6))
    ok = d < 0.03 and abs(pa.get("0", 0) - 0.5) < 0.05 and abs(pl.get("0", 0) - 0.5) < 0.05
    print("PASS" if ok else "FAIL", ": TEST C")
    results["C"] = ("PASS" if ok else "FAIL", d)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Aer error TEST C:", repr(e))
    results["C"] = ("FAIL (sim error)", None)

# ================= TEST D =================
print("\n============================================================")
print("TEST D: H -> measure c0 -> reset -> H -> measure c1")
print("============================================================")
try:
    nd = QuantumCircuit(1, 2)
    nd.h(0)
    nd.measure(0, 0)
    nd.reset(0)
    nd.h(0)
    nd.measure(0, 1)
    ld = lower_reset_to_measure_x(nd)
    print_info("Native D", nd)
    print_info("Lowered D", ld)
    nm = nd.count_ops().get("measure", 0)
    lm = ld.count_ops().get("measure", 0)
    print(f"structural: native measures={nm}, lowered measures={lm}")
    struct_ok = (ld.count_ops().get("reset", 0) == 0) and (lm == nm)
    print("structural reuse (no duplicate measurement):", struct_ok)
    sa, ha = run_hex(nd, 20000)
    sl, hl = run_hex(ld, 20000)
    # compare ONLY final bit c1 -> clbit index 1 in both (no aux expected)
    print("lowered num_clbits:", ld.num_clbits, "(expect 2 if reused)")
    pa = marginal_probs(nd, ha, [1])
    pl = marginal_probs(ld, hl, [1])
    show_counts("Native D (c1 only)", sa, pa)
    show_counts("Lowered D (c1 only)", sl, pl)
    d = max_abs_diff(pa, pl)
    print("max abs prob diff (final bit c1):", round(d, 6))
    ok = struct_ok and d < 0.03
    print("PASS" if ok else "FAIL", ": TEST D")
    results["D"] = ("PASS" if ok else "FAIL", d)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Aer error TEST D:", repr(e))
    results["D"] = ("FAIL (sim error)", None)

# ================= TEST E =================
print("\n============================================================")
print("TEST E: GidNET integration (ideal Aer)")
print("============================================================")
try:
    qc5 = QuantumCircuit(4, 4)
    qc5.x(0)
    qc5.h(1)
    qc5.z(2)
    qc5.s(3)
    qc5.measure(0, 0)
    qc5.measure(1, 1)
    qc5.measure(2, 2)
    qc5.measure(3, 3)
    print_info("Original static E", qc5)
    gidnet = GidNET(qc5)
    native_e = gidnet.compile_to_dynamic_circuit(iterations=20, draw=False)
    lowered_e = lower_reset_to_measure_x(native_e)
    print_info("Native GidNET E", native_e)
    print_info("Lowered GidNET E", lowered_e)
    sa, ha = run_hex(native_e, 20000)
    sl, hl = run_hex(lowered_e, 20000)
    if lowered_e.num_clbits == native_e.num_clbits:
        print("bit ordering: same num_clbits, comparing FULL count strings.")
        tgt_n = list(range(native_e.num_clbits))
        tgt_l = list(range(lowered_e.num_clbits))
        why = "full-string meaningful (no aux bits added; lowering reused measurements)"
    else:
        print("bit ordering: lowered has extra aux bits; comparing only original clbits.")
        tgt_n = list(range(native_e.num_clbits))
        tgt_l = list(range(native_e.num_clbits))  # originals are first in lowered
        why = "aux bits marginalized out; originals are lowered.clbits[0:N]"
    print(why)
    pa = marginal_probs(native_e, ha, tgt_n)
    pl = marginal_probs(lowered_e, hl, tgt_l)
    # print top outcomes only to keep log readable
    print("Native E top counts:", dict(sorted(sa.items(), key=lambda x: -x[1])[:8]))
    print("Lowered E top counts:", dict(sorted(sl.items(), key=lambda x: -x[1])[:8]))
    print("Native E num keys:", len(pa), "Lowered E num keys:", len(pl))
    d = max_abs_diff(pa, pl)
    print("max abs prob diff:", round(d, 6))
    ok = d < 0.05
    print("PASS" if ok else "FAIL", ": TEST E")
    results["E"] = ("PASS" if ok else "FAIL", d)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Aer error TEST E:", repr(e))
    results["E"] = ("FAIL (sim error)", None)

print("\n============================================================")
print("SUMMARY")
print("============================================================")
print("Qiskit version:", qiskit.__version__)
print("qiskit-aer version:", qiskit_aer.__version__)
for k in ["A", "B", "C", "D", "E"]:
    print(k, results.get(k))
print("Note: ideal simulator only; no noise; 95 benchmarks not run.")
