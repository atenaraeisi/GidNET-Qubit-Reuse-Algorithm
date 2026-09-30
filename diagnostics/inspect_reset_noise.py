"""Diagnostic: how backend-derived Aer NoiseModel treats reset/measure/X."""
import warnings
import qiskit
import qiskit_aer
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

print("Qiskit version:", qiskit.__version__)
print("qiskit-aer version:", qiskit_aer.__version__)

# ============ PART 1 ============
print("\n============================================================")
print("PART 1: basic model inspection")
print("============================================================")
try:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from qiskit.providers.fake_provider import FakeKolkata
        backend = FakeKolkata()
    for w in caught:
        print("warning:", str(w.message)[:300])
except Exception as e:
    print("FakeKolkata import failed:", repr(e))
    raise

try:
    bname = backend.name() if callable(getattr(backend, "name", None)) else backend.name
except Exception as e:
    bname = f"<name lookup failed: {e}>"
print("backend name:", bname)

noise_model = NoiseModel.from_backend(backend)
print("noise_instructions:", noise_model.noise_instructions)
print("basis_gates:", noise_model.basis_gates)
print("is_ideal():", noise_model.is_ideal())
print("'reset' in noise_instructions:", 'reset' in noise_model.noise_instructions)
print("'measure' in noise_instructions:", 'measure' in noise_model.noise_instructions)
print("'x' in noise_instructions:", 'x' in noise_model.noise_instructions)

# ============ PART 2 ============
print("\n============================================================")
print("PART 2: NoiseModel contents")
print("============================================================")
s = str(noise_model)
print("str(noise_model) [truncated to 4000 chars]:")
print(s[:4000])
d = noise_model.to_dict()
print("\nto_dict top-level keys:", list(d.keys()))
errs = d.get("errors", [])
print("num error entries:", len(errs))
from collections import Counter
cnt = Counter()
for e in errs:
    ops = e.get("operations", e.get("operation", "?"))
    typ = e.get("type", "?")
    cnt[(str(typ), str(ops))] += 1
print("error entries by (type, operations):")
for k, v in sorted(cnt.items()):
    print(" ", k, "count:", v)
# detailed per-op presence
for target in ["reset", "measure", "x"]:
    hits = [e for e in errs if target in str(e.get("operations", e.get("operation", "")))]
    print(f"\nentries mentioning '{target}': {len(hits)}")
    for e in hits[:3]:
        print("  type:", e.get("type"), "operations:", e.get("operations", e.get("operation")),
              "qubits:", str(e.get("qubits", e.get("qubit", "?")))[:200],
              "instructions-keys:", [k for k in e.keys()][:10])
# private structures (diagnostic read-only)
for attr in ["_local_quantum_errors", "_nonlocal_quantum_errors", "_default_quantum_errors",
             "_readout_errors", "_local_readout_errors", "_default_readout_error"]:
    if hasattr(noise_model, attr):
        try:
            v = getattr(noise_model, attr)
            print(f"{attr}: type={type(v).__name__} len/cnt={len(v) if hasattr(v,'__len__') else v}")
            if isinstance(v, dict) and len(v) < 20:
                print("  keys:", list(v.keys())[:20])
        except Exception as e:
            print(attr, "inspect failed:", e)
# explicit checks
has_q_reset = any(e.get("type") == "qerror" and "reset" in str(e.get("operations", e.get("operation",""))) for e in errs)
has_q_measure = any(e.get("type") == "qerror" and "measure" in str(e.get("operations", e.get("operation",""))) for e in errs)
has_ro_measure = any(e.get("type") == "roerror" for e in errs)
has_q_x = any(e.get("type") == "qerror" and "x" in str(e.get("operations", e.get("operation",""))) for e in errs)
print("\nANY explicit QuantumError on reset:", has_q_reset)
print("ANY explicit QuantumError on measure:", has_q_measure)
print("ANY ReadoutError present:", has_ro_measure)
print("ANY explicit QuantumError on x:", has_q_x)

# ============ PART 3 ============
print("\n============================================================")
print("PART 3: backend reset information")
print("============================================================")
try:
    conf = backend.configuration()
    print("basis_gates:", conf.basis_gates)
    print("reset supported (in basis_gates):", 'reset' in conf.basis_gates)
    print("dt:", getattr(conf, "dt", "MISSING"), "dtm:", getattr(conf, "dtm", "MISSING"))
except Exception as e:
    print("configuration inspect failed:", repr(e))
try:
    props = backend.properties()
    print("properties date:", props.last_update_date if hasattr(props, "last_update_date") else "MISSING")
    # gate entries
    gates = props.gates if hasattr(props, "gates") else []
    from collections import Counter as C2
    gc = C2(g.name for g in gates)
    print("gate name counts:", dict(gc))
    for gname in ["reset", "measure", "x", "sx"]:
        mg = [g for g in gates if g.name == gname]
        print(f"gate '{gname}' entries: {len(mg)}")
        for g in mg[:2]:
            print("  qubits:", g.qubits, "params:", [(p.name, p.value) for p in g.parameters])
    # readout + durations for qubit 0
    try:
        print("qubit0 readout_error:", props.readout_error(0))
    except Exception as e:
        print("readout_error(0) MISSING:", e)
    try:
        print("qubit0 readout_length:", props.readout_length(0))
    except Exception as e:
        print("readout_length(0) MISSING:", e)
    try:
        print("qubit0 gate_error x:", props.gate_error("x", 0))
    except Exception as e:
        print("gate_error x MISSING:", e)
    try:
        print("qubit0 gate_length x:", props.gate_length("x", 0))
    except Exception as e:
        print("gate_length x MISSING:", e)
    try:
        print("reset error lookup:", props.gate_error("reset", 0))
    except Exception as e:
        print("reset error/calibration MISSING:", repr(e)[:300])
except Exception as e:
    print("properties inspect failed:", repr(e))
print("has target attr:", hasattr(backend, "target"))

# ============ PART 4 ============
print("\n============================================================")
print("PART 4: minimal noisy experiment (100k shots)")
print("============================================================")
SHOTS = 100000
SEED = 12345
# A: X reset measure
qa = QuantumCircuit(1, 1)
qa.x(0)
qa.reset(0)
qa.measure(0, 0)
# B: X measure-aux cond-X measure-final (mirrors lowering)
from qiskit.circuit import ClassicalRegister
qb = QuantumCircuit(1, 2)
qb.x(0)
qb.measure(0, 0)  # aux
qb.x(0).c_if(qb.clbits[0], 1)
qb.measure(0, 1)  # final
print("--- Circuit A (native) ---")
print(qa)
print("count_ops A:", dict(qa.count_ops()))
print("--- Circuit B (measure+cond-X) ---")
print(qb)
print("count_ops B:", dict(qb.count_ops()))
try:
    ta = transpile(qa, backend=backend)
    tb = transpile(qb, backend=backend)
    print("--- Transpiled A ---")
    print(ta)
    print("count_ops transpiled A:", dict(ta.count_ops()))
    print("reset survives transpilation A:", dict(ta.count_ops()).get("reset", 0) > 0)
    print("--- Transpiled B ---")
    print(tb)
    print("count_ops transpiled B:", dict(tb.count_ops()))
except Exception as e:
    import traceback
    traceback.print_exc()
    print("transpile failed:", repr(e))
    ta, tb = qa, qb
noisy = AerSimulator(noise_model=noise_model, seed_simulator=SEED)
try:
    ra = noisy.run(ta, shots=SHOTS).result().get_counts()
    rb = noisy.run(tb, shots=SHOTS).result().get_counts()
    print("noisy counts A:", ra)
    print("noisy counts B (aux + final):", rb)
    # P(final=1): A single bit; B final is clbit index 1
    def p1_single(counts):
        t = sum(counts.values())
        return sum(c for k, c in counts.items() if k.replace(" ", "")[-1] == "1") / t if " " not in list(counts)[0] else None
    # robust via hex
    resa = noisy.run(ta, shots=SHOTS, seed_simulator=SEED).result()
    resb = noisy.run(tb, shots=SHOTS, seed_simulator=SEED).result()
    ha = resa.data(0)["counts"]
    hb = resb.data(0)["counts"]
    def marg_final(hexd, idx):
        tot = float(sum(hexd.values()))
        ones = sum(c for hk, c in hexd.items() if (int(hk, 16) >> idx) & 1)
        return ones / tot
    # find final index: A -> 0; B -> 1 (clbits[1])
    pa1 = marg_final(ha, 0)
    pb1 = marg_final(hb, 1)
    print(f"P(final=1) native reset A: {pa1:.6f}")
    print(f"P(final=1) measure+cond-X B (final bit only): {pb1:.6f}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("noisy run failed:", repr(e))

# ============ PART 5 ============
print("\n============================================================")
print("PART 5: explicit custom reset-noise sanity check")
print("============================================================")
try:
    from qiskit_aer.noise.errors import pauli_error
    perr = pauli_error([("X", 0.5), ("I", 0.5)])
    custom = NoiseModel()
    custom.add_quantum_error(perr, "reset", [0])
    print("custom noise_instructions:", custom.noise_instructions)
    print("custom is_ideal:", custom.is_ideal())
    q0 = QuantumCircuit(1, 1)
    q0.x(0)
    q0.reset(0)
    q0.measure(0, 0)
    ideal = AerSimulator(seed_simulator=SEED)
    noisy_c = AerSimulator(noise_model=custom, seed_simulator=SEED)
    ci = ideal.run(q0, shots=20000).result().get_counts()
    cn = noisy_c.run(q0, shots=20000).result().get_counts()
    print("ideal counts:", ci)
    print("custom-reset-noise counts:", cn)
    ti = sum(cn.values())
    p1 = sum(c for k, c in cn.items() if k.replace(" ", "")[-1] == "1") / ti
    print(f"P(1) with explicit 50% X-on-reset: {p1:.4f} (expect ~0.5 if applied)")
    print("Aer applies explicit reset noise:", 0.3 < p1 < 0.7)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("custom reset-noise check failed:", repr(e))

print("\n============================================================")
print("END (95 benchmarks not run; no source modified)")
print("============================================================")
