#!/usr/bin/env python3
"""Generate GidNET reuse plans for new benchmarks, exactly the way Experiment 1 did.

Protocol copied from experiment1_physical_variation.ipynb:
  1. load the .qasm, reject classical operands/conditions, REMOVE IDLE QUBITS (prepare_gidnet_input),
  2. for seed in 0..19: random.seed(seed); np.random.seed(seed);
        g = GidNET(qc_compact); circ = g.compile_to_dynamic_circuit(iterations=20, draw=False)
     plan = canonical(g.qubit_reuse_sequences)   (order inside a chain kept, order of chains ignored)
  3. sanity: every logical qubit exactly once; #reuse boundaries == #resets in the dynamic circuit,
  4. keep the MINIMUM-width plans, one per distinct canonical plan (P0, P1, ...), with their frequency.

Index convention: chains are in COMPACT qubit indices.  To make every downstream script
(exp4c1_build_table / exp4d / exp5) agree, the compacted circuit is also written to
--compact-dir/<same file name>; use that folder as --qasm-dir downstream (there compaction is the
identity map, which exp4d_features.py asserts).

A circuit is only usable when width < #active qubits (some reuse found) and it has >= --min-plans
distinct minimum-width plans (otherwise there is little to repair); both facts are reported.

Run in the GidNET environment (Qiskit 0.45.3), from the GidNET project root:
  python make_plans.py --bench-dir benchmarks2 --circuits-csv scan_out/selected.csv \
      --compact-dir benchmarks_compact --out-dir plans_out --timeout 1800
Outputs: plans_out/plans_new.csv (circuit, plan_id, seed, width, logical_depth, reset_count, frequency,
reuse_chains), gidnet_runs.csv (every seed), circuit_report.csv (status per circuit).  Resumable.
"""
import argparse, json, os, random, subprocess, sys, time

import numpy as np
import pandas as pd


def canonical(chains):
    return tuple(sorted(tuple(int(q) for q in c) for c in chains))


def prepare_gidnet_input(qc):
    from qiskit import QuantumCircuit
    for item in qc.data:
        if len(item.clbits) > 0:
            raise ValueError(f'classical operands in {item.operation.name}')
        if getattr(item.operation, 'condition', None) is not None:
            raise ValueError(f'conditioned operation {item.operation.name}')
    active = sorted({qc.find_bit(q).index for item in qc.data for q in item.qubits})
    o2n = {o: n for n, o in enumerate(active)}
    compact = QuantumCircuit(len(active))
    for item in qc.data:
        compact.append(item.operation.copy(), [o2n[qc.find_bit(q).index] for q in item.qubits], [])
    return compact, dict(declared=qc.num_qubits, active=len(active), identity=active == list(range(len(active))))


def qasm_text(qc):
    try:
        return qc.qasm()                       # Qiskit 0.x
    except AttributeError:
        from qiskit import qasm2               # Qiskit >= 1.0
        return qasm2.dumps(qc)


def worker(a):
    """Runs in a subprocess so that a hung/slow GidNET call can be killed by the parent's timeout."""
    from qiskit import QuantumCircuit
    from gidnet.qubitreuse import GidNET
    qc = QuantumCircuit.from_qasm_file(os.path.join(a.bench_dir, a.circuit))
    compact, meta = prepare_gidnet_input(qc)
    os.makedirs(a.compact_dir, exist_ok=True)
    open(os.path.join(a.compact_dir, a.circuit), 'w').write(qasm_text(compact))
    runs = []
    for seed in range(a.n_seeds):
        random.seed(seed); np.random.seed(seed)
        t0 = time.time()
        g = GidNET(compact)
        circ = g.compile_to_dynamic_circuit(iterations=a.iterations, draw=False)
        seqs = g.qubit_reuse_sequences
        if not seqs:            # None or empty: GidNET found no reuse (seen on dense circuits: qft, adr4, misex1, ...)
            if int(circ.num_qubits) != compact.num_qubits or circ.count_ops().get('reset', 0):
                raise RuntimeError('qubit_reuse_sequences is None but the dynamic circuit shows reuse')
            seqs = [[q] for q in range(compact.num_qubits)]
        chains = [list(map(int, c)) for c in seqs]
        flat = [q for c in chains for q in c]
        sane = (sorted(flat) == list(range(compact.num_qubits))
                and sum(len(c) - 1 for c in chains) == circ.count_ops().get('reset', 0))
        runs.append(dict(seed=seed, width=int(circ.num_qubits), logical_depth=int(circ.depth()),
                         reset_count=int(circ.count_ops().get('reset', 0)), sane=bool(sane),
                         reuse_chains=json.dumps([list(c) for c in canonical(chains)]),
                         seconds=round(time.time() - t0, 2)))
    json.dump(dict(meta=meta, runs=runs), open(a.json_out, 'w'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench-dir', default='benchmarks2')
    ap.add_argument('--circuits-csv', default='scan_out/selected.csv')
    ap.add_argument('--circuits', default='', help='comma list (overrides csv)')
    ap.add_argument('--compact-dir', default='benchmarks_compact')
    ap.add_argument('--out-dir', default='plans_out')
    ap.add_argument('--n-seeds', type=int, default=20)
    ap.add_argument('--iterations', type=int, default=20)
    ap.add_argument('--timeout', type=int, default=1800, help='seconds per circuit (all seeds)')
    ap.add_argument('--min-plans', type=int, default=3)
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--circuit', default='')
    ap.add_argument('--json-out', dest='json_out', default='')
    a = ap.parse_args()
    if a.worker:
        return worker(a)

    os.makedirs(a.out_dir, exist_ok=True)
    names = a.circuits.split(',') if a.circuits else list(pd.read_csv(a.circuits_csv).circuit)
    rep_path = os.path.join(a.out_dir, 'circuit_report.csv')
    runs_path = os.path.join(a.out_dir, 'gidnet_runs.csv')
    done = set(pd.read_csv(rep_path).circuit) if os.path.exists(rep_path) else set()
    for name in names:
        if name in done:
            continue
        tmp = os.path.join(a.out_dir, f'_{name}.json')
        cmd = [sys.executable, os.path.abspath(__file__), '--worker', '--circuit', name, '--json-out', tmp,
               '--bench-dir', a.bench_dir, '--compact-dir', a.compact_dir,
               '--n-seeds', str(a.n_seeds), '--iterations', str(a.iterations)]
        t0 = time.time()
        rec = dict(circuit=name, status='ok', error='')
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout, cwd=os.getcwd())
            if r.returncode != 0 or not os.path.exists(tmp):
                rec.update(status='error', error=(r.stderr or r.stdout)[-300:].replace('\n', ' | '))
        except subprocess.TimeoutExpired:
            rec.update(status='timeout')
        rec['seconds'] = round(time.time() - t0, 1)
        if rec['status'] == 'ok':
            d = json.load(open(tmp)); os.remove(tmp)
            runs = pd.DataFrame(d['runs']); runs.insert(0, 'circuit', name)
            runs.to_csv(runs_path, mode='a', index=False, header=not os.path.exists(runs_path))
            m = d['meta']
            minw = int(runs.width.min())
            mw = runs[runs.width == minw]
            nplans = mw.reuse_chains.nunique()
            rec.update(declared=m['declared'], active=m['active'], compact_is_identity=m['identity'],
                       min_width=minw, widths_seen=json.dumps(runs.width.value_counts().sort_index().to_dict()),
                       n_min_width_plans=nplans, all_sane=bool(runs.sane.all()),
                       reuse_found=minw < m['active'],
                       usable=bool(minw < m['active'] and nplans >= a.min_plans and runs.sane.all()))
        prev = pd.read_csv(rep_path) if os.path.exists(rep_path) else pd.DataFrame()
        pd.concat([prev, pd.DataFrame([rec])], ignore_index=True).to_csv(rep_path, index=False)  # columns differ per row
        print(f"[{name}] {rec['status']} {rec['seconds']}s "
              + (f"active={rec.get('active')} min_width={rec.get('min_width')} plans={rec.get('n_min_width_plans')} "
                 f"usable={rec.get('usable')}" if rec['status'] == 'ok' else rec['error']), flush=True)

    rep = pd.read_csv(rep_path)
    runs = pd.read_csv(runs_path) if os.path.exists(runs_path) else pd.DataFrame()
    rows = []
    for name in (rep[rep.usable == True].circuit if 'usable' in rep else []):
        r = runs[runs.circuit == name]
        mw = r[r.width == r.width.min()]
        freq = mw.groupby('reuse_chains').size()
        uniq = mw.sort_values('seed').drop_duplicates('reuse_chains').reset_index(drop=True)
        for i, row in uniq.iterrows():
            rows.append(dict(circuit=name, plan_id=f'P{i}', seed=row.seed, width=row.width,
                             logical_depth=row.logical_depth, reset_count=row.reset_count,
                             frequency=int(freq[row.reuse_chains]), reuse_chains=row.reuse_chains))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out_dir, 'plans_new.csv'), index=False)
    print(f'\n{len(rep)} circuits processed; usable: {int(rep.usable.sum())}; plans written: {len(out)}')
    print(rep.reindex(columns=['circuit', 'status', 'active', 'min_width', 'n_min_width_plans', 'usable', 'seconds']).to_string(index=False))


if __name__ == '__main__':
    main()
