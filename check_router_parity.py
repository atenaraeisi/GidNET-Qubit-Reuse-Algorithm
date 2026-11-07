#!/usr/bin/env python3
"""Parity gate: does router_topo:route_hh7 reproduce the cost_table you already have?

Re-routes a random sample of (circuit, plan, seed) from an existing 4C-1 directory with the hh7 router of
this experiment (through the same route_task as the table builder) and compares swap_count / routed_depth
with the stored values.  Pass = 100% identical.  If not, fix router_topo._route to match your
router_template.py before running anything else (same Qiskit version, same transpile arguments).

  python check_router_parity.py --c1-dir exp4c1_all --qasm-dir benchmarks_compact --n 12
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import exp4c1_build_table as b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--c1-dir', default='exp4c1_all')
    ap.add_argument('--qasm-dir', default='benchmarks_compact')
    ap.add_argument('--router', default='router_topo:route_hh7')
    ap.add_argument('--n', type=int, default=12, help='number of plans to re-route')
    ap.add_argument('--seeds', default='42,43,100')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    cat = pd.read_csv(os.path.join(a.c1_dir, 'catalog.csv'))
    cost = pd.read_csv(os.path.join(a.c1_dir, 'cost_table.csv'))
    cost = cost[cost.status == 'ok']
    seeds = [int(s) for s in a.seeds.split(',')]
    rng = np.random.default_rng(a.seed)
    # spread over circuits: round-robin
    by_c = {c: g for c, g in cat.groupby('circuit')}
    picks, circs = [], sorted(by_c)
    i = 0
    while len(picks) < min(a.n, len(cat)):
        g = by_c[circs[i % len(circs)]]
        r = g.iloc[int(rng.integers(len(g)))]
        if (r.circuit, r.sig) not in {(p.circuit, p.sig) for p in picks}:
            picks.append(r)
        i += 1
    ref = cost.set_index(['circuit', 'sig', 'seed'])
    n_cmp = n_ok = 0
    bad = []
    for r in picks:
        plan = b.canonical(b.parse_chains(r.plan_json))
        rows, _ = b.route_task(r.circuit, r.sig, plan, seeds, a.qasm_dir, a.router)
        for row in rows:
            key = (row['circuit'], row['sig'], row['seed'])
            if key not in ref.index or row['status'] != 'ok':
                bad.append((key, row['status'], 'missing')); continue
            n_cmp += 1
            ok = (row['swap_count'] == ref.loc[key, 'swap_count'] and row['routed_depth'] == ref.loc[key, 'routed_depth'])
            n_ok += ok
            if not ok:
                bad.append((key, (row['swap_count'], row['routed_depth']),
                            (ref.loc[key, 'swap_count'], ref.loc[key, 'routed_depth'])))
    print(f'parity: {n_ok}/{n_cmp} identical')
    for x in bad[:10]:
        print('  MISMATCH', x)
    print('PARITY OK' if n_cmp and n_ok == n_cmp and not bad else 'PARITY FAILED - fix router_topo before continuing')
    sys.exit(0 if (n_cmp and n_ok == n_cmp and not bad) else 1)


if __name__ == '__main__':
    main()
