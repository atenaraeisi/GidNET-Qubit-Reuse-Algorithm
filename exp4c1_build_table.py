#!/usr/bin/env python3
"""Experiment 4C-1, step 1: build the full cost table.

For every source plan (circuit, plan_id) this script
  1. regenerates the valid one-step neighbours with reuse_adversarial.py
     (operators suffix_swap, chain_transfer, node_swap_reconnect,
     segment_reverse; validity = reference_static_feasible, independent of GidNET),
  2. deduplicates them into a catalog of distinct plans (+ the originals),
  3. compiles every distinct plan with the Experiment-3 parity adapter and routes
     it once per seed with YOUR router function,
  4. writes cost_table.csv, catalog.csv, pools.csv, features.csv, manifest.json.

Run from the project root, in the Qiskit 0.45.3 environment used for Experiment 3,
with reuse_adversarial.py, experiment3_pipeline_adapter.py and your router module
on the Python path.  The run is resumable: finished (circuit, sig) tasks are skipped.

Example:
  python exp4c1_build_table.py --plans experiment2_plan_features.csv \
      --qasm-dir . --out-dir exp4c1_out --router router_template:route --workers 4
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from reuse_adversarial import (  # noqa: E402
    canonical, pair_boundaries, reference_static_feasible, unique_mutations,
)

OPS_ALL = ['suffix_swap', 'chain_transfer', 'node_swap_reconnect', 'segment_reverse']
POOL_A = {'suffix_swap'}
POOL_B = {'suffix_swap', 'chain_transfer'}
POOL_C = set(OPS_ALL)


def sig_of(plan) -> str:
    return hashlib.sha1(json.dumps([list(c) for c in plan]).encode()).hexdigest()[:24]


def parse_chains(x):
    v = x
    if isinstance(x, str):
        try:
            v = json.loads(x)
        except Exception:
            v = ast.literal_eval(x)
    return tuple(tuple(int(q) for q in c) for c in v)


def parse_seeds(s: str):
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


@lru_cache(maxsize=8)
def load_qc(path: str):
    from qiskit import QuantumCircuit
    return QuantumCircuit.from_qasm_file(path)


# ----------------------------------------------------------------- features
def interaction_features(qc, plan):
    """Cheap pre-routing features: load on the quotient interaction graph."""
    chain_of = {q: i for i, c in enumerate(plan) for q in c}
    wdeg = np.zeros(len(plan))
    for inst in qc.data:
        if inst.operation.name in ('barrier', 'measure'):
            continue
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if len(qs) != 2:
            continue
        a, b = chain_of.get(qs[0]), chain_of.get(qs[1])
        if a is None or b is None or a == b:
            continue
        wdeg[a] += 1
        wdeg[b] += 1
    mean = wdeg.mean() if len(wdeg) else 0.0
    return {
        'max_wdeg': float(wdeg.max()) if len(wdeg) else 0.0,
        'wdeg_cv': float(wdeg.std() / mean) if mean > 0 else 0.0,
        'n_chains': len(plan),
        'n_boundaries': len(pair_boundaries(plan)),
    }


# ----------------------------------------------------------------- catalog
def build_catalog(plans: pd.DataFrame, qasm_dir: str, args):
    catalog = {}   # (circuit, sig) -> dict
    pool_rows = []
    for _, row in plans.iterrows():
        circuit, plan_id = str(row['circuit']), str(row['plan_id'])
        qc = load_qc(os.path.join(qasm_dir, circuit))
        orig = canonical(parse_chains(row[args.chains_col]))
        width = len(orig)
        if 'width' in row and not pd.isna(row['width']) and int(row['width']) != width:
            print(f'[warn] {circuit}/{plan_id}: width column {row["width"]} != chains {width}')
        if not reference_static_feasible(qc, orig, width):
            print(f'[warn] {circuit}/{plan_id}: original plan fails static feasibility; skipped')
            continue
        osig = sig_of(orig)
        catalog.setdefault((circuit, osig), {'plan': orig, 'ops': set(), 'is_original': True})
        catalog[(circuit, osig)]['is_original'] = True
        found = {}   # sig -> set(ops)
        for op in OPS_ALL:
            for cand in unique_mutations(orig, op):
                if reference_static_feasible(qc, cand, width):
                    cc = canonical(cand)
                    s = sig_of(cc)
                    found.setdefault(s, (cc, set()))[1].add(op)
        pool_rows.append(dict(circuit=circuit, source_plan_id=plan_id, sig=osig,
                              is_original=True, in_A=True, in_B=True, in_C=True))
        for s, (cc, ops) in found.items():
            ent = catalog.setdefault((circuit, s), {'plan': cc, 'ops': set(), 'is_original': False})
            ent['ops'] |= ops
            pool_rows.append(dict(circuit=circuit, source_plan_id=plan_id, sig=s, is_original=False,
                                  in_A=bool(ops & POOL_A), in_B=bool(ops & POOL_B), in_C=True))
        print(f'[catalog] {circuit}/{plan_id}: {len(found)} valid neighbours')
    return catalog, pd.DataFrame(pool_rows)


# ----------------------------------------------------------------- routing task
def route_task(circuit, sig, plan, seeds, qasm_dir, router_spec):
    """Compile one plan and route it for every seed.  Runs in a worker process."""
    from experiment3_pipeline_adapter import build_dynamic_circuit, normalize_for_routing
    mod, fn = router_spec.split(':')
    route = getattr(importlib.import_module(mod), fn)
    qc = load_qc(os.path.join(qasm_dir, circuit))
    rows, feat = [], {'circuit': circuit, 'sig': sig, **interaction_features(qc, plan)}
    try:
        dyn = build_dynamic_circuit(qc, plan)
        norm = normalize_for_routing(dyn)
        feat['logical_depth'] = int(dyn.depth())
        feat['norm_depth'] = int(norm.depth())
        feat['dyn_width'] = int(dyn.num_qubits)
    except Exception as e:  # compile failure is recorded, never silently dropped
        err = f'compile_error: {type(e).__name__}: {e}'
        return [dict(circuit=circuit, sig=sig, seed=s, swap_count=np.nan, routed_depth=np.nan,
                     routing_time=np.nan, status=err) for s in seeds], feat
    for s in seeds:
        t0 = time.perf_counter()
        try:
            res = route(norm, int(s))
            if isinstance(res, dict):
                swap, depth = res['swap_count'], res['routed_depth']
            else:
                swap, depth = res
            rows.append(dict(circuit=circuit, sig=sig, seed=s, swap_count=float(swap),
                             routed_depth=float(depth), routing_time=time.perf_counter() - t0,
                             status='ok'))
        except Exception as e:
            rows.append(dict(circuit=circuit, sig=sig, seed=s, swap_count=np.nan, routed_depth=np.nan,
                             routing_time=time.perf_counter() - t0,
                             status=f'route_error: {type(e).__name__}: {e}'))
    return rows, feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plans', required=True, help='CSV with circuit, plan_id, width, reuse_chains')
    ap.add_argument('--chains-col', default='reuse_chains')
    ap.add_argument('--qasm-dir', default='.')
    ap.add_argument('--out-dir', default='exp4c1_out')
    ap.add_argument('--router', required=True, help='module:function, f(routing_input_circuit, seed)')
    ap.add_argument('--discovery-seeds', default='42-46')
    ap.add_argument('--heldout-seeds', default='100-129')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--circuits', default='', help='comma list to restrict (smoke test)')
    ap.add_argument('--max-plans-per-circuit', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = parse_seeds(args.discovery_seeds) + parse_seeds(args.heldout_seeds)
    plans = pd.read_csv(args.plans)
    if args.circuits:
        plans = plans[plans['circuit'].isin(args.circuits.split(','))]
    if args.max_plans_per_circuit:
        plans = plans.groupby('circuit').head(args.max_plans_per_circuit)

    cat_path = os.path.join(args.out_dir, 'catalog.csv')
    pools_path = os.path.join(args.out_dir, 'pools.csv')
    if os.path.exists(cat_path) and os.path.exists(pools_path):
        print('[resume] reusing existing catalog.csv / pools.csv')
        cdf = pd.read_csv(cat_path)
        catalog = {(r.circuit, r.sig): {'plan': canonical(parse_chains(r.plan_json))} for r in cdf.itertuples()}
    else:
        catalog, pools = build_catalog(plans, args.qasm_dir, args)
        pd.DataFrame([dict(circuit=c, sig=s, plan_json=json.dumps([list(x) for x in v['plan']]),
                           is_original=v['is_original'], produced_by='|'.join(sorted(v['ops'])))
                      for (c, s), v in catalog.items()]).to_csv(cat_path, index=False)
        pools.to_csv(pools_path, index=False)
        print(f'[catalog] {len(catalog)} distinct plans; {len(pools)} pool memberships')

    cost_path = os.path.join(args.out_dir, 'cost_table.csv')
    feat_path = os.path.join(args.out_dir, 'features.csv')
    done = set()
    if os.path.exists(cost_path):
        old = pd.read_csv(cost_path, usecols=['circuit', 'sig', 'seed'])
        cnt = old.groupby(['circuit', 'sig']).seed.nunique()
        done = {k for k, v in cnt.items() if v >= len(seeds)}
    tasks = [(c, s, v['plan']) for (c, s), v in catalog.items() if (c, s) not in done]
    print(f'[route] {len(tasks)} plans x {len(seeds)} seeds to run ({len(done)} already done)')

    def sink(rows, feat):
        pd.DataFrame(rows).to_csv(cost_path, mode='a', index=False, header=not os.path.exists(cost_path))
        pd.DataFrame([feat]).to_csv(feat_path, mode='a', index=False, header=not os.path.exists(feat_path))

    t0 = time.time()
    if args.workers <= 1:
        for i, (c, s, p) in enumerate(tasks, 1):
            sink(*route_task(c, s, p, seeds, args.qasm_dir, args.router))
            print(f'[route] {i}/{len(tasks)}  {time.time() - t0:.0f}s')
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(route_task, c, s, p, seeds, args.qasm_dir, args.router) for c, s, p in tasks]
            for i, f in enumerate(as_completed(futs), 1):
                sink(*f.result())
                print(f'[route] {i}/{len(tasks)}  {time.time() - t0:.0f}s')

    import qiskit
    cost = pd.read_csv(cost_path)
    manifest = dict(qiskit=qiskit.__version__, seeds=seeds, router=args.router,
                    n_plans=len(catalog), n_rows=len(cost),
                    n_failed=int((cost.status != 'ok').sum()),
                    failures=cost[cost.status != 'ok'].status.value_counts().head(10).to_dict())
    json.dump(manifest, open(os.path.join(args.out_dir, 'manifest.json'), 'w'), indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
