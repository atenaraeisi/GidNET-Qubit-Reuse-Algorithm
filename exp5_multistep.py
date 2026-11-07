#!/usr/bin/env python3
"""Experiment 5 (ceiling): do 2- and 3-step Repair moves raise the 1-step ceiling?

Stages (run in order, all resumable):
  enumerate : BFS over valid plans up to --max-depth steps from every ORIGINAL plan
              (same 4 operators + same independent feasibility check as 4C-1).
              Writes exp5_out/catalog5.csv (distinct plans), reach.csv (source, sig, dist)
              and prints how many plans / how much routing it implies.
  route     : routes every distinct plan not already in exp4c1_out/cost_table.csv with
              the same router + the same 35 seeds (42-46 discovery, 100-129 held-out).
  analyze   : ceiling per step count d = 1,2,3 (see below).  No routing needed.

Ceiling definitions, per source plan s and neighbourhood R_d(s) = plans within <= d steps (incl. s):
  sel_disc   : pick argmin of the 5-seed DISCOVERY mean over R_d; move only if it beats s on
               discovery; gain = held-out(s) - held-out(pick) on 30 seeds the choice never saw.
               (honest ceiling: includes winner's curse correction)
  oracle_held: argmin of held-out mean  (optimistic upper bound, in-sample)
  nbr_mean   : mean held-out gain of a RANDOM plan in R_d \\ {s}  (no selection at all)
  eqK_disc   : like sel_disc but candidates sub-sampled to K = |R_1 \\ {s}| (same candidate count
               as the one-step pool) -> separates "multi-step reaches better regions" from
               "more candidates -> larger best-of-N".
  vs_best_orig: held-out(best existing original by discovery) - held-out(pick): >0 means the
               multi-step pick beats simply choosing another GidNET output in that circuit.

Example (real run, Qiskit 0.45.3 env, from the GidNET project root):
  python exp5_multistep.py enumerate --plans experiment2_plan_features.csv --qasm-dir <benchmarks> \
      --c1-dir exp4c1_out --out-dir exp5_out --max-depth 3
  python exp5_multistep.py route --c1-dir exp4c1_out --out-dir exp5_out --qasm-dir <benchmarks> \
      --router router_template:route --workers 4
  python exp5_multistep.py analyze --c1-dir exp4c1_out --out-dir exp5_out
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

DISC = list(range(42, 47))
HELD = list(range(100, 130))
METRICS = ['routed_depth', 'swap_count']


def _lazy():
    import exp4c1_build_table as b
    return b


# ------------------------------------------------------------------ enumerate
def enumerate_stage(a):
    b = _lazy()
    from reuse_adversarial import canonical, reference_static_feasible, unique_mutations
    os.makedirs(a.out_dir, exist_ok=True)
    plans = pd.read_csv(a.plans)
    if a.circuits:
        plans = plans[plans.circuit.isin(a.circuits.split(','))]
    cat1 = pd.read_csv(os.path.join(a.c1_dir, 'catalog.csv'))
    plan_of = {(r.circuit, r.sig): b.parse_chains(r.plan_json) for r in cat1.itertuples()}
    ops = b.OPS_ALL
    qcs, memo = {}, {}      # memo: (circuit, sig) -> list of valid neighbour sigs

    def neighbours(circuit, sig):
        k = (circuit, sig)
        if k in memo:
            return memo[k]
        if circuit not in qcs:
            qcs[circuit] = None if a.mock_qc else b.load_qc(os.path.join(a.qasm_dir, circuit))
        plan = plan_of[k]
        width = len(plan)
        out = set()
        for op in ops:
            for cand in unique_mutations(plan, op):
                cc = canonical(cand)
                s = b.sig_of(cc)
                if s in out:
                    continue
                if (circuit, s) in plan_of or reference_static_feasible(qcs[circuit], cc, width):
                    plan_of.setdefault((circuit, s), cc)
                    out.add(s)
        out.discard(sig)
        memo[k] = sorted(out)
        return memo[k]

    reach, t0, n = [], time.time(), 0
    for r in plans.itertuples():
        circuit = str(r.circuit)
        orig = canonical(b.parse_chains(getattr(r, a.chains_col)))
        osig = b.sig_of(orig)
        plan_of.setdefault((circuit, osig), orig)
        seen, frontier = {osig: 0}, [osig]
        for d in range(1, a.max_depth + 1):
            nxt = []
            for s in frontier:
                for t in neighbours(circuit, s):
                    if t not in seen:
                        seen[t] = d
                        nxt.append(t)
            if a.max_per_depth and len(nxt) > a.max_per_depth:      # optional cap -> sampled ceiling
                rng = np.random.default_rng(a.seed + len(reach))
                keep = set(rng.choice(sorted(nxt), a.max_per_depth, replace=False))
                for s in nxt:
                    if s not in keep:
                        seen.pop(s)
                nxt = sorted(keep)
            frontier = nxt
        for s, d in seen.items():
            reach.append(dict(circuit=circuit, source_plan_id=str(r.plan_id), source_sig=osig, sig=s, dist=d))
        n += 1
        cnt = {d: sum(1 for v in seen.values() if v == d) for d in range(a.max_depth + 1)}
        print(f'[enum] {n}/{len(plans)} {circuit}/{r.plan_id}: plans by distance {cnt}  {time.time() - t0:.0f}s', flush=True)

    reach = pd.DataFrame(reach)
    reach.to_csv(os.path.join(a.out_dir, 'reach.csv'), index=False)
    keys = sorted({(r.circuit, r.sig) for r in reach.itertuples()})
    orig_keys = {(r.circuit, r.source_sig) for r in reach.itertuples()}
    pd.DataFrame([dict(circuit=c, sig=s, plan_json=json.dumps([list(x) for x in plan_of[(c, s)]]),
                       is_original=(c, s) in orig_keys) for c, s in keys]
                 ).to_csv(os.path.join(a.out_dir, 'catalog5.csv'), index=False)
    # work estimate
    have = set()
    cp = os.path.join(a.c1_dir, 'cost_table.csv')
    t_per = np.nan
    if os.path.exists(cp):
        c1 = pd.read_csv(cp)
        okc = c1[c1.status == 'ok']
        have = {k for k, v in okc.groupby(['circuit', 'sig']).seed.nunique().items() if v >= len(DISC + HELD)}
        t_per = okc.routing_time.mean()
    new = [k for k in keys if k not in have]
    print(f'\n[enum] distinct plans in reach sets: {len(keys)}; already routed: {len(keys) - len(new)}; to route: {len(new)}')
    print(reach.groupby(['circuit', 'dist']).sig.nunique().unstack(fill_value=0).to_string())
    print(f'[enum] routing jobs = {len(new) * 35}; est. CPU time = {len(new) * 35 * t_per / 3600:.2f} h '
          f'(mean routing_time {t_per:.2f}s/job from 4C-1; divide by --workers)')


# ------------------------------------------------------------------ route
def route_stage(a):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    b = _lazy()
    cat5 = pd.read_csv(os.path.join(a.out_dir, 'catalog5.csv'))
    seeds = DISC + HELD
    have = set()
    for d in (a.c1_dir, a.out_dir):
        p = os.path.join(d, 'cost_table.csv')
        if os.path.exists(p):
            c = pd.read_csv(p, usecols=['circuit', 'sig', 'seed', 'status'])
            c = c[c.status == 'ok']
            have |= {k for k, v in c.groupby(['circuit', 'sig']).seed.nunique().items() if v >= len(seeds)}
    tasks = [(r.circuit, r.sig, b.parse_chains(r.plan_json)) for r in cat5.itertuples() if (r.circuit, r.sig) not in have]
    print(f'[route] {len(tasks)} plans x {len(seeds)} seeds')
    cost_path, feat_path = (os.path.join(a.out_dir, f) for f in ('cost_table.csv', 'features.csv'))

    def sink(rows, feat):
        pd.DataFrame(rows).to_csv(cost_path, mode='a', index=False, header=not os.path.exists(cost_path))
        pd.DataFrame([feat]).to_csv(feat_path, mode='a', index=False, header=not os.path.exists(feat_path))

    t0 = time.time()
    if a.workers <= 1:
        for i, (c, s, p) in enumerate(tasks, 1):
            sink(*b.route_task(c, s, p, seeds, a.qasm_dir, a.router))
            print(f'[route] {i}/{len(tasks)} {time.time() - t0:.0f}s', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(b.route_task, c, s, p, seeds, a.qasm_dir, a.router) for c, s, p in tasks]
            for i, f in enumerate(as_completed(futs), 1):
                sink(*f.result())
                print(f'[route] {i}/{len(tasks)} {time.time() - t0:.0f}s', flush=True)
    cost = pd.read_csv(cost_path)
    print('[route] failures:', int((cost.status != 'ok').sum()))


# ------------------------------------------------------------------ analyze
def load_means(a):
    frames = []
    for d in (a.c1_dir, a.out_dir):
        p = os.path.join(d, 'cost_table.csv')
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    c = pd.concat(frames)
    c = c[c.status == 'ok'].drop_duplicates(['circuit', 'sig', 'seed'])
    out = {}
    for m in METRICS:
        dm = c[c.seed.isin(DISC)].groupby(['circuit', 'sig'])[m].agg(['mean', 'count'])
        hm = c[c.seed.isin(HELD)].groupby(['circuit', 'sig'])[m].agg(['mean', 'count'])
        j = dm.join(hm, lsuffix='_d', rsuffix='_h', how='inner')
        j = j[(j.count_d == len(DISC)) & (j.count_h == len(HELD))]
        out[m] = {k: (v.mean_d, v.mean_h) for k, v in j.iterrows()}
    return out


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def analyze_stage(a):
    reach = pd.read_csv(os.path.join(a.out_dir, 'reach.csv'))
    means = load_means(a)
    rng = np.random.default_rng(a.seed)
    orig_by_circ = reach.groupby('circuit').source_sig.unique().to_dict()
    rows = []
    for m in METRICS:
        mm = means[m]
        for (circ, psid, ssig), g in reach.groupby(['circuit', 'source_plan_id', 'source_sig']):
            if (circ, ssig) not in mm:
                continue
            d_s, h_s = mm[(circ, ssig)]
            origs = [s for s in orig_by_circ[circ] if (circ, s) in mm]
            bo = min(origs, key=lambda s: mm[(circ, s)][0])
            h_bo = mm[(circ, bo)][1]
            k1 = int(((g.dist == 1) & g.sig.map(lambda s: (circ, s) in mm)).sum())
            for D in range(1, a.max_depth + 1):   # reach sets saturate: keep the source at larger D (same sources for every D)
                R = [s for s in g[(g.dist <= D) & (g.dist >= 1)].sig if (circ, s) in mm]
                if not R:
                    continue
                disc = np.array([mm[(circ, s)][0] for s in R])
                held = np.array([mm[(circ, s)][1] for s in R])
                i = int(np.argmin(disc))
                pick_h = held[i] if disc[i] < d_s else h_s            # accept rule
                if k1 and len(R) > k1:                                # equal-K control
                    gs = []
                    for _ in range(a.reps):
                        j = rng.choice(len(R), k1, replace=False)
                        t = j[np.argmin(disc[j])]
                        gs.append(h_s - (held[t] if disc[t] < d_s else h_s))
                    eqk = float(np.mean(gs))
                else:
                    eqk = h_s - pick_h
                rows.append(dict(metric=m, circuit=circ, source=psid, D=D, n_cand=len(R), K1=k1,
                                 base_held=h_s,
                                 sel_disc=h_s - pick_h, oracle_held=h_s - min(h_s, held.min()),
                                 nbr_mean=h_s - held.mean(), eqK_disc=eqk,
                                 vs_best_orig=h_bo - pick_h, orig_vs_best_orig=h_bo - h_s,
                                 picked_new=bool(disc[i] < d_s)))
    ps = pd.DataFrame(rows)
    ps.to_csv(os.path.join(a.out_dir, 'per_source.csv'), index=False)
    cols = ['sel_disc', 'oracle_held', 'nbr_mean', 'eqK_disc', 'vs_best_orig']
    summ = []
    for (m, D), g in ps.groupby(['metric', 'D']):
        pc = g.groupby('circuit')
        rec = dict(metric=m, D=D, n_sources=len(g), mean_cands=g.n_cand.mean(),
                   frac_moved=g.picked_new.mean(), base=pc.base_held.mean().mean())
        for c in cols:
            per = pc[c].mean()
            rec[c] = per.mean()
            rec[c + '_rel%'] = 100 * (pc.apply(lambda x: x[c].mean() / x.base_held.mean())).mean()
            lo, hi = boot_ci(g[c].values)      # sources treated as independent (they share neighbours)
            rec[c + '_ci_lo'], rec[c + '_ci_hi'] = lo, hi
        rec['circuits_positive'] = int((pc.sel_disc.mean() > 0).sum())
        rec['n_circuits'] = g.circuit.nunique()
        summ.append(rec)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(a.out_dir, 'ceiling_summary.csv'), index=False)
    pc = ps.groupby(['metric', 'D', 'circuit']).agg(sel_disc=('sel_disc', 'mean'), eqK_disc=('eqK_disc', 'mean'),
                                                    oracle_held=('oracle_held', 'mean'),
                                                    vs_best_orig=('vs_best_orig', 'mean'),
                                                    n_cand=('n_cand', 'mean'))
    pc.to_csv(os.path.join(a.out_dir, 'ceiling_per_circuit.csv'))
    pd.set_option('display.width', 200)
    show = ['metric', 'D', 'n_sources', 'mean_cands', 'frac_moved', 'sel_disc', 'sel_disc_rel%', 'sel_disc_ci_lo',
            'sel_disc_ci_hi', 'eqK_disc_rel%', 'oracle_held_rel%', 'nbr_mean_rel%', 'vs_best_orig', 'circuits_positive']
    print(summ[show].round(2).to_string(index=False))

    dec = {}
    for m in METRICS:
        s = summ[summ.metric == m].set_index('D')
        if 1 not in s.index:
            continue
        top = int(s.index.max())
        dec[m] = dict(rel1=float(s.loc[1, 'sel_disc_rel%']), relmax=float(s.loc[top, 'sel_disc_rel%']),
                      relmax_eqK=float(s.loc[top, 'eqK_disc_rel%']), max_depth=top,
                      growth_pp=float(s.loc[top, 'sel_disc_rel%'] - s.loc[1, 'sel_disc_rel%']),
                      beats_best_original=float(s.loc[top, 'vs_best_orig']))
    d = dec.get('routed_depth', {})
    dec['verdict'] = ('ceiling >= threshold' if d and d['relmax'] >= a.target_pct else
                      'ceiling below threshold: multi-step does not rescue the effect size' if d else 'n/a')
    dec['target_pct'] = a.target_pct
    json.dump(dec, open(os.path.join(a.out_dir, 'decision.json'), 'w'), indent=2)
    print(json.dumps(dec, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['enumerate', 'route', 'analyze'])
    ap.add_argument('--plans', default='experiment2_plan_features.csv')
    ap.add_argument('--chains-col', default='reuse_chains')
    ap.add_argument('--qasm-dir', default='.')
    ap.add_argument('--c1-dir', default='exp4c1_out')
    ap.add_argument('--out-dir', default='exp5_out')
    ap.add_argument('--router', default='router_template:route')
    ap.add_argument('--max-depth', type=int, default=3)
    ap.add_argument('--max-per-depth', type=int, default=0, help='cap new plans per source per level (0 = none)')
    ap.add_argument('--circuits', default='')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--reps', type=int, default=200)
    ap.add_argument('--target-pct', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mock-qc', action='store_true', help='testing only: skip loading qasm')
    a = ap.parse_args()
    {'enumerate': enumerate_stage, 'route': route_stage, 'analyze': analyze_stage}[a.stage](a)


if __name__ == '__main__':
    main()
