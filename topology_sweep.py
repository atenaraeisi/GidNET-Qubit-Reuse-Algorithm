#!/usr/bin/env python3
"""Experiment 8 (1): does the topology change what Repair can do?

For every topology (hh7 = baseline of 4A, grid, line) this script
  1. reuses the plan catalog / pools of your 4C-1 directory (the PLANS do not depend on the hardware),
  2. re-routes every plan x 35 seeds on that topology  (exp4c1_build_table.py, resumable),
  3. reruns the 4D predictability analysis on the new cost table,
  4. summarises everything in topology_summary.csv and topology_cross.csv.

  python topology_sweep.py --c1-dir exp4c1_all --features exp4d_results_all/exp4d_features_all.csv \
         --qasm-dir benchmarks_compact --out-dir topo_out --topologies hh7,grid,line --workers 4

Stages can be rerun: finished routing is skipped.  Use --summarize-only to only rebuild the tables.
Run check_router_parity.py FIRST (hh7 must reproduce your existing cost table).

Outputs
  topo_out/<topo>/cost_table.csv, features.csv, ... , exp4d/{policy_summary,feature_signal,decision}.json|csv
  topology_summary.csv : per topology x objective -> opportunity (plan spread), hotspot signal, policy gains (macro %),
                         oracle ceilings, and the gain of the best original plan over the mean plan
  topology_cross.csv   : per circuit x objective, Spearman correlation of plan cost across topology pairs
                         (is a 'good' plan good everywhere, or is the preference hardware-specific?)
"""
import argparse, json, os, shutil, subprocess, sys
import numpy as np
import pandas as pd

ENV = None
HERE = os.path.dirname(os.path.abspath(__file__))
OBJS = ['swap_count', 'routed_depth']
POLICIES = ['random', 'rule_hotspot', 'rule_logical_depth', 'ridge_nb', 'ridge_nb|accept',
            'ridge_all|accept', 'oracle_disc5', 'oracle_insample']


def find(name, extra_dirs=()):
    for d in (*extra_dirs, HERE, os.getcwd(), os.path.join(HERE, '..', 'exp4c1'), os.path.join(HERE, '..', 'exp4d')):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError(name + ' (put it next to topology_sweep.py)')


def run(cmd):
    print('$', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, env=ENV)
    if r.returncode:
        raise SystemExit(f'command failed ({r.returncode}): {" ".join(cmd)}')


def mean_costs(cost_path):
    c = pd.read_csv(cost_path)
    c = c[c.status == 'ok']
    return c.groupby(['circuit', 'sig'])[OBJS].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--c1-dir', required=True, help='existing 4C-1 dir with catalog.csv and pools.csv')
    ap.add_argument('--features', required=True, help='exp4d_features*.csv (plan features)')
    ap.add_argument('--qasm-dir', default='benchmarks_compact')
    ap.add_argument('--out-dir', default='topo_out')
    ap.add_argument('--topologies', default='hh7,grid,line')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--discovery-seeds', default='42-46')
    ap.add_argument('--heldout-seeds', default='100-129')
    ap.add_argument('--boot', type=int, default=500)
    ap.add_argument('--summarize-only', action='store_true')
    a = ap.parse_args()
    topos = a.topologies.split(',')
    os.makedirs(a.out_dir, exist_ok=True)
    build = find('exp4c1_build_table.py'); pred = find('exp4d_predictability.py'); find('router_topo.py')
    dummy = os.path.join(a.out_dir, '_plans_unused.csv')
    pd.DataFrame(columns=['circuit', 'plan_id', 'width', 'reuse_chains']).to_csv(dummy, index=False)
    global ENV
    ENV = dict(os.environ, PYTHONPATH=os.pathsep.join([HERE, os.getcwd(), os.environ.get('PYTHONPATH', '')]))

    for t in topos:
        d = os.path.join(a.out_dir, t)
        os.makedirs(d, exist_ok=True)
        if a.summarize_only:
            continue
        for f in ('catalog.csv', 'pools.csv'):      # same plans on every topology; features.csv is rebuilt by the router stage
            if not os.path.exists(os.path.join(d, f)):
                shutil.copy(os.path.join(a.c1_dir, f), os.path.join(d, f))
        run([sys.executable, build, '--plans', dummy, '--qasm-dir', a.qasm_dir, '--out-dir', d,
             '--router', f'router_topo:route_{t}', '--workers', str(a.workers),
             '--discovery-seeds', a.discovery_seeds, '--heldout-seeds', a.heldout_seeds])
        run([sys.executable, pred, '--c1-dir', d, '--features', a.features,
             '--out-dir', os.path.join(d, 'exp4d'), '--boot', str(a.boot), '--no-gbr',
             '--disc-seeds', a.discovery_seeds])

    # ---------------- summary ----------------
    rows = []
    costs = {}
    for t in topos:
        d = os.path.join(a.out_dir, t)
        cp = os.path.join(d, 'cost_table.csv')
        if not os.path.exists(cp):
            print('skip', t, '(no cost table)'); continue
        mc = mean_costs(cp); costs[t] = mc
        cat = pd.read_csv(os.path.join(d, 'catalog.csv'))
        orig = set(map(tuple, cat[cat.is_original.astype(bool)][['circuit', 'sig']].itertuples(index=False, name=None)))
        ps = pd.read_csv(os.path.join(d, 'exp4d', 'policy_summary.csv'))
        fs = pd.read_csv(os.path.join(d, 'exp4d', 'feature_signal.csv'))
        n_fail = int((pd.read_csv(cp).status != 'ok').sum())
        for o in OBJS:
            r = dict(topology=t, objective=o, n_plans=len(mc), n_failed_rows=n_fail)
            # opportunity: how much do plans of the same circuit differ?  (mean over circuits of (max-min)/mean)
            g = mc[o].groupby('circuit')
            r['plan_spread_pct'] = float(((g.max() - g.min()) / g.mean()).mean() * 100)
            r['mean_cost'] = float(mc[o].mean())
            # hotspot signal
            h = fs[(fs.objective == o) & (fs.feature == 'max_weighted_degree')]
            if len(h):
                r['hotspot_pooled_spearman'] = float(h.pooled_spearman.iloc[0])
                r['hotspot_mean_circuit_spearman'] = float(h.mean_circuit_spearman.iloc[0])
                r['hotspot_circuits_neg_pos'] = f'{int(h.circuits_negative_sign.iloc[0])}/{int(h.circuits_positive_sign.iloc[0])}'
            for p in POLICIES:
                s = ps[(ps.objective == o) & (ps.policy == p)]
                if len(s):
                    r[f'gain_{p}'] = float(s.macro_rel_gain_pct.iloc[0])
                    r[f'ci_{p}'] = f'[{s.ci_lo.iloc[0]:.2f},{s.ci_hi.iloc[0]:.2f}]'
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out_dir, 'topology_summary.csv'), index=False)

    # cross-topology agreement of plan quality, per circuit
    cross = []
    ts = list(costs)
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            ci, cj = costs[ts[i]], costs[ts[j]]
            idx = ci.index.intersection(cj.index)
            for circ in sorted({k[0] for k in idx}):
                sub = [k for k in idx if k[0] == circ]
                if len(sub) < 4:
                    continue
                for o in OBJS:
                    x, y = ci.loc[sub, o], cj.loc[sub, o]
                    rho = x.rank().corr(y.rank()) if x.nunique() > 1 and y.nunique() > 1 else np.nan
                    cross.append(dict(circuit=circ, objective=o, topo_a=ts[i], topo_b=ts[j], n_plans=len(sub), spearman=rho))
    cr = pd.DataFrame(cross)
    cr.to_csv(os.path.join(a.out_dir, 'topology_cross.csv'), index=False)

    pd.set_option('display.width', 250, 'display.max_columns', 40)
    print('\n=== topology_summary ===')
    show = ['topology', 'objective', 'plan_spread_pct', 'hotspot_pooled_spearman', 'hotspot_circuits_neg_pos',
            'gain_rule_hotspot', 'gain_ridge_nb|accept', 'gain_oracle_disc5', 'gain_oracle_insample']
    print(out[[c for c in show if c in out.columns]].round(2).to_string(index=False))
    if len(cr):
        print('\n=== cross-topology Spearman of plan quality (median over circuits) ===')
        print(cr.groupby(['objective', 'topo_a', 'topo_b']).spearman.agg(['median', 'count']).round(2).to_string())


if __name__ == '__main__':
    main()
