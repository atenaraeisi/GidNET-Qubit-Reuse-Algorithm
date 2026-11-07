#!/usr/bin/env python3
"""Experiment 4C-1, step 2: budgeted policy simulation on the full cost table.

Budget unit = ONE routing run (one SABRE seed on one plan).  The delivered result of a
policy is the cheapest routed circuit it observed, so no selection bias exists and no
extra held-out evaluation is needed.  Gain = (expected cost of Original-only best-of-R)
- (expected cost of the policy at the same R); positive = better.

Policies (all always contain the Original):
  orig_only          R seeds on the Original (computed exactly, order statistics)
  random_suffix      each run picks a random plan from Original + suffix pool
  random_expanded    same, from Original + (suffix + chain_transfer) pool
  guided_<rk>_m<k>   rank the expanded pool with a cheap pre-routing score, spread the
                     runs round-robin over Original + top-k   (rk = hotspot | learned)
  sh_suffix / sh_expanded   successive halving on observed mean cost
  oracle_*           all runs on the truly best plan (in-sample upper bound)

Rankers: 'hotspot' = reduction of max weighted degree; 'learned' = ridge regression on
cheap delta-features trained leave-one-circuit-out (never sees the test circuit).

Inputs : exp4c1_out/{cost_table,catalog,pools,features}.csv (from exp4c1_build_table.py)
Outputs: policy_results_per_source.csv, policy_summary.csv, feature_correlations.csv,
         paired_vs_random.csv, decision.json
"""
from __future__ import annotations

import argparse
import json
import os
from math import comb

import numpy as np
import pandas as pd

OBJS = ['swap_count', 'routed_depth']
FEATS = ['d_max_wdeg', 'd_wdeg_cv', 'depth_ratio', 'n_changed']


# ------------------------------------------------------------ exact baseline
def exact_orig_bok(x, R):
    x = np.sort(np.asarray(x, float))
    S = len(x)
    if R >= S:
        return float(x[0])
    i = np.arange(1, S - R + 2)
    w = np.array([comb(S - k, R - 1) for k in i], float) / comb(S, R)
    return float((x[: S - R + 1] * w).sum())


# ------------------------------------------------------------ policies
class Runs:
    """Draw seeds without replacement per plan from a cost matrix X (P x S)."""
    def __init__(self, X, rng):
        self.X, self.rng = X, rng
        self.perm = [rng.permutation(X.shape[1]) for _ in range(X.shape[0])]
        self.ptr = np.zeros(X.shape[0], int)
        self.sum = np.zeros(X.shape[0])
        self.best = np.inf

    def run(self, p):
        if self.ptr[p] >= self.X.shape[1]:
            return False
        v = self.X[p, self.perm[p][self.ptr[p]]]
        self.ptr[p] += 1
        self.sum[p] += v
        self.best = min(self.best, v)
        return True

    def mean(self, p):
        return self.sum[p] / max(self.ptr[p], 1)


def pol_random(X, R, rng, ctx):
    r = Runs(X, rng)
    for _ in range(R):
        avail = np.flatnonzero(r.ptr < X.shape[1])
        if len(avail) == 0:
            break
        r.run(int(rng.choice(avail)))
    return r.best


def pol_guided(X, R, rng, ctx):
    score, m = ctx['score'], ctx['m']
    order = np.argsort(-score[1:], kind='stable')[:m] + 1
    active = [0] + [int(i) for i in order]
    r = Runs(X, rng)
    for i in range(R):
        if not r.run(active[i % len(active)]):
            break
    return r.best


def pol_sh(X, R, rng, ctx):
    r = Runs(X, rng)
    P = X.shape[0]
    active = list(range(P))
    budget = R
    if P > R:                        # not enough runs to try everyone once
        active = [0] + [int(i) for i in rng.choice(np.arange(1, P), R - 1, replace=False)] if R > 1 else [0]
    while budget > 0 and active:
        for p in list(active):
            if budget == 0:
                break
            if r.run(p):
                budget -= 1
        if all(r.ptr[p] >= X.shape[1] for p in active):
            break
        if len(active) > 1:
            active.sort(key=r.mean)
            active = active[: max(1, (len(active) + 1) // 2)]
    return r.best


def pol_oracle(X, R, rng, ctx):
    r = Runs(X, rng)
    p = int(np.argmin(X.mean(1)))
    for _ in range(R):
        if not r.run(p):
            break
    return r.best


# ------------------------------------------------------------ ranker utils
def ridge_fit(Xtr, ytr, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Z = (Xtr - mu) / sd
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (ytr - ytr.mean()))
    return lambda Xte: ((Xte - mu) / sd) @ w + ytr.mean()


def boundaries(plan_json):
    plan = json.loads(plan_json)
    return {(a, b) for c in plan for a, b in zip(c, c[1:])}


# ------------------------------------------------------------ bootstrap
def strat_boot(df, col, rng, n):
    groups = [g[col].to_numpy() for _, g in df.groupby('circuit')]
    point = float(np.mean([g.mean() for g in groups]))
    boots = [np.mean([rng.choice(g, len(g)).mean() for g in groups]) for _ in range(n)]
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ------------------------------------------------------------ data assembly
def build_sources(obj, cost, pools, feats, catalog):
    piv = cost.pivot_table(index=['circuit', 'sig'], columns='seed', values=obj).dropna()
    S = piv.shape[1]
    fe = feats.set_index(['circuit', 'sig'])
    bset = {(r.circuit, r.sig): boundaries(r.plan_json) for r in catalog.itertuples()}
    out = []
    for (circ, src), g in pools.groupby(['circuit', 'source_plan_id']):
        o = g[g.is_original]
        if o.empty or (circ, o.sig.iloc[0]) not in piv.index:
            continue
        osig = o.sig.iloc[0]
        c = g[~g.is_original & g.apply(lambda r: (r.circuit, r.sig) in piv.index, axis=1)]
        sigs = [osig] + c.sig.tolist()
        X = np.vstack([piv.loc[(circ, s)].to_numpy() for s in sigs])
        f0 = fe.loc[(circ, osig)]
        rows = []
        for s in sigs:
            f = fe.loc[(circ, s)]
            rows.append([f['max_wdeg'] - f0['max_wdeg'], f['wdeg_cv'] - f0['wdeg_cv'],
                         f['logical_depth'] / f0['logical_depth'] if f0['logical_depth'] else 1.0,
                         len(bset[(circ, s)] ^ bset[(circ, osig)])])
        out.append(dict(circuit=circ, source=src, X=X,
                        inA=np.array([True] + c.in_A.tolist()), inB=np.array([True] + c.in_B.tolist()),
                        F=np.nan_to_num(np.array(rows, float)),
                        gain=X[0].mean() - X.mean(1)))
    return out, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', default='exp4c1_out')
    ap.add_argument('--out-dir', default='exp4c1_results')
    ap.add_argument('--R', default='1,2,3,5,10,15,30')
    ap.add_argument('--reps', type=int, default=200)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    Rs = [int(x) for x in args.R.split(',')]

    cost = pd.read_csv(os.path.join(args.in_dir, 'cost_table.csv'))
    n_bad = int((cost.status != 'ok').sum())
    if n_bad:
        print(f'[warn] {n_bad} failed routing rows excluded (plans with any missing seed are dropped)')
    cost = cost[cost.status == 'ok']
    pools = pd.read_csv(os.path.join(args.in_dir, 'pools.csv'))
    feats = pd.read_csv(os.path.join(args.in_dir, 'features.csv'))
    catalog = pd.read_csv(os.path.join(args.in_dir, 'catalog.csv'))

    res_rows, corr_rows = [], []
    for obj in OBJS:
        sources, S = build_sources(obj, cost, pools, feats, catalog)
        print(f'[{obj}] {len(sources)} sources, {S} seeds')
        if max(Rs) > S:
            raise SystemExit(f'max R={max(Rs)} exceeds number of seeds {S}')

        # ---- feature correlations (delta feature vs true gain, candidates only)
        allrows = pd.DataFrame([dict(circuit=s['circuit'], **dict(zip(FEATS, f)), gain=g)
                                for s in sources for f, g in zip(s['F'][1:], s['gain'][1:])])
        for circ, d in list(allrows.groupby('circuit')) + [('ALL', allrows)]:
            for f in FEATS:
                corr_rows.append(dict(objective=obj, circuit=circ, feature=f, n=len(d),
                                      spearman=d[f].corr(d['gain'], method='spearman')))

        # ---- learned ranker, leave-one-circuit-out
        pred = {}
        for circ in {s['circuit'] for s in sources}:
            tr = allrows[allrows.circuit != circ]
            model = ridge_fit(tr[FEATS].to_numpy(float), tr['gain'].to_numpy(float))
            for i, s in enumerate(sources):
                if s['circuit'] == circ:
                    pred[i] = model(s['F'])

        # ---- simulate
        for i, s in enumerate(sources):
            rng = np.random.default_rng([args.seed, i, OBJS.index(obj)])
            X = s['X']
            idx_suf = np.flatnonzero(s['inA'])
            idx_exp = np.flatnonzero(s['inB'])
            policies = [('random_suffix', 'suffix', pol_random, {}),
                        ('random_expanded', 'expanded', pol_random, {}),
                        ('sh_suffix', 'suffix', pol_sh, {}),
                        ('sh_expanded', 'expanded', pol_sh, {}),
                        ('oracle_suffix', 'suffix', pol_oracle, {}),
                        ('oracle_expanded', 'expanded', pol_oracle, {})]
            for rk, score in (('hotspot', -s['F'][idx_exp, 0]), ('learned', pred[i][idx_exp])):
                for m in (1, 2, 3):
                    sc = np.asarray(score, float)
                    policies.append((f'guided_{rk}_m{m}', 'expanded', pol_guided, {'score': sc, 'm': m}))
            for R in Rs:
                base = exact_orig_bok(X[0], R)
                for name, pool, fn, ctx in policies:
                    idx = idx_suf if pool == 'suffix' else idx_exp
                    Xp = X[idx]
                    vals = [fn(Xp, R, rng, ctx) for _ in range(args.reps)]
                    gain = base - float(np.mean(vals))
                    res_rows.append(dict(objective=obj, circuit=s['circuit'], source=s['source'],
                                         policy=name, R=R, orig_bok=base, policy_mean=float(np.mean(vals)),
                                         gain=gain, rel_gain_pct=100 * gain / base,
                                         n_candidates=len(idx) - 1))
            print(f'  [{obj}] source {i + 1}/{len(sources)} done')

    res = pd.DataFrame(res_rows)
    res.to_csv(os.path.join(args.out_dir, 'policy_results_per_source.csv'), index=False)
    pd.DataFrame(corr_rows).to_csv(os.path.join(args.out_dir, 'feature_correlations.csv'), index=False)

    # ---- summary with stratified bootstrap
    brng = np.random.default_rng(args.seed + 1)
    summ = []
    for (obj, pol, R), d in res.groupby(['objective', 'policy', 'R']):
        macro, lo, hi = strat_boot(d, 'gain', brng, args.boot)
        cm = d.groupby('circuit').gain.mean()
        summ.append(dict(objective=obj, policy=pol, R=R, macro_gain=macro, ci_lo=lo, ci_hi=hi,
                         micro_gain=d.gain.mean(), macro_rel_pct=d.groupby('circuit').rel_gain_pct.mean().mean(),
                         win_rate=float((d.gain > 1e-9).mean()), circuits_positive=int((cm > 0).sum()),
                         n_circuits=int(len(cm)), n_sources=int(len(d))))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(args.out_dir, 'policy_summary.csv'), index=False)

    # ---- paired comparison of every guided / SH policy against random_expanded
    pair = []
    for (obj, R), d in res.groupby(['objective', 'R']):
        w = d.pivot_table(index=['circuit', 'source'], columns='policy', values='gain').reset_index()
        for pol in [c for c in w.columns if c.startswith(('guided', 'sh_'))]:
            w['diff'] = w[pol] - w['random_expanded']
            macro, lo, hi = strat_boot(w[['circuit', 'diff']], 'diff', brng, args.boot)
            pair.append(dict(objective=obj, R=R, policy=pol, vs='random_expanded',
                             macro_diff=macro, ci_lo=lo, ci_hi=hi))
    pair = pd.DataFrame(pair)
    pair.to_csv(os.path.join(args.out_dir, 'paired_vs_random.csv'), index=False)

    # ---- pre-registered decision rule
    non_oracle = [p for p in summ.policy.unique() if not p.startswith('oracle')]
    decision = {}
    for pol in non_oracle:
        ok = True
        for obj in OBJS:
            for R in (5, 10):
                r = summ[(summ.policy == pol) & (summ.objective == obj) & (summ.R == R)]
                if r.empty or not (r.ci_lo.iloc[0] > 0 and r.circuits_positive.iloc[0] >= 3):
                    ok = False
        decision[pol] = ok
    guided_beats_random = {}
    for pol in [p for p in non_oracle if p.startswith('guided')]:
        ok = True
        for obj in OBJS:
            for R in (5, 10):
                r = pair[(pair.policy == pol) & (pair.objective == obj) & (pair.R == R)]
                if r.empty or not (r.ci_lo.iloc[0] > 0):
                    ok = False
        guided_beats_random[pol] = ok
    verdict = dict(passes_equal_budget_rule=decision,
                   guided_beats_random_expanded=guided_beats_random,
                   GO=any(decision.values()))
    json.dump(verdict, open(os.path.join(args.out_dir, 'decision.json'), 'w'), indent=2)

    show = summ[summ.R.isin([1, 5, 10, 30])].sort_values(['objective', 'R', 'macro_gain'], ascending=[True, True, False])
    print(show[['objective', 'R', 'policy', 'macro_gain', 'ci_lo', 'ci_hi', 'circuits_positive', 'win_rate']]
          .round(3).to_string(index=False))
    print(json.dumps(verdict, indent=2))


if __name__ == '__main__':
    main()
