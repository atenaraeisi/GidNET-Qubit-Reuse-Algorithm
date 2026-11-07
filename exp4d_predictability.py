#!/usr/bin/env python3
"""Experiment 4D, step 2: can CHEAP pre-routing information pick a better plan? (single routing run)

Setting (your real Repair): from one valid plan, move to a neighbour chosen BEFORE routing, route once.
Expected cost of that single run = mean over routing seeds, so no routing budget/selection bias is involved.

Data: the 4C-1 cost table (every plan x 35 seeds) + plan features (Experiment-2 features for every plan,
from exp4d_features.py, plus logical/normalised depth from the 4C-1 features.csv).

Protocol (strict):
  * features are z-scored WITHIN each circuit, so circuits are comparable;
  * target = relative gain % of neighbour vs its source plan (positive = better);
  * models are trained on the OTHER circuits only (leave-one-circuit-out) and are scored on the held-out
    circuit with the 30 held-out routing seeds; no seed of the test circuit is used for training/selection;
  * oracle references pick the neighbour with the 5 discovery seeds (needs routing, so NOT achievable);
  * 'accept' variant = move only if the model predicts a positive gain, otherwise keep the source plan.

Outputs (out-dir): policy_results_per_source.csv, policy_summary.csv, per_circuit.csv, feature_signal.csv,
ridge_coefficients.csv, pair_accuracy.csv, decision.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

OBJS = ['swap_count', 'routed_depth']
ID_COLS = {'circuit', 'sig'}


def parse_seeds(s):
    out = []
    for p in s.split(','):
        if '-' in p:
            a, b = p.split('-'); out.extend(range(int(a), int(b) + 1))
        elif p:
            out.append(int(p))
    return out


def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    ym = y.mean()
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return (lambda T: ((T - mu) / sd) @ w + ym), w


def pick_lambda(train, cols, lams, target):
    """inner leave-one-circuit-out: choose lambda by pooled correlation with the true gain"""
    circs = train.circuit.unique()
    best, best_s = lams[0], -9
    for lam in lams:
        sc = []
        for c in circs:
            tr, va = train[train.circuit != c], train[train.circuit == c]
            if len(tr) < 5 or len(va) < 3:
                continue
            f, _ = ridge_fit(tr[cols].to_numpy(float), tr[target].to_numpy(float), lam)
            p = f(va[cols].to_numpy(float))
            sc.append(np.corrcoef(p, va[target])[0, 1] if np.std(p) > 0 and va[target].std() > 0 else 0)
        s = np.nanmean(sc) if sc else -9
        if s > best_s:
            best, best_s = lam, s
    return best


def strat_boot(df, col, rng, n):
    groups = [g[col].to_numpy() for _, g in df.groupby('circuit')]
    point = float(np.mean([g.mean() for g in groups]))
    boots = [np.mean([rng.choice(g, len(g)).mean() for g in groups]) for _ in range(n)]
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--c1-dir', default='exp4c1_out')
    ap.add_argument('--features', default='exp4d_features.csv')
    ap.add_argument('--out-dir', default='exp4d_results')
    ap.add_argument('--disc-seeds', default='42-46')
    ap.add_argument('--lams', default='1,10,100,1000')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--min-gain-pct', type=float, default=2.0, help='suggested go/no-go threshold')
    ap.add_argument('--max-harm-pct', type=float, default=0.5)
    ap.add_argument('--min-capture', type=float, default=0.5)
    ap.add_argument('--min-circuit-frac', type=float, default=0.6)
    ap.add_argument('--no-gbr', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    DISC = parse_seeds(args.disc_seeds)
    lams = [float(x) for x in args.lams.split(',')]

    cost = pd.read_csv(os.path.join(args.c1_dir, 'cost_table.csv')); cost = cost[cost.status == 'ok']
    pools = pd.read_csv(os.path.join(args.c1_dir, 'pools.csv'))
    f1 = pd.read_csv(os.path.join(args.c1_dir, 'features.csv'))
    f4 = pd.read_csv(args.features)
    PF = f4.merge(f1[['circuit', 'sig', 'logical_depth', 'norm_depth']], on=['circuit', 'sig'])
    cols = [c for c in PF.columns if c not in ID_COLS]
    PF[cols] = PF[cols].astype(float)
    # within-circuit z-score (constant columns -> 0, NaN -> 0)
    Z = PF.copy()
    for c in PF.circuit.unique():
        m = PF.circuit == c
        mu, sd = PF.loc[m, cols].mean(), PF.loc[m, cols].std(ddof=0).replace(0, np.nan)
        Z.loc[m, cols] = ((PF.loc[m, cols] - mu) / sd).fillna(0.0)
    drop = [c for c in cols if Z[c].abs().sum() == 0]
    cols = [c for c in cols if c not in drop]
    Zi = Z.set_index(['circuit', 'sig'])[cols]
    RAW = PF.set_index(['circuit', 'sig'])
    print(f'{len(PF)} plans, {len(cols)} usable features (dropped constant: {drop})')

    SMALL = [c for c in ['max_weighted_degree', 'weighted_degree_cv', 'logical_depth'] if c in cols]
    all_rows, sig_rows, coef_rows, acc_rows = [], [], [], []

    for obj in OBJS:
        P = cost.pivot_table(index=['circuit', 'sig'], columns='seed', values=obj).dropna()
        n_seeds = P.shape[1]
        allm, allv = P.mean(axis=1), P.var(axis=1)
        heldcols = [s for s in P.columns if s not in DISC]
        held = P[heldcols].mean(axis=1); disc = P[[s for s in DISC if s in P.columns]].mean(axis=1)

        # ---- pair tables (neighbours) -----------------------------------------------------------
        nb = []
        for (c, src), g in pools.groupby(['circuit', 'source_plan_id']):
            o = g[g.is_original].sig.iloc[0]
            if (c, o) not in P.index:
                continue
            for s in g[g.in_B & ~g.is_original].sig:
                if (c, s) not in P.index:
                    continue
                d = Zi.loc[(c, s)].to_numpy() - Zi.loc[(c, o)].to_numpy()
                row = dict(circuit=c, src=src, o=o, s=s,
                           gain_all=100 * (allm[(c, o)] - allm[(c, s)]) / allm[(c, o)],
                           gain_held=100 * (held[(c, o)] - held[(c, s)]) / held[(c, o)],
                           gain_disc=100 * (disc[(c, o)] - disc[(c, s)]) / disc[(c, o)],
                           se=100 * np.sqrt(allv[(c, o)] / n_seeds + allv[(c, s)] / n_seeds) / allm[(c, o)],
                           raw_dhot=RAW.loc[(c, s)].max_weighted_degree - RAW.loc[(c, o)].max_weighted_degree)
                row.update({f'd_{k}': v for k, v in zip(cols, d)})
                nb.append(row)
        nb = pd.DataFrame(nb)
        dcols = [f'd_{c}' for c in cols]
        dsmall = [f'd_{c}' for c in SMALL]

        # ---- all ordered pairs inside each circuit (more training data) --------------------------
        ap_rows = []
        for c, g in PF.groupby('circuit'):
            sigs = [s for s in g.sig if (c, s) in P.index]
            Zc = Zi.loc[[(c, s) for s in sigs]].to_numpy(); mc = allm[[(c, s) for s in sigs]].to_numpy()
            for i in range(len(sigs)):
                for j in range(len(sigs)):
                    if i == j:
                        continue
                    r = dict(circuit=c, gain_all=100 * (mc[i] - mc[j]) / mc[i])
                    r.update({k: v for k, v in zip(dcols, Zc[j] - Zc[i])})
                    ap_rows.append(r)
        allpairs = pd.DataFrame(ap_rows)

        # ---- univariate signal --------------------------------------------------------------------
        for f_ in dcols:
            pooled = nb[f_].corr(nb.gain_all, method='spearman')
            per = nb.groupby('circuit').apply(lambda x: x[f_].corr(x.gain_all, method='spearman'), include_groups=False)
            sig_rows.append(dict(objective=obj, feature=f_[2:], pooled_spearman=pooled,
                                 mean_circuit_spearman=per.mean(), circuits_negative_sign=int((per < 0).sum()),
                                 circuits_positive_sign=int((per > 0).sum()),
                                 n_pairs=len(nb)))

        # ---- leave-one-circuit-out models ---------------------------------------------------------
        models = {}   # name -> (series of predictions aligned with nb.index)
        pred = {k: pd.Series(np.nan, index=nb.index) for k in
                ['ridge_nb', 'ridge_all', 'ridge_small_all'] + ([] if args.no_gbr else ['gbr_all'])}
        for c in nb.circuit.unique():
            te = nb.circuit == c
            tr_nb = nb[~te]
            tr_all = allpairs[allpairs.circuit != c]
            # subsample all-pairs training set for speed if huge
            if len(tr_all) > 40000:
                tr_all = tr_all.sample(40000, random_state=args.seed)
            lam = pick_lambda(tr_nb, dcols, lams, 'gain_all')
            f, w = ridge_fit(tr_nb[dcols].to_numpy(float), tr_nb.gain_all.to_numpy(float), lam)
            pred['ridge_nb'][te] = f(nb.loc[te, dcols].to_numpy(float))
            lam2 = pick_lambda(tr_all.sample(min(len(tr_all), 8000), random_state=1), dcols, lams, 'gain_all')
            f, w2 = ridge_fit(tr_all[dcols].to_numpy(float), tr_all.gain_all.to_numpy(float), lam2)
            pred['ridge_all'][te] = f(nb.loc[te, dcols].to_numpy(float))
            f, _ = ridge_fit(tr_all[dsmall].to_numpy(float), tr_all.gain_all.to_numpy(float), 10.0)
            pred['ridge_small_all'][te] = f(nb.loc[te, dsmall].to_numpy(float))
            if not args.no_gbr:
                from sklearn.ensemble import GradientBoostingRegressor
                gb = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.7,
                                               random_state=args.seed)
                gb.fit(tr_all[dcols].to_numpy(float), tr_all.gain_all.to_numpy(float))
                pred['gbr_all'][te] = gb.predict(nb.loc[te, dcols].to_numpy(float))
            coef_rows.append(dict(objective=obj, heldout_circuit=c, lam_nb=lam, lam_all=lam2,
                                  **{f'nb:{k}': v for k, v in zip(cols, w)},
                                  **{f'all:{k}': v for k, v in zip(cols, w2)}))
        nb['p_ridge_nb'], nb['p_ridge_all'] = pred['ridge_nb'], pred['ridge_all']
        nb['p_ridge_small_all'] = pred['ridge_small_all']
        if not args.no_gbr:
            nb['p_gbr_all'] = pred['gbr_all']
        # simple rules (score = predicted gain; ties broken at random)
        nb['p_rule_hotspot'] = -nb.raw_dhot
        if 'd_logical_depth' in nb:
            nb['p_rule_logical_depth'] = -nb['d_logical_depth']
        pnames = [c for c in nb.columns if c.startswith('p_')]

        # pair accuracy on pairs whose true gain is clearly non-zero (|t| > 2 over the 35 seeds)
        clear = nb[(nb.gain_all.abs() / nb.se) > 2]
        for p in pnames:
            sgn = np.sign(nb[p] if not p.startswith('p_rule') else nb[p])
            acc_rows.append(dict(objective=obj, model=p[2:], n_clear_pairs=len(clear),
                                 sign_accuracy_on_clear_pairs=float((np.sign(clear[p]) == np.sign(clear.gain_all)).mean()),
                                 within_source_spearman=float(nb.groupby(['circuit', 'src']).apply(
                                     lambda x: x[p].corr(x.gain_all, method='spearman') if len(x) > 2 else np.nan,
                                     include_groups=False).mean())))

        # ---- single-shot policies per source -------------------------------------------------------
        for (c, src), g in nb.groupby(['circuit', 'src']):
            r0 = dict(objective=obj, circuit=c, source=src, n_candidates=len(g))
            res = dict(random=g.gain_held.mean(),
                       oracle_disc5=g.loc[g.gain_disc.idxmax()].gain_held,
                       oracle_insample=g.gain_held.max())
            jitter = rng.random(len(g)) * 1e-9
            for p in pnames:
                sc = g[p].to_numpy() + jitter
                k = int(np.nanargmax(sc))
                res[p[2:]] = g.iloc[k].gain_held
                if not p.startswith('p_rule'):
                    res[p[2:] + '|accept'] = g.iloc[k].gain_held if g[p].iloc[k] > 0 else 0.0
            for k_, v in res.items():
                all_rows.append(dict(**r0, policy=k_, rel_gain_pct=v))

    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(args.out_dir, 'policy_results_per_source.csv'), index=False)
    pd.DataFrame(sig_rows).to_csv(os.path.join(args.out_dir, 'feature_signal.csv'), index=False)
    pd.DataFrame(coef_rows).to_csv(os.path.join(args.out_dir, 'ridge_coefficients.csv'), index=False)
    pd.DataFrame(acc_rows).to_csv(os.path.join(args.out_dir, 'pair_accuracy.csv'), index=False)

    summ = []
    for (obj, pol), d in res.groupby(['objective', 'policy']):
        macro, lo, hi = strat_boot(d, 'rel_gain_pct', rng, args.boot)
        cm = d.groupby('circuit').rel_gain_pct.mean()
        summ.append(dict(objective=obj, policy=pol, macro_rel_gain_pct=macro, ci_lo=lo, ci_hi=hi,
                         circuits_positive=int((cm > 0).sum()), n_circuits=len(cm), n_sources=len(d)))
    summ = pd.DataFrame(summ)
    orc = summ[summ.policy == 'oracle_disc5'].set_index('objective').macro_rel_gain_pct
    summ['capture_of_oracle'] = summ.apply(lambda r: r.macro_rel_gain_pct / orc[r.objective] if orc[r.objective] > 0 else np.nan, axis=1)
    summ.to_csv(os.path.join(args.out_dir, 'policy_summary.csv'), index=False)
    res.groupby(['objective', 'policy', 'circuit']).rel_gain_pct.mean().reset_index().to_csv(
        os.path.join(args.out_dir, 'per_circuit.csv'), index=False)

    # ---- suggested go / no-go -------------------------------------------------------------------------
    decision = {}
    for pol in summ.policy.unique():
        if pol.startswith('oracle') or pol == 'random':
            continue
        r = {o: summ[(summ.objective == o) & (summ.policy == pol)].iloc[0] for o in OBJS}
        passed_obj = []
        for o in OBJS:
            other = r[[x for x in OBJS if x != o][0]]
            ok = (r[o].macro_rel_gain_pct >= args.min_gain_pct and r[o].ci_lo > 0
                  and other.macro_rel_gain_pct >= -args.max_harm_pct
                  and r[o].circuits_positive >= np.ceil(args.min_circuit_frac * r[o].n_circuits)
                  and (r[o].capture_of_oracle >= args.min_capture))
            if ok:
                passed_obj.append(o)
        decision[pol] = dict(passes=bool(passed_obj), passing_objectives=passed_obj,
                             **{f'{o}_gain_pct': round(float(r[o].macro_rel_gain_pct), 3) for o in OBJS},
                             **{f'{o}_capture': round(float(r[o].capture_of_oracle), 3) for o in OBJS})
    out = dict(thresholds=dict(min_gain_pct=args.min_gain_pct, max_harm_pct=args.max_harm_pct,
                               min_capture=args.min_capture, min_circuit_frac=args.min_circuit_frac),
               note='deployable policies are the "|accept" variants and the rules; CIs treat sources as independent '
                    '(they share neighbours) and use only the available circuits - treat as optimistic.',
               GO=any(v['passes'] for v in decision.values()), policies=decision)
    json.dump(out, open(os.path.join(args.out_dir, 'decision.json'), 'w'), indent=2)

    pd.set_option('display.width', 220)
    show = summ.sort_values(['objective', 'macro_rel_gain_pct'], ascending=[True, False])
    print(show[['objective', 'policy', 'macro_rel_gain_pct', 'ci_lo', 'ci_hi', 'circuits_positive', 'capture_of_oracle']].round(2).to_string(index=False))
    print(json.dumps({k: v for k, v in out.items() if k != 'policies'}, indent=1))


if __name__ == '__main__':
    main()
