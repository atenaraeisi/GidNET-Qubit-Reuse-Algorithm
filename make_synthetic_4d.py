"""Positive/negative control for exp4d_predictability.py: synthetic cost table where routed cost depends
(or does not depend) on plan features.  usage: python make_synthetic_4d.py OUT_DIR signal|null"""
import json, os, sys
import numpy as np, pandas as pd
out, mode = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(5)
seeds = list(range(42, 47)) + list(range(100, 130))
cost, f1, f4, cat, pools = [], [], [], [], []
for ci, circ in enumerate(['a.qasm', 'b.qasm', 'c.qasm', 'd.qasm', 'e.qasm']):
    n_plans = 30
    sigs = [f'{circ}_{i}' for i in range(n_plans)]
    hot = rng.normal(60 + 10 * ci, 6, n_plans); cv = rng.normal(.4, .08, n_plans); ld = rng.normal(170, 8, n_plans)
    extra = rng.normal(0, 1, (n_plans, 3))
    base = 100 + 40 * ci
    true = base + (0.8 * (hot - hot.mean()) + 0.5 * (ld - ld.mean()) if mode == 'signal' else 0) + rng.normal(0, 1.5, n_plans)
    for i, s in enumerate(sigs):
        for sd in seeds:
            v = true[i] + rng.normal(0, 4)
            cost.append(dict(circuit=circ, sig=s, seed=sd, swap_count=v / 3, routed_depth=v, status='ok'))
        f1.append(dict(circuit=circ, sig=s, max_wdeg=hot[i], wdeg_cv=cv[i], logical_depth=ld[i], norm_depth=ld[i] + 3))
        f4.append(dict(circuit=circ, sig=s, max_weighted_degree=hot[i], weighted_degree_cv=cv[i],
                       e1=extra[i, 0], e2=extra[i, 1], e3=extra[i, 2]))
        cat.append(dict(circuit=circ, sig=s, plan_json='[[0]]', is_original=i < 8, produced_by='x'))
    for src in range(8):                       # 8 sources, each with 6 random neighbours
        for k in rng.choice([j for j in range(8, n_plans)], 6, replace=False):
            pools.append(dict(circuit=circ, source_plan_id=f'P{src}', sig=sigs[k], is_original=False, in_A=False, in_B=True, in_C=True))
        pools.append(dict(circuit=circ, source_plan_id=f'P{src}', sig=sigs[src], is_original=True, in_A=True, in_B=True, in_C=True))
pd.DataFrame(cost).to_csv(f'{out}/cost_table.csv', index=False); pd.DataFrame(f1).to_csv(f'{out}/features.csv', index=False)
pd.DataFrame(cat).to_csv(f'{out}/catalog.csv', index=False); pd.DataFrame(pools).to_csv(f'{out}/pools.csv', index=False)
pd.DataFrame(f4).to_csv(f'{out}/feat4d.csv', index=False)
