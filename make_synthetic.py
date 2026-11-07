"""Smoke test: writes a synthetic exp4c1_out/ so exp4c1_simulate.py can be run without Qiskit."""
import json, os, sys
import numpy as np, pandas as pd
out = sys.argv[1] if len(sys.argv) > 1 else 'synth_out'
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(1)
seeds = list(range(42, 47)) + list(range(100, 130))
cost, feats, cat, pools = [], [], [], []
for ci, circ in enumerate(['c1.qasm', 'c2.qasm', 'c3.qasm']):
    for p in range(8):
        base = 60 + 20 * ci + rng.normal(0, 4)
        osig = f'{circ}_P{p}_orig'
        plans = [(osig, 0.0, 0.0, 0)]
        for k in range(int(rng.integers(2, 9))):
            plans.append((f'{circ}_P{p}_n{k}', rng.normal(0, 2), rng.normal(0, 0.1), int(rng.integers(2, 8))))
        for sig, dmx, dcv, nch in plans:
            true_mean = base + 0.8 * dmx + rng.normal(0, 1)      # hotspot weakly informative
            for s in seeds:
                v = true_mean + rng.normal(0, 3)
                cost.append(dict(circuit=circ, sig=sig, seed=s, swap_count=v, routed_depth=3.5 * v + rng.normal(0, 6),
                                 routing_time=1.0, status='ok'))
            feats.append(dict(circuit=circ, sig=sig, max_wdeg=50 + dmx, wdeg_cv=0.5 + dcv, n_chains=6,
                              n_boundaries=10, logical_depth=100 + nch, norm_depth=110, dyn_width=6))
            plan = [[0, 1 + nch], [2, 3]]
            cat.append(dict(circuit=circ, sig=sig, plan_json=json.dumps(plan), is_original=sig == osig, produced_by='x'))
            pools.append(dict(circuit=circ, source_plan_id=f'P{p}', sig=sig, is_original=sig == osig,
                              in_A=bool(rng.random() < .5) or sig == osig, in_B=True, in_C=True))
pd.DataFrame(cost).to_csv(f'{out}/cost_table.csv', index=False)
pd.DataFrame(feats).to_csv(f'{out}/features.csv', index=False)
pd.DataFrame(cat).drop_duplicates(['circuit', 'sig']).to_csv(f'{out}/catalog.csv', index=False)
pd.DataFrame(pools).to_csv(f'{out}/pools.csv', index=False)
print('synthetic table written to', out)
