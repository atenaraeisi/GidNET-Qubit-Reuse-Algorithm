#!/usr/bin/env python3
"""OFFLINE diagnostic of a routing-free TopologyScorer on Experiment-8 pools.

IMPORTANT: This ranks *all already-enumerated candidates* from the historical
catalog. It is NOT a deployable repair search and NOT a fresh held-out test.
Only held-out routing seed labels 100..129 are read AFTER selection.
The one original included in each historical pool is an explicit No Repair.
"""
from __future__ import annotations
import argparse
import io
import json
import zipfile
import pathlib
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from topology_embedding_v1 import (TopologyScorer,graph_for_topology,read_twoq_qasm2_compact)


def read_zip_csv(z, path):return pd.read_csv(io.BytesIO(z.read(path)))


def macro_avg(df,name):
    return float(df.groupby('circuit',sort=False)[name].mean().mean())


def stage(a):
    out=pathlib.Path(a.out_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Use new empty --out-dir: {out}')
    out.mkdir(parents=True,exist_ok=True)
    metrics=[];decisions=[];ranks=[];validation=[]
    seedset=list(range(100,130))
    with zipfile.ZipFile(a.topo_zip) as z, zipfile.ZipFile(a.qasm_zip) as qz:
        names=[x.strip() for x in a.topologies.split(',') if x.strip()]
        for name in names:
            base=f'topo_out/{name}/'
            catalog=read_zip_csv(z,base+'catalog.csv')
            pool=read_zip_csv(z,base+'pools.csv')
            savedfeat=read_zip_csv(z,base+'features.csv')
            costs=read_zip_csv(z,base+'cost_table.csv')
            if costs.duplicated(['circuit','sig','seed']).any():raise AssertionError('Repeated routing keys')
            if (costs.status!='ok').any():raise AssertionError('Incomplete cost table')
            evalcost=costs[costs.seed.isin(seedset)].groupby(['circuit','sig'],as_index=False).agg(real_swap=('swap_count','mean'))
            nseeds=costs[costs.seed.isin(seedset)].groupby(['circuit','sig']).seed.nunique()
            if nseeds.min()!=30 or nseeds.max()!=30:raise AssertionError('Routing seed mismatch')
            score_rows=[]
            g=graph_for_topology(name)
            for circuit, cg in catalog.groupby('circuit',sort=True):
                qtext=qz.read('benchmarks_compact/'+circuit)
                # Write only to an ephemeral path for the strict QASM2 parser.
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.qasm') as h:
                    h.write(qtext);h.flush()
                    pairs,n=read_twoq_qasm2_compact(h.name)
                scorer=TopologyScorer(pairs,g,starts=a.starts,local_steps=a.local_steps)
                for r in cg.itertuples(index=False):
                    plan=json.loads(r.plan_json)
                    if set(x for c in plan for x in c)!=set(range(n)):
                        raise AssertionError(f'Active compact mapping mismatch for {circuit}')
                    result=scorer.score(plan)
                    score_rows.append(dict(circuit=circuit,sig=r.sig,score=result.score,
                           relative_score=result.score/max(1,result.cross_gates),
                           quotient_unique_edges=result.unique_edges,hhi=result.hhi,
                           max_degree=result.max_degree,
                           greedy_mapping=json.dumps(result.placement)))
            df=pd.DataFrame(score_rows).merge(savedfeat,on=['circuit','sig'],validate='one_to_one').merge(evalcost,on=['circuit','sig'],validate='one_to_one')
            df['topology']=name
            if len(df)!=len(catalog):raise AssertionError('Missing plans or invalid merges')
            metrics.extend(df.to_dict('records'))
            rank_recs=[]
            for circuit,d in df.groupby('circuit'):
                for col in ['relative_score','quotient_unique_edges','max_wdeg','logical_depth','wdeg_cv']:
                    x=d[col].to_numpy(float);y=d.real_swap.to_numpy(float)
                    rho=float(spearmanr(x,y).statistic) if len(np.unique(x))>1 and len(np.unique(y))>1 else np.nan
                    rank_recs.append(dict(topology=name,circuit=circuit,feature=col,spearman_rho=rho,n_plans=len(d)))
            ranks.extend(rank_recs)
            dfi=df.set_index(['circuit','sig'],drop=False)
            for (circ,src_id),p in pool.groupby(['circuit','source_plan_id'],sort=False):
                orig=p[p.is_original]
                if len(orig)!=1:raise AssertionError('Each pool must have exactly one Original')
                baseline_sig=orig.iloc[0]['sig']
                pool_sigs=list(p.sig.unique())
                options=dfi.loc[[(circ,s) for s in pool_sigs]].copy()
                original=dfi.loc[(circ,baseline_sig)]
                rules={
                    'no_repair':None,
                    'topology_embed':('relative_score',),
                    'structural_lex':('quotient_unique_edges','neg_hhi','max_degree'),
                    'min_hotspot':('max_wdeg',),
                    'min_logical_depth':('logical_depth',)
                }
                options['neg_hhi']=-options['hhi']
                for method,cols in rules.items():
                    selected=baseline_sig
                    if cols is not None:
                        # Strict improvement; avoid arbitrary winner among tied scores.
                        original_val=tuple(float(original[col] if col!='neg_hhi' else -original.hhi) for col in cols)
                        best=options.sort_values(list(cols),ascending=True,kind='mergesort').iloc[0]
                        chosen_val=tuple(float(best[col]) for col in cols)
                        if chosen_val<original_val:
                            selected=str(best.sig)
                    pick=dfi.loc[(circ,selected)]
                    gain=float(original.real_swap-pick.real_swap)
                    decisions.append(dict(topology=name,circuit=circ,source_plan_id=src_id,
                        method=method,original_sig=baseline_sig,pick_sig=selected,
                        n_pool=len(pool_sigs),original_swaps=float(original.real_swap),
                        picked_swaps=float(pick.real_swap),gain_abs=gain,
                        gain_pct=100*gain/original.real_swap if original.real_swap>0 else np.nan,
                        original_metric=float(original.relative_score),picked_metric=float(pick.relative_score)))
            print(f'[score] {name}: {len(df)} plans, {len(pool.groupby(["circuit","source_plan_id"]))} originals',flush=True)
    dfm=pd.DataFrame(metrics);dfd=pd.DataFrame(decisions);dfr=pd.DataFrame(ranks)
    dfm.to_csv(out/'plan_topology_scores.csv',index=False)
    dfd.to_csv(out/'pool_decisions.csv',index=False)
    dfr.to_csv(out/'feature_spearman_by_circuit.csv',index=False)
    summary=[]
    for (top,method),d in dfd.groupby(['topology','method']):
        good=d.gain_abs>1e-9;bad=d.gain_abs < -1e-9
        summary.append(dict(topology=top,method=method,
            n_origins=len(d),n_circuits=d.circuit.nunique(),
            n_repair=int((d.pick_sig!=d.original_sig).sum()),
            n_better=int(good.sum()),n_harmed=int(bad.sum()),
            mean_gain_abs=float(d.gain_abs.mean()),
            mean_circuit_gain_abs=macro_avg(d,'gain_abs'),
            mean_circuit_gain_pct=macro_avg(d[d.original_swaps>0],'gain_pct') if (d.original_swaps>0).all() else np.nan))
    dfs=pd.DataFrame(summary).sort_values(['topology','method'])
    dfs.to_csv(out/'policy_summary.csv',index=False)
    with (out/'manifest.json').open('w',encoding='utf8') as h:
        json.dump(dict(status='retrospective_catalog_diagnostic_NOT_deployable',
            topologies=names,starts=a.starts,local_steps=a.local_steps,
            n_plans=len(dfm),n_pool_decisions=len(dfd),routing_eval_seeds=seedset,
            warning='All catalog candidates were already collected; this is NOT a fair candidate-generation comparison. Original search already used these benchmark circuits. No final-test generalization claim.'),h,indent=2)
    print('\n'+dfs.to_string(index=False, float_format=lambda x:f'{x:+.4f}'))
    print('[PASS] offline catalog-only diagnostic; no Routing executed')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--topo-zip',required=True)
    p.add_argument('--qasm-zip',required=True)
    p.add_argument('--out-dir',required=True)
    p.add_argument('--topologies',default='grid,line')
    p.add_argument('--starts',type=int,default=4)
    p.add_argument('--local-steps',type=int,default=1)
    stage(p.parse_args())
