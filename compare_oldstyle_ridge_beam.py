#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact historical LOCO Ridge vs same Ridge + cost-labeled valid depth-2..5 pairs.

BOTH use the original 4C1 one-step B pool and its 35-seed labels.
MULTI adds additional deep valid mutation pairs with the same 35-seed labels,
the same 25 features and 4C1 per-circuit normalizer; exact same LOCO fitting,
lambda cross-validation, and fixed-budget cap_search. No new routing in compare.

This is a historical 15-circuit exploratory comparison, not independent final test.
The old baseline is refitted/loaded with exact original 4C1 helper functions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import adaptive_repair_cached as old
from build_oldstyle_multistep import atomic_csv,origins

ARMS=('old_one_step','old_plus_deep')
POLICIES=('adaptive_beam','random_beam')


def fingerprint_new(old_fp, deep_path, implementation):
    h=hashlib.sha256()
    h.update(b'old-plus-valid-deep-35seed-v1')
    h.update(old_fp.encode())
    for path in [deep_path,Path(implementation)]:
        h.update(path.read_bytes())
    return h.hexdigest()


def make_training(c1,Z,cols,deep_path,obj):
    original,_=old.build_training(c1,Z,cols,obj)
    deep=pd.read_csv(deep_path)
    required={'circuit','discovery_depth','gain_'+obj}|{'d_'+s for s in cols}
    if not required.issubset(deep.columns):
        raise ValueError(f'{deep_path} missing: {sorted(required-set(deep.columns))}')
    deep=deep[deep.discovery_depth.between(2,5)].copy()
    deep['gain_all']=pd.to_numeric(deep['gain_'+obj],errors='raise')
    deep=deep[['circuit','gain_all']+['d_'+s for s in cols]]
    if deep.empty:
        raise ValueError('No complete 35-seed deep training pairs. Run build --stage status/features; '
                         'optionally route a LIMITED number of missing plans first.')
    if not np.isfinite(deep[['gain_all']+['d_'+s for s in cols]].to_numpy(float)).all():
        raise ValueError('Deep training has non-finite targets/features')
    if deep.circuit.nunique()<2:
        raise ValueError('Deep data comes from <2 circuits. Discover more circuits for LOCO training.')
    both=pd.concat([original,deep],ignore_index=True)
    return original, both, deep


def model_pair(circuit,obj,columns,oldtrain,both,oldfp,multifp,old_model_dir,multi_model_dir,model_mode,lams):
    import exp4d_predictability as oldfit
    def fetch(train,dir,fp):
        mdl,state,path=old.get_ridge_model(dir,model_mode,lambda:train,oldfit,obj,circuit,columns,lams,fp)
        return mdl,state,str(path)
    o=fetch(oldtrain,old_model_dir,oldfp)
    m=fetch(both,multi_model_dir,multifp)
    return {'old_one_step':o,'old_plus_deep':m}


def load_all_held(paths):
    """Load historical 30-seed routing costs; deep routing output is optional.

    The features stage can reuse previously cached full 35-seed costs, so
    no deep_routed_cost.csv is created unless stage route was actually run.
    """
    present = [Path(p) for p in paths if Path(p).is_file()]
    if not present:
        raise FileNotFoundError('No historical routing cost table found')
    return old.load_heldout(present)


def resume_state(out,args):
    out.mkdir(parents=True,exist_ok=True)
    conf=out/'config.json';path=out/'per_source_checkpoint.csv'
    params={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
            if k not in ('self_test','resume','out_dir','progress')}
    paths=[args.c1_dir/x for x in ('catalog.csv','pools.csv','features.csv','cost_table.csv')]
    paths+=[args.feature_file,args.deep_pairs_file,args.space_dir/'cost_table.csv']
    for p in paths:
        if not p.exists(): raise FileNotFoundError(str(p))
    params['input_files']={str(p.resolve()):[p.stat().st_size,p.stat().st_mtime_ns] for p in paths}
    if conf.exists():
        if not args.resume:raise RuntimeError(f'Existing run: use --resume or a NEW --out-dir: {out}')
        if json.loads(conf.read_text())!=params:raise RuntimeError('Input files/options changed; cannot resume this run')
        if not path.exists():
            print('RESUME: no checkpoint yet; restarting first source',flush=True)
            return []
        rows=pd.read_csv(path,dtype={'source_plan_id':str}).to_dict('records')
        key=['objective','policy','arm','circuit','source_plan_id']
        df=pd.DataFrame(rows)
        if not df.empty and df.duplicated(key).any():raise RuntimeError('Duplicate rows in checkpoint')
        print('RESUME',len(rows),'rows',flush=True)
        return rows
    if path.exists():raise RuntimeError('Checkpoint exists without config; choose new out-dir')
    conf.write_text(json.dumps(params,indent=2),encoding='utf-8')
    return []


def generate_outputs(rows,out):
    df=pd.DataFrame(rows)
    if df.empty:raise ValueError('No source result rows')
    atomic_csv(df,out/'per_source.csv')
    per=df.groupby(['objective','arm','policy','circuit'],as_index=False).agg(
        gain_pct=('actual_gain_pct','mean'),n_sources=('source_plan_id','nunique'),
        n_eval=('evaluated','sum'),harm_rate=('harm','mean'),
        n_scored=('n_scored','mean'))
    atomic_csv(per,out/'per_circuit.csv')
    macro=per.groupby(['objective','arm','policy'],as_index=False).agg(
        mean_gain_pct=('gain_pct','mean'),n_circuits=('circuit','nunique'),
        n_evaluated_circuits=('gain_pct','count'))
    atomic_csv(macro,out/'macro_summary.csv')
    pairs=df.pivot_table(index=['objective','policy','circuit','source_plan_id'],columns='arm',
                         values='actual_gain_pct',aggfunc='first').reset_index()
    pairs['delta_deep_minus_old_pp']=pairs['old_plus_deep']-pairs['old_one_step']
    atomic_csv(pairs,out/'paired_delta.csv')
    print('\nRESULTS:\n',macro.to_string(index=False),flush=True)
    if df.evaluated.eq(0).any():
        print(f'WARNING: {int(df.evaluated.eq(0).sum())} selected-plan costs missing; averages incomplete',flush=True)


def run(args):
    if args.max_depth<2:raise ValueError('max-depth must be >=2')
    c1=args.c1_dir
    catalog,Z,stats,cols=old.prepare_features(c1,args.feature_file)
    oldfp=old.training_fingerprint(c1,args.feature_file,args.lams)
    multifp=fingerprint_new(oldfp,args.deep_pairs_file,__file__)
    origs=origins(c1,args.circuits,args.max_sources)
    out=args.out_dir
    logs=resume_state(out,args)
    done={(str(x['objective']),str(x['policy']),str(x['arm']),str(x['circuit']),str(x['source_plan_id']))
          for x in logs}
    print('EVALUATION ON SAME HISTORICAL ORIGINALS:',origs.shape[0],flush=True)
    print('NO ROUTING during prediction or model selection; held-out lookup after selection only.',flush=True)
    held=load_all_held([c1/'cost_table.csv',args.space_dir/'cost_table.csv',args.deep_data_dir/'deep_routed_cost.csv'])
    for objective in args.objectives.split(','):
        objective=objective.strip()
        if objective not in old.OBJECTIVES:raise ValueError(f'Invalid objective {objective}')
        orig_train,combined,deep=make_training(c1,Z,cols,args.deep_pairs_file,objective)
        print('TRAIN',objective,'old pairs',len(orig_train),'added depth2-5',len(deep),
              'source circuits with deep labels',deep.circuit.nunique(),flush=True)
        train_lams=args.lams
        for circ,originals in origs.groupby('circuit',sort=True):
            models=model_pair(circ,objective,cols,orig_train,combined,oldfp,multifp,
                args.old_model_dir,args.multi_model_dir,args.model_mode,train_lams)
            if args.progress:print('CIRCUIT',circ,'sources',len(originals),'models',
                    {a:models[a][1] for a in ARMS},flush=True)
            for row in originals.itertuples(index=False):
                sid=str(row.source_plan_id);start=row.sig
                required={(objective,p,a,circ,sid) for p in POLICIES for a in ARMS}
                if required.issubset(done):
                    if args.progress:print('SKIP',circ,sid,flush=True)
                    continue
                if required & done:
                    logs=[x for x in logs if (str(x['objective']),str(x['policy']),str(x['arm']),
                         str(x['circuit']),str(x['source_plan_id'])) not in required]
                    done={(str(x['objective']),str(x['policy']),str(x['arm']),str(x['circuit']),str(x['source_plan_id']))
                          for x in logs}
                for policy in POLICIES:
                    # EXACT historical seed, same for both arms to provide paired comparison.
                    seed=args.seed+sum(ord(z) for z in f'{circ}:{sid}:{policy}')
                    for arm in ARMS:
                        model,state,_=models[arm]
                        # Fresh adapter gives each arm the same starting state.
                        adapter=old.RepositoryAdapter(circ,catalog,args.qasm_dir,cols,stats,Z)
                        start_vec=adapter.vector(start,adapter.plan_of[(circ,start)])
                        def pred(sig,plan,_model=model,_start=start_vec,_adapter=adapter):
                            return float(_model((_adapter.vector(sig,plan)-_start)[None,:])[0])
                        t0=time.perf_counter()
                        answer=old.cap_search(start,
                            lambda sig,rng:_adapter_gen(adapter,sig,rng,args),
                            pred,random.Random(seed),
                            mode='random' if policy=='random_beam' else 'guided',
                            beam_width=args.beam_width,max_depth=args.max_depth,
                            budget=args.budget,per_node_cap=args.per_expand)
                        secs=time.perf_counter()-t0
                        pick=answer['pick'];lookup=held[objective]
                        base=float(lookup.get((circ,start),np.nan));cost=float(lookup.get((circ,pick),np.nan))
                        gain=100*(base-cost)/base if np.isfinite(base) and np.isfinite(cost) and base>0 else np.nan
                        logs.append(dict(objective=objective,policy=policy,arm=arm,
                            circuit=circ,source_plan_id=sid,source_sig=start,pick_sig=pick,
                            actual_gain_pct=gain,predicted_gain_pct=answer['predicted_gain_pct'],
                            base_cost=base,pick_cost=cost,evaluated=int(np.isfinite(gain)),
                            harm=int(np.isfinite(gain) and gain<0),n_scored=answer['n_scored'],
                            max_depth_reached=answer['max_depth_reached'],search_sec=secs,search_seed=seed))
                    if args.progress:
                        print(f'  {circ}/{sid} {policy}: '
                              + ' / '.join(f'{x["arm"]} {x["actual_gain_pct"]:+.3f}%'
                                            for x in logs[-2:]),flush=True)
                # Checkpoint a COMPLETE (source,objective) only (4 rows).
                atomic_csv(pd.DataFrame(logs),out/'per_source_checkpoint.csv')
                done.update(required)
    generate_outputs(logs,out)
    manifest={'kind':'historical old-style 35-seed LOCO comparison (not independent final test)',
              'changed_factor':'add depth-2..5 training pairs only',
              'old_training':'original B-pool one-step pairs from original 4C1 build_training',
              'deep_training':'complete 35-seed cached/routed depth2..5, no original B duplicates',
              'heldout':'circuit excluded from fit; route 30-seed heldout lookup only after plan selection',
              'same_search_seed':'historical adaptive_repair_cached.py formula',
              'train_deep_pair_count':len(pd.read_csv(args.deep_pairs_file)),
              'note':'Deep sampling may be biased by cache coverage; report coverage; not a final heldout-family result',
              'settings':{k:str(v) for k,v in vars(args).items()}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if args.reference_old:
        ref=pd.read_csv(args.reference_old,dtype={'source_plan_id':str})
        prior=ref[ref.policy.isin(POLICIES) & ref.objective.isin(args.objectives.split(','))]
        current=pd.DataFrame(logs);cur=current[current.arm=='old_one_step']
        merged=cur.merge(prior[['objective','policy','circuit','source_plan_id','pick_sig']],
            on=['objective','policy','circuit','source_plan_id'],suffixes=('_new','_old'))
        print('REFERENCE CHECK: matched rows',len(merged),'of',len(cur),
              'same historical selected plan',int((merged.pick_sig_new==merged.pick_sig_old).sum()),flush=True)
        if len(merged)!=len(cur):print('WARNING: incomplete matching of old reference rows',flush=True)
    print('Saved',out.resolve(),flush=True)


def _adapter_gen(adapter,sig,rng,args):
    return adapter.generate(sig,rng,args.ops.split(','),
                            per_expand=args.per_expand,max_attempts=args.max_attempts_per_expand)


def self_test():
    with tempfile.TemporaryDirectory() as d:
        out=Path(d)
        logs=[]
        for arm,gain in [('old_one_step',1.),('old_plus_deep',2.)]:
            logs.append(dict(objective='swap_count',policy='adaptive_beam',arm=arm,
                             circuit='a',source_plan_id='P0',actual_gain_pct=gain,
                             evaluated=1,harm=0,n_scored=4))
        generate_outputs(logs,out)
        assert (out/'macro_summary.csv').exists()
        df=pd.read_csv(out/'paired_delta.csv')
        assert abs(df.iloc[0].delta_deep_minus_old_pp-1.)<1e-10
        sample = pd.DataFrame([
            {'circuit':'smoke.qasm','sig':'sig1','seed':seed,'status':'ok',
             'swap_count':2.,'routed_depth':4.} for seed in old.HELD
        ])
        costfile = out/'cost_table.csv'
        sample.to_csv(costfile,index=False)
        costs = load_all_held([costfile, out/'deep_routed_cost.csv'])
        assert abs(costs['swap_count'][('smoke.qasm','sig1')] - 2.) < 1e-10
    print('compare_oldstyle_ridge_beam self-test PASSED')


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--c1-dir',type=Path,default=Path('exp4c1_all'))
    ap.add_argument('--space-dir',type=Path,default=Path('space_out'))
    ap.add_argument('--qasm-dir',type=Path,default=Path('benchmarks_compact'))
    ap.add_argument('--feature-file',type=Path,default=Path('exp4d_features_all.csv'))
    ap.add_argument('--deep-data-dir',type=Path,default=Path('oldstyle_ms_data'))
    ap.add_argument('--deep-pairs-file',type=Path,default=None)
    ap.add_argument('--out-dir',type=Path,default=Path('oldstyle_ms_compare'))
    ap.add_argument('--old-model-dir',type=Path,default=Path('adaptive_ridge_models'))
    ap.add_argument('--multi-model-dir',type=Path,default=Path('oldstyle_ms_models'))
    ap.add_argument('--model-mode',choices=('auto','load','retrain'),default='auto')
    ap.add_argument('--objectives',default='swap_count')
    ap.add_argument('--circuits',default='')
    ap.add_argument('--max-sources',type=int,default=0)
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--lams',default='1,10,100,1000')
    ap.add_argument('--beam-width',type=int,default=5)
    ap.add_argument('--max-depth',type=int,default=5)
    ap.add_argument('--budget',type=int,default=120)
    ap.add_argument('--per-expand',type=int,default=40)
    ap.add_argument('--max-attempts-per-expand',type=int,default=500)
    ap.add_argument('--ops',default='suffix_swap,chain_transfer')
    ap.add_argument('--reference-old',type=Path,default=None,
                    help='Optional prior adaptive_out/per_source.csv to audit exact old reproduction')
    ap.add_argument('--progress',action='store_true')
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args()
    if a.self_test:self_test();return
    a.lams=[float(x) for x in a.lams.split(',')]
    if a.deep_pairs_file is None:a.deep_pairs_file=a.deep_data_dir/'deep_pairs_35seed.csv'
    run(a)
if __name__=='__main__':main()
