#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare 1-step-vs-multistep-trained Ridge INSIDE the same fixed-budget beam search.

Run from the original GidNET root, with this script alongside
adaptive_repair_cached.py and train_multistep_ridge_25.py.

Inputs:
 - ml_train_prepare_v1/{selected_candidates,labels,features,compiled_features}.csv
 - ml_split_v1/split_ledger.csv
 - exp4c1_all/{catalog,pools,features,cost_table}.csv
 - space_out/cost_table.csv
 - exp4d_features_all.csv and benchmarks_compact/*.qasm

Scientific protocol:
 * Refit BOTH Ridge arms on the SAME frozen training data; exclude the entire
   evaluation circuit FAMILY from fit, even when it occurs in the train split.
 * One-step arm trains only move depth=1; multi-step arm trains depths 1..5.
 * Both score the same 25-feature ordered vector, use the same operators and
   beam budgets; final selected cost is looked up after Search, NEVER as a score.
 * This is an exploratory same-search comparison on the old 15-circuit cache,
   not a newly independent final-test or a direct numerical reproduction of
   old 2.76/2.94% models (which had different training data and protocols).
 * Test normalization uses the old unlabeled 4C1 catalog, whereas the new
   trainer used a unique TRAIN candidate catalog: distribution shift possible.
 * Both search arms are treated identically under this limitation.

No Qiskit/routing is needed for --self-test. Real run needs original GidNET env.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import adaptive_repair_cached as baseline
import train_multistep_ridge_25 as multi

ARMS=("one_step","multi_step")
POLICIES=("adaptive_beam","random_beam")


def stable_seed(seed, circuit, source_id, policy, repeat):
    label = f"{seed}|{circuit}|{source_id}|{policy}|{repeat}"
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "big")


def resolve_family(c, ledger, train, optional_map, *, allow_unknown=False):
    # old 4C1 uses plain filename; frozen ledger uses 'CaQR::filename.qasm'.
    hits=ledger[ledger.circuit_id.astype(str).map(lambda s:s.endswith('::'+c) or s==c)]
    if len(hits)>1:
        caqr=hits[hits.circuit_id.astype(str).str.startswith('CaQR::')]
        if len(caqr)==1: hits=caqr
    if len(hits)==1:
        return str(hits.iloc[0].family_group), str(hits.iloc[0].split), str(hits.iloc[0].circuit_id)
    if c in optional_map:
        return optional_map[c], 'manual_family_map', c
    if not allow_unknown:
        raise ValueError(f"Could not uniquely map {c} to frozen split ledger. "
                         "Provide --family-map (CSV columns: circuit,family_group); "
                         "do not assume an unseen family while claiming held-out validation.")
    # Strictly exploratory fallback: exact filename must be absent from all train data.
    if train.circuit_id.astype(str).map(lambda s: s.endswith('::'+c) or s==c).any():
        raise ValueError(f"Unknown family for {c}, but same circuit exists in TRAIN: leakage risk")
    return '__unknown__'+c, 'unknown_exploratory', c


def make_family_map(path):
    if not path:
        return {}
    t=pd.read_csv(path)
    need={'circuit','family_group'}
    if not need.issubset(t.columns) or t.circuit.duplicated().any():
        raise ValueError('Family map must have unique circuit,family_group rows')
    return dict(zip(t.circuit.astype(str), t.family_group.astype(str)))


def fit_arms(data, cols, objective, eval_family, alpha):
    tr=data[(data.family_group != eval_family) & (data.is_original.astype(int)==0)].copy()
    tr=tr[np.isfinite(tr[objective+'_gain_pct'])].copy()
    if objective=='swap_count':
        tr=tr[tr.baseline_swap>0].copy()
    output={}
    for arm in ARMS:
        a=tr[tr.discovery_move_depth==1] if arm=='one_step' else tr[tr.discovery_move_depth.between(1,5)]
        if a.empty or a.family_group.nunique()<2:
            raise ValueError(f"Not enough other-family training examples: {objective}/{eval_family}/{arm}")
        if (a.family_group==eval_family).any():
            raise AssertionError('Leak: target family included in training')
        mdl=multi.fit_model(a,cols,objective,alpha)
        output[arm]=(mdl, len(a), int(a.family_group.nunique()))
    return output


def get_evaluation_sources(c1_dir, circuits, max_sources):
    orig=pd.read_csv(c1_dir/'pools.csv')
    orig=orig[orig.is_original.astype(bool)][['circuit','source_plan_id','sig']]
    orig=orig.drop_duplicates(['circuit','source_plan_id'])
    if circuits:
        wanted=set(s.strip() for s in circuits.split(',') if s.strip())
        missing=wanted-set(orig.circuit)
        if missing: raise ValueError(f"No old 4C1 originals for circuits: {sorted(missing)}")
        orig=orig[orig.circuit.isin(wanted)]
    if max_sources:
        orig=orig.groupby('circuit',sort=False).head(max_sources)
    if orig.empty:
        raise ValueError('No Original Reuse Plans found')
    return orig


def run(args):
    tr,ledger,delta_cols=multi.read_training_data(args.train_dir,args.split_ledger)
    if delta_cols!=['d_'+x for x in multi.FEATURES]:
        raise ValueError('Wrong feature ordering in new Ridge trainer')
    c1=Path(args.c1_dir)
    catalog,Z,stats,allcols=baseline.prepare_features(c1, Path(args.feature_file))
    missing=set(multi.FEATURES)-set(allcols)
    if missing:
        raise ValueError(f'Old GidNET evaluator cannot compute the required 25 features: {sorted(missing)}')
    # Only the 25 columns used by BOTH trained arms, in the canonical order.
    Z=Z.loc[:,multi.FEATURES]
    sources=get_evaluation_sources(c1,args.circuits,args.max_sources)
    fmap=make_family_map(args.family_map)
    if not args.allow_previously_studied_test:
        # Historical 15-circuit evaluation can include frozen FINAL_TEST circuits.
        # Never spend untouched final-test data without explicit approval.
        for c in sources.circuit.unique():
            fam,split,cid=resolve_family(c,ledger,tr,fmap,allow_unknown=args.allow_unknown_family)
            if split in ('test','final_test'):
                raise ValueError(f'{cid} is frozen {split}. Pass --allow-previously-studied-test '
                                 'only for explicitly exploratory / historical comparison.')
    held=baseline.load_heldout([c1/'cost_table.csv',Path(args.space_dir)/'cost_table.csv'])
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    logs=[]
    for objective in args.objectives.split(','):
        objective=objective.strip()
        if objective not in multi.OBJECTIVES:
            raise ValueError(f'Unsupported objective: {objective}')
        print('OBJECTIVE',objective,flush=True)
        for circuit,blocks in sources.groupby('circuit',sort=True):
            fam,split,full_id=resolve_family(circuit,ledger,tr,fmap,allow_unknown=args.allow_unknown_family)
            trained=fit_arms(tr,delta_cols,objective,fam,args.alpha)
            print(f'  CIRCUIT={circuit} family={fam} split={split} sources={len(blocks)}',flush=True)
            for row in blocks.itertuples(index=False):
                origin=row.sig
                for rep in range(args.repeats):
                    for policy in POLICIES:
                        # Same initial RNG seed across both arms for a given policy;
                        # guided path can diverge with model predictions (as intended).
                        seed=stable_seed(args.seed,circuit,str(row.source_plan_id),policy,rep)
                        for arm in ARMS:
                            # Fresh adapter for EVERY run: no cross-policy cache advantage.
                            setup_start=time.perf_counter()
                            adapter=baseline.RepositoryAdapter(
                                circuit,catalog,Path(args.qasm_dir),multi.FEATURES,stats,Z)
                            if (circuit,origin) not in adapter.plan_of:
                                raise ValueError(f'Original missing: {circuit}/{row.source_plan_id}/{origin}')
                            origin_v=adapter.vector(origin,adapter.plan_of[(circuit,origin)])
                            setup_secs=time.perf_counter()-setup_start
                            model,ntrain,nfamilies=trained[arm]
                            def predict(sig,plan,_m=model,_o=origin_v):
                                d=adapter.vector(sig,plan)-_o
                                z=(d-_m['feature_mean'])/_m['feature_sd']
                                return float(z@_m['weights']+_m['intercept'])
                            counters_before=dict(adapter.counters)
                            start=time.perf_counter()
                            result=baseline.cap_search(
                                origin,
                                lambda sig,rng:adapter.generate(
                                    sig,rng,args.ops.split(','),per_expand=args.per_expand,
                                    max_attempts=args.max_attempts_per_expand),
                                predict,random.Random(seed),
                                mode='guided' if policy=='adaptive_beam' else 'random',
                                beam_width=args.beam_width,max_depth=args.max_depth,
                                budget=args.budget,per_node_cap=args.per_expand)
                            secs=time.perf_counter()-start
                            chosen=result['pick']
                            base=float(held[objective].get((circuit,origin),np.nan))
                            cost=float(held[objective].get((circuit,chosen),np.nan))
                            if np.isfinite(base) and base>0 and np.isfinite(cost):
                                gain=100*(base-cost)/base
                            else: gain=float('nan')
                            logs.append(dict(objective=objective,arm=arm,policy=policy,
                                circuit=circuit,family_group=fam,frozen_split=split,
                                original_full_id=full_id,source_plan_id=row.source_plan_id,
                                source_sig=origin,pick_sig=chosen,repeat=rep,search_seed=seed,
                                n_train_pairs=ntrain,n_train_families=nfamilies,
                                predicted_gain_pct=result['predicted_gain_pct'],
                                actual_gain_pct=gain,original_mean_cost=base,pick_mean_cost=cost,
                                n_scored=result['n_scored'],n_seen=result['n_seen'],
                                max_depth_reached=result['max_depth_reached'],
                                generated=int(adapter.counters['generated'])-counters_before.get('generated',0),
                                checker_calls=int(adapter.counters['checker_calls'])-counters_before.get('checker_calls',0),
                                rejected=int(adapter.counters['rejected'])-counters_before.get('rejected',0),
                                search_time_sec=secs,adapter_setup_sec=setup_secs,
                                search_plus_setup_sec=secs+setup_secs,moved=int(chosen!=origin),
                                harm=int(np.isfinite(gain) and gain<0),
                                no_repair=int(chosen==origin),
                                evaluated=int(np.isfinite(gain))))
                        if args.progress:
                            print(f'  {circuit} {row.source_plan_id} {policy} rep={rep}: '
                                  +'/'.join(f'{x["arm"]}:{x["actual_gain_pct"]:.3f}%'
                                            for x in logs[-2:]),flush=True)
    result_df=pd.DataFrame(logs)
    result_df.to_csv(out/'per_source.csv',index=False)
    # Report NA and zero-base explicitly; never silently treat them as zero gain.
    by_circuit=result_df.groupby(['objective','arm','policy','circuit'],as_index=False).agg(
        gain_pct=('actual_gain_pct','mean'),n_eval=('evaluated','sum'),
        n_sources=('source_plan_id','nunique'),harm_rate=('harm','mean'),
        no_repair_rate=('no_repair','mean'),mean_scored=('n_scored','mean'),
        mean_time_sec=('search_time_sec','mean'))
    by_circuit.to_csv(out/'per_circuit.csv',index=False)
    summary=by_circuit.groupby(['objective','arm','policy'],as_index=False).agg(
        gain_pct_equal_circuit=('gain_pct','mean'),n_circuits=('circuit','nunique'),
        n_circuits_evaluated=('gain_pct','count'),mean_scored=('mean_scored','mean'),
        mean_search_time_sec=('mean_time_sec','mean'))
    summary.to_csv(out/'macro_summary.csv',index=False)
    pairs=result_df.pivot(index=['objective','policy','circuit','source_plan_id','repeat'],
                          columns='arm',values='actual_gain_pct')
    pairs=pairs.reset_index()
    pairs['delta_multi_minus_one_pp']=pairs['multi_step']-pairs['one_step']
    pairs.to_csv(out/'paired_delta.csv',index=False)
    manifest=dict(status='EXPLORATORY: older 15-circuit cache, not final independent evaluation',
        n_training_circuits=tr.circuit_id.nunique(),n_training_families=tr.family_group.nunique(),
        features=multi.FEATURES,alpha=args.alpha,
        train_scheme='exclude entire eval circuit family for BOTH arms; same train data',
        eval_test_feature_normalization='unlabeled 4C1 catalog per circuit (different candidate pool from trainer)',
        train_arm_one_step='depth 1',train_arm_multi_step='depth 1..5',
        routing_cost='cached original/selected 30 seeds 100..129, read after selection',
        comparability='Compare two arms against each other here; old 2.76/2.94 historical training differs',
        params={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        n_missing_rows=int(result_df.evaluated.eq(0).sum()))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\n'+summary.to_string(index=False),flush=True)
    if (result_df.evaluated==0).any():
        print('WARNING: Missing or undefined held-out costs: see per_source.csv; summary is incomplete.',flush=True)
    print(f'\nSaved {out.resolve()}',flush=True)


def self_test():
    baseline.self_test()
    multi.self_test()
    assert stable_seed(7,'c','p','adaptive_beam',0)==stable_seed(7,'c','p','adaptive_beam',0)
    assert stable_seed(7,'c','p','adaptive_beam',0)!=stable_seed(7,'c','p','random_beam',0)
    print('runner self-test passed')


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--train-dir',type=Path,default=Path('ml_train_prepare_v1'))
    ap.add_argument('--split-ledger',type=Path,default=Path('ml_split_v1/split_ledger.csv'))
    ap.add_argument('--c1-dir',type=Path,default=Path('exp4c1_all'))
    ap.add_argument('--space-dir',type=Path,default=Path('space_out'))
    ap.add_argument('--feature-file',type=Path,default=Path('exp4d_features_all.csv'))
    ap.add_argument('--qasm-dir',type=Path,default=Path('benchmarks_compact'))
    ap.add_argument('--out-dir',type=Path,default=Path('adaptive_ms25_beam_compare'))
    ap.add_argument('--circuits',default='')
    ap.add_argument('--max-sources',type=int,default=0)
    ap.add_argument('--objectives',default='swap_count',help='swap_count or swap_count,routed_depth')
    ap.add_argument('--alpha',type=float,default=10.)
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--repeats',type=int,default=1)
    ap.add_argument('--beam-width',type=int,default=5)
    ap.add_argument('--max-depth',type=int,default=5)
    ap.add_argument('--budget',type=int,default=120)
    ap.add_argument('--per-expand',type=int,default=40)
    ap.add_argument('--max-attempts-per-expand',type=int,default=500)
    ap.add_argument('--ops',default='suffix_swap,chain_transfer')
    ap.add_argument('--family-map',type=Path,default=None)
    ap.add_argument('--allow-unknown-family',action='store_true',
                    help='EXPLORATORY ONLY: unlisted eval circuit must not occur in train')
    ap.add_argument('--allow-previously-studied-test',action='store_true',
                    help='EXPLORATORY ONLY: permit frozen final_test in historical 15 (not pristine)')
    ap.add_argument('--progress',action='store_true')
    args=ap.parse_args()
    if args.self_test: self_test();return
    if args.repeats<1 or args.max_sources<0 or args.alpha<0: ap.error('Invalid repeats/sources/alpha')
    for p in [args.c1_dir/'catalog.csv',args.c1_dir/'pools.csv',args.c1_dir/'features.csv',
              args.c1_dir/'cost_table.csv',args.space_dir/'cost_table.csv',
              args.feature_file,args.split_ledger,
              args.train_dir/'selected_candidates.csv',args.train_dir/'labels.csv',
              args.train_dir/'features.csv',args.train_dir/'compiled_features.csv']:
        if not p.is_file(): ap.error('Missing input: '+str(p))
    if not args.qasm_dir.is_dir(): ap.error('Missing QASM folder: '+str(args.qasm_dir))
    run(args)

if __name__=='__main__': main()
