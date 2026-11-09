#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build depth-2..5 labeled repair examples compatible with the ORIGINAL 4C1 Ridge.

Stage discover: cost-blind bounded randomized traversal of valid reuse plans;
                resumable per original source. NO ROUTING.
Stage status:   audit already-routed 35-seed coverage and missing plan count.
Stage route:    optionally route a bounded number of missing distinct plans, with
                exactly the OLD router's 35 seeds; resumable per plan.
Stage features: make deep training pair rows using ORIGINAL catalog normalizer;
                B-pool 1-step pairs remain untouched and are added at fit time.

Requires GidNET originals and Qiskit 0.45.3 for discover/features/route;
status/self-test works without Qiskit. Designed for small controlled trials.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import adaptive_repair_cached as old

SEEDS = old.DISC + old.HELD


def key_seed(*x):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, x)).encode()).digest()[:8], 'big')


def atomic_csv(df, path):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    df.to_csv(temp,index=False)
    os.replace(temp,path)


def origins(c1, circuits='', max_sources=0):
    o=pd.read_csv(Path(c1)/'pools.csv')
    o=o[o.is_original.astype(bool)][['circuit','source_plan_id','sig']].drop_duplicates(['circuit','source_plan_id'])
    if circuits:
        want={s.strip() for s in circuits.split(',') if s.strip()}
        if not want.issubset(set(o.circuit)):
            raise ValueError(f'Unknown original circuit(s): {want-set(o.circuit)}')
        o=o[o.circuit.isin(want)]
    if max_sources:
        o=o.groupby('circuit',sort=False).head(max_sources)
    if o.empty: raise ValueError('No sources to process')
    return o.sort_values(['circuit','source_plan_id']).reset_index(drop=True)


def cost_means(paths, objective):
    # Merge costs only for identical circuit/signature/seed; prefer old 4C1,
    # then old exact-space cache, then independently generated deep-route file.
    frames=[]
    for p in paths:
        if Path(p).exists():
            a=pd.read_csv(p)
            necessary={'circuit','sig','seed','status',objective}
            if not necessary.issubset(a.columns):
                raise ValueError(f'{p} missing columns: {necessary-set(a.columns)}')
            frames.append(a[a.seed.isin(SEEDS)])
    if not frames: raise FileNotFoundError('No cost CSVs found')
    costs=pd.concat(frames,ignore_index=True)
    costs=costs[costs.status.eq('ok') & np.isfinite(pd.to_numeric(costs[objective],errors='coerce'))]
    costs=costs.drop_duplicates(['circuit','sig','seed'],keep='first')
    p=costs.pivot_table(index=['circuit','sig'],columns='seed',values=objective,aggfunc='first')
    good=p.reindex(columns=SEEDS).dropna(subset=SEEDS)
    return good[SEEDS].mean(axis=1), len(costs)


def load_artifacts(out):
    r=Path(out)/'reach.csv'; c=Path(out)/'catalog5.csv'
    if r.exists()!=c.exists(): raise RuntimeError('Only one of reach/catalog5 exists: do not resume inconsistent files')
    if not r.exists():
        return pd.DataFrame(columns=['circuit','source_plan_id','source_sig','sig','discovery_depth']),pd.DataFrame(columns=['circuit','sig','plan_json'])
    reach=pd.read_csv(r,dtype={'source_plan_id':str})
    cat=pd.read_csv(c)
    if reach.duplicated(['circuit','source_plan_id','sig']).any() or cat.duplicated(['circuit','sig']).any():
        raise RuntimeError('Corrupt duplicated cache rows')
    return reach,cat


def discover(args):
    orig=origins(args.c1_dir,args.circuits,args.max_sources)
    out=args.out_dir
    out.mkdir(parents=True,exist_ok=True)
    fp=out/'discover_config.json'
    cfg={'c1':str(args.c1_dir.resolve()),'features':str(args.feature_file.resolve()),
         'circuits':args.circuits,'max_sources':args.max_sources,'seed':args.seed,
         'max_depth':args.max_depth,'beam_width':args.beam_width,'budget':args.budget,
         'per_expand':args.per_expand,'max_attempts_per_expand':args.max_attempts_per_expand,
         'ops':args.ops}
    if fp.exists():
        if json.loads(fp.read_text())!=cfg: raise ValueError('Discovery settings changed; choose a new --out-dir')
        if not args.resume: raise RuntimeError('Existing discovery: add --resume, or choose new --out-dir')
    else:
        if (out/'reach.csv').exists(): raise RuntimeError('Existing reach.csv without config; choose new --out-dir')
        fp.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    reach,cat=load_artifacts(out)
    catalog,Z,stats,cols=old.prepare_features(args.c1_dir,args.feature_file)
    completed={(str(a),str(b)) for a,b in zip(reach.circuit,reach.source_plan_id)}
    total=len(orig)
    for circuit,group in orig.groupby('circuit',sort=True):
        if all((circuit,str(r.source_plan_id)) in completed for r in group.itertuples(index=False)):
            continue
        adapter=old.RepositoryAdapter(circuit,catalog,args.qasm_dir,cols,stats,Z)
        for source in group.itertuples(index=False):
            sid=str(source.source_plan_id)
            if (circuit,sid) in completed:
                print('RESUME SKIP',circuit,sid,flush=True);continue
            start=str(source.sig)
            seen={start};frontier=[start];found=[]
            rng=random.Random(key_seed(args.seed,circuit,sid))
            # Reserve an approximately equal sampling quota for EACH depth.
            # Without this, the first 1-2 layers can exhaust the entire budget,
            # making a nominal 'multistep' training set contain no deep plans.
            quotas=[args.budget//args.max_depth]*args.max_depth
            for qi in range(args.budget%args.max_depth): quotas[qi]+=1
            for depth in range(1,args.max_depth+1):
                layer=[];quota=quotas[depth-1]
                for here in frontier:
                    if len(layer)>=quota:break
                    for sig,plan in adapter.generate(here,rng,args.ops.split(','),
                                         per_expand=min(args.per_expand,quota),
                                         max_attempts=args.max_attempts_per_expand):
                        if sig in seen: continue
                        seen.add(sig)
                        found.append((sig,depth,plan))
                        layer.append(sig)
                        if len(layer)>=quota:break
                if not layer:break
                rng.shuffle(layer)
                frontier=layer[:args.beam_width]
            newreach=[dict(circuit=circuit,source_plan_id=sid,source_sig=start,sig=sig,
                           discovery_depth=depth) for sig,depth,_ in found]
            # Also a completion marker for zero-neighbor sources.
            newreach.append(dict(circuit=circuit,source_plan_id=sid,source_sig=start,
                                 sig=start,discovery_depth=0))
            newcat=[dict(circuit=circuit,sig=sig,
                         plan_json=json.dumps([list(z) for z in plan]))
                    for sig,depth,plan in found]
            reach=pd.concat([reach,pd.DataFrame(newreach)],ignore_index=True)
            cat=pd.concat([cat,pd.DataFrame(newcat)],ignore_index=True).drop_duplicates(['circuit','sig'],keep='first')
            atomic_csv(cat,out/'catalog5.csv')
            atomic_csv(reach,out/'reach.csv')
            completed.add((circuit,sid))
            depth_hist={d:sum(x[1]==d for x in found) for d in range(1,args.max_depth+1)}
            print(f'DISCOVER {len(completed)}/{total} {circuit} {sid}: {len(found)} plans {depth_hist} [checkpoint]',flush=True)
    print('Discovery saved:',out.resolve(),flush=True)
    status(args)


def status(args):
    reach,cat=load_artifacts(args.out_dir)
    if reach.empty:
        print('No reach.csv. Run --stage discover first.');return
    deep=reach[reach.discovery_depth.between(2,5)]
    print(f'SOURCES={reach[["circuit","source_plan_id"]].drop_duplicates().shape[0]} '
          f'DISCOVERED_DEPTH_2_TO_5_PAIRS={len(deep)} DISTINCT={deep[["circuit","sig"]].drop_duplicates().shape[0]}')
    paths=[args.c1_dir/'cost_table.csv',args.space_dir/'cost_table.csv',args.out_dir/'deep_routed_cost.csv']
    for obj in args.objectives.split(','):
        obj=obj.strip()
        means,_=cost_means(paths,obj)
        cand=deep[['circuit','sig']].drop_duplicates()
        hits=sum((c,s) in means.index for c,s in cand.itertuples(index=False,name=None))
        print(f'{obj}: 35-SEED CACHED DISTINCT={hits}/{len(cand)} missing={len(cand)-hits}')
    if not deep.empty:
        print('DISCOVERY DEPTH COUNTS:',deep.discovery_depth.value_counts().sort_index().to_dict())


def route(args):
    if args.max_new_plans<1:
        raise ValueError('To avoid accidental expensive routing, pass --max-new-plans N (e.g. 10).')
    reach,cat=load_artifacts(args.out_dir)
    if reach.empty:raise ValueError('Run discover first')
    means,_=cost_means([args.c1_dir/'cost_table.csv',args.space_dir/'cost_table.csv',args.out_dir/'deep_routed_cost.csv'],'swap_count')
    available=set(means.index)
    deep=reach[reach.discovery_depth.between(2,5)][['circuit','sig','discovery_depth']].drop_duplicates(['circuit','sig'])
    missing=[(c,s) for c,s,d in deep.itertuples(index=False,name=None) if (c,s) not in available]
    print(f'{len(missing)} missing 35-seed plans; about {len(missing)*35} router calls for ALL.',flush=True)
    if not missing: return
    # Spread a small routing budget across circuits rather than spending it all
    # on the alphabetically first circuit; deterministic, independent of cost.
    buckets=defaultdict(list)
    for c,s,d in deep.itertuples(index=False,name=None):
        if (c,s) not in available:
            buckets[c].append((d,s))
    for c in buckets:
        rr=random.Random(key_seed(args.seed,c,'routing-subset'))
        rr.shuffle(buckets[c])
    ordered=[]
    while len(ordered)<len(missing):
        progressed=False
        for c in sorted(buckets):
            if buckets[c]:
                d,s=buckets[c].pop()
                ordered.append((c,s))
                progressed=True
        if not progressed:break
    missing=ordered
    if not args.router:raise ValueError('--router MODULE:FUNCTION required for route')
    from exp4c1_build_table import parse_chains, route_task
    lookup={(r.circuit,r.sig):r.plan_json for r in cat.itertuples(index=False)}
    n=0
    for circuit,sig in missing:
        if n>=args.max_new_plans:break
        if (circuit,sig) not in lookup:raise ValueError(f'Missing plan_json {circuit}/{sig}')
        rows,_=route_task(circuit,sig,parse_chains(lookup[(circuit,sig)]),SEEDS,
                          str(args.qasm_dir),args.router)
        path=args.out_dir/'deep_routed_cost.csv'
        pd.DataFrame(rows).to_csv(path,mode='a',header=not path.exists(),index=False)
        n+=1
        ok=sum(x['status']=='ok' for x in rows)
        print(f'ROUTE {n}/{min(len(missing),args.max_new_plans)} {circuit}/{sig}: ok={ok}/35 [checkpoint]',flush=True)
    status(args)


def features(args):
    reach,cat=load_artifacts(args.out_dir)
    if reach.empty:raise ValueError('Run discover first')
    oldcatalog,Z,stats,cols=old.prepare_features(args.c1_dir,args.feature_file)
    if len(cols)!=25:
        print(f'WARNING: {len(cols)} old usable features, not 25; order preserved',flush=True)
    origp=origins(args.c1_dir)
    origs={(str(r.circuit),str(r.source_plan_id)):str(r.sig) for r in origp.itertuples(index=False)}
    all_old_B=set()
    pool=pd.read_csv(args.c1_dir/'pools.csv',dtype={'source_plan_id':str})
    for r in pool[(pool.in_B.astype(bool)) & (~pool.is_original.astype(bool))].itertuples(index=False):
        all_old_B.add((str(r.circuit),str(r.source_plan_id),str(r.sig)))
    pairs=reach[reach.discovery_depth.between(2,5)].copy()
    pairs=pairs[~pairs.apply(lambda r:(str(r.circuit),str(r.source_plan_id),str(r.sig)) in all_old_B,axis=1)]
    full_cost={}
    paths=[args.c1_dir/'cost_table.csv',args.space_dir/'cost_table.csv',args.out_dir/'deep_routed_cost.csv']
    for obj in ['swap_count','routed_depth']:
        full_cost[obj],_=cost_means(paths,obj)
    cat_lookup={(r.circuit,r.sig):r.plan_json for r in cat.itertuples(index=False)}
    out_rows=[]
    counts=defaultdict(int)
    for circuit,group in pairs.groupby('circuit',sort=True):
        adapter=None
        for row in group.itertuples(index=False):
            sid=str(row.source_plan_id);key=(circuit,sid)
            orig=origs.get(key)
            if orig is None:raise ValueError(f'Unknown original {key}')
            if (circuit,orig) not in Z.index:
                counts['original_feature_missing']+=1;continue
            gains={};ok=True
            for obj in ['swap_count','routed_depth']:
                mm=full_cost[obj]
                if (circuit,orig) not in mm.index or (circuit,row.sig) not in mm.index:
                    ok=False;break
                base=float(mm[(circuit,orig)]);cand=float(mm[(circuit,row.sig)])
                if not (np.isfinite(base) and base>0 and np.isfinite(cand)):
                    ok=False;break
                gains[obj]=100*(base-cand)/base
            if not ok:
                counts['missing_35_seed_cost']+=1;continue
            if adapter is None:
                adapter=old.RepositoryAdapter(circuit,oldcatalog,args.qasm_dir,cols,stats,Z)
            if (circuit,row.sig) not in cat_lookup:
                raise ValueError(f'No catalog plan for discovered {circuit}/{row.sig}')
            from exp4c1_build_table import parse_chains
            plan=parse_chains(cat_lookup[(circuit,row.sig)])
            # Only signatures obtained through independent validity checks at discover stage.
            vec=adapter.vector(row.sig,plan)
            src_vec=adapter.vector(orig,adapter.plan_of[(circuit,orig)])
            delta=vec-src_vec
            item={'circuit':circuit,'source_plan_id':sid,'source_sig':orig,'sig':row.sig,
                  'discovery_depth':int(row.discovery_depth),
                  'gain_swap_count':gains['swap_count'],'gain_routed_depth':gains['routed_depth']}
            for col,v in zip(cols,delta):item['d_'+col]=float(v)
            out_rows.append(item)
            counts['labeled']+=1
        print('FEATURES',circuit,'cumulative',dict(counts),flush=True)
    output=args.out_dir/'deep_pairs_35seed.csv'
    header=['circuit','source_plan_id','source_sig','sig','discovery_depth',
            'gain_swap_count','gain_routed_depth']+['d_'+c for c in cols]
    atomic_csv(pd.DataFrame(out_rows,columns=header),output)
    metadata={'n_new_pairs':len(out_rows),'counts':dict(counts),'columns':cols,
       'normalization':'original exp4c1_all catalog, same as historical Ridge',
       'labels':'same complete 35 seeds 42..46+100..129 as original Ridge',
       'old_B_untouched':True,
       'warning':'discovery depth is encountered valid path length; not certified minimal graph distance'}
    (args.out_dir/'deep_pairs_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(f'Saved {len(out_rows)} NEW labeled depth 2..5 examples to {output}',flush=True)
    if len(out_rows)==0:
        print('NO DEEP TRAINING EXAMPLES: run status, optionally route bounded missing plans. Do not claim Multi-Step training yet.',flush=True)


def self_test():
    assert key_seed('x',10)==key_seed('x',10)!=key_seed('x',11)
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'x.csv';atomic_csv(pd.DataFrame([{'a':1}]),path)
        assert pd.read_csv(path).iloc[0].a==1
        # Exact 35-seed coverage only, not 34 and not error statuses
        data=[]
        for sig,seeds in [('full',SEEDS),('partial',SEEDS[:-1])]:
            for seed in seeds:data.append(dict(circuit='c',sig=sig,seed=seed,status='ok',swap_count=3.0))
        pd.DataFrame(data).to_csv(Path(d)/'cost.csv',index=False)
        mm,_=cost_means([Path(d)/'cost.csv'],'swap_count')
        assert ('c','full') in mm.index and ('c','partial') not in mm.index
    print('build_oldstyle_multistep self-test PASSED')


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage',choices=['discover','status','route','features','self-test'],required=True)
    ap.add_argument('--c1-dir',type=Path,default=Path('exp4c1_all'))
    ap.add_argument('--space-dir',type=Path,default=Path('space_out'))
    ap.add_argument('--qasm-dir',type=Path,default=Path('benchmarks_compact'))
    ap.add_argument('--feature-file',type=Path,default=Path('exp4d_features_all.csv'))
    ap.add_argument('--out-dir',type=Path,default=Path('oldstyle_ms_data'))
    ap.add_argument('--circuits',default='')
    ap.add_argument('--max-sources',type=int,default=0)
    ap.add_argument('--max-depth',type=int,default=5)
    ap.add_argument('--budget',type=int,default=80)
    ap.add_argument('--beam-width',type=int,default=5)
    ap.add_argument('--per-expand',type=int,default=30)
    ap.add_argument('--max-attempts-per-expand',type=int,default=500)
    ap.add_argument('--ops',default='suffix_swap,chain_transfer')
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--objectives',default='swap_count')
    ap.add_argument('--max-new-plans',type=int,default=0,
                    help='Route at most N missing distinct plans; required on route stage')
    ap.add_argument('--router',default='router_template:route')
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if args.stage=='self-test':self_test();return
    if args.stage=='discover':discover(args)
    elif args.stage=='status':status(args)
    elif args.stage=='route':route(args)
    elif args.stage=='features':features(args)

if __name__=='__main__':main()
