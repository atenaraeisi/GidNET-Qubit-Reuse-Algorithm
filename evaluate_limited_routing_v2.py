#!/usr/bin/env python3
"""Fair-seed offline limited-routing pilot for fixed candidate pools.

A post-hoc DEVELOPMENT diagnostic, not an untouched test set.
- Candidate generation and Ridge prediction are frozen.
- Selection ONLY uses discovery seed costs (42..46).
- Final gain ONLY uses held-out seeds (100..129).
- In every common-seed comparison, all candidate plans see the SAME seed.
- The Original is assumed already routed on discovery seeds and costs zero extra calls.
- Uses no new routing; all costs must be precomputed.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

DISC=(42,43,44,45,46)
HELD=tuple(range(100,130))
KEY=['objective','policy','circuit','source_plan_id','source_sig']

def seed_hash(*items):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,items)).encode()).digest()[:8],'big')

def costs_as_dict(paths):
    pieces=[]
    needed={'circuit','sig','seed','swap_count'}
    for p in paths:
        p=Path(p)
        if not p.is_file(): raise FileNotFoundError(f'Missing cost table: {p}')
        d=pd.read_csv(p)
        if not needed.issubset(d.columns):raise ValueError(f'{p}: missing columns {sorted(needed-set(d.columns))}')
        if 'status' in d.columns:d=d[d.status.eq('ok')]
        d=d[d.seed.isin(DISC+HELD)]
        pieces.append(d[['circuit','sig','seed','swap_count']])
    d=pd.concat(pieces,ignore_index=True)
    if d.swap_count.isna().any():raise ValueError('Missing SWAP count in a successful row')
    dup=d.groupby(['circuit','sig','seed']).swap_count.nunique()
    if (dup>1).any():raise ValueError(f'Contradictory cost rows: {(dup>1).sum()}')
    d=d.drop_duplicates(['circuit','sig','seed'])
    return d.pivot(index=['circuit','sig'],columns='seed',values='swap_count').to_dict('index')

def macro(df):
    pc=(df.groupby(['method','budget','circuit'],as_index=False).agg(
        gain_pct=('heldout_gain_pct','mean'),harm_rate=('heldout_harm','mean'),
        no_repair_rate=('is_original','mean'),mean_calls=('candidate_calls','mean'),
        n_decisions=('heldout_gain_pct','size'),n_sources=('source_plan_id','nunique')))
    sm=(pc.groupby(['method','budget'],as_index=False).agg(n_circuits=('circuit','size'),
        gain_pct=('gain_pct','mean'),harm_rate=('harm_rate','mean'),
        no_repair_rate=('no_repair_rate','mean'),mean_calls=('mean_calls','mean'),
        n_decisions=('n_decisions','sum')))
    return pc,sm

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trace',type=Path,default=Path('adaptive_traced/candidate_trace.csv'))
    ap.add_argument('--gap',type=Path,default=Path('beam_selection_gap/per_source_oracle_gap.csv'))
    ap.add_argument('--c1-cost',type=Path,default=Path('exp4c1_all/cost_table.csv'))
    ap.add_argument('--space-cost',type=Path,default=Path('space_out/cost_table.csv'))
    ap.add_argument('--out-dir',type=Path,default=Path('limited_routing_v2'))
    ap.add_argument('--trials',type=int,default=5)
    ap.add_argument('--allow-partial',action='store_true')
    args=ap.parse_args()
    if args.trials<1:raise ValueError('trials must be positive')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    trace=pd.read_csv(args.trace)
    gap=pd.read_csv(args.gap)
    for d,colset in [(trace,set(KEY+['candidate_sig','predicted_gain_pct'])),(gap,set(KEY+['selected_sig','original_heldout_cost']))]:
        if not colset.issubset(d.columns):raise ValueError(f'Missing {sorted(colset-set(d.columns))}')
    trace=trace[(trace.objective=='swap_count') & (trace.policy=='random_beam')].drop_duplicates(KEY+['candidate_sig'])
    gap=gap[(gap.objective=='swap_count') & (gap.policy=='random_beam')]
    if gap.duplicated(KEY).any():raise ValueError('Duplicate sources in gap')
    by_source={k:v for k,v in trace.groupby(KEY)}
    cost=costs_as_dict([args.c1_cost,args.space_cost])
    out=[];missing=[]
    methods=['ridge_current','top1_common1seed','top1_3seed','top3_common1seed','top5_common1seed',
             'random5_common1seed','top2_2seed','top2_2seed_confirm1','top3_screen2_refine2',
             'repeat_best_5seed']
    for row in gap.itertuples(index=False):
        k=tuple(getattr(row,x) for x in KEY)
        g=by_source.get(k)
        if g is None:
            missing.append((row.circuit,row.source_plan_id,'Missing candidate trace'));continue
        g=g[(g.candidate_sig.ne(row.source_sig))&g.predicted_gain_pct.notna()]
        g=g.sort_values(['predicted_gain_pct','candidate_sig'],ascending=[False,True])
        ranked=g.candidate_sig.tolist()
        if not ranked:
            missing.append((row.circuit,row.source_plan_id,'No candidates'));continue
        orig=cost.get((row.circuit,row.source_sig))
        required=[row.source_sig,row.selected_sig]+ranked
        if any((row.circuit,s) not in cost or any(pd.isna(cost[(row.circuit,s)].get(z,np.nan)) for z in DISC+HELD) for s in set(required)):
            missing.append((row.circuit,row.source_plan_id,'Missing some 35-seed costs'));continue
        base=np.mean([orig[z] for z in HELD])
        if base<=0:
            missing.append((row.circuit,row.source_plan_id,'Original has zero SWAP'));continue
        if not np.isclose(base,float(row.original_heldout_cost),atol=1e-6):
            raise ValueError(f'Original heldout mismatch: {row.circuit} {row.source_plan_id}')
        def route(sig,seed):return float(cost[(row.circuit,sig)][seed])
        def avg(sig,seeds):return float(np.mean([route(sig,z) for z in seeds]))
        def choose(pool,seeds):
            """Compare all candidates against same seed set and original, keeping original as fallback."""
            if not pool:return row.source_sig
            winner=min(pool,key=lambda s:(avg(s,seeds), s))
            return winner if avg(winner,seeds)<avg(row.source_sig,seeds) else row.source_sig
        def record(m,b,t,chosen,calls):
            h=avg(chosen,HELD)
            gain=100*(base-h)/base
            out.append({'objective':'swap_count','policy':'random_beam','method':m,'budget':b,
                'trial':t,'circuit':row.circuit,'source_plan_id':row.source_plan_id,
                'source_sig':row.source_sig,'chosen_sig':chosen,'candidate_calls':calls,
                'is_original':int(chosen==row.source_sig),'heldout_gain_pct':gain,
                'heldout_harm':int(gain<0),'baseline_heldout_cost':base,'chosen_heldout_cost':h})
        record('ridge_current',0,0,row.selected_sig,0)
        for t in range(args.trials):
            s1=DISC[t%5];s2=DISC[(t+1)%5];s3=DISC[(t+2)%5]
            p1=ranked[:1];p2=ranked[:2];p3=ranked[:3];p5=ranked[:5]
            record('top1_common1seed',1,t,choose(p1,[s1]),len(p1))
            record('top1_3seed',3,t,choose(p1,[s1,s2,s3]),3)
            record('top3_common1seed',3,t,choose(p3,[s1]),len(p3))
            record('top5_common1seed',5,t,choose(p5,[s1]),len(p5))
            rng=np.random.default_rng(seed_hash(*k,t,'random_common'))
            rnd=[ranked[i] for i in rng.choice(len(ranked),size=min(5,len(ranked)),replace=False)]
            record('random5_common1seed',5,t,choose(rnd,[s1]),len(rnd))
            record('top2_2seed',4,t,choose(p2,[s1,s2]),2*len(p2))
            # Compare top-2 on two common seeds, then spend an extra call to veto
            # the provisional winner if its 3-seed mean is not better than Original.
            prelim=choose(p2,[s1,s2])
            confirmed=(prelim if prelim==row.source_sig or
                avg(prelim,[s1,s2,s3])<avg(row.source_sig,[s1,s2,s3]) else row.source_sig)
            record('top2_2seed_confirm1',5,t,confirmed,2*len(p2)+(prelim!=row.source_sig))
            # Screen three using common s1. Refine top two screen-ranked using
            # common s2; final choice uses their two-seed mean and Original fallback.
            finalists=sorted(p3,key=lambda s:(route(s,s1),s))[:2]
            refined=choose(finalists,[s1,s2])
            record('top3_screen2_refine2',5,t,refined,len(p3)+len(finalists))
            record('repeat_best_5seed',5,t,choose(p1,DISC),5)
    miss=pd.DataFrame(missing,columns=['circuit','source_plan_id','reason'])
    miss.to_csv(args.out_dir/'missing_source_coverage.csv',index=False)
    if missing and not args.allow_partial:
        raise SystemExit(f'{len(missing)} incomplete sources. No evaluation produced. See {args.out_dir}/missing_source_coverage.csv')
    if not out:raise SystemExit('No fully covered source records')
    result=pd.DataFrame(out)
    result.to_csv(args.out_dir/'per_source_limited_routing_v2.csv',index=False)
    pc,summary=macro(result)
    pc.to_csv(args.out_dir/'per_circuit_summary.csv',index=False)
    summary.to_csv(args.out_dir/'macro_summary.csv',index=False)
    # Paired per-circuit difference vs zero-extra-routing Ridge baseline.
    wide=pc.pivot(index='circuit',columns='method',values='gain_pct')
    comp=[]
    for m in methods:
        if m=='ridge_current':continue
        d=wide[m]-wide['ridge_current']
        comp.append({'method':m,'delta_vs_ridge_pp':d.mean(), 'n_circuits':len(d),
                     'improved_circuits':int((d>1e-10).sum()),
                     'worsened_circuits':int((d<-1e-10).sum()),
                     'tied_circuits':int((abs(d)<=1e-10).sum())})
    pd.DataFrame(comp).sort_values('delta_vs_ridge_pp',ascending=False).to_csv(
        args.out_dir/'paired_vs_ridge.csv',index=False)
    (args.out_dir/'manifest.json').write_text(json.dumps({
        'source':'Frozen previous Random Beam candidates',
        'methods':methods,'selection_seeds':DISC,'heldout_seeds':HELD,
        'trials':args.trials,'complete_sources':len(result[['circuit','source_plan_id']].drop_duplicates()),
        'incomplete_sources':len(missing),'partial':bool(missing),
        'caveats':['The 15 circuits were used in method development and do not form an untouched external test.',
            'Different trials reuse discovery and heldout data; n_decisions is not independent sample size.',
            'Comparisons include an assumed already-routed Original (not charged against extra-routing budget).',
            'This v2 uses same seeds across candidates before comparing their costs.'],
        },indent=2),encoding='utf-8')
    print('\nFAIR-SEED OFFLINE LIMITED ROUTING V2 (macro: equal per circuit)')
    print(summary.sort_values(['gain_pct'],ascending=False).round(3).to_string(index=False))
    print('\nPAIRED vs Ridge (percentage points; + is improvement)')
    print(pd.DataFrame(comp).sort_values('delta_vs_ridge_pp',ascending=False).round(3).to_string(index=False))
    print(f'\nCoverage: {len(result[["circuit","source_plan_id"]].drop_duplicates())}/{len(gap)} sources, incomplete={len(missing)}')
    print('Output:',args.out_dir.resolve())
    if missing: print('WARNING: INCOMPLETE 15-circuit coverage, exploratory only.')

if __name__=='__main__':main()
