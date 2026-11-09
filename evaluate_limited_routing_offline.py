#!/usr/bin/env python3
"""Offline, leak-free limited-routing selection evaluation for GidNET reuse repair.

Use existing per-plan routing seed data: seeds 42..46 are discovery/selection only,
100..129 are held-out/evaluation only. Never rank or choose using held-out data.
Candidate pool is frozen from adaptive_traced/candidate_trace.csv.

Test policies:
 - ridge_current: frozen selected plan from per_source_oracle_gap.csv (0 new routes)
 - top{3,5,10}_one_seed: top predicted gain candidates, one DISC routing each
 - random{3,5,10}_one_seed: same budget, random distinct candidates, one DISC routing each
 - repeat_best_5_seed: top Ridge candidate with 5 different DISC seeds (5 routes)

Original baseline routing is assumed cached/shared and NOT counted as a candidate
routing call. All outcomes include no-repair Original fallback based on the
corresponding discovery seed(s).
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

DISC=(42,43,44,45,46)
HELD=tuple(range(100,130))
KEY=['objective','policy','circuit','source_plan_id','source_sig']


def stable_seed(*items):
    data='|'.join(map(str,items)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(data).digest()[:8],'big')


def read_cost(files):
    frames=[]
    needed={'circuit','sig','seed','swap_count'}
    for p in files:
        path=Path(p)
        if not path.is_file():
            raise FileNotFoundError(f'Missing routing cost table: {path}')
        v=pd.read_csv(path)
        if not needed.issubset(v.columns):
            raise ValueError(f'{path}: missing {sorted(needed-set(v.columns))}')
        if 'status' in v: v=v[v.status.eq('ok')]
        v=v[v.seed.isin(DISC+HELD)]
        frames.append(v[['circuit','sig','seed','swap_count']])
    x=pd.concat(frames,ignore_index=True)
    if x.swap_count.isna().any():
        raise ValueError('NA swap_count in status=ok rows')
    mismatch=x.groupby(['circuit','sig','seed']).swap_count.nunique()
    if (mismatch>1).any():
        raise ValueError(f'Contradictory duplicate route costs: {(mismatch>1).sum()}')
    x=x.drop_duplicates(['circuit','sig','seed'])
    return x.pivot(index=['circuit','sig'],columns='seed',values='swap_count')


def macro(frame):
    # one vote per circuit; trials and sources averaged *within* circuit
    per_circuit=(frame.groupby(['method','budget','circuit'],as_index=False)
                 .agg(gain_pct=('heldout_gain_pct','mean'),
                      harm_rate=('heldout_harm','mean'),
                      no_repair_rate=('is_original','mean'),
                      mean_calls=('candidate_calls','mean'),
                      n_decisions=('heldout_gain_pct','size'),
                      n_sources=('source_plan_id','nunique')))
    summary=(per_circuit.groupby(['method','budget'],as_index=False)
             .agg(n_circuits=('circuit','size'),
                  gain_pct=('gain_pct','mean'),
                  harm_rate=('harm_rate','mean'),
                  no_repair_rate=('no_repair_rate','mean'),
                  mean_calls=('mean_calls','mean'),
                  n_decisions=('n_decisions','sum')))
    return per_circuit,summary


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trace',type=Path,default=Path('adaptive_traced/candidate_trace.csv'))
    ap.add_argument('--gap',type=Path,default=Path('beam_selection_gap/per_source_oracle_gap.csv'))
    ap.add_argument('--c1-cost',type=Path,default=Path('exp4c1_all/cost_table.csv'))
    ap.add_argument('--space-cost',type=Path,default=Path('space_out/cost_table.csv'))
    ap.add_argument('--out-dir',type=Path,default=Path('limited_routing_offline'))
    ap.add_argument('--budgets',type=int,nargs='+',default=[3,5,10])
    ap.add_argument('--trials',type=int,default=5,help='repeat evaluations with rotated discovery seed and random subset')
    ap.add_argument('--allow-partial',action='store_true',help='use only complete sources if routing costs are missing; for development diagnostics only')
    args=ap.parse_args()
    args.out_dir.mkdir(exist_ok=True,parents=True)
    if args.trials<1 or any(k<1 for k in args.budgets):raise ValueError('Positive trials and budgets required')

    candidates=pd.read_csv(args.trace)
    gap=pd.read_csv(args.gap)
    required={'objective','policy','circuit','source_plan_id','source_sig','candidate_sig','predicted_gain_pct'}
    if not required.issubset(candidates.columns):raise ValueError('Wrong candidate trace format')
    candidates=candidates[(candidates.objective=='swap_count')&(candidates.policy=='random_beam')].copy()
    gap=gap[(gap.objective=='swap_count')&(gap.policy=='random_beam')].copy()
    if gap.duplicated(KEY).any():raise ValueError('Duplicate source keys in gap')
    candidates=candidates.drop_duplicates(KEY+['candidate_sig'])
    by_source={key:x for key,x in candidates.groupby(KEY)}
    routing=read_cost([args.c1_cost,args.space_cost])
    cost=routing.to_dict('index') # (circuit,sig) -> {seed: cost}
    # Validate signed heldout labels of original to detect mismatched experiment runs.
    missing=[]; records=[]; source_counts={}
    for row in gap.itertuples(index=False):
        key=tuple(getattr(row,k) for k in KEY)
        g=by_source.get(key)
        if g is None:
            missing.append({'circuit':row.circuit,'source_plan_id':row.source_plan_id,'reason':'candidate_trace missing source'})
            continue
        g=(g[g.candidate_sig.ne(row.source_sig)]
           .sort_values(['predicted_gain_pct','candidate_sig'],ascending=[False,True]).copy())
        g=g[g.predicted_gain_pct.notna()]
        if g.empty:
            missing.append({'circuit':row.circuit,'source_plan_id':row.source_plan_id,'reason':'no candidate'})
            continue
        original=cost.get((row.circuit,row.source_sig))
        if original is None or any(pd.isna(original.get(s)) for s in DISC+HELD):
            missing.append({'circuit':row.circuit,'source_plan_id':row.source_plan_id,'reason':'original lacks 35 routing seeds'})
            continue
        baseheld=float(np.mean([original[s] for s in HELD]))
        if baseheld<=0:
            missing.append({'circuit':row.circuit,'source_plan_id':row.source_plan_id,'reason':'zero original SWAP baseline'})
            continue
        if abs(baseheld-float(row.original_heldout_cost))>1e-6:
            raise ValueError(f'Baseline cost disagreement for {row.circuit}/{row.source_plan_id}: {baseheld} vs {row.original_heldout_cost}; check datasets!')
        # Every candidate requested by a policy must have all 35 routings: otherwise do
        # not silently censor candidates, since this biases policy comparisons.
        kmax=max(args.budgets)
        needed=set(g.head(kmax).candidate_sig)
        random_pool=g.candidate_sig.to_list()
        # Random policy may sample any candidate from entire source pool.
        needed.update(random_pool)
        incomplete=[s for s in needed if (row.circuit,s) not in cost or
                   any(pd.isna(cost[(row.circuit,s)].get(seed)) for seed in DISC+HELD)]
        if incomplete:
            missing.append({'circuit':row.circuit,'source_plan_id':row.source_plan_id,'reason':f'{len(incomplete)}/{len(needed)} candidates lack complete 35-seed costs'})
            continue
        source_counts.setdefault(row.circuit,0)
        source_counts[row.circuit]+=1
        def append(method,budget,trial,chosen,calls):
            is_original=chosen==row.source_sig
            h=float(np.mean([cost[(row.circuit,chosen)][s] for s in HELD]))
            gain=(baseheld-h)/baseheld*100
            records.append({'objective':'swap_count','policy':'random_beam','method':method,'budget':budget,
                'trial':trial,'circuit':row.circuit,'source_plan_id':row.source_plan_id,
                'source_sig':row.source_sig,'chosen_sig':chosen,'candidate_calls':calls,
                'is_original':int(is_original),'heldout_gain_pct':gain,
                'heldout_harm':int(gain<0),'baseline_heldout_cost':baseheld,'chosen_heldout_cost':h})
        # Existing method was selected without using 30-heldout labels.
        # Use its already recorded signature; do not use oracle gain as a selector.
        append('ridge_current',0,0,row.selected_sig,0)
        ranked=g.candidate_sig.to_list()
        for trial in range(args.trials):
            rng=np.random.default_rng(stable_seed(*key,trial,'random'))
            for k in args.budgets:
                # Score K candidate routes with one discovery seed per candidate.
                top=ranked[:k]
                randoms=[ranked[i] for i in rng.choice(len(ranked),size=min(k,len(ranked)),replace=False)]
                for method,pool in [(f'top{k}_1seed',top),(f'random{k}_1seed',randoms)]:
                    # Candidate j evaluated on seed rotating with trial; all
                    # discovery seeds distinct where possible (5 available).
                    measured=[(float(cost[(row.circuit,s)][DISC[(trial+j)%len(DISC)]]),j,s)
                              for j,s in enumerate(pool)]
                    val,j,chosen=min(measured,key=lambda a:(a[0],a[1]))
                    # Compare to Original on same seed as winning candidate.
                    seed=DISC[(trial+j)%len(DISC)]
                    if val>=float(original[seed]): chosen=row.source_sig
                    append(method,k,trial,chosen,len(pool))
            # Five repeated routing seeds for ONE best-ranked candidate.
            s=ranked[0]
            m=float(np.mean([cost[(row.circuit,s)][v] for v in DISC]))
            b=float(np.mean([original[v] for v in DISC]))
            append('repeat_best_5seed',5,trial,s if m<b else row.source_sig,5)

    missing_df=pd.DataFrame(missing,columns=['circuit','source_plan_id','reason'])
    missing_df.to_csv(args.out_dir/'missing_source_coverage.csv',index=False)
    if missing and not args.allow_partial:
        print(f'INCOMPLETE: {len(missing)} sources lack compatible 35-seed data. See missing_source_coverage.csv')
        print('NO EVALUATION produced: add matching cost tables or use --allow-partial for diagnostic only.')
        print('First examples:\n',missing_df.head(12).to_string(index=False))
        raise SystemExit(2)
    if not records:raise SystemExit('No complete sources available.')
    df=pd.DataFrame(records)
    df.to_csv(args.out_dir/'per_source_limited_routing.csv',index=False)
    per_circuit, summary=macro(df)
    per_circuit.to_csv(args.out_dir/'per_circuit_summary.csv',index=False)
    summary.to_csv(args.out_dir/'macro_summary.csv',index=False)
    (args.out_dir/'manifest.json').write_text(json.dumps({
        'source':'random_beam candidates from a frozen, previously completed search',
        'selection_seeds':DISC,'heldout_seeds':HELD,
        'routing_budget':'extra candidate routings, Original baseline assumed previously routed',
        'methodology':'one discovery seed per plan; first K by Ridge prediction vs K random plans; 5-seed repeat',
        'n_missing_sources':len(missing),'partial':bool(missing),
        'caveat':'15 circuits reused in method development. Do not call untouched test; final evaluation must use external/new circuit families.',
        'n_source_records':len(df[['circuit','source_plan_id']].drop_duplicates()),
    },ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nOFFLINE Limited Routing results (macro mean: equal per circuit)')
    print(summary.sort_values(['budget','method']).round(3).to_string(index=False))
    print('\nCoverage:',len(df[['circuit','source_plan_id']].drop_duplicates()),'sources /',len(gap),'expected;',len(missing),'incomplete')
    print('Files written to:',args.out_dir.resolve())
    if missing:print('WARNING: PARTIAL-SOURCE RESULT IS NOT PAPER QUALITY.')

if __name__=='__main__':main()
