#!/usr/bin/env python3
"""Analyze selected-plan Ridge errors vs exact-space structural exchange distance.

Optional traced candidate output adds *observed operator move depth* diagnostics.
All costs/labels are used for POST-HOC evaluation only, never selection.

Run from repository root:
  python analyze_adaptive_errors.py \
    --per-source adaptive_full/per_source.csv \
    --reach space_out/reach.csv \
    --out-dir adaptive_diagnostics

Optional after running the traced search script:
  python analyze_adaptive_errors.py \
    --per-source adaptive_traced/per_source.csv \
    --reach space_out/reach.csv \
    --candidates adaptive_traced/candidate_trace.csv \
    --out-dir adaptive_diagnostics_traced
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ['objective','circuit','source_plan_id']

def grouped(df, group_cols):
    d = df.copy()
    d['error'] = d['predicted_gain_pct'] - d['actual_gain_pct']
    d['abs_error'] = d['error'].abs()
    d['squared_error'] = d['error']**2
    d['is_harmful'] = d['actual_gain_pct'] < 0
    out = d.groupby(group_cols, dropna=False).agg(
        n=('actual_gain_pct','size'),
        circuits=('circuit','nunique'),
        mean_pred_pct=('predicted_gain_pct','mean'),
        mean_actual_pct=('actual_gain_pct','mean'),
        optimism_pp=('error','mean'),
        mae_pp=('abs_error','mean'),
        mse_pp2=('squared_error','mean'),
        harmful_fraction=('is_harmful','mean'),
    ).reset_index()
    out['rmse_pp'] = np.sqrt(out.pop('mse_pp2'))
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--per-source', required=True, type=Path)
    ap.add_argument('--reach', required=True, type=Path)
    ap.add_argument('--candidates', type=Path, default=None)
    ap.add_argument('--out-dir', type=Path, default=Path('adaptive_diagnostics'))
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)

    per=pd.read_csv(args.per_source)
    reach=pd.read_csv(args.reach, usecols=['circuit','source_sig','sig','dist'])
    reach=reach.rename(columns={'sig':'pick_sig','dist':'exchange_distance'})
    if reach.duplicated(['circuit','source_sig','pick_sig']).any():
        raise ValueError('reach.csv has duplicate (circuit, source_sig, sig) keys')
    augmented=per.merge(reach, on=['circuit','source_sig','pick_sig'],how='left',validate='many_to_one')
    if augmented.exchange_distance.isna().any():
        examples=augmented[augmented.exchange_distance.isna()].head(3)
        raise ValueError('Some selected plans absent from reach.csv. Check exact-space dataset and signatures.\n' + examples.to_string(index=False))
    if augmented.heldout_missing.any() or augmented.actual_gain_pct.isna().any():
        raise ValueError('Missing heldout selected-plan labels: diagnose inputs before aggregating')
    augmented.exchange_distance=augmented.exchange_distance.astype(int)
    augmented['signed_error_pp']=augmented.predicted_gain_pct-augmented.actual_gain_pct
    augmented['abs_error_pp']=augmented.signed_error_pp.abs()
    augmented.to_csv(args.out_dir/'selected_with_exchange_distance.csv',index=False)
    by_dist=grouped(augmented,['objective','policy','exchange_distance'])
    by_dist.to_csv(args.out_dir/'selected_error_by_exchange_distance.csv',index=False)

    # Avoid misleading pooled statistics: these macro summaries weight circuits equally.
    circ=augmented.groupby(['objective','policy','circuit'],as_index=False).agg(
        mean_gain_pct=('actual_gain_pct','mean'), mean_pred_pct=('predicted_gain_pct','mean'),
        mean_scored=('n_scored','mean'), mean_time_sec=('search_time_sec','mean'))
    circ.to_csv(args.out_dir/'per_circuit_policy.csv',index=False)
    macro=circ.groupby(['objective','policy'],as_index=False).agg(
        macro_gain_pct=('mean_gain_pct','mean'),
        macro_pred_pct=('mean_pred_pct','mean'),
        macro_scored=('mean_scored','mean'))
    macro.to_csv(args.out_dir/'macro_policy.csv',index=False)
    sel=augmented[augmented.policy.isin(['adaptive_beam','random_beam'])]
    w=sel.pivot(index=KEY,columns='policy',values=['actual_gain_pct','pick_sig','n_scored'])
    rows=[]
    for obj,part in w.groupby(level=0):
        d=part['actual_gain_pct']['adaptive_beam'].astype(float) - part['actual_gain_pct']['random_beam'].astype(float)
        same=part['pick_sig']['adaptive_beam']==part['pick_sig']['random_beam']
        rows.append(dict(objective=obj,n=len(d),adaptive_better=int((d>1e-8).sum()),
            random_better=int((d < -1e-8).sum()),tie=int((d.abs()<=1e-8).sum()),
            identical_picks=int(same.sum()),mean_delta_pp=float(d.mean()),
            median_delta_pp=float(d.median())))
    paired=pd.DataFrame(rows)
    paired.to_csv(args.out_dir/'adaptive_vs_random_paired.csv',index=False)

    # Note: a selection-only error analysis is conditional on model selection,
    # and cannot establish how error changes causally with distance.
    trace_msg='No candidate trace supplied: operator-step diagnostics are not available.'
    if args.candidates is not None:
        candidates=pd.read_csv(args.candidates)
        needed={'objective','policy','circuit','source_plan_id','source_sig',
                'candidate_sig','discovery_move_depth','predicted_gain_pct','actual_gain_pct'}
        missing=needed-set(candidates.columns)
        if missing:
            raise ValueError('Candidate trace missing columns: '+str(sorted(missing)))
        if candidates.actual_gain_pct.isna().any():
            trace_msg='Candidate trace has missing heldout labels; excluded missing rows from aggregate.'
        t=candidates.dropna(subset=['actual_gain_pct']).copy()
        by_move=grouped(t,['objective','policy','discovery_move_depth'])
        by_move.to_csv(args.out_dir/'candidate_error_by_move_depth.csv',index=False)
        by_move_all=grouped(t,['objective','discovery_move_depth'])
        by_move_all.to_csv(args.out_dir/'candidate_error_by_move_depth_pooled.csv',index=False)
        trace_msg=f'{len(t)} labeled candidate visits analyzed, grouped by discovery path length. Duplicates across policies/sources are not independent.'
    # Simple plot without explicit colors.
    try:
        import matplotlib.pyplot as plt
        for obj in sorted(by_dist.objective.unique()):
            d=by_dist[(by_dist.objective==obj) & (by_dist.policy=='adaptive_beam')]
            fig,ax=plt.subplots(figsize=(8,4.4))
            ax.plot(d.exchange_distance,d.mean_pred_pct,marker='o',label='Predicted gain')
            ax.plot(d.exchange_distance,d.mean_actual_pct,marker='s',label='Actual heldout gain')
            ax.axhline(0,lw=0.8,ls='--')
            ax.set_xlabel('Reuse-edge exchange distance (NOT operator move count)')
            ax.set_ylabel('Mean percentage gain (%)')
            ax.set_title(f'{obj}: selected-plan prediction vs heldout outcome')
            ax.legend();ax.grid(alpha=.15);fig.tight_layout()
            fig.savefig(args.out_dir/f'selected_gain_by_exchange_distance_{obj}.png',dpi=150)
            plt.close(fig)
    except ImportError:
        pass
    with (args.out_dir/'README_results.txt').open('w',encoding='utf-8') as f:
        f.write('Adaptive Ridge error diagnostics\n\n')
        f.write('Exchange distance = number of changed reuse edges, NOT sequential operator steps.\n')
        f.write('Selected-plan error is subject to selection bias; do NOT interpret trends causally.\n')
        f.write('Use per-circuit_policy.csv / macro_policy.csv for circuit-weighted policy gains.\n')
        f.write('Most distance bins have uneven circuit composition and small sample sizes.\n')
        f.write('The candidate trace uses heldout routing labels POST-HOC only.\n')
        f.write(trace_msg+'\n')
    print('\nMacro gains, equal circuit weights:')
    print(macro.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nPaired comparisons:')
    print(paired.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nAdaptive selected SWAP by exchange distance:')
    print(by_dist[(by_dist.objective=='swap_count') & (by_dist.policy=='adaptive_beam')][
        ['exchange_distance','n','circuits','mean_pred_pct','mean_actual_pct','optimism_pp','mae_pp']]
          .to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\n'+trace_msg)
    print('\nResults:',args.out_dir)

if __name__=='__main__':main()
