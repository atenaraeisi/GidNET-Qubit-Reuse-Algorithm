#!/usr/bin/env python3
"""Train-only, family-disjoint OOF pilot: fixed-width reuse plan gain prediction.

Does not touch validation/final-test circuits.  No Qiskit, routing, or web access.
A positive prediction can select a valid candidate; otherwise retain the original.

This is a STATIC CANDIDATE-SET RANKING prototype, NOT learned beam expansion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

METRICS = ('swap_count', 'routed_depth')
KEY = ['circuit_id', 'original_sig', 'sig']
PLAN_KEY = ['circuit_id', 'sig']
FORBIDDEN_FEATURE_FRAGMENTS = ('swap_count', 'routed_depth', 'routing_time', 'baseline_mean', 'candidate_mean', 'abs_gain', 'gain_pct', 'harm')


def read_csv(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Missing required file: {p}')
    return pd.read_csv(p, encoding='utf-8-sig')


def unique(df, cols, label):
    if df.duplicated(cols).any():
        bad = df.loc[df.duplicated(cols, keep=False), cols].head(4).to_dict('records')
        raise ValueError(f'Non-unique {label}: {bad}')


def jdump(path, data):
    def cvt(v):
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, (np.bool_,)): return bool(v)
        if isinstance(v, dict): return {str(k):cvt(val) for k,val in v.items()}
        if isinstance(v, (list,tuple)): return [cvt(x) for x in v]
        return v
    path.write_text(json.dumps(cvt(data), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def audit_data(d, labels, routes, comp, split):
    needed = {'circuit_id','original_sig','sig','is_original','discovery_move_depth',
              'objective','baseline_mean','candidate_mean','abs_gain','n_seeds'}
    if not needed <= set(labels): raise ValueError(f'labels.csv missing {sorted(needed-set(labels))}')
    if set(labels.objective.unique())!=set(METRICS): raise ValueError('Unexpected objectives in labels')
    unique(labels, KEY+['objective'], 'labels')
    all_seeds = sorted(routes.seed.unique().tolist())
    if len(all_seeds)!=5 or set(all_seeds)!={42,43,44,45,46}:
        raise ValueError(f'This pilot requires exact development seeds 42..46; got {all_seeds}')
    unique(routes, PLAN_KEY+['seed'], 'routing jobs')
    if len(routes)!=5810 or len(routes[~routes.status.eq('ok')]):
        raise ValueError(f'Unexpected training routing completeness / status: {len(routes)} rows')
    if routes.groupby(PLAN_KEY).seed.nunique().ne(5).any():
        raise ValueError('Missing routing seed for one or more plans')
    unique(comp, PLAN_KEY, 'compiled features')
    if len(comp[~comp.status.eq('ok')]): raise ValueError('Compiled errors')
    if not {'circuit_id','split','family_group'}<=set(split):
        raise ValueError('split_ledger.csv requires circuit_id, split, family_group')
    unique(split, ['circuit_id'], 'split ledger')
    if set(labels.circuit_id) - set(split.loc[split.split.eq('train'),'circuit_id']):
        raise ValueError('LEAKAGE: labels contain circuits outside frozen training set')
    if labels['n_seeds'].ne(5).any(): raise ValueError('One or more labels are not 5-seed')
    if len(labels)!=2886: raise ValueError(f'Unexpected label rows: {len(labels)} != 2886')
    for obj in METRICS:
        part=labels.loc[labels.objective.eq(obj)].copy()
        means=routes.groupby(PLAN_KEY)[obj].mean().rename('route_mean').reset_index()
        z=part.merge(means, on=PLAN_KEY,how='left',validate='many_to_one')
        if z.route_mean.isna().any() or not np.allclose(z.candidate_mean,z.route_mean,rtol=0,atol=1e-7):
            raise ValueError(f'Mean route cost disagree with labels for {obj}')
        bas=means.rename(columns={'sig':'original_sig','route_mean':'orig_mean'})
        z=z.merge(bas,on=['circuit_id','original_sig'],how='left',validate='many_to_one')
        if z.orig_mean.isna().any() or not np.allclose(z.baseline_mean,z.orig_mean,rtol=0,atol=1e-7):
            raise ValueError(f'Original route baseline disagree with labels for {obj}')
        if not np.allclose(z.abs_gain,z.baseline_mean-z.candidate_mean,rtol=0,atol=1e-7):
            raise ValueError(f'Gain sign error for {obj}')
    keys=labels.loc[labels.objective.eq('swap_count'),KEY].sort_values(KEY).reset_index(drop=True)
    other=labels.loc[labels.objective.eq('routed_depth'),KEY].sort_values(KEY).reset_index(drop=True)
    if not keys.equals(other): raise ValueError('Objective-specific label keys differ')
    # Per-circuit features may be shared by several original plans. No problem.
    if set(map(tuple,comp[PLAN_KEY].to_numpy()))!=set(map(tuple,routes[PLAN_KEY].drop_duplicates().to_numpy())):
        raise ValueError('Compiled-feature plan keys != routing plan keys')
    return {'routing_rows':len(routes),'unique_plans':routes[PLAN_KEY].drop_duplicates().shape[0],
            'label_rows':len(labels),'source_candidate_rows':len(keys),
            'source_originals':int(labels.query("objective=='swap_count' and is_original==1").shape[0]),
            'seed_set':all_seeds,'n_circuits':labels.circuit_id.nunique()}


def features_for_labels(labels, comp, full_dir=None):
    # All predictors are available before physical routing. NEVER merge route labels into predictors.
    feature_columns=[]
    c=comp[PLAN_KEY+['logical_depth','norm_depth','dyn_width']].copy()
    c=c.rename(columns={k:'compiled_'+k for k in ('logical_depth','norm_depth','dyn_width')})
    c0=c.rename(columns={'sig':'original_sig',
            **{f'compiled_{k}':f'orig_compiled_{k}' for k in ('logical_depth','norm_depth','dyn_width')}})
    df=labels.merge(c,on=PLAN_KEY,how='left',validate='many_to_one')
    df=df.merge(c0,on=['circuit_id','original_sig'],how='left',validate='many_to_one')
    for k in ('logical_depth','norm_depth','dyn_width'):
        a='compiled_'+k; b='orig_compiled_'+k
        if df[a].isna().any() or df[b].isna().any(): raise ValueError(f'Missing compiled {k}')
        feature_columns.extend((a,b))
        dc='delta_compiled_'+k
        df[dc]=df[a]-df[b]
        feature_columns.append(dc)
    if full_dir is not None:
        f=read_csv(full_dir/'features.csv')
        pairs=read_csv(full_dir/'feature_pairs.csv')
        unique(f,PLAN_KEY,'structural features')
        unique(pairs,KEY,'feature pairs')
        if set(map(tuple,f[PLAN_KEY].to_numpy()))!=set(map(tuple,comp[PLAN_KEY].to_numpy())):
            raise ValueError('Plan keys differ across features.csv and compiled_features.csv')
        # Never trust a selected-pair join unless the exact candidate and original are present.
        if set(map(tuple,pairs[KEY].to_numpy()))!=set(map(tuple,labels.query("objective=='swap_count'")[KEY].to_numpy())):
            raise ValueError('Pair keys differ across feature_pairs.csv and labels.csv')
        structural=[x for x in f.columns if x not in PLAN_KEY]
        if len(structural)!=25:
            raise ValueError(f'Expected 25 structural features; got {len(structural)}: {structural}')
        f=f.rename(columns={k:'struct_'+k for k in structural})
        f0=f.rename(columns={'sig':'original_sig',**{'struct_'+k:'orig_struct_'+k for k in structural}})
        df=df.merge(f,on=PLAN_KEY,how='left',validate='many_to_one')
        df=df.merge(f0,on=['circuit_id','original_sig'],how='left',validate='many_to_one')
        # Operator and path length vary for duplicate plans, so use only path length from labels.
        for k in structural:
            x='struct_'+k; y='orig_struct_'+k; delta='delta_struct_'+k
            df[delta]=pd.to_numeric(df[x],errors='coerce')-pd.to_numeric(df[y],errors='coerce')
            feature_columns.extend((y,delta))
    df['move_depth']=df['discovery_move_depth']
    feature_columns.append('move_depth')
    for col in feature_columns:
        if any(s in col for s in FORBIDDEN_FEATURE_FRAGMENTS):
            raise ValueError(f'ROUTING LABEL LEAKAGE in feature {col}')
        df[col]=pd.to_numeric(df[col], errors='coerce').replace([np.inf,-np.inf],np.nan)
    if len(df)!=len(labels): raise ValueError('Unexpected row multiplicity after feature join')
    return df, feature_columns


def models():
    return {
        'ridge': Pipeline([
            ('impute',SimpleImputer(strategy='constant',fill_value=0,add_indicator=True)),
            ('scale',StandardScaler()),('regressor',Ridge(alpha=30))
        ]),
        'hgb': Pipeline([
            ('regressor',HistGradientBoostingRegressor(
                max_iter=100,learning_rate=0.05,max_leaf_nodes=7,
                max_depth=3,min_samples_leaf=25,l2_regularization=10,
                early_stopping=False,random_state=2026))
        ])
    }


def sample_weights(part):
    # Each family has equal total weight, each original within a family equal,
    # and each candidate within an original equal. This is NOT row-level uniform.
    p=part[['family_group','circuit_id','original_sig']].copy()
    p['oid']=p['circuit_id']+'::'+p['original_sig']
    famn=p[['family_group','oid']].drop_duplicates().groupby('family_group').oid.size()
    origsize=p.groupby('oid').oid.size()
    w=np.array([1.0/(famn[g]*origsize[o]) for g,o in zip(p.family_group,p.oid)],dtype=float)
    return w / w.mean()


def stable_pick(parts, n):
    h=hashlib.sha256('::'.join(map(str,parts)).encode()).digest()
    return int.from_bytes(h[:8],'big')%n

def pick_baseline(part, heuristic):
    if len(part)==0: return None
    if heuristic=='oracle':
        pp=part.sort_values(['abs_gain','sig'],ascending=[False,True]).iloc[0]
        return pp if pp.abs_gain>0 else None
    if heuristic=='random_one':
        # Reserve index 0 for No Repair.
        i=stable_pick((part.iloc[0].circuit_id,part.iloc[0].original_sig,part.iloc[0].objective),len(part)+1)
        return part.iloc[i-1] if i else None
    if heuristic=='logical_depth_rule':
        part=part.sort_values(['delta_compiled_norm_depth','sig'],ascending=[True,True])
        p=part.iloc[0]
        return p if p.delta_compiled_norm_depth<0 else None
    if heuristic in ('ridge','hgb'):
        part=part.sort_values(['pred_'+heuristic,'sig'],ascending=[False,True])
        p=part.iloc[0]
        return p if p['pred_'+heuristic]>0 else None
    if heuristic=='no_repair': return None
    raise KeyError(heuristic)


def cv_for_objective(df, feature_cols, objective, max_folds):
    d=df[df.objective==objective].copy()
    original=d[d.is_original.eq(1)].copy()
    candidates=d[d.is_original.eq(0)].copy()
    if len(candidates)==0: raise ValueError(f'No candidate data for {objective}')
    groups=candidates.family_group
    splits=GroupKFold(n_splits=min(max_folds,groups.nunique()))
    candidates['cv_fold']=-1
    for name in models(): candidates['pred_'+name]=np.nan
    for k,(tri,tei) in enumerate(splits.split(candidates[feature_cols],candidates.abs_gain,groups)):
        tr=candidates.iloc[tri].copy(); te=candidates.iloc[tei].copy()
        if set(tr.family_group) & set(te.family_group): raise AssertionError('Family leakage')
        for name,pipe in models().items():
            pipe.fit(tr[feature_cols],tr.abs_gain,regressor__sample_weight=sample_weights(tr))
            candidates.loc[te.index,'pred_'+name]=pipe.predict(te[feature_cols])
        candidates.loc[te.index,'cv_fold']=k
    if candidates[['pred_ridge','pred_hgb']].isna().any().any():
        raise ValueError('Missing OOF model scores')
    selected=[]
    variants=['no_repair','random_one','logical_depth_rule','ridge','hgb','oracle']
    gb=candidates.groupby(['circuit_id','original_sig'],sort=False)
    for row in original.itertuples(index=False):
        group=gb.get_group((row.circuit_id,row.original_sig)) if (row.circuit_id,row.original_sig) in gb.groups else candidates.iloc[0:0]
        best=max(0.0, float(group.abs_gain.max())) if len(group) else 0.0
        for method in variants:
            pick=pick_baseline(group,method)
            true_gain=float(pick.abs_gain) if pick is not None else 0.0
            pred=float(pick['pred_'+method]) if pick is not None and method in ('ridge','hgb') else None
            pct=(100.0*true_gain / float(row.baseline_mean)) if row.baseline_mean>0 else None
            selected.append(dict(objective=objective,method=method,circuit_id=row.circuit_id,
               family_group=row.family_group,original_sig=row.original_sig,
               candidate_sig=(pick.sig if pick is not None else row.original_sig),
               chosen_move_depth=(int(pick.discovery_move_depth) if pick is not None else 0),
               n_candidates=len(group),baseline_mean=float(row.baseline_mean),
               selected_true_gain=true_gain, selected_gain_pct=pct,
               oracle_gain=best,regret=best-true_gain,
               harm=int(true_gain<-1e-8),no_repair=int(pick is None),
               predicted_gain=pred))
    oof=candidates[KEY+['family_group','discovery_move_depth','cv_fold','abs_gain','pred_ridge','pred_hgb']].copy()
    summary={}
    for name in models():
        p=candidates['pred_'+name].to_numpy(dtype=float)
        y=candidates.abs_gain.to_numpy(dtype=float)
        summary[name]={'oof_mae':float(mean_absolute_error(y,p)),
                       'oof_r2':float(r2_score(y,p))}
    return oof,pd.DataFrame(selected),summary


def summaries(selected):
    z=[]
    for (obj,method),p in selected.groupby(['objective','method']):
        s=p['selected_gain_pct']
        p_circ=p.groupby('circuit_id')['selected_true_gain'].mean()
        p_family=p.groupby('family_group')['selected_true_gain'].mean()
        z.append(dict(objective=obj,method=method,n_originals=len(p),n_circuits=p.circuit_id.nunique(),
                      n_family_groups=p.family_group.nunique(),
                      n_pct_defined=int(s.notna().sum()),
                      mean_gain_abs_by_original=float(p.selected_true_gain.mean()),
                      mean_gain_pct_by_original=float(s.mean()) if s.notna().any() else None,
                      mean_gain_abs_by_circuit=float(p_circ.mean()),
                      mean_gain_abs_by_family=float(p_family.mean()),
                      harm_count=int(p.harm.sum()),
                      no_repair_count=int(p.no_repair.sum()),
                      mean_regret=float(p.regret.mean())))
    return pd.DataFrame(z).sort_values(['objective','method']).reset_index(drop=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train-dir',type=Path,required=True,help='ml_train_prepare_v1 (only train)')
    ap.add_argument('--split-ledger',type=Path,required=True,help='ml_split_v1/split_ledger.csv')
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--feature-mode',choices=['full','compiled'],default='full')
    ap.add_argument('--cv-folds',type=int,default=5)
    a=ap.parse_args()
    if a.cv_folds<2: ap.error('--cv-folds must be at least 2')
    a.out_dir.mkdir(parents=True,exist_ok=True)
    labels=read_csv(a.train_dir/'labels.csv')
    routes=read_csv(a.train_dir/'routing_results.csv')
    comp=read_csv(a.train_dir/'compiled_features.csv')
    split=read_csv(a.split_ledger)
    qc=audit_data(a.train_dir,labels,routes,comp,split)
    sub=split[['circuit_id','family_group','split']]
    labels=labels.merge(sub,on='circuit_id',how='left',validate='many_to_one')
    data, feature_cols=features_for_labels(labels,comp, a.train_dir if a.feature_mode=='full' else None)
    if data['family_group'].isna().any(): raise ValueError('Missing family group')
    if any(t in c for c in feature_cols for t in FORBIDDEN_FEATURE_FRAGMENTS):
        raise ValueError('Leakage from physical routing metrics into predictor!')
    outputs=[]; oofs=[]; prediction_summary={}
    for obj in METRICS:
        oof, choices, p=cv_for_objective(data,feature_cols,obj,a.cv_folds)
        outputs.append(choices);oofs.append(oof);prediction_summary[obj]=p
    selected=pd.concat(outputs,ignore_index=True)
    oof=pd.concat(oofs,ignore_index=True)
    result=summaries(selected)
    selected.to_csv(a.out_dir/'oof_selected_by_original.csv',index=False,encoding='utf-8-sig')
    oof.to_csv(a.out_dir/'oof_candidate_predictions.csv',index=False,encoding='utf-8-sig')
    result.to_csv(a.out_dir/'oof_method_comparison.csv',index=False,encoding='utf-8-sig')
    # Fit frozen starter models on ALL TRAIN candidates, for future validation only;
    # never import labels from validation/final test here.
    models_out={}
    for obj in METRICS:
        tr=data.loc[data.objective.eq(obj)&data.is_original.eq(0)]
        for name,pipe in models().items():
            pipe.fit(tr[feature_cols],tr.abs_gain,regressor__sample_weight=sample_weights(tr))
            target=a.out_dir/f'{obj}_{name}_{a.feature_mode}_TRAIN_ONLY.joblib'
            joblib.dump({'model':pipe,'features':feature_cols,'objective':obj,
                         'feature_mode':a.feature_mode,'scope':'TRAIN_ONLY'},target)
            models_out[obj+'_'+name]=target.name
    summary=dict(qc,feature_mode=a.feature_mode,n_predictor_columns=len(feature_cols),
             feature_columns=feature_cols,cv_split_method='GroupKFold, family_group only (TRAIN data)',
             cv_folds=min(a.cv_folds,data.loc[data.is_original.eq(0),'family_group'].nunique()),
             oof_regression=prediction_summary,
             method_comparison=result.where(pd.notna(result),None).to_dict('records'),
             trained_models=models_out,
             warning='Train-only grouped OOF experiment. NOT untouched validation or final-test evaluation. '
                    'All candidates were discovered OFFLINE without the model; NOT learned beam expansion. '
                    'OOF selection is on 5-seed averaged labels and may be noisy on unseen routing seeds.')
    jdump(a.out_dir/'learning_summary.json',summary)
    print(f"[PASS] {qc['routing_rows']} routing jobs, {qc['label_rows']} labels, {len(feature_cols)} predictors")
    print(f"[PASS] family-disjoint {summary['cv_folds']}-fold CV, models=Ridge/HGB")
    cols=['objective','method','mean_gain_pct_by_original','mean_gain_abs_by_original','harm_count','no_repair_count','mean_regret']
    print(result[cols].round(3).to_string(index=False))
    print(f'[saved] {a.out_dir.resolve()}')


if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'[FAIL] {type(exc).__name__}: {exc}',file=sys.stderr)
        raise
