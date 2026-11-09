#!/usr/bin/env python3
"""Train-only family-disjoint OOF pairwise learning-to-rank for fixed-width reuse.

Python 3.10+, numpy/pandas/scikit-learn; imports the audited data loader from
train_reuse_ranking_v1.py which MUST be in the same directory when deployed.

RANK: learn sign(gain_i - gain_j) for valid plans belonging to SAME original.
GATE: require held-out model P(winner beats original) >= pre-registered threshold.
Both use only structural/logical features, NEVER physical costs at inference.
This is offline candidate selection, NOT trained beam expansion or final test.
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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import clone
import joblib

# This script is already present in the user's pilot_code from the last stage.
import train_reuse_ranking_v1 as base

OBJECTIVES = ('swap_count', 'routed_depth')
THRESHOLDS = (0.50, 0.65, 0.75)
METHODS = ('pair_logit', 'pair_hgb')
GROUP_KEYS = ('circuit_id', 'original_sig')
EPSILON = 1e-8


def save_json(path, obj):
    def convert(x):
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, (np.bool_,)): return bool(x)
        if isinstance(x, dict): return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)): return [convert(v) for v in x]
        return x
    Path(path).write_text(json.dumps(convert(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def stable_rng(*parts):
    b = '::'.join(str(x) for x in parts).encode('utf-8')
    seed = int.from_bytes(hashlib.sha256(b).digest()[:8], 'big') % (2**32)
    return np.random.default_rng(seed)


def source_id(x):
    return str(x['circuit_id']) + '::' + str(x['original_sig'])


def load_data(train_dir, ledger, full=True):
    labels = base.read_csv(train_dir / 'labels.csv')
    routes = base.read_csv(train_dir / 'routing_results.csv')
    comp = base.read_csv(train_dir / 'compiled_features.csv')
    split = base.read_csv(ledger)
    qc = base.audit_data(train_dir, labels, routes, comp, split)
    labels = labels.merge(split[['circuit_id','family_group','split']],
                          on='circuit_id', how='left', validate='many_to_one')
    data, features = base.features_for_labels(labels, comp, train_dir if full else None)
    if len(data) != len(labels): raise ValueError('Feature join changed row count')
    if data.family_group.isna().any() or data.split.ne('train').any():
        raise ValueError('LEAKAGE: only frozen train split is permitted')
    if len(set(features)) != len(features): raise ValueError('Duplicated predictors')
    return data, features, qc


def weighted_pairs(d, x, margin, max_nonanchor, seed):
    """Per-original comparisons + reversed orientation, weighted equally per family/original.

    Every original-candidate pair is retained if it has a non-tied target;
    candidate-candidate pairs are bounded for predictable runtime.
    """
    key_to_indices = d.groupby(['circuit_id','original_sig'], sort=True).indices
    count_orig = (d[['circuit_id','original_sig','family_group']].drop_duplicates()
                 .groupby('family_group').size().to_dict())
    all_x, all_y, all_w = [], [], []
    details = {'originals_with_pairs':0, 'n_oriented_pairs':0, 'n_anchor_oriented_pairs':0}
    for key, idx in key_to_indices.items():
        loc = np.asarray(idx, dtype=np.int64)
        sub = d.iloc[loc]
        orig_pos = np.flatnonzero(sub.is_original.to_numpy(dtype=int)==1)
        if len(orig_pos)!=1: raise ValueError(f'Expected one Original per source: {key}')
        oi = int(orig_pos[0]); gains = sub.abs_gain.to_numpy(dtype=float)
        others = [k for k in range(len(loc)) if k != oi]
        anchors = [(oi,k) for k in others if abs(gains[oi]-gains[k]) > margin]
        combinations = []
        for i in range(len(others)):
            for j in range(i+1,len(others)):
                a,b = others[i],others[j]
                if abs(gains[a]-gains[b])>margin:
                    combinations.append((a,b))
        rng = stable_rng(seed, key[0], key[1])
        if len(combinations)>max_nonanchor:
            selected = np.sort(rng.choice(len(combinations), max_nonanchor, replace=False))
            combinations=[combinations[i] for i in selected]
        pairs=anchors+combinations
        if not pairs: continue
        details['originals_with_pairs'] += 1
        details['n_anchor_oriented_pairs'] += 2*len(anchors)
        # Weight one family equally, then each Original equally within family, then each pair.
        family = str(sub.family_group.iloc[0]); orig_weight = 1.0/count_orig[family]
        weight = orig_weight/len(pairs)
        for a,b in pairs:
            diff=x[loc[a]]-x[loc[b]]
            better=int(gains[a]>gains[b])
            avg=0.5*(x[loc[a]]+x[loc[b]])
            all_x.append(np.concatenate([diff,avg])); all_y.append(better); all_w.append(weight)
            all_x.append(np.concatenate([-diff,avg])); all_y.append(1-better); all_w.append(weight)
    if not all_x:
        raise ValueError('No informative within-Original pairs (perhaps all gains tied)')
    xx=np.asarray(all_x,dtype=np.float32)
    yy=np.asarray(all_y,dtype=np.int8)
    weights=np.asarray(all_w,dtype=np.float64)
    weights *= len(weights)/weights.sum()
    details['n_oriented_pairs']=len(yy)
    if len(np.unique(yy)) != 2: raise AssertionError('Pair labels not two-class')
    if not np.isfinite(xx).all(): raise AssertionError('nonfinite training feature')
    return xx, yy, weights, details


def get_models():
    return {
        'pair_logit':Pipeline([
            ('scale', StandardScaler()),
            ('classifier', LogisticRegression(C=0.3,max_iter=1000,random_state=2026))
        ]),
        'pair_hgb':Pipeline([
            ('classifier',HistGradientBoostingClassifier(
                max_iter=90,learning_rate=0.05,max_depth=3,max_leaf_nodes=7,
                min_samples_leaf=35,l2_regularization=12,early_stopping=False,
                random_state=2026))
        ])
    }


def fit_pair_models(train, cols, margin, max_nonanchor, seed):
    # The imputer is fitted ONLY on this fold's training family groups.
    # Original rows are included because they are known at inference time too.
    try:
        imp = SimpleImputer(strategy='constant', fill_value=0.0, keep_empty_features=True)
    except TypeError:  # sklearn before 1.2
        imp = SimpleImputer(strategy='constant', fill_value=0.0)
    x = imp.fit_transform(train[cols].to_numpy(dtype=np.float64)).astype(np.float32)
    if x.shape[1]!=len(cols): raise ValueError('imputer dropped predictors')
    pairx, pairy, pairw, counts=weighted_pairs(train,x,margin,max_nonanchor,seed)
    fitted={}
    for name, model in get_models().items():
        model.fit(pairx,pairy,classifier__sample_weight=pairw)
        fitted[name]=model
    return imp, fitted, counts


def symmetric_win_prob(model, pair_features):
    # Pair input is [x_i - x_j, (x_i + x_j)/2].
    # Inverse orientation changes the first half only, not its shared context.
    pair_features=np.asarray(pair_features,dtype=np.float32)
    ncols=pair_features.shape[1]//2
    inverse=pair_features.copy()
    inverse[:,:ncols]*=-1
    p=model.predict_proba(pair_features)[:,1]
    reverse=model.predict_proba(inverse)[:,1]
    out=0.5*(p+1.0-reverse)
    return np.clip(out,0.0,1.0)


def model_rank_group(model, imp, sub, cols):
    """Score candidates with symmetric pairwise Borda; P(candidate > Original) gates action."""
    sub=sub.sort_values('sig').reset_index(drop=True)
    ox=np.flatnonzero(sub.is_original.to_numpy(dtype=int)==1)
    if len(ox)!=1: raise ValueError('Original row absent or duplicated')
    oi=int(ox[0]); n=len(sub)
    x=imp.transform(sub[cols].to_numpy(dtype=np.float64)).astype(np.float32)
    if n==1:
        return sub.assign(borda=0.5,p_vs_original=0.5),oi
    # Pairwise votes, enforcing p(i>j)+p(j>i)=1.
    delta=(x[:,None,:]-x[None,:,:]).reshape(-1,x.shape[1])
    avg=(0.5*(x[:,None,:]+x[None,:,:])).reshape(-1,x.shape[1])
    pair_features=np.concatenate([delta,avg],axis=1)
    p=symmetric_win_prob(model,pair_features).reshape(n,n)
    np.fill_diagonal(p,0.5)
    if not np.allclose(p + p.T,1,atol=1e-5):
        raise AssertionError('Pairwise symmetry invariant failed')
    borda=(p.sum(axis=1)-0.5)/(n-1)
    return sub.assign(borda=borda,p_vs_original=p[:,oi]), oi


def choose_pairwise(ranked, oi, gate):
    origin=ranked.iloc[oi]
    if origin.objective=='swap_count' and origin.baseline_mean<=EPSILON:
        return None
    c=ranked[ranked.is_original.eq(0)].copy()
    c=c[(c.p_vs_original >= gate - EPSILON) &
        (c.borda > float(origin.borda) + EPSILON)]
    if c.empty: return None
    # Deterministic tie breaker: prefer lower pre-route norm_depth, fewer edits, then sig.
    c=c.sort_values(['borda','delta_compiled_norm_depth','discovery_move_depth','sig'],
                    ascending=[False,True,True,True],kind='mergesort')
    return c.iloc[0]


def choose_rule(group):
    c=group[group.is_original.eq(0)].sort_values(
        ['delta_compiled_norm_depth','sig'],ascending=[True,True],kind='mergesort')
    if c.empty or c.iloc[0].delta_compiled_norm_depth >= 0: return None
    return c.iloc[0]


def decision_row(group,pick, method, gate=None):
    o=group[group.is_original.eq(1)].iloc[0]
    all_c=group[group.is_original.eq(0)]
    oracle=max(0.0,float(all_c.abs_gain.max())) if not all_c.empty else 0.0
    gain=float(pick.abs_gain) if pick is not None else 0.0
    pct=100.0*gain/float(o.baseline_mean) if float(o.baseline_mean)>EPSILON else None
    return dict(objective=str(o.objective),method=method,circuit_id=str(o.circuit_id),
       original_sig=str(o.original_sig),family_group=str(o.family_group),
       candidate_sig=(str(pick.sig) if pick is not None else str(o.original_sig)),
       chosen_move_depth=int(pick.discovery_move_depth) if pick is not None else 0,
       baseline_mean=float(o.baseline_mean),n_candidates=len(all_c),
       selected_true_gain=gain, selected_gain_pct=pct, oracle_gain=oracle,
       regret=oracle-gain, harm=int(gain < -EPSILON), no_repair=int(pick is None),
       gate_threshold=gate,
       chosen_borda=float(pick.borda) if pick is not None and hasattr(pick,'borda') else None,
       chosen_p_vs_original=float(pick.p_vs_original) if pick is not None and hasattr(pick,'p_vs_original') else None)


def summarize(choices):
    rows=[]
    for (objective, method), g in choices.groupby(['objective','method']):
        fam_gain=g.groupby('family_group').selected_true_gain.mean()
        circ_gain=g.groupby('circuit_id').selected_true_gain.mean()
        pct=g.selected_gain_pct
        harmed=g.loc[g.harm.eq(1),'selected_true_gain']
        opportunities=int((g.oracle_gain>EPSILON).sum())
        missed=int(((g.oracle_gain>EPSILON)&g.no_repair.eq(1)).sum())
        rows.append(dict(objective=objective,method=method,
             n_originals=len(g),n_circuits=g.circuit_id.nunique(),n_families=g.family_group.nunique(),
             mean_gain_abs_by_original=float(g.selected_true_gain.mean()),
             mean_gain_abs_by_circuit=float(circ_gain.mean()),
             mean_gain_abs_by_family=float(fam_gain.mean()),
             mean_gain_pct_by_original=float(pct.mean()) if pct.notna().any() else None,
             n_pct_defined=int(pct.notna().sum()),
             harm_count=int(g.harm.sum()),repair_count=int((1-g.no_repair).sum()),
             no_repair_count=int(g.no_repair.sum()),
             missed_opportunity_count=missed,opportunity_count=opportunities,
             mean_regret=float(g.regret.mean()),
             total_harm_abs=float(-harmed.sum()) if len(harmed) else 0.0))
    return pd.DataFrame(rows).sort_values(['objective','method']).reset_index(drop=True)


def check_original_group_integrity(d):
    if d.duplicated(['objective','circuit_id','original_sig','sig']).any():
        raise ValueError('Duplicated source/candidate rows')
    for key,part in d.groupby(['objective','circuit_id','original_sig']):
        if int(part.is_original.sum())!=1: raise ValueError(f'Bad original multiplicity: {key}')
        original=part.loc[part.is_original.eq(1)].iloc[0]
        if str(original.sig)!=str(original.original_sig):
            raise ValueError(f'Original signature mismatch: {key}')
        if not np.allclose(part.baseline_mean.to_numpy(),float(original.candidate_mean)):
            raise ValueError(f'Inconsistent routing baseline: {key}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', type=Path, required=True)
    parser.add_argument('--split-ledger', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--feature-mode',choices=['full','compiled'],default='full')
    parser.add_argument('--cv-folds',type=int,default=5)
    parser.add_argument('--margin',type=float,default=0.2,
                        help='Omit comparisons of plans with nearly equal mean route costs')
    parser.add_argument('--max-candidate-pairs',type=int,default=140,
                        help='Cap candidate-candidate pairs per Original; all original-candidate pairs remain')
    parser.add_argument('--prior-selected',type=Path,default=None,
                        help='Optional ml_model_cv_full_v1/oof_selected_by_original.csv for same-source comparison')
    args=parser.parse_args()
    if args.cv_folds<2 or args.margin<0 or args.max_candidate_pairs<1:
        parser.error('Invalid cv-folds/margin/max-candidate-pairs')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    d, cols, qc=load_data(args.train_dir,args.split_ledger,args.feature_mode=='full')
    check_original_group_integrity(d)
    if set(d.objective.unique())!=set(OBJECTIVES): raise ValueError('Missing an objective')
    all_decisions=[]; scored=[]; counts={}; fold_report=[]
    for objective in OBJECTIVES:
        dataset=d[d.objective.eq(objective)].copy().reset_index(drop=True)
        originals=dataset[dataset.is_original.eq(1)].copy().reset_index(drop=True)
        # GroupKFold is on ORIGINALS, not rows, to include no-candidate sources.
        k=min(args.cv_folds,originals.family_group.nunique())
        if k<2: raise ValueError('Need two or more family groups')
        folds=GroupKFold(n_splits=k)
        groups_for_fold={}
        for fold, (_, test_oi) in enumerate(folds.split(originals,groups=originals.family_group)):
            for item in originals.iloc[test_oi].itertuples(index=False):
                groups_for_fold[(str(item.circuit_id),str(item.original_sig))]=fold
        dataset['fold']=[groups_for_fold[(str(c),str(o))] for c,o in zip(dataset.circuit_id,dataset.original_sig)]
        if len(groups_for_fold)!=len(originals): raise AssertionError('Incorrect CV original coverage')
        grouped=dataset.groupby(list(GROUP_KEYS),sort=True)
        counts[objective]={}
        for fold in range(k):
            tr=dataset[dataset.fold.ne(fold)].copy().reset_index(drop=True)
            te=dataset[dataset.fold.eq(fold)].copy().reset_index(drop=True)
            if set(tr.family_group) & set(te.family_group): raise AssertionError('LEAKAGE: family crosses fold')
            imp, models, info=fit_pair_models(tr,cols,args.margin,args.max_candidate_pairs,2026+fold)
            counts[objective][str(fold)]=info
            fold_report.append({'objective':objective,'fold':fold,
                 'train_families':int(tr.family_group.nunique()),'test_families':int(te.family_group.nunique()),
                 'train_originals':int(tr.is_original.sum()),'test_originals':int(te.is_original.sum()),
                 **info})
            for _, group in te.groupby(list(GROUP_KEYS),sort=True):
                group=group.copy().reset_index(drop=True)
                # Fixed-label baselines, identical in meaning to the previous v1.
                all_decisions.append(decision_row(group,None,'no_repair'))
                all_decisions.append(decision_row(group,choose_rule(group),'logical_depth_rule'))
                cand=group[group.is_original.eq(0)]
                if not cand.empty:
                    best=cand.sort_values(['abs_gain','sig'],ascending=[False,True]).iloc[0]
                    oracle_pick=best if best.abs_gain>0 else None
                else: oracle_pick=None
                all_decisions.append(decision_row(group,oracle_pick,'oracle'))
                for model_name, model in models.items():
                    rank, oi=model_rank_group(model,imp,group,cols)
                    rank['objective']=objective
                    rank['cv_fold']=fold
                    rank['pairwise_model']=model_name
                    scored.append(rank[['objective','cv_fold','pairwise_model','circuit_id',
                        'family_group','original_sig','sig','is_original','discovery_move_depth',
                        'abs_gain','borda','p_vs_original']].copy())
                    for gate in THRESHOLDS:
                        selected=choose_pairwise(rank,oi,gate)
                        all_decisions.append(decision_row(rank,selected,
                             f'{model_name}_gate_{gate:.2f}',gate))
    result=pd.DataFrame(all_decisions)
    expected=len(OBJECTIVES)*int(qc['source_originals'])*(3+len(METHODS)*len(THRESHOLDS))
    if len(result)!=expected: raise AssertionError(f'Unexpected decisions {len(result)} vs {expected}')
    outscore=pd.concat(scored,ignore_index=True)
    uniq=outscore.drop_duplicates(['objective','pairwise_model','circuit_id','original_sig','sig'])
    if len(uniq)!=len(outscore): raise AssertionError('OOF scored row duplication')
    compared=summarize(result)
    result.to_csv(args.out_dir/'oof_pairwise_selected.csv',index=False,encoding='utf-8-sig')
    outscore.to_csv(args.out_dir/'oof_pairwise_candidate_scores.csv',index=False,encoding='utf-8-sig')
    compared.to_csv(args.out_dir/'oof_pairwise_method_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(fold_report).to_csv(args.out_dir/'cv_pair_training_counts.csv',index=False,encoding='utf-8-sig')
    prior_comparison=None
    if args.prior_selected:
        prior=base.read_csv(args.prior_selected)
        needed={'objective','method','circuit_id','original_sig','selected_true_gain','no_repair'}
        if not needed<=set(prior.columns):
            raise ValueError('Prior selection file missing required columns')
        prior_orig=set(tuple(x) for x in prior.loc[prior.method.eq('no_repair'),
                                                  ['objective','circuit_id','original_sig']].to_numpy())
        now_orig=set(tuple(x) for x in result.loc[result.method.eq('no_repair'),
                                                  ['objective','circuit_id','original_sig']].to_numpy())
        if prior_orig!=now_orig: raise ValueError('Prior OOF evaluated different Originals')
        prior=prior[prior.method.isin(['ridge','hgb','random_one'])].copy()
        merged_summary=summarize(pd.concat([result,prior],ignore_index=True))
        merged_summary.to_csv(args.out_dir/'comparison_with_previous_oof.csv',index=False,encoding='utf-8-sig')
        prior_comparison='comparison_with_previous_oof.csv'
    # Fit only on TRAIN data. Cross-validated model selection is exploratory; no Val/Test files read.
    model_names={}
    for objective in OBJECTIVES:
        tr=d[d.objective.eq(objective)].copy().reset_index(drop=True)
        imp,fitted, info=fit_pair_models(tr,cols,args.margin,args.max_candidate_pairs,2026)
        for name,model in fitted.items():
            fn=f'{objective}_{name}_{args.feature_mode}_TRAIN_ONLY.joblib'
            joblib.dump({'model':model,'imputer':imp,'feature_columns':cols,
                         'objective':objective,'feature_mode':args.feature_mode,
                         'gate_thresholds_exploratory':THRESHOLDS,
                         'scope':'TRAIN_ONLY, candidate reranking; NO Validation/Test'},args.out_dir/fn)
            model_names[f'{objective}::{name}']=fn
    summary={'protocol':'family_disjoint_pairwise_rank_and_gate_v2',
       'data_checks':qc,'feature_mode':args.feature_mode,'n_predictors':len(cols),
       'pair_features':'[xi-xj, (xi+xj)/2] to preserve source context',
       'margin':args.margin,'max_candidate_candidate_pairs':args.max_candidate_pairs,
       'cv_folds':args.cv_folds,'gate_thresholds_predeclared':list(THRESHOLDS),
       'fold_training_pairs':counts,'models_saved':model_names,
       'comparison_with_previous_oof':prior_comparison,
       'method_comparison':compared.where(pd.notna(compared),None).to_dict('records'),
       'warnings':[
           'TRAIN-ONLY GroupKFold OOF; Validation/Final Test were not read.',
           'Predeclared gates are exploratory. Do NOT select a threshold using its own OOF performance and report that performance as unbiased. Use nested CV or untouched Validation for threshold selection.',
           'Only sampled, offline valid candidates are reranked; beam expansion is NOT learned.',
           'Five-seed mean routing costs are noisy labels, not held-out physical evaluation.',
           'Oracle uses true labels and is not a deployable method.',
           'Candidate-candidate training pairs share sources and do not constitute independent samples.'
       ]}
    save_json(args.out_dir/'pairwise_learning_summary.json',summary)
    print(f"[PASS] training only: {qc['n_circuits']} circuits, {qc['source_originals']} originals, {qc['label_rows']} label rows, {len(cols)} predictors")
    print(f"[PASS] {args.cv_folds}-fold family-disjoint pairwise CV; thresholds {THRESHOLDS} are PREDECLARED, not tuned")
    print(compared[['objective','method','mean_gain_abs_by_original','mean_gain_pct_by_original',
                    'harm_count','no_repair_count','mean_regret']].round(3).to_string(index=False))
    print(f'[saved] {args.out_dir.resolve()}')


if __name__=='__main__':
    try: main()
    except Exception as e:
        print(f'[FAIL] {type(e).__name__}: {e}',file=sys.stderr)
        raise
