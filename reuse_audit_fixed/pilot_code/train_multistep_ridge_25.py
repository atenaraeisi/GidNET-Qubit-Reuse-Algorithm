#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controlled 1-step versus 1..5-step Ridge training, original 25 features.

Training data: previously frozen TRAIN split (ml_split_v1) and 5 discovery
seeds (42..46). No Qiskit, routing, or generation. Validation/final test
are never accessed by this program. Family-group out-of-fold evaluation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

STRUCT23 = [
    'quotient_unique_edges', 'quotient_density', 'max_degree',
    'max_weighted_degree', 'max_edge_weight',
    'mean_raw_jaccard_incompat', 'max_raw_jaccard_incompat',
    'top2_raw_jaccard_incompat', 'mean_raw_weighted_incompat',
    'max_raw_weighted_incompat', 'mean_chain_jaccard_incompat',
    'max_chain_jaccard_incompat', 'top2_chain_jaccard_incompat',
    'mean_chain_weighted_incompat', 'max_chain_weighted_incompat',
    'mean_first_2q_local_delay', 'mean_first_2q_layer_delay',
    'edge_weight_cv', 'max_edge_share', 'edge_weight_hhi',
    'normalized_edge_entropy', 'edge_weight_concentration',
    'weighted_degree_cv',
]
FEATURES = STRUCT23 + ['logical_depth', 'norm_depth']
assert len(FEATURES) == 25
KEY = ['circuit_id', 'original_sig', 'sig']
GROUP_KEY = ['circuit_id', 'original_sig']
OBJECTIVES = ('swap_count', 'routed_depth')


def fail(msg):
    raise ValueError(msg)


def ensure_columns(df, columns, name):
    missed = sorted(set(columns) - set(df.columns))
    if missed:
        fail(f'{name} missing columns: {missed}')


def read_csv(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'Missing file: {p.resolve()}')
    return pd.read_csv(p, encoding='utf-8-sig', low_memory=False)


def check_unique(df, keys, name):
    if df.duplicated(keys).any():
        rows = df.loc[df.duplicated(keys, keep=False), keys].head(5)
        fail(f'{name} has duplicate keys {keys}:\n{rows}')


def ensure_source_consistency(labels, selected):
    l = labels[labels.objective == 'swap_count'][KEY + ['discovery_move_depth','is_original']]
    m = selected[KEY + ['discovery_move_depth','is_original']].merge(
        l, on=KEY, how='outer', suffixes=('_selected','_label'), indicator=True,
        validate='one_to_one')
    if (m['_merge'] != 'both').any():
        fail(f'Candidate and SWAP label signature keys disagree: {m._merge.value_counts().to_dict()}')
    for c in ['discovery_move_depth','is_original']:
        if not (m[c+'_selected'].astype(int) == m[c+'_label'].astype(int)).all():
            fail(f'Disagreement in {c} between labels and selected_candidates')


def read_training_data(train_dir, split_path, require_seeds=5):
    train_dir = Path(train_dir)
    ledger = read_csv(split_path)
    ensure_columns(ledger, ['circuit_id','split','family_group'], 'split_ledger')
    check_unique(ledger,['circuit_id'],'split_ledger')
    train_keys = set(ledger.loc[ledger.split == 'train','circuit_id'].astype(str))
    if not train_keys:
        fail('No circuits in frozen train split')

    selected = read_csv(train_dir / 'selected_candidates.csv')
    labels = read_csv(train_dir / 'labels.csv')
    struct = read_csv(train_dir / 'features.csv')
    compiled = read_csv(train_dir / 'compiled_features.csv')
    ensure_columns(selected, KEY + ['discovery_move_depth','is_original'], 'selected_candidates')
    ensure_columns(labels, KEY + ['objective','gain_pct','abs_gain','baseline_mean',
                                  'n_seeds','discovery_move_depth','is_original'], 'labels')
    ensure_columns(struct, ['circuit_id','sig'] + STRUCT23, 'features (23 structural columns)')
    ensure_columns(compiled, ['circuit_id','sig','logical_depth','norm_depth'], 'compiled_features')
    for name, df in [('selected_candidates',selected),('labels',labels),
                     ('features',struct),('compiled_features',compiled)]:
        bad = set(df.circuit_id.astype(str)) - train_keys
        if bad:
            fail(f'{name} includes {len(bad)} NON-TRAIN circuits! First: {sorted(bad)[:5]}')

    if not labels.objective.isin(OBJECTIVES).all():
        fail('Unexpected objective; never train a weighted SWAP + Depth mixture.')
    check_unique(selected,KEY,'selected_candidates')
    check_unique(labels,KEY+['objective'],'labels')
    check_unique(struct,['circuit_id','sig'],'features')
    check_unique(compiled,['circuit_id','sig'],'compiled_features')
    ensure_source_consistency(labels,selected)
    if not (labels.n_seeds == require_seeds).all():
        fail('Labels are not all 5-seed discovery labels (42..46). Stop to inspect routing/labels.')
    if not selected.discovery_move_depth.between(0,5).all():
        fail('Found candidate deeper than 5 or negative-depth entry')
    if not (selected.loc[selected.is_original.astype(int)==1,'discovery_move_depth']==0).all():
        fail('Original has nonzero move depth')
    if not (selected.loc[selected.is_original.astype(int)==0,'discovery_move_depth'].between(1,5)).all():
        fail('Non-original has invalid discovery move depth')

    pair = selected.merge(ledger[['circuit_id','split','family_group']],
                          on='circuit_id',how='left',validate='many_to_one')
    pair = pair.merge(labels[labels.objective == 'swap_count'][KEY + ['baseline_mean']],
                      on=KEY,how='left',validate='one_to_one').rename(columns={'baseline_mean':'baseline_swap'})
    pair = pair.merge(labels[labels.objective == 'routed_depth'][KEY + ['baseline_mean']],
                      on=KEY,how='left',validate='one_to_one').rename(columns={'baseline_mean':'baseline_depth'})
    if pair[['baseline_swap','baseline_depth']].isna().any().any():
        fail('Missing label records for one objective')
    if (pair.split != 'train').any() or pair.family_group.isna().any():
        fail('Non-train or group-null row')

    allfeat = struct[['circuit_id','sig']+STRUCT23].merge(
        compiled[['circuit_id','sig','logical_depth','norm_depth']],
        on=['circuit_id','sig'],how='inner',validate='one_to_one')
    pair = pair.merge(allfeat,on=['circuit_id','sig'],how='left',validate='many_to_one',indicator=True)
    if (pair['_merge'] != 'both').any():
        fail('Some candidates have missing 25-feature vectors. Recompute TRAIN features, not routing.')
    pair.drop(columns=['_merge'],inplace=True)
    for f in FEATURES:
        pair[f] = pd.to_numeric(pair[f],errors='coerce')
        pair[f] = pair[f].replace([np.inf, -np.inf],np.nan)
    if pair[FEATURES].isna().all(axis=1).any():
        fail('At least one candidate has no usable features')

    # Match old adaptive Ridge: normalize all fixed catalog plan vectors
    # within each circuit before forming delta(Candidate - Original).
    # These statistics are computed from unlabeled candidate vectors only.
    # Unique (circuit, sig) is used to avoid plan-frequency weighting.
    unique_f = pair.drop_duplicates(['circuit_id','sig'])[['circuit_id','sig']+FEATURES].copy()
    unique_f[FEATURES] = unique_f[FEATURES].astype(float)
    for circuit, idx in unique_f.groupby('circuit_id').groups.items():
        subset = unique_f.loc[idx, FEATURES]
        mu = subset.mean(axis=0).fillna(0)
        filled = subset.fillna(mu)
        sd = filled.std(axis=0,ddof=0).replace(0,1).fillna(1)
        unique_f.loc[idx,FEATURES] = ((filled-mu)/sd).to_numpy(float)
    if not np.isfinite(unique_f[FEATURES].to_numpy(float)).all():
        fail('Feature normalization produced non-finite values')
    zcols = {f:'z_'+f for f in FEATURES}
    unique_f.rename(columns=zcols,inplace=True)
    pair = pair.drop(columns=FEATURES).merge(unique_f,on=['circuit_id','sig'],how='left',validate='many_to_one')

    src = pair[pair.is_original.astype(int)==1].copy()
    check_unique(src,GROUP_KEY,'originals')
    base = src[GROUP_KEY + list(zcols.values())].rename(
        columns={f'z_{f}':f'orig_z_{f}' for f in FEATURES})
    pair = pair.merge(base,on=GROUP_KEY,how='left',validate='many_to_one')
    if pair[[f'orig_z_{f}' for f in FEATURES]].isna().any().any():
        fail('Some candidate groups lack an Original feature vector')
    Xcols = [f'd_{f}' for f in FEATURES]
    for f in FEATURES:
        pair['d_'+f] = pair['z_'+f] - pair['orig_z_'+f]
    if not np.isfinite(pair[Xcols].to_numpy(float)).all():
        fail('Nonfinite normalized differences')

    wide = pair[KEY + ['family_group','discovery_move_depth','is_original',
                       'baseline_swap','baseline_depth'] + Xcols].copy()
    for objective in OBJECTIVES:
        l=labels[labels.objective==objective][KEY+['gain_pct','abs_gain']].rename(
            columns={'gain_pct':objective+'_gain_pct', 'abs_gain':objective+'_abs_gain'})
        wide = wide.merge(l,on=KEY,how='left',validate='one_to_one')
    if wide[[x+'_abs_gain' for x in OBJECTIVES]].isna().any().any():
        fail('Missing training labels')
    # No test/validation rows were accessed. Original is a zero-gain option.
    return wide, ledger, Xcols


def weights_per_family_source(frame):
    # Each family has equal mass; within family, each Original has equal mass;
    # within Original, each training Candidate has equal mass.
    # original_sig may be reused across circuits, so count composite unique groups accurately
    original_size=frame.groupby(['family_group','circuit_id','original_sig'])['sig'].transform('size')
    nsrc=frame[['family_group','circuit_id','original_sig']].drop_duplicates().groupby('family_group').size()
    nfamily=frame['family_group'].nunique()
    mass=frame['family_group'].map(1.0/nsrc).to_numpy(float)/original_size.to_numpy(float)
    # one unit of mass per family, then rescale sum(weights)==n_rows
    mass = mass * (len(frame)/mass.sum())
    return mass


def fit_model(train, cols, objective, alpha):
    X=train[cols].to_numpy(float)
    y=train[objective+'_gain_pct'].to_numpy(float)
    if not np.isfinite(y).all():
        fail('Non-finite label supplied to fit_model')
    mu=X.mean(axis=0);sd=X.std(axis=0)
    sd[(~np.isfinite(sd)) | (sd==0)]=1.0
    Z=(X-mu)/sd
    weights=weights_per_family_source(train)
    reg=Ridge(alpha=alpha,fit_intercept=True)
    reg.fit(Z,y,sample_weight=weights)
    return {'feature_mean':mu,'feature_sd':sd,'weights':reg.coef_,
            'intercept':float(reg.intercept_),'alpha':float(alpha)}


def predict(m, frame, cols):
    return ((frame[cols].to_numpy(float)-m['feature_mean'])/m['feature_sd']) @ m['weights'] + m['intercept']


def evaluation_rows(data, objective, arm, folds, alpha, cols):
    valid = data[data.is_original.astype(int)==0].copy()
    if objective=='swap_count':
        valid=valid[valid.baseline_swap > 0].copy()
    # Keep all depths on the evaluation side; train subset varies.
    valid=valid[np.isfinite(valid[objective+'_gain_pct'])].copy()
    all_groups=np.array(valid.family_group)
    all_rows=[]
    for foldno,(i_train,i_test) in enumerate(folds):
        tr=valid.iloc[i_train].copy(); te=valid.iloc[i_test].copy()
        if arm=='one_step':
            tr=tr[tr.discovery_move_depth == 1].copy()
        elif arm=='multi_step':
            tr=tr[tr.discovery_move_depth.between(1,5)].copy()
        else:
            fail('Unknown arm')
        if tr.empty or tr.family_group.nunique() < 2:
            fail(f'Fold {foldno}: insufficient train families for {arm}/{objective}')
        model=fit_model(tr,cols,objective,alpha)
        te['predicted_gain_pct']=predict(model,te,cols)
        te['fold']=foldno
        te['arm']=arm
        te['objective']=objective
        all_rows.append(te)
    return pd.concat(all_rows,ignore_index=True)


def decisions_for_oof(oof, objective):
    rows=[]
    for (c,osig),block in oof.groupby(GROUP_KEY,sort=False):
        block=block.sort_values(['predicted_gain_pct','sig'],ascending=[False,True])
        top=block.iloc[0]
        pred=float(top.predicted_gain_pct)
        chosen=(pred>0)
        gain=float(top[objective+'_gain_pct']) if chosen else 0.0
        if not np.isfinite(gain):
            fail('Nonfinite true gain in selection')
        rows.append({'circuit_id':c, 'original_sig':osig,
                     'family_group':top.family_group, 'objective':objective,
                     'arm':top.arm,'fold':int(top.fold),
                     'selected_sig':top.sig if chosen else osig,
                     'selected_depth':int(top.discovery_move_depth) if chosen else 0,
                     'predicted_gain_pct':pred if chosen else 0.0,
                     'actual_gain_pct':gain,'harm':int(gain<0),
                     'no_repair':int(not chosen),
                     'best_seen_gain_pct':max(0.0,block[objective+'_gain_pct'].max()),
                     'n_candidates':len(block)})
    return pd.DataFrame(rows)


def macro_summary(all_dec):
    out=[]
    for (objective,arm), g in all_dec.groupby(['objective','arm']):
        circuit=g.groupby('circuit_id').actual_gain_pct.mean()
        family=g.groupby('family_group').actual_gain_pct.mean()
        out.append({'objective':objective,'arm':arm,'n_circuits':g.circuit_id.nunique(),
                    'n_families':g.family_group.nunique(),'n_originals':len(g),
                    'gain_pct_equal_circuit':float(circuit.mean()),
                    'gain_pct_equal_family':float(family.mean()),
                    'gain_pct_equal_original':float(g.actual_gain_pct.mean()),
                    'harm_rate':float(g.harm.mean()),'no_repair_rate':float(g.no_repair.mean()),
                    'mean_oracle_seen_gain_pct':float(g.best_seen_gain_pct.mean())})
    return pd.DataFrame(out).sort_values(['objective','arm'])


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def run(args):
    data,ledger,cols=read_training_data(args.train_dir,args.split_ledger)
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    cand=data[data.is_original.astype(int)==0].copy()
    sizes=cand.groupby('discovery_move_depth').size().to_dict()
    print('TRAIN data:', data.circuit_id.nunique(),'circuits,',
          data.family_group.nunique(),'families,',
          data[GROUP_KEY].drop_duplicates().shape[0],'originals')
    print('Candidate counts per move depth:',sizes)
    print('Feature count:',len(cols),'(the same 23 structural + 2 depth as original Ridge)')
    print('Reserved split data NEVER read:', ledger.split.value_counts().to_dict())
    outputs=[]; preds=[];decisions=[]
    for objective in OBJECTIVES:
        valid=cand.copy()
        if objective=='swap_count':
            valid=valid[valid.baseline_swap>0].copy()
        valid=valid[np.isfinite(valid[objective+'_gain_pct'])].copy()
        ngrp=valid.family_group.nunique()
        if ngrp<3:
            fail(f'Not enough family groups for GroupKFold for {objective}')
        folds=list(GroupKFold(n_splits=min(args.folds,ngrp)).split(valid,groups=valid.family_group))
        for arm in ['one_step','multi_step']:
            p=evaluation_rows(data,objective,arm,folds,args.alpha,cols)
            d=decisions_for_oof(p,objective)
            preds.append(p[KEY+['family_group','discovery_move_depth','objective','arm','fold',
                                'predicted_gain_pct',objective+'_gain_pct']].rename(columns={objective+'_gain_pct':'true_gain_pct'}))
            decisions.append(d)
            fit_rows=valid if arm=='multi_step' else valid[valid.discovery_move_depth==1]
            mdl=fit_model(fit_rows,cols,objective,args.alpha)
            path=out / f'{objective}_{arm}_TRAIN_ONLY.json'
            write_json(path,{
                'protocol':'fixed_train_family_cv_25delta_ridge_v1',
                'objective':objective,'arm':arm,'target':'percent_gain_vs_original',
                'feature_names':cols,'raw_feature_order':FEATURES,
                'input':'per-circuit standardized feature difference Candidate minus Original',
                'alpha':mdl['alpha'],'intercept':mdl['intercept'],
                'weights':mdl['weights'].tolist(),
                'feature_mean':mdl['feature_mean'].tolist(),
                'feature_sd':mdl['feature_sd'].tolist(),
                'training_circuits':sorted(fit_rows.circuit_id.unique().tolist()),
                'n_training_rows':len(fit_rows),
                'training_seed_set':[42,43,44,45,46],
                'scope':'TRAIN ONLY; not independently evaluated or production validated'})
            outputs.append(str(path))
    all_dec=pd.concat(decisions,ignore_index=True)
    all_pred=pd.concat(preds,ignore_index=True)
    all_dec.to_csv(out/'oof_selected_by_original.csv',index=False)
    all_pred.to_csv(out/'oof_candidate_predictions.csv',index=False)
    macro=macro_summary(all_dec)
    macro.to_csv(out/'oof_macro_summary.csv',index=False)
    by_circuit=all_dec.groupby(['objective','arm','circuit_id'],as_index=False).agg(
        gain_pct=('actual_gain_pct','mean'),harm_rate=('harm','mean'),
        n_orig=('original_sig','size'))
    by_circuit.to_csv(out/'oof_per_circuit.csv',index=False)
    pairing=all_dec.pivot(index=GROUP_KEY,columns=['objective','arm'],values='actual_gain_pct')
    pair_out=[]
    for objective in OBJECTIVES:
        p=pairing[objective].dropna(subset=['one_step','multi_step'])
        for (cid,osig),row in p.iterrows():
            pair_out.append({'objective':objective,'circuit_id':cid,'original_sig':osig,
                             'one_step_gain_pct':row.one_step,'multi_step_gain_pct':row.multi_step,
                             'delta_pp':row.multi_step-row.one_step})
    pd.DataFrame(pair_out).to_csv(out/'paired_delta.csv',index=False)
    write_json(out/'manifest.json',{
        'status':'TRAIN ONLY cross-validation; not heldout external validation',
        'n_train_circuits':int(data.circuit_id.nunique()),
        'n_train_family_groups':int(data.family_group.nunique()),
        'n_source_plans':int(data[GROUP_KEY].drop_duplicates().shape[0]),
        'candidate_depth_counts':{str(k):int(v) for k,v in sorted(sizes.items())},
        'n_features':25,'features':FEATURES,
        'development_seeds':[42,43,44,45,46],
        'CV':'GroupKFold family_group; same heldout candidates for both arms',
        'arms':{'one_step':'train depth=1; score candidates depth1..5',
                'multi_step':'train depth1..5; score candidates depth1..5'},
        'alpha_fixed':args.alpha,
        'notes':['No validation/final-test data accessed','No routing performed',
                 'One-step and multi-step training candidate distributions differ by construction',
                 'OOF is not independent final-test evidence',
                 'New 28-circuit training dataset is different from old 15-circuit Ridge-Beam benchmark',
                 'The selected-plan evaluation is fixed-pool, not re-running model-guided beam search']})
    print('\nOUT-OF-FAMILY CV SUMMARY (training families only):')
    print(macro.round(4).to_string(index=False))
    print('\nSaved:',out.resolve())
    print('Files: oof_macro_summary.csv, paired_delta.csv, oof_per_circuit.csv,')
    print('       oof_selected_by_original.csv, oof_candidate_predictions.csv, 4 TRAIN_ONLY model JSON')


def self_test():
    # Deterministic regression and source balancing sanity test.
    x=pd.DataFrame({'family_group':['a','a','a','b','b','c'],
                    'circuit_id':['1','1','1','2','2','3'],
                    'original_sig':['o','o','o','p','p','q'],
                    'sig':['x','y','z','u','v','w'],
                    'd_f1':[1.,2.,3.,4.,5.,6.],
                    'swap_count_gain_pct':[1.,2.,3.,-1.,-2.,0.]})
    w=weights_per_family_source(x)
    assert len(w)==6 and np.all(np.isfinite(w)) and np.all(w>0)
    m=fit_model(x,['d_f1'],'swap_count',10.0)
    p=predict(m,x,['d_f1'])
    assert len(p)==6 and np.all(np.isfinite(p))
    assert len(FEATURES)==25 and len(set(FEATURES))==25
    print('self-test passed: 25 features, weights, fit/predict')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train-dir',default='ml_train_prepare_v1')
    ap.add_argument('--split-ledger',default='ml_split_v1/split_ledger.csv')
    ap.add_argument('--out-dir',default='ml_ridge_multistep25_v1')
    ap.add_argument('--alpha',type=float,default=10.0,
                    help='Fixed Ridge regularization for both arms; avoid tuning on outer folds')
    ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--self-test',action='store_true')
    a=ap.parse_args()
    if a.self_test:
        self_test()
    else:
        if a.alpha<0 or a.folds<2: fail('alpha must be nonnegative, folds >=2')
        run(a)
