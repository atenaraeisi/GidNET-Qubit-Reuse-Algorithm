#!/usr/bin/env python3
"""TRAIN-ONLY v4 SHADOW AUDIT: identical policy selection with post-decision counterfactual logging.

The ranker is trained on within-Original routing-cost comparisons. The gate is
trained on RANKER-SELECTED CANDIDATES obtained OUT-OF-FOLD within the gate's
training families. It learns P(gain > gate_margin), not pairwise probabilities.

Outer folds are NEVER used for model/threshold/feature-mode selection. Final
TRAIN-only policy is selected by inner CV and frozen independently. This is
sampled offline candidate-set reranking, NOT learned beam expansion.

Dependencies in same folder: train_pairwise_repair_v2.py, train_reuse_ranking_v1.py.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import train_pairwise_repair_v2 as pair
import train_reuse_ranking_v1 as base

KEY = ['circuit_id', 'original_sig']
OBJECTIVES = ('swap_count', 'routed_depth')
FEATURE_MODES = ('compiled', 'structural', 'full')
GATES = (0.50, 0.65, 0.75)
RANKERS = ('pair_logit', 'pair_hgb')
SEED = 2026
RANK_META = ('rank_borda', 'rank_borda_gap', 'rank_p_vs_original', 'rank_n_candidates')
UTILITY = 'mean_family_gain_minus_lambda_mean_family_harm'


def save_json(path, obj):
    def convert(x):
        if isinstance(x, np.integer): return int(x)
        if isinstance(x, np.floating): return float(x)
        if isinstance(x, np.bool_): return bool(x)
        if isinstance(x, dict): return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [convert(v) for v in x]
        return x
    Path(path).write_text(json.dumps(convert(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def family_folds(d, k):
    origins = d[d.is_original.eq(1)][KEY+['family_group']].drop_duplicates().reset_index(drop=True)
    if origins.duplicated(KEY).any(): raise AssertionError('Duplicate source Original')
    n = origins.family_group.nunique()
    if not 2 <= k <= n: raise ValueError(f'{k} folds needs >= {k} families, only {n}')
    seen = set()
    for fold, (tri, vai) in enumerate(GroupKFold(n_splits=k).split(origins, groups=origins.family_group)):
        trf = set(origins.iloc[tri].family_group)
        vaf = set(origins.iloc[vai].family_group)
        if trf & vaf or seen & vaf: raise AssertionError('Family overlap in CV')
        seen.update(vaf)
        yield fold, trf, vaf
    if seen != set(origins.family_group): raise AssertionError('Missing family in CV')


def select_features(all_columns, mode):
    if mode == 'compiled':
        features = [c for c in all_columns if 'compiled_' in c or c == 'move_depth']
    elif mode == 'structural':
        features = [c for c in all_columns if 'struct_' in c or c == 'move_depth']
    elif mode == 'full':
        features = list(all_columns)
    else: raise ValueError(mode)
    if not features: raise AssertionError(f'Empty feature selection {mode}')
    for c in features:
        if any(fragment in c for fragment in base.FORBIDDEN_FEATURE_FRAGMENTS):
            raise AssertionError(f'Physical-label predictor leakage: {c}')
    return features


def fit_ranker(tr, cols, ranker, margin, pair_cap, seed):
    # Consistent with previous pairwise pipeline. Only train the requested ranker.
    try:
        imputer = SimpleImputer(strategy='constant', fill_value=0.0, keep_empty_features=True)
    except TypeError:
        imputer = SimpleImputer(strategy='constant', fill_value=0.0)
    X = imputer.fit_transform(tr[cols].to_numpy(dtype=np.float64)).astype(np.float32)
    if X.shape[1] != len(cols): raise AssertionError('Imputer dropped a ranker feature')
    px, py, pw, info = pair.weighted_pairs(tr, X, margin, pair_cap, seed)
    model = pair.get_models()[ranker]
    model.fit(px, py, classifier__sample_weight=pw)
    return imputer, model, info


def top_candidate_with_rank_scores(group, cols, imputer, model):
    """No physical cost/ground-truth label is used in candidate selection."""
    ranked, oi = pair.model_rank_group(model, imputer, group, cols)
    c = ranked[ranked.is_original.eq(0)]
    if c.empty: return ranked, None, None
    c = c.sort_values(['borda', 'delta_compiled_norm_depth', 'discovery_move_depth', 'sig'],
                      ascending=[False, True, True, True], kind='mergesort')
    winner = c.iloc[0]
    return ranked, winner, {
        'rank_borda': float(winner.borda),
        'rank_borda_gap': float(winner.borda - ranked.iloc[oi].borda),
        'rank_p_vs_original': float(winner.p_vs_original),
        'rank_n_candidates': int(len(c)),
    }


def gate_example(winner, meta, features, *, training):
    record = {c: float(winner[c]) if pd.notna(winner[c]) else np.nan for c in features}
    record.update(meta)
    record.update(circuit_id=str(winner.circuit_id), original_sig=str(winner.original_sig),
                  family_group=str(winner.family_group), candidate_sig=str(winner.sig))
    if training: record['selected_true_gain'] = float(winner.abs_gain)
    return record


def crossfit_gate_examples(tr, cols, ranker, folds, margin, pair_cap, seed):
    """Gate-training outcomes from ranker predictions NEVER fitted on that family's labels."""
    examples = []
    eligible_sources = set(tuple(v) for v in tr.loc[tr.is_original.eq(1), KEY].to_numpy())
    seen = set()
    audit=[]
    for k, trf, vaf in family_folds(tr, min(folds, tr.family_group.nunique())):
        fit = tr[tr.family_group.isin(trf)].reset_index(drop=True)
        held = tr[tr.family_group.isin(vaf)].reset_index(drop=True)
        imp, mdl, counts = fit_ranker(fit, cols, ranker, margin, pair_cap, seed+17*k)
        for key, group in held.groupby(KEY, sort=True):
            if key in seen: raise AssertionError('Gate OOF duplicate Original')
            seen.add(key)
            _, selected, meta = top_candidate_with_rank_scores(group, cols, imp, mdl)
            if selected is None: continue
            examples.append(gate_example(selected, meta, cols, training=True))
        audit.append(dict(fold=k, train_families=len(trf), heldout_families=sorted(vaf), **counts))
    if seen != eligible_sources: raise AssertionError('Not all Original sources cross-fitted for gate')
    return pd.DataFrame(examples), audit


def gate_features(cols):
    return [*cols, *RANK_META]


def gate_weights(examples):
    num_fam = examples.groupby('family_group').size()
    w = np.asarray([1.0/num_fam[f] for f in examples.family_group], dtype=np.float64)
    return w/w.mean()


def fit_gate(examples, cols, improvement_margin):
    """Small, heavily regularized logistic gate on ranker-selected OOF candidates."""
    f = gate_features(cols)
    if examples.empty:
        return None, {'n_gate_examples':0, 'n_positive':0, 'positive_fraction':None,
                      'gate_model':'always_no_repair'}
    if examples.duplicated(KEY).any(): raise AssertionError('Duplicate gate-training Original')
    y = (examples.selected_true_gain.to_numpy(dtype=float) > improvement_margin + 1e-10).astype(int)
    x = examples[f].replace([np.inf, -np.inf], np.nan)
    w = gate_weights(examples)
    if len(np.unique(y)) == 1:
        # Constant class: never invent a confidence from the ranker.
        info_model='constant_class'
        model = None
        constant = int(y[0])
    else:
        info_model='regularized_logistic'
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(C=0.10, max_iter=1500, random_state=SEED))
        ])
        model.fit(x, y, classifier__sample_weight=w)
        constant = None
    artifact = dict(model=model, constant_class=constant, features=f)
    return artifact, {'n_gate_examples':int(len(examples)), 'n_positive':int(y.sum()),
                       'positive_fraction':float(y.mean()),'gate_model':info_model}


def gate_predict(artifact, x):
    if artifact is None: return np.zeros(len(x),dtype=float)
    if artifact['constant_class'] is not None:
        return np.full(len(x), float(artifact['constant_class']), dtype=float)
    return artifact['model'].predict_proba(x[artifact['features']])[:,1]


def selected_winners_for_heldout(held, cols, imputer, model):
    """One best candidate per Original with no prior gain/baseline routing lookup."""
    groups = {}
    examples = []
    for key, group in held.groupby(KEY, sort=True):
        ranked, winner, meta = top_candidate_with_rank_scores(group, cols, imputer, model)
        groups[key] = (ranked, winner)
        if winner is not None:
            examples.append(gate_example(winner, meta, cols, training=False))
    return groups, pd.DataFrame(examples)


def policies_for(objective, rankers, feature_modes, gates):
    result = ['no_repair']
    if objective == 'routed_depth': result.append('logical_depth_rule')
    result += [f'{r}__{m}__gate_{t:.2f}' for r in rankers for m in feature_modes for t in gates]
    return result


def parse_policy(p):
    if p in ('no_repair', 'logical_depth_rule'): return (p, None, None)
    pieces = p.split('__')
    if len(pieces)!=3 or pieces[0] not in RANKERS or pieces[1] not in FEATURE_MODES or not pieces[2].startswith('gate_'):
        raise ValueError(f'Unknown policy {p}')
    return (pieces[0], pieces[1], float(pieces[2][5:]))


def simple_baselines(group):
    cand = group[group.is_original.eq(0)]
    best = cand.sort_values(['abs_gain','sig'],ascending=[False,True],kind='mergesort').iloc[0] if not cand.empty else None
    oracle = best if best is not None and best.abs_gain > 0 else None
    return {
        'no_repair': pair.decision_row(group, None, 'no_repair'),
        'logical_depth_rule': pair.decision_row(group, pair.choose_rule(group), 'logical_depth_rule'),
        'oracle': pair.decision_row(group, oracle, 'oracle')
    }


def score_choices(rows, objective, penalty, policies):
    values=[]
    d=pd.DataFrame(rows)
    for p in policies:
        g=d[d.method.eq(p)]
        if g.empty: raise AssertionError(f'Missing policy {p}')
        fam_gain=g.groupby('family_group').selected_true_gain.mean()
        fam_harm=g.assign(harm_abs=(-g.selected_true_gain).clip(lower=0)).groupby('family_group').harm_abs.mean()
        if g.duplicated(KEY).any(): raise AssertionError('Duplicate OOF policy decision')
        if len(fam_gain)!=d.family_group.nunique(): raise AssertionError('Missing OOF family')
        value=float(fam_gain.mean()-penalty*fam_harm.mean())
        values.append(dict(objective=objective, method=p, mean_family_gain=float(fam_gain.mean()),
                           mean_family_harm=float(fam_harm.mean()), utility=value,
                           repair_count=int(g.no_repair.eq(0).sum()), harm_count=int(g.harm.sum()),
                           n_originals=len(g), n_families=len(fam_gain)))
    priority={p:i for i,p in enumerate(policies)}
    result=pd.DataFrame(values)
    result['_priority']=result.method.map(priority)
    result=result.sort_values(['utility','_priority'],ascending=[False,True],kind='mergesort')
    return result.drop(columns=['_priority']).reset_index(drop=True)



def shadow_winner(winner):
    """Audit-only ground truth, obtained AFTER the routing-free winner is selected.

    The feature is NEVER presented to ranker/gate, never used to select a policy,
    and never used to choose a threshold. This merely logs the counterfactual
    the ranker would have offered if the Gate were disabled.
    """
    if winner is None:
        return dict(pre_gate_candidate_sig=None, pre_gate_true_gain=np.nan,
                    pre_gate_move_depth=np.nan, pre_gate_rank_borda=np.nan,
                    pre_gate_p_vs_original=np.nan)
    return dict(pre_gate_candidate_sig=str(winner.sig),
                pre_gate_true_gain=float(winner.abs_gain),
                pre_gate_move_depth=int(winner.discovery_move_depth),
                pre_gate_rank_borda=float(winner.borda),
                pre_gate_p_vs_original=float(winner.p_vs_original))


def write_shadow_attribution(decisions, output_dir, gate_margin):
    """Observational error decomposition for OUTER-fold policies only."""
    z = decisions[decisions.method.eq('nested_selected')].copy()
    if not len(z): raise AssertionError('Missing nested-selected decisions')
    z['has_candidate'] = z.n_candidates.astype(int).gt(0)
    z['is_ranker_policy'] = z.selected_inner_policy.astype(str).str.startswith('pair_')
    z['oracle_has_benefit'] = z.oracle_gain.astype(float).gt(gate_margin + 1e-10)
    z['pre_gate_would_help'] = z.pre_gate_true_gain.astype(float).gt(gate_margin + 1e-10)
    z['gate_rejected'] = z.no_repair.astype(int).eq(1) & z.has_candidate & z.is_ranker_policy
    z['gate_rejected_good'] = z.gate_rejected & z.pre_gate_would_help
    z['ranker_missed_available_gain'] = (z.is_ranker_policy & z.has_candidate & z.oracle_has_benefit &
                                         ~z.pre_gate_would_help)
    z['gate_accepted_bad'] = (z.is_ranker_policy & z.no_repair.astype(int).eq(0) &
                              ~z.pre_gate_would_help)
    z['gate_accepted_good'] = (z.is_ranker_policy & z.no_repair.astype(int).eq(0) &
                               z.pre_gate_would_help)
    # A positive Oracle on a Gate rejection does NOT imply a gate false negative:
    # the ranker may have proposed a negative candidate.
    columns = ['objective','outer_fold','selected_inner_policy','circuit_id','family_group','original_sig',
               'n_candidates','oracle_gain','pre_gate_candidate_sig','pre_gate_true_gain',
               'pre_gate_move_depth','pre_gate_rank_borda','pre_gate_p_vs_original',
               'gate_predicted_prob','gate_threshold','no_repair','selected_true_gain',
               'has_candidate','is_ranker_policy','oracle_has_benefit','pre_gate_would_help',
               'gate_rejected_good','ranker_missed_available_gain','gate_accepted_bad','gate_accepted_good']
    z[columns].to_csv(output_dir/'outer_ranker_gate_attribution.csv',index=False,encoding='utf-8-sig')
    report=[]
    for objective,g in z.groupby('objective',sort=True):
        r=g[g.is_ranker_policy & g.has_candidate]
        report.append(dict(objective=objective,gate_margin=float(gate_margin),
                           originals=int(len(g)),ranker_policy_sources=int(len(r)),
                           ranker_proposed_good=int(r.pre_gate_would_help.sum()),
                           ranker_proposed_nonpositive=int((~r.pre_gate_would_help).sum()),
                           ranker_missed_available_gain=int(r.ranker_missed_available_gain.sum()),
                           gate_rejected_good=int(r.gate_rejected_good.sum()),
                           gate_accepted_good=int(r.gate_accepted_good.sum()),
                           gate_accepted_bad=int(r.gate_accepted_bad.sum()),
                           ranker_policy_gate_rejected=int(r.gate_rejected.sum()),
                           policy_final_gain_total=float(g.selected_true_gain.sum())))
    save_json(output_dir/'shadow_attribution_summary.json',dict(
       note=('Audit-only after routing-free ranker selection. A pre-gate positive is defined by '
             'true five-seed gain > specified margin. These truth values are not model features. '
             'Results are outer-OOF on TRAIN families, NOT untouched validation.'),
       rows=report))

def one_split_evaluate(tr, va, allcols, rankers, feature_modes, gates, crossfit_folds,
                       margin, cap, gate_margin, seed):
    """Inner fold: gate training uses cross-fitted ranker choices from tr only."""
    if set(tr.family_group) & set(va.family_group): raise AssertionError('Train/heldout family leakage')
    objective=str(tr.objective.iloc[0]); policies=policies_for(objective, rankers, feature_modes, gates)
    evaluations=[]
    baseline_groups={key: simple_baselines(g) for key,g in va.groupby(KEY,sort=True)}
    for key, b in baseline_groups.items():
        for p in policies:
            if p in ('no_repair','logical_depth_rule'):
                evaluations.append(b[p])
    audit=[]
    for mode in feature_modes:
        cols=select_features(allcols,mode)
        for ranker in rankers:
            gate_train, cfaudit = crossfit_gate_examples(tr,cols,ranker,crossfit_folds,margin,cap,seed)
            gate_artifact, ginfo = fit_gate(gate_train, cols, gate_margin)
            imp, mdl, rank_info = fit_ranker(tr,cols,ranker,margin,cap,seed+901)
            groups, candidates = selected_winners_for_heldout(va, cols, imp, mdl)
            if candidates.empty:
                probs_by_key={}
            else:
                probabilities=gate_predict(gate_artifact,candidates)
                probs_by_key={(str(row.circuit_id),str(row.original_sig)):float(p)
                              for row,p in zip(candidates.itertuples(index=False),probabilities)}
            for key,(ranked,winner) in groups.items():
                prob=probs_by_key.get(key)
                for threshold in gates:
                    selected=winner if winner is not None and prob is not None and prob>=threshold else None
                    method=f'{ranker}__{mode}__gate_{threshold:.2f}'
                    row=pair.decision_row(ranked,selected,method,threshold)
                    row['gate_predicted_prob']=prob
                    row.update(shadow_winner(winner))
                    evaluations.append(row)
            audit.append(dict(objective=objective, mode=mode, ranker=ranker,
                              train_families=int(tr.family_group.nunique()),
                              heldout_families=int(va.family_group.nunique()),
                              heldout_originals=int(va.is_original.sum()),
                              rank_training_pairs=rank_info['n_oriented_pairs'],
                              **ginfo, crossfit_fold_details=cfaudit))
    result=pd.DataFrame(evaluations)
    if len(result)!=int(va.is_original.sum())*len(policies):
        raise AssertionError(f'Expected {int(va.is_original.sum())*len(policies)}, got {len(result)}')
    if result.duplicated(['method',*KEY]).any(): raise AssertionError('Duplicate evaluations')
    return result, audit


def inner_choose_policy(tr, allcols, rankers, modes, gates, nfolds, crossfit_folds,
                        margin, cap, gate_margin, penalty, seed):
    obj=str(tr.objective.iloc[0]); policies=policies_for(obj,rankers,modes,gates)
    all_oof=[]; fold_audit=[]
    for idx, train_f, hold_f in family_folds(tr,nfolds):
        tr2=tr[tr.family_group.isin(train_f)].reset_index(drop=True)
        va2=tr[tr.family_group.isin(hold_f)].reset_index(drop=True)
        out,audit=one_split_evaluate(tr2,va2,allcols,rankers,modes,gates,
                                     crossfit_folds,margin,cap,gate_margin,seed+idx*1003)
        all_oof.append(out)
        fold_audit.append(dict(inner_fold=idx, training_families=sorted(train_f),
                               heldout_families=sorted(hold_f), models=audit))
    decisions=pd.concat(all_oof,ignore_index=True)
    if decisions.duplicated(['method',*KEY]).any(): raise AssertionError('Repeated source inner-OOF')
    if len(decisions)!=int(tr.is_original.sum())*len(policies): raise AssertionError('Incomplete inner-OOF')
    scores=score_choices(decisions,obj,penalty,policies)
    return str(scores.iloc[0].method), scores, fold_audit


def evaluate_locked_policy(tr, held, allcols, policy, crossfit_folds,
                           margin, cap, gate_margin, seed):
    name,mode,threshold=parse_policy(policy)
    if policy in ('no_repair','logical_depth_rule'):
        return [simple_baselines(g)[policy] for _,g in held.groupby(KEY,sort=True)], None, {}
    cols=select_features(allcols,mode)
    gate_train, cfaudit=crossfit_gate_examples(tr,cols,name,crossfit_folds,margin,cap,seed)
    gate, gate_info=fit_gate(gate_train,cols,gate_margin)
    imp, model, rank_info=fit_ranker(tr,cols,name,margin,cap,seed+901)
    groups,candidates=selected_winners_for_heldout(held,cols,imp,model)
    probs_by_key={}
    if not candidates.empty:
        probs=gate_predict(gate,candidates)
        probs_by_key={(str(row.circuit_id),str(row.original_sig)):float(p)
                      for row,p in zip(candidates.itertuples(index=False),probs)}
    results=[]
    for key,(ranked,winner) in groups.items():
        prob=probs_by_key.get(key)
        selected=winner if winner is not None and prob is not None and prob >= threshold else None
        row=pair.decision_row(ranked,selected,'nested_selected',threshold)
        row['gate_predicted_prob']=prob
        row.update(shadow_winner(winner))
        results.append(row)
    return results,dict(imputer=imp,ranker=model,gate=gate,feature_cols=cols),dict(gate=gate_info,ranker=rank_info)


def run_nested(data, columns, qc, output_dir, outer_folds, inner_folds, crossfit_folds,
               margin, cap, gate_margin, harm_penalty, rankers, feature_modes, gates):
    all_rows=[]; policy_rows=[]; inner_rows=[]; gate_audits=[]; final_locks={}
    for objective in OBJECTIVES:
        d=data[data.objective.eq(objective)].reset_index(drop=True)
        print(f'[start] {objective}: originals={int(d.is_original.sum())}, families={d.family_group.nunique()}',flush=True)
        for k,trf,tef in family_folds(d,outer_folds):
            tr=d[d.family_group.isin(trf)].reset_index(drop=True)
            te=d[d.family_group.isin(tef)].reset_index(drop=True)
            chosen,scores,audits=inner_choose_policy(tr,columns,rankers,feature_modes,gates,
                  inner_folds,crossfit_folds,margin,cap,gate_margin,harm_penalty,seed=SEED+10000*k)
            scores['objective']=objective; scores['outer_fold']=k; inner_rows.append(scores)
            gate_audits.append(dict(objective=objective,outer_fold=k,inner_audits=audits))
            chosen_rows,_,ainfo=evaluate_locked_policy(tr,te,columns,chosen,crossfit_folds,
                                                       margin,cap,gate_margin,seed=SEED+20000*k)
            policy_rows.append(dict(objective=objective,outer_fold=k,
                          inner_policy=chosen,inner_utility=float(scores.iloc[0].utility),
                          heldout_families='|'.join(sorted(tef)),
                          train_originals=int(tr.is_original.sum()),heldout_originals=int(te.is_original.sum()),
                          gate_train_sources=ainfo.get('gate',{}).get('n_gate_examples',0)))
            for row in chosen_rows:
                row['method']='nested_selected'; row['selected_inner_policy']=chosen
                row['outer_fold']=k
                all_rows.append(row)
            for key,group in te.groupby(KEY,sort=True):
                baselines=simple_baselines(group)
                for name in ('no_repair','logical_depth_rule','oracle'):
                    row=baselines[name];row['outer_fold']=k
                    all_rows.append(row)
            print(f'  [outer {k+1}/{outer_folds}] chosen={chosen}',flush=True)
        final_chosen,final_scores, final_audits=inner_choose_policy(d,columns,rankers,feature_modes,gates,
                  inner_folds,crossfit_folds,margin,cap,gate_margin,harm_penalty,seed=SEED+90000)
        final_scores['objective']=objective;final_scores['outer_fold']='all_training_for_final_lock'
        inner_rows.append(final_scores)
        gate_audits.append(dict(objective=objective,outer_fold='all_training_for_final_lock',inner_audits=final_audits))
        rname,mode,threshold=parse_policy(final_chosen)
        model_file=None
        if rname not in ('no_repair','logical_depth_rule'):
            cols=select_features(columns,mode)
            gate_train,cfaudit=crossfit_gate_examples(d,cols,rname,crossfit_folds,margin,cap,SEED+91000)
            gate,ginfo=fit_gate(gate_train,cols,gate_margin)
            imp,mdl,info=fit_ranker(d,cols,rname,margin,cap,SEED+91901)
            model_file=f'{objective}_rank_plus_gate_TRAIN_ONLY.joblib'
            joblib.dump(dict(protocol='train_only_independent_gate_v4',objective=objective,
                            policy=final_chosen,threshold=threshold,imputer=imp,ranker=mdl,
                            gate=gate,rank_features=cols,gate_features=gate_features(cols)),output_dir/model_file)
        final_locks[objective]=dict(selected_method=final_chosen,threshold=threshold,
                          model_file=model_file,score=float(final_scores.iloc[0].utility),
                          no_repair_fallback=True)
        print(f'  [final TRAIN-only] {objective}: {final_chosen}',flush=True)
    decisions=pd.DataFrame(all_rows)
    expected = int(qc['source_originals']) * len(OBJECTIVES) * 4
    if len(decisions)!=expected: raise AssertionError(f'Expected {expected} decision rows, got {len(decisions)}')
    if decisions.duplicated(['objective','method',*KEY]).any(): raise AssertionError('Duplicate outer-OOF results')
    for (obj,method),g in decisions.groupby(['objective','method']):
        if len(g)!=int(qc['source_originals']) or g.family_group.nunique()!=data.family_group.nunique():
            raise AssertionError(f'Incomplete outer-fold groups {obj}:{method}')
    comparison=pair.summarize(decisions)
    decisions.to_csv(output_dir/'nested_outer_decisions.csv',index=False,encoding='utf-8-sig')
    write_shadow_attribution(decisions, output_dir, gate_margin)
    comparison.to_csv(output_dir/'nested_method_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(policy_rows).to_csv(output_dir/'outer_fold_policies.csv',index=False,encoding='utf-8-sig')
    pd.concat(inner_rows,ignore_index=True).to_csv(output_dir/'inner_policy_scores.csv',index=False,encoding='utf-8-sig')
    save_json(output_dir/'fold_gate_audit.json',gate_audits)
    save_json(output_dir/'final_train_only_policy_lock.json',dict(protocol='train_only_independent_gate_v4',
            feature_ablation=list(feature_modes),ranker_types=list(rankers),threshold_grid=list(gates),
            objective_policies=final_locks,training_data_only=True,
            warning='Nested outer CV evaluates policy SELECTION PROCEDURE, not frozen final policies; Validation and Final Test not read.'))
    save_json(output_dir/'independent_gate_summary.json',dict(protocol='train_only_independent_gate_v4',
            data_checks=qc, n_predictors=len(columns),outer_folds=outer_folds,
            inner_folds=inner_folds,gate_crossfit_folds=crossfit_folds,
            feature_ablation=list(feature_modes),ranker_types=list(rankers),
            gate_thresholds=list(gates),rank_pair_margin=margin,gate_improvement_margin=gate_margin,
            harm_penalty=harm_penalty,utility=UTILITY,method_comparison=comparison.to_dict('records'),
            outer_policy_counts=[{'objective':o,'method':m,'count':int(n)} for (o,m),n in
                 pd.DataFrame(policy_rows).groupby(['objective','inner_policy']).size().items()],
            final_train_only_policies=final_locks,
            warnings=[
                'Train-only NESTED grouped CV; external Validation and Final Test untouched.',
                'Gate is fit on cross-fitted ranker-selected candidate outcomes from training families only.',
                'Ranking model does NOT use baseline physical SWAP/depth at inference, including zero-baseline guard.',
                'Gate probabilities are not empirically calibrated; a threshold is a model score threshold.',
                'Source Originals with no discovered candidates always produce No Repair.',
                'Only offline candidate reranking; no learned beam search yet.',
                'Oracle uses routed true labels and is only an optimistic bound.',
                'Only 14 training families, labels average 5 routing seeds: estimates are unstable.',
                'Threshold/feature selection inside inner CV, never on outer heldout folds.',
            ]))
    return comparison


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir',type=Path,required=True)
    p.add_argument('--split-ledger',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--feature-mode',choices=['full','compiled'],default='full',
                   help='Full 60-feature input enables compiled/structural/full ablation; use full for main experiment')
    p.add_argument('--rankers',default='pair_logit',help='Comma-separated: pair_logit[,pair_hgb]')
    p.add_argument('--outer-folds',type=int,default=5)
    p.add_argument('--inner-folds',type=int,default=3)
    p.add_argument('--gate-crossfit-folds',type=int,default=3)
    p.add_argument('--harm-penalty',type=float,default=1.0,
                   help='Predeclared utility; do not tune after seeing outer-CV results')
    p.add_argument('--pair-margin',type=float,default=0.2)
    p.add_argument('--gate-margin',type=float,default=0.2)
    p.add_argument('--max-candidate-pairs',type=int,default=140)
    args=p.parse_args(argv)
    rankers=tuple(x.strip() for x in args.rankers.split(',') if x.strip())
    if not rankers or len(set(rankers))!=len(rankers) or set(rankers)-set(RANKERS):p.error('Bad --rankers')
    if args.outer_folds<2 or args.inner_folds<2 or args.gate_crossfit_folds<2 or args.harm_penalty<0 or args.pair_margin<0 or args.gate_margin<0:
        p.error('Invalid fold/penalty/margin settings')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        p.error('Output directory must be new/empty to avoid mixing experiments')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    d,allcols,qc=pair.load_data(args.train_dir,args.split_ledger,args.feature_mode=='full')
    pair.check_original_group_integrity(d)
    if set(d.objective.unique())!=set(OBJECTIVES):raise ValueError('Missing objective')
    # Note: compiled mode is only for smoke, skips structural/full ablation.
    feature_modes=FEATURE_MODES if args.feature_mode=='full' else ('compiled',)
    for mode in feature_modes:select_features(allcols,mode)
    if d.family_group.nunique() < args.outer_folds+args.inner_folds-1:
        raise ValueError('Not enough independent families for nested grouped folds')
    print(f'[QC PASS] {qc["routing_rows"]} routing jobs, {qc["label_rows"]} labels, '
          f'{qc["source_originals"]} sources, {d.family_group.nunique()} families; modes={feature_modes}; rankers={rankers}',flush=True)
    summary=run_nested(d,allcols,qc,args.out_dir,args.outer_folds,args.inner_folds,
                       args.gate_crossfit_folds,args.pair_margin,args.max_candidate_pairs,
                       args.gate_margin,args.harm_penalty,rankers,feature_modes,GATES)
    print(summary[['objective','method','mean_gain_abs_by_original','mean_gain_abs_by_family',
                   'harm_count','no_repair_count','mean_regret']].round(3).to_string(index=False))
    print(f'[saved] {args.out_dir.resolve()}',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        print(f'[FAIL] {type(e).__name__}: {e}',file=sys.stderr)
        raise
