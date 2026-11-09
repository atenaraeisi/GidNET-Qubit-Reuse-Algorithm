#!/usr/bin/env python3
"""Nested family-group CV for validity-preserving quantum reuse plan selection.

TRAIN split ONLY. Inner GroupKFold selects policy/model/gate; outer GroupKFold
estimates the selection PROCEDURE on completely unseen circuit families.
No routing, no Qiskit, no validation/final-test reading, no learned beam expansion.

Requires train_pairwise_repair_v2.py and train_reuse_ranking_v1.py beside script.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer

import train_pairwise_repair_v2 as pair
import train_reuse_ranking_v1 as base

SEED = 2026
GATES = (0.50, 0.65, 0.75)
KEY = ['circuit_id', 'original_sig']
SCORE_POLICY = 'mean_of_family_mean_gain_minus_lambda_times_mean_of_family_mean_harm'


def families_folds(data: pd.DataFrame, n: int):
    """Original-level splitting keeps no-candidate originals in CV."""
    orig = data[data.is_original.eq(1)][KEY + ['family_group']].drop_duplicates().reset_index(drop=True)
    if orig.duplicated(KEY).any():
        raise ValueError('Non-unique source Originals')
    nf = orig.family_group.nunique()
    if n < 2 or nf < n:
        raise ValueError(f'Cannot have {n} folds with {nf} families')
    cv = GroupKFold(n_splits=n)
    seen = set()
    for ix, (tr, te) in enumerate(cv.split(orig, groups=orig.family_group)):
        train_f = set(orig.iloc[tr].family_group)
        test_f = set(orig.iloc[te].family_group)
        assert not train_f.intersection(test_f)
        assert not (seen & test_f), 'Family repeated across held-out folds'
        seen.update(test_f)
        yield ix, train_f, test_f
    assert seen == set(orig.family_group), 'Some families not evaluated'


def policies_for(objective: str):
    options = ['no_repair']
    if objective == 'routed_depth':
        options.append('logical_depth_rule')
    options += [f'{model}_gate_{gate:.2f}'
                for model in ('pair_logit', 'pair_hgb') for gate in GATES]
    return options


def parse_policy(name: str):
    if name in ('no_repair', 'logical_depth_rule'):
        return name, None
    a, sep, g = name.partition('_gate_')
    if not sep or a not in ('pair_logit', 'pair_hgb') or float(g) not in GATES:
        raise ValueError(f'Unexpected policy: {name}')
    return a, float(g)


def fixed_baselines(group: pd.DataFrame):
    origin = group[group.is_original.eq(1)]
    if len(origin) != 1:
        raise ValueError('Must contain exactly one original plan')
    cand = group[group.is_original.eq(0)].copy()
    if cand.empty:
        oracle_pick = None
    else:
        best = cand.sort_values(['abs_gain', 'sig'], ascending=[False, True], kind='mergesort').iloc[0]
        oracle_pick = best if best.abs_gain > 0 else None
    return {
        'no_repair': pair.decision_row(group, None, 'no_repair'),
        'logical_depth_rule': pair.decision_row(group, pair.choose_rule(group), 'logical_depth_rule'),
        'oracle': pair.decision_row(group, oracle_pick, 'oracle'),
    }


def evaluate_pair_policies(train: pd.DataFrame, test: pd.DataFrame, features,
                           margin: float, cap: int, seed: int):
    """Fit on train families only, evaluate all preset policies on held-out families."""
    imp, models, count = pair.fit_pair_models(train, features, margin, cap, seed)
    results = []
    for _, group in test.groupby(KEY, sort=True):
        group = group.reset_index(drop=True)
        baselines = fixed_baselines(group)
        for p in policies_for(str(group.objective.iloc[0])):
            if p in baselines:
                results.append(baselines[p]); continue
            name, gate = parse_policy(p)
            rank, oi = pair.model_rank_group(models[name], imp, group, features)
            selected = pair.choose_pairwise(rank, oi, gate)
            results.append(pair.decision_row(rank, selected, p, gate))
    return pd.DataFrame(results), count


def selection_scores(all_oof: pd.DataFrame, objective: str, lam: float):
    """One family one vote; penalize magnitude, not just number, of harmful repairs."""
    rows = []
    required = set(policies_for(objective))
    for policy, group in all_oof.groupby('method', sort=True):
        family_gain = group.groupby('family_group').selected_true_gain.mean()
        family_harm = group.assign(
            harm_abs=(-group.selected_true_gain).clip(lower=0)
        ).groupby('family_group').harm_abs.mean()
        n_families = all_oof.family_group.nunique()
        if family_gain.size != n_families or family_harm.size != n_families:
            raise AssertionError('Not all inner-held-out families have decisions')
        mean_gain = float(family_gain.mean())
        mean_harm = float(family_harm.mean())
        rows.append({
            'objective': objective, 'method': policy,
            'n_families': n_families, 'n_originals': len(group),
            'mean_gain_abs_by_family': mean_gain,
            'mean_harm_abs_by_family': mean_harm,
            'score': mean_gain - lam * mean_harm,
            'harm_count': int(group.harm.sum()),
            'repair_count': int(group.no_repair.eq(0).sum()),
            'mean_regret': float(group.regret.mean()),
        })
    z = pd.DataFrame(rows)
    if set(z.method) != required:
        raise AssertionError(f'Policy coverage: expected {required}, got {set(z.method)}')
    # Simplest policy wins exact/nearly-exact ties. Tie priority is predeclared.
    priority = {p: i for i, p in enumerate(policies_for(objective))}
    z['_priority'] = z.method.map(priority)
    z = z.sort_values(['score', '_priority'], ascending=[False, True],
                      kind='mergesort').reset_index(drop=True)
    return z.drop(columns=['_priority'])


def choose_with_inner_cv(train: pd.DataFrame, features, objective: str, inner_folds: int,
                         margin: float, cap: int, lam: float, seed: int):
    out, audit = [], []
    for f, tr_f, va_f in families_folds(train, inner_folds):
        tr = train[train.family_group.isin(tr_f)].reset_index(drop=True)
        va = train[train.family_group.isin(va_f)].reset_index(drop=True)
        assert set(tr.family_group).isdisjoint(set(va.family_group))
        choices, pair_count = evaluate_pair_policies(tr, va, features, margin, cap, seed + 31*f)
        out.append(choices)
        audit.append({'inner_fold': f, 'train_families': len(tr_f),
                      'heldout_families': sorted(va_f),
                      'train_originals': int(tr.is_original.sum()),
                      'heldout_originals': int(va.is_original.sum()),
                      **pair_count})
    oof = pd.concat(out, ignore_index=True)
    n_orig = int(train.is_original.sum())
    expected = n_orig * len(policies_for(objective))
    if len(oof) != expected:
        raise AssertionError(f'Expected {expected} inner OOF policy rows, got {len(oof)}')
    if oof.duplicated(['method', *KEY]).any():
        raise AssertionError('Duplicated inner OOF decision')
    scores = selection_scores(oof, objective, lam)
    return str(scores.iloc[0].method), scores, audit


def fit_single_pair(train: pd.DataFrame, features, name: str, margin: float, cap: int, seed: int):
    try:
        imp = SimpleImputer(strategy='constant', fill_value=0.0, keep_empty_features=True)
    except TypeError:
        imp = SimpleImputer(strategy='constant', fill_value=0.0)
    x = imp.fit_transform(train[features].to_numpy(dtype=np.float64)).astype(np.float32)
    if x.shape[1] != len(features):
        raise ValueError('Imputer dropped feature columns')
    px, py, pw, info = pair.weighted_pairs(train, x, margin, cap, seed)
    model = pair.get_models()[name]
    model.fit(px, py, classifier__sample_weight=pw)
    return imp, model, info


def assess_selected_policy(train, test, features, policy, margin, cap, seed):
    method, gate = parse_policy(policy)
    imp = model = None
    if gate is not None:
        imp, model, _ = fit_single_pair(train, features, method, margin, cap, seed)
    results = []
    for _, group in test.groupby(KEY, sort=True):
        group = group.reset_index(drop=True)
        if policy == 'no_repair':
            pick = None
        elif policy == 'logical_depth_rule':
            pick = pair.choose_rule(group)
        else:
            rank, oi = pair.model_rank_group(model, imp, group, features)
            pick = pair.choose_pairwise(rank, oi, gate)
            group = rank
        row = pair.decision_row(group, pick, 'nested_selected', gate)
        row['selected_inner_policy'] = policy
        results.append(row)
    return results


def assess_regressions(train, test, features):
    """Frozen regression baselines. Their hyperparameters and positive-score gate are fixed."""
    tr = train[train.is_original.eq(0)].copy()
    te = test[test.is_original.eq(0)].copy()
    models = base.models()
    if len(tr) == 0:
        raise ValueError('No training candidates for regression baseline')
    for name, model in models.items():
        model.fit(tr[features], tr.abs_gain, regressor__sample_weight=base.sample_weights(tr))
        if len(te):
            te['pred_' + name] = model.predict(te[features])
    out = []
    grouped = {k: v for k, v in te.groupby(KEY, sort=True)}
    for k, orig_group in test.groupby(KEY, sort=True):
        cand = grouped.get(k, te.iloc[:0]).copy()
        basegroup = orig_group.reset_index(drop=True)
        for name in models:
            choice = base.pick_baseline(cand, name)
            out.append(pair.decision_row(basegroup, choice, name))
    return out


def add_context(rows, outer_fold, heldout_f):
    for r in rows:
        r['outer_fold'] = outer_fold
        r['outer_heldout_families'] = '|'.join(sorted(heldout_f))
    return rows


def fold_data(data, train_f, test_f):
    tr = data[data.family_group.isin(train_f)].copy().reset_index(drop=True)
    te = data[data.family_group.isin(test_f)].copy().reset_index(drop=True)
    if not tr.family_group.isin(test_f).sum() == 0:
        raise AssertionError('Family leakage')
    return tr, te


def confidence_summary(outer: pd.DataFrame):
    """Cluster-bootstrap descriptive 95% interval for outer OOF mean family gain.
    With only 14 training families intervals are unstable; not final test CIs.
    """
    results = []
    rng = np.random.default_rng(4545)
    for obj, d in outer.groupby('objective'):
        ref = d[d.method == 'nested_selected'].groupby('family_group').selected_true_gain.mean()
        fams = list(ref.index)
        by = {}
        for method, g in d.groupby('method'):
            s = g.groupby('family_group').selected_true_gain.mean().reindex(fams)
            if s.isna().any():
                raise AssertionError('Some outer method lacks a family')
            by[method] = s.to_numpy(dtype=float)
        ix = rng.integers(0, len(fams), size=(2000, len(fams)))
        for method, vals in by.items():
            boots = vals[ix].mean(axis=1)
            diff = vals - by.get('logical_depth_rule', np.zeros(len(vals))) if obj == 'routed_depth' else vals
            boots_diff = diff[ix].mean(axis=1)
            results.append(dict(objective=obj, method=method,
                 family_mean_gain=float(vals.mean()),
                 family_bootstrap_95_low=float(np.quantile(boots, 0.025)),
                 family_bootstrap_95_high=float(np.quantile(boots, 0.975)),
                 family_diff_vs_reference=float(diff.mean()),
                 family_diff_bootstrap_95_low=float(np.quantile(boots_diff, 0.025)),
                 family_diff_bootstrap_95_high=float(np.quantile(boots_diff, 0.975)),
                 reference_method=('logical_depth_rule' if obj == 'routed_depth' else 'no_repair')))
    return pd.DataFrame(results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir', type=Path, required=True)
    p.add_argument('--split-ledger', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--feature-mode', choices=['full', 'compiled'], default='full')
    p.add_argument('--outer-folds', type=int, default=5)
    p.add_argument('--inner-folds', type=int, default=3)
    p.add_argument('--harm-penalty', type=float, default=1.0,
                   help='PREDECLARED: utility = family mean gain minus penalty × family mean harm magnitude')
    p.add_argument('--margin', type=float, default=0.2)
    p.add_argument('--max-candidate-pairs', type=int, default=140)
    args = p.parse_args()
    if args.outer_folds < 2 or args.inner_folds < 2 or args.harm_penalty < 0 or args.margin < 0 or args.max_candidate_pairs < 1:
        p.error('Invalid CV fold / risk penalty / pair limits')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        p.error('Output folder exists and is not empty; use a NEW --out-dir to avoid mixing results')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data, cols, qc = pair.load_data(args.train_dir, args.split_ledger, args.feature_mode == 'full')
    pair.check_original_group_integrity(data)
    if data.objective.nunique() != 2:
        raise ValueError('Both objectives must be present')
    all_outer = []
    per_fold = []
    inner_scores = []
    inner_audits = []
    final_locks = {}
    for obj in pair.OBJECTIVES:
        d = data[data.objective.eq(obj)].reset_index(drop=True)
        n_family = d.family_group.nunique()
        if n_family - (n_family // args.outer_folds + 1) < args.inner_folds:
            raise ValueError('Not enough training families for inner CV')
        print(f'[start] objective={obj}, families={n_family}, originals={int(d.is_original.sum())}', flush=True)
        for outer_ix, train_f, held_f in families_folds(d, args.outer_folds):
            tr, te = fold_data(d, train_f, held_f)
            chosen, scored, audits = choose_with_inner_cv(
                tr, cols, obj, args.inner_folds, args.margin,
                args.max_candidate_pairs, args.harm_penalty, SEED + outer_ix * 100)
            scored.insert(0, 'outer_fold', outer_ix)
            scored['objective'] = obj
            inner_scores.append(scored)
            for inner in audits:
                inner_audits.append(dict(objective=obj, outer_fold=outer_ix, **inner))
            per_fold.append(dict(objective=obj, outer_fold=outer_ix,
                n_outer_train_families=len(train_f), n_outer_test_families=len(held_f),
                heldout_families='|'.join(sorted(held_f)),
                n_train_originals=int(tr.is_original.sum()), n_test_originals=int(te.is_original.sum()),
                inner_chosen_policy=chosen,
                inner_chosen_score=float(scored.loc[scored.method.eq(chosen), 'score'].iloc[0])))
            rows = assess_selected_policy(tr, te, cols, chosen,
                                          args.margin, args.max_candidate_pairs,
                                          SEED + 900 + outer_ix)
            # Frozen, non-learned baselines and optimistic oracle are evaluated on exact same test originals.
            for _, group in te.groupby(KEY, sort=True):
                rows.extend(fixed_baselines(group.reset_index(drop=True)).values())
            rows.extend(assess_regressions(tr, te, cols))
            all_outer.extend(add_context(rows, outer_ix, held_f))
            print(f'  [outer {outer_ix+1}/{args.outer_folds}] held-out={len(held_f)} families, inner chose={chosen}', flush=True)
        # The nested OOF result estimates the SELECTION PROCEDURE. A final, train-only policy
        # is chosen from full Training using its own inner CV for later frozen Validation.
        final_chosen, final_scored, final_audit = choose_with_inner_cv(
            d, cols, obj, args.inner_folds, args.margin, args.max_candidate_pairs,
            args.harm_penalty, SEED + 4000)
        final_scored.insert(0, 'outer_fold', 'all_training_for_final_lock')
        final_scored['objective'] = obj
        inner_scores.append(final_scored)
        for inner in final_audit:
            inner_audits.append(dict(objective=obj, outer_fold='all_training_for_final_lock', **inner))
        kind, gate = parse_policy(final_chosen)
        model_path = None
        if gate is not None:
            imp, model, info = fit_single_pair(d, cols, kind, args.margin, args.max_candidate_pairs, SEED + 8000)
            model_path = f'{obj}_frozen_nested_policy_TRAIN_ONLY.joblib'
            joblib.dump(dict(objective=obj, scope='TRAIN_ONLY',
                  selection_rule='nested_inner_group_CV_family_weighted_utility',
                  selected_method=final_chosen, model_name=kind, gate=gate,
                  imputer=imp, model=model, feature_columns=cols,
                  margin=args.margin, max_candidate_pairs=args.max_candidate_pairs),
                  args.out_dir / model_path)
        final_locks[obj] = dict(selected_method=final_chosen,
            selected_gate=gate, model_file=model_path,
            final_inner_score=float(final_scored.loc[final_scored.method.eq(final_chosen), 'score'].iloc[0]),
            no_repair_fallback=True)
        print(f'  [final TRAIN-only policy] {obj}: {final_chosen}', flush=True)
    result = pd.DataFrame(all_outer)
    num_orig = qc['source_originals']
    methods = set(result.method)
    required = {'nested_selected', 'no_repair', 'logical_depth_rule', 'oracle', 'ridge', 'hgb'}
    if methods != required:
        raise AssertionError(f'Wrong outer methods {methods}')
    if len(result) != num_orig * len(pair.OBJECTIVES) * len(required):
        raise AssertionError(f'Wrong decision row count: {len(result)}')
    if result.duplicated(['objective', 'method', *KEY]).any():
        raise AssertionError('Duplicated held-out decision')
    for (obj, method), g in result.groupby(['objective', 'method']):
        if g.family_group.nunique() != data.family_group.nunique() or len(g) != num_orig:
            raise AssertionError('Family or original missing in outer CV')
    summary_methods = pair.summarize(result)
    confidence = confidence_summary(result)
    fold_sel = pd.DataFrame(per_fold)
    # Preserve ALL true label decisions; these are TRAIN outer-fold labels, never Validation/Test.
    result.to_csv(args.out_dir/'nested_outer_decisions.csv', index=False, encoding='utf-8-sig')
    summary_methods.to_csv(args.out_dir/'nested_method_comparison.csv', index=False, encoding='utf-8-sig')
    fold_sel.to_csv(args.out_dir/'outer_fold_policies.csv', index=False, encoding='utf-8-sig')
    pd.concat(inner_scores, ignore_index=True).to_csv(args.out_dir/'inner_policy_scores.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(inner_audits).to_csv(args.out_dir/'inner_fold_audit.csv', index=False, encoding='utf-8-sig')
    confidence.to_csv(args.out_dir/'nested_family_bootstrap_exploratory.csv', index=False, encoding='utf-8-sig')
    pair.save_json(args.out_dir/'final_train_only_policy_lock.json', dict(
        protocol='nested_family_group_cv_v3',
        model_selection_criterion=SCORE_POLICY,
        lambda_harm=args.harm_penalty,
        feature_mode=args.feature_mode, predictor_columns=cols,
        policies=final_locks,
        warning='Frozen only after nested TRAIN CV. NOT validated on separate Validation/Test.'))
    pair.save_json(args.out_dir/'nested_learning_summary.json', dict(
        protocol='nested_family_group_cv_v3',
        training_data_qc=qc, feature_mode=args.feature_mode,
        n_predictors=len(cols), outer_folds=args.outer_folds,
        inner_folds=args.inner_folds, pair_margin=args.margin,
        max_candidate_pairs=args.max_candidate_pairs,
        harm_penalty=args.harm_penalty,
        inner_selection_criterion=SCORE_POLICY,
        threshold_grid=list(GATES),
        selectable_policies={obj:policies_for(obj) for obj in pair.OBJECTIVES},
        outer_policy_selection_counts={obj:dict(Counter(
            fold_sel.loc[fold_sel.objective.eq(obj), 'inner_chosen_policy']))
            for obj in pair.OBJECTIVES},
        final_train_only_policy=final_locks,
        outer_method_comparison=summary_methods.where(pd.notna(summary_methods),None).to_dict('records'),
        warnings=[
            'Nested OUTER GroupKFold estimates policy-selection procedure, NOT any one frozen final method.',
            'Inner GroupKFold selects model+gate using only families within each outer-train partition.',
            'Full-TRAIN final policy is selected separately; it has NOT been tested on untouched Validation.',
 '14 families only; family bootstrap intervals are descriptive and unstable, not conclusive.',
            '5-seed training labels are noisy and no new routing seeds were evaluated.',
            'Only reranking sampled offline valid neighbors; NOT learned beam expansion.',
            'Oracle uses ground-truth route costs and is NOT deployable.',
            'SWAP relative gains with zero baseline are undefined.',
            'The probabilities of pairwise classifiers are NOT calibrated confidence scores.',
            'Pair observations share Original/circuit; they are NOT independent.',
        ]))
    print('[PASS] Nested family CV completed. OUTER holdouts were never used for inner selection.', flush=True)
    print(summary_methods[['objective','method','mean_gain_abs_by_original',
            'mean_gain_abs_by_family','harm_count','mean_regret']].round(3).to_string(index=False),flush=True)
    print(f'[saved] {args.out_dir.resolve()}',flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'[FAIL] {type(e).__name__}: {e}', file=sys.stderr)
        raise
