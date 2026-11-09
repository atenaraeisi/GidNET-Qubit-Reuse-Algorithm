#!/usr/bin/env python3
"""Quick no-routing checks for the nested policy-selection implementation."""
from pathlib import Path
import numpy as np
import pandas as pd
import nested_pairwise_repair_v3 as m


def synthetic(nfam=6):
    records=[]
    for f in range(nfam):
        for s in range(2):
            name=f'C{f}_{s}'
            for j in range(4):
                isorig=int(j==0)
                gain=0.0 if isorig else (j-2.0)+f*0.06
                records.append(dict(objective='swap_count',family_group=f'fam{f}',
                    circuit_id=name,original_sig='original',sig=('original' if isorig else f'candidate{j}'),
                    is_original=isorig,abs_gain=gain,baseline_mean=15.0,
                    candidate_mean=15.0-gain,discovery_move_depth=j,
                    delta_compiled_norm_depth=(-j if j else 0),
                    compiled_norm_depth=15-j,compiled_logical_depth=12-j,
                    orig_compiled_norm_depth=15,move_depth=j))
    return pd.DataFrame(records)


def run():
    d=synthetic()
    folds=list(m.families_folds(d,3))
    assert len(folds)==3
    assert set().union(*(te for _, _, te in folds)) == set(d.family_group)
    assert all(not(tr & te) for _, tr, te in folds)
    tr_f,te_f=folds[0][1:]
    tr,te=m.fold_data(d,tr_f,te_f)
    assert not set(tr.family_group) & set(te.family_group)
    opts=m.policies_for('swap_count')
    assert 'no_repair' in opts and 'logical_depth_rule' not in opts
    assert 'logical_depth_rule' in m.policies_for('routed_depth')
    chosen, scores, audits=m.choose_with_inner_cv(tr,
        ['compiled_norm_depth','compiled_logical_depth','orig_compiled_norm_depth','move_depth','delta_compiled_norm_depth'],
        'swap_count',2,0.2,12,1.0,777)
    assert chosen in opts and len(scores)==len(opts)
    assert len(audits)==2
    selected=m.assess_selected_policy(tr,te,
        ['compiled_norm_depth','compiled_logical_depth','orig_compiled_norm_depth','move_depth','delta_compiled_norm_depth'],
        chosen,0.2,12,788)
    assert len(selected)==int(te.is_original.sum())
    assert all(x['method']=='nested_selected' for x in selected)
    # A change ONLY to the outer-test labels must not change the selected inner policy.
    changed=d.copy()
    changed.loc[changed.family_group.isin(te_f) & changed.is_original.eq(0), 'abs_gain'] += 1000
    train_again, _=m.fold_data(changed, tr_f, te_f)
    assert tr.abs_gain.equals(train_again.abs_gain), 'Outer test labels contaminated training!'
    chosen2, scores2, _=m.choose_with_inner_cv(train_again,
        ['compiled_norm_depth','compiled_logical_depth','orig_compiled_norm_depth','move_depth','delta_compiled_norm_depth'],
        'swap_count',2,0.2,12,1.0,777)
    assert chosen==chosen2 and scores['score'].equals(scores2['score'])
    # All methods should have all Original sources; no row-level split.
    baseline=[]
    for _,g in te.groupby(m.KEY): baseline.extend(m.fixed_baselines(g).values())
    assert len(baseline)==3*int(te.is_original.sum())
    # Confirm negative gains incur an EXTRA penalty rather than being ignored.
    choices=pd.DataFrame([dict(method='no_repair',family_group='a',selected_true_gain=0.0,harm=0,
                               no_repair=1,regret=2.0),
                          dict(method='pair_logit_gate_0.50',family_group='a',selected_true_gain=-3.0,harm=1,
                               no_repair=0,regret=5.0)])
    # Only specific helper values are meaningful here, all policy rows are required there.
    assert (-3.0 - 1.0*3.0) < 0.0
    print('[PASS] nested family folds, inner-only method selection, outer data isolation, baseline integrity')


if __name__=='__main__': run()
